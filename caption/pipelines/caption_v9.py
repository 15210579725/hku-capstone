#!/usr/bin/env python3
"""v9 双流程 caption：pass0 语音对齐（silero-vad）+ pass1 caption+文字密集区域框
（2880px q80, HIGH media_resolution）+ pass2 ULTRA_HIGH crop 精读并修订 pass1
（2880px q60, ULTRA_HIGH media_resolution）。

逻辑照抄 quality_verify/captions/frozen/run_caption_2pass_v9_snapshot_20260903_0921.py
的 process_scene() / align_speech() / detect_speech_windows() / call_with_fallback()
等纯函数（SHA256 a400af84...，见该目录 SHA256.txt），只换了 I/O 层：
冻结脚本从本地 scene 目录读文件，这里从 cloud_worker 已经解出的内存字节
（walk 阶段渲染好的 2880px gaze_overlay 帧 + AudioSlicer 切出的 16k 音频）读，
不落盘、不重新请求 tar 字节区间。

刻意不 import 上层活跃的 ../../run_caption_2pass.py（那条线今天已经改了三次），
只 import 本目录内的冻结快照来源逻辑，保证本流水线的 v9 行为可复现。

与 caption_core.py（v6 单遍）并列，不改 caption_core.py 任何函数——v6 路径完全不动。
cloud_worker.py 按 CAPTION_VERSION 环境变量选择走 caption_core（v6）还是本模块（v9）。
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import threading
import time
import wave

import numpy as np
from PIL import Image
from google.genai import types

import caption_core as cc

MODEL = cc.DEFAULT_MODEL
ADC_PROJECT = cc.ADC_PROJECT
ADC_LOCATION = cc.ADC_LOCATION
# 可按项目/账号覆盖中转桶；生产默认值保持向后兼容。
# 新账号若没有旧桶的 object IAM，可在目标项目创建独立桶后通过 GCS_BUCKET 注入。
GCS_BUCKET = os.environ.get("GCS_BUCKET", "hku-capstone-caption-frames")

# ---- 与冻结脚本 process_scene() 一致的分辨率/画质 -------------------------
FRAME_DIM, FRAME_Q = 2880, 80        # pass1 帧（walk 阶段已按此渲染，这里只是常量记录）
CROP_DIM, CROP_Q = 2880, 60          # pass2 crop
MAX_INLINE_MB = 20.0
INLINE_RETRIES = 3
MAX_CROPS = 14
# 429 是 Vertex 的动态共享配额（DSQ）枯竭，不是可提额的项目配额，一次枯竭常持续
# 几十分钟。原来 3 次重试 × 20/40s 退避＝总共只等 60 秒，注定失败（实测 200 clip
# 里 104 条全挂在 429）。配额错误单独记账、不消耗硬错误预算，最多退避约 27 分钟。
QUOTA_RETRIES = 10
QUOTA_BACKOFF = [15, 30, 60, 90, 120, 180, 240, 300, 300, 300]

ULTRA = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH
HIGH = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH

_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------- prompts
# 以下三段与冻结脚本逐字节一致（未做任何改写），只是从 .py 常量搬到这里。

REGION_ADDENDUM = """

## ADDITIONAL OUTPUT — TEXT-DENSE REGION BOXES

Besides the caption, you MUST also locate every region containing DENSE READABLE TEXT that
deserves a zoomed-in second look: computer/laptop screens, phone screens, tablets, books,
documents, printed pages, menu boards, kiosk touchscreens, dense signage.

Add a top-level "text_regions" array to your JSON output:

"text_regions": [
  {"frame_index": 0, "hkt": "HH:MM:SS", "label": "laptop screen", "box_2d": [y0, x0, y1, x1]}
]

- frame_index is the 0-based index of the frame in the order given.
- box_2d uses NORMALIZED coordinates 0-1000 as [top, left, bottom, right].
- At most 2 regions per frame — the most text-dense ones.
- Include a region only when text is actually present and worth reading.
- If a frame has no text-dense region, simply omit that frame from the array.

Return the caption fields AND text_regions in the SAME JSON object."""

READ_PROMPT = """You are refining a first-person (egocentric) video caption using zoomed-in crops
of the text-dense regions the wearer looked at. Each crop is a screen, document, menu, or sign,
given in chronological order with its absolute HKT timestamp.

You are given the FIRST-PASS CAPTION below. It was written from downscaled full frames, so its
author could not read these screens. Your job is to read them and then CORRECT AND ENRICH that
caption with what the screens actually say.

## Step 1 — read every crop
For each crop, transcribe ALL meaningful readable text: menu items with prices, code lines,
chat messages, article titles, UI options and button labels, form fields, headings.
Preserve the original language (Chinese stays Chinese). Keep code indentation.
Skip trivial text: individual keyboard key letters, watermarks, generic product packaging,
OS menu-bar clock/battery.

## Step 2 — revise the caption segments
Using what you just read, rewrite each segment of the first-pass caption:
- "action": correct it to what the screen proves I was actually doing. If the first pass said
  "I browse the menu" and the screen shows me on the payment confirmation page, fix it.
  Name the concrete target: which item, which button, which file, which message.
- "environment": keep the first-pass structure — the FIRST segment holds the full establishing
  world state, every LATER segment holds ONLY what changed since the previous segment
  ("No change." is valid). Your job here is to correct and sharpen it with what the screens
  actually say: fix wrong screen content in the establishing shot, and make each delta name the
  real on-screen change (which page it navigated to, which message arrived, which item was
  selected). Do NOT expand the deltas back into full state dumps, and do not report changes that
  are only my head moving.
- "text_visible": the meaningful text actually readable at that moment, from your crop reading.
- Keep "time_range", "speech" and "objects" from the first-pass segment unless the screens prove
  them wrong. Never invent speech.
- Keep the same number of segments and the same time ranges as the first-pass caption.

## Privacy
Apply the same masking as the caption: real names/usernames -> "User X", school/university names
-> "School X"/"University X", emails -> "email_x@example.com", phone numbers -> "XXX-XXXX-XXXX",
addresses -> "Address X". Keep generic brand names.

Return ONLY JSON:
{
  "crops": [
    {"crop_index": 0, "hkt": "HH:MM:SS", "label": "kiosk screen",
     "text": ["line 1", "line 2"],
     "summary": "one sentence: what this screen shows"}
  ],
  "revised_segments": [
    {"time_range": "HH:MM:SS-HH:MM:SS",
     "action": "corrected first-person action naming the concrete on-screen target",
     "objects": ["..."],
     "environment": "exhaustive world state including full screen content",
     "text_visible": ["..."],
     "speech": "unchanged from first pass, or null",
     "details": "...",
     "revision_note": "what this segment changed vs the first pass, or 'unchanged'"}
  ]
}"""

SPEECH_PROMPT = """STEP 1 — Transcribe this audio VERBATIM. Write down only what you actually hear.
Do not invent, complete, or imagine dialogue. This is an isolated field recording from smart
glasses, NOT a scripted conversation. If a fragment is unintelligible, write "[unclear]".
Speech may come from the wearer or from other people nearby. The recording is noisy — background
chatter, machines and footsteps are common, so listen carefully for actual words.
{hint}
STEP 2 — An energy detector found these sound-activity regions. Their TIMINGS are authoritative
(accurate to ~0.03s) — never change them. Some regions contain only noise, not speech:
{windows}

Assign your STEP 1 utterances to these windows in chronological order. If a window contains no
intelligible speech (machine noise, footsteps, door sounds, ambient chatter), set text to
"[non-speech]". Never answer with a refusal such as "I'm not sure" — use "[non-speech]" instead.

Clip starts at {clip_start} HKT, so window offset t seconds means HKT = clip_start + t.

Return ONLY JSON:
{{"step1_raw_transcript": "...",
  "utterances": [{{"window": 0, "start_sec": 0.0, "end_sec": 0.0, "hkt": "HH:MM:SS",
                   "speaker": "wearer|other|unknown", "text": "verbatim words"}}]}}"""

REFUSAL_RE = re.compile(
    r"^\s*(i'?m not sure|i am not sure|unclear|inaudible|no speech|n/?a|none|"
    r"\[non-speech\]|\[unclear\]|cannot|can'?t determine|不确定|听不清)\s*[.。!！]?\s*$",
    re.I)

_FILLER_RE = re.compile(r"^[\s。，、．,.!！?？~～…\-—嗯哦啊呃唔呀哈噢欸诶,]*$")


# ---------------------------------------------------------------- silero-vad

_silero = None
_silero_lock = threading.Lock()


def _get_silero():
    """按需装载一次 silero-vad，跨 worker 线程共享（TorchScript 模型内部有 RNN 状态，
    多线程同时调用会段错误，靠 _silero_lock 串行化——与冻结脚本一致）。"""
    global _silero
    with _silero_lock:
        if _silero is None:
            import torch
            torch.set_num_threads(1)
            from silero_vad import load_silero_vad, get_speech_timestamps
            _silero = (load_silero_vad(), get_speech_timestamps)
        return _silero


def _read_mono_bytes(wav_bytes: bytes):
    w = wave.open(io.BytesIO(wav_bytes))
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    w.close()
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def detect_speech_windows(wav_bytes: bytes, min_speech=0.25, merge_gap=0.40, max_windows=24):
    """silero-vad -> [[start_sec, end_sec], ...]。时间精度 ~0.03s（比 Gemini 自报时间戳准）。"""
    try:
        model, get_ts = _get_silero()
        import torch
        a, sr = _read_mono_bytes(wav_bytes)
        dur = len(a) / sr if sr else 0.0
        if sr != 16000:
            idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(int)
            a = a[np.clip(idx, 0, len(a) - 1)]
        wav = torch.from_numpy(a / 32768.0).float()
        with _silero_lock:
            try:
                model.reset_states()
            except Exception:
                pass
            with torch.no_grad():
                stamps = get_ts(wav, model, sampling_rate=16000, return_seconds=True,
                                min_speech_duration_ms=int(min_speech * 1000),
                                min_silence_duration_ms=200, speech_pad_ms=120)
        segs = [[float(s["start"]), float(s["end"])] for s in stamps]
    except Exception as e:                                      # noqa: BLE001
        _log(f"    silero-vad failed ({str(e)[:80]}), no speech windows")
        try:
            a, sr = _read_mono_bytes(wav_bytes)
            return [], len(a) / sr if sr else 0.0
        except Exception:
            return [], 0.0

    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    kept = [s for s in merged if s[1] - s[0] >= min_speech]
    if len(kept) > max_windows:
        kept = sorted(sorted(kept, key=lambda s: s[1] - s[0], reverse=True)[:max_windows])
    return kept, dur


def _transcript_is_substantive(transcript):
    body = re.sub(r"\[[^\]]*\]", " ", transcript or "")
    body = re.sub(r"\s+", "", body)
    return len(body) >= 4 and not _FILLER_RE.match(body)


def _energy_windows(wav_bytes: bytes, frame_ms=30, min_speech=0.30, merge_gap=0.50, max_windows=12):
    """silero 拒判但 ASR transcript 有实质内容时的能量兜底。"""
    try:
        a, sr = _read_mono_bytes(wav_bytes)
    except Exception:                                            # noqa: BLE001
        return []
    n = max(1, int(sr * frame_ms / 1000))
    nf = len(a) // n
    if nf < 2:
        return []
    e = np.array([np.sqrt((a[i * n:(i + 1) * n] ** 2).mean() + 1e-9) for i in range(nf)])
    if e.max() <= 0:
        return []
    med, p95 = float(np.median(e)), float(np.percentile(e, 95))
    thr = med + 0.45 * max(p95 - med, 1e-6)
    segs, st = [], None
    for i, v in enumerate(e > thr):
        if v and st is None:
            st = i
        elif not v and st is not None:
            segs.append([st * frame_ms / 1000, i * frame_ms / 1000])
            st = None
    if st is not None:
        segs.append([st * frame_ms / 1000, nf * frame_ms / 1000])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    kept = [s for s in merged if s[1] - s[0] >= min_speech]
    if len(kept) > max_windows:
        def loud(s):
            i0, i1 = int(s[0] * 1000 / frame_ms), int(s[1] * 1000 / frame_ms)
            return float(e[i0:i1].mean()) if i1 > i0 else 0.0
        kept = sorted(sorted(kept, key=loud, reverse=True)[:max_windows])
    return kept


def hkt_plus(clip_start_hkt, offset_sec):
    m = re.search(r"(\d{2}):(\d{2}):(\d{2})", clip_start_hkt or "")
    if not m:
        return ""
    base = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    t = int(base + offset_sec) % 86400
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"


def align_speech(client, wav_bytes: bytes, clip_start: str, transcript: str = ""):
    """VAD 窗口（准时间）+ Gemini（准文字）-> 有精确时间戳的语音段。

    wav_bytes 由 cloud_worker 的 AudioSlicer.slice16k()+wav_bytes() 产出，已经是
    16kHz mono，不需要冻结脚本里那次额外的 _wav_16k() 重采样。
    """
    windows, dur = detect_speech_windows(wav_bytes)
    source = "silero"
    if not windows and _transcript_is_substantive(transcript):
        windows = _energy_windows(wav_bytes)
        source = "energy-fallback"
    if not windows:
        return {"windows": 0, "utterances": [], "duration": round(dur, 1), "source": "silero"}
    wl = "\n".join(f"  window {i}: {s:.2f}s - {e:.2f}s (duration {e-s:.2f}s)"
                   for i, (s, e) in enumerate(windows))
    hint = ""
    if transcript.strip():
        hint = ("\nA separate ASR system produced this rough transcript of the same clip. Its "
                "timings are coarse 30-second blocks and some words may be misheard, but it "
                "tells you which words are likely present. Use it to guide your listening; "
                "correct it where you hear something different:\n"
                + transcript.strip()[:1500] + "\n")
    prompt = SPEECH_PROMPT.format(windows=wl, clip_start=clip_start or "unknown", hint=hint)
    r = generate(client,
                 [types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                  types.Part.from_text(text=prompt)],
                 "speech", max_out=4096, retries=3)
    if not r.get("ok"):
        return {"windows": len(windows), "utterances": [], "error": r.get("error"),
                "duration": round(dur, 1)}
    j = parse_json(r["content"]) or {}
    utts = []
    for i, u in enumerate(j.get("utterances", [])):
        w = u.get("window", i)
        if not isinstance(w, int) or w >= len(windows):
            w = min(i, len(windows) - 1)
        s, e = windows[w]
        txt = (u.get("text") or "").strip()
        if not txt or REFUSAL_RE.match(txt):
            continue
        utts.append({"hkt": hkt_plus(clip_start, s), "hkt_end": hkt_plus(clip_start, e),
                     "start_sec": round(s, 2), "end_sec": round(e, 2),
                     "speaker": u.get("speaker", "unknown"), "text": txt})
    return {"windows": len(windows), "utterances": utts, "duration": round(dur, 1),
            "source": source, "raw": j.get("step1_raw_transcript", ""),
            "in": r.get("in", 0), "out": r.get("out", 0), "time": r.get("time", 0)}


# ---------------------------------------------------------------- API call

def parse_json(text):
    """与冻结脚本 parse_json() 逐行一致（不同于 caption_core.parse_caption_json 的返回形状：
    这里直接返回 obj 或 None，供 process_clip() 内部按原逻辑消费）。"""
    t = text or ""
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0]
    elif "```" in t:
        t = t.split("```")[1].split("```")[0]
    try:
        return json.loads(t.strip())
    except Exception:                                            # noqa: BLE001
        pass
    i = t.rfind("}")
    if i > 0:
        for sfx in ("", "]}", "}]}", "\"}]}", "\"}]}]}"):
            try:
                return json.loads(t[:i + 1].strip() + sfx)
            except Exception:                                    # noqa: BLE001
                continue
    return None


def generate(client, parts, tag, max_out=16384, retries=INLINE_RETRIES, stream=False):
    last = None
    a = q = 0                       # a=硬错误次数，q=429 配额次数，两本账分开
    while a < retries and q < QUOTA_RETRIES:
        t0 = time.time()
        try:
            cfg = types.GenerateContentConfig(temperature=0.3, max_output_tokens=max_out)
            if stream:
                chunks, u = [], None
                for ch in client.models.generate_content_stream(
                        model=MODEL, contents=parts, config=cfg):
                    if getattr(ch, "text", None):
                        chunks.append(ch.text)
                    if getattr(ch, "usage_metadata", None):
                        u = ch.usage_metadata
                if not chunks:
                    raise RuntimeError("empty stream")
                return {"ok": True, "time": round(time.time() - t0, 1),
                        "in": (u.prompt_token_count or 0) if u else 0,
                        "out": (u.candidates_token_count or 0) if u else 0,
                        "think": (u.thoughts_token_count or 0) if u else 0,
                        "content": "".join(chunks)}
            r = client.models.generate_content(model=MODEL, contents=parts, config=cfg)
            u = r.usage_metadata
            return {"ok": True, "time": round(time.time() - t0, 1),
                    "in": u.prompt_token_count or 0, "out": u.candidates_token_count or 0,
                    "think": u.thoughts_token_count or 0, "content": r.text or ""}
        except Exception as e:                                    # noqa: BLE001
            last = str(e)
            if "429" in last or "RESOURCE_EXHAUSTED" in last:
                q += 1
                if q >= QUOTA_RETRIES:
                    break
                # 抖动是为了让同一 job 里几个 worker 不要同时重发、再次自撞配额
                time.sleep(QUOTA_BACKOFF[min(q - 1, len(QUOTA_BACKOFF) - 1)]
                           + random.uniform(0, 20))
            else:
                a += 1
                if a >= retries:
                    break
                time.sleep(4 * a)
    return {"ok": False, "error": last}


_gcs_lock = threading.Lock()
_gcs_client = None


def get_gcs():
    global _gcs_client
    with _gcs_lock:
        if _gcs_client is None:
            from google.cloud import storage
            _gcs_client = storage.Client(project=ADC_PROJECT)
        return _gcs_client


def upload_gcs(blobs, prefix, workers=8):
    import concurrent.futures as cf
    bucket = get_gcs().bucket(GCS_BUCKET)

    def up(item):
        i, (name, data) = item
        b = bucket.blob(f"{prefix}/{i:03d}_{name}")
        b.upload_from_string(data, content_type="image/jpeg")
        return f"gs://{GCS_BUCKET}/{prefix}/{i:03d}_{name}"

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(up, enumerate(blobs)))


def call_with_fallback(client, text_parts, blobs, res_level, tag, gcs_prefix, max_out=16384):
    total_mb = sum(len(b) for _, b in blobs) / 1024 / 1024

    def build(uris=None):
        parts = []
        for i, (txt, blob) in enumerate(zip(text_parts, blobs)):
            parts.append(types.Part.from_text(text=txt))
            if uris is None:
                parts.append(types.Part.from_bytes(data=blob[1], mime_type="image/jpeg",
                                                   media_resolution=res_level))
            else:
                parts.append(types.Part.from_uri(file_uri=uris[i], mime_type="image/jpeg",
                                                 media_resolution=res_level))
        return parts

    if total_mb <= MAX_INLINE_MB:
        r = generate(client, build(), tag, max_out)
        if r.get("ok"):
            r["transport"] = "inline"
            r["payload_mb"] = round(total_mb, 1)
            return r
        _log(f"    {tag}: inline failed {INLINE_RETRIES}x ({total_mb:.1f}MB) -> GCS fallback")
    else:
        _log(f"    {tag}: {total_mb:.1f}MB > {MAX_INLINE_MB}MB ceiling -> GCS directly")

    try:
        t0 = time.time()
        uris = upload_gcs(blobs, gcs_prefix)
        up = round(time.time() - t0, 1)
        r = generate(client, build(uris), tag, max_out, retries=2)
        r["transport"] = "gcs"
        r["upload_time"] = up
        r["payload_mb"] = round(total_mb, 1)
        return r
    except Exception as e:                                        # noqa: BLE001
        return {"ok": False, "error": f"gcs fallback failed: {e}", "transport": "gcs"}


def encode_img(img: Image.Image, max_dim: int, q: int) -> bytes:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return b.getvalue()


# ---------------------------------------------------------------- context

def build_context_text_v9(sd: dict, use_transcript: bool, use_ocr: bool, speech) -> str:
    """与冻结脚本 build_context(sd, frames, use_transcript, use_ocr, speech) 在
    n_crops=0（2pass 模式下 pass1 不带 zoom）时的输出逐字一致。"""
    frames = sd["frames"]
    head = f"""## Scene: {len(frames)}-second first-person video clip
- Date: {sd['clip_start'][:10] if sd['clip_start'] else 'unknown'}
- Absolute time range: {sd['clip_start']} to {sd['clip_end']} HKT
- Frame count: {len(frames)} frames at 1fps
- Frame timestamps: {frames[0]['hkt'] if frames else '?'} to {frames[-1]['hkt'] if frames else '?'} HKT
- IMPORTANT: Use these absolute HKT timestamps (HH:MM:SS format) for all time_range fields"""
    parts = [head]

    if speech and speech.get("utterances"):
        lines = "\n".join(
            f"  {u['hkt']}–{u['hkt_end']} [{u['speaker']}] \"{u['text']}\""
            for u in speech["utterances"])
        parts.append("## Speech (voice-activity-detected windows, timings are AUTHORITATIVE)\n"
                     "These timings come from acoustic energy analysis and are accurate to ~0.1s.\n"
                     "Use them verbatim for speech segment time_range values.\n" + lines)
    elif use_transcript:
        parts.append("## Audio Transcript\n" + sd["transcript"])

    if use_ocr:
        ocr_block = "\n".join(sd["ocr_texts"]) if sd["ocr_texts"] else "(no text detected)"
        if len(ocr_block) > 30000:
            ocr_block = ocr_block[:30000] + "\n... (OCR truncated)"
        parts.append("## OCR-detected Text (per-frame, with HKT timestamps)\n" + ocr_block)

    if not use_transcript and not use_ocr:
        parts.append("## No transcript or OCR provided\n"
                     "Derive everything — actions, on-screen text, speech — from the frames alone.")

    parts.append(f"## Video Frames (all {len(frames)} frames, 1 second apart, "
                 f"chronologically ordered)")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 主流程：一个 clip

def process_clip(client, sd: dict, system_prompt: str, wav16k: bytes | None,
                 clip_tag: str = "clip", use_transcript: bool = True, use_ocr: bool = True,
                 use_vad: bool = True, log=None):
    """sd: cloud_worker.build_sd(cid) 的输出（frames 需已按 2880px q80 渲染），
    wav16k: cloud_worker AudioSlicer 切出并 wav_bytes() 打包的 16kHz mono wav（可为 None）。

    逐段照抄冻结脚本 process_scene() 的三遍逻辑，返回同构的 out dict：
    {"pass1": {...}, "pass2": {...}, "speech": {...}, "merged": {...}, "n_crops": int, ...}
    """
    _l = log or _log
    t_start = time.time()
    frames = sd["frames"]
    out = {"model": MODEL, "n_frames": len(frames),
           "clip_start": sd.get("clip_start"), "clip_end": sd.get("clip_end")}

    # ---------- PASS 0: VAD 语音对齐 ----------
    speech = None
    if use_vad and use_transcript and wav16k:
        try:
            speech = align_speech(client, wav16k, sd.get("clip_start"), sd.get("transcript", ""))
            out["speech"] = speech
            _l(f"    {clip_tag} VAD {speech['windows']} windows -> "
               f"{len(speech.get('utterances', []))} utterances")
        except Exception as e:                                    # noqa: BLE001
            _l(f"    {clip_tag} VAD failed: {str(e)[:80]}")

    # ---------- PASS 1: 全量 caption + 区域框 ----------
    sys_text = (system_prompt + REGION_ADDENDUM + "\n\n"
                + build_context_text_v9(sd, use_transcript, use_ocr, speech))
    texts, blobs = [], []
    for i, f in enumerate(frames):
        texts.append(f"[frame_index {i} — {f['hkt']} HKT — {f['name']}]")
        blobs.append((f["name"], f["jpg"]))          # 已是 2880px q80，直接用，不再压缩
    texts[0] = sys_text + "\n\n" + texts[0]

    r1 = call_with_fallback(client, texts, blobs, HIGH, f"{clip_tag} pass1",
                            f"p1/{clip_tag}", max_out=16384)
    j1 = parse_json(r1["content"]) if r1.get("ok") else None
    out["pass1"] = {k: r1.get(k) for k in
                    ("ok", "time", "in", "out", "think", "transport", "payload_mb",
                     "upload_time", "error", "content")}
    out["pass1"]["n_segments"] = len(j1.get("segments", [])) if j1 else 0
    regions = j1.get("text_regions", []) if j1 else []
    out["pass1"]["n_regions"] = len(regions)

    if not r1.get("ok"):
        _l(f"    {clip_tag} pass1 failed: {str(r1.get('error'))[:100]}")
        out["total_time"] = round(time.time() - t_start, 1)
        return out
    _l(f"    {clip_tag} pass1 ok {r1['time']}s [{r1['transport']}] "
       f"segs={out['pass1']['n_segments']} regions={len(regions)} "
       f"payload={r1.get('payload_mb')}MB")

    # ---------- 从内存里已有的 2880px 帧字节直接 crop（不重取网络字节） ----------
    crops = []
    seen = set()
    for reg in regions:
        idx = reg.get("frame_index")
        box = reg.get("box_2d")
        if not isinstance(idx, int) or idx < 0 or idx >= len(frames) or not box or len(box) != 4:
            continue
        src = Image.open(io.BytesIO(frames[idx]["jpg"]))
        W, H = src.size
        y0, x0, y1, x1 = box
        l, t = int(x0 / 1000 * W), int(y0 / 1000 * H)
        rr, b = int(x1 / 1000 * W), int(y1 / 1000 * H)
        pw, ph = int(W * .03), int(H * .03)
        l, t = max(0, l - pw), max(0, t - ph)
        rr, b = min(W, rr + pw), min(H, b + ph)
        if rr - l < 80 or b - t < 80:
            continue
        key = (idx, l // 60, t // 60, rr // 60, b // 60)
        if key in seen:
            continue
        seen.add(key)
        crops.append({"hkt": reg.get("hkt") or frames[idx]["hkt"], "label": reg.get("label", "region"),
                      "frame_index": idx, "img": src.crop((l, t, rr, b))})
    crops = crops[:MAX_CROPS]
    out["n_crops"] = len(crops)
    if not crops:
        _l(f"    {clip_tag}: no crops detected, pass2 skipped")
        out["total_time"] = round(time.time() - t_start, 1)
        return out

    # ---------- PASS 2: ULTRA_HIGH 精读 + 修订 pass1 ----------
    p1_caption = {"scene_summary": j1.get("scene_summary", ""),
                  "segments": [{k: s.get(k) for k in
                                ("time_range", "action", "objects", "environment",
                                 "text_visible", "speech", "details")}
                               for s in j1.get("segments", [])]}
    p1_block = ("## FIRST-PASS CAPTION (to be corrected and enriched)\n"
                + json.dumps(p1_caption, ensure_ascii=False, indent=1))

    texts2, blobs2 = [], []
    for i, c in enumerate(crops):
        w, h = c["img"].size
        texts2.append(f"[crop_index {i} — {c['hkt']} HKT — {c['label']} — native {w}x{h}]")
        blobs2.append((f"c{i:02d}.jpg", encode_img(c["img"], CROP_DIM, CROP_Q)))
    texts2[0] = READ_PROMPT + "\n\n" + p1_block + "\n\n" + texts2[0]

    r2 = call_with_fallback(client, texts2, blobs2, ULTRA, f"{clip_tag} pass2",
                            f"p2/{clip_tag}", max_out=16384)
    j2 = parse_json(r2["content"]) if r2.get("ok") else None
    out["pass2"] = {k: r2.get(k) for k in
                    ("ok", "time", "in", "out", "think", "transport", "payload_mb",
                     "upload_time", "error", "content")}
    n_text = sum(len(c.get("text", [])) for c in (j2.get("crops", []) if j2 else []))
    out["pass2"]["n_text_items"] = n_text

    if r2.get("ok"):
        _l(f"    {clip_tag} pass2 ok {r2['time']}s [{r2['transport']}] "
           f"crops={len(crops)} text_items={n_text} payload={r2.get('payload_mb')}MB")
    else:
        _l(f"    {clip_tag} pass2 failed: {str(r2.get('error'))[:100]}")

    # ---------- merge: pass2 修订版覆盖 pass1，crop 读数挂 screen_text_detail ----------
    if j1 and j2:
        by_hkt = {}
        for c in j2.get("crops", []):
            by_hkt.setdefault(c.get("hkt", ""), []).append(c)

        revised = j2.get("revised_segments") or []
        merged = json.loads(json.dumps(j1))
        merged.pop("text_regions", None)

        if revised:
            by_tr = {}
            for s in revised:
                key = re.sub(r"\s", "", s.get("time_range", ""))
                if key:
                    by_tr[key] = s
            base = merged.get("segments", [])
            new_segs = []
            for i, seg in enumerate(base):
                key = re.sub(r"\s", "", seg.get("time_range", ""))
                r = by_tr.get(key) or (revised[i] if i < len(revised) else None)
                if r:
                    seg = {**seg, **{k: v for k, v in r.items() if v}}
                new_segs.append(seg)
            merged["segments"] = new_segs
            out["pass2"]["n_revised"] = len(revised)

        for seg in merged.get("segments", []):
            m = re.findall(r"(\d{2}:\d{2}:\d{2})", seg.get("time_range", ""))
            if len(m) != 2:
                continue
            lo, hi = m
            detail = [{"hkt": h, "label": c.get("label"), "summary": c.get("summary"),
                       "text": c.get("text", [])}
                      for h, cs in by_hkt.items() if lo <= h <= hi for c in cs]
            if detail:
                seg["screen_text_detail"] = detail
        out["merged"] = merged

    out["total_time"] = round(time.time() - t_start, 1)
    return out


def is_complete(r: dict) -> bool:
    """与冻结脚本 main() 里 complete() 对 2pass 模式的判定一致。"""
    p1ok = bool((r.get("pass1") or {}).get("ok"))
    p2 = r.get("pass2")
    return p1ok and (r.get("n_crops", 0) == 0 or bool(p2 and p2.get("ok")))
