#!/usr/bin/env python3
"""Dense video captioning for first-person POV footage — PRODUCTION ENTRY POINT.

Production entry supports the approved single-pass default (`--mode 1pass`) and retains
the v9 two-pass path (`--mode 2pass`) as a backup.

PIPELINE
  Pass 0  silero-vad finds speech windows; Gemini fills in the words.
          VAD owns the timing because Gemini's own audio timestamps drift 1.8-3.8s
          while VAD windows land within ~0.03s. Falls back to an energy detector
          when silero rejects faint distant speech but the ASR transcript has real
          words — filler-only transcripts ("嗯。") stay dropped as ASR hallucination.
  Pass 1  Full prompt.txt caption at 2880px q80, plus normalized box_2d coordinates
          for every text-dense region (screens, documents, menus, signage).
  Pass 2  Crops those regions from the 2880px originals and reads them at
          ULTRA_HIGH media_resolution, then REVISES pass-1's segments with what the
          screens actually say — correcting actions, not just appending text.
  Merge   Pass-2 revisions win; crop readings attach as screen_text_detail.

WHY THESE NUMBERS (see 过程debug/ for the experiments)
  media_resolution is the real quality lever, not JPEG pixels — ULTRA_HIGH costs 4.1x
  the input tokens of LOW on identical bytes. Crops stay at native 2880 q60: the
  resolution matrix showed q85 doubles payload for no consistent gain, and 1440 loses
  breadcrumbs, tab bars and inline comments on crops whose native size exceeds 1440.
  Grid stitching was tested and rejected — cells shrink text past legibility.

OUTPUT  gemini_test_results/v8_2pass_<timestamp>[_tag].json
  Each record carries pass1.content, pass2.content and merged as three complete
  caption payloads, so nothing is lost to the merge.

USAGE
  python run_caption_2pass.py                          # all text-dense scenes
  python run_caption_2pass.py S14_self_order_kiosk     # one scene
  python run_caption_2pass.py --workers 3 --rounds 5   # production defaults
  python run_caption_2pass.py --no-transcript --no-ocr # vision-only ablation
  python run_caption_2pass.py --merge OLD.json --scenes A,B   # resume failures

KNOWN LIMIT
  2880px q80 pushes 30 frames to 33-44MB, past the 20MB inline ceiling, so every
  scene goes through GCS and each call drops ~15-20% of the time at random. --rounds
  retries with decreasing concurrency; expect several rounds before 16/16.

ALSO PRESENT: --mode 2pass
  The original v9 two-call architecture remains available as a backup. Its results,
  comparison videos and full experiment record live in 单遍版测试_待审核/.

CHANGED SINCE v9 WAS LOCKED (affects 2pass too)
  hkt_of() previously matched only the _HKT filename suffix, so clips recorded on the
  mainland (_BJ) had every frame timestamp silently blanked — S12 and S13 went to the
  model with unlabelled frames. Now matches any 2-4 letter timezone suffix.
"""

import argparse, base64, io, json, os, re, sys, threading, time, wave
import concurrent.futures as cf
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
SCENES_ROOT = ROOT / "mimo_v25_audit_30x30s_20260821_COMPLETE" / "scenes"
PROMPT_FILE = ROOT / "prompt.txt"
RESULTS_DIR = ROOT / "gemini_test_results"
CROP_DIR = ROOT / "caption_crops"
RESULTS_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

PROJECT = "project-0e21c343-2d89-405c-9d5"
LOCATION = "global"
MODEL = "gemini-3.7-flash"
GCS_BUCKET = "hku-capstone-caption-frames"

FRAME_DIM = 2880          # pass-1 frame encode: native, downscaling visibly hurt pass-1 quality
FRAME_Q = 80
CROP_DIM = 2880           # pass-2 crop encode: keep crops at native size, never downscale
CROP_Q = 60               # matrix test: q85 doubles payload for no consistent gain
MAX_INLINE_MB = 20.0      # inline ceiling before pre-emptive GCS
INLINE_RETRIES = 3        # inline attempts before GCS fallback
MAX_CROPS = 14

# ---- one-pass mode ---------------------------------------------------------
# Full frames carry the action and the spatial layout; a zoom crop per readable
# frame carries the text. Cheap frames plus one zoom each beat expensive frames:
# with a zoom on every readable second the model never has to read a screen off
# the downscaled frame, which is where the character errors came from.
# media_resolution fixes the token count, so pixels beyond its budget are pure
# bandwidth: 1024px frames and 1280px crops score identically to 1440/1600 while
# halving the payload, which is what keeps the request on the reliable inline path.
ONE_FRAME_DIM, ONE_FRAME_Q = 1024, 72
ONE_CROP_DIM, ONE_CROP_Q = 1280, 62
ONE_FRAME_RES = "medium"     # 560 tokens per frame (approved production default)
ONE_CROP_RES = "medium"      # 560 tokens per crop
ONE_MIN_OCR = 40             # characters, not bytes: below this a frame holds nothing to zoom into
DETECT_WORK = 1440           # region detector's working resolution

ULTRA = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH
HIGH = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
MEDIUM = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM
LOW = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
RES_LEVEL = {"low": LOW, "medium": MEDIUM, "high": HIGH, "ultra": ULTRA}
RES_TOKENS = {"low": 280, "medium": 560, "high": 1120, "ultra": 2240}

TEXT_DENSE_THRESHOLD = 5000
ALWAYS_INCLUDE = {"S01_atomic_action", "S14_self_order_kiosk", "S27_kitchen_fill_water"}

_print_lock = threading.Lock()
_gcs_lock = threading.Lock()
_gcs_client = None


def log(msg):
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------- prompts

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


# ---------------------------------------------------------------- helpers

def hkt_of(fname):
    """Wall-clock time from a frame filename.

    The suffix is not always _HKT — clips recorded on the mainland carry _BJ — and
    matching only _HKT silently blanked every timestamp for those scenes, leaving
    the model with unlabelled frames and unanchored zoom crops.
    """
    m = re.match(r"\d{4}-\d{2}-\d{2}_(\d{2})-(\d{2})-(\d{2})_[A-Z]{2,4}", fname)
    return f"{m.group(1)}:{m.group(2)}:{m.group(3)}" if m else ""


def encode(img, max_dim, q):
    if isinstance(img, Path):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return b.getvalue()


def parse_json(text):
    t = text
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0]
    elif "```" in t:
        t = t.split("```")[1].split("```")[0]
    try:
        return json.loads(t.strip())
    except Exception:
        pass
    i = t.rfind("}")
    if i > 0:
        for sfx in ("", "]}", "}]}", "\"}]}", "\"}]}]}"):
            try:
                return json.loads(t[:i + 1].strip() + sfx)
            except Exception:
                continue
    return None


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


_silero = None
_silero_lock = threading.Lock()


def _get_silero():
    """Load the silero-vad model once, shared across worker threads."""
    global _silero
    with _silero_lock:
        if _silero is None:
            import torch
            torch.set_num_threads(1)  # avoid nested thread pools under our own workers
            from silero_vad import load_silero_vad, get_speech_timestamps
            _silero = (load_silero_vad(), get_speech_timestamps)
        return _silero


def _read_mono(wav_path):
    w = wave.open(str(wav_path))
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    w.close()
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def detect_speech_windows(wav_path, min_speech=0.25, merge_gap=0.40, max_windows=24):
    """silero-vad -> [[start_sec, end_sec], ...].

    A real neural speech detector, unlike the energy heuristic it replaces: it separates
    speech from the constant ambient noise in these field recordings instead of firing on
    any loud frame. Timing comes from here rather than from Gemini because Gemini's own
    audio timestamps drift 1.8-3.8s, while VAD windows land within ~0.03s.
    """
    try:
        model, get_ts = _get_silero()
        import torch
        a, sr = _read_mono(wav_path)
        dur = len(a) / sr if sr else 0.0
        if sr != 16000:  # silero expects 16 kHz
            idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(int)
            a = a[np.clip(idx, 0, len(a) - 1)]
        wav = torch.from_numpy(a / 32768.0).float()
        # The TorchScript model carries internal RNN state and segfaults when several
        # worker threads call it at once — serialise inference and reset between clips.
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
    except Exception as e:
        log(f"    silero-vad failed ({str(e)[:60]}), no speech windows")
        try:
            a, sr = _read_mono(wav_path)
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


# Filler-only ASR output — Whisper emits these on silence and ambient noise.
_FILLER_RE = re.compile(r"^[\s。，、．,.!！?？~～…\-—嗯哦啊呃唔呀哈噢欸诶,]*$")


def _transcript_is_substantive(transcript):
    """True when the ASR transcript carries real words, not just Whisper's noise fillers."""
    body = re.sub(r"\[[^\]]*\]", " ", transcript or "")
    body = re.sub(r"\s+", "", body)
    return len(body) >= 4 and not _FILLER_RE.match(body)


def _energy_windows(wav_path, frame_ms=30, min_speech=0.30, merge_gap=0.50, max_windows=12):
    """Energy fallback for faint, distant speech that silero-vad scores below threshold."""
    try:
        a, sr = _read_mono(wav_path)
    except Exception:
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


def _wav_16k(wav_path):
    """Downsample to 16 kHz mono — plenty for speech, ~3x smaller payload, far fewer SSL drops."""
    raw = Path(wav_path).read_bytes()
    try:
        w = wave.open(str(wav_path))
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        w.close()
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        if sr > 16000:
            idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(int)
            a = a[np.clip(idx, 0, len(a) - 1)]
        buf = io.BytesIO()
        out = wave.open(buf, "wb")
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.clip(a, -32768, 32767).astype(np.int16).tobytes())
        out.close()
        return buf.getvalue()
    except Exception:
        return raw


def align_speech(client, wav_path, clip_start, transcript=""):
    """VAD windows (exact timing) + Gemini (exact words) -> accurately timed utterances."""
    windows, dur = detect_speech_windows(wav_path)
    source = "silero"
    if not windows and _transcript_is_substantive(transcript):
        # silero rejects faint distant speech (S05, S14). When the ASR transcript still
        # carries real words, retry with the looser energy detector so those clips are
        # not silently dropped. Filler-only transcripts ("嗯。") stay dropped — those are
        # Whisper hallucinating on ambient noise, and silero is right to reject them.
        windows = _energy_windows(wav_path)
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
    audio = _wav_16k(wav_path)
    r = generate(client,
                 [types.Part.from_bytes(data=audio, mime_type="audio/wav"),
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
            "in": r.get("in", 0), "out": r.get("out", 0)}


def get_gcs():
    global _gcs_client
    with _gcs_lock:
        if _gcs_client is None:
            from google.cloud import storage
            _gcs_client = storage.Client(project=PROJECT)
        return _gcs_client


def upload_gcs(blobs, prefix, workers=8):
    """blobs: list of (name, bytes) -> list of gs:// URIs."""
    bucket = get_gcs().bucket(GCS_BUCKET)

    def up(item):
        i, (name, data) = item
        b = bucket.blob(f"{prefix}/{i:03d}_{name}")
        b.upload_from_string(data, content_type="image/jpeg")
        return f"gs://{GCS_BUCKET}/{prefix}/{i:03d}_{name}"

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(up, enumerate(blobs)))


# ---------------------------------------------------------------- scenes

def all_scenes():
    out = []
    for d in sorted(SCENES_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        sub = next((x for x in d.iterdir() if x.is_dir()), None)
        if not sub:
            continue
        n = 0
        for f in (sub / "picture" / "ocr_text").glob("*.txt"):
            lines = [l for l in f.read_text().split("\n")
                     if not l.startswith(("capture_time", "capture_epoch", "frame_index"))]
            n += sum(len(l.strip()) for l in lines)
        out.append({"name": d.name, "ocr_chars": n, "subdir": sub})
    return out


def select_scenes(filt=None):
    a = all_scenes()
    if filt:
        return [s for s in a if s["name"] == filt]
    return [s for s in a if s["ocr_chars"] >= TEXT_DENSE_THRESHOLD or s["name"] in ALWAYS_INCLUDE]


def load_scene(sc):
    sub = sc["subdir"]
    gz = sub / "picture" / "gaze_overlay"
    fd = gz if gz.exists() else sub / "picture" / "masked"
    frames = sorted(fd.glob("*.jpg"))
    raw_dir = sub / "picture" / "masked"
    raw_frames = sorted(raw_dir.glob("*.jpg")) if raw_dir.exists() else frames
    transcript = (sub / "audio" / "transcript.txt").read_text().strip()
    ocr = []
    for f in sorted((sub / "picture" / "ocr_text").glob("*.txt")):
        lines = [l for l in f.read_text().split("\n")
                 if not l.startswith(("capture_time", "capture_epoch", "frame_index"))]
        c = "\n".join(lines).strip()
        if c:
            ocr.append(f"[{hkt_of(f.stem)} {f.stem}]: {c}")
    mf = sub / "_audit" / "metadata.json"
    meta = json.loads(mf.read_text()) if mf.exists() else {}
    return {"frames": frames, "raw_frames": raw_frames, "transcript": transcript,
            "ocr": ocr, "start": meta.get("clip_start_hkt", ""),
            "end": meta.get("clip_end_hkt", "")}


def build_context(sd, frames, use_transcript=True, use_ocr=True, speech=None,
                  n_crops=0):
    head = f"""## Scene: {len(frames)}-second first-person video clip
- Date: {sd['start'][:10] if sd['start'] else 'unknown'}
- Absolute time range: {sd['start']} to {sd['end']} HKT
- Frame count: {len(frames)} frames at 1fps{f", plus {n_crops} zoomed screen crops" if n_crops else ""}
- Frame timestamps: {hkt_of(frames[0].name)} to {hkt_of(frames[-1].name)} HKT
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
        ocr_block = "\n".join(sd["ocr"]) or "(no text detected)"
        if len(ocr_block) > 30000:
            ocr_block = ocr_block[:30000] + "\n... (OCR truncated)"
        parts.append("## OCR-detected Text (per-frame, with HKT timestamps)\n" + ocr_block)

    if not use_transcript and not use_ocr:
        parts.append("## No transcript or OCR provided\n"
                     "Derive everything — actions, on-screen text, speech — from the frames alone.")

    parts.append(f"## Video Frames (all {len(frames)} frames, 1 second apart, "
                 f"chronologically ordered)"
                 + (" with ZOOM crops interleaved" if n_crops else ""))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- API call

def generate(client, parts, tag, max_out=16384, retries=INLINE_RETRIES, stream=False):
    """One model call. stream=True keeps the socket busy while a long answer is
    produced — an exhaustive per-zoom transcription runs for minutes and a silent
    connection that long gets dropped as 'Server disconnected without sending a
    response'."""
    last = None
    for a in range(retries):
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
        except Exception as e:
            last = str(e)
            if a < retries - 1:
                # 429 needs a much longer cool-off than a dropped socket
                quota = "429" in last or "RESOURCE_EXHAUSTED" in last
                time.sleep((20 * (a + 1)) if quota else (4 * (a + 1)))
    return {"ok": False, "error": last}


def call_with_fallback(client, text_parts, blobs, res_level, tag, gcs_prefix, max_out=16384):
    """text_parts: list of (position, text) interleaved with blobs.

    blobs: list of (name, bytes). Tries inline first; on repeated failure re-sends
    the same images as GCS URIs.
    """
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

    # inline unless obviously over the ceiling
    if total_mb <= MAX_INLINE_MB:
        r = generate(client, build(), tag, max_out)
        if r.get("ok"):
            r["transport"] = "inline"
            r["payload_mb"] = round(total_mb, 1)
            return r
        log(f"      {tag}: inline failed {INLINE_RETRIES}x ({total_mb:.1f}MB) → GCS fallback")
    else:
        log(f"      {tag}: {total_mb:.1f}MB > {MAX_INLINE_MB}MB ceiling → GCS directly")

    try:
        t0 = time.time()
        uris = upload_gcs(blobs, gcs_prefix)
        up = round(time.time() - t0, 1)
        r = generate(client, build(uris), tag, max_out, retries=2)
        r["transport"] = "gcs"
        r["upload_time"] = up
        r["payload_mb"] = round(total_mb, 1)
        return r
    except Exception as e:
        return {"ok": False, "error": f"gcs fallback failed: {e}", "transport": "gcs"}



# ---------------------------------------------------------------- region detection
# What the two-pass pipeline buys with its first API call is coordinates: WHERE the
# readable text sits in each frame. That is recoverable on the CPU for nothing —
# text is a dense field of short high-contrast strokes, and the eye tracker already
# says which surface is being read. Detecting locally collapses the two calls into one.

def _text_mask(img_bgr, work=DETECT_WORK):
    """Binary map marking pixels that belong to a text line."""
    h, w = img_bgr.shape[:2]
    sc = work / max(h, w)
    small = cv2.resize(img_bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # a percentile cut, not Otsu: one specular highlight drags a global Otsu
    # threshold above the text everywhere else in the frame
    bw = ((grad > max(float(np.percentile(grad, 88)), 8)).astype(np.uint8)) * 255
    lines = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    keep = np.zeros_like(lines)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if ch < 3 or ch > small.shape[0] * 0.20 or cw < 8:
            continue
        if area / float(cw * ch) < 0.18:      # hollow outline, not glyphs
            continue
        keep[lab == i] = 1
    return keep, small.shape[:2]


def _box_iou(a, b):
    t, l = max(a[0], b[0]), max(a[1], b[1])
    bo, r = min(a[2], b[2]), min(a[3], b[3])
    if bo <= t or r <= l:
        return 0.0
    inter = (bo - t) * (r - l)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _grow(mask, t, l, b, r, H, W, step=0.06, max_side=0.72, tries=3):
    """Extend the box while text keeps spilling over its edges."""
    for _ in range(tries):
        band = max(2, int(min(H, W) * 0.02))
        edge = (mask[max(0, t - band):t, l:r].sum() + mask[b:min(H, b + band), l:r].sum()
                + mask[t:b, max(0, l - band):l].sum() + mask[t:b, r:min(W, r + band)].sum())
        inside = mask[t:b, l:r].sum()
        if inside <= 0 or edge < inside * 0.12:
            break
        if (b - t) >= H * max_side and (r - l) >= W * max_side:
            break
        gh, gw = int(H * step), int(W * step)
        t, l = max(0, t - gh), max(0, l - gw)
        b, r = min(H, b + gh), min(W, r + gw)
    return t, l, b, r


def text_box_at_gaze(mask, shape, gaze=None, base_frac=0.44, max_side=0.72):
    """The box around whatever text the wearer is looking at, normalised 0-1.

    Free-roaming text detection loses to whatever is glossiest in frame — a
    portable air cooler out-scores a dim kiosk menu on raw edge energy. The eye
    tracker already says which surface is being read, so the box is pinned there
    and the text map only nudges its centre and decides how far it grows.
    """
    H, W = shape
    side = int(min(H, W) * base_frac)
    if gaze is not None:
        cx, cy = gaze[0] * W, gaze[1] * H
    else:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())
    for _ in range(2):
        t = int(max(0, min(H - side, cy - side / 2)))
        l = int(max(0, min(W - side, cx - side / 2)))
        sub = mask[t:t + side, l:l + side]
        if sub.sum() <= 0:
            break
        ys, xs = np.nonzero(sub)
        cx += 0.45 * (l + float(xs.mean()) - cx)
        cy += 0.45 * (t + float(ys.mean()) - cy)
    t = int(max(0, min(H - side, cy - side / 2)))
    l = int(max(0, min(W - side, cx - side / 2)))
    b, r = min(H, t + side), min(W, l + side)
    t, l, b, r = _grow(mask, t, l, b, r, H, W, max_side=max_side)
    m = float(mask[t:b, l:r].sum())
    return {"box": (t / H, l / W, b / H, r / W), "mass": m,
            "density": m / max(1.0, (b - t) * (r - l))}


def load_gaze(sub):
    """frame filename -> normalised (x, y) gaze point, empty when unavailable."""
    import csv
    p = Path(sub) / "eye_tracking" / "gaze.csv"
    if not p.exists():
        return {}
    out = {}
    for row in csv.DictReader(open(p)):
        try:
            out[row["frame_file"]] = (float(row["gaze_x"]), float(row["gaze_y"]))
        except (ValueError, KeyError):
            continue
    return out


def ocr_char_counts(sub):
    """Per-frame recognised-character count — a free gate on which frames hold text."""
    out = {}
    for i, f in enumerate(sorted((Path(sub) / "picture" / "ocr_text").glob("*.txt"))):
        body = [l for l in f.read_text().split("\n")
                if not l.startswith(("capture_time", "capture_epoch", "frame_index"))]
        out[i] = len(re.sub(r"\s", "", "".join(body)))
    return out


def detect_zoom_crops(sub, frames, min_ocr=ONE_MIN_OCR, max_crops=0,
                      crop_res=ONE_CROP_RES, res_top=None, top_n=0):
    """One zoom crop per frame that actually has something readable in it."""
    gaze_px = load_gaze(sub)
    oc = ocr_char_counts(sub)
    crops = []
    for i, f in enumerate(frames):
        if oc.get(i, 0) < min_ocr:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        H0, W0 = img.shape[:2]
        mask, shape = _text_mask(img)
        g = None
        if f.name in gaze_px:
            gx, gy = gaze_px[f.name]
            g = (gx / W0, gy / H0)
        w = text_box_at_gaze(mask, shape, g)
        if not w or w["mass"] <= 0:
            continue
        t, l, b, r = w["box"]
        src = Image.open(f)
        W, H = src.size
        crops.append({"frame_index": i, "hkt": hkt_of(f.name), "ocr_chars": oc.get(i, 0),
                      "density": round(w["density"], 4), "res": crop_res,
                      "img": src.crop((int(l * W), int(t * H), int(r * W), int(b * H)))})
    if max_crops and len(crops) > max_crops:
        crops = sorted(sorted(crops, key=lambda c: -c["ocr_chars"])[:max_crops],
                       key=lambda c: c["frame_index"])
    # resolution follows how much text the frame holds: the OCR count is already on
    # disk and beats the detector's density score, which under-rates a terminal
    # dialog of 10px text — exactly the crop that needs the extra tokens
    if res_top and top_n:
        for c in sorted(crops, key=lambda c: -c["ocr_chars"])[:top_n]:
            c["res"] = res_top
    return crops


ZOOM_ADDENDUM = """

## ZOOMED SCREEN CROPS — YOUR ONLY RELIABLE SOURCE OF ON-SCREEN TEXT

Interleaved with the frames you will find ZOOM parts, each marked
[ZOOM of frame_index N - HH:MM:SS HKT]. A ZOOM is the text-dense part of that exact
frame, enlarged. The full frames are downscaled and their screen text is NOT reliably
legible; the ZOOMs are. Whenever you describe what a screen shows at time T, read it
off the ZOOM whose frame_index is nearest to T - never guess from the full frame.

For every ZOOM you must actually read out:
- code: the visible lines, with identifiers and comments as written
- menus / kiosks / shop screens: every item name and its price
- chat / messaging: sender names and message text
- browsers / editors: tab titles, breadcrumbs, sidebar and file-tree entries, status bars
- documents / slides / articles: headings and body text
- buttons, dialogs, dropdowns: the exact labels of every visible option

## REQUIRED EXTRA OUTPUT — ONE VERBATIM TRANSCRIPTION PER ZOOM

Alongside the caption fields, add a top-level "screen_text" array with ONE entry per
ZOOM you were given, in the order given:

"screen_text": [
  {"zoom_index": 0, "hkt": "HH:MM:SS", "label": "laptop screen",
   "text": ["every readable line, verbatim, one per element"],
   "summary": "one sentence: what this screen shows"}
]

"text" is a full transcription, not a sample: list every meaningful line you can read
in that ZOOM — each menu row with its price, each visible code line with its
indentation, each chat message, each sidebar entry, each tab title, each button
label. Keep the original language and the original wording; do not translate,
summarise or merge lines. Skip only genuinely trivial text: individual keyboard key
letters, watermarks, the OS clock and battery indicator. If a ZOOM shows the same
screen as the previous one, transcribe it again anyway — the timestamps differ and
the small differences are the point.

The per-segment "text_visible" field stays as it is: the handful of lines that matter
at that moment. "screen_text" is the exhaustive record; "text_visible" is the summary.

"screen_text" is an ADDITION, not a replacement. The caption fields keep their full
depth: the first segment's "environment" is still the long establishing description
of the whole world (physical space, every object with state and position, complete
screen content, world text, media playing, ambient audio), and later segments still
carry their deltas. Do not shorten "environment", "action" or "details" to make room
for the transcription — write both in full.

A ZOOM shows the state of the world at its own timestamp. Text you read in a ZOOM
belongs in that moment's segment: in "text_visible", and woven into "environment" as
the screen's actual content. Do not attribute a ZOOM's content to a time range it does
not cover, and do not carry a screen's old content forward once a later ZOOM shows it
changed. If a ZOOM is blurred past reading, say nothing about its text rather than
inventing plausible-looking content.

PRIVACY, LAST STEP BEFORE YOU ANSWER. Zooming in is exactly when real names,
usernames, school and university names, emails, phone numbers, addresses and
credentials become legible, and they hide inside ordinary-looking strings - chat
thread titles, browser tabs, file paths, sidebar history entries, window titles.
Re-read every string you are about to put in "text_visible", "environment",
"details" and "action", and apply the masking rules to each one:
  "HKU成绩证明申请网址"        -> "University X成绩证明申请网址"
  "zhang_wei - Google Chrome"  -> "User X - Google Chrome"
  "/Users/limei/hw5/train.py"  -> "/Users/User X/hw5/train.py"
Quoting a string verbatim off a screen is not an exemption from masking it."""


def process_scene_1pass(client, sc, prompt, frame_dim, frame_q, crop_dim, crop_q,
                        use_transcript=True, use_ocr=False, use_vad=True,
                        frame_res=ONE_FRAME_RES, crop_res=ONE_CROP_RES,
                        res_top=None, top_n=0, max_crops=0):
    """Caption a scene in a single API call, using locally detected zoom crops."""
    name = sc["name"]
    t_start = time.time()
    sd = load_scene(sc)
    frames = sd["frames"]
    out = {"scene": name, "model": MODEL, "n_frames": len(frames), "mode": "1pass",
           "clip_start": sd["start"], "clip_end": sd["end"],
           "config": {"frame_dim": frame_dim, "frame_q": frame_q,
                      "crop_dim": crop_dim, "crop_q": crop_q,
                      "frame_res": frame_res, "crop_res": crop_res,
                      "res_top": res_top, "top_n": top_n,
                      "use_transcript": use_transcript, "use_ocr": use_ocr,
                      "use_vad": use_vad}}

    speech = None
    if use_vad and use_transcript:
        wav = sc["subdir"] / "audio" / "anonymized.wav"
        if wav.exists():
            try:
                speech = align_speech(client, wav, sd["start"], sd["transcript"])
                out["speech"] = speech
                log(f"  · {name} VAD {speech['windows']} windows -> "
                    f"{len(speech['utterances'])} utterances")
            except Exception as e:
                log(f"  · {name} VAD failed: {str(e)[:60]}")

    t0 = time.time()
    crops = detect_zoom_crops(sc["subdir"], sd["raw_frames"], max_crops=max_crops,
                              crop_res=crop_res, res_top=res_top, top_n=top_n)
    det_time = round(time.time() - t0, 1)
    by_frame = {}
    for c in crops:
        by_frame.setdefault(c["frame_index"], []).append(c)

    sys_text = (prompt + (ZOOM_ADDENDUM if crops else "") + "\n\n"
                + build_context(sd, frames, use_transcript, use_ocr, speech,
                                n_crops=len(crops)))

    texts, blobs, levels = [], [], []
    for i, f in enumerate(frames):
        texts.append(f"[frame_index {i} — {hkt_of(f.name)} HKT]")
        blobs.append((f.name, encode(f, frame_dim, frame_q)))
        levels.append(frame_res)
        for c in by_frame.get(i, []):
            w, h = c["img"].size
            texts.append(f"[ZOOM of frame_index {i} — {c['hkt']} HKT — text-dense "
                         f"region, native {w}x{h} — READ ALL TEXT HERE]")
            blobs.append((f"zoom_{i:02d}.jpg", encode(c["img"], crop_dim, crop_q)))
            levels.append(c["res"])
    texts[0] = sys_text + "\n\n" + texts[0]

    est = sum(RES_TOKENS[l] for l in levels)
    # the exhaustive per-zoom transcription roughly quintuples the answer, so the
    # single call needs far more output room than the two-pass halves did
    r = call_mixed_resolution(client, texts, blobs, levels, f"{name} 1pass",
                              f"one/{name}", max_out=32768)
    j = parse_json(r["content"]) if r.get("ok") else None
    out["pass1"] = {k: r.get(k) for k in ("ok", "time", "in", "out", "think",
                                          "transport", "payload_mb", "upload_time",
                                          "error")}
    out["pass1"]["content"] = r.get("content", "")
    out["pass1"]["n_segments"] = len(j.get("segments", [])) if j else 0
    out["n_crops"] = len(crops)
    out["detect_time"] = det_time
    out["est_image_tokens"] = est
    out["crops"] = [{k: c[k] for k in ("frame_index", "hkt", "ocr_chars", "density", "res")}
                    for c in crops]
    if j:
        shots = j.pop("screen_text", []) or []
        by_hkt = {}
        for i, sh in enumerate(shots):
            hk = sh.get("hkt") or (crops[i]["hkt"] if i < len(crops) else "")
            if hk:
                by_hkt.setdefault(hk, []).append(sh)
        for seg in j.get("segments", []):
            m = re.findall(r"(\d{2}:\d{2}:\d{2})", seg.get("time_range", ""))
            if len(m) != 2:
                continue
            lo, hi = m
            det = [{"hkt": h, "label": sh.get("label"), "summary": sh.get("summary"),
                    "text": sh.get("text", [])}
                   for h, shs in by_hkt.items() if lo <= h <= hi for sh in shs]
            if det:
                seg["screen_text_detail"] = det
        out["merged"] = j
        out["n_screen_text"] = sum(len(sh.get("text", [])) for sh in shots)
        out["pass1"]["n_text_items"] = (
            sum(len(s.get("text_visible") or []) for s in j.get("segments", []))
            + out["n_screen_text"])
    if r.get("ok"):
        log(f"  ✓ {name} 1pass {r['time']}s [{r['transport']}] "
            f"segs={out['pass1']['n_segments']} zooms={len(crops)} "
            f"img_tok≈{est} in={r.get('in')}")
    else:
        log(f"  ✗ {name} 1pass failed: {str(r.get('error'))[:70]}")
    out["total_time"] = round(time.time() - t_start, 1)
    return out


def call_mixed_resolution(client, text_parts, blobs, levels, tag, gcs_prefix,
                          max_out=16384, stream=True):
    """Like call_with_fallback, but each image carries its own media_resolution."""
    total_mb = sum(len(b) for _, b in blobs) / 1024 / 1024

    def build(uris=None):
        parts = []
        for i, (txt, blob) in enumerate(zip(text_parts, blobs)):
            parts.append(types.Part.from_text(text=txt))
            lvl = RES_LEVEL[levels[i]]
            if uris is None:
                parts.append(types.Part.from_bytes(data=blob[1], mime_type="image/jpeg",
                                                   media_resolution=lvl))
            else:
                parts.append(types.Part.from_uri(file_uri=uris[i], mime_type="image/jpeg",
                                                 media_resolution=lvl))
        return parts

    if total_mb <= MAX_INLINE_MB:
        r = generate(client, build(), tag, max_out, stream=stream)
        if r.get("ok"):
            r["transport"] = "inline"
            r["payload_mb"] = round(total_mb, 1)
            return r
        log(f"      {tag}: inline failed {INLINE_RETRIES}x ({total_mb:.1f}MB) → GCS fallback")
    else:
        log(f"      {tag}: {total_mb:.1f}MB > {MAX_INLINE_MB}MB ceiling → GCS directly")
    try:
        t0 = time.time()
        uris = upload_gcs(blobs, gcs_prefix)
        up = round(time.time() - t0, 1)
        r = generate(client, build(uris), tag, max_out, retries=2, stream=stream)
        r["transport"] = "gcs"
        r["upload_time"] = up
        r["payload_mb"] = round(total_mb, 1)
        return r
    except Exception as e:
        return {"ok": False, "error": f"gcs fallback failed: {e}", "transport": "gcs"}


# ---------------------------------------------------------------- two-pass

def crops_from_regions(regions, frames, max_crops=MAX_CROPS, open_frame=Image.open):
    """Pass-1's `text_regions` -> croppable images, cut from the native-resolution frames.

    Lifted verbatim out of process_scene so the cloud pipeline (pipelines/batch_worker.py)
    crops with exactly the same geometry. `frames` is anything indexable by frame_index
    whose entries carry `.name` and can be turned into a PIL Image by `open_frame`.
    """
    crops = []
    seen = set()
    for reg in regions:
        idx = reg.get("frame_index")
        box = reg.get("box_2d")
        if idx is None or idx >= len(frames) or not box or len(box) != 4:
            continue
        src = open_frame(frames[idx])
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
        crops.append({"hkt": reg.get("hkt") or hkt_of(frames[idx].name),
                      "label": reg.get("label", "region"),
                      "frame_index": idx, "img": src.crop((l, t, rr, b))})

    return crops[:max_crops]


def merge_pass2(j1, j2):
    """Pass-2's revisions win over pass-1; crop readings attach as screen_text_detail.

    Lifted verbatim out of process_scene. Returns (merged, n_revised).
    """
    by_hkt = {}
    for c in j2.get("crops", []):
        by_hkt.setdefault(c.get("hkt", ""), []).append(c)

    revised = j2.get("revised_segments") or []
    merged = json.loads(json.dumps(j1))
    merged.pop("text_regions", None)

    if revised:
        # index revisions by time_range, falling back to positional order
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
    return merged, len(revised)


def process_scene(client, sc, prompt, crop_dim, crop_q, frame_dim, frame_q,
                  use_transcript=True, use_ocr=True, use_vad=True):
    name = sc["name"]
    t_start = time.time()
    sd = load_scene(sc)
    frames = sd["frames"]
    out = {"scene": name, "model": MODEL, "n_frames": len(frames),
           "clip_start": sd["start"], "clip_end": sd["end"],
           "config": {"frame_dim": frame_dim, "frame_q": frame_q,
                      "crop_dim": crop_dim, "crop_q": crop_q,
                      "use_transcript": use_transcript, "use_ocr": use_ocr,
                      "use_vad": use_vad}}

    # ---------- PASS 0 : VAD-anchored speech alignment ----------
    speech = None
    if use_vad and use_transcript:
        wav = sc["subdir"] / "audio" / "anonymized.wav"
        if wav.exists():
            try:
                speech = align_speech(client, wav, sd["start"], sd["transcript"])
                out["speech"] = speech
                log(f"  · {name} VAD {speech['windows']} windows → "
                    f"{len(speech['utterances'])} utterances")
            except Exception as e:
                log(f"  · {name} VAD failed: {str(e)[:60]}")

    # ---------- PASS 1 : full caption + region boxes ----------
    sys_text = (prompt + REGION_ADDENDUM + "\n\n"
                + build_context(sd, frames, use_transcript, use_ocr, speech))
    texts, blobs = [], []
    for i, f in enumerate(frames):
        texts.append(f"[frame_index {i} — {hkt_of(f.name)} HKT — {f.name}]")
        blobs.append((f.name, encode(f, frame_dim, frame_q)))
    # prepend system text into the first part
    texts[0] = sys_text + "\n\n" + texts[0]

    r1 = call_with_fallback(client, texts, blobs, HIGH, f"{name} pass1",
                            f"p1/{name}", max_out=16384)
    j1 = parse_json(r1["content"]) if r1.get("ok") else None
    out["pass1"] = {k: r1.get(k) for k in
                    ("ok", "time", "in", "out", "think", "transport", "payload_mb",
                     "upload_time", "error")}
    out["pass1"]["content"] = r1.get("content", "")
    out["pass1"]["n_segments"] = len(j1.get("segments", [])) if j1 else 0
    regions = j1.get("text_regions", []) if j1 else []
    out["pass1"]["n_regions"] = len(regions)

    if not r1.get("ok"):
        log(f"  ✗ {name} pass1 failed: {str(r1.get('error'))[:70]}")
        out["total_time"] = round(time.time() - t_start, 1)
        return out
    log(f"  ✓ {name} pass1 {r1['time']}s [{r1['transport']}] "
        f"segs={out['pass1']['n_segments']} regions={len(regions)}")

    # ---------- crop from 2880px originals ----------
    crops = crops_from_regions(regions, frames)
    out["n_crops"] = len(crops)
    if not crops:
        log(f"  · {name} no crops detected, pass2 skipped")
        out["total_time"] = round(time.time() - t_start, 1)
        return out

    cdir = CROP_DIR / name
    cdir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, c in enumerate(crops):
        w, h = c["img"].size
        p = cdir / f"c{i:02d}_{c['hkt'].replace(':','-')}_{re.sub(r'[^a-zA-Z0-9]+','_',c['label'])[:22]}.jpg"
        c["img"].convert("RGB").save(p, quality=92)
        meta.append({"crop_index": i, "hkt": c["hkt"], "label": c["label"],
                     "frame_index": c["frame_index"], "native": f"{w}x{h}", "path": str(p)})
    out["crops"] = meta

    # ---------- PASS 2 : read text at ULTRA_HIGH ----------
    # hand pass-1's caption to pass-2 so it can correct the behaviour, not just read text
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
        blobs2.append((f"c{i:02d}.jpg", encode(c["img"], crop_dim, crop_q)))
    texts2[0] = READ_PROMPT + "\n\n" + p1_block + "\n\n" + texts2[0]

    r2 = call_with_fallback(client, texts2, blobs2, ULTRA, f"{name} pass2",
                            f"p2/{name}", max_out=16384)
    j2 = parse_json(r2["content"]) if r2.get("ok") else None
    out["pass2"] = {k: r2.get(k) for k in
                    ("ok", "time", "in", "out", "think", "transport", "payload_mb",
                     "upload_time", "error")}
    out["pass2"]["content"] = r2.get("content", "")
    n_text = 0
    if j2:
        for c in j2.get("crops", []):
            n_text += len(c.get("text", []))
    out["pass2"]["n_text_items"] = n_text

    if r2.get("ok"):
        log(f"  ✓ {name} pass2 {r2['time']}s [{r2['transport']}] "
            f"crops={len(crops)} text_items={n_text}")
    else:
        log(f"  ✗ {name} pass2 failed: {str(r2.get('error'))[:70]}")

    # ---------- merge : pass-2 revised segments win, crop text attached ----------
    if j1 and j2:
        merged, n_revised = merge_pass2(j1, j2)
        if n_revised:
            out["pass2"]["n_revised"] = n_revised
        out["merged"] = merged

    out["total_time"] = round(time.time() - t_start, 1)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--scenes", help="comma-separated scene names to run")
    ap.add_argument("--merge", help="merge results into this existing v7 JSON")
    ap.add_argument("--workers", type=int, default=6, help="parallel scenes (default 6)")
    ap.add_argument("--crop-dim", type=int, default=CROP_DIM)
    ap.add_argument("--crop-q", type=int, default=CROP_Q)
    ap.add_argument("--frame-dim", type=int, default=FRAME_DIM)
    ap.add_argument("--frame-q", type=int, default=FRAME_Q)
    ap.add_argument("--no-transcript", action="store_true", help="ablation: withhold audio transcript")
    ap.add_argument("--no-ocr", action="store_true", help="ablation: withhold OCR text")
    ap.add_argument("--no-vad", action="store_true", help="skip VAD-anchored speech alignment")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--rounds", type=int, default=4,
                    help="auto-retry rounds for scenes that did not complete (default 4)")
    ap.add_argument("--mode", choices=("2pass", "1pass"), default="1pass",
                    help="1pass (default, production): one call with locally detected zoom crops. "
                         "2pass: caption+boxes, then a second "
                         "call that re-reads ULTRA crops and revises the segments. "
                         "1pass: one call with locally detected zoom crops, ~1/3 the "
                         "tokens — evaluated 2026-09-03 and approved for production")
    ap.add_argument("--crop-res", default=ONE_CROP_RES,
                    choices=("low", "medium", "high", "ultra"),
                    help="1pass: media_resolution for zoom crops (default medium)")
    ap.add_argument("--frame-res", default=ONE_FRAME_RES,
                    choices=("low", "medium", "high", "ultra"),
                    help="1pass: media_resolution for full frames (default low)")
    ap.add_argument("--res-top", choices=("high", "ultra"), default=None,
                    help="1pass: upgrade the N most text-dense crops to this level")
    ap.add_argument("--top-n", type=int, default=0, help="1pass: how many crops to upgrade")
    ap.add_argument("--max-crops", type=int, default=0,
                    help="1pass: cap zoom crops per scene (0 = one per readable frame)")
    args = ap.parse_args()

    if args.list:
        for s in all_scenes():
            mark = " ★" if s["ocr_chars"] >= TEXT_DENSE_THRESHOLD else ""
            mark += " ●" if s["name"] in ALWAYS_INCLUDE else ""
            print(f"  {s['name']:42s} ocr={s['ocr_chars']:6d}{mark}")
        return 0

    prompt = PROMPT_FILE.read_text().strip()
    if args.scenes:
        want = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes = [s for s in all_scenes() if s["name"] in want]
        missing = want - {s["name"] for s in scenes}
        if missing:
            print(f"unknown scenes: {', '.join(sorted(missing))}")
    else:
        scenes = select_scenes(args.scene)
    if not scenes:
        print(f"no scene matched '{args.scene or args.scenes}'")
        return 1

    use_tr = not args.no_transcript
    one = args.mode == "1pass"
    # the zoom crops replace the OCR block outright: they read the same screens at a
    # resolution the OCR service never had, and the block costs 8-19k tokens a scene
    use_ocr = (not args.no_ocr) and not one
    use_vad = not args.no_vad
    if one:
        args.frame_dim = args.frame_dim if args.frame_dim != FRAME_DIM else ONE_FRAME_DIM
        args.frame_q = args.frame_q if args.frame_q != FRAME_Q else ONE_FRAME_Q
        args.crop_dim = args.crop_dim if args.crop_dim != CROP_DIM else ONE_CROP_DIM
        args.crop_q = args.crop_q if args.crop_q != CROP_Q else ONE_CROP_Q
    print(f"Prompt: {len(prompt)} chars + {'zoom' if one else 'region'} addendum")
    print(f"Model: {MODEL} | ADC {PROJECT} @ {LOCATION}")
    print(f"Mode: {args.mode}")
    if one:
        print(f"Frames: {args.frame_dim}px q{args.frame_q} @ {args.frame_res.upper()} "
              f"({RES_TOKENS[args.frame_res]} tok each)")
        print(f"Zooms:  {args.crop_dim}px q{args.crop_q} @ {args.crop_res.upper()} "
              f"({RES_TOKENS[args.crop_res]} tok each)"
              + (f", top {args.top_n} at {args.res_top.upper()}" if args.res_top else ""))
    else:
        print(f"Pass1 frames: {args.frame_dim}px q{args.frame_q} (default media_resolution)")
        print(f"Pass2 crops:  {args.crop_dim}px q{args.crop_q} ULTRA_HIGH + segment revision")
    print(f"Inputs: transcript={use_tr} ocr={use_ocr} vad={use_vad}")
    print(f"Scenes: {len(scenes)} | workers: {args.workers}\n")

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    t0 = time.time()

    def submit(ex, sc):
        if one:
            return ex.submit(process_scene_1pass, client, sc, prompt,
                             args.frame_dim, args.frame_q, args.crop_dim, args.crop_q,
                             use_tr, use_ocr, use_vad, args.frame_res, args.crop_res,
                             args.res_top, args.top_n, args.max_crops)
        return ex.submit(process_scene, client, sc, prompt, args.crop_dim,
                         args.crop_q, args.frame_dim, args.frame_q,
                         use_tr, use_ocr, use_vad)

    def run_batch(batch, workers):
        got = []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {submit(ex, sc): sc["name"] for sc in batch}
            for fu in cf.as_completed(futs):
                try:
                    got.append(fu.result())
                except Exception as e:
                    log(f"  ✗ {futs[fu]} crashed: {e}")
                    got.append({"scene": futs[fu], "error": str(e)})
        return got

    def complete(r):
        if one:
            return bool((r.get("pass1") or {}).get("ok") and r.get("merged"))
        return (r.get("pass1") or {}).get("ok") and (
            (r.get("pass2") or {}).get("ok") or r.get("n_crops") == 0)

    best = {}
    for rnd in range(args.rounds):
        pending = [sc for sc in scenes if not complete(best.get(sc["name"], {}))]
        if not pending:
            break
        # 33-44MB payloads all go through GCS; drops are random, so back off the
        # concurrency each round instead of hammering at the same rate.
        workers = max(1, args.workers - rnd)
        if rnd:
            log(f"\n--- round {rnd+1}: {len(pending)} scene(s) left, {workers} worker(s) ---")
        for r in run_batch(pending, workers):
            cur = best.get(r["scene"])
            if cur is None or (complete(r) and not complete(cur)):
                best[r["scene"]] = r
    results = list(best.values())

    if args.merge:
        base = json.loads(Path(args.merge).read_text())
        new = {r["scene"]: r for r in results}
        out = []
        for r in base:
            cand = new.pop(r["scene"], None)
            # keep whichever record actually has a successful pass1
            if cand and (cand.get("pass1") or {}).get("ok"):
                out.append(cand)
            else:
                out.append(r)
        out.extend(new.values())
        results = out
        print(f"\nMerged into {Path(args.merge).name}")

    results.sort(key=lambda r: r["scene"])
    tag = time.strftime("%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    f = RESULTS_DIR / f"{'v10_1pass' if one else 'v8_2pass'}_{tag}.json"
    f.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    dt = time.time() - t0
    print(f"\n{'='*74}")
    print(f"Saved: {f}")
    print(f"Wall time: {dt:.0f}s for {len(scenes)} scenes ({dt/max(1,len(scenes)):.0f}s/scene avg)")
    if one:
        print(f"{'Scene':40s} {'OK':>4s} {'Seg':>4s} {'Zoom':>5s} {'Text':>5s} "
              f"{'ImgTok':>7s} {'In':>7s} {'Tport':>6s}")
        ok1 = 0
        for r in results:
            p1 = r.get("pass1", {})
            ok1 += 1 if complete(r) else 0
            print(f"  {r['scene']:40s} {'✓' if complete(r) else '✗':>4s} "
                  f"{p1.get('n_segments', 0):4d} {r.get('n_crops', 0):5d} "
                  f"{p1.get('n_text_items', 0):5d} {r.get('est_image_tokens', 0):7d} "
                  f"{p1.get('in') or 0:7d} {p1.get('transport', '-'):>6s}")
        print(f"\nComplete: {ok1}/{len(results)}")
    else:
        print(f"{'Scene':40s} {'P1':>5s} {'Seg':>4s} {'Reg':>4s} {'Crop':>5s} {'Text':>5s} {'Tport':>6s}")
        ok1 = ok2 = 0
        for r in results:
            p1, p2 = r.get("pass1", {}), r.get("pass2", {})
            ok1 += 1 if p1.get("ok") else 0
            ok2 += 1 if p2.get("ok") else 0
            print(f"  {r['scene']:40s} "
                  f"{'✓' if p1.get('ok') else '✗':>5s} "
                  f"{p1.get('n_segments', 0):4d} {p1.get('n_regions', 0):4d} "
                  f"{r.get('n_crops', 0):5d} {p2.get('n_text_items', 0):5d} "
                  f"{p1.get('transport', '-'):>6s}")
        print(f"\nPass1 ok: {ok1}/{len(results)}   Pass2 ok: {ok2}/{len(results)}")
    def tok(r, p, k):
        return (r.get(p) or {}).get(k) or 0
    tot_in = sum(tok(r, "pass1", "in") + tok(r, "pass2", "in") for r in results)
    tot_out = sum(tok(r, "pass1", "out") + tok(r, "pass2", "out") for r in results)
    print(f"Tokens: in={tot_in:,} out={tot_out:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
