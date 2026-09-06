#!/usr/bin/env python3
"""帧渲染 + context 组装 + Gemini 调用。

context 结构与 ../run_caption.py 的 v6 保持一字不差（Scene 头 / Audio Transcript /
OCR 块 / 逐帧 [HKT — filename] / 收尾指令），换的只是数据来源。

注视圈参数照抄 ../mimo_v25_audit_30x30s_20260821_COMPLETE/_tools/render_gaze_overlay.py，
保证和 caption/ 目录已有的 gaze_overlay 产物一致。区别只在于：那边先在 2880² 上画圈、
再由 run_caption.py 压到 1440；这边直接在 1440 上按 scale=1440/2880 等比画，
省掉一次全分辨率解码，模型看到的圈相对位置与相对大小相同。
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from datetime import datetime, timedelta

from PIL import Image, ImageDraw, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- 与 _tools/render_gaze_overlay.py 一致 -------------------------------
GAZE_COLOR = (0, 255, 128)          # 当前帧注视点
GAZE_INVALID_COLOR = (255, 60, 60)  # 无效注视
TRAIL_COLOR = (255, 200, 0)         # 轨迹线
POINT_RADIUS = 18                   # @2880
TRAIL_POINTS = 5
NATIVE_DIM = 2880

# ---- 与 ../run_caption.py 一致 -------------------------------------------
MAX_DIM = 1440
JPEG_QUALITY = 60
ADC_PROJECT = (os.environ.get("ADC_PROJECT")
               or os.environ.get("GOOGLE_CLOUD_PROJECT")
               or "project-0e21c343-2d89-405c-9d5")
ADC_LOCATION = "global"
DEFAULT_MODEL = "gemini-3.7-flash"
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.3

# inline 上限 20MB，留安全余量
INLINE_LIMIT = 18 << 20
DOWNGRADE_DIMS = [1440, 1200, 1024]

FRAME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_[A-Z]{2,4}__frame_(\d+)\.(?:jpg|jpeg|png|txt)$",
    re.I)
# 时区标记不是只有 HKT：内地录制写 BJ（帧名后缀同理，见 FRAME_RE）。写死 HKT 会让
# 整条 transcript 一行都匹配不上、被静默丢掉 —— 4-27_hkt0930 冒烟实测 transcript=0 区间。
# 同一个坑在 quality_verify/select/scripts/verify_frames.py 里也记过（20/117 误判 FAIL）。
TR_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [A-Z]{2,4})? -> "
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [A-Z]{2,4})?\]\s*(.*)$")
# 布局 B 的相对偏移格式：[00:00:00-00:00:30] 或 [00:00-00:30]
TR_REL_RE = re.compile(
    r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[-–—]\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*(.*)$")
TR_REL_BASE = datetime(2000, 1, 1)

OCR_SKIP_PREFIX = ("capture_time", "capture_epoch", "frame_index")


# ---------------------------------------------------------------- 时间解析

def frame_time(name: str):
    m = FRAME_RE.search(os.path.basename(name))
    if not m:
        return None
    return datetime.strptime(
        f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S")


def frame_index(name: str):
    m = FRAME_RE.search(os.path.basename(name))
    return int(m.group(5)) if m else None


def hkt_hms(name: str) -> str:
    t = frame_time(name)
    return t.strftime("%H:%M:%S") if t else ""


# ---------------------------------------------------------------- 帧渲染

def build_trails(gaze_rows: dict):
    """预先算出每帧要画的轨迹点（@2880 坐标），让渲染与帧顺序无关。

    tar 里 masked 帧是**时间倒序**的，而 _tools/render_gaze_overlay.py 的轨迹依赖
    正序历史。先按 frame_index 升序把历史算好，渲染就能乱序并行。
    语义与原脚本一致：只有 valid 帧入历史，invalid 帧不清空历史。

    返回 {frame_file: (points, valid)}；points 含当前点，最多 TRAIL_POINTS+1 个。
    """
    out = {}
    hist = []
    for name in sorted(gaze_rows, key=lambda n: (frame_index(n) if frame_index(n) is not None else 0)):
        r = gaze_rows[name]
        valid = bool(r.get("gaze_x") and r.get("in_bounds") == "True")
        if valid:
            hist.append((float(r["gaze_x"]), float(r["gaze_y"])))
            if len(hist) > TRAIL_POINTS + 1:
                del hist[:-(TRAIL_POINTS + 1)]
        out[name] = (list(hist) if valid else [], valid)
    return out


def render_frame(jpg_bytes: bytes, points, valid: bool,
                 size: int = MAX_DIM, quality: int = JPEG_QUALITY,
                 as_image: bool = False):
    """解码 → 降到 size → 叠注视圈 → JPEG。

    points 为 None 表示这条录制根本没有眼动数据（不画任何标记）。
    """
    im = Image.open(io.BytesIO(jpg_bytes))
    im.draft("RGB", (size, size))          # libjpeg 在 DCT 域降采样
    im = im.convert("RGB")
    if max(im.size) != size:
        im = im.resize((size, size), Image.LANCZOS)

    scale = im.width / NATIVE_DIM
    draw = ImageDraw.Draw(im)

    if valid and points:
        pts = [(x * scale, y * scale) for x, y in points]
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                alpha = (i + 1) / len(pts)
                w = max(1, round(6 * alpha * scale))
                c = tuple(int(v * alpha) for v in TRAIL_COLOR)
                draw.line([pts[i], pts[i + 1]], fill=c, width=w)
        cx, cy = pts[-1]
        r = POINT_RADIUS * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=GAZE_COLOR, width=max(1, round(3 * scale)))
        d = 4 * scale
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], fill=GAZE_COLOR)
    elif points is not None:
        a, b = 10 * scale, 50 * scale
        w = max(1, round(3 * scale))
        draw.rectangle([a, a, b, b], outline=GAZE_INVALID_COLOR, width=w)
        draw.line([a, a, b, b], fill=GAZE_INVALID_COLOR, width=max(1, round(2 * scale)))
        draw.line([b, a, a, b], fill=GAZE_INVALID_COLOR, width=max(1, round(2 * scale)))

    if as_image:
        # pass2 的 crop 要在原分辨率上裁，编码再解码一遍纯属浪费；直接把图给出去。
        return im
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def recompress(jpg_bytes: bytes, size: int, quality: int) -> bytes:
    im = Image.open(io.BytesIO(jpg_bytes))
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    if max(im.size) != size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


# ---------------------------------------------------------------- OCR / transcript

def clean_ocr(text: str) -> str:
    lines = [l for l in text.split("\n") if not l.startswith(OCR_SKIP_PREFIX)]
    return "\n".join(lines).strip()


def parse_transcript(text: str):
    """→ [(start_dt, end_dt, 原始行)]

    transcript 有**两种**格式，跟 tar 布局对应（见 CLAUDE.md「tar 内部布局」）：
      A 绝对时钟  [2026-05-17 21:30:00 HKT -> 2026-05-17 21:30:30 HKT] 文本
      B 相对偏移  [00:00:00-00:00:30] 文本        ← 无日期、无时区
    只认 A 会让 B 的整条 transcript 一行都匹配不上、被静默丢成 0 区间
    （4-27_hkt0930 冒烟实测）。B 用一个固定基准日期合成 datetime 就行 ——
    下游 transcript_for_window() 本来就只用相对偏移对齐，绝对值不参与计算。
    """
    out = []
    for line in text.splitlines():
        m = TR_RE.match(line)
        if m:
            a = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            b = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S")
            out.append((a, b, line))
            continue
        m = TR_REL_RE.match(line)
        if m:
            def _sec(h, mi, s):
                return int(h) * 3600 + int(mi) * 60 + int(s or 0)
            a = TR_REL_BASE + timedelta(seconds=_sec(*m.group(1, 2, 3)))
            b = TR_REL_BASE + timedelta(seconds=_sec(*m.group(4, 5, 6)))
            out.append((a, b, line))
    return out


def transcript_for_window(intervals, rel_start: float, dur: float):
    """按**相对偏移**对齐（首个区间为基准），不用绝对时钟。

    _tools/extract_30s_from_masked_tar.py 踩过这个坑：legacy 包的 masked 绝对时钟
    会整体平移，但帧/音频的相对偏移是对的。
    """
    if not intervals:
        return []
    base = intervals[0][0]
    w0 = base + timedelta(seconds=rel_start)
    w1 = w0 + timedelta(seconds=dur)
    return [line for a, b, line in intervals if a < w1 and b > w0]


# ---------------------------------------------------------------- context

def build_context_text(sd: dict) -> str:
    """与 ../run_caption.py 的 build_context_text 完全一致。"""
    first_hkt = sd["frames"][0]["hkt"] if sd["frames"] else "?"
    last_hkt = sd["frames"][-1]["hkt"] if sd["frames"] else "?"

    ocr_block = "\n".join(sd["ocr_texts"]) if sd["ocr_texts"] else "(no text detected)"
    if len(ocr_block) > 30000:
        ocr_block = ocr_block[:30000] + "\n... (OCR truncated, remaining frames omitted)"

    return f"""## Scene: 30-second first-person video clip
- Date: {sd['clip_start'][:10] if sd['clip_start'] else 'unknown'}
- Absolute time range: {sd['clip_start']} to {sd['clip_end']} HKT
- Frame count: {sd['n_frames']} frames at 1fps
- Frame timestamps: {first_hkt} to {last_hkt} HKT
- IMPORTANT: Use these absolute HKT timestamps (HH:MM:SS format) for all time_range fields

## Audio Transcript
{sd['transcript']}

## OCR-detected Text (per-frame, with HKT timestamps)
{ocr_block}

## Video Frames (all {sd['n_frames']} frames, 1 second apart, chronologically ordered)"""


def build_parts_vertex(sd: dict, system_prompt: str):
    """与 ../run_caption.py 的 build_parts_vertex 完全一致。"""
    from google import genai
    context = build_context_text(sd)
    parts = [genai.types.Part.from_text(text=system_prompt + "\n\n" + context)]
    for frame in sd["frames"]:
        parts.append(genai.types.Part.from_text(text=f"[{frame['hkt']} HKT — {frame['name']}]"))
        parts.append(genai.types.Part.from_bytes(data=frame["jpg"], mime_type="image/jpeg"))
    parts.append(genai.types.Part.from_text(
        text="Now produce the fine-grained, first-person dense temporal caption JSON with absolute HKT timestamps."))
    return parts


def payload_bytes(sd: dict) -> int:
    return sum(len(f["jpg"]) for f in sd["frames"])


def enforce_inline_limit(sd: dict, limit: int = INLINE_LIMIT, log=print):
    """20MB inline 守卫：超限就整段降分辨率重压。返回实际用的最长边。"""
    dim = DOWNGRADE_DIMS[0]
    for d in DOWNGRADE_DIMS[1:]:
        if payload_bytes(sd) <= limit:
            break
        dim = d
        log(f"    payload {payload_bytes(sd)/1e6:.1f}MB > 上限，降到 {d}px 重压")
        for f in sd["frames"]:
            f["jpg"] = recompress(f["jpg"], d, JPEG_QUALITY)
    return dim


# ---------------------------------------------------------------- Gemini

def make_client(project: str = ADC_PROJECT, location: str = ADC_LOCATION):
    from google import genai
    return genai.Client(vertexai=True, project=project, location=location)


def run_one(client, model: str, parts, retries: int = 3, log=print) -> dict:
    """与 ../run_caption.py 的 run_one_vertex 同参数，加了退避重试。"""
    from google import genai
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=model, contents=parts,
                config=genai.types.GenerateContentConfig(
                    temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS),
            )
            u = resp.usage_metadata
            return {
                "ok": True, "model": model, "api": "adc",
                "time": round(time.time() - t0, 1),
                "in": u.prompt_token_count or 0,
                "out": u.candidates_token_count or 0,
                "think": getattr(u, "thoughts_token_count", 0) or 0,
                "content": resp.text or "",
                "attempts": attempt + 1,
            }
        except Exception as e:                              # noqa: BLE001
            last_err = e
            msg = str(e)
            transient = any(s in msg for s in
                            ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED",
                             "UNAVAILABLE", "DEADLINE", "Timeout", "timed out",
                             "Connection", "reset"))
            if attempt == retries - 1 or not transient:
                break
            wait = min(2 ** attempt * 5, 60)
            log(f"    重试 {attempt+1}/{retries} ({msg[:80]}) {wait}s 后")
            time.sleep(wait)
    return {"ok": False, "model": model, "api": "adc", "error": str(last_err)[:600]}


# ---------------------------------------------------------------- JSON 解析

def parse_caption_json(text: str):
    """剥 ``` 围栏 + 截断补救。返回 (obj, note)。"""
    if not text:
        return None, "empty"
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    i = s.find("{")
    if i > 0:
        s = s[i:]
    try:
        return json.loads(s), "ok"
    except json.JSONDecodeError:
        pass
    # 截断补救：砍到最后一个完整 segment，再补上结构
    seg = s.find('"segments"')
    if seg >= 0:
        last = s.rfind("}")
        while last > seg:
            cand = s[:last + 1]
            for tail in ("]}", "}]}", '"}]}'):
                try:
                    return json.loads(cand + tail), "salvaged"
                except json.JSONDecodeError:
                    continue
            last = s.rfind("}", 0, last)
    return None, "unparseable"
