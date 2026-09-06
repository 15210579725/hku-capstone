"""HF single-pass caption path: medium full frames plus medium gaze crops."""
from __future__ import annotations

import io, json, re, time
from PIL import Image
from google import genai
from google.genai import types
import caption_core as cc

MEDIUM = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM
FRAME_DIM, FRAME_Q = 1440, 60
CROP_DIM, CROP_Q = 1440, 70
MIN_OCR_CHARS = 40

ZOOM_PROMPT = """
## ZOOM CROPS
The request contains interleaved ZOOM parts, each from the exact timestamp shown.
The full frames are for action and spatial context; use the zoom crop as the reliable
source for readable on-screen text. Transcribe every meaningful visible line, menu item,
price, code line, message, tab, breadcrumb, button and dialog option. Preserve the original
language and mask personal names, schools, emails, phone numbers, addresses and credentials.
Return a top-level screen_text array with one object per zoom, each containing zoom_index,
hkt, label, text (verbatim lines), and summary. Keep environment/action/details complete.
"""

def _crop(frame: bytes, gaze: dict | None) -> bytes:
    im = Image.open(io.BytesIO(frame)).convert("RGB")
    w, h = im.size
    if gaze:
        try:
            gx, gy = float(gaze.get("gaze_x")), float(gaze.get("gaze_y"))
            # gaze coordinates are native 2880-space in the tar metadata
            cx, cy = gx * w / 2880.0, gy * h / 2880.0
        except (TypeError, ValueError):
            cx, cy = w / 2, h / 2
    else:
        cx, cy = w / 2, h / 2
    side = int(min(w, h) * 0.62)
    l = max(0, min(w - side, int(cx - side / 2)))
    t = max(0, min(h - side, int(cy - side / 2)))
    out = io.BytesIO()
    im.crop((l, t, l + side, t + side)).resize((CROP_DIM, CROP_DIM), Image.LANCZOS).save(
        out, format="JPEG", quality=CROP_Q)
    return out.getvalue()

def _json(text):
    s = text or ""
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m: s = m.group(1)
    i = s.find("{")
    if i >= 0: s = s[i:]
    try: return json.loads(s)
    except Exception: return None

def process_clip(client, sd: dict, system_prompt: str, model: str, gaze_rows: dict,
                 log=print) -> dict:
    parts = [types.Part.from_text(text=system_prompt + "\n\n" + ZOOM_PROMPT + "\n\n" +
                                  cc.build_context_text(sd))]
    zooms = []
    fallback_zooms = 0
    for i, frame in enumerate(sd["frames"]):
        parts.append(types.Part.from_text(text=f"[frame_index {i} — {frame['hkt']} HKT]"))
        parts.append(types.Part.from_bytes(data=frame["jpg"], mime_type="image/jpeg",
                                           media_resolution=MEDIUM))
        ocr = next((x for x in sd["ocr_texts"] if f"{frame['hkt']}" in x), "")
        has_text = len(re.sub(r"\s", "", ocr)) >= MIN_OCR_CHARS
        # Some tar layouts place OCR at the beginning rather than the tail. Keep
        # the one-pass path useful in that layout by sampling a few gaze crops.
        use_fallback = not sd["ocr_texts"] and fallback_zooms < 4
        if has_text or use_fallback:
            z = _crop(frame["jpg"], gaze_rows.get(frame["name"]))
            zi = len(zooms); zooms.append((i, frame["hkt"]))
            fallback_zooms += int(use_fallback)
            parts.append(types.Part.from_text(text=f"[ZOOM index {zi} — frame_index {i} — {frame['hkt']} HKT]"))
            parts.append(types.Part.from_bytes(data=z, mime_type="image/jpeg",
                                               media_resolution=MEDIUM))
    parts.append(types.Part.from_text(text="Now return only the dense caption JSON with absolute HKT timestamps."))
    t0 = time.time()
    try:
        r = client.models.generate_content(model=model, contents=parts,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=32768))
        u = r.usage_metadata
        content = r.text or ""
        parsed = _json(content)
        return {"ok": parsed is not None, "model": model, "content": content,
                "parsed": parsed, "zooms": len(zooms),
                "in": u.prompt_token_count or 0, "out": u.candidates_token_count or 0,
                "think": getattr(u, "thoughts_token_count", 0) or 0,
                "time": round(time.time()-t0, 1), "media_resolution": "MEDIUM"}
    except Exception as e:
        return {"ok": False, "model": model, "error": str(e)[:600], "zooms": len(zooms),
                "time": round(time.time()-t0, 1), "media_resolution": "MEDIUM"}
