#!/usr/bin/env python3
"""审核视频：1440px 带注视圈的帧 + 字幕面板 + 归一化音频。

布局/配色/编码参数沿用 ../render_caption_video.py（Pillow 逐帧画面板、`-g 1
-tune stillimage` 消除 1fps H.264 块状噪点），只是把写死的 scenes 目录换成
pull 下来的 clip 包，并按 1440px 等比缩掉一半字号。

被 pipeline.py 的 render 子命令调用，也可以单独跑：
  python render_review.py --rec <rec> --sample 8
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

VIDEO_WIDTH = 1440
PANEL_HEIGHT = 470
PANEL_BG = (26, 26, 34)
TEXT_COLOR = (230, 230, 235)
MUTED_COLOR = (160, 158, 168)
ACCENT = (27, 158, 151)
SPEECH_COLOR = (255, 220, 130)

FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def get_font(size):
    for fp in FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


F_LABEL, F_TIME, F_ACTION = get_font(34), get_font(25), get_font(28)
F_DETAIL, F_OBJ = get_font(24), get_font(23)

TR_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})\s*[–\-~]\s*(\d{1,2}):(\d{2}):(\d{2})")


def secs(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def seg_span(seg):
    m = TR_RE.match((seg.get("time_range") or "").strip())
    if not m:
        m2 = re.match(r"(\d{1,2}):(\d{2}):(\d{2})", (seg.get("time_range") or "").strip())
        if not m2:
            return None
        a = secs(*m2.groups())
        return a, a + 3
    return secs(*m.groups()[:3]), secs(*m.groups()[3:])


def wrap(text, font, max_w):
    if not text:
        return []
    lines, cur = [], ""
    for tok in re.findall(r"\S+\s*", str(text)):
        test = cur + tok
        if font.getbbox(test.strip())[2] > max_w and cur:
            lines.append(cur.strip())
            cur = tok
        else:
            cur = test
    if cur.strip():
        lines.append(cur.strip())
    return lines


def draw_panel(seg, header, w, h):
    panel = Image.new("RGB", (w, h), PANEL_BG)
    d = ImageDraw.Draw(panel)
    pad = 26
    maxw = w - pad * 2
    y = 14
    d.text((pad, y), header, font=F_LABEL, fill=ACCENT)
    y += 46
    if not seg:
        d.text((pad, y), "(这一秒没有对应 segment)", font=F_DETAIL, fill=MUTED_COLOR)
        return panel
    d.text((pad, y), seg.get("time_range", ""), font=F_TIME, fill=ACCENT)
    y += 34
    for line in wrap(seg.get("action", ""), F_ACTION, maxw):
        if y > h - 40:
            break
        d.text((pad, y), line, font=F_ACTION, fill=TEXT_COLOR)
        y += 34
    y += 6
    if seg.get("speech") and y < h - 110:
        for line in wrap(f'"{seg["speech"]}"', F_DETAIL, maxw):
            if y > h - 84:
                break
            d.text((pad, y), line, font=F_DETAIL, fill=SPEECH_COLOR)
            y += 30
        y += 4
    if seg.get("environment") and y < h - 84:
        for line in wrap("环境: " + str(seg["environment"])[:220], F_DETAIL, maxw):
            if y > h - 56:
                break
            d.text((pad, y), line, font=F_DETAIL, fill=MUTED_COLOR)
            y += 30
    tv = seg.get("text_visible") or []
    if tv and y < h - 56:
        s = " · ".join(tv[:5]) if isinstance(tv, list) else str(tv)
        for line in wrap("文字: " + s, F_OBJ, maxw):
            if y > h - 28:
                break
            d.text((pad, y), line, font=F_OBJ, fill=(120, 200, 195))
            y += 29
    ob = seg.get("objects") or []
    if ob and y < h - 28:
        s = " · ".join(ob[:6]) if isinstance(ob, list) else str(ob)
        for line in wrap("物体: " + s, F_OBJ, maxw):
            if y > h - 6:
                break
            d.text((pad, y), line, font=F_OBJ, fill=(150, 150, 160))
            y += 29
    return panel


def normalize_audio(src: Path, dst: Path):
    r = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-af",
                        "loudnorm=I=-14:TP=-1:LRA=11:print_format=json",
                        "-f", "null", "/dev/null"], capture_output=True, text=True)
    try:
        e = r.stderr
        stats = json.loads(e[e.rfind("{"):e.rfind("}") + 1])
        if stats.get("input_i") in ("-inf", "inf") or stats.get("target_offset") in ("-inf", "inf"):
            raise ValueError("silent")
        af = ("loudnorm=I=-14:TP=-1:LRA=11:"
              f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
              f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
              f"offset={stats['target_offset']}:linear=true")
        cmd = ["ffmpeg", "-y", "-i", str(src), "-af", af, "-ar", "48000", str(dst)]
    except Exception:                                          # noqa: BLE001
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "48000", str(dst)]
    subprocess.run(cmd, capture_output=True, check=True)


def render(args):
    rec = args.rec
    date = args.date
    if not date:
        cands = [p for p in OUT.glob("*/" + rec) if p.is_dir()]
        if not cands:
            sys.exit(f"找不到 out/*/{rec}，先 `pipeline.py pull --rec {rec} --sample N`")
        date = cands[0].parent.name
    recdir = OUT / date / rec
    full = json.loads((recdir / "captions_full.json").read_text())
    segs_by_clip = {}
    for s in full["segments"]:
        segs_by_clip.setdefault(s.get("clip_id"), []).append(s)

    clip_tars = sorted((recdir / "clips").glob("clip_*.tar"))
    if args.clips:
        keep = {int(x) for x in args.clips.split(",")}
        clip_tars = [p for p in clip_tars if int(p.stem.split("_")[1]) in keep]
    if not clip_tars:
        sys.exit(f"{recdir}/clips 里没有 clip 包，先 `pipeline.py pull --rec {rec} --sample N`")
    if args.sample and len(clip_tars) > args.sample:
        step = max(1, len(clip_tars) // args.sample)
        clip_tars = clip_tars[::step][:args.sample]

    out_path = Path(args.output) if args.output else (
        recdir / f"review_{rec[:24]}_{len(clip_tars)}clips.mp4")
    print(f"渲染 {len(clip_tars)} 个 clip → {out_path.name}")

    with tempfile.TemporaryDirectory(prefix="povreview_") as td:
        td = Path(td)
        comp = td / "comp"
        comp.mkdir()
        pcm = bytearray()
        n = 0
        for ct in clip_tars:
            cid = int(ct.stem.split("_")[1])
            segs = sorted(segs_by_clip.get(cid, []), key=lambda s: seg_span(s) or (0, 0))
            with tarfile.open(ct) as tf:
                names = sorted(m.name for m in tf.getmembers()
                               if m.name.startswith("frames/") and m.name.endswith(".jpg"))
                a = next((m for m in tf.getmembers() if m.name == "audio_16k.wav"), None)
                if a:
                    with wave.open(io.BytesIO(tf.extractfile(a).read())) as w:
                        pcm += w.readframes(w.getnframes())
                base = None
                for fn in names:
                    img = Image.open(io.BytesIO(tf.extractfile(fn).read())).convert("RGB")
                    if img.width != VIDEO_WIDTH:
                        img = img.resize((VIDEO_WIDTH,
                                          int(img.height * VIDEO_WIDTH / img.width)),
                                         Image.LANCZOS)
                    m = re.search(r"_(\d{2})-(\d{2})-(\d{2})_", fn)
                    t = secs(*m.groups()) if m else 0
                    if base is None:
                        base = t
                    seg = next((s for s in segs
                                if (sp := seg_span(s)) and sp[0] <= t < sp[1]), None)
                    if seg is None and segs:
                        seg = min(segs, key=lambda s: abs((seg_span(s) or (t, t))[0] - t))
                    hdr = f"clip_{cid:04d}   {m.group(1)}:{m.group(2)}:{m.group(3)} HKT" if m \
                        else f"clip_{cid:04d}"
                    panel = draw_panel(seg, hdr, VIDEO_WIDTH, PANEL_HEIGHT)
                    th = img.height + PANEL_HEIGHT
                    if th % 2:
                        th += 1
                    cv = Image.new("RGB", (VIDEO_WIDTH, th), PANEL_BG)
                    cv.paste(img, (0, 0))
                    cv.paste(panel, (0, img.height))
                    cv.save(comp / f"f_{n:06d}.jpg", quality=88)
                    n += 1
            print(f"  clip_{cid:04d} ✓ ({len(names)} 帧)", flush=True)

        raw = td / "raw.wav"
        with wave.open(str(raw), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(bytes(pcm))
        norm = td / "norm.wav"
        try:
            normalize_audio(raw, norm)
        except Exception:                                      # noqa: BLE001
            norm = raw

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-framerate", "1", "-i", str(comp / "f_%06d.jpg"),
               "-i", str(norm),
               "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-g", "1", "-tune", "stillimage",
               "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit("ffmpeg 失败")
    print(f"✓ {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {n} 帧)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rec", required=True)
    p.add_argument("--date")
    p.add_argument("--sample", type=int, default=8)
    p.add_argument("--clips")
    p.add_argument("--output")
    render(p.parse_args())
