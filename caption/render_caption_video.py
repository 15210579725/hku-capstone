#!/usr/bin/env python3
"""Render caption review videos: scene frames + normalized audio + model caption overlays.

Usage:
  python render_caption_video.py S01_atomic_action \
    -c "Flash #1B9E97:gemini_test_results/s01_rerun.json" \
    -c "Pro #D4923A:gemini_test_results/pro_retry_compressed.json"

  # Batch all 3 test scenes:
  python render_caption_video.py --batch

Subtitle panel rendered via Pillow (no libass needed).
Re-run with different --caption args to swap models — video frames are not modified.
Audio loudness-normalized to -14 LUFS (EBU R128 two-pass).
"""

import argparse, json, os, re, subprocess, sys, tempfile, textwrap, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCENES_ROOT = Path("/Users/mac/Desktop/hku capstone/caption/mimo_v25_audit_30x30s_20260821_COMPLETE/scenes")
OUTPUT_DIR = Path("/Users/mac/Desktop/hku capstone/caption/caption_review_videos")
RESULTS_DIR = Path("/Users/mac/Desktop/hku capstone/caption/gemini_test_results")

DEFAULT_COLORS = ["#1B9E97", "#D4923A", "#7B68EE", "#E06060"]
PANEL_BG = (26, 26, 34)
PANEL_HEIGHT = 1180
VIDEO_WIDTH = 2880
TEXT_COLOR = (230, 230, 235)
MUTED_COLOR = (160, 158, 168)

FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for fp in FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


FONT_LABEL = get_font(67, bold=True)
FONT_TIME = get_font(50, bold=True)
FONT_ACTION = get_font(56, bold=True)
FONT_DETAIL = get_font(48)
FONT_OBJ = get_font(45)
FONT_TEXTVIS = get_font(45)
FONT_ENV = get_font(38)
FONT_ENV_HEAD = get_font(42, bold=True)


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def find_scene_subdir(scene_name: str) -> Path:
    scene_dir = SCENES_ROOT / scene_name
    return next(d for d in scene_dir.iterdir() if d.is_dir())


def parse_caption_arg(arg: str) -> dict:
    parts = arg.split(":", 1)
    if len(parts) == 1:
        return {"label": Path(parts[0]).stem, "color": None, "path": parts[0]}
    label_part, path = parts
    tokens = label_part.strip().split()
    color = None
    label_tokens = []
    for t in tokens:
        if t.startswith("#") and len(t) in (4, 7):
            color = t
        else:
            label_tokens.append(t)
    return {"label": " ".join(label_tokens) or Path(path).stem, "color": color, "path": path.strip()}


def _v7_entry(data, scene_name):
    """Return the v7 two-pass record for a scene, or None if this isn't a v7 file."""
    if not isinstance(data, list) or not data:
        return None
    if "pass1" not in data[0] and "merged" not in data[0]:
        return None
    for item in data:
        sc = item.get("scene", "")
        if scene_name in sc or sc in scene_name:
            return item
    return None


def _v7_captions(entry, want: str) -> dict:
    """Extract a caption dict from a v7 record.

    want='merged'  -> pass-1 caption with screen_text_detail attached
    want='screen'  -> pass-2 crop readings rendered as pseudo-segments
    """
    if want == "screen":
        raw = (entry.get("pass2") or {}).get("content", "")
        parsed = _loads_loose(raw) or {}

        # Group crops by timestamp, then hold each reading until the next one
        # arrives — a crop is a point sample, but the screen it shows persists.
        by_t = {}
        for c in parsed.get("crops", []):
            h = (c.get("hkt") or "").strip()
            if not re.match(r"^\d{1,2}:\d{2}:\d{2}$", h):
                continue
            by_t.setdefault(h, []).append(c)

        stamps = sorted(by_t)
        clip_end = (entry.get("clip_end") or "")[-8:]
        segs = []
        for i, h in enumerate(stamps):
            nxt = stamps[i + 1] if i + 1 < len(stamps) else (clip_end or h)
            if nxt <= h:  # last crop: hold for one second minimum
                hh, mm, ss = (int(x) for x in h.split(":"))
                nxt = time.strftime("%H:%M:%S", time.gmtime(hh * 3600 + mm * 60 + ss + 1))
            group = by_t[h]
            lines, labels = [], []
            for c in group:
                lab = c.get("label", "screen")
                labels.append(lab)
                summ = c.get("summary")
                if summ:
                    lines.append(f"[{lab}] {summ}")
                lines.extend(c.get("text", []) or [])
            segs.append({
                "time_range": f"{h}–{nxt}",
                "action": " / ".join(dict.fromkeys(labels)),
                "objects": list(dict.fromkeys(labels)),
                "environment": None,
                "text_visible": lines,
                "speech": None,
                "details": "",
            })
        return {"scene_summary": f"Pass 2 · {len(segs)} crop readings (ULTRA_HIGH)",
                "segments": segs, "activity_chain": ""}

    merged = entry.get("merged")
    if isinstance(merged, dict) and merged.get("segments"):
        return merged
    raw = (entry.get("pass1") or {}).get("content", "")
    return _loads_loose(raw) or {"scene_summary": "", "segments": [], "activity_chain": ""}


def _loads_loose(text: str):
    if not text:
        return None
    t = text
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0]
    elif "```" in t:
        t = t.split("```")[1].split("```")[0]
    try:
        return json.loads(t.strip())
    except (json.JSONDecodeError, AttributeError):
        pass
    i = t.rfind("}")
    if i > 0:
        for sfx in ("", "]}", "}]}", "\"}]}"):
            try:
                return json.loads(t[:i + 1].strip() + sfx)
            except (json.JSONDecodeError, AttributeError):
                continue
    return None


def load_captions(path: str, scene_name: str, label: str = "") -> dict:
    with open(path) as f:
        data = json.load(f)

    entry = _v7_entry(data, scene_name)
    if entry is not None:
        want = "screen" if "screen" in label.lower() or "pass2" in label.lower() else "merged"
        return _v7_captions(entry, want)

    label_lower = label.lower()
    model_hint = ""
    if "flash" in label_lower:
        model_hint = "flash"
    elif "pro" in label_lower:
        model_hint = "pro"

    text = ""
    if isinstance(data, list):
        for item in data:
            scene = item.get("scene", "")
            model = item.get("model", "")
            scene_match = scene_name in scene or scene in scene_name
            model_match = not model_hint or model_hint in model.lower()
            if scene_match and model_match:
                text = item.get("content", "")
                break
        if not text:
            for item in data:
                scene = item.get("scene", "")
                if scene_name in scene or scene in scene_name:
                    text = item.get("content", "")
                    break
        if not text and data:
            text = data[0].get("content", "")
    elif isinstance(data, dict):
        text = data.get("content", json.dumps(data, ensure_ascii=False))

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        # Try to salvage truncated JSON by closing brackets
        salvaged = text.strip()
        for closer in (']}"', ']}', ']', '}'):
            try:
                # Find the last complete segment object
                last_brace = salvaged.rfind('}')
                if last_brace > 0:
                    attempt = salvaged[:last_brace+1] + ']}'
                    parsed = json.loads(attempt)
                    if parsed.get("segments"):
                        print(f"    (salvaged {len(parsed['segments'])} segments from truncated JSON)")
                        return parsed
            except (json.JSONDecodeError, AttributeError):
                continue
        return {"scene_summary": str(text)[:200], "segments": [], "activity_chain": ""}


def _parse_time_range(tr: str, base_seconds: float = 0) -> tuple[float, float] | None:
    """Parse time_range supporting both relative (0s-5s) and absolute HKT (17:57:12–17:57:15)."""
    tr = tr.strip()
    # Absolute HKT: "HH:MM:SS–HH:MM:SS" or "HH:MM:SS-HH:MM:SS"
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2}):(\d{2})", tr)
    if m:
        s1 = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        s2 = int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
        return (s1 - base_seconds, s2 - base_seconds)
    # Relative: "0s-5s" or "0-5"
    parts = tr.replace("s", "").split("-")
    try:
        start = float(parts[0])
        end = float(parts[1]) if len(parts) > 1 else 30.0
        return (start, end)
    except (ValueError, IndexError):
        return None


def _compute_base_seconds(segments: list) -> float:
    """Find the earliest absolute timestamp to use as base offset."""
    for seg in segments:
        tr = seg.get("time_range", "")
        m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})", tr)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return 0


def get_active_segment(segments: list, t: float) -> dict | None:
    base = _compute_base_seconds(segments)
    for seg in segments:
        parsed = _parse_time_range(seg.get("time_range", ""), base)
        if parsed is None:
            continue
        start, end = parsed
        if start <= t < end:
            return seg
    return None


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    if not text:
        return []
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


def render_caption_panel(captions_list: list, t: float, panel_w: int, panel_h: int) -> Image.Image:
    """Render the caption panel for time t as a Pillow Image."""
    panel = Image.new("RGB", (panel_w, panel_h), PANEL_BG)
    draw = ImageDraw.Draw(panel)

    n_models = len(captions_list)
    col_w = panel_w // n_models
    padding = 40

    for i, cap in enumerate(captions_list):
        x_start = col_w * i + padding
        max_text_w = col_w - padding * 2
        y = 28
        color = hex_to_rgb(cap["color"])

        # Column separator line
        if i > 0:
            draw.line([(col_w * i, 22), (col_w * i, panel_h - 22)], fill=(50, 48, 58), width=3)

        # Model label
        draw.text((x_start, y), cap["label"].upper(), font=FONT_LABEL, fill=color)
        y += 90

        seg = get_active_segment(cap["data"].get("segments", []), t)
        if not seg:
            draw.text((x_start, y), "(no segment)", font=FONT_DETAIL, fill=MUTED_COLOR)
            continue

        # Time range
        tr = seg.get("time_range", "")
        draw.text((x_start, y), tr, font=FONT_TIME, fill=color)
        y += 68

        # Dedicated environment panel: the world-state text fills the whole column.
        # It runs long (up to ~1200 chars), so it gets its own small font and, where
        # the model used "Heading: ..." sections, one visual break per section.
        if cap.get("field") == "environment":
            env = str(seg.get("environment") or "").strip()
            if not env:
                draw.text((x_start, y), "(no environment)", font=FONT_DETAIL, fill=MUTED_COLOR)
                continue
            for chunk in re.split(r"(?<=[.。])\s+(?=[A-Z一-鿿][^:：]{2,28}[:：])", env):
                chunk = chunk.strip()
                if not chunk:
                    continue
                m = re.match(r"^([^:：]{2,28}[:：])\s*(.*)$", chunk, re.S)
                head, rest = (m.group(1), m.group(2)) if m else ("", chunk)
                if head:
                    if y > panel_h - 60:
                        break
                    draw.text((x_start, y), head, font=FONT_ENV_HEAD, fill=color)
                    y += 50
                for line in wrap_text(rest, FONT_ENV, max_text_w):
                    if y > panel_h - 46:
                        break
                    draw.text((x_start, y), line, font=FONT_ENV, fill=TEXT_COLOR)
                    y += 46
                y += 8
            continue

        # Action (bold)
        action = seg.get("action", "")
        for line in wrap_text(action, FONT_ACTION, max_text_w):
            draw.text((x_start, y), line, font=FONT_ACTION, fill=TEXT_COLOR)
            y += 68
        y += 12

        # Speech
        speech = seg.get("speech")
        if speech and y < panel_h - 225:
            speech_str = f'"{speech}"'
            for line in wrap_text(speech_str, FONT_DETAIL, max_text_w):
                if y > panel_h - 170:
                    break
                draw.text((x_start, y), line, font=FONT_DETAIL, fill=(255, 220, 130))
                y += 59
            y += 6

        # Details
        details = seg.get("details", "")
        if details and y < panel_h - 170:
            for line in wrap_text(details[:200], FONT_DETAIL, max_text_w):
                if y > panel_h - 112:
                    break
                draw.text((x_start, y), line, font=FONT_DETAIL, fill=MUTED_COLOR)
                y += 59
            y += 8

        # Objects
        objects = seg.get("objects", [])
        if objects and y < panel_h - 112:
            obj_str = " · ".join(objects[:6])
            for line in wrap_text(obj_str, FONT_OBJ, max_text_w):
                if y > panel_h - 68:
                    break
                draw.text((x_start, y), line, font=FONT_OBJ, fill=(color[0]//2+80, color[1]//2+80, color[2]//2+80))
                y += 56
            y += 6

        # Visible text
        text_vis = seg.get("text_visible", [])
        if text_vis and y < panel_h - 68:
            tv_str = " · ".join(text_vis[:5])
            for line in wrap_text(tv_str, FONT_TEXTVIS, max_text_w):
                if y > panel_h - 22:
                    break
                draw.text((x_start, y), line, font=FONT_TEXTVIS, fill=color)
                y += 56

    return panel


def normalize_audio(audio_path: Path, output_path: Path):
    """Two-pass EBU R128 loudness normalization to -14 LUFS."""
    probe_cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json",
        "-f", "null", "/dev/null"
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    stderr = result.stderr

    try:
        json_start = stderr.rfind("{")
        json_end = stderr.rfind("}") + 1
        stats = json.loads(stderr[json_start:json_end])

        if stats.get("input_i") in ("-inf", "inf") or stats.get("target_offset") in ("-inf", "inf"):
            # Silent or near-silent audio — just copy without normalization
            norm_cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "48000", str(output_path)]
        else:
            norm_cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-af", (f"loudnorm=I=-14:TP=-1:LRA=11:"
                        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
                        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
                        f"offset={stats['target_offset']}:linear=true"),
                "-ar", "48000", str(output_path)
            ]
    except (json.JSONDecodeError, KeyError):
        norm_cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-af", "loudnorm=I=-14:TP=-1:LRA=11",
            "-ar", "48000", str(output_path)
        ]

    subprocess.run(norm_cmd, capture_output=True, check=True)


def build_video(scene_name: str, captions_list: list, output_path: Path):
    subdir = find_scene_subdir(scene_name)
    gaze_dir = subdir / "picture" / "gaze_overlay"
    masked_dir = subdir / "picture" / "masked"
    frame_dir = gaze_dir if gaze_dir.exists() else masked_dir
    audio_file = subdir / "audio" / "anonymized.wav"

    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        print(f"  ✗ No frames in {frame_dir}")
        return False

    n_frames = len(frame_files)

    with tempfile.TemporaryDirectory(prefix="caption_rv_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Normalize audio
        print(f"  [1/3] Normalizing audio...", flush=True)
        norm_audio = tmpdir / "norm.wav"
        normalize_audio(audio_file, norm_audio)

        # Render composite frames (video + caption panel)
        print(f"  [2/3] Rendering {n_frames} composite frames...", flush=True)
        composite_dir = tmpdir / "composite"
        composite_dir.mkdir()

        for idx, frame_path in enumerate(frame_files):
            t = float(idx)

            # Load and resize video frame
            img = Image.open(frame_path)
            orig_w, orig_h = img.size
            scale_h = int(orig_h * VIDEO_WIDTH / orig_w)
            if img.size != (VIDEO_WIDTH, scale_h):
                img = img.resize((VIDEO_WIDTH, scale_h), Image.LANCZOS)

            # Render caption panel
            panel = render_caption_panel(captions_list, t, VIDEO_WIDTH, PANEL_HEIGHT)

            # Composite
            total_h = scale_h + PANEL_HEIGHT
            if total_h % 2:
                total_h += 1
            composite = Image.new("RGB", (VIDEO_WIDTH, total_h), PANEL_BG)
            composite.paste(img, (0, 0))
            composite.paste(panel, (0, scale_h))

            composite.save(composite_dir / f"frame_{idx:04d}.jpg", quality=92)

        # Encode with ffmpeg
        print(f"  [3/3] Encoding video...", flush=True)
        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", "1",
            "-i", str(composite_dir / "frame_%04d.jpg"),
            "-i", str(norm_audio),
            "-vf", "format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "1", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path)
        ]
        r = subprocess.run(encode_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ✗ ffmpeg encode failed: {r.stderr[-300:]}")
            return False

    # Verify
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(output_path)]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    dur = float(info["format"]["duration"])
    size_mb = int(info["format"]["size"]) / 1024 / 1024
    print(f"  ✓ {output_path.name}: {dur:.1f}s, {size_mb:.1f}MB")
    return True


def discover_scenes_from_results(results_path: str) -> list[dict]:
    """Build batch config dynamically from a results JSON file."""
    with open(results_path) as f:
        data = json.load(f)

    # v7 two-pass: one panel for the caption, one for the crop text readings
    if data and ("pass1" in data[0] or "merged" in data[0]):
        configs = []
        for item in sorted(data, key=lambda x: x.get("scene", "")):
            if not (item.get("pass1") or {}).get("ok"):
                continue
            caps = [{"label": "Caption", "color": "#1B9E97", "path": results_path}]
            if (item.get("pass2") or {}).get("ok"):
                caps.append({"label": "Screen Text", "color": "#D4923A", "path": results_path})
            configs.append({"scene": item["scene"], "captions": caps})
        return configs

    scene_models = {}
    for item in data:
        if item.get("error"):
            continue
        scene = item["scene"]
        model = item["model"]
        if scene not in scene_models:
            scene_models[scene] = []
        scene_models[scene].append(model)

    configs = []
    for scene in sorted(scene_models):
        captions = []
        for model in scene_models[scene]:
            if "flash" in model.lower():
                captions.append({"label": "Flash", "color": "#1B9E97", "path": results_path})
            elif "pro" in model.lower():
                captions.append({"label": "Pro", "color": "#D4923A", "path": results_path})
            else:
                captions.append({"label": model[:12], "color": DEFAULT_COLORS[len(captions) % len(DEFAULT_COLORS)], "path": results_path})
        configs.append({"scene": scene, "captions": captions})
    return configs


def find_latest_results() -> str:
    """Find the latest results file, newest pipeline version first."""
    for prefix in ("v8_2pass_", "v7_2pass_", "v6_results_", "v4_results_",
                   "v3_results_", "v2_results_"):
        candidates = sorted(RESULTS_DIR.glob(f"{prefix}*.json"), reverse=True)
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError("No results JSON found in gemini_test_results/")


def main():
    parser = argparse.ArgumentParser(description="Render caption review videos")
    parser.add_argument("scene", nargs="?", help="Scene name (e.g. S01_atomic_action)")
    parser.add_argument("--caption", "-c", action="append", help="'label [#color]:path'")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--batch", action="store_true", help="Render all scenes from results")
    parser.add_argument("--results", "-r", help="Results JSON path (default: latest v3)")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT_JSON", "RIGHT_JSON"),
                        help="side-by-side ablation: two v7/v8 result files")
    parser.add_argument("--compare-labels", nargs=2, default=["with transcript+OCR", "vision only"],
                        help="panel labels for --compare")
    parser.add_argument("--compare-field", default=None, choices=("environment",),
                        help="--compare: put this field in both panels instead of the "
                             "action/speech/objects layout")
    parser.add_argument("--env", metavar="RESULTS_JSON",
                        help="environment review: left=action, right=full world-state text")
    parser.add_argument("--suffix", default="_review", help="output filename suffix")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.env:
        path = args.env
        with open(path) as f:
            data = json.load(f)
        scenes = [r["scene"] for r in data if (r.get("pass1") or {}).get("ok") and r.get("merged")]
        print(f"Environment review: {path}\nScenes: {len(scenes)}")
        results = []
        for i, scene in enumerate(scenes):
            print(f"\n[{i+1}/{len(scenes)}] {'='*50}\n{scene}\n{'='*50}")
            caps = []
            for label, color, field in (("ACTION", "#1B9E97", None),
                                        ("ENVIRONMENT (world state)", "#7D9BD4", "environment")):
                d = load_captions(path, scene, "")
                c = {"label": label, "color": color, "path": path, "data": d}
                if field:
                    c["field"] = field
                caps.append(c)
                n = len(d.get("segments", []))
                if field:
                    chars = sum(len(str(s.get("environment") or "")) for s in d.get("segments", []))
                    print(f"  [{label}]: {n} segments, {chars} env chars")
                else:
                    print(f"  [{label}]: {n} segments")
            out = OUTPUT_DIR / f"{scene}{args.suffix}.mp4"
            results.append((scene, build_video(scene, caps, out), out))
        print(f"\n{'='*60}\nENV SUMMARY ({len(results)} scenes)\n{'='*60}")
        for scene, ok, _ in results:
            print(f"  {'✓' if ok else '✗'} {scene}")
        print(f"\n  {sum(1 for _, ok, _ in results if ok)}/{len(results)} succeeded")
        return 0 if all(ok for _, ok, _ in results) else 1

    if args.compare:
        left, right = args.compare
        ll, rl = args.compare_labels
        with open(left) as f:
            ldata = json.load(f)
        lscenes = [r["scene"] for r in ldata if (r.get("pass1") or {}).get("ok")]
        with open(right) as f:
            rscenes = {r["scene"] for r in json.load(f) if (r.get("pass1") or {}).get("ok")}
        scenes = [s for s in lscenes if s in rscenes]
        print(f"Compare: {ll}  vs  {rl}")
        print(f"Scenes in both: {len(scenes)}")
        results = []
        for i, scene in enumerate(scenes):
            print(f"\n[{i+1}/{len(scenes)}] {'='*50}\n{scene}\n{'='*50}")
            caps = []
            for path, label, color in ((left, ll, "#1B9E97"), (right, rl, "#D4923A")):
                d = load_captions(path, scene, label)
                c = {"label": label, "color": color, "path": path, "data": d}
                if args.compare_field:
                    c["field"] = args.compare_field
                caps.append(c)
                segs = d.get("segments", [])
                if args.compare_field == "environment":
                    chars = sum(len(str(x.get("environment") or "")) for x in segs)
                    print(f"  [{label}]: {len(segs)} segments, {chars} env chars")
                else:
                    print(f"  [{label}]: {len(segs)} segments")
            out = OUTPUT_DIR / f"{scene}{args.suffix}.mp4"
            results.append((scene, build_video(scene, caps, out), out))
        print(f"\n{'='*60}\nCOMPARE SUMMARY ({len(results)} scenes)\n{'='*60}")
        for scene, ok, _ in results:
            print(f"  {'✓' if ok else '✗'} {scene}")
        print(f"\n  {sum(1 for _, ok, _ in results if ok)}/{len(results)} succeeded")
        return 0 if all(ok for _, ok, _ in results) else 1

    if args.batch:
        results_path = args.results or find_latest_results()
        print(f"Results: {results_path}")
        batch_config = discover_scenes_from_results(results_path)
        print(f"Scenes: {len(batch_config)}")

        results = []
        for i, cfg in enumerate(batch_config):
            scene = cfg["scene"]
            print(f"\n[{i+1}/{len(batch_config)}] {'='*50}\n{scene}\n{'='*50}")
            caps = []
            for c in cfg["captions"]:
                c["data"] = load_captions(c["path"], scene, c.get("label", ""))
                caps.append(c)
                print(f"  [{c['label']}]: {len(c['data'].get('segments',[]))} segments")
            out = OUTPUT_DIR / f"{scene}{args.suffix}.mp4"
            ok = build_video(scene, caps, out)
            results.append((scene, ok, out))

        print(f"\n{'='*60}\nBATCH SUMMARY ({len(results)} scenes)\n{'='*60}")
        for scene, ok, out in results:
            print(f"  {'✓' if ok else '✗'} {scene}")
        ok_count = sum(1 for _, ok, _ in results if ok)
        print(f"\n  {ok_count}/{len(results)} succeeded")
        return 0 if all(ok for _, ok, _ in results) else 1

    if not args.scene:
        parser.error("scene or --batch required")

    captions_list = []
    for i, cap_arg in enumerate(args.caption or []):
        parsed = parse_caption_arg(cap_arg)
        parsed["color"] = parsed["color"] or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        parsed["data"] = load_captions(parsed["path"], args.scene, parsed.get("label", ""))
        captions_list.append(parsed)
        print(f"  [{parsed['label']}]: {len(parsed['data'].get('segments',[]))} segments")

    output = Path(args.output) if args.output else OUTPUT_DIR / f"{args.scene}_review.mp4"
    ok = build_video(args.scene, captions_list, output)
    print(f"\n{'✓' if ok else '✗'} {output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
