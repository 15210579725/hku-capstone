"""生成情绪激活可视化 HTML：每个窗口的文本 + 6 情绪颜色条。"""
import csv
import re
import html as html_mod
from pathlib import Path

EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
EMO_COLORS = {
    "happiness": "#FFD700",
    "sadness":   "#4169E1",
    "anger":     "#DC143C",
    "fear":      "#8B008B",
    "disgust":   "#228B22",
    "surprise":  "#FF8C00",
}
EMO_CN = {
    "happiness": "快乐",
    "sadness":   "悲伤",
    "anger":     "愤怒",
    "fear":      "恐惧",
    "disgust":   "厌恶",
    "surprise":  "惊讶",
}

WIN_N = 8
WIN_STRIDE = 4


def parse_events(path):
    pat = re.compile(r"\[(\d{2}:\d{2}:\d{2})\s*->\s*(\d{2}:\d{2}:\d{2})\]\s*(.+)")
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                events.append({"t_start": m.group(1), "t_end": m.group(2), "text": m.group(3).strip()})
    return events


def make_windows(events):
    windows = []
    for i in range(0, max(1, len(events) - WIN_N + 1), WIN_STRIDE):
        chunk = events[i:i + WIN_N]
        windows.append({
            "idx": len(windows),
            "t_start": chunk[0]["t_start"],
            "t_end": chunk[-1]["t_end"],
            "texts": [e["text"] for e in chunk],
        })
    return windows


def norm_score(val, vmin, vmax):
    if vmax == vmin:
        return 0.5
    return max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))


def main():
    base = Path(__file__).parent
    events = parse_events(base / "data" / "day1_L2_events.txt")
    windows = make_windows(events)

    with open(base / "output" / "scores_day1_emocirc.csv", encoding="utf-8") as f:
        scores = list(csv.DictReader(f))

    assert len(scores) == len(windows), f"scores {len(scores)} != windows {len(windows)}"

    ranges = {}
    for emo in EMOTIONS:
        vals = [float(r[emo]) for r in scores]
        ranges[emo] = (min(vals), max(vals))

    rows_html = []
    for w, s in zip(windows, scores):
        text_lines = "<br>".join(html_mod.escape(t) for t in w["texts"])

        bars = []
        for emo in EMOTIONS:
            val = float(s[emo])
            n = norm_score(val, ranges[emo][0], ranges[emo][1])
            color = EMO_COLORS[emo]
            pct = int(n * 100)
            bars.append(
                f'<div class="bar-row">'
                f'<span class="emo-label" style="color:{color}">{EMO_CN[emo]}</span>'
                f'<div class="bar-bg">'
                f'<div class="bar-fill" style="width:{pct}%;background:{color};opacity:0.7"></div>'
                f'</div>'
                f'<span class="bar-val">{val:+.3f}</span>'
                f'</div>'
            )

        dominant_emo = max(EMOTIONS, key=lambda e: float(s[e]))
        dominant_val = float(s[dominant_emo])
        dominant_n = norm_score(dominant_val, ranges[dominant_emo][0], ranges[dominant_emo][1])
        border_color = EMO_COLORS[dominant_emo]
        border_alpha = 0.3 + 0.7 * dominant_n

        rows_html.append(
            f'<div class="win-card" style="border-left:5px solid {border_color};'
            f'border-left-color:rgba({_hex_to_rgb(border_color)},{border_alpha:.2f})">'
            f'<div class="win-header">'
            f'<span class="win-time">{w["t_start"]} → {w["t_end"]}</span>'
            f'<span class="win-id">窗口 {w["idx"]}</span>'
            f'</div>'
            f'<div class="win-body">'
            f'<div class="win-text">{text_lines}</div>'
            f'<div class="win-bars">{"".join(bars)}</div>'
            f'</div>'
            f'</div>'
        )

    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Emotion Circuits — Day1 L2 情绪激活可视化</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; background:#1a1a2e; color:#eee; padding:20px; }}
h1 {{ text-align:center; margin-bottom:8px; font-size:1.4em; color:#fff; }}
.subtitle {{ text-align:center; color:#888; margin-bottom:20px; font-size:0.9em; }}
.legend {{ display:flex; justify-content:center; gap:18px; margin-bottom:20px; flex-wrap:wrap; }}
.legend-item {{ display:flex; align-items:center; gap:5px; font-size:0.85em; }}
.legend-dot {{ width:12px; height:12px; border-radius:50%; }}
.controls {{ text-align:center; margin-bottom:16px; }}
.controls select, .controls input {{ padding:4px 8px; border-radius:4px; border:1px solid #555; background:#2a2a4a; color:#eee; margin:0 4px; }}
.win-card {{ background:#16213e; border-radius:8px; margin-bottom:12px; padding:14px; transition:all 0.2s; }}
.win-card:hover {{ background:#1a2744; transform:translateX(2px); }}
.win-header {{ display:flex; justify-content:space-between; margin-bottom:8px; }}
.win-time {{ color:#aaa; font-size:0.85em; font-family:monospace; }}
.win-id {{ color:#666; font-size:0.8em; }}
.win-body {{ display:flex; gap:16px; }}
.win-text {{ flex:1; font-size:0.88em; line-height:1.6; color:#ccc; }}
.win-bars {{ width:280px; flex-shrink:0; }}
.bar-row {{ display:flex; align-items:center; margin-bottom:4px; gap:6px; }}
.emo-label {{ width:32px; font-size:0.75em; text-align:right; font-weight:600; }}
.bar-bg {{ flex:1; height:14px; background:#0f3460; border-radius:3px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:3px; transition:width 0.3s; }}
.bar-val {{ width:52px; font-size:0.7em; color:#888; font-family:monospace; text-align:right; }}
.filter-highlight .win-card {{ opacity:0.3; }}
.filter-highlight .win-card.highlighted {{ opacity:1; }}
</style>
</head>
<body>
<h1>Do LLMs "Feel?" — Day1 L2 情绪激活可视化</h1>
<p class="subtitle">Llama-3.2-3B-Instruct · MLP 情绪方向投影 · 500 窗口 × 6 情绪 · LUCIA Day1</p>
<div class="legend">
{"".join(f'<div class="legend-item"><div class="legend-dot" style="background:{EMO_COLORS[e]}"></div>{EMO_CN[e]} {e}</div>' for e in EMOTIONS)}
</div>
<div class="controls">
<label>高亮情绪: <select id="emoFilter">
<option value="">全部显示</option>
{"".join(f'<option value="{e}">{EMO_CN[e]}</option>' for e in EMOTIONS)}
</select></label>
<label>阈值 ≥ <input type="range" id="threshold" min="0" max="100" value="70" style="width:120px">
<span id="threshVal">0.70</span></label>
</div>
<div id="container">
{"".join(rows_html)}
</div>
<script>
const cards = document.querySelectorAll('.win-card');
const scores = {_scores_json(scores)};
const ranges = {_ranges_json(ranges)};
const emoFilter = document.getElementById('emoFilter');
const threshold = document.getElementById('threshold');
const threshVal = document.getElementById('threshVal');
function update() {{
    const emo = emoFilter.value;
    const thr = threshold.value / 100;
    threshVal.textContent = thr.toFixed(2);
    const container = document.getElementById('container');
    if (!emo) {{
        container.classList.remove('filter-highlight');
        cards.forEach(c => c.classList.remove('highlighted'));
        return;
    }}
    container.classList.add('filter-highlight');
    cards.forEach((c, i) => {{
        const val = scores[i][emo];
        const r = ranges[emo];
        const norm = (val - r[0]) / (r[1] - r[0] || 1);
        c.classList.toggle('highlighted', norm >= thr);
    }});
}}
emoFilter.addEventListener('change', update);
threshold.addEventListener('input', update);
</script>
</body>
</html>"""

    out = base / "viz" / "emotion_activation_day1.html"
    out.write_text(page, encoding="utf-8")
    print(f"Generated: {out} ({len(rows_html)} windows)")


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _scores_json(scores):
    import json
    return json.dumps([{e: float(r[e]) for e in EMOTIONS} for r in scores])


def _ranges_json(ranges):
    import json
    return json.dumps({e: list(v) for e, v in ranges.items()})


if __name__ == "__main__":
    main()
