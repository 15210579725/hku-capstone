#!/usr/bin/env python3
"""
Prompt construction for NextAct points.

Templates live in prompts/ as plain text and are loaded here — never inline
a prompt in a runner script.

    from prompt_builder import build_prompt
    system, user = build_prompt(point, k=10)
"""

import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# EgoLife and NextMe deliberately share one template, so the two datasets are
# measured under identical instructions.
TEMPLATE_FILE = PROMPT_DIR / "predict_prompt.txt"
_CACHE = {}


def load_template(source: str = "") -> str:
    if "t" not in _CACHE:
        _CACHE["t"] = TEMPLATE_FILE.read_text("utf-8")
    return _CACHE["t"]


def _date_prefix(dates):
    """Token shown in the timestamp example when the point spans several days."""
    if len(dates) <= 1:
        return ""
    if all(re.fullmatch(r"\d{2}-\d{2}", d) for d in dates):
        return "MM-DD "
    if all(re.fullmatch(r"day\d+", d) for d in dates):
        return "dayN "
    return "DATE "


def _render_segments(segments, label):
    lines = []
    for i, seg in enumerate(segments, 1):
        n = f" ({seg['n']} events)" if "n" in seg else ""
        lines.append(f"  {label} {i}: {seg['date']} {seg['start']} – {seg['end']}{n}")
    return "\n".join(lines) if lines else f"  {label} 1: (unknown)"


def build_prompt(point: dict, k: int) -> tuple:
    """Return (system, user) for one NextAct point."""
    ctx_segments = point.get("context_segments", [])
    gt_segments = point.get("gt_segments", [])

    dates = {s["date"] for s in ctx_segments} | {s["date"] for s in gt_segments}
    date_prefix = _date_prefix(dates)

    template = load_template()
    system = template.format(
        k=k,
        context_segments=_render_segments(ctx_segments, "Session"),
        n_context_segments=len(ctx_segments) or 1,
        target_segments=_render_segments(gt_segments[:1] if k == 1 else gt_segments, "Segment"),
        date_prefix=date_prefix,
        action_example=f"[{date_prefix}HH:MM:SS -> {date_prefix}HH:MM:SS] ",
    )

    ctx = point["context_raw"]
    user = (f"Event history ({len(ctx)} events, oldest first):\n\n"
            + "\n".join(ctx))
    return system, user


if __name__ == "__main__":
    from nextact_points import load_nextact
    pts = load_nextact()
    for src in ("egolife", "nextme"):
        p = next(x for x in pts if x["source"] == src and len(x["context_segments"]) >= 1)
        if src == "nextme":
            p = max((x for x in pts if x["source"] == "nextme"),
                    key=lambda x: len(x["context_segments"]))
        s, u = build_prompt(p, k=10)
        print("=" * 70)
        print(f"{src}  id={p['id']}  level={p['level']}  ctx段={len(p['context_segments'])}")
        print("=" * 70)
        print(s)
        print("--- USER (前 3 行) ---")
        print("\n".join(u.splitlines()[:5]))
        print()
