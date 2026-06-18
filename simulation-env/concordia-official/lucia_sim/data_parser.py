"""
Data parser for LUCIA L1 event data.

Parses first-person event descriptions from L1 granularity files and builds
decision points for use in Concordia simulation. LUCIA is the main agent;
all other characters (Jake, Katrina, Shure, Tasha, Alice, etc.) are
environment entities whose actions appear as observations to LUCIA.

Event line format:  [HH:MM:SS -> HH:MM:SS] 中文第一人称描述
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_BASE = Path("/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data")
LUCIA_L1_DAY1 = DATA_BASE / "A4_LUCIA" / "L1" / "day1" / "events.txt"

# ── Regex for line parsing ──────────────────────────────────────────────────
_LINE_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2})\s*->\s*(\d{2}:\d{2}:\d{2})\]\s*(.+)$"
)

# ── Classification patterns ────────────────────────────────────────────────
# Order matters: observation and internal are checked first; action is the
# fallback so it does not need an exhaustive keyword list.

# Observation: things LUCIA perceives from the environment or other people.
# "我听到X说…", "我看到…", "我听X讲话/说话/说…", "我听着X讲话"
# BUT NOT "我听完后…" or "我听了X的话笑了笑" — those are reactions (→ action).
_OBSERVATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"^我看到"),
    re.compile(r"^我看见"),
    re.compile(r"^我听到"),
    # "我听X讲话/说话/说Y/讲/继续讲" — pure listening, no reactive verb
    re.compile(r"^我听着?[一-鿿A-Za-z]+(?:讲话|说话|说|讲|继续讲|给我解释)"),
    # "我听着大家…" style
    re.compile(r"^我听着大家"),
    re.compile(r"^我听他们"),
    re.compile(r"^我听着她们"),
]

# Internal: LUCIA's subjective feelings, thoughts, guesses.
# Must match when LUCIA is the subject, not when quoting others
# (e.g., 我听到Shure说"我觉得…" is observation, already caught above).
_INTERNAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^我感到"),
    re.compile(r"^我觉得"),
    re.compile(r"^我想到"),
    re.compile(r"^我猜"),
    re.compile(r"^我有一些尴尬"),     # "我有一些尴尬地不知所措"
    re.compile(r"^我不知道"),          # uncertainty / internal monologue
]


# ── Core functions ──────────────────────────────────────────────────────────

def classify_event(text: str) -> str:
    """Classify the event text (without timestamp) into observation/action/internal."""
    for pat in _OBSERVATION_PATTERNS:
        if pat.search(text):
            return "observation"
    for pat in _INTERNAL_PATTERNS:
        if pat.search(text):
            return "internal"
    return "action"


def parse_l1_events(
    filepath: str | Path = LUCIA_L1_DAY1,
) -> list[dict[str, str]]:
    """
    Parse an L1 events file into a list of event dicts.

    Returns:
        List of {"start": "HH:MM:SS", "end": "HH:MM:SS", "text": str, "type": str}
    """
    filepath = Path(filepath)
    events: list[dict[str, str]] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            start, end, text = m.group(1), m.group(2), m.group(3)
            events.append({
                "start": start,
                "end": end,
                "text": text,
                "type": classify_event(text),
            })
    return events


def _time_in_range(
    t: str,
    time_start: Optional[str],
    time_end: Optional[str],
) -> bool:
    """Check if HH:MM:SS (or HH:MM) time falls within [time_start, time_end]."""
    if time_start and t[:len(time_start)] < time_start:
        return False
    if time_end and t[:len(time_end)] > time_end:
        return False
    return True


def build_decision_points(
    events: list[dict[str, str]],
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
) -> list[dict]:
    """
    Build decision points from parsed L1 events.

    A decision point is formed whenever LUCIA performs an **action**:
      - All preceding observations (since the last action) are accumulated
        as context.
      - The action itself is the ground-truth prediction target.
      - Internal events are included in the observation context (they inform
        LUCIA's state but are not prediction targets).

    Args:
        events: output of parse_l1_events()
        time_start: optional lower bound "HH:MM" or "HH:MM:SS" (inclusive)
        time_end:   optional upper bound "HH:MM" or "HH:MM:SS" (inclusive)

    Returns:
        List of decision-point dicts:
        {
            "step": int,              # 0-indexed
            "timestamp": str,         # HH:MM:SS of the action
            "observations": [str],    # accumulated observation/internal texts with timestamps
            "gt_action": str,         # ground truth action text (no timestamp)
            "gt_raw": str,            # full line including [start -> end]
        }
    """
    # Filter by time range
    filtered = [
        e for e in events
        if _time_in_range(e["start"], time_start, time_end)
    ]

    decision_points: list[dict] = []
    pending_obs: list[str] = []  # accumulated observations since last action
    step = 0

    for ev in filtered:
        ts_prefix = f"[{ev['start']} -> {ev['end']}]"
        full_line = f"{ts_prefix} {ev['text']}"

        if ev["type"] in ("observation", "internal"):
            pending_obs.append(full_line)
        else:
            # action → create a decision point
            decision_points.append({
                "step": step,
                "timestamp": ev["start"],
                "observations": list(pending_obs),  # copy
                "gt_action": ev["text"],
                "gt_raw": full_line,
            })
            step += 1
            # Do NOT clear pending_obs — observations accumulate across
            # decision points so each point sees ALL prior context.
            # However, per the spec: "ALL prior observations since the
            # last action", so we clear after emitting.
            pending_obs.clear()

    return decision_points


def get_morning_decision_points(
    time_start: str = "11:09",
    time_end: str = "11:15",
    filepath: str | Path = LUCIA_L1_DAY1,
) -> list[dict]:
    """Convenience function: parse + build decision points for a short morning window."""
    events = parse_l1_events(filepath)
    return build_decision_points(events, time_start=time_start, time_end=time_end)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _print_stats(events: list[dict[str, str]]) -> None:
    """Print classification statistics."""
    from collections import Counter
    counts = Counter(e["type"] for e in events)
    total = len(events)
    print(f"总行数: {total}")
    for typ in ("observation", "action", "internal"):
        n = counts.get(typ, 0)
        pct = n / total * 100 if total else 0
        print(f"  {typ:12s}: {n:6d}  ({pct:5.1f}%)")


def _print_decision_points(dps: list[dict], n: int = 5) -> None:
    """Pretty-print first n decision points."""
    for dp in dps[:n]:
        print(f"\n{'='*60}")
        print(f"Step {dp['step']}  @  {dp['timestamp']}")
        print(f"--- 上下文观测 ({len(dp['observations'])} 条) ---")
        for obs in dp["observations"]:
            print(f"  {obs}")
        print(f"--- 真实动作 ---")
        print(f"  {dp['gt_raw']}")


if __name__ == "__main__":
    print("=" * 60)
    print("LUCIA L1 Day1 — 全天统计")
    print("=" * 60)

    all_events = parse_l1_events()
    _print_stats(all_events)

    # Morning window stats
    morning_events = [
        e for e in all_events
        if _time_in_range(e["start"], "11:09", "11:15")
    ]
    print(f"\n早间窗口 (11:09-11:15): {len(morning_events)} 行")
    _print_stats(morning_events)

    # Decision points for the morning window
    dps = build_decision_points(all_events, time_start="11:09", time_end="11:15")
    print(f"\n决策点数量: {len(dps)}")
    print("\n前 5 个决策点:")
    _print_decision_points(dps, n=5)
