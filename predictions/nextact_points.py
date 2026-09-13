#!/usr/bin/env python3
"""
NextAct 1500 evaluation points — single source of truth.

NextMe 1000: nextme_benchmark/benchmark.json (200/level x L1-L5, cross-VRS,
             gt_events_k10 already present). Context text is read from
             caption-result/{rec}/hierarchy/{level}/events.txt (last 50 lines).

EgoLife 500: the already-fixed selection from dataset/benchmark_1k.jsonl
             (seed=42, stratified by participant). Its stored ground_truth is
             only 3 events, so GT is re-read from the source day file at
             cutoff_line and extended to 10, continuing into the participant's
             next day when the current day runs out.

Points are never re-sampled here — only their GT length is extended.
"""

import json, random, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
CAPTION_DIR = ROOT / "caption-result"
BENCHMARK_JSON = Path(__file__).resolve().parent / "nextme_benchmark" / "benchmark.json"
EGOLIFE_SOURCE = DATASET_DIR / "benchmark_1k.jsonl"

LEVELS = ["L1", "L2", "L3", "L4", "L5"]
MAX_CONTEXT = 50
MAX_GT = 10
EGOLIFE_PER_LEVEL = 100
SEED = 42

EVENT_PAT = re.compile(r"\[(\d{2}:\d{2}:\d{2})\s*->\s*(\d{2}:\d{2}:\d{2})\]\s*(.+)")
REC_PAT = re.compile(r"(\d+)-(\d+)_hkt(\d{2})(\d{2})-(\d{2})(\d{2})_(\d+)m_(.+)")


def strip_ts(t):
    t = re.sub(r"^\[\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*->\s*\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*", "", t).strip()
    t = re.sub(r"^\[\d{2}:\d{2}:\d{2}\s*->\s*\d{2}:\d{2}:\d{2}\]\s*", "", t).strip()
    return t


def _read_lines(path):
    return [l.rstrip("\n") for l in path.read_text("utf-8").splitlines() if l.strip()]


# ─── NextMe ───────────────────────────────────────────────────────────

def _rec_sort_key(name):
    m = REC_PAT.match(name)
    if not m:
        return (99, 99, 99, 99)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _rec_meta(name):
    m = REC_PAT.match(name)
    if not m:
        return {"date": "00-00", "start": "??:??", "end": "??:??"}
    return {
        "date": f"{int(m.group(1)):02d}-{int(m.group(2)):02d}",
        "start": f"{m.group(3)}:{m.group(4)}",
        "end": f"{m.group(5)}:{m.group(6)}",
    }


_REC_ORDER = None


def _recordings_in_order():
    global _REC_ORDER
    if _REC_ORDER is None:
        names = [d.name for d in CAPTION_DIR.iterdir()
                 if d.is_dir() and REC_PAT.match(d.name)]
        _REC_ORDER = sorted(names, key=_rec_sort_key)
    return _REC_ORDER


def _level_lines(rec, level, cache):
    key = (rec, level)
    if key not in cache:
        f = CAPTION_DIR / rec / "hierarchy" / level / "events.txt"
        cache[key] = _read_lines(f) if f.exists() else []
    return cache[key]


def _build_context(recording, level, cache, gt_recordings=()):
    """Collect the MAX_CONTEXT events preceding the GT.

    Walks backwards from the context recording across earlier recordings. If
    the timeline runs out (context recording sits at the very start), fills
    forward instead — but never at or past the earliest GT recording, so no
    ground-truth material can leak in.
    """
    order = _recordings_in_order()
    try:
        idx = order.index(recording)
    except ValueError:
        return [], []

    chunks = []          # (recording, lines) newest-first while collecting
    collected = 0
    for i in range(idx, -1, -1):
        rec = order[i]
        lines = _level_lines(rec, level, cache)
        if not lines:
            continue
        take = lines[-(MAX_CONTEXT - collected):]
        chunks.append((rec, take))
        collected += len(take)
        if collected >= MAX_CONTEXT:
            break

    chunks.reverse()     # chronological

    if collected < MAX_CONTEXT and gt_recordings:
        gt_first = min(order.index(g) for g in gt_recordings if g in order)
        for i in range(idx + 1, gt_first):
            lines = _level_lines(order[i], level, cache)
            if not lines:
                continue
            take = lines[:MAX_CONTEXT - collected]
            chunks.append((order[i], take))
            collected += len(take)
            if collected >= MAX_CONTEXT:
                break
    ctx_raw, segments = [], []
    for rec, lines in chunks:
        meta = _rec_meta(rec)
        first_start = next((m.group(1) for m in (EVENT_PAT.match(l) for l in lines) if m), meta["start"])
        last_end = next((m.group(2) for m in reversed([EVENT_PAT.match(l) for l in lines]) if m), meta["end"])
        segments.append({"recording": rec, "date": meta["date"],
                         "start": first_start, "end": last_end, "n": len(lines)})
        ctx_raw.extend(f"[{meta['date']} {l[1:]}" if l.startswith("[") else l for l in lines)
    return ctx_raw, segments


def _extend_gt_forward(gt_raw, gt_recordings, level, cache):
    """Top a short GT up to MAX_GT using recordings after the last GT recording."""
    order = _recordings_in_order()
    idxs = [order.index(g) for g in gt_recordings if g in order]
    if not idxs:
        return gt_raw
    out = list(gt_raw)
    for i in range(max(idxs) + 1, len(order)):
        lines = _level_lines(order[i], level, cache)
        if not lines:
            continue
        out.extend(lines[:MAX_GT - len(out)])
        if len(out) >= MAX_GT:
            break
    return out


def load_nextme_points():
    bench = json.loads(BENCHMARK_JSON.read_text())
    cache = {}
    points = []
    for level in LEVELS:
        for p in bench.get(level, []):
            gt_recordings = p.get("gt_recordings_k10", [])
            ctx_raw, ctx_segments = _build_context(
                p["context_recording"], level, cache, gt_recordings)
            gt_raw = p.get("gt_events_k10") or p.get("gt_events_k3", [])
            if len(gt_raw) < MAX_GT:
                gt_raw = _extend_gt_forward(gt_raw, gt_recordings, level, cache)
            if not ctx_raw or not gt_raw:
                continue
            points.append({
                "id": p["id"],
                "source": "nextme",
                "level": level,
                "participant": "user",
                "context_raw": ctx_raw,
                "context_text": [strip_ts(c) for c in ctx_raw],
                "context_segments": ctx_segments,
                "context_date": ctx_segments[0]["date"] if ctx_segments else p["context_date"],
                "context_start": ctx_segments[0]["start"] if ctx_segments else p["context_start"],
                "context_end": ctx_segments[-1]["end"] if ctx_segments else p["context_end"],
                "gt_raw": gt_raw,
                "gt_text": [strip_ts(g) for g in gt_raw],
                "gt_segments": p.get("gt_segments_k10") or p.get("gt_segments_k3", []),
                "gt_recordings": p.get("gt_recordings_k10", []),
            })
    return points


# ─── EgoLife ──────────────────────────────────────────────────────────

def _select_egolife_500():
    """Reproduce the original 500-point EgoLife selection (seed 42, stratified
    by participant), so historical point identities stay stable. Original
    implementation: archive/superseded_20260913/run_full_benchmark.py."""
    all_data = [json.loads(l) for l in EGOLIFE_SOURCE.read_text().splitlines() if l.strip()]
    egolife = [d for d in all_data if d.get("source") == "egolife"]

    by_participant = defaultdict(list)
    for e in egolife:
        by_participant[e["participant"]].append(e)

    selected = []
    per_part = 500 // len(by_participant)
    remainder = 500 % len(by_participant)
    random.seed(42)
    for i, (part, points) in enumerate(sorted(by_participant.items())):
        n = per_part + (1 if i < remainder else 0)
        n = min(n, len(points))
        random.shuffle(points)
        selected.extend(points[:n])
    return selected[:500]


_DAY_PAT = re.compile(r"^(?P<prefix>egolife/[^/]+/L\d+)/day(?P<day>\d+)/events\.txt$")


def _day_chain(file_ref):
    """Return [(day_number, path)] for this participant/level, ascending."""
    m = _DAY_PAT.match(file_ref)
    if not m:
        return [], None
    prefix = m.group("prefix")
    cur_day = int(m.group("day"))
    base = DATASET_DIR / prefix
    days = sorted(
        (int(d.name[3:]), d / "events.txt")
        for d in base.iterdir()
        if re.fullmatch(r"day\d+", d.name) and (d / "events.txt").exists()
    )
    return days, cur_day


def _extend_gt(point, cache):
    """Read MAX_GT ground-truth lines from cutoff_line, continuing across days."""
    days, cur_day = _day_chain(point["file"])
    if not days:
        return None, []

    def lines_of(path):
        key = str(path)
        if key not in cache:
            cache[key] = _read_lines(path)
        return cache[key]

    gt, spans = [], []
    start_idx = next((i for i, (d, _) in enumerate(days) if d == cur_day), None)
    if start_idx is None:
        return None, []

    offset = point["cutoff_line"]
    for di in range(start_idx, len(days)):
        day_num, path = days[di]
        lines = lines_of(path)
        if offset >= len(lines):
            offset = 0
            continue
        take = lines[offset:offset + (MAX_GT - len(gt))]
        if take:
            gt.extend(take)
            spans.append({"day": day_num, "first_line": offset, "n": len(take)})
        offset = 0
        if len(gt) >= MAX_GT:
            break
    return gt, spans


def _time_bounds(raw_lines):
    """(start, end) HH:MM:SS from a list of '[s -> e] text' lines."""
    starts = [m.group(1) for m in (EVENT_PAT.match(l) for l in raw_lines) if m]
    ends = [m.group(2) for m in (EVENT_PAT.match(l) for l in raw_lines) if m]
    return (starts[0] if starts else "??:??:??", ends[-1] if ends else "??:??:??")


def _alloc(n_total, n_parts):
    """Split n_total across n_parts as evenly as possible."""
    base, rem = divmod(n_total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _participants():
    return sorted(d.name for d in (DATASET_DIR / "egolife").iterdir()
                  if d.is_dir() and re.match(r"A\d_", d.name))


def _concat_days(participant, level):
    """[(day_number, line)] for all days of one participant/level, chronological."""
    base = DATASET_DIR / "egolife" / participant / level
    if not base.is_dir():
        return []
    days = sorted((int(d.name[3:]), d / "events.txt") for d in base.iterdir()
                  if re.fullmatch(r"day\d+", d.name) and (d / "events.txt").exists())
    out = []
    for day_num, path in days:
        out.extend((day_num, l) for l in _read_lines(path))
    return out


def _segments_from_tagged(tagged):
    """Group [(day, line)] into per-day segments with time bounds."""
    segments = []
    for day, line in tagged:
        m = EVENT_PAT.match(line)
        start = m.group(1) if m else "??:??:??"
        end = m.group(2) if m else "??:??:??"
        if segments and segments[-1]["date"] == f"day{day}":
            segments[-1]["end"] = end
            segments[-1]["n"] += 1
        else:
            segments.append({"recording": f"day{day}", "date": f"day{day}",
                             "start": start, "end": end, "n": 1})
    return segments


def _build_egolife_level(level, n_points, rng):
    """Sample n_points cutoffs per level across participants, cross-day."""
    parts = _participants()
    quota = _alloc(n_points, len(parts))
    points = []
    for part, want in zip(parts, quota):
        tagged = _concat_days(part, level)
        lo, hi = MAX_CONTEXT, len(tagged) - MAX_GT
        if hi <= lo:
            continue
        candidates = range(lo, hi)
        picks = sorted(rng.sample(list(candidates), min(want, len(candidates))))
        for cut in picks:
            ctx_t = tagged[cut - MAX_CONTEXT:cut]
            gt_t = tagged[cut:cut + MAX_GT]
            ctx_raw = [l for _, l in ctx_t]
            gt_raw = [l for _, l in gt_t]
            ctx_segs = _segments_from_tagged(ctx_t)
            gt_segs = _segments_from_tagged(gt_t)
            points.append({
                "id": f"ego_{level}_{part.split('_')[0]}_{cut}",
                "source": "egolife",
                "level": level,
                "participant": part,
                "context_raw": ctx_raw,
                "context_text": [strip_ts(c) for c in ctx_raw],
                "context_segments": ctx_segs,
                "context_date": ctx_segs[0]["date"],
                "context_start": ctx_segs[0]["start"],
                "context_end": ctx_segs[-1]["end"],
                "gt_raw": gt_raw,
                "gt_text": [strip_ts(g) for g in gt_raw],
                "gt_segments": gt_segs,
                "cutoff_line": cut,
                "source_file": f"egolife/{part}/{level}",
            })
    return points


def load_egolife_points():
    """EgoLife 500 = 100 per level x L1-L5.

    L1 reuses 100 of the previously fixed 500-point selection (so existing
    ds-v4-flash L1 predictions stay valid); L2-L5 are sampled from each
    participant's days concatenated into one timeline.
    """
    rng = random.Random(SEED)
    points = _reuse_egolife_l1(EGOLIFE_PER_LEVEL)
    for level in ["L2", "L3", "L4", "L5"]:
        points.extend(_build_egolife_level(level, EGOLIFE_PER_LEVEL, rng))
    return points


def _reuse_egolife_l1(n_points):
    selected = _select_egolife_500()
    by_part = defaultdict(list)
    for p in selected:
        by_part[p["participant"]].append(p)
    parts = sorted(by_part)
    quota = _alloc(n_points, len(parts))
    kept = []
    for part, want in zip(parts, quota):
        kept.extend(by_part[part][:want])

    cache = {}
    points = []
    for p in kept:
        gt_raw, spans = _extend_gt(p, cache)
        if not gt_raw:
            gt_raw = p["ground_truth"]
            spans = []
        ctx = p["context"]
        ctx_start, ctx_end = _time_bounds(ctx)
        gt_start, gt_end = _time_bounds(gt_raw)
        day_label = f"day{spans[0]['day']}" if spans else p["session"]
        points.append({
            "id": p["id"],
            "source": "egolife",
            "level": "L1",
            "participant": p["participant"],
            "context_raw": ctx,
            "context_text": [strip_ts(c) for c in ctx],
            "context_date": day_label,
            "context_start": ctx_start,
            "context_end": ctx_end,
            "context_segments": [{"recording": p["file"], "date": day_label,
                                  "start": ctx_start, "end": ctx_end,
                                  "n": len(ctx)}],
            "gt_raw": gt_raw,
            "gt_text": [strip_ts(g) for g in gt_raw],
            "gt_segments": [{"date": f"day{s['day']}", "start": gt_start, "end": gt_end}
                            for s in spans[:1]] or [{"date": day_label, "start": gt_start, "end": gt_end}],
            "gt_day_spans": spans,
            "source_file": p["file"],
            "cutoff_line": p["cutoff_line"],
        })
    return points


def load_nextact():
    return load_egolife_points() + load_nextme_points()


if __name__ == "__main__":
    import collections
    pts = load_nextact()
    print(f"NextAct total: {len(pts)}")
    by = collections.Counter((p["source"], p["level"]) for p in pts)
    for key in sorted(by):
        print(f"  {key[0]:<8} {key[1]}: {by[key]}")
    for k in (1, 10):
        ok = sum(1 for p in pts if len(p["gt_text"]) >= k and len(p["context_text"]) >= k)
        print(f"  k={k} viable: {ok}/{len(pts)}")
    ego = [p for p in pts if p["source"] == "egolife"]
    cross = [p for p in ego if len(p.get("gt_day_spans", [])) > 1]
    print(f"  EgoLife 跨天取 GT 的点: {len(cross)}")
    short = [p for p in pts if len(p["gt_text"]) < 10]
    print(f"  GT 不足 10 条的点: {len(short)}  {[p['id'] for p in short][:5]}")
