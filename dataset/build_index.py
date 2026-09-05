#!/usr/bin/env python3
"""Build dataset index and 1k benchmark test points.

Outputs:
  data_index.json      - full inventory of all event files
  benchmark_1k.jsonl   - 1000 sampled L1 test points with context + ground truth
"""
import json, os, random, re

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXT_WINDOW = 50
GT_ACTIONS = 3
SEED = 42
TOTAL_POINTS = 1000
EGOLIFE_POINTS = 800
NEXTME_POINTS = 200

def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)

def read_lines(path):
    with open(path) as f:
        return [line.rstrip('\n') for line in f]

def build_data_index():
    """Scan all event files and build a structured index."""
    index = {"egolife": {}, "nextme": {}}

    ego_dir = os.path.join(DATASET_DIR, 'egolife')
    for participant in sorted(os.listdir(ego_dir)):
        p_dir = os.path.join(ego_dir, participant)
        if not os.path.isdir(p_dir) or participant.startswith('.'):
            continue
        index["egolife"][participant] = {}
        for level in sorted(os.listdir(p_dir)):
            l_dir = os.path.join(p_dir, level)
            if not os.path.isdir(l_dir):
                continue
            index["egolife"][participant][level] = {}
            for day in sorted(os.listdir(l_dir)):
                d_dir = os.path.join(l_dir, day)
                ef = os.path.join(d_dir, 'events.txt')
                if os.path.isfile(ef):
                    n = count_lines(ef)
                    rel = os.path.relpath(ef, DATASET_DIR)
                    index["egolife"][participant][level][day] = {
                        "file": rel,
                        "events": n,
                    }

    nm_dir = os.path.join(DATASET_DIR, 'nextme')
    for rec in sorted(os.listdir(nm_dir)):
        ef = os.path.join(nm_dir, rec, 'events.txt')
        if os.path.isfile(ef):
            n = count_lines(ef)
            rel = os.path.relpath(ef, DATASET_DIR)
            index["nextme"][rec] = {
                "file": rel,
                "events": n,
            }

    return index

def sample_points_from_file(file_path, rel_path, source, participant, day_or_rec, n_points, rng):
    """Sample n_points evaluation points from a single events.txt file."""
    lines = read_lines(file_path)
    total = len(lines)
    min_cutoff = CONTEXT_WINDOW
    max_cutoff = total - GT_ACTIONS
    if max_cutoff <= min_cutoff:
        return []

    candidates = list(range(min_cutoff, max_cutoff))
    if len(candidates) <= n_points:
        selected = candidates
    else:
        selected = sorted(rng.sample(candidates, n_points))

    points = []
    for cutoff in selected:
        ctx_start = max(0, cutoff - CONTEXT_WINDOW)
        context = lines[ctx_start:cutoff]
        gt = lines[cutoff:cutoff + GT_ACTIONS]

        point = {
            "source": source,
            "participant": participant,
            "session": day_or_rec,
            "file": rel_path,
            "cutoff_line": cutoff,
            "context_start_line": ctx_start,
            "n_context": len(context),
            "n_gt": len(gt),
            "context": context,
            "ground_truth": gt,
        }
        points.append(point)

    return points

def build_benchmark(data_index, n_egolife=EGOLIFE_POINTS, n_nextme=NEXTME_POINTS):
    """Sample balanced benchmark points from both datasets."""
    rng = random.Random(SEED)

    ego_l1_files = []
    for participant, levels in data_index["egolife"].items():
        if "L1" not in levels:
            continue
        for day, info in levels["L1"].items():
            fpath = os.path.join(DATASET_DIR, info["file"])
            if info["events"] > CONTEXT_WINDOW + GT_ACTIONS:
                ego_l1_files.append((fpath, info["file"], participant, day, info["events"]))

    nm_files = []
    for rec, info in data_index["nextme"].items():
        fpath = os.path.join(DATASET_DIR, info["file"])
        if info["events"] > CONTEXT_WINDOW + GT_ACTIONS:
            nm_files.append((fpath, info["file"], rec, info["events"]))

    # Distribute EgoLife points proportionally by file size
    total_ego_events = sum(f[4] for f in ego_l1_files)
    ego_points = []
    allocated = 0
    for i, (fpath, rel, participant, day, n_events) in enumerate(ego_l1_files):
        if i == len(ego_l1_files) - 1:
            alloc = n_egolife - allocated
        else:
            alloc = max(1, round(n_egolife * n_events / total_ego_events))
            alloc = min(alloc, n_egolife - allocated)
        allocated += alloc
        pts = sample_points_from_file(fpath, rel, "egolife", participant, day, alloc, rng)
        ego_points.extend(pts)

    # Distribute Next-Me points proportionally
    total_nm_events = sum(f[3] for f in nm_files)
    nm_points = []
    allocated = 0
    for i, (fpath, rel, rec, n_events) in enumerate(nm_files):
        if i == len(nm_files) - 1:
            alloc = n_nextme - allocated
        else:
            alloc = max(1, round(n_nextme * n_events / total_nm_events))
            alloc = min(alloc, n_nextme - allocated)
        allocated += alloc
        pts = sample_points_from_file(fpath, rel, "nextme", "user", rec, alloc, rng)
        nm_points.extend(pts)

    all_points = ego_points + nm_points
    # Assign IDs
    for i, pt in enumerate(all_points):
        pt["id"] = f"eval_{i:04d}"

    return all_points

def main():
    print("Building data index...")
    data_index = build_data_index()

    ego_files = sum(
        len(days)
        for p in data_index["egolife"].values()
        for days in p.values()
    )
    nm_files = len(data_index["nextme"])
    print(f"  EgoLife: {len(data_index['egolife'])} participants, {ego_files} files")
    print(f"  Next-Me: {nm_files} recordings")

    idx_path = os.path.join(DATASET_DIR, 'data_index.json')
    with open(idx_path, 'w') as f:
        json.dump(data_index, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {idx_path}")

    print("\nBuilding 1k benchmark...")
    points = build_benchmark(data_index)

    ego_count = sum(1 for p in points if p["source"] == "egolife")
    nm_count = sum(1 for p in points if p["source"] == "nextme")
    print(f"  EgoLife points: {ego_count}")
    print(f"  Next-Me points: {nm_count}")
    print(f"  Total: {len(points)}")

    bench_path = os.path.join(DATASET_DIR, 'benchmark_1k.jsonl')
    with open(bench_path, 'w') as f:
        for pt in points:
            f.write(json.dumps(pt, ensure_ascii=False) + '\n')
    print(f"  Saved: {bench_path}")

    # Print distribution summary
    print("\n--- EgoLife distribution ---")
    from collections import Counter
    ego_dist = Counter(p["participant"] for p in points if p["source"] == "egolife")
    for k in sorted(ego_dist):
        print(f"  {k}: {ego_dist[k]} points")

    print("\n--- Next-Me distribution ---")
    nm_dist = Counter(p["session"] for p in points if p["source"] == "nextme")
    for k in sorted(nm_dist):
        print(f"  {k}: {nm_dist[k]} points")

if __name__ == '__main__':
    main()
