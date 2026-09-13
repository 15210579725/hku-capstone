#!/usr/bin/env python3
"""
Repeat-last-action baseline on the fixed NextAct 1500 points.

Prediction for k = the last k context events, verbatim. Scored with the same
embedding SED + random-baseline normalization as any model.

k=10 is run first: its text set is a strict superset of k=1's, so with the
persistent embedding cache k=1 costs zero API calls.
"""

import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nextact_points import load_nextact
from nextme_benchmark.scorer import BenchmarkScorer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LEVELS = ["L1", "L2", "L3", "L4", "L5"]


def build_samples(points, k):
    samples = []
    for p in points:
        gt = p["gt_text"][:k]
        pred = p["context_text"][-k:]
        if not gt or not pred:
            continue
        samples.append({
            "id": p["id"], "ground_truth": gt, "prediction": pred,
            "level": p["level"], "k": k,
            "_source": p["source"],
        })
    return samples


def summarize(scores):
    """{(source, level): {...}} plus per-source and overall rows."""
    buckets = defaultdict(list)
    for s in scores:
        buckets[(s["source"], s["level"])].append(s["normalized"])
        buckets[(s["source"], "ALL")].append(s["normalized"])
        buckets[("ALL", s["level"])].append(s["normalized"])
        buckets[("ALL", "ALL")].append(s["normalized"])
    out = {}
    for key, vals in buckets.items():
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        out[f"{key[0]}|{key[1]}"] = {
            "n": len(vals),
            "normalized": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        }
    return out


def print_table(all_summaries, ks):
    header = f"{'Dataset':<10}{'Level':<8}" + "".join(f"{'k='+str(k):>22}" for k in ks)
    print("\n" + "=" * len(header))
    print("  Repeat-Last-Action Baseline @ NextAct 1500  (normalized: 0=random, 1=perfect)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = ([("nextme", lv) for lv in LEVELS] + [("nextme", "ALL")]
            + [("egolife", lv) for lv in LEVELS] + [("egolife", "ALL")]
            + [("ALL", lv) for lv in LEVELS] + [("ALL", "ALL")])
    seen = set()
    for src, lv in rows:
        key = f"{src}|{lv}"
        if key in seen:
            continue
        seen.add(key)
        cells = []
        present = False
        for k in ks:
            e = all_summaries[k].get(key)
            if e:
                present = True
                cells.append(f"{e['normalized']:>12.4f} ±{e['std']:<8.4f}")
            else:
                cells.append(f"{'—':>22}")
        if not present:
            continue
        n = all_summaries[ks[0]].get(key, {}).get("n", "")
        if src == "ALL" and lv == "ALL":
            print("-" * len(header))
        print(f"{src:<10}{lv:<8}" + "".join(cells) + f"   n={n}")
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 1],
                    help="run order; k=10 first fills the cache for k=1")
    ap.add_argument("--limit", type=int, help="debug: only first N points")
    ap.add_argument("--output", default="repeat_baseline_nextact.json")
    args = ap.parse_args()

    points = load_nextact()
    if args.limit:
        points = points[:args.limit]
    print(f"Loaded {len(points)} NextAct points")
    dist = defaultdict(int)
    for p in points:
        dist[(p["source"], p["level"])] += 1
    for key in sorted(dist):
        print(f"  {key[0]:<9}{key[1]}: {dist[key]}")

    scorer = BenchmarkScorer(concurrency=args.concurrency)
    if scorer._disk is not None:
        print(f"Embedding disk cache: {scorer._disk.count()} vectors at {scorer._disk.path}")

    all_summaries, all_scores = {}, {}
    for k in args.ks:
        samples = build_samples(points, k)
        print(f"\n### k={k}: scoring {len(samples)} points "
              f"(concurrency={args.concurrency})")
        scores = scorer.score_batch(samples)
        for s, smp in zip(scores, samples):
            s["source"] = smp["_source"]
        all_summaries[k] = summarize(scores)
        all_scores[k] = scores

    print_table(all_summaries, args.ks)

    out = RESULTS_DIR / args.output
    out.write_text(json.dumps({
        "benchmark": "NextAct 1500 (fixed points: NextMe 1000 = 200/level x L1-L5, EgoLife 500 = L1)",
        "baseline": "repeat-last-k-context-events",
        "summaries": {str(k): all_summaries[k] for k in args.ks},
        "scores": {str(k): all_scores[k] for k in args.ks},
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")
    if scorer._disk is not None:
        print(f"Embedding disk cache now holds {scorer._disk.count()} vectors")


if __name__ == "__main__":
    main()
