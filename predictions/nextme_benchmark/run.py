#!/usr/bin/env python3
"""
NextMe Behavior Prediction Benchmark — Main Entry Point.

Predict + Score in one command:
    python -m nextme_benchmark.run --model deepseek-v4-flash --k 3
    python -m nextme_benchmark.run --model deepseek-v4-flash --k 10 --levels L1 L2
    python -m nextme_benchmark.run --model deepseek-v4-flash --k 3 --max-points 5  # quick test

Score existing predictions:
    python -m nextme_benchmark.run --score-only predictions.json
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="NextMe Behavior Prediction Benchmark")

    # Model config
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Model name (default: deepseek-v4-flash)")
    parser.add_argument("--api-key", help="API key (or set DEEPSEEK_API_KEY env)")
    parser.add_argument("--base-url", default="https://api.deepseek.com",
                        help="API base URL")
    parser.add_argument("--api-key-file",
                        help="File containing API key (one line)")

    # Benchmark config
    parser.add_argument("--k", type=int, default=3, choices=[3, 10],
                        help="Prediction length (default: 3)")
    parser.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4", "L5"],
                        help="Levels to evaluate (default: all)")
    parser.add_argument("--max-points", type=int,
                        help="Max points per level (for quick testing)")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="API call concurrency")

    # Paths
    parser.add_argument("--benchmark", help="Benchmark JSON path")
    parser.add_argument("--prompt-template", help="Prompt template path")
    parser.add_argument("--caption-dir", help="Caption-result directory")
    parser.add_argument("--eval-config", help="Embedding eval config path")
    parser.add_argument("--baselines", help="Baselines JSON path")
    parser.add_argument("--output", "-o", help="Output file for results")

    # Score-only mode
    parser.add_argument("--score-only", help="Score existing predictions JSON file (skip prediction)")

    args = parser.parse_args()

    # ─── Score-only mode ──────────────────────────────────────────────
    if args.score_only:
        from nextme_benchmark.scorer import BenchmarkScorer
        scorer = BenchmarkScorer(args.eval_config, args.baselines)
        preds = json.loads(Path(args.score_only).read_text())

        samples = []
        for p in preds:
            samples.append({
                "id": p["id"],
                "ground_truth": p["ground_truth"],
                "prediction": p["prediction"],
                "level": p["level"],
                "k": p["k"],
            })

        print(f"Scoring {len(samples)} predictions...")
        results = scorer.score_batch(samples)
        _print_summary(results, args.levels)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nScores saved to {args.output}")
        return

    # ─── Predict + Score mode ─────────────────────────────────────────
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key and args.api_key_file:
        api_key = Path(args.api_key_file).read_text().strip()
    if not api_key:
        kf = Path.home() / ".config" / "deepseek" / "api_key"
        if kf.exists():
            api_key = kf.read_text().strip()
    if not api_key:
        sys.exit("No API key: set --api-key, --api-key-file, DEEPSEEK_API_KEY, or ~/.config/deepseek/api_key")

    from nextme_benchmark.predictor import BenchmarkPredictor
    from nextme_benchmark.scorer import BenchmarkScorer

    predictor = BenchmarkPredictor(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        benchmark_path=args.benchmark,
        prompt_template_path=args.prompt_template,
        caption_result_dir=args.caption_dir,
        concurrency=args.concurrency,
    )

    print(f"{'='*60}")
    print(f"NextMe Behavior Prediction Benchmark")
    print(f"  Model:  {args.model}")
    print(f"  k:      {args.k}")
    print(f"  Levels: {' '.join(args.levels)}")
    pts_desc = f"max {args.max_points}/level" if args.max_points else "200/level (full)"
    print(f"  Points: {pts_desc}")
    print(f"{'='*60}\n")

    t0 = time.time()
    predictions = predictor.predict_benchmark(
        k=args.k, levels=args.levels, max_points=args.max_points)
    pred_time = time.time() - t0

    print(f"\nPrediction done: {len(predictions)} results in {pred_time:.1f}s")

    # Save predictions
    out_dir = Path(args.output).parent if args.output else Path(".")
    pred_file = args.output or f"predictions_{args.model}_k{args.k}.json"
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {pred_file}")

    # Score
    print(f"\nScoring {len(predictions)} predictions...")
    scorer = BenchmarkScorer(args.eval_config, args.baselines)
    samples = [{"id": p["id"], "ground_truth": p["ground_truth"],
                "prediction": p["prediction"], "level": p["level"],
                "k": p["k"]} for p in predictions]
    scores = scorer.score_batch(samples)
    score_time = time.time() - t0 - pred_time

    _print_summary(scores, args.levels)

    # Save scores
    score_file = pred_file.replace("predictions_", "scores_")
    with open(score_file, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"\nScores saved to {score_file}")
    print(f"Total time: {time.time()-t0:.1f}s (predict {pred_time:.1f}s + score {score_time:.1f}s)")


def _print_summary(scores, levels):
    """Print per-level and overall summary table."""
    print(f"\n{'='*60}")
    print(f"{'Level':<6} {'N':>4} {'Raw SED':>10} {'Normalized':>12} {'±Std':>8}")
    print(f"{'-'*60}")

    all_raw, all_norm = [], []
    for lv in ["L1", "L2", "L3", "L4", "L5"]:
        if lv not in levels:
            continue
        lv_scores = [s for s in scores if s["level"] == lv]
        if not lv_scores:
            continue
        raws = [s["raw"] for s in lv_scores]
        norms = [s["normalized"] for s in lv_scores if s["normalized"] is not None]
        all_raw.extend(raws)
        all_norm.extend(norms)

        raw_mean = np.mean(raws)
        if norms:
            norm_mean = np.mean(norms)
            norm_std = np.std(norms)
            print(f"{lv:<6} {len(lv_scores):>4} {raw_mean:>10.4f} {norm_mean:>12.4f} {norm_std:>8.4f}")
        else:
            print(f"{lv:<6} {len(lv_scores):>4} {raw_mean:>10.4f} {'N/A':>12}")

    print(f"{'-'*60}")
    if all_norm:
        print(f"{'ALL':<6} {len(all_raw):>4} {np.mean(all_raw):>10.4f} "
              f"{np.mean(all_norm):>12.4f} {np.std(all_norm):>8.4f}")
    print(f"{'='*60}")
    print(f"Normalized: 0 = random, 1 = perfect")


if __name__ == "__main__":
    main()
