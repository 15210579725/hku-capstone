#!/usr/bin/env python3
"""
Unified benchmark runner for EgoLife & Next-Me action prediction.

Reads benchmark_1k.jsonl, calls an LLM to predict next actions,
outputs predictions.jsonl compatible with metric/eval_kit/evaluate.py.

Usage:
  # Predict (default: deepseek-v4-flash)
  python3 run_benchmark.py --n 10              # test 10 points
  python3 run_benchmark.py --n 1000            # full 1k benchmark
  python3 run_benchmark.py --model deepseek-chat --n 10  # use v4-pro

  # Evaluate predictions (requires metric/eval_kit)
  python3 ../metric/eval_kit/evaluate.py -i predictions.jsonl -c ../metric/eval_kit/config.json -m both
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_FILE = SCRIPT_DIR / "benchmark_1k.jsonl"

PREDICT_PROMPT = """You are observing a person's daily activities from their first-person perspective. Based on the recent activity context below, predict the next 3 most likely actions this person will take.

## Recent Activity Context
{context}

## Instructions
- Predict exactly 3 next actions, ranked by likelihood.
- Each action should be a single sentence describing what the person does next.
- Use the same language as the context (Chinese for EgoLife, English for Next-Me).
- Format each prediction as: [estimated_start -> estimated_end] action description
- Output ONLY the 3 predictions, one per line, numbered 1-3."""

def load_benchmark(path: Path, n: int = None, offset: int = 0):
    points = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                points.append(json.loads(line))
    if offset:
        points = points[offset:]
    if n:
        points = points[:n]
    return points

def build_prompt(point: dict) -> str:
    context = "\n".join(point["context"])
    return PREDICT_PROMPT.format(context=context)

def parse_predictions(raw: str) -> list:
    """Extract up to 3 action predictions from model output."""
    lines = raw.strip().split('\n')
    preds = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        num_match = re.match(r'^(\d+)[.)\]、]\s*(.*)', line)
        if not num_match:
            continue
        rest = num_match.group(2).strip()
        time_match = re.match(r'\[([^\]]+)\]\s*(.*)', rest)
        if time_match:
            rest = time_match.group(2).strip()
        if rest and not rest.startswith(('We ', 'Need ', 'This ')):
            preds.append(rest)
        if len(preds) >= 3:
            break
    return preds

def call_deepseek(prompt: str, api_key: str, model: str = "deepseek-v4-flash",
                  base_url: str = "https://api.deepseek.com/v1") -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"].get("content", "")
                reasoning = data["choices"][0]["message"].get("reasoning_content", "")
                return content or reasoning or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

def run_benchmark(args):
    points = load_benchmark(BENCHMARK_FILE, n=args.n, offset=args.offset)
    print(f"Loaded {len(points)} benchmark points")

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: No API key. Use --api-key or set DEEPSEEK_API_KEY env var")
        sys.exit(1)

    results = []
    errors = 0

    def process_point(pt):
        prompt = build_prompt(pt)
        raw = call_deepseek(prompt, api_key, model=args.model, base_url=args.base_url)
        preds = parse_predictions(raw)
        gt_texts = []
        for g in pt["ground_truth"]:
            m = re.match(r'\[[^\]]+\]\s*(.*)', g)
            gt_texts.append(m.group(1) if m else g)
        pred_texts = preds[:3]
        while len(pred_texts) < 3:
            pred_texts.append("")
        return {
            "id": pt["id"],
            "source": pt["source"],
            "participant": pt["participant"],
            "session": pt["session"],
            "prediction": pred_texts,
            "ground_truth": gt_texts,
            "raw_response": raw[:500],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_point, pt): pt for pt in points}
        for i, fut in enumerate(as_completed(futures)):
            pt = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                valid = sum(1 for p in result["prediction"] if p)
                print(f"  [{i+1}/{len(points)}] {result['id']}: {valid}/3 predictions OK")
            except Exception as e:
                errors += 1
                print(f"  [{i+1}/{len(points)}] {pt['id']}: ERROR {e}")

    results.sort(key=lambda x: x["id"])

    out_path = SCRIPT_DIR / args.output
    with open(out_path, 'w') as f:
        for r in results:
            row = {
                "id": r["id"],
                "prediction": r["prediction"],
                "ground_truth": r["ground_truth"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"\nDone: {len(results)} predictions, {errors} errors")
    print(f"Output: {out_path}")
    print(f"\nTo evaluate:")
    print(f"  python3 ../metric/eval_kit/evaluate.py -i {args.output} -c ../metric/eval_kit/config.json -m both")

    if args.save_full:
        full_path = SCRIPT_DIR / args.save_full
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full results: {full_path}")

def main():
    parser = argparse.ArgumentParser(description="Run benchmark predictions")
    parser.add_argument("--n", type=int, default=10, help="Number of points to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=5, help="Concurrent API workers")
    parser.add_argument("--output", default="predictions.jsonl", help="Output file")
    parser.add_argument("--save-full", default="predictions_full.json", help="Full results with raw response")
    args = parser.parse_args()
    run_benchmark(args)

if __name__ == "__main__":
    main()
