#!/usr/bin/env python3
"""
Run a model over the NextAct 1500 points and score best-of-3.

Points come from nextact_points.load_nextact(); prompts come from
prompt_builder (templates in prompts/). Scoring always takes the highest raw
score among the 3 candidate trajectories.

    python3 run_model_benchmark.py --model deepseek-v4-flash --ks 10 1
"""

import argparse, json, os, re, statistics, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nextact_points import load_nextact, strip_ts
from prompt_builder import build_prompt
from nextme_benchmark.scorer import BenchmarkScorer

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LEVELS = ["L1", "L2", "L3", "L4", "L5"]


def get_api_key(explicit=None, key_file=None):
    """Any OpenAI-compatible endpoint: --api-key > --api-key-file > env > default file."""
    if explicit:
        return explicit.strip()
    if key_file:
        return Path(key_file).expanduser().read_text().strip()
    for var in ("MODEL_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()
    default = Path.home() / ".config" / "deepseek" / "api_key"
    if default.exists():
        return default.read_text().strip()
    return ""


def parse_candidates(text, k):
    """Extract up to 3 trajectories of k actions from the model's JSON reply."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return []
    out = []
    for cand in obj.get("candidates", []):
        acts = [a for a in cand.get("actions", []) if isinstance(a, str) and a.strip()]
        if acts:
            out.append([strip_ts(a) for a in acts[:k]])
    return out


def predict_point(client, model, point, k, max_tokens, retries=3):
    system, user = build_prompt(point, k)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.7, max_tokens=max_tokens)
            msg = resp.choices[0].message
            text = (msg.content or "") or getattr(msg, "reasoning_content", "") or ""
            cands = parse_candidates(text, k)
            if cands:
                return {"id": point["id"], "source": point["source"],
                        "level": point["level"], "k": k,
                        "ground_truth": point["gt_text"][:k],
                        "all_predictions": cands}
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1 + attempt)
    return None


def run_k(client, model, points, k, concurrency, max_tokens):
    print(f"\n### Predicting k={k}: {len(points)} points (concurrency={concurrency})",
          flush=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(predict_point, client, model, p, k, max_tokens): p
                for p in points}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            if r:
                out.append(r)
            done += 1
            if done % 100 == 0 or done == len(points):
                print(f"  {done}/{len(points)} ({time.time()-t0:.0f}s, {len(out)} ok)",
                      flush=True)
    out.sort(key=lambda r: r["id"])
    return out


def summarize(scores):
    buckets = defaultdict(list)
    for s in scores:
        for key in ((s["source"], s["level"]), (s["source"], "ALL"),
                    ("ALL", s["level"]), ("ALL", "ALL")):
            buckets[key].append(s["normalized"])
    out = {}
    for key, vals in buckets.items():
        vals = [v for v in vals if v is not None]
        if vals:
            out[f"{key[0]}|{key[1]}"] = {
                "n": len(vals),
                "normalized": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }
    return out


def print_table(title, summaries, ks):
    header = f"{'Dataset':<10}{'Level':<8}" + "".join(f"{'k='+str(k):>22}" for k in ks)
    print("\n" + "=" * len(header))
    print(f"  {title}  (normalized: 0=random, 1=perfect)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = ([("nextme", lv) for lv in LEVELS] + [("nextme", "ALL")]
            + [("egolife", lv) for lv in LEVELS] + [("egolife", "ALL")]
            + [("ALL", lv) for lv in LEVELS] + [("ALL", "ALL")])
    for src, lv in rows:
        key = f"{src}|{lv}"
        cells, present, n = [], False, ""
        for k in ks:
            e = summaries.get(k, {}).get(key)
            if e:
                present = True
                n = e["n"]
                cells.append(f"{e['normalized']:>12.4f} ±{e['std']:<8.4f}")
            else:
                cells.append(f"{'—':>22}")
        if not present:
            continue
        if src == "ALL" and lv == "L1":
            print("-" * len(header))
        print(f"{src:<10}{lv:<8}" + "".join(cells) + f"   n={n}")
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com",
                    help="Any OpenAI-compatible endpoint")
    ap.add_argument("--api-key", help="API key (else --api-key-file or env)")
    ap.add_argument("--api-key-file", help="File containing the API key")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 1])
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--emb-concurrency", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    key = get_api_key(args.api_key, args.api_key_file)
    if not key:
        sys.exit("No API key: pass --api-key / --api-key-file, or set "
                 "MODEL_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY")
    client = OpenAI(api_key=key, base_url=args.base_url)

    points = load_nextact()
    if args.limit:
        points = points[:args.limit]
    print(f"NextAct points: {len(points)}")

    scorer = BenchmarkScorer(concurrency=args.emb_concurrency)
    tag = args.tag or args.model
    summaries, coverage = {}, {}

    for k in args.ks:
        preds = run_k(client, args.model, points, k, args.concurrency, args.max_tokens)
        pred_file = RESULTS_DIR / f"nextact_{tag}_k{k}_predictions.json"
        pred_file.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
        print(f"  Saved predictions: {pred_file}")

        print(f"  Scoring {len(preds)} points best-of-3...", flush=True)
        samples = [{"id": p["id"], "ground_truth": p["ground_truth"],
                    "all_predictions": p["all_predictions"],
                    "level": p["level"], "k": k} for p in preds]
        scores = scorer.score_batch(samples)
        for s, p in zip(scores, preds):
            s["source"] = p["source"]
        summaries[k] = summarize(scores)
        coverage[k] = f"{len(preds)}/{len(points)}"
        (RESULTS_DIR / f"nextact_{tag}_k{k}_scores.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2))

    print_table(f"{args.model} best-of-3 @ NextAct 1500", summaries, args.ks)
    print("Coverage: " + "  ".join(f"k={k}: {v}" for k, v in coverage.items()))

    (RESULTS_DIR / f"nextact_{tag}_summary.json").write_text(json.dumps({
        "model": args.model, "scoring": "best-of-3",
        "benchmark": "NextAct 1500 (NextMe 200/level + EgoLife 100/level, L1-L5)",
        "coverage": coverage,
        "summaries": {str(k): summaries[k] for k in args.ks},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
