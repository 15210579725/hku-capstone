#!/usr/bin/env python3
"""
NextMe Behavior Prediction Scorer.

Computes Soft Edit Distance (SED) between ground-truth and predicted action
sequences using embedding-based cosine cost, then normalizes against
per-level random baselines.

Usage:
    from nextme_benchmark.scorer import BenchmarkScorer
    scorer = BenchmarkScorer()
    result = scorer.score(ground_truth=["I cook dinner", "I eat dinner"],
                          prediction=["I prepare food", "I have a meal"],
                          level="L3")
    print(result)  # {"raw": 0.62, "normalized": 0.37, "level": "L3", ...}
"""

import json, math, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import numpy as np

# ─── eval_kit import ──────────────────────────────────────────────────
_EVAL_KIT_DIR = Path(__file__).resolve().parent.parent.parent / "metric" / "eval_kit"
if str(_EVAL_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_KIT_DIR))
from evaluate import load_config, call_embedding_api

# ─── Default paths ────────────────────────────────────────────────────
_DEFAULT_EVAL_CONFIG = Path(os.path.expanduser("~/.config/hku-capstone/eval_config.json"))
_BASELINES_FILE = Path(__file__).resolve().parent / "baselines.json"


# ─── SED core ─────────────────────────────────────────────────────────

def cosine_cost_matrix(gt_embs: np.ndarray, pr_embs: np.ndarray) -> List[List[float]]:
    """Build cost matrix: cost[i][j] = 1 - cosine(gt_i, pr_j), clamped to [0, 1]."""
    # gt_embs: (M, D), pr_embs: (N, D), both L2-normalized
    sim = gt_embs @ pr_embs.T  # (M, N)
    cost = 1.0 - sim
    cost = np.clip(cost, 0.0, 1.0)
    return cost.tolist()


def soft_edit_distance(len_a: int, len_b: int, cost_matrix: List[List[float]]) -> float:
    """SED with substitution cost from cost_matrix, insert/delete cost = 1.0."""
    dp = [[0.0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = float(i)
    for j in range(len_b + 1):
        dp[0][j] = float(j)
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost_matrix[i - 1][j - 1],
            )
    return dp[len_a][len_b]


def sed_score(gt_len: int, pr_len: int, cost_matrix: List[List[float]]) -> float:
    """1 - normalized SED. Range [0, 1], higher is better."""
    sed = soft_edit_distance(gt_len, pr_len, cost_matrix)
    max_len = max(gt_len, pr_len)
    return 1.0 - sed / max_len if max_len > 0 else 1.0


# ─── Embedding helper ─────────────────────────────────────────────────

def embed_texts(texts: List[str], emb_cfg: dict, concurrency: int = 4) -> np.ndarray:
    """Embed texts → (N, D) L2-normalized float64 array."""
    batch_size = min(emb_cfg.get("batch_size", 32), 8)
    total_batches = math.ceil(len(texts) / batch_size)
    batch_results = [None] * total_batches

    def _fetch(b):
        start = b * batch_size
        batch = texts[start:start + batch_size]
        matrix = call_embedding_api(batch, emb_cfg, timeout=180, max_attempts=6)
        return b, start, matrix

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(_fetch, b): b for b in range(total_batches)}
        done_count = 0
        for f in as_completed(futs):
            b, start, matrix = f.result()
            batch_results[b] = (start, matrix)
            done_count += 1
            if done_count % 50 == 0 or done_count == total_batches:
                print(f"  Embedding: {done_count}/{total_batches} batches", flush=True)

    dim = batch_results[0][1].shape[1]
    embeddings = np.empty((len(texts), dim), dtype=np.float64)
    for start, matrix in batch_results:
        embeddings[start:start + matrix.shape[0]] = matrix

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return embeddings / norms


# ─── BenchmarkScorer ──────────────────────────────────────────────────

class BenchmarkScorer:
    """Score predictions against ground truth with random-baseline normalization."""

    def __init__(self, config_path: Optional[str] = None,
                 baselines_path: Optional[str] = None):
        cfg_path = Path(config_path) if config_path else _DEFAULT_EVAL_CONFIG
        self.config = load_config(cfg_path)
        self.emb_cfg = self.config.get("embedding", {})
        self.concurrency = self.config.get("concurrency", 64)

        bl_path = Path(baselines_path) if baselines_path else _BASELINES_FILE
        if bl_path.exists():
            self.baselines = json.loads(bl_path.read_text())
        else:
            self.baselines = {}

        self._emb_cache: Dict[str, np.ndarray] = {}

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embed with dedup cache."""
        new_texts = [t for t in texts if t not in self._emb_cache]
        if new_texts:
            unique = list(dict.fromkeys(new_texts))
            embs = embed_texts(unique, self.emb_cfg, self.concurrency)
            for t, e in zip(unique, embs):
                self._emb_cache[t] = e
        return np.array([self._emb_cache[t] for t in texts])

    def score_single(self, ground_truth: List[str], prediction: List[str],
                     level: str, k: Optional[int] = None) -> dict:
        """
        Score one (GT, PR) pair.

        Args:
            ground_truth: list of GT action texts
            prediction: list of predicted action texts
            level: "L1" through "L5"
            k: prediction length for baseline lookup (default: len(prediction))

        Returns:
            {"raw": float, "normalized": float, "level": str, "k": int,
             "gt_len": int, "pr_len": int, "random_baseline": float}
        """
        if k is None:
            k = len(prediction)

        all_texts = list(dict.fromkeys(ground_truth + prediction))
        embs = self._get_embeddings(all_texts)
        idx = {t: i for i, t in enumerate(all_texts)}

        gt_emb = embs[[idx[t] for t in ground_truth]]
        pr_emb = embs[[idx[t] for t in prediction]]

        cost = cosine_cost_matrix(gt_emb, pr_emb)
        raw = sed_score(len(ground_truth), len(prediction), cost)

        # Normalize
        baseline_key = f"{level}_k{k}"
        rand_baseline = self.baselines.get(baseline_key, {}).get("random_sed")
        if rand_baseline is not None and rand_baseline < 1.0:
            normalized = max(0.0, (raw - rand_baseline) / (1.0 - rand_baseline))
        else:
            normalized = None

        return {
            "raw": round(raw, 4),
            "normalized": round(normalized, 4) if normalized is not None else None,
            "level": level,
            "k": k,
            "gt_len": len(ground_truth),
            "pr_len": len(prediction),
            "random_baseline": rand_baseline,
        }

    def score_batch(self, samples: List[dict]) -> List[dict]:
        """
        Score a batch of samples.

        Each sample: {"ground_truth": [...], "prediction": [...], "level": "L3",
                      "k": 3, "id": "optional_id"}

        Returns list of score dicts (same order), with "id" preserved.
        """
        # Collect all texts for batch embedding
        all_texts = []
        for s in samples:
            all_texts.extend(s["ground_truth"])
            all_texts.extend(s["prediction"])
        all_texts = list(dict.fromkeys(all_texts))

        if all_texts:
            self._get_embeddings(all_texts)

        results = []
        for s in samples:
            r = self.score_single(
                s["ground_truth"], s["prediction"],
                s["level"], s.get("k"))
            if "id" in s:
                r["id"] = s["id"]
            results.append(r)
        return results

    def score(self, ground_truth: List[str], prediction: List[str],
              level: str, k: Optional[int] = None) -> dict:
        """Alias for score_single."""
        return self.score_single(ground_truth, prediction, level, k)

    def clear_cache(self):
        """Clear embedding cache."""
        self._emb_cache.clear()


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    """Quick test: score a sample pair from stdin or args."""
    import argparse
    parser = argparse.ArgumentParser(description="Score GT vs PR action sequences")
    parser.add_argument("-i", "--input", help="JSONL file with {ground_truth, prediction, level, k} per line")
    parser.add_argument("--config", help="Eval config path")
    parser.add_argument("--baselines", help="Baselines JSON path")
    args = parser.parse_args()

    scorer = BenchmarkScorer(args.config, args.baselines)

    if args.input:
        samples = []
        with open(args.input) as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"Scoring {len(samples)} samples...")
        results = scorer.score_batch(samples)
        for r in results:
            print(json.dumps(r, ensure_ascii=False))

        # Summary
        raws = [r["raw"] for r in results]
        norms = [r["normalized"] for r in results if r["normalized"] is not None]
        print(f"\nRaw SED:    {np.mean(raws):.4f} ± {np.std(raws):.4f}")
        if norms:
            print(f"Normalized: {np.mean(norms):.4f} ± {np.std(norms):.4f}")
    else:
        print("Usage: python scorer.py -i samples.jsonl")
        print("Each line: {\"ground_truth\": [...], \"prediction\": [...], \"level\": \"L3\", \"k\": 3}")


if __name__ == "__main__":
    main()
