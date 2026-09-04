#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行为预测标准评测工具 (Behavior Prediction Evaluation Toolkit)

评测模型对未来 K 步行为的预测质量，支持:
  1. Embedding 方法 — 嵌入余弦相似度 + Soft Edit Distance
  2. LLM Judge 方法 — 大模型语义评分 (0-9) + Soft Edit Distance

用法:
  python evaluate.py -i predictions.jsonl -c config.json -m both
  python evaluate.py -i predictions.jsonl -c config.json -m embedding
  python evaluate.py -i predictions.jsonl -c config.json -m judge
  python evaluate.py -i predictions.jsonl --dry-run   # 仅验证输入格式
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Prompts ───────────────────────────────────────────────────────

EMBEDDING_INSTRUCTION = (
    "Represent the sentence primarily by its core action verb and "
    "the object being acted on. Prioritize verb-object alignment and "
    "downweight modifiers, tense, style, and incidental context."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert judge evaluating semantic similarity between "
    "a ground-truth human action and a predicted action.\n\n"
    "[Rubric (0-9)]\n"
    "9: Perfect — intent, verb, and noun all match (synonyms count).\n"
    "7-8: Strong — correct verb, slightly generalized noun within same category.\n"
    "4-6: Moderate — prerequisite action or correct location but incomplete intent.\n"
    "2-3: Marginal — only broad spatial context matches.\n"
    "0-1: No match — unrelated action or contradicts ground truth.\n\n"
    "[Rules]\n"
    "- Focus on core action verb + target object + underlying intent.\n"
    "- Ignore writing style, tense, tone, sentence structure differences.\n"
    "- Synonyms and paraphrases of the same action = high score.\n\n"
    "Output ONLY a single integer 0-9. No explanation."
)

# ─── I/O Utilities ─────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析错误 {path}:{i}: {e}")
    return rows


def write_jsonl(path: Path, rows: list):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ─── Input Validation ──────────────────────────────────────────────

def normalize_samples(raw_samples: List[dict]) -> List[dict]:
    samples = []
    ids = set()
    for i, s in enumerate(raw_samples):
        sid = s.get("id", s.get("point_id", f"sample_{i+1}"))
        if sid in ids:
            raise ValueError(f"重复 id: '{sid}'")
        ids.add(sid)

        gt = s.get("ground_truth", s.get("gt"))
        pred = s.get("prediction", s.get("pred"))
        if gt is None or pred is None:
            raise ValueError(f"样本 '{sid}': 缺少 ground_truth/prediction 字段")

        if isinstance(gt, str):
            gt = [gt]
        elif isinstance(gt, list) and gt and isinstance(gt[0], dict):
            gt = [item.get("text", str(item)) for item in gt]
        if isinstance(pred, str):
            pred = [pred]
        elif isinstance(pred, list) and pred and isinstance(pred[0], dict):
            pred = [item.get("text", str(item)) for item in pred]

        gt = [str(x).strip() for x in gt]
        pred = [str(x).strip() for x in pred]
        if not gt or not pred:
            raise ValueError(f"样本 '{sid}': ground_truth/prediction 不能为空")
        if any(not x for x in gt) or any(not x for x in pred):
            raise ValueError(f"样本 '{sid}': 包含空字符串")

        samples.append({
            "id": str(sid),
            "ground_truth": gt,
            "prediction": pred,
            "context": s.get("context", ""),
        })
    return samples


# ─── API: Embedding ────────────────────────────────────────────────

def _embedding_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1"):
        return base + "/embeddings"
    return base + "/v1/embeddings"


def call_embedding_api(
    texts: List[str],
    config: dict,
    timeout: int = 120,
    max_attempts: int = 4,
) -> np.ndarray:
    url = _embedding_endpoint(config["base_url"])
    instruction = config.get("instruction", EMBEDDING_INSTRUCTION)
    dimensions = config.get("dimensions", 4096)

    if instruction:
        formatted = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    else:
        formatted = texts

    payload: Dict[str, Any] = {
        "model": config["model"],
        "input": formatted,
        "encoding_format": "float",
    }
    if dimensions:
        payload["dimensions"] = dimensions
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.load(resp)
            data = sorted(result.get("data", []), key=lambda x: x.get("index", -1))
            matrix = np.array([d["embedding"] for d in data], dtype=np.float32)
            if matrix.shape[0] != len(texts):
                raise RuntimeError(f"返回向量数 {matrix.shape[0]} != 请求数 {len(texts)}")
            return matrix
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            error_msg = f"HTTP {e.code}: {detail}"
            if e.code not in {408, 429, 500, 502, 503, 504} or attempt == max_attempts:
                raise RuntimeError(f"Embedding API 调用失败: {error_msg}")
            delay = min(30.0, 2 ** (attempt - 1) + random.random())
            print(f"  embedding 第 {attempt}/{max_attempts} 次失败，{delay:.0f}s 后重试: {error_msg}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        except Exception as e:
            if attempt < max_attempts:
                delay = min(30.0, 2 ** (attempt - 1) + random.random())
                print(f"  embedding 第 {attempt}/{max_attempts} 次失败，{delay:.0f}s 后重试: {e}",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
            else:
                raise RuntimeError(f"Embedding API 调用失败 ({max_attempts} 次): {e}")
    raise AssertionError("unreachable")


# ─── API: LLM Judge ───────────────────────────────────────────────

def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _parse_judge_score(raw: str) -> Optional[int]:
    m = re.search(r"^\s*(\d)\s*$", raw, re.MULTILINE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b(\d)\b", raw)
    if nums:
        return int(nums[-1])
    return None


def call_judge_api(
    gt: str,
    pred: str,
    config: dict,
    context: str = "",
    timeout: int = 60,
    max_attempts: int = 3,
) -> Tuple[Optional[int], str, float]:
    url = _chat_endpoint(config["base_url"])
    parts = []
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Ground truth (gt): {gt}")
    parts.append(f"Prediction (pred): {pred}")
    user_msg = "\n".join(parts)

    payload: Dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": config.get("max_tokens", 1024),
        "temperature": 0.1,
    }
    if config.get("extra_body"):
        payload.update(config["extra_body"])
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        })
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.load(resp)
            elapsed = time.time() - t0
            msg = result["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content and msg.get("reasoning_content"):
                content = msg["reasoning_content"].strip()
            score = _parse_judge_score(content)
            if score is not None:
                return score, "", elapsed
            return None, f"parse_failed: {content[:100]}", elapsed
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            if attempt < max_attempts:
                time.sleep(2 * attempt)
            else:
                return None, f"HTTP {e.code}: {detail}", 0.0
        except Exception as e:
            if attempt < max_attempts:
                time.sleep(2 * attempt)
            else:
                return None, str(e)[:200], 0.0
    return None, "unreachable", 0.0


# ─── Soft Edit Distance ───────────────────────────────────────────

def soft_edit_distance(len_a: int, len_b: int, cost_matrix: List[List[float]]) -> float:
    """
    Soft Edit Distance (SED)。
    cost_matrix[i][j] = 替换代价, 范围 [0, 1] (0=完全匹配, 1=完全不同)。
    插入/删除代价固定为 1.0。
    """
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


def sed_score(gt_len: int, pred_len: int, cost_matrix: List[List[float]]) -> float:
    """1 - normalized SED。范围 [0, 1]，越高越好。"""
    sed = soft_edit_distance(gt_len, pred_len, cost_matrix)
    max_len = max(gt_len, pred_len)
    return 1.0 - sed / max_len if max_len > 0 else 1.0


# ─── Embedding Evaluator ──────────────────────────────────────────

def evaluate_embedding(samples: List[dict], config: dict) -> List[dict]:
    all_texts = list(dict.fromkeys(
        t for s in samples for t in s["ground_truth"] + s["prediction"]
    ))
    text_to_idx = {t: i for i, t in enumerate(all_texts)}

    batch_size = min(config.get("batch_size", 32), 64)
    dimensions = config.get("dimensions", 4096)
    total_batches = math.ceil(len(all_texts) / batch_size)
    embeddings = np.empty((len(all_texts), dimensions), dtype=np.float32)

    print(f"  共 {len(all_texts)} 个唯一文本，分 {total_batches} 批请求嵌入...")
    for b in range(total_batches):
        start = b * batch_size
        batch_texts = all_texts[start : start + batch_size]
        sys.stderr.write(f"\r  嵌入 batch {b+1}/{total_batches}")
        sys.stderr.flush()
        matrix = call_embedding_api(batch_texts, config)
        actual_dim = matrix.shape[1]
        if b == 0 and actual_dim != dimensions:
            dimensions = actual_dim
            embeddings = np.empty((len(all_texts), dimensions), dtype=np.float32)
        embeddings[start : start + len(batch_texts)] = matrix
    sys.stderr.write("\n")

    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings.astype(np.float64) / norms

    results = []
    for s in samples:
        gt, pred = s["ground_truth"], s["prediction"]
        cosine_matrix: List[List[float]] = []
        cost_matrix: List[List[float]] = []
        for g in gt:
            cos_row, cost_row = [], []
            for p in pred:
                gi, pi = text_to_idx[g], text_to_idx[p]
                cosine = float(np.dot(normalized[gi], normalized[pi]))
                cosine = max(-1.0, min(1.0, cosine))
                cos_row.append(round(cosine, 4))
                cost_row.append(1.0 - cosine)
            cosine_matrix.append(cos_row)
            cost_matrix.append(cost_row)

        score = sed_score(len(gt), len(pred), cost_matrix)
        sed_val = soft_edit_distance(len(gt), len(pred), cost_matrix)

        results.append({
            "id": s["id"],
            "score": round(score, 4),
            "soft_edit_distance": round(sed_val, 4),
            "cosine_matrix": cosine_matrix,
        })
    return results


# ─── LLM Judge Evaluator ──────────────────────────────────────────

def evaluate_judge(samples: List[dict], config: dict, concurrency: int = 10) -> List[dict]:
    pairs: List[Tuple[str, str, str]] = []
    pair_key_to_idx: Dict[Tuple[str, str], int] = {}
    for s in samples:
        ctx = s.get("context", "")
        for g in s["ground_truth"]:
            for p in s["prediction"]:
                key = (g, p)
                if key not in pair_key_to_idx:
                    pair_key_to_idx[key] = len(pairs)
                    pairs.append((g, p, ctx))

    scores: List[Optional[int]] = [None] * len(pairs)
    errors: List[str] = [""] * len(pairs)
    done = [0]
    total = len(pairs)

    def _run(idx: int, gt: str, pred: str, ctx: str):
        s, err, elapsed = call_judge_api(gt, pred, config, context=ctx)
        return idx, s, err

    print(f"  共 {total} 个 (gt, pred) 对需要评分，并发数={concurrency}...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(_run, i, g, p, c): i for i, (g, p, c) in enumerate(pairs)}
        for f in as_completed(futs):
            idx, s, err = f.result()
            scores[idx] = s
            errors[idx] = err
            done[0] += 1
            ok = sum(1 for x in scores[:done[0]] if x is not None)
            sys.stderr.write(f"\r  Judge [{done[0]}/{total}] ok={sum(1 for x in scores if x is not None)}")
            sys.stderr.flush()
    sys.stderr.write("\n")

    n_fail = sum(1 for s in scores if s is None)
    if n_fail:
        print(f"  ⚠ {n_fail}/{total} 对评分失败", file=sys.stderr)

    results = []
    for s in samples:
        gt, pred = s["ground_truth"], s["prediction"]
        judge_matrix: List[List[Optional[int]]] = []
        cost_matrix: List[List[float]] = []
        has_error = False
        for g in gt:
            j_row: List[Optional[int]] = []
            c_row: List[float] = []
            for p in pred:
                idx = pair_key_to_idx[(g, p)]
                judge_score = scores[idx]
                if judge_score is None:
                    has_error = True
                    j_row.append(None)
                    c_row.append(1.0)
                else:
                    j_row.append(judge_score)
                    c_row.append(1.0 - judge_score / 9.0)
            judge_matrix.append(j_row)
            cost_matrix.append(c_row)

        score = sed_score(len(gt), len(pred), cost_matrix)
        sed_val = soft_edit_distance(len(gt), len(pred), cost_matrix)

        result: Dict[str, Any] = {
            "id": s["id"],
            "score": round(score, 4),
            "soft_edit_distance": round(sed_val, 4),
            "judge_matrix": judge_matrix,
        }
        if has_error:
            result["has_error"] = True
        results.append(result)
    return results


# ─── Summary Statistics ────────────────────────────────────────────

def compute_summary(results: List[dict], method: str) -> dict:
    scores = np.array([r["score"] for r in results], dtype=np.float64)
    seds = np.array([r["soft_edit_distance"] for r in results], dtype=np.float64)
    n_errors = sum(1 for r in results if r.get("has_error"))
    summary: Dict[str, Any] = {
        "method": method,
        "n_samples": len(results),
        "score_mean": round(float(scores.mean()), 4),
        "score_std": round(float(scores.std()), 4),
        "score_median": round(float(np.median(scores)), 4),
        "score_min": round(float(scores.min()), 4),
        "score_max": round(float(scores.max()), 4),
        "sed_mean": round(float(seds.mean()), 4),
    }
    if n_errors:
        summary["n_errors"] = n_errors
    return summary


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="行为预测标准评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py -i predictions.jsonl -c config.json -m embedding
  python evaluate.py -i predictions.jsonl -c config.json -m judge
  python evaluate.py -i predictions.jsonl -c config.json -m both
  python evaluate.py -i predictions.jsonl --dry-run
""",
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="输入 JSONL 文件")
    parser.add_argument("-c", "--config", type=Path,
                        default=Path(__file__).resolve().parent / "config.json",
                        help="配置文件 (默认: 同目录 config.json)")
    parser.add_argument("-m", "--method", choices=["embedding", "judge", "both"],
                        default="both", help="评测方法 (默认: both)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="输出目录 (默认: 输入文件同目录)")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖并发数")
    parser.add_argument("--dry-run", action="store_true", help="仅验证输入格式，不调 API")
    args = parser.parse_args()

    # ── 加载输入 ──
    raw_samples = load_jsonl(args.input)
    samples = normalize_samples(raw_samples)
    step_counts = set()
    for s in samples:
        step_counts.add(len(s["ground_truth"]))
        step_counts.add(len(s["prediction"]))
    max_k = max(step_counts)
    print(f"✓ 加载 {len(samples)} 条样本 (步数范围: {min(step_counts)}-{max_k})")

    if args.dry_run:
        print("✓ 输入格式验证通过 (--dry-run)")
        for s in samples[:3]:
            print(f"  {s['id']}: gt={len(s['ground_truth'])}步, pred={len(s['prediction'])}步")
        if len(samples) > 3:
            print(f"  ... (共 {len(samples)} 条)")
        return

    # ── 加载配置 ──
    config = load_config(args.config)
    concurrency = args.concurrency or config.get("concurrency", 10)
    output_dir = args.output_dir or args.input.resolve().parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Any] = {"input": str(args.input), "n_samples": len(samples)}
    combined: Dict[str, dict] = {}
    for s in samples:
        combined[s["id"]] = {
            "id": s["id"],
            "ground_truth": s["ground_truth"],
            "prediction": s["prediction"],
            "n_steps_gt": len(s["ground_truth"]),
            "n_steps_pred": len(s["prediction"]),
        }

    # ── Embedding 评测 ──
    if args.method in ("embedding", "both"):
        emb_cfg = config.get("embedding", {})
        if not emb_cfg.get("api_key"):
            print("\n⚠ embedding.api_key 未设置，跳过 embedding 评测", file=sys.stderr)
        else:
            print("\n══════ Embedding 评测 ══════")
            t0 = time.time()
            emb_results = evaluate_embedding(samples, emb_cfg)
            elapsed = time.time() - t0
            emb_summary = compute_summary(emb_results, "embedding")
            emb_summary["elapsed_seconds"] = round(elapsed, 1)
            summaries["embedding"] = emb_summary
            for r in emb_results:
                combined[r["id"]]["embedding"] = {
                    "score": r["score"],
                    "soft_edit_distance": r["soft_edit_distance"],
                    "cosine_matrix": r["cosine_matrix"],
                }
            print(f"  得分: {emb_summary['score_mean']:.4f} ± {emb_summary['score_std']:.4f}")
            print(f"  中位: {emb_summary['score_median']:.4f}  "
                  f"范围: [{emb_summary['score_min']:.4f}, {emb_summary['score_max']:.4f}]")
            print(f"  耗时: {elapsed:.1f}s")

    # ── LLM Judge 评测 ──
    if args.method in ("judge", "both"):
        judge_cfg = config.get("judge", {})
        if not judge_cfg.get("api_key"):
            print("\n⚠ judge.api_key 未设置，跳过 LLM Judge 评测", file=sys.stderr)
        else:
            print("\n══════ LLM Judge 评测 ══════")
            t0 = time.time()
            judge_results = evaluate_judge(samples, judge_cfg, concurrency)
            elapsed = time.time() - t0
            judge_summary = compute_summary(judge_results, "judge")
            judge_summary["elapsed_seconds"] = round(elapsed, 1)
            summaries["judge"] = judge_summary
            for r in judge_results:
                combined[r["id"]]["judge"] = {
                    "score": r["score"],
                    "soft_edit_distance": r["soft_edit_distance"],
                    "judge_matrix": r["judge_matrix"],
                }
                if r.get("has_error"):
                    combined[r["id"]]["judge"]["has_error"] = True
            print(f"  得分: {judge_summary['score_mean']:.4f} ± {judge_summary['score_std']:.4f}")
            print(f"  中位: {judge_summary['score_median']:.4f}  "
                  f"范围: [{judge_summary['score_min']:.4f}, {judge_summary['score_max']:.4f}]")
            if judge_summary.get("n_errors"):
                print(f"  ⚠ {judge_summary['n_errors']} 条评分失败")
            print(f"  耗时: {elapsed:.1f}s")

    # ── 写出结果 ──
    results_list = [combined[s["id"]] for s in samples]
    results_path = output_dir / "eval_results.jsonl"
    write_jsonl(results_path, results_list)

    summary_path = output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    print(f"\n═══ 结果 ═══")
    print(f"  详细: {results_path}")
    print(f"  汇总: {summary_path}")

    if "embedding" in summaries and "judge" in summaries:
        e = summaries["embedding"]["score_mean"]
        j = summaries["judge"]["score_mean"]
        print(f"\n  Embedding 均分: {e:.4f}")
        print(f"  Judge 均分:     {j:.4f}")


if __name__ == "__main__":
    main()
