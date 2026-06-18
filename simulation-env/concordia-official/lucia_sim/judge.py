"""LLM Judge evaluation for LUCIA simulation predictions.

Uses DeepSeek to score each (prediction, gt_action) pair on 4 dimensions:
- semantic (0-4): How close is the predicted action to the ground truth
- plausibility (0-2): Is the prediction reasonable regardless of GT match
- temporal (0-2): Time granularity match
- granularity (0-2): Detail level match

Usage:
    cd concordia-official
    source .venv/bin/activate
    PYTHONPATH=. python lucia_sim/judge.py lucia_sim/output/morning_11_09_15/predictions.json
"""

import os
import asyncio
import json
import re
import sys
import time

from openai import AsyncOpenAI


API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

MAX_CONCURRENT = 10

JUDGE_PROMPT = """你是行为预测评估专家。评估AI agent的行为预测准确度。

背景：AI扮演Lucia，预测她下一步做什么。预测可能是第三人称，GT是第一人称"我..."。忽略人称差异，只比较核心动作。

4个维度：
semantic(0-4): 核心动作语义相似度。4=完全匹配 3=核心相同细节不同 2=相关但不同 1=同场景不同动作 0=无关
plausibility(0-2): 预测合理性。2=很合理 1=一般 0=不合理
temporal(0-2): 原子程度。2=单一原子动作 1=2-3个动作 0=长段复杂
granularity(0-2): 具体程度匹配。2=匹配 1=略偏 0=差距大

你必须在最后一行输出分数，格式严格如下（不要省略任何维度）：
scores: semantic=X plausibility=X temporal=X granularity=X

只输出这一行分数，不要解释。"""


def _parse_scores(txt: str) -> dict:
    """Parse scores from model output in either JSON or key=value format."""
    dims = ["semantic", "plausibility", "temporal", "granularity"]
    # Try JSON first
    m = re.search(r'\{[^}]+\}', txt)
    if m:
        try:
            obj = json.loads(m.group())
            if all(d in obj for d in dims):
                return {d: int(obj[d]) for d in dims}
        except (json.JSONDecodeError, ValueError):
            pass
    # Try key=value format: "semantic=2 plausibility=1 ..."
    # Also match Chinese patterns like "semantic应该给2" "semantic是2"
    scores = {}
    for dim in dims:
        m = re.search(rf'{dim}\s*[=:：应该给是为]\s*(\d+)', txt)
        if m:
            scores[dim] = int(m.group(1))
    if len(scores) == len(dims):
        return scores
    # Try finding the last occurrence of each dim (final answer after reasoning)
    if len(scores) < len(dims):
        for dim in dims:
            if dim not in scores:
                matches = re.findall(rf'{dim}\s*[=:：应该给是为]\s*(\d+)', txt)
                if matches:
                    scores[dim] = int(matches[-1])
    if len(scores) == len(dims):
        return scores
    # Fill missing with 0 if at least one was found
    if scores:
        for dim in dims:
            scores.setdefault(dim, 0)
        return scores
    return {d: 0 for d in dims} | {"error": txt}


async def judge_one(client, pred_item, semaphore):
    async with semaphore:
        user_msg = (
            f"模型预测：{pred_item['prediction']}\n"
            f"真实发生：{pred_item['gt_action']}"
        )
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            msg = resp.choices[0].message
            txt = (msg.content or "").strip()
            if not txt:
                txt = (getattr(msg, "reasoning_content", None) or "").strip()
            scores = _parse_scores(txt)
        except Exception as e:
            scores = {"semantic": 0, "plausibility": 0, "temporal": 0,
                      "granularity": 0, "error": str(e)}
        return {"step": pred_item["step"], "timestamp": pred_item.get("timestamp", ""),
                "prediction": pred_item["prediction"], "gt_action": pred_item["gt_action"],
                "scores": scores}


async def run_judge(predictions):
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    t0 = time.time()
    results = await asyncio.gather(
        *[judge_one(client, p, semaphore) for p in predictions]
    )
    elapsed = time.time() - t0
    results.sort(key=lambda x: x["step"])

    # Aggregate
    dims = ["semantic", "plausibility", "temporal", "granularity"]
    totals = {d: 0 for d in dims}
    n = len(results)
    for r in results:
        for d in dims:
            totals[d] += r["scores"].get(d, 0)

    avgs = {d: totals[d] / n if n else 0 for d in dims}
    max_scores = {"semantic": 4, "plausibility": 2, "temporal": 2, "granularity": 2}
    pcts = {d: avgs[d] / max_scores[d] * 100 for d in dims}

    print(f"\n{'='*60}")
    print(f"LLM Judge Results ({n} predictions, {elapsed:.1f}s)")
    print(f"{'='*60}")
    for d in dims:
        print(f"  {d:14s}: {avgs[d]:.2f} / {max_scores[d]}  ({pcts[d]:.1f}%)")
    overall = sum(avgs[d] / max_scores[d] for d in dims) / len(dims) * 100
    print(f"  {'overall':14s}: {overall:.1f}%")
    print(f"{'='*60}")

    return results, avgs


def main():
    if len(sys.argv) < 2:
        print("Usage: python judge.py <predictions.json>")
        sys.exit(1)

    pred_path = sys.argv[1]
    with open(pred_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        predictions = data
    else:
        predictions = data.get("predictions", data)

    predictions = [p for p in predictions if p.get("prediction")]

    print(f"Judging {len(predictions)} predictions from {pred_path}...")
    results, avgs = asyncio.run(run_judge(predictions))

    # Save results
    output_path = pred_path.replace(".json", "_judged.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "averages": avgs,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
