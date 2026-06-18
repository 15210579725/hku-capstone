"""scripts/run_score_real.py — GPU 用 Qwen2.5-3B 对窗口真实打分 → scores_L2_real.parquet。

conda activate qwen && python scripts/run_score_real.py \
    --artifact outputs/directions/emotion_vectors.pt \
    --windows artifacts/windows_L2_count.parquet \
    --model /root/autodl-tmp/Qwen2.5-3B-Instruct \
    [--day 1]   # 不给 day 则全 7 天
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=config.AGENT_B_DIRECTIONS)
    ap.add_argument("--windows", default=config.artifact("windows_L2_count.parquet"))
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--day", type=int, default=None, help="只打某天；省略=全部")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_recenter", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    payload = score.load_nested_artifact(args.artifact)
    print(f"[score_real] artifact model={payload['model_name']} "
          f"best_layer={payload['best_layer']}", flush=True)

    wdf = pd.read_parquet(args.windows)
    if args.day is not None:
        wdf = wdf[wdf["day"] == args.day].reset_index(drop=True)
    print(f"[score_real] scoring {len(wdf)} windows "
          f"(day={args.day or 'ALL'})", flush=True)

    provider = score.RealResidualProvider(
        artifact=payload,
        model_id=args.model,
        chat_template_signature=payload.get("chat_template_signature"),
        lang=payload.get("stimulus_lang", "zh"),
        batch_size=args.batch_size,
    )

    tag = args.tag or ("real_day%d" % args.day if args.day is not None else "real")
    sdf = score.score_windows_real(
        wdf, payload, provider,
        recenter=not args.no_recenter,
        write=True, tag=tag, level=config.DEFAULT_LEVEL, mode="count",
    )
    print(f"[score_real] done: {len(sdf)} rows, tag={tag}", flush=True)
    print(sdf[config.EMOTIONS].describe().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
