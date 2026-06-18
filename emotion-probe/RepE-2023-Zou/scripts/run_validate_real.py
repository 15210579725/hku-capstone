"""scripts/run_validate_real.py — 本地：真实分归一化 + 金标准验证 + 出图。

读 artifacts/scores_L2_count_real.parquet（GPU 真实打分）→ normalize（z + 分位）
→ validate（对齐 feelings_all_days 金标准，命中率/相关/κ/分歧 top-K）
→ 与 random baseline 对照（同方向、随机残差打分）
→ visualize（day1 六情绪真实曲线叠金标准事件点 + 分歧表）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import (normalize, parse_feelings, validate, visualize,
                 direction_artifact, windowize, score)  # noqa: E402


def main():
    real_path = config.artifact("scores_L2_count_real.parquet")
    sdf_real = pd.read_parquet(real_path)
    # 契约别名：也存一份 scores_L2_real.parquet
    sdf_real.to_parquet(config.artifact("scores_L2_real.parquet"), index=False)
    print(f"[validate_real] loaded {len(sdf_real)} real-scored windows", flush=True)

    feel = parse_feelings.parse_feelings(write=False)

    # --- 真实分：归一化 + 验证 ---
    norm_real = normalize.normalize(sdf_real, write=True, tag="real")
    res_real = validate.run_validation(feel, norm_real, tag="real", write=True)
    m_real = res_real["metrics"]

    # --- random baseline：同窗、随机残差、同方向 artifact 打分 ---
    art = direction_artifact.make_mock_artifact(
        hidden_dim=2048, write=False)  # 随机正交方向
    wdf = windowize.build_windows(write=False)
    prov_rnd = score.MockResidualProvider(art, inject_strength=0.0)  # 纯随机
    sdf_rnd = score.score_windows(wdf, art, prov_rnd, write=False, tag="random")
    norm_rnd = normalize.normalize(sdf_rnd, write=False, tag="random")
    res_rnd = validate.run_validation(feel, norm_rnd, tag="random", write=True)
    m_rnd = res_rnd["metrics"]

    # --- 对照表 ---
    keys = ["strict_hit_rate", "qualitative_hit_rate", "polarity_pearson",
            "polarity_spearman", "binary_accuracy", "cohen_kappa",
            "n_aligned", "strict_n"]
    compare = {k: {"real": m_real.get(k), "random": m_rnd.get(k)} for k in keys}
    with open(config.artifact("validation_compare_real_vs_random.json"), "w",
              encoding="utf-8") as f:
        json.dump({"real": m_real, "random": m_rnd, "compare": compare}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== REAL vs RANDOM ===")
    for k in keys:
        print(f"  {k:24s} real={compare[k]['real']}  random={compare[k]['random']}")

    # --- 出图（真实分）---
    p1 = visualize.plot_day_curves(norm_real, feel, day=1)
    aligned = validate.align_feelings_to_windows(feel, norm_real)
    top = validate.divergence_topk(aligned, write=True)
    p2 = visualize.plot_divergence_table(top)
    print(f"\n[validate_real] figures: {p1} | {p2}", flush=True)
    return compare


if __name__ == "__main__":
    main()
