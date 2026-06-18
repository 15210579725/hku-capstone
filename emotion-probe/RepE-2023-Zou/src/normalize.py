"""交付物⑤：两套归一化。

A) 当天时间 z-score：每 day、每情绪列独立 (x-μ)/σ（σ=0 置 0）→ 画日内曲线。
B) 刺激分布分位映射 [0,1]：用经验 CDF 把原始分映射到 [0,1]。
   刺激分布来源：mock 阶段用全体窗口分数近似；GPU 阶段换 Agent B 抽向量刺激打分分布
   （ref_distribution 参数，默认 self）。
两套都输出后缀 _z / _q，写回 scores。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def add_zscore(sdf: pd.DataFrame, emotions=None) -> pd.DataFrame:
    """每 day、每情绪列独立 z-score → 新增 {emo}_z 列。"""
    emotions = emotions or config.EMOTIONS
    out = sdf.copy()
    for e in emotions:
        out[f"{e}_z"] = 0.0
    for day, idx in out.groupby("day").groups.items():
        for e in emotions:
            vals = out.loc[idx, e].astype(float)
            mu, sd = vals.mean(), vals.std(ddof=0)
            out.loc[idx, f"{e}_z"] = (vals - mu) / sd if sd > 0 else 0.0
    return out


def quantile_map(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """经验 CDF 分位映射到 [0,1]：每个 x 在 ref 分布中的分位。"""
    ref_sorted = np.sort(np.asarray(ref, dtype=float))
    n = len(ref_sorted)
    if n == 0:
        return np.zeros_like(x, dtype=float)
    # searchsorted 右插：排名 / n
    ranks = np.searchsorted(ref_sorted, np.asarray(x, dtype=float), side="right")
    return ranks / n


def add_quantile(sdf: pd.DataFrame, emotions=None, ref_distribution: dict = None) -> pd.DataFrame:
    """每情绪列分位映射 → 新增 {emo}_q ∈ [0,1]。

    ref_distribution: {emo: ref array}；默认用本列全体（self）。
    """
    emotions = emotions or config.EMOTIONS
    out = sdf.copy()
    for e in emotions:
        ref = ref_distribution[e] if (ref_distribution and e in ref_distribution) else out[e].values
        out[f"{e}_q"] = quantile_map(out[e].values, ref)
    return out


def normalize(sdf: pd.DataFrame, ref_distribution: dict = None,
              write: bool = True, tag: str = "mock",
              level: str = config.DEFAULT_LEVEL, mode: str = None) -> pd.DataFrame:
    mode = mode or config.WINDOW_MODE
    out = add_zscore(sdf)
    out = add_quantile(out, ref_distribution=ref_distribution)
    if write:
        path = config.artifact(f"scores_norm_{level}_{mode}_{tag}")
        try:
            out.to_parquet(path + ".parquet", index=False)
        except Exception:
            out.to_csv(path + ".csv", index=False)
    return out


if __name__ == "__main__":
    from src import direction_artifact, windowize, score
    art = direction_artifact.make_mock_artifact()
    wdf = windowize.build_windows(write=False)
    prov = score.MockResidualProvider(art, inject_strength=1.6)
    sdf = score.score_windows(wdf, art, prov, write=False)
    out = normalize(sdf, write=False)
    # 自测：z 每天每列均值≈0 std≈1；q ∈ [0,1]
    for e in config.EMOTIONS:
        for day, g in out.groupby("day"):
            z = g[f"{e}_z"]
            if z.std(ddof=0) > 0:
                assert abs(z.mean()) < 1e-6, f"{e} day{day} z mean !=0"
                assert abs(z.std(ddof=0) - 1) < 1e-6, f"{e} day{day} z std !=1"
        q = out[f"{e}_q"]
        assert (q >= 0).all() and (q <= 1).all(), f"{e}_q out of [0,1]"
    print("normalize self-test passed; z mean~0 std~1, q in [0,1]")
    print("example happiness_z range:", round(out['happiness_z'].min(), 3),
          round(out['happiness_z'].max(), 3),
          "| q range:", round(out['happiness_q'].min(), 3), round(out['happiness_q'].max(), 3))
