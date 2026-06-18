"""交付物⑥：验证。把探针窗口按时间戳对齐金标准时间线，算指标。

- 对齐：每条自陈用 mid_time 找覆盖窗（t_start<=mid<=t_end）；多窗取最近中点；无覆盖记 miss。
- 情绪时刻命中率：自陈映射到情绪 e 的时刻，对应窗的 e_z 是否 > 阈值（默认 0.5σ）→ 命中率。
- 极性相关：探针极性分(正情绪和−负情绪和) vs 自陈极性，Pearson + Spearman。
- 二值一致率 / κ：探针二值 vs 自陈二值 → accuracy + Cohen's κ。
- 分歧 top-K：|探针−自陈| 排序导表。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def align_feelings_to_windows(feelings: pd.DataFrame, scores_norm: pd.DataFrame) -> pd.DataFrame:
    """每条自陈对齐到覆盖其 mid_time 的窗口（同 day）。无覆盖取最近中点窗。"""
    aligned = []
    sc_by_day = {d: g.reset_index(drop=True) for d, g in scores_norm.groupby("day")}
    for _, fr in feelings.iterrows():
        day = int(fr["day"])
        mid = int(fr["mid_time"])
        if day not in sc_by_day or mid < 0:
            aligned.append({**fr.to_dict(), "win_id": None, "aligned": False})
            continue
        g = sc_by_day[day]
        cover = g[(g["t_start"] <= mid) & (g["mid_time"] >= mid) | (g["t_start"] <= mid) & (g["t_end"] >= mid)]
        if cover.empty:
            # 取最近中点窗
            j = (g["mid_time"] - mid).abs().idxmin()
            win = g.loc[j]
            covered = False
        else:
            # 多窗取中点最近
            j = (cover["mid_time"] - mid).abs().idxmin()
            win = cover.loc[j]
            covered = True
        rec = {**fr.to_dict()}
        rec["win_id"] = win["win_id"]
        rec["aligned"] = True
        rec["covered"] = covered
        for e in config.EMOTIONS:
            rec[f"probe_{e}_z"] = float(win[f"{e}_z"])
            rec[f"probe_{e}_q"] = float(win[f"{e}_q"])
        aligned.append(rec)
    return pd.DataFrame(aligned)


def compute_metrics(aligned: pd.DataFrame, tag: str = "mock") -> dict:
    A = aligned[aligned["aligned"]].copy()
    metrics = {"tag": tag, "n_feelings": int(len(aligned)), "n_aligned": int(len(A))}

    # --- 情绪时刻命中率（仅非 neutral、非争议 计入严格命中率）---
    strict = A[(A["mapped_emotion"] != config.NEUTRAL) & (~A["controversial"].astype(bool))]
    hits = []
    for _, r in strict.iterrows():
        e = r["mapped_emotion"]
        if e not in config.EMOTIONS:
            continue
        hits.append(1 if r[f"probe_{e}_z"] > config.HIT_THRESHOLD_SIGMA else 0)
    metrics["strict_hit_rate"] = round(float(np.mean(hits)), 4) if hits else None
    metrics["strict_n"] = len(hits)

    # 含争议项的定性命中率（仅参考）
    qual = A[A["mapped_emotion"] != config.NEUTRAL]
    qhits = []
    for _, r in qual.iterrows():
        e = r["mapped_emotion"]
        if e in config.EMOTIONS:
            qhits.append(1 if r[f"probe_{e}_z"] > config.HIT_THRESHOLD_SIGMA else 0)
    metrics["qualitative_hit_rate"] = round(float(np.mean(qhits)), 4) if qhits else None
    metrics["qualitative_n"] = len(qhits)

    # --- 极性相关：探针极性分 vs 自陈极性 ---
    pos = [f"probe_{e}_z" for e in config.POSITIVE_EMOTIONS]
    neg = [f"probe_{e}_z" for e in config.NEGATIVE_EMOTIONS]
    probe_pol = A[pos].sum(axis=1) - A[neg].sum(axis=1)
    self_pol = A["polarity"].astype(float)
    valid = self_pol != 0
    if valid.sum() > 2 and probe_pol[valid].std() > 0:
        pr, _ = pearsonr(probe_pol[valid], self_pol[valid])
        sr, _ = spearmanr(probe_pol[valid], self_pol[valid])
        metrics["polarity_pearson"] = round(float(pr), 4)
        metrics["polarity_spearman"] = round(float(sr), 4)
        metrics["polarity_n"] = int(valid.sum())
    else:
        metrics["polarity_pearson"] = None
        metrics["polarity_spearman"] = None

    # --- 二值一致率 / κ（对每个情绪：探针该情绪 z>阈值 vs 自陈是否标注该情绪）---
    y_true, y_pred = [], []
    for e in config.EMOTIONS:
        self_e = (A["mapped_emotion"] == e).astype(int)
        probe_e = (A[f"probe_{e}_z"] > config.HIT_THRESHOLD_SIGMA).astype(int)
        y_true.extend(self_e.tolist())
        y_pred.extend(probe_e.tolist())
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        metrics["binary_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics["cohen_kappa"] = round(float(cohen_kappa_score(y_true, y_pred)), 4)
    else:
        metrics["binary_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4) if y_true else None
        metrics["cohen_kappa"] = None
    return metrics


def _val_out(name: str, tag: str = "real") -> str:
    """真实/随机产物落模型目录 runs/<tag>/validation/，mock 留共享 artifacts/。"""
    if tag and ("mock" in tag or "signal" in tag):
        return config.artifact(name)
    return config.model_artifact("validation", name)


def divergence_topk(aligned: pd.DataFrame, k: int = config.DIVERGENCE_TOPK,
                    write: bool = True, tag: str = "real") -> pd.DataFrame:
    """找探针与自陈最分歧的案例：|探针主导负面 z − 自陈极性对应| 大。"""
    A = aligned[aligned["aligned"]].copy()
    pos = [f"probe_{e}_z" for e in config.POSITIVE_EMOTIONS]
    neg = [f"probe_{e}_z" for e in config.NEGATIVE_EMOTIONS]
    A["probe_polarity_score"] = A[pos].sum(axis=1) - A[neg].sum(axis=1)
    # 分歧 = 探针极性符号 与 自陈极性 不一致的程度
    A["self_polarity"] = A["polarity"].astype(float)
    A["divergence"] = (A["probe_polarity_score"] - A["self_polarity"]).abs()
    A["probe_top_emotion"] = A[[f"probe_{e}_z" for e in config.EMOTIONS]].idxmax(axis=1).str.replace("probe_", "").str.replace("_z", "")
    cols = ["day", "time_str", "category", "content", "mapped_emotion", "polarity",
            "probe_top_emotion", "probe_polarity_score", "divergence", "win_id"] + \
           [f"probe_{e}_z" for e in config.EMOTIONS]
    top = A.sort_values("divergence", ascending=False).head(k)[cols].reset_index(drop=True)
    if write:
        top.to_csv(_val_out("divergence_topk.csv", tag), index=False)
    return top


def run_validation(feelings: pd.DataFrame, scores_norm: pd.DataFrame,
                   tag: str = "mock", write: bool = True) -> dict:
    aligned = align_feelings_to_windows(feelings, scores_norm)
    metrics = compute_metrics(aligned, tag=tag)
    if write:
        with open(_val_out(f"validation_metrics_{tag}.json", tag), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        divergence_topk(aligned, write=True, tag=tag)
    return {"metrics": metrics, "aligned": aligned}


if __name__ == "__main__":
    from src import direction_artifact, windowize, score, normalize, parse_feelings
    art = direction_artifact.make_mock_artifact()
    wdf = windowize.build_windows(write=False)
    feel = parse_feelings.parse_feelings(write=False)

    res = {}
    for tag, strength in [("signal", 1.6), ("random", 0.0)]:
        prov = score.MockResidualProvider(art, inject_strength=strength)
        sdf = score.score_windows(wdf, art, prov, write=False)
        norm = normalize.normalize(sdf, write=False)
        r = run_validation(feel, norm, tag=tag, write=False)
        res[tag] = r["metrics"]
        print(f"=== {tag} ===")
        print(json.dumps(r["metrics"], ensure_ascii=False, indent=2))
    # 对照：信号注入 hit_rate / |相关| 应优于随机
    sig, rnd = res["signal"], res["random"]
    print("\nsignal strict_hit_rate", sig["strict_hit_rate"], "vs random", rnd["strict_hit_rate"])
    assert sig["strict_hit_rate"] > (rnd["strict_hit_rate"] or 0) + 0.05, "管线不敏感"
    print("validate self-test passed: signal > random")
