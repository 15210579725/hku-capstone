"""scripts/plot_all_days.py — 用真实分给全 7 天各出情绪随时间曲线图 + 7 天总览。

分钟级滑窗逐窗 z 分抖动剧烈，直接画是“毛刺墙”。本脚本：
  - 原始细粒度曲线 → 淡背景（保留你要的细粒度信息）
  - 滑动均值平滑曲线 → 主曲线（突出一天的情绪走势）
  - 自陈事件 → 着色竖线
输出到 runs/<MODEL_TAG>/figures/：
  day{1..7}_emotion_curves.png、all_days_overview.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import parse_feelings, visualize  # noqa: E402

EMO = config.EMOTIONS
COL = visualize.EMO_COLORS


def _smooth(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(window=k, center=True, min_periods=max(1, k // 3)).mean()


def _draw(ax, g, fr, k, raw_bg=True, legend=False):
    g = g.sort_values("mid_time")
    x = g["mid_time"] / 3600.0
    for e in EMO:
        if raw_bg:
            ax.plot(x, g[f"{e}_z"], color=COL[e], linewidth=0.5, alpha=0.12)
        ax.plot(x, _smooth(g[f"{e}_z"], k), color=COL[e], linewidth=2.0,
                alpha=0.95, label=e if legend else None)
    for _, r in fr.iterrows():
        emo = r["mapped_emotion"]
        if emo == config.NEUTRAL:
            continue
        ax.axvline(r["mid_time"] / 3600.0, color=COL.get(emo, "#888"),
                   linestyle="--", alpha=0.35, linewidth=0.8)
    ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth_k", type=int, default=21,
                    help="滑动平滑窗口（窗数），越大越平滑")
    args = ap.parse_args()
    k = args.smooth_k

    norm = pd.read_parquet(
        config.model_artifact("scores", "scores_norm_L2_count_real.parquet"))
    feel = parse_feelings.parse_feelings(write=False)
    days = sorted(int(d) for d in norm["day"].unique())

    paths = []
    # 1) 逐天单图
    for d in days:
        g = norm[norm["day"] == d]
        fr = feel[(feel["day"] == d) & (feel["start"] >= 0)]
        fig, ax = plt.subplots(figsize=(16, 7))
        _draw(ax, g, fr, k, raw_bg=True, legend=True)
        ax.set_xlabel("time of day (hour)")
        ax.set_ylabel("emotion score (z within day)")
        ax.set_title(f"day{d} probe emotion trajectory (real Qwen, smooth k={k}, "
                     f"{len(g)} windows) + self-report markers (dashed)")
        ax.legend(loc="upper right", ncol=3, fontsize=9)
        fig.tight_layout()
        p = config.model_artifact("figures", f"day{d}_emotion_curves.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
        print(f"[plot] day{d} -> {p}", flush=True)

    # 2) 7 天总览
    fig, axes = plt.subplots(2, 4, figsize=(26, 11), sharey=True)
    axes = axes.flatten()
    for idx, d in enumerate(days):
        g = norm[norm["day"] == d]
        fr = feel[(feel["day"] == d) & (feel["start"] >= 0)]
        _draw(axes[idx], g, fr, k, raw_bg=True, legend=(idx == 0))
        axes[idx].set_title(f"day{d}  ({len(g)} windows)")
        axes[idx].set_xlabel("hour")
    for j in range(len(days), len(axes)):
        axes[j].axis("off")
    axes[0].set_ylabel("emotion z-score")
    axes[4].set_ylabel("emotion z-score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=6, fontsize=12)
    fig.suptitle(
        f"Emotion probe trajectories over 7 days ({config.MODEL_TAG}, real Qwen "
        f"residuals, z within day, smooth k={k}) + self-report markers (dashed)",
        fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    overview = config.model_artifact("figures", "all_days_overview.png")
    fig.savefig(overview, dpi=130)
    plt.close(fig)
    paths.append(overview)
    print(f"[plot] overview -> {overview}", flush=True)

    print("\n生成图片：")
    for p in paths:
        print(f"  {p}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
