"""交付物⑦：可视化。

A) 日内 6 情绪曲线（默认 day1，标注最密 276 条）：x=时间，6 条 z 曲线，
   叠加 feelings 事件点（按 mapped_emotion 着色竖线）。→ day1_emotion_curves.png
B) 分歧 top-K 表 → divergence_topk.png

matplotlib Agg 后端；中文字体缺失则退化英文标注，axes.unicode_minus=False。
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

# 尝试常见中文字体；失败则纯英文标注
_CN_FONT = None
for cand in ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
             "Songti SC", "Hiragino Sans GB"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        _CN_FONT = cand
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue

EMO_COLORS = {
    "happiness": "#f1c40f", "sadness": "#3498db", "anger": "#e74c3c",
    "fear": "#9b59b6", "disgust": "#27ae60", "surprise": "#e67e22",
    "neutral": "#95a5a6",
}


def _sec_to_hm(s):
    return f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}"


def _fig_out(name: str, label: str = "real") -> str:
    """真实/随机图落模型目录 runs/<tag>/figures/，mock 留共享 artifacts/。"""
    if label and ("mock" in label or "signal" in label):
        return config.artifact(name)
    return config.model_artifact("figures", name)


def plot_day_curves(scores_norm, feelings, day=None, write=True, label="real"):
    day = day or config.PRIMARY_VALIDATION_DAY
    g = scores_norm[scores_norm["day"] == day].sort_values("mid_time")
    fr = feelings[(feelings["day"] == day) & (feelings["start"] >= 0)]

    fig, ax = plt.subplots(figsize=(16, 7))
    for e in config.EMOTIONS:
        ax.plot(g["mid_time"] / 3600.0, g[f"{e}_z"], label=e,
                color=EMO_COLORS[e], linewidth=1.1, alpha=0.85)
    # 叠加自陈事件竖线（按 mapped_emotion 着色；neutral 不画线避免遮挡）
    plotted = set()
    for _, r in fr.iterrows():
        emo = r["mapped_emotion"]
        if emo == config.NEUTRAL:
            continue
        ax.axvline(r["mid_time"] / 3600.0, color=EMO_COLORS.get(emo, "#888"),
                   linestyle="--", alpha=0.35, linewidth=0.8)
        plotted.add(emo)
    ax.set_xlabel("time of day (hour)")
    ax.set_ylabel("emotion score (z-normalized within day)")
    title = f"day{day} probe emotion curves ({label}) + self-report markers (dashed)"
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = _fig_out(f"day{day}_emotion_curves.png", label)
    if write:
        fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_divergence_table(topk_df, write=True, label="real"):
    df = topk_df.copy()
    show_cols = ["day", "time_str", "category", "mapped_emotion",
                 "probe_top_emotion", "probe_polarity_score", "divergence"]
    df = df[show_cols].round(3)
    # content 单独列（可能长），截断
    contents = topk_df["content"].astype(str).str.slice(0, 22).tolist()
    df.insert(4, "content", contents)

    fig, ax = plt.subplots(figsize=(15, 0.5 + 0.42 * len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    ax.set_title(f"Top-K probe vs self-report divergence ({label})", pad=12)
    fig.tight_layout()
    path = _fig_out("divergence_topk.png", label)
    if write:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    from src import (direction_artifact, windowize, score, normalize,
                     parse_feelings, validate)
    art = direction_artifact.make_mock_artifact()
    wdf = windowize.build_windows(write=False)
    feel = parse_feelings.parse_feelings(write=False)
    prov = score.MockResidualProvider(art, inject_strength=1.6)
    sdf = score.score_windows(wdf, art, prov, write=False)
    norm = normalize.normalize(sdf, write=False)
    aligned = validate.align_feelings_to_windows(feel, norm)
    top = validate.divergence_topk(aligned, write=False)

    p1 = plot_day_curves(norm, feel, day=1)
    p2 = plot_divergence_table(top)
    print("cn font:", _CN_FONT)
    for p in (p1, p2):
        sz = os.path.getsize(p)
        print(p, sz, "bytes")
        assert sz > 0
    print("visualize self-test passed")
