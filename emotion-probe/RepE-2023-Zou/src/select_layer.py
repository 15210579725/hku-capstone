"""src/select_layer.py — 按 test 刺激上各层分类准确率，给每情绪选最优层。

准确率口径逐字复刻 emotion_concept.ipynb 的 results 计算：
  对每层、每情绪，H_test 是 test 集每样本在该层的投影标量（pipeline 已 transform）。
  test 集口径：目标恒在偶数位置（pair[0]），与其 1 个对照配对 → 两两一组。
    sign==-1 用 min、否则用 max；命中 = 该组极值 == 组内第 0 个（目标）。
  acc = mean(命中)。随机基线 0.5（二选一）。

最优层 = 该情绪 acc 最高的层（并列取更靠中层 / 第一个）。
产出每情绪 best_layer / best_layer_acc + 全层 acc 曲线（写 select_layer/）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

EMOTIONS = config.EMOTIONS


def layer_accuracy(H_test_emotion, layer: int, sign: int) -> float:
    """单情绪单层准确率（复刻 notebook results[layer][emotion]）。"""
    vals = []
    for H in H_test_emotion:
        v = np.asarray(H[layer]).reshape(-1)
        vals.append(float(v[0]))
    # 两两一组：目标在第 0 个
    pairs = [vals[i:i + 2] for i in range(0, len(vals), 2)]
    pairs = [p for p in pairs if len(p) == 2]
    eval_func = min if sign == -1 else max
    cors = [1.0 if eval_func(p) == p[0] else 0.0 for p in pairs]
    return float(np.mean(cors)) if cors else float("nan")


def accuracy_table(rep_readers, H_tests, hidden_layers) -> dict:
    """全层 × 6 情绪准确率表 {layer: {emotion: acc}}。"""
    table = {int(layer): {} for layer in hidden_layers}
    for layer in hidden_layers:
        for emo in EMOTIONS:
            s = rep_readers[emo].direction_signs[layer]
            sign = int(np.sign(np.asarray(s).reshape(-1)[0]) or 1)
            table[int(layer)][emo] = layer_accuracy(H_tests[emo], layer, sign)
    return table


def select_best_layers(table: dict, hidden_layers) -> tuple[dict, dict]:
    """每情绪选 acc 最高层。并列时取靠中层（|depth| 居中），稳定可复现。"""
    layers = [int(l) for l in hidden_layers]
    mid = np.median([abs(l) for l in layers])
    best_layer, best_acc = {}, {}
    for emo in EMOTIONS:
        accs = [(l, table[l][emo]) for l in layers]
        max_acc = max(a for _, a in accs)
        # 并列：选 |layer| 最接近中层的
        tied = [l for l, a in accs if a >= max_acc - 1e-9]
        best = min(tied, key=lambda l: (abs(abs(l) - mid), abs(l)))
        best_layer[emo] = int(best)
        best_acc[emo] = float(max_acc)
    return best_layer, best_acc


def run_select(rep_readers, H_tests, hidden_layers, out_dir: str) -> dict:
    """算准确率表 + 选层，落盘 select_layer/。返回 best_layer/best_acc/table。"""
    table = accuracy_table(rep_readers, H_tests, hidden_layers)
    best_layer, best_acc = select_best_layers(table, hidden_layers)

    os.makedirs(out_dir, exist_ok=True)
    # 全层曲线（emotion -> [(layer, acc)...]）
    curves = {
        emo: [[int(l), round(table[int(l)][emo], 4)] for l in hidden_layers]
        for emo in EMOTIONS
    }
    summary = {
        "emotions": EMOTIONS,
        "best_layer": best_layer,
        "best_layer_acc": {e: round(best_acc[e], 4) for e in EMOTIONS},
        "random_baseline": 0.5,
        "accuracy_curves": curves,
    }
    with open(os.path.join(out_dir, "select_layer.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[select_layer] best layers:", best_layer, flush=True)
    print("[select_layer] best acc   :",
          {e: round(best_acc[e], 3) for e in EMOTIONS}, flush=True)
    return {"best_layer": best_layer, "best_layer_acc": best_acc, "table": table}
