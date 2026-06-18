"""io_artifact — 写/读 Agent C 的交接契约 emotion_vectors.pt（+ manifest.json）。

契约（torch.save 一个 dict，见 plan §3，保持稳定）：
{
  "model_name": str,                  # 抽取所用模型（smoke 时为 0.5B）
  "num_hidden_layers": int, "hidden_size": int, "rep_token": -1,
  "stimulus_lang": "zh" | "en",
  "use_chat_template": bool,
  "chat_template_signature": str,     # apply_chat_template 前缀样例
  "patch_info": dict,                 # device-agnostic 补丁状态
  "emotions": [6],
  "layers": [int,...],                # 抽取覆盖层（负索引口径）
  "layer_index_convention": str,
  "directions": {emotion: {layer:int -> np.ndarray (hidden_size,)}},      # PCA comp0，未单位归一
  "direction_signs": {emotion: {layer:int -> int(±1)}},
  "best_layer": {emotion: int},
  "best_layer_acc": {emotion: float},
  "H_train_means": {emotion: {layer:int -> np.ndarray (1,hidden_size)}},  # recenter 复现用
}

打分（Agent C）：score = (h @ d)/‖d‖ × sign，
  d = directions[emotion][best_layer[emotion]]，
  sign = direction_signs[emotion][best_layer[emotion]]，
  h = 该情绪 best_layer 最后 token 隐状态（模板须与 chat_template_signature 同分布）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

ARTIFACT_VERSION = "1.0"


def _to_np1d(x):
    """统一成 (hidden_size,) float32 ndarray。"""
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)


def save_artifact(
    path: str,
    *,
    model_name: str,
    num_hidden_layers: int,
    hidden_size: int,
    stimulus_lang: str,
    use_chat_template: bool,
    chat_template_signature: str,
    patch_info: dict,
    emotions: list,
    layers: list,
    directions: dict,
    direction_signs: dict,
    best_layer: dict,
    best_layer_acc: dict,
    H_train_means: dict,
    rep_token: int = -1,
):
    """规范化并 torch.save 交接 artifact。directions 落 comp0 的 (hidden_size,) 向量。"""
    norm_directions = {
        emo: {int(layer): _to_np1d(vec) for layer, vec in layer_map.items()}
        for emo, layer_map in directions.items()
    }
    norm_signs = {
        emo: {int(layer): int(np.sign(s) or 1) for layer, s in layer_map.items()}
        for emo, layer_map in direction_signs.items()
    }
    norm_means = {
        emo: {
            int(layer): np.asarray(m, dtype=np.float32).reshape(1, -1)
            for layer, m in layer_map.items()
        }
        for emo, layer_map in H_train_means.items()
    }

    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": model_name,
        "num_hidden_layers": int(num_hidden_layers),
        "hidden_size": int(hidden_size),
        "rep_token": int(rep_token),
        "stimulus_lang": stimulus_lang,
        "use_chat_template": bool(use_chat_template),
        "chat_template_signature": chat_template_signature,
        "patch_info": patch_info,
        "emotions": list(emotions),
        "layers": [int(x) for x in layers],
        "layer_index_convention": (
            "negative (HF hidden_states tuple index; -1=last transformer layer; "
            "tuple has num_hidden_layers+1 entries incl. embedding at idx 0)"
        ),
        "directions": norm_directions,
        "direction_signs": norm_signs,
        "best_layer": {emo: int(l) for emo, l in best_layer.items()},
        "best_layer_acc": {emo: float(a) for emo, a in best_layer_acc.items()},
        "H_train_means": norm_means,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    return payload


def load_artifact(path: str) -> dict:
    """读回 artifact（weights_only=False，含 numpy 对象）。"""
    return torch.load(path, map_location="cpu", weights_only=False)


def write_manifest(path: str, payload: dict, extra: dict | None = None):
    """写人读 manifest.json：字段说明 + 打分示例 + 各情绪 best_layer/acc 概览。"""
    per_emotion = {
        emo: {
            "best_layer": payload["best_layer"].get(emo),
            "best_layer_acc": round(float(payload["best_layer_acc"].get(emo, float("nan"))), 4),
            "sign_at_best": int(
                payload["direction_signs"][emo][payload["best_layer"][emo]]
            ),
        }
        for emo in payload["emotions"]
    }
    manifest = {
        "artifact_version": payload["artifact_version"],
        "artifact_file": os.path.basename_for_manifest if False else "emotion_vectors.pt",
        "model_name": payload["model_name"],
        "num_hidden_layers": payload["num_hidden_layers"],
        "hidden_size": payload["hidden_size"],
        "rep_token": payload["rep_token"],
        "stimulus_lang": payload["stimulus_lang"],
        "use_chat_template": payload["use_chat_template"],
        "layers_covered": payload["layers"],
        "layer_index_convention": payload["layer_index_convention"],
        "chat_template_signature": payload["chat_template_signature"],
        "patch_info": payload["patch_info"],
        "per_emotion": per_emotion,
        "scoring": {
            "formula": "score = (h @ d) / norm(d) * sign",
            "where": {
                "d": "directions[emotion][best_layer[emotion]]  (hidden_size,) float32, NOT unit-normalized",
                "sign": "direction_signs[emotion][best_layer[emotion]]  in {-1,+1}",
                "h": "last-token hidden state at best_layer for the SAME chat-template distribution",
                "recenter": "optional: h - H_train_means[emotion][layer] before projecting (RepE transform 复现)",
            },
            "python_example": (
                "import torch, numpy as np; "
                "art = torch.load('emotion_vectors.pt', weights_only=False); "
                "emo='happiness'; L=art['best_layer'][emo]; "
                "d=art['directions'][emo][L]; s=art['direction_signs'][emo][L]; "
                "score = float(h @ d) / float(np.linalg.norm(d)) * s"
            ),
        },
    }
    if extra:
        manifest.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
