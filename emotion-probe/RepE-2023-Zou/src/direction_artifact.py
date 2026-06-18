"""交付物④前置：Agent B 方向 artifact 的契约定义 + mock 生成器 + 适配层。

契约依据 plan_A_RepE.md（RepE 路线）：
  rep_reading_pipeline.get_directions(...) → 每层方向；打分 = H·direction/‖direction‖ × direction_signs。
Agent B 真实产出预期落在 outputs/directions/emotion_vectors.pt。

适配层 `adapt_artifact` 隔离字段名差异：等 B 真实契约出来只改这一处。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


@dataclass
class Artifact:
    """标准化后的方向 artifact（打分器只认这个结构）。"""
    emotions: list                          # 6 情绪顺序
    directions: dict                        # emotion -> np.ndarray[hidden_dim]
    best_layer: dict                        # emotion -> int
    direction_signs: dict                   # emotion -> +1/-1
    hidden_dim: int
    model_id: str = config.MODEL_ID
    rep_token: int = config.REP_TOKEN
    norm: str = "unit"                       # 方向是否已单位归一化
    meta: dict = field(default_factory=dict)

    def unit_direction(self, emo: str) -> np.ndarray:
        d = self.directions[emo]
        nrm = np.linalg.norm(d)
        return d / nrm if nrm > 0 else d


def adapt_artifact(raw: dict) -> Artifact:
    """适配层：把 Agent B 真实字段名映射到标准 Artifact。

    等拿到 B 真实契约只改本函数（字段别名表）。当前按假设契约直读。
    """
    # 字段别名容错（B 可能用不同命名）
    def pick(d, *keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    emotions = pick(raw, "emotions", "emotion_order", default=config.EMOTIONS)
    raw_dirs = pick(raw, "directions", "emotion_directions", "vectors", default={})
    directions = {e: np.asarray(v, dtype=np.float32) for e, v in raw_dirs.items()}
    best_layer = pick(raw, "best_layer", "best_layers", "layer", default={})
    signs = pick(raw, "direction_signs", "signs", default={e: 1 for e in emotions})
    hidden_dim = int(pick(raw, "hidden_dim", default=(
        len(next(iter(directions.values()))) if directions else config.MOCK_HIDDEN_DIM)))
    return Artifact(
        emotions=list(emotions),
        directions=directions,
        best_layer={e: int(best_layer.get(e, -20)) for e in emotions},
        direction_signs={e: int(signs.get(e, 1)) for e in emotions},
        hidden_dim=hidden_dim,
        model_id=pick(raw, "model_id", default=config.MODEL_ID),
        rep_token=int(pick(raw, "rep_token", default=config.REP_TOKEN)),
        norm=pick(raw, "norm", default="unit"),
        meta={"source": "agent_b"},
    )


def load_artifact(path: str = config.AGENT_B_DIRECTIONS) -> Artifact:
    """加载 Agent B 真实 artifact（.pt 或 .npz 或 .json），经适配层标准化 + 校验。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent B artifact not found: {path}")
    if path.endswith(".pt"):
        import torch  # 仅真实阶段需要
        raw = torch.load(path, map_location="cpu")
        # torch tensor -> numpy
        if "directions" in raw and isinstance(raw["directions"], dict):
            raw["directions"] = {
                k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                for k, v in raw["directions"].items()
            }
    elif path.endswith(".npz"):
        npz = np.load(path, allow_pickle=True)
        raw = {k: npz[k].item() if npz[k].dtype == object else npz[k] for k in npz.files}
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    art = adapt_artifact(raw)
    _validate(art)
    return art


def _validate(art: Artifact):
    assert set(art.emotions) >= set(config.EMOTIONS), "emotions 不全"
    for e in config.EMOTIONS:
        assert e in art.directions, f"缺方向 {e}"
        assert art.directions[e].shape == (art.hidden_dim,), f"{e} 维度不符"
        assert art.direction_signs[e] in (-1, 1), f"{e} sign 非法"


def make_mock_artifact(hidden_dim: int = config.MOCK_HIDDEN_DIM, seed: int = 0,
                       write: bool = True) -> Artifact:
    """生成符合契约的随机方向（单位化）+ 随机 best_layer + 随机 sign，供本地自测。

    关键：6 个方向两两近正交（用 QR 分解），让 mock 信号注入彼此可分。
    """
    rng = np.random.default_rng(seed)
    emos = config.EMOTIONS
    # 随机矩阵 QR 取正交基的前 6 列做方向，保证近正交
    M = rng.standard_normal((hidden_dim, len(emos)))
    Q, _ = np.linalg.qr(M)
    directions = {}
    for i, e in enumerate(emos):
        v = Q[:, i].astype(np.float32)
        v = v / np.linalg.norm(v)
        directions[e] = v
    art = Artifact(
        emotions=list(emos),
        directions=directions,
        best_layer={e: int(rng.integers(-22, -16)) for e in emos},
        direction_signs={e: int(rng.choice([-1, 1])) for e in emos},
        hidden_dim=hidden_dim,
        norm="unit",
        meta={"source": "mock", "seed": seed},
    )
    _validate(art)
    if write:
        meta = {
            "emotions": art.emotions,
            "best_layer": art.best_layer,
            "direction_signs": art.direction_signs,
            "hidden_dim": art.hidden_dim,
            "model_id": art.model_id,
            "norm": art.norm,
            "meta": art.meta,
        }
        with open(config.artifact("mock_artifact_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return art


if __name__ == "__main__":
    art = make_mock_artifact()
    print("mock artifact ok: hidden_dim", art.hidden_dim, "emotions", art.emotions)
    print("best_layer", art.best_layer)
    print("signs", art.direction_signs)
    # 检查近正交
    import itertools
    cos = []
    for a, b in itertools.combinations(config.EMOTIONS, 2):
        cos.append(abs(float(art.directions[a] @ art.directions[b])))
    print("max abs cosine between directions:", round(max(cos), 6))
    assert max(cos) < 1e-5
    print("direction_artifact self-test passed")
