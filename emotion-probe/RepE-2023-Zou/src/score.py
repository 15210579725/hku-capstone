"""交付物④：打分器（真实残差 / mock 残差双模式）。

打分逻辑（与 RepE 一致）：
  h = residual_at(best_layer[e], text)
  raw_score[e] = sign[e] * (h · direction[e] / ‖direction[e]‖)
输出逐窗 × 6 情绪分 + 时间戳矩阵（动态量化情绪表）。
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import direction_artifact, emotion_mapping  # noqa: E402


# ---------------------------------------------------------------------------
# 残差提供者抽象
# ---------------------------------------------------------------------------
class ResidualProvider:
    def residual(self, text: str, layer: int) -> np.ndarray:
        raise NotImplementedError


class MockResidualProvider(ResidualProvider):
    """文本→可复现伪残差。

    不用纯随机：对窗文本命中映射词典的情绪，在对应方向上注入弱信号，
    使 mock 分数与金标准弱相关 → 验证 validate 能算出非平凡指标。
    inject_strength=0 时退化为纯随机 baseline（用于对照）。
    """

    def __init__(self, artifact: direction_artifact.Artifact,
                 inject_strength: float = 1.6, noise_std: float = 1.0,
                 seed: int = 7):
        self.art = artifact
        self.hidden_dim = artifact.hidden_dim
        self.inject = inject_strength
        self.noise_std = noise_std
        self.seed = seed
        self._cache = {}

    def _text_seed(self, text: str) -> int:
        h = hashlib.sha256((str(self.seed) + "|" + (text or "")).encode("utf-8")).hexdigest()
        return int(h[:16], 16)

    def residual(self, text: str, layer: int) -> np.ndarray:
        # 同一文本（与层无关，mock 简化）缓存，保证复现
        key = text or ""
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng(self._text_seed(text))
        h = rng.standard_normal(self.hidden_dim).astype(np.float32) * self.noise_std
        if self.inject > 0 and text:
            # 命中词典 → 沿对应情绪方向注入信号（含 sign，使投影后为正）
            hits = {}
            for kw_list, emo, pol, ctrl in emotion_mapping.RULES:
                for kw in kw_list:
                    if kw in text:
                        hits[emo] = hits.get(emo, 0) + 1
            for emo, cnt in hits.items():
                if emo not in self.art.directions:
                    continue
                d = self.art.unit_direction(emo)
                sign = self.art.direction_signs[emo]
                # 注入方向乘 sign，使最终 raw_score = sign*(h·d) 为正向冲高
                h = h + self.inject * np.log1p(cnt) * sign * d
        self._cache[key] = h
        return h


class RealResidualProvider(ResidualProvider):
    """GPU 阶段实现：加载 Qwen2.5-3B-Instruct（fp16，不量化），套与 Agent B 抽向量时
    完全一致的中文 chat 模板，取 best_layer 最后 token 残差（hidden state）。

    护栏：模板必须与 B 训练分布同分布 —— 复用 stimuli._zh_chat_prefix 渲染 prompt，
    并用 artifact 里的 chat_template_signature 校验（同一渲染器、同一 tokenizer）。
    取隐状态口径与 RepReadingPipeline._get_hidden_states 完全一致：
      outputs.hidden_states[layer][:, rep_token=-1, :]，bfloat16→float。
    """

    def __init__(self, artifact, model_id=config.MODEL_ID, dtype=None,
                 chat_template_signature: str = None, lang: str = "zh",
                 batch_size: int = 16):
        self.art = artifact
        self.model_id = model_id
        self.lang = lang
        self.batch_size = batch_size
        self._expected_sig = chat_template_signature
        self._dtype = dtype
        self._model = None
        self._tok = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self._dtype or torch.float16
        self._tok = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left", legacy=False)
        if self._tok.pad_token_id is None:
            self._tok.pad_token_id = self._tok.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto").eval()

        # 模板同分布校验：用同一渲染器生成签名与 artifact 比对
        from src import stimuli
        live_sig = stimuli.chat_template_signature(self._tok, lang=self.lang)
        if self._expected_sig is not None and live_sig != self._expected_sig:
            raise RuntimeError(
                "chat_template_signature 不一致：打分模板与抽向量分布不符\n"
                f"  expected(B): {self._expected_sig!r}\n  live:        {live_sig!r}")
        self._render = lambda emo, scenario: (
            stimuli._zh_chat_prefix(self._tok, emo, scenario)
            if self.lang == "zh"
            else stimuli._en_chat_prefix(self._tok, emo, scenario))

    def _last_token_hidden(self, prompts, layer: int) -> np.ndarray:
        """对一批 prompt 取该层最后 token 隐状态 (B, hidden) float32（口径同 RepE）。"""
        import torch
        self._lazy_load()
        device = next(self._model.parameters()).device
        enc = self._tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer][:, -1, :]  # rep_token=-1
        if h.dtype == torch.bfloat16:
            h = h.float()
        return h.float().cpu().numpy()

    def residual(self, text: str, layer: int, emotion: str = None) -> np.ndarray:
        """单条文本在 best_layer 的最后 token 残差。emotion 决定 chat 模板的情绪槽。"""
        self._lazy_load()
        emo = emotion or config.EMOTIONS[0]
        prompt = self._render(emo, text or "")
        return self._last_token_hidden([prompt], layer)[0]


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
def project_onto_direction(h: np.ndarray, d_unit: np.ndarray) -> float:
    # d 已单位化 → dot 即投影
    return float(h @ d_unit)


def score_windows(windows: pd.DataFrame, artifact: direction_artifact.Artifact,
                  provider: ResidualProvider, write: bool = True,
                  tag: str = "mock", level: str = config.DEFAULT_LEVEL,
                  mode: str = None) -> pd.DataFrame:
    mode = mode or config.WINDOW_MODE
    emos = config.EMOTIONS
    out_rows = []
    for _, w in windows.iterrows():
        text = w["text"]
        row = {
            "win_id": w["win_id"],
            "day": int(w["day"]),
            "t_start": int(w["t_start"]),
            "t_end": int(w["t_end"]),
            "mid_time": int(w["mid_time"]),
        }
        for e in emos:
            layer = artifact.best_layer[e]
            h = provider.residual(text, layer)
            d_unit = artifact.unit_direction(e)
            raw = artifact.direction_signs[e] * project_onto_direction(h, d_unit)
            row[e] = raw
        out_rows.append(row)
    sdf = pd.DataFrame(out_rows)
    if write:
        path = config.artifact(f"scores_{level}_{mode}_{tag}")
        try:
            sdf.to_parquet(path + ".parquet", index=False)
        except Exception:
            sdf.to_csv(path + ".csv", index=False)
    return sdf


# ---------------------------------------------------------------------------
# 真实阶段：嵌套 emotion_vectors.pt → 打分（emotion-aware，按 best_layer 投影 × sign）
# ---------------------------------------------------------------------------
def load_nested_artifact(path: str):
    """读 io_artifact 契约的嵌套 emotion_vectors.pt（directions[emo][layer]）。

    Returns: payload dict（含 directions/direction_signs/best_layer/H_train_means/...）。
    """
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def best_direction(payload, emo: str):
    """取某情绪 best_layer 的方向(单位化)、sign、layer、H_train_mean。"""
    L = int(payload["best_layer"][emo])
    d = np.asarray(payload["directions"][emo][L], dtype=np.float32).reshape(-1)
    nrm = np.linalg.norm(d)
    d_unit = d / nrm if nrm > 0 else d
    sign = int(payload["direction_signs"][emo][L])
    mean = None
    if "H_train_means" in payload and emo in payload["H_train_means"]:
        mean = np.asarray(payload["H_train_means"][emo][L], dtype=np.float32).reshape(-1)
    return d_unit, sign, L, mean


def score_windows_real(windows: pd.DataFrame, payload, provider: "RealResidualProvider",
                       recenter: bool = True, write: bool = True,
                       tag: str = "real", level: str = config.DEFAULT_LEVEL,
                       mode: str = None) -> pd.DataFrame:
    """真实打分：逐情绪在其 best_layer 批量前向，取最后 token 残差，recenter→投影×sign。

    与 RepE transform 一致：可选 recenter（减 H_train_means）后投影到 PCA comp0。
    输出逐窗 × 6 情绪真实分（schema 同 mock：win_id/day/t_start/t_end/mid_time + 6 列）。
    """
    mode = mode or config.WINDOW_MODE
    emos = config.EMOTIONS
    texts = windows["text"].fillna("").tolist()
    n = len(texts)
    base = {
        "win_id": windows["win_id"].tolist(),
        "day": windows["day"].astype(int).tolist(),
        "t_start": windows["t_start"].astype(int).tolist(),
        "t_end": windows["t_end"].astype(int).tolist(),
        "mid_time": windows["mid_time"].astype(int).tolist(),
    }
    sdf = pd.DataFrame(base)

    bs = provider.batch_size
    provider._lazy_load()
    for e in emos:
        d_unit, sign, L, mean = best_direction(payload, e)
        scores = np.empty(n, dtype=np.float32)
        for i in range(0, n, bs):
            chunk = texts[i:i + bs]
            prompts = [provider._render(e, t) for t in chunk]
            H = provider._last_token_hidden(prompts, L)  # (b, hidden)
            if recenter and mean is not None:
                H = H - mean[None, :]
            proj = H @ d_unit  # (b,)
            scores[i:i + bs] = sign * proj
        sdf[e] = scores
        print(f"[score_real] {e}: layer={L} sign={sign} done ({n} windows)", flush=True)

    if write:
        path = config.artifact(f"scores_{level}_{mode}_{tag}")
        try:
            sdf.to_parquet(path + ".parquet", index=False)
        except Exception:
            sdf.to_csv(path + ".csv", index=False)
    return sdf


if __name__ == "__main__":
    from src import windowize
    art = direction_artifact.make_mock_artifact()
    wdf = windowize.build_windows(write=False)
    prov_signal = MockResidualProvider(art, inject_strength=1.6)
    prov_random = MockResidualProvider(art, inject_strength=0.0)
    s_sig = score_windows(wdf, art, prov_signal, tag="mock_signal")
    s_rnd = score_windows(wdf, art, prov_random, tag="mock_random", write=False)
    print("score matrix shape:", (len(s_sig), len(config.EMOTIONS)))
    assert s_sig[config.EMOTIONS].notna().all().all()
    # 信号注入版：命中"开心"的窗，happiness 分应显著高于均值
    mask = wdf["text"].str.contains("开心|笑", regex=True, na=False)
    if mask.sum() > 5:
        hi = s_sig.loc[mask.values, "happiness"].mean()
        lo = s_sig.loc[~mask.values, "happiness"].mean()
        print(f"happiness mean (hit={mask.sum()}): {hi:.3f} vs others: {lo:.3f}")
        assert hi > lo + 0.3, "信号注入未生效"
    print("score self-test passed")
