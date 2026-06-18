"""stimuli — 构造 6 情绪成对刺激（中/英两套）+ Qwen chat 模板前缀。

复刻 RepE `examples/primary_emotions/utils.py::primary_emotions_concept_dataset`
的成对 / shuffle / labels / train-test 切分口径（已逐字核对 /tmp/repe_ref）：

  - 每情绪 c_e = 目标情绪句；o_e = 其它 5 情绪句拼接后 shuffle，与 c_e 配对。
  - 每对 [c, o] 内部 random.shuffle，train_labels 记 [s==true_s for s in pair] → [bool,bool]。
  - data = np.concatenate(pairs) 展平成 [d0a,d0b,d1a,d1b,...]，对内相邻、对间拼接。
    → get_directions 内 relative = H[::2]-H[1::2] 即「对内求差」。
  - test 用未 shuffle 的 data_（目标恒在位置 0），labels=[[1,0]*len(test)]。

模板两套：
  - zh：走 Qwen `tokenizer.apply_chat_template`，system+user 中文「判断这段经历中
        {情绪}的程度」+ scenario，add_generation_prompt=True 出前缀喂 get_directions。
  - en：保留 RepE 原始 template_str 做跨语言消融（也可套 chat 模板，默认原始）。
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

# 情绪英文 -> 中文名（用于中文 prompt）
EMOTION_ZH = {
    "happiness": "快乐",
    "sadness": "悲伤",
    "anger": "愤怒",
    "fear": "恐惧",
    "disgust": "厌恶",
    "surprise": "惊讶",
}

# RepE 原始英文 concept 模板（消融用）
EN_TEMPLATE = (
    "{user_tag} Consider the {emotion} of the following scenario:\n"
    "Scenario: {scenario}\nAnswer: {assistant_tag} "
)


def _load_raw(data_dir: str, cap: int = 200) -> dict:
    """加载 6 情绪 json，去重并截断到 cap（复刻 list(set(...))[:200]）。"""
    raw = {}
    for emo in EMOTIONS:
        with open(os.path.join(data_dir, f"{emo}.json"), encoding="utf-8") as f:
            raw[emo] = list(set(json.load(f)))[:cap]
    return raw


def _zh_chat_prefix(tokenizer, emotion_en: str, scenario: str) -> str:
    """用 Qwen apply_chat_template 生成中文 concept 判断的前缀（含 generation prompt）。"""
    emo_zh = EMOTION_ZH[emotion_en]
    messages = [
        {
            "role": "system",
            "content": "你是一个善于体察情绪的助手，会仔细判断一段经历中蕴含的情绪。",
        },
        {
            "role": "user",
            "content": (
                f"请判断下面这段经历中“{emo_zh}”这种情绪的程度。\n"
                f"经历：{scenario}\n回答："
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _en_chat_prefix(tokenizer, emotion_en: str, scenario: str) -> str:
    """英文 chat 模板版本（消融，可选）。"""
    messages = [
        {
            "role": "system",
            "content": "You are an assistant skilled at perceiving the emotion in a scenario.",
        },
        {
            "role": "user",
            "content": (
                f"Consider the {emotion_en} of the following scenario:\n"
                f"Scenario: {scenario}\nAnswer:"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_concept_dataset(
    data_dir: str,
    tokenizer=None,
    lang: str = "zh",
    use_chat_template: bool = True,
    cap: int = 200,
    seed: int = 0,
):
    """构造 6 情绪成对 concept 数据集（train/test），复刻 RepE 口径。

    Args:
        data_dir: 含 {emotion}.json 的目录（emotions_zh 或 emotions_en）。
        tokenizer: Qwen tokenizer（use_chat_template=True 时必需）。
        lang: "zh" 或 "en"，决定 prompt 文案。
        use_chat_template: True 走 apply_chat_template；False 走 EN_TEMPLATE 原始串。
        cap: 每情绪去重后截断句数。
        seed: 复现用（RepE 内部固定 random.seed(0)）。

    Returns:
        formatted_data: {emotion: {'train':{'data','labels'}, 'test':{'data','labels'}}}
        其中 train data 已展平为 [d0a,d0b,...] 交替；labels 为 list[[bool,bool]]。
    """
    random.seed(seed)
    raw = _load_raw(data_dir, cap=cap)

    def render(emotion_en: str, scenario: str) -> str:
        if use_chat_template:
            assert tokenizer is not None, "use_chat_template=True 需要 tokenizer"
            if lang == "zh":
                return _zh_chat_prefix(tokenizer, emotion_en, scenario)
            return _en_chat_prefix(tokenizer, emotion_en, scenario)
        # 原始 RepE 英文模板（user_tag/assistant_tag 留空，与源码默认一致）
        return EN_TEMPLATE.format(emotion=emotion_en, scenario=scenario, user_tag="", assistant_tag="")

    formatted = {}
    for emo in EMOTIONS:
        c_e = raw[emo]
        o_e = np.concatenate([v for k, v in raw.items() if k != emo])
        random.shuffle(o_e)

        # 配对：与源码一致用 zip，长度取 min(len(c_e), len(o_e))
        pairs = [[c, o] for c, o in zip(c_e, o_e)]

        train_labels = []
        for pair in pairs:
            true_s = pair[0]
            random.shuffle(pair)
            train_labels.append([s == true_s for s in pair])

        train_flat = np.concatenate(pairs).tolist()  # 对内 shuffle 后展平
        # test：未 shuffle，目标恒在位置 0（与源码 data_ 一致）
        test_flat = np.concatenate(
            [[c, o] for c, o in zip(c_e, o_e)]
        ).tolist()

        train_data = [render(emo, s) for s in train_flat]
        test_data = [render(emo, s) for s in test_flat]

        formatted[emo] = {
            "train": {"data": train_data, "labels": train_labels},
            "test": {"data": test_data, "labels": [[1, 0] * len(test_data)]},
        }
    return formatted


def chat_template_signature(tokenizer, lang: str = "zh") -> str:
    """返回一条 apply_chat_template 渲染样例（写进 artifact，供 Agent C 对齐模板分布）。"""
    if lang == "zh":
        return _zh_chat_prefix(tokenizer, "happiness", "<SCENARIO>")
    return _en_chat_prefix(tokenizer, "happiness", "<SCENARIO>")
