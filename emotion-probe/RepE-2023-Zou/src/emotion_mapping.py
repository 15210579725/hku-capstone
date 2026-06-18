"""交付物③配套：细情绪 → 6 基本情绪 / 极性 的可审计映射词典。

策略 = 关键词词典 + 极性（确定性、可解释，不用模型）。
多命中取优先级最先匹配；未命中 → neutral。
D2 争议项（尴尬/无聊/手足无措/不知所措/纠结）通过 config.CONTROVERSIAL_AS_NEUTRAL 开关，
默认按争议词典映射（尴尬类→fear，无聊/累→sadness），并标记 controversial=True，
不计入严格命中率，仅定性。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# 词典：按优先级排序，先命中先取。每项 (关键词列表, 基本情绪, 极性, 是否争议)
# 极性：+1 正向 / -1 负向 / 0 中性
RULES = [
    # --- 明确正向 ---
    (["开心", "高兴", "愉快", "快乐", "喜悦"], "happiness", +1, False),
    (["笑", "好笑", "搞笑"], "happiness", +1, False),
    (["赞叹", "佩服", "厉害", "夸", "称赞"], "happiness", +1, False),
    (["舒服", "放松", "惬意", "自在"], "happiness", +1, False),
    (["喜欢", "享受", "满足", "欣慰"], "happiness", +1, False),
    (["期待", "盼望", "希望"], "happiness", +1, False),
    (["好奇", "感兴趣", "想知道"], "happiness", +1, False),
    (["感动", "温暖", "感慨"], "happiness", +1, False),

    # --- 惊讶（弱极性） ---
    (["震惊", "惊讶", "惊奇", "意外", "吃惊", "没想到"], "surprise", 0, False),

    # --- 明确负向：焦虑/恐惧类 ---
    (["害怕", "恐惧", "恐怖", "可怕"], "fear", -1, False),
    (["担心", "担忧", "忧虑"], "fear", -1, False),
    (["着急", "焦急", "焦虑", "急"], "fear", -1, False),
    (["紧张", "不安", "忐忑"], "fear", -1, False),

    # --- 明确负向：悲伤类 ---
    (["难过", "伤心", "失落", "沮丧", "失望"], "sadness", -1, False),
    (["孤独", "寂寞"], "sadness", -1, False),

    # --- 明确负向：愤怒类 ---
    (["生气", "愤怒", "恼火", "气愤"], "anger", -1, False),
    (["烦", "烦躁", "厌烦", "不耐烦"], "anger", -1, False),

    # --- 明确负向：厌恶类 ---
    (["恶心", "嫌弃", "厌恶", "反感", "脏"], "disgust", -1, False),

    # --- 争议项（D2，可一键改判 neutral）---
    (["尴尬"], "fear", -1, True),
    (["手足无措", "不知所措", "无措"], "fear", -1, True),
    (["纠结", "犹豫", "矛盾"], "fear", -1, True),
    (["无聊", "乏味", "没意思"], "sadness", -1, True),
    (["累", "疲惫", "疲倦", "困"], "sadness", -1, True),
]


def map_content(content: str):
    """单条自陈 → (mapped_emotion, polarity, mapped_rule, controversial)。"""
    if not content:
        return config.NEUTRAL, 0, None, False
    for keywords, emo, pol, controversial in RULES:
        for kw in keywords:
            if kw in content:
                if controversial and config.CONTROVERSIAL_AS_NEUTRAL:
                    return config.NEUTRAL, 0, kw, True
                return emo, pol, kw, controversial
    return config.NEUTRAL, 0, None, False


def export_mapping_table(write: bool = True) -> dict:
    table = {
        "emotions": config.EMOTIONS,
        "neutral": config.NEUTRAL,
        "controversial_as_neutral": config.CONTROVERSIAL_AS_NEUTRAL,
        "rules": [
            {
                "keywords": kw,
                "mapped_emotion": emo,
                "polarity": pol,
                "controversial": ctrl,
            }
            for kw, emo, pol, ctrl in RULES
        ],
        "notes": (
            "多命中取优先级最先匹配；未命中→neutral。"
            "争议项(尴尬/手足无措/不知所措/纠结/无聊/累)默认按词典映射并标 controversial=True，"
            "不计入严格命中率，仅定性。可通过 config.CONTROVERSIAL_AS_NEUTRAL 一键改判 neutral。"
        ),
    }
    if write:
        with open(config.artifact("emotion_mapping_table.json"), "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=2)
    return table


if __name__ == "__main__":
    export_mapping_table()
    tests = ["我感到有一些尴尬", "我有点开心", "我很担心明天", "我觉得有点无聊",
             "我在想这件事", "我笑了出来", "我震惊了"]
    for t in tests:
        print(t, "->", map_content(t))
