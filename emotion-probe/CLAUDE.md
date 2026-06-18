# CLAUDE.md — emotion-probe 容器导览

三条情绪探针复现线的容器目录。逐线细节见各子目录的 `CLAUDE.md`。

## 结构与定位

- `RepE-2023-Zou/` — **方法母本**。Representation Engineering(Zou 2023)reading 向量法:成对刺激取最后 token 残差 → 每层方向 → 残差投影打分。工程最成熟,带 mock 自测 + pytest,本机已验证可跑。
- `do_llm_feels_2025-wang/` — 复现 EmotionCircuits-LLM(Llama-3.2-3B)。在母本的"读方向"之上,进一步**归因到 MLP 神经元/注意力头**并用 steering 做**因果**验证。
- `anthropic-2026-gemma-version/` — 在母本方法上**换 gemma 模型**、扩到 **8 情绪**;含快速线(layer23)与忠实复现线(layer28)两套等价实现。

## 关键共性

1. 三线对 `transformers` 版本要求不同(RepE 锁 4.40.2),**必须各自独立建环境**。
2. 三线各自产出独立的情绪方向向量 `*.pt`(均被 `.gitignore` 排除,GPU 重跑可再生)。
3. 共享同一研究问题与 LUCIA/EgoLife 日记 + feelings 金标准数据(原始数据路径多为仓库外硬编码,见各子 CLAUDE.md)。

## 方法演进脉络

```
RepE(读出情绪方向,可分类/可对齐)
   ├─→ do_llm_feels(定位电路 + steering,可干预)
   └─→ gemma-version(换模型 + 多情绪,泛化性)
```

动手前先读对应子目录的 README + CLAUDE。
