# HKU Capstone — LLM 有没有「情感」?

> 用**机制可解释性(mechanistic interpretability)**的方法,把"大语言模型有没有情感"这个看似哲学的问题,拆成一条**可证伪、可测量**的证据链。

## 核心命题

我们不追问 LLM 有没有"主观感受",而是问一个能用实验回答的问题:

> **LLM 内部是否存在可定位、可读取、可干预的情绪表征?**

整个项目用四个递进的证据层级来回答它:

```
1. 可分类   probe 能从隐藏层读出情绪          →  RepE 在 Llama-2-13b 上 6 情绪分类 97.3%(随机 16.67%)
2. 可对齐   读数对得上真人自陈情绪            →  Qwen2.5-3B 对 438 条金标准命中 0.545(2.8× 随机),Pearson 0.386
3. 可干预   注入/消融情绪方向能改变模型输出    →  EmotionCircuits 在 Llama-3.2-3B 上定位电路 + steering 因果验证
4. 可行为化 情绪驱动的 agent 行为接近真人      →  Concordia / 生成式智能体仿真,动作预测一致率
```

## 仓库结构

```
hku capstone/
├── emotion-probe/                       # 核心:三条情绪探针复现线(按方法论文脉络命名)
│   ├── RepE-2023-Zou/                    # 【旗舰】Representation Engineering 读向量法,Qwen2.5-3B
│   │                                     #   中文日记 → 逐时刻 6 情绪曲线,438 条自陈金标准验证
│   ├── do_llm_feels_2025-wang/           # EmotionCircuits 复现,Llama-3.2-3B
│   │                                     #   六情绪电路级定位(MLP neurons / attn heads)+ steering 因果
│   └── anthropic-2026-gemma-version/     # Gemma 系探针,8 情绪(含 curiosity)
│                                         #   对比故事 → diff-of-means 方向 → 日记打分
├── simulation-env/                      # 「情绪 → 行为」证据链:多智能体行为仿真
│   ├── concordia-official/               # Google DeepMind Concordia + 自有 lucia_sim(EgoLife 行为预测)
│   └── Generative Agents/                # 斯坦福生成式智能体 + 自有 world/(Smallville 行为回放)
└── README.md / CLAUDE.md                # 本文 / 面向开发者的全局导览
```

每个子目录都有自己的 `README.md`(怎么用)和 `CLAUDE.md`(架构与坑),**先读对应子目录的文档再动手。**

## 三条探针线一览

| 子项目 | 模型 | 方法 | 角色 |
|--------|------|------|------|
| **RepE-2023-Zou**(旗舰) | Qwen2.5-3B-Instruct(主)/ Llama-2-13b-chat(官方复现) | RepE reading 向量 / LAT,残差投影打分 | 方法母本;已本机验证可跑(见下) |
| **do_llm_feels_2025-wang** | Llama-3.2-3B | diff-of-means 方向 → 神经元/注意力头归因 → steering | 电路级定位 + 因果干预 |
| **anthropic-2026-gemma-version** | gemma-4-E4B | diff-of-means + 中性 PCA 去混杂,逐 token 投影 | 多情绪 + 跨模型验证 |

> RepE 是方法论母本,后两条线是它在"电路定位/因果"与"不同模型/多情绪"方向上的延伸。

## 快速开始

### 0. 通用前提

- **Python 3.11/3.12**(各子项目略有差异,见各自 README)。
- 所有重型推理(抽隐藏态、跑模型)需要 **GPU**;行为仿真线(concordia / 生成式智能体)用 DeepSeek API 后端,**不需要 GPU**。
- 各子项目环境相互独立,请**按子目录 README 分别建虚拟环境**,不要混用。

### 1. 最省事的"看得见结果"入口(无需 GPU / 模型)

旗舰线带一个 mock 自测,用伪残差跑通完整流水线,已在本机验证:

```bash
cd emotion-probe/RepE-2023-Zou
python3 -m pip install -r env/requirements_probe.txt   # 注意 transformers==4.40.2 版本约束
python3 scripts/run_mock_selftest.py                   # 端到端 8 阶段,signal>random
python3 -m pytest tests/                                # 6 passed
```

### 2. 真实复现

各线的真实抽取/打分/仿真命令见各子目录 README。

## ⚠️ Clone 后必读

本仓库是多个研究脚本 + 三个打包的官方框架的集合,**不是一个一键即跑的单体应用**。Clone 后请注意:

1. **大文件未入库,需自行生成/下载。** 模型权重、情绪向量(`*.pt`)、海量仿真回放数据、大型分析 CSV 已被 `.gitignore` 排除。如何重新生成见各子目录 README(通常是在 GPU 上重跑对应抽取阶段)。
2. **部分脚本含硬编码绝对路径。** 多个项目的数据路径/输出路径写死成原作者本机或服务器路径(如 `DATA_ROOT`、`/root/...`、`/autodl-fs/...`)。运行前需按各 README 的提示改成你自己的路径。
3. **API 密钥走环境变量。** 仓库内已清理所有明文密钥,改为从环境变量读取(见 `.env.example`)。运行前自行设置,**切勿提交明文 key**。
4. **打包的官方框架已去除各自 `.git`**,作为普通文件纳入,保留了组员的本地改动;上游来源见各子目录 README 的"数据来源/出处"。

## 环境变量

复制 `.env.example` 并填入你自己的 key(或直接 `export`):

```bash
export TRANSLATE_API_KEY="..."   # gemma 线:生成/翻译故事(OpenAI 兼容端点)
export DEEPSEEK_API_KEY="..."    # 仿真线 + 部分翻译:DeepSeek API
```

## 致谢 / 上游

本项目复现并扩展了以下工作(均已注明出处,打包副本仅供组内复现):
- Representation Engineering — Zou et al. 2023, `andyzoujm/representation-engineering`
- EmotionCircuits-LLM — `gagan3012/EmotionCircuits-LLM`
- gemma-emotional-probes(官方参考)
- Concordia — `google-deepmind/concordia`
- Generative Agents — `joonspk-research/generative_agents`
