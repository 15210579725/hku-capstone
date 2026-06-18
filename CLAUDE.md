# CLAUDE.md — HKU Capstone 全局导览

面向后续 AI/开发者的仓库导览。逐子项目的细节见各自目录的 `CLAUDE.md`。

## 这个仓库是什么

一个研究型 capstone:用机制可解释性方法探究「LLM 有没有可定位、可读取、可干预的情绪表征」。由**三条探针复现线 + 两套行为仿真**组成,共同搭起一条四级证据链(可分类 → 可对齐 → 可干预 → 可行为化)。

## 顶层布局与依赖关系

```
emotion-probe/          探针线(读取/定位情绪表征)
  RepE-2023-Zou/          ← 方法母本(RepE reading 向量)。其余两线都是它的延伸
  do_llm_feels_2025-wang/ ← 在 RepE 基础上做电路定位 + steering 因果
  anthropic-2026-gemma-version/ ← 在 RepE 基础上换模型(gemma)+ 扩到 8 情绪
simulation-env/         仿真线(把情绪表征接到 agent 行为上)
  concordia-official/     ← Concordia + lucia_sim,EgoLife 数据驱动行为预测
  Generative Agents/      ← 斯坦福生成式智能体 + world/,Smallville 行为回放
```

- 三条探针线**产出物相互独立**(各自的方向向量 `*.pt`),但共享同一套研究问题与 LUCIA/EgoLife 日记数据。
- 仿真线消费日记数据 + DeepSeek 后端,验证"情绪/状态 → 行为"是否一致;**不依赖探针线的产物即可独立运行**。

## 跨项目的共性约定与坑

1. **API 密钥**:全部从环境变量读取,仓库内无明文。`TRANSLATE_API_KEY`(gemma 翻译/生成,OpenAI 兼容端点)、`DEEPSEEK_API_KEY`(仿真 + 部分翻译)。改代码时不要写回明文。
2. **硬编码绝对路径**:多个脚本把数据/输出路径写死成原作者本机或 AutoDL 服务器路径。这是已知技术债;新机器上跑需改路径常量(各子 CLAUDE.md 标了具体位置)。
3. **大文件策略**:`*.pt/*.bin/*.safetensors`、海量仿真 `storage/`、大 CSV 经 `.gitignore` 排除。它们是**可再生产物**(GPU 重跑抽取/仿真即可),不是代码。改 `.gitignore` 前先确认不会带回大文件或明文 key。
4. **打包的官方框架**:RepE、EmotionCircuits、Concordia、生成式智能体均已去掉各自 `.git`,作为普通文件纳入主仓,保留了本地改动。它们各自的上游 URL/commit 记在子 README。
5. **transformers 版本敏感**:RepE 线锁 `4.40.2`(同时满足 Qwen2.5 加载与 RepE pipeline 子类);不同探针线对 transformers 版本要求不同,**务必各自独立建环境**。

## 验证现状

- `emotion-probe/RepE-2023-Zou`:本机已跑通 `run_mock_selftest.py`(8 阶段 mock 全链路)+ `pytest`(6 passed),不依赖 GPU/模型。真实抽取需 GPU。
- 其余线需各自的模型 + GPU/API,未在本机端到端验证;文档基于真实代码与作者已落盘的结果撰写。

## 给后续工作的提示

- 动任何一条线之前,先读该目录的 `README.md`(用法)+ `CLAUDE.md`(架构/坑)。
- 涉及"跑起来"的任务,优先用 RepE 线的 mock 自测确认基础环境,再逐步上 GPU。
- 修复硬编码路径、补齐各线的 `requirements.txt`、统一 key 注入,是让"clone 即跑"更顺的高价值方向。
