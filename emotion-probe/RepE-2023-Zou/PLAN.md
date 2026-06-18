# Emotion-Probe — 用 RepE reading 向量做动态情绪量化（执行计划 + 进度）

> 论文：Representation Engineering (Zou et al. 2023, arXiv:2310.01405) · 仓库 andyzoujm/representation-engineering (MIT)
> 数据：`/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA`（7 天第一人称中文日记 L1–L5 多粒度 + `feelings_all_days.tsv` 438 条带时间戳情绪自陈金标准）

## 目标
把"我一天做了什么"的中文 caption → **逐时刻、6 基本情绪（happiness/sadness/anger/fear/disgust/surprise）的连续量化曲线**，回答"哪些时刻、什么情绪反应"，并用 `feelings_all_days.tsv` 金标准验证探针读数。

## 已锁定决策
- 官方复现：**最忠实 fp16 13B**（Llama-2-13b-chat，AutoDL 北京B区 vGPU-48GB）。
- 中文主力模型：**Qwen2.5-3B-Instruct**（探针不量化）。
- 取向量：**concept(reading) / LAT** —— 成对刺激取最后 token 残差 → 每层 PCA 方向 + `direction_signs`；打分 = 残差在方向上的投影标量。
- 打分窗口：**L1/L2 滑窗**（默认 N=8 事件 / stride=4，另备 ~3min 时间窗）。
- transformers 版本坑：RepE ≥4.42 失效、Qwen2.5 需 ≥4.37 → Qwen 阶段锁 **4.40.2**（重叠窗口）；Llama 忠实复现锁 **4.35.2**。
- RepE `rep_readers.py` 硬编码 `.cuda()`（4 处）→ monkeypatch 成 device-agnostic。

## 六阶段流水线
1. **官方复现**（Llama-2-13b fp16 跑 `emotion_concept`）：验 per-layer 6 分类准确率中层达峰 80–90%+（随机 17%）。
2. **抽 6 情绪向量**（Qwen2.5-3B，中文化刺激，LAT 抽全层方向）。
3. **选层**：每情绪选分类分离度最优层（预期中层 ~50–70% 深度）。
4. **打分**：caption → L2 滑窗 → 套同分布中文 chat 模板 → 投影 6 方向 → 逐窗×6 情绪矩阵（z-score + 分位 [0,1] 两套归一化）。
5. **验证**：按时间戳对齐 `feelings_all_days.tsv`，算命中率 / Pearson-Spearman / κ / 分歧 top-K（"嘴上平静、内部某情绪高"）。
6. **产出**：动态情绪表 + 日内 6 情绪曲线（叠金标准事件点）+ 选层曲线 + 分歧表。

## 执行编排（3 并行 agent，我做监控指挥）
- **服务器/复现线**：SSH 上 AutoDL（北京B区 vGPU-48GB，已开机）→ 装钉死 4.35.2 环境 + 下 Llama-2-13b → 跑 `emotion_concept` 复现；顺带备好 Qwen2.5-3B 环境（4.40.2）。
- **B · 情绪向量+选层线**：中文化 6 情绪刺激 + Qwen2.5-3B LAT 抽全层方向 + 选层 → 产出 `outputs/directions/emotion_vectors.pt` 交接 artifact（`directions[emotion][layer]`、`direction_signs`、`best_layer`、`model_name`、`chat_template_signature`）。
- **C · 数据流水线+打分+验证线**：解析 caption（健壮正则，跳脏行）→ L2 滑窗 → 金标准映射（可审计词典）→ 打分器（消费 B 的 artifact）→ 归一化 → 验证 → 可视化；本地 mock 残差先跑通全链路。
- **合流**：B 的向量 + C 的打分器 → 落到 GPU 跑真实抽取与打分 → 汇总动态情绪表 + 金标准验证。

## 当前进度
- ✅ 调研论文 + 代码 + 数据，流水线定稿，决策锁定。
- ✅ AutoDL 北京B区 GPU（RTX 4090 48GB）已租，PyTorch2.1/py3.10/cu12.1。
- ✅ **阶段1 官方复现成功**：Llama-2-13b fp16 跑 emotion_concept，峰值层 -26、6 情绪均值准确率 **97.3%**（随机 16.67%），完全复现论文趋势。
- ✅ **阶段2-3 中文向量+选层**：Qwen2.5-3B 抽 6 情绪全层方向，最优层 -12~-15（中层），选层准确率 0.95~1.00。
- ✅ **阶段4 真实打分**：本地 caption 全 7 天 3653 窗 × 6 情绪，Qwen-3B 真实残差投影。
- ✅ **阶段5 金标准验证**：对 438 条 feelings，真实探针命中率 **0.545 vs 随机 0.197（2.8 倍）**，极性相关 Pearson 0.386 / Spearman 0.415（随机 ~0），κ 0.051（随机 -0.01）。
- ✅ **阶段6 可视化**：逐天 6 情绪曲线（原始细粒度 + 滑动平滑）+ 7 天总览 + 分歧 top-K。
- ✅ **项目重构**：共享流水线（src/scripts/config）+ 按模型分目录（runs/<MODEL_TAG>/）。换模型只改 config.MODEL_TAG，见 runs/README.md。

## 目录结构
```
emotion-probe/
├── src/  scripts/  tests/  config.py   # 共享流水线（所有模型复用，config.MODEL_TAG 切换）
├── data/emotions_{zh,en}/              # 6 情绪刺激（中/英）
├── artifacts/                          # 模型无关共享中间产物（caption 解析/L2 滑窗/金标准/mock）
├── runs/<MODEL_TAG>/                   # 各模型专属产物：directions/select_layer/scores/validation/figures
├── docs/                               # plan_A_RepE / plan_B_SLM_NeuralMRI / RUNBOOK
└── PLAN.md
```

