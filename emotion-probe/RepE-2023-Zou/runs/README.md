# runs/ — 按模型分目录的实验产物

每个模型一个子目录，**共享流水线代码（`src/`、`scripts/`、`config.py`）不变**，
只有各模型自己的产物落在这里。换模型只改一处：`config.MODEL_TAG`（或设环境变量
`EMOPROBE_MODEL_TAG`），所有产物会自动写到 `runs/<新 tag>/`。

## 目录结构（每个模型一致）
```
runs/<model-tag>/
├── directions/      emotion_vectors.pt + manifest.json   ← 6 情绪全层方向 + best_layer + signs
├── select_layer/    select_layer.json                    ← 各层分类准确率 + 选层
├── scores/          scores_L2_*_real.parquet             ← 逐窗 × 6 情绪真实分（原始/z/分位）
├── validation/      validation_metrics_real.json 等       ← 对 feelings 金标准的验证指标
└── figures/         day{1..7}_emotion_curves.png 等        ← 逐天情绪曲线 + 7 天总览 + 分歧表
```

## 已完成的模型
- **qwen2.5-3b-instruct**：Qwen2.5-3B-Instruct（fp16 不量化）。最优层 -12~-15，
  选层准确率 0.95~1.00；对 438 条金标准验证命中率 0.545（随机 0.197）、极性相关 0.39/0.42。

## 跑一个新模型（端到端）
GPU 实例（AutoDL，conda env 见 docs/RUNBOOK）上：
```bash
export EMOPROBE_MODEL_TAG=<新模型短名>          # 例如 llama3-8b-instruct
# 1. 抽 6 情绪方向 + 选层 → runs/<tag>/directions, select_layer
python scripts/run_extract.py --model <模型路径> --lang zh
# 2. 用本地 caption 窗口真实打分 → runs/<tag>/scores
python scripts/run_score_real.py --model <模型路径>          # 省略 --day = 全 7 天
# 3. 验证（对 feelings 金标准）→ runs/<tag>/validation
python scripts/run_validate_real.py
# 4. 出图（逐天 + 7 天总览）→ runs/<tag>/figures
python scripts/plot_all_days.py --smooth_k 21
```
共享中间产物（caption 解析、L2 滑窗、金标准时间线）在根级 `artifacts/`，跨模型复用、无需重跑。

## 注意
- 6 情绪顺序固定 `[happiness, sadness, anger, fear, disgust, surprise]`，方向 artifact 须一致。
- 打分用的中文 chat 模板必须与抽向量时同分布（`chat_template_signature` 自动校验）。
- 探针模型一律不量化（避免污染残差读数）。
