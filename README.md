# HKU Capstone — 第一人称行为预测

从第一人称 (egocentric) POV 眼镜视频生成 dense caption，并评测行为预测质量。

## 仓库结构

```
caption/        视频 → 结构化 caption（Gemini 三遍精读）
metric/         行为预测评测工具（Embedding + LLM Judge）
```

---

## caption/ — Dense Video Captioning

从 HuggingFace 上的 POV 录制 tar 文件，通过 Gemini 生成逐秒 caption。**不需要下载视频到本地**——通过 HTTP Range 直接远程读取。

### 三遍架构

1. **Pass 0** — silero-vad 语音检测（精度 ~0.03s），Gemini 填词
2. **Pass 1** — 2880px 全帧 caption + 文字区域框选
3. **Pass 2** — 裁切文字区域 ULTRA_HIGH 精读，修订 Pass 1

### 用法

```bash
cd caption
pip install -r requirements.txt
```

编辑 `config.json`，填入 HuggingFace token 和 Google Cloud 项目 ID（[新用户 $300 免费](https://cloud.google.com)）。

```bash
python run_pipeline.py list                          # 查看可用录制
python run_pipeline.py run --day 2026-05-18          # 处理一天
python run_pipeline.py run --tar aria/xxx.tar        # 处理单条
python run_pipeline.py run --tar aria/xxx.tar --max-clips 3  # 冒烟测试
```

输出：`caption_output/<date>/<recording>/captions_full.json`（含每段 action / environment / speech / text_visible）。

详见 [caption/README.md](caption/README.md)。

### 已交付 Caption 数据

可直接用于分析的 caption 结果位于 [`caption-result/`](caption-result/)。这是 2026-09-07 从 Hugging Face `mmm8383/pov-captions` 拉取的**当时已完成部分**，不是全部计划数据已经完成：HF 上共有 217 条录制、44,938 个 caption JSON 文件；按文件大小 `>= 2500 bytes` 选出 42,552 个候选，最终发布 42,304 个顶层 `ok=true` 的 clip（候选内覆盖率 99.4%），过滤 248 个失败 clip，共含 228,494 个细分 segment。每条录制目录内有：

- `captions.jsonl`：逐 clip 的结构化结果，推荐程序读取；每行一个 JSON。顶层包含 `clip_id`、HKT 时间范围、`model`、`recording` 和 `ok`；语义结果在 `parsed`（兼容旧结果时也可检查 `merged`）中，包括 `scene_summary`、`activity_chain`、`screen_text` 和 `segments`，每个 segment 含 `action`、`environment`、`speech`、`text_visible` 等字段。
- `captions.txt`：便于人工快速浏览的纯文本版本。

批次概况、HF 来源快照和每条录制的候选内覆盖率见 [`caption-result/index.json`](caption-result/index.json)。其中 `target_clips` / `selected_candidate_clips` 指本次从 HF 选中的候选数，历史计划目标另存为 `source_planned_target_clips`。文本中的敏感信息（凭据、token、邮箱、电话、姓名、地址、支付信息、URL/本地路径等）已替换为 `XXX`，完整计数型脱敏审计见 [`caption-result/redaction-audit.json`](caption-result/redaction-audit.json)，不含原始匹配值。发布副本中的 `content_raw` 是结构化结果的冗余原始模型输出，已统一置为 `XXX` 以缩小隐私暴露面；分析请使用 `merged` 或 `parsed`。注意：`clip_index` 可能稀疏；分析时以实际 JSONL 行数及 `ok` 字段为准，不要假设编号连续，也不要把这个快照当作剩余 HF 作业已经完成。

最小读取示例：

```python
import json
from pathlib import Path

for path in Path("caption-result").glob("*/captions.jsonl"):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("ok")]
    parsed = (rows[0].get("merged") or rows[0].get("parsed") or {}) if rows else {}
    print(path.parent.name, len(rows), parsed.get("scene_summary", ""))
```

---

## metric/ — 行为预测评测

评测 "预测未来 K 步行为" 的质量。两种方法，**API key 已预填，装完依赖直接跑**。

| 方法 | 原理 | 模型 |
|------|------|------|
| **Embedding** | 文本→向量，余弦相似度 | Qwen3-Embedding-8B (SiliconFlow) |
| **LLM Judge** | 大模型 0-9 语义打分 | Gemini-3.7-flash (RightCode) |

多步预测用 **Soft Edit Distance**：传统编辑距离只能精确匹配，SED 用语义相似度作为替换代价，允许"近似匹配"得部分分。

### 用法

```bash
cd metric/eval_kit
pip install -r requirements.txt
```

准备输入（JSONL，每行一个样本）：

```jsonl
{"id": "001", "ground_truth": ["我打开冰箱", "我拿出水"], "prediction": ["我走到冰箱前", "我拿了饮料"]}
```

运行：

```bash
python evaluate.py -i input.jsonl -m both       # Embedding + Judge
python evaluate.py -i input.jsonl -m embedding   # 仅 Embedding
python evaluate.py -i input.jsonl -m judge       # 仅 Judge
python evaluate.py -i input.jsonl --dry-run      # 验证格式
```

输出 `eval_results.jsonl`（逐条得分）和 `eval_summary.json`（汇总统计）。

> **额度说明**：预填的 key 是项目组的限量额度，用完需自行注册充值。Embedding 在 [SiliconFlow](https://cloud.siliconflow.com) 注册（有免费额度），Judge 在 [RightCode](https://www.right.codes) 注册（付费）。这两个 API 兼容 OpenAI 接口，拿到后也可以直接调 embedding 或 LLM 做其他事。

详见 [metric/eval_kit/README.md](metric/eval_kit/README.md)。
