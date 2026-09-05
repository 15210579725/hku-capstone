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

可直接用于分析的 caption 结果位于 [`caption-result/`](caption-result/)。本批包含 24 条录制；每条录制目录内有：

- `captions.jsonl`：逐 clip 的结构化结果，推荐程序读取；每行一个 JSON，包含 `clip_id`、HKT 时间范围、`scene_summary`、`activity_chain`、`segments`、`speech`、`text_visible` 以及 `has_pass2` 等字段。
- `captions.txt`：便于人工快速浏览的纯文本版本。

批次概况和每条录制的完成覆盖率见 [`caption-result/index.json`](caption-result/index.json)。注意：`index.json` 中的 `clips_done` / `coverage` 是交付时的实际统计，部分长录制只完成了部分 clip，分析时请按 JSONL 中的记录数和 `ok` 字段筛选。

最小读取示例：

```python
import json
from pathlib import Path

for path in Path("caption-result").glob("*/captions.jsonl"):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("ok")]
    print(path.parent.name, len(rows), rows[0]["scene_summary"] if rows else "")
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
