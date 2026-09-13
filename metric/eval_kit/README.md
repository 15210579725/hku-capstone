# 行为预测标准评测工具

评测模型对未来 K 步行为的预测质量。支持两种评测方法，可单独使用或组合使用。

## 评测方法

### 1. Embedding (嵌入余弦相似度)

将 ground-truth 和 prediction 文本通过 embedding 模型转化为向量，用余弦相似度衡量语义距离。

- **单步 (K=1)**: 直接计算 cosine similarity，得分范围 [0, 1]
- **多步 (K>1)**: 计算 Soft Edit Distance — 在传统编辑距离的基础上，用 `1 - cosine` 作为替换代价（而非 0/1 的精确匹配），从而允许"近似匹配"的步骤获得部分得分

**推荐模型**: Qwen3-Embedding-8B（在 4 人标注的 350 点基准上 pairwise accuracy 0.9290，超过所有 LLM-judge）

### 2. LLM Judge (大模型语义评分)

用大模型对每对 (ground-truth, prediction) 进行 0-9 的语义相似度打分。

- **单步 (K=1)**: judge 分数 / 9 即为得分
- **多步 (K>1)**: 构建评分矩阵，用 `1 - score/9` 作为 Soft Edit Distance 的替换代价

**推荐 Prompt**: P3NT（0-9 量表，无 CoT，直出整数。在 6 版 prompt × 3 模型横评中综合最优）

### Soft Edit Distance (SED)

传统编辑距离只能精确匹配，语义相近但措辞不同的步骤被判为完全不同。SED 用连续的语义代价替换二元匹配：

```
替换代价 = 1 - similarity(gt[i], pred[j])
插入/删除代价 = 1.0

最终得分 = 1 - SED / max(len_gt, len_pred)
```

当 K=1 时，SED 退化为直接的相似度分数。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

仅需 `numpy`（Python 3.8+）。

### 2. 配置 API

`config.json` 中的 `embedding.api_key` 和 `judge.api_key` 均为空，需要使用自己的 API key。

建议将配置复制到仓库外，填入所需的 key，再通过 `--config` 指定该文件：

```bash
mkdir -p ~/.config/hku-capstone
cp config.json ~/.config/hku-capstone/eval-kit.json
chmod 600 ~/.config/hku-capstone/eval-kit.json
# 编辑该文件，填写 embedding.api_key 和 judge.api_key
python evaluate.py -i predictions.jsonl --config ~/.config/hku-capstone/eval-kit.json -m both
```

不要将填写了真实密钥的配置提交到 Git。

> **💡 直接调用 API**
> 拿到这两个 API 后，除了评测，你也可以直接调用它们做其他事：
> - **Embedding API** (SiliconFlow): 兼容 OpenAI 接口，可直接用 `openai` SDK 调用 `Qwen/Qwen3-Embedding-8B` 生成文本向量，做检索/聚类/相似度计算等。
> - **LLM API** (RightCode): 兼容 OpenAI 接口，可直接调用 `gemini-3.7-flash` 做文本生成、分析、翻译等任意 LLM 任务。

只用一种方法的话，另一种 `api_key` 留空即可。

### 3. 准备输入文件

JSONL 格式，每行一个样本：

```jsonl
{"id": "sample_01", "ground_truth": ["我拿起手机"], "prediction": ["我拿出手机查看消息"]}
{"id": "sample_02", "ground_truth": ["我打开冰箱", "我拿出水"], "prediction": ["我走到冰箱前", "我拿了饮料"]}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本唯一标识 |
| `ground_truth` | list[string] | K 步 ground-truth 行为 |
| `prediction` | list[string] | K 步模型预测行为 |
| `context` | string (可选) | 上下文信息 |

`ground_truth` 和 `prediction` 也接受单个字符串（自动视为 K=1）。

### 4. 运行评测

```bash
# 两种方法都运行
python evaluate.py -i predictions.jsonl -m both

# 仅 embedding
python evaluate.py -i predictions.jsonl -m embedding

# 仅 LLM judge
python evaluate.py -i predictions.jsonl -m judge

# 先验证输入格式
python evaluate.py -i predictions.jsonl --dry-run
```

### 5. 查看结果

运行后在输入文件同目录生成：

- `eval_results.jsonl` — 每条样本的详细得分
- `eval_summary.json` — 汇总统计（均值/标准差/中位数等）

## 推荐平台

### Embedding 平台

| 平台 | 模型 | 说明 |
|------|------|------|
| **SiliconFlow (硅基流动)** ★推荐 | Qwen/Qwen3-Embedding-8B | 免费额度充足，效果最佳 |
| 阿里云百炼 (DashScope) | text-embedding-v3 | base_url 改为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，config 中 `instruction` 设为空字符串 |
| OpenAI | text-embedding-3-small | config 中 `instruction` 设为空字符串，`dimensions` 设为 1536 |

**SiliconFlow 注册**: 访问 https://cloud.siliconflow.com 注册账号，在控制台创建 API Key。注册即送免费额度。

### LLM Judge 平台

默认模型为 **Gemini-3.7-flash**（在我们的 350 点基准上排序准确率最高）。通过 **RightCode 中转站**访问，**付费**。使用前需自行配置 `judge.api_key`。

| 中转站 | 说明 | config 配置 |
|--------|------|------------|
| **[RightCode](https://www.right.codes)** ★推荐 | 支持 Gemini / GPT / Claude 等多家模型 | base_url: `https://www.right.codes/gemini/v1`，model: `gemini-3.7-flash` |

在 RightCode 控制台获取自己的 API Key，填写到仓库外配置副本的 `judge.api_key`。

**备选模型** (如不想用 RightCode):

| 平台 | 模型 | config 修改 |
|------|------|------------|
| 阿里云百炼 | qwen3.5-flash | base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| DeepSeek | deepseek-chat | base_url: `https://api.deepseek.com/v1` |
| OpenAI | gpt-4o-mini | base_url: `https://api.openai.com/v1` |

## 输出格式

### eval_results.jsonl

```json
{
  "id": "multi_01",
  "ground_truth": ["我打开冰箱", "我拿出水", "我关上冰箱门"],
  "prediction": ["我走到冰箱前", "我拿了饮料", "我关好冰箱"],
  "n_steps_gt": 3,
  "n_steps_pred": 3,
  "embedding": {
    "score": 0.8234,
    "soft_edit_distance": 0.5298,
    "cosine_matrix": [[0.72, 0.45, 0.33], [0.41, 0.85, 0.37], [0.31, 0.35, 0.91]]
  },
  "judge": {
    "score": 0.8519,
    "soft_edit_distance": 0.4444,
    "judge_matrix": [[6, 3, 2], [3, 8, 3], [2, 2, 9]]
  }
}
```

### eval_summary.json

```json
{
  "input": "predictions.jsonl",
  "n_samples": 100,
  "embedding": {
    "method": "embedding",
    "n_samples": 100,
    "score_mean": 0.7845,
    "score_std": 0.1234,
    "score_median": 0.8012,
    "score_min": 0.3456,
    "score_max": 0.9876,
    "sed_mean": 0.2155,
    "elapsed_seconds": 12.3
  },
  "judge": { "..." : "..." }
}
```

## 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | (必填) | 输入 JSONL 文件路径 |
| `--config` | `-c` | `config.json` | 配置文件路径 |
| `--method` | `-m` | `both` | 评测方法: `embedding` / `judge` / `both` |
| `--output-dir` | `-o` | 输入文件同目录 | 输出目录 |
| `--concurrency` | | 10 | 覆盖并发数 |
| `--dry-run` | | | 仅验证输入格式 |

## 基准数据

在 350 个人工标注的行为预测点（4 位标注者）上：

| 方法 | Pairwise Accuracy | Top-1 Accuracy |
|------|-------------------|----------------|
| Embedding (Qwen3-8B cosine) | **0.9290** | 0.9021 |
| LLM Judge: Gemini-3.7 P3NT | 0.9044 | 0.9155 |
| LLM Judge: GPT-5.6 P3NT | 0.8936 | **0.9217** |
| LLM Judge: Qwen3.5 P3NT | 0.8794 | 0.9200 |
| 人类标注者互评 | 0.8383 | — |

- Embedding 在排序一致性上超过所有方法（包括人类标注者）
- LLM Judge 在 Top-1 准确率上更优
- 两种方法互补，建议同时使用
