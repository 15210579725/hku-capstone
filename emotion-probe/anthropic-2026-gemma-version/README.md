# Emotion Probe — Gemma 版(anthropic-2026-gemma-version)

港大 capstone「LLM 有没有情感」课题下的一条子线。本线在 **Gemma 4 E4B** 上复现并延伸
Anthropic 论文 *"Emotion Concepts and their Function in a Large Language Model"*
(Sofroniew et al., Transformer Circuits, 2026)所描述的「情绪表达探针(expression probe)」方法,
并把它扩展到一组 **8 种情绪(含 curiosity 好奇心)**,最后用这些探针方向给中文/英文日记片段逐 token 打分并可视化。

> 参考实现:Ryan Codrai 的开源仓库 `gemma-emotional-probes`,已原样打包在本目录
> `gemma-emotional-probes-main/`,供对照与「忠实复现线」直接调用。

---

## 1. 本线在大项目中的角色

- 这是旗舰 RepE(Representation Engineering / difference-of-means)思路在 **Gemma 模型 + 多情绪** 上的落地。
- 方法核心:用「对比故事」造数据 → 抽隐藏态 → 求 difference-of-means 方向 → 用中性文本 PCA 去混杂
  → 归一化得到每个情绪在每层的方向向量 → 用方向给任意文本逐 token 投影打分。
- 8 种情绪:`sad, curious, excited, bored, anxious, happy, tormented, angry`。

---

## 2. 目录结构

```
anthropic-2026-gemma-version/
├── README.md                     # 本文件
├── CLAUDE.md                     # 面向后续开发者/AI 的架构说明
│
├── scripts/                      # 【自有】数据准备(英文故事生成/下载 + 翻译)
│   ├── download_hf_stories.py        # 从 HuggingFace 下载 7 情绪 + neutral 英文故事
│   ├── generate_curiosity_stories.py # 用 gpt-5.4 API 生成 curiosity 故事(HF 数据集没有)
│   ├── translate_stories.py          # 英→中翻译(asyncio 版)
│   ├── translate_stories_v2.py       # 英→中翻译(同步 + 自适应批次,更稳)
│   └── translate_segments_en.py      # 中文日记片段→英文(保留时间戳前缀)
│
├── extraction/                   # 【自有 / 快速线】抽隐藏态 + 算情绪向量
│   ├── extract_8emotions.py          # GPU:8 情绪 + neutral 的逐层 mean 激活
│   └── compute_vectors.py            # 由激活算出 en_vectors.pt(diff-of-means + 中性 PCA 去混杂)
│
├── official_run_staging/         # 【忠实复现线】对官方脚本只做「接口级」改动后的运行版
│   ├── extract_story_activations.py       # GPU:8 情绪故事 → 全 42 层激活
│   ├── extract_neutral_story_activations.py # GPU:1200 中性故事 → (1200,42,2560)
│   └── compute_expression_vectors.py      # → emotion_vectors_all_layers.pt
│
├── scoring/                      # 【自有】日记打分
│   ├── score_diary.py                # GPU:layer 23 逐 token 投影打分 → diary_scores.json
│   └── diary_scores.json             # 已生成的打分结果(已提交)
│
├── visualization/                # 【自有】可视化产物与批处理
│   ├── batch_diary.py                # 把 25 个日记片段(中+英)POST 给官方 Flask 可视化器
│   ├── data/diary_scores.{json,js}   # 打分结果(JSON / 供静态页用的 JS 包装)
│   ├── index*.html, diary_index.html # 静态可视化页面
│   ├── figures/*.png                 # 情绪曲线、雷达图、总览图
│   └── diary_html/                   # Flask 可视化器导出的逐片段 HTML + summary
│
├── data/                         # 数据(中小体积,已提交)
│   ├── stories_en/<emotion>.json     # 8 情绪英文故事,各 1200 条(list[str])
│   ├── stories_zh/<emotion>.json     # 对应中文翻译
│   ├── neutral/neutral_{en,zh}.json  # 中性故事(PCA 去混杂用)
│   ├── selected_segments.json        # 25 个中文日记片段(带 expected_emotions)
│   └── selected_segments_en.json     # 上者的英文版
│
├── vectors/                      # 【大文件,被 .gitignore 排除】见第 6 节
│   ├── en_vectors.pt                  (≈496M)  快速线产物
│   ├── emotion_vectors_all_layers.pt  (≈496M)  忠实复现线产物
│   ├── neutral_stats.json             (已提交)  逐层统计,z-score 用
│   └── diary_zscore_stats.json        (已提交)  layer28 上 zh/en 的 μ/σ,可视化归一化用
│
└── gemma-emotional-probes-main/  # 【官方参考】Ryan Codrai 原仓库,原样保留
    ├── visualise.py                  # Flask 可视化器(本线忠实复现线直接用它)
    ├── extraction/                   # 官方抽取/计算脚本(staging 即由此改写)
    └── agents/                       # 官方数据生成 agent(本线未跑,用 HF 现成数据)
```

> 自有代码(`scripts/ extraction/ scoring/ visualization/`)与官方代码
> (`gemma-emotional-probes-main/`)严格分开;`official_run_staging/` 是「对官方脚本只改路径/数据接口、不改算法」的桥接层。

---

## 3. 环境依赖与模型

- **Python 3.10+**;关键依赖:
  ```bash
  pip install torch transformers tqdm pandas huggingface_hub openai requests flask
  ```
  (本目录没有 requirements.txt;按上面这行装即可。)
- **模型:`google/gemma-4-E4B`**(42 层,hidden_dim 2560)。
  - 需要 **GPU**(脚本用 `device_map="cuda"`、`bfloat16`);权重约 15GB。
  - 需要 **HuggingFace 访问权限 / 已接受 Gemma 许可**,并把权重缓存到本地。
  - 抽取/打分脚本设了 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`,默认从本地缓存离线加载,
    因此请先在有网环境 `huggingface-cli download google/gemma-4-E4B` 缓存好。

> ⚠️ 关于模型:目录文档约定本线为「Gemma」线,但**所有可执行脚本里写死的 `MODEL_ID` 是
> `google/gemma-4-E4B`**(不是 Gemma-2-9B)。请以代码为准;若要换模型,需同时改 `MODEL_ID`、
> 确认 `NUM_LAYERS`/`HIDDEN_DIM` 与层访问路径 `model.model.language_model.layers`。

---

## 4. 环境变量(API 密钥)

仅 `scripts/` 下的「生成 / 翻译」脚本需要调用外部 LLM API。它们统一从环境变量读 key,**仓库内不含任何明文 key**:

```bash
export TRANSLATE_API_KEY="你的key"     # 生成 curiosity 故事 + 所有翻译脚本都用它
```

说明:
- 这些脚本通过 **OpenAI 兼容接口** 调用模型 `gpt-5.4`,`base_url` 在脚本里写死为
  `https://once.novai.su/v1`。若你的中转地址不同,直接改脚本顶部的 `BASE_URL`/`API_BASE_URL` 常量。
- 本线脚本**不使用** `DEEPSEEK_API_KEY`,也**不读** `TRANSLATE_API_BASE_URL`(它们属于本仓库其他子线的约定)。
- GPU 阶段(extraction/scoring/visualise)**不需要任何 API key**,纯本地推理。

---

## 5. 端到端运行

> 标注:🖥️ = 需 GPU;🔑 = 需 `TRANSLATE_API_KEY`;💻 = 本地 CPU 即可。
> 多数脚本顶部把输入/输出路径**写死成 AutoDL 服务器路径**(如 `/root/...`、`/autodl-fs/...`、
> `/Users/mac/Desktop/hku capstone/gemma-probes/...`)。clone 后请按自己的环境**修改脚本顶部的路径常量**再运行。

### 阶段 A — 准备故事数据(可跳过:`data/` 已含成品)
```bash
# A1 🔑(可选) 下载 7 情绪 + neutral 英文故事(curiosity 不在 HF 数据集里)
python scripts/download_hf_stories.py
# A2 🔑 生成 curiosity 英文故事(100 主题 × 12 变体 ≈ 1200)
python scripts/generate_curiosity_stories.py
# A3 🔑 英文故事 → 中文(二选一,v2 更稳)
python scripts/translate_stories_v2.py
# A4 🔑 中文日记片段 → 英文(用于中英对照可视化)
python scripts/translate_segments_en.py
```

### 阶段 B — 抽激活 + 算向量(任选一条线)

快速线(layer 23 打分用):
```bash
# B1 🖥️ 8 情绪 + neutral 的逐层 mean 激活 → en_*.pt
python extraction/extract_8emotions.py
# B2 💻 由激活算情绪方向 → vectors/en_vectors.pt
python extraction/compute_vectors.py
```

忠实复现线(官方方法,可视化器用):
```bash
# B1' 🖥️ 8 情绪故事 → 全 42 层激活
python official_run_staging/extract_story_activations.py
# B2' 🖥️ 1200 中性故事 → (1200,42,2560)
python official_run_staging/extract_neutral_story_activations.py
# B3' 💻 → vectors/emotion_vectors_all_layers.pt
python official_run_staging/compute_expression_vectors.py
```

### 阶段 C — 打分 / 可视化
```bash
# C1 🖥️ 快速线:给 25 个日记片段逐 token 打分(layer 23) → diary_scores.json
python scoring/score_diary.py

# C2 🖥️ 忠实复现线:先起官方 Flask 可视化器(读 emotion_vectors_all_layers.pt)
python gemma-emotional-probes-main/visualise.py     # 监听 :8080
# 另开一个进程把 25 个片段(中+英)POST 进去,导出逐片段 HTML + summary
python visualization/batch_diary.py                  # 默认 layer 28、probe_mode=expression
```
静态结果可直接看 `visualization/index.html` / `diary_index.html` 与 `visualization/figures/*.png`。

---

## 6. 被 .gitignore 排除的大文件如何重新生成

仓库根 `.gitignore` 用 `*.pt` 排除了所有张量文件。本目录两个需要重建的产物:

| 文件 | 体积 | 重新生成方式 |
|---|---|---|
| `vectors/en_vectors.pt` | ≈496M | 跑阶段 B 快速线:`extract_8emotions.py` → `compute_vectors.py` |
| `vectors/emotion_vectors_all_layers.pt` | ≈496M | 跑阶段 B 忠实复现线:`extract_story_activations.py` + `extract_neutral_story_activations.py` → `compute_expression_vectors.py` |

两者都需要 🖥️ GPU 跑 Gemma 4 E4B 抽隐藏态。中间激活文件(`en_*.pt` / `activations_all_layers/*.pt` /
`neutral_activations.pt`)更大,同样被忽略,按上表脚本重算即可,无需提交。

> `vectors/neutral_stats.json`、`vectors/diary_zscore_stats.json` 是小 JSON,已随仓库提交,不必重建。

---

## 7. 数据来源

- **英文情绪故事(7 种)+ 中性故事**:HuggingFace 数据集
  [`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes)
  (官方论文配套数据,`expression/stories.parquet`、`expression/neutral_stories.parquet`)。
- **curiosity 故事**:HF 数据集没有,用 `gpt-5.4` 现生成(`generate_curiosity_stories.py`)。
- **中文故事 / 英文日记**:由上述英文/中文经 `gpt-5.4` 翻译得到。
- **日记片段 `selected_segments.json`**:25 段第一人称中文日记(带时间戳与 `expected_emotions` 标注),
  来自本组自有日记数据,作为探针的下游测试集。
- **方法 / 官方代码**:Anthropic Transformer Circuits 2026 论文 + Ryan Codrai 的
  `gemma-emotional-probes`(本目录 `gemma-emotional-probes-main/`)。
</content>
</invoke>
