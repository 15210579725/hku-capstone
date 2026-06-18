# Emotion-Probe · RepE 线（Representation Engineering, Zou et al. 2023）

> 用机制可解释性方法探究「LLM 有没有情感」——本目录是整个 HKU capstone 的**旗舰复现线**。
> 论文：Representation Engineering: A Top-Down Approach to AI Transparency（Zou et al. 2023, arXiv:2310.01405）
> 官方仓库：`andyzoujm/representation-engineering`（MIT），已固定在 `third_party/representation-engineering/`。

## 1. 这条线在总课题里的角色

总课题问的是「LLM 内部是否存在可读出的情绪表征」。本线给出的回答方式是：

把**中文第一人称日记 caption**（"我一天做了什么"）→ 用 RepE 的 **reading / concept（LAT）向量**方法
→ 读出**逐时刻、6 基本情绪（happiness / sadness / anger / fear / disgust / surprise）的连续量化曲线**
→ 再用 **438 条带时间戳的情绪自陈金标准**验证探针读数是否对得上。

核心数学很轻：成对刺激取最后 token 残差 → 每层 PCA 求方向 → 打分 = 残差在方向上的投影标量
（`score = sign · (h · d / ‖d‖)`）。它天然就是一个"连续情绪值"，不需要再训分类头。

本线是方法论母本，capstone 的另外两条线（`do_llm_feels`、`gemma` 线）是它在不同模型 / 不同论文上的延伸。

- **官方复现**：Llama-2-13b-chat（fp16，不量化），跑通论文 `emotion_concept`。
- **中文主力模型**：`Qwen/Qwen2.5-3B-Instruct`（探针**不量化**，量化会污染残差读数）。

## 2. 目录结构

```
RepE-2023-Zou/
├── config.py                 # 全局配置：路径 / 6 情绪顺序 / 滑窗参数 / MODEL_TAG（换模型只改这里）
├── src/                      # 共享流水线（所有模型复用，与模型无关）
│   ├── parse_captions.py     #  ① 解析 events.txt（健壮正则，跳脏行）
│   ├── windowize.py          #  ② L2 滑窗（默认 N=8 / stride=4，另备 ~3min 时间窗）
│   ├── parse_feelings.py     #  ③ 解析 438 条 feelings 金标准时间线
│   ├── emotion_mapping.py    #  ③ 细情绪→6 基本情绪 的可审计词典
│   ├── stimuli.py            #  ② 构 6 情绪成对刺激 + Qwen 中文 chat 模板（复刻 RepE 口径）
│   ├── extract.py            #  ② 用 rep-reading pipeline 抽全层 LAT 方向
│   ├── select_layer.py       #  ③ 各层分类准确率 → 每情绪选最优层
│   ├── io_artifact.py        #  ② 写/读交接契约 emotion_vectors.pt + manifest.json
│   ├── direction_artifact.py #  ④ 方向 artifact 契约 + mock 生成器 + 适配层
│   ├── score.py              #  ④ 打分器（真实残差 / mock 残差双模式）
│   ├── normalize.py          #  ⑤ 两套归一化（按天 z-score + 分位 [0,1]）
│   ├── validate.py           #  ⑥ 对齐金标准，算命中率 / 相关 / κ / 分歧 top-K
│   ├── visualize.py          #  ⑦ 日内曲线 + 分歧表
│   └── repe_compat.py        #  让 RepE 在 CPU/MPS/任意 device 跑通的兼容补丁
├── scripts/
│   ├── run_mock_selftest.py  # 本地端到端自测（不需 GPU/模型，用 mock 残差）
│   ├── run_extract.py        # 【需 GPU】抽 6 情绪方向 + 选层 → emotion_vectors.pt
│   ├── run_score_real.py     # 【需 GPU】Qwen 对窗口真实打分
│   ├── run_validate_real.py  # 真实分归一化 + 金标准验证 + 出图（本地，CPU 即可）
│   └── plot_all_days.py      # 7 天情绪曲线总览
├── tests/test_pipeline.py    # 6 个单元自测（pytest 或直接 python 跑）
├── data/emotions_{zh,en}/    # 6 情绪刺激语料（中文自建 / 英文取自 RepE 原仓）
├── artifacts/                # 模型无关共享中间产物（caption 解析 / 滑窗 / 金标准 / mock）
├── runs/<MODEL_TAG>/         # 各模型专属产物：directions/select_layer/scores/validation/figures
├── docs/                     # plan_A_RepE / plan_B_SLM_NeuralMRI / RUNBOOK_AutoDL
├── env/requirements_probe.txt# 探针环境依赖（含版本坑说明）
├── third_party/representation-engineering/  # 官方 RepE（pip install -e）
└── PLAN.md                   # 执行计划与进度
```

## 3. 环境与依赖

依赖清单见 `env/requirements_probe.txt`。**头号版本坑：`transformers==4.40.2` 必须钉死**——

- RepE 老代码在 transformers **≥4.42 会 break**（issue #56：pipeline 内部改动使 `RepReadingPipeline` 子类失效）；
- Qwen2.5 又需要 transformers **≥4.37**；
- 两者重叠窗口锁 **4.40.2**（Qwen 阶段）。官方 Llama-2 忠实复现另锁 **4.35.2**（见 `docs/RUNBOOK_AutoDL_RepE复现.md`）。

```bash
# 建议 conda 隔离环境
conda create -n emoprobe python=3.11 -y && conda activate emoprobe
pip install -r env/requirements_probe.txt
# 安装官方 RepE 本体（editable）
pip install -e third_party/representation-engineering
```

说明：
- `torch` 在 Apple Silicon 走默认 wheel（CPU + MPS）即可跑本地 mock 自测与验证；GPU 端在服务器用对应 CUDA wheel。
- 本项目主要是 **GPU 本地推理，不需要任何 API key**。若后续接入需要密钥的服务，一律从环境变量读取，**不要写明文**。

## 4. 如何运行

### 4.1 先跑本地 mock 自测（验证环境，不需 GPU / 不需模型）

mock 用可复现的伪残差（对命中情绪词典的窗注入弱信号），端到端验证整条流水线是否通：

```bash
python3 scripts/run_mock_selftest.py     # 8 阶段全链路：parse→window→feelings→artifact→score→normalize→validate→visualize
python3 -m pytest tests/                  # 6 passed（也可 python3 tests/test_pipeline.py）
```

本机已实跑通过：mock 自测 8 阶段全过、`signal > random`、artifacts 落盘；pytest **6 passed**。
判据是"信号注入版指标显著优于纯随机基线"，证明流水线对情绪信号敏感、非平凡。

> 注意：mock 自测会读取原始 LUCIA 日记（解析 caption、构滑窗、解析 feelings）。
> 原始数据路径在 `config.py::DATA_ROOT`（默认 `/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA/`，**在仓库之外**）。
> 若 clone 到别的机器没有这份数据：`artifacts/` 下已提交了**预生成的解析产物**
> （`windows_L2_count.parquet`、`feelings_timeline.parquet` 等，未被 .gitignore 排除），
> 下游归一化 / 验证 / 出图可直接复用，无需重新解析。需要重跑解析时把 `DATA_ROOT` 指到本地实际路径即可。

### 4.2 GPU 真实抽取 / 打分 / 验证

GPU 实例（AutoDL，环境搭建见 `docs/RUNBOOK_AutoDL_RepE复现.md`）上：

```bash
export EMOPROBE_MODEL_TAG=qwen2.5-3b-instruct          # 产物会落到 runs/<tag>/

# 【需 GPU】① 抽 6 情绪全层方向 + 选层 → runs/<tag>/directions/emotion_vectors.pt + select_layer/
python scripts/run_extract.py --model /root/autodl-tmp/Qwen2.5-3B-Instruct --lang zh \
    --out_select runs/$EMOPROBE_MODEL_TAG/select_layer
#   （--out_artifact 默认即 runs/<tag>/directions/emotion_vectors.pt；
#     --out_select 的默认值是历史遗留的 outputs/select_layer，建议如上显式指到 runs/<tag>/select_layer）

# 【需 GPU】② 用本地 caption 窗口真实打分 → runs/<tag>/scores/（省略 --day 即全 7 天）
python scripts/run_score_real.py --model /root/autodl-tmp/Qwen2.5-3B-Instruct

# 【CPU 即可】③ 归一化 + 对 feelings 金标准验证（含 random baseline 对照）→ runs/<tag>/validation/
python scripts/run_validate_real.py

# 【CPU 即可】④ 逐天 + 7 天总览情绪曲线 → runs/<tag>/figures/
python scripts/plot_all_days.py --smooth_k 21
```

哪些步骤需要 GPU：**只有 ①抽向量 和 ②真实打分**需要加载 Qwen 做前向（GPU）。
③验证 和 ④出图 只读已落盘的分数 parquet，本地 CPU 即可。

### 4.3 已有的真实结果（qwen2.5-3b-instruct，可直接查看）

`runs/qwen2.5-3b-instruct/` 下已提交一套真实跑通的产物（除 `.pt` 向量外，parquet/json/png 都在）：

- 选层：最优层 **-12 ~ -15**（中层），各情绪选层准确率 **0.95 ~ 1.00**（随机基线 0.5）。
- 金标准验证（438 条 feelings）：严格命中率 **0.545 vs 随机 0.197**（约 2.8 倍）；
  极性相关 Pearson **0.386** / Spearman **0.415**（随机 ≈ -0.05）；Cohen's κ **0.051**（随机 ≈ -0.01）。
- 规模参考：L2 滑窗共 **3653** 个窗（7 天）；caption 解析 14631 行成功率 99.99%（仅 1 行截断跳过）；
  feelings 438 条，中性占比 0.78。

## 5. 切换模型（config.MODEL_TAG）

换模型**只改一处**：`config.MODEL_TAG`，或设环境变量 `EMOPROBE_MODEL_TAG`（优先级更高）。

```bash
export EMOPROBE_MODEL_TAG=llama3-8b-instruct   # 例
```

所有模型专属产物会自动写到 `runs/<新 tag>/{directions,select_layer,scores,validation,figures}/`；
共享流水线代码（`src/`、`scripts/`、`config.py`）与模型无关的中间产物（`artifacts/`）都不变、跨模型复用。
约束：6 情绪顺序固定 `[happiness, sadness, anger, fear, disgust, surprise]`，方向 artifact 须一致；
打分用的中文 chat 模板必须与抽向量时同分布（`chat_template_signature` 自动校验）；探针一律不量化。

## 6. 被 .gitignore 排除的大文件如何重新生成

仓库根 `.gitignore` 排除了模型权重 / 向量 / 张量（`*.pt *.pth *.bin *.safetensors *.npz *.npy` 等）。
本线**唯一被排除的关键产物**是：

```
runs/<MODEL_TAG>/directions/emotion_vectors.pt     ← 6 情绪全层方向 + best_layer + direction_signs + H_train_means
```

重新生成方法：在 GPU 上跑 **§4.2 的 ①**（`scripts/run_extract.py`）即可重抽。
该 `.pt` 是 GPU 阶段抽向量的产物，CPU 端无法直接复现（需加载 Qwen 做前向）。
同目录的 `manifest.json`（人读的字段说明 + 打分公式示例 + 各情绪 best_layer/acc）未被排除、可直接看。

> 其余产物（`scores_*.parquet`、`validation_*.json`、`figures/*.png`、`select_layer.json`、
> `artifacts/*.parquet`）都**不在排除列表**，已随仓库提交，clone 后即可查看 / 复跑下游。

## 7. 数据来源

- **中文情绪刺激** `data/emotions_zh/*.json`（6 文件，每情绪约 139~161 句）：中文化自建语料，
  复用 RepE 的成对 / 配方机制（目标情绪 vs 其它 5 情绪），用于抽 6 情绪方向。
- **英文情绪刺激** `data/emotions_en/*.json`：取自 RepE 官方 `data/emotions/`，做跨语言一致性消融。
- **LUCIA 日记 caption**（`config.DATA_ROOT/<level>/day{d}/events.txt`）：7 天第一人称中文日记，
  多粒度（L1~L5），行格式 `[HH:MM:SS -> HH:MM:SS] 中文描述`。**在仓库之外**，是探针打分的输入文本。
- **feelings 金标准**（`config.DATA_ROOT/feelings_all_days.tsv`）：438 条带时间戳的情绪自陈，
  列 `day / line_no / time / category / content`，是验证探针读数的真值。**在仓库之外**。

LUCIA 原始日记与 feelings 金标准不随本仓上传；它们的解析产物已落在 `artifacts/`。
若要在新机器上从原始数据重跑，请把 `config.py::DATA_ROOT` 指向你本地的 A4_LUCIA 目录。
```
