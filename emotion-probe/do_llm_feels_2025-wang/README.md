# do_llm_feels — EmotionCircuits 复现与扩展线

> HKU Capstone · 机制可解释性子课题「LLM 有没有情感」
> 本线复现并扩展 **EmotionCircuits-LLM**(论文 *Do LLMs "Feel"? Emotion Circuits Discovery and Control*,arXiv:2510.11328,MBZUAI),
> 在 **Llama-3.2-3B-Instruct** 上定位六种情绪的内部表征与电路,并用 steering(注入/消融)验证因果性。

---

## 1. 本线在项目中的角色

整个 capstone 用机制可解释性方法探究「LLM 内部是否存在可定位、可干预的情绪表征」。三条线:

| 线 | 目录(`emotion-probe/` 下) | 方法侧重 |
|---|---|---|
| 旗舰线 RepE | `RepE-2023-Zou/` | Representation Engineering:reading vector / 表征层面的读出与控制 |
| **本线 EmotionCircuits** | `do_llm_feels_2025-wang/`(本目录) | **电路级定位 + 因果干预**:diff-of-means 情绪方向 → 定位 MLP 神经元 / attention head → steering 验证 |
| Gemma 线 | `anthropic-2026-gemma-version/` | 探针(probe)方向,另一模型族 |

本线可视作 RepE「表征读出」思路向「电路级定位 + 因果干预」的延伸:用 **diff-of-means** 得到每种情绪在残差流的方向向量(对应 RepE 的 reading vector),再进一步把该方向归因到具体 **MLP 神经元**和**注意力头**,整合成「情绪电路」,并通过直接调控电路实现可控情绪生成。

六种情绪:`anger / disgust / fear / happiness / sadness / surprise`。

---

## 2. 目录结构(区分自有代码 vs 打包的官方仓)

```
do_llm_feels_2025-wang/
├── README.md                     # 本文件
├── CLAUDE.md                     # 面向后续 AI / 开发者的架构说明
│
├── score_emocirc.py              # ★ 自有:用官方情绪方向向量给「真实第一人称叙事」打情绪分
├── gen_viz.py                    # ★ 自有:把打分结果渲染成交互式 HTML
│
├── data/
│   └── day1_L2_events.txt        # ★ 自有输入:LUCIA Day1 L2 第一人称中文事件流(2006 条带时间戳)
├── output/
│   └── scores_day1_emocirc.csv   # ★ 自有产物:500 窗口 × 6 情绪投影分(已随仓库提交)
├── viz/
│   └── emotion_activation_day1.html  # ★ 自有产物:可视化(已提交)
├── directions/                   # ⚠ 被 .gitignore 排除(*.pt),clone 后不存在
│   ├── emo_directions_mlp.pt        #   = repo 内 02 阶段产物的本地副本(MD5 一致,冗余)
│   └── emo_directions_attention.pt
│
└── repo/
    └── EmotionCircuits-LLM/      # 打包的官方复现仓(commit 16b3441)
        ├── README.md             # 官方七阶段 pipeline 说明(权威命令来源)
        ├── quick_start.py        # 官方一行命令情绪调控 demo
        ├── environment_simple.yml# conda 环境(name: emotion_circuits)
        ├── data/
        │   ├── sev.jsonl         # SEV(Scenario-Event with Valence)训练集
        │   └── test_set.jsonl    # 测试集
        ├── scripts/01_…07_…/     # 七阶段 pipeline 脚本
        └── outputs/llama32_3b/   # 已复现的中间/最终产物(部分大文件被排除,见 §6)
```

`★` = 本线自有代码/数据;`repo/EmotionCircuits-LLM/` = 原样打包的官方仓,二次开发请优先改自有脚本,勿改官方脚本逻辑。

---

## 3. 环境依赖与模型

### 3.1 模型(需 GPU)

- **Llama-3.2-3B-Instruct**(`meta-llama/Llama-3.2-3B-Instruct`,28 层,hidden=3072)。
- 需 HuggingFace 授权;通过环境变量提供 token:
  ```bash
  export HF_TOKEN="<你的 HuggingFace token>"   # 不要写进代码或提交
  ```
- 硬件:CUDA 11.8+,显存 ≥ 8GB(推荐 16GB+),磁盘 ≥ 20GB。CPU 也能跑 `quick_start.py`(加 `--device cpu`),但完整 pipeline 实际需 GPU。

### 3.2 Python 环境

用官方 `environment_simple.yml`(关键版本:torch 2.4.1 / transformers 4.46.3 / numpy 1.24.4 / pandas 1.5.3):

```bash
cd repo/EmotionCircuits-LLM
conda env create -f environment_simple.yml
conda activate emotion_circuits
```

> 自有脚本 `score_emocirc.py` / `gen_viz.py` 只依赖 `torch / transformers / numpy / pandas`,在同一环境即可运行。

### 3.3 API 密钥(仅 GPT 标注步骤需要)

官方 pipeline 的 **01 / 03 / 07** 阶段有 GPT 标注脚本(`*_label_*_with_gpt.py`),用 GPT-4o-mini 给生成文本打情绪标签。

⚠ **注意**:这些脚本里 `OpenAI(api_key="Your OpenAI API Key")` 是**明文占位符**,需自行改成从环境变量读取,严禁把真实密钥提交进仓库:

```bash
export OPENAI_API_KEY="<你的 key>"     # 然后把脚本里的 api_key 改为 os.environ["OPENAI_API_KEY"]
```

只跑 `quick_start.py` 或本线 `score_emocirc.py` 时**不需要** OpenAI key。

---

## 4. 快速开始

### 4.1 官方一行命令 demo(情绪调控)

```bash
cd repo/EmotionCircuits-LLM
python quick_start.py --input_text "My girlfriend forgot my birthday again." --emotion anger --scale 0.8
```

> ⚠ **clone 后无法直接跑**:`quick_start.py` 依赖 05 阶段的差分向量
> (`emo_diff_all.npz`、`attention_emotion_diff/emo_diff/{emotion}/L*.npy`),这些是 `*.npz/*.npy`,
> **已被顶层 `.gitignore` 排除**。需先在 GPU 上重跑 **Step 05**(见 §6)再用 demo。
> 电路配置 `global_circuit/{emotion}.json` 是随仓库提交的,不用重新生成。

### 4.2 本线扩展:给真实第一人称叙事打情绪分

把官方情绪方向向量应用到 `data/day1_L2_events.txt`(真实生活事件流),输出每个时间窗口的六情绪强度:

```bash
# 默认路径是 AutoDL 服务器绝对路径,本地/他机请用参数覆盖
python score_emocirc.py \
  --events     data/day1_L2_events.txt \
  --model      <本地 Llama-3.2-3B-Instruct 路径或 meta-llama/Llama-3.2-3B-Instruct> \
  --directions repo/EmotionCircuits-LLM/outputs/llama32_3b/02_emotion_directions/emo_directions_mlp.pt \
  --output     output/scores_day1_emocirc.csv \
  --layers     11-20
```

流程:解析事件 → 切窗口(N=8,stride=4)→ Llama last-token 残差(层 11–20)→ 投影到归一化情绪方向 → CSV。
所用 `emo_directions_mlp.pt` 在 repo 内是**已提交**的(见 §6),clone 后即可用,无需 GPU 之外的额外准备。

```bash
# 渲染交互式 HTML(纯 CPU,读 data/ 和 output/,写 viz/)
python gen_viz.py
# → viz/emotion_activation_day1.html
```

---

## 5. 完整复现 pipeline(需 GPU,命令来自官方 README)

在 `repo/EmotionCircuits-LLM/` 下依次执行。标 🖥 = 需 GPU,标 🔑 = 需 OpenAI key。

```bash
cd repo/EmotionCircuits-LLM

# 01 基于提示的情绪激发生成 🖥🔑
python scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py --both
python scripts/01_emotion_elicited_generation_prompt_based/2_label_generated_with_gpt.py --both
python scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py --both

# 02 情绪方向提取(diff-of-means)🖥  →  emo_directions_{mlp,attention}.pt
python scripts/02_emotion_direction_extraction/1_dump_residual_aligned_sublayer_activations.py \
  --input_path outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl
python scripts/02_emotion_direction_extraction/2_compute_emotion_directions.py

# 03 基于方向向量的 steering 生成(验证方向因果性)🖥🔑
python scripts/03_emotion_elicited_generation_steer_based/1_steer_with_emotion_direction.py
python scripts/03_emotion_elicited_generation_steer_based/2_label_steered_with_gpt.py
python scripts/03_emotion_elicited_generation_steer_based/3_generate_accuracy_stats.py

# 04 局部组件识别(神经元 / 注意力头贡献)🖥  →  contrib_mean_*.csv(大文件)
python scripts/04_local_components_identification/1_compute_neuron_contrib.py
python scripts/04_local_components_identification/2_compute_head_contrib.py

# 05 情绪差分向量计算 🖥  →  emo_diff_all.npz + L{layer}.npy(quick_start 依赖)
python scripts/05_emotion_diff_vector_computation/1_dump_interv_points_activations.py
python scripts/05_emotion_diff_vector_computation/2_compute_emotion_mlp_diff.py
python scripts/05_emotion_diff_vector_computation/3_compute_emotion_attn_diff.py

# 06 情绪电路整合 🖥  →  global_circuit/{emotion}.json + global_ref/*.npy
python scripts/06_emotion_circuit_integration/1_analyze_emotion_direction_similarity.py
python scripts/06_emotion_circuit_integration/2_compute_sigma_from_residuals.py
python scripts/06_emotion_circuit_integration/3_compute_sublayer_importance_multi_alpha.py
python scripts/06_emotion_circuit_integration/4_analyze_sublayer_importance.py
python scripts/06_emotion_circuit_integration/5_integrate_global_circuit.py

# 07 基于电路的情绪生成(最终验证,论文报 99.41% 准确率)🖥🔑
python scripts/07_emotion_elicited_generation_circuit_based/1_enhance_global_circuit.py
python scripts/07_emotion_elicited_generation_circuit_based/2_visualize_global_circuit.py
python scripts/07_emotion_elicited_generation_circuit_based/3_baseline_text_generation.py
python scripts/07_emotion_elicited_generation_circuit_based/4_circuit_steer_all_valences.py
python scripts/07_emotion_elicited_generation_circuit_based/5_label_circuit_emotion_text.py
python scripts/07_emotion_elicited_generation_circuit_based/6_generate_accuracy_stats.py
```

各阶段输出目录见官方 README §"Full Pipeline"。情绪电路规模:每情绪 392 个 MLP 神经元 + 168 个注意力头,覆盖全部 28 层,重要层集中在 15–27。

---

## 6. 被 .gitignore 排除的大文件 & 如何重新生成

顶层仓库 `/Users/.../hku capstone/.gitignore` 排除了张量/向量与一张神经元大表。clone 后这些文件缺失,需在 **GPU 上重跑对应阶段**重新生成。

| 缺失文件 | 排除规则 | 重新生成方式(在 `repo/EmotionCircuits-LLM/` 下) |
|---|---|---|
| `directions/*.pt`(本线) | `*.pt` | 跑 **Step 02** 得 `outputs/llama32_3b/02_emotion_directions/emo_directions_{mlp,attention}.pt`,再复制到 `directions/`。**注意**:repo 内那两份 `.pt` 已被嵌套 `.gitignore` 例外保留并提交,clone 后存在,本线脚本可直接用 repo 路径,`directions/` 副本仅为冗余便利 |
| `04…/mlp_neurons/contrib_mean_*.csv`(6 个,各 ~26–28MB) | `**/04_local…/mlp_neurons/contrib_mean_*.csv` | 跑 **Step 04 / 1_compute_neuron_contrib.py**(依赖 01 的 `accepted.jsonl` + 02 的 `emo_directions_mlp.pt`) |
| `05…/mlp_emotion_diff/emo_diff_all.npz`、`05…/attention_emotion_diff/emo_diff/{emotion}/L*.npy` | `*.npz` / `*.npy` | 跑 **Step 05**(3 个脚本)。⭐ `quick_start.py` 依赖这两类文件,clone 后必须先跑 05 |
| `06…/global_ref/*.npy`、`06…/emotion_direction_similarity/*_similarity_matrix.npy` | `*.npy` | 跑 **Step 06** |

> 仍随仓库提交的「轻产物」:`02` 两个 `.pt`(嵌套 gitignore 例外)、`global_circuit/{emotion}.json`、`head_importance_*.csv`、各 `*_summary*.json`、`importance_*.csv`、`*.png`、`sev.jsonl`/`test_set.jsonl`、本线 `scores_day1_emocirc.csv` 与 HTML。

---

## 7. 数据来源

- **`repo/.../data/sev.jsonl`、`test_set.jsonl`**:官方 **SEV(Scenario-Event with Valence)** 数据集,随官方仓打包。每条含 theme / scenario / event(positive·neutral·negative)。用于 01 阶段情绪文本生成。
- **`data/day1_L2_events.txt`**:本线自有输入,**LUCIA Day1 L2 第一人称中文生活事件流**(2006 条,`[HH:MM:SS -> HH:MM:SS] 文本`),用于把官方情绪方向迁移到真实第一人称叙事上做情绪强度刻画。

---

## 8. 参考

- 论文:*Do LLMs "Feel"? Emotion Circuits Discovery and Control*,arXiv:2510.11328(MBZUAI,Chenxi Wang et al.)
- 官方仓:打包于 `repo/EmotionCircuits-LLM/`(commit 16b3441);上游见仓库 README 内链接。
- 模型:[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
