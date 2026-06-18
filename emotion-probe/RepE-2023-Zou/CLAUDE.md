# CLAUDE.md — Emotion-Probe · RepE 线（面向后续 AI / 开发者）

本文件给后续接手的 AI / 开发者，说明这条线的**架构、关键设计决策与坑、数据流、与其它子项目的关系**。
面向使用者的「怎么跑」见 `README.md`；执行计划与进度见 `PLAN.md`；GPU 复现细节见 `docs/`。

## 0. 一句话定位

RepE reading（LAT / concept）向量法：中文日记 caption → 逐时刻 6 基本情绪连续量化曲线
→ 用 438 条带时间戳 feelings 金标准验证。它是 capstone 的**方法论母本**，
`do_llm_feels` 与 `gemma` 线是它在不同模型 / 不同论文上的延伸（见 §6）。

## 1. 六阶段流水线架构与 src 模块职责

数据流是一条单向链，所有阶段都吃 `config.py` 的全局常量（路径 / 6 情绪顺序 / 滑窗参数 / MODEL_TAG）：

```
                         ┌─────────────── 共享中间产物 artifacts/（模型无关，已提交）
LUCIA events.txt ──①parse_captions──②windowize── windows_L2_count.parquet
feelings_all_days.tsv ──③parse_feelings(+emotion_mapping)── feelings_timeline.parquet
emotions_zh/*.json ──②stimuli──②extract(rep-reading)──③select_layer──②io_artifact──┐
                                                                                     ▼
                                          runs/<TAG>/directions/emotion_vectors.pt（GPU 产物，.gitignore 排除 *.pt）
windows + emotion_vectors.pt ──④score(RealResidualProvider)── runs/<TAG>/scores/scores_*_real.parquet
                              ──⑤normalize(z + 分位)── scores_norm_*.parquet
                              ──⑥validate(对齐 feelings)── runs/<TAG>/validation/*.json
                              ──⑦visualize / plot_all_days── runs/<TAG>/figures/*.png
```

| 阶段 | 模块 | 职责 / 关键不变量 |
|---|---|---|
| ① 解析 caption | `src/parse_captions.py` | 行格式 `[HH:MM:SS -> HH:MM:SS] 文本`；健壮正则（箭头空格可有可无、小时 1~2 位）；脏行/截断行标 `parsed_ok=False` 跳过、不抛异常。时间转当天零点起秒数。 |
| ② 滑窗 | `src/windowize.py` | 默认事件计数窗 N=8 / stride=4（重叠 50%）；备选 ~3min 时间窗。末尾残窗保留标 `partial=True`。窗 schema：`win_id/day/t_start/t_end/mid_time/n_events/text/partial`。 |
| ③ 金标准 | `src/parse_feelings.py` + `src/emotion_mapping.py` | 438 条自陈 → 时间线；细情绪→6 基本情绪用**可审计关键词词典**（确定性、不用模型），多命中取优先级最先匹配，未命中→neutral。 |
| ② 刺激 | `src/stimuli.py` | 逐字复刻 RepE `primary_emotions_concept_dataset` 的成对/shuffle/labels/train-test 口径；中文走 Qwen `apply_chat_template`，英文保留原始模板做消融。 |
| ② 抽取 | `src/extract.py` | rep-reading pipeline，`hidden_layers=range(-1,-num_layers,-1)`（全层负索引），`rep_token=-1 / n_difference=1 / direction_method='pca'`。产 `RepReader`（directions / direction_signs / H_train_means）。 |
| ③ 选层 | `src/select_layer.py` | 复刻 notebook results：test 集目标恒在偶位，两两一组，`sign==-1`用 min 否则 max，命中=极值==组内第 0 个。每情绪选 acc 最高层，并列取更靠中层。 |
| ② 落盘 | `src/io_artifact.py` | 写交接契约 `emotion_vectors.pt` + 人读 `manifest.json`（含打分公式与 python 示例）。 |
| ④ 契约+mock | `src/direction_artifact.py` | 标准 `Artifact` dataclass + `adapt_artifact` 适配层（隔离字段名差异）+ `make_mock_artifact`（QR 取近正交方向，供本地自测）。 |
| ④ 打分 | `src/score.py` | `MockResidualProvider`（信号注入 / 纯随机双模式）与 `RealResidualProvider`（加载 Qwen 取 best_layer 最后 token 残差）。打分 `raw = sign · (h · d/‖d‖)`。 |
| ⑤ 归一化 | `src/normalize.py` | 两套：按天 z-score（`{e}_z`，画曲线）+ 经验 CDF 分位映射 [0,1]（`{e}_q`，跨情绪可比）。 |
| ⑥ 验证 | `src/validate.py` | feelings 按 `mid_time` 对齐覆盖窗；情绪时刻命中率（z>0.5σ）、极性 Pearson/Spearman、二值一致率 + Cohen's κ、分歧 top-K。 |
| ⑦ 可视化 | `src/visualize.py` + `scripts/plot_all_days.py` | 日内 6 情绪 z 曲线叠 feelings 事件竖线；分歧表；7 天总览（原始细粒度淡背景 + 滑动平滑主曲线）。matplotlib Agg，缺中文字体退化英文。 |

## 2. 关键设计决策与坑（务必先读）

### 2.1 transformers 版本（最致命）
- RepE 自定义 `RepReadingPipeline` 在 transformers **≥4.42 失效**（issue #56，pipeline 内部改动）；
  Qwen2.5 又需 **≥4.37**。→ **Qwen 阶段钉死 `transformers==4.40.2`**（重叠窗口），见 `env/requirements_probe.txt`。
- 官方 Llama-2-13b 忠实复现另锁 **4.35.2**（见 `docs/RUNBOOK_AutoDL_RepE复现.md`），加载加 `attn_implementation="eager"` 防 flash-attn 报错。
- 升级 transformers 前务必想清楚：宁可不升。

### 2.2 RepE 的 `.cuda()` monkeypatch（`src/repe_compat.py`）
- RepE 源码 `repe/rep_readers.py` 的 `project_onto_direction` / `recenter` **两个函数硬编码 `.cuda()`**（共 4 处调用），本地无 NVIDIA GPU（macOS CPU/MPS）必崩。
- `repe_compat.apply_compat()` 把这两个模块级函数 monkeypatch 成 device-agnostic 版本
  （以 `direction.device` 为锚点，H 跟随；语义不变 `H·dir/‖dir‖`），GPU 端同样安全；并幂等注册 rep-reading pipeline。
- `get_signs / transform` 内是按名字从模块全局查这两个函数，所以**替换模块属性即可全局生效**。
- 不要去改 `third_party/` 源码——补丁集中在 `repe_compat.py`，patch 状态会写进 artifact 的 `patch_info`。

### 2.3 两套归一化（z-score / 分位），都算、各有用途
- **按天 z-score**（`{e}_z`）：每 day 每情绪列独立 (x−μ)/σ → 画日内相对走势、命中率判据（z>0.5σ）。
- **经验 CDF 分位 [0,1]**（`{e}_q`）：跨情绪可比、对比自陈。ref 分布默认用本列全体；GPU 阶段可换成刺激打分分布。
- 二者输出同存，下游按需取。**不要只留一套**——曲线要 z、跨情绪比较要 q。

### 2.4 选层策略
- 不是固定取某层，而是**每情绪各选 test 分类准确率最高的层**，并列时取 `|layer|` 最接近中层中位数的（稳定可复现）。
- Qwen2.5-3B（36 层）实测最优层落在 **-12 ~ -15**（中层 ~50~70% 深度），acc 0.95~1.00，符合 RepE「中层达峰」结论。
- 末层语言信号强、易被中文跨语言污染，故取中层。

### 2.5 模板同分布护栏
- 抽向量与真实打分**必须用同一渲染器、同一 tokenizer 的 chat 模板**，否则残差不可比。
- `stimuli.chat_template_signature` 生成一条样例签名写进 artifact；`RealResidualProvider._lazy_load` 加载时重新渲染签名与 artifact 比对，不一致直接 `RuntimeError`。改模板时这条护栏会拦你。

### 2.6 探针不量化
- 量化会污染残差读数（投影标量是连续情绪值，对数值精度敏感）。Qwen 探针**一律 fp16 不量化**。
- 省钱用的 8bit 只在官方 Llama-2 复现「验趋势」时可接受（见 RUNBOOK），正式探针不用。

### 2.7 mock 的意义
- `MockResidualProvider`：对命中情绪词典的窗，沿对应方向注入弱信号（含 sign，使投影后为正），其余纯随机。
- 自测判据是 **signal > random**（命中率 / 极性相关显著优于纯随机基线）——证明流水线对信号敏感、非平凡，
  而**不依赖 GPU / 真实模型**就能验证整条链路接口正确。`inject_strength=0` 退化为纯随机 baseline。

## 3. 数据流中的路径约定（config.py）

- `DATA_ROOT`：LUCIA 原始日记根，**绝对路径、在仓库之外**（换机器需改）。caption/feelings 严格只读。
- `ARTIFACTS_DIR`（`artifacts/`）：**模型无关**共享中间产物（解析/滑窗/金标准/mock），跨模型复用，已提交。
- `RUNS_DIR/<MODEL_TAG>/`：**模型专属**产物（directions/select_layer/scores/validation/figures）。
  `MODEL_TAG` 由 `EMOPROBE_MODEL_TAG` 环境变量或默认 `qwen2.5-3b-instruct` 决定，import config 时自动建目录。
- 只有 `runs/<TAG>/directions/emotion_vectors.pt` 被 `.gitignore`（`*.pt`）排除，需 GPU 重抽；其余 parquet/json/png 均提交。
- `validate.py` / `visualize.py` 里有个约定：tag 含 `mock`/`signal` 的产物落共享 `artifacts/`，真实/random 的落 `runs/<TAG>/`。

> 已知小瑕疵（不影响主流程）：`scripts/run_extract.py` 的 `--out_select` 默认值是历史遗留的
> `PROJECT_ROOT/outputs/select_layer`，与现行 `runs/<TAG>/select_layer` 不一致；跑时显式传
> `--out_select runs/<TAG>/select_layer`（或用 `config.SELECT_LAYER_DIR`）。脚本 docstring 里
> 个别示例路径还是旧的 `outputs/directions/`，以 argparse 默认值 + `runs/README.md` 为准。

## 4. 交接契约 emotion_vectors.pt（torch.save 一个 dict）

打分器只认这个结构（字段见 `src/io_artifact.py` 顶部 docstring）。核心字段：

```
directions[emotion][layer]      # PCA comp0, (hidden_size,), 未单位归一
direction_signs[emotion][layer] # ±1
best_layer[emotion]             # int（负索引）
H_train_means[emotion][layer]   # (1, hidden) recenter 复现用
chat_template_signature         # 模板同分布校验锚
```

打分公式（与 RepE 一致）：`score = sign · (h @ d) / ‖d‖`，可选先 `h - H_train_means` recenter（复现 RepE transform）。
`direction_artifact.adapt_artifact` 是字段别名适配层——若上游换了命名，**只改这一处**。

## 5. 验证指标口径（避免误读）

- `strict_hit_rate`：仅非 neutral、非争议项计入（real 0.545 / random 0.197）。
- 极性相关：探针极性分（正情绪和 − 负情绪和）vs 自陈极性，Pearson/Spearman（real 0.386/0.415 / random ≈ -0.05）。
- `cohen_kappa` 偏低（0.051）属正常：feelings 中性占比 0.78、严格命中样本仅 66 条，类别极不平衡；
  看相对优势（real ≫ random）而非绝对值。`binary_accuracy` real≈random 也是不平衡导致（多数类是 neutral）——别被它误导，主看命中率与极性相关。
- 核心 finding 是**分歧 top-K**：找"嘴上平静/正向、内部某负面情绪 z 高"的对齐"洗白"案例（探针比自陈对负面更敏感）。

## 6. 与其它子项目的关系

- 本线（RepE）是 **方法论母本**：reading 向量 + 投影打分 + 金标准验证这套范式。
- `do_llm_feels` 线：把同一范式迁到另一篇论文 / 另一套模型上的延伸。
- `gemma` 线：把探针范式落到 Gemma 系模型（见 capstone 顶层记忆：layer 选择、RRF/z-score 坑、token 波动等结论）。
- 三条线共享"抽方向 → 选层 → 投影打分 → 对自陈金标准验证"的骨架，差异在模型 / 论文 / 抽取细节。
  改本线流水线时注意接口稳定性（6 情绪顺序、artifact 契约、chat 模板同分布护栏），下游线可能复用。

## 7. 给后续 AI 的操作提醒

- 改任何 `src/` 模块后，先跑 `python3 -m pytest tests/`（6 passed）+ `python3 scripts/run_mock_selftest.py`（signal>random）保证没破链路。
- 不需要 GPU 就能验证绝大部分逻辑（mock 残差）；只有抽向量 / 真实打分需 Qwen 前向。
- 6 情绪顺序、负索引层口径、`chat_template_signature` 校验是硬约束，别动。
- 原始 LUCIA 数据只读；一切产出写 `artifacts/`（共享）或 `runs/<TAG>/`（模型专属）。
- 不要升 transformers、不要量化探针、不要改 `third_party/` 源码（补丁走 `repe_compat.py`）。
