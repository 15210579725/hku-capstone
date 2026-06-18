# Emotion-Probe 复现实施方案 A — 基于 RepE（HKU Capstone）

> 论文：Representation Engineering (Zou et al. 2023, arXiv:2310.01405)
> 仓库：https://github.com/andyzoujm/representation-engineering (MIT)
> 数据：`/Users/mac/Desktop/emotion-action predict/事件粒度迭代/A4_LUCIA_DAY1/`
> 项目根：`/Users/mac/Desktop/hku capstone/emotion-probe`

## 0. 已核实事实
- 项目/数据真实路径在 `/Users/mac/...`（`/Users/xox406` 只是别名，不可写）。
- 主数据 `daily_B_最细事件.txt`：1001 行，11:09:56 → 22:02:39（约 11h），中文第一人称带时间戳。
- 同目录 `daily_A_原子动作.txt`（7322 行，最细）、`daily_C_中等粒度.txt`（265 行，已按事件聚合，每行一个时间段小结）。
- RepE 真实 API（照抄，勿臆造）：
  - `from repe import repe_pipeline_registry; repe_pipeline_registry()` 注册 `"rep-reading"` / `"rep-control"`。
  - `from utils import primary_emotions_concept_dataset`（`examples/primary_emotions/utils.py`）；模板 `'{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '`。
  - `rep_reading_pipeline.get_directions(train_data, rep_token=-1, hidden_layers=..., n_difference=1, train_labels=..., direction_method='pca', batch_size=...)`。
  - 打分：`rep_reading_pipeline(test_data, rep_token=-1, hidden_layers=[L], rep_reader=rep_reader)` → `{layer: H}`；概念分 = `project_onto_direction`（`H·direction/‖direction‖`）再乘 `direction_signs[layer]`。
  - `data/emotions/`：happiness/sadness/anger/fear/disgust/surprise.json（各 ~200 英文场景句）。
- **关键工程事实**：`repe/` 内无任何硬编码 `.cuda()`，device 由 HF Pipeline 管理 → **MPS 接入成本低**。

## 1. 推荐模型
- **主选 `Qwen/Qwen2.5-3B-Instruct`**：中文母语级；fp16 ≈6GB 权重，MPS 总占用约 7–9GB（16GB 够，需关大内存应用）；必须 Instruct（chat 模板 + 自陈指令遵循）；**探针不量化**（量化污染残差读数）。
- **备选 A `Qwen/Qwen2.5-1.5B-Instruct`（Apache-2.0）**：fp16 ≈3GB，先用它跑通最小 demo，再切 3B 跑正式实验，代码全共用，只改 model id。
- 备选 B（Yi-1.5-6B / glm-4-9b）中文更强但 16GB MPS 吃力，仅做一次性对照。

## 2. 环境与 MPS 接入
```bash
conda create -n emoprobe python=3.11 -y && conda activate emoprobe
pip install torch torchvision torchaudio        # Apple Silicon 默认即 MPS
pip install transformers accelerate scikit-learn matplotlib pandas tqdm
git clone https://github.com/andyzoujm/representation-engineering.git third_party/representation-engineering
pip install -e third_party/representation-engineering
```
要点：`device = "mps" if torch.backends.mps.is_available() else "cpu"`；`from_pretrained(..., torch_dtype=torch.float16).to(device)`（手动 `.to("mps")` 比 `device_map="auto"` 稳）；`pipeline("rep-reading", model=..., tokenizer=..., device=device)`（唯一可能试错点：某些 transformers 版本要整数 device id → 退路是不传 device、模型已 `.to("mps")`）；用 **fp16 不用 bf16**；跑前 `export PYTORCH_ENABLE_MPS_FALLBACK=1`。PCA 在 CPU/sklearn 算，与 device 无关。

## 3. 构建 4 个情绪方向
| 目标 | 来源 | 说明 |
|---|---|---|
| happiness（合并开心/幸福） | 现成 `happiness.json` | 合并为单一正向维度；valence/arousal 拆分列 future work |
| sadness | 现成 `sadness.json` | 直接用 |
| anxiety（焦虑，**自建**） | 新建 `anxiety.json` | RepE 默认六情绪无焦虑；强调"预期性/不确定"而非急性威胁 |
| fear（对照） | 现成 `fear.json` | 验证 anxiety≠fear（方向余弦） |

**中文策略（推荐 c 双语混合，主力中文）**：把 RepE 英文场景句翻成中文、在中文域内构造正负对、模板中文化（用 `tokenizer.apply_chat_template`）；保留英文版做跨语言一致性消融。
中文模板示例：`请判断下面这段经历中的「{emotion}」程度。\n经历：{scenario}\n回答：`
焦虑语料自建：Qwen 批量生成"持续担忧型"日常场景 → 人工剔除与 fear/sadness 重叠 → 复用 RepE pair 机制（其他情绪作负样本）。
训练：扫全层选最优层（预期中间层 ~50%，3B 36 层→约 -18~-22）；`rep_token=-1`；方向符号用 `direction_signs` 自动定，勿手翻。

## 4. 窗口化 + per-window 打分
- **默认窗口 = `daily_C_中等粒度.txt`（265 行，每行已语义完整）**；备选 `daily_B` 滑窗（每 N=5 行 / 每 5 分钟）。正则 `^\[(\d2:\d2:\d2) -> (\d2:\d2:\d2)\]\s*(.*)$`。
- 每窗套**与训练同分布的中文 chat 模板** → `rep_reading_pipeline` 取 best_layer 投影 × `direction_signs` = 原始情绪分。
- 归一化：A) 沿时间 z-score（画日内曲线）；B) 用刺激分布 min-max/分位映射到 [0,1]（跨情绪 + 对比自陈）。两者都算。

## 5. 探针 vs LLM 自陈 对比
- 自陈 prompt（数字孪生当事人，强制 JSON 0–10 打分 开心/悲伤/焦虑/恐惧 + 主要情绪）。探针走 forward、自陈走 generate，**分两遍但同窗口文本同模板前缀**。
- 指标：Pearson/Spearman、二值化一致率/Cohen's κ、divergence=|z(探针)−z(自陈)| 取 top-K。
- **核心 finding**：找"嘴上平静/开心、内部 sadness/anxiety 高"的对齐"洗白"案例。
- 可视化：双轴时间序列、4×1 分面散点+回归、分歧 top-K 表、一致性热图。

## 6. 目录结构
```
emotion-probe/
├── third_party/representation-engineering/   # clone + pip install -e
├── data/{emotions_en/, emotions_zh/(含自建 anxiety.json), daily/}
├── src/config.py data_prep.py build_directions.py select_layers.py
│      parse_daily.py score_trajectory.py self_report.py compare.py utils_mps.py
├── artifacts/   └── notebooks/explore.ipynb
```

## 7. 风险/坑
MPS pipeline device 参数兼容（退路：模型 `.to("mps")` + 不传 device）｜中文跨语言污染（主力中文+末层语言信号强故取中间层）｜anxiety≈fear（训 fear 对照算余弦）｜窗口太短信号弱（用 C 粒度 / 滑窗 N≥5）｜自陈被对齐洗白（这是 finding 不是 bug）｜16GB OOM（先 1.5B、batch 1~8、关浏览器）｜PCA 符号/归一化不一致（用 direction_signs + 统一归一化）｜模板训练/推理不一致（都用 apply_chat_template）｜不改原始数据（软链只读）。

## 8. 里程碑（合计 ~8.5–11 天）
M0 环境 0.5d → M1 英文两情绪 MPS 跑通 1d → M2 四情绪中文方向+选层+验证 anxiety≠fear 2–3d → M3 日内曲线（切 3B）1.5d → M4 自陈 vs 探针对比 2d → M5 消融+报告 1.5–2d。M1 是关键卡点（MPS device 参数）。

## 9. 相对 SLM/Neural-MRI 路线
RepE 优势：自带 6 情绪刺激 + reading/control pipeline、**MPS 接入成本低**、reading 投影标量天然就是"连续情绪值"。建议 **RepE 做主干**，借用 2604.04064 的两个结论：(1) 选层锚定中间层；(2) 若中文 comprehension 分离度不够，切 generation-based 抽取（RepE 的 `emotion_function` 风格中文版）。组合优于二选一。

### 关键文件
- `third_party/.../examples/primary_emotions/utils.py`（模板与 pair 构造，中文化基准）
- `third_party/.../repe/rep_reading_pipeline.py`（get_directions / 投影 / device 接入点）
- `third_party/.../repe/rep_readers.py`（PCARepReader / direction_signs / project_onto_direction）
- 数据 `daily_C_中等粒度.txt`（默认窗口）、`daily_B_最细事件.txt`（高分辨率滑窗消融）
