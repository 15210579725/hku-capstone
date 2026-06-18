# Emotion-Probe 复现实施方案 B — 基于 SLM 情绪向量 / Neural-MRI（HKU Capstone）

> 论文：Extracting and Steering Emotion Representations in Small Language Models (Jihoon Jeong, 2026, arXiv:2604.04064)
> 仓库：https://github.com/JihoonJeong/Neural-MRI (MIT)
> 数据：`/Users/mac/Desktop/emotion-action predict/事件粒度迭代/A4_LUCIA_DAY1/daily_B_最细事件.txt`
> 项目根：`/Users/mac/Desktop/hku capstone/emotion-probe`

## 0. 一句话结论
- **路径：抽核心逻辑自写脚本（~200–400 行），不要起整套 Neural-MRI Web 平台**（React/D3/WebSocket/SAELens/Docker 与本任务错位，16GB M1 起服务是负担）。仓库当"参考实现 + 刺激语料来源"。核心数学=差值均值+单位归一化+残差投影。
- **模型：主选 Qwen2.5-1.5B-Instruct**，备选 3B-Instruct；论文恰在 Qwen 上发现跨语言情绪纠缠（给你跨语言迁移的学术依据）。
- **设备：抽取与投影用 CPU + fp32，不要 MPS。** ⚠️ **TransformerLens 官方警告 MPS 会"静默产生错误 logits/激活"(issue #1178)**，对依赖残差投影的本任务致命；1.5B 在 M1 CPU 上单 forward 秒级，够用。
- **抽取方法：理解式（主结果）+ 生成式（论文复现亮点）。**

## 1. 推荐模型
- **主选 `Qwen/Qwen2.5-1.5B-Instruct`**：中文优先（硬约束）；instruct 必需（生成式抽取仅 instruct + 自陈需指令遵循）；论文跨语言纠缠发现就在 Qwen2.5 上（可引用支点）；TransformerLens 走 `qwen2` 架构加载；CPU fp32 ≈7–9GB 安全。
- 备选 1 `Qwen2.5-3B-Instruct`（论文 main 封顶 3B，因 TransformerLens 架构限制）；备选 2 `Qwen2.5-0.5B` / `gpt2-medium`（仅冒烟）。
- 正式全程 instruct；可加载 1.5B base 只跑理解式做对照。

## 2. 环境与接入
```bash
git clone https://github.com/JihoonJeong/Neural-MRI "/Users/mac/Desktop/hku capstone/Neural-MRI-ref"  # 只读参考+取英文刺激
pip install "transformer_lens>=2.0" torch transformers numpy pandas matplotlib scipy scikit-learn plotly
# 不需要 fastapi/uvicorn/saelens/react/docker
```
设备：`DEVICE="cpu"; DTYPE=torch.float32`（规避 #1178）。加载 `HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device="cpu", dtype=torch.float32)`。
**降级方案**：若 TransformerLens 对 Qwen2.5 加载报错 → 直接 HF transformers + `register_forward_hook` 到 `model.model.layers[L]` 抓 hidden_states，等价且无损（任务 1/2 不强依赖 steering）。

## 3. 抽取 3 情绪向量 + neutral 基线
映射：开心/幸福→`happy`(可加权 blissful/enthusiastic/grateful)，焦虑→`anxious`(nervous 近邻)，悲伤→`sad`，基线→`neutral`。
> 注：用户列 4 标签但实为 3 情绪（开心/幸福合一）+ neutral 基线（数学必需，非第 4 情绪）。脚本"情绪列表可配置"，要凑满 4 正式情绪可补 `angry`，改一行配置。

差值均值（论文原文）：
```
a_e = mean_passages( resid_中间层_lasttoken(passage) );  a_neu = 同 for neutral
v_e = (a_e - a_neu);  v_e = v_e / ‖v_e‖    # 单位归一化
# 生成式取"生成序列中点 token"；理解式取"最后 token"；都取中间层(~50%)
```
中间层：1.5B 28 层 → 取 **L14（resid_post）**。**复现 U 形层扫描**：{7,14,21,28} 层各抽一组，用 Cohen's d / Mann–Whitney 算分离度画 U 形确认 50% 最佳。
中文刺激（A+B 都做做消融）：A) 用论文英文 passages 抽向量再验证对中文是否迁移（论文预言会，省事且有学术价值）；B) 自建中文第一人称刺激（与 daily_B 同分布）。

## 4. 窗口化 + per-window 投影（任务 1）
窗口：W1 固定行窗（每 10 行步长 5，~200 窗）/ W2 每 5 分钟时间窗（**推荐主用**）/ W3 直接用 daily_C 每行一窗（最省事）。
打分：`h = resid_L14_lasttoken(x)`；`score_e = dot(h, v_e)`（v_e 已单位归一化）。**h 与 v_e 必须同层同 hook 同 token 位置**。
归一化：1) 沿时间 z-score（情绪轨迹主读数）；2) 沿情绪 softmax（主导情绪占比，堆叠面积图）。
可视化：多线图（11:00–22:00 三情绪 + 事件标注）、堆叠面积、热力图；可选 plotly 交互。

## 5. 探针 vs 自陈（任务 2）
复用仓库已有 `Emotion detect/prompt_template.txt` 中文模板，改造为数字孪生当事人自陈，强制 JSON 0–100（开心/焦虑/悲伤 + 说明）。同一 Qwen 既做探针又做自陈 = "同一大脑两个读数"。
指标：Pearson/Spearman（逐情绪沿时间）、|z(probe)−z(self)| top-k 分歧、相邻窗变化方向一致率。
**核心论点**：探针对负面情绪比自陈更敏感（自陈受 RLHF 报喜不报忧）→ "内部表征 vs 对齐后输出"分离。
可视化：双轴叠图、散点+回归、分歧 top-k 表。

## 6. 目录结构
```
emotion-probe/
├── config.py                 # 模型/设备(cpu+fp32)/层(14)/情绪列表/窗口参数
├── data/{stimuli_en.json, stimuli_zh.json}
├── src/model_loader.py activations.py extract_vectors.py layer_sweep.py
│      windowize.py project_scores.py self_report.py compare.py visualize.py
├── scripts/00_smoke_gpt2.sh 01_build_vectors.sh 02_layer_sweep.sh
│           03_trajectory.sh 04_self_report_compare.sh
└── outputs/
```
`activations.py` 是地基（定义"层/hook/token 位置"三要素，抽向量与投影共用，决定数值正确性）。

## 7. 风险/坑
**TransformerLens MPS #1178 静默错误（致命→强制 CPU+fp32）**｜TransformerLens 对 Qwen2.5 支持不全（降级 HF hook）｜中文刺激质量｜跨语言不迁移（回退中文原生）｜生成式耗时（passage 减到 3–5）｜自陈报喜不报忧（这是 feature）｜窗口超长（限 token / 用 daily_C）｜投影尺度不可比（强制 z-score+softmax）｜情绪向量相关（报余弦，必要时 Gram-Schmidt 正交化）。

## 8. 里程碑（合计 ~8–10 工作日）
M0 环境+gpt2 冒烟 0.5–1d → M1 向量抽取 1d → M2 层扫描 U 形 0.5d → M3 任务1 日内轨迹 1–1.5d → M4 中文刺激+跨语言对照 1d → M5 生成式抽取复现亮点 1d → M6 任务2 自陈对比 1.5d → M7 报告+消融 1.5–2d。

## 9. 相对 RepE 路线
差值均值：实现极低、确定性、可解释，但多概念可能相关、投影打分需自己封装；RepE LAT-PCA：天然正交多概念、reading 对任意输入打分更现成，但有随机性。建议 **本方案差值均值做主线（契合指定论文+简单稳健），RepE-PCA 做 §7 对照消融**（激活已抓好，换个降维零额外成本）。组合 > 二选一。
