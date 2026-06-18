# CLAUDE.md — concordia-official（面向后续 AI / 开发者）

## 0. 一句话定位

本目录 = **Google DeepMind Concordia 官方框架（@873dddd 原样打包，约 2 处本地改动）** + 项目自有的 **`lucia_sim/`** 行为预测仿真模块。在 capstone 总课题「LLM 有没有情感」的证据链里，它是「**情绪 → 行为**」中的**行为仿真环节**：用 LUCIA 第一人称 caption（EgoLife L1）驱动一个 LLM agent 做下一步动作预测，再与真实动作（GT）比对一致率。LLM 后端 = DeepSeek（远程 API），**本环节不依赖 GPU**。

面向使用者的安装/运行/数据说明见 `README.md`，本文件只讲架构、设计决策与坑。

## 1. lucia_sim 架构与核心模块

复用 Concordia 完整组件栈：`Config → Simulation → (GM + Entity) → Sequential Engine`。

| 模块 | 职责 |
|------|------|
| `data_parser.py` | 解析 L1 events.txt（`[HH:MM:SS -> HH:MM:SS] 中文`），用正则关键词分类 observation / internal / action；`build_decision_points()` 把流式事件切成决策点：每遇到一个 action 就发射一个决策点（累积的 obs/internal 作上下文，action 作 GT），发射后清空累积 |
| `ego_replay_gm.py` | 自定义 GM prefab，数据驱动回放（核心，见 §2） |
| `deepseek_model.py` | `DeepSeekLanguageModel`：直接实现 Concordia 的 `LanguageModel` 接口（`sample_text` / `sample_choice`），用 OpenAI SDK 打 DeepSeek（见 §4） |
| `run_simulation.py` | 主入口：建 model + embedder → 解析数据 → 配 prefab/instances → `Simulation.play()` → 抽取预测、写 HTML/JSON/TXT |
| `judge.py` | LLM Judge：异步并发（信号量 10）对每条 (prediction, gt_action) 用 DeepSeek 打 4 维分（semantic 0-4 / plausibility 0-2 / temporal 0-2 / granularity 0-2），含 JSON 与 `key=value` 双解析回退 |

## 2. 如何挂接 Concordia 组件栈（关键）

`run_simulation.py` 注册两个 prefab：
- `basic__Entity`（官方 `prefabs/entity/basic.py`）→ 实例 **Lucia**（带中文 goal）；
- `ego_replay__GameMaster`（自定义 `EgoReplayGameMaster`）→ GM，参数携带 `decision_points`。

GM 的精妙之处在 **`EgoReplayScript`（一个 `ContextComponent` 实例，被注册到 4 个标准 GM key 上）**：

```
next_action_spec_key  ┐
terminator_key        ├── 同一个 EgoReplayScript 实例
make_observation_key  │
resolution_key        ┘
next_acting_key       ── NextActingInFixedOrder(['Lucia'])  （官方组件）
act_component         ── SwitchAct（官方，按 output_type 路由到对应组件）
```

`EgoReplayScript.pre_act()` 按 `action_spec.output_type` 分流（数据驱动，几乎不调用 LLM 做环境判定）：

```
1. TERMINATE         → step_idx 耗尽决策点则返回 'Yes' 结束
2. MAKE_OBSERVATION  → 吐出当前决策点累积的观察文本（首步附角色介绍 _ROLE_INTRODUCTION）
3. NEXT_ACTING       → 'Lucia'（由 NextActingInFixedOrder 给）
4. NEXT_ACTION_SPEC  → 返回 free 类型中文行动指令（要求第一人称单一原子动作）
5. entity.act(spec)  → 这一步才真正调用 DeepSeek，让 Lucia 生成动作预测
6. pre_observe()     → 捕获带 PUTATIVE_EVENT_TAG 的 putative 事件，存进 _current_prediction
7. RESOLVE           → 记录 {step,timestamp,prediction,gt_action}，返回 'Lucia: <GT动作>' 推进，step_idx++
```

预测结果存在组件内 `self._predictions`；`run_simulation._extract_predictions()` 通过 `event_resolution.DEFAULT_RESOLUTION_COMPONENT_KEY` 从 GM 取回 `EgoReplayScript`，调 `get_predictions()` 落盘。

设计要点：**GM 完全数据驱动**——它不让 LLM 判定环境后果，只负责「喂观察 / 收预测 / 回放 GT」。唯一的 LLM 调用发生在 Lucia entity 生成预测时。这是 ego-replay 范式与官方「GM 用 LLM 裁决世界」用法的最大区别。

## 3. 数据流

```
events.txt (L1, ~1.5万行/day1)
   │ parse_l1_events() 正则解析 + classify_event() 分类
   ▼
[{start,end,text,type}]  (type ∈ observation/internal/action)
   │ build_decision_points(time_start, time_end)  按时间窗过滤 + 切分
   ▼
[{step,timestamp,observations[],gt_action,gt_raw}]  ← 决策点
   │ 注入 EgoReplayGameMaster.params['decision_points']
   ▼
Simulation.play()  逐步回放（每步 1 决策点，见 §2）
   ▼
predictions.json / interaction_log.txt / simulation_log.html
   │ judge.py 异步打分
   ▼
predictions_judged.json（averages + per-step scores）
```

默认窗口 11:09–11:15 → 64 个决策点（已落地于 `lucia_sim/output/morning_11_09_15/`，64 条预测的 judge 均值约 semantic 0.77/4、plausibility 1.22/2、temporal 1.72/2、granularity 1.50/2）。

## 4. 关键决策与坑

- **为何自写 `DeepSeekLanguageModel` 而非用官方 `openai.gpt_model.GptLanguageModel`**：官方 `GptLanguageModel` 面向 GPT-5/4o，会发 `reasoning_effort` / `verbosity` 等参数（见 `concordia/contrib/language_models/openai/gpt_model_multimodal.py`），DeepSeek 端会拒绝。自写包装只发标准 chat.completions 参数，规避不兼容。
- **模型名**：仿真用 `deepseek-v4-pro`，judge 用 `deepseek-v4-flash`。endpoint：仿真 sync 用 `https://api.deepseek.com/v1`，judge async 用 `https://api.deepseek.com`。**若 DeepSeek 账户下模型名不同，需改 `deepseek_model.py` / `judge.py` 顶部常量**。
- **预测文本残留 "Lucia " 前缀**：`pre_observe()` 只剥离 ` Lucia:` / ` Lucia --` 形式，实际 putative 事件常是 `Lucia 我...`（空格无冒号），导致 `predictions.json` 里 prediction 形如 `"Lucia 我戳了一下手机。"`。judge prompt 已声明「忽略人称差异」，影响有限；如需更干净的预测文本可在此处加正则。
- **数据路径硬编码**：`data_parser.py` 的 `DATA_BASE` 是作者本机绝对路径，且 `run_simulation.py` 未暴露数据路径 CLI 参数。换机器必须改常量或放置同路径数据（README §5 已强调）。
- **embedder**：本地 `sentence-transformers/all-MiniLM-L6-v2`（极小，CPU 即可；首次会联网下载权重）。官方 examples 用的是 `all-mpnet-base-v2`，此处特意换成更小的。
- **官方框架的本地改动（约 2 处）**：整个目录是作为普通文件提交进上层仓库的（git 里全部为新增，无上游 remote，故**无法用 `git diff` 精确定位**）。已知相对官方“正经发行版”的项目侧定制集中在：① `examples/games/run.py` 是一个统一 runner（argparse 选 game/scenario/api_type）；② `concordia/contrib/language_models/together/together_ai_model.py` 含 DeepSeek（`deepseek-ai/DeepSeek-V4-Pro`）条目。**改动是否恰好这两处无法从 git 证实**——但无论如何，本项目的全部业务逻辑都隔离在 `lucia_sim/`，对 `concordia/` 框架本体改动极小，升级框架风险低。

## 5. 常用命令

```bash
source .venv/bin/activate
export DEEPSEEK_API_KEY=...

# 跑仿真（默认 11:09-11:15 = 64 步）
PYTHONPATH=. python lucia_sim/run_simulation.py
PYTHONPATH=. python lucia_sim/run_simulation.py --time-start 11:09 --time-end 11:30 --output-dir lucia_sim/output/run2
PYTHONPATH=. python lucia_sim/run_simulation.py --max-steps 5     # 冒烟测试

# 评估
PYTHONPATH=. python lucia_sim/judge.py lucia_sim/output/morning_11_09_15/predictions.json

# 仅数据统计（不调 LLM）
PYTHONPATH=. python lucia_sim/data_parser.py
```

## 6. 与其它子项目的关系

同处 `~/Desktop/hku capstone/` 下：

- **`emotion-probe/`（探针线）**：用 RepE 等方法读 LLM 内部情绪表征，是证据链「情绪」上游。本目录是其下游「行为」环节——理想闭环是：探针证明模型内部有情绪状态 → 本仿真检验该状态是否外显为符合情境的行为。两者目前数据接口未直接打通，是并行推进的两条线。
- **Generative Agents 类工作**：本目录直接站在 Concordia（GM + Entity 生成式 agent 范式）肩上，ego-replay 是把「自由生成」约束成「对真值数据的可评测预测」的一种变体。
- **`simulation-env/`**：本目录所在的仿真大类目录；`_server_edit/` / `old_file/` 为历史/服务器编辑产物，与本仿真无直接依赖。

## 7. 技术栈

Python 3.12 · `gdm-concordia` 2.4.0（editable 源码安装）· openai SDK（2.43.0，打 DeepSeek）· sentence-transformers（5.6.0）· numpy。`.venv` 已 gitignore，需自建。
