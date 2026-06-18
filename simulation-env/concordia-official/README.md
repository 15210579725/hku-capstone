# LUCIA 行为预测仿真环境（基于 Concordia）

> 本目录 = **Google DeepMind Concordia 官方框架**（@873dddd 原样打包）+ 项目自有的 **`lucia_sim/`** 仿真模块。
> 下文「安装 / 运行」均针对 `lucia_sim/`；官方框架仅作底座，无需单独跑通。
> 官方框架的完整文档见同目录 `CHEATSHEET.md` 与上游仓库 `google-deepmind/concordia`。

## 1. 这是什么 / 在总课题中的位置

HKU capstone 的总主题是用机制可解释性方法探究「LLM 有没有情感」，整条证据链为：

```
情绪刺激 → 探针读取内部表征(emotion-probe) → 情绪状态 → 行为表现(本目录)
```

本目录是「**情绪 → 行为**」证据链中的**行为仿真环节**。做法是 **ego-replay（第一人称回放）**：

- 取 EgoLife 真人实验中 LUCIA（一位 mentor）的 **L1 第一人称 caption** 数据；
- 把数据切成一连串「决策点」：每个决策点 = 此前累积的观察（observation / internal）+ 一个真实动作（action，作为 ground truth）；
- 在每个决策点，让 LLM 扮演的 LUCIA agent 接收观察，**预测她下一步会做什么**；
- 预测完后，环境**不采用**预测，而是按真实 GT 动作继续推进；
- 最后用 **LLM Judge** 给「预测 vs 真实动作」打分，统计一致率。

LLM 后端用 **DeepSeek**（远程 API）。本环节本身**不需要 GPU**：唯一的本地模型是极小的句向量 embedder（all-MiniLM-L6-v2），CPU 即可；真正吃 GPU 的是探针线 `emotion-probe`，不在本目录。

## 2. 目录结构

```
concordia-official/
├── concordia/          # ★ 官方 Concordia 框架（@873dddd 原样打包），底座
├── examples/           # ★ 官方示例（games / notebooks 等）
├── bin/                # ★ 官方安装/测试脚本（install.sh、test.sh ...）
├── setup.py            # ★ 官方打包配置（包名 gdm-concordia 2.4.0）
├── pyproject.toml      # ★ 官方构建/lint 配置
├── requirements.txt    # ★ 官方 pip-compile 锁文件（含 hash，体量大）
│
├── lucia_sim/          # ☆ 本项目自有仿真模块（重点）
│   ├── run_simulation.py   # 仿真主入口
│   ├── data_parser.py      # L1 caption 解析 + 构建决策点
│   ├── ego_replay_gm.py    # 自定义 Game Master（数据回放）
│   ├── deepseek_model.py   # DeepSeek API 的 LanguageModel 适配
│   ├── judge.py            # LLM Judge 评估（4 维度打分）
│   └── output/             # 已有的几次运行结果（morning_11_09_15 / test3 ...）
│
├── README.md           # 本文件
└── CLAUDE.md           # 面向后续 AI / 开发者的架构说明
```

★ = 官方框架，原样保留（约 2 处本地改动，见 `CLAUDE.md`）；☆ = 项目自有代码。

## 3. 环境与依赖

- **Python 3.12**（见 `.python-version`；`setup.py` 要求 `>=3.12`）。
- `.venv/` 已被 `.gitignore` 排除，**clone 后需自建虚拟环境**。

```bash
cd "simulation-env/concordia-official"

# 1) 建并激活虚拟环境（Python 3.12）
python3.12 -m venv .venv
source .venv/bin/activate

# 2) 安装官方框架本体（editable，装入核心依赖：absl-py / numpy / pandas 等）
pip install -e .

# 3) 安装 lucia_sim 额外依赖（官方 setup 不含这两项）
pip install openai sentence-transformers
```

> 已验证可用版本：`openai==2.43.0`、`sentence-transformers==5.6.0`。
>
> 也可严格复刻官方锁定环境：`bash bin/install.sh`（按 `requirements.txt` 全 hash 安装，会拉入 torch / transformers 等，体量很大，对 lucia_sim 非必需）。

## 4. 环境变量（API 密钥）

DeepSeek 密钥**统一从环境变量读取，代码中不写明文**：

```bash
export DEEPSEEK_API_KEY="你的_deepseek_key"
```

- 仿真（`run_simulation.py` → `deepseek_model.py`）和评估（`judge.py`）都从 `DEEPSEEK_API_KEY` 取值。
- 未设置时，创建 `DeepSeekLanguageModel` 会抛 `DEEPSEEK_API_KEY not set`，judge 调用也会失败。
- 建议把上面的 `export` 写进 `~/.zshrc`，或每次运行前手动 export。

## 5. 数据来源与路径（重要）

数据为 **EgoLife / LUCIA 的 L1 第一人称事件 caption**。`data_parser.py` 里**硬编码**了绝对路径：

```
/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA/L1/day1/events.txt
```

行格式：`[HH:MM:SS -> HH:MM:SS] 中文第一人称描述`，全天约 1.5 万行。解析时按关键词把每行分类为 observation / internal / action（见 `data_parser.py`）。

> ⚠️ **可移植性提示**：该路径是作者本机绝对路径，且 `run_simulation.py` 当前**没有提供数据路径参数**。其他人 clone 后必须二选一：
> 1. 把 `events.txt` 放到上面这个完全相同的路径；或
> 2. 修改 `lucia_sim/data_parser.py` 顶部的 `DATA_BASE` / `LUCIA_L1_DAY1` 常量指向本地数据。

## 6. 如何运行仿真

确保已激活 `.venv` 且已 `export DEEPSEEK_API_KEY`，在 `concordia-official/` 目录下运行（需 `PYTHONPATH=.` 让 `lucia_sim` 可导入）：

```bash
# 默认窗口 11:09–11:15（该窗口 → 64 个决策点），输出到 ./output
PYTHONPATH=. python lucia_sim/run_simulation.py

# 指定时间窗口 + 限制步数 + 指定输出目录
PYTHONPATH=. python lucia_sim/run_simulation.py \
    --time-start 11:09 --time-end 11:15 \
    --max-steps 5 \
    --output-dir lucia_sim/output/my_run

# 仅看数据解析统计（不调用 LLM，可离线验证数据可读）
PYTHONPATH=. python lucia_sim/data_parser.py
```

参数（`run_simulation.py`）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--time-start` | `11:09` | L1 窗口起点 `HH:MM`（含） |
| `--time-end`   | `11:15` | L1 窗口终点 `HH:MM`（含） |
| `--max-steps`  | 全部决策点 | 最大仿真步数（每步 = 1 个决策点） |
| `--output-dir` | `./output` | 输出目录（相对当前 cwd） |

每次运行产出（写入 `--output-dir`）：

- `simulation_log.html` — Concordia 官方结构化 HTML 日志
- `predictions.json` — 每步 `{step, timestamp, prediction, gt_action}`
- `interaction_log.txt` — 人类可读的逐步对照日志

## 7. 评估（LLM Judge）

对 `predictions.json` 逐条用 DeepSeek 打分（4 维度：semantic / plausibility / temporal / granularity，异步并发 10）：

```bash
PYTHONPATH=. python lucia_sim/judge.py lucia_sim/output/morning_11_09_15/predictions.json
```

结果写到同目录 `*_judged.json`，并在终端打印各维度均值与总体百分比。

示例（`lucia_sim/output/morning_11_09_15/`，64 条预测的真实结果）：

```
semantic     : 0.77 / 4
plausibility : 1.22 / 2
temporal     : 1.72 / 2
granularity  : 1.50 / 2
```

（语义一致率偏低属预期：原子级行为预测本就很难，本环节关注的是「行为分布/趋势能否复现」，而非逐条命中。）

## 8. 官方框架说明（简述）

`concordia/` 是 Google DeepMind 的生成式社会仿真库（包名 `gdm-concordia` 2.4.0）。核心范式：一个 **Game Master (GM)** 模拟环境、若干 **Entity** 用自然语言描述意图，GM 把意图解析为结果，由 **Engine**（默认 sequential）驱动回合循环。`lucia_sim` 正是复用这套 `Config → Simulation → GM + Entity → Engine` 组件栈，只把 GM 换成「数据回放」逻辑。官方游戏示例见 `examples/games/`（`python -m concordia.examples.games.run --game=pub_coordination`），深入文档见 `CHEATSHEET.md`。
