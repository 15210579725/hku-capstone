# Generative Agents 仿真环境（HKU Capstone · 情绪→行为 证据链）

本目录是 capstone「用机制可解释性方法探究 LLM 有没有情感」总课题里的**行为仿真子环境**。
它做两件事：

1. **官方斯坦福生成式智能体（Smallville）** —— 原仓库 `joonspk-research/generative_agents @fe05a71`，
   已打包为普通文件放在 `official_repo/`（含 reverie 后端 + Django 前端），并打了一组补丁让前端能跑在 Python 3.12 / Django 4.2。
2. **项目自定义流水线（`world/` + 根目录脚本）** —— 把 EgoLife 真实参与者 **Lucia (A4_LUCIA)** 一天的
   第一人称 caption，重建成 Smallville 风格的可仿真世界，让 LLM 扮演 Lucia 自主决策下一步动作，
   并测「模型动作 vs 真人动作」的一致率。它复用官方的 Phaser 2-D 前端做回放可视化。

在整条「情绪→行为」证据链中，本子环境负责**行为侧**：提供智能体在具体情境下的动作序列，
供与情感探针（见 `../emotion-probe/`）、社会仿真（见 `../concordia-official/`）的结果相互印证。

---

## 一、目录结构

```
Generative Agents/
├── README.md                  # 本文件（面向使用者）
├── CLAUDE.md                  # 面向后续 AI/开发者的架构与坑位说明
│
├── build_world.py             # ① 由 L5/L4/L3 caption 搭世界树+角色+人设, 切决策步
├── perception.py              # ② 给每步挂 L1 防泄露感知窗口（只取"看到/听到"）
├── gen_phaser_sim.py          # ③ 世界时间线 → 官方前端 demo 回放格式
├── rollout.py                 # ④ LLM 扮 Lucia 出动作: tf 逐点预测 / fr 自由仿真
├── judge.py                   # ⑤ LLM 语义评审一致率（一致/部分/分支/不一致）
├── build_viz.py               # 备选: 生成自包含 viz.html（双击即开, 不需 server）
│
├── world/                     # 【项目自有数据】构建好的世界与决策步（已入库, 可直接用）
│   ├── world_knowledge.json   #   世界树 + 6 角色 + Lucia 人设 + 情绪倾向
│   ├── decision_steps.json    #   181 个 L3 决策步（decision_steps_day1.json 同内容）
│   └── steps_perc_day1.json   #   每步附 L1 防泄露感知窗口（rollout/judge 直接读这个）
├── results_tf.json / _judged  # 【已跑结果】TF 逐点预测 + 评审后
├── results_fr.json / _judged  # 【已跑结果】FR 自由仿真 + 评审后
│
└── official_repo/             # 【斯坦福官方仓库】@fe05a71, 已打补丁, 见下
    ├── README.md              #   官方原始 README（权威运行说明）
    ├── requirements.txt       #   官方后端依赖（Python 3.x / Django 2.2 / openai 0.27）
    ├── reverie/
    │   ├── compress_sim_storage.py   # 把 storage 跑出的回放压成 demo
    │   └── backend_server/    #   仿真后端核心
    │       ├── reverie.py     #     主循环（fork 基线→逐步推进→存盘）
    │       ├── maze.py / path_finder.py
    │       └── persona/       #     感知-检索-计划-反思-对话 认知模块 + prompt 模板
    └── environment/frontend_server/   # Django 前端（地图渲染/回放/demo）
        ├── manage.py
        ├── storage/           #   仿真输出（见"数据策略", 大部分被 .gitignore 排除）
        ├── compressed_storage/#   压缩后的 demo（全部被排除, 可重新生成）
        ├── static_dirs/       #   前端地图/sprite 资源（约 39M, 已保留）
        └── frontend_server/settings/{base,local}.py
```

> 自定义脚本编号 ①–⑤ 即下文流水线步骤。`world/` 是项目在官方框架之外**额外构建**的产物。

---

## 二、环境与依赖

本目录涉及**两套相互独立**的运行环境，按需要安装：

### A. 前端可视化环境 `.venv_fe`（跑官方 Django 前端 + 自定义流水线脚本）

- Python **3.12**（仓库里的 `.venv_fe` 即此版本，已被 `.gitignore` 排除，需自建）
- 关键包：`Django==4.2.16`、`django-cors-headers`、`numpy`、`openai`
- 自定义脚本 `rollout.py` / `judge.py` 也在这个环境里跑（只额外用到 `openai` 库）

```bash
cd "simulation-env/Generative Agents"
python3.12 -m venv .venv_fe
.venv_fe/bin/pip install "Django==4.2.16" django-cors-headers numpy openai
```

> 官方前端原本写给 Django 2.2 / Python 3.x，本仓库已打补丁让它在 Python 3.12 + Django 4.2 下跑通
> （补丁清单见第五节）。`official_repo/environment/frontend_server/requirements.txt` 是官方原始钉版，仅供参考，不要照装。

### B. 后端仿真环境（仅当你要**真正跑官方斯坦福仿真**时才需要）

- 按官方 `official_repo/requirements.txt`（Django 2.2 / `openai==0.27.0` / numpy 1.25 等），建议单独 venv
- 后端 `reverie.py` 通过 `from utils import *` 读取 OpenAI key（见下）

### LLM API 密钥（**只从环境变量读，仓库内无明文**）

| 用途 | 读取方式 | 模型 |
|------|----------|------|
| 自定义流水线 `rollout.py` / `judge.py` | 环境变量 `DEEPSEEK_API_KEY` | `deepseek-v4-pro`（base_url `https://api.deepseek.com`）|
| 官方 `reverie` 后端 | `reverie/backend_server/utils.py` 里的 `openai_api_key`（该文件被 .gitignore 排除，需自建）| `gpt-3.5-turbo` / GPT-4 |

```bash
# 自定义流水线（推荐用环境变量，绝不写进代码）
export DEEPSEEK_API_KEY="sk-..."

# 官方后端：在 reverie/backend_server/ 下新建 utils.py（参考 official_repo/README.md 第 1 步）
#   openai_api_key = "sk-..."        # 你的 OpenAI key
#   key_owner      = "Your Name"
#   maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
#   ... (其余路径常量见官方 README 模板)
```

---

## 三、如何运行

### 路径 1：项目自定义流水线（Lucia 一致率实验）

```bash
cd "simulation-env/Generative Agents"
export DEEPSEEK_API_KEY="sk-..."

# ① + ② 构建世界与感知窗口（world/ 已入库, 若只想复跑可跳过这两步）
python3 build_world.py          # → world/world_knowledge.json, world/decision_steps*.json
python3 perception.py day1      # → world/steps_perc_day1.json

# ④ LLM 扮 Lucia 出动作
python3 rollout.py tf           # teacher-forcing 逐点预测（并发, 干净测一致率）→ results_tf.json
python3 rollout.py fr           # free rollout 自由仿真（顺序滚动, 允许走分支）→ results_fr.json

# ⑤ 一致率评审
python3 judge.py results_tf.json   # → results_tf_judged.json + 终端汇总
python3 judge.py results_fr.json

# ③ 生成官方前端可吃的 demo 回放（写入 compressed_storage/lucia_day1_morning/）
python3 gen_phaser_sim.py
```

> ⚠️ **`build_world.py` / `perception.py` 依赖外部 EgoLife caption 数据**：硬编码路径
> `/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA/L{1-5}/day1`。
> 该数据**不在本仓库内**。clone 后若没有这份数据，请直接用已入库的 `world/*.json`
> 跳到第 ④/⑤ 步（rollout/judge 只读 `world/steps_perc_day1.json`，不碰外部路径）。

### 路径 2：官方 2-D 前端可视化（回放 Lucia demo）

```bash
.venv_fe/bin/python official_repo/environment/frontend_server/manage.py runserver 8000
# 浏览器打开:
#   首页自检:   http://localhost:8000/
#   Lucia demo: http://localhost:8000/demo/lucia_day1_morning/0/2/
```

角色映射：Lucia=Isabella(IR)、Jake=Klaus、Shure=Ryan Park(RP)、Katrina=Maria Lopez(ML)、
Alice=Hailey(HJ)、Tasha=Abigail(AC)。Lucia 头顶 emoji=当前动作；气泡=真实 GT + 🤖AI 预测 + 一致标签；相机跟随 Lucia。

> 无需起 server 的轻量版：`python3 build_viz.py` 生成自包含 `viz.html`，双击即可看俯视轨迹+指标。

### 路径 3：跑一场全新的官方斯坦福仿真（需后端环境 + OpenAI key）

参见 `official_repo/README.md`。简述：起前端 server → 在 `reverie/backend_server/` 运行 `python reverie.py`
→ fork 基线（如 `base_the_ville_isabella_maria_klaus`）→ 输入新仿真名 → `run <步数>` → `fin` 存盘
→ `compress_sim_storage.py` 压成 demo → `/demo/...` 回放。

---

## 四、被 .gitignore 排除的数据 —— 如何获取

为控制仓库体积，以下大文件**不入库**（规则见仓库根 `.gitignore` 与 `official_repo/.gitignore`）：

| 被排除内容 | 体量 | 性质 | 如何获取 |
|-----------|------|------|---------|
| `storage/July1_the_ville_isabella_maria_klaus-step-3-*`（**21 个运行目录**）| 约 1GB | 官方已完成仿真的**回放 demo**（每步一个 JSON）| 不必要；可从官方仓库重新下载，或自己跑一场新仿真（路径 3）生成 |
| `compressed_storage/*`（含 `lucia_day1_morning`、`July1_*`）| 约 23M | 压缩后的回放 demo | `lucia_day1_morning` 由 `python3 gen_phaser_sim.py` 重新生成；官方 demo 用 `compress_sim_storage.py` 压出 |
| `.venv_fe/` | — | 前端虚拟环境 | 按第二节 A 自建 |

**已保留、可直接开新仿真的启动模板**（`.gitignore` 用 `!` 白名单显式保留）：

- `storage/base_the_ville_isabella_maria_klaus/` —— 3 智能体基线（Isabella / Maria / Klaus）
- `storage/base_the_ville_n25/` —— 25 智能体基线

→ 这两个 base 模板是跑任何新官方仿真的起点（`reverie.py` 提示 fork 时填它们的名字）。
另：`static_dirs/`（前端地图/sprite 资源，约 39M）**已保留**，前端可视化开箱可用。

---

## 五、官方前端跑在 Python 3.12 / Django 4.2 的补丁（共约 16 处）

为在新版 Python/Django 跑通官方前端，对 `official_repo/` 做了一组本地改动，主要类别：

- venv 装 `Django==4.2.16` + `django-cors-headers` + `numpy`（替代官方钉的 Django 2.2）
- `frontend_server/urls.py`：`from django.conf.urls import url` → `from django.urls import re_path as url`
- `translator/views.py`：`from django.contrib.staticfiles.templatetags.staticfiles import static` → `from django.templatetags.static import static`
- 所有模板的 `{% load staticfiles %}` → `{% load static %}`（home/demo/base/landing/persona_state/path_tester 等）
- `frontend_server/settings/{base,local}.py`：移除 `'storages'` app
- `templates/base.html`：jQuery 必须在 bootstrap **之前**加载
- `templates/demo/main_script.html`：`all_movement[step]` 加 undefined 守卫（回放循环不崩）+ 相机改为跟随 Lucia
- `translator/views.py` 默认 demo 路由指向自定义/示例仿真

---

## 六、结果（day1 上午, L3, 181 决策步, deepseek-v4-pro）

| 指标 | TF 逐点预测 | FR 自由仿真 |
|------|------------|------------|
| 一致率（一致 + 0.5×部分）| **37.6%** | 32.3% |
| 严格一致率（仅一致）| 24.3% | 22.1% |
| 合理分支率 | 12.2% | 9.9% |

- 高一致场景：拼图 0.53 / 白板讨论 0.51 / 点外卖 0.50
- 低一致场景：精细微动作（组装硬盘 0.17、整理烘焙 0.18）
- 主要偏差信号：**模型比真人 Lucia 更主动、更爱说话**（真人多是安静配合 / 等待 / 小动作）

---

## 七、数据来源

- **官方代码与地图资源**：`joonspk-research/generative_agents @fe05a71`（Apache-2.0，见 `official_repo/LICENSE`）
- **Lucia 真人 caption 数据**：EgoLife 数据集参与者 A4_LUCIA，本地路径
  `/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA/L{1-5}/day1`（外部，不入库）
- 当前覆盖范围：**day1 上午 11:09–14:17**（出门采购前），181 个 L3 决策步
