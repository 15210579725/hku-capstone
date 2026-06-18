# CLAUDE.md —— Generative Agents 仿真环境（开发者/AI 指南）

面向后续接手的 AI 与开发者。先读本目录 `README.md`（使用者视角），本文件补充**架构、关键决策、坑位与数据流**。

---

## 1. 这是什么 / 在大课题里的位置

- 总课题：**用机制可解释性方法探究 LLM 有没有情感**，核心是「情绪 → 行为」证据链。
- 本子环境 = **行为侧**：用斯坦福生成式智能体（Smallville）框架做「情境 → 动作」仿真。
- 两块拼起来：
  - `official_repo/`：原封斯坦福仓库 `joonspk-research/generative_agents @fe05a71`（已打补丁），保留**完整官方能力**（reverie 后端真跑多智能体仿真 + Django/Phaser 前端可视化）。
  - `world/` + 根目录 5 个脚本：**项目自有**，把 EgoLife 真人 Lucia 的一天重建成可仿真世界，让 LLM 扮演她并测一致率，复用官方前端做回放。

---

## 2. 架构

### 2.1 官方部分 `official_repo/`

```
reverie/backend_server/        仿真后端（不依赖前端即可推进逻辑）
  reverie.py                   主循环: fork 基线 → 逐 step 推进 → 与前端交换 movement → 存盘(fin)
  maze.py / path_finder.py     Smallville 网格地图 + A* 寻路
  persona/                     单个智能体的认知架构（论文核心）
    cognitive_modules/         perceive → retrieve → plan → reflect → execute → converse
    memory_structures/         associative(记忆流) / spatial(空间记忆) / scratch(短期)
    prompt_template/           各认知步的 GPT prompt（gpt_structure.py 封装 OpenAI 调用）
environment/frontend_server/   Django 前端
  manage.py                    起 server: /(自检) /simulator_home /replay/<sim> /demo/<sim>/<step>/<speed>
  translator/views.py          各路由视图（demo() 读 compressed_storage 回放）
  storage/                     后端跑出的仿真原始输出（fork 自 base, 每 step 一组 JSON）
  compressed_storage/          compress_sim_storage.py 压缩后的 demo（带 sprite, 给 /demo 用）
  static_dirs/assets/          地图 tileset + 角色 sprite（约 39M, 已保留）
```

后端与前端**通过文件系统的 storage 目录通信**（reverie 写 movement，前端轮询读），不是 HTTP RPC。

### 2.2 自定义部分（根目录 + `world/`）

五步流水线，编号与脚本对应：

| 步 | 脚本 | 职责 | 读 | 写 |
|----|------|------|----|----|
| ① | `build_world.py` | L5/L4 取场景骨架, L3 切 181 决策步, 内置世界树/角色/人设 | 外部 EgoLife `A4_LUCIA/L{3,4,5}` | `world/world_knowledge.json`, `world/decision_steps*.json` |
| ② | `perception.py` | 给每步挂 L1 防泄露感知窗口 | 外部 `A4_LUCIA/L1` + `world/decision_steps_day1.json` | `world/steps_perc_day1.json` |
| ③ | `gen_phaser_sim.py` | 决策步 → 官方 demo 回放格式（6 人映射 sprite）| `results_tf_judged.json`, `world/steps_perc_day1.json` | `compressed_storage/lucia_day1_morning/{master_movement,meta}.json` |
| ④ | `rollout.py {tf,fr}` | LLM 扮 Lucia 出下一步动作 | `world/{world_knowledge,steps_perc_day1}.json` | `results_tf.json` / `results_fr.json` |
| ⑤ | `judge.py <results>` | LLM 评审一致率（一致 1.0 / 部分 0.5 / 分支 0.3 / 不一致 0.0）| 上一步结果 + `steps_perc_day1.json` | `*_judged.json` + 终端汇总 |
| — | `build_viz.py` | 备选: 内嵌数据生成自包含 `viz.html`（无需 server）| world + results | `viz.html` |

`tf`（teacher-forcing）= 每步用真实历史回填、并发独立预测 → 干净测一致率；
`fr`（free rollout）= 用模型自己输出顺序滚动 → 模拟 Lucia 的上午、允许走分支。

---

## 3. 数据流

```
EgoLife A4_LUCIA caption (外部, L1–L5)
        │  build_world.py(③L5/L4骨架+L3切步)  perception.py(L1感知窗口)
        ▼
world/{world_knowledge, decision_steps, steps_perc_day1}.json   ← 已入库, 可作复跑起点
        │  rollout.py tf/fr  (DeepSeek 扮 Lucia)
        ▼
results_tf.json / results_fr.json
        │  judge.py  (DeepSeek 评审)
        ▼
results_*_judged.json  ──┬─ build_viz.py ─→ viz.html (独立)
                         └─ gen_phaser_sim.py ─→ official_repo/.../compressed_storage/lucia_day1_morning/
                                                        │  manage.py runserver
                                                        ▼
                                            /demo/lucia_day1_morning/0/2/  (Phaser 2-D 回放)
```

官方独立路径（与上面无关）：`reverie.py` fork `base_*` → 写 `storage/<sim>` → 前端 `/replay` 或压缩后 `/demo`。

---

## 4. 关键决策与坑

### 4.1 LLM key：两套机制别混
- **自定义脚本**（rollout/judge）：`os.environ["DEEPSEEK_API_KEY"]` + `base_url=https://api.deepseek.com`，模型 `deepseek-v4-pro`，用 `openai` 库的 `OpenAI()` 新接口。
- **官方后端**：`gpt_structure.py` 里 `from utils import *; openai.api_key = openai_api_key`，用 `openai==0.27` 旧接口。`utils.py` 被 `.gitignore` 排除，clone 后需照 `official_repo/README.md` 第 1 步自建。
- 仓库内**无任何明文 key**，务必保持。

### 4.2 防泄露（自定义流水线的核心正确性）
- 决策步要预测的是 Lucia 的 `gt_action`，所以**绝不能**把 `gt_action` 放进 context。
- `perception.py` 只取 `我看到/我听到`（环境），剔除 `我做/我说`（那是答案）。
- `build_world.py:fix_self()` 修一个数据坑：佩戴者自述常被 diarize 错标成「我听到 Lucia 说 X」，要还原成「我说 X」，否则会把自己的话当成别人说的环境信息。

### 4.3 官方前端的 Python 3.12 / Django 4.2 补丁（约 16 处，已应用）
官方钉死 Django 2.2 / Python 3.x。为在 3.12 跑通做了一组改动，**升级 Django 时若回退会全部复发**：
- `urls.py`: `conf.urls.url` → `urls.re_path as url`
- `translator/views.py`: `staticfiles.templatetags.staticfiles.static` → `templatetags.static.static`
- 各模板 `{% load staticfiles %}` → `{% load static %}`
- `settings/{base,local}.py`: 去掉 `'storages'` app
- `base.html`: jQuery 须在 bootstrap 之前
- `demo/main_script.html`: `all_movement[step]` 加 undefined 守卫（回放循环越界不崩）+ 相机跟随 Lucia
- 默认 demo 路由调整
> `official_repo/{requirements.txt, environment/frontend_server/requirements.txt}` 是官方原始钉版，**不要照装**；前端实际用 `.venv_fe`（Django 4.2.16）。

### 4.4 storage 数据策略（务必理解，否则会误删/误传）
- `.gitignore` 规则（仓库根 + `official_repo/.gitignore` 双重）：
  - `storage/*` 排除，但白名单 `!base_the_ville_isabella_maria_klaus`、`!base_the_ville_n25` 保留 → **base 模板入库**，可直接 fork 开新仿真。
  - 21 个 `July1_the_ville_*-step-3-*` 运行目录（约 1GB）排除 → 是官方已完成仿真的回放 demo，可重抓或自跑。
  - `compressed_storage/` 整目录排除 → 含 `lucia_day1_morning`（用 `gen_phaser_sim.py` 重生）和 July1 demo。
  - `static_dirs/`（39M 地图资源）**保留**（注意：`official_repo/.gitignore` 写了 `static_dirs/*`，但仓库根 .gitignore 未排除该路径，实际以入库内容为准——前端资源已在仓库里）。
- 自定义脚本产物 `gen_phaser_sim.py` 写进被排除的 `compressed_storage/`，所以**重新跑前端 demo 前必须先跑一遍 `gen_phaser_sim.py`**。

### 4.5 外部数据依赖
- `build_world.py` / `perception.py` 顶部硬编码 `DATA = "/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA"`，换机器要改。
- 但 `world/*.json` 已入库，**复跑 rollout/judge/viz 无需外部数据**。改写覆盖范围（如新增 day2）才需要原始 caption。

### 4.6 角色 ↔ sprite 固定映射（gen_phaser_sim.py:CAST）
Lucia→Isabella Rodriguez, Jake→Klaus Mueller, Shure→Ryan Park, Katrina→Maria Lopez,
Alice→Hailey Johnson, Tasha→Abigail Chen。chat 气泡必须传 `[[说话人,内容],...]` 列表，传整串会被前端逐字符 garble。

---

## 5. 与其它子项目的关系

- `../emotion-probe/`：情感探针（RepE-2023-Zou、Gemma 探针等），从模型内部激活读「情绪」信号 —— 证据链**情绪侧**。
- `../concordia-official/`：另一套（DeepMind Concordia）社会智能体仿真 —— 平行的行为仿真对照。
- 本目录提供**「情境→动作」行为序列**，与 emotion-probe 的情绪信号对齐，论证「情绪是否驱动/解释行为」。
- 当前仅 day1 上午、单受试者 Lucia、单模型 deepseek-v4-pro，属可扩展的概念验证（PoC）。

---

## 6. 给后续 AI 的提醒

- 改任何官方文件前，先确认是否会破坏 4.3 的补丁；升级 Django 要重打补丁。
- 不要把 `storage/July1_*`、`compressed_storage/`、`.venv_fe/` 加回 git。
- 不要在代码里写 key；新增脚本沿用 `os.environ` 读取。
- 跑前端 demo 前先 `gen_phaser_sim.py`；它依赖 `results_tf_judged.json`，所以顺序是 rollout→judge→gen_phaser_sim→runserver。
