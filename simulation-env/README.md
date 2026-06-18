# simulation-env — 「情绪 → 行为」行为仿真

本目录是 capstone 证据链的第 4 环(**可行为化**):把情绪/状态接到多智能体仿真里,检验"情绪驱动的 agent 行为是否接近真人"。两套独立的仿真框架,均用 **DeepSeek API** 作 LLM 后端,**不需要 GPU**。

## 两套仿真

| 子目录 | 框架 | 自有部分 | 一句话 |
|--------|------|----------|--------|
| [`concordia-official/`](concordia-official/) | Google DeepMind Concordia | `lucia_sim/` | 用 EgoLife/LUCIA 第一人称 caption 驱动:agent 接收观察 → 预测动作 → 与真实 GT 比一致率 |
| [`Generative Agents/`](Generative%20Agents/) | 斯坦福 Generative Agents | `world/` | 把真人 Lucia 的一天重建进 Smallville,让 agent 扮演并测行为一致率(TF 37.6% / FR 32.3%) |

## 怎么用

两套环境相互独立,进入各子目录按其 `README.md` 建虚拟环境与运行。共性:

- 后端 LLM 用 DeepSeek,需设 `export DEEPSEEK_API_KEY="..."`(官方生成式智能体后端另用 OpenAI key,见其 README)。
- 虚拟环境(`.venv` / `.venv_fe`)已被 `.gitignore` 排除,需自建。
- 部分脚本含硬编码数据路径,运行前需按子目录说明调整。

## 大文件说明

- `Generative Agents/` 的海量仿真回放数据(`storage/` 下 ~1GB 的已完成运行、`compressed_storage/`)已被排除;**仅保留 `base_the_ville_*` 两个启动模板**,足以开新仿真。被排除的回放 demo 可重新跑仿真生成,或从官方仓库获取。
