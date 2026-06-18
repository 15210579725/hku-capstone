# CLAUDE.md — simulation-env 容器导览

两套多智能体行为仿真,对应证据链第 4 环(可行为化)。逐套细节见各子目录 `CLAUDE.md`。

## 结构与定位

- `concordia-official/` — Google DeepMind Concordia 官方框架 + 自有 `lucia_sim/`。采用 **ego-replay 范式**:GM 完全数据驱动(不让 LLM 裁决环境),唯一的 LLM 调用是 agent 生成动作预测,再用 GT 推进、异步打分。
- `Generative Agents/` — 斯坦福生成式智能体 + 自有 `world/`。把 EgoLife 真人 Lucia 的 day1 重建成 Smallville,让 agent 扮演并测一致率,复用官方 Phaser 前端做回放 demo。

## 关键共性与坑

1. **后端 = DeepSeek API,本环节不需要 GPU**(与探针线相反)。`DEEPSEEK_API_KEY` 走环境变量;生成式智能体的官方后端另读自建 `utils.py`(被 `.gitignore` 排除),仓库内无明文 key。
2. **硬编码数据路径**:两套都把 EgoLife/LUCIA 数据路径写死成原作者本机路径,且未提供命令行参数覆盖。新机器上跑必须放置同路径数据或改常量。
3. **官方框架已去 `.git`**:作为普通文件纳入主仓,业务逻辑隔离在 `lucia_sim/` 与 `world/`,对官方本体改动很小。上游 URL/commit 见各子 README。
4. **数据策略**:生成式智能体的 `storage/` 海量回放(~1GB)与 `compressed_storage` 已排除,仅留 `base_the_ville_*` 启动模板;`static_dirs`(39M 前端资源)保留以便前端可渲染。

## 与其它子项目关系

- 仿真线**独立于探针线**即可运行(不消费 `*.pt` 方向向量),共享的是同一套 EgoLife/LUCIA 日记数据与"情绪/状态 → 行为"的研究问题。
