# CLAUDE.md — predictions/（NextAct 评测，维护者笔记）

同学用的上手文档是 `README.md`。本文件只记**代码里看不出来的东西**：评测集怎么造出来的、哪些口径不能动、踩过哪些坑。

动手前先读完。历史上最大的错误来源是**凭记忆重建已有资产**——先读文件再动手。

---

## 1. 评测点已定稿，只能扩不能换

NextAct = **NextMe 1000（200/粒度 × L1–L5）+ EgoLife 500（100/粒度 × L1–L5）= 1500**。

所有已发布数字都绑定这批点。可以扩展某点的 GT 长度或补 context，**不能重新采样**。

唯一入口 `nextact_points.load_nextact()`，返回统一 schema：
`id` `source` `level` `participant` `context_raw` `context_text` `context_segments` `context_date/start/end` `gt_raw` `gt_text` `gt_segments`。
`*_text` 去掉了时间戳前缀（打分用），`*_raw` 保留（喂 prompt 用）。

### NextMe 1000 — `nextme_benchmark/benchmark.json`

- 结构 `{"L1": [...200], ..., "L5": [...200]}`，每点自带 `gt_events_k3` / `gt_events_k10` / `gt_segments_k*` / `gt_recordings_k*`。
- 天生跨 VRS：`context_recording` 与 `gt_recordings_*` 是不同录制，中间有相机关闭的间隙。
- **context 文本不在 json 里**，运行时从 `../caption-result/{rec}/hierarchy/{level}/events.txt` 读。

### EgoLife 500

- L1 的 100 点取自既有 500 点选择（`_reuse_egolife_l1`，源 `../dataset/benchmark_1k.jsonl`，seed=42 分层）。
- L2–L5 各 100 点由 `_build_egolife_level` 新建：每个参与者 7 天同粒度 events **跨天拼成一条时间轴**再采 cutoff，seed=42。
- 参与者均衡 17/17/17/17/16/16。跨天拼接后各人 L1≈105k、L2≈21k、L3≈6.6k、L4≈1.2k、L5≈250 条，L5 也够。

### 跨 VRS context 组装（`_build_context`）

单条录制在粗粒度上根本不够 50 条（L3 中位 14、L4 中位 4、L5 中位 3）。
做法：按 `M-D_hktHHMM` 解析全库 528 条录制的时间序，从 context 录制**向前跨录制回溯**补满 50 条；若 context 录制位于时间轴最开头则改为向后补，但**严格停在最早的 GT 录制之前**。

**改动 context 逻辑后必须重跑泄漏检查**：context 用到的录制序号必须全部 < 最早 GT 录制序号。当前实测 0 泄漏、context∩GT 录制集合为空。

### 凑不满的 6 个点（数据边界，不是 bug）

`L4_0001` `L4_0002` `L5_0001` `L5_0002` `L5_0003` 的 context 录制就是全库第 0 条、GT 在第 1–3 条，前无可回溯中间无录制；`L5_0108` 的 GT 录制是第 526/527 条（共 528），其后无 L5 事件、GT 仅 5 条。
按 k' fallback 惯例用实际可得长度打分，SED 按 `max(len_gt,len_pred)` 归一天然处理不等长，**不丢点**。

---

## 2. 不可变更的口径

- **永远 best-of-3**：3 条候选轨迹分别打分取 **raw 最高**，与模型给的 probability 无关。只取第一条会低约 35%。scorer 看到 sample 里有 `all_predictions` 就自动走 best-of-N。
- **归一化** `max(0, (raw - random_sed) / (1 - random_sed))`，`random_sed` 按 `{level}_k{k}` 从 `nextme_benchmark/baselines.json` 取（L1–L5 × k1/k3/k10 全齐）。
- **两个数据集共用同一份 prompt** `prompts/predict_prompt.txt`，`prompt_builder.TEMPLATE_FILE` 单一入口、不按 source 分叉。语言差异靠一条通用规则解决（"用与 context 相同的语言"），实测 EgoLife 出中文、NextMe 出英文。
- `_date_prefix()` 按 segment 日期格式自动选提示符：`MM-DD `（NextMe）/ `dayN `（EgoLife）/ 空（单日）。

---

## 3. Embedding 与缓存

- 配置 `~/.config/hku-capstone/eval_config.json`（含 key，**禁止写进仓库/记忆/日志**）。
- `scorer.embed_texts` **硬编码 `batch_size = min(cfg.batch_size, 8)`**。SiliconFlow 会大量抛 `IncompleteRead`（4096 维 × 8 条 ≈ 700KB 响应被截断），靠 `max_attempts=6` 兜住。**调大 batch 会加剧截断，要提速只加并发**（实测 96 稳定，1500 点只触发 14 次重试）。
- `emb_cache.py`：sqlite + float32，键 `model|dim|sha1(text)`。scorer 顺序为 **内存 → sqlite → API**。
- **k=1 的文本是 k=10 的严格子集（实测 100%）**，所以 `--ks 10 1` 能让 k=1 零 API 调用。

---

## 4. 踩过的坑

1. **"EgoLife k=10 不可用" / "只有 L1 能做 k=10"** — 都是假的。前者真因是 `benchmark_1k.jsonl` 只存了 3 条 GT，回源重读即可；后者是 `benchmark.json` 五个粒度本来就都带 `gt_events_k10`。**没读文件就下结论的典型。**
2. **自行重新采样评测点** — 破坏所有历史可比性。
3. **每个 k 新建 scorer** — 内存缓存随实例销毁，白烧 embedding 配额。整轮共用一个实例。
4. **DeepSeek 402 Insufficient Balance** — 余额耗尽时所有调用秒失败、`predict_point` 静默返回 None，表现为「0 ok」而非报错。排查方式是单独裸调一次看异常。
5. **表格漏行** — `print_table` 的 `rows` 白名单写漏会导致有数据却不打印。改表结构记得同步 `rows`。
6. **prompt 变了就不能复用旧预测** — 曾统计出「只有 617/1000 点 context 有变动」想省着跑，但 prompt 本身加了时间段和语言规则，旧预测全产自旧 prompt，混表即口径污染。只能整批重跑。

---

## 5. 当前结果

`results/nextact_ds-v4-flash_*`（模型，best-of-3）、`results/repeat_baseline_nextact.json`（baseline）、`results/nextact_model_vs_baseline.json`（同点对比）。分数表见 `README.md`。
覆盖率 k=1 1493/1500、k=10 1455/1500（少数点模型返回无法解析）。

`results/legacy/` 与 `archive/superseded_20260913/` 是**已废弃口径**，不要引用：旧 best-of-1 评分、纯 L1 的 EgoLife 结果、旧 prompt 的预测、被取代的 runner 与 predictor。

---

## 6. 维护约定

每次改代码/配置/prompt/数据流程或跑出新结果，立刻更新本文件与 `README.md`（如涉及同学用法）。不写 API key / token 明文，只写「已配置」及存放位置。
