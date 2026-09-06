# CLAUDE.md — Dense Video Captioning Pipeline

对第一人称（egocentric）POV 眼镜视频做 dense video captioning，产出结构化 JSON caption，
并渲染审核视频。**生产版 = 单遍版（`--mode 1pass`，默认；整帧和 zoom crop 均 MEDIUM）。**
两遍 v9（`--mode 2pass`）保留为 backup。

---

## 目录布局

```
caption/
  run_caption_2pass.py     ← 唯一生产入口，默认两遍版
  prompt.txt               ← caption system prompt（v9，未改动）
  render_caption_video.py  ← 审核视频渲染
  CLAUDE.md                ← 本文件

  单遍版测试_待审核/        ← 单遍架构评估与对比产出，现已获批为默认版
  过程debug/               ← 推导出配置的实验脚本与数据，见其 README.md
  gemini_test_results/     ← caption 结果 JSON
  caption_crops/           ← pass2 裁出的屏幕区域图
  caption_review_videos/   ← 渲染好的审核视频
  mimo_v25_audit_30x30s_20260821_COMPLETE/scenes/  ← 30 个 30s 场景数据

  quality_verify/ pipelines/ transcript/ ARIN7600/ 等  ← 独立子项目，各有自己的 CLAUDE.md
```

---

## 快速开始

```bash
# 全部文字密集场景（生产默认，两遍版）
python run_caption_2pass.py --workers 3 --rounds 5 --tag v9

# 单场景
python run_caption_2pass.py S14_self_order_kiosk

# 断点续跑：只补失败的场景，合并进已有结果
python run_caption_2pass.py --merge gemini_test_results/OLD.json --scenes S05_screen_text,S14_self_order_kiosk

# 消融：不给 transcript / OCR
python run_caption_2pass.py --no-transcript --no-ocr --tag ablation

# 渲染审核视频
python render_caption_video.py --env gemini_test_results/LATEST.json --suffix _env   # 左动作 右环境
python render_caption_video.py --batch                                              # 左caption 右屏幕文字
python render_caption_video.py --compare A.json B.json --suffix _ablation            # 左右对比两个结果
python render_caption_video.py --compare A.json B.json --compare-field environment   # 只比环境描述
```

---

## 锁定配置（两遍版）

| 项 | 值 | 依据 |
|---|---|---|
| 模型 | `gemini-3.7-flash` | — |
| 认证 | ADC Vertex AI，project `project-0e21c343-2d89-405c-9d5`，location `global` | 免费 $300 额度 |
| Pass1 帧 | 2880px q80，默认 media_resolution | 降到 1440 让 pass1 效果明显变差 |
| Pass2 crop | 2880px q60，**ULTRA_HIGH** media_resolution | 分辨率矩阵实测：q85 payload 翻倍无收益；1440 会丢面包屑/标签栏/行内注释 |
| 帧源 | `gaze_overlay/`（绿色注视圈+黄色轨迹），fallback `masked/` | — |
| 语音 | silero-vad 定时间 + Gemini 定文字，能量 VAD 兜底 | Gemini 自身时间戳漂移 1.8-3.8s，VAD 锚定后降到 0.03s |
| 传输 | inline ≤20MB，失败 3 次回退 GCS | GCS 不省 token、不加速，只作兜底 |
| 并发 | 3 workers × 8 GCS 上传线程 | 更高会打满网络导致 SSL EOF；也会从共享配额拿到 429 |
| 重试 | `--rounds 5`，每轮递减并发 | 429 用 20/40/60s 长退避，掉线用 4/8/12s |
| 视频编码 | `-g 1 -tune stillimage` | 1fps 幻灯片内容在帧间预测下会出块状伪影 |

---

## 三遍架构

**Pass 0 — 语音对齐**
silero-vad 找出语音窗口（时间精度 ~0.03s），Gemini 只负责填词并判断是否人声。
silero 无结果但 transcript 有实质内容时退回能量 VAD；只有「嗯。」这类填充词的直接丢弃
（那是 Whisper 在噪音上的幻觉，16 个场景里有 11 个）。

**Pass 1 — caption + 区域框**
完整 `prompt.txt` + REGION_ADDENDUM，输出 caption 和每个文字密集区域的 `box_2d` 归一化坐标。

**Pass 2 — 精读 + 修订**
按坐标从 2880px 原图 crop（padding 3%），ULTRA_HIGH 精读，然后**用读到的内容改写 pass1 的
segment**——纠正 action、补全 environment、更新 text_visible，输出 `revised_segments` 并带
`revision_note`。

**Merge** — pass2 修订版覆盖 pass1，crop 读数作为 `screen_text_detail` 挂在对应 segment。

输出 JSON 每条记录同时保留 `pass1.content` / `pass2.content` / `merged` 三份完整内容。

---

## prompt 要点（v9，未改动）

**Rule 4 环境字段 = world model 训练数据**
- **首段**：完整建立性描述，覆盖物理空间 / 所有物体（颜色·材质·状态·位置）/ 每块屏幕完整内容 /
  世界文字 / 播放媒体 / 环境音。这是 agent 加载的初始世界状态，要求详尽。
- **后续每段**：只写相对上一段的变化，`"No change."` 是合法值。
- **忽略头部晃动造成的伪变化**：物体因转头进出画面不算变化；只报屏幕内容切换、物体被移动/开关、
  人员进出、光照/位置/环境音变化。

实测效果：首段 1073 字符，后续每段 153 字符，部分段落是 "No change."。

其余规则：第一人称、动作粒度合并、菜单/弹窗必须枚举具体选项、语音必须原文引用并纠正同音字、
OCR 可能不准需自行判断、绝对 HKT 时间戳、隐私脱敏（含密码/门禁码/token 等凭据）、注视点优先。

---

## 已知限制

**2880 q80 的可靠性代价**
30 帧 payload 涨到 33-44MB，超过 20MB inline 上限，每个场景都必须走 GCS，
每次调用约 15-20% 随机掉线。一个场景要 4 次调用全成功才完整，需多轮重试才到 16/16。

**其他**
- transcript 时间戳是 30s 块级，可能与 clip 时间不对齐（如 S13 差 8 分钟）
- 部分场景音频近乎静音（S02），loudnorm `-inf` 已兜底
- macOS ffmpeg 无 libass，字幕面板用 Pillow 逐帧渲染
- 全 I 帧编码让视频从 ~25MB 涨到 ~80MB
- `/Users/mac` 这类 macOS 默认账户路径偶有残留（16 场景中 3 处）。"mac" 非真实姓名，暂未处理
- OCR 一致率不能单独当质量指标：昏暗小字屏幕上模型常比本地 OCR 读得更准更多

---

## 已修 bug（2026-09-03）

**时间戳后缀只匹配 `_HKT`**：内地录制的 clip 文件名是 `_BJ`，导致 S12/S13 全部帧时间戳为空，
模型拿到的是无标注帧。已改为 `_[A-Z]{2,4}`。修复后这两个场景需重跑才生效。

---

## 消融结论：transcript 留、OCR 可去

两轮全量对比（同 prompt 同配置，只切输入），取两边都完整的 6 个场景：

| 指标 | 有 transcript+OCR | 纯视觉 |
|---|---|---|
| **语音段** | **11** | **0** |
| crop 读数 | 339 | **473** |
| text_visible | 220 | **250** |

没有 transcript 完全捕捉不到说了什么，不可替代；但文字识别纯视觉反而更强，
说明**两遍 crop 已完全取代 OCR**。

---

## 单遍版（`--mode 1pass`）：当前默认

2026-09-03 做过一次单遍架构评估：本地注视点锚定的区域检测替代 pass1 框选，
整帧 LOW + 每个可读帧一张 MEDIUM zoom，一次调用出结果。
实测省 41% 成本、16/16 首轮可靠、14/16 场景文字提取更多、真值准确率相同，
但首段环境描述短约 13%。当前配置为整帧 MEDIUM、zoom crop MEDIUM。

HF 生产流水线使用 `CAPTION_VERSION=v10` 的单遍 worker；v9 两遍版仍可通过
`--caption-version v9` 显式运行。

## 2026-09-05 — 单遍获批并接入 HF

- 本地 `run_caption_2pass.py` 默认改为 `--mode 1pass`；整帧 `ONE_FRAME_RES=medium`，
  zoom crop `ONE_CROP_RES=medium`。两遍 v9 保留为 `--mode 2pass` backup。
- 新增 `pipelines/caption_onepass.py`，HF `cloud_worker.py` 以 `CAPTION_VERSION=v10`
  执行一次 Gemini 调用，整帧和注视点 crop 均逐 part `MEDIUM`；OCR 缺失布局最多保底
  4 个注视点 crop，避免 tar 布局 B 没有 zoom。
- 冒烟证据：HF job `6a9bbabbe686246ca69a355f` 用 `project-1a1608e8-53ad-4178-bf7`
  + `gemini-3.5-flash` 成功，`pipeline=v10-1pass`、`media_resolution=MEDIUM`、1 segment；
  HF job `6a9bbbc8e686246ca69a3588` 再验证 zoom，`ok=true`、`n_zooms=1`。
- 已按两个 ADC profile 并行提交剩余窗口：`20260905-v10-adc1` 使用
  `project-1a1608e8-53ad-4178-bf7`，`20260905-v10-adc2` 使用
  `project-0e21c343-2d89-405c-9d5`，均为 `gemini-3.5-flash`、`ENTRY=cloud_worker.py`。
  截至提交核验，两组 job 均在 HF `RUNNING/SCHEDULING`；未完成前不要宣称生产全量交付。

**当前不是默认**，全部产出（32 个左右对比视频、结果 JSON、完整实验记录）在
`单遍版测试_待审核/`，等审核后再决定。代码保留在主脚本里是因为它复用同一套
VAD / 传输 / 重试 / 场景加载，不跑就不会触发。

---

## 迭代历史

- v1-v2：10 帧采样、第三人称、粗粒度
- v3：全 30 帧、第一人称、绝对 HKT、1024px
- v4：隐私脱敏、语音引用强化、2.8K 视频
- v5：gaze_overlay 帧、选项枚举、媒体环境描述
- v6：ADC 默认、1440px、OCR/语音判断、动作合并、视频编码修复（`run_caption.py`，已归档）
- v7：两遍 crop 架构、ULTRA_HIGH 精读
- v8：pass1 2880 q80、pass2 修订行为、environment 改 world model 导向
- **v9（当前生产版）**：environment 改「首段建立 + 后续增量」、silero-vad 语音对齐、
  自动多轮重试、凭据脱敏；2026-09-03 补修 `_BJ` 时间戳 bug
- v10 单遍版：已评估未采用，见 `单遍版测试_待审核/`

## 2026-09-04 — 前 50 小时 caption 续跑（新 ADC / HF CPU）

- 用户要求恢复 2026-04-15 起的前 50 小时窗口（实际选取 24 条录制 / 52.7 小时）；已完成的 7 条跳过，提交其余 17 条 HF `cpu-upgrade` job。
- 新 ADC 项目 `project-1a1608e8-53ad-4178-bf7` 无权访问旧中转桶，因此在目标项目创建 `hku-capstone-caption-frames-bf7-20260904`；单请求 GCS 上传 + Vertex Batch 创建/立即取消探针通过，探针对象已清理。生产任务通过 `GCS_BUCKET` 注入该桶。
- `pipelines/caption_v9.py` 支持 `GCS_BUCKET` 环境变量覆盖；`pipelines/batch_worker.py` 新增终止态断点检查，明确 `CANCELLED/FAILED/EXPIRED/SUCCEEDED` 时清除旧 pass1/pass2 断点并按缺失 clip 重建请求，避免把取消任务误接续为失败。
- 本轮 job 使用 `gemini-3.7-flash`、v9 Batch、24 caption workers、24 GCS upload workers、12 speech workers；ADC JSON 仅作为 HF secret 注入，不写入仓库。
- 首次重提的系统性失败根因：`caption_core.py` 曾把 `ADC_PROJECT` 硬编码为旧项目，导致 Batch create 403；现已改为优先读取 `ADC_PROJECT`/`GOOGLE_CLOUD_PROJECT` 环境变量。r3 重提时显式注入目标项目，抽查 job 已确认环境变量正确。
- 当前状态：r3 的 17 个 job 已提交并运行，后台监督器负责终态、预算阈值（>$270 仅停止本轮 job）、结果拉回 `pipelines/out/`；完成后补写 manifest 和验收记录。未完成前不要宣称 24/24 交付。

## 2026-09-05 — Qwen/Doubao 10 场景 v10 批处理（第二轮：高分辨率+prompt boost）
- 目标：提升 caption 对屏幕内容（代码、聊天、文档）的描述精度。用户反馈旧版 Doubao "描述电脑过程太泛泛"。
- **分辨率调优结果**：
  - Qwen 图像 token 在 2048px 达上限 2502/张（2880px 不再增加），故 Qwen 设 2048px q85。
  - Doubao 1440px 实测 token 反而比 1024px 少（1296 vs 1567/image），且耗时 4× → 证实 Doubao 内部固定缩放，提分辨率无意义。Doubao 保持 1024px q85。
- **Doubao prompt boost**：在 user message 追加"Screen Content Requirements"指令（要求 text_visible ≥5-10 项、引用完整代码/聊天文本、aim for 5000+ tokens）。效果：completion tokens 平均提升 1.7×，text_visible 项数提升 2-3×。
- 场景与帧源不变：S01-S22 共 10 场景，gaze_overlay 30 帧/场景。
- 结果：20/20 全部成功（3 个 Doubao 场景需 JSON 修复：text_visible 漏 `[]` 括号）。
  - Qwen（2048px）：img=75060 tokens/场景，completion 平均 5328 tokens（旧 4847），quality 因更高分辨率文本捕获更具体。
  - Doubao（1024px+boost）：completion 平均 3844 tokens（旧 2341, +64%），text_visible 每段 5-10 项（旧 2-3 项）。
- 旧版 1024px 结果备份在 `_backup_1024px/`。
- 审核视频：`caption_model_batch_20260905/review_videos/`，30 mp4（重新渲染）。
- 踩坑：Doubao 生成的 JSON 频繁把 text_visible 写成裸字符串列表（漏 `[]`），已在 `extract_json` 加 `_repair_json` 兜底。
