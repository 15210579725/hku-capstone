# CLAUDE.md — HKU Capstone 全局导览

面向后续 AI/开发者的仓库导览。逐子项目的细节见各自目录的 `CLAUDE.md`。

## 这个仓库是什么

一个研究型 capstone:用机制可解释性方法探究「LLM 有没有可定位、可读取、可干预的情绪表征」。由**三条探针复现线 + 两套行为仿真**组成,共同搭起一条四级证据链(可分类 → 可对齐 → 可干预 → 可行为化)。

## 顶层布局与依赖关系

```
emotion-probe/          探针线(读取/定位情绪表征)
  RepE-2023-Zou/          ← 方法母本(RepE reading 向量)。其余两线都是它的延伸
  do_llm_feels_2025-wang/ ← 在 RepE 基础上做电路定位 + steering 因果
  anthropic-2026-gemma-version/ ← 在 RepE 基础上换模型(gemma)+ 扩到 8 情绪
simulation-env/         仿真线(把情绪表征接到 agent 行为上)
  concordia-official/     ← Concordia + lucia_sim,EgoLife 数据驱动行为预测
  Generative Agents/      ← 斯坦福生成式智能体 + world/,Smallville 行为回放
```

- 三条探针线**产出物相互独立**(各自的方向向量 `*.pt`),但共享同一套研究问题与 LUCIA/EgoLife 日记数据。
- 仿真线消费日记数据 + DeepSeek 后端,验证"情绪/状态 → 行为"是否一致;**不依赖探针线的产物即可独立运行**。

## 跨项目的共性约定与坑

1. **API 密钥**:全部从环境变量读取,仓库内无明文。`TRANSLATE_API_KEY`(gemma 翻译/生成,OpenAI 兼容端点)、`DEEPSEEK_API_KEY`(仿真 + 部分翻译)。改代码时不要写回明文。
2. **硬编码绝对路径**:多个脚本把数据/输出路径写死成原作者本机或 AutoDL 服务器路径。这是已知技术债;新机器上跑需改路径常量(各子 CLAUDE.md 标了具体位置)。
3. **大文件策略**:`*.pt/*.bin/*.safetensors`、海量仿真 `storage/`、大 CSV 经 `.gitignore` 排除。它们是**可再生产物**(GPU 重跑抽取/仿真即可),不是代码。改 `.gitignore` 前先确认不会带回大文件或明文 key。
4. **打包的官方框架**:RepE、EmotionCircuits、Concordia、生成式智能体均已去掉各自 `.git`,作为普通文件纳入主仓,保留了本地改动。它们各自的上游 URL/commit 记在子 README。
5. **transformers 版本敏感**:RepE 线锁 `4.40.2`(同时满足 Qwen2.5 加载与 RepE pipeline 子类);不同探针线对 transformers 版本要求不同,**务必各自独立建环境**。

## 验证现状

- `emotion-probe/RepE-2023-Zou`:本机已跑通 `run_mock_selftest.py`(8 阶段 mock 全链路)+ `pytest`(6 passed),不依赖 GPU/模型。真实抽取需 GPU。
- 其余线需各自的模型 + GPU/API,未在本机端到端验证;文档基于真实代码与作者已落盘的结果撰写。

## 给后续工作的提示

- 动任何一条线之前,先读该目录的 `README.md`(用法)+ `CLAUDE.md`(架构/坑)。
- 涉及"跑起来"的任务,优先用 RepE 线的 mock 自测确认基础环境,再逐步上 GPU。
- 修复硬编码路径、补齐各线的 `requirements.txt`、统一 key 注入,是让"clone 即跑"更顺的高价值方向。

## 2026-08-17 — MiMo caption 首个脱敏 AutoDL-FS 样本测试
- 队友代码：`/Users/mac/Desktop/hku capstone/caption/ARIN7600`，来自 `OldLigant/ARIN7600`，本次 clone HEAD `ba53a06`；实际使用 `AriaRealLife/caption_pipeline_arl.py`，未修改其 caption prompt/解析逻辑。
- 测试正式 tar：`/autodl-fs/data/data-masking/masked-data/6-22_hkt1027-1030_3m_084fb5e2-24ab-402c-a31c-f4fbd05ec510.tar`。选它是因为较早两个 1min 小包存在历史日期/时间命名错位；该包 masked 图时间与 HKT 一致。
- 本次取前 180 张 1Hz masked 原图（2026-06-22 10:27:45–10:30:44 HKT）+ `audio/anonymized.wav`，正好 36×5s scenes / 180s。
- MiMo：`mimo-v2.5`，thinking disabled。先 probe 成功，再全量；最终 `captions.jsonl` 36/36 scenes、unique=36、missing=0、最终 recovery 全为 ok；共 self_actions=85、others=34、speech=37、ocr=0。续跑 12 API workers 的最后 15 scenes 40.3s 完成，0 final failure。
- 原始队友 caption：`/Users/mac/Desktop/hku capstone/caption/mimo_test_6-22_084fb5e2/captions/captions.jsonl`
- 衍生绝对 HKT caption（不改原始字段，只追加 review 时间/源帧）：`captions_hkt.jsonl`
- 人类可读审核：`captions_readable.txt`
- 烧录字幕：`captions_review.ass` / `captions_review_1440.ass`
- MiMo key 未写入项目记忆/仓库；运行后移到 `/Users/mac/.config/hku-capstone/mimo.env`，mode 600。测试目录只保留 `.env.example`。
- 队友脚本写 `clips.parquet` 时因 venv 未装可选 `pyarrow/fastparquet` 出 warning，但不影响 JSON caption；本次没有为非必要 parquet 额外改环境。
- 审核视频采用远端正式 tar 内 2880×2880 masked 原图作为源、匿名音频作为声音，输出编码缩放到 1440×1440；中文字幕依赖 westc 测试实例安装的 `fonts-noto-cjk`。

### 2026-08-17 MiMo review video 最终结果补充
- 前述“远端 2880 原图缩 1440 烧录”方案未作为最终产物：westc 当前容器 cgroup memory.max=2GiB，且同实例百度下载产生大量 file page cache；1440/1080 远端 ffmpeg 会逼近/触发 cgroup OOM，因此主动放弃不完整远端视频，不拿 partial 给审核。
- 最终审核视频在 Mac 本地生成，使用**MiMo caption 实际看到的 180 张 1024×1024 masked 帧**（它们仅由正式 tar 的 2880×2880 masked 原图缩放得到，没有生成/重绘）+ 同一 `audio/anonymized.wav`。画面上方完整保留 1024×1024 输入帧，下方 416px 独立 caption 面板，不遮挡画面。
- caption 面板保持模型原文，按 5s scene 展示绝对 HKT、SELF / OTHERS / SPEECH / ENV / PSY；使用 STHeiti Medium 中文字体。Pillow 仅安装在 `/Users/mac/Desktop/hku capstone/caption/.venv-mimo` 用于审核帧渲染。
- 最终视频：`/Users/mac/Desktop/hku capstone/caption/mimo_test_6-22_084fb5e2/review_180s_captioned.mp4`
- QA：180.000s，1024×1440，1fps H.264 + AAC，27,936,552 bytes；ffmpeg 全片 decode OK；2s/62s/122s 抽帧 caption panel 均检测到有效文字像素。

## 2026-08-17 — MiMo caption 全中文翻译 + 审核视频
- 用户要求：所有 caption 翻译成中文后重新合成审核视频。
- 原始 caption 保留不动：`caption/mimo_test_6-22_084fb5e2/captions/captions.jsonl`。
- 新增翻译脚本：`caption/mimo_test_6-22_084fb5e2/captions/translate_captions_zh.py`；使用已配置的 MiMo `mimo-v2.5`，6 batches × 6 scenes 并行翻译，严格保留 clip_id/time/time_end/lang/数组顺序。
- 翻译覆盖全部语义 caption 字段：self_actions、others、environment、speech（含 speaker）、psychology（awareness/emotion/note）、ocr、tags；并用中文结构重建 `narrative`，避免 narrative 中残留英文副本。
- 中文结果：`caption/mimo_test_6-22_084fb5e2/captions/captions_zh.jsonl`，36/36 scenes；语义字段共检查 645 个字符串，英文残留 0 条。
- 中文可读版：`caption/mimo_test_6-22_084fb5e2/captions/captions_readable_zh.txt`，栏目和时间标题也全部中文（香港时间、我的动作、他人动作、语音、环境、心理、标签）。
- 中文审核帧：`caption/mimo_test_6-22_084fb5e2/review_frames_zh/`，180/180；上方保留 MiMo 实际看到的 1024×1024 masked 帧，下方 416px 中文 caption 面板，不遮挡画面。
- 最终全中文视频：`caption/mimo_test_6-22_084fb5e2/review_180s_captioned_zh.mp4`。规格：180.000s、1024×1440、1fps H.264 + AAC、27,876,190 bytes；ffmpeg 全片 decode OK；2s/62s/122s 抽检字幕面板均非空。
- 原匿名音频继续使用 `input/mimo_input/audio/anonymized.wav`；未修改/删除任何原始正式数据或英文 caption。

## 2026-09-04 — metric/eval_kit + caption/run_pipeline.py 推送 GitHub

### eval_kit（行为预测评测工具）
- 路径：`metric/eval_kit/`，commit `dbbbabce`
- 三种评分器：SED（软编辑距离，无需 API）、Embedding（默认 SiliconFlow Qwen3-Embedding-8B）、LLM-Judge（推荐 RightCode Gemini-3.7-flash）
- 输入 JSONL 格式，输出逐样本得分 + 汇总报告
- 配置：`config.json`（填 API key 即用）

### run_pipeline.py（统一 dense video captioning 流水线）
- 路径：`caption/run_pipeline.py`，commit `4448c70c`
- 合并 `pipelines/` 下 6 个模块（remote_tar / caption_core / caption_v9 / cloud_worker / merge / pipeline）为单文件 ~2400 行
- v9 三遍架构：silero-vad 语音对齐 + 2880px q80 pass1 + ULTRA_HIGH crop pass2 精读修订
- 支持两种 tar 布局（A/B）自动适配、两种 transcript 时间格式
- 端到端冒烟通过：1 分钟录制（布局 B），1 clip，全链路 126s 完成，产出 json/jsonl/txt/report
- CLI：`list` 列出所有天和录制，`run --day/--tar` 处理指定数据
- 依赖：numpy/Pillow/requests/huggingface_hub/google-genai；torch/silero-vad 可选
- 凭据不入仓库：config.json 中 hf_token 和 google_project 均为空模板

## 2026-09-04 — Google Cloud ADC（b34927743）配置与 Gemini 3.7 Flash 视频验收

- 目标账号：`b34927743@gmail.com`；项目：`project-1a1608e8-53ad-4178-bf7`（My First Project）。
- ADC 已通过 Safari 的 Google OAuth 流程重新生成到本机默认路径 `/Users/mac/.config/gcloud/application_default_credentials.json`；未把 ADC、刷新令牌或 OAuth 授权码写入项目。
- 为多账号切换保留 profile：`/Users/mac/.config/gcloud/adc-profiles/b34927743.json`；原有 `meng15210579725.json` 未删除。使用时通过 `GOOGLE_APPLICATION_CREDENTIALS` 显式选择。
- 只读核验：OAuth userinfo 返回目标账号；Resource Manager 项目为 ACTIVE；`aiplatform.googleapis.com` 已启用；Cloud Billing `billingEnabled=true`，账单账号关联状态已确认。
- 低成本 Vertex 端到端验收：`gemini-3.7-flash` global endpoint 的文本/思考/图像/音频/视频 6 项均 HTTP 200；视频 fixture 返回“从红色过渡到蓝色”。证据归档于 `google_accounts/b34927743/`：`video_proof.json`、`gemini37_vertex_adc_test.json`、`synthetic_red_blue.mp4`、`README.md`。
- 未完成项：尚未配置其他 Google 账号；后续新增账号应各自生成独立 ADC profile，禁止复制或提交凭据内容。

## 2026-09-04 — English paper draft migrated to an Overleaf-ready Git workspace

- Goal: consolidate the supplied `/Users/mac/Downloads/Hku capstone (1).zip` paper outline and all 19 PNG figures into an English LaTeX draft, and document the privacy/ethics workflow from the companion masking repository.
- Workspace: `paper/overleaf/` (independent Git repository with `overleaf` remote set to the user-provided project URL). The workspace contains `main.tex`, seven section files, `refs.bib`, `README.md`, `source_manifest.tsv`, `.gitignore`, and 19 semantically renamed PNGs. The four source MP4s and two HTML review pages remain outside the Overleaf workspace.
- Key decisions: use an article-style local compile fallback with an optional ICLR style hook; label current metric values as pilot results; write `deface CenterFace` and local OpenAI Privacy Filter/OPF checkpoint rather than claiming cloud OpenAI API transfer; describe audio order as Rubber Band voice-only transform first, then original-signal ASR/OPF/ForcedAligner and privacy-span muting in the transformed waveform.
- Verification: `pdflatex` + `bibtex` + two final `pdflatex` passes completed with exit code 0; output `paper/overleaf/main.pdf` is 13 pages, Letter size. Figure inventory is 19/19 and the manifest contains SHA-256 values. Overleaf `git ls-remote` was attempted read-only with prompts disabled and returned `could not read Username`; no push was claimed or performed.
- Unfinished: Overleaf Git authentication token is still required for pull/push; official ICLR style file, final authors/affiliations, final participant split, external bibliography metadata, confidence intervals, and final prompt/model freeze remain to be completed by the team.
- Next step: review the English draft and figure selection in Overleaf, then run `git pull --rebase overleaf main` and push with `git push overleaf HEAD:main` after authenticating with an Overleaf Git token; do not place the token or raw media in the repository.

## 2026-09-04 — Overleaf Git push completed

- User generated the Overleaf Git authentication token interactively; the token was not sent to chat, printed, or written to the repository.
- The Overleaf remote default branch was verified as `main` with one starter commit. The local paper repository merged that unrelated starter history while retaining the English `main.tex` and all paper files.
- Push completed successfully: remote `refs/heads/main` now points to commit `956382c` (`Document Overleaf main branch sync`), whose parent is the paper merge commit `36dd5baa3de7b6545ec799fba303e88d7ecaf744`.
- Remote verification: Overleaf file tree shows `main.tex`, seven `sections/*.tex` files, `refs.bib`, `README.md`, `source_manifest.tsv`, and all 19 PNGs under `figures/`; the online PDF preview shows the NextMe-800 title and 13 pages after a fresh recompile.
- Remaining review: replace the optional article fallback with the official ICLR style if required, finalize authors/affiliations and bibliography, and decide whether embedded Chinese labels inside source screenshots must be redrawn in English.

## 2026-09-04 — ICLR 2027 style and reviewer-facing ethics revision

- Downloaded the official ICLR 2027 style package from the conference author-guidelines URL and added the exact `iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`, `fancyhdr.sty`, and `math_commands.tex` files to `paper/overleaf/`.
- `main.tex` now uses `\\usepackage{iclr2027_conference,times}` with anonymous submission mode (`\\iclrfinalcopy` remains commented). A natbib style-name registration is included so the official author-year `.bst` compiles under the current TeX Live release.
- Added the ICLR-required AI use statement and reviewer-facing Ethics and Reproducibility statements. Rewrote `sections/06_ethics_privacy.tex` around the threat model, consent/governance, native-resolution CenterFace/PP-OCR workflow, local OPF text protection, Rubber Band + ASR/ForcedAligner audio order, fail-safe muting, audit gates, and residual linkage risks.
- Added the previously omitted `figures/judge_stability.png` to the appendix; all 19 manifest figures are now referenced by LaTeX.
- Verification: clean four-stage `pdflatex`/`bibtex` build succeeds with official style, output is 13 Letter pages, all 19 figure references resolve, no secret-pattern scan or `git diff --check` findings. Official ICLR 2027 guidance says initial main text is at most 9 pages, references are unlimited, and double-blind anonymity is mandatory.
- Pushed to Overleaf `main` as commit `7fb6caa` and recompiled in the browser. The page shows the anonymous ICLR 2027 header, the new AI/Ethics/Reproducibility statements, the detailed privacy section, and all 19 figures. Authors/affiliations remain intentionally blank.
- Remaining review: the main-text body currently reaches the 9-page submission boundary before references; final shortening may be needed after real citations, author metadata, and final results are inserted. Decide whether embedded Chinese labels inside source screenshots must be redrawn in English.

## 2026-09-04 — Long-horizon dataset comparison added

- Goal: compare NextMe-800 with the datasets explicitly listed in the supplied capstone PDF and make the meaning of “wild” operational for an ICLR reviewer.
- Modified `paper/overleaf/sections/02_dataset.tex`: added a compact source-backed comparison table covering NextMe-800, EgoMonth, EgoLife, LongNAP, LSC/NTCIR Lifelog, Ego4D, KrishnaCam, and CASTLE; added a concise ecological/temporal/sensor definition of wildness and a source/code-audit note.
- Modified `paper/overleaf/refs.bib` with dataset paper/project citations; added `paper/overleaf/DATASET_SOURCES.md` with primary URLs and the evidence checked for each row.
- Moved the privacy-control evidence matrix to the appendix so the main text remains within the ICLR nine-page boundary while retaining the detailed ethics workflow in Section 6.
- Verification: clean official-style four-pass LaTeX/BibTeX build succeeds; PDF is 14 pages total with the main text ending on page 9, statements/references/appendix afterward; `git diff --check` is clean; rendered page 3 confirms the dataset table is readable and has no overflow.
- Key caveat: the current artifact-backed NextMe-800 statistic is approximately 750 hours over four months; the separate “1k hours” note is explicitly labeled as a planning target, and final participant count/duration/split remain to be frozen.
- Delivery: committed as `65ad3e9da7e81cb79e2b6afa5cfdabd7e07f3ac4` and pushed to Overleaf `refs/heads/main`; read-only `git ls-remote` matches the local commit. Overleaf was refreshed and recompiled successfully; the online preview shows the revised dataset section and the page-3 comparison table. Local rendering verified the appendix privacy matrix and the main-text page-9 boundary.
- Remaining next step: freeze final participant/duration/split statistics and replace provisional bibliography/author metadata before camera-ready submission.

## 2026-09-05 — Caption 结果交付

- 目标：将 `/Users/mac/Desktop/hku capstone/caption/pipelines/out/_delivery_20260905` 的可分析 caption 数据同步到 GitHub 仓库 `caption-result/`。
- 修改：新增 `caption-result/`（24 条录制，各含 `captions.jsonl`、`captions.txt`，并保留 `index.json`）；根目录 `README.md` 增加数据目录、字段说明、覆盖率提示和 JSONL 读取示例。
- 验证：交付目录共 49 个文件、约 2.6 MB；提交前将检查 JSONL 可解析、文件数量与索引一致。
- 未完成项：GitHub 推送后的远端目录和 README 链接待最终核验。
- 下一步：提交并推送后检查远端 `main` 内容。

## 2026-09-05 — 统一评测数据集 dataset/ 推送 GitHub

- 目标：合并 EgoLife（6 人 × 7 天 × 5 层级）与 Next-Me（24 条 Aria 眼镜录制）为统一评测数据集，固定 1k 测试点索引。
- 目录结构：`dataset/egolife/{A1-A6}/L{1-5}/day{1-7}/events.txt` + `dataset/nextme/{recording_id}/events.txt`
- A4_LUCIA 使用 Claude-Opus 标注版本（替代 ds-flash），其余 5 人使用 DS-Flash 标注版本。
- Next-Me caption JSONL 转换为统一 `[HH:MM:SS -> HH:MM:SS] action` 格式，共 2107 条事件。
- `benchmark_1k.jsonl`：1000 个固定评测点（800 EgoLife + 200 Next-Me），L1 粒度，seed=42，50 条上下文 + 3 条 GT。
- `data_index.json`：全量 234 个事件文件清单。
- `run_benchmark.py`：统一预测脚本，默认 ds-v4-flash，输出与 `metric/eval_kit/evaluate.py` 兼容的 JSONL。
- 端到端验证：ds-v4-flash 10 点（5 EgoLife + 5 Next-Me）预测 + eval_kit 格式兼容 dry-run 均 PASS。
- 提交 `0dbfc1de`，已推送 `origin/main`。
- 总数据量 ~47MB（234 个 events.txt + 索引 + 脚本）。

## 2026-09-06 — 100 小时 Caption 脱敏交付（待 GitHub 推送）

- `caption-result/` 已准备替换为 133 条录制的脱敏版本：12,068 个成功 clip；37 条失败 clip 已过滤；前 100 小时为 11,813 / 11,943（98.9%）。
- 每条录制含 `captions.jsonl` 与 `captions.txt`，根目录 `index.json`；`_redaction_audit.json` 记录脱敏类别计数与验证摘要，不含原始匹配值。
- 脱敏掩码为 `XXX`，覆盖凭据、邮箱、电话、姓名、地址、支付信息、URL/本地路径等；输出 JSONL 已逐行验证可解析，且仅保留顶层 `ok=true` 记录。
- 分析注意：`clip_index` 可能稀疏，时间戳与录制覆盖范围请以各条记录字段为准；前 100 小时窗口并非 100% 覆盖。
- 当前仅完成本地替换与待审查 staging；GitHub 推送及远端实时核验尚未完成。

## 2026-09-07 — HF 已完成 Caption 快照同步（待远端核验）

- 目标：把 `mmm8383/pov-captions` 当前已完成的 caption 回拉到本地，过滤失败行、全量脱敏并同步到仓库 `caption-result/`。
- 快照边界：HF 当时有 217 条录制、44,938 个 caption JSON；大小筛选得到 42,552 个候选，发布 42,304 个 `ok=true` clip、过滤 248 个失败 clip，共 228,494 个 segment。这是已完成部分的时间点快照，不代表全部计划作业完成。
- 修改文件：替换 `caption-result/`；更新根 `README.md`；修复 `caption/pipelines/redact_delivery.py` 的姓名传播、HF token、邮箱边界和发布索引重算逻辑；新增/扩展 `caption/pipelines/test_redact_delivery.py`。
- 关键决策：发布 coverage 定义为本次选中候选中的成功比例；旧计划目标保存在 `source_planned_target_clips`。敏感字段与 speech 说话人信号驱动人名传播，避免把 UI/模型标签误当人名。`content_raw` 与结构化结果重复且扩大隐私面，发布副本统一置为 `XXX`，分析使用 `merged` / `parsed`。
- 本地验证：最终 release9 含 217 个 JSONL 与 217 个 TXT；42,304 行与 42,304 个 TXT clip 标题一一对应；全量 JSON 可解析且仅有 `ok=true`；228,494 个 segment；模型分布为 `gemini-3.5-flash` 3,252、`gemini-3.5-flash-lite` 2、`gemini-3.6-flash` 5、`gemini-3.7-flash` 39,045。独立复扫通用 PII、敏感姓名字段、源派生 180 个强姓名信号和 5 个弱说话人信号均为 0；`content_raw` 非遮罩数为 0；最大单文件 5,902,384 bytes。回归测试 10/10 通过。
- 质量修复：独立审计发现 `lite` 曾被误当成人名并破坏 2 条 model 标识；已把该技术标签加入非人名集合，加回归测试并从只读 HF 快照完整重建，最终模型分布与源数据完全一致。
- 安全提示：排查 LaunchAgent 时曾让包含 `CONTROL_PLANE_API_KEY` 的完整环境出现在本地工具输出；没有写入仓库，但该 key 应轮换。后续只允许过滤后的 `launchctl print`。
- GitHub 同步：数据发布提交为 `d02ff91cc238d2ca8e7103dea6b6b02231f8406f`（`Sync completed HF caption snapshot`）；`git push origin main` 退出码为 0，随后 `git ls-remote origin refs/heads/main` 返回同一精确 SHA。
- 完成状态：HF 已完成部分已回拉到本地并经脱敏发布；远端数据提交的 `caption-result/` 树已核验为 436 个文件（217 JSONL、217 TXT、`index.json`、`redaction-audit.json`）。本段后续作为独立小型进度记录提交，不改变已发布数据。
