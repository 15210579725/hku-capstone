# CLAUDE.md — POV Caption 云端流水线

`mmm8383/pov-data` 的脱敏 tar → HF Job 上原地解包 → Gemini 3.7-flash dense caption →
产物回 `mmm8383/pov-captions` → 本地取回、合并整天、渲染审核视频。

上级 `../CLAUDE.md` 记的是 v6 prompt / 分辨率测试 / 单点验证脚本，先读它再读这里。

## 为什么全在云上跑（2026-09-02 实测，不要再试本地拉 tar）

| 约束 | 实测 |
|---|---|
| 本机 → HF 带宽 | **2.3 MB/s**，单连接与 8 并发相同（总带宽封顶）；Clash 代理与直连一样；**Xet 通道更慢**（1.4–1.8 MB/s）|
| 本机主盘剩余 | 8.4 GB |

一条 43 分钟录制 9.58 GB 本地要拉 70 分钟，226 分钟那条 42.66 GB 要 5.2 小时。
HF Job 在 HF 内网里读同一个 repo，`cpu-upgrade` $0.03/小时。

## 目录

```
pipeline.py        本地 CLI（选片/提交 job/看状态/拉结果/合并/渲染）
remote_tar.py      tar 访问层：FileSource(挂载) + HttpRangeSource(Range)，同一接口
cloud_worker.py    HF Job 里跑的主流程
caption_core.py    帧渲染 + context 组装 + Gemini 调用（context 与 ../run_caption.py 等价）
merge.py           clip → 录制 → 整天
render_review.py   审核视频（复用 ../render_caption_video.py 的面板与编码参数）
prompts/v6.txt     ../prompt.txt 的冻结快照
out/<date>/<rec>/  拉回本地的产物
```

云端 `mmm8383/pov-captions`（private）：
`_code/`、`captions/<rec>/clip_XXXX.json`、`captions/<rec>/captions_full.{json,jsonl,txt}`、
`captions/<rec>/{manifest,report}.json`、`clips/<rec>/clip_XXXX.tar`（30 帧 + audio_16k.wav + meta）。

## 常用命令

```bash
python pipeline.py days
python pipeline.py scan --month 2026-05 --limit 12      # 候选：时长/大小/眼动/语音密度
python pipeline.py run --tar aria/2026-05-18/5-18_...tar --max-clips 3   # 冒烟
python pipeline.py run --tar aria/2026-05-18/5-18_...tar                # 整条
python pipeline.py run --day 2026-05-18                 # 一天全部
python pipeline.py run --tar ... --only-failed          # 只重跑失败的 clip
python pipeline.py status --all ; python pipeline.py logs <job_id>
python pipeline.py pull --rec <rec> --sample 8          # caption + 抽样 clip 包
python pipeline.py merge --day 2026-05-18
python pipeline.py render --rec <rec> --sample 8
```

## tar 内部布局（**两种**，不要假设顺序）

```
<rec>/eye_tracking/*.jpg      眼部相机 1Hz ~22KB/张 —— 跳过（want=False 时不发请求）
<rec>/audio/anonymized.wav    48kHz mono 16bit
<rec>/audio/transcript.txt    [YYYY-MM-DD HH:MM:SS HKT -> ... HKT] 文本，30 秒块
<rec>/picture/masked/*.jpg    2880²、~3.1MB/帧、**按时间倒序**、文件名走 GNU @LongLink
<rec>/picture/ocr_text/*.txt  每帧一份，前三行是 capture_time/capture_epoch/frame_index
<rec>/eye_tracking/gaze.csv   append 在**最末**（12 条录制没有）
```

**上面那种是布局 A。2026-09-03 实测：4-15 起 50 小时窗口的 24 条里，A 只有 16 条，
另外 8 条是布局 B** —— `picture/ocr_text/`（**正序**）在最前，然后 `picture/masked/`，
`audio/` 在靠后位置，`eye_tracking/gaze.csv` 仍在最末：

```
布局 A（16/24）  eye_tracking/*.jpg → audio/ → picture/masked → picture/ocr_text → gaze.csv
布局 B（8/24）   picture/ocr_text（正序）→ picture/masked → audio/ → gaze.csv
```

`locate_sections()` 靠「二分找 eye_tracking 末尾、紧跟着就是 audio」定位，在布局 B 上
直接抛 `RuntimeError: 找不到 audio/anonymized.wav`（4-27_hkt0930 冒烟实测），
而且 `read_tail_adaptive()` 在 B 上永远读不到 ocr（ocr 在头部，不在尾部）→ `ocr=0 帧`。
**别给每种布局打补丁**：`remote_tar.walk_headers()` 只读 header 不读数据地把整条 tar
索引一遍（64KB 窗口，最大那条 ~4 万 member 只读 ~2GB），之后按偏移各取所需，
布局怎么排都无所谓。`batch_worker.index_tar()` 就是这么用的。

判布局是不是 B 的最省办法：读开头 2MB，看第一个非目录 member 在 `picture/` 还是
`eye_tracking/` 下。

## 踩过的坑 / 必须知道的约定

1. **masked 帧是时间倒序的**，而注视轨迹依赖正序历史。`caption_core.build_trails()`
   先按 `frame_index` 升序把每帧的轨迹点算好，渲染才能乱序并行。别改成边渲染边累积历史。
2. **clip_id = frame_index // clip_seconds**，不依赖总帧数，所以 `--max-clips`
   只取 tar 开头（= 录制末尾）那几个 clip 时编号仍与全量一致，能无缝续跑。
3. **transcript 按相对偏移对齐**（首个区间为基准），不能用绝对时钟 ——
   `_tools/extract_30s_from_masked_tar.py` 踩过，legacy 包的 masked 绝对时钟会整体平移。
4. **注视圈参数照抄** `../mimo_v25_audit_30x30s_20260821_COMPLETE/_tools/render_gaze_overlay.py`：
   `(0,255,128)` 空心圈 r=18 width=3 + 中心实心点 r=4，`(255,200,0)` 轨迹回看 5 帧渐隐，
   无效注视左上角红叉。区别只在于那边先在 2880² 画再压到 1440，这边直接在 1440 上按
   `scale=1440/2880` 等比画（省一次全分辨率解码，模型看到的相对位置/大小一致）。
5. **哪些录制没有眼动**：`_work/todo.txt` 列的 521 条有；不在里面的 12 条确认无 MPS 眼动
   （含 4-15 和 5-09_daf330ff 两个 v5 确定性 UUID）。判定也可以按 `_work/verify_coverage.py`
   的办法读 tar 尾部 8MB 找 `eye_tracking/gaze.csv`。
6. **`_work/gaze_bundle.tar` 里只有 524 个 csv**，同样没有那 12 条。
7. Gemini **inline 上限 20MB**：`caption_core.enforce_inline_limit` 在 18MB 处兜底降到
   1200/1024 重压。实测 1440px × 30 帧才 ~4.5MB（每帧 ~158KB），正常摸不到。
8. **tar 文件名的日期 ≠ 录制内部时钟**：`5-18_hkt2130-2214` 这条内部帧名和 transcript
   都是 `2026-05-17`。`out/` 按**内部日期**归档，所以合并要用 `merge --day 2026-05-17`。
9. **`hf jobs run` 会把 `bash -lc` 的 `-l` 当成自己的 `--label` 吃掉** → 只能用 `bash -c`。
10. **HF snapshot 目录里是指向 `blobs/` 的符号链接**，直接 `python cloud_worker.py` 会把
   blobs 目录当 `sys.path[0]`，同目录的 `caption_core` import 不到 → 先 `cp -rL` 到普通目录再跑。
11. HF commit 有频率限制，caption 每积 25 个提交一次、clip 包按 25 个一批，别一个文件一次 commit。

## 后续升级路线

上级目录 2026-09-02 下半场已把 caption 推进到 **v7 两遍 crop**（`../run_caption_2pass.py`）：
pass1 出完整 caption + `text_regions` 框，pass2 按框从 2880px 原图 crop 再用
ULTRA_HIGH 精读文字，屏幕文字提取量提升 5–8 倍。本流水线目前按用户要求先对齐
**v6 单遍**（`caption_core` 的 context 与 `../run_caption.py` 等价）。要升 v7 时：
`caption_core` 里加 pass2，`cloud_worker.do_clip` 改成两遍，帧包已经存的是 1440px，
crop 需要的 2880px 原图得在 walk 阶段一并留下（或按需重取那几帧的字节区间）。
注意上级记的两个坑：`ULTRA_HIGH` 只在 per-Part 级别有（全局 config 只有 LOW/MEDIUM/HIGH），
以及并发别开太大（8 workers × 32 上传线程会打满网络导致 SSL EOF）。


## 2026-09-02 下半场 — 吞吐/成本实验（750 小时可行性）

### 单价必须用对，之前算错了 2.5 倍
`gemini-3.7-flash` 官方价 **$0.75/M in、$3.75/M out**（2026-12-31 前引导价，之后翻倍），
Batch 半价。**thinking token 按 output 计费**，早期漏算了。这条 43.5 分钟录制真实花 **$3.20**
（≈$4.42/小时视频），不是最初报的 $1.20。`merge.py` 的 PRICE_IN/PRICE_OUT 已修正，
`tokens.out` 已改名 `out_billable` 并含 think。

### 瓶颈定性：是 Vertex 的动态共享配额在降速，不是我们这边
逐个排除：
- **不是带宽**：768px 把 payload 从 5.06MB 砍到 0.96MB（1/5），吞吐反而没变好
- **不是客户端 CPU/GIL**：30 帧 base64+json 序列化只要 0.023s
- **不是 429**：87 次调用只重试 3 次
- **不是连接池**：每线程独立 client 比共享 client 更慢（被时间效应掩盖）
- **是 DSQ**：5 组顺序跑的实验性能单调衰减（0.435→0.060 clip/s）、延迟单调上升（22→68s），
  且冷却 20 分钟后没恢复；最后连一句 "ok" 都要 14.2s
- **多区域这条路堵死**：`gemini-3.7-flash` 只在 `global` 端点存在，19 个区域端点全 404

### 并发是按客户端算的，多开 job 线性扩容
4 个 job 同秒起跑各跑 20 clip：各自 2.14–2.39 clip/s（没被摊薄），
**聚合 9.16 clip/s ≈ 2000 万输入 tok/分**，429 率 7.5%（重试能兜）。
所以 750 小时的吞吐不是问题，**问题是成本和模型质量**。

### 模型对照（同 11 clip，同 prompt，同 OCR/transcript 上下文）
| 模型 | 段数 | 语音 | 文字条 | 物体 | 描述字数 | $/clip | 中位延迟 |
|---|---|---|---|---|---|---|---|
| gemini-3.7-flash（基线） | 5.2 | 1.5 | 9.5 | 11.0 | 1060 | 0.0323 | 27s |
| gemini-3.5-flash-lite | 4.0 | 1.3 | 9.0 | 7.8 | 878 | 0.0132 | 6.5s |
| gemini-3.1-flash-lite | 3.9 | 0.9 | 9.3 | 6.3 | 798 | 0.0106 | 8.5s |
| 3.5-lite + 15帧(stride2) | 4.0 | 1.4 | **6.9** | 8.3 | 885 | 0.0078 | 8.7s |

- **3.1-flash-lite 不能用**：clip_0000 里把「从床上抱起笔电走到餐桌」整条物理动作链丢了，
  只说「看屏幕、视线移动」。这正是数据集最核心的行为信号。
- **减帧的代价很集中**：描述质量不变，但**屏幕文字提取掉 23%**（帧少 = 抓文字机会少）。
- **`gemini-3.8-flash` 别用**：thinking 输出 5,965 tok，比 3.7-flash 还贵 40%。

### PII 合规：所有模型都不 100% 守 Rule 8，lite 差 10 倍
`merge.py` 新增 `pii_audit()`，结果进 `report.json`。实测：
- 3.7-flash：458 段里 4 处命中（2 处明文 `HKU`）/ 2 处正确匿名 → **0.9% 段落泄漏**
- 3.5-flash-lite：~44 段里 4 处命中 / **0 处匿名** → **~9%，是基线 10 倍**
**花钱跑全量之前必须先加固 Rule 8 + 加正则后处理兜底**，否则公开数据集会带明文校名/院系。

### Batch API 已验证可用（750 小时的主选项）
24 clip 端到端 16 分钟 `SUCCEEDED`，输出 24/24 有内容、token 与在线完全一致
（in=38,856 / out=1,866）、caption 结构与 HKT 时间轴正常。**用的就是被降速的 3.7-flash
也照跑 → Batch 是独立容量池**。GCS 上传 38 MB/s，固定开销约 4.4 分钟排队。
生产时 JSONL 走 inline base64（1440px 全量约 620GB，768px 约 124GB，token 完全相同）。

## 自动关机 / 防漏钱（2026-09-02）

HF Job 是脚本跑完就退，正常不常驻。三处真会漏钱的已封上：
1. **挂死空转** → `cloud_worker` 内置 `Watchdog` 线程，连续 N 分钟无进展就 `os._exit(75)`
   （`IDLE_TIMEOUT_MIN` 默认 15，8 个关键节点 `beat()` 打点）；`--timeout` 按分片时长自动算、封顶 24h
2. **Vertex batch 遗留**（按 token 计费，弃置照样烧）→ `pipeline.py ps` 一屏看 HF job + batch；
   `pipeline.py stop` 一键砍在跑的 job **和**未结束的 batch
3. **GCS 中转文件堆积** → `pipeline.py gcs-lifecycle`，已对
   `gs://hku-capstone-caption-frames` 的 `batch_in/` `batch_out/` 设 7 天自动删并验证生效。
   **修过一个真实生产漏洞**：`cmd_gcs_lifecycle()` 原来只覆盖 `batch_in/batch_out/` 前缀，
   没盖 v9 用的 `p1/p2/` inline-fallback 前缀（pass1 payload 22-25MB 必超 20MB inline
   上限，每次都会真的写 GCS）——已补上 `p1/`/`p2/` 前缀并对生产 bucket 实际执行过一次
   `gcs-lifecycle`（7 天自动删除，非破坏性，未立即删任何东西）。详见
   `../quality_verify/CLAUDE.md`。

## 分片调度（全量用）

`run --all --shards 8`：按时长贪心均衡（532 条 / 47,688 分钟 → 8 片各 5,961 分钟），
一个 job 顺序处理多条 tar（`TAR_LIST`），每条跑完清盘不跨 tar 累积，单条挂掉不影响后续。
`--skip-done`（默认开）跳过 OUT_REPO 上已有 `report.json` 的录制 → 断点续跑天然成立。
`--dry-run` 只打印分片方案。

**踩到的坑**：把 `merge.py` 的 `tokens.out` 改名成 `out_billable` 时漏改了 `cloud_worker`
的日志行，导致每条录制在 caption 全部跑完并上传后才抛 `KeyError: 'out'` 被判失败
（数据没丢，但 manifest/captions_full 永远写不出）。**改共享数据结构的字段名，
必须 grep 全项目所有读取点**。是两条 1 分钟小录制的分片冒烟抓出来的 —— 全量跑之前一定要先冒烟。

### 按 clip 挑选的窗口调度（`quality_verify/` 200 clip 子集用，2026-09-03）

不是新调度器，是复用上面 `run --shards N` 的原班机制，加一层白名单：

```bash
python pipeline.py run --clips-file quality_verify/select/candidates_250.json \
    --shards 8 --clip-seconds 15 --workers 6 --dry-run
```

- `--clips-file`：给候选片段产物（`select/candidates_250.json` 那种 schema：
  `clips[].source.{tar,cloud_clip_id,frame_index_start}`），只 caption 里面列出的 clip 窗口。
  `targets` 自动收窄成这些 clip 落在的 tar（不给 `--tar/--day/--all` 时）。
- `load_wanted_clips()` 会**硬校验** `cloud_clip_id == frame_index_start // clip_seconds`，
  对不上直接 `sys.exit`——这是防「同一个编号在不同切法下指向不同窗口」的静默错位，
  `--clip-seconds` 必须和产出候选那边切 clip 用的秒数一致（SELECT 侧当前是 15）。
- 分片配平方式换了：不再按 tar 分钟数配平，改按**目标 clip 数**配平（因为成本现在由
  clip 数决定，不是 tar 时长）；`--dry-run` 每片打印 tar 数、目标 clip 数、`walk` 分钟数
  （按 tar 全长估，∼1s/tar-分钟，这段 I/O 躲不掉）+ `caption` 分钟数
  （`--sec-per-clip`× clip 数 / `--workers`，默认 20s/clip 是占位值，换 caption 版本后
  要用真实冒烟数字回填）。
- 真正省钱省时间的是 `cloud_worker.py`：`WANTED_CLIPS_JSON` 环境变量（`pipeline.py submit()`
  自动从分片里的 `wanted_clips_for_shard` 序列化过去）在 `process_one()` 算完 `todo` 之后
  按 `{rec: [clip_id,...]}` 过滤——**只砍 Gemini 调用，tar 的 walk/渲染阶段不变**（整条
  tar 还是要走一遍，云端 209MB/s 读速下这个 I/O 本来就快，没必要为了省它去改 tar 内部
  跳读逻辑）。不给 `--clips-file` 时 `wanted_clips=None`，v6 全量路径和历史行为完全不变。
- 250 候选 clip 分布在 117 条不同 tar 里（平均 2.14 clip/tar），实测 `--shards 8`
  比 `--shards 4` 更均衡也更快出活（8 片时单片 walk 24-34min，4 片时单片 walk 55-64min，
  两者 timeout 都留了 ≥3× 余量），8 片够用，不必再退 4 片。

## ⚠️ 与主线 v9 的版本关系（2026-09-03）

本目录的 `caption_core.py` 对齐的是 **v6 单遍**（1440px q60 / 30 帧 / 默认 media_resolution）。
在这些吞吐实验进行的同时，`caption/` 主线已锁定 **v9 三遍架构**
（`../run_caption_2pass.py`：silero-vad 语音对齐 + 2880px q80 pass1 + 2880px ULTRA_HIGH
crop 精读并修订 pass1），`../run_caption.py` 已归档进 `../过程debug/早期caption版本/`。

**成本要换算**：按主线记的两遍实测 token（in 73,125 / out 5,438 每 clip）折算，
v9 每 clip ≈ **$0.0752，是 v6 的 2.04 倍** → 全量 795 小时**在线 $7,182 / Batch $3,591**。
本目录 `merge.py` 的估价是按实际 usage 算的，不写死版本，所以数字本身没错，
但拿 v6 的 $/clip 外推全量时必须乘 2。

**v9 的两条生产约束很可能是本机带宽造成的，不是架构限制**：

| v9 约束（主线 CLAUDE.md 记的） | 本机依据 | HF Job 实测 | 倍数 |
|---|---|---|---|
| 只能 3 workers，更高打满网络 SSL EOF | 上行 2.3 MB/s | 聚合 27.1 MB/s | 11.8× |
| GCS 上传纯开销 2–27s | 32 线程 4.5 MB/s | 38.2 MB/s | 8.5× |
| 每调用掉线 15–20%，5 轮后 12/16 | 33–44MB payload 在 2.3MB/s 上要传 2–5 分钟 | **未测** | — |

**下一步**：把 `run_caption_2pass.py` 接进本目录的云端调度（解包/分片/断点续跑/看门狗/Batch
都是现成的，只换 caption 那一层），在 HF Job 上跑一遍 16 场景基准，直接对比本机的 12/16。
掉线率若塌到个位数，v9 的 `--workers 3` / `--rounds 5` 都能放开。

吞吐报告（含全部实测数据与图表）：`throughput_report.html`，浏览器直接打开。

## 状态

### 2026-09-02 — 流水线建成并端到端跑通

片源 `aria/2026-05-18/5-18_hkt2130-2214_43m_53b730f3-2fe6-402b-a233-d9849ff782b0.tar`
（43 分钟 / 9.58 GB / 2609 帧 / 87 clip，有眼动，transcript 密度 120.4 字/分，
8 个候选里第一，第二名 72 字/分）。原定的 4-15 因无眼动排除。

**实测数字**（`cpu-upgrade`，挂载 `-v hf://datasets/mmm8383/pov-data:/data:ro` 生效）

| 项 | 结果 |
|---|---|
| 云端读 tar | **209–235 MB/s**（本机是 2.3 MB/s，差 ~90 倍）|
| 整条端到端 | **10 分 04 秒**（解包 44s + caption 530s @24 并发 + 上传合并）|
| clip 成功率 | **87/87**，458 segments，121 个带语音原文 |
| token / 费用 | in=3.38M out=73K，**$1.20**（约 $0.014/clip、$1.65/小时视频）|
| 帧体积 | 2880² 原图 ~3.5MB → 1440px q60 **~158KB**；30 帧一个 clip ≈ 4.7MB，远低于 20MB inline 上限 |
| 冒烟（3 clip） | 45s / $0.04 |
| `--only-failed` | 87 个里精确挑出 1 个失败，只重跑它，1 分 19 秒 |

本地产物 `out/2026-05-17/5-18_.../`：`captions_full.{json,jsonl,txt}`、`report.json`、
`clips/`（抽样的 clip 包）、`review_..._8clips.mp4`（8 clip / 240 帧 / 73.8 MB，
均匀覆盖 21:30→22:05）；整天合并 `out/2026-05-17/day_2026-05-17.{json,jsonl,txt}`。

注意：本机主盘紧张（跑完剩 5.7 GB）。`out/` 只占 160 MB，大头是 HF 缓存
（xet chunk ~626 MB + pov-captions 快照 62 MB），可清但按用户规则不自动删。

### 下一步可做

- 跑更多天：`run --day <date>` 一次提交一天全部录制，`merge --day` 合成整天
- 226 分钟的 4-15（无眼动）：`has_gaze=false` 时会自动不画圈，但 prompt v6 的
  Rule 6/11 是围绕注视圈写的，要跑得先出一版去掉 gaze 规则的 `prompts/v6_nogaze.txt`
- 升 v7 两遍 crop（见上一节）

## 2026-09-03 — v9 双流程接上 Vertex Batch（`batch_worker.py`）

跑 4-15 起 52.7 小时（24 条录制 / 3162 分钟 / 618 GB / ~6325 clip）的正式 caption。
`cloud_worker.py`（在线）之外新增 `batch_worker.py`（Batch），caption 逻辑不重写，
prompt / build_context / align_speech / parse_json / encode_img 全部 import 自
`caption_v9.py`（它锁在 `quality_verify/captions/frozen/` 的 SHA256 冻结快照上）。

### 为什么帧走 gs:// URI 而不是 inline base64
**Batch 单请求 payload 上限 10 MB**（官方文档），v9 pass1 是 2880px q80 × 30 帧 =
33–44 MB，inline 这条路堵死。改用 `fileData` + `gs://`（Vertex 单文件上限 30 MB），
token 计数完全相同。已用 1 行 1 请求的预检验证：`SUCCEEDED`，`in=2216` ≈ 2 帧 × 1120，
说明**逐 part 的 `mediaResolution: HIGH` 确实生效**。

线格式用 SDK 的 `types.Part.from_uri(...).to_json_dict()` 生成再转 camelCase，
不手写 —— `to_json_dict()` 出的是 snake_case，protobuf JSON 两种都收，但 camelCase
是 `bench2_worker` 已实测跑通的形状。

### 四个必须知道的坑（都是这次实测踩出来的）

1. **tar 有两种布局，别按顺序定位段**。24 条里 16 条是文档记的布局 A，8 条是布局 B
   （`picture/ocr_text` 正序在最前 → `picture/masked` → `audio/` 在靠后 → `gaze.csv`）。
   `locate_sections()` 在 B 上直接抛「找不到 audio/anonymized.wav」，
   `read_tail_adaptive()` 在 B 上读到 `ocr=0 帧`。
   → 新增 `remote_tar.walk_headers()`：只读 header 不读数据地扫全表（64KB 窗口），
   `batch_worker.index_tar()` 按名字认段。已对拍 `tarfile` 验证名字/大小/偏移全精确。
   注意「audio 无」的初判多半是假阴性 —— 是定位函数找错了地方，不是真没有。

2. **帧字节必须顺序读，不能按偏移随机读**。为了布局无关我一度改成随机读，
   功能对但丢了 `SeqStream` 的 8MB 预取：顺序 209 MB/s → 随机 ~6 MB/s。
   解包从 10 帧/s 掉到 0.2–0.9 帧/s，**4 条大录制被看门狗当挂死杀掉**。
   → 分类用索引（布局无关），搬字节用 `iter_members` 顺序预取（快）。
   修复后实测稳定 **13 帧/s 不衰减**（26,307 帧那条从 3min 到 12min 都是 13.2 帧/s）。
   另外遍历起点必须是 0，不能用索引里的 `data_offset` —— 那是**数据**偏移，
   header 在它前 512 字节，从数据中间起步会吞掉第一帧、连累整个 clip
   （本地合成 tar 实测 95→94）。

3. **`FileSource.read` 必须线程安全**。原来是共享句柄 `seek()+read()`，
   24 线程并发时交错读到错位字节，表现为 PIL `cannot identify image file`，
   786 帧坏 18 帧、连累 8/27 个 clip 作废。→ 改 `os.pread`（原子）。
   回归：合成文件 + 已知 sha256，1/24/64 线程并发读全部一致；旧写法 120 个 member
   的小测试就能复现损坏。

4. **transcript 有两种时间格式**。A 是 `[YYYY-MM-DD HH:MM:SS HKT -> ... HKT]`，
   B 是 `[00:00:00-00:00:30]`（相对偏移，无日期无时区）；时区标记还有 `BJ`。
   只认一种会让整条 transcript **静默丢成 0 区间**。`caption_core.parse_transcript()`
   现在两种都认。`TR_RE` 的时区放宽为 `[A-Z]{2,4}`，`FRAME_RE` 同步。

### Batch 的真实周转：单批 5 分钟 ≠ 全量并发时 5 分钟
冒烟（19 clip，独占）pass1 5.1 分钟 / pass2 4.6 分钟。但全量按「一条录制一个 batch」
拆成 24 个并发提交后，**20 个 batch 同时 RUNNING，最久排到 91 分钟**。
同一个 project 下多个 batch 会互相排队 —— Batch 绕开的是**在线 DSQ 降速**，
不是批次之间的容量竞争。Google 官方建议是「合成一个大 batch 比拆成很多小的吞吐好」，
按录制拆分换来的是故障隔离和流水线化，代价就是排队。**冒烟的周转数字不能外推到全量。**

### 成本（实测）
冒烟单价 **$0.0371/clip**（in 63,620 / out 6,974 每 clip，crop 10 张/clip），
比按 6 张 crop 的预估高 26%，差异主要在 output（thinking 占大头）。
真实素材混合后降到 **$0.0360/clip** → 全量约 $228，占 $300 的 76%。

### HF Jobs 余额会静默砍掉全部 job
第一次提交 24 个 job 后约 10 分钟全部 `CANCELED`。不是被别人砍的 ——
`hf jobs run` 一个最便宜的探针 job 会直接返回
`Pre-paid credit balance is insufficient`。
诱因是我给 5 条长录制切了 `cpu-performance`（$1.90/h，是 `cpu-upgrade` 的 63 倍），
5 条并跑就是 ~$9.5/h。**`--big-flavor` 默认已改回 `cpu-upgrade`**，要用大机型
必须先确认余额。全部走 `cpu-upgrade` 时整个全量只要约 20 job-小时 ≈ $0.6。

### 仓库是跟其它任务共用的，交付前必须审计
`captions/<rec>/` 下会混进别的流水线写的 clip：实测有另一会话写入的 v6 单遍 clip，
以及 v9 **在线**版撞 `429 RESOURCE_EXHAUSTED` 失败的 clip。worker 会把它们当
`done_caps` 跳过，坏 clip 就永久留在交付里。
→ `pipeline.py audit` 按 `mode` 和 `ok` 查出「异版/失败」clip；
`pipeline.py repair` 删掉它们并重跑对应录制（只补这几个 clip，不重算整条）。
`pipeline.py cost` 也已按本批窗口过滤，否则预算闸门会被别的任务的花销污染。

### 新增命令
```bash
python pipeline.py plan-batch --from 2026-04-15 --hours 52.7   # 选片 + 成本预估
python pipeline.py run-batch  --from 2026-04-15 --hours 52.7 --run-id <id>
python pipeline.py cost       # 本批窗口真实花销 vs 预算（--all-repo 看全仓库）
python pipeline.py audit      # 成功率 / 版本混杂 / PII / 缺口
python pipeline.py repair --dry-run   # 列出异版/失败 clip；去掉 --dry-run 才动手
```

### PII：v9 也会漏，公开前必须过一道正则脱敏
按录制拆解实测：`4-27_hkt1817`（v9）17 段里 `school_plain` 14 处（明文 `HKU`/`香港大学`），
`4-25_hkt1843`（v9）281 段里 `phone_plain` 13 处（`2605 3928` 这类真实香港 8 位电话）。
**不是 v6 才漏，v9 同样漏** —— 印证了吞吐报告里那条「Rule 8 匿名不达标」的警告。
是文本层面的问题，caption 出来后正则重刷即可，不必重跑模型；但**过这道处理之前不能公开**。

## 2026-09-04 — 双 ADC 并行跑前 100 小时 caption

**目标**：`mmm8383/pov-data` 从 2026-04-15 起前 100 小时素材出 v9 双遍 caption，
两个 GCP 项目并行、各自 $250 闸门、5–6 小时内跑完。

**波次划分**（按日期切，互不重叠）

| 波 | 项目 | 凭据 | 桶 | run_id | 范围 |
|---|---|---|---|---|---|
| A | `project-1a1608e8-53ad-4178-bf7`（项目号 988763814276） | `~/.config/gcloud/adc-profiles/b34927743.json` | `hku-capstone-caption-frames-bf7-20260904` | `20260904-bf7-r3` | 4-15→4-30，24 条 / 52.7h |
| B | `project-0e21c343-2d89-405c-9d5`（项目号 812215571837） | `~/.config/gcloud/adc-profiles/meng15210579725.json` | `hku-capstone-caption-frames` | `20260904-0e21-w2` | 5-03→5-11，35 条 / 47.9h |

合计 59 条 / 100.6h / 约 12075 clip。

**切 ADC 必须三个一起切**（2026-09-04 血的教训）

r2 那 16 个 job 全挂在 `403 PERMISSION_DENIED / aiplatform.batchPredictionJobs.create`：
只换了凭据文件和 GCS 桶，`ADC_PROJECT` 没换，于是拿 bf7 的凭据去 0e21 项目建 batch。
`caption_core.ADC_PROJECT` 是 import 时从 `ADC_PROJECT`/`GOOGLE_CLOUD_PROJECT` 读的，
默认值写死 0e21，不显式传就一定串项目。正确姿势：

```bash
ADC_FILE=~/.config/gcloud/adc-profiles/<账号>.json \
python pipeline.py run-batch --from 2026-05-03 --hours 47.9 \
  --run-id <run_id> --no-push-code \
  --env ADC_PROJECT=<项目 ID> --env GCS_BUCKET=<该项目下的桶>
```

`ADC_PATH` 已改成可被 `ADC_FILE` 环境变量覆盖（原来写死默认凭据文件）。

**`--skip-done` 原来是错的**

`done_recordings()` 只看 `captions/<rec>/report.json` 在不在。4-24/4-25/4-27 有 5 条录制
早年被 `--max-clips 3` 冒烟跑过，report.json 早就存在 → run-batch 把它们整条永久跳过，
287 分钟素材一个 clip 都没真跑。新增 `done_recordings_covered(picked, clip_seconds)`
按「ok 的 clip 数 ≥ 应有数量 × 90%」判定，`cmd_run_batch` 已切过去。
补发的 4 条：`4-24_hkt1012` / `4-25_hkt1942` / `4-27_hkt1819` / `4-27_hkt1944`。

**真实单价（66 个 v9_2pass_batch clip 实测，不再用外推值）**

- $/clip 均值 **0.0331**、中位 0.0329、p90 0.0480、最大 0.0547
- batch_in 均 54,968 tok / batch_out 均 5,955 tok / crop 均 6.7 个（最大 14）
- 外推 12075 clip ≈ **$399**，分到两波 A≈$209 / B≈$190，都在 $250 以下

**周转实测**：pass1 batch 4–10 分钟、pass2 batch 5–6 分钟。真正的瓶颈是解包+上传
（13–18 MB/s/job，271 clip 的录制要 833s），最长的 4-26（438m/80GB）约 90 分钟。

**监控**：`watch_dual.py` + `_watch/tick.sh`，15 分钟一轮。
从 HF 上新增的 `clip_*.json` 增量累加真实 token 花销（缓存在 `_watch/cost_state.json`），
按录制日期归波次分别计费；任一波触到 `--cap` 只停那一波的 HF job 和**本 run_id 的**
Vertex batch（靠 dest 前缀过滤，不碰同项目里别的任务的作业）。
手动停某一波：`python3 watch_dual.py --stop-wave A`。

**遗留**

- bf7 的桶**没有生命周期规则**（0e21 那个有 7 天清 `batch_in/`/`batch_out/`/`p1/`/`p2/`）。
  A 波约 618 GB 中转帧会一直存着，$0.02/GB/月。加规则属删除操作，等用户授权。
- batch 被 CANCELLED 时 `predictions.jsonl` 里已完成（且已计费）的行会被丢弃重发。
  `_batch_is_terminal()` 只保证不丢数据，没做部分回收。跑批中途不动 worker 代码。
- PII：prompt Rule 8 匿名化不达标，跑完要对文本做一遍正则脱敏后处理，不用重跑模型。

### 看门狗在 pass0 误杀大录制（2026-09-04，本批最大的一个坑）

**症状**：55 个 job 并跑时，**小录制（≤130 clip）一路顺利走到 pass1/pass2，所有
≥160 clip 的大录制全部 ERROR**，日志最后一行清一色是

```
!! 看门狗：25.3 分钟无进展（最后一步：解包上传 10100/10139 帧），主动退出
```

一小时内死了 28 条，占窗口的一半。

**根因不是上传，是 pass0**。注意那句「最后一步」记的还是**上传**阶段的打点——
说明进 pass0 之后一次点都没打过。`run_speech()` 原来写的是 `if i % 50 == 0: beat(...)`，
338 clip 的 4-17 在高并发下连第一个 50 都没跑完，看门狗就按「25 分钟无进展」把
整条录制杀了。pass0 走**在线** API，几十个 job 一起打会被 DSQ 拖慢，
`v9.generate()` 的 429 退避是 20/40/60s，单个 clip 拖几分钟很正常。

**修法（两条都必要）**

1. `run_speech()` 改成**每完成一个 clip 就 `beat()`**。`beat()` 只是更新一个时间戳，
   每轮调没有代价；「慢但在推进」本来就不该被判成挂死。
2. 加 `SPEECH_DEADLINE_MIN`（默认 20，本批重跑用 15）。超时只 `cancel()` **还没开跑**的
   future，已在飞的让它自然结束——强行 `shutdown(cancel_futures=True)` 会让解释器退出时
   join 卡住。放弃的 clip 没有 speech 块，`build_context` 本来就有「只用 transcript」的分支。

**重跑不该重传**：这批死的时候帧早就 100% 传上 GCS 了（日志都是 N/N 帧、没有 dropped clip），
但 `unpack(do_upload=not resume_p1)` 只在**已提交 pass1** 时才复用，死在 pass0 的一律重传。
新增状态位 `frames_uploaded`（整批传完时存下 clip 列表），续跑时按同一模板推 URI，
4-15 那条从 21 分钟上传直接降到 0。**只存「整批成功」的那份列表**：`unpack` 会把有帧
没传上去的 clip 整个丢掉，存这份才能保证重建出来的 gs:// URI 个个真实存在。

**捡回掉队录制**：`_watch/repair2.py`。**不要读 job 日志**去判断上传是否完整
（22 条逐个 `hf jobs logs` 把本机内存吃爆，后台任务被系统杀过一次）；直接列 GCS：
帧是 1 fps、clip 30 秒，**一个 clip 目录正好 30 个 blob 才算完整**，末尾 clip 允许不足。
三道去重：覆盖率已达标的 / 还有 job 在跑的 / 30 分钟冷却名单内的都跳过。
`_watch/autorepair2.sh` 是它的 15 分钟循环。

**Codex 的 supervisor**：`/private/tmp/caption_supervisor_bf7.py` 只轮询 + $270 闸门 +
终态后 `pull_all()`，**不会重投 job**，和 repair2 不冲突。它只盯最初那 17 个 job id，
重跑的新 id 不在它视野里。

### Vertex Batch 周转不可控：共享池排队（2026-09-04 实测）

官方文档（[batch inference with Gemini]）明确：Gemini batch **没有按项目的配额**，
用的是**所有客户共享的资源池**，按实时供需动态分配；容量饱和时请求排队，
队列最长 **72 小时**，SLA 是「**开始运行后 24 小时内**完成大多数作业」，
超过 24 小时未完成的会被取消、只对已完成的请求计费。单个 batch 最多 20 万条请求。

本批实测的退化曲线（同一个 run，请求规模相近）：

| 提交时间 | pass1/pass2 周转 |
|---|---|
| 14:39–14:51 | 4–34 分钟（早期，池子空） |
| 14:52–15:13 | 60–80 分钟 |
| bf7 14:32 起 | 90–99 分钟 |

**两个项目同步退化**，所以不是「新项目配额低」，就是共享池排队。
**别把「batch 几分钟就回来」当成可依赖的前提去排期**——那是池子空的时候的运气。
按 SLA 排期要留到小时级；要卡死几小时内出结果，batch 给不了保证。

推论：往同一个池子里堆更多并发 batch job 并不会更快（池子是全球共享的），
拆小批次也没用。真正的杠杆只有「早点提交」和「减少串行轮数」——
v9 是 pass1 → crop → pass2 两轮**串行** batch，等于要吃两次排队。

**连带风险**：`BATCH_TIMEOUT_MIN`（默认 240 分钟）到点会 `batches.cancel()` 止损，
但**被 cancel 的 batch 里已完成的行是已经计费的**，重跑那部分等于重复付钱
（`_batch_is_terminal()` 只保证不丢数据，没做部分回收）。周转膨胀时这个超时容易被误触发，
排期紧的时候要么调大它，要么接受这笔重复计费。

### 别按分钟估 clip 数：有录制不是 1fps（2026-09-04）

`plan-batch` 一直用「时长 × 60 ÷ clip_seconds」估 clip 数和成本，前提是**所有素材 1fps**。
`5-09_hkt2126-2155_29m_daf330ff-93d8-5ba8-a55b-92bcc2f16f5f` 不是：29 分钟解出
**52886 帧**（~30fps）。`clip_id = frame_index // 30` 把它切成 **1763 个 clip**，
每个 clip 的 30 帧只覆盖**真实 1 秒**——发给模型的 30 张图几乎完全相同。

- 语义错：标称 30 秒的窗口实际 1 秒，caption 没有意义
- 成本错：估 58 个 clip、实际 1763 个，**低估 30 倍**；按 $0.0331/clip 是 **$58**，
  占 100h 批次预算的 15%，标的却只有 29 分钟素材
- 还把 1763 个请求塞进本就拥堵的共享 batch 队列

**筛法**：算 `size/minutes`。全窗口 58 条的中位数是 **0.200 GB/分钟**，
这条是 **1.08**（5.4 倍），是唯一的离群点。已加进 `_watch/repair2.py` 的 `EXCLUDE`。
以后开新批次前先跑一遍这个离群检查，别等花完钱才发现。
（这条也是 12 条无 MPS 眼动录制之一，见 [[reference_aria_frame_bj_clock_skew]] 记的 v5 确定性 UUID。）

## 2026-09-05 12:30 HKT — 两个 ADC 合力先跑前 50 小时

**目标变更**：用户要求「先用两个 adc 账号合力把前 50 小时跑出来，然后再往后；
并发合理即可（不要过多 429），今天跑完」。原来的分工是 A(bf7)=4 月 / B(0e21)=5 月，
这条日期线作废。

**开工时实况**（04:25 UTC 实测）
- 前 50h = 24 条录制 / 3162 分钟 / 6348 clip，HF 上只落了 144 个 clip = **2.3%**。
  之前 watch 里看到的 472 个 clip 全是 5 月那批（在前 50h 之外）。
- 花销 bf7 $3.73 / 0e21 $5.24，合计 $9 —— **钱完全不是瓶颈，Vertex batch 的服务速率才是**。

**在线通路第三次证伪**：拿真实的 pass1 请求（30 帧 gs:// fileData @HIGH）
并发 8 打 16 个，1.3 秒内 **16/16 全是 429 RESOURCE_EXHAUSTED**。
在线不是 batch 的退路，别再试。探针脚本 `/tmp/probe_online.py` 的写法可以复用：
直接从 GCS 上已有的 `*_pass1.jsonl` 取 request 原样重放，不用另造 payload。

**发现 1：有 4 个无人认领的重复 batch 在白烧队列位**
bf7 上存在 `20260904-bf7-r4` 这一套 batch（4-15/4-17/4-20/4-21），创建 7 小时
**产出 0 行**，而同样这 4 条录制在 `-r3` 下另有一套正在出结果。r4 的 HF job 早已
消失，没人轮询它。已全部 cancel。
→ 教训：换 run_id 重跑前先查旧 run_id 下有没有 batch 还活着，孤儿 batch 不会自己死。

**发现 2：新提交的 batch 被服务的速度比老的高一个数量级**
| batch | 提交时间 | 已完成/总行 | 折算 |
|---|---|---|---|
| 4-21 (654 行) | 03:33 | 157 | ~175 行/h |
| 4-17 (338 行) | 03:30 | 74 | ~82 行/h |
| 4-27_1602 (97 行) | 18:46 | 96 | ~10 行/h |
| 4-22_1412 (205 行) | 18:49 | 181 | ~19 行/h |
老 batch 像是排在队列后面出不来。**推论（待 20 分钟对拍验证）：与其干等老 batch，
不如 cancel 掉重发换一个新队列位**——`harvest()` 会把已计费的行捞回来，只重发缺的，
不会重复付钱。

**动作：把 8 条录制从 bf7 搬到 0e21**
挑的是 bf7 上「已完成行 ≤5」的（抛弃的已计费行合计 17 行，可忽略）：
4-15 / 4-22_0821 / 4-30 / 4-23_1900 / 4-24 / 4-25_1942 / 4-27_1819 / 4-27_1944，
共 2166 clip ≈ 前 50h 的 34%。新 run_id `20260905-0e21-f50`，桶用 0e21 的
`hku-capstone-caption-frames`，帧在 0e21 侧重传（只花时间不花 token）。
脚本 `_watch/move_to_0e21.py`。
- **必须连旧的 HF job 一起 cancel**：只 cancel batch 的话，旧 job 会看到
  batch 变 CANCELLED（`_batch_is_terminal` 认它是终止态）→ 在 bf7 上重新提交一遍
  = 重复付费。本次 `move_to_0e21.py` 里按名字取消 HF job 没匹配上（JobInfo 没有
  `name` 属性），是手动按 job id 补的 8 次 `hf jobs cancel`。
- 分完之后：bf7 留 11 条前 50h，0e21 = 8 条前 50h + 26 条 5 月。

**pass0 并发下调**：`--speech-workers 8 → 3`、`SPEECH_DEADLINE_MIN 15 → 8`。
在线语音对齐并发 8 已经硬 429，拿不到就退 transcript，别让 pass0 卡住真正值钱的 pass1。
`_watch/repair2.py` 与 `move_to_0e21.py` 都已改。

**波次归属改成显式名单**：`watch_dual.py` 和 `_watch/repair2.py` 里各加了一份
`MOVED_TO_B`（8 条），`wave_of()` 先查名单再看日期。不改的话这 8 条的花销会记到
A 头上、两边的 $250 闸门都算错，`repair2.py` 也会把它们塞回 bf7。
`repair2.py` 的窗口同时从 100.6h 收到 50h：5 月那批中途死掉本轮不自动捡。

**旁路发现**：0e21 项目里有两个不属于本流水线的 batch
（`gs://hku-capstone-caption-frames/rokid-caption-batch/phone/outputs/…`，
03:10 与 03:16 创建）。不是 caption 流水线建的，没动它，但它们同样占着这个项目的额度。

## 2026-09-05 14:00 HKT — 根因：3.7-flash 对 credit 项目零额度，改混跑 3.6/3.5

**这是今天所有"排队慢/停摆"现象的唯一根因，不是队列、不是并发、不是区域。**

### 先修一个会导致误判的测量 bug
`out_pass1/` 下每次 batch 尝试都会新建一个 `prediction-model-<ISO时间戳>/` 子目录，
**旧的不会清掉**。把整个 dest 一锅端数行数，会把上一次已取消 batch 的结果当成当前进度。
我因此得出过两个错误结论（"新 batch 服务快""全线 0 行/h"）。
正确姿势：只数**时间戳与该 batch 的 create_time 相差 10 分钟以内**的那个子目录。
按此重测，当时 38 个 RUNNING batch **本次产出全部是 0 行**，最老的已 10 小时。

### 根因证据（同一把凭据、同一个项目，只换模型）
| 模型 | Vertex 在线 | Vertex batch（5 行探针） |
|---|---|---|
| gemini-3.7-flash | **429**（连 `say hi` 5 token 都过不去） | **25 分钟 0/5**（两个项目都是） |
| gemini-3.6-flash | OK 7.5s | **5.6 分钟 5/5** |
| gemini-3.5-flash | OK 1.2s | **6.3 分钟 5/5** |
| gemini-2.5-flash | OK 0.8s | — |
| gemini-3.5-flash-lite | OK 0.9s | — |
3.7-pro / 3-flash / 2.0-flash / 3.5-pro / 3.7-flash-lite 都是 **404 模型不存在**。

其他已排除的路：
- **换区域没用**：`gemini-3.7-flash` 只在 `location="global"` 存在，us-central1 报 404。
- **换 key 没用**：用户给的 Gemini Developer API key 对 3.7 和 3.5 都是 429（免费层耗尽）；
  google-cloud key（AIza…）是 403 无权限。
- **不是 Google 事故**：status.cloud.google.com 当日无 Vertex 相关事件。
- 官方口径：Gemini batch 无每项目配额，是**跨全体客户的共享池按实时供需动态分配**。
  免费额度项目在 3.7-flash 这种新模型上被分到 0，就是这个机制的结果。

### 用户决策与当前部署
用户定：「三个模型混着跑，能用哪个就用哪个。3.7-flash, 3.6-flash, 3.5-flash」。
- 前 50h 的 19 条待跑录制：**10 条 gemini-3.6-flash + 9 条 gemini-3.5-flash 交替分配**
  （一个模型的池子塌了只影响一半）。分配表落盘 `_watch/model_assign.json`，
  `_watch/repair2.py` 读它，**不在表里的一律退到 3.5-flash**。
- 切换脚本 `_watch/switch_model.py`（`--models` / `--stride` / `--offset` / `--apply`）：
  **沿用原 run_id**，所以 GCS 上已传的帧原样复用（最大那条 20.6 GB，重传要 12 分钟）；
  先 cancel 旧 batch + 旧 HF job，worker 续跑时看到 CANCELLED → harvest → 用新模型只发缺的 clip。
- 5 月那批（0e21 上 26 条）的 3.7 batch **原样留着排队**，不占额外的钱；
  万一 Google 放量就是白捡的 3.7 数据。
- 每个 clip 的记录里本来就有 `model` 字段（`batch_worker.py:867`），混跑后可按模型区分。

### 两个操作教训
1. **切模型必须连旧 HF job 一起 cancel**。只 cancel batch 的话，旧 job 还在轮询；
   而只 cancel job 不 cancel batch 的话，新 job 续跑会看到 `pass1` 还是 RUNNING、
   直接接回旧模型的 batch，**模型切换变成空操作**。
2. **手动提交后要往 `_watch/repaired.log` 写冷却记录**。自动捡回循环只看
   `hf jobs ps` 的在跑集合，我手动 cancel 完到新 job 起来之间有个窗口，
   循环正好在这个窗口 tick 就会重复起一个 job，两个 job 抢同一份状态 → 重复提交 batch。
   本次 4-30 就被起了两个，已砍掉一个。

## 2026-09-05 16:00 HKT — batch 周转不可预测，给 worker 加了在线模式

### batch 周转实测（同一份真实 pass1 请求，只改行数/模型）
| 探针 | 结果 |
|---|---|
| 3.6-flash 5 行 | **5.6 分钟 SUCCEEDED** |
| 3.5-flash 5 行 | **6.3 分钟 SUCCEEDED** |
| 3.5-flash 30 行 | 23 分钟仍 0 产出 |
| 3.7-flash 5 行 | **101 分钟仍 0 产出** |
| 19 条真实录制（100–900 行） | 提交 60 分钟后全部 0 产出 |
周转不随行数线性增长，更像在等调度窗口。**batch 无法保证当天交付。**

在线则稳定：3.5-flash 并发 8、真实 30 帧 payload，15/16 成功、**零 429**、
p50 25 秒、**894 clip/h（单项目）**。

### 改动：`batch_worker.py` 新增 `online_run()`
- `--env ONLINE=1` 时，pass1/pass2 不提交 batch，改用线程池
  （`ONLINE_WORKERS`，默认 12）把**同一份 JSONL 逐行重放到在线接口**。
  请求体逐字复用 `write_pass*_jsonl` 的产物，所以两条路发出去的 payload 完全相同
  （同样的 gs:// fileData、mediaResolution、generationConfig），返回结构也和
  `collect()` 一模一样（复用 `row_payload` + `row_clip_id`），下游零改动。
- 退避沿用 v9 本地版：429 走 20/40/60s 慢退避，掉线走 4/8/12s 快重试，最多 6 次。
- 计费按实际通路选价：`lane = "online" if c["online"] else "batch"`；
  clip 记录里的 `mode` 相应写成 `v9_2pass_online`。**在线是 batch 的两倍价**
  （$0.75/$3.75 vs $0.375/$1.875），前 50h 全走在线约 $340，仍在两个 $250 内。
- 备份 `scratchpad/batch_worker.py.bak3`。

### 顺带清理
- **5 月那批（26 条）的 3.7 batch 全部取消 + HF job 取消**：它们十几小时零产出，
  却占着 0e21 的并发槽位，而前 50h 有 8 条录制在同一个项目排队。零产出=零计费，
  帧和状态都还在 GCS/HF，前 50h 跑完原样续跑即可。取消后 0e21 只剩 8 个 batch。
- `watch_dual.py` 的 `wave_of()` 改成**前缀匹配**：`hf_jobs()` 拿到的 rec 是从
  job 名 `povcap-<rec[:28]>` 反解的 28 字符前缀，和 `MOVED_TO_B` 里的全名对不上，
  导致搬到 B 的 8 条被全记到 A 波（显示 job A19/B0）。
- `_watch/batchgrow.py` 的停滞门槛 25 → 75 分钟：Vertex 会**先把输入行的骨架写出来**
  （见过 `0/252` 这种），再逐步回填响应，25 分钟判停滞会误报。
- 监控加密到三层：`tick3.sh` 5 分钟（进度+预算闸门）、`batchgrow.py` 10 分钟
  （只 list 对象字节数、不下载 jsonl，所以能高频）、`repair2.py` 8 分钟（自动捡回）。
  **老的 `tick2.sh` 不能覆写**（正在被 bash 执行），所以另起了 `tick3.sh`。

### 质量验证：换模型不破坏下游
3.5-flash 的探针输出 5/5 可被 `v9.parse_json` 解析，segment 结构、`environment` 的
PHYSICAL SPACE / OBJECTS / SCREEN CONTENT 三段式都对，细节密度在线
（认出 USB 风扇插在左侧 USB-C 口、终端窗口尺寸 210x54）。

## 2026-09-05 深夜 — 转向 v10 单遍在线，加模型降级阶梯

**用户决策**：「用 v10 单遍在线跑通，监督，随时 debug，直到完成 50 小时」。
在此之前先停掉了所有 batch job 与四个监控循环，并把 HF 上的 caption 全量拉到本地。

### 关键事实：在线 429 是「逐模型 + 随时间漂移」的
同一把凭据、同一个项目，只换模型就通/不通，且几小时内会翻转：
| 时刻 | bf7 | 0e21 |
|---|---|---|
| 13:00 | 3.7 挂，3.6/3.5 通 | 同左 |
| 16:00 | 3.7/3.6/3.5 **全挂**，只剩 lite | 只剩 3.5 通 |
所以**任何写死单模型的跑法都会周期性整条录制全废**。实测 4-21：654 个 clip
全部 429，白解包 71 GB / 804 秒，零产出。

### 修复 1：`caption_core.run_one()` 加模型降级阶梯（已推 `_code/`）
- `MODEL_LADDER` 环境变量，默认 `3.7-flash,3.6-flash,3.5-flash,3.5-flash-lite`。
- **429/404 时立刻换下一个模型，不睡**；整条阶梯都不可用才退避重来。
- 真正用上的模型写回 `result["model"]`，另记 `fallback_from`，每个 clip 来源可查。
- 实测：bf7 一路降到 lite（1.9s 成功），0e21 降到 3.5-flash（2.3s）。

### 修复 2：重试深度 3→7 轮、退避上限 60→240s（已推 `_code/`）
429 是**分钟级**漂移窗口，原来 3 轮最多等 15 秒就判死，是 96% 失败率的直接原因。

### 修复 3：`pipeline.py` 加 `--no-skip-done`
`--skip-done` 原本 `default=True` 且**没有关闭开关**，想对已有 report.json 的录制
用 `--only-failed` 补跑会被整条跳过。

### 并发实测（带阶梯，真实 30 帧 payload，0e21）
| 并发 | 成功率 | 吞吐 |
|---|---|---|
| 4 | 70% | 265 clip/h |
| 12 | **75%** | **895 clip/h** |
高并发**没有**恶化成功率 —— 因为瓶颈是共享池的额度漂移，不是我们打得太猛。
按单项目 895 clip/h × 2 个项目算，6348 个 clip 约 3.5 小时。

### 补跑用 ONLY_FAILED，不要用 repair
`pipeline.py repair` 会**删** HF 上的 clip，且它的「异版」判定是以 v9 为目标的
（`mode != v9_2pass_batch and pipeline != v9-2pass` 就算异版），演练显示会删掉
3039 个 clip。改用 worker 自带的 `ONLY_FAILED=1`：只重跑 `ok=false` 的 clip，
**不删任何东西**，已成功的 645 个（含 526 个 v9 两遍）原样保留。

### 两个自己犯的统计错误（都已更正）
1. 把「已落盘 clip 文件数」当成「成功数」，据此得出「4-20 已 242/242 完成」
   以及后来「Codex 覆盖了好数据」的结论 —— **都不成立**。判据必须是 `ok` 字段。
2. 下午说「batch 健康约 3000 clip/h」，那个数把落盘的**失败**记录也算进去了。
真实构成（本地 3655 个 clip）：成功 645，失败 3010，其中 **3008 个是 429**，
失败横跨 3.5-flash(2074) / 3.7-flash(653) / 3.6-flash(283)。
按通路：v9 两遍 batch **326/327**、v9 两遍 **200/200**、v10 单遍在线 **117/3126**。

### 本地副本
`out/_delivery_20260905/` —— 112 条录制 / 3655 个 clip / 19 MB，
每条录制一个 `captions.jsonl`（完整记录）+ `captions.txt`（可读）+ 根 `index.json`。
另有 `out/_public_20260905/` 是脱敏版（2260 clip，用户后来说不需要脱敏，留档备用）。
脱敏工具 `deid.py`：实测清掉 3474 处 HKU/HKUL、69 处中文姓名、14 个手机号、
9 个邮箱、10 个 github handle、36 处本机路径；公开论文/课程网站/公众人物**故意保留**
（用户定的原则：只脱结构化身份标识符，工作与浏览内容保持可读）。

## 2026-09-05 深夜（续）— v10 补跑跑通，监控改成零下载

### 验证跑结果（4-27_1730，91 clip）
**89 ✓ / 2 ✗ = 97.8%**，caption 阶段 143 秒（≈2290 clip/h），整 job 277 秒含解包。
$2.5849 / 90 clip = **$0.0287/clip**。修复（模型阶梯 + 重试加深）确认生效。

### 启动 v10 的四个坑（`_watch/launch_v10.py` 已固化）
1. `--tar` 要**仓库内 tar 路径** `aria/<日期>/<录制>.tar`，传录制名会 404。
2. `--skip-done` 原本 `default=True` 且**没有关闭开关** → 已加 `--no-skip-done`，
   否则对有 report.json 的录制做 `--only-failed` 补跑会被整条跳过、变成空操作。
3. **`run` 子命令不接受 `--env`**（只有 `run-batch` 有），传了直接
   "unrecognized arguments" 退出。ADC_PROJECT 由 pipeline.py 从 ADC JSON 的
   `quota_project_id` 自动带上，RUN_RETRIES 在 caption_core 里已默认 7，都不用传。
4. 提交脚本的输出过滤器要能捕获失败，否则 8 个 job 一个没起来还以为起了。

### 监控自己是限流元凶 → 改成**文件大小判据**，零下载
逐个下载几千个 clip json 判 `ok`，把 HF 打到 429（HEAD 都被限，重试等 38–73 秒），
还和 job 自己的 commit 抢那 128 次/小时的配额。
改用大小判据：**429 失败记录中位 778 字节（最大 13305），成功记录最小 7881 字节**。
拿 2260 个已知真值验证：**阈值 6000 时 56/56 真成功全部命中、0 漏判、4 个误判（0.2%）**。
大小由 `list_repo_tree` 免费带出，一条录制一次请求 —— 全量一轮从「几分钟 + 限流」
降到 **8 秒 / 零下载**。`_watch/rate_f50.py` 与 `launch_v10.needs()` 都已改。

### 当前部署
- `_watch/tick_v10.sh` 7 分钟一轮：可用 clip 数 / 速率 / ETA / 已完成录制数 /
  新增 ERROR（按 job id 去重）。
- 自动补位循环 15 分钟一轮：在跑 job < 6 就补到 8，缺口大的优先，两个项目交替。
- **不要覆写 `tick2.sh`/`tick3.sh`**（历史版本，可能仍被 bash 执行）。

### 待办
用户称两个账号各剩约 $100，而按 token 统计只花了 $9，差额可能来自
**已计费但未收回的 batch 行**（今天累计取消过约 100 个 batch）。需要用户核对账单页，
这决定剩下约 6000 个 clip（预估 $171）够不够跑完。

## 2026-09-05 深夜（三）— 连锁故障：阶梯加错模块 → 打爆 HF 提交配额

### 故障链（环环相扣，单看每一环都像别的问题）
1. **模型降级阶梯加错了模块**。我把它加在 `caption_core.run_one`，但 **v10 单遍走的是
   `caption_onepass.process_clip`** —— 那里直接 `client.models.generate_content(...)`，
   try/except 里任何异常就 `ok=False`，**无重试无阶梯**。所以阶梯对当前跑法完全没生效，
   日志里 `in=None out=None 0.5s` 就是四个模型一个都没试、429 直接返回。
2. clip 大量失败 → **失败记录照样要写 HF**，`flush_captions` 每积压 **25** 个就提交一次。
3. 8 个 job 并发提交 → 打爆 HF 的 **128 次/小时**仓库提交配额。
4. commit 重试耗尽 → `raise RuntimeError` → **job 整个崩掉**（一次崩 5 个，最后 0 个存活）。
5. 连推代码都推不上去 —— 推送本身也是一次 commit，被同一个配额挡住。

### 三处修复（已推 `_code/`）
* `caption_onepass.process_clip` 改走 `cc.run_one`，接上阶梯与 7 轮重试。
* `run_one` 加 `max_tokens` 参数，单遍传 **32768**。不加的话会套用 caption_core 的
  `MAX_OUTPUT_TOKENS=8192`，把单遍输出上限砍到 1/4、**静默截断长 caption**。
* `flush_captions` 的提交触发阈值 **25 → 200**（`FLUSH_EVERY` 可调），提交次数降到 1/8。
  （`COMMIT_CHUNK=250` 与 429 分钟级退避此前已在线上生效。）
* 并发从 8 降到 **3**。

### 经验
* **改共享代码前先确认调用链走哪条路**。这条流水线有三个入口
  （`cloud_worker`→v6/v10、`batch_worker`→v9 两遍、`caption_onepass`→v10 单遍），
  改错模块会得到「改了但完全没效果」这种最难查的结果。
* **HF 提交配额是全局串行资源**：job 的 caption 提交、状态保存、代码推送、
  以及我自己的监控下载全都算在同一个 128 次/小时里。监控用大小判据零下载之后，
  剩下的压力主要来自 job 数 × 提交频率 —— 这两个都要压。
* `hf jobs logs` 会把整个日志读进内存，大 job 上会被系统按内存压力杀掉
  （本轮被杀过一次）。查进度优先用 `list_repo_tree` 的大小判据，别拉日志。

## 2026-09-05 — v10 单遍的真实单价与验算

`captions/<rec>/report.json` 里的 `est_cost_usd` **已经是在线价**，不要再翻倍。
以 4-21 验算：`tokens.in=12.478M × $0.75 + tokens.out_billable=1.108M × $3.75 = $13.52`
= `est_cost_usd 13.514` ✓。

**$0.031/clip**（有眼动录制；无眼动的约一半，因为不画注视圈、zoom 少一半）。
前 50h 全量 6348 个 clip ≈ **$197**。这个数比按 4-27_1730（无眼动，$0.0287）
外推的低 25%，也和按 in=40k 猜的 $0.041 不同 —— **别用单条录制外推，读 report**。

report.json 的可用字段：`clip_count / clip_ok / clip_failed / failed_clip_ids /
segment_count / tokens{in,out_billable} / est_cost_usd / model / has_gaze / pii_audit`。
汇总花销只要读这 23 个文件，不用碰 clip json。

**跑完后仍需过一遍 `deid.py`**：report 自带的 pii_audit 显示 4-21 一条就有 189 处命中
（31 明文校名 / 85 电话样式 / 71 泛化校名 / 2 邮箱）。

## 2026-09-05 — 重试深度：漂移窗口要深，持续饱和要浅（方向相反）

同一个 `RUN_RETRIES` 参数，两种场景下的最优方向**完全相反**，今晚两个方向各调了一次：

* **短暂漂移窗口**（某个模型几分钟内没额度）→ 要**深度重试**扛过去。
  3 轮最多等 15 秒就判死，是 96% 失败率的直接原因，所以调到 7 轮 / 退避上限 240s。
* **持续饱和**（所有模型都排不上）→ 要**快速失败**。实测失败的 clip 耗 **324 秒**
  （整条阶梯 × 7 轮退避）才放弃，成功的只要 15–32 秒，worker 槽位被长退避锁死。
  16 worker、成功率 35% 时：7 轮 ≈ 92 clip/h，2 轮 ≈ 1090 clip/h，**差 12 倍**。
  失败的 clip 反正会被下一轮 `ONLY_FAILED` 捡回来，深度重试纯属浪费槽位。
  → 改回 `RUN_RETRIES=2`、退避上限 15s。

**判据**：比较日志里失败 clip 与成功 clip 的耗时。一旦失败明显更慢，就说明重试在做无用功，
应该让它快速失败、把并发让给可能成功的请求，靠多轮补跑收敛。

**改参数时不要杀正在跑的 job**：`FLUSH_EVERY=200` 意味着每个 job 攒着最多 200 个
**已计费但未提交**的 clip，杀掉就是白花钱。让它自然结束，新代码由自动补位起的新 job 带上。

## 2026-09-05 深夜 — bf7 触发 Spend cap，产能减半

```
403 PERMISSION_DENIED
Spend cap breached for project: projects/988763814276
for service: aiplatform.googleapis.com
```
**bf7（project-1a1608e8-53ad-4178-bf7）已被停用**，不是限流是花完。
`_watch/launch_v10.py` 已改成**只用 0e21**，恢复 bf7 前不要改回交替。

这也解释了额度账为什么对不上：我按落盘 clip 的 token 统计出「只花了 $9」，
但真实花销远高于此 —— **大量已计费的 batch 行从来没被收回过**
（今天累计取消过约 100 个 batch，它们跑完的部分照样计费，结果却没进 HF）。
**教训：取消 batch 不等于不花钱，按落盘结果统计花销会严重低估。**

### 一个反直觉的判据：探针 429 ≠ 没额度
0e21 单发探针四个模型全 429，但同时 5 个 job 正稳定产出 1107 clip/h ——
**429 是因为容量被我们自己的 job（5×16=80 并发）吃满了**，探针抢不到槽位。
我据此误停过一次自动补位。**判断有没有额度要看 job 的实际产出速率，
不要用单发探针**；探针只在**没有 job 在跑**时才有参考价值。

### 当前状态（供续接）
- 前 50h 可用 **2473/6348 = 39%**，已完成 4/24 条录制，速率约 1150 clip/h。
- 只剩 0e21 一个项目，补位上限 5 个 job。
- 本地全量副本 `out/_delivery_20260905/`。
- 待用户确认：bf7 的 spend cap 是「$300 免费额度耗尽」还是「控制台可调上限」。
  若是后者，解除即可恢复一倍产能。

## 2026-09-05 — job 超时 1h 太短，导致反复重新解包

`pipeline.py run` 的 `--timeout` 默认 **1h**。大录制（4-26 有 877 clip、438 分钟素材）
一小时跑不完就被杀，下一轮补位又要**从头解包十几分钟**才能继续 caption。
症状是 `已完成` 长期停在 4/24 —— 不是跑不动，是每条大录制都在「解包 → 跑 45 分钟 →
被杀 → 再解包」里空转，解包开销占每轮约 25%。
`_watch/launch_v10.py` 加了 `--timeout`（默认 **5h**），让一个 job 能一次跑完整条录制。

**改这类参数时不要杀正在跑的 job**：`FLUSH_EVERY=200` 下每个 job 缓冲着最多 200 个
已计费未提交的 clip，杀掉就是白花钱；让它自然结束，新配置由下一轮补位带上。

## 2026-09-05 — 收敛机制是「多轮补跑」，优化点是轮次空档而非并发

每个 job 用 `ONLY_FAILED=1` 扫一遍某条录制，**只能把剩余失败 clip 的一部分转成功**
（撞 429 的仍然失败），然后正常 COMPLETED 退出 —— 4-20 一轮只从 0 跑到 22/241。
所以 `已完成 4/24` 长期不动不是卡住，是**大录制需要多轮才能收敛**。

由此推论：**吞吐的决定因素是「单位时间能跑多少轮」，不是单 job 多快。**
- 补位间隔 15 分钟、job 9 分钟就退 → 每轮空转 6 分钟。已把补位改成 **5 分钟一轮**。
- **加并发不是解法**：并发越高 429 越多、每轮转化率越低，总量未必涨。
  实测 5-6 个 job 时约 1100 clip/h 是个合理工作点。

**判断 job 是否健康别只看存活数**：正常 COMPLETED 且很快退出是这个机制下的预期行为，
不代表出错；真正要报警的是 ERROR 状态和「一轮下来一个 clip 都没转成功」。

## 2026-09-05 收盘 — 两个项目额度全部耗尽，停在 48.3%

```
bf7  : 403 Spend cap breached for project: projects/988763814276
0e21 : 403 This API method requires billing to be enabled
```
两个项目共用结算账号 `01ED25-401537-125236`，同时失效 —— 大概率 $300 免费额度整体用尽。
**已停掉所有 HF job 与补位循环**（403 下继续跑只烧 HF 计算费、零产出）。

### 最终状态
* 前 50h **可用 caption 3068/6348 = 48.3%**（开工时 2.3%）。
* 完整跑完 4 条（都是十几分钟的小录制），另 16 条部分完成。
* 本地全量副本 `out/_delivery_20260905/`：每条录制一个 `captions.jsonl` + `captions.txt`，
  根目录 `index.json`。脱敏版在 `out/_public_20260905/`（用户说不需要脱敏，留档）。

### 恢复方式（额度解决后一条命令续跑）
```
python _watch/launch_v10.py --hours 50 --workers 16 --max-jobs 6
```
* `ONLY_FAILED=1` 只补 `ok=false` 的 clip，**已产出的 3068 个不会重复付费**。
* 解包好的帧还在 GCS，续跑不重传。
* 换新 GCP 项目只需改 `_watch/launch_v10.py` 顶部的 `CREDS` 字典
  （以及 `bf7` 已被硬编码排除的那处注释）。
* 监控：`_watch/tick_v10.sh`（7 分钟，零下载）+ 5 分钟补位循环，重新 arm 即可。

### 待办
* 跑完后过一遍 `deid.py`（report 自带的 pii_audit 显示单条录制就有上百处命中）。
* 挑一条 `pull --sample` + `render` 出审核视频做肉眼验收。

## 2026-09-06 — ADC 死后改走 rightcode 反代，gemini-3.7-flash 终于可用

### 两个反代的实况
| 代理 | base_url | 状态 |
|---|---|---|
| **rightcode** | `https://www.rightapi.ai/gemini/v1` | **可用**，gemini-3.7-flash |
| NovAI | `https://once.novai.su/v1` | 403「用户额度不足，剩余 ¥-0.67」欠费 |
两者的端点与 key 定义在 `caption/过程debug/早期caption版本/run_caption.py` 的
`PROXY_ENDPOINTS`，都走 **OpenAI SDK**（不是 Vertex 客户端），支持 `RIGHTCODE_API_KEY`
/ `NOVAI_API_KEY` 环境变量覆盖。
rightcode **没有余额查询接口**（`/api/user/self`、`/dashboard/billing/*` 全 404），
额度只能去它后台看 —— 这是当前最大的未知风险。

### 并发实测（真实 30 帧负载、生产同款 v9 prompt、temperature 0.3）
| 并发 | 成功率 | 单请求中位 | 吞吐 |
|---|---|---|---|
| 1 | 2/2 | 25s | 145 clip/h |
| **8** | **8/8** | 29s | **850 clip/h** |
| 16 | 16/16 | 38s | 874 clip/h |
并发 8→16 吞吐不涨、延迟涨 30% ⇒ **服务端已饱和，最优工作点并发 8–12，约 860 clip/h**。
零 429、零失败，比 ADC 稳定得多（ADC 最好时也有 25% 的 429）。
token 规模与 Vertex 一致（in 4–5.6 万），说明 payload 没被代理改写。

### 接入方式：只换协议外壳，内容逐字不变
`caption_core.py` 新增：
* `CAPTION_API`（adc | rightcode | novai）、`PROXY_BASE_URL`、`PROXY_API_KEY` 三个环境变量。
* `_parts_to_openai(parts)`：把 `google.genai` 的 Part 列表转成 OpenAI content 数组
  （`Part.text` → text 项；`Part.inline_data` → `image_url` 的 base64 data URI）。
* `run_one_proxy()`：返回结构与 `run_one` **完全一致**，下游无需区分。
* `run_one()` 入口按 `CAPTION_API` 分流；**代理通路不走模型降级阶梯**
  （阶梯是为绕开 Vertex 逐模型的额度漂移，反代是单一计费池，换模型无意义）。
关键点：两条通路**共用同一份 `parts`**，所以发出去的 prompt / 30 帧 / 参数逐字相同，
`ONLY_FAILED` 续跑与下游解析零改动。
`pipeline.py`：pip 装 `openai`；把 `CAPTION_API` / `PROXY_BASE_URL` / `PROXY_API_KEY`
从本机环境透传进 job env —— **key 不写进 `_code/`**，免得进 git 历史。

### 云端验证
`4-20` 一条：**148/187 全部 ✓，零失败**，in≈41k、out 2–10k、单 clip 18–28s。

### 启动方式（本机需先设三个环境变量）
```
export CAPTION_API=rightcode
export PROXY_BASE_URL=https://www.rightapi.ai/gemini/v1
export PROXY_API_KEY=<从 run_caption.py 的 PROXY_ENDPOINTS 取>
python _watch/launch_v10.py --hours 50 --workers 10 --max-jobs 6
```

## 2026-09-06 — prompt 加 action_brief 字段（精简动作）

用户要求：每个 segment 除 `action` 外再输出一个精简版 `action_brief`。
**新 prompt 在 `caption/prompt_brief.txt`，原 `prompt.txt` 未改动**（便于对照/回退）。

### 规则演进（第一版压过头了，用户当场纠正）
**v1（错）**：「恰好一个主要动词 + 恰好一块名词 + 3-7 词」。
结果把最值钱的信息删掉了：
* `I read the Claude Code panel explanation about TanhTransformedDistribution and
  change-of-variables log probability calculation.` → `I read Claude explanations.` ❌ 主题没了
* `I type '感觉ai好慢' into the WeChat chat with 'babe' ...` → `I send a WeChat message.` ❌ 引语和收件人都没了

**v2（对）**：「**一个主要动词，但保留所有承载信息的成分**」。
* 保留：引语原文、收件人/说话人、具体主题或标题、app/品牌名、被操作的具体对象。
* 只删：设备名、手/姿势/方位、"on the screen"/"in the input box" 这类界面脚手架、
  已被其它动词蕴含的冗余动词。
* **不设词数上限** —— "as short as possible, but not one bit of signal shorter"。
* prompt 里写进用户给的 4 个标定例 + 3 个反例（反例就是 v1 的错误输出）。

### 实测（16 场景 × rightcode/novai，`caption_model_batch_20260905/{brief,brief2}/`）
| | v1 旧规则 | v2 新规则 |
|---|---|---|
| 长度中位 | 5 词 | **7-8 词** |
| 最长 | **67 词**（语音段把整段引语塞进去） | **21 词** |
| 覆盖率 | 100% | 100% |
变长是对的（多出来的是主题/引语/收件人），同时最长值收敛 —— 语音段不再无脑塞完整引语。

### 一个容易误判的现象
按关键词比对新旧两轮时会看到 `action_brief=None`：**不是规则没执行，是这一轮模型的
分段变了**（segments 84→75 / 77→74），那条 action 被合并进别的段。
**分段本身有随机性**，跨轮次比对要按时间轴对齐，不能按关键词硬匹配。

### 跑批工具
* `caption_model_batch_20260905/run_gemini.py` —— 复用 `run_v10.py` 的场景加载/消息构造/
  解析落盘，只换 MODELS 为两家反代；`SCENES_16` 是带 `_review.mp4` 的 16 个场景。
* `render_gemini.py` —— 复用 `render_review.py` 的绘制原语，包一层把 `▸ action_brief`
  高亮显示在完整 action 之上；输出到 `caption_review_videos/`。

## 2026-09-06 — 定稿架构：v11 prompt + 双反代 + autopilot 分阶段

### 通路（ADC 已彻底废弃）
两个 GCP 项目额度耗尽后，caption 调用全部走 **OpenAI 兼容反代**：
| 反代 | base_url | 备注 |
|---|---|---|
| rightcode | `https://www.rightapi.ai/gemini/v1` | 主力，实测并发 8-16 吞吐饱和 ~870 clip/h |
| novai | `https://once.novai.su/v1` | 快一倍、命中 prompt 缓存；**语音逐字保真度明显更好** |
key 在 `caption/过程debug/早期caption版本/run_caption.py` 的 `PROXY_ENDPOINTS`，
用 `_watch/env_proxies.sh` 注入环境变量，**不写进 `_code/`**。

**novai 官方限额 RPM 430 / TPM 172K / RPD 8.6K**。按单请求 ~47K token 算 TPM 只够
约 2 并发，但 **2026-09-06 实测超限 40 倍照跑**，所以按用户要求恢复大额度。
收紧不用改代码：`NOVAI_WORKERS=2`、`NOVAI_MAX_JOBS=1` 两个环境变量即可。
（别把 800h 阶段建立在「限额不生效」这个假设上 —— 它可能像 Vertex 配额一样随时变。）

### 供应商质量差异（S09 实测，已用 transcript 逐字比对验证）
* **rightcode 会截断语音**：S09 里 5 段语音有 4 段用 `...` 截断，而源 transcript 没有省略号 —— 是在丢用户的话。
* **novai 逐字保真**：6/6 段完整，还多抓到 rightcode 完全漏掉的同事提问。
要语音信号的话 novai 明显更可靠。

### v11 prompt（用户审核通过）
`prompts/v11.txt` / `v11_nogaze.txt` = v9 + 两节新规则，与 `caption/prompt_brief.txt` 逐字一致：
1. **VERBATIM SPEECH**：说的话必须一字不差进 `action`，不省略/不改写/不翻译，
   保留口头重复（「默认的默认的」），`speech` 字段同样全文。
2. **action_brief**：一个主要动词，但**保留所有承载信息的成分**（引语、收件人、
   具体主题、品牌、对象），只删设备名/手/姿势/界面脚手架。**不设词数上限**。
   → 用户纠正过一次：第一版加了「一块名词 + 3-7 词」的硬约束，把主题和引语都删没了。
   prompt 里写进了 4 个标定例 + 3 个反例（反例就是第一版的错误输出）。
worker 侧用 `PROMPT_SET` 环境变量选版本（默认 v9，本批传 v11），
`cloud_worker.py` 的 `_pf = f"{_ps}.txt"`；**`push_code()` 必须带上 v11 两份**，
否则 job 里会 FileNotFoundError。
**已验证上线**：17:48 的提交里 8/8 个 clip 带 `action_brief`。

### autopilot（`_watch/autopilot.py`，8 分钟一轮）
一轮做四件事：算覆盖率（文件大小判据，零下载）→ 达 97% 推进阶段 →
按新增 ERROR / 吞吐变化调并发 → 补位到目标 job 数。
* 阶段：**50h → 100h → 800h**，自动推进。
* 调并发：出 ERROR 或吞吐掉 >40% → workers-4/jobs-1；连续两轮干净 → workers+2/jobs+1。
  边界 6-20 workers / 3-10 jobs。
* **首轮必须只播种 ERROR 不反应**，否则历史失败会把并发一把砍到底（踩过）。
* 「已跑完的算跑完」：`ONLY_FAILED=1` 只补 `ok=false` 的 clip，阶段推进不回头重跑。

### 短窗口速率读数不可信（今晚栽了两次）
`FLUSH_EVERY=200` 意味着 job 攒够 200 个 clip 才提交，多个 job 同时 flush 会让
8 分钟窗口显示出 4000-5000 clip/h 这种远超理论上限（两家合计约 1100 clip/h）的数。
**判断速率要看多轮平均，或直接看 job 日志里的 clip/h。**

## 2026-09-06 — 调度器 v2 与「大小判据」的标定错误

### 大小判据阈值必须标定，不能拍脑袋（本轮最贵的一个 bug）
用「clip json 字节数 ≥ N 即成功」代替逐个下载判 `ok`，方向是对的（省下几千次下载、
避免把 HF 打限流），但**我把 N 拍成了 6000，没做标定**。
分层抽样 84 个 clip 对照真实 `ok` 字段：
* `ok=True` 最小 **2568 字节** —— 静态场景只有 1 个 segment，文件天然就小
  （例：「I sit at a library study table and read through a resume document.」4548 字节）
* `ok=False` 典型 995 字节（另有 1 个 27KB 的异常失败记录）
* **阈值 6000：84 个里误判 27 个成功为失败（32%）**；阈值 **2500：0 漏判，1/84 误判**

连锁后果（花了几小时才追到）：
1. 覆盖率被严重低估，50h 窗口其实早就达标了，却显示 93.6% 卡住不动。
2. `launch_v10.needs()` 用同一阈值 → **一直在追幽灵缺口**：起 job → 解包 47 GB →
   worker 侧 `ONLY_FAILED` 按真实 `ok` 判定发现没活干 → 空跑退出。
   这就是「4 个 job 在跑、0 clip/h」的真相，不是供应商退化。
3. 空跑被调度器当成吞吐崩塌，触发并发振荡。
**教训：任何代理指标（proxy metric）上线前必须用真值分层标定一次。**

### 调度器 v2（v1 的三个设计错误）
v1 轨迹（用户指出的）：`零ERROR→加并发` 一路加到 20w×10j，然后 `0 clip/h` 时**还在加**。
1. **`--workers` 是每 job 的并发，被当成全局旋钮**：6 job × 20 worker = 120 并发，
   而 rightcode 实测 8-16 就饱和。每次「加并发」都在加剧超订。
   → v2 改**总并发预算** `TOTAL_BASE=24`，`per_job = total // slots`（下限 2）。
2. **拿单轮覆盖率增量当吞吐**：`FLUSH_EVERY=200` 让 clip 成批砸下来，单轮读数在
   0 和 5000 之间乱跳。→ v2 用 EMA 且**只展示不决策**。
3. **「零 ERROR 就加并发」是单向棘轮**，加到出事为止；窗口尾部「零 ERROR」
   只是没活干。→ v2 **只在出错时降**，无错时朝基线缓慢回升，不主动突破基线。

其它两点：
* **槽位数受「有缺口的录制数」约束**（一条录制 = 一个 job），尾部填不满是正常现象，
  不是故障信号，别拿它触发降速。
* **停滞推进**：连续 5 轮无进展且覆盖率 ≥85% 就进下一阶段，否则最后几个反复失败的
  clip 会让 autopilot 卡在 95% 一整晚，后面的阶段永远开不了工。

## 2026-09-06 收官 — 50h 与 100h 两个窗口全部完成

| 窗口 | 结果 |
|---|---|
| 50h | 100%（6347 clip） |
| **100h** | **100%（12180 clip / 133 条录制）** |
本地交付 `out/_delivery_100h/`（205 MB）：每条录制 `captions.jsonl` + `captions.txt`，
根目录 `index.json`。**只含成功 clip**，失败记录已排除。

### 数据集是混合的，交付前要说清
* **带 `action_brief`（v11）: 8007 = 66.1%**；其余 4098 个是 v11 上线前跑的 v9 结果，
  caption 有效但没有精简动作字段。
* 模型分布：gemini-3.7-flash 8846 / 3.5-flash 3252 / 3.6-flash 5 / 3.5-flash-lite 2。
* 每个 clip 自带 `model` 字段，`action_brief` 有无可逐条判定 —— 可过滤，不是脏数据。
* 补齐那 4098 个（v11 重跑）约 $125 / 2 小时。

### 稳定期实测数据（可作为后续估算基准）
* 单 job 3 worker ≈ **225 clip/h**；8 job 满编 ≈ 1800-2400 clip/h。
* clip 级失败率稳定 **0-1%**（走 rightcode + novai 双反代）。
* 单价 ~$0.031/clip。

### 800h 阶段已加闸门，不会自动开跑
`_watch/autopilot.py` 里 `ALLOW_800=1` 才放行。原因：
526 条录制 / **95988 clip** / 约 **$3000** / 约 **48 小时**连续运行 ——
比今晚任何一次预算高一个数量级，而今晚已经烧穿两个 $300 账号，
且两家反代都**没有余额查询接口**，等于全程盲飞。
放行前还要解决：① 预算确认；② 监控要迁到 launchd（现在的 Monitor 随会话结束而死，
撑不了 48 小时）；③ novai 的 RPD 8.6K/天 意味着它最多只能承担约 9% 的量，
rightcode 一旦挂掉几乎全线停摆。

### 仍未做：脱敏
`deid.py` 已写好并验证（3474 处 HKU/HKUL、69 处中文姓名、14 个手机号、9 个邮箱、
10 个 github handle、36 处本机路径全部清零；公开论文/课程网站/公众人物故意保留）。
**但至今没对任何交付数据跑过** —— 现有 caption 里有明文校名、人名、电话和完整屏幕内容。

## 2026-09-06 — 800h 阶段：三个「指标正常但底下出血」的 bug

### 1. 解析 `hf jobs ps` 文本在 launchd 下必然失效（最隐蔽）
CLI 在**交互式终端**下输出制表符分隔；在 **launchd 这类精简环境**下输出的是
**对齐且截断**的表格（`povcap-6...`、无制表符）。按 `\t` 切分永远得到 0 个字段 →
`job_stats()` 每轮都算出「0 个 job 在跑」→ daemon 反复重复提交。
**实测后果：8 条录制各起了 3 个实例，24 个 job 干 8 个 job 的活。**
→ 改用 `HfApi().list_jobs()`，job 名在 `j.labels["name"]`，状态在 `j.status.stage`。
**教训：任何常驻/守护进程里都不要解析 CLI 的人类可读输出。**

### 2. 冷却表只写不读
`launch_v10.py` 里写了 `repaired.log`（注释还专门解释了竞态），但**从来没读过**，
纯装饰。新 job 要约 30 秒才出现在列表里，这个窗口期第二个实例就会重复提交。
→ 加了 30 分钟冷却读取。

### 3. autopilot 没有单实例锁
手动跑一次 + launchd daemon 同时 tick，各自提交一遍。
→ 加 `fcntl.flock` 排他锁，拿不到就跳过本轮（不排队）。

### 换到 API 后要重新播种 ERROR 基线
旧的 seen 文件是坏解析建的（一直空），换 API 后一次性返回 104 个历史 ERROR，
全被当「新增」触发降并发。**任何改变数据源的修复，都要同步重置基线状态。**

### 800h 规模的其它准备（已确认）
* **解包开销不是瓶颈**：87% 的 clip 在 ≥60 分钟录制里；<10 分钟的仅 30 条 / 246 clip（0.3%）。
* **job 超时 5h → 8h**：最长 438 分钟录制 877 clip，3 worker 下要 3.9h，余量太薄。
* **休眠已被 caffeinate 阻止**（`sleep 0`），40 小时连续跑无虞。
* **HF API 配额 1000 次/5 分钟**：覆盖率扫描必须**一次递归列举** `captions/`，
  按录制逐个列（526 次）会直接打爆并阻塞启动器。
