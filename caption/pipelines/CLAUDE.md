# CLAUDE.md — POV Caption 云端流水线

从 HuggingFace 数据仓库读取 POV 眼镜录制 tar → HF Job 上原地解包 →
Gemini dense caption → 产物回 HF private dataset → 本地拉回、合并、渲染审核视频。

## 目录

```
pipeline.py          本地 CLI（选片/提交 job/看状态/拉结果/合并/渲染）
cloud_worker.py      HF Job 里跑的 worker（v10 单遍默认）
caption_core.py      帧渲染 + context 组装 + Gemini 调用
caption_onepass.py   v10 单遍（整帧 + zoom crop，一次调用）
caption_v9.py        v9 双遍（pass0 语音 + pass1 + pass2 精读，备用）
remote_tar.py        tar 访问层：FileSource(挂载) + HttpRangeSource(Range)
merge.py             clip → 录制 → 整天
deid.py              PII 脱敏
render_review.py     审核视频
prompts/prompt.txt   当前 prompt（含 action_brief + 语音逐字保真）
prompts/prompt_nogaze.txt  无眼动变体
archive/             历史文件（batch_worker, bench_worker, 旧 prompt 等）
out/<date>/<rec>/    拉回本地的产物
```

## 常用命令

```bash
python pipeline.py days                                    # 列出所有天
python pipeline.py scan --month 2026-05 --limit 12         # 候选排序
python pipeline.py run --tar <tar路径> --max-clips 3       # 冒烟
python pipeline.py run --tar <tar路径>                     # 整条
python pipeline.py run --day 2026-05-18                    # 一天全部
python pipeline.py run --tar <tar路径> --only-failed       # 只补失败
python pipeline.py status --all                            # 查看 HF job 状态
python pipeline.py logs <job_id>                           # 查看 job 日志
python pipeline.py pull --rec <rec> --sample 8             # 拉回结果
python pipeline.py merge --day 2026-05-18                  # 合并整天
python pipeline.py render --rec <rec> --sample 8           # 审核视频
```

## 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `HF_TOKEN` | HuggingFace READ token | — |
| `GOOGLE_CLOUD_PROJECT` | GCP 项目 ID（ADC 模式） | — |
| `CAPTION_API` | 调用方式：`adc` / `rightcode` / `novai` | `adc` |
| `PROXY_BASE_URL` | 反代 base URL（非 adc 时必填） | — |
| `PROXY_API_KEY` | 反代 API key（非 adc 时必填） | — |
| `CAPTION_VERSION` | caption 版本：`v10`（单遍）/ `v9`（双遍） | `v10` |
| `ONLY_FAILED` | `1` = 只补跑 `ok=false` 的 clip | `0` |

### 凭据

两种方式（二选一）：

**方式 A — Google Cloud ADC（Vertex AI）**：
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id
```

**方式 B — OpenAI 兼容反代**：
```bash
export CAPTION_API=rightcode
export PROXY_BASE_URL=https://www.rightapi.ai/gemini/v1
export PROXY_API_KEY=your-key
```

## tar 内部布局

有两种布局，程序自动适配（`walk_headers` 扫全表，不依赖顺序）：

```
布局 A: eye_tracking/*.jpg → audio/ → picture/masked → picture/ocr_text → gaze.csv
布局 B: picture/ocr_text → picture/masked → audio/ → gaze.csv
```

## 踩过的坑

1. **masked 帧是时间倒序的**，注视轨迹依赖正序。`build_trails()` 先按 frame_index 升序算好再渲染。
2. **clip_id = frame_index // clip_seconds**，不依赖总帧数。
3. **transcript 有两种时间格式**：绝对 HKT 和相对偏移 `[00:00:00-00:00:30]`，两种都会出现。
4. **tar 文件名的日期 ≠ 录制内部时钟**：`out/` 按内部日期归档。
5. **Gemini inline 上限 20MB**：正常 1440px × 30 帧约 4.5MB，不会触发。
6. **`FileSource.read` 必须用 `os.pread`** 保证线程安全，共享句柄 seek+read 会交错读错位。

## 适配其他数据集

要在其他数据集（如 CASTLE）上跑：

1. 将视频数据打成 tar 上传到 HF dataset repo
2. 修改 `pipeline.py` 顶部的 `SRC_REPO`（源数据仓库）和 `OUT_REPO`（产物仓库）
3. tar 内部需要有 `picture/masked/*.jpg`（帧图片）和 `audio/anonymized.wav`（音频）
4. 可选：`picture/ocr_text/*.txt`（每帧 OCR）、`eye_tracking/gaze.csv`（眼动数据）
5. 帧文件名需包含时间戳（格式 `*_YYYY-MM-DD_HH-MM-SS_HKT_*.jpg` 或类似）
