# POV Dense Video Captioning Pipeline

从 HuggingFace 私有仓库 `mmm8383/pov-data` 读取 POV 眼镜录制数据，通过 Google Gemini 生成 v9 三遍 dense video caption。

## 前置条件

### 1. Python 环境

```bash
pip install -r requirements.txt
```

可选依赖（提升语音时间戳精度）：
```bash
pip install torch silero-vad
```

### 2. HuggingFace Token

在 https://huggingface.co/settings/tokens 创建一个有 READ 权限的 token。需要有 `mmm8383/pov-data` 仓库的访问权限。

### 3. Google Cloud 配置

1. 安装 [gcloud CLI](https://cloud.google.com/sdk/docs/install)
2. 登录并设置应用默认凭据：
   ```bash
   gcloud auth application-default login
   ```
3. 在 [Google Cloud Console](https://console.cloud.google.com) 创建项目并启用 Vertex AI API
4. 将项目 ID 填入 `config.json` 的 `google_project` 字段

> Google Cloud 新用户有 $300 免费额度，足够处理大量数据。

### 4. 配置文件

编辑 `config.json`，填入必要的凭据：

```json
{
  "hf_token": "hf_xxxxx",
  "google_project": "your-project-id",
  "google_location": "global",
  "model": "gemini-3.7-flash",
  "output_dir": "./caption_output",
  "workers": 3,
  "clip_seconds": 30,
  "rounds": 3
}
```

| 字段 | 说明 |
|---|---|
| `hf_token` | HuggingFace READ token |
| `google_project` | Google Cloud 项目 ID |
| `google_location` | Vertex AI 区域，默认 `global` |
| `model` | Gemini 模型名，默认 `gemini-3.7-flash` |
| `output_dir` | 输出目录 |
| `workers` | caption 并发数（建议 3，过高会触发限流） |
| `clip_seconds` | 每个 clip 秒数（默认 30） |
| `rounds` | 失败 clip 的重试轮次 |

也可以通过环境变量设置 `HF_TOKEN` 和 `GOOGLE_CLOUD_PROJECT`。

## 使用方法

### 列出可用数据

```bash
python run_pipeline.py list
```

输出按天分组的录制列表，显示每天的录制数、总时长和文件大小。

### 处理一天的全部录制

```bash
python run_pipeline.py run --day 2026-05-18
```

### 处理单条录制

```bash
python run_pipeline.py run --tar aria/2026-05-18/5-18_hkt2130-2214_43m_xxx.tar
```

### 冒烟测试（只处理前几个 clip）

```bash
python run_pipeline.py run --tar aria/2026-05-18/xxx.tar --max-clips 3
```

### 其他选项

```bash
--workers 6        # 调整并发数
--output-dir ./out # 指定输出目录
--no-vad           # 禁用 silero-vad 语音对齐
--no-ocr           # 不使用 OCR 文本
--clip-seconds 15  # 自定义 clip 时长
```

## 输出结构

```
caption_output/
  2026-05-18/
    <recording_name>/
      captions_full.json    # 完整 JSON（含全部 segments）
      captions_full.jsonl   # 每行一个 segment
      captions_full.txt     # 人类可读文本版
      report.json           # 统计报告（clip 成功率、token 用量、PII 审计等）
      clips/
        clip_0000.json      # 每个 clip 的详细结果（含 pass1/pass2 原始内容）
    day_2026-05-18.json     # 整天合并结果（如果处理了多条录制）
    day_2026-05-18.txt
    day_2026-05-18.jsonl
```

## 架构说明

### v9 三遍 Caption

1. **Pass 0（语音对齐）**：silero-vad 检测语音窗口（精度 ~0.03s），Gemini 填词并判断说话人
2. **Pass 1（全量 caption + 区域框）**：2880px 帧 + 完整 prompt → 结构化 caption + 文字密集区域的 `box_2d` 坐标
3. **Pass 2（精读修订）**：按坐标从原图裁切文字区域，ULTRA_HIGH 分辨率精读，修订 Pass 1 的描述

### 数据访问

通过 HTTP Range 请求直接从 HuggingFace 读取 tar 文件，不需要下载整个文件到本地。支持两种 tar 内部布局（Layout A 和 Layout B），自动适配。

### 成本估算

使用 `gemini-3.7-flash`（Vertex AI 引导价 $0.75/M input, $3.75/M output）：
- 每个 30 秒 clip 约 $0.04-0.08
- 每小时视频约 $3-5

## 常见问题

**Q: 报错 `找不到 audio/anonymized.wav`**
A: 这是 tar 布局 B 的正常降级处理，程序会自动用 `walk_headers` 兜底扫描。如果最终报错，说明该录制确实没有音频。

**Q: `429 RESOURCE_EXHAUSTED`**
A: Vertex AI 的共享配额限流。程序内置了自动退避重试（最长等待 5 分钟），降低 `--workers` 可减少触发。

**Q: `silero-vad` 不可用**
A: 安装 `torch` 和 `silero-vad` 即可。不装也能运行，语音对齐会退化到能量 VAD 或直接使用粗粒度 transcript 时间戳。
