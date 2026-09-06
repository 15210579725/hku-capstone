# POV Dense Video Captioning Pipeline

对第一人称（egocentric）POV 视频做 dense video captioning，产出带时间戳的结构化 JSON。

当前生产架构：**v10 单遍**（`cloud_worker.py` + `caption_onepass.py`），每个 30 秒 clip 一次 Gemini 调用，
整帧 + 注视点 zoom crop 均 MEDIUM 分辨率。prompt 包含 `action_brief`（精简动作）和语音逐字保真规则。

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
# 可选（提升语音时间戳精度）：
pip install torch silero-vad
```

### 2. 凭据

需要两样东西：

**HuggingFace Token**：在 https://huggingface.co/settings/tokens 创建 READ token，
且账号要有数据仓库（如 `mmm8383/pov-data`）的访问权限。

**Gemini API**：有两种方式（二选一）：
- **Google Cloud ADC**（免费 $300）：`gcloud auth application-default login`，填 `config.json` 里的 `google_project`
- **OpenAI 兼容反代**（如 rightcode）：设环境变量 `CAPTION_API=rightcode`、`PROXY_BASE_URL`、`PROXY_API_KEY`

### 3. 配置

编辑 `config.json`：

```json
{
  "hf_token": "hf_xxxxx",
  "google_project": "your-project-id",
  "google_location": "global"
}
```

也可以通过环境变量 `HF_TOKEN`、`GOOGLE_CLOUD_PROJECT` 设置。

### 4. 运行

所有命令在 `pipelines/` 目录下执行。

```bash
# 列出数据仓库里所有天和录制
python pipeline.py days

# 按语音密度排序候选录制
python pipeline.py scan --month 2026-05 --limit 12

# 冒烟测试（3 个 clip）
python pipeline.py run --tar <tar路径> --max-clips 3

# 跑整条录制
python pipeline.py run --tar <tar路径>

# 跑一整天
python pipeline.py run --day 2026-05-18

# 只补跑失败的 clip
python pipeline.py run --tar <tar路径> --only-failed

# 拉回结果 + 抽样 clip 包
python pipeline.py pull --rec <rec> --sample 8

# 合并整天
python pipeline.py merge --day 2026-05-18

# 渲染审核视频
python pipeline.py render --rec <rec> --sample 8
```

## 目录结构

```
pipelines/
  pipeline.py          主 CLI（选片/提交 HF Job/拉结果/合并/渲染）
  cloud_worker.py      HF Job 里跑的 worker（v10 单遍默认）
  caption_core.py      帧渲染 + context 组装 + Gemini 调用
  caption_onepass.py   v10 单遍（整帧 + zoom crop，一次 Gemini 调用）
  caption_v9.py        v9 双遍（pass0 语音 + pass1 caption + pass2 精读）
  remote_tar.py        tar 访问层（本地挂载 / HTTP Range）
  merge.py             clip → 录制 → 整天合并
  deid.py              PII 脱敏
  render_review.py     审核视频渲染
  prompts/prompt.txt   caption prompt（含 action_brief + 语音逐字）
  prompts/prompt_nogaze.txt  无眼动录制用的 prompt 变体
  archive/             历史文件（旧 worker/prompt/实验脚本）

prompt.txt             本地入口用的 prompt（与 prompts/prompt.txt 相同）
run_caption_2pass.py   本地 caption 入口（不经过 HF Job）
render_caption_video.py 本地审核视频渲染
config.json            凭据和配置模板
```

## 输出格式

每个 clip 产出一个 JSON，核心字段：

```json
{
  "segments": [
    {
      "time": "12:34:56 HKT",
      "time_end": "12:35:02 HKT",
      "action": "I type '感觉ai好慢' into the WeChat chat...",
      "action_brief": "I send '感觉ai好慢' to babe on WeChat.",
      "environment": "PHYSICAL SPACE: ...\nOBJECTS: ...\nSCREEN CONTENT: ...",
      "text_visible": ["Tab: Claude Code", "Terminal: python train.py"],
      "speech": [{"speaker": "me", "text": "感觉ai好慢"}]
    }
  ]
}
```

## 成本参考

| 模型 | 每 clip | 每小时视频 |
|---|---|---|
| gemini-3.7-flash (Vertex) | ~$0.031 | ~$3.7 |
| gemini-3.5-flash (Vertex) | ~$0.020 | ~$2.4 |
| 反代 (rightcode) | 取决于定价 | — |

## 适配其他数据集

本流水线的数据源是 HuggingFace 上的 tar 包。要跑其他数据集（如 CASTLE）：

1. 把视频数据打成同样的 tar 结构上传到 HF dataset repo
2. 修改 `pipeline.py` 顶部的 `SRC_REPO` 和 `OUT_REPO`
3. 如果帧格式/命名不同，可能需要调整 `cloud_worker.py` 的解包逻辑
