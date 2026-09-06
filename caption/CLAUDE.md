# CLAUDE.md — Dense Video Captioning Pipeline

对第一人称（egocentric）POV 眼镜视频做 dense video captioning，产出结构化 JSON caption。

**生产版 = v10 单遍**（`pipelines/cloud_worker.py` + `caption_onepass.py`，`CAPTION_VERSION=v10` 默认）。
整帧和注视点 zoom crop 均 MEDIUM 分辨率，一次 Gemini 调用出结果。

---

## 目录布局

```
caption/
  prompt.txt               ← caption prompt（v11：action_brief + 语音逐字保真）
  run_caption_2pass.py     ← 本地 caption 入口
  render_caption_video.py  ← 本地审核视频渲染
  config.json              ← 凭据模板
  CLAUDE.md                ← 本文件

  pipelines/               ← 云端流水线（HF Job），见其 CLAUDE.md
  archive/                 ← 历史文件
```

---

## Prompt 要点

当前使用 `pipelines/prompts/prompt.txt`（v11），要点：

- **Rule 4 环境字段 = world model 训练数据**：首段完整建立性描述（物理空间/物体/屏幕内容/文字/环境音），后续每段只写变化，`"No change."` 合法
- **VERBATIM SPEECH**：说的话必须一字不差进 `action`，不省略/不改写/不翻译，保留口头重复
- **action_brief**：一个主要动词，保留所有承载信息的成分（引语、收件人、主题、品牌），只删设备名/手势/界面脚手架，不设词数上限
- 其它规则：第一人称、绝对 HKT 时间戳、隐私脱敏、注视点优先

无眼动的录制自动使用 `prompt_nogaze.txt`（去掉了注视圈相关规则）。

---

## 锁定配置

| 项 | 值 |
|---|---|
| 模型 | `gemini-3.7-flash`（或 3.5-flash / 3.6-flash） |
| 认证 | ADC Vertex AI 或 OpenAI 兼容反代 |
| 帧分辨率 | 1440px q60，media_resolution MEDIUM |
| zoom crop | 1440px q70，MEDIUM，注视点中心 62% 裁切 |
| 并发 | cloud_worker 内 3-16 workers |
| 重试 | run_one 内 3-7 轮退避重试 |

---

## 快速开始

```bash
cd pipelines/

# 列出所有天
python pipeline.py days

# 跑一条录制
python pipeline.py run --tar aria/2026-05-18/5-18_hkt2130-2214_43m_xxx.tar

# 冒烟（3 clip）
python pipeline.py run --tar <tar> --max-clips 3

# 补跑失败
python pipeline.py run --tar <tar> --only-failed

# 拉回结果
python pipeline.py pull --rec <rec> --sample 8

# 渲染审核视频
python pipeline.py render --rec <rec>
```
