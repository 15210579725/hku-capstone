# AutoDL 上原始复现 RepE 情绪实验 · 可执行手册

> 在 AutoDL 租 GPU，**先原始复现** RepE（Zou et al. 2023, arXiv:2310.01405）官方仓库 `andyzoujm/representation-engineering` 的两个情绪 notebook，跑通后再换中文小模型 + 4 情绪扩展。
> 价格为近期挂牌区间，动态价以 AutoDL 控制台→算力市场实时报价为准。

## 0. 速览 + 三条铁律

| 实验 | notebook | 模型(论文原配) | 显存 | 推荐卡 |
|---|---|---|---|---|
| 检测/reading | `emotion_concept.ipynb` | Llama-2-13b-chat(13B fp16) | ≈26 GB | A100 40G(忠实) 或 4090 24G+8bit(省钱) |
| steering | `emotion_function.ipynb` | Mistral-7B-Instruct-v0.1(7B fp16) | ≈14 GB | 4090/3090 24G |

**三条会卡死复现的坑：**
1. **国内连不上 huggingface.co** → 必须 `source /etc/network_turbo` 或走 `hf-mirror.com` / ModelScope。
2. **Llama-2 是 gated 模型** → 用非 gated 镜像 `NousResearch/Llama-2-13b-chat-hf` 或 ModelScope `modelscope/Llama-2-13b-chat-ms`，免去 Meta 申请。
3. **RepE 代码老，新版 transformers 必 break** → **必须 pin `transformers==4.35.2`**（issue #56 实证 4.42+ 失效）。

## 1. GPU 选型 + 成本

近期挂牌价(¥/时)：RTX 4090 24G ≈1.98 ｜ RTX 3090 24G ≈1.3–2.0 ｜ L20 48G ≈3.68 ｜ A100 40G ≈2.8–4 / 80G ≈4–6.68。会员/学生多 95 折。

- **Mistral-7B fp16 steering(≈14G)** → 4090/3090 24G 绰绰有余。
- **Llama-2-13B fp16 reading(≈26G)** → 24G fp16 装不下，两条路：
  - **省钱**：4090 24G + **8bit 量化 13B**(≈14G)，精度略损但够验证。
  - **忠实**：A100 40G fp16 13B，完全按论文原配。

单轮(非训练，前向+PCA+少量生成)1–3h。**省钱路线总花费 ≈¥4–6，忠实 ≈¥6–9。**
> 省钱关键：先用 **无卡模式(¥0.1/时)** 下好模型、装好环境，再切正常开机只跑推理 → GPU 计费可压到 30–60 分钟，单轮 **≈¥2–4**。

## 2. 国内访问 HuggingFace

**法一·学术加速(最省事，2026 仍有效)**
```bash
source /etc/network_turbo   # 加速 github/huggingface.co；不加速 pypi；取消用 unset http_proxy https_proxy
```
**法二·hf-mirror 镜像(稳，推荐)**
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
# 旧 CLI：huggingface-cli download ...   新版(>=v1.0)：hf download ...
apt update && apt install -y aria2 git-lfs
wget https://hf-mirror.com/hfd/hfd.sh && chmod +x hfd.sh
./hfd.sh NousResearch/Llama-2-13b-chat-hf --tool aria2c -x 8 --local-dir /root/autodl-tmp/Llama-2-13b-chat-hf
```
**法三·ModelScope(魔搭，国内最快，无需 token)**
```bash
pip install modelscope
modelscope download --model AI-ModelScope/Mistral-7B-Instruct-v0.1 --local_dir /root/autodl-tmp/Mistral-7B-Instruct-v0.1
modelscope download --model modelscope/Llama-2-13b-chat-ms --local_dir /root/autodl-tmp/Llama-2-13b-chat-ms
```
**gated 替代**：Llama-2 用 `NousResearch/Llama-2-13b-chat-hf`(权重与官方一致，非 gated)；Mistral 走 ModelScope。下载目录与 HF 兼容，`from_pretrained("/root/autodl-tmp/xxx")` 直接加载。
**推荐组合**：Llama-2 走 hf-mirror 的 NousResearch，Mistral 走 ModelScope。

## 3. 环境搭建（最关键：版本钉死）

镜像选 **PyTorch 2.1.x / Python 3.10 / CUDA 12.1**；数据盘 ≥80 GB；模型代码都放 `/root/autodl-tmp`(系统盘只 30G)。
```bash
cd /root/autodl-tmp
conda create -n repe python=3.10 -y && conda activate repe
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
source /etc/network_turbo
git clone https://github.com/andyzoujm/representation-engineering.git
cd representation-engineering && pip install -e .
# 版本钉死（RepE 老代码，新版 transformers 必 break；issue #56 实证 4.42+ 失效）
pip install "transformers==4.35.2" "tokenizers==0.15.0" "accelerate==0.25.0" "torch==2.1.2" \
            "numpy==1.26.4" "scikit-learn==1.3.2" "datasets==2.16.1" sentencepiece protobuf
pip install "bitsandbytes==0.41.3"   # 仅省钱路线 8bit 需要
```
验证：
```python
import transformers; print(transformers.__version__)   # 4.35.2
from repe import repe_pipeline_registry; repe_pipeline_registry()   # 不报错=pipeline 注册成功
```
> 兜底：原仓库实在跑不动时改用 `vgel/repeng`(派生自官方、对接现代 HF API)，但偏离"原始复现"，仅 plan B。

## 4. 原始复现步骤

- 控制台→JupyterLab 打开 `examples/primary_emotions/emotion_concept.ipynb`；kernel 用 repe 环境(`python -m ipykernel install --user --name repe`)；久任务用 SSH+tmux。
- 改模型路径为本地：
  - reading：`model_name_or_path = "/root/autodl-tmp/Llama-2-13b-chat-hf"`
  - steering：`model_name_or_path = "/root/autodl-tmp/Mistral-7B-Instruct-v0.1"`
- 六情绪英文刺激 `data/emotions/*.json` 仓库自带，无需另下。
- **reading 预期产出**：per-layer 情绪分类准确率曲线(中层~50% 深度达峰，六分类应 80–90%+，随机仅 17%)；沿叙事的情绪 reading score 曲线。
- **steering 预期产出**：注入快乐/愤怒/悲伤向量后，同一中性 prompt 输出明显带对应情绪；`coef` 调强度，过大文本崩坏属正常。
- **判定成功**：复现出论文的趋势和量级即可，不必逐数字对齐(镜像/量化有微小差异)。

## 5. 常见坑

| 坑 | 应对 |
|---|---|
| gated 401 | 换 NousResearch 镜像 / ModelScope |
| transformers break(`_use_flash_attention_2`、pipeline KeyError) | pin 4.35.2；加载加 `attn_implementation="eager"` |
| 13B fp16 OOM(24G) | 8bit 量化 或 换 A100 40G |
| 8bit 写法 | 用 `BitsAndBytesConfig`(下)，别用已弃用的 `load_in_8bit=True` |
| HF 下载断 | `--resume-download` / hfd.sh 断点续传 |
| notebook import/路径 | 在 `examples/primary_emotions/` 启 kernel；确认 `pip install -e .` 生效 |
| 计费 | 无卡模式(¥0.1/时)下模型装环境，再切开机跑；跑完立即关机 |

8bit 加载 13B（省钱路线，替换 notebook 的 `from_pretrained`）：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/Llama-2-13b-chat-hf",
    quantization_config=bnb, device_map="auto", attn_implementation="eager")
```
无卡模式工作流：关机→「无卡模式开机」(¥0.1/时)→下模型/装环境/验证→关机→正常开机(4090/A100)→只跑推理→跑完关机。

## 6. 下一步衔接（复现跑通后）

1. **换模型**：`Qwen/Qwen2.5-1.5B-Instruct` 或 `Qwen2.5-3B-Instruct`(ModelScope 直接下，LlamaForCausalLM 兼容架构)。若 4.35.2 加载新 Qwen 报错，小幅升 transformers 测试。
2. **情绪扩 4 类**：开心/幸福、焦虑、悲伤、愤怒/暴躁。前三+愤怒可改写自 `data/emotions/`，**"焦虑"需自建中文刺激**(仿 `utils.py` 模板写中文版)。
3. **接中文 egocentric 数据**：`daily_C_中等粒度.txt` 做窗口 → 探针读日内情绪轨迹 → 与"自陈"对比一致性。
4. **GPU 优势**：CUDA 上无 M1 MPS 限制，fp16/8bit/bf16 全可用，13B 也能 A100 fp16 直跑。

## 7.（可选）第二篇 Neural-MRI / SLM 情绪 steering

arXiv:2604.04064 + github.com/JihoonJeong/Neural-MRI(MIT)。技术栈 FastAPI+TransformerLens+PyTorch+SAELens，Python 3.11+，`uv` 管依赖。支持小模型：GPT-2/Pythia-1.4B/**Qwen-2.5-3B(8.4G)**/Llama-3.2-3B；**Mistral-7B 被标 TL unsupported**。TransformerLens 当前 3.4.0，CUDA 上易装易用；MPS 静默错误(#1178)是 Apple Silicon 专属，**CUDA 不走该路径不受影响**(基于根因推断，原 issue 未实测 CUDA)。

## 关键来源
AutoDL 文档(价格/省钱/network_turbo/镜像/JupyterLab) ｜ hf-mirror.com ｜ NousResearch/Llama-2-13b-chat-hf(非 gated) ｜ ModelScope ｜ RepE issue #56(4.42+ break) ｜ vgel/repeng ｜ arXiv:2604.04064 / Neural-MRI ｜ TransformerLens issue #1178。
