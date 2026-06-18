# emotion-probe — 情绪探针复现线

本目录是 capstone 的核心:用**表征读取/定位**的方法,检验 LLM 内部是否存在可读出、可干预的情绪表征。包含三条按方法论文脉络命名的复现线。

## 三条线

| 子目录 | 模型 | 方法 | 一句话 |
|--------|------|------|--------|
| [`RepE-2023-Zou/`](RepE-2023-Zou/) | Qwen2.5-3B / Llama-2-13b | RepE reading 向量(LAT),残差投影 | **旗舰/方法母本**:中文日记 → 逐时刻 6 情绪曲线,真人自陈金标准验证 |
| [`do_llm_feels_2025-wang/`](do_llm_feels_2025-wang/) | Llama-3.2-3B | diff-of-means → 神经元/头归因 → steering | 六情绪**电路级定位 + 因果干预** |
| [`anthropic-2026-gemma-version/`](anthropic-2026-gemma-version/) | gemma-4-E4B | diff-of-means + 中性 PCA 去混杂 | **多情绪(8 类)+ 跨模型**验证 |

RepE 线是方法母本,后两条线分别在"电路/因果"与"换模型/多情绪"方向上延伸。

## 怎么用

三条线**环境相互独立**(尤其 transformers 版本要求不同),请进入各子目录、按其 `README.md` 分别建虚拟环境与运行。

最快验证基础环境(无需 GPU/模型):

```bash
cd RepE-2023-Zou
python3 -m pip install -r env/requirements_probe.txt
python3 scripts/run_mock_selftest.py   # 端到端 mock,本机已验证通过
```

## 共性说明

- 情绪向量等大文件(`*.pt`)已被 `.gitignore` 排除,需在 GPU 上重跑抽取生成,方法见各子目录 README。
- 部分脚本含硬编码数据路径,运行前需按子目录说明调整。
- 详细架构与坑见各子目录的 `CLAUDE.md`。
