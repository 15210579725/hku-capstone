# CLAUDE.md — do_llm_feels(EmotionCircuits 线)开发者/AI 指南

面向后续接手的 AI 与开发者。先读本文件,再动代码。**改动优先落在自有脚本,勿改 `repo/EmotionCircuits-LLM/` 内官方脚本逻辑**(它是原样打包的复现仓,改了会偏离论文)。

---

## 1. 这是什么 / 复现架构

复现并扩展 **EmotionCircuits-LLM**(arXiv:2510.11328,MBZUAI)。核心命题:LLM 内部是否存在**可定位、可干预**的情绪表征。方法链:

```
diff-of-means 情绪方向(残差流)→ 归因到 MLP 神经元 / attention head → 整合成"情绪电路" → steering(注入/消融)验证因果
```

模型:**Llama-3.2-3B-Instruct**(28 层,hidden=3072,`MODEL_NAME="llama32_3b"`)。
六情绪:`anger / disgust / fear / happiness / sadness / surprise`。

官方七阶段 pipeline(`repo/EmotionCircuits-LLM/scripts/01…07/`),阶段职责:

| 阶段 | 职责 | 关键产物 |
|---|---|---|
| 01 prompt_based | 情绪引导 prompt 生成文本 + GPT 标注筛选 | `…/labeled/sev/accepted.jsonl` |
| 02 direction_extraction | 残差对齐激活 → **diff-of-means 情绪方向** | `emo_directions_{mlp,attention}.pt` |
| 03 steer_based | 用方向向量 steering 生成,验证方向因果 | `steered_outputs.jsonl` + 准确率 |
| 04 local_components | 逐神经元/逐头对"沿情绪方向推进"的贡献 | `contrib_mean_{emo}.csv`、`head_importance_{emo}.csv` |
| 05 diff_vector | 情绪 vs 中性激活差分向量(干预用) | `emo_diff_all.npz`、`attention…/emo_diff/{emo}/L*.npy` |
| 06 circuit_integration | σ / 子层重要性 / 整合全局电路 | `global_circuit/{emo}.json`、`global_ref/v_ref_{emo}.npy` |
| 07 circuit_based | 用电路做情绪生成 + 标注,最终验证 | `circuit_steer_*_outputs.jsonl`(论文 99.41%) |

04 神经元贡献方法(脚本头注释):`beta = W_down @ v_emo[L]`,`c⁽ⁿ⁾ = H⁽ⁿ⁾ * beta`(H 为 SwiGLU 门后 last-token 激活)。

---

## 2. 自有脚本与官方 repo 的关系

本线在官方 02 阶段产物(情绪方向 `.pt`)之上做**下游应用扩展**,不重写 pipeline。

### `score_emocirc.py`(自有)
- **输入**:`data/day1_L2_events.txt`(真实第一人称叙事)+ 官方 `emo_directions_mlp.pt`。
- **逻辑**:`parse_events` → `make_windows(N=8, stride=4)` → `build_prompt`(Llama chat 模板)→ 取 `hidden_states[L+1]` 的 **last-token 残差**(默认层 11–20)→ 投影到**归一化**情绪方向 → 各层投影均值 → 每窗口 6 情绪分 → CSV。
- **关键映射**:`hidden_states[0]` 是 embedding,`hidden_states[L+1]` 对应第 L 层输出,故代码里 `hs_indices = [l+1 for l in layers]`。
- **与官方关系**:消费 02 阶段 `.pt`,不依赖 04/05/06。`directions["dirs"][emo]` 形状 `(28, 3072)`。

### `gen_viz.py`(自有)
- 读 `data/day1_L2_events.txt` + `output/scores_day1_emocirc.csv`,生成 `viz/emotion_activation_day1.html`。
- 纯字符串拼 HTML,**无外部依赖、无网络**(适合离线/上传);每情绪按自身 min-max 归一化成颜色条,内嵌 JS 做"高亮情绪+阈值"筛选。
- 强约束:`assert len(scores) == len(windows)` —— 改窗口参数(N/stride)须两脚本同步,否则断言失败。

### `quick_start.py`(官方)
- 加载 `global_circuit/{emo}.json`(电路成员)+ 05 的 `emo_diff_all.npz` / attention `L*.npy`(残差方向),对生成做电路级注入。`HF_TOKEN` 从环境变量读。

---

## 3. 数据流总览

```
官方 pipeline(repo 内):
 sev.jsonl ─01─ accepted.jsonl ─02─ emo_directions_{mlp,attn}.pt
                                    ├─03─ steering 验证
                                    ├─04─ contrib_mean_*.csv / head_importance_*.csv
                                    └─05─ emo_diff_all.npz / L*.npy ─06─ global_circuit/*.json ─07─ 99.41%

本线扩展(自有):
 day1_L2_events.txt ─(score_emocirc.py: 窗口化+last-token残差+投影 emo_directions_mlp.pt)→
   scores_day1_emocirc.csv ─(gen_viz.py)→ emotion_activation_day1.html
```

---

## 4. 关键决策与坑(务必牢记)

1. **`quick_start.py` clone 后不能直接跑**。它依赖 `*.npz`/`*.npy`(05 产物),被顶层 `.gitignore` 排除。必须先在 GPU 重跑 Step 05。`global_circuit/*.json` 是提交的,够。文档已显著标注。
2. **`.pt` 的双层 gitignore**。顶层 `.gitignore` 排 `*.pt`;但 `repo/EmotionCircuits-LLM/.gitignore` 用 `!outputs/*/02_emotion_directions/emo_directions_*.pt` 例外保留,故 repo 那两份方向 `.pt` **已提交**、clone 后存在。本目录 `directions/*.pt` 仍被排除(顶层规则,无例外)——它只是 repo 那份的**冗余副本**(MD5 一致)。`score_emocirc.py` 用 repo 路径即可,不必依赖 `directions/`。
3. **大 CSV 排除规则**:`**/04_local_components_identification/mlp_neurons/contrib_mean_*.csv`(6 个各 ~26–28MB)。重生成 = Step 04 / `1_compute_neuron_contrib.py`。注意 `head_importance_*.csv` 很小、**未**被排除。
4. **脚本默认路径是 AutoDL 服务器绝对路径**(`/root/autodl-tmp/…`、conda env `emocirc`)。他机运行必须用 `--model / --directions / --output` 覆盖。
5. **GPT 标注脚本有明文 key 占位符**:01/03/07 的 `*_label_*_with_gpt.py` 里 `api_key="Your OpenAI API Key"`。改成读 `os.environ["OPENAI_API_KEY"]`,**严禁提交真实密钥**(顶层 gitignore 已排 `.env`/`secrets.*`)。
6. **层索引 off-by-one**:`hidden_states` 含 embedding 层,层 L 的输出在 index `L+1`。`score_emocirc.py` 已处理,改投影层时别忘。
7. **窗口参数一致性**:`WIN_N=8 / WIN_STRIDE=4` 在两脚本中各定义一份,改一处要改两处。
8. **conda 环境名分歧**:官方 `environment_simple.yml` 名为 `emotion_circuits`;自有脚本注释里写的是服务器上的 `emocirc`。二者是同类环境,以实际机器为准。

---

## 5. 与其它线的关系

- **RepE-2023-Zou(旗舰线,`emotion-probe/RepE-2023-Zou/`)**:本线是其"表征读出/控制"思路在**电路级定位 + 因果干预**上的延伸。diff-of-means 情绪方向 ≈ RepE reading vector,但进一步归因到 MLP 神经元/attn 头并做电路级 steering。
- **Gemma 线(`emotion-probe/anthropic-2026-gemma-version/`)**:探针(probe)方向、另一模型族,与本线互为方法对照。

---

## 6. 验证小抄

```bash
# 确认情绪方向 .pt 结构(应为 keys=[dirs,layers,hidden,emotions,type]; 每情绪 (28,3072))
python -c "import torch; d=torch.load('repo/EmotionCircuits-LLM/outputs/llama32_3b/02_emotion_directions/emo_directions_mlp.pt', map_location='cpu', weights_only=False); print(d['emotions'], d['layers'], d['hidden'])"

# 确认哪些大文件被忽略(在仓库根运行)
git -C "$(git rev-parse --show-toplevel)" check-ignore -v \
  emotion-probe/do_llm_feels_2025-wang/directions/emo_directions_mlp.pt

# 本线产物自检:CSV 行数应 = 窗口数+1(表头),当前 501 行 = 500 窗口
wc -l output/scores_day1_emocirc.csv
```
