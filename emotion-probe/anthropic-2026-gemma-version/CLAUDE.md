# CLAUDE.md — Gemma 情绪探针线(开发者 / AI 指南)

面向后续接手的开发者或 AI。先读 `README.md` 了解怎么跑;本文件讲**架构、数据流、关键决策与坑**。

---

## 1. 这条线是什么

在 `google/gemma-4-E4B`(42 层,hidden 2560)上复现 Anthropic 2026 论文
*Emotion Concepts and their Function in a Large Language Model* 的「expression probe」,
并扩展到 **8 情绪**:`sad, curious, excited, bored, anxious, happy, tormented, angry`。
方法是 RepE 系的 **difference-of-means + 中性 PCA 去混杂**:

```
每情绪故事(各 1200) ─抽隐藏态─► 每情绪逐层 mean
                                      │  raw_dir = emotion_mean − global_mean
中性故事(1200)     ─抽隐藏态─► 中性激活 ─SVD─► 取累计方差≥50% 的 top-k PC
                                      │  dir = raw_dir − PCₖ·(PCₖᵀ·raw_dir)   (投影掉混杂)
                                      ▼  vec = dir / ‖dir‖
                              每层每情绪一个单位方向向量
日记文本 ─抽隐藏态(中心化)─► score = (h − global_mean) · vec   逐 token 打分
```

---

## 2. 两条并行子流水线(重要)

本目录里其实有**两套等价但独立**的实现,产物是两个不同的 `.pt`,别混用:

| | 快速自建线 | 忠实复现线 |
|---|---|---|
| 抽激活 | `extraction/extract_8emotions.py` | `official_run_staging/extract_story_activations.py` + `extract_neutral_story_activations.py` |
| 算向量 | `extraction/compute_vectors.py` | `official_run_staging/compute_expression_vectors.py` |
| 产物 | `vectors/en_vectors.pt` | `vectors/emotion_vectors_all_layers.pt` |
| 产物结构 | `{vectors:[8,42,2560](tensor), emotions, global_means:[42,2560], neutral_pcs:list}` | `{vectors:{layer:{emotion:tensor}}, global_means:{layer:tensor}, pcs:{layer:tensor}}` |
| 下游 | `scoring/score_diary.py`(**layer 23**,直接点积) | `gemma-emotional-probes-main/visualise.py` Flask + `visualization/batch_diary.py`(**layer 28**) |

- 两条线算法实质相同(diff-of-means + 中性 PCA 50% 方差去混杂),差别只在**张量组织方式 / 落地层 / 下游消费方式**。
- `official_run_staging/` 是「对 `gemma-emotional-probes-main/extraction/` 官方脚本只做接口级改动
  (改路径、限定 8 情绪、`padding_side='right'`、离线缓存)、**不动算法**」的运行版。脚本头部 docstring 写明了每一处改动,改它时务必保持这一纪律。
- 选层结论:经验上 **layer 28 优于 layer 23**(见组内 memory)。`visualise.py` 里 `TARGET_LAYER=23` 只是默认,
  `batch_diary.py` 实际传 `layer=28`;`score_diary.py` 仍写死 23,要复刻最优结果应改成 28。

---

## 3. 各目录职责

- `scripts/` — 数据准备。下载/生成英文故事、英↔中翻译、日记中→英。只有这层调外部 API。
- `extraction/` — 快速线:抽隐藏态 + 算向量。
- `official_run_staging/` — 忠实复现线的抽取+计算(官方算法)。
- `scoring/` — 用 `en_vectors.pt` 给日记逐 token 打分。
- `visualization/` — 批处理 + 静态页 + 图表;`batch_diary.py` 把片段喂给官方 Flask 可视化器。
- `data/` — 故事(8×1200 en/zh)、中性故事、25 个日记片段(中/英)。
- `vectors/` — 两个大向量 `.pt`(被 .gitignore)+ 两个小 stats JSON(已提交)。
- `gemma-emotional-probes-main/` — 官方原仓库,**只读参考 + 提供 `visualise.py`**,不要在里面改业务代码。

---

## 4. 关键决策与坑

1. **模型名义 vs 实际**:课题描述称「Gemma 线」,但代码里 `MODEL_ID` 一律是
   `google/gemma-4-E4B`(非 Gemma-2-9B)。所有维度(42 层 / 2560 / `model.model.language_model.layers`
   这种多模态包装下的层路径)都按 E4B 写死。换模型要同步改这三处。

2. **脚本路径全是服务器绝对路径,且指向旧目录名**:
   - GPU 脚本写死 `/root/gemma-probes-capstone/...`、`/autodl-fs/data/...`;
   - `scripts/` 多数写死 `/Users/mac/Desktop/hku capstone/gemma-probes/...`(注意是 `gemma-probes`,
     **不是**当前 `emotion-probe/anthropic-2026-gemma-version/`);
   - 只有 `download_hf_stories.py` 用了 `PROJECT_ROOT` 相对路径。
   - ⇒ clone 后**必须改脚本顶部路径常量**才能跑通,别指望开箱即用。

3. **数据格式两套,容易踩**:
   - 仓库 `data/stories_en/<emotion>.json` 是**纯 list[str]**(`download_hf_stories.py` 的输出格式);
   - 但 `extract_8emotions.py` 期望 `/root/data/stories/<emotion>.json` 里是 `{"stories":[...]}`,
     `official_run_staging/extract_story_activations.py` 期望 `{"emotion":..., "stories":[...]}`。
   - ⇒ 上服务器抽激活前,数据被**重新包装过**。复现时要么改抽取脚本的读取逻辑,要么先把 list 包成 dict。

4. **去混杂用的是「中性文本 PCA」而非 mean-only**:对中性激活做 SVD,投影掉累计方差 ≥50% 的主成分,
   目的是去掉「文本长度/句法/位置」等与情绪无关的共同方向。阈值 `VARIANCE_THRESHOLD=0.50` 两条线一致。

5. **token offset 50**:抽故事/中性激活时跳过前 50 个 token 再做 mean(论文方法「token 50 起平均」)。
   配合 `padding_side='right'`——offset 切片 `act[j, 50:seq_len]` 只在右 padding 下才对,而 gemma 分词器**默认左 padding**,
   所以 staging 脚本显式设右 padding。改 padding 会悄悄算错向量。

6. **z-score 归一化**:`vectors/diary_zscore_stats.json` 存了 layer 28 上 zh/en 各情绪的 μ/σ,
   `neutral_stats.json` 存逐层统计。可视化的 `_z` 变体(`diary_html/out_*_z`、`summary_z.json`)用它把
   原始投影分标准化,便于跨情绪/中英比较。这两个 JSON 是离线算好直接提交的,**没有对应的生成脚本在本目录**——
   要重算需自行写(用中性激活的逐 token 分布)。

7. **curiosity 是自造数据**:HF 数据集只有 7 情绪,curiosity 用 `gpt-5.4` 现生成,
   prompt 明确要求「不出现 curious/curiosity 等词,只靠行为/提问/探索体现」,以免探针学到词面而非语义。

8. **score_diary.py 的 hook 兼容性**:E4B 的 decoder layer 输出有时是 tensor、有时是 tuple,
   hook 里做了 `isinstance(output, tuple)` 分支,迁移到别的模型时注意这点。

---

## 5. 数据流(端到端)

```
HF stories.parquet ─┐
gpt-5.4(curiosity)─┼─► data/stories_en/*.json (8×1200, list)
                    └─► (翻译) data/stories_zh/*.json
neutral parquet ──────► data/neutral/neutral_en.json ──(翻译)─► neutral_zh.json

[GPU] stories+neutral ─► 激活(*.pt, 被忽略) ─► vectors/en_vectors.pt
                                            └► vectors/emotion_vectors_all_layers.pt

selected_segments.json ─(翻译)─► selected_segments_en.json
        │
        ├─[GPU 快速线] score_diary.py(layer23) ─► scoring/diary_scores.json ─► visualization/data/*
        └─[GPU 复现线] visualise.py(Flask) ←POST← batch_diary.py(layer28, 中+英)
                                              └─► visualization/diary_html/{out_zh,out_en,*_z}/ + summary*.json
```

---

## 6. 与其他子线的关系

- **RepE 旗舰线**:本线是旗舰 RepE(reading-vector / difference-of-means)方法的一个**模型特化 + 多情绪**实例
  ——把同一套「对比数据→方向向量→投影打分」搬到 Gemma 4 E4B,并新增 curiosity 这一情绪。
- **`do_llm_feels` 线**:同属「LLM 有没有情感」大课题下的并行探索;那条线侧重行为/表征层面的「是否有感受」问法,
  本线提供**机制可解释性侧的方向探针证据**(情绪在残差流里是否线性可分、能否迁移到中英日记)。
- 三线接口约定见组内 memory(`emotion-probe-agent-orchestration`);注意历史上有过 `src` 重名冲突,
  本目录已自成体系(scripts/extraction/scoring/visualization),合流时按目录名而非 `src` 引用。
</content>
