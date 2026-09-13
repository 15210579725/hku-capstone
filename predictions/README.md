# NextAct — Egocentric Next-Action Prediction Benchmark

Give a model 50 events of someone's day, ask it what happens next. Score how close it got.

**1,500 fixed evaluation points**, uniform across five granularity levels and two datasets:

| Dataset | Points | Per level | What makes it hard |
|---|---|---|---|
| **NextMe** | 1,000 | 200 × L1–L5 | Context and ground truth come from **different recordings**, with camera-off gaps (often days) in between |
| **EgoLife** | 500 | 100 × L1–L5 | Ground truth follows the context directly — the easier setting |

L1 is the finest granularity (a few seconds per event); L5 is the coarsest (whole activities). Every point supports **k=1** (predict the next action) and **k=10** (predict the next ten).

---

## Quick start

```bash
cd predictions

# 1. Check your setup — runs 2 points end to end, ~30 seconds
python3 run_model_benchmark.py --limit 2 --ks 1 --concurrency 2 --tag smoke

# 2. Full run: 1,500 points x k=1 and k=10
python3 run_model_benchmark.py --model deepseek-v4-flash --ks 10 1
```

Results land in `results/nextact_<tag>_*.json` and a score table is printed.

### Evaluating your own model

Any OpenAI-compatible endpoint works:

```bash
python3 run_model_benchmark.py \
  --model your-model-name \
  --base-url https://your-endpoint/v1 \
  --api-key-file ~/.config/your-provider/api_key \
  --tag your-model
```

The key can also come from `MODEL_API_KEY`, `OPENAI_API_KEY`, or `DEEPSEEK_API_KEY`.

Useful flags: `--ks 10 1` (which k to run — put 10 first, see caching below), `--concurrency 60` (model calls in flight), `--limit N` (first N points only), `--tag NAME` (names the output files).

---

## Setup

**Python deps:** `openai`, `numpy`.

**Embedding credentials** — scoring embeds every sentence, so this file must exist:

```
~/.config/hku-capstone/eval_config.json
```

```json
{
  "embedding": {
    "api_key": "<your SiliconFlow key>",
    "base_url": "https://api.siliconflow.com/v1",
    "model": "Qwen/Qwen3-Embedding-8B",
    "dimensions": 4096,
    "batch_size": 32,
    "instruction": "Represent the sentence primarily by its core action verb and the object being acted on. Prioritize verb-object alignment and downweight modifiers, tense, style, and incidental context."
  },
  "concurrency": 64
}
```

Never commit this file or paste a key into the repo.

**Source data:** `../caption-result/` (NextMe recordings) and `../dataset/egolife/` (EgoLife). Both are in this repository.

---

## How scoring works

**Soft Edit Distance over sentence embeddings.** Predicted and ground-truth action sequences are aligned with an edit distance whose substitution cost is `1 - cosine(embedding_gt, embedding_pred)`; insertion and deletion cost 1.

```
raw        = 1 - SED / max(len_gt, len_pred)
normalized = max(0, (raw - random_baseline) / (1 - random_baseline))
```

**Read the normalized number.** `0` means no better than random predictions; `1` means perfect. Per-level random baselines live in `nextme_benchmark/baselines.json`.

**Best-of-3.** The model returns three candidate trajectories; the one with the highest raw score is kept. This is deliberate and applies to every model — scoring only the first candidate lowers scores by roughly 35%, so never compare a best-of-1 number against these.

---

## Reference scores

`deepseek-v4-flash`, best-of-3, normalized. The baseline repeats the last *k* context events verbatim as its prediction — beat it or your model has learned nothing.

| Dataset | Level | k=1 model | k=1 baseline | k=10 model | k=10 baseline |
|---|---|---|---|---|---|
| NextMe | L1 | 0.276 | 0.165 | 0.155 | 0.117 |
| NextMe | L2 | 0.268 | 0.143 | 0.154 | 0.117 |
| NextMe | L3 | 0.275 | 0.132 | 0.157 | 0.103 |
| NextMe | L4 | 0.349 | 0.177 | 0.137 | 0.084 |
| NextMe | L5 | 0.331 | 0.152 | 0.106 | 0.052 |
| **NextMe** | **ALL** | **0.300** | **0.154** | **0.143** | **0.096** |
| EgoLife | L1 | 0.394 | 0.254 | 0.241 | 0.222 |
| EgoLife | L2 | 0.399 | 0.256 | 0.260 | 0.234 |
| EgoLife | L3 | 0.464 | 0.321 | 0.267 | 0.242 |
| EgoLife | L4 | 0.372 | 0.247 | 0.234 | 0.194 |
| EgoLife | L5 | 0.368 | 0.256 | 0.188 | 0.162 |
| **EgoLife** | **ALL** | **0.399** | **0.267** | **0.238** | **0.211** |
| **Overall** | **ALL** | **0.333** | **0.192** | **0.176** | **0.135** |

Three things this table says: the model beats the baseline in every cell; its lead at k=1 (+0.11 to +0.18) largely evaporates at k=10 (+0.02 to +0.06), so **long-horizon prediction is the open problem**; and EgoLife scores higher than NextMe throughout because its ground truth directly follows its context, not because models handle it better — the gain over baseline is nearly identical on both.

To reproduce the baseline column:

```bash
python3 baselines/run_repeat_baseline.py
```

---

## Editing the prompt

`prompts/predict_prompt.txt` — a single template, deliberately shared by both datasets so they are measured under identical instructions. One rule covers the language difference ("write in the same language as the context events"), which keeps EgoLife in Chinese and NextMe in English without branching.

The template receives: `{k}`, `{context_segments}`, `{n_context_segments}`, `{target_segments}`, `{date_prefix}`, `{action_example}`. Literal JSON braces must be escaped as `{{` / `}}`.

Every prompt states both the **context time windows** and the **target prediction windows**, and tells the model that the gaps between them are unrecorded.

---

## Cost and runtime

A full 1,500-point run takes roughly **8 minutes of model calls** at `--concurrency 60`, plus embedding time for scoring.

Embeddings are cached to `.emb_cache/embeddings.sqlite` and reused across runs and across *k*. Two consequences worth knowing:

- **Run k=10 before k=1.** The k=1 text set is a strict subset of k=10's, so it then costs zero embedding calls.
- **Re-scoring is nearly free.** Only genuinely new sentences hit the API.

If the embedding API returns `IncompleteRead`, that is normal — responses are large and the client retries up to 6 times. Raise `--emb-concurrency` to go faster, but leave the internal batch size at 8; larger batches make truncation worse.

---

## Repository layout

```
predictions/
├── run_model_benchmark.py     Main entry — evaluate a model
├── prompts/predict_prompt.txt The prompt (shared by both datasets)
├── baselines/                 Reference baselines
├── results/                   Output; results/legacy/ holds superseded runs
└── archive/                   Superseded scripts, kept for reference only
```

Internal modules you should not normally need to touch: `nextact_points.py` (builds the 1,500 points), `prompt_builder.py` (fills the template), `emb_cache.py` (embedding cache), `nextme_benchmark/` (scorer, benchmark points, random baselines).

**The evaluation points are fixed.** Do not re-sample them — every published number depends on this exact set.

Maintainers: see `CLAUDE.md` for construction details, scoring invariants, and known pitfalls.
