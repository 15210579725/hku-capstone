# Dataset: EgoLife & Next-Me Unified Action Prediction Benchmark

Two egocentric (first-person) datasets merged into a single evaluation framework for action prediction research.

## Directory Structure

```
dataset/
├── egolife/                        # EgoLife dataset (6 participants × 7 days × 5 levels)
│   ├── A1_JAKE/L{1-5}/day{1-7}/events.txt
│   ├── A2_ALICE/...
│   ├── A3_TASHA/...
│   ├── A4_LUCIA/...                # Claude-Opus version (replaces ds-flash)
│   ├── A5_KATRINA/...
│   └── A6_SHURE/...
├── nextme/                         # Next-Me dataset (24 recording sessions)
│   ├── {recording_id}/events.txt
│   └── ...
├── benchmark_1k.jsonl              # 1000 evaluation test points (fixed index)
├── data_index.json                 # Full data inventory
├── run_benchmark.py                # Prediction runner (DeepSeek API)
├── build_index.py                  # Index generation script
└── convert_nextme.py               # Caption → events.txt converter
```

## Data Sources

### EgoLife (800 points)
- **Source**: [EgoLife dataset](https://huggingface.co/datasets/lmms-lab/EgoLife) — 6 participants cohabiting for 7 days
- **Event levels**: L1 (atomic 1-3s) → L2 (small events) → L3 (tasks) → L4 (activity segments) → L5 (half-day summary)
- **Language**: Chinese
- **Format**: `[HH:MM:SS -> HH:MM:SS] 第一人称描述`
- **A4_LUCIA** uses the Claude-Opus annotated version; other 5 participants use DeepSeek-Flash annotations

### Next-Me (200 points)
- **Source**: First-person Aria glasses recordings of the dataset author
- **Sessions**: 24 recordings across April–May 2026, ~2107 action segments
- **Language**: English (with occasional Chinese speech transcriptions)
- **Format**: `[HH:MM:SS -> HH:MM:SS] action description`
- **Converted from**: structured JSONL captions (video + gaze + audio → dense captioning pipeline)

## Benchmark: benchmark_1k.jsonl

1000 fixed evaluation points for standardized action prediction evaluation.

### Sampling
- **EgoLife**: 800 points, proportionally distributed across 6 participants × 7 days (L1 level)
- **Next-Me**: 200 points from 11 sessions with sufficient coverage (≥53 events)
- **Seed**: 42 (deterministic)

### Each test point contains:
```json
{
  "id": "eval_0000",
  "source": "egolife",
  "participant": "A1_JAKE",
  "session": "day1",
  "file": "egolife/A1_JAKE/L1/day1/events.txt",
  "cutoff_line": 500,
  "context_start_line": 450,
  "n_context": 50,
  "n_gt": 3,
  "context": ["[11:24:00 -> 11:24:02] ...", ...],
  "ground_truth": ["[11:24:02 -> 11:24:05] ...", ...]
}
```

## Quick Start

### 1. Run predictions (DeepSeek V4 Flash)
```bash
cd dataset/

# Test with 10 points
python3 run_benchmark.py --n 10

# Full 1k benchmark
python3 run_benchmark.py --n 1000 --workers 10

# Custom model
python3 run_benchmark.py --n 10 --model deepseek-chat --api-key YOUR_KEY
```

### 2. Evaluate predictions
```bash
# Install: pip install numpy
python3 ../metric/eval_kit/evaluate.py \
  -i predictions.jsonl \
  -c ../metric/eval_kit/config.json \
  -m both
```

### 3. Use your own model
Output format (`predictions.jsonl`):
```json
{"id": "eval_0000", "prediction": ["action1", "action2", "action3"], "ground_truth": ["gt1", "gt2", "gt3"]}
```

## Evaluation Metrics

Uses `metric/eval_kit/`:
- **SED** (Soft Edit Distance): alignment-based sequence similarity, no API needed
- **Embedding**: Qwen3-Embedding-8B cosine similarity (SiliconFlow API)
- **LLM Judge**: 0-9 semantic scoring (Gemini-3.7-Flash via RightCode)

## Data Statistics

| Dataset | Participants | Sessions | L1 Events | Benchmark Points |
|---------|-------------|----------|-----------|-----------------|
| EgoLife | 6 | 42 (7 days each) | ~620K | 800 |
| Next-Me | 1 | 24 recordings | ~2.1K | 200 |
| **Total** | **7** | **66** | **~622K** | **1000** |
