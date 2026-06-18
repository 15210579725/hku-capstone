"""用 EmotionCircuits-LLM 论文方法对 day1 L2 事件打情绪分。

用法（在服务器上）：
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate emocirc
  python /root/autodl-tmp/emotion-circuits-L1/score_emocirc.py

流程：
  1. 解析 events.txt → 切 N=8/stride=4 窗口
  2. 加载预提取的 MLP 情绪方向向量（emo_directions_mlp.pt）
  3. 加载 Llama-3.2-3B-Instruct，取中间层 last-token 残差
  4. 投影到 6 情绪方向 → 输出 CSV
"""
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
WIN_N = 8
WIN_STRIDE = 4


def parse_events(path):
    pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2})\s*->\s*(\d{2}:\d{2}:\d{2})\]\s*(.+)")
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                t_start, t_end, text = m.groups()
                events.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "text": text.strip(),
                })
    return events


def time_to_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def make_windows(events, n=WIN_N, stride=WIN_STRIDE):
    windows = []
    for i in range(0, max(1, len(events) - n + 1), stride):
        chunk = events[i:i + n]
        text = " / ".join(e["text"] for e in chunk)
        windows.append({
            "win_idx": len(windows),
            "t_start": time_to_sec(chunk[0]["t_start"]),
            "t_end": time_to_sec(chunk[-1]["t_end"]),
            "n_events": len(chunk),
            "text": text,
        })
    return windows


def build_prompt(text, emotion_zh="情绪"):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are an assistant skilled at detecting emotions in first-person narratives.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Please assess the emotional intensity in the following experience:\n{text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@torch.no_grad()
def extract_residuals(model, tokenizer, texts, layers, batch_size=8, device="cuda"):
    """提取指定层的 last-token 残差。返回 {layer_idx: [N, hidden_dim]}"""
    all_residuals = {l: [] for l in layers}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

        for layer_idx in layers:
            hs = hidden_states[layer_idx]  # (batch, seq, hidden)
            seq_lens = inputs.attention_mask.sum(dim=1)  # (batch,)
            last_tokens = []
            for b in range(hs.shape[0]):
                last_tokens.append(hs[b, seq_lens[b] - 1, :].cpu().numpy())
            all_residuals[layer_idx].extend(last_tokens)

        if (i // batch_size) % 10 == 0:
            print(f"  batch {i // batch_size}/{(len(texts) - 1) // batch_size + 1}", flush=True)

    return {l: np.stack(v) for l, v in all_residuals.items()}


def score_with_directions(residuals, directions, layers_to_use=None):
    """
    residuals: {layer_idx: [N, hidden_dim]}
    directions: dict from emo_directions_mlp.pt — dirs[emotion] shape [num_layers, hidden_dim]

    对每个情绪，选最优层（论文用中间层 ~11-20），投影到方向向量。
    """
    dirs = directions["dirs"]
    num_layers = directions["layers"]

    if layers_to_use is None:
        layers_to_use = list(range(11, 21))

    scores_per_emotion = {}
    for emo in EMOTIONS:
        d = dirs[emo]  # [num_layers, hidden_dim]
        layer_scores = []
        for layer_idx in layers_to_use:
            if layer_idx not in residuals:
                continue
            h = residuals[layer_idx]  # [N, hidden_dim]
            v = d[layer_idx]  # [hidden_dim]
            v_norm = v / (np.linalg.norm(v) + 1e-12)
            proj = h @ v_norm  # [N]
            layer_scores.append(proj)

        if layer_scores:
            scores_per_emotion[emo] = np.mean(layer_scores, axis=0)
        else:
            scores_per_emotion[emo] = np.zeros(len(next(iter(residuals.values()))))

    return scores_per_emotion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str,
                        default="/root/autodl-tmp/emotion-circuits-L1/data/day1_L2_events.txt")
    parser.add_argument("--model", type=str,
                        default="/root/autodl-tmp/Llama-3.2-3B-Instruct")
    parser.add_argument("--directions", type=str,
                        default="/root/autodl-tmp/emotion-circuits-L1/EmotionCircuits-LLM/outputs/llama32_3b/02_emotion_directions/emo_directions_mlp.pt")
    parser.add_argument("--output", type=str,
                        default="/root/autodl-tmp/emotion-circuits-L1/output/scores_day1_emocirc.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layers", type=str, default="11-20",
                        help="投影层范围，如 11-20")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 解析层范围
    l_start, l_end = map(int, args.layers.split("-"))
    layers_to_use = list(range(l_start, l_end + 1))
    # hidden_states index: 0=embedding, 1=layer0, ..., 28=layer27
    # 对应 layer_idx 在 hidden_states 中的索引是 layer+1
    hs_indices = [l + 1 for l in layers_to_use]

    print(f"[1/4] 解析事件文件 {args.events}", flush=True)
    events = parse_events(args.events)
    print(f"  共 {len(events)} 条事件", flush=True)

    print(f"[2/4] 切窗口 N={WIN_N} stride={WIN_STRIDE}", flush=True)
    windows = make_windows(events)
    print(f"  共 {len(windows)} 个窗口", flush=True)

    print(f"[3/4] 加载模型 {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    prompts = [build_prompt(w["text"]) for w in windows]
    print(f"  提取层 {layers_to_use} 的 last-token 残差...", flush=True)
    residuals = extract_residuals(model, tokenizer, prompts, hs_indices,
                                  batch_size=args.batch_size)
    # 重映射 key: hs_index → layer_idx
    residuals_mapped = {l: residuals[l + 1] for l in layers_to_use if (l + 1) in residuals}

    print(f"[4/4] 加载情绪方向向量 {args.directions}", flush=True)
    directions = torch.load(args.directions, map_location="cpu", weights_only=False)
    print(f"  方向向量 emotions={directions.get('emotions')} layers={directions.get('layers')} hidden={directions.get('hidden')}", flush=True)

    scores = score_with_directions(residuals_mapped, directions, layers_to_use)

    # 构建 DataFrame
    rows = []
    for i, w in enumerate(windows):
        row = {
            "win_id": f"d1_w{i}",
            "day": 1,
            "win_idx": w["win_idx"],
            "t_start": w["t_start"],
            "t_end": w["t_end"],
            "n_events": w["n_events"],
        }
        for emo in EMOTIONS:
            row[emo] = float(scores[emo][i])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\n完成！结果保存到 {args.output}", flush=True)
    print(df[EMOTIONS].describe().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
