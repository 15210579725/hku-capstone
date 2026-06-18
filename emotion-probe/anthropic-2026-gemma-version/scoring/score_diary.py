"""
Score diary segments using pre-computed emotion vectors.

For each segment:
1. Concatenate events into one text
2. Run through Gemma 4 E4B, capture layer 23 activations
3. For each token: center activation and compute dot product with 8 emotion vectors
4. Output per-token scores
"""

import os

# Set HuggingFace cache
os.environ["HF_HOME"] = "/autodl-fs/data/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/autodl-fs/data/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/autodl-fs/data/huggingface_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============ CONFIG ============
MODEL_ID = "google/gemma-4-E4B"
VECTORS_PATH = "/root/gemma-probes-capstone/en_vectors.pt"
SEGMENTS_PATH = "/root/gemma-probes-capstone/selected_segments.json"
OUTPUT_PATH = "/root/gemma-probes-capstone/diary_scores.json"
TARGET_LAYER = 23
MAX_SEQ_LEN = 2048
# ================================


def main():
    print("=" * 60)
    print("Diary Segment Scoring - Gemma 4 E4B Layer 23")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load emotion vectors
    print(f"\nLoading vectors from {VECTORS_PATH}...")
    vec_data = torch.load(VECTORS_PATH, map_location="cpu")
    vectors = vec_data["vectors"]  # [8, 42, 2560]
    emotions = vec_data["emotions"]  # ['sad', 'curious', ...]
    global_means = vec_data["global_means"]  # [42, 2560]

    # Extract layer 23 vectors and mean
    layer_vectors = vectors[:, TARGET_LAYER, :].float()  # [8, 2560]
    layer_mean = global_means[TARGET_LAYER].float()  # [2560]

    print(f"  Emotions: {emotions}")
    print(f"  Layer vectors shape: {layer_vectors.shape}")
    print(f"  Layer mean norm: {layer_mean.norm().item():.4f}")

    # Load segments
    print(f"\nLoading segments from {SEGMENTS_PATH}...")
    with open(SEGMENTS_PATH, "r") as f:
        segments = json.load(f)
    print(f"  Loaded {len(segments)} segments")

    # Load model
    print(f"\nLoading model: {MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Register hook on layer 23
    captured = {}

    def hook_fn(module, input, output):
        # Gemma 4 E4B layers output a plain tensor, not a tuple
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        captured["hidden"] = hidden.detach()  # [batch, seq_len, hidden_dim]

    hook = model.model.language_model.layers[TARGET_LAYER].register_forward_hook(hook_fn)

    # Move vectors to device
    layer_vectors = layer_vectors.to(device)
    layer_mean = layer_mean.to(device)

    # Process segments
    results = []
    total_start = time.time()

    for seg_idx, segment in enumerate(segments):
        seg_id = segment.get("id", f"seg_{seg_idx:02d}")
        events = segment["events"]

        # Concatenate events
        text = "\n".join(events)

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)

        seq_len = inputs["input_ids"].shape[1]

        # Forward pass
        with torch.no_grad():
            _ = model(**inputs)

        # Get activations
        hidden = captured["hidden"]  # [1, seq_len, 2560]
        hidden = hidden[0].float()  # [seq_len, 2560]

        # Center activations
        hidden_centered = hidden - layer_mean.unsqueeze(0)  # [seq_len, 2560]

        # Compute scores: dot product with each emotion vector
        # layer_vectors: [8, 2560], hidden_centered: [seq_len, 2560]
        scores = hidden_centered @ layer_vectors.T  # [seq_len, 8]

        # Decode tokens
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Build output
        scores_dict = {}
        for i, emotion in enumerate(emotions):
            scores_dict[emotion] = scores[:, i].cpu().tolist()

        result = {
            "id": seg_id,
            "expected_emotions": segment.get("expected_emotions", []),
            "num_tokens": seq_len,
            "tokens": tokens,
            "scores": scores_dict,
        }
        results.append(result)

        # Progress
        elapsed = time.time() - total_start
        print(f"  [{seg_idx+1}/{len(segments)}] {seg_id}: {seq_len} tokens | {elapsed:.1f}s elapsed")

        # Clear
        captured.clear()

    # Remove hook
    hook.remove()

    # Save results
    print(f"\nSaving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(OUTPUT_PATH) / (1024**2)
    print(f"  Saved: {OUTPUT_PATH} ({file_size:.2f} MB)")

    # Summary
    total_time = time.time() - total_start
    total_tokens = sum(r["num_tokens"] for r in results)
    print(f"\nDone! {len(results)} segments, {total_tokens} total tokens in {total_time:.1f}s")

    # Print sample scores for first segment
    print("\n--- Sample: First segment top scores ---")
    first = results[0]
    print(f"  ID: {first['id']}, Tokens: {first['num_tokens']}")
    print(f"  Expected: {first['expected_emotions']}")
    # Mean score per emotion
    print("  Mean scores per emotion:")
    for emotion in emotions:
        mean_score = sum(first["scores"][emotion]) / len(first["scores"][emotion])
        print(f"    {emotion:>10}: {mean_score:.4f}")


if __name__ == "__main__":
    main()
