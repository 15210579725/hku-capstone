#!/root/miniconda3/bin/python3.10
"""
Extract activations from Gemma 4 E4B for 8 emotions + neutral.
Saves per-layer mean activations for each emotion.
"""

import os

# Set HuggingFace cache to the location where model is already downloaded
os.environ["HF_HOME"] = "/autodl-fs/data/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/autodl-fs/data/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/autodl-fs/data/huggingface_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Don't attempt network access

import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============ CONFIG ============
MODEL_ID = "google/gemma-4-E4B"
EMOTIONS = ["sad", "curious", "excited", "bored", "anxious", "happy", "tormented", "angry"]
STORIES_DIR = "/root/data/stories"
NEUTRAL_PATH = "/root/data/neutral_stories/neutral_stories.json"
OUTPUT_DIR = "/root/gemma-probes-capstone/activations"
BATCH_SIZE = 32
MAX_LENGTH = 512
MIN_TOKEN_OFFSET = 50  # Skip first 50 tokens
NUM_LAYERS = 42
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_emotion_stories(emotion):
    """Load stories for a given emotion."""
    path = os.path.join(STORIES_DIR, f"{emotion}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["stories"]


def load_neutral_stories():
    """Load neutral stories."""
    with open(NEUTRAL_PATH, "r") as f:
        data = json.load(f)
    return data["stories"]


def get_output_path(emotion):
    return os.path.join(OUTPUT_DIR, f"en_{emotion}.pt")


def extract_activations(model, tokenizer, stories, device):
    """
    Extract mean activations (skipping first MIN_TOKEN_OFFSET tokens) for each layer.
    Returns dict: layer_idx (int) -> tensor [N_stories, hidden_dim]
    """
    # Storage for accumulated activations per layer
    layer_activations = {i: [] for i in range(NUM_LAYERS)}

    # Hook storage
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is hidden states
            hidden = output[0]  # [batch, seq_len, hidden_dim]
            captured[layer_idx] = hidden.detach()
        return hook_fn

    # Register hooks
    for i, layer in enumerate(model.model.language_model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    num_stories = len(stories)
    num_batches = (num_stories + BATCH_SIZE - 1) // BATCH_SIZE

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, num_stories)
        batch_texts = stories[batch_start:batch_end]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        attention_mask = inputs["attention_mask"]  # [batch, seq_len]

        with torch.no_grad():
            _ = model(**inputs)

        # Process captured activations
        for layer_idx in range(NUM_LAYERS):
            hidden = captured[layer_idx]  # [batch, seq_len, hidden_dim]
            batch_size_actual = hidden.shape[0]
            seq_len = hidden.shape[1]

            # Create mask: skip first MIN_TOKEN_OFFSET tokens AND respect attention mask
            valid_mask = attention_mask.clone()  # [batch, seq_len]
            if seq_len > MIN_TOKEN_OFFSET:
                valid_mask[:, :MIN_TOKEN_OFFSET] = 0
            else:
                # If sequence is shorter than offset, use all tokens (edge case)
                pass

            # valid_mask: [batch, seq_len] -> [batch, seq_len, 1]
            mask_expanded = valid_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

            # Mean over valid tokens per sample
            sum_hidden = (hidden.float() * mask_expanded).sum(dim=1)  # [batch, hidden_dim]
            count = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
            mean_hidden = sum_hidden / count  # [batch, hidden_dim]

            layer_activations[layer_idx].append(mean_hidden.cpu())

        # Clear captured
        captured.clear()

        # Progress
        elapsed = time.time() - start_time
        if batch_idx > 0:
            avg_time = elapsed / (batch_idx + 1)
            remaining = avg_time * (num_batches - batch_idx - 1)
            print(f"  Batch {batch_idx+1}/{num_batches} | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
        else:
            print(f"  Batch {batch_idx+1}/{num_batches} | First batch: {elapsed:.1f}s | "
                  f"Est. total: {elapsed * num_batches:.1f}s")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate all batches per layer -> [N_stories, hidden_dim]
    result = {}
    for layer_idx in range(NUM_LAYERS):
        result[layer_idx] = torch.cat(layer_activations[layer_idx], dim=0)  # float32

    return result


def main():
    print("=" * 60)
    print("Gemma 4 E4B Activation Extraction - 8 Emotions + Neutral")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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

    # Verify layer access
    layers = model.model.language_model.layers
    print(f"Number of layers: {len(layers)}")
    assert len(layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {len(layers)}"

    # Process emotions
    all_tasks = EMOTIONS + ["neutral"]
    total_start = time.time()

    for task_idx, emotion in enumerate(all_tasks):
        output_path = get_output_path(emotion)

        # Resume logic
        if os.path.exists(output_path):
            print(f"\n[{task_idx+1}/{len(all_tasks)}] SKIP {emotion} (already exists: {output_path})")
            continue

        print(f"\n{'='*60}")
        print(f"[{task_idx+1}/{len(all_tasks)}] Processing: {emotion}")
        print(f"{'='*60}")

        # Load stories
        if emotion == "neutral":
            stories = load_neutral_stories()
        else:
            stories = load_emotion_stories(emotion)

        print(f"  Loaded {len(stories)} stories")

        # Extract
        activations = extract_activations(model, tokenizer, stories, device)

        # Report shape
        sample_tensor = activations[0]
        print(f"  Output shape per layer: {sample_tensor.shape}")
        print(f"  Hidden dim: {sample_tensor.shape[1]}")

        # Save
        torch.save(activations, output_path)
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"  Saved: {output_path} ({file_size:.1f} MB)")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    # List outputs
    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".pt"):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath) / (1024**2)
            print(f"  {f} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
