"""
Extract residual stream activations for the 1200 neutral stories across all 42
Gemma-4-E4B layers. For each story we average the residual stream over token
positions 50 onwards (matching the emotion-story extraction and the paper's
methodology). Produces a single (1200, 42, 2560) tensor.

Adapted from the official extraction/extract_neutral_story_activations.py.
INTERFACE-ONLY changes vs. the official script:
  * INPUT_PATH points at this project's neutral_stories.json (the official
    script read a neutral_stories.csv with a "story" column; our data is JSON
    of the form {"stories": [...]}). The story list is read with json instead
    of pandas — same list of strings, just a different container.
  * OUTPUT_PATH repointed to this project's path.
  * HF_HUB_CACHE / HF_HUB_OFFLINE so the locally-cached model loads offline.

The main algorithm is UNCHANGED: forward hooks on the language-model layers,
right padding, TOKEN_SKIP=50, BATCH_SIZE=32, max_length=512, masked mean over
real token positions >= 50, saved as a (N, 42, 2560) float32 tensor.
"""

import os
os.environ.setdefault("HF_HUB_CACHE", "/autodl-fs/data/huggingface_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B"
INPUT_PATH = Path("/root/data/neutral_stories/neutral_stories.json")
OUTPUT_PATH = Path("/autodl-fs/data/official_run/neutral_activations.pt")
TOKEN_SKIP = 50  # start averaging from this token position
NUM_LAYERS = 42
BATCH_SIZE = 32


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def main():
    print(f"Loading tokenizer + model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = get_layers(model)
    assert len(layers) == NUM_LAYERS, f"expected {NUM_LAYERS} layers, got {len(layers)}"

    print(f"Reading stories from {INPUT_PATH}...")
    stories = json.loads(INPUT_PATH.read_text())["stories"]
    print(f"  {len(stories)} stories, batch size {BATCH_SIZE}")

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = act.detach()
        return hook_fn

    hooks = [layers[i].register_forward_hook(make_hook(i)) for i in range(NUM_LAYERS)]

    all_means = torch.zeros(len(stories), NUM_LAYERS, 2560, dtype=torch.float32)

    try:
        for start in tqdm(range(0, len(stories), BATCH_SIZE), desc="Extracting"):
            batch = stories[start : start + BATCH_SIZE]
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                model(**inputs)

            # Build a mask that's 1 for real tokens at positions >= TOKEN_SKIP, else 0.
            mask = inputs["attention_mask"].clone()  # (B, L)
            mask[:, :TOKEN_SKIP] = 0
            mask = mask.to(torch.float32)
            counts = mask.sum(dim=1).clamp(min=1)  # (B,)

            for layer_idx in range(NUM_LAYERS):
                act = captured[layer_idx].float()  # (B, L, 2560)
                # Masked mean over sequence dim.
                summed = (act * mask.unsqueeze(-1)).sum(dim=1)  # (B, 2560)
                means = summed / counts.unsqueeze(-1)  # (B, 2560)
                all_means[start : start + len(batch), layer_idx] = means.cpu()
    finally:
        for h in hooks:
            h.remove()

    print(f"Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_means, OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"Saved ({size_mb:.1f} MB). Shape: {tuple(all_means.shape)}")


if __name__ == "__main__":
    main()
