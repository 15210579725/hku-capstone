"""Extract activations at all 42 layers for the project's 8 emotions from Gemma 4 E4B.

Adapted from the official extraction/extract_story_activations.py.
INTERFACE-ONLY changes vs. the official script:
  * EMOTIONS restricted to the project's 8 emotions. (The official script
    derived the full emotion set by globbing every *.json in DATA_DIR; here we
    only keep the 8 we care about — same parsing, just filtered.)
  * DATA_DIR / ACTIVATIONS_DIR repointed to this project's real paths.
  * HF_HUB_CACHE / HF_HUB_OFFLINE so the locally-cached 15 GB model loads
    without any network access.
  * tokenizer.padding_side = "right"  -- the official *neutral*-story script
    sets this explicitly, and the offset-50 slice  act[j, 50:seq_len]  is only
    correct under right padding (the gemma-4 tokenizer DEFAULTS TO LEFT). This
    keeps the story extraction consistent with the neutral extraction and with
    the paper's "average residual stream over tokens 50 onwards" methodology.

The main algorithm is UNCHANGED: forward hooks on
model.model.language_model.layers, MIN_TOKEN_OFFSET=50, BATCH_SIZE=32,
max_length=512, mean over the token dimension from offset 50 to seq_len,
per-emotion stacked (n_stories, 2560) tensors saved as {emotion}.pt.
"""

import os
os.environ.setdefault("HF_HUB_CACHE", "/autodl-fs/data/huggingface_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B"
DATA_DIR = Path("/root/data/stories")
ACTIVATIONS_DIR = Path("/autodl-fs/data/official_run/activations_all_layers")
NUM_LAYERS = 42
MIN_TOKEN_OFFSET = 50
BATCH_SIZE = 32

# --- interface: the project's 8 emotions (replaces the implicit 171-emotion glob) ---
EMOTIONS = ["sad", "curious", "excited", "bored", "anxious", "happy", "tormented", "angry"]

ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # match neutral extraction; offset-50 slice needs right padding
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()

layers = model.model.language_model.layers

# Hook all 42 layers
captured = {}
hooks = []
for i in range(NUM_LAYERS):
    def make_hook(idx):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            captured[idx] = act.detach().cpu()
        return hook_fn
    hooks.append(layers[i].register_forward_hook(make_hook(i)))

# Load all stories grouped by emotion
print("Loading stories...")
stories_by_emotion = defaultdict(list)
for path in DATA_DIR.glob("*.json"):
    data = json.loads(path.read_text())
    if data["emotion"] not in EMOTIONS:  # interface: keep only the 8 project emotions
        continue
    for story in data["stories"]:
        stories_by_emotion[data["emotion"]].append(story)

emotions = sorted(stories_by_emotion.keys())
print(f"Loaded {len(emotions)} emotions, {sum(len(s) for s in stories_by_emotion.values())} stories total\n")

start = time.time()
for emotion in emotions:
    out_file = ACTIVATIONS_DIR / f"{emotion}.pt"
    if out_file.exists():
        print(f"  Skipping {emotion}")
        continue

    stories = stories_by_emotion[emotion]
    layer_activations = defaultdict(list)

    for i in tqdm(range(0, len(stories), BATCH_SIZE), desc=emotion, leave=True):
        batch_stories = stories[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_stories, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(model.device)

        attention_mask = inputs["attention_mask"].cpu()

        with torch.no_grad():
            model(**inputs)

        for layer_idx in range(NUM_LAYERS):
            act = captured[layer_idx]
            for j in range(act.shape[0]):
                seq_len = attention_mask[j].sum().item()
                if seq_len < MIN_TOKEN_OFFSET + 10:
                    continue
                avg = act[j, MIN_TOKEN_OFFSET:seq_len, :].mean(dim=0)
                layer_activations[layer_idx].append(avg.float())

    result = {idx: torch.stack(vecs) for idx, vecs in layer_activations.items() if vecs}
    torch.save(result, out_file)
    elapsed = time.time() - start
    print(f"  {emotion}: {len(layer_activations[0])} stories, 42 layers | {elapsed / 60:.1f}m elapsed")

for h in hooks:
    h.remove()

total = time.time() - start
print(f"\nDone. {total / 60:.1f} minutes total.")
