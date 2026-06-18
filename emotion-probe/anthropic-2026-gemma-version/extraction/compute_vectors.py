"""
Compute expression vectors from activation files.

For each layer:
1. Compute per-emotion mean activations
2. Compute global mean across all emotions
3. Raw direction = emotion_mean - global_mean
4. PCA confound removal using neutral activations
5. Normalize

Output: en_vectors.pt
"""

import torch
import os

# Configuration
ACTIVATION_DIR = "/autodl-fs/data/activations"
OUTPUT_PATH = "/root/gemma-probes-capstone/en_vectors.pt"
NUM_LAYERS = 42
HIDDEN_DIM = 2560
VARIANCE_THRESHOLD = 0.50  # Keep PCs explaining 50% of variance

EMOTIONS = ['sad', 'curious', 'excited', 'bored', 'anxious', 'happy', 'tormented', 'angry']
EMOTION_FILES = [f"en_{e}.pt" for e in EMOTIONS]
NEUTRAL_FILE = "en_neutral.pt"


def load_activations(filename):
    """Load activation file, returns dict {layer_idx: tensor[1200, 2560]}"""
    path = os.path.join(ACTIVATION_DIR, filename)
    return torch.load(path, map_location='cpu')


def compute_pca_top_k(neutral_centered, variance_threshold=0.50):
    """
    Compute SVD on centered neutral activations.
    Return V[:, :k] where k PCs explain >= variance_threshold of total variance.
    """
    # SVD: neutral_centered = U @ diag(S) @ V^T
    U, S, Vh = torch.linalg.svd(neutral_centered, full_matrices=False)
    # V in torch.linalg.svd: Vh is V^T, so V = Vh.T
    V = Vh.T  # shape [2560, min(1200, 2560)]

    # Determine k: cumulative variance explained
    variance = S ** 2
    total_var = variance.sum()
    cumvar = torch.cumsum(variance, dim=0) / total_var

    # Find smallest k such that cumvar[k-1] >= threshold
    k = int((cumvar >= variance_threshold).nonzero(as_tuple=True)[0][0].item()) + 1

    return V[:, :k], k


def project_out(direction, V_k):
    """
    Remove confound directions from the raw direction.
    direction: [2560]
    V_k: [2560, k]
    """
    # Project direction onto the subspace spanned by V_k, then subtract
    projection = V_k @ (V_k.T @ direction)
    return direction - projection


def main():
    print("Loading activation files...")

    # Load all emotion activations
    emotion_data = {}
    for emotion, filename in zip(EMOTIONS, EMOTION_FILES):
        print(f"  Loading {filename}...")
        emotion_data[emotion] = load_activations(filename)

    # Load neutral activations
    print(f"  Loading {NEUTRAL_FILE}...")
    neutral_data = load_activations(NEUTRAL_FILE)

    print("\nComputing expression vectors...")

    # Output tensors
    vectors = torch.zeros(len(EMOTIONS), NUM_LAYERS, HIDDEN_DIM)
    global_means = torch.zeros(NUM_LAYERS, HIDDEN_DIM)
    neutral_pcs = []  # Store V[:, :k] per layer

    for layer_idx in range(NUM_LAYERS):
        # 1. Compute per-emotion mean for this layer
        emotion_means = {}
        for emotion in EMOTIONS:
            layer_acts = emotion_data[emotion][layer_idx]  # [1200, 2560]
            emotion_means[emotion] = layer_acts.mean(dim=0)  # [2560]

        # 2. Global mean = average of all emotion means
        all_means = torch.stack([emotion_means[e] for e in EMOTIONS])  # [8, 2560]
        global_mean = all_means.mean(dim=0)  # [2560]
        global_means[layer_idx] = global_mean

        # 3. Raw directions
        raw_directions = {}
        for emotion in EMOTIONS:
            raw_directions[emotion] = emotion_means[emotion] - global_mean

        # 4. PCA confound removal using neutral activations
        neutral_acts = neutral_data[layer_idx]  # [1200, 2560]
        neutral_mean = neutral_acts.mean(dim=0)
        neutral_centered = neutral_acts - neutral_mean  # [1200, 2560]

        V_k, k = compute_pca_top_k(neutral_centered, VARIANCE_THRESHOLD)
        neutral_pcs.append(V_k)

        # 5. Project out confounds and normalize
        for i, emotion in enumerate(EMOTIONS):
            direction = project_out(raw_directions[emotion], V_k)
            direction = direction / direction.norm()
            vectors[i, layer_idx] = direction

        if (layer_idx + 1) % 7 == 0 or layer_idx == 0:
            print(f"  Layer {layer_idx:2d}: k={k} PCs removed")

    # Save results
    result = {
        'vectors': vectors,
        'emotions': EMOTIONS,
        'global_means': global_means,
        'neutral_pcs': neutral_pcs,
    }

    torch.save(result, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  vectors shape: {vectors.shape}")
    print(f"  emotions: {EMOTIONS}")
    print(f"  global_means shape: {global_means.shape}")
    print(f"  neutral_pcs: {len(neutral_pcs)} layers")

    # Quick validation: cosine similarity at layer 23
    print("\n--- Cosine Similarity Matrix (Layer 23) ---")
    layer_vecs = vectors[:, 23, :]  # [8, 2560]
    cos_sim = torch.mm(layer_vecs, layer_vecs.T)  # Already normalized, so this is cosine sim

    # Print header
    short_names = ['sad', 'cur', 'exc', 'bor', 'anx', 'hap', 'tor', 'ang']
    header = "        " + "  ".join(f"{n:>5}" for n in short_names)
    print(header)
    for i, emotion in enumerate(EMOTIONS):
        row = f"{emotion:>8}"
        for j in range(len(EMOTIONS)):
            row += f"  {cos_sim[i, j].item():5.2f}"
        print(row)


if __name__ == "__main__":
    main()
