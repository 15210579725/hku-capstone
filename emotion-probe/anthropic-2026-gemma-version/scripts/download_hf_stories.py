"""
Download emotion stories from HuggingFace dataset ryancodrai/emotion-probes.

Filters for 7 target emotions + neutral, saves as JSON files.
"""

import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "ryancodrai/emotion-probes"
TARGET_EMOTIONS = ["sad", "excited", "bored", "anxious", "happy", "tormented", "angry"]

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_EN_DIR = PROJECT_ROOT / "data" / "stories_en"
NEUTRAL_DIR = PROJECT_ROOT / "data" / "neutral"


def main():
    # Create output directories
    STORIES_EN_DIR.mkdir(parents=True, exist_ok=True)
    NEUTRAL_DIR.mkdir(parents=True, exist_ok=True)

    # Download stories parquet
    print("Downloading expression/stories.parquet ...")
    stories_path = hf_hub_download(REPO_ID, "expression/stories.parquet", repo_type="dataset")
    df = pd.read_parquet(stories_path)
    print(f"  Total stories: {len(df)}, columns: {df.columns.tolist()}")

    # Filter and save each target emotion
    print("\nFiltering target emotions:")
    for emotion in TARGET_EMOTIONS:
        subset = df[df["emotion"] == emotion]["story"].tolist()
        out_path = STORIES_EN_DIR / f"{emotion}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  {emotion}: {len(subset)} stories -> {out_path}")

    # Download and save neutral stories
    print("\nDownloading expression/neutral_stories.parquet ...")
    neutral_path = hf_hub_download(REPO_ID, "expression/neutral_stories.parquet", repo_type="dataset")
    df_neutral = pd.read_parquet(neutral_path)
    neutral_stories = df_neutral["story"].tolist()

    out_path = NEUTRAL_DIR / "neutral_en.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(neutral_stories, f, ensure_ascii=False, indent=2)
    print(f"  neutral: {len(neutral_stories)} stories -> {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
