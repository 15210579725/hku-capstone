"""
Translate English emotion stories to Chinese using GPT-5.4 API.
Uses ThreadPoolExecutor for concurrency (more stable than asyncio for this API).
Has resume logic and saves progress periodically.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# API Configuration
BASE_URL = "https://once.novai.su/v1"
API_KEY = os.environ.get("TRANSLATE_API_KEY", "")
MODEL = "gpt-5.4"

# Translation settings
BATCH_SIZE = 10  # stories per API call
MAX_WORKERS = 5  # concurrent threads
SAVE_EVERY = 50  # save progress every N stories
MAX_RETRIES = 6  # max retries per request
BASE_DELAY = 3  # base delay for exponential backoff

# File mappings
BASE_DIR = Path("/Users/mac/Desktop/hku capstone/gemma-probes")
FILES = [
    (BASE_DIR / "data/stories_en/sad.json", BASE_DIR / "data/stories_zh/sad.json"),
    (BASE_DIR / "data/stories_en/curious.json", BASE_DIR / "data/stories_zh/curious.json"),
    (BASE_DIR / "data/stories_en/excited.json", BASE_DIR / "data/stories_zh/excited.json"),
    (BASE_DIR / "data/stories_en/bored.json", BASE_DIR / "data/stories_zh/bored.json"),
    (BASE_DIR / "data/stories_en/anxious.json", BASE_DIR / "data/stories_zh/anxious.json"),
    (BASE_DIR / "data/stories_en/happy.json", BASE_DIR / "data/stories_zh/happy.json"),
    (BASE_DIR / "data/stories_en/tormented.json", BASE_DIR / "data/stories_zh/tormented.json"),
    (BASE_DIR / "data/stories_en/angry.json", BASE_DIR / "data/stories_zh/angry.json"),
    (BASE_DIR / "data/neutral/neutral_en.json", BASE_DIR / "data/neutral/neutral_zh.json"),
]

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the following English short stories to Chinese. "
    "Maintain the emotional tone, narrative style, and literary quality. "
    "Output ONLY the translated stories as a JSON array of strings, one per input story. "
    "Do not add explanations."
)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0, max_retries=0)


def translate_batch(batch_idx: int, stories: list[str]) -> tuple[int, list[str]]:
    """Translate a batch of stories with retry logic. Returns (batch_idx, translations)."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(stories, ensure_ascii=False)},
                ],
                temperature=0.3,
                max_tokens=16000,
            )
            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                content = content.strip()

            translated = json.loads(content)
            if isinstance(translated, list) and len(translated) == len(stories):
                return (batch_idx, translated)
            else:
                print(f"  [Batch {batch_idx}] Expected {len(stories)} translations, got {len(translated) if isinstance(translated, list) else 'non-list'}. Retrying...")

        except json.JSONDecodeError as e:
            print(f"  [Batch {batch_idx}] JSON parse error attempt {attempt+1}: {e}")
        except Exception as e:
            error_msg = str(e)[:120]
            print(f"  [Batch {batch_idx}] API error attempt {attempt+1}: {error_msg}")

        # Exponential backoff
        delay = BASE_DELAY * (2 ** attempt)
        print(f"  [Batch {batch_idx}] Waiting {delay}s before retry {attempt+2}...")
        time.sleep(delay)

    print(f"  [Batch {batch_idx}] FAILED after {MAX_RETRIES} retries.")
    return (batch_idx, ["[TRANSLATION FAILED]"] * len(stories))


def translate_file(input_path: Path, output_path: Path):
    """Translate a single file with resume logic."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")

    # Load source stories
    with open(input_path, "r", encoding="utf-8") as f:
        stories = json.load(f)

    total = len(stories)
    print(f"Total stories: {total}")

    # Resume logic
    translated = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            translated = json.load(f)
        # Remove trailing failed translations
        while translated and translated[-1] == "[TRANSLATION FAILED]":
            translated.pop()
        print(f"Resuming from story {len(translated)}/{total}")

    if len(translated) >= total:
        print("Already complete! Skipping.")
        return

    start_idx = len(translated)
    remaining_stories = stories[start_idx:]

    # Create batches with indices
    batches = []
    for i in range(0, len(remaining_stories), BATCH_SIZE):
        batches.append(remaining_stories[i:i + BATCH_SIZE])

    print(f"Remaining: {len(remaining_stories)} stories in {len(batches)} batches")

    # Process batches using ThreadPoolExecutor
    processed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit in chunks for periodic saving
        chunk_size = SAVE_EVERY // BATCH_SIZE  # batches per save checkpoint

        for chunk_start in range(0, len(batches), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(batches))
            chunk_batches = batches[chunk_start:chunk_end]

            futures = {
                executor.submit(translate_batch, chunk_start + i, batch): i
                for i, batch in enumerate(chunk_batches)
            }

            # Collect results in order
            results = [None] * len(chunk_batches)
            for future in as_completed(futures):
                idx = futures[future]
                batch_idx, result = future.result()
                results[idx] = result

            # Extend translated list in order
            for result in results:
                if result:
                    translated.extend(result)
                    processed += len(result)

            # Save progress
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False, indent=2)

            print(f"  Progress: {len(translated)}/{total} ({len(translated)*100//total}%)")

    print(f"Completed: {input_path.name} ({len(translated)} stories)")


def main():
    print("=" * 60)
    print("Story Translation: English -> Chinese")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}")
    print("=" * 60)

    start_time = time.time()

    for input_path, output_path in FILES:
        if not input_path.exists():
            print(f"\n[SKIP] File not found: {input_path}")
            continue
        translate_file(input_path, output_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All done! Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
