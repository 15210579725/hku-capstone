"""
Translate English emotion stories to Chinese using GPT-5.4 API.
- Synchronous requests for stability
- 5 stories per batch (falls back to 3, then 1 on repeated failures)
- Aggressive retry with exponential backoff
- Saves after every batch
- Resume from existing progress
"""

import json
import os
import time
import requests

# API Configuration
API_BASE_URL = "https://once.novai.su/v1"
API_KEY = os.environ.get("TRANSLATE_API_KEY", "")
MODEL = "gpt-5.4"

SYSTEM_PROMPT = (
    "Translate the following English short stories to Chinese. "
    "Maintain emotional tone and narrative style. "
    "Output ONLY a JSON array of translated strings."
)

# Directories
BASE_DIR = "/Users/mac/Desktop/hku capstone/gemma-probes"
STORIES_EN_DIR = os.path.join(BASE_DIR, "data/stories_en")
STORIES_ZH_DIR = os.path.join(BASE_DIR, "data/stories_zh")
NEUTRAL_EN_PATH = os.path.join(BASE_DIR, "data/neutral/neutral_en.json")
NEUTRAL_ZH_PATH = os.path.join(BASE_DIR, "data/neutral/neutral_zh.json")

# Settings
BATCH_SIZE = 5
MAX_RETRIES = 5
BASE_BACKOFF = 2  # seconds
SLEEP_BETWEEN_REQUESTS = 0.5


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate_batch(stories, batch_size_override=None):
    """Translate a batch of stories. Returns list of translated strings or None on failure."""
    actual_batch_size = batch_size_override or len(stories)

    user_content = json.dumps(stories, ensure_ascii=False)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 429:
                wait_time = BASE_BACKOFF ** (attempt + 1)
                print(f"    Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Parse the JSON array from response
            # Sometimes the model wraps in ```json ... ```
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3].strip()
                elif "```" in content:
                    content = content[:content.rfind("```")].strip()

            translated = json.loads(content)

            if isinstance(translated, list) and len(translated) == len(stories):
                return translated
            elif isinstance(translated, list) and len(translated) > 0:
                # Partial match - pad or truncate
                print(f"    Warning: expected {len(stories)} translations, got {len(translated)}")
                if len(translated) >= len(stories):
                    return translated[:len(stories)]
                # If we got fewer, retry
                wait_time = BASE_BACKOFF ** (attempt + 1)
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"    Invalid response format. Retrying...")
                wait_time = BASE_BACKOFF ** (attempt + 1)
                time.sleep(wait_time)
                continue

        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
            wait_time = BASE_BACKOFF ** (attempt + 1)
            print(f"    Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"    Connection error: {e}")
            wait_time = BASE_BACKOFF ** (attempt + 1)
            print(f"    Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
        except requests.exceptions.Timeout:
            print(f"    Request timeout")
            wait_time = BASE_BACKOFF ** (attempt + 1)
            print(f"    Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP error: {e}")
            wait_time = BASE_BACKOFF ** (attempt + 1)
            print(f"    Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"    Unexpected error: {type(e).__name__}: {e}")
            wait_time = BASE_BACKOFF ** (attempt + 1)
            print(f"    Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

    return None


def translate_file(en_path, zh_path, label):
    """Translate a single file with resume support."""
    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"  Source: {en_path}")
    print(f"  Output: {zh_path}")

    # Load source
    stories_en = load_json(en_path)
    total = len(stories_en)
    print(f"  Total stories: {total}")

    # Load existing progress
    stories_zh = load_json(zh_path)
    done = len(stories_zh)
    print(f"  Already translated: {done}")

    if done >= total:
        print(f"  COMPLETE - skipping")
        return done

    # Translate remaining stories in batches
    current_batch_size = BATCH_SIZE
    consecutive_failures = 0

    i = done
    while i < total:
        batch_end = min(i + current_batch_size, total)
        batch = stories_en[i:batch_end]

        print(f"  [{i+1}-{batch_end}/{total}] Translating {len(batch)} stories (batch_size={current_batch_size})...", end=" ", flush=True)

        result = translate_batch(batch)

        if result is not None:
            stories_zh.extend(result)
            save_json(zh_path, stories_zh)
            print(f"OK ({len(stories_zh)}/{total})")
            consecutive_failures = 0
            # Reset batch size back to normal after success
            if current_batch_size < BATCH_SIZE:
                current_batch_size = min(current_batch_size + 1, BATCH_SIZE)
            i = batch_end
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        else:
            consecutive_failures += 1
            print(f"FAILED (attempt {consecutive_failures})")

            if consecutive_failures >= 3 and current_batch_size > 3:
                current_batch_size = 3
                print(f"    Reducing batch size to {current_batch_size}")
            elif consecutive_failures >= 5 and current_batch_size > 1:
                current_batch_size = 1
                print(f"    Reducing batch size to {current_batch_size}")
            elif consecutive_failures >= 8:
                # Skip this batch entirely
                print(f"    SKIPPING batch [{i+1}-{batch_end}] after {consecutive_failures} failures")
                # Fill with placeholder
                stories_zh.extend(["[TRANSLATION FAILED]"] * len(batch))
                save_json(zh_path, stories_zh)
                i = batch_end
                consecutive_failures = 0
                time.sleep(5)
            else:
                # Wait longer before next attempt
                wait = min(30, BASE_BACKOFF ** consecutive_failures)
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)

    final_count = len(stories_zh)
    print(f"  Finished {label}: {final_count}/{total} translated")
    return final_count


def main():
    print("=" * 60)
    print("Story Translation Script v2")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max retries: {MAX_RETRIES}")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(STORIES_ZH_DIR, exist_ok=True)

    # Define all files to process
    emotion_files = ["sad", "curious", "excited", "bored", "anxious", "happy", "tormented", "angry"]

    results = {}

    # Process emotion files
    for emotion in emotion_files:
        en_path = os.path.join(STORIES_EN_DIR, f"{emotion}.json")
        zh_path = os.path.join(STORIES_ZH_DIR, f"{emotion}.json")

        if not os.path.exists(en_path):
            print(f"\n  WARNING: {en_path} not found, skipping")
            continue

        count = translate_file(en_path, zh_path, emotion)
        results[emotion] = count

    # Process neutral file
    if os.path.exists(NEUTRAL_EN_PATH):
        count = translate_file(NEUTRAL_EN_PATH, NEUTRAL_ZH_PATH, "neutral")
        results["neutral"] = count

    # Final summary
    print("\n" + "=" * 60)
    print("TRANSLATION SUMMARY")
    print("=" * 60)
    for name, count in results.items():
        print(f"  {name}: {count}/1200")

    total_done = sum(results.values())
    total_expected = len(results) * 1200
    print(f"\n  Total: {total_done}/{total_expected}")
    print("=" * 60)


if __name__ == "__main__":
    main()
