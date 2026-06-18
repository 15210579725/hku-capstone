"""
Translate Chinese diary segments to English for the "English version" emotion
visualization comparison.

For each segment in selected_segments.json:
  - Each event looks like "[HH:MM:SS -> HH:MM:SS] 中文叙述"
  - We strip the timestamp prefix, translate ONLY the Chinese narrative to English,
    then re-attach the ORIGINAL timestamp prefix untouched.
  - First person "我" -> "I", colloquial, faithful to the original meaning.

Features:
  - Batched translation (BATCH_SIZE narratives per API call), synchronous requests.
  - Exponential backoff retry; falls back to smaller batches on repeated failures.
  - Saves progress after EVERY segment (resume / breakpoint support).

Output: data/selected_segments_en.json -- identical structure to the source,
        only the `events` strings have English narratives (timestamps preserved).
"""

import json
import os
import re
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "https://once.novai.su/v1"
API_KEY = os.environ.get("TRANSLATE_API_KEY", "")
MODEL = "gpt-5.4"

BASE_DIR = "/Users/mac/Desktop/hku capstone/gemma-probes"
SRC_PATH = os.path.join(BASE_DIR, "data/selected_segments.json")
OUT_PATH = os.path.join(BASE_DIR, "data/selected_segments_en.json")

BATCH_SIZE = 10
MAX_RETRIES = 6
BASE_BACKOFF = 2  # seconds
SLEEP_BETWEEN_REQUESTS = 0.4

# Regex to split "[HH:MM:SS -> HH:MM:SS] narrative" into (timestamp, narrative).
TS_RE = re.compile(r"^(\[\d{2}:\d{2}:\d{2}\s*->\s*\d{2}:\d{2}:\d{2}\])\s*(.*)$", re.S)

SYSTEM_PROMPT = (
    "You are a professional translator. You will receive a JSON array of short "
    "first-person diary narratives written in Chinese (a person describing what "
    "they see, hear, and do moment by moment). Translate each entry into natural, "
    "colloquial English.\n"
    "Rules:\n"
    "1. Translate first person '我' as 'I'.\n"
    "2. Keep personal names (e.g. Lucia, Jake, Katrina, Shure) exactly as written.\n"
    "3. If a narrative quotes speech in Chinese quotation marks, translate the "
    "quoted speech into English and keep it in quotes.\n"
    "4. Stay faithful to the original meaning; keep it casual and spoken.\n"
    "5. Do NOT add timestamps, numbering, or any extra commentary.\n"
    "Output ONLY a JSON array of translated English strings, same length and same "
    "order as the input array."
)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic save to avoid corrupting progress


def split_event(event):
    """Return (timestamp_prefix, narrative). timestamp_prefix may be '' if no match."""
    m = TS_RE.match(event)
    if m:
        return m.group(1), m.group(2)
    return "", event


def _extract_json_array(content):
    content = content.strip()
    if content.startswith("```"):
        # strip ```json ... ``` fences
        content = content.split("\n", 1)[1] if "\n" in content else content
        if "```" in content:
            content = content[: content.rfind("```")]
        content = content.strip()
    # Fallback: grab the outermost [...] if there is leading/trailing chatter.
    if not content.startswith("["):
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
    return json.loads(content)


def translate_narratives(narratives):
    """Translate a list of Chinese narratives. Returns list of same length, or None."""
    user_content = json.dumps(narratives, ensure_ascii=False)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            translated = _extract_json_array(content)

            if isinstance(translated, list) and len(translated) == len(narratives):
                return [str(t) for t in translated]

            print(
                f"    Length mismatch: expected {len(narratives)}, "
                f"got {len(translated) if isinstance(translated, list) else 'non-list'}"
            )
        except Exception as e:  # noqa: BLE001 -- broad on purpose, we retry everything
            print(f"    {type(e).__name__}: {e}")

        wait = min(60, BASE_BACKOFF ** (attempt + 1))
        print(f"    Retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
        time.sleep(wait)

    return None


def translate_events(events):
    """Translate all events of one segment, preserving timestamp prefixes."""
    prefixes = []
    narratives = []
    for ev in events:
        pre, nar = split_event(ev)
        prefixes.append(pre)
        narratives.append(nar)

    out = [None] * len(events)
    i = 0
    while i < len(narratives):
        batch_size = BATCH_SIZE
        # Adaptive shrinking: try BATCH_SIZE, then 5, then 1.
        for size in (BATCH_SIZE, 5, 1):
            end = min(i + size, len(narratives))
            chunk = narratives[i:end]
            result = translate_narratives(chunk)
            if result is not None:
                for k, tr in enumerate(result):
                    pre = prefixes[i + k]
                    out[i + k] = f"{pre} {tr}".strip() if pre else tr
                i = end
                batch_size = size
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                break
        else:
            # Every batch size failed for this position -- keep original to avoid
            # losing the entry, mark it, and move on by 1.
            print(f"    !! Failed to translate event index {i}; keeping original.")
            out[i] = events[i]
            i += 1
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Translate Chinese diary segments -> English")
    print(f"Model: {MODEL}  |  Batch size: {BATCH_SIZE}")
    print("=" * 64)

    segments = load_json(SRC_PATH)
    if segments is None:
        raise SystemExit(f"Source not found: {SRC_PATH}")
    total = len(segments)
    total_events = sum(len(s["events"]) for s in segments)
    print(f"Source: {total} segments, {total_events} events")

    # Resume: output is a list aligned by segment order; done = len(existing).
    out_segments = load_json(OUT_PATH) or []
    done = len(out_segments)
    if done:
        print(f"Resuming: {done} segments already translated.")

    for idx in range(done, total):
        seg = segments[idx]
        n_ev = len(seg["events"])
        print(f"\n[{idx + 1}/{total}] {seg.get('id', '?')} -- {n_ev} events ...")

        new_seg = dict(seg)  # preserve all other fields (id, expected_emotions, reason, ...)
        new_seg["events"] = translate_events(seg["events"])

        out_segments.append(new_seg)
        save_json(OUT_PATH, out_segments)  # save after EVERY segment
        print(f"    saved ({len(out_segments)}/{total})")

    translated_events = sum(len(s["events"]) for s in out_segments)
    print("\n" + "=" * 64)
    print(f"DONE: {len(out_segments)}/{total} segments, {translated_events} events")
    print(f"Output: {OUT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
