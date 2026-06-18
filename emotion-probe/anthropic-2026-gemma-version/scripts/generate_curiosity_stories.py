"""
Generate ~1200 curiosity-themed emotional stories using GPT-5.4 API.
100 topics x 12 variations = 1200 stories total.
Supports resume: skips already-generated stories if output file exists.
"""

import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

# --- Configuration ---
BASE_URL = "https://once.novai.su/v1"
API_KEY = os.environ.get("TRANSLATE_API_KEY", "")
MODEL = "gpt-5.4"
OUTPUT_PATH = Path("/Users/mac/Desktop/hku capstone/gemma-probes/data/stories_en/curious.json")
MAX_CONCURRENT = 5  # concurrent API requests (conservative to avoid rate limits)
VARIATIONS_PER_TOPIC = 12
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are a creative fiction writer. Write a short story (100-150 words) that conveys a sense of curiosity without ever using the word "curious", "curiosity", or direct synonyms like "inquisitive". Instead, show the emotion through characters' actions, questions, explorations, body language, and internal monologue. The story should make the reader FEEL the curiosity without being told about it."""

# 100 diverse topics to spark curiosity-themed stories
TOPICS = [
    "abandoned building", "strange letter", "new neighbor", "unknown sound at night",
    "locked drawer", "mysterious map", "unusual animal behavior", "coded message",
    "old photograph", "forgotten room", "unmarked package", "distant lighthouse",
    "hidden door behind a bookshelf", "unfamiliar scent in the hallway",
    "a journal found in a secondhand book", "radio signal from nowhere",
    "footprints leading into the woods", "a key with no lock",
    "whispering walls in an old house", "a clock that runs backwards",
    "a stranger who knows your name", "light under a closed door",
    "a painting that changes overnight", "an unopened time capsule",
    "a song no one else can hear", "symbols carved into a tree",
    "a telescope pointed at nothing", "a child's drawing of the future",
    "a library book overdue by 50 years", "a shadow with no source",
    "a phone call from an unknown number", "a garden that blooms at midnight",
    "an elevator to an unlisted floor", "a mirror that shows a different room",
    "a recipe written in a dead language", "a road not on any map",
    "a satellite dish on a remote cabin", "a well that echoes differently",
    "a cat that stares at an empty corner", "a compass that points south",
    "an old radio playing a station that doesn't exist", "a trapdoor under the carpet",
    "a handprint on a foggy window from the outside", "a lighthouse with no keeper",
    "a street that wasn't there yesterday", "a notebook with tomorrow's date",
    "an attic noise in a single-story house", "a bottle washed ashore with coordinates",
    "a face in a crowd that looks exactly like you", "a tunnel behind a waterfall",
    "a staircase that goes deeper than the basement should allow",
    "a music box that plays unfamiliar tunes", "a bird that taps on the same window daily",
    "a shopkeeper who claims to remember your past life",
    "a watch that stops at the same time every day",
    "a photograph of a place you've never been but recognize",
    "a message scratched into the underside of a desk",
    "a tree that grows in an impossible pattern",
    "a door in the forest that stands alone with no walls",
    "a typewriter that types by itself at night",
    "an ancient coin found in modern concrete",
    "a child who speaks a language no one recognizes",
    "a boat drifting upriver against the current",
    "a constellation that doesn't match any star chart",
    "a recording of your own voice saying things you never said",
    "a neighbor's window that glows blue every midnight",
    "a book with blank pages that fill themselves when read aloud",
    "a subway station not on the official map",
    "a weather vane that always points at you",
    "a flower growing through a crack in a sealed vault",
    "an echo that returns different words",
    "a chess piece found in an archaeological dig",
    "a photograph where everyone is looking at something outside the frame",
    "a room temperature that drops in one exact spot",
    "a stranger's diary left on a park bench",
    "a dog that barks at nothing every evening at seven",
    "a postcard from a place that doesn't exist",
    "a fingerprint on the inside of a sealed jar",
    "a violin that plays one note when no one touches it",
    "a window that shows a different season than outside",
    "a pile of stones arranged in perfect geometric patterns",
    "a phone that receives texts from your own number",
    "a hallway that seems longer when you walk alone",
    "a cloud formation that appears in the same shape daily",
    "a door knocker that's warm to the touch in winter",
    "a drawing found inside a wall during renovation",
    "a tape recorder with no tape that still plays sound",
    "a path through the forest that always leads back to start",
    "a penny from a year that hasn't happened yet",
    "a fish tank where the fish swim in synchronized patterns",
    "a candle that burns without melting",
    "an intersection where birds never fly over",
    "a voice that whispers your name in an empty elevator",
    "a train that arrives at a station not on its route",
    "a reflection in water that doesn't match the sky",
    "a mailbox that receives letters addressed to no one",
    "a stain on the ceiling that looks like a map",
    "a breeze that carries the scent of a place far away",
    "a crack in the pavement that glows faintly at dusk",
    "a swing in an abandoned playground that moves on its own",
    "a vending machine that dispenses unlabeled items",
    "a bookmark in a library book with a personal note to you",
]

NUM_TOPICS = len(TOPICS)
print(f"Loaded {NUM_TOPICS} topics")


def load_existing_stories() -> list[str]:
    """Load existing stories from the output file for resume support."""
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    return []


def save_stories(stories: list[str]):
    """Save stories to the output file."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stories, f, ensure_ascii=False, indent=2)


async def generate_story(client: AsyncOpenAI, topic: str, variation: int, semaphore: asyncio.Semaphore) -> str | None:
    """Generate a single story for a given topic and variation with retry logic."""
    async with semaphore:
        user_prompt = f"Topic: {topic}\nVariation #{variation + 1}: Write a unique short story inspired by this topic that evokes a deep sense of curiosity."
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.9,
                    max_tokens=300,
                )
                story = response.choices[0].message.content.strip()
                return story
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = 2 ** attempt + 1
                    await asyncio.sleep(wait_time)
                    continue
                elif "500" in error_str or "502" in error_str or "503" in error_str:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"  [ERROR] topic='{topic}' var={variation}: {e}")
                    return None
        print(f"  [FAILED after {MAX_RETRIES} retries] topic='{topic}' var={variation}")
        return None


async def main():
    print("=" * 60)
    print("Curiosity Story Generator")
    print(f"Target: {NUM_TOPICS} topics x {VARIATIONS_PER_TOPIC} variations = {NUM_TOPICS * VARIATIONS_PER_TOPIC} stories")
    print("=" * 60)

    # Load existing stories for resume
    existing_stories = load_existing_stories()
    num_existing = len(existing_stories)
    total_target = NUM_TOPICS * VARIATIONS_PER_TOPIC

    if num_existing >= total_target:
        print(f"Already have {num_existing} stories (target: {total_target}). Done!")
        return

    if num_existing > 0:
        print(f"Resuming: found {num_existing} existing stories, need {total_target - num_existing} more.")

    # Figure out which (topic_idx, variation_idx) pairs to generate
    # Stories are stored in order: topic_0_var_0, topic_0_var_1, ..., topic_0_var_11, topic_1_var_0, ...
    skip_count = num_existing
    tasks_to_generate = []
    for t_idx, topic in enumerate(TOPICS):
        for v_idx in range(VARIATIONS_PER_TOPIC):
            if skip_count > 0:
                skip_count -= 1
                continue
            tasks_to_generate.append((t_idx, topic, v_idx))

    print(f"Generating {len(tasks_to_generate)} stories...")
    print()

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    all_stories = list(existing_stories)  # start with existing
    batch_size = 50  # save every N stories
    generated_count = 0
    start_time = time.time()

    # Process in batches for progress reporting and incremental saves
    for batch_start in range(0, len(tasks_to_generate), batch_size):
        batch = tasks_to_generate[batch_start:batch_start + batch_size]
        coros = [generate_story(client, topic, v_idx, semaphore) for (_, topic, v_idx) in batch]
        results = await asyncio.gather(*coros)

        for result in results:
            if result:
                all_stories.append(result)
                generated_count += 1

        # Save progress
        save_stories(all_stories)

        elapsed = time.time() - start_time
        rate = generated_count / elapsed if elapsed > 0 else 0
        print(f"  Progress: {len(all_stories)}/{total_target} stories "
              f"(+{generated_count} new, {rate:.1f} stories/sec, {elapsed:.0f}s elapsed)")

    elapsed_total = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Done! Total stories: {len(all_stories)}")
    print(f"New stories generated: {generated_count}")
    print(f"Time elapsed: {elapsed_total:.1f}s")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
