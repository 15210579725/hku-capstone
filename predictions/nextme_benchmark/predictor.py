#!/usr/bin/env python3
"""
NextMe Behavior Predictor — model-agnostic prediction interface.

Loads benchmark points, builds cross-VRS prompts with context from the
recording hierarchy, and calls a configurable LLM to generate predictions.

Usage:
    from nextme_benchmark.predictor import BenchmarkPredictor
    predictor = BenchmarkPredictor(model="deepseek-v4-flash",
                                   api_key="...", base_url="...")
    results = predictor.predict_benchmark(k=3, max_points=10)
"""

import json, os, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from openai import OpenAI

# ─── Paths ────────────────────────────────────────────────────────────
_PKG_DIR = Path(__file__).resolve().parent
_BENCHMARK_FILE = _PKG_DIR / "benchmark.json"
_PROMPT_TEMPLATE_FILE = _PKG_DIR / "prompt.txt"
_CAPTION_RESULT_DIR = Path(__file__).resolve().parent.parent.parent / "caption-result"

EVENT_PAT = re.compile(r"\[(\d{2}:\d{2}:\d{2})\s*->\s*(\d{2}:\d{2}:\d{2})\]\s*(.+)")
REC_PAT = re.compile(r"(\d+)-(\d+)_hkt(\d{2})(\d{2})-(\d{2})(\d{2})_(\d+)m_(.+)")

MAX_CONTEXT = 50


def _parse_events(path, date_str):
    events = []
    for line in path.read_text("utf-8").strip().splitlines():
        m = EVENT_PAT.match(line.strip())
        if m:
            events.append({
                "start": m.group(1), "end": m.group(2),
                "text": m.group(3).strip(), "raw": line.strip(),
                "date": date_str,
                "raw_with_date": f"[{date_str} {m.group(1)} -> {date_str} {m.group(2)}] {m.group(3).strip()}"
            })
    return events


def _parse_date_str(rec_name):
    m = REC_PAT.match(rec_name)
    if m:
        return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return "00-00"


def _strip_ts(t):
    t = re.sub(r"^\[\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*->\s*\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*", "", t).strip()
    t = re.sub(r"^\[\d{2}:\d{2}:\d{2}\s*->\s*\d{2}:\d{2}:\d{2}\]\s*", "", t).strip()
    return t


class BenchmarkPredictor:
    """Generate predictions for benchmark points using an LLM."""

    def __init__(self, model: str, api_key: str, base_url: str,
                 benchmark_path: Optional[str] = None,
                 prompt_template_path: Optional[str] = None,
                 caption_result_dir: Optional[str] = None,
                 concurrency: int = 50,
                 temperature: float = 0.7,
                 max_tokens: int = 8192):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.concurrency = concurrency
        self.temperature = temperature
        self.max_tokens = max_tokens

        bp = Path(benchmark_path) if benchmark_path else _BENCHMARK_FILE
        self.benchmark = json.loads(bp.read_text())

        tp = Path(prompt_template_path) if prompt_template_path else _PROMPT_TEMPLATE_FILE
        self.prompt_template = tp.read_text("utf-8")

        self.caption_dir = Path(caption_result_dir) if caption_result_dir else _CAPTION_RESULT_DIR

    def _load_context(self, point: dict, level: str) -> List[dict]:
        """Load context events from the recording hierarchy."""
        rec_dir = self.caption_dir / point["context_recording"] / "hierarchy" / level / "events.txt"
        date_str = point["context_date"]
        if rec_dir.exists():
            events = _parse_events(rec_dir, date_str)
            return events[-MAX_CONTEXT:]
        return []

    def _build_prompt(self, point: dict, context: List[dict], k: int) -> tuple:
        """Build (system, user) prompt for a benchmark point."""
        segments = point.get(f"gt_segments_k{k}", point.get("gt_segments_k3", []))
        ctx_date = point["context_date"]

        seg_lines = []
        for i, seg in enumerate(segments, 1):
            seg_lines.append(f"  Segment {i}: {seg['date']} {seg['start']} – {seg['end']}")
        target_segments_str = "\n".join(seg_lines)

        dates_in_play = set([ctx_date] + [s["date"] for s in segments])
        cross_day = len(dates_in_play) > 1

        date_prefix = "MM-DD " if cross_day else ""
        action_example = f"[{date_prefix}HH:MM:SS -> {date_prefix}HH:MM:SS] action description"

        system = self.prompt_template.format(
            k=k,
            context_date=ctx_date,
            context_start=point["context_start"],
            context_end=point["context_end"],
            target_segments=target_segments_str,
            date_prefix=date_prefix,
            action_example=action_example,
        )

        if cross_day:
            ctx_lines = [e["raw_with_date"] for e in context]
        else:
            ctx_lines = [e["raw"] for e in context]

        user = (f"Event history (most recent {len(context)} events "
                f"from recording {ctx_date} {point['context_start']}–{point['context_end']}):\n\n"
                + "\n".join(ctx_lines))

        return system, user

    def _call_model(self, system: str, user: str, retries: int = 3) -> Optional[dict]:
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens)
                text = resp.choices[0].message.content.strip()
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if m:
                        return json.loads(m.group())
                    raise
            except Exception:
                if attempt == retries - 1:
                    return None
                time.sleep(1)
        return None

    def predict_point(self, point: dict, level: str, k: int) -> Optional[dict]:
        """Predict for a single benchmark point. Returns prediction dict or None."""
        context = self._load_context(point, level)
        if not context:
            return None

        system, user = self._build_prompt(point, context, k)
        result = self._call_model(system, user)

        if not result or "candidates" not in result:
            return None

        best = result["candidates"][0]
        if "actions" not in best:
            return None

        pr_raw = best["actions"][:k]
        pr_text = [_strip_ts(a) for a in pr_raw]

        return {
            "id": point["id"],
            "level": level,
            "k": k,
            "ground_truth": point[f"gt_events_k{k}"][:k],
            "prediction": pr_text,
            "prediction_raw": pr_raw,
            "probability": best.get("probability"),
            "all_candidates": result["candidates"],
        }

    def predict_benchmark(self, k: int = 3, levels: Optional[List[str]] = None,
                          max_points: Optional[int] = None,
                          progress_callback=None) -> List[dict]:
        """
        Run predictions on benchmark points.

        Args:
            k: prediction length (3 or 10)
            levels: which levels to run (default: all L1-L5)
            max_points: limit per level (default: all 200)
            progress_callback: callable(done, total) for progress updates
        """
        if levels is None:
            levels = ["L1", "L2", "L3", "L4", "L5"]

        tasks = []
        for lv in levels:
            points = self.benchmark.get(lv, [])
            if max_points:
                points = points[:max_points]
            for pt in points:
                gt = pt.get(f"gt_events_k{k}", [])
                if len(gt) < k:
                    continue
                tasks.append((pt, lv, k))

        results = []
        t0 = time.time()

        def _do(idx):
            pt, lv, kv = tasks[idx]
            return idx, self.predict_point(pt, lv, kv)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futs = {pool.submit(_do, i): i for i in range(len(tasks))}
            done = 0
            for f in as_completed(futs):
                done += 1
                idx, r = f.result()
                if r:
                    results.append(r)
                if progress_callback:
                    progress_callback(done, len(tasks))
                elif done % 50 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    print(f"  {done}/{len(tasks)} done ({elapsed:.1f}s, "
                          f"{len(results)} ok)")

        results.sort(key=lambda r: r["id"])
        return results
