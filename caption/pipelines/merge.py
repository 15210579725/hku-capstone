#!/usr/bin/env python3
"""clip → 整条录制 → 整天 的合并。云端和本地共用。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")

# gemini-3.7-flash 官方价（$/1M token，2026-12-31 前的引导价；之后翻倍到 1.50/7.50）
# Batch API 半价。thinking token 按 output 计费，必须算进去。
PRICE_IN, PRICE_OUT = 0.75, 3.75


def load_clips(caps_dir: Path):
    out = []
    for p in sorted(Path(caps_dir).glob("clip_*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:                                     # noqa: BLE001
            continue
    return sorted(out, key=lambda c: c.get("clip_id", 0))


def _seg_start(seg, fallback):
    m = TIME_RE.search(seg.get("time_range") or "")
    return m.group(0) if m else fallback


def merge_recording(clips: list, manifest: dict | None = None) -> dict:
    segments, summaries = [], []
    for c in clips:
        parsed = c.get("parsed") or {}
        base = (c.get("clip_start_hkt") or "")[11:] or "00:00:00"
        summaries.append({
            "clip_id": c.get("clip_id"),
            "time": f"{(c.get('clip_start_hkt') or '')[11:]}–{(c.get('clip_end_hkt') or '')[11:]}",
            "ok": c.get("ok"),
            "summary": parsed.get("scene_summary", ""),
            "activity_chain": parsed.get("activity_chain", ""),
        })
        for s in parsed.get("segments", []) or []:
            s = dict(s)
            s["clip_id"] = c.get("clip_id")
            s["_sort"] = _seg_start(s, base)
            segments.append(s)
    segments.sort(key=lambda s: (s["clip_id"], s["_sort"]))
    for s in segments:
        s.pop("_sort", None)

    ok = [c for c in clips if c.get("ok")]
    bad = [c for c in clips if not c.get("ok")]
    tin = sum((c.get("usage") or {}).get("in") or 0 for c in clips)
    tout = sum(((c.get("usage") or {}).get("out") or 0)
               + ((c.get("usage") or {}).get("think") or 0) for c in clips)
    date = (clips[0].get("clip_start_hkt") or "")[:10] if clips else ""
    return {
        "recording": (manifest or {}).get("recording") or (clips[0].get("recording") if clips else ""),
        "date": date,
        "clip_count": len(clips),
        "clip_ok": len(ok),
        "clip_failed": len(bad),
        "failed_clip_ids": [c.get("clip_id") for c in bad],
        "segment_count": len(segments),
        "time_start_hkt": clips[0].get("clip_start_hkt") if clips else "",
        "time_end_hkt": clips[-1].get("clip_end_hkt") if clips else "",
        "has_gaze": bool(clips and clips[0].get("has_gaze")),
        "model": clips[0].get("model") if clips else "",
        "tokens": {"in": tin, "out_billable": tout},
        "est_cost_usd": round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 4),
        "pii_audit": pii_audit(segments),
        "manifest": manifest or {},
        "clip_summaries": summaries,
        "segments": segments,
    }


# prompt Rule 8 要求把校名/人名/邮箱等换成占位符。实测各模型都不是 100% 守规矩：
# 3.7-flash 全量 458 段里漏了 2 处「HKU」，lite 系列漏得多得多。
# 花几千刀跑全量之前先量出来，别等公开了才发现。
PII_PATTERNS = {
    "school_plain": r"(?i)\buniversity of hong kong\b|\bHKU\b|港大|香港大学",
    "school_generic": r"(?i)\b[A-Z][a-z]+ (?:University|College)\b",
    "email_plain": r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_plain": r"\b(?:\+?86)?1[3-9]\d{9}\b|\b\d{4}[- ]\d{4}\b",
}
ANON_PATTERNS = r"University X|School X|User [A-Z]\b|email_x@|XXX-XXXX-XXXX|Address X"


def pii_audit(segments: list) -> dict:
    blob = json.dumps(segments, ensure_ascii=False)
    out = {k: len(re.findall(p, blob)) for k, p in PII_PATTERNS.items()}
    out["anonymized_placeholders"] = len(re.findall(ANON_PATTERNS, blob))
    out["total_hits"] = sum(v for k, v in out.items() if k != "anonymized_placeholders")
    return out


def render_text(merged: dict) -> str:
    L = [f"# {merged['recording']}",
         f"日期 {merged['date']}   {merged['time_start_hkt']} → {merged['time_end_hkt']}",
         f"clip {merged['clip_ok']}/{merged['clip_count']} 成功   segments {merged['segment_count']}   "
         f"眼动 {'有' if merged['has_gaze'] else '无'}   模型 {merged['model']}",
         ""]
    by_clip = {}
    for s in merged["segments"]:
        by_clip.setdefault(s.get("clip_id"), []).append(s)
    for cs in merged["clip_summaries"]:
        cid = cs["clip_id"]
        L.append(f"── clip_{cid:04d}  {cs['time']}  {'' if cs['ok'] else '[FAILED]'}")
        if cs["summary"]:
            L.append(f"   « {cs['summary']} »")
        for s in by_clip.get(cid, []):
            L.append(f"   [{s.get('time_range','')}] {s.get('action','')}")
            if s.get("speech"):
                L.append(f"       语音: {s['speech']}")
            if s.get("environment"):
                L.append(f"       环境: {s['environment']}")
            tv = s.get("text_visible")
            if tv:
                L.append(f"       文字: {', '.join(tv) if isinstance(tv, list) else tv}")
            ob = s.get("objects")
            if ob:
                L.append(f"       物体: {', '.join(ob) if isinstance(ob, list) else ob}")
            if s.get("details"):
                L.append(f"       细节: {s['details']}")
        L.append("")
    return "\n".join(L)


def write_recording_outputs(outdir: Path, merged: dict):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "captions_full.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2))
    with open(outdir / "captions_full.jsonl", "w") as f:
        for s in merged["segments"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (outdir / "captions_full.txt").write_text(render_text(merged))
    report = {k: merged[k] for k in
              ("recording", "date", "clip_count", "clip_ok", "clip_failed",
               "failed_clip_ids", "segment_count", "time_start_hkt", "time_end_hkt",
               "has_gaze", "model", "tokens", "est_cost_usd", "pii_audit")}
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _start_dt(m):
    try:
        return datetime.strptime(m["time_start_hkt"], "%Y-%m-%d %H:%M:%S")
    except Exception:                                         # noqa: BLE001
        return datetime.min


def merge_day(merged_list: list, date: str) -> dict:
    ms = sorted(merged_list, key=_start_dt)
    segs = []
    for m in ms:
        for s in m["segments"]:
            s = dict(s)
            s["recording"] = m["recording"]
            segs.append(s)
    return {
        "date": date,
        "recordings": [{"recording": m["recording"], "start": m["time_start_hkt"],
                        "end": m["time_end_hkt"], "clips": m["clip_count"],
                        "segments": m["segment_count"], "has_gaze": m["has_gaze"]}
                       for m in ms],
        "recording_count": len(ms),
        "segment_count": len(segs),
        "clip_ok": sum(m["clip_ok"] for m in ms),
        "clip_count": sum(m["clip_count"] for m in ms),
        "est_cost_usd": round(sum(m["est_cost_usd"] for m in ms), 4),
        "segments": segs,
    }


def render_day_text(day: dict) -> str:
    L = [f"# {day['date']} 全天 caption",
         f"{day['recording_count']} 条录制   clip {day['clip_ok']}/{day['clip_count']}   "
         f"segments {day['segment_count']}", ""]
    for r in day["recordings"]:
        L.append(f"  · {r['recording']}  {r['start'][11:]}→{r['end'][11:]}  "
                 f"{r['segments']} segs  眼动{'有' if r['has_gaze'] else '无'}")
    L.append("")
    cur = None
    for s in day["segments"]:
        if s.get("recording") != cur:
            cur = s.get("recording")
            L.append(f"\n══ {cur}")
        L.append(f"   [{s.get('time_range','')}] {s.get('action','')}")
        if s.get("speech"):
            L.append(f"       语音: {s['speech']}")
    return "\n".join(L)


def write_day_outputs(outdir: Path, day: dict):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"day_{day['date']}.json").write_text(
        json.dumps(day, ensure_ascii=False, indent=2))
    (outdir / f"day_{day['date']}.txt").write_text(render_day_text(day))
    with open(outdir / f"day_{day['date']}.jsonl", "w") as f:
        for s in day["segments"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
