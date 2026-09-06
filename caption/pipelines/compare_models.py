#!/usr/bin/env python3
"""同一批 clip 换模型跑，出并排对照，用来定「750 小时用哪个模型」。

用本地已 pull 下来的 clip 包（out/<date>/<rec>/clips/clip_XXXX.tar，里面就是
1440px 带注视圈的帧），所以不用再解一次 tar。3.7-flash 的结果直接取
captions_full.json 里已有的，不重复花钱。

  python compare_models.py --rec <rec> --models gemini-3.5-flash-lite,gemini-3.1-flash-lite
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import caption_core as cc

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

PRICE = {   # $/1M token，(in, out)；out 含 thinking
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def load_clip_tar(p: Path):
    with tarfile.open(p) as tf:
        names = sorted(m.name for m in tf.getmembers()
                       if m.name.startswith("frames/") and m.name.endswith(".jpg"))
        frames = [{"name": Path(n).name, "hkt": cc.hkt_hms(Path(n).name),
                   "jpg": tf.extractfile(n).read()} for n in names]
        meta = json.loads(tf.extractfile("meta.json").read())
    return frames, meta


def fetch_context(tar_path: str):
    """clip 包里没有 OCR/transcript，从远端 tar 拉一次（尾部 ~17MB + 100KB）。
    不补的话 lite 模型少了上下文，对照就不公平。"""
    import remote_tar as rt
    import cloud_worker as cw
    src = rt.HttpRangeSource(tar_path)
    entries, buf, base = cw.read_tail_adaptive(src)
    _, ocr = cw.extract_tail_payload(entries, buf, base)
    del buf
    sec = rt.locate_sections(src)
    tr = sec["transcript"]
    txt = src.read(tr["data_offset"], tr["size"]).decode("utf-8", "replace") if tr else ""
    src.close()
    return ocr, cc.parse_transcript(txt)


def build_sd(frames, meta, ocr_texts, intervals):
    t0 = cc.frame_time(frames[0]["name"])
    ocrs = []
    for fr in frames:
        got = ocr_texts.get(cc.frame_index(fr["name"]))
        if got and got[1]:
            ocrs.append(f"[{fr['hkt']} {got[0]}]: {got[1]}")
    rel = cc.frame_index(frames[0]["name"])
    return {"frames": frames, "ocr_texts": ocrs,
            "transcript": "\n".join(cc.transcript_for_window(intervals, rel, len(frames))),
            "n_frames": len(frames),
            "clip_start": meta.get("clip_start_hkt") or t0.strftime("%Y-%m-%d %H:%M:%S"),
            "clip_end": meta.get("clip_end_hkt")
            or (t0 + timedelta(seconds=len(frames))).strftime("%Y-%m-%d %H:%M:%S")}


def stats(parsed):
    if not parsed:
        return {"segments": 0, "speech": 0, "text_items": 0, "chars": 0, "objects": 0}
    segs = parsed.get("segments", []) or []
    tv = sum(len(s.get("text_visible") or []) for s in segs)
    ob = sum(len(s.get("objects") or []) for s in segs)
    ch = sum(len(str(s.get("action", ""))) + len(str(s.get("details", "")))
             + len(str(s.get("environment", ""))) for s in segs)
    return {"segments": len(segs),
            "speech": sum(1 for s in segs if s.get("speech")),
            "text_items": tv, "objects": ob, "chars": ch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--date")
    ap.add_argument("--models", default="gemini-3.5-flash-lite,gemini-3.1-flash-lite")
    ap.add_argument("--clips", help="逗号分隔的 clip id；默认用本地全部 clip 包")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tar", required=True, help="源 tar 路径，用来补 OCR/transcript 上下文")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="帧抽稀：2 = 每 2 秒 1 帧（15 帧/clip），直接砍一半 input token")
    args = ap.parse_args()

    date = args.date
    if not date:
        cands = [p for p in OUT.glob("*/" + args.rec) if p.is_dir()]
        if not cands:
            sys.exit(f"找不到 out/*/{args.rec}")
        date = cands[0].parent.name
    recdir = OUT / date / args.rec
    tars = sorted((recdir / "clips").glob("clip_*.tar"))
    if args.clips:
        keep = {int(x) for x in args.clips.split(",")}
        tars = [p for p in tars if int(p.stem.split("_")[1]) in keep]
    if not tars:
        sys.exit("本地没有 clip 包，先 pipeline.py pull --rec ... --sample N")

    full = json.loads((recdir / "captions_full.json").read_text())
    base_model = full.get("model", "gemini-3.7-flash")
    base_by_clip = {}
    for s in full["segments"]:
        base_by_clip.setdefault(s["clip_id"], []).append(s)

    prompt = (HERE / "prompts" / "v6.txt").read_text().strip()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"对照 {len(tars)} 个 clip × {len(models)} 个模型（基线 {base_model} 用已有结果）\n")

    print("拉 OCR/transcript 上下文（尾部 Range，一次性）...", flush=True)
    ocr_texts, intervals = fetch_context(args.tar)
    print(f"  OCR {len(ocr_texts)} 帧, transcript {len(intervals)} 区间\n")
    loaded = []
    for p in tars:
        frames, meta = load_clip_tar(p)
        if args.frame_stride > 1:
            frames = frames[::args.frame_stride]
        loaded.append((int(p.stem.split("_")[1]),
                       build_sd(frames, meta, ocr_texts, intervals)))

    client = cc.make_client()
    results = {}

    def one(job):
        model, cid, sd = job
        parts = cc.build_parts_vertex(sd, prompt)
        t = time.perf_counter()
        r = cc.run_one(client, model, parts, retries=3, log=lambda *a: None)
        dt = time.perf_counter() - t
        parsed, note = (cc.parse_caption_json(r.get("content", "")) if r["ok"] else (None, "err"))
        return model, cid, {"ok": r["ok"] and parsed is not None, "t": round(dt, 1),
                            "in": r.get("in"), "out": (r.get("out") or 0) + (r.get("think") or 0),
                            "parsed": parsed, "note": note, "error": r.get("error", "")[:120]}

    jobs = [(m, cid, sd) for m in models for cid, sd in loaded]
    with ThreadPoolExecutor(args.workers) as ex:
        for model, cid, res in ex.map(one, jobs):
            results.setdefault(model, {})[cid] = res
            print(f"  {model:24s} clip_{cid:04d} {'✓' if res['ok'] else '✗'} "
                  f"{res['t']:5.1f}s in={res['in']} out={res['out']}")

    # 汇总表
    print(f"\n{'模型':26s} {'段数':>5} {'语音':>5} {'文字条':>7} {'物体':>5} {'描述字数':>9} "
          f"{'in':>7} {'out':>6} {'延迟':>6} {'$/clip':>8}")
    rows = []
    agg_base = {"segments": 0, "speech": 0, "text_items": 0, "objects": 0, "chars": 0}
    for cid, _ in loaded:
        s = stats({"segments": base_by_clip.get(cid, [])})
        for k in agg_base:
            agg_base[k] += s[k]
    bi = sum((c.get("usage") or {}).get("in") or 0 for c in [])  # 基线 token 从 report 取
    rep = json.loads((recdir / "report.json").read_text())
    n = len(loaded)
    b_in = rep["tokens"]["in"] // rep["clip_count"]
    b_out = rep["tokens"].get("out_billable", rep["tokens"].get("out", 0)) // rep["clip_count"]
    pin, pout = PRICE.get(base_model, (0.75, 3.75))
    print(f"{base_model + ' (基线)':26s} {agg_base['segments']/n:5.1f} {agg_base['speech']/n:5.1f} "
          f"{agg_base['text_items']/n:7.1f} {agg_base['objects']/n:5.1f} {agg_base['chars']/n:9.0f} "
          f"{b_in:7d} {b_out:6d} {'—':>6} "
          f"{b_in/1e6*pin + b_out/1e6*pout:8.4f}")
    rows.append((base_model, b_in, b_out))
    for m in models:
        rs = results[m]
        agg = {"segments": 0, "speech": 0, "text_items": 0, "objects": 0, "chars": 0}
        tin = tout = tt = ok = 0
        for cid, r in rs.items():
            if not r["ok"]:
                continue
            ok += 1
            s = stats(r["parsed"])
            for k in agg:
                agg[k] += s[k]
            tin += r["in"] or 0
            tout += r["out"] or 0
            tt += r["t"]
        if not ok:
            print(f"{m:26s} 全部失败")
            continue
        pin, pout = PRICE.get(m, (0.75, 3.75))
        ci = tin / ok / 1e6 * pin + tout / ok / 1e6 * pout
        print(f"{m:26s} {agg['segments']/ok:5.1f} {agg['speech']/ok:5.1f} "
              f"{agg['text_items']/ok:7.1f} {agg['objects']/ok:5.1f} {agg['chars']/ok:9.0f} "
              f"{tin//ok:7d} {tout//ok:6d} {tt/ok:6.1f} {ci:8.4f}")
        rows.append((m, tin // ok, tout // ok))

    outp = recdir / "model_compare.json"
    outp.write_text(json.dumps(
        {"base_model": base_model, "clips": [c for c, _ in loaded],
         "base": {str(c): base_by_clip.get(c, []) for c, _ in loaded},
         "models": {m: {str(c): r["parsed"] for c, r in rs.items()}
                    for m, rs in results.items()}}, ensure_ascii=False, indent=2))

    # 逐段并排（取第一个 clip）
    cid0 = loaded[0][0]
    print(f"\n===== clip_{cid0:04d} 逐段并排 =====")
    print(f"\n--- {base_model} (基线) ---")
    for s in base_by_clip.get(cid0, [])[:6]:
        print(f"  [{s.get('time_range','')}] {s.get('action','')}")
        if s.get("speech"):
            print(f"      语音: {s['speech']}")
        if s.get("text_visible"):
            print(f"      文字: {', '.join(s['text_visible'][:6])}")
    for m in models:
        r = results[m].get(cid0)
        print(f"\n--- {m} ---")
        if not r or not r["ok"]:
            print("  (失败)", (r or {}).get("error", ""))
            continue
        for s in (r["parsed"].get("segments") or [])[:6]:
            print(f"  [{s.get('time_range','')}] {s.get('action','')}")
            if s.get("speech"):
                print(f"      语音: {s['speech']}")
            if s.get("text_visible"):
                print(f"      文字: {', '.join(s['text_visible'][:6])}")
    print(f"\n完整对照落盘: {outp}")


if __name__ == "__main__":
    main()
