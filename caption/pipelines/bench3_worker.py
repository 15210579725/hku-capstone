#!/usr/bin/env python3
"""第三轮：并发上限是按 GCP 项目算，还是按客户端/进程算？

这是 750 小时能不能在 1–2 天跑完的决定性问题：
  · 如果按客户端算 → 多开 HF Job 就能线性扩容量，一个项目够用
  · 如果按项目算   → 开再多 job 也是白搭，必须开多个 GCP 项目分摊

做法：同时起 N 个一模一样的 job，各自跑同样多的 clip，记录各自吞吐和
起止的墙上时钟。拿单 job 基线（bench2 里 gemini-3.5-flash-lite 0.301 clip/s）对比：
每个 job 仍然 ~0.3 → 线性扩；每个掉到 ~0.3/N → 项目级封顶。

环境变量：BENCH3_MODEL（默认 gemini-3.5-flash-lite）、BENCH3_N（默认 20 clip）、
BENCH3_TAG（分组标签，只用于日志辨认）。其余同 cloud_worker.py。
"""
from __future__ import annotations

import json
import os
import statistics as st
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import caption_core as cc
import cloud_worker as cw
import remote_tar as rt

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", "/tmp/povbench3"))


def log(*a):
    print(*a, flush=True)


def main():
    c = cw.cfg()
    cw.setup_adc()
    model = os.environ.get("BENCH3_MODEL", "gemini-3.5-flash-lite")
    n_want = int(os.environ.get("BENCH3_N", "20"))
    tag = os.environ.get("BENCH3_TAG", "solo")
    log(f"=== BENCH3 tag={tag} model={model} n={n_want} conc={c['workers']} ===")

    # 只解包够用的 clip（从 tar 头部拿，成本最低）
    src = rt.open_source(c["tar"], mount_root=c["mount"], token=c["token"])
    entries, buf, base = cw.read_tail_adaptive(src)
    gaze_rows, ocr_texts = cw.extract_tail_payload(entries, buf, base)
    del buf
    trails = cc.build_trails(gaze_rows) if gaze_rows else {}
    sec = rt.locate_sections(src)
    tr = sec["transcript"]
    intervals = cc.parse_transcript(
        src.read(tr["data_offset"], tr["size"]).decode("utf-8", "replace") if tr else "")

    fdir = WORK / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    clip_frames: dict[int, list] = {}
    pool = ThreadPoolExecutor(c["render_workers"])
    pend = []

    def render(fn, data):
        pts, valid = trails.get(fn, ([], False)) if trails else (None, False)
        (fdir / fn).write_bytes(cc.render_frame(data, pts, valid))

    t0 = time.perf_counter()
    for name, size, data in rt.iter_members(
            src, start=sec["pictures_start"],
            want=lambda n: "/picture/masked/" in n and n.lower().endswith(".jpg")):
        if data is None:
            if "/picture/ocr_text/" in name and clip_frames:
                break
            continue
        fn = os.path.basename(name)
        fi = cc.frame_index(fn)
        if fi is None:
            continue
        clip_frames.setdefault(fi // c["clip_seconds"], []).append(fn)
        pend.append(pool.submit(render, fn, data))
        if len(pend) >= 32:
            for f in pend[:16]:
                f.result()
            pend = pend[16:]
        if len(clip_frames) > n_want:
            break
    for f in pend:
        f.result()
    pool.shutdown(wait=True)
    if len(clip_frames) > n_want:
        del clip_frames[min(clip_frames)]
    log(f"解包 {len(clip_frames)} clip, {time.perf_counter()-t0:.0f}s")

    system_prompt = (HERE / "prompts" / "v6.txt").read_text().strip()

    def build_sd(cid):
        names = sorted(clip_frames[cid], key=cc.frame_index)
        frames = [{"name": fn, "hkt": cc.hkt_hms(fn),
                   "jpg": (fdir / fn).read_bytes()} for fn in names]
        ocrs = []
        for fn in names:
            got = ocr_texts.get(cc.frame_index(fn))
            if got and got[1]:
                ocrs.append(f"[{cc.hkt_hms(fn)} {got[0]}]: {got[1]}")
        t0_ = cc.frame_time(names[0])
        return {"frames": frames, "ocr_texts": ocrs,
                "transcript": "\n".join(cc.transcript_for_window(
                    intervals, cc.frame_index(names[0]), len(names))),
                "n_frames": len(frames),
                "clip_start": t0_.strftime("%Y-%m-%d %H:%M:%S"),
                "clip_end": (t0_ + timedelta(seconds=len(names))).strftime("%Y-%m-%d %H:%M:%S")}

    # 同步起跑：所有 job 都等到下一个整分钟的 :00 才开始，保证真正重叠
    sync = os.environ.get("BENCH3_SYNC_EPOCH")
    if sync:
        wait = float(sync) - time.time()
        if wait > 0:
            log(f"同步等待 {wait:.0f}s 到统一起跑点")
            time.sleep(wait)

    client = cc.make_client()
    lat, tin, tout, errs = [], [], [], []
    lock = threading.Lock()

    def one(cid):
        parts = cc.build_parts_vertex(build_sd(cid), system_prompt)
        t = time.perf_counter()
        r = cc.run_one(client, model, parts, retries=2, log=lambda *a: None)
        dt = time.perf_counter() - t
        with lock:
            if r["ok"]:
                lat.append(dt); tin.append(r["in"]); tout.append((r["out"] or 0) + (r["think"] or 0))
            else:
                errs.append(r.get("error", "")[:100])

    ids = sorted(clip_frames)
    start_wall = time.time()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(c["workers"]) as ex:
        list(ex.map(one, ids))
    wall = time.perf_counter() - t0
    end_wall = time.time()

    n = len(lat)
    res = {"tag": tag, "model": model, "conc": c["workers"], "n": len(ids), "ok": n,
           "fail": len(errs), "wall_s": round(wall, 1),
           "clips_per_s": round(len(ids) / wall, 3),
           "eff_parallel": round(sum(lat) / wall, 1) if wall else 0,
           "lat_med": round(st.median(lat), 1) if n else None,
           "input_tpm": round(sum(tin) / wall * 60) if wall else 0,
           "in_avg": round(sum(tin) / n) if n else None,
           "out_avg": round(sum(tout) / n) if n else None,
           "start_epoch": round(start_wall), "end_epoch": round(end_wall),
           "errors": errs[:2]}
    log("RESULT " + json.dumps(res, ensure_ascii=False))
    log(f"  {tag}: {res['clips_per_s']:.3f} clip/s  有效并发={res['eff_parallel']}  "
        f"中位={res['lat_med']}s  {res['input_tpm']/1000:.0f}K tok/分  fail={res['fail']}")
    log("BENCH3_DONE")


if __name__ == "__main__":
    main()
