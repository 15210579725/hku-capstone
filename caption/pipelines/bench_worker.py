#!/usr/bin/env python3
"""吞吐基准：在 HF Job 里一次问清楚 caption 阶段到底被什么卡住。

背景：整条 43 分钟录制里 84 次调用总耗时 3040 worker-秒、墙上时间 530 秒，
有效并发只有 5.7（配了 24）。重试只有 3 次 → 不是 429。本地实测请求体
base64+json 序列化只要 0.023s → 不是 GIL/CPU。剩下的嫌疑是上行带宽和 SDK 连接池。

测什么：
  egress   裸上行带宽（1 路 / 8 路并发）
  A  1440q60 30帧 C=24 共享 client      ← 复现基线
  B  1440q60 30帧 C=24 每线程独立 client ← 测 SDK 连接池
  C  1440q60 30帧 C=64 每线程独立 client ← 测并发天花板
  D   768q60 30帧 C=24 每线程独立 client ← payload 降到 1/3，测带宽假设
  E  1440q35 30帧 C=24 每线程独立 client ← 只降 JPEG 质量
每组用互不重叠的 clip，避免隐式缓存污染。同时记录 input token 以验证
「token 与像素无关」在生产配置下是否成立。

环境变量同 cloud_worker.py，另加 BENCH_ONLY=1 时不写任何产物到 repo。
"""
from __future__ import annotations

import io
import json
import os
import statistics as st
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import caption_core as cc
import cloud_worker as cw
import remote_tar as rt

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", "/tmp/povbench"))


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 裸带宽

def egress_probe(mb=40, conc=1):
    """往 Cloudflare 的公共上传端点打数据，量 job 的裸上行。"""
    import requests
    payload = os.urandom(mb * 1024 * 1024 // conc)

    def one(_):
        t = time.perf_counter()
        try:
            requests.post("https://speed.cloudflare.com/__up", data=payload, timeout=300)
        except Exception as e:                                # noqa: BLE001
            return None, str(e)[:60]
        return time.perf_counter() - t, None

    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, range(conc)))
    wall = time.perf_counter() - t0
    errs = [e for _, e in res if e]
    return {"mb": mb, "conc": conc, "wall": round(wall, 1),
            "mbps": round(mb / wall, 2), "errors": errs[:2]}


# ---------------------------------------------------------------- 每线程 client

_local = threading.local()


def thread_client():
    if not hasattr(_local, "c"):
        _local.c = cc.make_client()
    return _local.c


# ---------------------------------------------------------------- 一组测试

def run_config(name, clip_ids, frames_dir, meta, system_prompt, conc,
               shared_client, model):
    """返回这组的吞吐统计。frames_dir 决定用哪种画质的帧。"""
    shared = cc.make_client() if shared_client else None
    lat, tin, tout, tcache, errs, sent = [], [], [], [], [], []
    lock = threading.Lock()

    def one(cid):
        sd = meta["build_sd"](cid, frames_dir)
        nbytes = cc.payload_bytes(sd)
        parts = cc.build_parts_vertex(sd, system_prompt)
        client = shared if shared_client else thread_client()
        t = time.perf_counter()
        r = cc.run_one(client, model, parts, retries=2, log=lambda *a: None)
        dt = time.perf_counter() - t
        with lock:
            sent.append(nbytes)
            if r["ok"]:
                lat.append(dt)
                tin.append(r["in"])
                tout.append((r["out"] or 0) + (r["think"] or 0))
            else:
                errs.append(r.get("error", "")[:100])
        return r["ok"]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, clip_ids))
    wall = time.perf_counter() - t0

    n = len(lat)
    out = {
        "config": name, "conc": conc, "n": len(clip_ids), "ok": n, "fail": len(errs),
        "wall_s": round(wall, 1),
        "clips_per_s": round(len(clip_ids) / wall, 3),
        "eff_parallel": round(sum(lat) / wall, 1) if wall else 0,
        "lat_med": round(st.median(lat), 1) if n else None,
        "lat_p90": round(sorted(lat)[int(n * 0.9)], 1) if n else None,
        "payload_mb_avg": round(sum(sent) / max(1, len(sent)) / 1e6, 2),
        "upload_mbps": round(sum(sent) * 1.37 / 1e6 / wall, 2),   # base64 膨胀 4/3
        "tok_in_avg": round(sum(tin) / n) if n else None,
        "tok_out_avg": round(sum(tout) / n) if n else None,
        "errors": errs[:2],
    }
    log(f"  {name:32s} wall={out['wall_s']:6.1f}s  {out['clips_per_s']:.3f} clip/s  "
        f"有效并发={out['eff_parallel']:5.1f}  中位={out['lat_med']}s  "
        f"上行={out['upload_mbps']:5.2f}MB/s  payload={out['payload_mb_avg']}MB  "
        f"in_tok={out['tok_in_avg']}  fail={out['fail']}")
    return out


# ---------------------------------------------------------------- 主流程

def main():
    c = cw.cfg()
    cw.setup_adc()
    rec = c["rec"]
    model = c["model"]
    log(f"=== BENCH {rec} ===")

    results = {"recording": rec, "model": model, "at": time.strftime("%F %T")}

    # 0) 裸上行
    log("\n[0] 裸上行带宽")
    for conc in (1, 8):
        r = egress_probe(mb=40, conc=conc)
        log(f"  {conc} 路并发: {r['mbps']} MB/s ({r['wall']}s / {r['mb']}MB) {r['errors']}")
        results[f"egress_c{conc}"] = r

    # 1) 解包，三种画质各存一份
    src = rt.open_source(c["tar"], mount_root=c["mount"], token=c["token"])
    log(f"\n[1] 解包（源: {'挂载' if src.seekable else 'Range'}）")
    entries, buf, base = cw.read_tail_adaptive(src)
    gaze_rows, ocr_texts = cw.extract_tail_payload(entries, buf, base)
    del buf
    trails = cc.build_trails(gaze_rows) if gaze_rows else {}
    sec = rt.locate_sections(src)
    tr = sec["transcript"]
    intervals = cc.parse_transcript(
        src.read(tr["data_offset"], tr["size"]).decode("utf-8", "replace") if tr else "")

    VARIANTS = {"1440q60": (1440, 60), "768q60": (768, 60), "1440q35": (1440, 35)}
    dirs = {k: WORK / rec / k for k in VARIANTS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_per_clip = c["clip_seconds"]
    clip_frames: dict[int, list] = {}
    pool = ThreadPoolExecutor(c["render_workers"])
    pending = []

    def render_all(fname, data):
        pts, valid = trails.get(fname, ([], False)) if trails else (None, False)
        for k, (dim, q) in VARIANTS.items():
            (dirs[k] / fname).write_bytes(cc.render_frame(data, pts, valid, size=dim, quality=q))

    t0 = time.perf_counter()
    nb = 0
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
        clip_frames.setdefault(fi // n_per_clip, []).append(fn)
        nb += size
        pending.append(pool.submit(render_all, fn, data))
        if len(pending) >= 32:
            for f in pending[:16]:
                f.result()
            pending = pending[16:]
    for f in pending:
        f.result()
    pool.shutdown(wait=True)
    dt = time.perf_counter() - t0
    log(f"  {sum(len(v) for v in clip_frames.values())} 帧 / {len(clip_frames)} clip, "
        f"{nb/1e9:.2f} GB, {dt:.0f}s ({nb/dt/1e6:.0f} MB/s), 三种画质各一份")
    for k, d in dirs.items():
        tot = sum(p.stat().st_size for p in d.glob("*.jpg"))
        log(f"    {k}: 单帧均值 {tot/max(1,len(list(d.glob('*.jpg'))))/1024:.0f} KB")

    system_prompt = (HERE / "prompts" / "v6.txt").read_text().strip()

    def build_sd(cid, frames_dir):
        names = sorted(clip_frames[cid], key=cc.frame_index)
        frames = [{"name": fn, "hkt": cc.hkt_hms(fn),
                   "jpg": (frames_dir / fn).read_bytes()} for fn in names]
        ocrs = []
        for fn in names:
            got = ocr_texts.get(cc.frame_index(fn))
            if got and got[1]:
                ocrs.append(f"[{cc.hkt_hms(fn)} {got[0]}]: {got[1]}")
        t0_ = cc.frame_time(names[0])
        from datetime import timedelta
        return {"frames": frames, "ocr_texts": ocrs,
                "transcript": "\n".join(cc.transcript_for_window(
                    intervals, cc.frame_index(names[0]), len(names))),
                "n_frames": len(frames),
                "clip_start": t0_.strftime("%Y-%m-%d %H:%M:%S"),
                "clip_end": (t0_ + timedelta(seconds=len(names))).strftime("%Y-%m-%d %H:%M:%S")}

    meta = {"build_sd": build_sd}

    # 2) 矩阵（互不重叠的 clip）
    ids = sorted(clip_frames)
    plan = [
        ("A 1440q60 C24 共享client", 16, 24, True, "1440q60"),
        ("B 1440q60 C24 每线程client", 16, 24, False, "1440q60"),
        ("C 1440q60 C64 每线程client", 24, 64, False, "1440q60"),
        ("D  768q60 C24 每线程client", 16, 24, False, "768q60"),
        ("E 1440q35 C24 每线程client", 15, 24, False, "1440q35"),
    ]
    log(f"\n[2] 吞吐矩阵（{len(ids)} 个 clip 切成互不重叠的 5 组）")
    pos = 0
    runs = []
    for name, n, conc, shared, variant in plan:
        block = ids[pos:pos + n]
        pos += n
        if not block:
            log(f"  {name}: clip 不够，跳过")
            continue
        runs.append(run_config(name, block, dirs[variant], meta,
                               system_prompt, conc, shared, model))
    results["runs"] = runs

    log("\n=== 汇总 ===")
    base_cps = runs[0]["clips_per_s"] if runs else 1
    for r in runs:
        log(f"  {r['config']:32s} {r['clips_per_s']:.3f} clip/s "
            f"({r['clips_per_s']/base_cps:4.2f}× 基线)  有效并发 {r['eff_parallel']:5.1f}  "
            f"上行 {r['upload_mbps']:5.2f} MB/s  in_tok {r['tok_in_avg']}")

    out = WORK / "bench.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if os.environ.get("BENCH_ONLY") != "1":
        from huggingface_hub import HfApi
        api = HfApi(token=c["token"])
        cw.commit(api, c["out_repo"], [(out, f"_bench/bench_{time.strftime('%Y%m%d_%H%M%S')}.json")],
                  "throughput bench")
    log("BENCH_DONE")


if __name__ == "__main__":
    main()
