#!/usr/bin/env python3
"""第二轮吞吐实验：容量从哪来。

第一轮结论：瓶颈不是带宽、不是客户端、不是 payload 大小（768px 把 payload 降到 1/5
吞吐没变好，且 in_tok 完全相同），而是 **Vertex 对本项目的动态共享配额（DSQ）在降速**
——表现为延迟膨胀而不是 429。且 `gemini-3.7-flash` 只在 `global` 端点存在，
19 个区域端点全 404，多区域摊配额这条路堵死。

这一轮问三件事：
  R  冷却后在线吞吐能否恢复到起步水平（区分「突发额度耗尽」和「永久降速」）
  M  换模型能拿到多少容量 + 各自 token 成本（lite 系列可能便宜数倍）
  B  Batch API 端到端能否走通、周转多久（独立容量池 + 半价，是 750 小时的主选项）

环境变量同 cloud_worker.py，另加：
  BENCH2_MODELS   逗号分隔，默认 gemini-3.5-flash-lite,gemini-3.8-flash
  BENCH2_BATCH_N  Batch 测试用多少 clip，默认 24；设 0 跳过
  GCS_BUCKET      默认 hku-capstone-caption-frames
"""
from __future__ import annotations

import base64
import io
import json
import os
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import caption_core as cc
import cloud_worker as cw
import remote_tar as rt

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", "/tmp/povbench2"))
BUCKET = os.environ.get("GCS_BUCKET", "hku-capstone-caption-frames")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 在线吞吐

def online_block(name, model, clip_ids, frames_dir, build_sd, system_prompt, conc=24):
    client = cc.make_client()
    lat, tin, tout, errs = [], [], [], []
    import threading
    lock = threading.Lock()

    def one(cid):
        sd = build_sd(cid, frames_dir)
        parts = cc.build_parts_vertex(sd, system_prompt)
        t = time.perf_counter()
        r = cc.run_one(client, model, parts, retries=2, log=lambda *a: None)
        dt = time.perf_counter() - t
        with lock:
            if r["ok"]:
                lat.append(dt)
                tin.append(r["in"])
                tout.append((r["out"] or 0) + (r["think"] or 0))
            else:
                errs.append(r.get("error", "")[:120])
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, clip_ids))
    wall = time.perf_counter() - t0
    n = len(lat)
    tpm = sum(tin) / wall * 60 if wall else 0
    out = {"name": name, "model": model, "n": len(clip_ids), "ok": n, "fail": len(errs),
           "wall_s": round(wall, 1), "clips_per_s": round(len(clip_ids) / wall, 3),
           "eff_parallel": round(sum(lat) / wall, 1) if wall else 0,
           "lat_med": round(st.median(lat), 1) if n else None,
           "in_tok_avg": round(sum(tin) / n) if n else None,
           "out_tok_avg": round(sum(tout) / n) if n else None,
           "input_tpm": round(tpm), "errors": errs[:2]}
    log(f"  {name:34s} {out['clips_per_s']:.3f} clip/s  有效并发={out['eff_parallel']:5.1f}  "
        f"中位={out['lat_med']}s  输入 {out['input_tpm']/1000:.0f}K tok/分  "
        f"in={out['in_tok_avg']} out={out['out_tok_avg']}  fail={out['fail']}")
    return out


# ---------------------------------------------------------------- GCS

def gcs_upload(local: Path, obj: str, token: str):
    import requests
    url = (f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o"
           f"?uploadType=media&name={obj}")
    with open(local, "rb") as f:
        r = requests.post(url, data=f, timeout=1800,
                          headers={"Authorization": f"Bearer {token}",
                                   "Content-Type": "application/json"})
    r.raise_for_status()
    return f"gs://{BUCKET}/{obj}"


def gcp_token():
    import google.auth
    import google.auth.transport.requests
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


# ---------------------------------------------------------------- Batch

def batch_block(model, clip_ids, frames_dir, build_sd, system_prompt, timeout_s=3000):
    """把 clip 写成 JSONL 传 GCS，提交 Vertex batch，轮询到结束。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jl = WORK / f"batch_{stamp}.jsonl"
    t0 = time.perf_counter()
    nbytes = 0
    with open(jl, "w") as f:
        for cid in clip_ids:
            sd = build_sd(cid, frames_dir)
            parts = [{"text": system_prompt + "\n\n" + cc.build_context_text(sd)}]
            for fr in sd["frames"]:
                parts.append({"text": f"[{fr['hkt']} HKT — {fr['name']}]"})
                parts.append({"inlineData": {"mimeType": "image/jpeg",
                                             "data": base64.b64encode(fr["jpg"]).decode()}})
            parts.append({"text": "Now produce the fine-grained, first-person dense temporal "
                                  "caption JSON with absolute HKT timestamps."})
            row = {"request": {"contents": [{"role": "user", "parts": parts}],
                               "generationConfig": {"temperature": cc.TEMPERATURE,
                                                    "maxOutputTokens": cc.MAX_OUTPUT_TOKENS}}}
            line = json.dumps(row, ensure_ascii=False)
            nbytes += len(line)
            f.write(line + "\n")
    build_s = time.perf_counter() - t0
    log(f"  JSONL {jl.stat().st_size/1e6:.0f} MB / {len(clip_ids)} clip，构建 {build_s:.0f}s")

    tok = gcp_token()
    t0 = time.perf_counter()
    src = gcs_upload(jl, f"batch_in/{jl.name}", tok)
    up_s = time.perf_counter() - t0
    log(f"  上传 GCS {up_s:.0f}s ({jl.stat().st_size/1e6/up_s:.1f} MB/s) → {src}")

    from google import genai
    client = cc.make_client()
    dest = f"gs://{BUCKET}/batch_out/{stamp}/"
    t0 = time.perf_counter()
    job = client.batches.create(model=model, src=src,
                                config=genai.types.CreateBatchJobConfig(dest=dest))
    log(f"  batch 已提交: {job.name}  state={job.state}")
    last = None
    while time.perf_counter() - t0 < timeout_s:
        time.sleep(20)
        job = client.batches.get(name=job.name)
        s = str(job.state)
        if s != last:
            log(f"    [{time.perf_counter()-t0:5.0f}s] {s}")
            last = s
        if "SUCCEEDED" in s or "FAILED" in s or "CANCELLED" in s or "EXPIRED" in s:
            break
    wall = time.perf_counter() - t0
    ok = "SUCCEEDED" in str(job.state)
    log(f"  batch {'✓' if ok else '✗'} {job.state}  周转 {wall/60:.1f} 分钟"
        f"{'  err=' + str(job.error)[:200] if not ok else ''}")
    return {"model": model, "n": len(clip_ids), "state": str(job.state),
            "turnaround_min": round(wall / 60, 1), "jsonl_mb": round(jl.stat().st_size / 1e6, 1),
            "build_s": round(build_s), "upload_s": round(up_s), "dest": dest,
            "clips_per_s": round(len(clip_ids) / wall, 4) if ok else None}


# ---------------------------------------------------------------- 主流程

def main():
    c = cw.cfg()
    cw.setup_adc()
    rec = c["rec"]
    log(f"=== BENCH2 {rec} ===")
    results = {"recording": rec, "at": time.strftime("%F %T")}

    # 解包（1440q60 + 768q60 两份）
    src = rt.open_source(c["tar"], mount_root=c["mount"], token=c["token"])
    entries, buf, base = cw.read_tail_adaptive(src)
    gaze_rows, ocr_texts = cw.extract_tail_payload(entries, buf, base)
    del buf
    trails = cc.build_trails(gaze_rows) if gaze_rows else {}
    sec = rt.locate_sections(src)
    tr = sec["transcript"]
    intervals = cc.parse_transcript(
        src.read(tr["data_offset"], tr["size"]).decode("utf-8", "replace") if tr else "")

    VAR = {"1440q60": (1440, 60), "768q60": (768, 60)}
    dirs = {k: WORK / rec / k for k in VAR}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    clip_frames: dict[int, list] = {}
    pool = ThreadPoolExecutor(c["render_workers"])
    pending = []

    def render_all(fn, data):
        pts, valid = trails.get(fn, ([], False)) if trails else (None, False)
        for k, (dim, q) in VAR.items():
            (dirs[k] / fn).write_bytes(cc.render_frame(data, pts, valid, size=dim, quality=q))

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
        pending.append(pool.submit(render_all, fn, data))
        if len(pending) >= 32:
            for f in pending[:16]:
                f.result()
            pending = pending[16:]
    for f in pending:
        f.result()
    pool.shutdown(wait=True)
    log(f"解包 {len(clip_frames)} clip, {time.perf_counter()-t0:.0f}s")

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
        return {"frames": frames, "ocr_texts": ocrs,
                "transcript": "\n".join(cc.transcript_for_window(
                    intervals, cc.frame_index(names[0]), len(names))),
                "n_frames": len(frames),
                "clip_start": t0_.strftime("%Y-%m-%d %H:%M:%S"),
                "clip_end": (t0_ + timedelta(seconds=len(names))).strftime("%Y-%m-%d %H:%M:%S")}

    ids = sorted(clip_frames)
    pos = 0
    runs = []

    # R：3.7-flash 冷却后能否恢复
    log("\n[R] 冷却后恢复测试 gemini-3.7-flash")
    blk = ids[pos:pos + 12]; pos += 12
    runs.append(online_block("R 3.7-flash 恢复", "gemini-3.7-flash", blk,
                             dirs["1440q60"], build_sd, system_prompt))

    # M：换模型
    models = [m.strip() for m in os.environ.get(
        "BENCH2_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash").split(",") if m.strip()]
    log("\n[M] 换模型")
    for m in models:
        blk = ids[pos:pos + 12]; pos += 12
        if not blk:
            break
        runs.append(online_block(f"M {m}", m, blk, dirs["1440q60"],
                                 build_sd, system_prompt))
    results["online"] = runs

    # B：Batch
    bn = int(os.environ.get("BENCH2_BATCH_N", "24"))
    if bn:
        log(f"\n[B] Batch API（{bn} clip，用 768px 压小 JSONL；token 与 1440px 相同）")
        blk = ids[pos:pos + bn]; pos += bn
        try:
            results["batch"] = batch_block(c["model"], blk, dirs["768q60"],
                                           build_sd, system_prompt)
        except Exception as e:                                 # noqa: BLE001
            log(f"  batch 失败: {type(e).__name__}: {str(e)[:400]}")
            results["batch"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}

    log("\n=== 汇总 ===")
    for r in runs:
        log(f"  {r['name']:34s} {r['clips_per_s']:.3f} clip/s  "
            f"{r['input_tpm']/1000:6.0f}K tok/分  in={r['in_tok_avg']} out={r['out_tok_avg']}")
    if results.get("batch"):
        log(f"  batch: {results['batch']}")

    out = WORK / "bench2.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    from huggingface_hub import HfApi
    api = HfApi(token=c["token"])
    cw.commit(api, c["out_repo"],
              [(out, f"_bench/bench2_{time.strftime('%Y%m%d_%H%M%S')}.json")], "bench2")
    log("BENCH2_DONE")


if __name__ == "__main__":
    main()
