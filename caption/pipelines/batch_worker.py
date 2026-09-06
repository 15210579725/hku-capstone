#!/usr/bin/env python3
"""HF Job worker：一条录制走完 v9 双流程 caption，pass1/pass2 都走 Vertex Batch。

与同目录另外两个入口的分工
  cloud_worker.py + caption_core.py   v6 单遍，在线
  cloud_worker.py + caption_v9.py     v9 双遍，在线（CAPTION_VERSION=v9）
  batch_worker.py + caption_v9.py     v9 双遍，**Batch**  ← 本文件

caption 逻辑一行不重写：prompt / build_context / align_speech / parse_json /
encode_img 全部 import 自 caption_v9（它锁在 quality_verify/captions/frozen/ 的
SHA256 冻结快照上）。本文件只负责把「一次在线调用」翻译成「一行 batch 请求」。

为什么 pass1/pass2 走 Batch 而不是在线
  在线 3.7-flash 会被 Vertex 动态共享配额单调降速（实测 0.435 → 0.060 clip/s，
  冷却 20 分钟不恢复）；Batch 是独立容量池，同一个被降速的模型照跑不误，还半价。

为什么帧走 gs:// URI 而不是 inline base64
  Batch 单请求 payload 上限 10 MB，v9 pass1 是 2880px q80 × 30 帧 = 33–44 MB，
  inline 这条路堵死。冻结脚本的 call_with_fallback 超过 20MB 时本来就走
  Part.from_uri，token 计数完全相同 —— 只是把「每次调用临时传一遍」换成
  「传一次、pass1/pass2 复用」。

阶段
  A 解包   走 picture/masked/ → 2880px 画注视圈 q80 → 传 GCS → 记 tar 内偏移
  B pass0  silero-vad + Gemini 定语音（在线：payload 几十 KB，不吃 DSQ）
  C pass1  帧 URI @HIGH + prompt + REGION_ADDENDUM → Batch → 轮询
  D crop   按偏移从挂载的 tar 随机回读原帧 → 裁 → 传 GCS
  E pass2  crop URI @ULTRA + READ_PROMPT + pass1 回传 → Batch → 轮询
  F 合并   pass2 修订覆盖 pass1 → clip_XXXX.json → HF → manifest/report

环境变量（在 cloud_worker.cfg() 的基础上加）
  RUN_ID              本次全量跑的标识，决定 GCS 路径与断点续跑的归属
  GCS_BUCKET          默认 hku-capstone-caption-frames
  UPLOAD_WORKERS=24   GCS 上传并发
  SPEECH_WORKERS=12   pass0 在线并发
  BATCH_TIMEOUT_MIN=240   单个 batch 轮询硬上限，超时 cancel 并如实报告
  BATCH_POLL_SEC=30
  NO_SPEECH=0         1 = 跳过 pass0（只用 transcript）
  PROMPT_VARIANT      auto（默认，按有无眼动选）| v9 | v9_nogaze
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

from PIL import Image
from google.genai import types

import caption_core as cc
import caption_v9 as v9
import cloud_worker as cw
import merge as mg
import remote_tar as rt

log = cw.log
beat = cw.beat
WORK = cw.WORK
HERE = Path(__file__).resolve().parent

# Batch 半价；pass0 走在线算全价。thinking token 按 output 计费。
PRICE = {"batch": (0.375, 1.875), "online": (0.75, 3.75)}

CLIP_TAG = "__povclip"          # 埋在 GCS 路径里，用来从回显的 request 认领 clip


# ---------------------------------------------------------------- 配置

def cfg():
    c = cw.cfg()
    c.update({
        "run_id": os.environ.get("RUN_ID", time.strftime("%Y%m%d")),
        "bucket": os.environ.get("GCS_BUCKET", v9.GCS_BUCKET),
        "upload_workers": int(os.environ.get("UPLOAD_WORKERS", "24")),
        "speech_workers": int(os.environ.get("SPEECH_WORKERS", "12")),
        "batch_timeout_min": float(os.environ.get("BATCH_TIMEOUT_MIN", "240")),
        "batch_poll_sec": float(os.environ.get("BATCH_POLL_SEC", "30")),
        # ONLINE=1 时 pass1/pass2 不走 batch，直接用线程池打在线接口。
        # 2026-09-05：Vertex batch 对这两个 credit 项目的周转不可预测
        # （3.5-flash 5 行 6 分钟、30 行 23 分钟仍未完，3.7-flash 5 行 101 分钟仍未完），
        # 而在线并发 8 实测 894 clip/h、零 429。贵一倍，但能当天交付。
        "online": os.environ.get("ONLINE", "") not in ("", "0", "false"),
        "online_workers": int(os.environ.get("ONLINE_WORKERS", "12")),
        "no_speech": os.environ.get("NO_SPEECH", "0") == "1",
        "speech_deadline_min": float(os.environ.get("SPEECH_DEADLINE_MIN", "20")),
        "prompt_variant": os.environ.get("PROMPT_VARIANT", "auto"),
    })
    return c


# ---------------------------------------------------------------- GCS

class Gcs:
    """薄封装：上传 bytes / 文件、按前缀列。google-cloud-storage 自带重试，不另写。"""

    def __init__(self, bucket_name, project=v9.ADC_PROJECT, workers=24):
        from google.cloud import storage
        self.client = storage.Client(project=project)
        # requests/urllib3 默认连接池只有 10。上传线程比它多时，多出来的线程会不停
        # 抢连接并重建 TLS，吞吐会塌掉（实测 48 线程时从 10 帧/s 掉到 0.2 帧/s）。
        try:
            import requests
            ad = requests.adapters.HTTPAdapter(pool_connections=workers,
                                               pool_maxsize=workers * 2,
                                               max_retries=3)
            self.client._http.mount("https://", ad)
        except Exception as e:                                # noqa: BLE001
            print(f"[gcs] 调整连接池失败（继续）: {str(e)[:100]}", flush=True)
        self.bucket = self.client.bucket(bucket_name)
        self.name = bucket_name
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="gcs")

    def uri(self, obj):
        return f"gs://{self.name}/{obj}"

    def put(self, obj, data, content_type="image/jpeg"):
        self.bucket.blob(obj).upload_from_string(data, content_type=content_type)
        return self.uri(obj)

    def put_file(self, obj, path, content_type="application/json"):
        blob = self.bucket.blob(obj)
        blob.chunk_size = 8 << 20
        blob.upload_from_filename(str(path), content_type=content_type)
        return self.uri(obj)

    def read_text(self, obj):
        return self.bucket.blob(obj).download_as_text()

    def list_names(self, prefix):
        return [b.name for b in self.client.list_blobs(self.name, prefix=prefix)]

    def shutdown(self):
        self.pool.shutdown(wait=True)


GCS: Gcs | None = None


# ---------------------------------------------------------------- 线格式

def to_wire(obj):
    """SDK 的 to_json_dict() 出 snake_case；protobuf JSON 两种都收，但 camelCase
    是 bench2 已实测跑通的形状，统一转过去，少一个变量。"""
    if isinstance(obj, dict):
        out = {}
        for k, val in obj.items():
            head, *rest = k.split("_")
            out[head + "".join(w[:1].upper() + w[1:] for w in rest)] = to_wire(val)
        return out
    if isinstance(obj, list):
        return [to_wire(x) for x in obj]
    return obj


def part_uri(uri, res_level):
    p = types.Part.from_uri(file_uri=uri, mime_type="image/jpeg",
                            media_resolution=res_level)
    return to_wire(p.to_json_dict())


def part_text(text):
    return {"text": text}


GEN_CFG = {"temperature": 0.3, "maxOutputTokens": 16384}


# ---------------------------------------------------------------- 纯函数：裁图 / 合并
# 下面两段逐字取自冻结快照 run_caption_2pass_v9_snapshot_20260903_0921.py 的
# process_scene()（也就是 caption_v9.process_clip 内联的同一段），差别只有：
#   · crop 的原图不是内存里的 jpg 字节，而是按 tar 偏移回读后现渲染的 PIL Image
#     （load(idx) 回调），因为 Batch 模式下帧早已传走、不在本机内存里；
#   · 几何、padding、去重、上限完全不动。
# 已在本地用 16 个已落盘场景验证：与已有结果 15/15 逐字一致（1 个无 pass2）。

def crops_from_regions(regions, n_frames, hkt_of_idx, load, max_crops=None):
    crops = []
    seen = set()
    for reg in regions:
        idx = reg.get("frame_index")
        box = reg.get("box_2d")
        if not isinstance(idx, int) or idx < 0 or idx >= n_frames or not box or len(box) != 4:
            continue
        src = load(idx)
        W, H = src.size
        y0, x0, y1, x1 = box
        l, t = int(x0 / 1000 * W), int(y0 / 1000 * H)
        rr, b = int(x1 / 1000 * W), int(y1 / 1000 * H)
        pw, ph = int(W * .03), int(H * .03)
        l, t = max(0, l - pw), max(0, t - ph)
        rr, b = min(W, rr + pw), min(H, b + ph)
        if rr - l < 80 or b - t < 80:
            continue
        key = (idx, l // 60, t // 60, rr // 60, b // 60)
        if key in seen:
            continue
        seen.add(key)
        crops.append({"hkt": reg.get("hkt") or hkt_of_idx(idx),
                      "label": reg.get("label", "region"),
                      "frame_index": idx, "img": src.crop((l, t, rr, b))})
    return crops[:(max_crops or v9.MAX_CROPS)]


def merge_pass2(j1, j2):
    """pass2 的 revised_segments 覆盖 pass1，crop 读数挂 screen_text_detail。
    返回 (merged, n_revised)。"""
    by_hkt = {}
    for c in j2.get("crops", []):
        by_hkt.setdefault(c.get("hkt", ""), []).append(c)

    revised = j2.get("revised_segments") or []
    merged = json.loads(json.dumps(j1))
    merged.pop("text_regions", None)

    if revised:
        by_tr = {}
        for s in revised:
            key = re.sub(r"\s", "", s.get("time_range", ""))
            if key:
                by_tr[key] = s
        base = merged.get("segments", [])
        new_segs = []
        for i, seg in enumerate(base):
            key = re.sub(r"\s", "", seg.get("time_range", ""))
            r = by_tr.get(key) or (revised[i] if i < len(revised) else None)
            if r:
                seg = {**seg, **{k: v for k, v in r.items() if v}}
            new_segs.append(seg)
        merged["segments"] = new_segs

    for seg in merged.get("segments", []):
        m = re.findall(r"(\d{2}:\d{2}:\d{2})", seg.get("time_range", ""))
        if len(m) != 2:
            continue
        lo, hi = m
        detail = [{"hkt": h, "label": c.get("label"), "summary": c.get("summary"),
                   "text": c.get("text", [])}
                  for h, cs in by_hkt.items() if lo <= h <= hi for c in cs]
        if detail:
            seg["screen_text_detail"] = detail
    return merged, len(revised)


# ---------------------------------------------------------------- Batch 提交/轮询

def batch_submit(client, model, jsonl_path, gcs: Gcs, prefix, tag):
    n = sum(1 for _ in open(jsonl_path))
    mb = jsonl_path.stat().st_size / 1e6
    t0 = time.time()
    src = gcs.put_file(f"{prefix}/in/{jsonl_path.name}", jsonl_path,
                       content_type="application/jsonl")
    up = time.time() - t0
    log(f"  [{tag}] JSONL {mb:.1f} MB / {n} 请求，上传 {up:.0f}s "
        f"({mb/max(up,0.1):.1f} MB/s)")
    dest = gcs.uri(f"{prefix}/out_{tag}/")
    job = client.batches.create(model=model, src=src,
                                config=types.CreateBatchJobConfig(dest=dest))
    log(f"  [{tag}] batch 已提交 {job.name}  state={job.state}")
    return {"job_name": job.name, "dest": dest, "src": src, "n": n,
            "submitted_at": time.time()}


def batch_wait(client, sub, tag, timeout_min, poll_sec):
    """轮询到结束。每轮 beat()，看门狗不会把等 batch 误判成挂死。"""
    t0 = time.time()
    last = None
    job = None
    s = "UNKNOWN"
    while True:
        try:
            job = client.batches.get(name=sub["job_name"])
            s = str(job.state)
        except Exception as e:                                # noqa: BLE001
            log(f"    [{tag}] 查状态失败（继续轮询）: {str(e)[:120]}")
        if s != last:
            log(f"    [{tag}] [{(time.time()-t0)/60:5.1f}min] {s}")
            last = s
        beat(f"batch {tag} {s}")
        if any(k in s for k in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
            break
        if (time.time() - t0) / 60 > timeout_min:
            log(f"    [{tag}] !! 超过 {timeout_min:.0f} 分钟仍未结束，主动 cancel 止损")
            try:
                client.batches.cancel(name=sub["job_name"])
            except Exception as e:                            # noqa: BLE001
                log(f"    [{tag}] cancel 失败: {str(e)[:120]}")
            s = "TIMEOUT_CANCELLED"
            break
        time.sleep(poll_sec)
    wall = (time.time() - t0) / 60
    ok = "SUCCEEDED" in s
    log(f"  [{tag}] batch {'✓' if ok else '✗'} {s}  周转 {wall:.1f} 分钟"
        f"{'  err=' + str(getattr(job, 'error', ''))[:200] if not ok else ''}")
    return {"state": s, "ok": ok, "turnaround_min": round(wall, 1)}


def _batch_is_terminal(client, sub):
    """断点里的 Batch 是不是**失败性**终止 —— 只有这种才需要清断点重发。

    **SUCCEEDED 绝对不能算在内**：batch 成功了但 HF Job 在收取结果之前就死了
    （2026-09-04 HF 预付余额耗尽，几十个 job 被平台统一杀掉）是很常见的情况，
    这时结果好端端躺在 GCS 的 dest 里、**钱已经付过了**。把它当终止态清掉断点
    就会原样重发一遍，等于为同一批请求付两次钱。保留断点，让 batch_wait 立刻
    读到 SUCCEEDED 返回，再由 collect() 从 dest 把行读回来，一分钱不用多花。

    HF Job 被取消/重启后 /tmp 状态可能仍保留上一轮的 job_name。旧逻辑会无条件
    接着轮询，立刻读到 CANCELLED 就把整条录制误判失败，所以这里才需要判定。
    查询失败时保守地保留断点（让后续 batch_wait 自己处理）。
    """
    try:
        s = str(client.batches.get(name=sub["job_name"]).state)
    except Exception:                                      # noqa: BLE001
        return False
    return any(k in s for k in ("FAILED", "CANCELLED", "EXPIRED"))


def batch_rows(gcs: Gcs, dest):
    prefix = dest.split(f"gs://{gcs.name}/", 1)[1]
    names = [n for n in gcs.list_names(prefix) if n.endswith((".jsonl", ".json"))]
    for n in sorted(names):
        for line in gcs.read_text(n).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def row_clip_id(row):
    """从回显的 request 里认出这行是哪个 clip：图片 URI 路径带 __povclip0042/。"""
    try:
        blob = json.dumps(row.get("request", row), ensure_ascii=False)
    except Exception:                                         # noqa: BLE001
        return None
    i = blob.find(CLIP_TAG)
    if i < 0:
        return None
    digits = ""
    for ch in blob[i + len(CLIP_TAG):]:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else None


def row_payload(row):
    resp = row.get("response") or {}
    txt = ""
    for cand in (resp.get("candidates") or []):
        for p in ((cand.get("content") or {}).get("parts") or []):
            if p.get("text"):
                txt += p["text"]
    u = resp.get("usageMetadata") or resp.get("usage_metadata") or {}
    err = row.get("status") or resp.get("error") or row.get("error")
    return {
        "content": txt,
        "in": u.get("promptTokenCount") or u.get("prompt_token_count") or 0,
        "out": u.get("candidatesTokenCount") or u.get("candidates_token_count") or 0,
        "think": u.get("thoughtsTokenCount") or u.get("thoughts_token_count") or 0,
        "ok": bool(txt),
        "error": (str(err)[:300] if err and not txt else None),
    }


def collect(gcs, dest, expect_order, tag):
    rows = list(batch_rows(gcs, dest))
    by_cid, unknown = {}, []
    for r in rows:
        cid = row_clip_id(r)
        if cid is None:
            unknown.append(r)
        else:
            by_cid[cid] = row_payload(r)
    if unknown:
        left = [c for c in expect_order if c not in by_cid]
        log(f"  [{tag}] {len(unknown)} 行认不出 clip id，按行序兜底填 {len(left)} 个空位")
        for cid, r in zip(left, unknown):
            by_cid[cid] = row_payload(r)
    ok = sum(1 for v in by_cid.values() if v["ok"])
    log(f"  [{tag}] 回收 {len(by_cid)}/{len(expect_order)} 行，有正文 {ok}")
    return by_cid


def online_run(client, model, jsonl_path, tag, workers):
    """把 batch 的 JSONL 原样用在线接口打完，返回和 collect() 一模一样的结构。

    请求体逐字复用 write_pass*_jsonl 生成的那份，所以在线和 batch 两条路发出去的
    payload 完全相同（同样的 gs:// fileData、同样的 mediaResolution、同样的
    generationConfig），只是计费走在线价、结果不落 GCS。
    """
    rows = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]
    out, lock = {}, threading.Lock()
    t0 = time.time()
    done = [0]

    def one(row):
        cid = row_clip_id(row)
        req = row.get("request") or {}
        for attempt in range(6):
            try:
                resp = client.models.generate_content(
                    model=model, contents=req.get("contents"),
                    config=req.get("generationConfig") or {})
                d = resp.to_json_dict() if hasattr(resp, "to_json_dict") else {}
                return cid, row_payload({"response": d})
            except Exception as e:                              # noqa: BLE001
                msg = str(e)
                if attempt == 5:
                    return cid, {"content": "", "in": 0, "out": 0, "think": 0,
                                 "ok": False, "error": msg[:300]}
                # 429 慢退避（20/40/60s），掉线快重试（4/8/12s）—— 和 v9 本地版一致
                slow = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                time.sleep(min(60, 20 * (attempt + 1)) if slow else 4 * (attempt + 1))
        return cid, {"content": "", "in": 0, "out": 0, "think": 0, "ok": False,
                     "error": "retries exhausted"}

    log(f"  [{tag}] 在线直发 {len(rows)} 个请求，{workers} 并发")
    with ThreadPoolExecutor(workers, thread_name_prefix=f"on-{tag}") as ex:
        for fu in as_completed([ex.submit(one, r) for r in rows]):
            cid, payload = fu.result()
            with lock:
                done[0] += 1
                if cid is not None:
                    out[cid] = payload
            if done[0] % 20 == 0 or done[0] == len(rows):
                el = time.time() - t0
                beat(f"{tag} 在线 {done[0]}/{len(rows)}")
                log(f"  [{tag}] {done[0]}/{len(rows)}  {el/60:.1f}min  "
                    f"{done[0]/max(el,1)*3600:.0f} clip/h")
    ok = sum(1 for v in out.values() if v["ok"])
    log(f"  [{tag}] 在线完成 {len(out)}/{len(rows)} 行，有正文 {ok}，"
        f"用时 {(time.time()-t0)/60:.1f} 分钟")
    return out


def harvest(gcs, attempts, tag):
    """把历史失败尝试（CANCELLED/FAILED/EXPIRED）的 dest 里**已完成且已计费**的行捞回来。

    Batch 被 cancel 时，已经跑完的请求照样写进 predictions.jsonl，也照样计费
    （官方口径：只对已完成的请求收费）。2026-09-04 这批因为 BATCH_TIMEOUT_MIN=240
    在 6.7 小时周转下被误触发，两个项目一共 75 个 batch 被 cancel。不捞就等于
    为同一批请求付两次钱 —— 捞回来之后这些 clip 根本不用重发。
    """
    got = {}
    for a in attempts or []:
        dest, order = a.get("dest"), a.get("order") or []
        if not dest:
            continue
        try:
            rows = collect(gcs, dest, order, f"{tag}·历史")
        except Exception as e:                                # noqa: BLE001
            log(f"  [{tag}] 历史结果回收失败（跳过）: {str(e)[:120]}")
            continue
        for cid, v in rows.items():
            if v.get("ok") and cid not in got:
                got[cid] = v
    if got:
        log(f"  [{tag}] 从 {len(attempts)} 次历史尝试里捞回 {len(got)} 个已计费的 clip，不再重发")
    return got


# ---------------------------------------------------------------- prompt

def pick_prompt(c, has_gaze):
    """无 MPS 眼动的录制换 v9_nogaze —— 原版 prompt 有 4 处规则围绕绿色注视圈写，
    画面上没有圈却告诉模型有，会诱发「我正看着…」这类无依据描述。"""
    variant = c["prompt_variant"]
    if variant == "auto":
        variant = "v9" if has_gaze else "v9_nogaze"
    return variant, (HERE / "prompts" / f"{variant}.txt").read_text().strip()


# ---------------------------------------------------------------- A 解包

def index_tar(src):
    """扫一遍 header 建索引，按段归类。不假设任何布局顺序。

    数据集里至少有两种 tar 布局（见 remote_tar.walk_headers 的注释），而且有的
    录制**根本没有 audio/ 段**。所以这里只按名字认段，谁在前谁在后都无所谓。
    """
    t0 = time.time()
    frames, ocr = [], []
    gaze = wav = tr = None
    n = 0
    for name, size, off in rt.walk_headers(src):
        n += 1
        low = name.lower()
        if "/picture/masked/" in name and low.endswith((".jpg", ".jpeg")):
            frames.append((os.path.basename(name), size, off))
        elif "/picture/ocr_text/" in name and low.endswith(".txt"):
            ocr.append((os.path.basename(name), size, off))
        elif name.endswith("eye_tracking/gaze.csv"):
            gaze = {"data_offset": off, "size": size}
        elif name.endswith("audio/anonymized.wav"):
            wav = {"data_offset": off, "size": size}
        elif name.endswith("audio/transcript.txt"):
            tr = {"data_offset": off, "size": size}
        if n % 20000 == 0:
            beat(f"索引 {n} 个 member")
    log(f"索引: {n} 个 member / masked {len(frames)} / ocr {len(ocr)} / "
        f"gaze {'有' if gaze else '无'} / wav {'有' if wav else '无'} / "
        f"transcript {'有' if tr else '无'}  ({time.time()-t0:.0f}s)")
    return {"frames": frames, "ocr": ocr, "gaze": gaze, "wav": wav, "transcript": tr}


def read_gaze(src, entry):
    if not entry:
        return {}
    import csv as _csv
    blob = src.read(entry["data_offset"], entry["size"]).decode("utf-8", "replace")
    return {r["frame_file"]: r for r in _csv.DictReader(io.StringIO(blob))}


def read_ocr(src, entries, workers=16):
    """每帧一份小文本，并行按偏移取。返回 {frame_index: (stem, 正文)}。"""
    out = {}
    if not entries:
        return out

    def one(e):
        name, size, off = e
        txt = src.read(off, size).decode("utf-8", "replace")
        return name, cc.clean_ocr(txt)

    with ThreadPoolExecutor(workers, thread_name_prefix="ocr") as ex:
        for name, txt in ex.map(one, entries):
            i = cc.frame_index(name)
            if i is not None:
                out[i] = (name[:-4], txt)
    return out


def unpack(c, src, frame_entries, trails, has_gaze, gcs: Gcs, prefix, do_upload=True):
    """masked 帧 → 2880px 画注视圈 q80 → 传 GCS。

    分类靠 index_tar 的索引（布局无关），搬字节靠 iter_members 的顺序预取（快）。
    两者缺一不可：只用索引按偏移随机读会掉到 ~6 MB/s，只用顺序遍历又会被
    tar 布局差异坑到。

    帧不落盘：cpu-upgrade 只有 50 GB，最大那条录制 26,280 帧 × ~1.2MB ≈ 31 GB。

    do_upload=False 用于断点续跑：batch 已提交过、帧早在 GCS 上了，
    这时一个字节都不读，URI 按同一模板推出来，几秒重建现场。
    """
    n_per_clip = c["clip_seconds"]
    clip_frames: dict[int, list] = {}
    meta: dict[str, dict] = {}
    for fname, size, off in frame_entries:
        fi = cc.frame_index(fname)
        if fi is None:
            continue
        clip_frames.setdefault(fi // n_per_clip, []).append(fname)
        meta[fname] = {"off": off, "size": size, "uri": None}

    if c["max_clips"]:
        # masked 是时间倒序，tar 开头那几个 clip = 录制末尾；编号与全量一致
        keep = sorted(clip_frames)[:c["max_clips"]]
        for cid in [x for x in clip_frames if x not in keep]:
            for fn in clip_frames.pop(cid):
                meta.pop(fn, None)

    def frame_uri(cid, fname):
        return gcs.uri(f"{prefix}/f/{CLIP_TAG}{cid:04d}/{fname}")

    if not do_upload:
        for cid, names in clip_frames.items():
            for fn in names:
                meta[fn]["uri"] = frame_uri(cid, fn)
        log(f"索引复用（不重传）: {sum(len(v) for v in clip_frames.values())} 帧 / "
            f"{len(clip_frames)} clip")
        return clip_frames, meta

    t0 = time.time()
    n_up = up_bytes = done_n = 0
    last_beat = time.time()
    total = sum(len(v) for v in clip_frames.values())
    cid_of = {fn: cid for cid, names in clip_frames.items() for fn in names}
    log(f"待解包上传 {total} 帧，{c['upload_workers']} 并发")

    def one(cid, fname, data):
        pts, valid = trails.get(fname, ([], False)) if has_gaze else (None, False)
        jpg = cc.render_frame(data, pts, valid, size=v9.FRAME_DIM, quality=v9.FRAME_Q)
        gcs.put(f"{prefix}/f/{CLIP_TAG}{cid:04d}/{fname}", jpg)
        return fname, len(jpg), frame_uri(cid, fname)

    # 帧字节必须**顺序**读：SeqStream 有 8MB 分块预取，实测 209 MB/s；
    # 改成按偏移随机读只有 ~6 MB/s（挂载是网络盘，随机小读没有预取可言），
    # 慢 35 倍，直接把 4 条大录制拖到被看门狗判为挂死。
    # 布局无关性由上面的 index_tar 保证（谁在前谁在后都已经分好类），
    # 这里只借 iter_members 的顺序预取搬字节，非帧 member 会被跳过、不下载。
    # 从 0 开始走，别拿 index 里的 data_offset 当起点：那是**数据**偏移，
    # member header 在它前面 512 字节（长文件名还多一个 LongLink 头），
    # 从数据中间起步会吞掉第一帧、连累整个 clip 被判为不完整（本地实测 95→94）。
    # 从头走的代价可忽略：非帧 member 的数据不下载，只掠过 header。

    def is_frame(name):
        return "/picture/masked/" in name and name.lower().endswith((".jpg", ".jpeg"))

    pool = ThreadPoolExecutor(c["upload_workers"], thread_name_prefix="frame")
    pending = []

    def drain(keep):
        nonlocal n_up, up_bytes, done_n, last_beat
        while len(pending) > keep:
            fu = pending.pop(0)
            done_n += 1
            try:
                fname, ln, uri = fu.result()
            except Exception as e:                            # noqa: BLE001
                log(f"  帧失败: {str(e)[:120]}")
                continue
            meta[fname]["uri"] = uri
            n_up += 1
            up_bytes += ln
            # 看门狗要杀的是「一帧都不动」的真挂死，不是「慢但在推进」
            if done_n % 50 == 0 or time.time() - last_beat > 30:
                beat(f"解包上传 {n_up}/{total} 帧")
                last_beat = time.time()
            if done_n % 2000 == 0:
                el = time.time() - t0
                log(f"  进度 {n_up}/{total} 帧 {up_bytes/1e9:.1f} GB "
                    f"{el/60:.0f}min ({n_up/max(el,1):.1f} 帧/s)")

    try:
        for name, size, data, off in rt.iter_members(
                src, start=0, want=is_frame, with_offset=True):
            if data is None:
                continue
            fname = os.path.basename(name)
            cid = cid_of.get(fname)
            if cid is None:                     # 不在本轮范围（--max-clips 裁掉的）
                continue
            pending.append(pool.submit(one, cid, fname, data))
            if len(pending) >= c["upload_workers"] * 3:
                drain(c["upload_workers"])
        drain(0)
    finally:
        pool.shutdown(wait=True)

    dropped = []
    for cid in list(clip_frames):
        if any(not meta.get(f, {}).get("uri") for f in clip_frames[cid]):
            dropped.append(cid)
            for f in clip_frames.pop(cid):
                meta.pop(f, None)
    if dropped:
        log(f"  !! {len(dropped)} 个 clip 有帧没传上去，本轮跳过: {dropped[:10]}")

    dt = max(time.time() - t0, 0.1)
    log(f"解包+上传: {n_up}/{total} 帧 / {len(clip_frames)} clip，"
        f"传 {up_bytes/1e9:.2f} GB，{dt:.0f}s ({up_bytes/dt/1e6:.0f} MB/s)")
    return clip_frames, meta


# ---------------------------------------------------------------- clip 上下文

def build_sd(cid, names, meta, ocr_texts, intervals):
    """凑出 caption_v9.build_context_text_v9 要的 sd（cloud_worker.build_sd 的形状），
    只是 frames 里带的是 gs:// URI 而不是 jpg 字节。"""
    frames = [{"name": fn, "hkt": cc.hkt_hms(fn), "uri": meta[fn]["uri"]} for fn in names]
    ocr = []
    for fn in names:
        got = ocr_texts.get(cc.frame_index(fn))
        if got and got[1]:
            ocr.append(f"[{cc.hkt_hms(fn)} {got[0]}]: {got[1]}")
    rel = cc.frame_index(names[0])
    dur = len(names)
    t0 = cc.frame_time(names[0])
    return {
        "frames": frames,
        "ocr_texts": ocr,
        "transcript": "\n".join(cc.transcript_for_window(intervals, rel, dur)),
        "n_frames": dur,
        "clip_start": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "clip_end": (t0 + timedelta(seconds=dur)).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------- B pass0 语音

def run_speech(c, client, todo, clip_frames, meta, ocr_texts, intervals, slicer):
    """silero-vad 定时间 + Gemini 定文字。在线跑：payload 只有几十 KB、不吃 DSQ，
    而且能原样复用 caption_v9.align_speech（batch 化要多排一轮，不值）。"""
    if c["no_speech"] or slicer is None:
        return {}
    out = {}

    def one(cid):
        names = sorted(clip_frames[cid], key=cc.frame_index)
        pcm = slicer.slice16k(cc.frame_index(names[0]), len(names))
        if not pcm:
            return cid, None
        sd = build_sd(cid, names, meta, ocr_texts, intervals)
        try:
            return cid, v9.align_speech(client, cw.wav_bytes(pcm),
                                        sd["clip_start"], sd["transcript"])
        except Exception as e:                                # noqa: BLE001
            return cid, {"error": str(e)[:200], "utterances": []}

    t0 = time.time()
    n_utt = 0
    deadline = t0 + c["speech_deadline_min"] * 60
    expired = False
    # 每完成一个就打点。原来 `i % 50 == 0` 才打：2026-09-04 五十多个 job 并跑时
    # 在线 pass0 被拖慢，338 clip 的 4-17 连第一次打点都没熬到，看门狗按
    # 「25 分钟无进展」把整条录制杀了（另外 5 条同样死法）。beat() 只是更新一个
    # 时间戳，每轮都调没有代价，而「慢但在推进」本来就不该被判成挂死。
    with ThreadPoolExecutor(c["speech_workers"], thread_name_prefix="vad") as ex:
        futs = {ex.submit(one, cid): cid for cid in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            cid = futs[fu]
            beat(f"pass0 语音 {i}/{len(todo)}")
            # 超时只放弃**还没开跑**的，已在飞的让它自然结束（强行 shutdown 会让
            # 解释器退出时 join 卡住）。放弃的 clip 没有 speech 块，
            # build_context 本来就有「只用 transcript」的分支，不影响出片。
            if not expired and time.time() > deadline:
                expired = True
                n_cancel = sum(1 for f in futs if f.cancel())
                log(f"  !! pass0 超过 {c['speech_deadline_min']:.0f} 分钟，"
                    f"放弃剩余 {n_cancel} 个 clip 的语音对齐，退回 transcript")
            if fu.cancelled():
                continue
            try:
                cid, sp = fu.result()
            except Exception as e:                            # noqa: BLE001
                log(f"  语音 clip_{cid:04d} 崩了: {str(e)[:120]}")
                continue
            if sp:
                out[cid] = sp
                n_utt += len(sp.get("utterances", []))
    log(f"pass0 语音: {len(out)}/{len(todo)} clip 有窗口，{n_utt} 条话语 "
        f"({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------- C pass1

def write_pass1_jsonl(path, todo, clip_frames, meta, ocr_texts, intervals, speech, prompt):
    """每 clip 一行。文本/图片的交错顺序与冻结脚本 process_scene 的 pass1 一致，
    差别只在图片是 gs:// URI 而不是 inline bytes —— token 计数相同。"""
    with open(path, "w") as f:
        for cid in todo:
            names = sorted(clip_frames[cid], key=cc.frame_index)
            sd = build_sd(cid, names, meta, ocr_texts, intervals)
            use_tr = bool(sd["transcript"].strip()) or bool(speech.get(cid))
            sys_text = (prompt + v9.REGION_ADDENDUM + "\n\n"
                        + v9.build_context_text_v9(sd, use_tr, True, speech.get(cid)))
            frames = sd["frames"]
            texts = [f"[frame_index {i} — {fr['hkt']} HKT — {fr['name']}]"
                     for i, fr in enumerate(frames)]
            texts[0] = sys_text + "\n\n" + texts[0]
            parts = []
            for i, fr in enumerate(frames):
                parts.append(part_text(texts[i]))
                parts.append(part_uri(fr["uri"], v9.HIGH))
            f.write(json.dumps(
                {"request": {"contents": [{"role": "user", "parts": parts}],
                             "generationConfig": GEN_CFG}}, ensure_ascii=False) + "\n")
    return len(todo)


# ---------------------------------------------------------------- D crop

def build_crops(c, src, trails, has_gaze, pass1, clip_frames, meta, gcs: Gcs, prefix):
    """按阶段 A 记的偏移从挂载的 tar 随机回读原帧（实测 209 MB/s），裁出与本地版
    逐字相同的几何，再传 GCS 供 pass2 用。"""
    out = {}
    t0 = time.time()
    n_crop = 0

    def one(cid):
        j1 = pass1[cid].get("json")
        regions = (j1 or {}).get("text_regions") or []
        if not regions:
            return cid, []
        names = sorted(clip_frames[cid], key=cc.frame_index)
        cache = {}

        def load(idx):
            fn = names[idx]
            if fn not in cache:
                m = meta[fn]
                data = src.read(m["off"], m["size"])
                pts, valid = trails.get(fn, ([], False)) if has_gaze else (None, False)
                cache[fn] = cc.render_frame(data, pts, valid, size=v9.FRAME_DIM,
                                            quality=v9.FRAME_Q, as_image=True)
            return cache[fn]

        crops = crops_from_regions(regions, len(names),
                                   lambda i: cc.hkt_hms(names[i]), load)
        got = []
        for i, cr in enumerate(crops):
            w, h = cr["img"].size
            blob = v9.encode_img(cr["img"], v9.CROP_DIM, v9.CROP_Q)
            uri = gcs.put(f"{prefix}/c/{CLIP_TAG}{cid:04d}/{i:02d}.jpg", blob)
            got.append({"crop_index": i, "hkt": cr["hkt"], "label": cr["label"],
                        "frame_index": cr["frame_index"], "native": f"{w}x{h}",
                        "uri": uri})
        cache.clear()
        return cid, got

    todo_crop = [cid for cid in sorted(pass1) if cid in clip_frames]
    with ThreadPoolExecutor(max(4, c["render_workers"]), thread_name_prefix="crop") as ex:
        futs = {ex.submit(one, cid): cid for cid in todo_crop}
        for i, fu in enumerate(as_completed(futs), 1):
            cid = futs[fu]
            try:
                cid, got = fu.result()
            except Exception as e:                            # noqa: BLE001
                log(f"  crop clip_{cid:04d} 失败: {str(e)[:160]}")
                got = []
            if got:
                out[cid] = got
                n_crop += len(got)
            if i % 50 == 0:
                beat(f"裁图 {i}/{len(futs)}")
    log(f"crop: {len(out)} clip 有区域，共 {n_crop} 张 ({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------- E pass2

def write_pass2_jsonl(path, crops_by_clip, pass1):
    order = []
    with open(path, "w") as f:
        for cid in sorted(crops_by_clip):
            j1 = pass1[cid].get("json") or {}
            crops = crops_by_clip[cid]
            # 把 pass1 的 caption 交给 pass2，让它改行为判断，而不只是补文字
            p1_caption = {"scene_summary": j1.get("scene_summary", ""),
                          "segments": [{k: s.get(k) for k in
                                        ("time_range", "action", "objects", "environment",
                                         "text_visible", "speech", "details")}
                                       for s in j1.get("segments", [])]}
            p1_block = ("## FIRST-PASS CAPTION (to be corrected and enriched)\n"
                        + json.dumps(p1_caption, ensure_ascii=False, indent=1))
            texts = [f"[crop_index {m['crop_index']} — {m['hkt']} HKT — {m['label']} "
                     f"— native {m['native']}]" for m in crops]
            texts[0] = v9.READ_PROMPT + "\n\n" + p1_block + "\n\n" + texts[0]
            parts = []
            for i, m in enumerate(crops):
                parts.append(part_text(texts[i]))
                parts.append(part_uri(m["uri"], v9.ULTRA))
            f.write(json.dumps(
                {"request": {"contents": [{"role": "user", "parts": parts}],
                             "generationConfig": GEN_CFG}}, ensure_ascii=False) + "\n")
            order.append(cid)
    return order


# ---------------------------------------------------------------- 断点状态

def load_state(c):
    """batch 已提交就绝不重复提交 —— 重复提交 = 重复计费。"""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(c["out_repo"], f"_batch/{c['run_id']}/{c['rec']}.json",
                            repo_type="dataset", token=c["token"])
        st = json.loads(Path(p).read_text())
        log(f"接上已有状态: stage={st.get('stage')} "
            f"pass1={'有' if st.get('pass1') else '无'} pass2={'有' if st.get('pass2') else '无'}")
        return st
    except Exception:                                         # noqa: BLE001
        return {}


def save_state(api, c, st):
    p = WORK / f"state_{c['rec']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=2))
    try:
        cw.commit(api, c["out_repo"], [(p, f"_batch/{c['run_id']}/{c['rec']}.json")],
                  f"batch state {c['rec']} {st.get('stage')}")
    except Exception as e:                                    # noqa: BLE001
        log(f"  状态存 HF 失败（不致命）: {str(e)[:120]}")


# ---------------------------------------------------------------- F 记录与收尾

def build_record(c, cid, clip_frames, meta, ocr_texts, intervals, speech,
                 pass1, pass2, crops_by_clip, has_gaze, variant, caps_dir):
    names = sorted(clip_frames[cid], key=cc.frame_index)
    sd = build_sd(cid, names, meta, ocr_texts, intervals)
    p1 = pass1.get(cid, {})
    p2 = pass2.get(cid, {})
    sp = speech.get(cid) or {}
    j1, j2 = p1.get("json"), p2.get("json")

    merged, n_revised = (None, 0)
    if j1 and j2:
        merged, n_revised = merge_pass2(j1, j2)
    parsed = merged or j1

    bin_ = (p1.get("in") or 0) + (p2.get("in") or 0)
    bout = ((p1.get("out") or 0) + (p1.get("think") or 0)
            + (p2.get("out") or 0) + (p2.get("think") or 0))
    oin, oout = sp.get("in") or 0, sp.get("out") or 0
    lane = "online" if c.get("online") else "batch"      # pass1/pass2 实际走的通路
    cost = (bin_ / 1e6 * PRICE[lane][0] + bout / 1e6 * PRICE[lane][1]
            + oin / 1e6 * PRICE["online"][0] + oout / 1e6 * PRICE["online"][1])
    n_text = sum(len(cr.get("text", [])) for cr in ((j2 or {}).get("crops", []) or []))

    rec_out = {
        "clip_id": cid,
        "recording": c["rec"],
        "clip_index": f"clip_{cid:04d}",
        "mode": "v9_2pass_online" if c.get("online") else "v9_2pass_batch",
        "prompt_variant": variant,
        "n_frames": len(names),
        "frame_dim": v9.FRAME_DIM,
        "clip_start_hkt": sd["clip_start"],
        "clip_end_hkt": sd["clip_end"],
        "transcript_lines": len([l for l in sd["transcript"].splitlines() if l.strip()]),
        "ocr_entries": len(sd["ocr_texts"]),
        "has_gaze": has_gaze,
        "ok": bool(parsed),
        "model": c["model"],
        "speech": sp or None,
        "n_crops": len(crops_by_clip.get(cid, [])),
        "crops": crops_by_clip.get(cid, []),
        "pass1": dict({k: p1.get(k) for k in ("ok", "in", "out", "think", "error", "content")},
                      n_segments=len((j1 or {}).get("segments", [])),
                      n_regions=len((j1 or {}).get("text_regions", []))),
        "pass2": dict({k: p2.get(k) for k in ("ok", "in", "out", "think", "error", "content")},
                      n_text_items=n_text, n_revised=n_revised),
        # merge.py 只认 parsed/usage，保持它不用改就能合并
        "parsed": parsed,
        "usage": {"in": bin_ + oin, "out": bout + oout, "think": 0,
                  "batch_in": bin_, "batch_out": bout,
                  "online_in": oin, "online_out": oout},
        "est_cost_usd": round(cost, 6),
        "content_raw": p1.get("content", ""),
    }
    (caps_dir / f"clip_{cid:04d}.json").write_text(
        json.dumps(rec_out, ensure_ascii=False, indent=2))
    return rec_out


def finalize(c, api, caps_dir, rec, clip_frames, has_gaze, variant,
             intervals, ocr_texts, gaze_rows, t_all, st, turnaround=None):
    first_index = min(cc.frame_index(f) for v in clip_frames.values() for f in v)
    base_time = min(cc.frame_time(f) for v in clip_frames.values() for f in v)
    base_time -= timedelta(seconds=first_index)

    manifest = {
        "recording": rec, "tar": c["tar"], "src_repo": c["src_repo"],
        "run_id": c["run_id"], "mode": "v9_2pass_batch", "prompt_variant": variant,
        "clip_seconds": c["clip_seconds"], "has_gaze": has_gaze,
        "gaze_rows": len(gaze_rows), "ocr_frames": len(ocr_texts),
        "transcript_intervals": len(intervals),
        "recording_start_hkt": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames_processed": sum(len(v) for v in clip_frames.values()),
        "clips": sorted(clip_frames),
        "frame_dim": v9.FRAME_DIM, "crop_dim": v9.CROP_DIM,
        "model": c["model"], "max_clips": c["max_clips"],
        "batch_turnaround": turnaround or {},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    (caps_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    have = {int(p.stem.split("_")[1]) for p in caps_dir.glob("clip_*.json")}
    missing = sorted(set(clip_frames) - have)
    if missing:
        from huggingface_hub import hf_hub_download

        def fetch(cid):
            try:
                p = hf_hub_download(c["out_repo"], f"captions/{rec}/clip_{cid:04d}.json",
                                    repo_type="dataset", token=c["token"])
                (caps_dir / f"clip_{cid:04d}.json").write_bytes(Path(p).read_bytes())
            except Exception:                                 # noqa: BLE001
                pass
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(fetch, missing))
        log(f"补回 {len(missing)} 个云上已有的 clip 一起合并")

    clips = mg.load_clips(caps_dir)
    merged = mg.merge_recording(clips, manifest)
    # 真实计费按每 clip 记的 batch/online 拆分价算，别用 merge.py 的在线单价
    merged["est_cost_usd"] = round(sum(cl.get("est_cost_usd") or 0 for cl in clips), 4)
    merged["billing"] = "batch(pass1+pass2) + online(pass0)"
    rep = mg.write_recording_outputs(caps_dir, merged)
    rep["est_cost_usd"] = merged["est_cost_usd"]
    rep["billing"] = merged["billing"]
    rep["batch_turnaround"] = turnaround or {}
    (caps_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    log(f"合并: {rep['clip_ok']}/{rep['clip_count']} clip, {rep['segment_count']} segments, "
        f"in={merged['tokens']['in']} out={merged['tokens'].get('out_billable', 0)} "
        f"实际 ${merged['est_cost_usd']}")
    log(f"PII 审计: {merged.get('pii_audit')}")
    cw.commit(api, c["out_repo"],
              [(caps_dir / n, f"captions/{rec}/{n}") for n in
               ("manifest.json", "report.json", "captions_full.json",
                "captions_full.jsonl", "captions_full.txt") if (caps_dir / n).exists()],
              f"v9 batch manifest+merged {rec}")
    save_state(api, c, dict(st, stage="done", cost_usd=merged["est_cost_usd"]))
    beat("完成一条")
    log(f"完成，总耗时 {time.time()-t_all:.0f}s  DONE_OK")
    return manifest


# ---------------------------------------------------------------- 主流程

def process_one(c, api):
    rec = c["rec"]
    t_all = time.time()
    log(f"=== {rec} ===")
    log(f"run_id={c['run_id']} mount={c['mount']} clip={c['clip_seconds']}s "
        f"upload_workers={c['upload_workers']} max_clips={c['max_clips']}")

    caps_dir = WORK / rec / "captions"
    caps_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"batch_in/v9/{c['run_id']}/{rec}"      # 复用已验证的 7 天生命周期规则

    src = rt.open_source(c["tar"], mount_root=c["mount"], token=c["token"])
    log(f"源: {'挂载文件' if src.seekable else 'HTTP Range'}  {src.size/1e9:.2f} GB")
    if src.size < (50 << 20):
        log("!! tar 小于 50 MB，视为空包，跳过")
        return {"recording": rec, "skipped": "empty_tar"}

    t0 = time.time()
    idx = index_tar(src)
    if not idx["frames"]:
        log("!! tar 里没有 picture/masked/ 帧，跳过")
        return {"recording": rec, "skipped": "no_masked_frames"}
    gaze_rows = read_gaze(src, idx["gaze"])
    has_gaze = len(gaze_rows) > 0
    trails = cc.build_trails(gaze_rows) if has_gaze else {}
    ocr_texts = read_ocr(src, idx["ocr"])
    beat("索引与尾部数据就绪")
    log(f"gaze={len(gaze_rows)} 行, ocr={len(ocr_texts)} 帧 ({time.time()-t0:.1f}s)")

    variant, prompt = pick_prompt(c, has_gaze)
    log(f"prompt: {variant} ({len(prompt)} chars) + REGION_ADDENDUM")

    transcript_text = ""
    if idx["transcript"]:
        transcript_text = src.read(idx["transcript"]["data_offset"],
                                   idx["transcript"]["size"]).decode("utf-8", "replace")
    intervals = cc.parse_transcript(transcript_text)
    # 有的录制整条没有 audio/ 段（4-27_hkt0930 实测）。没有音频就没有 pass0 语音，
    # 没有 transcript 就别在 prompt 里塞一个空的「## Audio Transcript」标题。
    slicer = None
    if idx["wav"]:
        try:
            slicer = cw.AudioSlicer(src, idx["wav"])
        except Exception as e:                                # noqa: BLE001
            log(f"  wav 解析失败，pass0 跳过: {str(e)[:120]}")
    else:
        log("  这条录制没有 audio/ 段 → 无 pass0 语音")
    log(f"transcript={len(intervals)} 区间")
    beat("audio/transcript 就绪")

    st = load_state(c)
    st.setdefault("recording", rec)
    st.setdefault("run_id", c["run_id"])
    resume_p1 = st.get("pass1")

    # 先建立 client 检查旧断点。明确终止的 Batch 不能续接：清掉 pass1/pass2
    # 让下面按缺失 clip 重新组 JSONL；已上传到 OUT_REPO 的 clip 仍由 done_caps
    # 去重，不会整条录制重算。
    resume_client = None
    if resume_p1 or st.get("pass2"):
        from google import genai
        resume_client = genai.Client(vertexai=True, project=v9.ADC_PROJECT,
                                     location=v9.ADC_LOCATION)
        if resume_p1 and _batch_is_terminal(resume_client, resume_p1.get("sub", {})):
            log("  旧 pass1 Batch 失败性终止，转入历史待回收，只补真正缺的 clip")
            st.setdefault("harvest_pass1", []).append(
                {"dest": resume_p1["sub"].get("dest"), "order": resume_p1.get("order") or []})
            if st.get("pass2"):
                st.setdefault("harvest_pass2", []).append(
                    {"dest": st["pass2"]["sub"].get("dest"), "order": st["pass2"].get("order") or []})
            st.pop("pass1", None)
            st.pop("pass1_result", None)
            st.pop("pass2", None)
            st.pop("pass2_result", None)
            resume_p1 = None
        elif st.get("pass2") and _batch_is_terminal(resume_client, st["pass2"].get("sub", {})):
            log("  旧 pass2 Batch 失败性终止，转入历史待回收并重建缺失请求")
            st.setdefault("harvest_pass2", []).append(
                {"dest": st["pass2"]["sub"].get("dest"), "order": st["pass2"].get("order") or []})
            st.pop("pass2", None)
            st.pop("pass2_result", None)
        save_state(api, c, st)

    # 帧一旦整批传完就把 clip 清单记进状态。job 被看门狗杀掉重启时（2026-09-04
    # 有 6 条这么死的），帧还好端端在 GCS 上、7 天生命周期也没到，没有理由再花
    # 十几分钟重传十几 GB —— 按同一模板把 URI 推出来就行。
    # 只记「整批成功」的那份 clip 列表：unpack 会把有帧没传上去的 clip 整个丢掉，
    # 存这份列表才能保证续跑时重建出来的 URI 个个都真实存在。
    frames_ok = st.get("frames_uploaded")
    reuse = bool(resume_p1) or frames_ok is not None
    clip_frames, meta = unpack(c, src, idx["frames"], trails, has_gaze, GCS, prefix,
                               do_upload=not reuse)
    if not clip_frames:
        log("!! 没解出任何帧")
        return {"recording": rec, "skipped": "no_frames"}
    if frames_ok is not None:
        keep = set(frames_ok)
        for cid in [x for x in clip_frames if x not in keep]:
            for fn in clip_frames.pop(cid):
                meta.pop(fn, None)
        log(f"  续跑：沿用上轮已传的 {len(clip_frames)} 个 clip，不重传")
    elif not reuse:
        st["frames_uploaded"] = sorted(clip_frames)
        save_state(api, c, dict(st, stage="frames_uploaded"))

    done_caps = cw.existing_clip_ids(api, c["out_repo"], rec, "captions", ".json")
    # 先把历史失败尝试里已计费的 pass1 行捞回来，这些 clip 不用再发一遍
    salvaged1 = harvest(GCS, st.get("harvest_pass1"), "pass1") if st.get("harvest_pass1") else {}
    salvaged2 = harvest(GCS, st.get("harvest_pass2"), "pass2") if st.get("harvest_pass2") else {}
    todo = [x for x in sorted(clip_frames)
            if x not in done_caps and x not in salvaged1]
    if done_caps:
        log(f"云上已有 {len(done_caps)} 个 clip，本轮待跑 {len(todo)}")
    if not todo and not resume_p1 and not salvaged1:
        log("全部已完成")
        return finalize(c, api, caps_dir, rec, clip_frames, has_gaze, variant,
                        intervals, ocr_texts, gaze_rows, t_all, st)

    from google import genai
    client = resume_client or genai.Client(vertexai=True, project=v9.ADC_PROJECT,
                                           location=v9.ADC_LOCATION)

    # ---------- B pass0 ----------
    speech = {int(k): v for k, v in (st.get("speech_cache") or {}).items()}
    if not speech:
        speech = run_speech(c, client, todo, clip_frames, meta, ocr_texts, intervals, slicer)
        st["speech_cache"] = {str(k): v for k, v in speech.items()}
        save_state(api, c, dict(st, stage="speech_done"))

    # ---------- C pass1 ----------
    if not todo and not resume_p1:
        # 全部 clip 都从历史里捞回来了：一条新请求都不用发，直接进 crop/pass2。
        # 不加这个分支会去提交一个空 JSONL 的 batch。
        log("pass1: 本轮无新请求，全部沿用捞回的历史结果")
        sub1, order1, got1 = None, [], {}
        r1 = {"state": "SALVAGED", "ok": True, "turnaround_min": 0}
    elif resume_p1:
        sub1 = resume_p1["sub"]
        # 续跑时上一轮的 order 里可能有本轮被丢弃的 clip（帧没传全），过滤掉，
        # 否则后面 clip_frames[cid] 直接 KeyError
        order1 = [x for x in resume_p1["order"] if x in clip_frames]
        if len(order1) != len(resume_p1["order"]):
            log(f"  续跑：order 里有 {len(resume_p1['order']) - len(order1)} 个 clip "
                f"本轮不可用，已剔除")
        log(f"pass1 接上已提交的 batch {sub1['job_name']}")
    else:
        jl = WORK / f"{rec}_pass1.jsonl"
        write_pass1_jsonl(jl, todo, clip_frames, meta, ocr_texts, intervals, speech, prompt)
        beat("pass1 JSONL 就绪")
        order1 = todo
        if c["online"]:
            got1 = online_run(client, c["model"], jl, "pass1", c["online_workers"])
            sub1 = None
            r1 = {"state": "ONLINE_OK", "ok": True, "turnaround_min": 0}
        else:
            sub1 = batch_submit(client, c["model"], jl, GCS, prefix, "pass1")
            st["pass1"] = {"sub": sub1, "order": order1}
            save_state(api, c, dict(st, stage="pass1_submitted"))
        jl.unlink(missing_ok=True)

    if sub1 is not None:
        r1 = batch_wait(client, sub1, "pass1", c["batch_timeout_min"], c["batch_poll_sec"])
        if not r1["ok"]:
            save_state(api, c, dict(st, stage="pass1_failed", pass1_result=r1))
            raise RuntimeError(f"pass1 batch 未成功: {r1['state']}")
        got1 = collect(GCS, sub1["dest"], order1, "pass1")
    pass1 = {}
    # 捞回来的历史结果和本轮新跑的合并；同一个 clip 以本轮为准
    for cid, g in salvaged1.items():
        if cid in clip_frames:
            g = dict(g)
            g["json"] = v9.parse_json(g["content"]) if g["ok"] else None
            g["salvaged"] = True
            pass1[cid] = g
    for cid in order1:
        g = dict(got1.get(cid) or {"ok": False, "content": "", "in": 0, "out": 0,
                                   "think": 0, "error": "missing in batch output"})
        g["json"] = v9.parse_json(g["content"]) if g["ok"] else None
        pass1[cid] = g
    n_sal = sum(1 for g in pass1.values() if g.get("salvaged"))
    log(f"pass1: {sum(1 for g in pass1.values() if g.get('json'))}/{len(pass1)} 解析出 JSON"
        + (f"（其中 {n_sal} 个是捞回来的历史结果）" if n_sal else ""))

    # ---------- D crop ----------
    crops_by_clip = build_crops(c, src, trails, has_gaze, pass1, clip_frames, meta, GCS, prefix)
    # 历史里已经跑成功过 pass2 的 clip 不用再裁再发一遍
    if salvaged2:
        skip2 = [cid for cid in list(crops_by_clip) if cid in salvaged2]
        for cid in skip2:
            crops_by_clip.pop(cid, None)
        if skip2:
            log(f"  [pass2] {len(skip2)} 个 clip 已有历史结果，跳过重发")

    # ---------- E pass2 ----------
    pass2 = {}
    for cid, g in salvaged2.items():
        if cid in pass1:
            g = dict(g)
            g["json"] = v9.parse_json(g["content"]) if g["ok"] else None
            g["salvaged"] = True
            pass2[cid] = g
    r2 = {"state": "SKIPPED", "ok": False, "turnaround_min": 0}
    resume_p2 = st.get("pass2")
    if crops_by_clip or resume_p2:
        got2_online = None
        if resume_p2:
            sub2, order2 = resume_p2["sub"], resume_p2["order"]
            log(f"pass2 接上已提交的 batch {sub2['job_name']}")
        else:
            jl2 = WORK / f"{rec}_pass2.jsonl"
            order2 = write_pass2_jsonl(jl2, crops_by_clip, pass1)
            beat("pass2 JSONL 就绪")
            if c["online"]:
                got2_online = online_run(client, c["model"], jl2, "pass2",
                                         c["online_workers"])
                sub2 = None
                r2 = {"state": "ONLINE_OK", "ok": True, "turnaround_min": 0}
            else:
                got2_online = None
                sub2 = batch_submit(client, c["model"], jl2, GCS, prefix, "pass2")
                st["pass2"] = {"sub": sub2, "order": order2}
                save_state(api, c, dict(st, stage="pass2_submitted"))
            jl2.unlink(missing_ok=True)
        if sub2 is not None:
            r2 = batch_wait(client, sub2, "pass2", c["batch_timeout_min"],
                            c["batch_poll_sec"])
        if r2["ok"]:
            got2 = (got2_online if sub2 is None
                    else collect(GCS, sub2["dest"], order2, "pass2"))
            for cid in order2:
                g = got2.get(cid)
                if not g:
                    continue
                g = dict(g)
                g["json"] = v9.parse_json(g["content"]) if g["ok"] else None
                pass2[cid] = g
        else:
            log(f"!! pass2 未成功（{r2['state']}），本条只入库 pass1 结果")

    # ---------- F 落盘 ----------
    results = {cid: build_record(c, cid, clip_frames, meta, ocr_texts, intervals,
                                 speech, pass1, pass2, crops_by_clip,
                                 has_gaze, variant, caps_dir)
               for cid in order1 if cid in clip_frames}
    cw.flush_captions(api, c, rec, caps_dir, results, set(), final=True)
    st["pass1_result"], st["pass2_result"] = r1, r2
    save_state(api, c, dict(st, stage="captions_uploaded"))

    return finalize(c, api, caps_dir, rec, clip_frames, has_gaze, variant,
                    intervals, ocr_texts, gaze_rows, t_all, st,
                    turnaround={"pass1_min": r1["turnaround_min"],
                                "pass2_min": r2["turnaround_min"]})


def main():
    global GCS
    c = cfg()
    cw.setup_adc()
    cw.WD = cw.Watchdog(c["idle_min"] * 60)
    log(f"看门狗：{c['idle_min']:.0f} 分钟无进展则自动退出")
    GCS = Gcs(c["bucket"], workers=c["upload_workers"])
    log(f"GCS 桶 gs://{c['bucket']}  run_id={c['run_id']}  model={c['model']}")

    from huggingface_hub import HfApi
    api = HfApi(token=c["token"])
    tars = c["tars"]
    log(f"本分片 {len(tars)} 条录制")
    done, failed = [], []
    for i, tp in enumerate(tars, 1):
        log(f"\n########## [{i}/{len(tars)}] {Path(tp).stem} ##########")
        try:
            process_one(dict(c, tar=tp, rec=Path(tp).stem), api)
            done.append(tp)
        except Exception as e:                                # noqa: BLE001
            import traceback
            log(f"!! 这条挂了，继续下一条: {type(e).__name__}: {str(e)[:300]}")
            log(traceback.format_exc()[-1500:])
            failed.append(tp)
        finally:
            shutil.rmtree(WORK, ignore_errors=True)
    if cw.WD:
        cw.WD.stop()
    GCS.shutdown()
    log(f"\n分片结束: 成功 {len(done)} / 失败 {len(failed)}")
    if failed:
        log("失败清单: " + ", ".join(Path(f).stem for f in failed))
    log("SHARD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
