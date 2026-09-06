#!/usr/bin/env python3
"""在 HF Job 里跑的 worker：一条 tar 从解包到 caption 到上传，全在云上。

由 pipeline.py 提交，配置全走环境变量：

  必需  HF_TOKEN           写 OUT_REPO 用
        GOOGLE_ADC_JSON    单行 ADC 凭据，落 /tmp/adc.json
        TAR_PATH           aria/2026-05-18/5-18_....tar
  可选  SRC_REPO=mmm8383/pov-data
        OUT_REPO=mmm8383/pov-captions
        MOUNT_ROOT         -v 挂载点，存在就走本地文件（比 HTTP Range 快得多）
        CLIP_SECONDS=30    每个 clip 多少帧（1fps）
        MAX_CLIPS          只跑 tar 里最先出现的 N 个 clip（= 录制末尾），冒烟用
        WORKERS=24         caption 并发
        RENDER_WORKERS=8   解码/画圈并发
        MODEL=gemini-3.7-flash
        ONLY_FAILED=0      1 = 只重跑 ok=false 的 clip
        SKIP_CLIP_TARS=0   1 = 不生成/上传帧包（只要 caption 时更快）

产物写 OUT_REPO：
  captions/<rec>/clip_XXXX.json        逐 clip，断点续跑的单位
  captions/<rec>/captions_full.{json,jsonl,txt}
  captions/<rec>/{manifest,report}.json
  clips/<rec>/clip_XXXX.tar            30 张 1440px 帧 + audio_16k.wav + meta.json
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import shutil
import tarfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import caption_core as cc
import caption_v9 as cv9
import caption_onepass as c1p
import remote_tar as rt

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", "/tmp/povwork"))


def log(*a):
    print(*a, flush=True)


class Watchdog:
    """长时间没进展就自己退出。

    HF Job 跑完脚本就退，正常不会空烧；但网络卡死/SDK 无限重试时，job 会一直挂到
    --timeout 才被砍。看门狗把「挂死」的代价从几小时压到几分钟。
    """

    def __init__(self, idle_limit_s: float):
        self.idle_limit = idle_limit_s
        self.last = time.time()
        self.note = "启动"
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def beat(self, note: str = ""):
        self.last = time.time()
        if note:
            self.note = note

    def _run(self):
        while not self._stop.wait(20):
            idle = time.time() - self.last
            if idle > self.idle_limit:
                log(f"!! 看门狗：{idle/60:.1f} 分钟无进展（最后一步：{self.note}），"
                    f"主动退出，避免空转计费")
                os._exit(75)

    def stop(self):
        self._stop.set()


WD: Watchdog | None = None


def beat(note=""):
    if WD:
        WD.beat(note)


# ---------------------------------------------------------------- 配置

def cfg():
    tars = [x.strip() for x in os.environ.get("TAR_LIST", "").split(",") if x.strip()]
    c = {
        "tar": os.environ.get("TAR_PATH") or tars[0],
        "tars": tars or [os.environ["TAR_PATH"]],
        "src_repo": os.environ.get("SRC_REPO", "mmm8383/pov-data"),
        "out_repo": os.environ.get("OUT_REPO", "mmm8383/pov-captions"),
        "mount": os.environ.get("MOUNT_ROOT") or None,
        "clip_seconds": int(os.environ.get("CLIP_SECONDS", "30")),
        "max_clips": int(os.environ["MAX_CLIPS"]) if os.environ.get("MAX_CLIPS") else None,
        "workers": int(os.environ.get("WORKERS", "24")),
        "render_workers": int(os.environ.get("RENDER_WORKERS", "8")),
        "model": os.environ.get("MODEL", cc.DEFAULT_MODEL),
        "only_failed": os.environ.get("ONLY_FAILED", "0") == "1",
        "skip_clip_tars": os.environ.get("SKIP_CLIP_TARS", "0") == "1",
        "token": os.environ.get("HF_TOKEN"),
        "idle_min": float(os.environ.get("IDLE_TIMEOUT_MIN", "15")),
        # v9 双流程（pass0 语音对齐 + pass1 caption/区域框 + pass2 ULTRA_HIGH crop 修订）。
        # 默认 v6 单遍，向后兼容；caption_v9.py 是新增模块，不改 caption_core.py。
        "caption_version": os.environ.get("CAPTION_VERSION", "v10"),
        "rounds": int(os.environ.get("ROUNDS", "1")),          # 只对 v9 生效，v6 忽略
        "v9_use_ocr": os.environ.get("V9_USE_OCR", "1") == "1",
        "v9_use_vad": os.environ.get("V9_USE_VAD", "1") == "1",
        # {rec: [clip_id,...]} —— 给了就只对白名单里的 clip 发 caption 请求；tar 仍整条
        # walk（流式 I/O，云端 209MB/s，躲不掉也不值得躲），但省掉不需要窗口的 Gemini 调用，
        # 那才是全量跑的钱和时间大头。不给就是旧行为（整条 tar 全 caption），v6/v9 都兼容。
        "wanted_clips": (json.loads(os.environ["WANTED_CLIPS_JSON"])
                        if os.environ.get("WANTED_CLIPS_JSON") else None),
    }
    c["rec"] = Path(c["tar"]).stem
    return c


def setup_adc():
    raw = os.environ.get("GOOGLE_ADC_JSON")
    if not raw:
        raise SystemExit("缺 GOOGLE_ADC_JSON")
    p = Path("/tmp/adc.json")
    p.write_text(raw)
    p.chmod(0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", cc.ADC_PROJECT)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", cc.ADC_LOCATION)


# ---------------------------------------------------------------- 尾部：gaze + ocr

def read_tail_adaptive(src, want_frames_from_zero=True):
    """自适应扩大尾部窗口，直到把 ocr_text 全部（frame_00000）和 gaze.csv 都收进来。"""
    n = 16 << 20
    while True:
        entries, buf, base = rt.read_tail_entries(src, n)
        ocr = [e for e in entries if "/picture/ocr_text/" in e["name"]]
        idx = [cc.frame_index(e["name"]) for e in ocr]
        idx = [i for i in idx if i is not None]
        covered = bool(idx) and min(idx) == 0
        if covered or not want_frames_from_zero or n >= src.size or n >= (512 << 20):
            return entries, buf, base
        n = min(n * 2, src.size)


def extract_tail_payload(entries, buf, base):
    gaze_rows, ocr_texts = {}, {}
    for e in entries:
        s = e["data_offset"] - base
        if s < 0 or s + e["size"] > len(buf):
            continue
        blob = buf[s:s + e["size"]]
        if e["name"].endswith("eye_tracking/gaze.csv"):
            for r in csv.DictReader(io.StringIO(blob.decode("utf-8", "replace"))):
                gaze_rows[r["frame_file"]] = r
        elif "/picture/ocr_text/" in e["name"] and e["name"].endswith(".txt"):
            i = cc.frame_index(e["name"])
            if i is not None:
                ocr_texts[i] = (os.path.basename(e["name"])[:-4],
                                cc.clean_ocr(blob.decode("utf-8", "replace")))
    return gaze_rows, ocr_texts


# ---------------------------------------------------------------- 音频

def parse_wav_header(head: bytes):
    """从 RIFF 头解析参数与 data 块位置，不依赖 wave 模块能 seek 整个文件。"""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise RuntimeError("不是 WAV")
    pos = 12
    fmt = None
    while pos + 8 <= len(head):
        cid = head[pos:pos + 4]
        csize = int.from_bytes(head[pos + 4:pos + 8], "little")
        body = pos + 8
        if cid == b"fmt ":
            fmt = {
                "channels": int.from_bytes(head[body + 2:body + 4], "little"),
                "rate": int.from_bytes(head[body + 4:body + 8], "little"),
                "bits": int.from_bytes(head[body + 14:body + 16], "little"),
            }
        elif cid == b"data":
            if not fmt:
                raise RuntimeError("data 在 fmt 之前")
            fmt["data_offset"] = body
            fmt["data_size"] = csize
            fmt["width"] = fmt["bits"] // 8
            fmt["frame_bytes"] = fmt["width"] * fmt["channels"]
            return fmt
        pos = body + csize + (csize & 1)
    raise RuntimeError("没找到 data 块")


def resample_to_16k(pcm: bytes, rate: int, channels: int, width: int) -> bytes:
    """48k(或任意) → 16k mono。优先 audioop，缺了就用 numpy 盒式滤波抽取。"""
    try:
        import audioop
        if channels > 1:
            pcm = audioop.tomono(pcm, width, 0.5, 0.5)
        out, _ = audioop.ratecv(pcm, width, 1, rate, 16000, None)
        return out
    except Exception:                                        # noqa: BLE001
        import numpy as np
        a = np.frombuffer(pcm, dtype="<i2")
        if channels > 1:
            a = a.reshape(-1, channels).mean(axis=1)
        k = max(1, round(rate / 16000))
        n = (len(a) // k) * k
        a = a[:n].reshape(-1, k).mean(axis=1)
        return a.astype("<i2").tobytes()


def wav_bytes(pcm16k: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16k)
    return buf.getvalue()


class AudioSlicer:
    """按秒从远端 wav 里精确取一段（线性 PCM，可按字节切）。"""

    def __init__(self, src, wav_entry):
        self.src = src
        self.base = wav_entry["data_offset"]
        self.size = wav_entry["size"]
        self.h = parse_wav_header(src.read(self.base, 4096))

    def slice16k(self, start_sec: float, dur_sec: float) -> bytes:
        h = self.h
        off = self.base + h["data_offset"] + int(start_sec * h["rate"]) * h["frame_bytes"]
        n = int(dur_sec * h["rate"]) * h["frame_bytes"]
        n = min(n, self.base + h["data_offset"] + h["data_size"] - off)
        if n <= 0:
            return b""
        pcm = self.src.read(off, n)
        return resample_to_16k(pcm, h["rate"], h["channels"], h["width"])


def locate_sections_safe(src):
    """`rt.locate_sections()` 假设布局 A（eye_tracking → audio → picture/masked →
    picture/ocr_text → gaze.csv），靠二分找 eye_tracking 末尾、紧接着就是 audio 定位。
    实测至少还有布局 B（ocr_text 正序在前、audio 靠后）、以及**部分录制根本没有
    audio/ 段**，这两种都会让它直接抛 RuntimeError，导致整条录制（连同本来完好的
    画面/眼动）全部损失——2026-09-04 铺 200 clip 时 4-27_hkt1819-1940_ecab4eec
    就是这样丢的（day 4-27 在 pipelines/CLAUDE.md 记的坑列表里本就有前例）。

    先试原函数（多数 tar 都是布局 A，最快）；失败了退化到 `rt.walk_headers()`
    通用扫描 —— 不假设顺序，也不要求 audio 一定存在，找到 picture/masked 的
    起点就能继续正常 caption（只是没有 transcript 可用，has_gaze 不受影响）。
    """
    try:
        return rt.locate_sections(src)
    except Exception as e:
        log(f"locate_sections 失败({type(e).__name__}: {e})，退化到 walk_headers 通用扫描")
        wav = tr = pictures_start = None
        for name, size, off in rt.walk_headers(src):
            if pictures_start is None and "/picture/masked/" in name:
                pictures_start = off
            elif name.endswith("audio/anonymized.wav"):
                wav = {"data_offset": off, "size": size}
            elif name.endswith("audio/transcript.txt"):
                tr = {"data_offset": off, "size": size}
            if wav and tr and pictures_start is not None:
                break
        if pictures_start is None:
            raise RuntimeError("walk_headers 兜底也没找到 picture/masked/，"
                                "这条录制可能真的没有画面") from e
        log(f"  兜底定位: wav={'有' if wav else '无'} transcript={'有' if tr else '无'} "
            f"pictures_start={pictures_start}")
        return {"wav": wav, "transcript": tr, "pictures_start": pictures_start}


# ---------------------------------------------------------------- HF 上传

def commit(api, repo, pairs, msg):
    """pairs = [(本地路径, repo 内路径)]，一次 commit 提交完。"""
    from huggingface_hub import CommitOperationAdd
    if not pairs:
        return
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(l)) for l, r in pairs]
    # HF 的仓库提交限流是 **128 次/小时**，超了会 429 并提示「约 1 小时后重试」。
    # 2026-09-05 十几个 job 并发时打爆过：原来只重试 4 次共 50 秒，必然失败，
    # 然后 raise 让整个 job 崩掉 —— 已经算好（且已计费）的 caption 就丢了。
    # 现在对 429 用分钟级退避扛过限流窗口，非限流错误仍然快重试。
    waits = [10, 30, 60, 120, 240, 480, 600, 600, 600]
    for attempt, w in enumerate(waits):
        try:
            api.create_commit(repo_id=repo, repo_type="dataset",
                              operations=ops, commit_message=msg)
            return
        except Exception as e:                                # noqa: BLE001
            msg_s = str(e)
            limited = "429" in msg_s or "rate limit" in msg_s.lower()
            log(f"  commit 失败({attempt+1}/{len(waits)}"
                f"{'，限流' if limited else ''}): {msg_s[:160]}")
            time.sleep(w if limited else min(w, 20))
    raise RuntimeError(f"commit 反复失败: {msg}")


def existing_clip_ids(api, repo, rec, prefix, suffix):
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception:                                         # noqa: BLE001
        return set()
    out = set()
    pre = f"{prefix}/{rec}/clip_"
    for f in files:
        if f.startswith(pre) and f.endswith(suffix):
            try:
                out.add(int(f[len(pre):-len(suffix)]))
            except ValueError:
                pass
    return out


def remote_failed_clip_ids(api, repo, rec, ids):
    """把云上已有的 clip json 拉下来看 ok，返回失败的那些（--only-failed 用）。"""
    from huggingface_hub import hf_hub_download
    bad = set()

    def check(cid):
        try:
            p = hf_hub_download(repo, f"captions/{rec}/clip_{cid:04d}.json",
                                repo_type="dataset", token=api.token)
            return cid, bool(json.loads(Path(p).read_text()).get("ok"))
        except Exception:                                     # noqa: BLE001
            return cid, False
    with ThreadPoolExecutor(16) as ex:
        for cid, ok in ex.map(check, sorted(ids)):
            if not ok:
                bad.add(cid)
    return bad


# ---------------------------------------------------------------- 主流程

def main():
    global WD
    c = cfg()
    setup_adc()
    WD = Watchdog(c["idle_min"] * 60)
    log(f"看门狗：{c['idle_min']:.0f} 分钟无进展则自动退出")
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
            shutil.rmtree(WORK, ignore_errors=True)           # 每条后清盘，别跨 tar 累积
    if WD:
        WD.stop()
    log(f"\n分片结束: 成功 {len(done)} / 失败 {len(failed)}")
    if failed:
        log("失败清单: " + ", ".join(Path(f).stem for f in failed))
    log("SHARD_DONE")


def process_one(c, api):
    rec = c["rec"]
    is_v9 = c["caption_version"] == "v9"
    # v9 pass1/pass2 都要 2880px 原图（pass2 从 pass1 用过的同一张图上按框 crop，
    # 不再重取字节区间）；v6 沿用 1440px q60。
    stored_dim, stored_q = (cv9.FRAME_DIM, cv9.FRAME_Q) if is_v9 else (cc.MAX_DIM, cc.JPEG_QUALITY)
    log(f"=== {rec} ===")
    log(f"src={c['src_repo']} out={c['out_repo']} mount={c['mount']} "
        f"clip={c['clip_seconds']}s workers={c['workers']} max_clips={c['max_clips']} "
        f"caption_version={c['caption_version']} stored_frame={stored_dim}px")

    frames_dir = WORK / rec / "frames"
    caps_dir = WORK / rec / "captions"
    tars_dir = WORK / rec / "clips"
    for d in (frames_dir, caps_dir, tars_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    src = rt.open_source(c["tar"], mount_root=c["mount"], token=c["token"])
    log(f"源: {'挂载文件' if src.seekable else 'HTTP Range'}  {src.size/1e9:.2f} GB")

    # 1) 尾部：gaze.csv + 全部 ocr_text
    t0 = time.time()
    entries, buf, base = read_tail_adaptive(src)
    gaze_rows, ocr_texts = extract_tail_payload(entries, buf, base)
    del buf
    has_gaze = len(gaze_rows) > 0
    beat("读尾部 gaze/ocr")
    log(f"尾部: gaze={len(gaze_rows)} 行, ocr={len(ocr_texts)} 帧  ({time.time()-t0:.1f}s)")
    trails = cc.build_trails(gaze_rows) if has_gaze else {}

    # 2) audio/transcript 定位（locate_sections_safe：布局 B / 无音频录制走 walk_headers 兜底）
    t0 = time.time()
    sec = locate_sections_safe(src)
    tr_entry = sec["transcript"]
    transcript_text = ""
    if tr_entry:
        transcript_text = src.read(tr_entry["data_offset"], tr_entry["size"]).decode("utf-8", "replace")
    intervals = cc.parse_transcript(transcript_text)
    wav_desc = (f"wav@{sec['wav']['data_offset']} ({sec['wav']['size']/1e6:.0f}MB)"
                if sec["wav"] else "wav=无")
    log(f"段定位: {wav_desc}, transcript={len(intervals)} 区间, "
        f"pictures@{sec['pictures_start']}  ({time.time()-t0:.1f}s)")
    beat("定位 audio/transcript")
    slicer = None if (c["skip_clip_tars"] or sec["wav"] is None) else AudioSlicer(src, sec["wav"])

    # 3) 走 picture/masked/，解码 → 1440px + 注视圈
    n_per_clip = c["clip_seconds"]
    done_caps = existing_clip_ids(api, c["out_repo"], rec, "captions", ".json")
    done_tars = existing_clip_ids(api, c["out_repo"], rec, "clips", ".tar")
    if done_caps:
        log(f"云上已有 {len(done_caps)} 个 clip caption，将跳过（ONLY_FAILED={c['only_failed']}）")

    t0 = time.time()
    clip_frames: dict[int, list] = {}
    complete: list[int] = []
    n_bytes = 0
    stop = False

    # WANTED_CLIPS 前置到解包。原来白名单在下面第 ~530 行才生效，也就是先把整条
    # tar 的 544 个 clip 全解出来（读 27 GB + 渲染 8147 帧 2880px，实测 834s），
    # 再扔掉 541 个只留 3 个。iter_members 的 want() 返回 False 时连 data 都不读，
    # 所以把白名单塞进 want() 就能同时省掉读盘和渲染——补跑场景下是几十倍差别。
    # max_clips 是冒烟测试用的「解前 N 个 clip」，与白名单语义冲突，那条路径不启用。
    allow_cids = None
    if c.get("wanted_clips") is not None and not c["max_clips"]:
        allow_cids = set(c["wanted_clips"].get(rec, []))
        if not allow_cids:
            log("WANTED_CLIPS: 这条录制没有目标 clip，跳过")
            return
        log(f"WANTED_CLIPS 前置到解包：只解 {len(allow_cids)} 个 clip 的帧")

    def want(name):
        if "/picture/masked/" not in name or not name.lower().endswith((".jpg", ".jpeg")):
            return False
        if allow_cids is None:
            return True
        fi = cc.frame_index(os.path.basename(name))
        return fi is not None and (fi // n_per_clip) in allow_cids

    pool = ThreadPoolExecutor(c["render_workers"], thread_name_prefix="render")
    pending = []

    def render_and_store(fname, data):
        pts, valid = trails.get(fname, ([], False)) if has_gaze else (None, False)
        jpg = cc.render_frame(data, pts, valid, size=stored_dim, quality=stored_q)
        (frames_dir / fname).write_bytes(jpg)
        return fname, len(jpg)

    # chunk 必须跟着白名单一起调小。SeqStream 的 skip() 本身是免费的，但读 512 字节
    # 的 tar header 会触发 _schedule() 拉整整一个 chunk（默认 8MB × ahead 2 = 16MB），
    # 而帧间距只有 ~3.3MB，等于跳过等于没跳、27GB 照样全读。256KB chunk 让每个
    # header 只带 512KB 预取，白名单场景下读的字节数降一个量级。
    # 挂载源是 os.pread 随机读，小 chunk 没有额外代价；HTTP Range 源改小会变成
    # 大量小请求，所以只在 seekable（挂载）且启用白名单时才缩。
    _chunk = (256 << 10) if (allow_cids is not None and getattr(src, "seekable", False)) \
        else (8 << 20)
    for name, size, data in rt.iter_members(src, start=sec["pictures_start"], want=want,
                                            chunk=_chunk):
        if data is None:
            if "/picture/ocr_text/" in name and clip_frames:
                break              # 已经走到 ocr 段，masked 结束
            continue
        fname = os.path.basename(name)
        fi = cc.frame_index(fname)
        if fi is None:
            continue
        cid = fi // n_per_clip
        clip_frames.setdefault(cid, []).append(fname)
        n_bytes += size
        pending.append(pool.submit(render_and_store, fname, data))
        if len(pending) >= c["render_workers"] * 4:
            for f in pending[:c["render_workers"] * 2]:
                f.result()
            pending = pending[c["render_workers"] * 2:]
            beat(f"解包中 {len(clip_frames)} clip")
        if c["max_clips"] and len(clip_frames) > c["max_clips"]:
            stop = True
            break
    for f in pending:
        f.result()
    pool.shutdown(wait=True)

    if stop:
        # 最后启动的那个 clip 是不完整的，丢掉
        newest = min(clip_frames)
        for fn in clip_frames.pop(newest):
            (frames_dir / fn).unlink(missing_ok=True)

    total_frames = sum(len(v) for v in clip_frames.values())
    dt = time.time() - t0
    log(f"解包: {total_frames} 帧 / {len(clip_frames)} clip, 读 {n_bytes/1e9:.2f} GB, "
        f"{dt:.0f}s ({n_bytes/dt/1e6:.0f} MB/s)")
    if not clip_frames:
        raise RuntimeError("没解出任何帧")

    # 4) 组 clip → caption
    # 无 MPS 眼动的录制要换 nogaze prompt，避免模型在没有注视圈的画面上编造注视描述。
    _pf = "prompt.txt" if has_gaze else "prompt_nogaze.txt"
    system_prompt = (HERE / "prompts" / _pf).read_text().strip()
    log(f"prompt: {_pf}（has_gaze={has_gaze}）")
    client = cc.make_client()
    # 原来 first_index 和 base_time 各取一次 min，两个 min 可能落在不同帧上；
    # 白名单前置后 clip_frames 只剩子集，这种跨帧混取会让 recording_start_hkt 漂移。
    # 改成锁定「索引最小的那一帧」再同时取它的 index 和 time，结果与解了多少 clip 无关。
    _first = min((f for v in clip_frames.values() for f in v), key=cc.frame_index)
    first_index = cc.frame_index(_first)
    base_time = cc.frame_time(_first) - timedelta(seconds=first_index)  # 推回 frame_00000 的时刻

    def build_sd(cid):
        names = sorted(clip_frames[cid], key=cc.frame_index)
        frames = []
        for fn in names:
            frames.append({"name": fn, "hkt": cc.hkt_hms(fn),
                           "jpg": (frames_dir / fn).read_bytes()})
        ocrs = []
        for fn in names:
            fi = cc.frame_index(fn)
            got = ocr_texts.get(fi)
            if got and got[1]:
                ocrs.append(f"[{cc.hkt_hms(fn)} {got[0]}]: {got[1]}")
        rel = cc.frame_index(names[0])
        dur = len(names)
        lines = cc.transcript_for_window(intervals, rel, dur)
        t0_ = cc.frame_time(names[0])
        return {
            "frames": frames,
            "ocr_texts": ocrs,
            "transcript": "\n".join(lines),
            "n_frames": len(frames),
            "clip_start": t0_.strftime("%Y-%m-%d %H:%M:%S"),
            "clip_end": (t0_ + timedelta(seconds=dur)).strftime("%Y-%m-%d %H:%M:%S"),
        }

    todo = sorted(clip_frames)
    if c["only_failed"]:
        bad = remote_failed_clip_ids(api, c["out_repo"], rec, done_caps & set(todo))
        todo = [x for x in todo if x not in done_caps or x in bad]
        log(f"--only-failed: 云上 {len(done_caps)} 个已有，其中 {len(bad)} 个失败")
    else:
        todo = [x for x in todo if x not in done_caps]
    if c.get("wanted_clips") is not None:
        allow = set(c["wanted_clips"].get(rec, []))
        before = len(todo)
        todo = [x for x in todo if x in allow]
        log(f"WANTED_CLIPS: rec={rec} 白名单{len(allow)}个 → 待caption {before}→{len(todo)}")
    log(f"待 caption: {len(todo)} clip")

    results, uploaded = {}, set()

    def do_clip_v6(cid):
        sd = build_sd(cid)
        dim = cc.enforce_inline_limit(sd, log=lambda *a: None)
        parts = cc.build_parts_vertex(sd, system_prompt)
        r = cc.run_one(client, c["model"], parts, log=lambda *a: None)
        parsed, note = (cc.parse_caption_json(r.get("content", "")) if r["ok"] else (None, "api-error"))
        rec_out = {
            "clip_id": cid,
            "recording": rec,
            "clip_index": f"clip_{cid:04d}",
            "n_frames": sd["n_frames"],
            "frame_dim": dim,
            "clip_start_hkt": sd["clip_start"],
            "clip_end_hkt": sd["clip_end"],
            "payload_bytes": cc.payload_bytes(sd),
            "transcript_lines": len([l for l in sd["transcript"].splitlines() if l.strip()]),
            "ocr_entries": len(sd["ocr_texts"]),
            "has_gaze": has_gaze,
            "ok": bool(r["ok"] and parsed is not None),
            "parse_note": note,
            "usage": {k: r.get(k) for k in ("in", "out", "think", "time", "attempts")},
            "model": r["model"],
            "error": r.get("error"),
            "content_raw": r.get("content", ""),
            "parsed": parsed,
        }
        (caps_dir / f"clip_{cid:04d}.json").write_text(
            json.dumps(rec_out, ensure_ascii=False, indent=2))
        return cid, rec_out, sd

    def do_clip_v10(cid):
        sd = build_sd(cid)
        r = c1p.process_clip(client, sd, system_prompt, c["model"], gaze_rows,
                             log=lambda *a: None)
        parsed = r.get("parsed")
        rec_out = {
            "clip_id": cid, "recording": rec, "clip_index": f"clip_{cid:04d}",
            "n_frames": sd["n_frames"], "frame_dim": stored_dim,
            "clip_start_hkt": sd["clip_start"], "clip_end_hkt": sd["clip_end"],
            "payload_bytes": cc.payload_bytes(sd),
            "transcript_lines": len([l for l in sd["transcript"].splitlines() if l.strip()]),
            "ocr_entries": len(sd["ocr_texts"]), "has_gaze": has_gaze,
            "ok": bool(r.get("ok") and parsed is not None),
            "parse_note": "v10-1pass-medium" if parsed is not None else "api-error",
            "usage": {k: r.get(k) for k in ("in", "out", "think", "time")},
            "model": c["model"], "error": r.get("error"),
            "content_raw": r.get("content", ""), "parsed": parsed,
            "pipeline": "v10-1pass", "media_resolution": "MEDIUM",
            "n_zooms": r.get("zooms", 0),
        }
        (caps_dir / f"clip_{cid:04d}.json").write_text(
            json.dumps(rec_out, ensure_ascii=False, indent=2))
        return cid, rec_out, sd

    def do_clip_v9(cid):
        sd = build_sd(cid)
        wav16k = None
        if slicer is not None:
            first_idx = cc.frame_index(sorted(clip_frames[cid], key=cc.frame_index)[0])
            pcm = slicer.slice16k(first_idx, sd["n_frames"])
            if pcm:
                wav16k = wav_bytes(pcm)
        v9 = cv9.process_clip(client, sd, system_prompt, wav16k, clip_tag=f"clip_{cid:04d}",
                              use_transcript=True, use_ocr=c["v9_use_ocr"],
                              use_vad=c["v9_use_vad"], log=lambda *a: None)
        p1, p2, sp = v9.get("pass1") or {}, v9.get("pass2") or {}, v9.get("speech") or {}
        merged = v9.get("merged")
        parsed = merged
        if parsed is None and p1.get("ok"):
            # pass1 成功但没有 crop（n_crops==0）或 pass2 失败：仍保留 pass1 的 caption
            parsed = cv9.parse_json(p1.get("content", ""))
        usage = {
            "in": (p1.get("in") or 0) + (p2.get("in") or 0) + (sp.get("in") or 0),
            "out": (p1.get("out") or 0) + (p2.get("out") or 0) + (sp.get("out") or 0),
            "think": (p1.get("think") or 0) + (p2.get("think") or 0),
            "time": round((p1.get("time") or 0) + (p2.get("time") or 0) + (sp.get("time") or 0), 1),
            "attempts": None,
        }
        note = ("v9-merged" if merged is not None else
                ("v9-pass1-only" if parsed is not None else "v9-fail"))
        rec_out = {
            "clip_id": cid,
            "recording": rec,
            "clip_index": f"clip_{cid:04d}",
            "n_frames": sd["n_frames"],
            "frame_dim": stored_dim,
            "clip_start_hkt": sd["clip_start"],
            "clip_end_hkt": sd["clip_end"],
            "payload_bytes": cc.payload_bytes(sd),
            "transcript_lines": len([l for l in sd["transcript"].splitlines() if l.strip()]),
            "ocr_entries": len(sd["ocr_texts"]),
            "has_gaze": has_gaze,
            "ok": bool(cv9.is_complete(v9) and parsed is not None),
            "parse_note": note,
            "usage": usage,
            "model": c["model"],
            "error": p1.get("error") or p2.get("error"),
            "content_raw": p1.get("content", ""),
            "parsed": parsed,
            "pipeline": "v9-2pass",
            "n_crops": v9.get("n_crops", 0),
            "n_regions": p1.get("n_regions", 0),
            "pass1": p1,
            "pass2": p2 or None,
            "speech": sp,
        }
        (caps_dir / f"clip_{cid:04d}.json").write_text(
            json.dumps(rec_out, ensure_ascii=False, indent=2))
        return cid, rec_out, sd

    t0 = time.time()
    pending_uploads = []

    def handle_result(i, n_total, cid, rec_out, sd, n_ok, n_bad, allow_incremental_flush=True):
        results[cid] = rec_out
        beat(f"caption clip_{cid:04d}")
        n_ok += 1 if rec_out["ok"] else 0
        n_bad += 0 if rec_out["ok"] else 1
        segs = len((rec_out["parsed"] or {}).get("segments", []))
        extra = (f" tport1={rec_out['pass1'].get('transport')} n_crops={rec_out.get('n_crops')}"
                if rec_out.get("pipeline") == "v9-2pass" else "")
        log(f"  [{i}/{n_total}] clip_{cid:04d} {'✓' if rec_out['ok'] else '✗'} "
            f"{rec_out['clip_start_hkt'][11:]} segs={segs} "
            f"in={rec_out['usage'].get('in')} out={rec_out['usage'].get('out')} "
            f"{rec_out['usage'].get('time')}s{extra}")
        if not c["skip_clip_tars"] and cid not in done_tars:
            p = tars_dir / f"clip_{cid:04d}.tar"
            try:
                make_clip_tar(p, sd, frames_dir, slicer, cc.frame_index(sorted(
                    clip_frames[cid], key=cc.frame_index)[0]), rec_out)
                pending_uploads.append((p, f"clips/{rec}/clip_{cid:04d}.tar"))
                done_tars.add(cid)          # v9 多轮重试时别对同一 clip 重复生成/排队上传
            except Exception as e:                            # noqa: BLE001
                log(f"    clip 包生成失败: {str(e)[:160]}")
        if allow_incremental_flush and len(results) - len(uploaded) >= 25:
            flush_captions(api, c, rec, caps_dir, results, uploaded)
        return n_ok, n_bad

    n_ok = n_bad = 0
    if c["caption_version"] == "v10":
        with ThreadPoolExecutor(c["workers"], thread_name_prefix="cap") as ex:
            futs = {ex.submit(do_clip_v10, cid): cid for cid in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                cid = futs[fut]
                try:
                    cid, rec_out, sd = fut.result()
                except Exception as e:
                    n_bad += 1
                    log(f"  [{i}/{len(todo)}] clip_{cid:04d} 崩了: {str(e)[:200]}")
                    continue
                n_ok, n_bad = handle_result(i, len(todo), cid, rec_out, sd, n_ok, n_bad)
    elif not is_v9:
        with ThreadPoolExecutor(c["workers"], thread_name_prefix="cap") as ex:
            futs = {ex.submit(do_clip_v6, cid): cid for cid in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                cid = futs[fut]
                try:
                    cid, rec_out, sd = fut.result()
                except Exception as e:                        # noqa: BLE001
                    n_bad += 1
                    log(f"  [{i}/{len(todo)}] clip_{cid:04d} 崩了: {str(e)[:200]}")
                    continue
                n_ok, n_bad = handle_result(i, len(todo), cid, rec_out, sd, n_ok, n_bad)
    else:
        # 33-44MB(30帧)/17-22MB(15帧) 的 payload 走 GCS 时随机掉线 15-20%，与主线
        # run_caption_2pass.py 的 --rounds 一致：按轮退避并发，只重跑没完整的 clip。
        rounds = max(1, c["rounds"])
        best: dict[int, dict] = {}
        remaining = list(todo)
        for rnd in range(rounds):
            if not remaining:
                break
            workers = max(1, c["workers"] - rnd)
            if rnd:
                log(f"  --- v9 round {rnd+1}/{rounds}: {len(remaining)} clip 待重跑, "
                    f"workers={workers} ---")
            with ThreadPoolExecutor(workers, thread_name_prefix="cap") as ex:
                futs = {ex.submit(do_clip_v9, cid): cid for cid in remaining}
                for i, fut in enumerate(as_completed(futs), 1):
                    cid = futs[fut]
                    try:
                        cid, rec_out, sd = fut.result()
                    except Exception as e:                    # noqa: BLE001
                        log(f"  [{i}/{len(remaining)}] clip_{cid:04d} 崩了: {str(e)[:200]}")
                        continue
                    cur = best.get(cid)
                    better = cur is None or (rec_out["ok"] and not cur.get("ok"))
                    if better:
                        best[cid] = rec_out
                    # v9 同一 clip 可能跨轮重试多次，中途不增量 flush（避免把某轮的失败态
                    # 提前 commit 成"已上传"、后面轮次的成功版本反而漏传）；只在全部轮次
                    # 结束后做一次 final flush，写的是每个 clip 的最终最优状态。
                    n_ok, n_bad = handle_result(i, len(remaining), cid, best[cid], sd, n_ok, n_bad,
                                                allow_incremental_flush=False)
            remaining = [cid for cid in todo if not best.get(cid, {}).get("ok")]
        n_ok = sum(1 for v in best.values() if v.get("ok"))
        n_bad = len(todo) - n_ok
    log(f"caption: {n_ok} ✓ / {n_bad} ✗  ({time.time()-t0:.0f}s)")

    # 5) 上传
    flush_captions(api, c, rec, caps_dir, results, uploaded, final=True)
    if pending_uploads:
        t0 = time.time()
        for i in range(0, len(pending_uploads), 25):
            commit(api, c["out_repo"], pending_uploads[i:i + 25],
                   f"clips {rec} [{i}:{i+25}]")
        log(f"clip 包上传 {len(pending_uploads)} 个 ({time.time()-t0:.0f}s)")

    # 6) manifest / report / 合并
    manifest = {
        "recording": rec,
        "tar": c["tar"],
        "src_repo": c["src_repo"],
        "clip_seconds": n_per_clip,
        "has_gaze": has_gaze,
        "gaze_rows": len(gaze_rows),
        "ocr_frames": len(ocr_texts),
        "transcript_intervals": len(intervals),
        "recording_start_hkt": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames_processed": total_frames,
        "clips": sorted(clip_frames),
        "frame_dim": stored_dim,
        "caption_version": c["caption_version"],
        "model": c["model"],
        "max_clips": c["max_clips"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    (caps_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # 上一轮已经跑过、本轮跳过的 clip 得拉回来一起合并，否则 captions_full 会缺这几个
    missing = sorted((done_caps & set(clip_frames)) - set(results))
    if missing:
        from huggingface_hub import hf_hub_download

        def fetch(cid):
            p = hf_hub_download(c["out_repo"], f"captions/{rec}/clip_{cid:04d}.json",
                                repo_type="dataset", token=c["token"])
            (caps_dir / f"clip_{cid:04d}.json").write_bytes(Path(p).read_bytes())
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(fetch, missing))
        log(f"补回 {len(missing)} 个云上已有的 clip 一起合并")

    # 合并成整条录制的成品（本地 pull 时可直接取，不用重算）
    import merge as mg
    merged = mg.merge_recording(mg.load_clips(caps_dir), manifest)
    rep = mg.write_recording_outputs(caps_dir, merged)
    log(f"合并: {rep['clip_ok']}/{rep['clip_count']} clip, {rep['segment_count']} segments, "
        f"in={rep['tokens']['in']} out={rep['tokens'].get('out_billable', 0)} "
        f"估价 ${rep['est_cost_usd']}")
    commit(api, c["out_repo"],
           [(caps_dir / n, f"captions/{rec}/{n}") for n in
            ("manifest.json", "report.json", "captions_full.json",
             "captions_full.jsonl", "captions_full.txt")],
           f"manifest+merged {rec}")
    beat("完成一条")
    log(f"完成，总耗时 {time.time()-t_all:.0f}s")
    log("DONE_OK")
    return manifest


def flush_captions(api, c, rec, caps_dir, results, uploaded, final=False):
    todo = [cid for cid in sorted(results) if cid not in uploaded]
    pairs = [(caps_dir / f"clip_{cid:04d}.json", f"captions/{rec}/clip_{cid:04d}.json")
             for cid in todo]
    pairs = [(l, r) for l, r in pairs if Path(l).exists()]
    if not pairs:
        return
    # 一次 commit 装多少个 clip。原来是 40，十几个 job 并发时会把 HF 的
    # 128 次/小时提交配额打爆（4500 个 clip ÷ 40 = 113 次，再加各 job 的
    # save_state 就超了）。250 一批 → 同样的量只要 18 次。
    chunk = int(os.environ.get("COMMIT_CHUNK", "250"))
    for i in range(0, len(pairs), chunk):
        commit(api, c["out_repo"], pairs[i:i + chunk],
               f"captions {rec} [{i}:{i+chunk}]{' final' if final else ''}")
    uploaded.update(todo)


def make_clip_tar(path: Path, sd: dict, frames_dir: Path, slicer, first_index: int, meta: dict):
    """一个 clip 一个 tar：30 张 1440px 帧 + 16k 音频 + meta，本地只拉抽样的那几个。"""
    with tarfile.open(path, "w") as tf:
        for fr in sd["frames"]:
            p = frames_dir / fr["name"]
            tf.add(str(p), arcname=f"frames/{fr['name']}")
        if slicer is not None:
            pcm = slicer.slice16k(first_index, sd["n_frames"])
            if pcm:
                wb = wav_bytes(pcm)
                ti = tarfile.TarInfo("audio_16k.wav")
                ti.size = len(wb)
                tf.addfile(ti, io.BytesIO(wb))
        mb = json.dumps({k: v for k, v in meta.items()
                         if k not in ("content_raw",)}, ensure_ascii=False, indent=2).encode()
        ti = tarfile.TarInfo("meta.json")
        ti.size = len(mb)
        tf.addfile(ti, io.BytesIO(mb))


if __name__ == "__main__":
    sys.exit(main())
