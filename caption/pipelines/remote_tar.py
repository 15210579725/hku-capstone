#!/usr/bin/env python3
"""远端/本地 tar 的统一访问层。

两个后端同一接口：
  FileSource       本地文件，或 HF Job 里 `-v hf://datasets/...` 挂进来的只读 repo
  HttpRangeSource  HF resolve 端点的 HTTP Range，多线程顺序预取

mmm8383/pov-data 的脱敏包内部结构（本轮 Range 扫出来的）：
  <rec>/eye_tracking/*.jpg      眼部相机图，1Hz，~22KB/张 —— 跳过
  <rec>/audio/anonymized.wav    48kHz mono 16bit
  <rec>/audio/transcript.txt    [YYYY-MM-DD HH:MM:SS HKT -> ... HKT] 文本
  <rec>/picture/masked/*.jpg    2880²、~3.1MB/帧、**按时间倒序**、文件名走 GNU @LongLink
  <rec>/picture/ocr_text/*.txt  每帧一份
  <rec>/eye_tracking/gaze.csv   append 在最末（12 条录制没有）

跳过一个 member 时只推进 offset、不发任何请求，所以"不要 eye_tracking"是真的省带宽。
"""
from __future__ import annotations

import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

BLOCK = 512
REPO = "mmm8383/pov-data"


# ---------------------------------------------------------------- tar header

def parse_header(blk: bytes):
    """解析 512 字节 tar header。返回 None 表示不是 header（结束块/数据块）。"""
    if len(blk) < BLOCK or blk[257:262] != b"ustar":
        return None
    name = blk[0:100].rstrip(b"\0").decode("utf-8", "replace")
    try:
        size = int(blk[124:136].rstrip(b"\0 ").decode() or "0", 8)
    except ValueError:
        return None
    return {"name": name, "size": size, "type": chr(blk[156])}


def _align(n: int) -> int:
    return ((n + BLOCK - 1) // BLOCK) * BLOCK


def scan_headers(buf: bytes, base: int = 0):
    """在任意缓冲里按 512 对齐扫 header（用于二分探测/读尾部）。"""
    out = []
    for i in range(0, max(0, len(buf) - BLOCK + 1), BLOCK):
        h = parse_header(buf[i:i + BLOCK])
        if h:
            h["offset"] = base + i
            h["data_offset"] = base + i + BLOCK
            out.append(h)
    return out


def resolve_longlink(entries, buf: bytes, base: int):
    """把 scan_headers 的结果里的 GNU @LongLink 合并成真实文件名。"""
    out = []
    pending = None
    for h in entries:
        if h["type"] == "L":                       # ././@LongLink
            s = h["data_offset"] - base
            pending = buf[s:s + h["size"]].rstrip(b"\0").decode("utf-8", "replace")
            continue
        if pending:
            h = dict(h, name=pending)
            pending = None
        out.append(h)
    return out


# ---------------------------------------------------------------- sources

class FileSource:
    """本地文件 / 挂载进来的 repo 文件。

    read() 必须是线程安全的：batch_worker 按索引里的偏移用 24 个线程随机取帧。
    原来的实现是共享句柄 seek()+read()，两个线程一交错就读到错位的字节 ——
    实测表现为 PIL "cannot identify image file"，786 帧里坏 18 帧、连累 8 个 clip
    整个作废。os.pread 是原子的（偏移由参数给，不动文件位置），没有这个问题；
    平台不支持时退回加锁的 seek+read。
    """

    seekable = True

    def __init__(self, path):
        self.path = str(path)
        self.size = os.path.getsize(self.path)
        self._fh = open(self.path, "rb")
        self._fd = self._fh.fileno()
        self._lock = threading.Lock()
        self._pread = hasattr(os, "pread")

    def read(self, off: int, n: int) -> bytes:
        if n <= 0:
            return b""
        if self._pread:
            out = bytearray()
            while len(out) < n:                     # pread 允许短读，补齐到要的长度
                chunk = os.pread(self._fd, n - len(out), off + len(out))
                if not chunk:
                    break
                out += chunk
            return bytes(out)
        with self._lock:
            self._fh.seek(off)
            return self._fh.read(n)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class HttpRangeSource:
    """HF resolve 端点的 Range 读。CDN 重定向 URL 会缓存并在过期时重解析。"""

    seekable = False

    def __init__(self, path_in_repo: str, repo: str = REPO, token: str = None,
                 size: int = None, retries: int = 5):
        import requests
        from huggingface_hub import get_token
        self.requests = requests
        self.path = path_in_repo
        self.repo = repo
        self.token = token or os.environ.get("HF_TOKEN") or get_token()
        self.retries = retries
        self.base_url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path_in_repo}"
        self._sess = requests.Session()
        self._cdn = None
        self._cdn_at = 0.0
        self.size = size if size is not None else self._head_size()

    # -- 内部

    def _auth(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _head_size(self):
        r = self._sess.head(self.base_url, headers=self._auth(),
                            allow_redirects=True, timeout=60)
        r.raise_for_status()
        self._cdn, self._cdn_at = r.url, time.time()
        return int(r.headers["Content-Length"])

    def _url(self):
        # 签名 URL 有有效期，10 分钟重解析一次
        if self._cdn is None or time.time() - self._cdn_at > 600:
            try:
                r = self._sess.head(self.base_url, headers=self._auth(),
                                    allow_redirects=True, timeout=60)
                self._cdn, self._cdn_at = r.url, time.time()
            except Exception:
                return self.base_url
        return self._cdn

    # -- 接口

    def read(self, off: int, n: int) -> bytes:
        if n <= 0:
            return b""
        end = min(off + n, self.size) - 1
        if end < off:
            return b""
        last = None
        for attempt in range(self.retries):
            url = self._url()
            hdr = {"Range": f"bytes={off}-{end}"}
            if url == self.base_url or "huggingface.co" in url:
                hdr.update(self._auth())
            try:
                r = self._sess.get(url, headers=hdr, timeout=300)
                if r.status_code in (403, 401):        # 签名过期 → 重解析
                    self._cdn = None
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                data = r.content
                if len(data) != end - off + 1:
                    raise RuntimeError(f"short read {len(data)} != {end - off + 1}")
                return data
            except Exception as e:                      # noqa: BLE001
                last = e
                self._cdn = None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"range read failed off={off} n={n}: {last}")

    def close(self):
        try:
            self._sess.close()
        except Exception:
            pass


# ---------------------------------------------------------------- 顺序流

class SeqStream:
    """顺序读 + 免费跳过。跳过的区间不会被下载。"""

    def __init__(self, source, start: int = 0, chunk: int = 8 << 20, ahead: int = 8):
        self.src = source
        self.pos = start
        self.chunk = chunk
        self.ahead = max(1, ahead if not source.seekable else 2)
        self.ex = ThreadPoolExecutor(self.ahead, thread_name_prefix="tarfetch")
        self.futs = {}

    def _cidx(self, p):
        return p // self.chunk

    def _schedule(self):
        c0 = self._cidx(self.pos)
        for c in list(self.futs):
            if c < c0:
                self.futs.pop(c).cancel()
        for c in range(c0, c0 + self.ahead):
            off = c * self.chunk
            if off >= self.src.size:
                break
            if c not in self.futs:
                n = min(self.chunk, self.src.size - off)
                self.futs[c] = self.ex.submit(self.src.read, off, n)

    def read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0 and self.pos < self.src.size:
            self._schedule()
            c = self._cidx(self.pos)
            if c not in self.futs:
                break
            buf = self.futs[c].result()
            s = self.pos - c * self.chunk
            take = min(n, len(buf) - s)
            if take <= 0:
                break
            out += buf[s:s + take]
            self.pos += take
            n -= take
        return bytes(out)

    def skip(self, n: int):
        self.pos += n

    def seek(self, pos: int):
        self.pos = pos

    def close(self):
        for f in self.futs.values():
            f.cancel()
        self.futs.clear()
        self.ex.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- 成员遍历

def iter_members(source, start: int = 0, want=None, end: int = None,
                 chunk: int = 8 << 20, ahead: int = 8, with_offset: bool = False):
    """从 start 顺序遍历 member。

    want(name) -> bool 决定是否读取该 member 的数据：
      True  → yield (name, size, data_bytes)
      False → yield (name, size, None)，且这段字节不会被下载

    with_offset=True 时多 yield 一个 data_offset（member 数据在 tar 里的绝对偏移），
    调用方记下来就能事后用 source.read(offset, size) 随机回读这一个 member，
    不必重走整条 tar。batch_worker 的 pass2 crop 靠这个回取 2880px 原帧。
    """
    limit = source.size if end is None else min(end, source.size)
    st = SeqStream(source, start, chunk=chunk, ahead=ahead)
    try:
        pending_name = None
        while st.pos < limit:
            blk = st.read(BLOCK)
            if len(blk) < BLOCK:
                break
            h = parse_header(blk)
            if h is None:
                if blk.strip(b"\0") == b"":          # 结束块
                    break
                continue                              # 容错：跳过异常块
            name, size, typ = h["name"], h["size"], h["type"]
            if typ == "L":                            # GNU 长文件名
                raw = st.read(_align(size))
                pending_name = raw[:size].rstrip(b"\0").decode("utf-8", "replace")
                continue
            if pending_name:
                name, pending_name = pending_name, None
            padded = _align(size)
            off = st.pos                              # 数据段起点，读/跳之前取
            if size and want and want(name):
                data = st.read(size)
                st.skip(padded - size)
                yield (name, size, data, off) if with_offset else (name, size, data)
            else:
                st.skip(padded)
                yield (name, size, None, off) if with_offset else (name, size, None)
    finally:
        st.close()


def walk_headers(source, start: int = 0, end: int = None, window: int = 64 << 10):
    """只读 header、不读数据地把整条 tar 索引一遍，yield (name, size, data_offset)。

    为什么需要它：数据集里**不止一种 tar 布局**。文档记的那种是
    `eye_tracking/*.jpg → audio/ → picture/masked → picture/ocr_text → gaze.csv`，
    但实测还有 `picture/ocr_text（正序，在最前）→ picture/masked → gaze.csv`
    且**完全没有 audio/ 段**的（4-27_hkt0930 就是）。`locate_sections()` 靠
    「二分找 eye_tracking 末尾、紧接着就是 audio」的假设定位，在后一种上直接抛
    「找不到 audio/anonymized.wav」。与其给每种布局打补丁，不如不假设布局。

    窗口取 64KB 而不是 SeqStream 的 8MB：小 member（ocr_text ~2KB）一次读能吃下
    几十个 header，大 member（masked 帧 ~3MB）也只多读 64KB 就能跳过去。
    最大那条录制约 4 万个 member，总共只读 ~2 GB，挂载盘上几十秒。
    """
    limit = source.size if end is None else min(end, source.size)
    pos = start
    base, buf = -1, b""
    pending_name = None

    def ensure(p, need):
        nonlocal base, buf
        if base <= p and p + need <= base + len(buf):
            return True
        base = p
        buf = source.read(base, max(window, need))
        return len(buf) >= need

    while pos < limit:
        if not ensure(pos, BLOCK):
            break
        blk = buf[pos - base:pos - base + BLOCK]
        h = parse_header(blk)
        pos += BLOCK
        if h is None:
            if blk.strip(b"\0") == b"":          # 结束块
                break
            continue                              # 容错：跳过异常块
        name, size, typ = h["name"], h["size"], h["type"]
        padded = _align(size)
        if typ == "L":                            # GNU 长文件名，名字在数据段里
            if not ensure(pos, padded):
                break
            raw = buf[pos - base:pos - base + size]
            pending_name = raw.rstrip(b"\0").decode("utf-8", "replace")
            pos += padded
            continue
        if pending_name:
            name, pending_name = pending_name, None
        yield name, size, pos
        pos += padded


def read_tail_entries(source, n: int = 8 << 20):
    """读尾部 n 字节并解析出 member（用于 gaze.csv / ocr_text 段定位）。"""
    n = min(n, source.size)
    base = source.size - n
    buf = source.read(base, n)
    return resolve_longlink(scan_headers(buf, base), buf, base), buf, base


def find_boundary(source, pred_left, probe: int = 4 << 20, max_steps: int = 20):
    """二分找"左侧条件成立"的最后位置。

    pred_left(headers) -> True 表示这个探测窗口还在目标段的左边。
    返回 (lo, hi)：真实边界落在 [lo, hi) 内，且 hi - lo < probe。
    """
    lo, hi = 0, source.size
    for _ in range(max_steps):
        if hi - lo < probe:
            break
        mid = ((lo + hi) // 2) // BLOCK * BLOCK
        buf = source.read(mid, probe)
        hs = scan_headers(buf, mid)
        if pred_left(hs):
            lo = mid
        else:
            hi = mid
    return lo, hi


def locate_sections(source, probe: int = 4 << 20):
    """定位 audio/anonymized.wav、audio/transcript.txt、picture/ 的起始 offset。

    先二分到 eye_tracking 段的末尾（那之后紧接着就是 audio/），再顺着头往下读。
    对 seekable 源同样有效，只是探测更便宜。
    """
    def still_eye(hs):
        return any("eye_tracking/" in h["name"] and h["name"].endswith(".jpg") for h in hs)

    lo, _ = find_boundary(source, still_eye, probe=probe)

    wav = tr = None
    off = lo
    for _ in range(6):
        buf = source.read(off, probe)
        hs = resolve_longlink(scan_headers(buf, off), buf, off)
        for h in hs:
            if h["name"].endswith("audio/anonymized.wav"):
                wav = h
            elif h["name"].endswith("audio/transcript.txt"):
                tr = h
        if wav:
            break
        off += probe - BLOCK
    if wav is None:
        raise RuntimeError("找不到 audio/anonymized.wav")

    if tr is None:
        # transcript 紧跟在 wav 之后
        after = wav["data_offset"] + _align(wav["size"])
        buf = source.read(after, 256 * 1024)
        hs = resolve_longlink(scan_headers(buf, after), buf, after)
        for h in hs:
            if h["name"].endswith("audio/transcript.txt"):
                tr = h
                break
    pictures_start = wav["data_offset"] + _align(wav["size"])
    return {"wav": wav, "transcript": tr, "pictures_start": pictures_start}


def open_source(tar_path_in_repo: str, mount_root: str = None, token: str = None,
                size: int = None):
    """挂载可用就用挂载（HF Job 内网），否则退回 HTTP Range。"""
    if mount_root:
        p = os.path.join(mount_root, tar_path_in_repo)
        if os.path.exists(p):
            return FileSource(p)
    return HttpRangeSource(tar_path_in_repo, token=token, size=size)
