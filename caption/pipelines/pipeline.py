#!/usr/bin/env python3
"""POV caption 流水线的本地入口：选片 / 提交云端 job / 拉结果 / 合并 / 渲染审核视频。

重活全在 HF Job 上（本机到 HF 只有 ~2.3 MB/s，主盘也只剩 8 GB，拉 tar 不现实）。
本机只负责提交和取回小产物。

  python pipeline.py days                                  列出 HF 上有哪些天
  python pipeline.py scan --month 2026-05 --limit 12       候选片源（时长/大小/眼动/语音密度）
  python pipeline.py run --tar aria/2026-05-18/5-18_...tar 提交一条
  python pipeline.py run --day 2026-05-18                  提交一天的全部
  python pipeline.py run --tar ... --max-clips 5           冒烟
  python pipeline.py run --tar ... --only-failed           只重跑失败的 clip
  python pipeline.py status [--all]                        看 job
  python pipeline.py logs <job_id>
  python pipeline.py pull --rec <rec> [--sample 8]         拉 caption（+抽样 clip 包）
  python pipeline.py merge --day 2026-05-18                合成整天
  python pipeline.py render --rec <rec> --sample 8         审核视频
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SRC_REPO = "mmm8383/pov-data"
OUT_REPO = os.environ.get("POV_OUT_REPO", "mmm8383/pov-captions")
CODE_FILES = ["remote_tar.py", "caption_core.py", "caption_v9.py", "caption_onepass.py",
               "cloud_worker.py", "merge.py", "deid.py"]
FLAVOR = "cpu-upgrade"
IMAGE = "python:3.12"
# 两个 ADC 项目并行跑时，用 ADC_FILE 指到 ~/.config/gcloud/adc-profiles/<账号>.json，
# 同时用 --env ADC_PROJECT=<项目> --env GCS_BUCKET=<该项目下的桶> 把 job 侧也切过去。
# 2026-09-04 踩过：只换凭据不换 ADC_PROJECT，batch 会拿 A 的凭据去 B 项目建作业 → 403。
ADC_PATH = Path(os.environ.get("ADC_FILE")
                or Path.home() / ".config/gcloud/application_default_credentials.json")
JOBS_LOG = HERE / "out" / "_jobs.jsonl"

DUR_RE = re.compile(r"_(\d+)m_")


def api():
    from huggingface_hub import HfApi
    return HfApi()


def token():
    from huggingface_hub import get_token
    t = os.environ.get("HF_TOKEN") or get_token()
    if not t:
        sys.exit("没有 HF token：先 `hf auth login`")
    return t


def list_tars():
    a = api()
    info = a.repo_info(SRC_REPO, repo_type="dataset", files_metadata=True)
    out = []
    for s in info.siblings:
        if s.rfilename.startswith("aria/") and s.rfilename.endswith(".tar"):
            m = DUR_RE.search(s.rfilename)
            out.append({"path": s.rfilename, "size": s.size or 0,
                        "minutes": int(m.group(1)) if m else 0,
                        "day": s.rfilename.split("/")[1],
                        "rec": Path(s.rfilename).stem})
    return sorted(out, key=lambda r: r["path"])


def gaze_available():
    """todo.txt 里列的是有 MPS 眼动的那 521 条；剩下 12 条确认无眼动。"""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SRC_REPO, "_work/todo.txt", repo_type="dataset")
    return set(l.strip() for l in open(p) if l.strip())


# ---------------------------------------------------------------- 命令

def cmd_days(args):
    rows = list_tars()
    by = {}
    for r in rows:
        by.setdefault(r["day"], []).append(r)
    for d in sorted(by):
        rs = by[d]
        print(f"{d}  {len(rs)} 条  {sum(x['minutes'] for x in rs):>4} 分钟  "
              f"{sum(x['size'] for x in rs)/1e9:>6.1f} GB")
    print(f"\n共 {len(by)} 天 / {len(rows)} 条录制")


def cmd_scan(args):
    import remote_tar as rt
    rows = [r for r in list_tars() if r["path"].startswith(f"aria/{args.month}")] \
        if args.month else [r for r in list_tars() if r["day"] == args.day]
    if args.min_minutes:
        rows = [r for r in rows if r["minutes"] >= args.min_minutes]
    if args.max_minutes:
        rows = [r for r in rows if r["minutes"] <= args.max_minutes]
    have = gaze_available()
    rows = [r for r in rows if (r["path"] in have) or not args.gaze_only]
    rows.sort(key=lambda r: r["minutes"])
    rows = rows[:args.limit]
    tok = token()
    print(f"探测 {len(rows)} 条（每条几十 MB Range，估计 {len(rows)*0.6:.0f} 分钟）...\n")

    def probe(r):
        try:
            src = rt.HttpRangeSource(r["path"], token=tok, size=r["size"])
            sec = rt.locate_sections(src)
            tr = sec["transcript"]
            if not tr:
                return r, None
            txt = src.read(tr["data_offset"], tr["size"]).decode("utf-8", "replace")
            src.close()
            lines = [l for l in txt.splitlines() if l.strip()]
            chars = sum(len(re.sub(r"^\[.*?\]\s*", "", l).strip()) for l in lines)
            return r, {"kb": tr["size"] / 1024, "lines": len(lines), "chars": chars,
                       "density": chars / max(1, r["minutes"])}
        except Exception as e:                                # noqa: BLE001
            return r, {"err": str(e)[:60]}

    res = []
    with ThreadPoolExecutor(4) as ex:
        for r, t in ex.map(probe, rows):
            res.append((r, t))
            print(".", end="", flush=True)
    print("\n")
    print(f"{'时长':>5} {'GB':>6} {'眼动':>4} {'语音密度':>9} {'字数':>7}  文件")
    for r, t in sorted(res, key=lambda x: -(x[1] or {}).get("density", -1)):
        g = "有" if r["path"] in have else "无"
        if not t or t.get("err"):
            print(f"{r['minutes']:>4}m {r['size']/1e9:>6.2f} {g:>4} {'?':>9} {'?':>7}  "
                  f"{r['rec'][:52]}  {(t or {}).get('err','no transcript')}")
        else:
            print(f"{r['minutes']:>4}m {r['size']/1e9:>6.2f} {g:>4} "
                  f"{t['density']:>7.1f}字/分 {t['chars']:>7}  {r['rec'][:52]}")


def ensure_repo():
    a = api()
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        a.repo_info(OUT_REPO, repo_type="dataset")
    except RepositoryNotFoundError:
        print(f"新建 private dataset {OUT_REPO}")
        a.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)


def push_code():
    a = api()
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd(path_in_repo=f"_code/{f}", path_or_fileobj=str(HERE / f))
           for f in CODE_FILES]
    for _v in ("v9.txt", "v9_nogaze.txt", "v11.txt", "v11_nogaze.txt"):
        ops.append(CommitOperationAdd(path_in_repo=f"_code/prompts/{_v}",
                                      path_or_fileobj=str(HERE / "prompts" / _v)))
    a.create_commit(repo_id=OUT_REPO, repo_type="dataset", operations=ops,
                    commit_message="worker code")
    print(f"worker 代码已推到 {OUT_REPO}/_code/")


JOB_SCRIPT = """set -e
pip install -q --disable-pip-version-check huggingface_hub google-genai openai pillow requests numpy 2>&1 | tail -2
if [ "${CAPTION_VERSION:-v6}" = "v9" ]; then
  # v9 pass0 用 silero-vad 语音对齐，需要 torch；只装 CPU wheel（不带 CUDA runtime，
  # 比默认 PyPI wheel 小得多），装完再装依赖它的 silero-vad 和 pass2 GCS 兜底用的
  # google-cloud-storage。v6 路径完全不碰这几个包，体积/时间都不受影响。
  # torch 和 torchaudio 必须在同一条 pip 命令里从同一个 index 装，否则后面单独装
  # silero-vad 时它会把 torchaudio 当依赖从默认 PyPI 再装一次、versions 对不上，
  # torchaudio 的 native .so 加载失败（实测：clip 全部退化成 energy-fallback）。
  pip install -q --disable-pip-version-check torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
  pip install -q --disable-pip-version-check silero-vad google-cloud-storage 2>&1 | tail -3
fi
if [ "${CAPTION_VERSION:-v6}" = "v10" ]; then
  # v10 Batch worker uploads inputs to GCS before submitting Vertex Batch.
  pip install -q --disable-pip-version-check google-cloud-storage 2>&1 | tail -3
fi
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(os.environ['OUT_REPO'], repo_type='dataset',
                      allow_patterns=['_code/**'], token=os.environ['HF_TOKEN'])
open('/tmp/codedir', 'w').write(os.path.join(p, '_code'))
PY
# snapshot 里是指向 blobs 的符号链接，直接跑会把 blobs 目录当成 sys.path[0]，
# 同目录的 caption_core 就 import 不到；-L 解引用复制到普通目录再跑。
mkdir -p /tmp/code && cp -rL "$(cat /tmp/codedir)"/. /tmp/code/
cd /tmp/code
ls -la /data 2>/dev/null | head -3 || echo "no /data mount"
python "${ENTRY:-cloud_worker.py}"
"""


def submit(tar_path, args):
    tok = token()
    if not ADC_PATH.exists():
        sys.exit(f"找不到 ADC 凭据 {ADC_PATH}；先 `gcloud auth application-default login`")
    adc = json.dumps(json.loads(ADC_PATH.read_text()), separators=(",", ":"))

    envs = {
        "TAR_PATH": tar_path,
        "SRC_REPO": SRC_REPO,
        "OUT_REPO": OUT_REPO,
        "CLIP_SECONDS": str(args.clip_seconds),
        "WORKERS": str(args.workers),
        "RENDER_WORKERS": str(args.render_workers),
        "MODEL": args.model,
        # 2026-09-05：GCP 两个项目额度耗尽后改走 OpenAI 兼容反代（rightcode）。
        # key 走环境变量透传，**不写进 _code/**，免得进 git 历史。
        # 本机没设这几个变量时保持空字符串，worker 侧默认仍是 adc。
        # FLUSH_EVERY 200 是 8 job × 20 worker 时代为躲 HF 的 128 次/小时提交限流设的，
        # 代价是每个 job 要攒 45 分钟才提交一次，进度完全看不见（还被我误判成故障）。
        # 现在总并发降到 24、每 job 只有 3 worker，60 一批 ≈ 37 次/小时，安全且可见。
        "FLUSH_EVERY": os.environ.get("FLUSH_EVERY", "60"),
        "PROMPT_SET": os.environ.get("PROMPT_SET", "v9"),
        "CAPTION_API": os.environ.get("CAPTION_API", "adc"),
        "PROXY_BASE_URL": os.environ.get("PROXY_BASE_URL", ""),
        "PROXY_API_KEY": os.environ.get("PROXY_API_KEY", ""),
        "ONLY_FAILED": "1" if args.only_failed else "0",
        "SKIP_CLIP_TARS": "1" if args.no_clip_tars else "0",
        "IDLE_TIMEOUT_MIN": str(getattr(args, "idle_timeout_min", 15)),
        "CAPTION_VERSION": getattr(args, "caption_version", "v10"),
        "ROUNDS": str(getattr(args, "rounds", 1)),
        "V9_USE_OCR": "0" if getattr(args, "v9_no_ocr", False) else "1",
        "V9_USE_VAD": "0" if getattr(args, "v9_no_vad", False) else "1",
    }
    # The ADC JSON may be authorized for a different project than the historical
    # caption default. Pass its quota project explicitly before worker imports so
    # Vertex client construction cannot fall back to project-0e21... .
    if isinstance(json.loads(adc), dict):
        adc_project = json.loads(adc).get("quota_project_id")
        if adc_project:
            envs["ADC_PROJECT"] = adc_project
    if getattr(args, "tar_list", None):
        envs["TAR_LIST"] = ",".join(args.tar_list)
    if getattr(args, "wanted_clips_for_shard", None):
        envs["WANTED_CLIPS_JSON"] = json.dumps(args.wanted_clips_for_shard, separators=(",", ":"))
    if not args.no_mount:
        envs["MOUNT_ROOT"] = "/data"
    if args.max_clips:
        envs["MAX_CLIPS"] = str(args.max_clips)
    if getattr(args, "entry", None):
        envs["ENTRY"] = args.entry
    if getattr(args, "bench_only", False):
        envs["BENCH_ONLY"] = "1"
    for kv in (getattr(args, "env", None) or []):
        k, _, v = kv.partition("=")
        envs[k] = v

    cmd = ["hf", "jobs", "run", "--flavor", args.flavor, "--timeout", args.timeout,
           "--name", f"povcap-{Path(tar_path).stem[:28]}", "-d"]
    for k, v in envs.items():
        cmd += ["-e", f"{k}={v}"]
    if not args.no_mount:
        cmd += ["-v", f"hf://datasets/{SRC_REPO}:/data:ro"]

    fd, sp = tempfile.mkstemp(prefix="povsec_", suffix=".env")
    os.close(fd)
    Path(sp).chmod(0o600)
    Path(sp).write_text(f"HF_TOKEN={tok}\nGOOGLE_ADC_JSON={adc}\n")
    # 注意：不能用 `bash -lc`，`hf jobs run` 会把 -l 当成自己的 --label 吃掉
    cmd += ["--secrets-file", sp, IMAGE, "bash", "-c", JOB_SCRIPT]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        Path(sp).unlink(missing_ok=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        print(out)
        sys.exit(f"提交失败: {tar_path}")
    m = re.findall(r"\b[0-9a-f]{24}\b", out)
    job_id = m[-1] if m else out.strip().splitlines()[-1].strip()
    print(f"✓ {Path(tar_path).stem}  job={job_id}")
    JOBS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS_LOG, "a") as f:
        f.write(json.dumps({"job": job_id, "tar": tar_path,
                            "at": time.strftime("%Y-%m-%d %H:%M:%S"), "env": envs}) + "\n")
    return job_id


def load_wanted_clips(clips_file, clip_seconds):
    """读 select/candidates_250.json（或后续 clips_final.json）这类「按 clip 挑选」的
    产物，按 tar 分组出 clip_id 白名单。

    schema（Agent-SELECT 产出，只读不改）：{clips: [{source: {tar, cloud_clip_id,
    frame_index_start, ...}, duration_s, ...}]}。`cloud_clip_id` 形如 "clip_0493"，
    是 SELECT 侧按 `frame_index_start // clip_seconds` 算好的，必须与本次提交实际用的
    `--clip-seconds` 一致，否则白名单里的编号会对不上 cloud_worker 里重新算出的 cid
    （同一个数字指向了不同的时间窗口，caption 出来的内容和挑选时看到的完全不是一回事，
    且不会报错——这类静默错位正是本项目「改共享结构必须先核对」的教训要提防的那一种）。
    这里用抽样重新核算 `frame_index_start // clip_seconds` 做断言，而不是假定一致。
    """
    data = json.loads(Path(clips_file).read_text())
    by_tar: dict[str, set[int]] = {}
    bad = []
    for c in data["clips"]:
        src = c.get("source", {})
        tar = src.get("tar")
        ccid_s = src.get("cloud_clip_id")
        fi_start = src.get("frame_index_start")
        if not tar or not ccid_s or fi_start is None:
            continue
        cid = int(ccid_s.split("_")[-1])
        expect = fi_start // clip_seconds
        if cid != expect:
            bad.append((c.get("clip_id"), cid, expect))
            continue
        by_tar.setdefault(tar, set()).add(cid)
    if bad:
        sys.exit(f"--clips-file 里 {len(bad)} 个 clip 的 cloud_clip_id 与 "
                  f"--clip-seconds {clip_seconds} 算出来的窗口对不上（例：{bad[:3]}）——"
                  f"多半是 clip-seconds 传错了，SELECT 侧是按哪个切的就必须传哪个。")
    return by_tar


def cmd_run(args):
    ensure_repo()
    if not args.no_push_code:
        push_code()
    rows = list_tars()
    wanted_by_tar = None
    if args.clips_file:
        wanted_by_tar = load_wanted_clips(args.clips_file, args.clip_seconds)
        print(f"--clips-file: {len(wanted_by_tar)} 条录制 / "
              f"{sum(len(v) for v in wanted_by_tar.values())} 个目标 clip "
              f"（clip-seconds={args.clip_seconds}）")
    if args.tar:
        targets = [args.tar]
    elif args.day:
        targets = [r["path"] for r in rows if r["day"] == args.day]
    elif wanted_by_tar is not None:
        targets = sorted(wanted_by_tar)          # --clips-file 自己就是一份完整的片源清单
    elif args.all:
        targets = [r["path"] for r in rows]
    else:
        sys.exit("要给 --tar / --day / --all / --clips-file 之一")
    if wanted_by_tar is not None and (args.tar or args.day or args.all):
        before = len(targets)
        targets = [t for t in targets if t in wanted_by_tar]
        print(f"与 --clips-file 取交集: {before} → {len(targets)} 条")
    if not targets:
        sys.exit("没有匹配的 tar")
    if args.skip_done:
        before = len(targets)
        if wanted_by_tar is not None:
            # 窗口模式：按目标 clip 逐条核对，不能用 rec 级 report.json（见
            # done_wanted_clips() 的 docstring，53b730f3 就是这么被永久跳过的）。
            done_tars = done_wanted_clips({t: wanted_by_tar[t] for t in targets})
            targets = [t for t in targets if t not in done_tars]
        else:
            done = done_recordings()
            targets = [t for t in targets if Path(t).stem not in done]
        print(f"跳过已完成的 {before - len(targets)} 条")
    if not targets:
        print("全部已完成，无需提交")
        return

    mins = {r["path"]: r["minutes"] for r in rows}
    k = max(1, min(args.shards, len(targets)))

    if wanted_by_tar is None:
        # 全量 caption 模式：按录制时长均衡（贪心：长的先放最空的分片）——caption 成本
        # 与总帧数成正比，时长就是最好的代理指标。
        weight = lambda p: mins.get(p, 0)                        # noqa: E731
    else:
        # 窗口模式：一个 job 要不要 caption 的量由「目标 clip 数」决定，不是 tar 时长——
        # walk 全 tar 找那 1-3 个窗口本身很快（209MB/s，实测 43 分钟录制约 44s），
        # 真正花钱花时间的是发给 Gemini 的那几个 clip，按它来配平分片才有意义。
        weight = lambda p: len(wanted_by_tar.get(p, ()))          # noqa: E731
    order = sorted(targets, key=lambda p: -weight(p))
    shards = [[] for _ in range(k)]
    load = [0] * k
    for tp in order:
        i = load.index(min(load))
        shards[i].append(tp)
        load[i] += weight(tp)

    unit = "分钟" if wanted_by_tar is None else "个目标clip"
    print(f"{len(targets)} 条录制 / {sum(load)} {unit} → {k} 个分片（flavor={args.flavor}）")
    for i, sh in enumerate(shards):
        if not sh:
            continue
        if wanted_by_tar is None:
            # 实测 ~10 分钟/43 分钟录制，留 3 倍余量 + 30 分钟固定开销
            est_h = max(1, int(load[i] / 43 * 10 * 3 / 60) + 1)
            args.wanted_clips_for_shard = None
        else:
            # walk 開銷（每条 tar ~1 秒/分钟时长）+ caption 開銷（每个目标 clip 约
            # --sec-per-clip 秒 / --workers 并发摊薄）。--sec-per-clip 默认值是留白的
            # 估计旋钮：具体多快取决于最终定的 caption 版本（v9 两遍 vs 主线新定的
            # v10 单遍，二者单 clip 耗时不同），版本一定后应回填真实冒烟数字。
            walk_min = sum(mins.get(t, 0) for t in sh) * (44 / 43 / 60)
            cap_min = load[i] * args.sec_per_clip / max(1, args.workers) / 60
            est_h = max(1, int((walk_min + cap_min) * 3 / 60) + 1)   # 3 倍余量
            args.wanted_clips_for_shard = {Path(t).stem: sorted(wanted_by_tar.get(t, ()))
                                           for t in sh}
        args.timeout = f"{min(est_h, 24)}h"
        args.tar_list = sh
        if wanted_by_tar is None:
            print(f"  分片{i+1}: {len(sh)} 条 / {load[i]} 分钟 / timeout={args.timeout}")
        else:
            print(f"  分片{i+1}: {len(sh)} 条tar / {load[i]} 个目标clip / "
                  f"预估 walk={walk_min:.1f}min + caption={cap_min:.1f}min / "
                  f"timeout={args.timeout}")
        if args.dry_run:
            for x in sh[:3]:
                n = len(wanted_by_tar.get(x, ())) if wanted_by_tar is not None else mins.get(x, 0)
                print(f"      {Path(x).stem[:60]}  ({n}{'clip' if wanted_by_tar is not None else 'min'})")
            if len(sh) > 3:
                print(f"      … 共 {len(sh)} 条")
            continue
        submit(sh[0], args)


def done_recordings():
    """OUT_REPO 上已经有 report.json 的录制，认为已完成。

    只适用于「整条 tar 全量 caption」模式。**窗口模式（--clips-file）不能用这个**——
    report.json 是整条录制级别的，只要这条录制以前被任何一次全量/窗口跑过（哪怕跑的
    是完全不同的几个 clip_id），它就会被判定「已完成」而永远跳过，本轮真正要的
    clip_id 永远产不出来。这是 2026-09-04 铺 200 clip 时实测踩到的：`53b730f3` 早年
    v9smoke 跑过 clip_171/172/173，report.json 早就存在，导致后面三轮都把它整条跳过，
    这次要的 clip_118/125/158 一次都没被尝试过。窗口模式请用 `done_wanted_clips()`。
    """
    try:
        files = api().list_repo_files(OUT_REPO, repo_type="dataset")
    except Exception:                                          # noqa: BLE001
        return set()
    return {f.split("/")[1] for f in files
            if f.startswith("captions/") and f.endswith("/report.json")}


def done_recordings_covered(picked, clip_seconds, min_ratio=0.9):
    """比 done_recordings() 严的「已完成」判定：不只看 report.json 在不在，
    还要求这条录制云端 ok=True 的 clip 数达到应有数量的 min_ratio。

    2026-09-04 实测的坑：4-24/4-25/4-27 有 5 条录制早年被 --max-clips 3 冒烟跑过，
    report.json 早就存在，于是 run-batch --skip-done 把它们整条永久跳过 —— 288 分钟
    素材一个 clip 都没真跑。全量模式下必须按覆盖率判，不能按 report.json 存在判。
    """
    try:
        files = api().list_repo_files(OUT_REPO, repo_type="dataset")
    except Exception:                                          # noqa: BLE001
        return set()
    got = {}
    pat = re.compile(r"^captions/([^/]+)/clip_\d+\.json$")
    for f in files:
        m = pat.match(f)
        if m:
            got[m.group(1)] = got.get(m.group(1), 0) + 1
    done = set()
    for r in picked:
        want = max(1, r["minutes"] * 60 // clip_seconds)
        if got.get(r["rec"], 0) >= want * min_ratio:
            done.add(r["rec"])
    return done


def done_wanted_clips(wanted_by_tar):
    """窗口模式专用的 --skip-done：不看 rec 级 report.json，逐条 tar 核对它这次
    真正要的 clip_id 是否**全部**已经在云端且 ok=True，只有这样才算这条 tar 完成、
    可以跳过。存在 ≠ 成功：仓库是跟别的流水线（v6/v9 batch 等）共用的，`captions/CLAUDE.md`
    记过失败 clip 的 json 也会被上传（ok=False），只看文件存在会把失败 clip 永久卡住
    （`existing_clip_ids()`/`done_caps` 在 cloud_worker 那边也有同样的坑，未改，
    这里至少保证 pipeline.py 侧的分片调度不会把该重跑的 tar 判成「已完成」）。
    """
    try:
        files = set(api().list_repo_files(OUT_REPO, repo_type="dataset"))
    except Exception:                                          # noqa: BLE001
        return set()
    from huggingface_hub import hf_hub_download
    done = set()
    for tp, cids in wanted_by_tar.items():
        rec = Path(tp).stem
        if not all(f"captions/{rec}/clip_{c:04d}.json" in files for c in cids):
            continue
        all_ok = True
        for c in cids:
            try:
                p = hf_hub_download(OUT_REPO, f"captions/{rec}/clip_{c:04d}.json",
                                     repo_type="dataset", token=token())
                if not json.loads(Path(p).read_text()).get("ok"):
                    all_ok = False
                    break
            except Exception:                                  # noqa: BLE001
                all_ok = False
                break
        if all_ok:
            done.add(tp)
    return done


def cmd_bench(args):
    ensure_repo()
    if not args.no_push_code:
        push_code()
    n = getattr(args, "jobs", 1) or 1
    if n > 1:
        # 让所有 job 等到同一个墙上时钟起跑，保证真正并发重叠
        sync = int(time.time()) + 420
        args.env = (args.env or []) + [f"BENCH3_SYNC_EPOCH={sync}"]
        print(f"{n} 个 job 将在 {time.strftime('%H:%M:%S', time.localtime(sync))} 同步起跑")
    base_env = list(args.env or [])
    for i in range(n):
        args.env = base_env + ([f"BENCH3_TAG=job{i+1}of{n}"] if n > 1 else [])
        submit(args.tar, args)


def cmd_stop(args):
    """把还在跑的 job 全砍掉。HF Job 本身跑完就退，这个是给「跑飞了」兜底的。"""
    r = subprocess.run(["hf", "jobs", "ps"], capture_output=True, text=True)
    ids = [l.split("\t")[0] for l in r.stdout.splitlines()[1:] if l.strip() and "\t" in l]
    ids = [i for i in ids if re.fullmatch(r"[0-9a-f]{24}", i)]
    if args.job_ids:
        ids = args.job_ids
    if not ids:
        print("没有正在跑的 job")
    for j in ids:
        rr = subprocess.run(["hf", "jobs", "cancel", j], capture_output=True, text=True)
        print(f"  cancel {j}: {'✓' if rr.returncode == 0 else (rr.stderr or '')[:80]}")

    if not args.keep_batch:
        try:
            import caption_core as cc
            client = cc.make_client()
            live = []
            for b in client.batches.list(config={"page_size": 100}):
                s = str(b.state)
                if not any(k in s for k in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
                    live.append(b)
            if not live:
                print("Vertex batch: 没有未结束的任务")
            for b in live:
                client.batches.cancel(name=b.name)
                print(f"  cancel batch {b.name.split('/')[-1]}")
        except Exception as e:                                # noqa: BLE001
            print(f"  batch 检查失败（可忽略）: {str(e)[:120]}")


def cmd_ps(args):
    """一屏看清：HF job + Vertex batch 的在跑情况和累计花销。"""
    subprocess.run(["hf", "jobs", "ps"] + (["-a"] if args.all else []))
    try:
        import caption_core as cc
        client = cc.make_client()
        rows = []
        for b in client.batches.list(config={"page_size": 50}):
            rows.append((b.name.split("/")[-1], str(b.state).replace("JobState.JOB_STATE_", ""),
                         str(getattr(b, "create_time", ""))[:19]))
        print(f"\nVertex batch（{len(rows)} 个）")
        for i, (n, s, ct) in enumerate(rows[:15]):
            print(f"  {n:24s} {s:12s} {ct}")
        live = [r for r in rows if r[1] in ("PENDING", "QUEUED", "RUNNING")]
        print(f"  未结束: {len(live)}" + ("  ← 还在计费" if live else ""))
    except Exception as e:                                    # noqa: BLE001
        print(f"batch 列表失败: {str(e)[:120]}")


def cmd_gcs_lifecycle(args):
    """给中转桶设生命周期，避免 JSONL/预测结果/inline 兜底上传的图长期堆着收存储费。

    batch_in/ batch_out/ 是 Batch API 那条路的中转前缀；p1/ p2/ 是 v9 2pass
    inline 超限时的 GCS 兜底前缀（caption_v9.upload_gcs 的 gcs_prefix，与
    quality_verify/captions/frozen 冻结脚本一致）——v9 接入前这两个前缀没被
    覆盖到，200 clip 全量跑起来会有 24-33MB/clip 的图片式弃置在桶里不会自动删。
    """
    import json as _j
    prefixes = ["batch_in/", "batch_out/", "p1/", "p2/"]
    rule = {"lifecycle": {"rule": [
        {"action": {"type": "Delete"},
         "condition": {"age": args.days, "matchesPrefix": prefixes}}]}}
    f = Path(tempfile.gettempdir()) / "pov_lifecycle.json"
    f.write_text(_j.dumps(rule))
    r = subprocess.run(["gcloud", "storage", "buckets", "update",
                        f"gs://{args.bucket}", "--lifecycle-file", str(f)],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[:300])
    if r.returncode == 0:
        print(f"✓ gs://{args.bucket} 的 {' '.join(prefixes)} 将在 {args.days} 天后自动删除")


def cmd_status(args):
    cmd = ["hf", "jobs", "ps"] + (["-a"] if args.all else [])
    subprocess.run(cmd)


def cmd_logs(args):
    subprocess.run(["hf", "jobs", "logs", args.job_id])


def cmd_pull(args):
    from huggingface_hub import snapshot_download
    import merge as mg
    rec = args.rec
    pats = [f"captions/{rec}/**"]
    d = snapshot_download(OUT_REPO, repo_type="dataset", allow_patterns=pats,
                          token=token())
    caps = Path(d) / "captions" / rec
    if not caps.exists():
        sys.exit(f"{OUT_REPO} 上没有 captions/{rec}/")
    clips = mg.load_clips(caps)
    mf = caps / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    merged = mg.merge_recording(clips, manifest)
    date = merged["date"] or manifest.get("recording_start_hkt", "")[:10] or "unknown"
    outdir = OUT / date / rec
    rep = mg.write_recording_outputs(outdir, merged)
    print(f"caption → {outdir}")
    print(f"  clip {rep['clip_ok']}/{rep['clip_count']}  segments {rep['segment_count']}  "
          f"估价 ${rep['est_cost_usd']}")
    if rep["clip_failed"]:
        print(f"  失败 clip: {rep['failed_clip_ids']}")

    ids = pick_clip_ids(clips, args)
    if ids:
        print(f"拉 {len(ids)} 个 clip 包（每个约 5MB，本机 ~2.3MB/s）...")
        cdir = outdir / "clips"
        cdir.mkdir(exist_ok=True)
        from huggingface_hub import hf_hub_download
        for cid in ids:
            p = hf_hub_download(OUT_REPO, f"clips/{rec}/clip_{cid:04d}.tar",
                                repo_type="dataset", token=token())
            dst = cdir / f"clip_{cid:04d}.tar"
            if not dst.exists():
                dst.write_bytes(Path(p).read_bytes())
            print(f"  clip_{cid:04d}")
    return outdir


def pick_clip_ids(clips, args):
    ok = [c["clip_id"] for c in clips if c.get("ok")]
    if args.clips:
        return [int(x) for x in args.clips.split(",")]
    if args.sample:
        n = min(args.sample, len(ok))
        if n == 0:
            return []
        step = max(1, len(ok) // n)
        return ok[::step][:n]
    return []


def cmd_merge(args):
    import merge as mg
    day_dir = OUT / args.day
    if not day_dir.exists():
        sys.exit(f"{day_dir} 不存在，先 pull 每条录制")
    ms = []
    for rec_dir in sorted(day_dir.iterdir()):
        p = rec_dir / "captions_full.json"
        if p.is_file():
            ms.append(json.loads(p.read_text()))
    if not ms:
        sys.exit("这一天下面没有 captions_full.json")
    day = mg.merge_day(ms, args.day)
    mg.write_day_outputs(day_dir, day)
    print(f"整天 caption → {day_dir}/day_{args.day}.txt")
    print(f"  {day['recording_count']} 条录制  clip {day['clip_ok']}/{day['clip_count']}  "
          f"segments {day['segment_count']}  估价 ${day['est_cost_usd']}")


def cmd_render(args):
    import render_review
    render_review.render(args)


# ---------------------------------------------------------------- v9 Batch 全量

# Batch 半价（$/1M token）；pass0 语音走在线算全价。
BATCH_IN, BATCH_OUT = 0.375, 1.875
ONLINE_IN, ONLINE_OUT = 0.75, 3.75
# 每 clip 的 token 预估：帧 30 × 1120(HIGH) + prompt/OCR/transcript 约 6.3k；
# crop 保守取 6 × 2240(ULTRA) + pass1 回传。冒烟跑完用实测值替换。
EST = {"p1_in": 39900, "p2_in": 15500, "out": 3000, "p0_in": 2500, "p0_out": 300}
BUDGET_USD = float(os.environ.get("POV_BUDGET_USD", "300"))


def est_clip_cost():
    b = (EST["p1_in"] + EST["p2_in"]) / 1e6 * BATCH_IN + EST["out"] / 1e6 * BATCH_OUT
    o = EST["p0_in"] / 1e6 * ONLINE_IN + EST["p0_out"] / 1e6 * ONLINE_OUT
    return b + o


def select_window(args):
    """从 --from 那天起按时间顺序取够 --hours 小时。录制整条取、不切一半：
    clip_id = frame_index // 30 依赖整条帧序。"""
    rows = [r for r in list_tars() if r["day"] >= args.start]
    rows.sort(key=lambda r: (r["day"], r["path"]))
    want = args.hours * 60
    picked, cum = [], 0
    for r in rows:
        if cum >= want:
            break
        picked.append(r)
        cum += r["minutes"]
    return picked, cum


def cmd_plan_batch(args):
    picked, cum = select_window(args)
    have = gaze_available()
    per = est_clip_cost()
    total_clips = 0
    print(f"{'日期':>10} {'时长':>6} {'GB':>6} {'clip':>5} {'眼动':>4} {'prompt':>10}  录制")
    for r in picked:
        clips = max(1, r["minutes"] * 60 // args.clip_seconds)
        total_clips += clips
        g = r["path"] in have
        print(f"{r['day']:>10} {r['minutes']:>5}m {r['size']/1e9:>6.1f} {clips:>5} "
              f"{'有' if g else '无':>4} {'v9' if g else 'v9_nogaze':>10}  {r['rec'][:46]}")
    nogaze = [r for r in picked if r["path"] not in have]
    print()
    print(f"{len(picked)} 条录制 / {cum} 分钟 ({cum/60:.1f}h) / "
          f"{sum(r['size'] for r in picked)/1e9:.0f} GB / 约 {total_clips} clip")
    if nogaze:
        print(f"无眼动 {len(nogaze)} 条（{sum(r['minutes'] for r in nogaze)} 分钟）"
              f"→ 自动换 v9_nogaze prompt")
    est = total_clips * per
    print()
    print(f"预估 ${per:.4f}/clip × {total_clips} = ${est:.0f}"
          f"   (Batch pass1+pass2 半价 + 在线 pass0)")
    print(f"预算 ${BUDGET_USD:.0f}，占 {est / BUDGET_USD * 100:.0f}%")
    if est > BUDGET_USD:
        print(f"!! 预估超预算 ${est - BUDGET_USD:.0f}，先缩范围或等冒烟实测再决定")
    print("这是按 curated 文字密集场景外推的保守值，真实素材 crop 更少通常更便宜；"
          "先冒烟拿实测单价再放全量。")
    return picked


def cmd_run_batch(args):
    if args.tar:
        picked = [r for r in list_tars() if r["path"] == args.tar or r["rec"] == args.tar]
        if not picked:
            sys.exit(f"没找到 {args.tar}")
    else:
        picked, _ = select_window(args)
    if args.skip_done:
        done = done_recordings_covered(picked, args.clip_seconds)
        before = len(picked)
        skipped = [r["rec"] for r in picked if r["rec"] in done]
        picked = [r for r in picked if r["rec"] not in done]
        print(f"跳过已完成的 {before - len(picked)} 条（按 clip 覆盖率 ≥90% 判定）")
        for rec in skipped:
            print(f"    已完成 {rec[:56]}")
    if not picked:
        print("没有待跑的录制")
        return
    ensure_repo()
    if not args.no_push_code:
        push_code()
    print(f"提交 {len(picked)} 个 job（一条录制一个，flavor={args.flavor}，"
          f"run_id={args.run_id}）")
    if args.dry_run:
        for r in picked:
            print(f"  {r['day']} {r['minutes']:>4}m  {r['rec'][:60]}")
        return
    args.tar_list = None
    args.entry = "archive/batch_worker_v10.py"
    args.caption_version = "v10"
    base_env = list(args.env or [])
    base_flavor = args.flavor
    # 解包阶段是 CPU 瓶颈：2880² 解码 + q80 编码，8 vCPU 上实测 ~10 帧/s，
    # 26,280 帧的那条要 43 分钟。长录制换 32 vCPU 能把这段砍到十几分钟；
    # HF 计费按分钟，多花的是几美元 HF 额度，不占 GCP 的 $300。
    for r in picked:
        big = r["minutes"] >= args.big_minutes
        args.flavor = args.big_flavor if big else base_flavor
        # 并发只在真换了大机型时才翻倍。之前把 big_flavor 改回 cpu-upgrade 却漏了这行，
        # 48 线程挤在 8 vCPU 上把上传打到 0.2 帧/s，4 条大录制被看门狗误杀。
        upw = args.upload_workers * 2 if (big and args.big_flavor != base_flavor) \
            else args.upload_workers
        args.env = base_env + [
            f"RUN_ID={args.run_id}",
            f"UPLOAD_WORKERS={upw}",
            f"SPEECH_WORKERS={args.speech_workers}",
            f"BATCH_TIMEOUT_MIN={args.batch_timeout_min}",
            f"PROMPT_VARIANT={args.prompt_variant}",
        ]
        submit(r["path"], args)
    args.flavor = base_flavor


def cmd_cost(args):
    """汇总真实用量。仓库是跟其它任务共用的，所以默认只算本批窗口内、
    且确实由 v9 batch 流水线产出的部分 —— 否则预算闸门会被别人的花销污染。"""
    from huggingface_hub import hf_hub_download
    a = api()
    files = a.list_repo_files(OUT_REPO, repo_type="dataset")
    mine = {r["rec"] for r, in [(x,) for x in select_window(args)[0]]}
    reports = [f for f in files if f.startswith("captions/") and f.endswith("/report.json")]

    def get(f):
        try:
            p = hf_hub_download(OUT_REPO, f, repo_type="dataset", token=token())
            return f.split("/")[1], json.loads(Path(p).read_text())
        except Exception:                                     # noqa: BLE001
            return None, None
    with ThreadPoolExecutor(16) as ex:
        rows = [(r, d) for r, d in ex.map(get, reports) if d]

    in_win = [(r, d) for r, d in rows if r in mine]
    out_win = [(r, d) for r, d in rows if r not in mine]
    sel = rows if args.all_repo else in_win

    total = sum((d.get("est_cost_usd") or 0) for _, d in sel)
    clips = sum((d.get("clip_count") or 0) for _, d in sel)
    ok = sum((d.get("clip_ok") or 0) for _, d in sel)
    segs = sum((d.get("segment_count") or 0) for _, d in sel)
    pii = {}
    for _, d in sel:
        for k, v in (d.get("pii_audit") or {}).items():
            pii[k] = pii.get(k, 0) + v

    print(f"{'录制':<50} {'clip':>12} {'segs':>7} {'$':>8}")
    for r, d in sorted(sel, key=lambda x: -(x[1].get("est_cost_usd") or 0)):
        print(f"{r[:48]:<50} "
              f"{str(d.get('clip_ok', 0)) + '/' + str(d.get('clip_count', 0)):>12} "
              f"{d.get('segment_count', 0):>7} {d.get('est_cost_usd', 0):>8.2f}")
    print()
    scope = "整个仓库" if args.all_repo else f"本批窗口({args.start} 起 {args.hours}h)"
    print(f"[{scope}] {len(sel)}/{len(mine)} 条录制  clip {ok}/{clips}  segments {segs}")
    print(f"PII: {pii}")
    print(f"累计花销 ${total:.2f} / 预算 ${BUDGET_USD:.0f}  "
          f"({total / BUDGET_USD * 100:.0f}%，剩 ${BUDGET_USD - total:.2f})")
    if ok:
        print(f"实测单价 ${total / ok:.4f}/clip")
    if out_win and not args.all_repo:
        other = sum((d.get("est_cost_usd") or 0) for _, d in out_win)
        print(f"（仓库里另有 {len(out_win)} 条不在本批窗口的录制，${other:.2f}，"
              f"是其它任务的产物，未计入；--all-repo 可一并显示）")
    if total > BUDGET_USD * 0.9:
        print("!! 已用掉 90% 预算，该停了：python pipeline.py stop")


def cmd_audit(args):
    """本批窗口的交付审计：成功率 / 版本混杂 / PII / 缺口。

    仓库是跟其它任务共用的，同一条 `captions/<rec>/` 下可能混进别的流水线写的
    clip（实测另一会话把 v6 单遍的 clip 写进了本批两条录制）。这些 clip 没有
    `screen_text_detail`、prompt 也不同，混在 v9 数据集里必须能查出来。
    """
    from huggingface_hub import hf_hub_download
    a = api()
    files = a.list_repo_files(OUT_REPO, repo_type="dataset")
    picked, _ = select_window(args)
    want = {r["rec"]: r for r in picked}

    by_rec = {}
    for f in files:
        if f.startswith("captions/") and "/clip_" in f and f.endswith(".json"):
            rec = f.split("/")[1]
            if rec in want:
                by_rec.setdefault(rec, []).append(f)

    def load(f):
        try:
            p = hf_hub_download(OUT_REPO, f, repo_type="dataset", token=token())
            return f, json.loads(Path(p).read_text())
        except Exception:                                     # noqa: BLE001
            return f, None

    allf = [f for v in by_rec.values() for f in v]
    print(f"拉取 {len(allf)} 个 clip json 做审计…")
    got = {}
    with ThreadPoolExecutor(24) as ex:
        for f, d in ex.map(load, allf):
            if d:
                got[f] = d

    n_v9 = n_other = n_bad = 0
    mixed = {}
    pii_rows = []
    print(f"\n{'录制':<46} {'时长':>5} {'clip':>11} {'v9':>5} {'异版':>5} {'失败':>5} {'PII':>5}")
    for rec, r in sorted(want.items(), key=lambda kv: kv[1]["day"]):
        fs = by_rec.get(rec, [])
        exp = max(1, r["minutes"] * 60 // args.clip_seconds)
        v9 = other = bad = 0
        for f in fs:
            d = got.get(f) or {}
            if d.get("mode") == "v9_2pass_batch" or d.get("pipeline") == "v9-2pass":
                v9 += 1
            else:
                other += 1
                mixed.setdefault(rec, []).append(Path(f).stem)
            if not d.get("ok"):
                bad += 1
        n_v9 += v9; n_other += other; n_bad += bad
        rep = None
        try:
            if f"captions/{rec}/report.json" in files:
                rep = json.loads(open(hf_hub_download(
                    OUT_REPO, f"captions/{rec}/report.json", repo_type="dataset",
                    token=token())).read())
        except Exception:                                     # noqa: BLE001
            pass
        pii = (rep or {}).get("pii_audit", {})
        if rep:
            pii_rows.append((rec, pii))
        flag = "" if len(fs) >= exp else f"  缺 {exp - len(fs)}"
        print(f"{rec[:44]:<46} {r['minutes']:>4}m {len(fs):>4}/{exp:<6} {v9:>5} "
              f"{other:>5} {bad:>5} {pii.get('total_hits', 0):>5}{flag}")

    print(f"\n合计: v9 clip {n_v9}  异版 {n_other}  失败 {n_bad}")
    if mixed:
        print("\n版本混杂（非 v9_2pass_batch，多半是别的流水线写进同一路径的）:")
        for rec, cl in mixed.items():
            print(f"  {rec[:52]}: {len(cl)} 个 → {', '.join(sorted(cl)[:6])}")
        print("  修法: 删掉这些 clip json 后对该录制重跑，只会补这几个 clip")
    agg = {}
    for _, p in pii_rows:
        for k, v in p.items():
            agg[k] = agg.get(k, 0) + v
    print(f"\nPII 汇总（{len(pii_rows)} 条录制）: {agg}")
    if agg.get("total_hits"):
        print("  → 公开前需过一道正则脱敏后处理；这是文本层面的，不用重跑模型")


def bad_clips(files, rec, tok):
    """返回该录制里「不是本流水线产的」或「失败的」clip 路径。

    仓库是跟其它任务共用的：实测另一会话把 v6 单遍的 clip、以及 v9 **在线**版
    撞 429 失败的 clip，都写进了同一个 captions/<rec>/ 下。我的 worker 会把它们
    当成 done_caps 跳过，于是坏 clip 就永久留在交付里。
    """
    from huggingface_hub import hf_hub_download
    out = []
    cand = [f for f in files if f.startswith(f"captions/{rec}/clip_") and f.endswith(".json")]

    def chk(f):
        try:
            d = json.loads(Path(hf_hub_download(OUT_REPO, f, repo_type="dataset",
                                                token=tok)).read_text())
        except Exception:                                     # noqa: BLE001
            return f, "拉取失败"
        # 两条 v9 流水线的标记不同：batch 版写 mode，在线版写 pipeline
        if d.get("mode") != "v9_2pass_batch" and d.get("pipeline") != "v9-2pass":
            return f, f"异版({d.get('mode') or d.get('pipeline')})"
        if not d.get("ok"):
            return f, "失败"
        return f, None
    with ThreadPoolExecutor(24) as ex:
        for f, why in ex.map(chk, cand):
            if why:
                out.append((f, why))
    return out


def cmd_repair(args):
    """把交付里「异版 / 失败」的 clip 删掉并重跑对应录制。

    删了之后 worker 的 done_caps 就不再包含它们，process_one 只会补跑这几个 clip，
    整条录制不会重算。report.json 也一并删，否则 --skip-done 会跳过整条。
    """
    a = api()
    tok = token()
    files = a.list_repo_files(OUT_REPO, repo_type="dataset")
    picked, _ = select_window(args)
    todo = {}
    for r in picked:
        rec = r["rec"]
        bad = bad_clips(files, rec, tok)
        if bad:
            todo[rec] = (r, bad)
            print(f"{rec[:48]}: {len(bad)} 个待修 → "
                  f"{', '.join(sorted({w for _, w in bad}))}")
    if not todo:
        print("没有需要修的 clip")
        return
    n = sum(len(v[1]) for v in todo.values())
    print(f"\n共 {len(todo)} 条录制 / {n} 个 clip")
    if args.dry_run:
        print("--dry-run，未改动")
        return
    from huggingface_hub import CommitOperationDelete
    for rec, (r, bad) in todo.items():
        ops = [CommitOperationDelete(path_in_repo=f) for f, _ in bad]
        if f"captions/{rec}/report.json" in files:
            ops.append(CommitOperationDelete(path_in_repo=f"captions/{rec}/report.json"))
        a.create_commit(repo_id=OUT_REPO, repo_type="dataset", operations=ops,
                        commit_message=f"repair: 删掉 {len(bad)} 个异版/失败 clip 以便重跑 {rec}")
        print(f"  已清理 {rec[:48]}")
    if args.no_submit:
        print("\n--no-submit：只清理未提交，之后跑 run-batch 会自动补上")
        return
    args.tar_list = None
    args.entry = "cloud_worker.py"
    args.caption_version = "v10"
    for rec, (r, _) in todo.items():
        args.env = [f"RUN_ID={args.run_id}", f"UPLOAD_WORKERS={args.upload_workers}",
                    f"SPEECH_WORKERS={args.speech_workers}",
                    f"BATCH_TIMEOUT_MIN={args.batch_timeout_min}",
                    f"PROMPT_VARIANT={args.prompt_variant}"]
        submit(r["path"], args)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("days").set_defaults(fn=cmd_days)

    s = sub.add_parser("scan")
    s.add_argument("--month")
    s.add_argument("--day")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--min-minutes", type=int, default=30)
    s.add_argument("--max-minutes", type=int, default=100)
    s.add_argument("--gaze-only", action="store_true", default=True)
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("run")
    s.add_argument("--tar")
    s.add_argument("--day")
    s.add_argument("--all", action="store_true", help="全量 533 条")
    s.add_argument("--skip-done", action="store_true", default=True,
                   help="跳过 OUT_REPO 上已有 report.json 的录制（默认开）")
    # 原来只有 default=True 没有关闭开关，想对已有 report.json 的录制补跑失败 clip
    # （--only-failed）时会被整条跳过，只能改代码。2026-09-05 加。
    s.add_argument("--no-skip-done", dest="skip_done", action="store_false",
                   help="不跳过已有 report.json 的录制（配合 --only-failed 补跑）")
    s.add_argument("--clip-seconds", type=int, default=30)
    s.add_argument("--max-clips", type=int)
    s.add_argument("--workers", type=int, default=24)
    s.add_argument("--render-workers", type=int, default=8)
    s.add_argument("--model", default="gemini-3.7-flash")
    s.add_argument("--flavor", default=FLAVOR)
    s.add_argument("--timeout", default="4h")
    s.add_argument("--only-failed", action="store_true")
    s.add_argument("--no-clip-tars", action="store_true")
    s.add_argument("--no-mount", action="store_true")
    s.add_argument("--no-push-code", action="store_true")
    s.add_argument("--idle-timeout-min", type=float, default=15,
                   help="连续多少分钟没进展就让 job 自杀（默认 15）")
    s.add_argument("--shards", type=int, default=1, help="把待处理录制分成几个 job 并行")
    s.add_argument("--dry-run", action="store_true", help="只打印分片方案，不提交")
    s.add_argument("--caption-version", choices=("v6", "v9", "v10"), default="v10",
                   help="v10=单遍整帧+zoom均MEDIUM(默认)；v9=两遍ULTRA_HIGH crop backup；v6=旧单遍")
    s.add_argument("--rounds", type=int, default=5,
                   help="v9 专用：payload 大走 GCS 有随机掉线，按轮退避并发重跑未完成的 clip"
                        "（默认 5，对齐主线 run_caption_2pass.py；v6 忽略此参数）")
    s.add_argument("--v9-no-ocr", action="store_true", help="v9 消融：不给 OCR 文本块")
    s.add_argument("--v9-no-vad", action="store_true", help="v9 消融：跳过 pass0 语音对齐")
    s.add_argument("--clips-file",
                   help="按 clip 挑选的产物（如 quality_verify/select/candidates_250.json），"
                        "给了就只 caption 里面列出的窗口：targets 自动收窄成这些 clip 所在的"
                        "tar，分片按「目标 clip 数」而不是 tar 时长配平，且 caption 阶段只对"
                        "白名单里的 clip_id 发请求（tar 本身仍整条 walk，I/O 躲不掉但很快）")
    s.add_argument("--sec-per-clip", type=float, default=20.0,
                   help="--clips-file 模式下预估单 clip 墙钟秒数，只用来估分片时长，"
                        "不影响实际调用；不同 caption 版本/并发下应回填真实冒烟数字再用")
    s.set_defaults(fn=cmd_run, tar_list=None)

    s = sub.add_parser("bench")
    s.add_argument("--tar", default="aria/2026-05-18/5-18_hkt2130-2214_43m_"
                                    "53b730f3-2fe6-402b-a233-d9849ff782b0.tar")
    s.add_argument("--clip-seconds", type=int, default=30)
    s.add_argument("--max-clips", type=int)
    s.add_argument("--workers", type=int, default=24)
    s.add_argument("--render-workers", type=int, default=8)
    s.add_argument("--model", default="gemini-3.7-flash")
    s.add_argument("--flavor", default=FLAVOR)
    s.add_argument("--timeout", default="2h")
    s.add_argument("--only-failed", action="store_true")
    s.add_argument("--no-clip-tars", action="store_true", default=True)
    s.add_argument("--no-mount", action="store_true")
    s.add_argument("--no-push-code", action="store_true")
    s.add_argument("--bench-only", action="store_true")
    s.add_argument("--entry", default="archive/bench_worker.py")
    s.add_argument("--env", action="append", help="额外环境变量 K=V，可重复")
    s.add_argument("--jobs", type=int, default=1, help="同时起几个一模一样的 job")
    s.set_defaults(fn=cmd_bench, day=None)

    for name, fn in (("plan-batch", cmd_plan_batch), ("run-batch", cmd_run_batch)):
        s = sub.add_parser(name, help="v9 双流程 + Vertex Batch 的选片/提交")
        s.add_argument("--from", dest="start", default="2026-04-15",
                       help="从哪一天起（含），默认 2026-04-15")
        s.add_argument("--hours", type=float, default=52.7,
                       help="往后取够多少小时素材（录制整条取，默认 52.7）")
        s.add_argument("--tar", help="只跑这一条（冒烟用，给 tar 路径或 rec 名）")
        s.add_argument("--clip-seconds", type=int, default=30)
        s.add_argument("--run-id", default=time.strftime("%Y%m%d"),
                       help="本次跑的标识，决定 GCS 路径与断点续跑归属")
        s.add_argument("--max-clips", type=int, help="每条只跑前 N 个 clip（冒烟用）")
        s.add_argument("--upload-workers", type=int, default=24)
        s.add_argument("--speech-workers", type=int, default=12)
        s.add_argument("--batch-timeout-min", type=float, default=240,
                       help="单个 batch 轮询硬上限，超时自动 cancel 止损")
        s.add_argument("--prompt-variant", default="auto",
                       choices=("auto", "v9", "v9_nogaze"),
                       help="auto=按有无 MPS 眼动自动选（默认）")
        s.add_argument("--workers", type=int, default=24)
        s.add_argument("--render-workers", type=int, default=8)
        s.add_argument("--model", default="gemini-3.7-flash")
        s.add_argument("--flavor", default=FLAVOR)
        s.add_argument("--big-flavor", default=FLAVOR,
                       help="长录制用的机型。cpu-performance(32 vCPU) 能把解包从 43 分钟"
                            "压到十几分钟，但它 $1.90/h 是 cpu-upgrade 的 63 倍 —— "
                            "2026-09-03 就是 5 条长录制并跑把 HF 预付余额烧穿、"
                            "24 个 job 被平台统一 CANCELED。默认与普通机型一致，"
                            "确认余额充足再显式指定")
        s.add_argument("--big-minutes", type=int, default=200,
                       help="超过多少分钟算长录制，用 --big-flavor（默认 200）")
        s.add_argument("--timeout", default="8h")
        s.add_argument("--skip-done", action="store_true", default=True)
        s.add_argument("--no-clip-tars", action="store_true", default=True,
                       help="不生成/上传 clip 帧包（默认开，本批只要 caption）")
        s.add_argument("--no-mount", action="store_true")
        s.add_argument("--no-push-code", action="store_true")
        s.add_argument("--only-failed", action="store_true")
        s.add_argument("--idle-timeout-min", type=float, default=25,
                       help="等 batch 时每轮都会打点，25 分钟无进展才判挂死")
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--env", action="append", help="额外环境变量 K=V，可重复")
        s.set_defaults(fn=fn, day=None, all=False, tar_list=None, shards=1)

    s = sub.add_parser("repair", help="删掉异版/失败的 clip 并重跑对应录制")
    s.add_argument("--from", dest="start", default="2026-04-15")
    s.add_argument("--hours", type=float, default=52.7)
    s.add_argument("--clip-seconds", type=int, default=30)
    s.add_argument("--run-id", default=time.strftime("%Y%m%d"))
    s.add_argument("--upload-workers", type=int, default=24)
    s.add_argument("--speech-workers", type=int, default=12)
    s.add_argument("--batch-timeout-min", type=float, default=240)
    s.add_argument("--prompt-variant", default="auto")
    s.add_argument("--workers", type=int, default=24)
    s.add_argument("--render-workers", type=int, default=8)
    s.add_argument("--model", default="gemini-3.7-flash")
    s.add_argument("--flavor", default=FLAVOR)
    s.add_argument("--timeout", default="8h")
    s.add_argument("--max-clips", type=int)
    s.add_argument("--no-mount", action="store_true")
    s.add_argument("--only-failed", action="store_true")
    s.add_argument("--no-clip-tars", action="store_true", default=True)
    s.add_argument("--idle-timeout-min", type=float, default=25)
    s.add_argument("--no-submit", action="store_true", help="只清理，不提交 job")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_repair, env=None, day=None, all=False, shards=1,
                   skip_done=True, no_push_code=True, bench_only=False,
                   rounds=1, v9_no_ocr=False, v9_no_vad=False)

    s = sub.add_parser("audit", help="交付审计：成功率/版本混杂/PII/缺口")
    s.add_argument("--from", dest="start", default="2026-04-15")
    s.add_argument("--hours", type=float, default=52.7)
    s.add_argument("--clip-seconds", type=int, default=30)
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("cost", help="汇总真实花销，对着 $300 预算看还剩多少")
    s.add_argument("--from", dest="start", default="2026-04-15")
    s.add_argument("--hours", type=float, default=52.7)
    s.add_argument("--clip-seconds", type=int, default=30)
    s.add_argument("--all-repo", action="store_true",
                   help="连仓库里其它任务的产物一起算（默认只算本批窗口）")
    s.set_defaults(fn=cmd_cost)

    s = sub.add_parser("status")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("ps", help="HF job + Vertex batch 一屏总览")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_ps)

    s = sub.add_parser("stop", help="砍掉所有在跑的 job 和未结束的 batch")
    s.add_argument("job_ids", nargs="*")
    s.add_argument("--keep-batch", action="store_true", help="只砍 HF job，不动 Vertex batch")
    s.set_defaults(fn=cmd_stop)

    s = sub.add_parser("gcs-lifecycle", help="给 batch 中转桶设自动删除")
    s.add_argument("--bucket", default="hku-capstone-caption-frames")
    s.add_argument("--days", type=int, default=7)
    s.set_defaults(fn=cmd_gcs_lifecycle)

    s = sub.add_parser("logs")
    s.add_argument("job_id")
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("pull")
    s.add_argument("--rec", required=True)
    s.add_argument("--sample", type=int, default=0)
    s.add_argument("--clips")
    s.set_defaults(fn=cmd_pull)

    s = sub.add_parser("merge")
    s.add_argument("--day", required=True)
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("render")
    s.add_argument("--rec", required=True)
    s.add_argument("--date")
    s.add_argument("--sample", type=int, default=8)
    s.add_argument("--clips")
    s.add_argument("--output")
    s.set_defaults(fn=cmd_render)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
