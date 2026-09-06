#!/usr/bin/env python3
"""两个 ADC 项目并行跑 caption 时的监控 + 预算闸门。

一次 tick 做四件事：
  1. HF job 状态汇总（按波次分）
  2. 两个项目的 Vertex batch 状态汇总（只认本流水线的 batch，靠 dest 前缀过滤，
     不去碰同项目里别的任务的作业）
  3. 从 HF 上新增的 clip_*.json 增量累加真实 token 花销（Vertex 按 token 计费，
     这个数就是账单口径），本地缓存已数过的文件名，避免每轮重下上万个文件
  4. 任一项目累计花销触到 --cap 就只停那一个项目的 job + batch，另一个继续

用法: python watch_dual.py            一次 tick
      python watch_dual.py --stop-wave A   手动停某一波
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "_watch" / "cost_state.json"
OUT_REPO = "mmm8383/pov-captions"

# 波次定义：日期前缀 → 用哪个 ADC / 桶 / run_id
WAVES = {
    "A": {
        "project": "project-1a1608e8-53ad-4178-bf7",
        "cred": os.path.expanduser("~/.config/gcloud/adc-profiles/b34927743.json"),
        "run_ids": ["20260904-bf7-r3", "20260904-bf7-r2"],
        "label": "A(bf7) 前50h",
    },
    "B": {
        "project": "project-0e21c343-2d89-405c-9d5",
        "cred": os.path.expanduser("~/.config/gcloud/adc-profiles/meng15210579725.json"),
        "run_ids": ["20260904-0e21-w2", "20260905-0e21-f50"],
        "label": "B(0e21) 前50h一半 + 5-03→5-11",
    },
}


def rec_day(rec: str) -> str:
    """4-27_hkt1944-... → 2026-04-27"""
    m = re.match(r"(\d+)-(\d+)_", rec)
    return f"2026-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if m else "9999"


# 2026-09-05：用户要求两个账号合力先跑完前 50 小时，于是把 bf7 上「已完成行 ≤5」的
# 8 条 4 月录制搬到了 0e21（run_id 20260905-0e21-f50）。搬过之后波次就不再是
# 「A=4 月 / B=5 月」这条日期线了，必须按这份显式名单归属，否则这 8 条的花销会
# 被记到 A 头上、两边的 $250 闸门都算错。
MOVED_TO_B = {
    "4-15_hkt1323-1709_226m_544095a4-a5a0-5eda-b82e-99d1cdcaa61b",
    "4-22_hkt0821-1124_183m_4966a1d1-de55-4aac-9709-f82d7ca2fa43",
    "4-30_hkt1204-1623_258m_6d2452e7-8da7-44c6-9987-3f715b4ac1fe",
    "4-23_hkt1900-2106_125m_f6f92591-e213-4871-9269-e9e62ceefe12",
    "4-24_hkt1012-1142_89m_98cc61cf-ab34-4adb-a985-c5df54af9952",
    "4-25_hkt1942-2111_88m_d6699610-e76a-4c36-8635-977b34610dfd",
    "4-27_hkt1819-1940_81m_ecab4eec-fbea-454d-ae92-ff89650d7654",
    "4-27_hkt1944-2013_29m_bcbc8cc8-7df4-45da-b545-23139fb1b55e",
}


def wave_of(rec: str) -> str:
    # hf_jobs() 拿到的 rec 是从 job 名 `povcap-<rec[:28]>` 反解的**28 字符前缀**，
    # 和 MOVED_TO_B 里的全名对不上，所以两头都要按前缀比（2026-09-05 踩过：
    # 搬到 B 的 8 条被全记到 A 波，job 数显示 A19/B0）。
    if any(rec.startswith(m[:28]) or m.startswith(rec[:28]) for m in MOVED_TO_B):
        return "B"
    return "A" if rec_day(rec) < "2026-05-01" else "B"


# ------------------------------------------------------------------ 花销

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"counted": {}, "totals": {}}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, separators=(",", ":")))


def scan_cost(st, quiet=False):
    """增量把新出现的 clip json 的 usage 累加进 totals。counted 记 rec → 已数过的 clip_id 列表。"""
    from huggingface_hub import HfApi, hf_hub_download, get_token
    tok = get_token()
    api = HfApi(token=tok)
    files = api.list_repo_files(OUT_REPO, repo_type="dataset")
    pat = re.compile(r"^captions/([^/]+)/clip_(\d+)\.json$")
    want = []
    for f in files:
        m = pat.match(f)
        if not m:
            continue
        rec, cid = m.group(1), m.group(2)
        if cid in st["counted"].get(rec, []):
            continue
        want.append((f, rec, cid))

    def get(item):
        f, rec, cid = item
        try:
            d = json.loads(Path(hf_hub_download(OUT_REPO, f, repo_type="dataset",
                                                token=tok)).read_text())
            return rec, cid, d
        except Exception:                                       # noqa: BLE001
            return rec, cid, None

    new = 0
    if want:
        with ThreadPoolExecutor(24) as ex:
            for rec, cid, d in ex.map(get, want):
                if d is None:
                    continue
                st["counted"].setdefault(rec, []).append(cid)
                t = st["totals"].setdefault(rec, {"clips": 0, "ok": 0, "usd": 0.0,
                                                  "bin": 0, "bout": 0})
                t["clips"] += 1
                t["ok"] += 1 if d.get("ok") else 0
                t["usd"] += float(d.get("est_cost_usd") or 0)
                u = d.get("usage") or {}
                t["bin"] += int(u.get("batch_in") or 0)
                t["bout"] += int(u.get("batch_out") or 0)
                new += 1
    save_state(st)
    if not quiet:
        print(f"新增 clip {new} 个（本轮扫描 {len(files)} 个仓库文件）")
    return st


def cost_by_wave(st):
    out = {"A": {"usd": 0.0, "clips": 0, "ok": 0}, "B": {"usd": 0.0, "clips": 0, "ok": 0}}
    for rec, t in st["totals"].items():
        w = out[wave_of(rec)]
        w["usd"] += t["usd"]
        w["clips"] += t["clips"]
        w["ok"] += t["ok"]
    return out


# ------------------------------------------------------------------ HF job

def hf_jobs():
    r = subprocess.run(["hf", "jobs", "ps"], capture_output=True, text=True,
                       env={**os.environ, "PATH": os.environ["PATH"] + ":" + os.path.expanduser("~/.local/bin")})
    rows = []
    for ln in r.stdout.splitlines()[1:]:
        f = ln.split("\t")
        if len(f) >= 7 and f[1].startswith("povcap-"):
            rows.append({"id": f[0], "rec": f[1][len("povcap-"):], "status": f[5], "rt": f[6]})
    return rows


# ------------------------------------------------------------------ Vertex batch

def vertex_batches(wave):
    """只返回本流水线的 batch：dest 里带本波 run_id 的。"""
    w = WAVES[wave]
    env = dict(os.environ)
    env["GOOGLE_APPLICATION_CREDENTIALS"] = w["cred"]
    env["GOOGLE_CLOUD_PROJECT"] = w["project"]
    code = f'''
import os, json
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = {w["cred"]!r}
os.environ["GOOGLE_CLOUD_PROJECT"] = {w["project"]!r}
from google import genai
c = genai.Client(vertexai=True, project={w["project"]!r}, location="global")
run_ids = {w["run_ids"]!r}
out = []
for b in c.batches.list(config={{"page_size": 200}}):
    dest = ""
    try: dest = (b.dest.gcs_uri or "") if b.dest else ""
    except Exception: pass
    if run_ids and not any(r in dest for r in run_ids): continue
    out.append({{"name": b.name.split("/")[-1],
                 "state": str(b.state).split(".")[-1].replace("JOB_STATE_", ""),
                 "dest": dest}})
print(json.dumps(out))
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:                                           # noqa: BLE001
        return []


def stop_wave(wave, reason=""):
    """只停这一波：砍它的 HF job + 它自己 run_id 下未结束的 batch。别的任务不碰。"""
    w = WAVES[wave]
    print(f"!! 停波次 {w['label']}  {reason}")
    recs = [j for j in hf_jobs() if wave_of(j["rec"]) == wave and j["status"] == "RUNNING"]
    for j in recs:
        subprocess.run(["hf", "jobs", "cancel", j["id"]], capture_output=True, text=True,
                       env={**os.environ, "PATH": os.environ["PATH"] + ":" + os.path.expanduser("~/.local/bin")})
        print(f"   cancel job {j['id']} {j['rec'][:40]}")
    live = [b for b in vertex_batches(wave)
            if b["state"] in ("PENDING", "QUEUED", "RUNNING", "PAUSED")]
    if live:
        env = dict(os.environ)
        env["GOOGLE_APPLICATION_CREDENTIALS"] = w["cred"]
        env["GOOGLE_CLOUD_PROJECT"] = w["project"]
        code = f'''
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = {w["cred"]!r}
os.environ["GOOGLE_CLOUD_PROJECT"] = {w["project"]!r}
from google import genai
c = genai.Client(vertexai=True, project={w["project"]!r}, location="global")
for n in {[b["name"] for b in live]!r}:
    try:
        c.batches.cancel(name=f"projects/{w['project']}/locations/global/batchPredictionJobs/{{n}}")
        print("cancel batch", n)
    except Exception as e:
        print("cancel 失败", n, str(e)[:80])
'''
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        print("   " + (r.stdout or r.stderr).strip().replace("\n", "\n   "))


# ------------------------------------------------------------------ tick

def tick(cap):
    print(f"\n===== {time.strftime('%H:%M:%S')} =====")
    jobs = hf_jobs()
    st = scan_cost(load_state())
    cost = cost_by_wave(st)
    tripped = []
    for wv in ("A", "B"):
        w = WAVES[wv]
        js = [j for j in jobs if wave_of(j["rec"]) == wv]
        run = sum(1 for j in js if j["status"] == "RUNNING")
        bs = vertex_batches(wv)
        bc = {}
        for b in bs:
            bc[b["state"]] = bc.get(b["state"], 0) + 1
        c = cost[wv]
        print(f"{w['label']:<22} job RUNNING {run:>2}  batch {bc}")
        print(f"{'':22} clip {c['clips']:>5}（ok {c['ok']}） 花销 ${c['usd']:.2f} / ${cap:.0f}"
              f"  剩 ${cap - c['usd']:.2f}")
        if c["usd"] >= cap:
            tripped.append(wv)
    for wv in tripped:
        stop_wave(wv, f"累计 ${cost[wv]['usd']:.2f} ≥ 闸门 ${cap:.0f}")
    return cost, tripped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=250.0)
    ap.add_argument("--stop-wave", choices=("A", "B"))
    a = ap.parse_args()
    if a.stop_wave:
        stop_wave(a.stop_wave, "手动")
    else:
        tick(a.cap)
