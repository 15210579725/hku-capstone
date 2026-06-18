# -*- coding: utf-8 -*-
"""ego 感知层: 把真实 L1/L2 第一人称感知流, 按决策步时间窗回放给 agent.
防泄露核心: 只取"我看到/我听到"(环境), 剔除"我做/我说"(那是该步要预测的答案)."""
import re

DATA = "/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA"
LINE = re.compile(r"\[(\d\d):(\d\d):(\d\d)\s*->\s*(\d\d):(\d\d):(\d\d)\]\s*(.+)")

def _sec(h, m, s): return int(h)*3600 + int(m)*60 + int(s)

def load_layer(day, layer="L1"):
    rows = []
    for ln in open(f"{DATA}/{layer}/{day}/events.txt", encoding="utf-8"):
        m = LINE.match(ln.strip())
        if not m: continue
        g = m.groups()
        rows.append({"start": _sec(*g[0:3]), "end": _sec(*g[3:6]), "desc": g[6].strip()})
    return rows

def is_percept(desc):  # 只有这两类是"她看到/听到的环境"
    return desc.startswith("我看到") or desc.startswith("我听到")

def perceive_window(rows, t_start, t_end, max_lines=40, dedup=True):
    """取 [t_start, t_end) 内的真实感知行(仅 我看到/我听到), 防泄露剔除自身动作.
    用于某决策步: t_start=该步开始, t_end=该步开始(只看截至此刻) 或 上一步末->本步开始."""
    seen = set(); out = []
    for r in rows:
        if r["start"] < t_start or r["start"] >= t_end: continue
        if not is_percept(r["desc"]): continue          # 剔除 我做/我说 → 防泄露
        key = r["desc"]
        if dedup and key in seen: continue
        seen.add(key); out.append(r["desc"])
    if len(out) > max_lines:                              # 太长则首尾采样保留信息
        head = out[:max_lines*2//3]; tail = out[-max_lines//3:]
        out = head + ["…(中间略)…"] + tail
    return out

def build_perception_for_steps(day, steps, look_back_to_prev=True):
    """给每个决策步附上真实感知窗口 real_perception(list[str])."""
    rows = load_layer(day, "L1")
    enriched = []
    for k, s in enumerate(steps):
        g = LINE.match(f"[{s['time_start']} -> {s['time_end']}] x")
        ts = _sec(*s["time_start"].split(":"))
        # 窗口起点: 上一步结束(承接现场) 或 本步往前推 90s
        if look_back_to_prev and k > 0:
            prev_end = _sec(*steps[k-1]["time_end"].split(":"))
            win_start = min(prev_end, ts)
        else:
            win_start = ts - 90
        # 窗口终点: 本步开始时刻(只给"做动作之前"看到/听到的) + 含本步触发事件窗口的感知
        win_end = _sec(*s["time_end"].split(":"))   # 含本步窗口内的他人言行(背景回放), 但已剔除Lucia自身动作
        perc = perceive_window(rows, win_start, win_end)
        enriched.append({**s, "real_perception": perc})
    return enriched

if __name__ == "__main__":
    import json, sys
    day = sys.argv[1] if len(sys.argv)>1 else "day1"
    steps = json.load(open(f"world/decision_steps_{day}.json", encoding="utf-8"))
    en = build_perception_for_steps(day, steps)
    json.dump(en, open(f"world/steps_perc_{day}.json","w"), ensure_ascii=False, indent=2)
    s6 = en[6] if len(en)>6 else en[-1]
    print(f"{day}: {len(en)} steps enriched -> world/steps_perc_{day}.json")
    print(f"\n--- step{s6['step_id']} 真实感知窗口 ({len(s6['real_perception'])} 行, 防泄露) ---")
    for l in s6["real_perception"][:18]: print("  ", l)
    print(f"--- 该步 gt_action(答案,未泄露给感知): {s6['gt_action']}")
