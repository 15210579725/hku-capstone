"""交付物②：构滑窗。

A) 事件计数窗（默认）：N=8 条/窗，stride=4（重叠 50%）。
B) 时间窗（备选）：~3min 窗，1.5min step。
末尾残窗保留并标 partial=True，不丢数据。
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import parse_captions  # noqa: E402


def _join_text(texts) -> str:
    return " / ".join(t for t in texts if t)


def windowize_day_count(df: pd.DataFrame, day: int, n: int, stride: int) -> list:
    """事件计数窗。df 须已按 start 排序。"""
    wins = []
    n_rows = len(df)
    starts = list(range(0, max(n_rows - n, 0) + 1, stride))
    # 保证残窗：若最后一个起点未覆盖到末尾，追加一个对齐到末尾的残窗
    if n_rows > 0 and (not starts or starts[-1] + n < n_rows):
        last_start = max(n_rows - n, 0)
        if not starts or starts[-1] != last_start:
            starts.append(min(last_start, starts[-1] + stride) if starts else 0)
    seen = set()
    wid = 0
    for s in starts:
        e = min(s + n, n_rows)
        key = (s, e)
        if key in seen:
            continue
        seen.add(key)
        chunk = df.iloc[s:e]
        if chunk.empty:
            continue
        t_start = int(chunk["start"].iloc[0])
        t_end = int(chunk["end"].iloc[-1])
        wins.append(
            {
                "win_id": f"d{day}_w{wid}",
                "day": day,
                "win_idx": wid,
                "t_start": t_start,
                "t_end": t_end,
                "mid_time": (t_start + t_end) // 2,
                "n_events": int(len(chunk)),
                "text": _join_text(chunk["text"].tolist()),
                "partial": bool(len(chunk) < n),
            }
        )
        wid += 1
    return wins


def windowize_day_time(df: pd.DataFrame, day: int, win_minutes: float, step_minutes: float) -> list:
    """时间窗。按 start 秒切片。"""
    wins = []
    if df.empty:
        return wins
    win_s = win_minutes * 60.0
    step_s = step_minutes * 60.0
    t0 = int(df["start"].min())
    t_last = int(df["end"].max())
    wid = 0
    cur = t0
    while cur <= t_last:
        lo, hi = cur, cur + win_s
        mask = (df["start"] >= lo) & (df["start"] < hi)
        chunk = df[mask]
        if not chunk.empty:
            t_start = int(chunk["start"].iloc[0])
            t_end = int(chunk["end"].iloc[-1])
            wins.append(
                {
                    "win_id": f"d{day}_w{wid}",
                    "day": day,
                    "win_idx": wid,
                    "t_start": t_start,
                    "t_end": t_end,
                    "mid_time": int((lo + hi) // 2),
                    "n_events": int(len(chunk)),
                    "text": _join_text(chunk["text"].tolist()),
                    "partial": bool((hi > t_last)),
                }
            )
            wid += 1
        cur += step_s
    return wins


def build_windows(level: str = config.DEFAULT_LEVEL, mode: str = None,
                  n: int = None, stride: int = None,
                  win_minutes: float = None, step_minutes: float = None,
                  write: bool = True) -> pd.DataFrame:
    mode = mode or config.WINDOW_MODE
    n = n or config.WIN_N
    stride = stride or config.WIN_STRIDE
    win_minutes = win_minutes or config.WIN_MINUTES
    step_minutes = step_minutes or config.STEP_MINUTES

    layer = parse_captions.load_layer(level)
    all_wins = []
    for day, df in layer.items():
        if mode == "count":
            all_wins.extend(windowize_day_count(df, day, n, stride))
        elif mode == "time":
            all_wins.extend(windowize_day_time(df, day, win_minutes, step_minutes))
        else:
            raise ValueError(f"unknown mode {mode}")
    wdf = pd.DataFrame(all_wins)
    if write and not wdf.empty:
        path = config.artifact(f"windows_{level}_{mode}")
        try:
            wdf.to_parquet(path + ".parquet", index=False)
        except Exception:
            wdf.to_csv(path + ".csv", index=False)
    return wdf


def window_stats(wdf: pd.DataFrame, level: str = config.DEFAULT_LEVEL,
                 mode: str = None, write: bool = True) -> dict:
    mode = mode or config.WINDOW_MODE
    stats = {"level": level, "mode": mode, "total_windows": int(len(wdf)), "per_day": {}}
    for day, g in wdf.groupby("day"):
        stats["per_day"][f"day{int(day)}"] = {
            "n_windows": int(len(g)),
            "n_partial": int(g["partial"].sum()),
            "mean_n_events": round(float(g["n_events"].mean()), 2),
        }
    if write:
        with open(config.artifact(f"window_stats_{level}_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    wdf = build_windows()
    s = window_stats(wdf)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    # 自测断言：时间单调、t_start<=mid<=t_end、相邻窗重叠
    g1 = wdf[wdf["day"] == 1].reset_index(drop=True)
    assert (g1["t_start"] <= g1["mid_time"]).all() and (g1["mid_time"] <= g1["t_end"]).all()
    assert g1["t_start"].is_monotonic_increasing
    print("windowize self-test passed")
