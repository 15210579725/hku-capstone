"""交付物①：解析 events.txt caption 文件。

行格式：[HH:MM:SS -> HH:MM:SS] 中文描述
脏数据健壮处理：
  - 箭头无空格 `->` （L2 day1/day3 各几十行）；
  - 截断行（如 `[15:44:25 -> 15`）→ 记 warning 跳过，不抛异常；
  - 小时位 1~2 位。
"""
from __future__ import annotations

import json
import re
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# 健壮正则：箭头两侧空格可有可无，小时 1~2 位，分秒 2 位，文本可空
LINE_RE = re.compile(
    r"^\[\s*(\d{1,2}:\d{2}:\d{2})\s*->\s*(\d{1,2}:\d{2}:\d{2})\s*\]\s*(.*)$"
)


def _hms_to_sec(hms: str) -> int:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_events_file(path: str) -> pd.DataFrame:
    """解析单个 events.txt → DataFrame。

    列：idx, start, end, start_str, end_str, text, raw_line, parsed_ok
    start/end = 当天零点起的秒数 (int)，便于滑窗与对齐。
    匹配失败的行 parsed_ok=False（保留 raw_line），不抛异常。
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            line = raw.rstrip("\n")
            if not line.strip():
                continue  # 空行直接忽略，不计入
            m = LINE_RE.match(line)
            if m:
                start_str, end_str, text = m.group(1), m.group(2), m.group(3)
                rows.append(
                    {
                        "idx": i,
                        "start": _hms_to_sec(start_str),
                        "end": _hms_to_sec(end_str),
                        "start_str": start_str,
                        "end_str": end_str,
                        "text": text.strip(),
                        "raw_line": line,
                        "parsed_ok": True,
                    }
                )
            else:
                rows.append(
                    {
                        "idx": i,
                        "start": -1,
                        "end": -1,
                        "start_str": None,
                        "end_str": None,
                        "text": None,
                        "raw_line": line,
                        "parsed_ok": False,
                    }
                )
    df = pd.DataFrame(rows)
    return df


def load_layer(level: str = config.DEFAULT_LEVEL, days=None) -> dict:
    """加载某 level 的全部天 → {day: DataFrame(仅 parsed_ok 行，已排序)}。"""
    if days is None:
        days = config.DAYS
    out = {}
    for d in days:
        path = config.events_path(level, d)
        if not os.path.exists(path):
            continue
        df = parse_events_file(path)
        ok = df[df["parsed_ok"]].copy().sort_values("start").reset_index(drop=True)
        out[d] = ok
    return out


def parse_stats(level: str = config.DEFAULT_LEVEL, days=None, write: bool = True) -> dict:
    """统计每天行数 / 解析成功率 / 跳过行数，写 artifacts/parse_stats_{level}.json。"""
    if days is None:
        days = config.DAYS
    stats = {"level": level, "per_day": {}, "total_lines": 0, "total_skipped": 0}
    for d in days:
        path = config.events_path(level, d)
        if not os.path.exists(path):
            continue
        df = parse_events_file(path)
        n_total = len(df)
        n_ok = int(df["parsed_ok"].sum())
        n_skip = n_total - n_ok
        skipped_lines = df[~df["parsed_ok"]]["raw_line"].tolist()
        rate = n_ok / n_total if n_total else 0.0
        stats["per_day"][f"day{d}"] = {
            "total": n_total,
            "parsed_ok": n_ok,
            "skipped": n_skip,
            "success_rate": round(rate, 6),
            "skipped_examples": skipped_lines[:5],
        }
        stats["total_lines"] += n_total
        stats["total_skipped"] += n_skip
    stats["overall_success_rate"] = round(
        1 - stats["total_skipped"] / stats["total_lines"], 6
    ) if stats["total_lines"] else 0.0
    if write:
        with open(config.artifact(f"parse_stats_{level}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    s = parse_stats(config.DEFAULT_LEVEL)
    print(json.dumps(s, ensure_ascii=False, indent=2))
