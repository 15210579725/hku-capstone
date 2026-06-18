"""交付物③：金标准 feelings_all_days.tsv 解析 + 情绪映射。

列：day, line_no, time, category, content（tab 分隔，有表头，438 数据行）。
time 列 = [HH:MM:SS -> HH:MM:SS]，与 caption 同正则。
产出「真实情绪时间线」DataFrame，写 artifacts/feelings_timeline.parquet。
"""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import emotion_mapping  # noqa: E402

TIME_RE = re.compile(r"\[\s*(\d{1,2}:\d{2}:\d{2})\s*->\s*(\d{1,2}:\d{2}:\d{2})\s*\]")


def _hms_to_sec(hms: str) -> int:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _day_to_int(day_str) -> int:
    s = str(day_str).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def parse_feelings(write: bool = True) -> pd.DataFrame:
    df = pd.read_csv(config.FEELINGS_TSV, sep="\t", dtype=str).fillna("")
    assert list(df.columns) == ["day", "line_no", "time", "category", "content"], \
        f"unexpected columns: {list(df.columns)}"

    recs = []
    for _, r in df.iterrows():
        m = TIME_RE.search(r["time"])
        if m:
            start = _hms_to_sec(m.group(1))
            end = _hms_to_sec(m.group(2))
        else:
            start, end = -1, -1
        emo, pol, rule, ctrl = emotion_mapping.map_content(r["content"])
        recs.append(
            {
                "day": _day_to_int(r["day"]),
                "line_no": int(r["line_no"]) if str(r["line_no"]).isdigit() else -1,
                "time_str": r["time"],
                "start": start,
                "end": end,
                "mid_time": (start + end) // 2 if start >= 0 else -1,
                "category": r["category"],
                "content": r["content"],
                "mapped_emotion": emo,
                "polarity": pol,
                "mapped_rule": rule,
                "controversial": ctrl,
            }
        )
    tl = pd.DataFrame(recs)
    if write:
        emotion_mapping.export_mapping_table()
        try:
            tl.to_parquet(config.artifact("feelings_timeline.parquet"), index=False)
        except Exception:
            tl.to_csv(config.artifact("feelings_timeline.csv"), index=False)
    return tl


def feelings_stats(tl: pd.DataFrame, write: bool = True) -> dict:
    n = len(tl)
    emo_counts = tl["mapped_emotion"].value_counts().to_dict()
    neutral_ratio = emo_counts.get(config.NEUTRAL, 0) / n if n else 0.0
    stats = {
        "n_rows": int(n),
        "category_dist": tl["category"].value_counts().to_dict(),
        "mapped_emotion_dist": {k: int(v) for k, v in emo_counts.items()},
        "neutral_ratio": round(neutral_ratio, 4),
        "controversial_count": int(tl["controversial"].sum()),
        "per_day_counts": {f"day{int(k)}": int(v) for k, v in tl["day"].value_counts().sort_index().items()},
        "time_parse_failures": int((tl["start"] < 0).sum()),
    }
    if write:
        with open(config.artifact("feelings_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    tl = parse_feelings()
    s = feelings_stats(tl)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    # 自测断言：438 条全部有映射结果、无 NaN、时间全解析
    assert len(tl) == 438, f"expected 438 rows, got {len(tl)}"
    assert tl["mapped_emotion"].notna().all()
    assert s["time_parse_failures"] == 0
    print("parse_feelings self-test passed")
