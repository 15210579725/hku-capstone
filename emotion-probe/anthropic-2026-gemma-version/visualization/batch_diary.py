#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量把 25 个日记片段(中文+英文)POST 给官方 Flask 可视化器,
保存每个返回的自包含 HTML,并汇总情绪排名 top3。
"""
import json
import os
import re
import sys
import time

import requests

BASE_DIR = "/root/gemma-probes-capstone/official_run"
OUT_DIR = os.path.join(BASE_DIR, "diary_html")
OUT_ZH = os.path.join(OUT_DIR, "out_zh")
OUT_EN = os.path.join(OUT_DIR, "out_en")
URL = "http://127.0.0.1:8080/"
LAYER = "28"
PROBE_MODE = "expression"

EMO_RE = re.compile(r'class="emotion-item" data-emotion="([a-z]+)"')


def extract_top3(html):
    emos = EMO_RE.findall(html)
    return emos[:3]


def post_with_retry(text, retries=2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                URL,
                data={"text": text, "layer": LAYER, "probe_mode": PROBE_MODE},
                timeout=300,
            )
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa
            last_err = e
            print("    retry %d/%d after error: %s" % (attempt + 1, retries, e))
            time.sleep(3)
    raise last_err


def main():
    os.makedirs(OUT_ZH, exist_ok=True)
    os.makedirs(OUT_EN, exist_ok=True)

    with open(os.path.join(BASE_DIR, "selected_segments.json"), encoding="utf-8") as f:
        zh_data = json.load(f)
    with open(os.path.join(BASE_DIR, "selected_segments_en.json"), encoding="utf-8") as f:
        en_data = json.load(f)

    # 用 id 对齐英文
    en_by_id = {d["id"]: d for d in en_data}

    summary = []
    for i, seg in enumerate(zh_data):
        seg_id = seg["id"]
        print("[%d/%d] %s" % (i + 1, len(zh_data), seg_id))
        zh_text = "\n".join(seg["events"])
        en_seg = en_by_id.get(seg_id, {})
        en_text = "\n".join(en_seg.get("events", []))

        zh_top3, en_top3 = [], []

        # 中文
        try:
            zh_html = post_with_retry(zh_text)
            with open(os.path.join(OUT_ZH, "%s.html" % seg_id), "w", encoding="utf-8") as f:
                f.write(zh_html)
            zh_top3 = extract_top3(zh_html)
            print("    zh top3:", zh_top3)
        except Exception as e:  # noqa
            print("    ZH FAILED:", e)

        # 英文
        try:
            en_html = post_with_retry(en_text)
            with open(os.path.join(OUT_EN, "%s.html" % seg_id), "w", encoding="utf-8") as f:
                f.write(en_html)
            en_top3 = extract_top3(en_html)
            print("    en top3:", en_top3)
        except Exception as e:  # noqa
            print("    EN FAILED:", e)

        summary.append({
            "id": seg_id,
            "expected_emotions": seg.get("expected_emotions", []),
            "reason": seg.get("reason", ""),
            "zh_top3": zh_top3,
            "en_top3": en_top3,
        })

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nDONE. %d segments processed." % len(summary))


if __name__ == "__main__":
    main()
