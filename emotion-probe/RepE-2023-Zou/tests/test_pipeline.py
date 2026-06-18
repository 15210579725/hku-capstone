"""单元自测（纯 assert 脚本，无需 pytest 也可 `python3 tests/test_pipeline.py`）。

覆盖每个交付物的关键不变量。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from src import (parse_captions, windowize, parse_feelings, emotion_mapping,  # noqa: E402
                 direction_artifact, score, normalize, validate)


def test_parse_captions():
    s = parse_captions.parse_stats("L2", write=False)
    assert s["overall_success_rate"] >= 0.999
    assert s["per_day"]["day2"]["skipped"] == 1
    assert "[15:44:25 -> 15" in s["per_day"]["day2"]["skipped_examples"][0]
    assert s["per_day"]["day1"]["total"] == 2006
    print("ok test_parse_captions")


def test_windowize():
    wdf = windowize.build_windows("L2", "count", write=False)
    g = wdf[wdf["day"] == 1].reset_index(drop=True)
    assert (g["t_start"] <= g["mid_time"]).all()
    assert (g["mid_time"] <= g["t_end"]).all()
    assert g["t_start"].is_monotonic_increasing
    # 相邻窗重叠（计数窗 stride<N）
    assert (g["t_start"].iloc[1] <= g["t_end"].iloc[0])
    print("ok test_windowize")


def test_feelings_and_mapping():
    tl = parse_feelings.parse_feelings(write=False)
    assert len(tl) == 438
    assert tl["mapped_emotion"].notna().all()
    assert (tl["start"] >= 0).all()
    # 映射正确性抽查
    assert emotion_mapping.map_content("我有点开心")[0] == "happiness"
    assert emotion_mapping.map_content("我很担心")[0] == "fear"
    assert emotion_mapping.map_content("我在想事情")[0] == config.NEUTRAL
    assert emotion_mapping.map_content("我感到尴尬")[3] is True  # controversial
    print("ok test_feelings_and_mapping")


def test_artifact():
    art = direction_artifact.make_mock_artifact(write=False)
    assert art.hidden_dim == config.MOCK_HIDDEN_DIM
    for e in config.EMOTIONS:
        assert art.directions[e].shape == (art.hidden_dim,)
        assert abs(np.linalg.norm(art.directions[e]) - 1) < 1e-5
    print("ok test_artifact")


def test_score_and_normalize():
    art = direction_artifact.make_mock_artifact(write=False)
    wdf = windowize.build_windows("L2", "count", write=False)
    prov = score.MockResidualProvider(art, inject_strength=1.6)
    sdf = score.score_windows(wdf, art, prov, write=False)
    assert sdf[config.EMOTIONS].notna().all().all()
    assert sdf.shape[0] == len(wdf)
    norm = normalize.normalize(sdf, write=False)
    for e in config.EMOTIONS:
        assert ((norm[f"{e}_q"] >= 0) & (norm[f"{e}_q"] <= 1)).all()
    print("ok test_score_and_normalize")


def test_validate_sensitive():
    art = direction_artifact.make_mock_artifact(write=False)
    wdf = windowize.build_windows("L2", "count", write=False)
    feel = parse_feelings.parse_feelings(write=False)
    res = {}
    for tag, st in [("signal", 1.6), ("random", 0.0)]:
        prov = score.MockResidualProvider(art, inject_strength=st)
        sdf = score.score_windows(wdf, art, prov, write=False)
        norm = normalize.normalize(sdf, write=False)
        res[tag] = validate.run_validation(feel, norm, tag=tag, write=False)["metrics"]
    assert res["signal"]["strict_hit_rate"] > res["random"]["strict_hit_rate"] + 0.05
    print("ok test_validate_sensitive  (signal > random)")


if __name__ == "__main__":
    test_parse_captions()
    test_windowize()
    test_feelings_and_mapping()
    test_artifact()
    test_score_and_normalize()
    test_validate_sensitive()
    print("\nALL UNIT TESTS PASSED")
