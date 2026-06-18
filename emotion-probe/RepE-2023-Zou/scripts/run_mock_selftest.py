"""端到端 mock 自测：parse → windowize → feelings(+mapping) → mock artifact →
score(mock residual, 信号 vs 随机) → normalize(z+q) → validate → visualize。

全程 L2、默认窗口参数（N=8/stride=4）。跑完打印结构化汇报数据。
判据：无异常跑完 + 所有 assert 通过 + 信号注入指标显著优于随机 + artifact 落盘。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from src import (parse_captions, windowize, parse_feelings, emotion_mapping,  # noqa: E402
                 direction_artifact, score, normalize, validate, visualize)


def section(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


def main():
    LEVEL = config.DEFAULT_LEVEL
    MODE = config.WINDOW_MODE
    report = {}

    section(f"[1/8] PARSE CAPTIONS  level={LEVEL}")
    pstats = parse_captions.parse_stats(LEVEL)
    report["parse"] = pstats
    print(f"overall success rate: {pstats['overall_success_rate']}  "
          f"total skipped: {pstats['total_skipped']}")
    for d, v in pstats["per_day"].items():
        print(f"  {d}: total={v['total']:5d} ok={v['parsed_ok']:5d} "
              f"skip={v['skipped']} rate={v['success_rate']}")
    print(f"  day2 skipped example (truncated line): {pstats['per_day']['day2']['skipped_examples']}")

    section(f"[2/8] WINDOWIZE  mode={MODE}  N={config.WIN_N} stride={config.WIN_STRIDE}")
    wdf = windowize.build_windows(LEVEL, MODE)
    wstats = windowize.window_stats(wdf, LEVEL, MODE)
    report["windows"] = wstats
    print(f"total windows: {wstats['total_windows']}")
    for d, v in wstats["per_day"].items():
        print(f"  {d}: n_windows={v['n_windows']:4d} partial={v['n_partial']} "
              f"mean_events={v['mean_n_events']}")

    # 备选 ~3min 时间窗也实跑一遍，证明备选可用
    wdf_time = windowize.build_windows(LEVEL, "time")
    report["windows_time_mode_total"] = int(len(wdf_time))
    print(f"  [alt ~3min time-window] total windows: {len(wdf_time)}")

    section("[3/8] PARSE FEELINGS + EMOTION MAPPING (438 rows)")
    feel = parse_feelings.parse_feelings(write=True)
    fstats = parse_feelings.feelings_stats(feel)
    report["feelings"] = fstats
    print(f"n_rows={fstats['n_rows']}  neutral_ratio={fstats['neutral_ratio']}  "
          f"controversial={fstats['controversial_count']}")
    print(f"category dist: {fstats['category_dist']}")
    print(f"mapped_emotion dist (438-line mapping): {fstats['mapped_emotion_dist']}")
    print(f"per-day feelings: {fstats['per_day_counts']}")

    section("[4/8] MOCK DIRECTION ARTIFACT (Agent B contract)")
    art = direction_artifact.make_mock_artifact(hidden_dim=config.MOCK_HIDDEN_DIM, seed=0)
    print(f"hidden_dim={art.hidden_dim} emotions={art.emotions}")
    print(f"best_layer={art.best_layer}")
    print(f"direction_signs={art.direction_signs}")

    section("[5/8] SCORE (mock residual: signal-injection vs pure-random)")
    prov_sig = score.MockResidualProvider(art, inject_strength=1.6)
    prov_rnd = score.MockResidualProvider(art, inject_strength=0.0)
    s_sig = score.score_windows(wdf, art, prov_sig, tag="signal", level=LEVEL, mode=MODE)
    s_rnd = score.score_windows(wdf, art, prov_rnd, tag="random", level=LEVEL, mode=MODE, write=False)
    print(f"score matrix shape: ({len(s_sig)}, {len(config.EMOTIONS)})  (windows x emotions)")
    assert s_sig[config.EMOTIONS].notna().all().all(), "score has NaN"

    section("[6/8] NORMALIZE (z-score per day + quantile [0,1])")
    norm_sig = normalize.normalize(s_sig, tag="signal", level=LEVEL, mode=MODE)
    norm_rnd = normalize.normalize(s_rnd, tag="random", level=LEVEL, mode=MODE, write=False)
    for e in config.EMOTIONS[:1]:
        zc = norm_sig[f"{e}_z"]
        qc = norm_sig[f"{e}_q"]
        print(f"  {e}: z range [{zc.min():.2f},{zc.max():.2f}]  q range [{qc.min():.2f},{qc.max():.2f}]")
    # 校验
    for e in config.EMOTIONS:
        assert ((norm_sig[f"{e}_q"] >= 0) & (norm_sig[f"{e}_q"] <= 1)).all()

    section("[7/8] VALIDATE  (signal vs random baseline)")
    r_sig = validate.run_validation(feel, norm_sig, tag="signal", write=True)
    r_rnd = validate.run_validation(feel, norm_rnd, tag="random", write=True)
    m_sig, m_rnd = r_sig["metrics"], r_rnd["metrics"]
    report["validation_signal"] = m_sig
    report["validation_random"] = m_rnd

    def row(name, key, fmt="{}"):
        sv = m_sig.get(key)
        rv = m_rnd.get(key)
        print(f"  {name:24s} signal={fmt.format(sv)}   random={fmt.format(rv)}")

    print(f"  aligned feelings: {m_sig['n_aligned']}/{m_sig['n_feelings']}")
    row("strict_hit_rate", "strict_hit_rate")
    row("qualitative_hit_rate", "qualitative_hit_rate")
    row("polarity_pearson", "polarity_pearson")
    row("polarity_spearman", "polarity_spearman")
    row("binary_accuracy", "binary_accuracy")
    row("cohen_kappa", "cohen_kappa")

    # 敏感性判据：信号注入显著优于随机
    assert m_sig["strict_hit_rate"] > (m_rnd["strict_hit_rate"] or 0) + 0.05, \
        "pipeline not sensitive (hit_rate)"
    assert (m_sig["polarity_pearson"] or 0) > (m_rnd["polarity_pearson"] or 0), \
        "pipeline not sensitive (polarity)"
    print("  => SIGNAL > RANDOM on hit_rate & polarity  (pipeline is SENSITIVE, non-trivial)")

    section("[8/8] VISUALIZE")
    aligned = validate.align_feelings_to_windows(feel, norm_sig)
    top = validate.divergence_topk(aligned, write=True)
    p1 = visualize.plot_day_curves(norm_sig, feel, day=config.PRIMARY_VALIDATION_DAY)
    p2 = visualize.plot_divergence_table(top)
    for p in (p1, p2):
        sz = os.path.getsize(p)
        print(f"  {p}  ({sz} bytes)")
        assert sz > 0

    # 汇总落盘
    with open(config.artifact("mock_selftest_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    section("ARTIFACTS WRITTEN")
    for fn in sorted(os.listdir(config.ARTIFACTS_DIR)):
        fp = os.path.join(config.ARTIFACTS_DIR, fn)
        print(f"  {fn}  ({os.path.getsize(fp)} bytes)")

    section("ALL SELF-TESTS PASSED")
    print("end-to-end mock pipeline ran clean; signal > random baseline; artifacts on disk.")


if __name__ == "__main__":
    main()
