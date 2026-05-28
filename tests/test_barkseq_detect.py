import pandas as pd

from bark_detection.barkseq_detect import detect_barkseqs
from bark_detection.config import BarkConfig
from tests.conftest import make_timeline_row, timeline_from_rows


def _cfg() -> BarkConfig:
    cfg = BarkConfig()
    cfg.barkseq_threshold = 0.42
    cfg.merge_gap_sec = 0.5
    cfg.hop_size_sec = 0.25
    return cfg


def test_single_barkseq_region():
    cfg = _cfg()
    timeline = timeline_from_rows(
        [
            make_timeline_row(0.25, bark=0.1),
            make_timeline_row(0.5, bark=0.55, dog=0.7),
            make_timeline_row(0.75, bark=0.1),
            make_timeline_row(1.0, bark=0.05),
        ]
    )
    out = detect_barkseqs(timeline, duration_sec=2.0, cfg=cfg)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["start_time_sec"] <= 0.5 <= row["end_time_sec"]
    assert float(row["peak_time_sec"]) == 0.5
    assert float(row["max_combined_bark_score"]) == 0.55


def test_two_barkseqs_far_apart():
    cfg = _cfg()
    timeline = timeline_from_rows(
        [
            make_timeline_row(0.5, bark=0.55),
            make_timeline_row(1.0, bark=0.05),
            make_timeline_row(1.5, bark=0.05),
            make_timeline_row(2.0, bark=0.60),
        ]
    )
    out = detect_barkseqs(timeline, duration_sec=3.0, cfg=cfg)

    assert len(out) == 2
    assert float(out.iloc[0]["peak_time_sec"]) == 0.5
    assert float(out.iloc[1]["peak_time_sec"]) == 2.0


def test_merge_nearby_regions():
    cfg = _cfg()
    timeline = timeline_from_rows(
        [
            make_timeline_row(0.5, bark=0.55),
            make_timeline_row(0.75, bark=0.05),
            make_timeline_row(1.0, bark=0.50),
        ]
    )
    out = detect_barkseqs(timeline, duration_sec=2.0, cfg=cfg)

    assert len(out) == 1
    assert float(out.iloc[0]["start_time_sec"]) <= 0.375
    assert float(out.iloc[0]["end_time_sec"]) >= 1.125
