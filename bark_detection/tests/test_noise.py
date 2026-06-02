import pandas as pd

from bark_detection.config import BarkConfig
from bark_detection.noise import flag_noise
from tests.conftest import make_timeline_row, timeline_from_rows


def _cfg() -> BarkConfig:
    cfg = BarkConfig()
    cfg.speech_noise_threshold = 0.15
    cfg.music_noise_threshold = 0.15
    cfg.barkseq_threshold = 0.42
    return cfg


def _barkseq_row(
    *,
    start: float,
    end: float,
    max_combined: float,
    barkseq_id: int = 0,
) -> dict:
    return {
        "barkseq_id": barkseq_id,
        "start_time_sec": start,
        "end_time_sec": end,
        "peak_time_sec": (start + end) / 2,
        "duration_sec": end - start,
        "max_combined_bark_score": max_combined,
        "mean_combined_bark_score": max_combined,
    }


def test_noise_clean():
    cfg = _cfg()
    barkseqs = pd.DataFrame([_barkseq_row(start=0.375, end=0.625, max_combined=0.49)])
    timeline = timeline_from_rows(
        [make_timeline_row(0.5, speech=0.05, music=0.03)]
    )
    out = flag_noise(barkseqs, timeline, cfg)

    assert len(out) == 1
    assert out.iloc[0]["noise_reason"] == "clean"
    assert out.iloc[0]["noise_flag"] is False or out.iloc[0]["noise_flag"] == False
    assert float(out.iloc[0]["max_speech_score"]) == 0.05
    assert float(out.iloc[0]["max_music_score"]) == 0.03


def test_high_speech_strong_bark():
    cfg = _cfg()
    barkseqs = pd.DataFrame([_barkseq_row(start=0.375, end=0.625, max_combined=0.50)])
    timeline = timeline_from_rows(
        [make_timeline_row(0.5, speech=0.20, music=0.03)]
    )
    out = flag_noise(barkseqs, timeline, cfg)

    assert len(out) == 1
    assert out.iloc[0]["noise_flag"] in (True, 1)
    assert "speech" in str(out.iloc[0]["noise_reason"])


def test_high_music_strong_bark():
    cfg = _cfg()
    barkseqs = pd.DataFrame([_barkseq_row(start=0.375, end=0.625, max_combined=0.50)])
    timeline = timeline_from_rows(
        [make_timeline_row(0.5, speech=0.03, music=0.20)]
    )
    out = flag_noise(barkseqs, timeline, cfg)

    assert len(out) == 1
    assert out.iloc[0]["noise_flag"] in (True, 1)
    assert "music" in str(out.iloc[0]["noise_reason"])


def test_likely_noise_weak_bark():
    cfg = _cfg()
    barkseqs = pd.DataFrame([_barkseq_row(start=0.375, end=0.625, max_combined=0.35)])
    timeline = timeline_from_rows(
        [make_timeline_row(0.5, speech=0.20, music=0.03)]
    )
    out = flag_noise(barkseqs, timeline, cfg)

    assert len(out) == 1
    assert out.iloc[0]["noise_reason"] == "likely_noise"
