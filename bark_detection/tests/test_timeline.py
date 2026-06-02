import pandas as pd

from bark_detection.config import BarkConfig
from bark_detection.timeline import TIMELINE_COLUMNS, build_timeline


def _fake_panns_scores(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_timeline_monotonic_and_bark_mode():
    cfg = BarkConfig()
    cfg.combined_bark_mode = "bark"

    scores = _fake_panns_scores(
        [
            {
                "center_time_sec": 0.5,
                "dog_score": 0.7,
                "bark_score": 0.49,
                "speech_score": 0.05,
                "music_score": 0.03,
            },
            {
                "center_time_sec": 1.0,
                "dog_score": 0.2,
                "bark_score": 0.03,
                "speech_score": 0.15,
                "music_score": 0.02,
            },
            {
                "center_time_sec": 1.5,
                "dog_score": 0.6,
                "bark_score": 0.55,
                "speech_score": 0.04,
                "music_score": 0.04,
            },
        ]
    )

    out = build_timeline(scores, cfg)

    assert list(out.columns) == TIMELINE_COLUMNS
    assert out["time_sec"].is_monotonic_increasing
    pd.testing.assert_series_equal(
        out["combined_bark_score"],
        out["bark_score"],
        check_names=False,
    )


def test_timeline_max_bark_dog_mode():
    cfg = BarkConfig()
    cfg.combined_bark_mode = "max_bark_dog"

    scores = _fake_panns_scores(
        [
            {
                "center_time_sec": 0.5,
                "dog_score": 0.8,
                "bark_score": 0.3,
                "speech_score": 0.0,
                "music_score": 0.0,
            },
        ]
    )
    out = build_timeline(scores, cfg)
    assert float(out.iloc[0]["combined_bark_score"]) == 0.8
