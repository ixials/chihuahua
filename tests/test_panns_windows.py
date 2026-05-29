import pandas as pd

from bark_detection.config import BarkConfig
from bark_detection.panns_windows import generate_windows


def test_overlapping_windows_dogs1_duration():
    cfg = BarkConfig()
    duration = 5.014063
    df = generate_windows(duration, cfg)

    assert len(df) == 21
    assert float(df.iloc[0]["start_time_sec"]) == 0.0
    assert (df["start_time_sec"] >= 0).all()
    assert (df["end_time_sec"] >= 0).all()
    assert df["start_time_sec"].is_monotonic_increasing
    assert df["end_time_sec"].is_monotonic_increasing

    last_end = float(df.iloc[-1]["end_time_sec"])
    assert abs(last_end - duration) < 1e-6

    centers = (df["start_time_sec"] + df["end_time_sec"]) / 2.0
    pd.testing.assert_series_equal(
        centers.reset_index(drop=True),
        df["center_time_sec"].reset_index(drop=True),
        check_names=False,
    )

    durations = df["end_time_sec"] - df["start_time_sec"]
    assert (durations > 0).all()
    assert float(durations.iloc[-1]) < cfg.window_size_sec
    assert float(durations.iloc[-1]) > 0

    # True bounds only — last window ends at file duration, not padded to 1.0 s.
    assert float(df.iloc[-1]["end_time_sec"]) == duration
    assert float(df.iloc[-1]["start_time_sec"]) > duration - cfg.window_size_sec


def test_short_grid_dogs1_duration():
    cfg = BarkConfig()
    duration = 5.014063
    df = generate_windows(
        duration,
        cfg,
        window_size_sec=cfg.short_window_size_sec,
        hop_size_sec=cfg.short_hop_size_sec,
    )

    assert cfg.short_window_size_sec == 0.5
    assert cfg.short_hop_size_sec == 0.1
    assert len(df) == 51
    assert float(df.iloc[0]["start_time_sec"]) == 0.0
    assert float(df.iloc[0]["end_time_sec"]) == 0.5
    assert float(df.iloc[-1]["start_time_sec"]) == 5.0
    assert float(df.iloc[-1]["end_time_sec"]) == duration

    last_duration = float(df.iloc[-1]["end_time_sec"] - df.iloc[-1]["start_time_sec"])
    assert last_duration < cfg.short_window_size_sec
    assert last_duration > 0

    centers = (df["start_time_sec"] + df["end_time_sec"]) / 2.0
    pd.testing.assert_series_equal(
        centers.reset_index(drop=True),
        df["center_time_sec"].reset_index(drop=True),
        check_names=False,
    )
