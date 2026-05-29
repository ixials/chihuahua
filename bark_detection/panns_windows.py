"""M2: overlapping analysis windows for PANNs inference."""

import pandas as pd

from bark_detection.config import BarkConfig

_END_TOLERANCE_SEC = 1e-6


def generate_windows(
    duration_sec: float,
    cfg: BarkConfig,
    *,
    window_size_sec: float | None = None,
    hop_size_sec: float | None = None,
) -> pd.DataFrame:
    """Build overlapping windows covering [0, duration_sec].

    Windows start at t=0 and advance by ``hop_size_sec`` while
    ``start_time_sec < duration_sec``. Each row stores true audio bounds
    (no padding): ``end_time_sec = min(start + window_size_sec, duration_sec)``.

    Optional ``window_size_sec`` / ``hop_size_sec`` override ``cfg`` defaults
    (e.g. short grids via ``cfg.short_window_size_sec``).
    """
    window_size = cfg.window_size_sec if window_size_sec is None else window_size_sec
    hop_size = cfg.hop_size_sec if hop_size_sec is None else hop_size_sec

    rows: list[dict[str, float | int]] = []
    window_id = 0
    k = 0
    while True:
        start = k * hop_size
        if start >= duration_sec:
            break
        end = min(start + window_size, duration_sec)
        rows.append(
            {
                "window_id": window_id,
                "start_time_sec": start,
                "end_time_sec": end,
                "center_time_sec": (start + end) / 2.0,
            }
        )
        window_id += 1
        k += 1

    df = pd.DataFrame(rows, columns=[
        "window_id",
        "start_time_sec",
        "end_time_sec",
        "center_time_sec",
    ])

    if len(df) > 0:
        last_end = float(df.iloc[-1]["end_time_sec"])
        if abs(last_end - duration_sec) > _END_TOLERANCE_SEC:
            raise ValueError(
                f"last window end_time_sec ({last_end}) != duration_sec ({duration_sec})"
            )

    return df