"""M4: Candidate region detection.

Find every contiguous region in the smoothed RMS curve that exceeds the threshold.
Each region becomes one row in the returned DataFrame.
"""

import numpy as np
import pandas as pd

from bark_detection.config import BarkConfig


def find_candidates(rms_df: pd.DataFrame, threshold: float, cfg: BarkConfig) -> pd.DataFrame:
    """Find contiguous regions where rms_smoothed > threshold.

    For each region compute one row of a DataFrame with columns:
      - candidate_id          : 0-based integer index
      - start_time_sec        : float
      - peak_time_sec         : float — time_sec of argmax(rms_smoothed) within region
      - end_time_sec          : float
      - duration_sec          : end_time_sec - start_time_sec
      - rms_peak              : float — max of rms_smoothed within the region
      - normalized_rms_peak   : float — rms_peak / global max of rms_smoothed in (0, 1]

    Regions are detected on rms_smoothed (strict >). Times use the half-hop convention
    so that even a 1-frame region satisfies start_time_sec < peak_time_sec <= end_time_sec.
    """
    smoothed = rms_df["rms_smoothed"].to_numpy()
    times = rms_df["time_sec"].to_numpy()

    hop_sec = cfg.hop_ms / 1000.0
    global_max = float(smoothed.max())

    mask = smoothed > threshold  # strict comparison

    # Detect rising and falling edges using np.diff.
    # Prepend/append False so that regions at array boundaries are handled correctly.
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))  # +1 = rising edge, -1 = falling edge

    rising_edges = np.where(diff == 1)[0]    # index into original array (no offset needed)
    falling_edges = np.where(diff == -1)[0]  # first index *after* the region

    rows = []
    for cid, (start_idx, end_idx) in enumerate(zip(rising_edges, falling_edges)):
        # end_idx is the first frame outside the region (exclusive)
        last_idx = end_idx - 1
        region_smoothed = smoothed[start_idx:end_idx]
        argmax_local = int(np.argmax(region_smoothed))
        peak_idx = start_idx + argmax_local

        start_time = float(times[start_idx]) - hop_sec / 2.0
        end_time = float(times[last_idx]) + hop_sec / 2.0
        peak_time = float(times[peak_idx])

        # Clamp to valid audio range
        start_time = max(0.0, start_time)
        # duration_sec from metadata is not directly available here; clamp conservatively
        # using the last observed time + half hop as the upper bound
        audio_end = float(times[-1]) + hop_sec / 2.0
        end_time = min(end_time, audio_end)

        rms_peak = float(region_smoothed.max())
        normalized_rms_peak = rms_peak / global_max

        rows.append({
            "candidate_id": cid,
            "start_time_sec": start_time,
            "peak_time_sec": peak_time,
            "end_time_sec": end_time,
            "duration_sec": end_time - start_time,
            "rms_peak": rms_peak,
            "normalized_rms_peak": normalized_rms_peak,
        })

    columns = [
        "candidate_id",
        "start_time_sec",
        "peak_time_sec",
        "end_time_sec",
        "duration_sec",
        "rms_peak",
        "normalized_rms_peak",
    ]

    if rows:
        return pd.DataFrame(rows, columns=columns)
    else:
        return pd.DataFrame(columns=columns)
