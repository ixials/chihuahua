"""M3: smoothing + adaptive threshold for the RMS energy curve."""

import numpy as np
import pandas as pd

from bark_detection.config import BarkConfig


def smooth_rms(rms: np.ndarray, cfg: BarkConfig) -> np.ndarray:
    """Return a smoothed copy of `rms` per cfg.smoothing_method.

    Window length in frames = max(1, round(cfg.smoothing_window_ms / cfg.hop_ms)).
    - "moving_average": uniform kernel via pandas rolling mean (center=True).
    - "median":         pandas rolling median (center=True).

    Output has the same length as input, no NaN, all >= 0.
    """
    n = max(1, round(cfg.smoothing_window_ms / cfg.hop_ms))
    series = pd.Series(rms.astype(np.float64))

    if cfg.smoothing_method == "moving_average":
        smoothed = series.rolling(n, center=True, min_periods=1).mean()
    elif cfg.smoothing_method == "median":
        smoothed = series.rolling(n, center=True, min_periods=1).median()
    else:
        raise ValueError(
            f"Unknown smoothing_method: {cfg.smoothing_method!r}. "
            "Expected 'moving_average' or 'median'."
        )

    result = smoothed.to_numpy(dtype=np.float32)
    # Guarantee no NaN / negative values (should not occur with min_periods=1)
    result = np.where(np.isfinite(result), result, 0.0)
    result = np.maximum(result, 0.0)
    return result


def compute_thresholds(rms_smoothed: np.ndarray, cfg: BarkConfig) -> dict:
    """Compute both candidate threshold values and identify the operating point.

    Returns a dict with keys:
      mean_std_k2      float  — mean + 2 * std (always computed, k hard-coded to 2)
      percentile_90    float  — 90th percentile (always computed)
      operating_point  float  — whichever cfg.threshold_method selects
      operating_method str    — "mean_std" | "percentile"
      mean             float  — mean of rms_smoothed
      std              float  — std of rms_smoothed

    All floats rounded to 6 decimal places.
    """
    arr = rms_smoothed.astype(np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))

    mean_std_k2 = mu + 2.0 * sigma
    pct_90 = float(np.percentile(arr, 90.0))

    if cfg.threshold_method == "mean_std":
        operating_point = mu + cfg.mean_std_k * sigma
        operating_method = "mean_std"
    elif cfg.threshold_method == "percentile":
        operating_point = float(np.percentile(arr, cfg.percentile))
        operating_method = "percentile"
    else:
        raise ValueError(
            f"Unknown threshold_method: {cfg.threshold_method!r}. "
            "Expected 'mean_std' or 'percentile'."
        )

    return {
        "mean_std_k2": round(mean_std_k2, 6),
        "percentile_90": round(pct_90, 6),
        "operating_point": round(operating_point, 6),
        "operating_method": operating_method,
        "mean": round(mu, 6),
        "std": round(sigma, 6),
    }
