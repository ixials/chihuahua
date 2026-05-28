"""M6: Bark confidence scoring with BarkClassifier hook."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from bark_detection.config import BarkConfig


class BarkClassifier(Protocol):
    def score(self, wav: np.ndarray, sr: int, t_start: float, t_end: float) -> float: ...


def triangle(
    duration_sec: float,
    peak_sec: float,
    min_sec: float,
    max_sec: float,
) -> float:
    """Triangle scoring function for bark duration.

    Returns 0 outside [min_sec, max_sec].
    Linearly ramps 0 → 1 from min_sec to peak_sec.
    Linearly ramps 1 → 0 from peak_sec to max_sec.
    Value at peak_sec is exactly 1.0.
    """
    if duration_sec <= min_sec or duration_sec >= max_sec:
        return 0.0
    if duration_sec <= peak_sec:
        return (duration_sec - min_sec) / (peak_sec - min_sec)
    # duration_sec > peak_sec (and < max_sec)
    return (max_sec - duration_sec) / (max_sec - peak_sec)


def compute_bark_confidence(
    events_df: pd.DataFrame,
    wav: np.ndarray,
    sr: int,
    cfg: BarkConfig,
) -> pd.DataFrame:
    """Add a bark_confidence column to events_df.

    Heuristic (default path, classifier=None):
        duration_score  = triangle(duration_sec, peak=cfg.duration_score_peak_sec,
                                   min=0.08, max=1.2)
        bark_confidence = cfg.w_rms * normalized_rms_peak
                        + cfg.w_duration * duration_score

    Classifier blend (when cfg.classifier is not None):
        heuristic = as above
        bark_confidence = (1 - cfg.classifier_weight) * heuristic
                        + cfg.classifier_weight * cfg.classifier.score(wav, sr, t_start, t_end)

    Result is clipped to [0, 1] defensively.
    Returns a copy of events_df with the bark_confidence column appended.
    """
    df = events_df.copy()
    confidences: list[float] = []

    for _, row in df.iterrows():
        dur = float(row["duration_sec"])
        norm_rms = float(row["normalized_rms_peak"])

        dur_score = triangle(
            dur,
            peak_sec=cfg.duration_score_peak_sec,
            min_sec=0.08,
            max_sec=1.2,
        )

        heuristic = cfg.w_rms * norm_rms + cfg.w_duration * dur_score

        if cfg.classifier is not None:
            t_start = float(row["start_time_sec"])
            t_end = float(row["end_time_sec"])
            classifier_score = cfg.classifier.score(wav, sr, t_start, t_end)
            confidence = (
                (1.0 - cfg.classifier_weight) * heuristic
                + cfg.classifier_weight * classifier_score
            )
        else:
            confidence = heuristic

        confidences.append(float(np.clip(confidence, 0.0, 1.0)))

    df["bark_confidence"] = confidences
    return df
