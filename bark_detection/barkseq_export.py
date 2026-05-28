"""M7: Export canonical barkseqs.csv with confidence scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bark_detection.config import BarkConfig
from bark_detection import noise

BARKSEQS_COLUMNS = [
    "barkseq_id",
    "start_time_sec",
    "end_time_sec",
    "peak_time_sec",
    "duration_sec",
    "max_dog_score",
    "mean_dog_score",
    "max_bark_score",
    "mean_bark_score",
    "max_combined_bark_score",
    "mean_combined_bark_score",
    "max_speech_score",
    "max_music_score",
    "noise_flag",
    "noise_reason",
    "confidence",
    "method",
]


def _method_string(cfg: BarkConfig) -> str:
    model = cfg.panns_model_name.lower().replace("-", "")
    return f"panns_{model}_v1"


def compute_confidence(row: pd.Series, cfg: BarkConfig) -> float:
    """Multiplicative bark score with speech/music penalties, clipped to [0, 1]."""
    mean_combined = float(row["mean_combined_bark_score"])
    max_speech = float(row["max_speech_score"])
    max_music = float(row["max_music_score"])
    raw = mean_combined * (
        1.0 - cfg.speech_penalty * max_speech
    ) * (
        1.0 - cfg.music_penalty * max_music
    )
    return float(np.clip(raw, 0.0, 1.0))


def export_barkseqs(
    barkseqs_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    cfg: BarkConfig,
) -> pd.DataFrame:
    """Apply M6 noise flags and M7 confidence; return final export DataFrame."""
    flagged = noise.flag_noise(barkseqs_df, timeline_df, cfg)
    out = flagged.copy()
    out["confidence"] = out.apply(compute_confidence, axis=1, cfg=cfg)
    out["method"] = _method_string(cfg)
    return out[BARKSEQS_COLUMNS]
