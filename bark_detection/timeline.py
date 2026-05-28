"""M4: Collapse per-window PANNs scores into a bark score timeline."""

from __future__ import annotations

import pandas as pd

from bark_detection.config import BarkConfig

TIMELINE_COLUMNS = [
    "time_sec",
    "dog_score",
    "bark_score",
    "speech_score",
    "music_score",
    "combined_bark_score",
]


def _combined_bark_score(
    dog: pd.Series,
    bark: pd.Series,
    cfg: BarkConfig,
) -> pd.Series:
    mode = cfg.combined_bark_mode
    if mode == "bark":
        return bark
    if mode == "max_bark_dog":
        return pd.concat([bark, dog], axis=1).max(axis=1)
    raise ValueError(
        f"unknown combined_bark_mode {mode!r}; expected 'bark' or 'max_bark_dog'"
    )


def build_timeline(panns_scores_df: pd.DataFrame, cfg: BarkConfig) -> pd.DataFrame:
    """One row per PANNs window; time_sec = center_time_sec."""
    required = {
        "center_time_sec",
        "dog_score",
        "bark_score",
        "speech_score",
        "music_score",
    }
    missing = required - set(panns_scores_df.columns)
    if missing:
        raise ValueError(f"panns_scores_df missing columns: {sorted(missing)}")

    df = panns_scores_df.sort_values("center_time_sec").reset_index(drop=True)
    out = pd.DataFrame(
        {
            "time_sec": df["center_time_sec"].astype(float),
            "dog_score": df["dog_score"].astype(float),
            "bark_score": df["bark_score"].astype(float),
            "speech_score": df["speech_score"].astype(float),
            "music_score": df["music_score"].astype(float),
        }
    )
    out["combined_bark_score"] = _combined_bark_score(
        out["dog_score"], out["bark_score"], cfg
    )
    return out[TIMELINE_COLUMNS]
