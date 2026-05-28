"""M6: Flag Barkseqs likely contaminated by speech or music."""

from __future__ import annotations

import pandas as pd

from bark_detection.config import BarkConfig

NOISE_COLUMNS = [
    "max_speech_score",
    "max_music_score",
    "noise_flag",
    "noise_reason",
]


def _span_slice(
    timeline_df: pd.DataFrame,
    start_time_sec: float,
    end_time_sec: float,
) -> pd.DataFrame:
    return timeline_df[
        (timeline_df["time_sec"] >= start_time_sec)
        & (timeline_df["time_sec"] <= end_time_sec)
    ]


def _classify_noise(
    max_speech: float,
    max_music: float,
    max_combined: float,
    cfg: BarkConfig,
) -> tuple[bool, str]:
    high_speech = max_speech >= cfg.speech_noise_threshold
    high_music = max_music >= cfg.music_noise_threshold
    strong_bark = max_combined >= cfg.barkseq_threshold

    if not high_speech and not high_music:
        return False, "clean"

    if strong_bark:
        if high_speech and high_music:
            return True, "high_speech_and_music"
        if high_speech:
            return True, "high_speech"
        return True, "high_music"

    if high_speech or high_music:
        return True, "likely_noise"

    return False, "clean"


def flag_noise(
    barkseqs_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    cfg: BarkConfig,
) -> pd.DataFrame:
    """Add noise fields to each Barkseq row. Does not drop rows."""
    required_timeline = {"time_sec", "speech_score", "music_score"}
    missing = required_timeline - set(timeline_df.columns)
    if missing:
        raise ValueError(f"timeline_df missing columns: {sorted(missing)}")

    if barkseqs_df.empty:
        out = barkseqs_df.copy()
        for col in NOISE_COLUMNS:
            out[col] = pd.Series(dtype=object if col == "noise_reason" else float)
        return out

    timeline = timeline_df.sort_values("time_sec").reset_index(drop=True)
    rows: list[dict] = []

    for _, row in barkseqs_df.iterrows():
        start = float(row["start_time_sec"])
        end = float(row["end_time_sec"])
        span = _span_slice(timeline, start, end)
        if span.empty:
            raise ValueError(f"no timeline samples in [{start}, {end}]")

        max_speech = float(span["speech_score"].max())
        max_music = float(span["music_score"].max())
        max_combined = float(row["max_combined_bark_score"])
        noise_flag, noise_reason = _classify_noise(
            max_speech, max_music, max_combined, cfg
        )
        rows.append(
            {
                "max_speech_score": max_speech,
                "max_music_score": max_music,
                "noise_flag": noise_flag,
                "noise_reason": noise_reason,
            }
        )

    out = barkseqs_df.copy()
    noise_df = pd.DataFrame(rows)
    for col in NOISE_COLUMNS:
        out[col] = noise_df[col].values
    return out
