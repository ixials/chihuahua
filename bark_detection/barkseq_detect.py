"""M5: Detect Barkseq regions from a bark score timeline."""

from __future__ import annotations

import pandas as pd

from bark_detection.config import BarkConfig

BARKSEQ_COLUMNS = [
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
]


def _positive_regions(
    timeline_df: pd.DataFrame,
    duration_sec: float,
    cfg: BarkConfig,
) -> list[dict]:
    """Expand each above-threshold timeline point to [center ± hop/2]."""
    hop_half = cfg.hop_size_sec / 2.0
    threshold = cfg.barkseq_threshold

    df = timeline_df.sort_values("time_sec").reset_index(drop=True)
    mask = df["combined_bark_score"] > threshold

    regions: list[dict] = []
    for idx in df.index[mask]:
        center = float(df.loc[idx, "time_sec"])
        start = max(0.0, center - hop_half)
        end = min(float(duration_sec), center + hop_half)
        regions.append({"start_time_sec": start, "end_time_sec": end})

    return regions


def _merge_regions(regions: list[dict], merge_gap_sec: float) -> list[dict]:
    """Merge regions whose inter-region gap is <= merge_gap_sec."""
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r["start_time_sec"])
    merged: list[dict] = [dict(sorted_regions[0])]

    for region in sorted_regions[1:]:
        gap = float(region["start_time_sec"]) - float(merged[-1]["end_time_sec"])
        if gap <= merge_gap_sec:
            merged[-1]["end_time_sec"] = max(
                float(merged[-1]["end_time_sec"]),
                float(region["end_time_sec"]),
            )
        else:
            merged.append(dict(region))

    return merged


def _aggregate_region(
    timeline_df: pd.DataFrame,
    start_time_sec: float,
    end_time_sec: float,
) -> dict:
    """Compute peak time and score aggregates over timeline samples in the span."""
    span = timeline_df[
        (timeline_df["time_sec"] >= start_time_sec)
        & (timeline_df["time_sec"] <= end_time_sec)
    ]
    if span.empty:
        raise ValueError(
            f"no timeline samples in [{start_time_sec}, {end_time_sec}]"
        )

    peak_idx = span["combined_bark_score"].idxmax()
    peak_time_sec = float(span.loc[peak_idx, "time_sec"])

    return {
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "peak_time_sec": peak_time_sec,
        "duration_sec": round(end_time_sec - start_time_sec, 6),
        "max_dog_score": float(span["dog_score"].max()),
        "mean_dog_score": float(span["dog_score"].mean()),
        "max_bark_score": float(span["bark_score"].max()),
        "mean_bark_score": float(span["bark_score"].mean()),
        "max_combined_bark_score": float(span["combined_bark_score"].max()),
        "mean_combined_bark_score": float(span["combined_bark_score"].mean()),
    }


def detect_barkseqs(
    timeline_df: pd.DataFrame,
    duration_sec: float,
    cfg: BarkConfig,
) -> pd.DataFrame:
    """Find merged Barkseq regions where combined_bark_score exceeds threshold."""
    required = {
        "time_sec",
        "dog_score",
        "bark_score",
        "combined_bark_score",
    }
    missing = required - set(timeline_df.columns)
    if missing:
        raise ValueError(f"timeline_df missing columns: {sorted(missing)}")

    df = timeline_df.sort_values("time_sec").reset_index(drop=True)
    raw_regions = _positive_regions(df, duration_sec, cfg)
    merged_regions = _merge_regions(raw_regions, cfg.merge_gap_sec)

    rows: list[dict] = []
    for region in merged_regions:
        rows.append(
            _aggregate_region(
                df,
                float(region["start_time_sec"]),
                float(region["end_time_sec"]),
            )
        )

    if not rows:
        return pd.DataFrame(columns=BARKSEQ_COLUMNS)

    out = pd.DataFrame(rows)
    out.insert(0, "barkseq_id", range(len(out)))
    return out[BARKSEQ_COLUMNS]
