"""M9: Map Barkseq timestamps to video frame indices."""

from __future__ import annotations

import pandas as pd

from bark_detection.config import BarkConfig

FRAME_COLUMNS = [
    "start_frame",
    "end_frame",
    "peak_frame",
    "context_start_frame",
    "context_end_frame",
]


def time_to_frame(time_sec: float, fps: float, frame_count: int) -> int:
    """frame_id = round(time_sec * fps), clamped to [0, frame_count - 1]."""
    frame_id = int(round(time_sec * fps))
    return max(0, min(frame_id, frame_count - 1))


def align_frames(
    barkseqs_df: pd.DataFrame,
    fps: float,
    frame_count: int,
    cfg: BarkConfig,
) -> pd.DataFrame:
    """Add frame index columns to each Barkseq row."""
    if barkseqs_df.empty:
        out = barkseqs_df.copy()
        for col in FRAME_COLUMNS:
            out[col] = pd.Series(dtype=int)
        return out

    rows: list[dict] = []
    for _, row in barkseqs_df.iterrows():
        start_sec = float(row["start_time_sec"])
        end_sec = float(row["end_time_sec"])
        peak_sec = float(row["peak_time_sec"])
        ctx_start_sec = start_sec - cfg.pre_context_sec
        ctx_end_sec = end_sec + cfg.post_context_sec

        rows.append(
            {
                "start_frame": time_to_frame(start_sec, fps, frame_count),
                "end_frame": time_to_frame(end_sec, fps, frame_count),
                "peak_frame": time_to_frame(peak_sec, fps, frame_count),
                "context_start_frame": time_to_frame(
                    ctx_start_sec, fps, frame_count
                ),
                "context_end_frame": time_to_frame(ctx_end_sec, fps, frame_count),
            }
        )

    out = barkseqs_df.copy()
    frame_df = pd.DataFrame(rows)
    for col in FRAME_COLUMNS:
        out[col] = frame_df[col].values
    return out
