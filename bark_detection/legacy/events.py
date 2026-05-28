"""M5: Merge nearby candidates and apply duration rules."""

from __future__ import annotations

import pandas as pd

from bark_detection.config import BarkConfig


def merge_candidates(candidates_df: pd.DataFrame, cfg: BarkConfig) -> pd.DataFrame:
    """Merge candidate rows whose inter-event gap is <= cfg.merge_gap_ms.

    A "gap" is next.start_time_sec - prev.end_time_sec. Candidates are
    iterated in sorted-by-start order (input should already be sorted; sort
    defensively).

    When merging a chain of candidates [c_i, c_{i+1}, ...]:
      - start_time_sec = first.start_time_sec
      - end_time_sec   = last.end_time_sec
      - peak_time_sec  = peak_time_sec of the candidate with the largest rms_peak
      - rms_peak       = max(rms_peak of all merged candidates)
      - normalized_rms_peak = max(normalized_rms_peak)
      - duration_sec   = end_time_sec - start_time_sec
      - merged_from    = [list of original candidate_id values that fed this event]
                          (single-candidate events use a one-element list, e.g. [0])

    Returns a DataFrame with the same columns as the input plus `merged_from`,
    AND a new sequential `merged_id` column (0-based).
    """
    if candidates_df.empty:
        result = candidates_df.copy()
        result["merged_from"] = pd.Series(dtype=object)
        result["merged_id"] = pd.Series(dtype=int)
        return result

    df = candidates_df.sort_values("start_time_sec").reset_index(drop=True)
    gap_threshold_sec = cfg.merge_gap_ms / 1000.0

    merged_rows: list[dict] = []

    # Running buffer: accumulate candidates that belong to the same merged event.
    buf_ids: list[int] = [int(df.loc[0, "candidate_id"])]
    buf_start: float = float(df.loc[0, "start_time_sec"])
    buf_end: float = float(df.loc[0, "end_time_sec"])
    buf_peak_time: float = float(df.loc[0, "peak_time_sec"])
    buf_rms_peak: float = float(df.loc[0, "rms_peak"])
    buf_norm_peak: float = float(df.loc[0, "normalized_rms_peak"])

    def flush_buffer() -> dict:
        return {
            "start_time_sec": buf_start,
            "peak_time_sec": buf_peak_time,
            "end_time_sec": buf_end,
            "duration_sec": round(buf_end - buf_start, 6),
            "rms_peak": buf_rms_peak,
            "normalized_rms_peak": buf_norm_peak,
            "merged_from": list(buf_ids),
        }

    for i in range(1, len(df)):
        row = df.loc[i]
        gap = float(row["start_time_sec"]) - buf_end

        if gap <= gap_threshold_sec:
            # Extend the current buffer.
            buf_ids.append(int(row["candidate_id"]))
            buf_end = float(row["end_time_sec"])
            if float(row["rms_peak"]) > buf_rms_peak:
                buf_rms_peak = float(row["rms_peak"])
                buf_peak_time = float(row["peak_time_sec"])
            if float(row["normalized_rms_peak"]) > buf_norm_peak:
                buf_norm_peak = float(row["normalized_rms_peak"])
        else:
            # Flush the buffer and start a new one.
            merged_rows.append(flush_buffer())
            buf_ids = [int(row["candidate_id"])]
            buf_start = float(row["start_time_sec"])
            buf_end = float(row["end_time_sec"])
            buf_peak_time = float(row["peak_time_sec"])
            buf_rms_peak = float(row["rms_peak"])
            buf_norm_peak = float(row["normalized_rms_peak"])

    merged_rows.append(flush_buffer())

    result = pd.DataFrame(merged_rows)
    result.insert(0, "merged_id", range(len(result)))
    return result


def apply_duration_rules(
    merged_df: pd.DataFrame, cfg: BarkConfig
) -> tuple[pd.DataFrame, list[dict]]:
    """Classify each merged event by duration.

    For each row:
      - duration < cfg.min_duration_ms / 1000  => DROP. Record in dropped list:
          {"merged_id": ..., "merged_from": [...], "duration_sec": ...,
           "reason": "below_min_duration"}.
      - duration > cfg.max_duration_ms / 1000  => KEEP, set event_type="long_event".
          NOTE: Internal splitting by sub-peaks is a future enhancement (M5+ or later).
          For now, flagging is sufficient.
      - otherwise                              => KEEP, set event_type="bark".

    Returns (kept_df, dropped_list).
    The kept_df gets a new sequential `event_id` column (0-based) and an
    `event_type` column with values {"bark", "long_event"}. Preserves `merged_from`
    and all peak / duration / rms columns.
    """
    min_dur = cfg.min_duration_ms / 1000.0
    max_dur = cfg.max_duration_ms / 1000.0

    kept_rows: list[dict] = []
    dropped: list[dict] = []

    for _, row in merged_df.iterrows():
        dur = float(row["duration_sec"])
        if dur < min_dur:
            dropped.append(
                {
                    "merged_id": int(row["merged_id"]),
                    "merged_from": row["merged_from"],
                    "duration_sec": dur,
                    "reason": "below_min_duration",
                }
            )
        else:
            event_type = "long_event" if dur > max_dur else "bark"
            kept_rows.append(
                {
                    "start_time_sec": float(row["start_time_sec"]),
                    "peak_time_sec": float(row["peak_time_sec"]),
                    "end_time_sec": float(row["end_time_sec"]),
                    "duration_sec": dur,
                    "rms_peak": float(row["rms_peak"]),
                    "normalized_rms_peak": float(row["normalized_rms_peak"]),
                    "event_type": event_type,
                    "merged_from": row["merged_from"],
                }
            )

    if kept_rows:
        kept_df = pd.DataFrame(kept_rows)
        kept_df.insert(0, "event_id", range(len(kept_df)))
    else:
        kept_df = pd.DataFrame(
            columns=[
                "event_id",
                "start_time_sec",
                "peak_time_sec",
                "end_time_sec",
                "duration_sec",
                "rms_peak",
                "normalized_rms_peak",
                "event_type",
                "merged_from",
            ]
        )

    return kept_df, dropped
