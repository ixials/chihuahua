"""Phase 2: per-label vocalization score inspector.

Runs PANNs per window and exports individual label scores for diagnosis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bark_detection.config import BarkConfig
from bark_detection.panns_inference import VOCALIZATION_LABELS
from bark_detection.paths import intermediate_dir

# Full inspect set: all six vocalization labels + bare "Dog"
INSPECT_LABELS: list[str] = VOCALIZATION_LABELS + ["Dog"]


def label_to_column(label: str) -> str:
    """Convert a PANNs display label to a DataFrame column slug.

    Examples:
        "Bark"          -> "bark_score"
        "Bow-wow"       -> "bow_wow_score"
        "Whimper (dog)" -> "whimper_dog_score"
        "Dog"           -> "dog_score"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{slug}_score"


def inspect_score_columns() -> list[str]:
    """Return the ordered list of per-label score column names for INSPECT_LABELS."""
    return [label_to_column(label) for label in INSPECT_LABELS]


def nearest_window_row(df: pd.DataFrame, time_sec: float) -> pd.Series:
    """Return the row whose center_time_sec is closest to *time_sec*."""
    idx = (df["center_time_sec"] - time_sec).abs().idxmin()
    return df.loc[idx]


def format_window_summary(
    row: pd.Series,
    cfg: BarkConfig,
    query_time_sec: Optional[float] = None,
) -> str:
    """Return a human-readable diagnosis string for a single window row."""
    lines: list[str] = []

    if query_time_sec is not None:
        lines.append(
            f"Query: {query_time_sec:.3f}s  ->  nearest window center: {row['center_time_sec']:.3f}s"
        )
    else:
        lines.append(f"Window center: {row['center_time_sec']:.3f}s")

    lines.append(
        f"  window_id={int(row['window_id'])}"
        f"  [{row['start_time_sec']:.3f}s, {row['end_time_sec']:.3f}s]"
    )
    lines.append(
        f"  vocalization_max_score={row['vocalization_max_score']:.4f}"
        f"  max_bark_dog_score={row['max_bark_dog_score']:.4f}"
        f"  threshold={cfg.barkseq_threshold:.2f}"
    )

    for label in INSPECT_LABELS:
        col = label_to_column(label)
        score = float(row[col])
        marker = " *" if score >= cfg.barkseq_threshold else ""
        lines.append(f"    {label:<20s} {score:.4f}{marker}")

    return "\n".join(lines)


def print_inspect_summaries(
    df: pd.DataFrame,
    at_times: list[float],
    cfg: BarkConfig,
) -> None:
    """Print a window summary for each queried time."""
    for t in at_times:
        row = nearest_window_row(df, t)
        print(format_window_summary(row, cfg, query_time_sec=t))
        print()


def run_vocalization_inspect(
    run_dir: Path,
    cfg: BarkConfig,
    at_times: Optional[list[float]] = None,
    write_plot: bool = True,
) -> pd.DataFrame:
    """Run PANNs per window, export per-label vocalization scores, and optionally plot.

    Requires:
        - intermediate/panns_windows.csv
        - audio.wav

    Writes:
        - intermediate/vocalization_scores.csv
        - debug/vocalization_inspect.png  (if write_plot is True)

    Returns the vocalization scores DataFrame.
    """
    from bark_detection import viz
    from bark_detection.panns_inference import (
        _load_wav_mono,
        build_audio_tagging_16k,
        ensure_panns_assets,
        load_label_list,
        slice_window,
    )

    inter = intermediate_dir(run_dir)
    windows_path = inter / "panns_windows.csv"
    wav_path = run_dir / "audio.wav"

    if not windows_path.is_file():
        raise FileNotFoundError(f"missing panns_windows.csv: {windows_path}")
    if not wav_path.is_file():
        raise FileNotFoundError(f"missing audio.wav: {wav_path}")

    ensure_panns_assets()
    labels = load_label_list()
    name_to_index = {name: idx for idx, name in enumerate(labels)}

    missing = [l for l in INSPECT_LABELS if l not in name_to_index]
    if missing:
        raise KeyError(f"labels not found in PANNs label list: {missing}")

    label_indices: dict[str, int] = {l: name_to_index[l] for l in INSPECT_LABELS}
    vocalization_indices = [label_indices[l] for l in VOCALIZATION_LABELS]
    dog_col = label_to_column("Dog")

    tagger = build_audio_tagging_16k(cfg)
    wav, sr = _load_wav_mono(wav_path)
    if sr != cfg.target_sample_rate_hz:
        raise ValueError(
            f"expected sample rate {cfg.target_sample_rate_hz}, got {sr}"
        )

    windows_df = pd.read_csv(windows_path)

    rows: list[dict] = []
    for _, win in windows_df.iterrows():
        segment = slice_window(
            wav,
            sr,
            float(win["start_time_sec"]),
            float(win["end_time_sec"]),
            cfg.window_size_sec,
        )
        clipwise_output, _ = tagger.inference(segment[None, :])
        scores_arr = clipwise_output[0]

        row: dict = {
            "window_id": int(win["window_id"]),
            "start_time_sec": float(win["start_time_sec"]),
            "end_time_sec": float(win["end_time_sec"]),
            "center_time_sec": float(win["center_time_sec"]),
        }

        for label in INSPECT_LABELS:
            row[label_to_column(label)] = float(scores_arr[label_indices[label]])

        vocalization_max = float(np.max(scores_arr[vocalization_indices]))
        row["vocalization_max_score"] = vocalization_max
        row["max_bark_dog_score"] = max(vocalization_max, row[dog_col])

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = inter / "vocalization_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"vocalization scores: {len(df)} windows -> {out_path.name}")

    if at_times:
        print_inspect_summaries(df, at_times, cfg)

    if write_plot:
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        plot_path = debug_dir / "vocalization_inspect.png"
        viz.plot_vocalization_inspect(df, plot_path, cfg)
        print(f"vocalization inspect plot -> debug/vocalization_inspect.png")

    return df
