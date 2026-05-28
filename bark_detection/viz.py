"""Debug visualization for the bark detection pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bark_detection.config import BarkConfig

SCORE_SERIES = [
    ("combined_bark_score", "#E53935", 2.0, "combined_bark"),
    ("bark_score", "#FB8C00", 1.2, "bark"),
    ("dog_score", "#43A047", 1.0, "dog"),
    ("speech_score", "#1E88E5", 1.0, "speech"),
    ("music_score", "#8E24AA", 1.0, "music"),
]


def _plot_score_lines(ax: plt.Axes, timeline_df: pd.DataFrame) -> None:
    for column, color, width, label in SCORE_SERIES:
        if column in timeline_df.columns:
            ax.plot(
                timeline_df["time_sec"],
                timeline_df[column],
                linewidth=width,
                color=color,
                label=label,
                alpha=0.9 if column == "combined_bark_score" else 0.75,
            )


def plot_panns_score_timeline(
    timeline_df: pd.DataFrame,
    out_path: Path,
    cfg: BarkConfig,
    *,
    title: str | None = None,
) -> None:
    """Plot PANNs score timeline with barkseq threshold line."""
    fig, ax = plt.subplots(figsize=(12, 4))
    _plot_score_lines(ax, timeline_df)

    ax.axhline(
        cfg.barkseq_threshold,
        linestyle="--",
        linewidth=1.2,
        color="#424242",
        label=f"barkseq threshold ({cfg.barkseq_threshold:.2f})",
    )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or "PANNs score timeline")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_barkseq_overlay(
    timeline_df: pd.DataFrame,
    barkseqs_df: pd.DataFrame,
    out_path: Path,
    cfg: BarkConfig,
    *,
    title: str | None = None,
) -> None:
    """Plot score timeline with shaded Barkseq regions and peak markers."""
    fig, ax = plt.subplots(figsize=(12, 4))
    _plot_score_lines(ax, timeline_df)

    ax.axhline(
        cfg.barkseq_threshold,
        linestyle="--",
        linewidth=1.2,
        color="#424242",
        label=f"barkseq threshold ({cfg.barkseq_threshold:.2f})",
    )

    shade_colors = ["#FFCDD2", "#C8E6C9"]
    for i, row in barkseqs_df.iterrows():
        start = float(row["start_time_sec"])
        end = float(row["end_time_sec"])
        peak = float(row["peak_time_sec"])
        barkseq_id = int(row["barkseq_id"])
        color = shade_colors[int(barkseq_id) % len(shade_colors)]

        ax.axvspan(start, end, alpha=0.35, color=color, label=f"Barkseq {barkseq_id}")
        ax.axvline(peak, linestyle=":", linewidth=1.5, color=color, alpha=0.9)
        ax.scatter(
            [peak],
            [float(row["max_combined_bark_score"])],
            s=40,
            color=color,
            edgecolors="#212121",
            linewidths=0.6,
            zorder=5,
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or "PANNs scores with Barkseq regions")
    ax.grid(True, alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles: list = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rms(
    rms_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str | None = None,
    threshold: float | None = None,
    threshold_label: str | None = None,
) -> None:
    """Plot raw RMS (and optionally smoothed RMS + threshold line).

    If `rms_df` contains a column `rms_smoothed`, it is overlaid as a second
    line.  Raw `rms` is always drawn (lighter / thinner) as the first line.

    If `threshold` is not None, a horizontal dashed line is drawn.

    Always includes legend, grid, x-label "time (s)", y-label "RMS".
    Saves to `out_path` as a PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 3))

    # Raw RMS — lighter, thin
    ax.plot(
        rms_df["time_sec"],
        rms_df["rms"],
        linewidth=0.7,
        color="#90CAF9",
        alpha=0.8,
        label="raw",
    )

    # Smoothed RMS — if present
    if "rms_smoothed" in rms_df.columns:
        ax.plot(
            rms_df["time_sec"],
            rms_df["rms_smoothed"],
            linewidth=1.4,
            color="#1565C0",
            label="smoothed",
        )

    # Threshold line — if provided
    if threshold is not None:
        label = threshold_label if threshold_label is not None else f"threshold = {threshold:.4f}"
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.2,
            color="#E53935",
            label=label,
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMS")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
