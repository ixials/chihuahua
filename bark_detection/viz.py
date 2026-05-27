"""Debug visualization for the bark detection pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
