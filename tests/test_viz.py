import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd

from bark_detection.config import BarkConfig
from bark_detection import viz
from bark_detection.inspect_labels import INSPECT_LABELS, label_to_column


def test_debug_plots_created(tmp_path: Path):
    cfg = BarkConfig()
    timeline = pd.DataFrame(
        {
            "time_sec": [0.5, 1.0, 1.5],
            "dog_score": [0.7, 0.2, 0.6],
            "bark_score": [0.49, 0.03, 0.55],
            "speech_score": [0.05, 0.15, 0.04],
            "music_score": [0.03, 0.02, 0.04],
            "combined_bark_score": [0.49, 0.03, 0.55],
        }
    )
    barkseqs = pd.DataFrame(
        [
            {
                "barkseq_id": 0,
                "start_time_sec": 0.375,
                "end_time_sec": 0.625,
                "peak_time_sec": 0.5,
                "max_combined_bark_score": 0.49,
            },
            {
                "barkseq_id": 1,
                "start_time_sec": 1.375,
                "end_time_sec": 2.125,
                "peak_time_sec": 1.75,
                "max_combined_bark_score": 0.55,
            },
        ]
    )

    timeline_png = tmp_path / "panns_score_timeline.png"
    overlay_png = tmp_path / "barkseq_overlay.png"

    viz.plot_panns_score_timeline(timeline, timeline_png, cfg)
    viz.plot_barkseq_overlay(timeline, barkseqs, overlay_png, cfg)

    assert timeline_png.is_file()
    assert overlay_png.is_file()
    assert timeline_png.stat().st_size > 0
    assert overlay_png.stat().st_size > 0


def test_plot_vocalization_inspect_creates_file(tmp_path: Path):
    cfg = BarkConfig()

    # Build a minimal vocalization_scores DataFrame
    rows = []
    for i, center in enumerate([0.5, 1.0, 1.5]):
        row = {
            "window_id": i,
            "start_time_sec": center - 0.5,
            "end_time_sec": center + 0.5,
            "center_time_sec": center,
        }
        for label in INSPECT_LABELS:
            row[label_to_column(label)] = 0.1 * (i + 1)
        row["vocalization_max_score"] = 0.1 * (i + 1)
        row["max_bark_dog_score"] = 0.1 * (i + 1)
        rows.append(row)
    df = pd.DataFrame(rows)

    out_png = tmp_path / "vocalization_inspect.png"
    viz.plot_vocalization_inspect(df, out_png, cfg, title="test inspect plot")

    assert out_png.is_file()
    assert out_png.stat().st_size > 0
