"""Optional smoke test: synthetic pipeline through non-PANNs stages."""

import pytest
import pandas as pd

from bark_detection.barkseq_detect import detect_barkseqs
from bark_detection.barkseq_export import export_barkseqs
from bark_detection.config import BarkConfig
from bark_detection.panns_windows import generate_windows
from bark_detection.timeline import build_timeline
from tests.conftest import make_timeline_row, timeline_from_rows


@pytest.mark.skip(reason="integration smoke; run manually with pytest -k integration_smoke --runxfail")
def test_synthetic_pipeline_without_panns():
    cfg = BarkConfig()
    duration = 3.0

    windows = generate_windows(duration, cfg)
    assert len(windows) > 0

    fake_scores = windows.copy()
    fake_scores["dog_score"] = 0.1
    fake_scores["bark_score"] = 0.1
    fake_scores["speech_score"] = 0.05
    fake_scores["music_score"] = 0.05
    fake_scores.loc[fake_scores["center_time_sec"] == 0.5, "bark_score"] = 0.55
    fake_scores.loc[fake_scores["center_time_sec"] == 0.5, "dog_score"] = 0.7

    timeline = build_timeline(fake_scores, cfg)
    initial = detect_barkseqs(timeline, duration, cfg)
    assert len(initial) >= 1

    export = export_barkseqs(initial, timeline, cfg)
    assert "confidence" in export.columns
    assert (export["confidence"] >= 0).all()
    assert (export["confidence"] <= 1).all()
