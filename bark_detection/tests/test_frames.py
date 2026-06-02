import pandas as pd

from bark_detection.config import BarkConfig
from bark_detection.frames import align_frames, time_to_frame


def test_time_to_frame_round_and_clamp():
    assert time_to_frame(1.75, fps=30, frame_count=150) == 52
    assert time_to_frame(-1.0, fps=30, frame_count=150) == 0
    assert time_to_frame(999.0, fps=30, frame_count=150) == 149


def test_align_frames_dogs1_values():
    cfg = BarkConfig()
    cfg.pre_context_sec = 0.5
    cfg.post_context_sec = 0.5

    barkseqs = pd.DataFrame(
        [
            {
                "barkseq_id": 1,
                "start_time_sec": 1.375,
                "end_time_sec": 2.125,
                "peak_time_sec": 1.75,
            }
        ]
    )
    out = align_frames(barkseqs, fps=30.0, frame_count=150, cfg=cfg)
    row = out.iloc[0]

    assert int(row["peak_frame"]) == round(1.75 * 30)
    assert 0 <= int(row["start_frame"]) <= 149
    assert 0 <= int(row["end_frame"]) <= 149
    assert int(row["context_start_frame"]) == 26
    assert int(row["context_end_frame"]) == 79


def test_context_start_clamps_at_zero():
    cfg = BarkConfig()
    cfg.pre_context_sec = 0.5
    cfg.post_context_sec = 0.5

    barkseqs = pd.DataFrame(
        [
            {
                "barkseq_id": 0,
                "start_time_sec": 0.375,
                "end_time_sec": 0.625,
                "peak_time_sec": 0.5,
            }
        ]
    )
    out = align_frames(barkseqs, fps=30.0, frame_count=150, cfg=cfg)
    assert int(out.iloc[0]["context_start_frame"]) == 0
