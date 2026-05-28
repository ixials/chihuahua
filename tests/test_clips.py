from pathlib import Path

import pandas as pd
import pytest
import scipy.io.wavfile

from bark_detection.clips import compute_clip_bounds, extract_clips
from bark_detection.config import BarkConfig
from tests.conftest import write_mono_wav


def _barkseqs_two_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "barkseq_id": 0,
                "start_time_sec": 0.375,
                "end_time_sec": 0.625,
                "peak_time_sec": 0.5,
                "duration_sec": 0.25,
            },
            {
                "barkseq_id": 1,
                "start_time_sec": 1.375,
                "end_time_sec": 2.125,
                "peak_time_sec": 1.75,
                "duration_sec": 0.75,
            },
        ]
    )


def test_clip_extraction_padded(tmp_path: Path):
    cfg = BarkConfig()
    cfg.clip_pre_context_sec = 0.25
    cfg.clip_post_context_sec = 0.25
    cfg.prevent_clip_overlap = True

    wav_path = tmp_path / "audio.wav"
    write_mono_wav(wav_path, duration_sec=5.014063)

    barkseqs = _barkseqs_two_events()
    out_dir = tmp_path / "bark_event_clips"
    annotated, written = extract_clips(
        barkseqs, wav_path, out_dir, audio_duration_sec=5.014063, cfg=cfg
    )

    assert len(written) == 2
    for p in written:
        assert p.is_file()
        sr, data = scipy.io.wavfile.read(p)
        assert sr == 16000
        assert len(data) > 0

    row0 = annotated.iloc[0]
    assert row0["clip_path"] == "bark_event_clips/barkseq_000.wav"
    assert float(row0["clip_start_time_sec"]) == pytest.approx(0.125, abs=1e-6)
    assert float(row0["clip_end_time_sec"]) == pytest.approx(0.875, abs=1e-6)

    dur0 = len(scipy.io.wavfile.read(written[0])[1]) / 16000
    assert dur0 == pytest.approx(float(row0["clip_duration_sec"]), abs=0.01)

    sr1, data1 = scipy.io.wavfile.read(written[1])
    assert len(data1) / sr1 == pytest.approx(float(annotated.iloc[1]["clip_duration_sec"]), abs=0.01)


def test_neighbor_clips_do_not_overlap_when_gap_allows():
    cfg = BarkConfig()
    cfg.clip_pre_context_sec = 0.25
    cfg.clip_post_context_sec = 0.25
    cfg.prevent_clip_overlap = True

    barkseqs = _barkseqs_two_events()
    bounds = compute_clip_bounds(barkseqs, audio_duration_sec=5.014063, cfg=cfg)

    end0 = float(bounds.iloc[0]["clip_end_time_sec"])
    start1 = float(bounds.iloc[1]["clip_start_time_sec"])
    assert end0 <= start1
