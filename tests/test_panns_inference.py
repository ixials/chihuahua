"""Unit tests for PANNs inference helpers (no model download)."""

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile

from bark_detection.config import BarkConfig
from bark_detection.panns_inference import (
    _load_wav_mono,
    merge_window_scores,
    normalize_audio_peak,
)


def _score_row(
    window_id: int,
    start: float,
    end: float,
    *,
    dog: float = 0.0,
    bark: float = 0.0,
    animal: float = 0.0,
    speech: float = 0.0,
    music: float = 0.0,
) -> dict:
    return {
        "window_id": window_id,
        "start_time_sec": start,
        "end_time_sec": end,
        "center_time_sec": (start + end) / 2.0,
        "dog_score": dog,
        "bark_score": bark,
        "animal_score": animal,
        "speech_score": speech,
        "music_score": music,
    }


def test_normalize_audio_peak_scales_to_target():
    audio = np.array([0.25, -0.5, 0.125], dtype=np.float32)
    normalized = normalize_audio_peak(audio, target_peak=0.9)

    assert float(np.max(np.abs(normalized))) == pytest.approx(0.9)
    assert normalized[1] == pytest.approx(-0.9)
    assert normalized[0] == pytest.approx(0.45)


def test_normalize_audio_peak_silent_noop():
    audio = np.zeros(4, dtype=np.float32)
    normalized = normalize_audio_peak(audio, target_peak=0.9)
    np.testing.assert_array_equal(normalized, audio)


def test_load_wav_mono_normalizes_when_enabled(tmp_path):
    wav_path = tmp_path / "quiet.wav"
    samples = np.array([8192, -16384, 4096], dtype=np.int16)
    scipy.io.wavfile.write(wav_path, 16000, samples)

    cfg = BarkConfig(audio_normalize=True, audio_normalize_target_peak=0.9)
    audio, sr = _load_wav_mono(wav_path, cfg)

    assert sr == 16000
    assert float(np.max(np.abs(audio))) == pytest.approx(0.9)


def test_load_wav_mono_skips_normalization_when_disabled(tmp_path):
    wav_path = tmp_path / "quiet.wav"
    samples = np.array([8192, -16384, 4096], dtype=np.int16)
    scipy.io.wavfile.write(wav_path, 16000, samples)

    cfg = BarkConfig(audio_normalize=False)
    audio, _ = _load_wav_mono(wav_path, cfg)

    assert float(np.max(np.abs(audio))) == pytest.approx(16384 / 32768.0)


def test_merge_window_scores_takes_max_per_column():
    long_scores = pd.DataFrame(
        [
            _score_row(0, 0.0, 1.0, dog=0.2, bark=0.3, speech=0.1),
            _score_row(1, 1.0, 2.0, dog=0.5, bark=0.4, speech=0.2),
        ]
    )
    short_scores = pd.DataFrame(
        [
            _score_row(0, 0.0, 0.5, dog=0.8, bark=0.1, speech=0.05),
            _score_row(1, 0.5, 1.0, dog=0.1, bark=0.9, speech=0.3),
            _score_row(2, 1.0, 1.5, dog=0.6, bark=0.7, speech=0.15),
        ]
    )

    merged = merge_window_scores(long_scores, short_scores)

    assert len(merged) == 2
    assert float(merged.iloc[0]["dog_score"]) == pytest.approx(0.8)
    assert float(merged.iloc[0]["bark_score"]) == pytest.approx(0.9)
    assert float(merged.iloc[0]["speech_score"]) == pytest.approx(0.3)
    assert float(merged.iloc[1]["dog_score"]) == pytest.approx(0.6)
    assert float(merged.iloc[1]["bark_score"]) == pytest.approx(0.7)
    assert int(merged.iloc[0]["window_id"]) == 0
    assert float(merged.iloc[0]["start_time_sec"]) == 0.0


def test_merge_window_scores_no_overlap_keeps_long_values():
    long_scores = pd.DataFrame([_score_row(0, 0.0, 1.0, bark=0.4)])
    short_scores = pd.DataFrame([_score_row(0, 2.0, 2.5, bark=0.95)])

    merged = merge_window_scores(long_scores, short_scores)

    assert float(merged.iloc[0]["bark_score"]) == pytest.approx(0.4)
