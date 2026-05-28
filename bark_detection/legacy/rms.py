"""M2: framed RMS energy curve."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile

from bark_detection.config import BarkConfig


def load_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM mono WAV via scipy.io.wavfile.

    Returns (float32 in [-1, 1], sample_rate).
    """
    sample_rate, data = scipy.io.wavfile.read(wav_path)
    if data.ndim > 1:
        # Take the first channel if unexpectedly stereo
        data = data[:, 0]
    # Convert int16 → float32 in [-1, 1]
    data = data.astype(np.float32) / 32768.0
    return data, sample_rate


def compute_rms(wav: np.ndarray, sample_rate: int, cfg: BarkConfig) -> pd.DataFrame:
    """Framed RMS over `wav` using cfg.window_ms / cfg.hop_ms.

    Returns a DataFrame with columns `time_sec`, `rms`.  time_sec is the
    center of each frame (start_sample + window/2) / sample_rate.
    """
    window_samples = int(round(cfg.window_ms * sample_rate / 1000))
    hop_samples = int(round(cfg.hop_ms * sample_rate / 1000))

    num_samples = len(wav)

    # Build a sliding window view: shape (num_samples - window_samples + 1, window_samples)
    view = np.lib.stride_tricks.sliding_window_view(wav, window_shape=window_samples)

    # Take every hop_samples-th frame starting at index 0
    view_hopped = view[::hop_samples]  # shape (n_frames, window_samples)

    rms_values = np.sqrt(np.mean(view_hopped ** 2, axis=1)).astype(np.float32)

    n_frames = view_hopped.shape[0]
    expected_frames = (num_samples - window_samples) // hop_samples + 1
    assert n_frames == expected_frames, (
        f"RMS array length mismatch: got {n_frames}, expected {expected_frames}"
    )

    assert np.all(rms_values >= 0), "Some RMS values are negative"
    assert np.all(np.isfinite(rms_values)), "Some RMS values are non-finite"

    # time_sec = center of each frame
    start_samples = np.arange(n_frames) * hop_samples
    center_samples = start_samples + window_samples / 2.0
    time_sec = (center_samples / sample_rate).astype(np.float32)

    return pd.DataFrame({"time_sec": time_sec, "rms": rms_values})
