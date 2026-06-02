import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile

from bark_detection.config import BarkConfig


@pytest.fixture
def cfg() -> BarkConfig:
    return BarkConfig()


def make_timeline_row(
    time_sec: float,
    *,
    bark: float = 0.0,
    dog: float = 0.0,
    speech: float = 0.0,
    music: float = 0.0,
    combined: float | None = None,
) -> dict:
    c = combined if combined is not None else bark
    return {
        "time_sec": time_sec,
        "dog_score": dog,
        "bark_score": bark,
        "speech_score": speech,
        "music_score": music,
        "combined_bark_score": c,
    }


def timeline_from_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def write_mono_wav(path, duration_sec: float, sr: int = 16000) -> None:
    n = int(round(duration_sec * sr))
    samples = np.zeros(n, dtype=np.int16)
    scipy.io.wavfile.write(path, sr, samples)
