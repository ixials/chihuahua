"""M8: Extract one WAV clip per Barkseq with configurable audio context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile

from bark_detection.config import BarkConfig
from bark_detection.legacy.rms import load_wav_mono

CLIP_COLUMNS = [
    "clip_start_time_sec",
    "clip_end_time_sec",
    "clip_duration_sec",
    "clip_pre_context_actual_sec",
    "clip_post_context_actual_sec",
    "clip_path",
]


def _time_to_sample(time_sec: float, sample_rate: int) -> int:
    return int(round(time_sec * sample_rate))


def compute_clip_bounds(
    barkseqs_df: pd.DataFrame,
    audio_duration_sec: float,
    cfg: BarkConfig,
    clips_dir_name: str = "bark_event_clips",
) -> pd.DataFrame:
    """Compute padded clip time ranges; event start/end columns are unchanged."""
    if barkseqs_df.empty:
        out = barkseqs_df.copy()
        for col in CLIP_COLUMNS:
            out[col] = pd.Series(dtype=object if col == "clip_path" else float)
        return out

    df = barkseqs_df.sort_values("barkseq_id").reset_index(drop=True)
    n = len(df)
    rows: list[dict] = []

    for i in range(n):
        row = df.iloc[i]
        event_start = float(row["start_time_sec"])
        event_end = float(row["end_time_sec"])
        barkseq_id = int(row["barkseq_id"])

        clip_start = event_start - cfg.clip_pre_context_sec
        clip_end = event_end + cfg.clip_post_context_sec

        clip_start = max(0.0, clip_start)
        clip_end = min(float(audio_duration_sec), clip_end)

        if cfg.prevent_clip_overlap:
            if i > 0:
                prev_end = float(df.iloc[i - 1]["end_time_sec"])
                clip_start = max(clip_start, prev_end)
            if i < n - 1:
                next_start = float(df.iloc[i + 1]["start_time_sec"])
                clip_end = min(clip_end, next_start)

        if clip_end <= clip_start:
            clip_end = min(float(audio_duration_sec), clip_start + 1.0 / cfg.target_sample_rate_hz)

        pre_actual = event_start - clip_start
        post_actual = clip_end - event_end
        clip_duration = clip_end - clip_start

        rows.append(
            {
                "clip_start_time_sec": round(clip_start, 6),
                "clip_end_time_sec": round(clip_end, 6),
                "clip_duration_sec": round(clip_duration, 6),
                "clip_pre_context_actual_sec": round(pre_actual, 6),
                "clip_post_context_actual_sec": round(post_actual, 6),
                "clip_path": f"{clips_dir_name}/barkseq_{barkseq_id:03d}.wav",
            }
        )

    out = barkseqs_df.copy()
    clip_df = pd.DataFrame(rows)
    for col in CLIP_COLUMNS:
        out[col] = clip_df[col].values
    return out


def extract_clips(
    barkseqs_df: pd.DataFrame,
    wav_path: Path,
    out_dir: Path,
    audio_duration_sec: float,
    cfg: BarkConfig,
) -> tuple[pd.DataFrame, list[Path]]:
    """Write padded WAV clips and return barkseqs_df with clip_* columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = compute_clip_bounds(
        barkseqs_df, audio_duration_sec, cfg, clips_dir_name=out_dir.name
    )

    wav, sr = load_wav_mono(wav_path)
    if sr != cfg.target_sample_rate_hz:
        raise ValueError(
            f"expected sample rate {cfg.target_sample_rate_hz}, got {sr} from {wav_path}"
        )

    written: list[Path] = []
    n_samples = len(wav)

    for _, row in annotated.iterrows():
        start_i = _time_to_sample(float(row["clip_start_time_sec"]), sr)
        end_i = _time_to_sample(float(row["clip_end_time_sec"]), sr)
        start_i = max(0, min(start_i, n_samples - 1))
        end_i = max(start_i + 1, min(end_i, n_samples))

        clip = wav[start_i:end_i]
        out_path = out_dir / Path(str(row["clip_path"])).name
        pcm = np.clip(clip, -1.0, 1.0)
        scipy.io.wavfile.write(
            out_path,
            sr,
            (pcm * 32767.0).astype(np.int16),
        )
        written.append(out_path)

    return annotated, written
