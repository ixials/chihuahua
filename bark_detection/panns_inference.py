"""M3: PANNs AudioSet tagging per overlapping window."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile

from bark_detection.config import BarkConfig
from bark_detection import panns_windows
from bark_detection.paths import intermediate_dir

PANNS_DATA_DIR = Path.home() / "panns_data"
LABELS_FILENAME = "class_labels_indices.csv"
CHECKPOINT_FILENAME = "Cnn14_16k_mAP=0.438.pth"
LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)
CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
)
MIN_CHECKPOINT_BYTES = 300_000_000

VOCALIZATION_LABELS: list[str] = [
    "Bark",
    "Yip",
    "Bow-wow",
    "Howl",
    "Growling",
    "Whimper (dog)",
]

SCORE_COLUMN_LABELS: dict[str, list[str]] = {
    "dog": ["Dog"],
    "bark": VOCALIZATION_LABELS,
    "animal": ["Animal"],
    "speech": ["Speech"],
    "music": ["Music"],
}


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"saved {dest}")


def ensure_panns_assets() -> None:
    """Download PANNs label CSV and Cnn14_16k checkpoint via urllib."""
    labels_path = PANNS_DATA_DIR / LABELS_FILENAME
    if not labels_path.is_file():
        _download_url(LABELS_URL, labels_path)

    checkpoint_path = PANNS_DATA_DIR / CHECKPOINT_FILENAME
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size < MIN_CHECKPOINT_BYTES
    ):
        _download_url(CHECKPOINT_URL, checkpoint_path)


def load_label_list() -> list[str]:
    """Return AudioSet display names after assets exist."""
    ensure_panns_assets()
    from panns_inference.config import labels

    return list(labels)


def resolve_label_mapping(labels: list[str]) -> dict[str, dict[str, int]]:
    """Map score columns to label names and indices (no hard-coded indices)."""
    name_to_index = {name: idx for idx, name in enumerate(labels)}
    mapping: dict[str, dict[str, int]] = {}

    for score_col, label_names in SCORE_COLUMN_LABELS.items():
        col_map: dict[str, int] = {}
        for label_name in label_names:
            if label_name not in name_to_index:
                raise KeyError(
                    f"label {label_name!r} not found in PANNs label list "
                    f"({len(labels)} classes)"
                )
            col_map[label_name] = name_to_index[label_name]
        mapping[score_col] = col_map

    return mapping


def build_audio_tagging_16k(cfg: BarkConfig):
    """Construct Cnn14_16k AudioTagging wrapper."""
    ensure_panns_assets()
    import torch
    from panns_inference.config import classes_num
    from panns_inference.inference import AudioTagging
    from panns_inference.models import Cnn14

    model = Cnn14(
        sample_rate=16000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=classes_num,
    )
    checkpoint_path = PANNS_DATA_DIR / CHECKPOINT_FILENAME

    # PyTorch 2.6+ defaults weights_only=True; PANNs checkpoints need False.
    _orig_load = torch.load

    def _load_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_checkpoint
    try:
        tagger = AudioTagging(
            model=model,
            checkpoint_path=str(checkpoint_path),
            device=cfg.panns_device,
        )
    finally:
        torch.load = _orig_load

    return tagger


SCORE_COLUMNS = ("dog_score", "bark_score", "animal_score", "speech_score", "music_score")


def normalize_audio_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    """Scale *audio* so ``max(abs(audio)) == target_peak`` (no-op if silent)."""
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio.astype(np.float32, copy=False)
    return (audio * (target_peak / peak)).astype(np.float32)


def merge_window_scores(
    long_scores: pd.DataFrame,
    short_scores: pd.DataFrame,
    score_columns: tuple[str, ...] = SCORE_COLUMNS,
    short_window_weight: float = 1.0,
) -> pd.DataFrame:
    """Merge long- and short-grid PANNs scores with per-column max.

    For each long window, overlapping short windows (by time interval) can
    raise score columns; non-score columns come from the long grid row.
    Short scores are multiplied by *short_window_weight* before the max.
    """
    merged = long_scores.copy()
    for idx, long_row in long_scores.iterrows():
        start = float(long_row["start_time_sec"])
        end = float(long_row["end_time_sec"])
        overlap = short_scores[
            (short_scores["start_time_sec"] < end)
            & (short_scores["end_time_sec"] > start)
        ]
        for col in score_columns:
            long_val = float(long_row[col])
            short_val = float(overlap[col].max()) if len(overlap) else 0.0
            merged.at[idx, col] = max(long_val, short_val * short_window_weight)
    return merged


def _load_wav_mono(wav_path: Path, cfg: BarkConfig | None = None) -> tuple[np.ndarray, int]:
    sample_rate, data = scipy.io.wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    audio = data.astype(np.float32) / 32768.0
    if cfg is not None and cfg.audio_normalize:
        audio = normalize_audio_peak(audio, cfg.audio_normalize_target_peak)
    return audio, int(sample_rate)


def slice_window(
    wav: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    window_size_sec: float,
) -> np.ndarray:
    """Extract [start, end) and zero-pad to window_size_sec * sr samples."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = wav[start_sample:end_sample]
    target_len = int(window_size_sec * sr)
    if len(segment) < target_len:
        segment = np.pad(segment, (0, target_len - len(segment)), mode="constant")
    return segment.astype(np.float32)


def _score_at_indices(scores: np.ndarray, indices: list[int]) -> float:
    return float(np.max(scores[indices]))


def _top_k_labels(
    scores: np.ndarray, labels: list[str], k: int = 3
) -> list[tuple[str, float]]:
    ranked = np.argsort(scores)[::-1][:k]
    return [(labels[idx], float(scores[idx])) for idx in ranked]


def _write_label_list(path: Path, labels: list[str]) -> None:
    lines = [f"{idx}\t{name}" for idx, name in enumerate(labels)]
    path.write_text("\n".join(lines) + "\n")


def _infer_window_scores(
    wav: np.ndarray,
    sr: int,
    windows_df: pd.DataFrame,
    tagger,
    mapping: dict[str, dict[str, int]],
    labels: list[str],
    window_size_sec: float,
) -> list[dict[str, object]]:
    """Run PANNs on each row of *windows_df* and return score dicts."""
    dog_ix = list(mapping["dog"].values())
    bark_ix = list(mapping["bark"].values())
    animal_ix = list(mapping["animal"].values())
    speech_ix = list(mapping["speech"].values())
    music_ix = list(mapping["music"].values())

    rows: list[dict[str, object]] = []
    for _, win in windows_df.iterrows():
        segment = slice_window(
            wav,
            sr,
            float(win["start_time_sec"]),
            float(win["end_time_sec"]),
            window_size_sec,
        )
        clipwise_output, _ = tagger.inference(segment[None, :])
        scores = clipwise_output[0]

        top3 = _top_k_labels(scores, labels, k=3)
        row: dict[str, object] = {
            "window_id": int(win["window_id"]),
            "start_time_sec": float(win["start_time_sec"]),
            "end_time_sec": float(win["end_time_sec"]),
            "center_time_sec": float(win["center_time_sec"]),
            "dog_score": _score_at_indices(scores, dog_ix),
            "bark_score": _score_at_indices(scores, bark_ix),
            "animal_score": _score_at_indices(scores, animal_ix),
            "speech_score": _score_at_indices(scores, speech_ix),
            "music_score": _score_at_indices(scores, music_ix),
        }
        for rank, (label, score) in enumerate(top3, start=1):
            row[f"top_{rank}_label"] = label
            row[f"top_{rank}_score"] = score
        rows.append(row)
    return rows


def _write_label_mapping(path: Path, mapping: dict[str, dict[str, int]]) -> None:
    lines: list[str] = []
    for score_col, label_map in mapping.items():
        score_name = f"{score_col}_score"
        if score_col == "bark":
            label_str = ", ".join(label_map.keys())
            lines.append(
                f"{score_name} -> max({label_str}) -> "
                + ", ".join(f"{n}={i}" for n, i in label_map.items())
            )
        else:
            for label_name, index in label_map.items():
                lines.append(f"{score_name} -> {label_name} -> {index}")
    path.write_text("\n".join(lines) + "\n")


def run_panns_scores(run_dir: Path, cfg: BarkConfig) -> None:
    """Run PANNs tagging on each row of panns_windows.csv."""
    ensure_panns_assets()
    labels = load_label_list()
    mapping = resolve_label_mapping(labels)

    inter = intermediate_dir(run_dir)
    label_list_path = inter / "panns_label_list.txt"
    mapping_path = inter / "panns_label_mapping.txt"
    _write_label_list(label_list_path, labels)
    _write_label_mapping(mapping_path, mapping)

    for line in mapping_path.read_text().strip().splitlines():
        print(line)

    long_windows_df = pd.read_csv(inter / "panns_windows.csv")

    short_windows_path = inter / "panns_short_windows.csv"
    if short_windows_path.is_file():
        short_windows_df = pd.read_csv(short_windows_path)
    else:
        meta = json.loads((run_dir / "metadata.json").read_text())
        duration_sec = float(meta["duration_sec"])
        short_windows_df = panns_windows.generate_windows(
            duration_sec,
            cfg,
            window_size_sec=cfg.short_window_size_sec,
            hop_size_sec=cfg.short_hop_size_sec,
        )

    wav, sr = _load_wav_mono(run_dir / "audio.wav", cfg)
    if sr != cfg.target_sample_rate_hz:
        raise ValueError(
            f"expected sample rate {cfg.target_sample_rate_hz}, got {sr}"
        )

    tagger = build_audio_tagging_16k(cfg)

    long_rows = _infer_window_scores(
        wav,
        sr,
        long_windows_df,
        tagger,
        mapping,
        labels,
        cfg.window_size_sec,
    )
    long_scores_df = pd.DataFrame(long_rows)

    short_rows = _infer_window_scores(
        wav,
        sr,
        short_windows_df,
        tagger,
        mapping,
        labels,
        cfg.short_window_size_sec,
    )
    short_scores_df = pd.DataFrame(short_rows)
    short_scores_path = inter / "panns_short_scores.csv"
    short_scores_df.to_csv(short_scores_path, index=False)

    scores_df = merge_window_scores(
        long_scores_df,
        short_scores_df,
        short_window_weight=cfg.short_window_weight,
    )
    scores_path = inter / "panns_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    print(
        f"panns scores: {len(long_scores_df)} long + {len(short_scores_df)} short"
        f" -> merged {len(scores_df)} -> {scores_path.name}"
    )
