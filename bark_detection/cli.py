import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from bark_detection.audio_io import extract_audio, probe_metadata
from bark_detection.config import BarkConfig
from bark_detection import rms as rms_module
from bark_detection import viz
from bark_detection import threshold as threshold_module
from bark_detection import candidates as candidates_module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_extract(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    wav_path = run_dir / "audio.wav"
    extract_audio(video, wav_path, cfg)
    meta = probe_metadata(video, wav_path)
    _write_json(run_dir / "metadata.json", meta)


def _run_rms(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    wav_path = run_dir / "audio.wav"
    wav, sr = rms_module.load_wav_mono(wav_path)
    rms_df = rms_module.compute_rms(wav, sr, cfg)
    rms_df.to_csv(run_dir / "rms_values.csv", index=False)
    viz.plot_rms(
        rms_df,
        run_dir / "rms_debug_plot.png",
        title=f"RMS energy — {video.stem}",
    )


def _run_threshold(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    # 1. Read existing RMS CSV
    csv_path = run_dir / "rms_values.csv"
    rms_df = __import__("pandas").read_csv(csv_path)

    # 2. Smooth
    rms_smoothed = threshold_module.smooth_rms(rms_df["rms"].to_numpy(), cfg)
    rms_df["rms_smoothed"] = rms_smoothed

    # Rewrite CSV with three columns: time_sec, rms, rms_smoothed
    rms_df[["time_sec", "rms", "rms_smoothed"]].to_csv(csv_path, index=False)

    # 3. Compute thresholds
    thresholds = threshold_module.compute_thresholds(rms_smoothed, cfg)

    # 4. Update metadata.json
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["thresholding"] = {
        "method": cfg.threshold_method,
        "mean_std_k": cfg.mean_std_k,
        "percentile": cfg.percentile,
        "smoothing_method": cfg.smoothing_method,
        "smoothing_window_ms": cfg.smoothing_window_ms,
        "mean_std_k2": thresholds["mean_std_k2"],
        "percentile_90": thresholds["percentile_90"],
        "operating_point": thresholds["operating_point"],
        "operating_method": thresholds["operating_method"],
        "rms_smoothed_mean": thresholds["mean"],
        "rms_smoothed_std": thresholds["std"],
    }
    _write_json(meta_path, meta)

    # 5. Re-render plot
    op = thresholds["operating_point"]
    op_method = thresholds["operating_method"]
    viz.plot_rms(
        rms_df,
        run_dir / "rms_debug_plot.png",
        title=f"RMS energy — {video.stem}",
        threshold=op,
        threshold_label=f"{op_method} = {op:.4f}",
    )


def _run_candidates(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    csv_path = run_dir / "rms_values.csv"
    rms_df = pd.read_csv(csv_path)

    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    threshold = float(meta["thresholding"]["operating_point"])
    operating_method = meta["thresholding"]["operating_method"]

    cands_df = candidates_module.find_candidates(rms_df, threshold, cfg)
    cands_df.to_csv(run_dir / "bark_candidates.csv", index=False)

    print(
        f"candidates: {len(cands_df)}"
        f"  (threshold: {threshold:.6f}, operating_method: {operating_method})"
    )


_STAGE_ORDER = ["extract", "rms", "threshold", "candidates"]

_DISPATCH = {
    "extract": _run_extract,
    "rms": _run_rms,
    "threshold": _run_threshold,
    "candidates": _run_candidates,
}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_bark_detection")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output_dir", default=Path("outputs"), type=Path)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "extract", "rms", "threshold", "candidates"],
    )
    args = parser.parse_args(argv)

    video_path: Path = args.video
    if not video_path.exists():
        parser.error(f"video not found: {video_path}")

    cfg = BarkConfig()

    video_stem = video_path.stem
    run_dir = args.output_dir / video_stem
    run_dir.mkdir(parents=True, exist_ok=True)

    stages_to_run = _STAGE_ORDER if args.stage == "all" else [args.stage]

    for stage in stages_to_run:
        _DISPATCH[stage](video_path, run_dir, cfg)

    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
