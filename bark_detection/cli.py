import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from bark_detection.audio_io import extract_audio, probe_metadata
from bark_detection.config import BarkConfig
from bark_detection import viz
from bark_detection.legacy import candidates as candidates_module
from bark_detection.legacy import events as events_module
from bark_detection.legacy import rms as rms_module
from bark_detection.legacy import scoring as scoring_module
from bark_detection.legacy import threshold as threshold_module
from bark_detection import panns_inference
from bark_detection import panns_windows
from bark_detection import barkseq_detect
from bark_detection import barkseq_export
from bark_detection import clips
from bark_detection import frames
from bark_detection import noise
from bark_detection import timeline as timeline_module
from bark_detection.paths import intermediate_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _legacy_dir(run_dir: Path) -> Path:
    d = run_dir / "legacy_rms"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_extract(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    wav_path = run_dir / "audio.wav"
    extract_audio(video, wav_path, cfg)
    meta = probe_metadata(video, wav_path)
    _write_json(run_dir / "metadata.json", meta)


def _run_windows(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    duration_sec = float(meta["duration_sec"])

    windows_df = panns_windows.generate_windows(duration_sec, cfg)
    windows_df.to_csv(intermediate_dir(run_dir) / "panns_windows.csv", index=False)

    print(f"windows: {len(windows_df)}")


def _run_panns(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    panns_inference.run_panns_scores(run_dir, cfg)


def _run_timeline(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    inter = intermediate_dir(run_dir)
    scores_path = inter / "panns_scores.csv"
    if not scores_path.is_file():
        raise FileNotFoundError(f"missing panns scores: {scores_path}")

    panns_scores_df = pd.read_csv(scores_path)
    timeline_df = timeline_module.build_timeline(panns_scores_df, cfg)
    timeline_df.to_csv(inter / "bark_score_timeline.csv", index=False)

    peak = float(timeline_df["combined_bark_score"].max())
    print(
        f"timeline: {len(timeline_df)} samples"
        f"  combined_bark_mode={cfg.combined_bark_mode!r}"
        f"  peak={peak:.4f}"
    )


def _run_barkseqs(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    inter = intermediate_dir(run_dir)
    timeline_path = inter / "bark_score_timeline.csv"
    if not timeline_path.is_file():
        raise FileNotFoundError(f"missing bark score timeline: {timeline_path}")

    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    duration_sec = float(meta["duration_sec"])

    timeline_df = pd.read_csv(timeline_path)
    barkseqs_df = barkseq_detect.detect_barkseqs(timeline_df, duration_sec, cfg)
    barkseqs_df.to_csv(inter / "barkseqs_initial.csv", index=False)

    print(
        f"barkseqs: {len(barkseqs_df)}"
        f"  threshold={cfg.barkseq_threshold:.4f}"
        f"  merge_gap_sec={cfg.merge_gap_sec}"
    )


def _load_barkseq_inputs(run_dir: Path) -> tuple:
    import pandas as pd

    inter = intermediate_dir(run_dir)
    initial_path = inter / "barkseqs_initial.csv"
    timeline_path = inter / "bark_score_timeline.csv"
    if not initial_path.is_file():
        raise FileNotFoundError(f"missing barkseqs initial: {initial_path}")
    if not timeline_path.is_file():
        raise FileNotFoundError(f"missing bark score timeline: {timeline_path}")
    return pd.read_csv(initial_path), pd.read_csv(timeline_path)


def _run_noise(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    barkseqs_df, timeline_df = _load_barkseq_inputs(run_dir)
    flagged_df = noise.flag_noise(barkseqs_df, timeline_df, cfg)

    n_flagged = int(flagged_df["noise_flag"].sum())
    reasons = flagged_df["noise_reason"].value_counts().to_dict()
    print(
        f"noise: {len(flagged_df)} barkseqs  flagged={n_flagged}"
        f"  speech_thr={cfg.speech_noise_threshold:.2f}"
        f"  music_thr={cfg.music_noise_threshold:.2f}"
        f"  reasons={reasons}"
    )


def _run_export(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    barkseqs_df, timeline_df = _load_barkseq_inputs(run_dir)
    export_df = barkseq_export.export_barkseqs(barkseqs_df, timeline_df, cfg)
    export_df.to_csv(run_dir / "barkseqs.csv", index=False)

    min_conf = float(export_df["confidence"].min())
    max_conf = float(export_df["confidence"].max())
    print(
        f"export: {len(export_df)} barkseqs → barkseqs.csv"
        f"  confidence ∈ [{min_conf:.4f}, {max_conf:.4f}]"
        f"  method={export_df['method'].iloc[0] if len(export_df) else cfg.panns_model_name!r}"
    )


def _run_clips(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    barkseqs_path = run_dir / "barkseqs.csv"
    wav_path = run_dir / "audio.wav"
    meta_path = run_dir / "metadata.json"
    if not barkseqs_path.is_file():
        raise FileNotFoundError(f"missing barkseqs export: {barkseqs_path}")
    if not wav_path.is_file():
        raise FileNotFoundError(f"missing audio: {wav_path}")

    meta = json.loads(meta_path.read_text())
    duration_sec = float(meta["duration_sec"])

    barkseqs_df = pd.read_csv(barkseqs_path)
    out_dir = run_dir / "bark_event_clips"
    annotated_df, written = clips.extract_clips(
        barkseqs_df, wav_path, out_dir, duration_sec, cfg
    )
    annotated_df.to_csv(barkseqs_path, index=False)

    clip_durations = annotated_df["clip_duration_sec"].tolist()
    pre_actual = annotated_df["clip_pre_context_actual_sec"].tolist()
    post_actual = annotated_df["clip_post_context_actual_sec"].tolist()
    print(
        f"clips: {len(written)} wav files → {out_dir.name}/"
        f"  clip_durations_sec={clip_durations}"
        f"  pre_context_actual={pre_actual}"
        f"  post_context_actual={post_actual}"
    )


def _run_frames(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    barkseqs_path = run_dir / "barkseqs.csv"
    meta_path = run_dir / "metadata.json"
    if not barkseqs_path.is_file():
        raise FileNotFoundError(f"missing barkseqs export: {barkseqs_path}")

    meta = json.loads(meta_path.read_text())
    fps = float(meta["fps"])
    frame_count = int(meta["frame_count"])

    barkseqs_df = pd.read_csv(barkseqs_path)
    aligned_df = frames.align_frames(barkseqs_df, fps, frame_count, cfg)
    aligned_df.to_csv(barkseqs_path, index=False)

    if len(aligned_df) > 0:
        peak_frames = aligned_df["peak_frame"].tolist()
        print(
            f"frames: updated barkseqs.csv  fps={fps}  frame_count={frame_count}"
            f"  peak_frames={peak_frames}"
            f"  context ±{cfg.pre_context_sec}s"
        )
    else:
        print(f"frames: no barkseqs (fps={fps}, frame_count={frame_count})")


def _run_viz(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    inter = intermediate_dir(run_dir)
    timeline_path = inter / "bark_score_timeline.csv"
    barkseqs_path = run_dir / "barkseqs.csv"
    if not timeline_path.is_file():
        raise FileNotFoundError(f"missing bark score timeline: {timeline_path}")
    if not barkseqs_path.is_file():
        raise FileNotFoundError(f"missing barkseqs export: {barkseqs_path}")

    timeline_df = pd.read_csv(timeline_path)
    barkseqs_df = pd.read_csv(barkseqs_path)
    debug_dir = run_dir / "debug"
    stem = video.stem

    timeline_png = debug_dir / "panns_score_timeline.png"
    overlay_png = debug_dir / "barkseq_overlay.png"

    viz.plot_panns_score_timeline(
        timeline_df,
        timeline_png,
        cfg,
        title=f"PANNs score timeline — {stem}",
    )
    viz.plot_barkseq_overlay(
        timeline_df,
        barkseqs_df,
        overlay_png,
        cfg,
        title=f"Barkseq overlay — {stem}",
    )

    print(
        f"viz: 2 pngs → debug/"
        f"  barkseqs_shaded={len(barkseqs_df)}"
        f"  threshold={cfg.barkseq_threshold:.2f}"
    )


def _run_rms(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    legacy = _legacy_dir(run_dir)
    wav_path = run_dir / "audio.wav"
    wav, sr = rms_module.load_wav_mono(wav_path)
    rms_df = rms_module.compute_rms(wav, sr, cfg)
    rms_df.to_csv(legacy / "rms_values.csv", index=False)
    viz.plot_rms(
        rms_df,
        legacy / "rms_debug_plot.png",
        title=f"RMS energy — {video.stem}",
    )


def _run_threshold(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    legacy = _legacy_dir(run_dir)

    # 1. Read existing RMS CSV
    csv_path = legacy / "rms_values.csv"
    rms_df = __import__("pandas").read_csv(csv_path)

    # 2. Smooth
    rms_smoothed = threshold_module.smooth_rms(rms_df["rms"].to_numpy(), cfg)
    rms_df["rms_smoothed"] = rms_smoothed

    # Rewrite CSV with three columns: time_sec, rms, rms_smoothed
    rms_df[["time_sec", "rms", "rms_smoothed"]].to_csv(csv_path, index=False)

    # 3. Compute thresholds
    thresholds = threshold_module.compute_thresholds(rms_smoothed, cfg)

    # 4. Write thresholding block to legacy_rms/metadata.json (not root)
    meta = json.loads((run_dir / "metadata.json").read_text())
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
    _write_json(legacy / "metadata.json", meta)

    # 5. Re-render plot
    op = thresholds["operating_point"]
    op_method = thresholds["operating_method"]
    viz.plot_rms(
        rms_df,
        legacy / "rms_debug_plot.png",
        title=f"RMS energy — {video.stem}",
        threshold=op,
        threshold_label=f"{op_method} = {op:.4f}",
    )


def _run_candidates(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    legacy = _legacy_dir(run_dir)
    csv_path = legacy / "rms_values.csv"
    rms_df = pd.read_csv(csv_path)

    meta = json.loads((legacy / "metadata.json").read_text())
    threshold = float(meta["thresholding"]["operating_point"])
    operating_method = meta["thresholding"]["operating_method"]

    cands_df = candidates_module.find_candidates(rms_df, threshold, cfg)
    cands_df.to_csv(legacy / "bark_candidates.csv", index=False)

    print(
        f"candidates: {len(cands_df)}"
        f"  (threshold: {threshold:.6f}, operating_method: {operating_method})"
    )


def _run_events(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    legacy = _legacy_dir(run_dir)
    cands_path = legacy / "bark_candidates.csv"
    cands = pd.read_csv(cands_path)

    merged_df = events_module.merge_candidates(cands, cfg)
    kept_df, dropped = events_module.apply_duration_rules(merged_df, cfg)

    # Reorder columns to canonical order.
    col_order = [
        "event_id",
        "start_time_sec",
        "peak_time_sec",
        "end_time_sec",
        "duration_sec",
        "rms_peak",
        "normalized_rms_peak",
        "event_type",
        "merged_from",
    ]
    kept_df = kept_df[col_order]

    kept_df.to_csv(legacy / "bark_events_initial.csv", index=False)

    n_raw = len(cands)
    n_merged = len(merged_df)
    n_kept = len(kept_df)
    n_long = int((kept_df["event_type"] == "long_event").sum()) if n_kept > 0 else 0
    n_dropped = len(dropped)
    print(
        f"events: raw={n_raw} → merged={n_merged} → kept={n_kept}"
        f" (long_event={n_long}, dropped={n_dropped})"
    )


def _run_score(video: Path, run_dir: Path, cfg: BarkConfig) -> None:
    import pandas as pd

    legacy = _legacy_dir(run_dir)
    events_path = legacy / "bark_events_initial.csv"
    wav_path = run_dir / "audio.wav"

    events_df = pd.read_csv(events_path)
    wav, sr = rms_module.load_wav_mono(wav_path)

    scored_df = scoring_module.compute_bark_confidence(events_df, wav, sr, cfg)

    col_order = [
        "event_id",
        "start_time_sec",
        "peak_time_sec",
        "end_time_sec",
        "duration_sec",
        "rms_peak",
        "normalized_rms_peak",
        "bark_confidence",
        "event_type",
        "merged_from",
    ]
    scored_df = scored_df[col_order]

    scored_df.to_csv(legacy / "bark_events.csv", index=False)

    min_conf = float(scored_df["bark_confidence"].min())
    max_conf = float(scored_df["bark_confidence"].max())
    print(
        f"score: {len(scored_df)} events  "
        f"bark_confidence ∈ [{min_conf:.4f}, {max_conf:.4f}]"
    )


# Prompt 1–6 PANNs default: extract → … → frames → viz.
# Legacy RMS stages remain individually invocable via --stage.
_STAGE_ORDER = [
    "extract",
    "windows",
    "panns",
    "timeline",
    "barkseqs",
    "noise",
    "export",
    "clips",
    "frames",
    "viz",
]

_DISPATCH = {
    "extract": _run_extract,
    "windows": _run_windows,
    "panns": _run_panns,
    "timeline": _run_timeline,
    "barkseqs": _run_barkseqs,
    "noise": _run_noise,
    "export": _run_export,
    "clips": _run_clips,
    "frames": _run_frames,
    "viz": _run_viz,
    "rms": _run_rms,
    "threshold": _run_threshold,
    "candidates": _run_candidates,
    "events": _run_events,
    "score": _run_score,
}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_bark_detection")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output_dir", default=Path("outputs"), type=Path)
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "extract",
            "windows",
            "panns",
            "timeline",
            "barkseqs",
            "noise",
            "export",
            "clips",
            "frames",
            "viz",
            "rms",
            "threshold",
            "candidates",
            "events",
            "score",
        ],
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
