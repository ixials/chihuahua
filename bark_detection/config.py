from dataclasses import dataclass


@dataclass
class BarkConfig:
    # M1
    target_sample_rate_hz: int = 16000
    target_channels: int = 1
    # M2 (PANNs overlapping windows)
    window_size_sec: float = 1.0
    hop_size_sec: float = 0.25
    # M3 (PANNs inference)
    panns_model_name: str = "Cnn14_16k"
    panns_device: str = "cpu"
    # M4 (bark score timeline)
    combined_bark_mode: str = "max_bark_dog"  # or "bark" for precision-first
    # M5 (Barkseq detection from timeline)
    barkseq_threshold: float = 0.4  # high-recall default (was 0.42 on dogs1 bark-only)
    merge_gap_sec: float = 0.33
    # M6 (PANNs noise flagging on Barkseqs)
    speech_noise_threshold: float = 0.1
    music_noise_threshold: float = 0.1
    # M7 (PANNs Barkseq export confidence)
    speech_penalty: float = 0.6
    music_penalty: float = 0.6
    # M8 (Barkseq WAV clips)
    clip_pre_context_sec: float = 0.20
    clip_post_context_sec: float = 0.20
    prevent_clip_overlap: bool = True
    # M9 (frame alignment)
    pre_context_sec: float = 0.5
    post_context_sec: float = 0.5
    # M2 legacy (RMS framing)
    window_ms: int = 50
    hop_ms: int = 10
    # M3
    smoothing_method: str = "moving_average"  # or "median"
    smoothing_window_ms: int = 50
    threshold_method: str = "mean_std"        # or "percentile"
    mean_std_k: float = 2.0
    percentile: float = 90.0
    # M5
    merge_gap_ms: int = 200
    min_duration_ms: int = 120
    max_duration_ms: int = 1200
    # Legacy RMS M6 (scoring.py — not used by PANNs pipeline)
    w_rms: float = 0.6
    w_duration: float = 0.4
    duration_score_peak_sec: float = 0.20
    classifier: object | None = None
    classifier_weight: float = 0.5
