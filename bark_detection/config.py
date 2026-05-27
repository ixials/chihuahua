from dataclasses import dataclass


@dataclass
class BarkConfig:
    # M1
    target_sample_rate_hz: int = 16000
    target_channels: int = 1
    # M2
    window_ms: int = 50
    hop_ms: int = 10
    # M3
    smoothing_method: str = "moving_average"  # or "median"
    smoothing_window_ms: int = 50
    threshold_method: str = "mean_std"        # or "percentile"
    mean_std_k: float = 2.0
    percentile: float = 90.0
