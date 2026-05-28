import json
import subprocess
from pathlib import Path
from typing import Union

from bark_detection.config import BarkConfig


def extract_audio(video_path: Union[str, Path], wav_path: Union[str, Path], cfg: BarkConfig) -> None:
    """Run ffmpeg to extract standardized audio (mono, 16 kHz, 16-bit PCM)."""
    video_path = Path(video_path)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", str(cfg.target_channels),
        "-ar", str(cfg.target_sample_rate_hz),
        "-vn", "-acodec", "pcm_s16le", str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}):\n{e.stderr}") from e


def _ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed (exit {e.returncode}) for {path}:\n{e.stderr}") from e
    return json.loads(r.stdout)


def _parse_fps(r_frame_rate: str) -> float:
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/", 1)
        n, d = float(num), float(den)
        return n / d if d != 0 else 0.0
    return float(r_frame_rate)


def _first_stream(probe: dict, codec_type: str) -> dict:
    for s in probe.get("streams", []):
        if s.get("codec_type") == codec_type:
            return s
    raise RuntimeError(f"No {codec_type} stream found")


def _stream_duration(stream: dict, probe: dict) -> float:
    d = stream.get("duration")
    if d is not None:
        return float(d)
    return float(probe.get("format", {}).get("duration", 0.0))


def probe_metadata(video_path: Union[str, Path], wav_path: Union[str, Path]) -> dict:
    """Return the metadata.json payload: fps/frame_count from video, sample_rate/duration from WAV."""
    video_path = Path(video_path)
    wav_path = Path(wav_path)

    vprobe = _ffprobe(video_path)
    wprobe = _ffprobe(wav_path)

    vstream = _first_stream(vprobe, "video")
    astream_std = _first_stream(wprobe, "audio")

    fps = _parse_fps(vstream.get("r_frame_rate", "0/1"))
    frame_count = int(vstream.get("nb_frames", 0))
    wav_sample_rate = int(astream_std.get("sample_rate", 0))
    wav_duration = _stream_duration(astream_std, wprobe)

    return {
        "fps": round(fps, 6),
        "frame_count": int(frame_count),
        "sample_rate_hz": int(wav_sample_rate),
        "duration_sec": round(wav_duration, 6),
    }
