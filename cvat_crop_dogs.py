#!/usr/bin/env python3
"""
Build the minimal dog audio-video dataset we agreed on.

Expected input folder:

vidsAndAnnotation/
└── 16/
    └── 00/
        ├── 16_bark_000.mp4
        ├── annotations.xml
        └── <one aligned .wav file>

Output:

processed/
└── 16_bark_000/
    ├── crops/
    │   ├── 16_bark_000_track_0.mp4
    │   └── 16_bark_000_track_1.mp4
    ├── audio/
    │   ├── 16_bark_000_track_0.wav
    │   └── 16_bark_000_track_1.wav
    └── training_windows.csv

What it does:
- Automatically finds the one MP4, annotations.xml, and one WAV in --input-dir.
- Reads Barking / Not_Barking directly from the CVAT XML.
- Adds 10% padding around each dog box.
- Creates one 224x224 moving crop MP4 per dog track.
- Keeps the crop video on the original timeline.
- Writes black frames only when that dog is outside or has no box.
- Trims the WAV per track using ffmpeg.
- Creates training windows only inside continuous, visible, same-label sections.
- Uses 2-second windows with 1-second stride by default.
- Keeps shorter valid sections and marks them needs_padding=1.
- Does NOT create all_annotations.csv, frame_annotations.csv, summary.json,
  JPG crops, or other duplicate metadata.

Install:
    pip install opencv-python

Run:
    python build_dog_av_dataset.py \
        --input-dir vidsAndAnnotation/16/00 \
        --output-root processed \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import wave
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import subprocess

import cv2
import numpy as np


VALID_LABELS = {
    "barking": ("Barking", 1),
    "not_barking": ("Not_Barking", 0),
    "not barking": ("Not_Barking", 0),
}


@dataclass
class Annotation:
    frame: int
    source_track_id: int
    track_id: int
    label_text: str
    label: int
    outside: int
    occluded: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float


def normalize_vocalization(value: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if value is None:
        return None, None

    key = value.strip().lower().replace("-", "_")
    return VALID_LABELS.get(key, (None, None))


def parse_track_remaps(values: Iterable[str]) -> Dict[int, int]:
    remap: Dict[int, int] = {}

    for value in values:
        try:
            old_text, new_text = value.split(":", maxsplit=1)
            remap[int(old_text)] = int(new_text)
        except Exception as exc:
            raise ValueError(
                f"Invalid --track-remap '{value}'. Use OLD:NEW, such as 2:0."
            ) from exc

    return remap


def find_input_files(input_dir: Path) -> Tuple[Path, Path, Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    videos = sorted(input_dir.glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(
            f"Expected exactly one MP4 in {input_dir}, found {len(videos)}: "
            f"{[path.name for path in videos]}"
        )

    preferred_xml = input_dir / "annotations.xml"
    if preferred_xml.exists():
        xml_path = preferred_xml
    else:
        xml_files = sorted(input_dir.glob("*.xml"))
        if len(xml_files) != 1:
            raise ValueError(
                f"Expected annotations.xml or exactly one XML in {input_dir}, "
                f"found {len(xml_files)}."
            )
        xml_path = xml_files[0]

    wav_files = sorted(input_dir.glob("*.wav"))
    if len(wav_files) != 1:
        raise ValueError(
            f"Expected exactly one WAV in {input_dir}, found {len(wav_files)}: "
            f"{[path.name for path in wav_files]}"
        )

    return videos[0], xml_path, wav_files[0]


def find_track_level_vocalization(track: ET.Element) -> Optional[str]:
    for attribute in track.findall("./attribute"):
        if attribute.get("name", "").strip().lower() == "vocalization":
            return attribute.text
    return None


def find_box_vocalization(box: ET.Element) -> Optional[str]:
    for attribute in box.findall("./attribute"):
        if attribute.get("name", "").strip().lower() == "vocalization":
            return attribute.text
    return None


def parse_annotations(
    xml_path: Path,
    track_remap: Dict[int, int],
) -> Dict[int, Dict[int, Annotation]]:
    """
    Return:
        tracks[output_track_id][frame_number] = Annotation

    Mutable Vocalization values are carried forward if CVAT omits them
    on an interpolated box.
    """
    root = ET.parse(xml_path).getroot()
    tracks: Dict[int, Dict[int, Annotation]] = {}

    for track in root.findall("./track"):
        if track.get("label") != "Dog":
            continue

        source_track_id = int(track.get("id", "-1"))
        output_track_id = track_remap.get(source_track_id, source_track_id)
        current_vocalization = find_track_level_vocalization(track)

        boxes = sorted(
            track.findall("./box"),
            key=lambda element: int(element.get("frame", "0")),
        )

        for box in boxes:
            box_value = find_box_vocalization(box)
            if box_value is not None:
                current_vocalization = box_value

            label_text, label = normalize_vocalization(current_vocalization)

            frame = int(box.get("frame", "0"))
            outside = int(box.get("outside", "0"))

            if outside == 0 and label_text is None:
                raise ValueError(
                    f"Visible dog is missing Barking/Not_Barking: "
                    f"track={source_track_id}, frame={frame}"
                )

            # An outside marker may still carry the previous attribute.
            if label_text is None:
                label_text = "Not_Barking"
                label = 0

            annotation = Annotation(
                frame=frame,
                source_track_id=source_track_id,
                track_id=output_track_id,
                label_text=label_text,
                label=int(label),
                outside=outside,
                occluded=int(box.get("occluded", "0")),
                xtl=float(box.get("xtl", "0")),
                ytl=float(box.get("ytl", "0")),
                xbr=float(box.get("xbr", "0")),
                ybr=float(box.get("ybr", "0")),
            )

            track_frames = tracks.setdefault(output_track_id, {})

            if frame in track_frames:
                existing = track_frames[frame]

                if existing.outside == 0 and annotation.outside == 0:
                    existing_is_target = (
                        existing.source_track_id == output_track_id
                    )
                    incoming_is_target = (
                        annotation.source_track_id == output_track_id
                    )

                    if incoming_is_target and not existing_is_target:
                        track_frames[frame] = annotation
                    elif existing_is_target and not incoming_is_target:
                        pass
                    else:
                        raise ValueError(
                            "Track remapping found two visible annotations but "
                            "neither one is the canonical target track. "
                            f"output_track={output_track_id}, frame={frame}, "
                            f"source_tracks="
                            f"{existing.source_track_id},"
                            f"{annotation.source_track_id}"
                        )

                elif existing.outside == 1 and annotation.outside == 0:
                    track_frames[frame] = annotation

                elif existing.outside == 0 and annotation.outside == 1:
                    pass

                else:
                    existing_is_target = (
                        existing.source_track_id == output_track_id
                    )
                    incoming_is_target = (
                        annotation.source_track_id == output_track_id
                    )
                    if incoming_is_target and not existing_is_target:
                        track_frames[frame] = annotation
            else:
                track_frames[frame] = annotation

    if not tracks:
        raise ValueError(f"No Dog tracks found in {xml_path}")

    return tracks


def add_padding(
    annotation: Annotation,
    frame_width: int,
    frame_height: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    box_width = max(1.0, annotation.xbr - annotation.xtl)
    box_height = max(1.0, annotation.ybr - annotation.ytl)

    x1 = max(0, int(math.floor(annotation.xtl - padding * box_width)))
    y1 = max(0, int(math.floor(annotation.ytl - padding * box_height)))
    x2 = min(frame_width, int(math.ceil(annotation.xbr + padding * box_width)))
    y2 = min(frame_height, int(math.ceil(annotation.ybr + padding * box_height)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop after padding: track={annotation.track_id}, "
            f"frame={annotation.frame}, box={(x1, y1, x2, y2)}"
        )

    return x1, y1, x2, y2


def letterbox(image: np.ndarray, output_size: int) -> np.ndarray:
    """Resize without stretching, then pad to a square."""
    if image.size == 0:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    height, width = image.shape[:2]
    scale = min(output_size / width, output_size / height)

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    x_offset = (output_size - resized_width) // 2
    y_offset = (output_size - resized_height) // 2

    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized

    return canvas


def create_crop_videos(
    video_path: Path,
    tracks: Dict[int, Dict[int, Annotation]],
    track_bounds: Dict[int, Tuple[int, int]],
    crops_dir: Path,
    padding: float,
    crop_size: int,
) -> Tuple[float, int, int, int, Dict[int, Path]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Video has invalid FPS: {fps}")

    crops_dir.mkdir(parents=True, exist_ok=True)
    clip_name = video_path.stem

    crop_paths = {
        track_id: crops_dir / f"{clip_name}_track_{track_id}.mp4"
        for track_id in sorted(tracks)
    }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")


    # and each writer covers exactly its own frame range.
    frame_count = 0

    try:
        for track_id in sorted(tracks):
            if track_id not in track_bounds:
                continue

            track_start, track_end = track_bounds[track_id]

            writer = cv2.VideoWriter(
                str(crop_paths[track_id]),
                fourcc,
                fps,
                (crop_size, crop_size),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not create crop video: {crop_paths[track_id]}"
                )

            cap.set(cv2.CAP_PROP_POS_FRAMES, track_start)
            frame_index = track_start

            try:
                while frame_index <= track_end:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    annotation = tracks[track_id].get(frame_index)

                    if annotation is None or annotation.outside == 1:
                        crop_frame = np.zeros(
                            (crop_size, crop_size, 3),
                            dtype=np.uint8,
                        )
                    else:
                        x1, y1, x2, y2 = add_padding(
                            annotation=annotation,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            padding=padding,
                        )
                        dog_crop = frame[y1:y2, x1:x2]
                        crop_frame = letterbox(dog_crop, crop_size)

                    writer.write(crop_frame)
                    frame_index += 1

            finally:
                writer.release()

            # Track the total frames read across all tracks for the summary.
            # Use the last track's end as a proxy for total video length.
            frame_count = max(frame_count, frame_index)

    finally:
        cap.release()

    # Re-derive total frame count from video properties for the duration check.
    if reported_frames > 0 and frame_count != reported_frames:
        print(
            f"WARNING: OpenCV reported {reported_frames} frames but last "
            f"track read up to frame {frame_count}.",
            file=sys.stderr,
        )

    # Use reported_frames for duration if available, else use what we read.
    total_frames = reported_frames if reported_frames > 0 else frame_count

    return fps, frame_width, frame_height, total_frames, crop_paths


def get_visible_same_label_segments(
    track_frames: Dict[int, Annotation],
) -> List[Tuple[int, int, int, str]]:
    """
    Return continuous visible segments:
        start_frame, end_frame, numeric_label, text_label

    A segment ends when:
    - the next annotation frame is not consecutive,
    - the dog becomes outside/missing,
    - or Barking/Not_Barking changes.
    """
    visible = [
        annotation
        for _, annotation in sorted(track_frames.items())
        if annotation.outside == 0
    ]

    if not visible:
        return []

    segments: List[Tuple[int, int, int, str]] = []

    start_frame = visible[0].frame
    previous = visible[0]
    current_label = visible[0].label
    current_label_text = visible[0].label_text

    for annotation in visible[1:]:
        is_consecutive = annotation.frame == previous.frame + 1
        same_label = annotation.label == current_label

        if not is_consecutive or not same_label:
            segments.append(
                (
                    start_frame,
                    previous.frame,
                    current_label,
                    current_label_text,
                )
            )
            start_frame = annotation.frame
            current_label = annotation.label
            current_label_text = annotation.label_text

        previous = annotation

    segments.append(
        (
            start_frame,
            previous.frame,
            current_label,
            current_label_text,
        )
    )

    return segments


def make_windows(
    segment_start: int,
    segment_end: int,
    window_frames: int,
    stride_frames: int,
) -> List[Tuple[int, int, int]]:
    """
    Return:
        start_frame, end_frame, needs_padding

    Short segments produce one natural-length window.
    Long segments produce fixed-length windows plus an end-aligned final window.
    """
    segment_frames = segment_end - segment_start + 1

    if segment_frames < window_frames:
        return [(segment_start, segment_end, 1)]

    starts = list(
        range(
            segment_start,
            segment_end - window_frames + 2,
            stride_frames,
        )
    )

    final_start = segment_end - window_frames + 1
    if starts[-1] != final_start:
        starts.append(final_start)

    return [
        (start, start + window_frames - 1, 0)
        for start in starts
    ]


def write_training_windows(
    output_csv: Path,
    clip_name: str,
    tracks: Dict[int, Dict[int, Annotation]],
    crop_paths: Dict[int, Path],

    audio_paths: Dict[int, Path],
    clip_output_dir: Path,
    fps: float,
    window_seconds: float,
    stride_seconds: float,
    min_segment_seconds: float,
) -> int:
    window_frames = max(1, int(round(window_seconds * fps)))
    stride_frames = max(1, int(round(stride_seconds * fps)))
    min_segment_frames = max(1, int(math.ceil(min_segment_seconds * fps)))

    rows: List[dict] = []

    for track_id in sorted(tracks):
        sample_index = 0

        for segment_index, (
            segment_start,
            segment_end,
            label,
            label_text,
        ) in enumerate(get_visible_same_label_segments(tracks[track_id])):
            segment_length = segment_end - segment_start + 1

            if segment_length < min_segment_frames:
                continue

            for start_frame, end_frame, needs_padding in make_windows(
                segment_start=segment_start,
                segment_end=segment_end,
                window_frames=window_frames,
                stride_frames=stride_frames,
            ):
                start_sec = start_frame / fps
                end_sec = (end_frame + 1) / fps
                actual_duration = (end_frame - start_frame + 1) / fps

                rows.append({
                    "sample_id": f"{clip_name}_t{track_id}_w{sample_index:04d}",
                    "clip_name": clip_name,
                    "track_id": track_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_sec": f"{start_sec:.6f}",
                    "end_sec": f"{end_sec:.6f}",
                    "actual_duration_sec": f"{actual_duration:.6f}",
                    "label": label,
                    "label_text": label_text,
                    "crop_path": crop_paths[track_id]
                        .relative_to(clip_output_dir)
                        .as_posix(),

                    "audio_path": audio_paths[track_id]
                        .relative_to(clip_output_dir)
                        .as_posix(),
                    "needs_padding": needs_padding,
                })
                sample_index += 1

    fieldnames = [
        "sample_id",
        "clip_name",
        "track_id",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "actual_duration_sec",
        "label",
        "label_text",
        "crop_path",
        "audio_path",
        "needs_padding",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def check_wav_duration(wav_path: Path, video_duration: float) -> None:
    """Best-effort warning only. The WAV is still used if this check fails."""
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            wav_duration = wav_file.getnframes() / wav_file.getframerate()

        difference = abs(wav_duration - video_duration)
        if difference > 0.10:
            print(
                f"WARNING: WAV duration ({wav_duration:.3f}s) and video duration "
                f"({video_duration:.3f}s) differ by {difference:.3f}s. "
                "Make sure they start at the same time.",
                file=sys.stderr,
            )
    except (wave.Error, OSError, ZeroDivisionError):
        print(
            "WARNING: Could not verify WAV duration. The file will still be used.",
            file=sys.stderr,
        )


def process(
    input_dir: Path,
    output_root: Path,
    padding: float,
    crop_size: int,
    window_seconds: float,
    stride_seconds: float,
    min_segment_seconds: float,
    track_remap: Dict[int, int],
    overwrite: bool,
) -> Path:
    video_path, xml_path, source_wav_path = find_input_files(input_dir)
    clip_name = video_path.stem
    clip_output_dir = output_root / clip_name

    if clip_output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {clip_output_dir}\n"
                "Run again with --overwrite to replace it."
            )
        shutil.rmtree(clip_output_dir)

    crops_dir = clip_output_dir / "crops"
    audio_dir = clip_output_dir / "audio"
    crops_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video:       {video_path}")
    print(f"Annotations: {xml_path}")
    print(f"Existing WAV:{source_wav_path}")

    tracks = parse_annotations(
        xml_path=xml_path,
        track_remap=track_remap,
    )

    track_bounds: Dict[int, Tuple[int, int]] = {}

    for track_id, track_frames in tracks.items():
        visible_frames = [
            ann.frame
            for ann in track_frames.values()
            if ann.outside == 0
        ]

        if not visible_frames:
            continue

        track_bounds[track_id] = (
            min(visible_frames),
            max(visible_frames),
        )

    fps, width, height, total_frames, crop_paths = create_crop_videos(
        video_path=video_path,
        tracks=tracks,
        track_bounds=track_bounds,
        crops_dir=crops_dir,
        padding=padding,
        crop_size=crop_size,
    )


    video_duration = total_frames / fps
    check_wav_duration(source_wav_path, video_duration)

    # Trim WAV per track using ffmpeg
    audio_paths: Dict[int, Path] = {}

    for track_id, (start_frame, end_frame) in track_bounds.items():
        start_sec = start_frame / fps
        end_sec = (end_frame + 1) / fps

        output_wav_path = audio_dir / f"{clip_name}_track_{track_id}.wav"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_wav_path),
                "-ss",
                str(start_sec),
                "-to",
                str(end_sec),
                str(output_wav_path),
            ],
            check=True,
        )

        audio_paths[track_id] = output_wav_path

    windows_csv = clip_output_dir / "training_windows.csv"
    window_count = write_training_windows(
        output_csv=windows_csv,
        clip_name=clip_name,
        tracks=tracks,
        crop_paths=crop_paths,

        audio_paths=audio_paths,
        clip_output_dir=clip_output_dir,
        fps=fps,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_segment_seconds=min_segment_seconds,
    )

    print()
    print("Done.")
    print(f"Resolution:       {width}x{height}")
    print(f"FPS:              {fps:.6f}")
    print(f"Frames (total):   {total_frames}")
    print(f"Dog crop videos:  {len(crop_paths)}")
    print(f"Training windows: {window_count}")
    print(f"Output:           {clip_output_dir}")

    return clip_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create moving dog crop videos, trim aligned WAV audio per track, "
            "and create minimal synchronized training-window metadata from CVAT XML."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing one MP4, annotations.xml, and one WAV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed"),
        help="Output root. Default: processed",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Padding added on every box side. Default: 0.10",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Square crop-video size. Default: 224",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="Target training-window length. Default: 2.0",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=1.0,
        help="Stride for long same-label sections. Default: 1.0",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.4,
        help="Skip visible same-label sections shorter than this. Default: 0.4",
    )
    parser.add_argument(
        "--track-remap",
        action="append",
        default=[],
        metavar="OLD:NEW",
        help=(
            "Optional: merge a later CVAT track into the same physical dog. "
            "Repeat as needed. Example: --track-remap 2:0"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output folder for this clip.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if not 0.0 <= args.padding <= 1.0:
            raise ValueError("--padding must be between 0 and 1.")
        if args.crop_size <= 0:
            raise ValueError("--crop-size must be positive.")
        if args.window_seconds <= 0:
            raise ValueError("--window-seconds must be positive.")
        if args.stride_seconds <= 0:
            raise ValueError("--stride-seconds must be positive.")
        if args.min_segment_seconds < 0:
            raise ValueError("--min-segment-seconds cannot be negative.")

        process(
            input_dir=args.input_dir,
            output_root=args.output_root,
            padding=args.padding,
            crop_size=args.crop_size,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            min_segment_seconds=args.min_segment_seconds,
            track_remap=parse_track_remaps(args.track_remap),
            overwrite=args.overwrite,
        )
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())