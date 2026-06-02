# Prompt 5 — M8 clips + M9 frame alignment

> Grouped executor prompt. Stop after completion; deliver §4 review summary.

## Scope

Implement **M8** (WAV clip extraction) and **M9** (video frame alignment) for the PANNs pipeline.

## Prerequisites

- `barkseqs.csv`, `audio.wav`, `metadata.json` under `outputs/<stem>/`.

## M8 — Clips

**Module:** `bark_detection/clips.py`  
**CLI stage:** `clips`

- Read `barkseqs.csv` and `audio.wav` (mono 16 kHz)
- For each row, slice padded `[clip_start_time_sec, clip_end_time_sec]` → `bark_event_clips/barkseq_NNN.wav`
- Padding: `clip_pre_context_sec` / `clip_post_context_sec` (default 0.25 s), clamped to audio bounds and neighbor events when `prevent_clip_overlap=true`
- Add `clip_*` columns to `barkseqs.csv`; event `start_time_sec` / `end_time_sec` unchanged
- File count == row count in `barkseqs.csv`

## M9 — Frames

**Module:** `bark_detection/frames.py`  
**CLI stage:** `frames`

- Read `metadata.json` for `fps`, `frame_count`
- `frame_id = round(time_sec * fps)`, clamp to `[0, frame_count - 1]`
- Add columns to `barkseqs.csv`: `start_frame`, `end_frame`, `peak_frame`, `context_start_frame`, `context_end_frame`
- Context: `start - pre_context_sec`, `end + post_context_sec` (defaults 0.5 s each)

## CLI

Extend `_STAGE_ORDER` with `clips` and `frames` after `export`.

## Acceptance

- 2 clip files on `dogs1`; listen and verify barks
- `barkseqs.csv` updated with frame columns; peak_frame aligns with audible bark at 30 fps
