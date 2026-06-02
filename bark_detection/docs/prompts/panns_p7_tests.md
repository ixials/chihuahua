# Prompt 7 — M11 unit tests

> Grouped executor prompt. Stop after completion; deliver §4 review summary.

## Scope

Add fast synthetic unit tests under `tests/` — no real PANNs inference in default runs.

## Test files

- `test_panns_windows.py` — overlapping window generation
- `test_timeline.py` — combined_bark_score from fake scores
- `test_barkseq_detect.py` — single / split / merge cases
- `test_noise.py` — clean, high_speech, high_music, likely_noise
- `test_frames.py` — round × fps, clamping, context
- `test_clips.py` — padded WAV extraction, overlap prevention
- `test_viz.py` — PNG creation (Agg backend)
- `test_schema.py` — expected CSV columns
- `test_integration_smoke.py` — skipped by default

## Acceptance

`pytest tests/` passes in under a few seconds.
