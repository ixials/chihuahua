# Prompt 4 — M6 noise flagging + M7 final export

> Grouped executor prompt. Stop after completion; deliver §4 review summary from `docs/bark_detection_plan.md`.

## Scope

Implement **M6** (noise flagging) and **M7** (canonical `barkseqs.csv` export) for the PANNs pipeline.

## Prerequisites

- M1–M5 complete: `barkseqs_initial.csv`, `bark_score_timeline.csv` exist under `outputs/<stem>/`.

## M6 — Noise flagging

**Module:** `bark_detection/noise.py`  
**CLI stage:** `noise`

For each row in `barkseqs_initial.csv`, slice `bark_score_timeline.csv` where `start_time_sec ≤ time_sec ≤ end_time_sec` and compute:

- `max_speech_score`, `max_music_score`

Classify with config thresholds (`speech_noise_threshold`, `music_noise_threshold`, reuse `barkseq_threshold` for strong_bark):

| Condition | `noise_reason` |
|-----------|----------------|
| speech/music below thresholds | `clean` |
| strong bark + high speech/music | `high_speech` / `high_music` / `high_speech_and_music` |
| weak bark + high speech/music | `likely_noise` |

Set `noise_flag = (noise_reason != "clean")`. **Do not drop rows.**

## M7 — Final export

**Module:** `bark_detection/barkseq_export.py`  
**CLI stage:** `export`

Write `outputs/<stem>/barkseqs.csv` with full schema:

- All M5 timing/score columns
- M6 noise fields
- `confidence = clip(mean_combined * (1 - α * max_speech) * (1 - β * max_music), 0, 1)`
- `method = panns_<model>_v1`

Config: `speech_penalty` (α), `music_penalty` (β).

## CLI

Extend default `_STAGE_ORDER` to include `noise` and `export` after `barkseqs`.

## Acceptance

- Every Barkseq row has `noise_flag` and `noise_reason`
- `len(barkseqs.csv) == len(barkseqs_initial.csv)` (no silent drops)
- All `confidence` values ∈ [0, 1]
- Append M6/M7 config decisions to `docs/decisions.md`
- Record Prompt 4 results in `docs/bark_detection_plan.md`

## Inspect on dogs1

- 2 rows, both likely `clean` (low in-span speech/music)
- Confidence ~0.48 and ~0.54 with α=β=0.3
- Listen after M8 clips land; for now spot-check CSV rows vs audible barks
