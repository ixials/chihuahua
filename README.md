# Chihuahua

Audio-visual dog bark detection. The **bark_detection** module finds when barking happens in a video (PANNs pipeline, M1–M11).

## Run bark detection

```bash
python run_bark_detection.py --video files/"dogname.mp4" --output_dir outputs/
Only thing you change is the dogname.mp4
```

## Main output

**`outputs/<video_name>/barkseqs.csv`** — final Barkseq table (timings, scores, confidence, frame indices, clip paths).

## Output layout

```
outputs/<video_name>/
  audio.wav                 # extracted mono 16 kHz audio
  metadata.json             # fps, frame_count, duration, sample rate
  barkseqs.csv              # final result — start here
  bark_event_clips/         # one padded WAV per Barkseq (listening / features)
    barkseq_000.wav
    ...
  debug/                    # inspection plots
    panns_score_timeline.png
    barkseq_overlay.png
  intermediate/             # diagnostic / stage-to-stage CSVs (safe to ignore)
    panns_windows.csv
    panns_scores.csv
    bark_score_timeline.csv
    barkseqs_initial.csv
    panns_label_list.txt
    panns_label_mapping.txt
  legacy_rms/               # only if you run legacy --stage rms … score
```

**Final artifacts (top level):** `audio.wav`, `metadata.json`, `barkseqs.csv`, `bark_event_clips/`, `debug/`

**Intermediate artifacts:** everything under `intermediate/` — used by the pipeline between stages; useful for debugging thresholds and PANNs scores.

**Default detection (vocalization Phase 1):** expanded dog-vocal PANNs labels (Bark, Yip, Bow-wow, Howl, Growling, Whimper (dog)), threshold **0.30**, combined mode **`max_bark_dog`**, merge gap **0.35 s**. Use CLI flags below to revert toward precision-first settings.

## Fast tuning (after first run)

PANNs inference is slow. Once you have `panns_scores.csv` on disk you can skip it and iterate quickly on thresholds and detection settings.

```bash
# One full run first (slow - includes PANNs)
python run_bark_detection.py --video files/your_clip.mp4

# Fast retune - skip PANNs, try new threshold
python run_bark_detection.py --video files/your_clip.mp4 --from-stage timeline --threshold 0.30

# Try max_bark_dog mode
python run_bark_detection.py --video files/your_clip.mp4 --from-stage timeline --threshold 0.30 --combined-mode max_bark_dog
```

Available tuning flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--from-stage STAGE` | Run from this stage through viz, reusing earlier outputs | — |
| `--threshold FLOAT` | Override `barkseq_threshold` | 0.30 |
| `--combined-mode MODE` | `bark` or `max_bark_dog` | `max_bark_dog` |
| `--merge-gap FLOAT` | Override `merge_gap_sec` | 0.35 |

## Label inspection (Phase 2)

Diagnose missed or false-positive detections by examining individual PANNs vocalization label scores per window.

```bash
# After extract + windows exist (or after a full run):
python run_bark_detection.py --video files/your_clip.mp4 --stage inspect

# Query a missed sound at 2.3 s:
python run_bark_detection.py --video files/your_clip.mp4 --stage inspect --at-time 2.3

# Query multiple times:
python run_bark_detection.py --video files/your_clip.mp4 --stage inspect --at-time 2.3 --at-time 5.1
```

**Outputs:**

| Path | Description |
|------|-------------|
| `intermediate/vocalization_scores.csv` | Per-window scores for Bark, Yip, Bow-wow, Howl, Growling, Whimper (dog), Dog, plus `vocalization_max_score` and `max_bark_dog_score` aggregates. |
| `debug/vocalization_inspect.png` | Line plot of all seven label scores over time with the detection threshold. |

**`--at-time FLOAT`** (repeatable) — prints a human-readable summary for the window nearest to each queried time, showing every label score and flagging those above the threshold.

## Tests

```bash
pytest tests/
```

Fast synthetic tests only; no real PANNs inference in the default suite.

## Other

Command to extract audio from video:
```
ffmpeg -i files/dogs1.mp4 files/dogs1.wav
```

Command to combine audio and video:

```
ffmpeg -i files/dogs1-output.mp4 -i files/dogs1.wav -c:v copy -c:a aac -shortest files/dogs1-final.mp4
```