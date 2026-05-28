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