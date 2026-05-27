# Bark Event Detection — Plan

> Living plan document. Edit freely before approving each milestone. Implementation only proceeds for milestones you explicitly green-light.

---

## 1. Project context

The larger project is an **audio-visual dog-bark localization pipeline**. Given a video of multiple dogs, the eventual system will:

1. Detect and track every visible dog.
2. Detect bark events in the audio.
3. Assign each bark event to the correct tracked dog, or mark it `off-screen/unknown` if no visible dog matches.

This document covers **only step (2) — the audio bark-event detection module**. It answers a single question:

> *"When did bark-like sound events happen in the video?"*

It does **not** decide which dog barked. That belongs to a later module.

The repo currently contains:

```
audio_utils.py            # librosa + scipy STFT spectrogram helpers (kept as-is)
dog-bounding-box.ipynb    # YOLO/Ultralytics notebook for later vision work
files/dogs1.mp4           # 5 s, 30 fps, 150 frames, 44.1 kHz stereo (sample video)
files/dogs1.wav           # 44.1 kHz stereo (sample raw audio)
README.md
requirements.txt          # numpy, scipy, ultralytics, opencv-python, pillow, matplotlib, librosa
```

Audited environment: Python 3.11, `numpy/scipy/pandas/matplotlib/opencv` installed; `ffmpeg` and `ffprobe` available; **librosa is *not* installed** locally. We will proceed without installing librosa (see §8).

---

## 2. Research paper motivation

Two papers anchor the design:

- **"What's That Sound Right Now? Video-centric Audio-Visual Localization."** Uses **RMS audio energy** to find audio-active regions and sample clips/frames around sound events. This is why we start with an RMS-energy baseline to find candidate bark timestamps before introducing any learned classifier.

- **"Active Speakers in Context."** Uses **Mel-spectrogram audio features paired with per-track visual candidates** to classify each person as active/silent. Forward-looking analogue for us: each tracked dog will eventually receive a barking/not-barking score. The audio module here is therefore designed so a learned classifier (CAV-MAE, YAMNet, PANNs, or a custom bark/non-bark head) can be slotted into the confidence-scoring stage later — without rewriting the earlier stages.

---

## 3. Scope

**In scope (this document):**
- Extracting standardized mono 16 kHz audio from MP4.
- Computing an RMS energy curve.
- Smoothing and adaptively thresholding the curve.
- Producing contiguous candidate regions, then merged, duration-filtered, confidence-scored **bark events**.
- Grouping nearby events into **bark episodes**.
- Aligning all timestamps to video frame indices and producing context windows for later visual association.
- A debug plot, modular code, unit tests, and a code review.

**Explicitly out of scope (handled later):**
- Tracking which dog barked.
- Visual detection / tracking.
- A trained bark/non-bark classifier (but the pipeline must accept one as a drop-in).
- Multi-microphone source localization.

---

## 4. Milestone list (M0–M10)

| # | Milestone | Status |
|---|---|---|
| M0 | Inspect repo + write this plan | proposed |
| M1 | Audio extraction + metadata | ✅ completed 2026-05-27 |
| M2 | RMS energy curve | ✅ completed 2026-05-27 |
| M3 | Smoothing + adaptive threshold | ✅ completed 2026-05-27 |
| M4 | Candidate region detection | ✅ completed 2026-05-27 |
| M5 | Merge nearby peaks + duration rules | proposed |
| M6 | Bark confidence scoring (with classifier hook) | proposed |
| M7 | Bark episodes | proposed |
| M8 | Align events to video frames + context windows | proposed |
| M9 | Debug visualization | proposed |
| M10 | Tests + code review | proposed |

Each milestone is detailed in §5. We **stop after every milestone**, summarize, and wait for explicit approval before starting the next one.

---

## 5. Milestones in detail

### M0 — Inspect repo + write this plan

- **Goal:** Audit the environment, confirm dependencies, and produce this document.
- **Why it matters:** Aligns on scope and structure before any code is written. Avoids over-building.
- **Files created/edited:** `docs/bark_detection_plan.md` (this file).
- **Inputs:** Current repo state.
- **Outputs/artifacts:** This Markdown plan.
- **Acceptance criteria:** You can read this end-to-end and feel the plan matches your intent.
- **What you should inspect before approving M1:**
  1. The milestone list and scope statement.
  2. The file structure proposed in §7.
  3. The default parameters in §8.
  4. Library choice (scipy now, librosa/CAV-MAE/YAMNet/PANNs later).

---

### M1 — Audio extraction + metadata

- **Goal:** Convert MP4 audio into a standardized WAV (mono, 16 kHz) and capture video/audio metadata that every later stage will key off.
- **Why it matters:** All later timestamps must align to a known sample rate and to known video frames. Standardizing now means later stages never have to re-handle stereo/48 kHz/odd sample rates.
- **Files created/edited:**
  - `bark_detection/__init__.py`
  - `bark_detection/config.py` (`BarkConfig` dataclass — full parameter set, used by all stages)
  - `bark_detection/audio_io.py` (ffmpeg extraction + ffprobe metadata)
  - `bark_detection/cli.py` (initial CLI; supports the `extract` stage)
  - `run_bark_detection.py` (thin shim → `bark_detection.cli:main`)
  - `.gitignore` (add `outputs/`)
- **Inputs:** `--video <path.mp4>`, `--output_dir <dir>`.
- **Outputs/artifacts:**
  - `outputs/<video_stem>/audio.wav` — mono, 16 kHz, 16-bit PCM
  - `outputs/<video_stem>/metadata.json` — four fields the pipeline actually needs:
    ```json
    {
      "fps": 30.0,
      "frame_count": 150,
      "sample_rate_hz": 16000,
      "duration_sec": 5.014063
    }
    ```
- **Acceptance criteria:**
  - `ffprobe outputs/<stem>/audio.wav` reports `mono, 16 kHz`.
  - `metadata.json` fields match `ffprobe` on the source video (±1 frame / ±10 ms tolerance for duration).
  - Audio duration ≈ video duration; any discrepancy is reported but not silently fixed.
- **What you should inspect:**
  1. The exact ffmpeg command logged in `metadata.json`.
  2. The reported fps / frame_count / duration values.
  3. Whether the standardized WAV duration matches the source within tolerance.

#### M1 — results (2026-05-27, slimmed to Option B)
- Files: `bark_detection/{__init__.py, config.py, audio_io.py, cli.py}`, `run_bark_detection.py`, `.gitignore`.
- Run dir: `outputs/dogs1/` (artifacts: `audio.wav`, `metadata.json`).
- Source video: fps=30.0, frame_count=150.
- Standardized audio: sample_rate=16000 Hz, duration=5.014063 s.
- Δ(audio − video) = +14.063 ms — sub-frame at 30 fps; AAC priming, will revisit at M8 if needed.
- Acceptance: WAV mono/16 kHz ✅; metadata.json matches ffprobe ✅; duration delta noted ⚠ (sub-frame).

---

### M2 — RMS energy curve

- **Goal:** Compute a per-frame RMS energy curve from the standardized WAV.
- **Why it matters:** Barks tend to appear as short high-energy spikes. RMS is a cheap, interpretable **candidate event detector** — not a final bark classifier. Mirrors the RMS gating used in *"What's That Sound Right Now?"*
- **Files created/edited:**
  - `bark_detection/rms.py`
  - `bark_detection/cli.py` (adds the `rms` stage)
  - `bark_detection/viz.py` (first version of `rms_debug_plot.png`)
- **Inputs:** `outputs/<stem>/audio.wav`, config (`window_ms=50`, `hop_ms=10`).
- **Outputs/artifacts:**
  - `outputs/<stem>/rms_values.csv` with columns:
    - `time_sec`
    - `rms`
  - `outputs/<stem>/rms_debug_plot.png` (first version: raw RMS over time).
- **Acceptance criteria:**
  - RMS array length ≈ `floor((num_samples - window) / hop) + 1`.
  - All values ≥ 0; no NaN/Inf.
  - Visible spikes during the audible barks on the sample video.
- **What you should inspect:**
  1. Are window/hop sizes appropriate for short bark spikes? (default: 50 ms / 10 ms.)
  2. Does the plot show clear spikes at audible bark times?
  3. Is the CSV size reasonable (~100 rows/sec)?

#### M2 — results (2026-05-27)
- Files: `bark_detection/rms.py`, `bark_detection/viz.py`; extended `bark_detection/{config.py, cli.py}`.
- Artifacts: `outputs/dogs1/rms_values.csv`, `outputs/dogs1/rms_debug_plot.png`.
- RMS array length: 497; expected from formula `floor((80225 - 800) / 160) + 1 = 497`; match: PASS.
- RMS range: min=0.003431, max=0.29201, mean=0.030363; all finite & ≥ 0: PASS.
- CSV row count: 497; rows-per-second: ≈99.70.
- Plot saved at: `outputs/dogs1/rms_debug_plot.png`.
- Acceptance: all PASS.

---

### M3 — Smoothing + adaptive threshold

- **Goal:** Smooth the RMS curve and compute an adaptive energy threshold that separates "interesting" from "background."
- **Why it matters:** Raw RMS is noisy and the right threshold depends on the clip (a quiet room vs. a windy yard). An adaptive threshold (mean + k·σ or percentile) generalizes across clips without hand-tuning.
- **Files created/edited:**
  - `bark_detection/threshold.py`
  - extend `bark_detection/rms.py` (fills `rms_smoothed` column)
  - extend `bark_detection/viz.py`
- **Inputs:** `rms_values.csv`, config (`smoothing_method`, `smoothing_window_ms`, `threshold_method`, `mean_std_k`, `percentile`).
- **Outputs/artifacts:**
  - `rms_values.csv` updated with populated `rms_smoothed` (columns: `time_sec`, `rms`, `rms_smoothed`).
  - Threshold values written to `metadata.json` under a `thresholding` key (both `mean_std_k2` and `percentile_90`, plus the chosen `operating_point`).
  - `rms_debug_plot.png` updated with: raw RMS, smoothed RMS, threshold line.
- **Acceptance criteria:**
  - Smoothed curve has lower variance than raw curve (numeric check).
  - Threshold lies between `mean(rms_smoothed)` and `max(rms_smoothed)`.
- **Default recommendation:** smoothing = moving average (window 50 ms); threshold = `mean + 2·σ` of smoothed RMS. Reason: moving average is the simplest, fastest baseline and easy to reason about; `mean + 2σ` adapts to clip energy and is robust to long quiet stretches where percentile thresholds would still fire on noise. **We will also log the percentile-based threshold for comparison so you can choose.**
- **What you should inspect:**
  1. Plot of smoothed curve + threshold line — does it visibly sit *just above* background and *below* bark peaks?
  2. Compare `mean+2σ` vs. `90th percentile` values in `metadata.json`.

#### M3 — results (2026-05-27)
- Files: `bark_detection/threshold.py`; extended `bark_detection/{config.py, cli.py, viz.py}`.
- Artifacts: `outputs/dogs1/rms_values.csv` (now with `rms_smoothed`), `outputs/dogs1/metadata.json` (now with `thresholding`), `outputs/dogs1/rms_debug_plot.png` (updated).
- Variance: raw RMS = 0.002317, smoothed RMS = 0.002179 (smoothed < raw: PASS).
- Thresholds: `mean+2σ` = 0.123633, `90th percentile` = 0.053196. Operating point: `mean_std` (0.123633).
- Bound check: mean(rms_smoothed) = 0.030360 < operating_point 0.123633 < max(rms_smoothed) = 0.284253: PASS.
- Acceptance: all PASS.

---

### M4 — Candidate region detection

- **Goal:** Find every contiguous region where smoothed RMS exceeds the threshold.
- **Why it matters:** These are *candidate sound events* — not confirmed barks. We treat M4's output as recall-favored; later stages tighten precision.
- **Files created/edited:**
  - `bark_detection/candidates.py`
  - `bark_detection/cli.py` (adds the `candidates` stage)
- **Inputs:** `rms_values.csv`, threshold value, config.
- **Outputs/artifacts:**
  - `outputs/<stem>/bark_candidates.csv` with columns:
    - `candidate_id`
    - `start_time_sec`
    - `end_time_sec`
    - `peak_time_sec`
    - `duration_sec`
    - `rms_peak`
    - `normalized_rms_peak` (peak / global max RMS in clip)
- **Acceptance criteria:**
  - `start_time_sec < peak_time_sec ≤ end_time_sec` for every row.
  - `normalized_rms_peak ∈ (0, 1]`.
  - Number of candidates ≥ number of audible barks (recall-favored).
- **What you should inspect:**
  1. Total candidate count vs. your manual count of audible barks.
  2. A few candidate rows — do their times match what you hear?

#### M4 — results (2026-05-27)
- Files: `bark_detection/candidates.py`; extended `bark_detection/cli.py`.
- Artifacts: `outputs/dogs1/bark_candidates.csv`.
- Total candidates: 3.
- Threshold used: 0.123633, method: mean_std.
- Range checks: `start_time_sec < peak_time_sec ≤ end_time_sec` holds for all rows: PASS; `normalized_rms_peak ∈ (0, 1]` (min=0.632933, max=1.000000): PASS.
- Sample rows:
  ```
   candidate_id  start_time_sec  peak_time_sec  end_time_sec  duration_sec  rms_peak  normalized_rms_peak
              0           0.190          0.245         0.330         0.140  0.284253             1.000000
              1           1.480          1.505         1.540         0.060  0.179913             0.632933
              2           1.670          1.705         1.760         0.090  0.215209             0.757103
  ```
- Acceptance: all PASS.

---

### M5 — Merge nearby peaks + duration rules

- **Goal:** Merge candidates that are clearly the same event split by a brief dip, then drop noise-shaped events and flag overly-long events.
- **Why it matters:** RMS thresholding alone fragments single barks into 2–3 pieces and also picks up clicks/long noises. This stage applies physically-motivated bark-shape rules.
- **Files created/edited:**
  - `bark_detection/events.py`
  - `bark_detection/cli.py` (adds the `events` stage)
- **Inputs:** `bark_candidates.csv`, config (`merge_gap_ms` default 200; `min_duration_ms` 80; `max_duration_ms` 1200).
- **Outputs/artifacts:**
  - `outputs/<stem>/bark_events_initial.csv` (same schema as candidates plus `event_type ∈ {bark, long_event}` and `merged_from` list).
- **Acceptance criteria:**
  - After merging, no two events are within `merge_gap_ms` of each other.
  - Every kept event satisfies `min_duration_ms ≤ duration ≤ max_duration_ms` **unless** flagged `long_event`.
  - Long events are **flagged, not deleted**. If reasonable internal peaks exist, they may be split.
- **What you should inspect (summary will be printed):**
  1. Raw candidate count → after merge → after duration filter.
  2. A few examples of kept vs. removed events.

---

### M6 — Bark confidence scoring (with classifier hook)

- **Goal:** Assign each event a `bark_confidence ∈ [0, 1]` from cheap signal-shape features, and expose a clean seam for a future trained classifier.
- **Why it matters:** Downstream visual association needs a continuous score, not a binary label. Also, the architecture must allow swapping in CAV-MAE / YAMNet / PANNs / a custom bark/non-bark head later **without** changing earlier stages.
- **Files created/edited:**
  - `bark_detection/scoring.py`
- **Default formula (heuristic, no training required):**
  ```
  duration_score   = triangle(duration_sec, peak=0.2s, min=0.08s, max=1.2s)
  bark_confidence  = w_rms * normalized_rms_peak + w_dur * duration_score
  ```
  Defaults: `w_rms = 0.6`, `w_dur = 0.4`. SNR estimate is reserved as an additive term (off by default).
- **Classifier hook:**
  ```python
  class BarkClassifier(Protocol):
      def score(self, wav: np.ndarray, sr: int, t_start: float, t_end: float) -> float: ...
  ```
  If `cfg.classifier is not None`, `bark_confidence` becomes a config-weighted blend of the heuristic and the classifier score. Default classifier is `None`.
- **Inputs:** `bark_events_initial.csv`, `audio.wav`, config.
- **Outputs/artifacts:**
  - `outputs/<stem>/bark_events.csv` with columns:
    - `event_id`, `start_time_sec`, `peak_time_sec`, `end_time_sec`, `duration_sec`,
      `rms_peak`, `normalized_rms_peak`, `bark_confidence`, `event_type`
    - frame columns and context columns added later in M8
    - `episode_id` added in M7
- **Acceptance criteria:**
  - `bark_confidence ∈ [0, 1]`.
  - Holding `duration_score` constant, `bark_confidence` is monotonic in `normalized_rms_peak`.
- **What you should inspect:**
  1. The exact formula and weights.
  2. The classifier-hook protocol signature.

---

### M7 — Bark episodes

- **Goal:** Group temporally-close bark events into episodes/bursts.
- **Why it matters:** A barking burst is more visually identifiable than a single bark. Later, the dog-association module may prefer episode-level features.
- **Files created/edited:**
  - `bark_detection/episodes.py`
  - `bark_detection/cli.py` (adds the `episodes` stage)
- **Inputs:** `bark_events.csv`, config (`episode_gap_sec` default 0.75).
- **Outputs/artifacts:**
  - `outputs/<stem>/bark_episodes.csv` with columns:
    - `episode_id`, `start_time_sec`, `end_time_sec`, `number_of_barks`,
      `average_confidence`, `max_confidence`
    - frame columns added in M8
  - `bark_events.csv` updated with `episode_id`.
- **Acceptance criteria:**
  - Every event's `[start, end]` is contained within exactly one episode's `[start, end]`.
  - `number_of_barks` matches the count of events with that `episode_id`.
- **What you should inspect:**
  1. The grouping for a known burst on the sample clip.
  2. Whether the default `episode_gap_sec` (0.75 s) is right for your data.

---

### M8 — Align events to video frames + context windows

- **Goal:** Convert every timestamp to a video frame index and add a context window around each event for later visual association.
- **Why it matters:** Downstream dog-tracking work indexes by frame, not seconds. Context windows give the visual module a buffer to look for a mouth-open / motion cue *just before* and *just after* the bark.
- **Files created/edited:**
  - `bark_detection/frames.py`
- **Formula:** `frame_id = round(timestamp_sec * fps)`, clamped to `[0, frame_count - 1]`.
- **Context:** defaults `pre_context = 0.5 s`, `post_context = 0.5 s`.
- **Inputs:** `bark_events.csv`, `bark_episodes.csv`, `metadata.json`.
- **Outputs/artifacts:**
  - `bark_events.csv` updated with: `start_frame`, `peak_frame`, `end_frame`,
    `context_start_time_sec`, `context_end_time_sec`,
    `context_start_frame`, `context_end_frame`.
  - `bark_episodes.csv` updated with: `start_frame`, `end_frame`,
    `context_start_time_sec`, `context_end_time_sec`,
    `context_start_frame`, `context_end_frame`.
- **Acceptance criteria:**
  - `start_frame == round(start_time_sec * fps)` for every event.
  - `0 ≤ frame ≤ frame_count - 1` after clamping.
  - `context_start_frame ≤ start_frame` and `end_frame ≤ context_end_frame`.
- **What you should inspect:**
  1. Spot-check 2–3 events: open the video at `peak_frame`, confirm you see/hear the bark.

---

### M9 — Debug visualization

- **Goal:** Produce one publication-style debug plot that summarizes the pipeline result.
- **Why it matters:** Most of the value of a research module is the ability to *show* it works in a meeting or paper figure.
- **Files created/edited:**
  - `bark_detection/viz.py` (final version).
- **Outputs/artifacts:**
  - `outputs/<stem>/rms_debug_plot.png` showing:
    - raw RMS
    - smoothed RMS
    - threshold line
    - shaded detected event regions
    - peak markers
    - (optional) shaded episode regions in a lighter color
    - labels with `event_id` and `bark_confidence`
- **Acceptance criteria:** PNG opens; all listed layers visible; legend present.
- **What you should inspect:**
  1. Does the figure tell a clear story at a glance?
  2. Any visual layer worth adding (e.g. waveform under the RMS)?

---

### M10 — Tests + code review

- **Goal:** Add unit tests and run an independent code review focused on modularity and the classifier-plug-in seam.
- **Why it matters:** Locks in correct behavior before later modules start importing from this one.
- **Files created/edited:**
  - `tests/test_frames.py` — timestamp ↔ frame conversion + clamping.
  - `tests/test_rms.py` — RMS shape/output sanity on synthetic input.
  - `tests/test_events.py` — merging logic + min/max duration filtering.
  - `tests/test_episodes.py` — episode grouping logic.
- **Agents invoked:**
  - **test-automator** to author the tests.
  - **code-reviewer** to inspect modularity, clarity, edge cases, and whether `scoring.py`'s `BarkClassifier` hook is well-shaped.
- **Acceptance criteria:**
  - `pytest tests/` is green.
  - Code-reviewer flags no critical issues for the classifier-plug-in seam.
- **What you should inspect:**
  1. The test list and what each one asserts.
  2. The code-reviewer summary.

---

## 6. File structure

```
bark_detection/
  __init__.py
  config.py        # BarkConfig dataclass (all tunables; see §8)
  audio_io.py      # M1: ffmpeg extract → mono 16 kHz WAV; ffprobe → metadata.json
  rms.py           # M2: framed RMS curve → rms_values.csv
  threshold.py     # M3: smoothing + adaptive threshold
  candidates.py    # M4: contiguous above-threshold regions → bark_candidates.csv
  events.py        # M5: merge + duration filter → bark_events_initial.csv
  scoring.py       # M6: heuristic confidence + BarkClassifier hook → bark_events.csv
  episodes.py      # M7: group events → bark_episodes.csv
  frames.py        # M8: timestamp ↔ frame + context windows
  viz.py           # M9: rms_debug_plot.png
  cli.py           # argparse entry; supports --stage and --from-stage

tests/
  test_frames.py
  test_rms.py
  test_events.py
  test_episodes.py

docs/
  bark_detection_plan.md       # this document

run_bark_detection.py          # thin shim → bark_detection.cli.main()
outputs/                       # gitignored; flat per-video dir: outputs/<video_stem>/
```

Per-video artifacts (under `outputs/<video_stem>/`):

```
audio.wav                # M1
metadata.json            # M1
rms_values.csv           # M2
bark_candidates.csv      # M4
bark_events_initial.csv  # M5
bark_events.csv          # M6 / M8
bark_episodes.csv        # M7
rms_debug_plot.png       # M9
```

Each stage reads its predecessor's CSV and writes its own. The CLI supports `--stage <name>` (run one stage) and `--from-stage <name>` (resume from a stage), so you can re-tune any milestone without re-extracting audio.

Preferred command:

```bash
python run_bark_detection.py --video files/dogs1.mp4 --output_dir outputs/
```

---

## 7. (Folded into §6 above.)

---

## 8. Configuration parameters (`BarkConfig`)

All tunables live in one dataclass. Defaults are starting points, not commitments — every value can be overridden from the CLI.

| Group | Field | Default | Used in |
|---|---|---:|---|
| audio | `target_sample_rate_hz` | 16000 | M1 |
| audio | `target_channels` | 1 | M1 |
| rms | `window_ms` | 50 | M2 |
| rms | `hop_ms` | 10 | M2 |
| smoothing | `smoothing_method` | `"moving_average"` (`"median"` also supported) | M3 |
| smoothing | `smoothing_window_ms` | 50 | M3 |
| threshold | `threshold_method` | `"mean_std"` (`"percentile"` also supported) | M3 |
| threshold | `mean_std_k` | 2.0 | M3 |
| threshold | `percentile` | 90 | M3 |
| events | `merge_gap_ms` | 200 | M5 |
| events | `min_duration_ms` | 80 | M5 |
| events | `max_duration_ms` | 1200 | M5 |
| scoring | `w_rms` | 0.6 | M6 |
| scoring | `w_duration` | 0.4 | M6 |
| scoring | `duration_score_peak_sec` | 0.20 | M6 |
| scoring | `classifier` | `None` | M6 (optional) |
| scoring | `classifier_weight` | 0.5 | M6 (when classifier set) |
| episodes | `episode_gap_sec` | 0.75 | M7 |
| frames | `pre_context_sec` | 0.5 | M8 |
| frames | `post_context_sec` | 0.5 | M8 |

> **Implementation note:** `BarkConfig` in `bark_detection/config.py` only declares the fields that the currently-implemented milestones use. M1 has only `target_sample_rate_hz` and `target_channels`. Each later milestone adds its own fields when it lands. The table above is the *target* shape.

---

## 9. Library strategy: scipy now, librosa/CAV-MAE/YAMNet/PANNs later

### Now (M1–M9)

Use only what's already installed:

- `subprocess` + `ffmpeg` / `ffprobe` — audio extraction, metadata.
- `scipy.io.wavfile` — read the standardized 16 kHz mono WAV.
- `numpy` — framed RMS, smoothing.
- `scipy.signal` — optional median filter, peak helpers.
- `pandas` — CSV I/O for stage artifacts.
- `matplotlib` — debug plot.

This avoids installing librosa (which pulls `numba` + `audioread`) just to read a WAV. The existing `audio_utils.py` (which *does* use librosa) is **kept untouched** for the separate spectrogram/visualization workflow.

### Later (M6+ classifier plug-in)

The scoring stage exposes a single Protocol:

```python
class BarkClassifier(Protocol):
    def score(self, wav: np.ndarray, sr: int, t_start: float, t_end: float) -> float:
        """Return P(bark) ∈ [0, 1] for the clip wav[int(t_start*sr):int(t_end*sr)]."""
```

Adapters that satisfy this protocol can wrap:

- **YAMNet** (TF Hub) — built-in `Bark` class id; very fast first baseline.
- **PANNs** (`panns_inference`) — `Dog` / `Bark, woof, yip` AudioSet classes; PyTorch.
- **CAV-MAE** — audio-visual encoder; can produce audio-only embeddings; train a small head.
- **Custom bark/non-bark head** — log-Mel features + small classifier on collected data.

When `cfg.classifier` is provided, final confidence is:

```
final = (1 - α) * heuristic + α * classifier.score(...)
```

with `α = cfg.classifier_weight`. Earlier stages (M1–M5, M7–M9) **do not change** when a classifier is plugged in.

---

## 10. Stop/go checklist after every milestone

Before starting milestone `N+1`, I will:

1. Print a short summary of what M`N` produced.
2. List every file created/modified.
3. List every artifact in `outputs/<stem>/` and the columns/values you should glance at.
4. Flag any anomaly (sample-rate mismatch, suspicious counts, NaN, etc.).
5. Recommend the next step.
6. **Wait for your explicit "go M`N+1`" before continuing.**

You can at any point:

- Edit this plan and tell me to re-read it.
- Tell me to redo a milestone with different parameters.
- Skip ahead (not recommended) or back up.

---

## Open items for your review

- Sample video for M1: `files/dogs1.mp4` (only video present). Add more before later milestones if you have them.
- Confirm the default parameters in §8 are reasonable starting points.
- Confirm scipy-now / librosa-later library strategy in §9.
- Confirm the milestone gating workflow in §10 matches how you want to collaborate.
