# Bark Detection — Plan (PANNs-first)

> Living plan document. Edit freely before approving each prompt group. **Implementation only proceeds for groups you explicitly green-light** (see §4).

**Design pivot (2026-05-28):** The main detector is **PANNs** (Pretrained Audio Neural Networks) on overlapping audio windows, not RMS energy thresholding. RMS-based stages remain in the repo as **legacy modules** but are **not wired into the default CLI** after the PANNs pipeline lands.

---

## 1. Project context

The larger project is an **audio-visual dog bark localization pipeline**. Given a video of multiple dogs, the eventual system will:

1. Detect and track every visible dog (YOLO/tracking — **not this module**).
2. **Detect when barking / dog vocalization happens in the audio** — **this module**.
3. Assign each detected **Barkseq** to the correct tracked dog, or mark `off-screen/unknown` if no visible dog matches — **future work**.

This document covers **only step (2)** — the audio preprocessing and **Barkseq** detection module. It answers:

> *"When did dog-vocalization bursts (Barkseqs) occur in the video?"*

It does **not** assign Barkseqs to dogs. It does **not** build YOLO/tracking.

### Terminology

| Term | Definition |
|------|------------|
| **Barkseq** | A continuous sequence or burst of dog vocalization, possibly containing one or more individual barks. **Primary output unit.** |
| **Bark Unit** | One individual bark inside a Barkseq. **Deferred** — can be added later via sub-segmentation. |
| **combined_bark_score** | Scalar timeline score derived from PANNs dog/bark-related labels (exact formula fixed in M4 after label inspection in M3). |

### Repo anchors (unchanged)

```
audio_utils.py            # librosa spectrogram helpers (separate workflow; not bark_detection CLI)
dog-bounding-box.ipynb    # YOLO/Ultralytics — stage 1 vision (future)
files/dogs1.mp4           # 5 s sample video
run_bark_detection.py     # shim → bark_detection.cli:main
outputs/                  # gitignored; flat outputs/<video_stem>/
```

---

## 2. Research motivation

Two papers anchor the **overall** audio-visual system:

- **"What's That Sound Right Now?"** — model-based sound activity and clip sampling around events (not RMS-only gating as the final detector).
- **"Active Speakers in Context"** — per-track audio features + visual candidates → active/silent; forward analogue for stage 3 (assign Barkseq → dog).

**Adapted pipeline for this project:**

```
MP4 video
  → extract mono 16 kHz audio
  → (optional future: denoising / source separation)
  → model-based sound event detection (PANNs on overlapping windows)
  → Barkseq extraction from score timeline
  → optional noise flagging (speech / music)
  → per-Barkseq WAV clips
  → align timestamps to video frames
  → CSVs + metadata + debug plots
```

**Why PANNs-first:** Mentor-style pipelines use **learned tagging / SED**, not hand-tuned RMS thresholds, as the primary event finder. Existing RMS work (M2–M6 legacy) is **not wasted** — it remains available for optional baseline comparison, timing refinement, and debug plots, but is **demoted** from the default path.

---

## 3. Scope

**In scope**

- M1–M11 as defined below (extract → PANNs windows → inference → timeline → Barkseqs → noise flags → export → clips → frames → viz → tests).
- Inspectable CSV/JSON/PNG artifact at every stage; review checkpoint after each **prompt group** (§4).
- Flat `outputs/<video_stem>/` layout.

**Out of scope**

- Assigning Barkseqs to tracked dogs.
- YOLO / multi-dog tracking.
- Bark Unit sub-segmentation inside a Barkseq (future).
- RMS as the **main** detector (legacy code may stay on disk, unwired from default CLI).

---

## 4. Milestone list (M1–M11)

| # | Milestone | Status |
|---|---|---|
| M1 | Extract mono 16 kHz WAV + metadata | ✅ completed 2026-05-28 (reuse) |
| M2 | Overlapping PANNs windows | ✅ completed 2026-05-28 |
| M3 | PANNs inference per window | ✅ completed 2026-05-28 |
| M4 | Bark score timeline | ✅ completed 2026-05-28 |
| M5 | Detect Barkseq candidates | ✅ completed 2026-05-28 |
| M6 | Noise flagging (speech / music) | ✅ completed 2026-05-28 |
| M7 | Final Barkseq export + confidence | ✅ completed 2026-05-28 |
| M8 | Extract Barkseq WAV clips | ✅ completed 2026-05-28 |
| M9 | Align Barkseqs to video frames | ✅ completed 2026-05-28 |
| M10 | Debug visualizations | ✅ completed 2026-05-28 |
| M11 | Unit tests | ✅ completed 2026-05-28 |

**Process:** Implement in **seven grouped prompts** (not one prompt per milestone). Stop after each group, deliver a review summary (see below), and wait for explicit approval before starting the next group.

| Prompt | Milestones | Saved prompt file (target) |
|--------|------------|----------------------------|
| **Prompt 1** | M1 + M2 | `docs/prompts/panns_p1_extract_windows.md` |
| **Prompt 2** | M3 | `docs/prompts/panns_p2_inference.md` |
| **Prompt 3** | M4 + M5 | `docs/prompts/panns_p3_timeline_barkseqs.md` |
| **Prompt 4** | M6 + M7 | `docs/prompts/panns_p4_noise_export.md` |
| **Prompt 5** | M8 + M9 | `docs/prompts/panns_p5_clips_frames.md` |
| **Prompt 6** | M10 | `docs/prompts/panns_p6_viz.md` |
| **Prompt 7** | M11 | `docs/prompts/panns_p7_tests.md` |

**After each grouped prompt completes**, the executor must stop and report (in order):

1. **Files created/modified** — paths only.
2. **Artifacts generated** — CSV/JSON/PNG/WAV paths under `outputs/<stem>/`.
3. **Key numbers** — row counts, score ranges, Barkseq count, threshold used, min/max confidence, etc.
4. **What to inspect** — 2–4 concrete checks for the user (listen to a clip, open a plot, spot-check a CSV row, compare to audible ground truth on `dogs1`).
5. **PASS/FAIL** — against that group's acceptance criteria.
6. **Recommended next step** — do not start the next prompt group until the user approves.

### Legacy RMS milestones (superseded)

The following were completed under the old RMS-first plan and are **frozen as reference implementation** only:

| Old | Topic | Repo module | Default CLI |
|-----|--------|-------------|-------------|
| M2 | Framed RMS | `rms.py` | unwired |
| M3 | Smooth + threshold | `threshold.py` | unwired |
| M4 | RMS candidates | `candidates.py` | unwired |
| M5 | Merge + duration | `events.py` | unwired |
| M6 | Heuristic confidence | `scoring.py` | unwired |

Artifacts such as `legacy_rms/{rms_values.csv, bark_candidates.csv, bark_events*.csv}` on `outputs/dogs1/` remain useful for **comparison** against PANNs Barkseqs. Do not delete legacy modules until PANNs M11 tests pass unless the user asks.

---

## 5. Milestones in detail

### M1 — Extract mono 16 kHz WAV + metadata

- **Goal:** Standardize audio from MP4 and capture video timing metadata.
- **Files:** `bark_detection/audio_io.py`, `bark_detection/config.py`, `bark_detection/cli.py` (stage `extract`).
- **Outputs:**
  - `outputs/<stem>/audio.wav` — mono, 16 kHz, 16-bit PCM
  - `outputs/<stem>/metadata.json` — `fps`, `frame_count`, `sample_rate_hz`, `duration_sec`
- **Acceptance:** WAV is mono 16 kHz; metadata fields present and consistent with ffprobe.
- **Status:** Reuse existing implementation.

---

### M2 — Overlapping PANNs windows

- **Goal:** Define fixed-length overlapping analysis windows for PANNs inference.
- **Files:** `bark_detection/panns_windows.py` (new), extend `config.py` + `cli.py`.
- **Defaults:** `window_size_sec = 1.0`, `hop_size_sec = 0.25`.
- **Output:** `panns_windows.csv`

| Column | Description |
|--------|-------------|
| `window_id` | 0-based integer |
| `start_time_sec` | Window start (inclusive), ≥ 0 |
| `end_time_sec` | Window end, ≤ `duration_sec` — **true audio bounds, never padded** |
| `center_time_sec` | `(start_time_sec + end_time_sec) / 2` |

**Final-window behavior (two layers):**

- **CSV (`panns_windows.csv`):** Always store **true** `start_time_sec` / `end_time_sec` clipped to `[0, duration_sec]`. If the last hop produces a segment shorter than `window_size_sec`, `end_time_sec` is `duration_sec` (not padded). Timestamps in the CSV are the ground-truth span in the recording.
- **PANNs inference (M3):** When a window's true span is shorter than the model's required input length, **pad the waveform** (e.g. zero-pad at the end) only inside the inference path. Do not change CSV times to reflect padding. Document pad length and method in M3 results / `docs/decisions.md` if non-obvious.

- **Acceptance:** Windows cover `[0, duration_sec]`; consecutive centers spaced by `hop_size_sec`; no negative times; last row has `end_time_sec == duration_sec` (within float tolerance); CSV never contains padded end times.
- **Inspect:** Row count ≈ `ceil((duration - window) / hop) + 1`; spot-check first and last window: last `end_time_sec` equals audio duration; last window `duration < window_size_sec` if applicable.

#### Prompt 1 — results (2026-05-28, M1 + M2)

- **M1:** `files/dogs1.mp4` → `outputs/dogs1/{audio.wav, metadata.json}` — fps=30, frame_count=150, sample_rate_hz=16000, duration_sec=5.014063.
- **M2:** `panns_windows.csv` — 21 windows (`floor(5.014063/0.25)+1`); first `[0, 1.0]`; last `start=5.0`, `end=5.014063`, duration=0.014063 s (unpadded CSV).
- **CLI:** `--stage all` → `extract` → `windows` only.

---

### M3 — PANNs inference per window

- **Goal:** Run AudioSet tagging on every window; emit interpretable dog/bark/speech/music scores.
- **Files:** `bark_detection/panns_inference.py` (new), extend `config.py` + `cli.py`.
- **Dependency:** `panns-inference` + PyTorch. **Model:** `Cnn14_16k` at **16 kHz** (matches M1 WAV — do not resample to 32 kHz for the default path).
- **Label discovery (mandatory):** **Do not hard-code AudioSet label indices** (e.g. do not assume `Bark` is index 75) without loading the actual label list from the installed PANNs package at runtime.
  1. Load/print the full or searchable label list (or save to `outputs/<stem>/panns_label_list.txt` for inspection).
  2. Match by **exact `display_name` strings** (e.g. `Dog`, `Bark`, `Animal`, `Speech`, `Music`) and any secondary labels justified in the M3 report (e.g. `Yip`, `Bow-wow`).
  3. **Print and save** the resolved mapping: `score_column → label_name → index` in the Prompt 2 report and append the final mapping to **`docs/decisions.md`** (one decision-log entry).
  4. Implement `dog_score`, `bark_score`, `animal_score`, `speech_score`, `music_score` using only those verified indices.
- **Output:** `panns_scores.csv` — one row per `window_id`

| Column | Description |
|--------|-------------|
| `window_id`, `start_time_sec`, `end_time_sec`, `center_time_sec` | From M2 |
| `dog_score`, `bark_score`, `animal_score`, `speech_score`, `music_score` | ∈ [0, 1], no NaN/Inf |
| `top_1_label`, `top_1_score`, `top_2_label`, `top_2_score`, `top_3_label`, `top_3_score` | Argmax triplet over 527 classes |

- **Acceptance:** One row per window; all scores ∈ [0, 1]; no NaN/Inf; inference reproducible on `dogs1`.
- **Inspect:** Top labels on windows overlapping known barks (~0.25 s, ~1.5 s on sample clip).

#### Prompt 2 — results (2026-05-28, M3)

- **Label mapping** (runtime, `panns_label_mapping.txt`): Dog→74, Bark/Yip/Bow-wow→75/76/78 (max), Animal→72, Speech→0, Music→137.
- **Model:** Cnn14_16k @ 16 kHz; STFT 512/160 samples; pad short windows to 16000 samples in inference only.
- **Output:** `panns_scores.csv` — 21 rows; score range [0.0008, 0.9010]; no NaN/Inf.
- **Bark windows (audible ~0.25 s / ~1.5 s):** id=0 center=0.5 bark=0.490; id=4 center=1.5 bark=0.548; id=5 center=1.75 bark=0.605.
- **CLI:** `--stage all` → `extract` → `windows` → `panns`.

---

### M4 — Bark score timeline

- **Goal:** Collapse per-window PANNs scores into a uniform time series for segmentation.
- **Files:** `bark_detection/timeline.py` (new), extend `cli.py`.
- **Logic:** Resample or aggregate window scores onto a time grid (e.g. hop = `hop_size_sec` at each `center_time_sec`, or interpolate). Define:

```
combined_bark_score = f(dog_score, bark_score, animal_score)
```

Default proposal (tune in M4 if needed): `combined_bark_score = max(bark_score, dog_score)` or `0.7 * bark_score + 0.3 * dog_score` — **must be stated explicitly in M4 results** after inspecting `dogs1` plots.

- **Output:** `bark_score_timeline.csv`

| Column | Description |
|--------|-------------|
| `time_sec` | Timeline sample time |
| `dog_score`, `bark_score`, `combined_bark_score`, `speech_score`, `music_score` | Aligned aggregates |

- **Acceptance:** Monotonic time column; combined score peaks align with audible barks on sample clip.
- **Inspect:** Overlay known bark times vs peaks in `combined_bark_score`.

#### Prompt 3 — M4 results (2026-05-28)

- **Formula:** `combined_bark_score = bark_score` (`combined_bark_mode="bark"`).
- **Timeline:** 21 samples at window centers (hop 0.25 s); peak combined **0.605** at t=1.75 s; trough **0.027** at t=1.0 s between barks.
- **Key samples:** t=0.5 combined=0.490; t=1.0 combined=0.027; t=1.5 combined=0.548.

---

### M5 — Detect Barkseq candidates

- **Goal:** Find continuous **Barkseq** regions where `combined_bark_score` exceeds a threshold, merging nearby positives.
- **Files:** `bark_detection/barkseq_detect.py` (new), extend `cli.py`. Reuse merge-gap **pattern** from legacy `events.py` but on timeline scores, not RMS CSV.
- **Defaults:** `barkseq_threshold` (configurable, e.g. 0.15–0.35 after M4 calibration), `merge_gap_sec = 0.5`.
- **Output:** `barkseqs_initial.csv` — columns at minimum: `barkseq_id`, `start_time_sec`, `end_time_sec`, `peak_time_sec`, `duration_sec`, peak/mean score aggregates as available from timeline slice.
- **Acceptance:** No two kept regions separated by less than `merge_gap_sec`; each region has `duration_sec > 0`; count plausible on `dogs1` (expect bursts, not per-frame noise).
- **Inspect:** Compare count and timing vs legacy RMS `bark_events.csv` on same clip (debug only).

#### Prompt 3 — M5 results (2026-05-28)

- **Threshold:** `barkseq_threshold=0.42` (0.35 merges into one region via t=1.25 bridge at bark=0.403).
- **Merge:** `merge_gap_sec=0.5`; region half-width = hop/2 = **0.125 s** per positive center.
- **Output:** 2 Barkseqs — [0.375, 0.625] peak 0.5 s; [1.375, 2.125] peak 1.75 s.
- **Legacy RMS compare:** events [0.19, 0.33] and [1.48, 1.76] — same count, PANNs bounds wider/coarser (center-grid + threshold).

---

### M6 — Noise flagging

- **Goal:** Flag Barkseqs likely contaminated by speech or music; **do not delete aggressively**.
- **Files:** extend `bark_detection/barkseq_detect.py` or `noise.py`, extend `cli.py`.
- **New fields on each Barkseq row:**

| Field | Description |
|-------|-------------|
| `noise_flag` | bool |
| `noise_reason` | e.g. `high_speech`, `high_music`, `likely_noise`, `clean` |
| `max_speech_score`, `max_music_score` | Max over timeline inside Barkseq span |

- **Logic (initial):**
  - High bark/dog + high speech/music → **keep**, mark noisy (e.g. TV + dog).
  - Low bark/dog + high speech/music → `likely_noise`.
  - Thresholds configurable; document in M6 results.
- **Acceptance:** Every row has `noise_flag` and `noise_reason`; no silent drops without logging in report.
- **Inspect:** Flags on `dogs1` match intuition (background speech/music if present).

#### Prompt 4 — M6 results (2026-05-28)

- **Thresholds:** `speech_noise_threshold=0.15`, `music_noise_threshold=0.15`; `strong_bark` gate reuses `barkseq_threshold=0.42`.
- **Output:** 2 Barkseqs flagged — both `clean` (`noise_flag=False`); max in-span speech 0.058 / 0.057; max music 0.035 / 0.041.
- **Row count:** 2 (unchanged from `barkseqs_initial.csv`).

---

### M7 — Final Barkseq export

- **Goal:** Publish canonical `barkseqs.csv` with simple interpretable **confidence**.
- **Files:** extend `bark_detection/barkseq_export.py` or same module, `cli.py`.
- **Output:** `barkseqs.csv`

| Column | Description |
|--------|-------------|
| `barkseq_id` | 0-based |
| `start_time_sec`, `end_time_sec`, `peak_time_sec`, `duration_sec` | Timing |
| `max_dog_score`, `mean_dog_score` | Aggregates over span |
| `max_bark_score`, `mean_bark_score` | Aggregates over span |
| `max_combined_bark_score`, `mean_combined_bark_score` | Aggregates over span |
| `max_speech_score`, `max_music_score` | From M6 |
| `noise_flag`, `noise_reason` | From M6 |
| `confidence` | Simple: e.g. `mean_combined_bark_score * (1 - α * max_speech) * (1 - β * max_music)` with small α, β — **exact formula in M7 results** |
| `method` | e.g. `panns_cnn14_16k_v1` |

- **Acceptance:** `confidence ∈ [0, 1]`; schema stable; row count equals `barkseqs_initial` after any explicit drops (drops must be reported).
- **Inspect:** 2–3 sample rows on `dogs1` with audible validation.

#### Prompt 4 — M7 results (2026-05-28)

- **Formula:** `confidence = clip(mean_combined_bark_score × (1 − 0.3 × max_speech) × (1 − 0.3 × max_music), 0, 1)`.
- **Config:** `speech_penalty=0.3`, `music_penalty=0.3`; `method=panns_cnn14_16k_v1`.
- **Output:** `barkseqs.csv` — 2 rows; confidence **0.477** (id=0, peak 0.5 s) and **0.542** (id=1, peak 1.75 s).
- **Row count:** equals `barkseqs_initial.csv` (no drops).

---

#### M1 — results (legacy, 2026-05-27)

Reused: `audio.wav`, `metadata.json` on `outputs/dogs1/`.

---

### M8 — Extract Barkseq audio clips

- **Goal:** One WAV file per Barkseq for listening, classifier training, and stage-3 features.
- **Files:** `bark_detection/clips.py` (new), reuse `rms.load_wav_mono` or scipy reader, `cli.py`.
- **Output directory:** `outputs/<stem>/bark_event_clips/`
  - `barkseq_000.wav`, `barkseq_001.wav`, …
- **Clip bounds:** padded by `clip_pre_context_sec` / `clip_post_context_sec` (default **0.25 s** each), clamped to audio duration and neighboring Barkseq boundaries when `prevent_clip_overlap=true`. Event `start_time_sec` / `end_time_sec` unchanged.
- **CSV columns added:** `clip_start_time_sec`, `clip_end_time_sec`, `clip_duration_sec`, `clip_pre_context_actual_sec`, `clip_post_context_actual_sec`, `clip_path`.
- **Acceptance:** File count == row count in `barkseqs.csv`; clips include pre/post context where possible; actual padding recorded when clamped.
- **Inspect:** Listen to each clip on `dogs1`.

#### Prompt 5 — M8 results (2026-05-28, revised padded clips)

- **Padding config:** `clip_pre_context_sec=0.25`, `clip_post_context_sec=0.25`, `prevent_clip_overlap=true`.
- **Event spans unchanged:** id=0 [0.375, 0.625], id=1 [1.375, 2.125].
- **Clip spans:** id=0 [0.125, 0.875] (0.75 s); id=1 [1.125, 2.375] (1.25 s). Full 0.25 s padding on all sides — no boundary or neighbor clipping on `dogs1`.
- **File count:** 2.

---

### M9 — Align Barkseqs to video frames

- **Goal:** Add frame indices for downstream vision association.
- **Files:** `bark_detection/frames.py` (new), `cli.py`.
- **Formula:** `frame_id = round(timestamp_sec * fps)`, clamp to `[0, frame_count - 1]`.
- **Defaults:** `pre_context_sec = 0.5`, `post_context_sec = 0.5`.
- **Columns added to `barkseqs.csv`:**

| Column | Description |
|--------|-------------|
| `start_frame`, `end_frame`, `peak_frame` | Event frames |
| `context_start_frame`, `context_end_frame` | ± context around event |

- **Acceptance:** Spot-check: open video at `peak_frame` and hear/see bark; context contains event.
- **Inspect:** 2–3 rows on `dogs1` at 30 fps.

#### Prompt 5 — M9 results (2026-05-28)

- **Formula:** `frame_id = round(time_sec × fps)`, clamp `[0, frame_count − 1]`; context ± **0.5 s** (`pre_context_sec`, `post_context_sec`).
- **dogs1 (fps=30, frame_count=150):**
  - id=0: peak_frame **15** (0.5 s); start/end **11–19**; context **0–34**
  - id=1: peak_frame **52** (1.75 s); start/end **41–64**; context **26–79**
- **Output:** frame columns appended to `barkseqs.csv`.

---

### M10 — Debug visualizations

- **Goal:** Publication-style figures under `debug/`.
- **Files:** extend `bark_detection/viz.py`, `cli.py`.
- **Outputs:**
  - `debug/panns_score_timeline.png` — time on x-axis; `combined_bark_score`, `dog_score`, `bark_score`, `speech_score`, `music_score`; threshold line.
  - `debug/barkseq_overlay.png` — same scores + **shaded final Barkseq regions** + peak markers.
- **Acceptance:** PNGs open; legend; shaded regions match `barkseqs.csv`.
- **Inspect:** Visual story clear at a glance on `dogs1`.

#### Prompt 6 — M10 results (2026-05-28)

- **Outputs:** `debug/panns_score_timeline.png`, `debug/barkseq_overlay.png`.
- **Plotted series:** `combined_bark_score`, `bark_score`, `dog_score`, `speech_score`, `music_score`; threshold line at **0.42**.
- **Overlay:** 2 shaded regions [0.375, 0.625] and [1.375, 2.125]; peak markers at 0.5 s and 1.75 s.

---

### M11 — Unit tests

- **Goal:** Lock behavior before stage-3 integration.
- **Files:** `tests/test_audio_io.py`, `test_panns_windows.py`, `test_panns_scores.py`, `test_timeline.py`, `test_barkseq_detect.py`, `test_noise.py`, `test_frames.py`, `test_clips.py`.
- **Coverage (minimum):**
  - Audio extraction → mono 16 kHz WAV + metadata fields
  - PANNs window generation → correct start/end/center
  - `panns_scores.csv` → one row per window; scores ∈ [0, 1]; no NaN/Inf
  - Barkseq detection on **synthetic** score timelines
  - `merge_gap_sec` merges close positives
  - Noise flagging on synthetic speech/music patterns
  - Frame alignment math + clamping
  - Clip extraction → one file per Barkseq
- **Acceptance:** `pytest tests/` green.
- **Agents:** test-automator to author; optional code-reviewer after green.

#### Prompt 7 — M11 results (2026-05-28)

- **Tests:** 20 passed, 1 skipped (integration smoke) in **0.44 s**.
- **Coverage:** windows, timeline, barkseq detect, noise, frames, clips, viz, schema — all synthetic; no real PANNs inference.
- **Command:** `pytest tests/`

---

## 6. File structure (target)

**Do not create a second top-level package** (e.g. `panns_bark/`). This repo is small (~36 files); one package `bark_detection/` + one entry script stays simplest.

```
bark_detection/
  __init__.py
  config.py           # BarkConfig — PANNs fields; legacy RMS fields optional
  audio_io.py         # M1 (shared)
  panns_windows.py    # M2
  panns_inference.py  # M3
  timeline.py         # M4
  barkseq_detect.py   # M5–M6
  barkseq_export.py   # M7 (may merge with barkseq_detect)
  clips.py            # M8
  frames.py           # M9
  viz.py              # M10 (PANNs plots; RMS plot helpers may move to legacy/)
  cli.py              # default stages: extract → … → align → viz (PANNs only)
  legacy/             # RMS-first modules — unwired from default CLI (see §6.1)
    rms.py
    threshold.py
    candidates.py
    events.py
    scoring.py

docs/
  bark_detection_plan.md   # stays here — repo source of truth (do not move)
  decisions.md             # keep; append PANNs entries
  research_workflow.md       # keep; update cross-refs when needed
  prompts/
    panns_p1_*.md … panns_p7_*.md   # new grouped prompts
  legacy/                  # optional archive for stale RMS handoff/prompts (see §6.1)

tests/                # M11
run_bark_detection.py
outputs/<video_stem>/ # gitignored
```

### 6.1 Repository layout — legacy vs active (no big-bang move yet)

| Path | Role | When to populate |
|------|------|------------------|
| **`bark_detection/`** (root of package) | Active PANNs pipeline + shared `audio_io`, `config`, `cli` | Prompt groups 1–6 |
| **`bark_detection/legacy/`** | RMS detector modules only; optional manual `--stage` for baseline | Create folder early; **move** `rms.py` … `scoring.py` here when Prompt 1 rewires CLI (not before) |
| **`docs/bark_detection_plan.md`** | Living plan — **keep in `docs/`** | Already PANNs-first; do not relocate |
| **`docs/legacy/`** | Archive `handoff.md`, `prompts/m2–m7_*.md` after PANNs handoff exists | Optional; copy/move when convenient, not blocking |
| **`outputs/<stem>/`** | Shared `audio.wav`, `metadata.json` + PANNs artifacts | Reuse M1 outputs on `dogs1` |
| **`outputs/<stem>/legacy_rms/`** | Legacy RMS CSVs/plots from manual `--stage rms` … `score` | Written automatically when legacy stages run |

**What not to do**

- Do not delete legacy code until PANNs M11 tests pass (unless you explicitly ask).
- Do not move `bark_detection_plan.md` into the package — breaks links from `research_workflow.md` / `handoff.md`.
- Do not duplicate `audio_io` into a new package.

**Legacy folders (done):** RMS modules live in **`bark_detection/legacy/`**; stale handoff and RMS prompts live in **`docs/legacy/`**. Default CLI runs PANNs Prompt 1 (`extract` → `windows`); legacy RMS stages remain via `--stage rms` etc.

### Per-video artifact tree (final)

```
outputs/<video_name>/
  audio.wav
  metadata.json
  barkseqs.csv
  bark_event_clips/
    barkseq_000.wav
    barkseq_001.wav
    ...
  debug/
    panns_score_timeline.png
    barkseq_overlay.png
  intermediate/
    panns_windows.csv
    panns_scores.csv
    bark_score_timeline.csv
    barkseqs_initial.csv
    panns_label_list.txt
    panns_label_mapping.txt
  legacy_rms/                  # legacy manual --stage rms … score only
    rms_values.csv
    rms_debug_plot.png
    bark_candidates.csv
    bark_events_initial.csv
    bark_events.csv
    metadata.json              # M1 fields + thresholding block
```

**Legacy artifacts (manual `--stage rms` … `score`):** live under `legacy_rms/` — not at the run root.

### CLI

Default pipeline:

```bash
python run_bark_detection.py --video files/dogs1.mp4 --output_dir outputs/
```

Stages (target order): `extract` → `windows` → `panns` → `timeline` → `barkseqs` → `noise` → `export` → `clips` → `frames` → `viz`

Support `--stage <name>` and `--from-stage <name>` for iteration. Legacy RMS stages are **not** in default `_STAGE_ORDER`.

---

## 7. Configuration (`BarkConfig`)

Fields grow per milestone. Target defaults:

| Group | Field | Default | Milestone |
|-------|--------|--------:|-----------|
| audio | `target_sample_rate_hz` | 16000 | M1 |
| audio | `target_channels` | 1 | M1 |
| panns | `window_size_sec` | 1.0 | M2 |
| panns | `hop_size_sec` | 0.25 | M2 |
| panns | `model_name` | `"Cnn14_16k"` | M3 |
| panns | `device` | `"cpu"` | M3 |
| timeline | `combined_bark_formula` | TBD in M4 | M4 |
| barkseq | `barkseq_threshold` | TBD in M5 | M5 |
| barkseq | `merge_gap_sec` | 0.5 | M5 |
| noise | `speech_noise_threshold` | 0.15 | M6 |
| noise | `music_noise_threshold` | 0.15 | M6 |
| confidence | `speech_penalty` | 0.3 | M7 |
| confidence | `music_penalty` | 0.3 | M7 |
| frames | `pre_context_sec` | 0.5 | M9 |
| frames | `post_context_sec` | 0.5 | M9 |

Legacy RMS fields may remain in `config.py` for optional baseline runs but are unused by the default stage list.

---

## 8. Library strategy

### Default pipeline (M2–M10)

| Library | Use |
|---------|-----|
| `subprocess` + ffmpeg/ffprobe | M1 extract + metadata |
| `scipy.io.wavfile` | Read/write 16 kHz mono WAV |
| `numpy`, `pandas` | Arrays, CSVs |
| `matplotlib` | M10 debug plots |
| `torch` + `panns-inference` | M3 AudioSet tagging (`Cnn14_16k`) |

**PANNs notes**

- AudioSet labels are **multi-label sigmoid** scores in [0, 1], not softmax.
- Primary bark label: **`Bark`** — index resolved at M3 from the installed label list (do not hard-code).
- `Dog` (74) and `Animal` (72) are broader; prefer `Bark` for event detection.
- No built-in sliding window in `panns-inference` — M2 provides windows; M3 runs clip-wise inference per row.

### Legacy / optional

- Framed RMS path: `numpy` + `scipy` only (existing modules).
- `audio_utils.py` / librosa: separate notebook workflow; not required for PANNs CLI.

### Future (stage 3 / paper)

- YAMNet baseline for ablation (compare to PANNs Barkseqs).
- CAV-MAE or custom heads on Barkseq clips.
- Eval harness (IoU on annotated Barkseqs) — not in M11; add when annotations exist.

---

## 9. Working agreement

- **Grouped implementation:** Seven prompt groups (§4 table). **Stop after each group** — not after every milestone. Deliver the six-part review summary listed in §4 before requesting approval for the next group.
- **No commits** unless the user explicitly asks.
- **Do not invent** artifact filenames or CSV columns — use this plan; if something is missing, ask.
- **Do not pre-declare** all `BarkConfig` fields upfront — add fields when the prompt group that needs them lands.
- **Flat** `outputs/<stem>/` — no per-run subdirs until paper/repro PR.
- **Delegate each prompt group** to a `fullstack-developer` subagent using the saved prompt file for that group.
- **M3 (Prompt 2):** Inspect the real PANNs label list; print/save matched labels; write final index mapping to `docs/decisions.md` — never hard-code indices from memory or docs.

---

## 10. Comparison baseline (dogs1)

Legacy RMS pipeline on `files/dogs1.mp4` produced **2 events** at ~0.19–0.33 s and ~1.48–1.76 s (user-audible: one bark ~0.25 s, burst ~1.5 s). PANNs Barkseqs should be validated against the same audible ground truth, not necessarily identical boundaries (Barkseq vs single-bark merge semantics differ).

---

## 11. Next step

**This document is the PANNs-first plan.** Prompt 7 (M11) complete. PANNs pipeline M1–M11 implemented on `dogs1`.

Grouped prompt files (`docs/prompts/panns_p1_*.md` … `panns_p7_*.md`) need to be drafted before each group runs. Old RMS prompts (`m3`–`m7`) are **stale**.
