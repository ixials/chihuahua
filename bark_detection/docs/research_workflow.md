# Research Workflow — Bark Event Detection

> Living document. Companion to [`bark_detection_plan.md`](./bark_detection_plan.md). Edit freely.
>
> **Implementation plan** (`bark_detection_plan.md`) = source of truth for *code*: what to build, in what order, with what artifacts.
> **This doc** (`research_workflow.md`) = source of truth for *paper readiness*: what to **write down**, **measure**, **annotate**, and **figure out** at each milestone so that when you sit down to write the 4–6 page workshop paper, every number, decision, and figure is already on disk.

---

## §1 — Purpose & relation to the implementation plan

When this work ends, you should be able to write the paper in **a few sittings, not weeks**, because:

- Every methodological decision was logged when it was made (not reconstructed from memory months later).
- Every reported number came from a re-runnable script over re-runnable data.
- Every figure has a known source file and a known producing milestone.
- The ground truth, evaluation protocol, baseline, and ablations were defined *before* any results were collected — so the paper isn't accidentally cherry-picking.

This doc cross-links to specific milestones in the implementation plan; the implementation plan stays implementation-focused. Both are living documents and can be edited as you learn.

**Scope reminder** (matches `bark_detection_plan.md` §3): the *paper* is also scoped to **bark event detection in audio only**. "Which dog barked" is future work.

**Paper target**: workshop / short paper, 4–6 pages. That sets the rigor bar:
- 1 sample dataset (the `files/dogs1.mp4` clip + ideally 2–3 more short clips).
- 1 external baseline (YAMNet).
- 2 ablations.
- 1 annotator (you), documented as a limitation.

---

## §2 — Reproducibility infrastructure

The single most useful research habit: **any number in your paper must be regenerable from a recorded config + a recorded git SHA**. Cheap to set up, expensive to retrofit.

### §2.1 Run directory convention

> **Deferred until M10 / paper time.** M1–M9 use the flat layout `outputs/<video_stem>/`. The per-run subdirectory convention below kicks in when paper work begins.

Replace `outputs/<video_stem>/` from `bark_detection_plan.md` with a per-run subdirectory:

```
outputs/<video_stem>/<run_id>/
    audio.wav                 # M1
    metadata.json             # M1
    run_manifest.json         # new — see §2.2
    rms_values.csv            # M2
    bark_candidates.csv       # M4
    bark_events_initial.csv   # M5
    bark_events.csv           # M6 / M8
    bark_episodes.csv         # M7
    rms_debug_plot.png        # M9
    eval.json                 # new — see §4
```

where `run_id = YYYYMMDD-HHMMSS-<git_short_sha>` (e.g. `20260527-141530-ffb398f`). Two reasons:

- You don't overwrite yesterday's result when you re-run with a different param. The paper's Table 2 needs *several* runs side-by-side.
- The `run_id` is your citation key when emailing a collaborator or annotating a figure.

Keep `outputs/` gitignored; the `metadata.json` + `run_manifest.json` are small and *do* get archived (zipped) per published run.

### §2.2 `run_manifest.json` (one per run)

> **Deferred until M10.** Not produced by M1–M9. Re-add as a small PR when the paper work starts.

Written by `cli.py` at the start of every run, before any stage executes.

```json
{
  "run_id": "20260527-141530-ffb398f",
  "git": {
    "sha": "ffb398f9b8a...",
    "branch": "main",
    "dirty": false
  },
  "command": "python run_bark_detection.py --video files/dogs1.mp4 --output_dir outputs/",
  "config": { "...full BarkConfig dataclass as dict..." },
  "python": "3.11.4",
  "platform": "darwin-23.4.0",
  "host": "aryahb-mbp",
  "started_at": "2026-05-27T14:15:30Z",
  "finished_at": "2026-05-27T14:15:38Z",
  "wall_seconds": 8.2,
  "packages": { "numpy": "1.26.4", "scipy": "1.11.4", "...": "..." },
  "ffmpeg_version": "ffmpeg version 6.0",
  "pipeline_version": "0.1.0"
}
```

Notes:
- `git.dirty: true` is a yellow flag for any published number.
- `packages` is `{name: version}` for the packages in `requirements.txt` only — no need to dump the whole `pip freeze`.
- `config` is the **same** snapshot of `BarkConfig` that the stage code consumes — not a separate hand-written record.

### §2.3 Random seeds

M0–M10 has no training; no seeds needed in the initial heuristic pipeline. **But**:

- When you sample a subset of clips for annotation (§3), log the seed.
- When you later swap in a learned classifier (M6 hook), the classifier's adapter is responsible for logging its own seed inside `run_manifest.json` under `config.classifier.seed`.

### §2.4 Frozen requirements

`requirements.txt` is already pinned. Keep it that way. When you install a new package, **pin it in the same PR that adds the code that imports it** — never `pip install foo` in isolation.

---

## §3 — Ground-truth annotation protocol

You need labels before you can measure anything. Workshop scope: small, careful, fully documented.

### §3.1 Tool

**Audacity** label tracks. Free, available everywhere, exports a 3-column TSV: `start\tend\tlabel`. Simplest path.

Alternative tools considered: Praat TextGrid (phonetic-style, overkill), Label Studio (web UI, heavier setup). Audacity wins on friction.

### §3.2 Label schema

```
bark              # canonical, confident bark
yip               # high-pitched short bark variant
growl             # voiced but not a bark
bark_uncertain    # you'd flag this if asked "is it a bark?"
non_bark_noise    # foreground non-vocal sound you want the system to *not* fire on (door slam, etc.)
```

At evaluation time (§4), this collapses to **binary**: `{bark, yip, bark_uncertain}` → `bark` (positive); everything else → `non_bark` (negative). The finer-grained labels stay on disk so a reviewer asking "what about yips?" can be answered.

### §3.3 Storage

`annotations/<video_stem>.csv` (committed to git — small):

| column | type | notes |
|---|---|---|
| `start_time_sec` | float | seconds from start of audio |
| `end_time_sec` | float | seconds from start of audio |
| `label` | str | one of the five above |
| `annotator` | str | your initials |
| `notes` | str | optional; edge-case reasoning |

A small script `bark_detection/annotations.py` will convert Audacity's `.txt` export to this CSV. (Add this script as part of M10's tests milestone, or whenever you first annotate.)

### §3.4 Coverage

- **Minimum**: fully annotate `files/dogs1.mp4` (5 s).
- **Target**: 2–3 more short clips (10–30 s each) — record on a phone if needed. Even noisy phone audio is fine; document provenance.
- Total annotated audio for a workshop paper: **~60–90 s** is enough. Don't over-collect.

### §3.5 Annotation guide

Create `docs/annotation_guide.md` the first time you annotate. **One page**, structured as:

1. **What counts as a bark** — single-sentence definition.
2. **Boundaries** — start = onset of the voiced burst; end = energy returns to baseline.
3. **Edge cases** with 2–3 worked examples ("two barks 100 ms apart → two labels, not one", "tail of a bark with audible echo → end at primary energy decay, ignore echo", etc.).
4. **What to do when uncertain** — label `bark_uncertain` and leave a note.

Write this *before* you annotate. Update as you discover new edge cases — every update is a `docs/decisions.md` entry.

### §3.6 Sole-annotator caveat

Mention explicitly in the paper's Limitations section: one annotator, no inter-annotator agreement measured. This is acceptable at workshop scope but must be disclosed.

---

## §4 — Evaluation methodology

This section *is* the paper's Experimental Setup section. Lock it in **before** you look at any numbers, so you cannot accidentally tune to the test set.

### §4.1 Event-level matching

A predicted event `P = [p_start, p_end]` matches a ground-truth event `G = [g_start, g_end]` if:

```
IoU(P, G) = |P ∩ G| / |P ∪ G|   ≥   τ_iou       (default τ_iou = 0.5)
```

Greedy one-to-one assignment:
1. Compute IoU for every (P, G) pair.
2. Sort pairs by IoU desc.
3. Walk the list, accept a pair if neither side is already matched; stop when IoU < τ_iou.

Then:
- TP = matched pairs.
- FP = predicted events with no match.
- FN = ground-truth events with no match.

### §4.2 Primary metrics

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2·P·R / (P + R)

Report at:
1. The **operating point** (the configured threshold method, default `mean + 2σ`).
2. A **PR curve** (sweep threshold from `mean + 0σ` to `mean + 5σ` in 0.25σ steps, or equivalent percentile sweep).

### §4.3 Secondary metrics

- **Episode-level P/R/F1** (same matching rule but on M7's `bark_episodes.csv`).
- **Count error**: per clip, `|predicted_event_count − true_event_count|`. Easy to interpret.

### §4.4 Reporting format

`outputs/<video_stem>/<run_id>/eval.json`:

```json
{
  "tau_iou": 0.5,
  "events": {"tp": 7, "fp": 2, "fn": 1, "precision": 0.78, "recall": 0.88, "f1": 0.82},
  "episodes": {"tp": 3, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
  "count_error": 1
}
```

Aggregate across clips into `eval_summary.csv` (one row per clip + a final micro-averaged row).

### §4.5 Where it lives in code

Add `bark_detection/eval.py` as part of **M10** (test/review milestone). It's small: matching + metric computation + JSON/CSV output. No new milestone needed.

### §4.6 What *not* to do

- **Do not** tune any `BarkConfig` parameter *after* you've looked at eval numbers on a clip. If you do, that clip must move to a held-out set you don't re-eval on. Workshop scope: just commit the params, run once, report. Document any post-hoc tuning as a `docs/failures.md` entry.

---

## §5 — Baselines

Workshop scope: **one** external baseline + your heuristic pipeline.

### §5.1 YAMNet (the baseline)

- TF-Hub model `https://tfhub.dev/google/yamnet/1`. AudioSet ontology, class id 70 = `Bark`.
- Runs at 16 kHz mono — matches your standardized audio from M1.
- Outputs a 521-d class probability vector every ~0.48 s. The "Bark" channel is your detection score.

**Wrapping into the pipeline**:

```python
# bark_detection/baselines/yamnet.py
class YamnetBarkClassifier:
    def score(self, wav: np.ndarray, sr: int, t_start: float, t_end: float) -> float:
        # extract wav[int(t_start*sr):int(t_end*sr)], pad to >=0.96s,
        # run YAMNet, return max over the "Bark" channel
        ...
```

Satisfies the `BarkClassifier` Protocol from `bark_detection_plan.md` §9. No pipeline changes needed.

### §5.2 Two reporting modes for YAMNet

To make the comparison fair:

- **Mode A (event-level overlay)**: take the *heuristic*'s detected events, but replace `bark_confidence` with YAMNet's score for that event window. Tests whether YAMNet *re-ranks* better than the heuristic.
- **Mode B (standalone)**: run YAMNet over the whole clip with a fixed window stride, threshold the Bark channel, merge contiguous detections, score against ground truth. Tests YAMNet as an *independent* detector.

Both go in Table 1.

### §5.3 What's deferred

PANNs, CAV-MAE, custom-trained heads — all "future work" for the workshop paper. Mention each in one sentence in §8 (Future Work).

---

## §6 — Ablations

Workshop scope: **two** ablations, each one row in Table 2.

### §6.1 Ablation A — Threshold method

| Threshold method | Precision | Recall | F1 |
|---|---|---|---|
| `mean + 2σ` (default) | … | … | … |
| `90th percentile` | … | … | … |

`BarkConfig.threshold_method` already supports both. Just two runs, two `run_id`s.

### §6.2 Ablation B — Smoothing window

| Smoothing window | Precision | Recall | F1 |
|---|---|---|---|
| 0 ms (no smoothing) | … | … | … |
| 50 ms (default) | … | … | … |
| 100 ms | … | … | … |

Three runs.

### §6.3 What *not* to ablate (at workshop scope)

`merge_gap_ms`, `min_duration_ms`, `max_duration_ms`, `episode_gap_sec`, scoring weights `w_rms/w_duration` — log the chosen values in `docs/decisions.md` but don't sweep them in the paper. Save them for a future longer paper.

---

## §7 — Decision log & failure log

Two append-only Markdown files. The single highest leverage research habit, after reproducibility. Five-line entries, not essays.

### §7.1 `docs/decisions.md`

One entry per non-trivial decision. Template:

```markdown
## 2026-05-27 — Threshold method: `mean + 2σ` over `90th percentile`
**Context:** Need an adaptive threshold for RMS that works on quiet and noisy clips.
**Options:** (a) `mean + k·σ`; (b) percentile of smoothed RMS; (c) Otsu.
**Decision:** `mean + 2σ`, with percentile logged side-by-side for the ablation.
**Why:** Robust to long quiet stretches that fool percentile thresholds; one tunable k; matches RMS-gating idea from the *"What's That Sound Right Now?"* paper.
**Revisit if:** Ablation A shows percentile is materially better.
```

Append once per real decision. **Don't** log "renamed a variable"; **do** log "picked Audacity over Praat", "set `merge_gap_ms = 200`", "chose IoU ≥ 0.5 as the matching threshold".

### §7.2 `docs/failures.md`

One entry per thing that didn't work. Template:

```markdown
## 2026-05-29 — Median smoothing fragmented single barks
**What I tried:** `smoothing_method = "median"` with `smoothing_window_ms = 50`.
**What happened:** Each bark became 2–3 events; F1 dropped from 0.82 → 0.64 on `dogs1.mp4`.
**Why I think it happened:** Median over 50 ms is shorter than the typical bark plateau, so the interior dips below threshold.
**What I'm doing instead:** Stay on moving average; if I revisit median, use ≥150 ms.
```

These become the **Discussion & Limitations** section of the paper. Future-you will not remember any of this without these notes.

---

## §8 — Figures & tables registry

Workshop paper ≈ 4–5 figures + 1–2 tables. Each has a known owner and known source file:

| ID | Type | Caption stub | Produced by | Source artifact |
|---|---|---|---|---|
| F1 | Figure | Pipeline overview (block diagram) | manual (Mermaid → PNG) | `docs/figures/pipeline.png` |
| F2 | Figure | Waveform + RMS + smoothed RMS + threshold + detected events overlay | M9 | `outputs/<stem>/<run_id>/rms_debug_plot.png` |
| F3 | Figure | Ground-truth vs predicted events (timeline strips, one per clip) | M10 / `eval.py` | `outputs/<stem>/<run_id>/eval_timeline.png` |
| F4 | Figure | Precision–Recall curve over threshold sweep | M10 / `eval.py` | `outputs/<stem>/<run_id>/pr_curve.png` |
| T1 | Table | Per-clip & micro-averaged P/R/F1: heuristic vs YAMNet (Mode A & B) | M10 / `eval.py` | `eval_summary.csv` |
| T2 | Table | Ablations: threshold method × smoothing window | §6 + `eval.py` | `ablation_summary.csv` |

When a figure or table is first produced, **copy** the source artifact into `docs/figures/` with a stable name (`F2_rms_overlay.png`, etc.). Paper compiles against `docs/figures/` — never against a per-run path that might be deleted.

---

## §9 — Per-milestone paper artifact map

For each milestone in `bark_detection_plan.md`, this is what the *paper* gets out of it. Use this column as the extra item in the §10 stop/go checklist.

| Milestone | Paper artifact to produce or log |
|---|---|
| **M0** Inspect + plan | Decision entry: scope (event detection only). Related-work notes citing the two anchor papers (file: `docs/related_work.md`, just a scratch list to start). |
| **M1** Audio extraction | `run_manifest.json` schema goes live. `metadata.json` is the source for the paper's *Dataset* paragraph (sample rate, duration, fps, frame count). |
| **M2** RMS curve | First version of F2 (raw RMS only). Decision entry: window/hop choice (50/10 ms) and *why* (bark plateau ≈ 100–300 ms; need ≥ 5 frames per event). |
| **M3** Smoothing + threshold | F2 updated with smoothed curve + threshold line. Decision entries: smoothing method + threshold method. Log **both** `mean+2σ` and `90th percentile` values in `metadata.json` — this is Ablation A's data. |
| **M4** Candidates | Number to report: candidate count vs your manual count on the sample clip. Goes in the paper as the recall sanity-check sentence. |
| **M5** Merge + duration | Numbers: raw candidates → after-merge → after-duration counts. Decision entries: `merge_gap_ms`, `min_duration_ms`, `max_duration_ms` and the bark-shape reasoning. |
| **M6** Scoring + classifier hook | The scoring formula is **paper Equation (1)**. The `BarkClassifier` Protocol is the *Plug-in classifier* paragraph in Method. Histogram of `bark_confidence` over candidates → optional figure. |
| **M7** Episodes | One paragraph in Method: episode definition + the `episode_gap_sec = 0.75` rationale. Number: episodes per clip. |
| **M8** Frame alignment | One paragraph in Method: the `frame_id = round(t · fps)` formula + context-window definition. |
| **M9** Debug viz | F2 final committed to `docs/figures/F2_rms_overlay.png`. |
| **M10** Tests + eval + review | Everything in §4 fires: `eval.py` produces T1, F3, F4. §6 ablations produce T2. Code-reviewer notes seed `docs/failures.md` and the paper's Limitations section. |

Cross-reference: the column "What you should inspect" in `bark_detection_plan.md` is your *gate*; this column is your *paper export*. Both happen at the same checkpoint.

---

## §10 — Paper outline mapping

The workshop paper, section by section, with the artifacts that fill each subsection. Numbers are page targets, not hard limits.

### Abstract (1 paragraph)
- One-line problem, one-line approach, one headline F1 number (from T1).

### 1. Introduction (~0.75 pg)
- Motivation: audio-visual dog localization.
- Problem statement: bark *event detection* in the audio modality.
- Anchor papers: *What's That Sound Right Now?* (RMS gating), *Active Speakers in Context* (per-track classification).
- Scope sentence: "We address only the event-detection step; assignment to a specific dog is future work."
- Contributions (workshop-appropriate, 2 bullets is fine):
  1. A simple, transparent RMS-based pipeline for bark event detection with explicit hooks for learned classifiers.
  2. A baseline comparison against YAMNet and two ablations on a small annotated set.

### 2. Related Work (~0.75 pg)
- The two anchor papers (motivation).
- Pretrained audio classifiers: YAMNet, PANNs, CAV-MAE (one sentence each; YAMNet is used as a baseline, the others are future work).

### 3. Method (~1.5 pg)
- **3.1 Pipeline overview** — F1 (block diagram).
- **3.2 Audio standardization** — M1 paragraph.
- **3.3 RMS energy + smoothing + adaptive threshold** — M2/M3 paragraphs + F2 (referenced here, shown in Results).
- **3.4 Candidate regions and event refinement** — M4/M5 paragraphs.
- **3.5 Confidence scoring with a plug-in classifier** — M6 paragraph, Equation (1), the `BarkClassifier` Protocol.
- **3.6 Episode grouping and frame alignment** — M7/M8 paragraphs.

### 4. Experimental Setup (~0.5 pg)
- **Dataset** — §3 + per-clip provenance.
- **Annotation** — §3.5 + sole-annotator caveat.
- **Metrics** — §4.1–§4.3.
- **Baseline** — §5.
- **Implementation** — pinned versions from `requirements.txt`; commit SHA.

### 5. Results (~1 pg)
- **T1**: per-clip + micro-averaged P/R/F1, heuristic vs YAMNet (Mode A & B).
- **F2**: pipeline trace on a representative clip.
- **F3**: ground-truth vs predicted event timelines.
- **F4**: PR curve over threshold sweep.

### 6. Ablations (~0.5 pg)
- **T2**: threshold method × smoothing window.
- 2–3 sentences interpreting each.

### 7. Discussion & Limitations (~0.5 pg)
- Sole-annotator caveat (§3.6).
- Pulled directly from `docs/failures.md` entries.
- Where the heuristic fails (e.g., near non-bark vocalizations); where YAMNet fails (e.g., chronic over-firing on the `Animal` superclass).

### 8. Conclusion & Future Work (~0.25 pg)
- Recap.
- Future: PANNs/CAV-MAE classifier head, dog-track assignment, multi-mic source localization.

---

## §11 — Per-milestone workflow checklist

Run this at the end of **every** milestone, on top of `bark_detection_plan.md` §10. Designed to take < 2 minutes.

1. **Code artifacts** produced per the implementation plan. ✅
2. **`run_manifest.json`** exists for any run whose number you might cite. ✅
3. **Paper artifact** for this milestone produced (per §9 table above). ✅
4. **`docs/decisions.md` entry** added if a non-trivial choice was made this milestone. ✅
5. **`docs/failures.md` entry** added if anything didn't work this milestone. ✅
6. **Cross-link noted** — which paper section (§10) does this milestone now feed? Add the milestone ID into the paper outline if helpful.

Only after all six are ✅ do you signal "ready to go to milestone N+1" per `bark_detection_plan.md` §10.

---

## Open items (this doc)

- Confirm Audacity as the annotation tool (vs Praat / Label Studio).
- Confirm YAMNet as the single workshop-paper baseline.
- Confirm the two ablations chosen (threshold method, smoothing window).
- Add 2–3 more short clips to the dataset before M10 if possible.
- Decide a paper venue + deadline so the M10 schedule has a real target.
