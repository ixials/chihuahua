# Session handoff — bark detection module (M0–M5 executed)

> One-page brief for a fresh chat picking up at M6. Read this top to bottom (≈ 2 min), then paste `docs/prompts/m6_prompt.md` into the new chat to start M6.

This doc supersedes the previous handoff (which was about *drafting prompts*; that work is done — all per-milestone prompts now live in `docs/prompts/m2_prompt.md` … `m6_prompt.md`).

---

## 1. Project goal

Audio-visual dog bark localization pipeline. Three eventual stages:

1. Detect and track every visible dog.
2. **Detect bark events in the audio.** ← *this module*
3. Assign each bark event to the correct tracked dog (or mark off-screen / unknown).

**Right now we are only building stage 2** — a modular, research-grade audio bark event detector. Stage 1 (visual) lives in `dog-bounding-box.ipynb` and is unrelated to this work. Stage 3 (assignment) is future work.

## 2. Anchors

- **Implementation plan** — [`docs/bark_detection_plan.md`](./bark_detection_plan.md). Source of truth for *code*: milestones M0–M10, what each ships, acceptance criteria, the §8 `BarkConfig` parameter table.
- **Research workflow** — [`docs/research_workflow.md`](./research_workflow.md). Source of truth for *paper readiness*: reproducibility, annotation, eval, baselines, ablations, decision/failure logs, paper outline.
- **Decision log** — [`docs/decisions.md`](./decisions.md). Append-only, 4 entries today.
- **Per-milestone prompts** — `docs/prompts/m<N>_prompt.md` (M2 through M6 ready; M7–M10 still TBD).

**Two motivating papers** (plan §2):
- *What's That Sound Right Now? Video-centric Audio-Visual Localization* — RMS-energy gating to find sound events. That's why M2–M5 are RMS-first, no learned classifier.
- *Active Speakers in Context* — per-track audio features + visual candidates → active/silent classification. Forward analogue for stage 3.

## 3. Where we are right now

**M0–M5 complete and verified.** Pipeline runs end-to-end on `files/dogs1.mp4`:

```bash
python run_bark_detection.py --video files/dogs1.mp4 --output_dir outputs/
```

Final result on the 5 s sample clip: **2 bark events**, both `event_type=bark`:

| event_id | start | peak | end | dur | rms_peak | norm | event_type | merged_from |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0.190 | 0.245 | 0.330 | 0.140 | 0.284 | 1.000 | bark | [0] |
| 1 | 1.480 | 1.705 | 1.760 | 0.280 | 0.215 | 0.757 | bark | [1, 2] |

User audibly confirmed: one bark at ≈ 0.25 s, a burst at ≈ 1.5 s. Pipeline match is exact.

## 4. Files in flight (uncommitted M5 layer)

Per `git status --short` at end of session:

```
 M bark_detection/cli.py
 M bark_detection/config.py
 M docs/bark_detection_plan.md
 M docs/decisions.md
?? bark_detection/events.py
?? docs/prompts/m6_prompt.md
?? outputs/
```

**Last commit on `main`** — `adeb31a` ("Made steps 0,1,2,3,4 …"). M0–M4 are committed. M5 is uncommitted, waiting for user review. Remote: `github.com:ixials/chihuahua`.

User's stated rule: **no commits without explicit ask.**

## 5. What changed this session (milestone by milestone)

| Milestone | Status | New code | Artifact produced |
|---|---|---|---|
| **M0** | ✅ | none | `docs/bark_detection_plan.md` (audited repo, designed milestones) |
| **M1** | ✅ | `bark_detection/{__init__.py, config.py, audio_io.py, cli.py}`, `run_bark_detection.py` | `outputs/dogs1/{audio.wav, metadata.json}` |
| **M2** | ✅ | `bark_detection/rms.py`, `bark_detection/viz.py` | `outputs/dogs1/{rms_values.csv, rms_debug_plot.png}` |
| **M3** | ✅ | `bark_detection/threshold.py`; extended viz | `metadata.json[thresholding]`, `rms_smoothed` column, updated plot |
| **M4** | ✅ | `bark_detection/candidates.py` | `outputs/dogs1/bark_candidates.csv` (3 candidates) |
| **M5** | ✅ (uncommitted) | `bark_detection/events.py` | `outputs/dogs1/bark_events_initial.csv` (2 events) |

Every milestone after M0 was delegated to a `fullstack-developer` subagent with a self-contained prompt. The user reviewed the subagent's report before approving the next milestone.

## 6. Course corrections (the two slim passes — DO NOT undo)

Both were *deliberate* simplifications the user pushed back on. Documented here so the next chat doesn't re-introduce what we deleted.

### Trim 1 — "Option B" (after M1 first landed)

The original M1 produced three artifacts under `outputs/<stem>/<run_id>/`:
- `audio.wav`, `metadata.json` (12 fields), `run_manifest.json` (full provenance: git SHA, env, ffmpeg version, timings).

User's question: *"Do I really need the metadata? It's not like it does shit, does it?"*

Investigation showed only `fps` and `frame_count` are ever *read* downstream (M8). `run_manifest.json` is never read by any pipeline code.

**Decision:** drop `run_manifest.json` entirely. Drop the `<run_id>` per-run subdir (flat `outputs/<stem>/` instead). Shrink `metadata.json` to 4 fields. Both `research_workflow.md` §2.1 and §2.2 got "Deferred until M10 / paper time" notes — re-add as a small PR if/when paper work starts.

### Trim 2 — BarkConfig YAGNI (after Trim 1)

`BarkConfig` had **20 fields** pre-declared for M2–M8, but M1 only used 2. Plus an unused `as_dict()` method.

User's question: *"do i need all the files too for the m1 or are some things not needed?"*

**Decision:** trim `BarkConfig` to just the fields the *currently-implemented* milestones use. Each later milestone adds its own fields (M2 added 2, M3 added 5, M5 added 3). `__init__.py` became empty (re-export was unused).

`docs/bark_detection_plan.md` §8 keeps the *target* shape unchanged; `bark_detection/config.py` is just the *currently-implemented subset*. Per-milestone subagent prompts now include "add fields X, Y to BarkConfig" as part of the milestone scope.

## 7. Failed attempts / known caveats (carry forward)

Three caveats inherited from the original prompt-drafting session. Two are now resolved; one is queued for M6.

### ⚠ C1 — RESOLVED. M3 prompt mentioned a fabricated `threshold.json`
The `m3_prompt.md` told the executor to write a separate `threshold.json` file. `docs/bark_detection_plan.md` §M3 actually says the threshold lives inside `metadata.json` under a `thresholding` key.

**Resolution:** the M3 subagent prompt explicitly reconciled this — put the values under `metadata.json[thresholding]`, do NOT create `threshold.json`. The plan's §M3 was updated to match. Both candidate threshold values (`mean+2σ` AND `90th percentile`) are logged side-by-side so M10's Ablation A doesn't need a re-run.

**Lesson for the next chat:** every subagent prompt must include the rule "**Do NOT invent artifact filenames, columns, or fields that aren't in the plan. Quote them verbatim from the milestone's section. If something looks missing, surface it as an open question, do not silently add it.**"

### ⚠ C2 — RESOLVED. M5 left `merged_from` serialization unspecified
`m5_prompt.md` didn't pin the on-disk format for the `merged_from` column.

**Resolution:** the M5 executor used pandas' default list stringification — `"[0]"` for single-candidate events, `"[1, 2]"` for merged. Format is documented in §M5 results. The M6 prompt is already aware (see C3).

### ⚠ C3 — QUEUED for M6. `merged_from` parse-back behavior
`m6_prompt.md` tells the M6 executor to read `bark_events_initial.csv` and **surface any parse ambiguity in `merged_from` as an anomaly, not silently fix it**. It also notes that §M6's output schema does not list `merged_from`, so M6 may legitimately drop the column — the prompt asks the executor to surface this drop-vs-passthrough decision explicitly.

**Forecast for the next chat:** the format is `"[0]"` / `"[1, 2]"`, parseable with `ast.literal_eval`. The M6 executor will likely drop the column from `bark_events.csv` (per §M6's schema) but pass-through is also acceptable. Expect the subagent to flag this.

## 8. Codebase shape today

```
bark_detection/      738 lines, 9 files
  __init__.py     1   empty (was a re-export, removed in Trim 2)
  config.py      21   BarkConfig with 12 fields (M1+M2+M3+M5)
  audio_io.py    79   ffmpeg extract + ffprobe metadata (M1)
  rms.py         59   framed RMS via sliding_window_view (M2)
  threshold.py   77   smoothing + adaptive threshold (M3)
  candidates.py  89   above-threshold contiguous regions (M4)
  events.py     159   merge + duration filter + long_event flag (M5)
  cli.py        183   argparse, stage dispatch (`extract → rms → threshold → candidates → events`)
  viz.py         70   matplotlib raw + smoothed + threshold overlay
run_bark_detection.py   4-line shim → bark_detection.cli.main
```

## 9. Decisions logged (`docs/decisions.md`)

| Date | Topic | One-line decision |
|---|---|---|
| 2026-05-27 | RMS window/hop | 50 ms / 10 ms — bark plateau ≈ 100–300 ms, ≥ 5 frames per event |
| 2026-05-27 | Smoothing method | moving average over median — simplest baseline, preserves spike timing |
| 2026-05-27 | Threshold method | `mean + 2σ` over 90th percentile — adapts to clip energy, robust to quiet stretches |
| 2026-05-27 | Merge + duration rules | `merge_gap_ms=200, min=80, max=1200` — physically-motivated bark-shape rules |

Full text in `docs/decisions.md`. Both candidate threshold values are logged in `metadata.json[thresholding]` side-by-side for Ablation A.

## 10. Artifacts on disk (`outputs/dogs1/`)

```
audio.wav            mono 16 kHz 16-bit PCM (5.014 s)
metadata.json        fps, frame_count, sample_rate_hz, duration_sec, + thresholding block
rms_values.csv       time_sec, rms, rms_smoothed (497 rows ≈ 100 rows/sec)
rms_debug_plot.png   raw + smoothed + threshold line
bark_candidates.csv  3 rows: candidate regions above threshold
bark_events_initial.csv  2 rows: merged + duration-filtered events
```

`outputs/` is gitignored. The directory is per-video flat (no per-run subdir).

## 11. Working agreement — invariants for the next chat

Non-negotiables. Tell the next subagent these explicitly in its prompt:

- **Delegate every milestone to a `fullstack-developer` subagent.** Don't implement inline.
- **Use the saved prompt** in `docs/prompts/m<N>_prompt.md` as the starting point for the subagent's task description.
- **Reconcile against the saved prompts.** Each saved prompt mentions `<run_id>` subdir and `run_manifest.json` — both are **deferred until M10**. Tell the subagent to use the flat `outputs/<stem>/` layout and to skip any manifest update.
- **Do not pre-declare `BarkConfig` fields** beyond the current milestone. Each milestone grows the config by exactly the fields it uses.
- **Do NOT invent artifact filenames, columns, or fields** that aren't in the plan. Quote them verbatim from the milestone's section. If something looks missing, surface it as an open question, do not silently add it. (Carrying C1's lesson forward.)
- **No new packages.** numpy + scipy + pandas + matplotlib only, until M10 introduces YAMNet (`tensorflow-hub`).
- **No librosa.** `scipy.io.wavfile` is sufficient for everything before M10.
- **Stop after every milestone for user review.** Wait for explicit "go M<N+1>".
- **No commits** unless the user explicitly asks.
- **Update the docs** as part of each milestone: §4 status row → ✅, append a "M<N> — results" subsection at the end of §M<N>, append decision-log entry if a non-trivial choice was made.
- **Subagent reports back in the exact order** the saved prompt specifies (files / artifacts / headline numbers / PASS-FAIL / anomalies / next step). ≤ 500 words.

## 12. Next steps — M6 → M10

| # | Milestone | One-liner | Prompt file |
|---|---|---|---|
| M6 | Bark confidence scoring (+ classifier hook) | Add `bark_confidence ∈ [0,1]` per event from `normalized_rms_peak` + duration triangle; define `BarkClassifier` Protocol as the future seam for YAMNet / PANNs / CAV-MAE. Produces `bark_events.csv` (new file; M5's is `bark_events_initial.csv`). | `docs/prompts/m6_prompt.md` ✅ |
| M7 | Bark episodes | Group events within `episode_gap_sec=0.75` into bursts → `bark_episodes.csv`. Add `episode_id` column to `bark_events.csv`. | not yet drafted |
| M8 | Frame alignment + context windows | Convert every timestamp to a frame index via `metadata.json[fps]`. Add `start_frame / peak_frame / end_frame` and `context_*` columns. | not yet drafted |
| M9 | Final debug visualization | Update `rms_debug_plot.png` with shaded event regions, peak markers, episode bands. Stable copy → `docs/figures/F2_rms_overlay.png`. | not yet drafted |
| M10 | Tests + eval + code review | `tests/` (frames, rms, events, episodes); `bark_detection/eval.py` (IoU matching, P/R/F1); `bark_detection/baselines/yamnet.py` (Mode A + B); two ablations; `code-reviewer` agent pass. **YAMNet install lands here.** | not yet drafted |

Detailed acceptance criteria for each: `docs/bark_detection_plan.md` §M6–§M10.

## 13. How to start the next chat

1. Open a fresh Claude Code session in `/Users/aryahb/chihuahua`.
2. Tell it: **"Read `docs/handoff.md` first, then we'll continue."**
3. After it reads this file, paste the contents of [`docs/prompts/m6_prompt.md`](./prompts/m6_prompt.md) as the next message. That prompt is already self-contained and matches the per-milestone template the user has been using.
4. Remind the new chat to:
   - **Delegate M6 to a `fullstack-developer` subagent** (don't implement inline).
   - **Reconcile the saved prompt** against the deferred conventions in §11 (no `run_manifest.json`, no `<run_id>` subdir, no pre-declared M7+ fields, no inventing artifact names).
   - **Expect C3 to surface** — the subagent will likely flag a `merged_from` drop-vs-passthrough decision; that's correct behavior.

After M6 succeeds, the prompt files for M7–M10 still need to be drafted (per §12). The pattern from `docs/prompts/m<2..6>_prompt.md` is the template.
