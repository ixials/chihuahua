Use this prompt to start M6.

```
I reviewed docs/bark_detection_plan.md §M6 and docs/research_workflow.md §9
(row M6). Both match what I want.

Proceed with M6 only: Bark confidence scoring (with classifier hook).

Inputs:
- Prior-stage artifacts from M5: outputs/dogs1/bark_events_initial.csv
  (same schema as M4's candidates plus `event_type ∈ {bark, long_event}`
  and a `merged_from` list).
- Prior-stage artifacts from M1: outputs/dogs1/audio.wav (the scoring
  stage's signature accepts raw audio because the classifier hook
  receives `wav` + `sr` + `t_start` + `t_end`; with the default
  `classifier = None` only the heuristic path runs and `audio.wav` is
  not consumed numerically, but it must still be present on disk).
- Prior-stage artifacts from M1: outputs/dogs1/metadata.json.
- BarkConfig: use §8 defaults unless I override: none
  (w_rms=0.6, w_duration=0.4, duration_score_peak_sec=0.20,
   classifier=None, classifier_weight=0.5).

Constraints:
- No librosa, no new packages. Use numpy + scipy only.
- Use the flat layout `outputs/<stem>/` (per research_workflow.md §2.1's
  deferral banner — per-run subdirs and run_manifest.json kick in at
  M10, not now). Read from and write to outputs/dogs1/ directly.
- `BarkClassifier` is a `typing.Protocol`, NOT an implementation. Define
  it in `bark_detection/scoring.py` so future adapters (YAMNet, PANNs,
  CAV-MAE, custom heads) can satisfy it, but do NOT add any concrete
  classifier in M6. The default `classifier = None` path is the only
  one exercised on the sample clip — the classifier-blend code path
  must exist and be reachable (when `cfg.classifier is not None`) but
  is not invoked in this milestone's run.
- Quote the Protocol verbatim from §M6 of the plan:
      class BarkClassifier(Protocol):
          def score(self, wav: np.ndarray, sr: int, t_start: float, t_end: float) -> float: ...
  Do not rename or extend the signature.
- The `merged_from` column from M5 is read as-is and passed through to
  M6's output (or dropped if §M6's output schema does not list it —
  re-read §M6's output schema and follow it verbatim; the listed
  columns are `event_id, start_time_sec, peak_time_sec, end_time_sec,
  duration_sec, rms_peak, normalized_rms_peak, bark_confidence,
  event_type`, with frame/context columns deferred to M8 and
  `episode_id` deferred to M7). If `merged_from`'s serialization
  format from M5 is ambiguous when reading the CSV back in, do NOT
  silently re-pick a format — surface it as an anomaly in the report.
- Do not commit anything. Leave changes uncommitted for me to review.

After M6:
1. Update docs/bark_detection_plan.md:
   - Change M6's row in §4's status table to "✅ completed YYYY-MM-DD".
   - Append a short "M6 — results" subsection at the end of §M6 with the
     measured values listed below.
2. Produce the paper artifacts for M6 from research_workflow.md §9
   (row M6 lists three items; the first two are mandatory, the third
   is optional):
   - (Mandatory) Write the exact scoring formula with the §8 default
     weights filled in and explicitly label it as paper **Equation (1)**
     in the M6 — results subsection, so it can be lifted into Method
     §3.5 of the paper without rewording. With defaults the formula is:
         duration_score   = triangle(duration_sec, peak=0.20s, min=0.08s, max=1.2s)
         bark_confidence  = 0.6 * normalized_rms_peak + 0.4 * duration_score
     (When `cfg.classifier is not None`, the final blend is
         final = (1 - classifier_weight) * heuristic + classifier_weight * classifier.score(...)
      with `classifier_weight = 0.5` per §8 default.)
   - (Mandatory) Quote the `BarkClassifier` Protocol signature verbatim
     in the M6 — results subsection and explicitly mark it as the source
     text for the paper's Method §"Plug-in classifier" paragraph, so a
     future writing pass can lift it directly.
   - (Optional, only if you can produce it cheaply) A histogram figure
     of `bark_confidence` over the kept events. §9 row M6 lists this as
     "optional figure" only — do NOT invent a new mandatory artifact
     name. If produced, save it to outputs/dogs1/ with a clear filename
     and mention it in the report; if skipped, say so.
3. Append ONE decision-log entry to docs/decisions.md (workflow §7.1
   template) covering the scoring weights and duration-score triangle
   shape together: `w_rms = 0.6`, `w_duration = 0.4`, triangle
   `peak = 0.20 s, min = 0.08 s, max = 1.2 s`. Use the bark-shape
   reasoning from §M6 + §8 as the justification (RMS peak dominates;
   duration tempers extreme-short and extreme-long events; triangle
   peak ≈ 200 ms matches the typical bark plateau). Single entry, not
   four. This entry is MANDATORY — these are non-trivial picks, not
   defaults inherited unchanged from another paper.
4. If any non-trivial decision was made beyond the one above, append a
   5-line entry to docs/decisions.md (workflow §7.1 template).
5. If anything didn't work, append a 5-line entry to docs/failures.md
   (workflow §7.2 template). docs/failures.md does not yet exist on
   disk; create it with this first entry if needed.
6. Stop and report, in this order:
   a. Files created/modified (expected: `bark_detection/scoring.py`;
      extensions to `bark_detection/{config.py, cli.py}` for the new
      scoring config fields and the `score` stage).
   b. Output artifacts created (with paths), including
      outputs/dogs1/bark_events.csv. Note: per §M6 the output file is
      named `bark_events.csv`, NOT `bark_events_initial.csv` — M5's
      file is the input, M6's file is a new file.
   c. The headline numbers for this milestone:
      - The exact scoring formula with default weights filled in,
        explicitly labelled as paper Equation (1).
      - The `BarkClassifier` Protocol signature, quoted verbatim from
        §M6, explicitly marked for the paper's Method §"Plug-in
        classifier" paragraph.
      - Confirmation that `bark_confidence ∈ [0, 1]` for every output
        row; report the min and max observed values.
      - Monotonicity sanity check: sort the output rows by
        `normalized_rms_peak`, group into rough bins of similar
        `duration_score`, and confirm that within each bin
        `bark_confidence` is non-decreasing as `normalized_rms_peak`
        increases. Report any bin where this fails.
      - 2–3 sample event rows showing the new `bark_confidence` column
        (include `start_time_sec`, `duration_sec`, `normalized_rms_peak`,
        `bark_confidence`, `event_type`).
      - Path to the saved bark_events.csv.
      - (Optional) Path to a histogram of `bark_confidence` if produced;
        otherwise state that it was skipped (§9 row M6 lists it as
        optional, not mandatory).
   d. PASS/FAIL against each acceptance criterion in §M6, with measured values:
      - `bark_confidence ∈ [0, 1]` for every row (report min/max).
      - Holding `duration_score` constant, `bark_confidence` is monotonic
        in `normalized_rms_peak` (report the monotonicity check result).
   e. Any anomaly — including, explicitly, any ambiguity in re-reading
      M5's `merged_from` column from bark_events_initial.csv (M5 left
      its serialization format unspecified). If `merged_from` cannot be
      parsed unambiguously, flag it here; do not silently re-pick a format.
   f. Recommended next step.

If any acceptance criterion fails, report and stop. Do not silently work around it.

Do not start M7 until I approve.
```
