Use this prompt to start M4.

```
I reviewed docs/bark_detection_plan.md §M4 and docs/research_workflow.md §9
(row M4). Both match what I want.

Proceed with M4 only: Candidate region detection.

Inputs:
- Prior-stage artifacts from M3: outputs/dogs1/<run_id>/rms_values.csv
  (with both `rms` and `rms_smoothed` columns populated) and
  outputs/dogs1/<run_id>/metadata.json (threshold value lives under the
  `thresholding` key; use the operating-point threshold M3 chose).
  I will supply <run_id> at runtime.
- BarkConfig: use §8 defaults unless I override: none. M4 reads the
  threshold computed in M3; no new tunables.

Constraints:
- No librosa, no new packages. Use numpy + scipy only.
- Use the per-run subdirectory layout from research_workflow.md §2.1.
- Update run_manifest.json if any new config field is read.
- Do not commit anything. Leave changes uncommitted for me to review.

After M4:
1. Update docs/bark_detection_plan.md:
   - Change M4's row in §4's status table to "✅ completed YYYY-MM-DD".
   - Append a short "M4 — results" subsection at the end of §M4 with the
     measured values listed below.
2. Produce the paper artifact for M4 from research_workflow.md §9:
   - Report the candidate count so I can compare it against my manual
     count of audible barks on the sample clip (recall sanity check).
3. If any non-trivial decision was made, append a 5-line entry to
   docs/decisions.md (workflow §7.1 template). M4 is deterministic given
   M3's threshold, so this is expected to be skipped.
4. If anything didn't work, append a 5-line entry to docs/failures.md
   (workflow §7.2 template).
5. Stop and report, in this order:
   a. Files created/modified.
   b. Output artifacts created (with paths), including
      outputs/dogs1/<run_id>/bark_candidates.csv.
   c. The headline numbers for this milestone:
      - Total candidate count (for me to compare against my manual count
        of audible barks on the sample clip).
      - Range check: `start_time_sec < peak_time_sec ≤ end_time_sec`
        holds for every row (report row count checked and any violations).
      - Range check: `normalized_rms_peak ∈ (0, 1]` for every row
        (report min and max of `normalized_rms_peak`).
      - 2–3 sample candidate rows (with start/peak/end times) so I can
        spot-check against what I hear.
      - Path to the saved bark_candidates.csv.
   d. PASS/FAIL against each acceptance criterion in §M4, with measured values:
      - `start_time_sec < peak_time_sec ≤ end_time_sec` for every row.
      - `normalized_rms_peak ∈ (0, 1]`.
      - Number of candidates ≥ number of audible barks (recall-favored) —
        report the candidate count; I will confirm against my manual count.
   e. Any anomaly.
   f. Recommended next step.

If any acceptance criterion fails, report and stop. Do not silently work around it.

Do not start M5 until I approve.
```
