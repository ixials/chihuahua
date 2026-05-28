Use this prompt to start M2.

```
I reviewed docs/bark_detection_plan.md §M2 and docs/research_workflow.md §9
(row M2). Both match what I want.

Proceed with M2 only: RMS energy curve.

Inputs:
- Prior-stage artifacts from M1: outputs/dogs1/<run_id>/audio.wav and
  outputs/dogs1/<run_id>/metadata.json (I will supply <run_id> at runtime).
- BarkConfig: use §8 defaults unless I override: none (window_ms=50, hop_ms=10).

Constraints:
- No librosa, no new packages. Use numpy + scipy.io.wavfile only.
- Use the per-run subdirectory layout from research_workflow.md §2.1.
- Update run_manifest.json if any new config field is read.
- Do not commit anything. Leave changes uncommitted for me to review.

After M2:
1. Update docs/bark_detection_plan.md:
   - Change M2's row in §4's status table to "✅ completed YYYY-MM-DD".
   - Append a short "M2 — results" subsection at the end of §M2 with the
     measured values listed below.
2. Produce the paper artifact for M2 from research_workflow.md §9:
   - First version of figure F2 (raw RMS only) saved at
     outputs/dogs1/<run_id>/rms_debug_plot.png.
   - Append a decision-log entry to docs/decisions.md (workflow §7.1 template)
     for the window/hop choice (50/10 ms) with reasoning: bark plateau
     ≈ 100–300 ms; need ≥ 5 frames per event.
3. If any non-trivial decision was made beyond window/hop, append a 5-line
   entry to docs/decisions.md (workflow §7.1 template).
4. If anything didn't work, append a 5-line entry to docs/failures.md
   (workflow §7.2 template).
5. Stop and report, in this order:
   a. Files created/modified.
   b. Output artifacts created (with paths), including
      outputs/dogs1/<run_id>/rms_values.csv and
      outputs/dogs1/<run_id>/rms_debug_plot.png.
   c. The headline numbers for this milestone:
      - RMS array length, and the expected value from the formula
        floor((num_samples - window) / hop) + 1 (report both; confirm match).
      - min / max / mean of the RMS array.
      - Confirmation that there are no NaN/Inf values and all values ≥ 0.
      - Path to the saved rms_debug_plot.png (raw RMS only, first version).
      - rms_values.csv row count and rows-per-second (expected ~100 rows/sec).
   d. PASS/FAIL against each acceptance criterion in §M2, with measured values:
      - RMS array length ≈ floor((num_samples - window) / hop) + 1.
      - All values ≥ 0; no NaN/Inf.
      - Visible spikes during the audible barks on the sample video.
   e. Any anomaly.
   f. Recommended next step.

If any acceptance criterion fails, report and stop. Do not silently work around it.

Do not start M3 until I approve.
```
