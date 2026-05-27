Use this prompt to start M3.

```
I reviewed docs/bark_detection_plan.md §M3 and docs/research_workflow.md §9
(row M3). Both match what I want.

Proceed with M3 only: Smoothing + adaptive threshold.

Inputs:
- Prior-stage artifacts from M2: outputs/dogs1/<run_id>/rms_values.csv
  (with `rms` column populated, `rms_smoothed` column reserved/empty).
- Prior-stage artifacts from M1: outputs/dogs1/<run_id>/metadata.json and
  outputs/dogs1/<run_id>/audio.wav (I will supply <run_id> at runtime).
- BarkConfig: use §8 defaults unless I override: none
  (smoothing_method="moving_average", smoothing_window_ms=50,
   threshold_method="mean_std", mean_std_k=2.0, percentile=90).

Constraints:
- No librosa, no new packages. Use numpy + scipy only.
- Use the per-run subdirectory layout from research_workflow.md §2.1.
- Update run_manifest.json if any new config field is read.
- Do not commit anything. Leave changes uncommitted for me to review.

After M3:
1. Update docs/bark_detection_plan.md:
   - Change M3's row in §4's status table to "✅ completed YYYY-MM-DD".
   - Append a short "M3 — results" subsection at the end of §M3 with the
     measured values listed below.
2. Produce the paper artifact for M3 from research_workflow.md §9:
   - Update figure F2 at outputs/dogs1/<run_id>/rms_debug_plot.png to show
     raw RMS, smoothed RMS, and the threshold line.
   - Append TWO decision-log entries to docs/decisions.md (workflow §7.1
     template) — one for the smoothing-method pick and one for the
     threshold-method pick — using the reasoning from §M3's "Default
     recommendation" paragraph: moving average is the simplest, fastest
     baseline and easy to reason about; `mean + 2σ` adapts to clip energy
     and is robust to long quiet stretches where percentile thresholds
     would still fire on noise.
   - Log BOTH `mean+2σ` and `90th percentile` threshold values
     side-by-side in metadata.json under a `thresholding` key (e.g.,
     `thresholding.mean_std_k2`, `thresholding.percentile_90`, plus
     `thresholding.operating_point` naming which method was used). This
     captures Ablation A's data now so the paper's Table 2 has the
     numbers without a re-run.
3. If any non-trivial decision was made beyond the two above, append a
   5-line entry to docs/decisions.md (workflow §7.1 template).
4. If anything didn't work, append a 5-line entry to docs/failures.md
   (workflow §7.2 template).
5. Stop and report, in this order:
   a. Files created/modified.
   b. Output artifacts created/updated (with paths), including
      outputs/dogs1/<run_id>/rms_values.csv (now with `rms_smoothed`
      populated), outputs/dogs1/<run_id>/threshold.json, and
      outputs/dogs1/<run_id>/rms_debug_plot.png.
   c. The headline numbers for this milestone:
      - Variance of raw RMS vs variance of smoothed RMS (report both
        numbers; smoothed must be lower).
      - Threshold value at the operating point (`mean + 2σ` of
        `rms_smoothed`), and the percentile-based threshold value
        (90th percentile of `rms_smoothed`) — BOTH logged into
        metadata.json under `thresholding`.
      - Confirmation that the operating-point threshold lies strictly
        between `mean(rms_smoothed)` and `max(rms_smoothed)` (report
        all three numbers).
      - Path to the updated rms_debug_plot.png showing raw RMS,
        smoothed RMS, and the threshold line.
   d. PASS/FAIL against each acceptance criterion in §M3, with measured values:
      - Smoothed curve has lower variance than raw curve.
      - Threshold lies between `mean(rms_smoothed)` and `max(rms_smoothed)`.
   e. Any anomaly.
   f. Recommended next step.

If any acceptance criterion fails, report and stop. Do not silently work around it.

Do not start M4 until I approve.
```
