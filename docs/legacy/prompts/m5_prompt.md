Use this prompt to start M5.

```
I reviewed docs/bark_detection_plan.md §M5 and docs/research_workflow.md §9
(row M5). Both match what I want.

Proceed with M5 only: Merge nearby peaks + duration rules.

Inputs:
- Prior-stage artifacts from M4: outputs/dogs1/<run_id>/bark_candidates.csv
  (columns: candidate_id, start_time_sec, end_time_sec, peak_time_sec,
   duration_sec, rms_peak, normalized_rms_peak).
- Prior-stage artifacts from M1/M3: outputs/dogs1/<run_id>/metadata.json
  and the existing outputs/dogs1/<run_id>/run_manifest.json in the same
  run dir. I will supply <run_id> at runtime.
- BarkConfig: use §8 defaults unless I override: none
  (merge_gap_ms=200, min_duration_ms=80, max_duration_ms=1200).

Constraints:
- No librosa, no new packages. Use numpy + scipy only.
- Use the per-run subdirectory layout from research_workflow.md §2.1.
- Update run_manifest.json if any new config field is read.
- Do not commit anything. Leave changes uncommitted for me to review.

After M5:
1. Update docs/bark_detection_plan.md:
   - Change M5's row in §4's status table to "✅ completed YYYY-MM-DD".
   - Append a short "M5 — results" subsection at the end of §M5 with the
     measured values listed below.
2. Produce the paper artifact for M5 from research_workflow.md §9:
   - Log the three counts (raw candidates → after-merge → after-duration)
     so I can drop them into the paper's Method/Results section as the
     recall-funnel sentence.
   - Append ONE decision-log entry to docs/decisions.md (workflow §7.1
     template) covering all three parameters at once: merge_gap_ms=200,
     min_duration_ms=80, max_duration_ms=1200. Use the plan's
     "physically-motivated bark-shape rules" framing as the reasoning —
     single entry, not three.
3. If any non-trivial decision was made beyond the one above, append a
   5-line entry to docs/decisions.md (workflow §7.1 template).
4. If anything didn't work, append a 5-line entry to docs/failures.md
   (workflow §7.2 template).
5. Stop and report, in this order:
   a. Files created/modified.
   b. Output artifacts created (with paths), including
      outputs/dogs1/<run_id>/bark_events_initial.csv (same schema as
      candidates plus `event_type ∈ {bark, long_event}` and a
      `merged_from` list).
   c. The headline numbers for this milestone:
      - Three counts in sequence: raw candidate count → after-merge count
        → after duration-filter count.
      - Confirmation that no two output events are within `merge_gap_ms`
        (200 ms) of each other; report the minimum inter-event gap
        observed across the saved events.
      - Confirmation that every kept event satisfies
        `min_duration_ms ≤ duration ≤ max_duration_ms` UNLESS flagged
        `event_type = "long_event"`. Report the count of `long_event`
        rows separately.
      - 2–3 sample rows of kept events (showing the `merged_from` field
        if non-empty), and 1–2 rows of dropped events with the reason
        ("below min_duration" or "filtered out").
      - Path to the saved bark_events_initial.csv.
   d. PASS/FAIL against each acceptance criterion in §M5, with measured values:
      - After merging, no two events are within `merge_gap_ms` of each other.
      - Every kept event satisfies `min_duration_ms ≤ duration ≤ max_duration_ms`
        unless flagged `long_event`.
      - Long events are flagged, not deleted (and may be split if reasonable
        internal peaks exist).
   e. Any anomaly.
   f. Recommended next step.

If any acceptance criterion fails, report and stop. Do not silently work around it.

Do not start M6 until I approve.
```
