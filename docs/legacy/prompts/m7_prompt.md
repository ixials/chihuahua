Use this prompt to start M7.

```
I reviewed docs/bark_detection_plan.md §M7 and docs/research_workflow.md §9
(row M7). Both match what I want.

Proceed with M7 only: Bark episodes.

Inputs:
- Prior-stage artifacts from M6: outputs/dogs1/bark_events.csv with the
  M6 schema (10 columns: event_id, start_time_sec, peak_time_sec,
  end_time_sec, duration_sec, rms_peak, normalized_rms_peak,
  bark_confidence, event_type, merged_from). `merged_from` is read
  passively — M7 does not use it for grouping and does not emit it in
  bark_episodes.csv, but it must not be dropped from bark_events.csv.
- Prior-stage artifacts from M1: outputs/dogs1/metadata.json (for the
  total clip duration sanity check listed in step 6.e).
- BarkConfig: use §8 default unless I override: none
  (episode_gap_sec = 0.75).

Constraints:
- No librosa, no new packages. numpy + scipy only (pandas is already in
  use for CSV I/O).
- Use the flat layout `outputs/<stem>/` (per research_workflow.md §2.1's
  deferral banner — per-run subdirs and run_manifest.json kick in at
  M10, not now). Read from and write to outputs/dogs1/ directly.
- Episode grouping rule (prescribed — do NOT redesign):
      same_episode(event[i], event[i+1])
          iff
      event[i+1].start_time_sec - event[i].end_time_sec  <=  episode_gap_sec
  The test is `<=`, not `<`. Events whose gap equals exactly
  `episode_gap_sec` belong to the SAME episode. Sort events by
  `start_time_sec` before applying the rule. `episode_id` is 0-based
  and assigned in time order.
- Per-episode aggregates (prescribed — do NOT redesign):
      number_of_barks      = count of events in the episode
      average_confidence   = arithmetic mean of bark_confidence over those events
      max_confidence       = max of bark_confidence over those events
      start_time_sec       = min(start_time_sec) of those events
      end_time_sec         = max(end_time_sec) of those events
- bark_events.csv update is in-place: add an `episode_id` int column to
  the existing file. Do NOT modify, reorder, or drop any existing
  column (including `merged_from`). The file is overwritten with the
  new column appended.
- Do not commit anything. Leave changes uncommitted for me to review.

After M7:
1. Update docs/bark_detection_plan.md:
   - Change M7's row in §4's status table to "✅ completed YYYY-MM-DD".
   - Append a short "M7 — results" subsection at the end of §M7 with the
     measured values listed below.
2. Produce the paper artifact for M7 from research_workflow.md §9
   (row M7 lists ONE mandatory item — one Method paragraph plus the
   episodes-per-clip number; there are no optional artifacts):
   - (Mandatory) Draft the one-paragraph Method blurb covering:
       (a) the episode definition (events grouped when consecutive
           inter-event gap <= episode_gap_sec),
       (b) the `episode_gap_sec = 0.75` rationale (bark-burst spacing),
       (c) the `average_confidence` and `max_confidence` columns
           (what they aggregate over and why both are reported).
     Place this paragraph in the M7 — results subsection and explicitly
     mark it as the source text for the paper's Method §"Episodes"
     paragraph, so a future writing pass can lift it directly.
   - (Mandatory number) Report the episodes-per-clip count for dogs1.
3. Append ONE decision-log entry to docs/decisions.md (workflow §7.1
   template) justifying `episode_gap_sec = 0.75`. Reasoning to use:
   typical inter-bark gap inside a barking burst is ≈ 0.3–0.7 s;
   0.75 s gives slight slack so a within-burst pause does not split
   the burst; a longer value risks merging unrelated bursts, a shorter
   one risks splitting one burst into several episodes. Single entry.
   This entry is MANDATORY — `episode_gap_sec` is a non-trivial pick
   even though it is listed as a default in §8 (workflow §9 row M7
   explicitly calls for the rationale).
4. If any non-trivial decision was made beyond the one above, append a
   5-line entry to docs/decisions.md (workflow §7.1 template).
5. If anything didn't work, append a 5-line entry to docs/failures.md
   (workflow §7.2 template). docs/failures.md may not yet exist on
   disk; create it with this first entry if needed.
6. Stop and report, in this order:
   a. Files created/modified (expected: `bark_detection/episodes.py`
      new; extensions to `bark_detection/{config.py, cli.py}` for the
      new `episode_gap_sec` field and the `episodes` stage —
      `_STAGE_ORDER`, `_DISPATCH`, and the `--stage` argparse `choices`
      list must all gain "episodes").
   b. Output artifacts created (with paths):
      - outputs/dogs1/bark_episodes.csv (new file, 6 columns:
        episode_id, start_time_sec, end_time_sec, number_of_barks,
        average_confidence, max_confidence).
      - outputs/dogs1/bark_events.csv (overwritten in place with the
        new `episode_id` column appended — all other columns,
        including `merged_from`, preserved unchanged).
   c. Headline numbers for this milestone:
      - The Method paragraph in full (the text from step 2 item 1).
      - Episodes-per-clip count for dogs1 (single integer).
      - 2–3 sample rows from bark_episodes.csv showing all 6 columns.
      - A sample slice of bark_events.csv showing event_id alongside
        the new episode_id column populated correctly.
      - Paths to both files.
   d. PASS/FAIL against each acceptance criterion in §M7, with measured values:
      - Containment: every event's [start_time_sec, end_time_sec]
        is contained within exactly one episode's [start_time_sec,
        end_time_sec]. Report any violating event.
      - Count match: for every episode, `number_of_barks` equals the
        count of rows in bark_events.csv with that `episode_id`.
        Report any mismatch.
   e. Any anomaly — including, explicitly: does any episode's
      `end_time_sec` exceed the total clip duration from metadata.json?
      (No `merged_from` warning is needed for M7 since M7 does not
      parse or emit that column.)
   f. Recommended next step.

If any acceptance criterion fails, report and stop. Do not silently work around it.

Do not start M8 until I approve.
```
