# Decision log

> Append-only. One entry per non-trivial decision. Template from `research_workflow.md` §7.1.

## 2026-05-27 — RMS window/hop: 50 ms / 10 ms
**Context:** Need a framed RMS energy curve that resolves bark events.
**Options:** (a) 25/10 ms (finer); (b) 50/10 ms (default); (c) 100/25 ms (coarser).
**Decision:** 50 ms window with 10 ms hop.
**Why:** Bark plateaus are ≈ 100–300 ms; 50 ms window keeps the spike shape; 10 ms hop guarantees ≥ 5 frames per event for downstream contiguous-region detection.
**Revisit if:** events get fragmented (consider widening window) or sub-event structure matters (consider tightening hop).

## 2026-05-27 — Smoothing method: moving average over median
**Context:** Need to denoise the framed RMS curve before thresholding.
**Options:** (a) moving average; (b) median filter; (c) Savitzky–Golay.
**Decision:** Moving average with a 50 ms window (5 frames at 10 ms hop).
**Why:** Simplest, fastest baseline; easy to reason about; preserves spike timing acceptably for bark plateaus ≈ 100–300 ms; median over a 50 ms window risks fragmenting single barks (logged for revisit under Ablation B).
**Revisit if:** plot shows event fragmentation, or Ablation B reveals median is materially better.

## 2026-05-27 — Threshold method: mean + 2σ over 90th percentile
**Context:** Need an adaptive energy threshold separating "interesting" from "background" on the smoothed RMS curve.
**Options:** (a) mean + k·σ; (b) percentile of smoothed RMS; (c) Otsu.
**Decision:** mean + 2σ as the operating point; 90th percentile logged side-by-side for Ablation A.
**Why:** mean + 2σ adapts to per-clip energy; robust to long quiet stretches that would still trigger a fixed-percentile threshold from background noise; one tunable (k); matches the RMS-gating idea in "What's That Sound Right Now?".
**Revisit if:** Ablation A on the small annotated set shows percentile is materially better.

## 2026-05-27 — Event merge + duration rules: merge_gap_ms=200, min_duration_ms=80, max_duration_ms=1200
**Context:** Threshold crossings fragment single barks into multiple candidates and also pick up sub-bark clicks and atypical long noises.
**Options:** (a) tighter merge gap (100–150 ms) + tighter min (100 ms); (b) defaults from the plan (200/80/1200); (c) much looser (300/50/2000).
**Decision:** (b) — merge_gap_ms=200, min_duration_ms=80, max_duration_ms=1200.
**Why:** Physically-motivated bark-shape rules — a single bark plateau is ≈ 100–300 ms; bursts of consecutive barks rarely sit closer than 200 ms apart in audio; events shorter than 80 ms look like clicks/noise; events longer than 1200 ms are too long to be a single bark and are flagged (not deleted) for later splitting or manual review.
**Revisit if:** sample clips reveal a different inter-bark cadence or systematic mis-merges, or M10's Ablation B suggests retuning.

## 2026-05-28 — Confidence scoring weights + triangle shape: w_rms=0.6, w_duration=0.4, triangle peak=0.20s, min=0.08s, max=1.2s
**Context:** Need a heuristic `bark_confidence ∈ [0, 1]` for each event using only cheap signal-shape features available without a trained classifier.
**Options:** Equal weights (0.5/0.5); RMS-dominant (0.6/0.4); duration-dominant (0.3/0.7). Triangle peak at 0.15 s, 0.20 s, or 0.25 s; min/max set to match or widen the duration-filter bounds [80, 1200] ms.
**Decision:** `w_rms=0.6, w_duration=0.4`; triangle `peak=0.20s, min=0.08s, max=1.2s`.
**Why:** RMS peak is a more direct proxy for bark loudness and is the primary discriminant used in "What's That Sound Right Now?" — it deserves the larger weight. Duration tempers extremes: events that are very short (< 80 ms, click-like) or very long (> 1.2 s, non-bark) receive low duration_score, dragging confidence down even when RMS is high. Triangle min/max align with the M5 duration-filter bounds so the two stages are consistent. Triangle peak ≈ 200 ms matches the typical bark plateau and is the same value used in the plan's §M6 formula.
**Revisit if:** M10 ablation reveals duration weight should be higher, or if a trained classifier (YAMNet / PANNs) shows a different optimal blend via `classifier_weight`.

## 2026-05-28 — PANNs label mapping (Cnn14_16k, resolved at runtime)
**Context:** M3 must emit `dog_score`, `bark_score`, `animal_score`, `speech_score`, and `music_score` from AudioSet sigmoid outputs without hard-coded label indices.
**Options:** (a) hard-code indices from docs (e.g. Bark=75); (b) load `class_labels_indices.csv` and match exact `display_name` strings at runtime.
**Decision:** (b) — resolve indices from the installed PANNs label list after `ensure_panns_assets()` downloads `~/panns_data/class_labels_indices.csv`. Mapping on `dogs1` (527 classes):
- `dog_score` → `Dog` → **74**
- `bark_score` → `max(Bark, Yip, Bow-wow)` → **75**, **76**, **78**
- `animal_score` → `Animal` → **72**
- `speech_score` → `Speech` → **0**
- `music_score` → `Music` → **137**
**Why:** AudioSet label order is authoritative only from the CSV; docs can drift. `bark_score` uses the max of three dog-vocalization labels to catch yips and bow-wows, not only canonical `Bark`. Short final windows are zero-padded to `int(window_size_sec * sr)` (= 16000 samples) inside inference only; CSV times stay unpadded.
**Revisit if:** M4 timeline plots show `Dog`/`Animal` dominating over `Bark` on known bark windows, or if a finer label set (e.g. `Howl`, `Whimper`) improves recall.

## 2026-05-28 — combined_bark_score: bark_score alone (`combined_bark_mode="bark"`)
**Context:** M4 must collapse per-window PANNs scores into a timeline that peaks at the two audible barks on `dogs1` (~0.25 s and ~1.5 s; window centers at 0.5 s and 1.5 s with 1 s / 0.25 s hop).
**Options:** (a) `bark_score` alone; (b) `max(bark_score, dog_score)`; (c) weighted blend (e.g. `0.7 * bark + 0.3 * dog`); (d) include `animal_score` (rejected — too broad).
**Decision:** (a) — `combined_bark_score = bark_score`; config key `combined_bark_mode="bark"` (alternate `"max_bark_dog"` for ablation).
**Why:** On `dogs1`, `bark_score` peaks at the two bark windows (0.49 at center 0.5 s; 0.55–0.61 at 1.5–1.75 s) and drops to 0.03 at center 1.0 s between barks. `Animal` tops `top_1` almost everywhere but stays high (0.90) in the inter-bark trough — unusable. `dog_score` stays elevated during non-bark dog sounds (panting at ~3 s: dog ≈ 0.84, bark ≈ 0.01), so `max(bark, dog)` would create false peaks and raise the 1.0 s trough to 0.21. Bark-specific labels (`Bark`, `Yip`, `Bow-wow`) already gate on vocalization type.
**Revisit if:** clips where barks are quiet but dog presence is strong show missed peaks, or weighted/max modes improve M5 Barkseq recall on the annotated set.

## 2026-05-28 — Barkseq threshold 0.42 and merge_gap_sec 0.5
**Context:** M5 must segment `bark_score_timeline.csv` into Barkseqs on `dogs1` with two audible barks; legacy RMS found 2 events at ~0.19–0.33 s and ~1.48–1.76 s.
**Options:** (a) `barkseq_threshold=0.35`; (b) **0.42**; (c) 0.50; merge_gap 0.5 s vs 0.75 s.
**Decision:** `barkseq_threshold=0.42`, `merge_gap_sec=0.5`. Positive centers expand by `hop_size_sec/2` (= 0.125 s) each side.
**Why:** With `combined_bark_score=bark_score`, 0.35 keeps the t=1.25 window (bark=0.403) in one chain with the ~1.5 s burst, and 0.75 s gap after merging the first burst still yields **one** merged Barkseq. **0.42** drops center 0.75 (bark=0.417) so the first burst isolates; inter-burst gap 0.625→1.375 = **0.75 s > 0.5 s** → **2 Barkseqs**. 0.50 misses the first bark (0.49). Second Barkseq end extends to 2.125 s because t=2.0 (bark=0.522) stays above threshold — acceptable until M6/M7 tuning.
**Revisit if:** M10 eval shows systematic early/late boundaries; try `max_bark_dog` mode or per-clip threshold calibration.

## 2026-05-28 — PANNs M6 noise thresholds: speech/music 0.15
**Context:** M6 must flag Barkseqs contaminated by speech or music without dropping rows; `dogs1` has low in-span speech/music during barks but high speech (0.155) in the inter-bark trough at t=1.0 s (correctly excluded by M5).
**Options:** (a) 0.10 (more sensitive); (b) **0.15**; (c) 0.20 (fewer flags).
**Decision:** `speech_noise_threshold=0.15`, `music_noise_threshold=0.15`; `strong_bark` uses existing `barkseq_threshold=0.42`.
**Why:** 0.15 sits above in-span noise on `dogs1` (max speech ~0.058) while still catching moderate TV/speech overlap in longer clips. Weak-bark + high speech/music → `likely_noise`; strong bark + high speech → keep with `high_speech` flag.
**Revisit if:** annotated clips show missed contamination or excessive `likely_noise` on real TV/speech backgrounds.

## 2026-05-28 — PANNs M7 confidence: speech_penalty=0.3, music_penalty=0.3
**Context:** M7 needs an interpretable ranking score ∈ [0, 1] on final `barkseqs.csv` for downstream filtering before stage-3 dog assignment.
**Options:** (a) use `max_combined_bark_score` alone; (b) multiplicative penalties with α,β ∈ {0.2, **0.3**, 0.5}; (c) subtractive blend.
**Decision:** `confidence = clip(mean_combined_bark_score × (1 − 0.3 × max_speech_score) × (1 − 0.3 × max_music_score), 0, 1)`; `method=panns_cnn14_16k_v1`.
**Why:** `mean_combined` smooths burst tails; max speech/music penalties are conservative (one loud window inside a span pulls confidence down). On `dogs1`: id=0 → 0.477, id=1 → 0.542. Not a calibrated probability — hand-tuned rank score.
**Revisit if:** M8 clip listening or stage-3 eval shows poor rank ordering; tune α/β or switch base to `max_combined_bark_score`.

## 2026-05-28 — M8 clip padding: ±0.25 s, prevent overlap with neighbors
**Context:** Exact event-span clips are timestamp-accurate but harsh for listening; adjacent Barkseqs can steal each other's padding without bounds.
**Options:** (a) exact span only; (b) fixed padding clamped to audio; (c) **(b) + clamp to neighbor event boundaries** when `prevent_clip_overlap=true`.
**Decision:** `clip_pre_context_sec=0.25`, `clip_post_context_sec=0.25`, `prevent_clip_overlap=true`. Record `clip_*` columns on `barkseqs.csv`; keep `start_time_sec`/`end_time_sec` as detection bounds. M9 frame columns stay on event + visual context, not clip times.
**Why:** 250 ms pre-roll gives the ear context before the bark; neighbor clamp prevents overlapping WAV files when bursts are close. On `dogs1` gap 0.75 s between events — full padding retained (0.75 s and 1.25 s clips vs 0.25 s / 0.75 s event duration).
**Revisit if:** clips still feel tight; increase to 0.5 s or add separate `clip_padding_sec` symmetric knob.

## 2026-05-28 — M9 frame alignment: round × fps, context ±0.5 s
**Context:** Stage 3 needs frame indices to associate Barkseqs with YOLO tracks on video.
**Options:** floor vs round for frame index; context 0.25 s vs **0.5 s** vs 1.0 s.
**Decision:** `frame_id = round(time_sec × fps)` clamped to `[0, frame_count − 1]`; `pre_context_sec=0.5`, `post_context_sec=0.5`.
**Why:** Round matches intuitive “nearest frame” for peak alignment; ±0.5 s gives one second of context window at 30 fps (15 frames each side) for vision models. On `dogs1`: peak frames 15 and 52 at 0.5 s and 1.75 s.
**Revisit if:** stage-3 needs tighter crops or frame-accurate bbox sync; try floor/ceil or smaller context.

## 2026-05-28 — M10 debug plots: score timeline + Barkseq overlay
**Context:** Need inspectable figures tying PANNs scores to final Barkseq regions without opening CSVs.
**Options:** (a) single combined plot; (b) **timeline plot + overlay with shaded regions**; (c) interactive HTML.
**Decision:** (b) — `debug/panns_score_timeline.png` (5 score lines + threshold 0.42) and `debug/barkseq_overlay.png` (same + shaded event spans + peak dots). Legacy RMS plot stays in `legacy_rms/`.
**Why:** Overlay makes M5 segmentation visually auditable on `dogs1`; threshold line shows why t=1.0 s trough stays below detection.
**Revisit if:** plots get crowded on long clips; add clip-span shading or downsample timeline.
