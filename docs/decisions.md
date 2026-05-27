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
