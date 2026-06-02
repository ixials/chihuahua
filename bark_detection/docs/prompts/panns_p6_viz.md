# Prompt 6 — M10 debug visualizations

> Grouped executor prompt. Stop after completion; deliver §4 review summary.

## Scope

Implement **M10** — publication-style debug PNGs under `outputs/<stem>/debug/`.

## Prerequisites

- `bark_score_timeline.csv`, `barkseqs.csv`

## M10 — Viz

**Module:** extend `bark_detection/viz.py`  
**CLI stage:** `viz`

**Outputs:**

1. `debug/panns_score_timeline.png` — time vs combined_bark, bark, dog, speech, music; barkseq threshold line
2. `debug/barkseq_overlay.png` — same scores + shaded Barkseq regions (event start/end) + peak markers

## Acceptance

- PNGs open; legend present; shaded regions match `barkseqs.csv` event bounds
