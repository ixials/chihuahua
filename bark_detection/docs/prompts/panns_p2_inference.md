# Prompt 2 — M3 only: PANNs inference per window

Use this prompt in **Agent mode** (Plan mode cannot edit Python).

## Scope

- **In:** `outputs/<stem>/audio.wav`, `panns_windows.csv`, `metadata.json`
- **Out:** `panns_scores.csv`, `panns_label_list.txt`, `panns_label_mapping.txt`
- **Also:** append label-mapping decision to `docs/decisions.md`
- **Not in scope:** M4+ timeline, Barkseqs, clips, viz, tests

## Requirements

1. Add `bark_detection/panns_inference.py` per plan §M3.
2. Model: **Cnn14_16k** @ **16 kHz** (match M1 WAV). STFT inside model: window=512 samples, hop=160, mel=64, fmax=8000 Hz.
3. `ensure_panns_assets()` via **urllib** (not wget): `~/panns_data/class_labels_indices.csv` + `Cnn14_16k_mAP=0.438.pth`.
4. **No hard-coded label indices** — resolve from installed label list; `bark_score = max(Bark, Yip, Bow-wow)`.
5. **Padding (M3 only):** slice true `[start,end)` from WAV; if shorter than `window_size_sec * sr`, zero-pad to 16000 samples for inference. CSV times unchanged.
6. Extend `config.py`: `panns_model_name`, `panns_device` (default `cpu`).
7. CLI: stage `panns`; `_STAGE_ORDER = ["extract", "windows", "panns"]`.
8. `requirements.txt`: add `pandas`, `torch`, `panns-inference`.

## Report after run (dogs1)

Same six-part summary as Prompt 1. Include label mapping lines and bark_score at windows whose centers are near 0.25 s and 1.5 s.
