# BirdCLEF 2026 — Acoustic Species Identification in the Pantanal

**Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) — Kaggle  
**Task:** Multi-label bird species detection from passive acoustic monitoring soundscapes  
**Metric:** cMAP (competition mean Average Precision)  
**Final Score:** 0.657  

---

## Approach

The core model is a **ConvNeXt-based audio classifier** trained on mel spectrograms of 7-second audio windows. The submission pipeline processes each test soundscape by sliding a 7-second window at 5-second steps (2 seconds of overlap between consecutive chunks), using 1-second lookback context and reflection padding at boundaries.

**Preprocessing pipeline:**
- Audio loaded at 32kHz, mono
- Mel spectrogram: 128 mels, 40–15000 Hz frequency range
- Log-scale conversion via `power_to_db`, min-max normalised per chunk
- Resized to (128, 313) and tiled to 3 channels for ConvNeXt input

---

## Ensemble Attempts — Google Perch

The primary research goal was to build a **specialist/generalist ensemble** combining the fine-tuned ConvNeXt (specialist) with Google's Perch bird vocalization classifier (generalist), motivated by the hypothesis that Perch's broad pretraining would complement ConvNeXt on underrepresented Pantanal species.

Four Perch integration paths were attempted, all blocked by Kaggle's CPU-only submission environment:

| Approach | Outcome |
|---|---|
| Perch v8 via TF Hub | CPU inference too slow — submission timeout |
| Selective Perch (uncertain chunks only) | Timeout persisted; blending also degraded score to 0.507 |
| Perch v2 via TF Hub | `XlaCallModule` version incompatibility with Kaggle TF |
| `perch_v2_cpu` SavedModel | StableHLO v1.9.1 incompatible with TF 2.20.0 on Kaggle |

The score degradation from 0.657 to 0.507 observed during blending experiments was traced to a **preprocessing mismatch** — adding pre-emphasis filtering and changing to zero-padding caused the model to receive inputs outside its training distribution. Reverting to the original preprocessing restored the baseline.

---

## Ablation Results

| Configuration | Score |
|---|---|
| ConvNeXt alone — original preprocessing, STEP=5.0 | **0.657** |
| ConvNeXt + Perch blend — wrong preprocessing, STEP=7.0 | 0.507 |
| ConvNeXt + veto only — wrong preprocessing, STEP=7.0 | 0.507 |
| ConvNeXt + veto only — original preprocessing, STEP=5.0 | 0.657 |
| Perch v2 alone | Pipeline error (XLA incompatibility) |

**Key finding:** Preprocessing consistency with the training distribution is the dominant factor in inference performance. The 0.15 score drop from preprocessing changes exceeded any gain from the ensemble.

---

## Environment Constraints

All Kaggle competition submissions run on CPU only (no GPU access). This imposed hard limits on model complexity and inference time (120-minute wall clock limit). Perch v8 requires GPU for practical inference; Perch v2 CPU variants have StableHLO compatibility issues with the current Kaggle TF 2.20.0 environment — a known issue in the competition community at the time of submission.

---

## Takeaways

- Specialist models fine-tuned on domain-specific data can match or outperform large pretrained generalist models under CPU inference constraints
- Inference environment control is a first-class concern in competition ML — model quality is irrelevant if the pipeline cannot finish within time limits
- Negative results (failed ensemble, preprocessing sensitivity) are as informative as positive ones for understanding model behaviour

---

*First-year undergraduate project — Manipal Institute of Technology, 2026*
