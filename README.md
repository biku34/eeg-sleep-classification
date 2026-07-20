# Automated Sleep Stage Classification from Single-Channel EEG

**NETSIP 2026 · Track 2 (BCI Innovation Challenge) · Topic 6**

A sequence-aware deep-learning system that classifies every 30-second epoch of a
**single frontal EEG channel** into one of five sleep stages — **Wake, N1, N2, N3,
REM** — at an agreement level comparable to human sleep technicians.

> **Headline result:** Cohen's κ = **0.700** under subject-independent evaluation
> (human inter-scorer agreement ≈ 0.76), from a single Fpz-Cz electrode — the
> configuration realisable in consumer wearables. The model transfers to a
> *different* database (CAP) at κ **0.630 with no retraining**, and a causal,
> real-time demo reconstructs an unseen subject's whole night at **85%** accuracy.

---

## Results

All metrics are measured under **subject-independent** cross-validation (no person
appears in both training and test), so they reflect generalisation to new people —
not memorisation.

### Model ladder (Sleep-EDF, subject-wise 5-fold)

| Model | Accuracy | Macro-F1 | Cohen's κ |
|---|:---:|:---:|:---:|
| Spectral features + Random Forest (baseline) | 0.726 | 0.655 | 0.627 |
| 1-D CNN (learns features from raw signal) | 0.743 | 0.692 | 0.659 |
| **CNN + BiLSTM (temporal context)** | **0.777** | **0.728** | **0.700** |

### Per-class F1 (CNN + BiLSTM)

| Wake | N1 | N2 | N3 | REM |
|:---:|:---:|:---:|:---:|:---:|
| 0.893 | 0.474 | 0.805 | 0.705 | 0.766 |

N1 is the hardest, rarest stage and improves most as the models gain capability
(0.296 → 0.422 → 0.474 across the ladder). On single-channel EEG, N1 and REM both
present as theta activity, so N1↔REM confusion is an expected, documented limitation.

### Cross-database transfer (CAP Sleep Database, no retraining)

| | Accuracy | Macro-F1 | Cohen's κ |
|---|:---:|:---:|:---:|
| Sleep-EDF (in-domain) | 0.777 | 0.728 | 0.700 |
| **CAP (transfer)** | 0.727 | 0.662 | **0.630** |
| Drop | −0.050 | −0.066 | −0.070 |

Evaluated on 3,138 epochs from 3 unseen CAP subjects, using the F4-C4 derivation
(closest available to Fpz-Cz) resampled 512 → 100 Hz. The ~0.07 κ drop is graceful,
not catastrophic — evidence the model learned real sleep physiology rather than
dataset-specific quirks.

---

## Overview

Manual sleep staging is the bottleneck in polysomnography: a technician visually
scores ~960 epochs per night, ~2 hours of expert time. This project automates it
from a single frontal electrode, using a three-stage methodology:

1. **Interpretable baseline** — relative spectral band power (delta, theta, alpha,
   sigma, beta via Welch's method) + time-domain statistics, classified with a
   class-weighted Random Forest. Establishes a transparent reference and recovers
   physiologically meaningful biomarkers (e.g. sigma-band elevation for N2 spindles,
   delta dominance for N3).
2. **1-D CNN** — learns features directly from the raw 3000-sample epoch, with a wide
   first convolution kernel sized to capture sleep spindles and K-complexes.
3. **CNN + BiLSTM** — per-epoch CNN embeddings are passed through a bidirectional
   LSTM over sequences of consecutive epochs, converting isolated classification into
   sequence labelling. This encodes the inter-epoch context that AASM scoring rules
   explicitly depend on, and is the single largest source of improvement.

---

## Data

**Primary — Sleep-EDF Database Expanded v1.0.0, Sleep Cassette subset** (PhysioNet).
153 whole-night recordings from 78 healthy subjects, single-channel Fpz-Cz at 100 Hz.
Rechtschaffen & Kales stages 3 and 4 are merged into N3 (AASM); *Movement time* and
unscored epochs are discarded; each recording is cropped to 30 minutes either side of
the sleep period.

**External validation — CAP Sleep Database v1.0.0** (PhysioNet). Used only at test
time to measure cross-database robustness; the model is never trained on it.

Neither dataset is committed to the repo (see `.gitignore`). Download via the AWS
open mirror (fast, no credentials):

```bash
# Sleep-EDF (primary)
aws s3 sync --no-sign-request \
  s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ data/raw/sleep-cassette/

# CAP (a few normal subjects for the transfer test)
aws s3 cp --no-sign-request s3://physionet-open/capslpdb/1.0.0/n1.edf data/raw/cap/
aws s3 cp --no-sign-request s3://physionet-open/capslpdb/1.0.0/n1.txt data/raw/cap/
```

---

## Setup

Python 3.10/3.11.

```bash
pip install -r requirements.txt
```

Key dependencies: `mne` (EDF I/O + filtering), `numpy`, `scipy`, `scikit-learn`,
`torch`, `pandas`, `matplotlib`, `tqdm`. On Colab, only `mne` needs installing; do
**not** pin `numpy < 2.0` (it conflicts with Colab's pre-built packages — use
`numpy >= 2.0`).

---

## Reproducing the results

Run the pipeline stages in order (in `notebooks/sleep_pipeline.ipynb`, or the `src`
modules):

| Step | Produces |
|---|---|
| 1. Preprocess | `data/processed/*.npz` — one file of 30-s epochs per recording (~165k epochs) |
| 2. Baseline | Random Forest metrics + feature-importance biomarkers |
| 3. CNN | 1-D CNN metrics |
| 4. CNN + BiLSTM | Headline metrics + confusion matrix |
| 5. Leakage check | Demonstrates random-split accuracy inflation vs subject-split |
| 6. Demo | Causal hypnogram reconstruction of a held-out night (`results/demo_hypnogram.png`) |
| 7. CAP transfer | Cross-database metrics with the saved model |

Reproducibility notes:

- **All randomness is seeded** (`set_seed(42)` across Python, NumPy, PyTorch;
  deterministic cuDNN). The seed and every hyper-parameter live in
  `configs/config.yaml`.
- **Normalisation is per-epoch z-scoring** (each window standardised by its own
  statistics), so there is no train→test leakage of dataset statistics.
- **Epochs are exactly 3000 samples** (`tmax = 30 − 1/fs`).
- The reported κ uses subject-wise 5-fold cross-validation. For the exact
  leave-one-subject-out protocol in the brief, set the CV to `LeaveOneGroupOut`
  (slower; lands within ~0.01 κ of the 5-fold number).

---

## Evaluation protocol — why the numbers are honest

Adjacent 30-second epochs from the same night are near-identical. Splitting epochs
randomly leaks near-duplicates across the train/test boundary and inflates accuracy.
This project splits strictly **by subject** — every epoch from a person is on one
side of the split. An explicit experiment quantifies the inflation: a naive random
split reports higher accuracy/κ than the honest subject split on identical data.

Metrics reported: accuracy, macro-averaged F1, per-class F1, and Cohen's κ (the
standard sleep-medicine agreement metric), with full normalised confusion matrices.

---

## Limitations

- **Single channel by design.** The eye-movement (EOG) and chin-muscle (EMG)
  channels that separate N1 from REM are deliberately excluded to mimic wearables;
  consequently N1 is the weakest class and N1↔REM confusion is the dominant failure
  mode.
- **Healthy subjects.** Sleep-EDF Sleep Cassette contains healthy sleepers; the CAP
  transfer test probes robustness but generalisation to clinical populations
  (apnoea, narcolepsy) is not established here.
- **Montage sensitivity.** Cross-database performance drops mainly via Wake/REM → N1
  confusion, because CAP's F4-C4 derivation carries alpha/theta rhythms differently
  than Fpz-Cz.

---

## Acknowledgements & data citations

- Kemp B. et al. *Analysis of a sleep-dependent neuronal feedback loop: the
  slow-wave microcontinuity of the EEG.* Sleep-EDF Database Expanded, PhysioNet.
- Terzano M.G. et al. *Atlas, rules, and recording techniques for the scoring of
  cyclic alternating pattern (CAP) in human sleep.* CAP Sleep Database, PhysioNet.
- Goldberger A. et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000.

Implementation stack: Python, MNE-Python, NumPy, SciPy, scikit-learn, PyTorch.

## License

MIT — see [`LICENSE`](LICENSE).
