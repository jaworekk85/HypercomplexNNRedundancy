# Small RGB pilot

## Hypotheses

H1-H3 from PROJECT.md remain research hypotheses. This pilot checks image-data trainability and describes held-out activations; it does not test compression value.

## Frozen protocol

CIFAR-10 (Krizhevsky2009): 5000 stratified training, 1000 validation, 1000 calibration examples; all 10000 official test examples. Split seed 471; model seeds [2026, 2027, 2028].
RGB channels of one 32x32 pixel form one quaternion, with explicit zero alpha. ARGB follows the text of NiemczynowiczKycia2025; ABGR checks the channel order in the authors' notebook. See docs/INPUT_ENCODING_REVIEW.md. This is not a reproduction of their malaria CNN experiment.
Channel normalization is fit on the selected training images only. The real control receives the same RGB information without a padded zero channel or unused zero-input columns.
Each model is a single relu hidden layer and a real 10-logit readout. Adam, learning rate 0.001, batch size 128, 20 epochs; identical batch orders within each seed. Hidden weights have variance 2/(3*32*32), zero biases, and seeded GlorotUniform readouts.
Checkpoint selection minimizes validation cross-entropy, with initialization also eligible. Calibration is used for diagnostics only. All training/selection finished before the final test stage. Uncompiled checkpoints contain configuration and weights, not optimizer state.

## Measured results

Values below are means and sample standard deviations across seeds, not confidence intervals. Seeds share the same image splits.

| Model | Real trainable parameters | Hidden scalar features | Test accuracy (%) | Test cross-entropy | Selected epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| Quaternion (0,R,G,B) | 132490 | 128 | 37.78 ± 0.92 | 1.9169 ± 0.0197 | [8, 3, 4] |
| Quaternion (0,B,G,R) | 132490 | 128 | 37.25 ± 1.86 | 1.9119 ± 0.0183 | [6, 4, 3] |
| Real RGB | 132579 | 43 | 37.61 ± 1.12 | 1.8722 ± 0.0133 | [6, 4, 3] |

| Model | Effective rank, initial → selected | Constant scalar features, selected | Scalar pairs with correlation ≥0.9 (%) |
| --- | ---: | --- | ---: |
| Quaternion (0,R,G,B) | 38.36 → 20.61 | [0, 0, 0] | 0.008 |
| Quaternion (0,B,G,R) | 39.45 → 20.71 | [0, 0, 0] | 0.025 |
| Real RGB | 22.19 → 17.45 | [0, 0, 0] | 0.000 |

| Model | Maximum grouped cosine in each seed | Grouped pairs ≥0.9, mean fraction (%) | Pair-value correlation between calibration halves |
| --- | --- | ---: | --- |
| Quaternion (0,R,G,B) | 0.552, 0.562, 0.719 | 0.000 | 0.985, 0.987, 0.983 |
| Quaternion (0,B,G,R) | 0.560, 0.627, 0.722 | 0.000 | 0.980, 0.987, 0.984 |

Paired test accuracy differences (percentage points):

- ARGB_minus_RGB: +0.170 ± 1.293; per-seed differences [0.42, -1.23, 1.32].
- ABGR_minus_RGB: -0.357 ± 0.746; per-seed differences [0.17, -0.03, -1.21].
- ABGR_minus_ARGB: -0.527 ± 1.880; per-seed differences [-0.25, 1.2, -2.53].

All recorded first-step gradient/update checks and checkpoint roundtrips passed. Per-seed JSON files preserve kernel component norms and every check. No units were removed, no weights were masked, and actual compact-architecture savings are zero.

## Definitions

Scalar Pearson is computed on centered scalar hidden features. Grouped cosine centers each of the four components over calibration examples, then flattens the entire sample-by-component block per quaternion unit. Constant features/units are undefined and excluded. Histograms exclude diagonals and duplicate pairs. The threshold 0.9 was fixed before execution; it is a descriptive threshold, not a pruning rule.
Entropy effective rank is exp(-sum(p*log(p))) for the normalized eigenvalues p of the centered scalar-feature covariance. Normalized rank divides by min(number of examples minus one, number of scalar features). Raw eigenvalues and normalized ranks are in JSON.
Calibration-half stability compares matching pair values from even/odd calibration examples within the same model. It does not align individual neurons between independent training seeds. Similarity calculations follow the broad feature-redundancy motivation of Ayinde2019, with no new algorithm claimed.

## Interpretation and limits

In this run, none of the 6 trained quaternion models contained a calibration unit pair with centered grouped cosine at least 0.9. Maximum pair similarities ranged from 0.552 to 0.722. This is a negative observation for the chosen near-duplicate diagnostic, not a general negative result about QNN redundancy.

These are small dense models trained on only 10% of the official training set. The comparison matches real parameter budgets approximately; it does not match hidden width, compute, or attained validation quality. Consequently, accuracy differences and raw rank differences are descriptive pilot observations, not a demonstrated quaternion performance gain or a controlled compression result.
Both high scalar correlation and a low effective rank can coexist with modest whole-quaternion similarity. Neither establishes that a whole unit can be removed safely. An absent high-similarity signal under this metric does not refute H1 for other architectures, representations or datasets.
ARGB/ABGR changes are reported rather than used to select a winning test configuration. Three seeds describe initialization/shuffle variation on one fixed split; broader uncertainty and generalization remain untested. Pruning and Gates C-D remain open.

## Artifacts and provenance

comparison.csv and summary.json contain the numeric comparison; each seed directory holds its history, selected model, calibration activations, test logits and audit. split_indices.npz and data.json identify exact splits and training-only normalization. environment.freeze.txt and source hashes in summary.json record the software. Dataset archive SHA-256 is checked by the installed Keras loader.

Citations use references.bib: Krizhevsky2009, NiemczynowiczKycia2025, NiemczynowiczKycia2026, Ayinde2019. No source claims are inferred from an unexecuted notebook reproduction.

## Verification performed after execution

`python -m scripts.verify_rgb_pilot` passed **89 artifact checks across nine
runs**. It independently recomputed accuracy and cross-entropy from saved test
logits, counted actual HDF5 weight entries in each checkpoint, verified selection
of the minimum-validation-loss epoch, checked split disjointness and verified
the source, split and actual downloaded archive hashes. No retraining or new
model test inference was performed during this audit. See `verification.json`.

The executed pytest suite passed **26 tests**, including the offline image
training/checkpoint/activation integration check. Six deprecation warnings came
from the Keras/NumPy `__array__(copy=...)` serialization interface. Training also
logged a TensorFlow `reset_default_graph` deprecation warning through Keras
session cleanup. No numerical or model integrity checks failed. The separate
Phase 0 report continues to document the unmodified upstream tracking failure.
