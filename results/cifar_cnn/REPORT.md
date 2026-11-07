# CIFAR CNN quality gate

## Hypotheses and goal

Target: at least 80% CIFAR-10 test accuracy for a correctly trained quaternion CNN. The target is not a guaranteed result. H1-H3 about redundancy/compression remain open.

## Protocol

CIFAR-10 (Krizhevsky2009): 48,000 training images and the same disjoint 1000 validation/1000 calibration images as the Dense pilot. Use all 10,000 official test images only after both trainings and checkpoint choices finish. This official test set was already used by the earlier Dense pilot; it is not a newly collected blind benchmark.
One seed, 2026. Six 3x3 convolutions in three blocks, each Conv-BatchNorm-ReLU twice, MaxPool and scalar-channel Dropout2d. Global average pooling and a real ten-logit readout. Quaternion widths 8/16/32 (32/64/128 scalar channels); real widths 16/32/64. Ordinary per-component BatchNorm is used, with no claimed quaternion invariance.
Quaternion input is (0,R,G,B), component-blocked within every feature map. Kernels implement the standard right Hamilton product, following the tensor formulation motivation of NiemczynowiczKycia2026. Each connection has four learned real kernels; the expanded convolution matrix is derived, not independently trained. This is an existing model family, not a proposed novel architecture.
AdamW: lr 0.001, weight decay 0.0001, cosine schedule to 1e-05, 50 epochs, batch 128. Train-only random crop from four zero-padded pixels and horizontal flip. RGB statistics fit only on 48,000 training images. GPU mixed precision; the first-update audit uses float32. Dedicated augmentation and batch generators give the same sample order/augmentation stream across the two models.
Checkpoint selection minimizes clean validation cross-entropy. The calibration set is not used to optimize weights, choose epochs, or select the architecture. Last-epoch checkpoints include optimizer/scheduler/scaler and RNG states for resuming; selected inference checkpoints contain model weights and BatchNorm buffers.

## Measured results

| Model | Real trainable parameters | BN buffer elements | Best epoch | Selected train accuracy | Validation accuracy | Test accuracy | Test cross-entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quaternion | 73898 | 902 | 49 | 80.92% | 78.30% | 79.00% | 0.6080 |
| real | 72954 | 454 | 50 | 77.65% | 74.70% | 76.24% | 0.6870 |

| Model | Calibration tensor | Effective scalar rank (initial → selected) | Scalar pairs ≥0.9 | Grouped pairs ≥0.9 |
| --- | --- | ---: | ---: | ---: |
| quaternion | [1000, 16, 32, 4] | 21.91 → 33.07 | 0.000% | 0.000% |
| real | [1000, 16, 64] | 10.83 → 18.77 | 0.000% | not applicable |

Primary 80% target achieved: **False**.

## Checks and interpretation

All expected parameters, every quaternion convolution component's gradients/updates, and checkpoint output equivalence were checked before interpreting the run. Explicit audits and full shapes are in per-model JSON. Nontrainable BatchNorm running statistics are separated from real trainable parameters.
Similarity pools spatial positions and calibration examples. Stability comparisons split by disjoint whole-image halves before pooling, avoiding comparisons of different pixels from the same image as if they were held-out images. Constant features are excluded; 0.9 is a descriptive threshold rather than proof of removable channels.
This is a one-seed quality check. The real/quaternion budgets are close but not identical; compute and effective width differ. A higher accuracy than the Dense pilot cannot be attributed solely to architecture: training-set size, augmentation, optimization and framework also changed. No general quaternion performance gain is established. No units were pruned, no weights masked, and actual compact-architecture savings are zero.
## Reproducibility

config.json, data.json and split_indices.npz record choices and held-out indices. summary.json records source hashes, package versions, device and seed; environment.freeze.txt records the full GPU environment. Histories, selected/last checkpoints, calibration features and final test logits are saved per model.

## Resume and artifact verification

Training was explicitly paused by the user after epoch 11, then resumed from
epoch 12. Both models completed 50 epochs. The quaternion history retains the
earlier epochs, and each model records 18,750 optimizer update attempts. CUDA
mixed precision can skip individual optimizer steps on overflow, so these counts
are explicitly attempts rather than a claim that every attempt applied an update.

`python -m scripts.verify_cifar_cnn` passed **35 checks**. These include contiguous
epoch histories, update-attempt counts after resuming, actual saved parameter
and buffer counts, validation checkpoint selection, source hashes, and metrics
recomputed from saved test logits. No extra training or test inference was used
for this audit. See `verification.json`; pause/resume events are preserved in
`pause.json` and `resume.json`.

The prior suite of ten layer, gradient, serialization and interrupted-resume
tests passed in the same PyTorch 2.6.0+cu124 environment; see `pytest.xml`.
The 80% target was missed by one percentage point in the quaternion run.
