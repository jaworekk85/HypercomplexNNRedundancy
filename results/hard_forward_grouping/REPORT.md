# Discrete-forward learned grouping: one-seed pilot

## Hypotheses

Using a discrete permutation throughout training may avoid the soft-to-hard operator gap. Whether learned group membership improves held-out quality over paired fixed random membership is unconfirmed.

## Protocol

Seed 2026; unchanged CIFAR backbone, split, normalization, augmentation, optimizer and 50-epoch schedule. Exact matching of the routing logits at every adaptive forward; Sinkhorn straight-through surrogate in backward. This is a biased surrogate, not the true derivative of the discrete assignment. Learn assignments for 40 epochs, freeze for 10. Choose minimum original-BN validation CE in epochs 40-50. Reuse the paired fixed-random control's completed 50 epochs, selected by the same rule. Refit the same four downstream BNs, layer-wise on 48,000 training images, in both selected models. Validation/calibration each have 1,000 held-out images; no new official-test inference.

## Measured results

| Condition | Validation accuracy | Validation CE | Calibration accuracy | Calibration CE |
| --- | ---: | ---: | ---: | ---: |
| random_original | 78.00% | 0.622695 | 76.80% | 0.638498 |
| learned_original | 78.10% | 0.653201 | 76.70% | 0.653901 |
| random_recalibrated | 78.10% | 0.626831 | 77.20% | 0.632758 |
| learned_recalibrated | 77.80% | 0.662828 | 77.30% | 0.659881 |

Selected learned checkpoint: epoch 46. Final exported assignment changes 6/32 positions and creates 4/8 new unordered groups.
BN-refitted learned-minus-random effect: validation -0.30 pp; calibration +0.10 pp. Prespecified replication gate: False.

## Interpretation and limits

This tests one deterministic surrogate, seed, schedule and initialization. An unchanged final assignment is not evidence of useful learned membership. A positive gate only motivates replication; it is not statistical significance, a general gain, a novel method, or a compression result. A negative outcome does not rule out all grouping methods. Previously explored held-out data are not a new blind benchmark.

The learned training adds 1,024 real logit parameters: 74,922 vs 73,898. It also adds matching, Sinkhorn and surrogate-backward compute. Final fixed-index exports both have 73,898 real parameters and 934 buffer elements. No backbone channel is removed.

The unmodified trainer's legacy 'soft' keys mean adaptive DISCRETE routing in this run. Canonical phases and freeze metrics are in learned/training_discrete.csv and learned/freeze.json. Both complete freeze-boundary checkpoints and all resume RNG/optimizer states are retained.

Prior art: Bengio2013STE, Mena2018, Wang2023PSTE, Wang2019FLGC, Zhang2021StructuredShuffle in references.bib. This is not a faithful reproduction of those complete methods.

## Verified interpretation and decision

The selected exported assignment changes 6/32 scalar positions and creates 4/8 new unordered groups. Its BN-refitted learned-minus-fixed effect is -0.30 pp on validation and +0.10 pp on calibration.

The prespecified replication gate returns False: calibration gain False, validation quality True, actual membership change True. A positive gate only motivates replication; a negative gate preserves this unsuccessful configuration. Neither is a general method result or a novelty claim.

In this completed pilot, the primary calibration difference is one additional correct image out of 1,000, while validation has three fewer correct images. CE is worse for learned grouping on both held-out splits, before and after BN refitting. This does not demonstrate a useful learned-grouping advantage or establish equivalence; preserve the near-tied accuracy and unfavorable CE rather than claim a gain. The gate does not justify automatically launching replication or compression of this configuration.

The freeze has zero immediate prediction shift because routing was already discrete. This verifies the intended mechanism; it is not evidence of a learned-grouping advantage. Training adds 1,024 logit parameters and matching/surrogate compute; both fixed-index exports have 73,898 real parameters with no removed backbone channels.

Six implementation tests and 131 independent artifact checks pass. Numerical sources and paired reference artifacts are preserved. Prior context: Bengio2013STE, Mena2018, Wang2023PSTE, Wang2019FLGC and Zhang2021StructuredShuffle. See results/hard_forward_grouping/REPORT.md and configs/hard_forward_grouping.json.
