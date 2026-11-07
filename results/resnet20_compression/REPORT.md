# ResNet20 internal-channel compression

Status: completed; one seed2026.

## Hypothesis

At similar compact size, direct consumer-output fitting may retain more quality than shared-map folding after removing whole Q/scalar internal channels.

## Measured results

| Family | Fraction | Policy | Real trainable parameters | Validation before (%) | Validation after 3ep (%) |
|---|---:|---|---:|---:|---:|
| real | 0.25 | mapfold | 202738 | 86.20 | 89.20 |
| real | 0.25 | pruned | 202738 | 72.30 | 88.00 |
| real | 0.25 | direct | 202738 | 87.60 | 89.10 |
| real | 0.50 | mapfold | 135754 | 51.40 | 82.90 |
| real | 0.50 | pruned | 135754 | 25.10 | 83.00 |
| real | 0.50 | direct | 135754 | 70.70 | 84.90 |
| quaternion | 0.25 | mapfold | 204442 | 76.50 | 88.60 |
| quaternion | 0.25 | pruned | 204442 | 46.50 | 87.40 |
| quaternion | 0.25 | direct | 204442 | 84.10 | 88.30 |
| quaternion | 0.50 | mapfold | 137290 | 27.70 | 81.80 |
| quaternion | 0.50 | pruned | 137290 | 17.10 | 81.00 |
| quaternion | 0.50 | direct | 137290 | 53.30 | 82.90 |

## Interpretation and limits

Independent read-only audit PASSED: 146 frozen sources, 308 inputs, 12 cases,
24 actual checkpoint layouts/counts, four within-family matched-mask groups,
108 fold records (maximum error 7.63e-6) and 36 local direct-fit checks.
See independent_audit.json and scripts/review_resnet20_compression.py. This
audit did not independently replay all validation/check predictions; the
scientific runner separately reloaded and replayed each final validation metric.
All twelve repairs recorded 1125 successful / 1125 attempted AMP steps.

ONE source seed, BOTH model families, fractions25/50percent, THREE same-victim policies, maximum12 matched3ep repairs. Nine internal residual-block channels only, preserving residual outputs/shortcuts. Greedy activation-reconstruction-times-outgoing-energy selection and original ridge map fit on500 fit images, fixed victims per family/fraction; mapfold chain chooses masks, pruned/direct reuse exactly. Direct refits next convolution's pre-BN outputs against current partially compressed reference, Q Hamilton-constrained or unconstrained real, centered on the same mapfold kernel.500 separate check images, 1000 validation, no official-test inference. No new baselines, algebras, hyperparameter sweep, or automatic replication.

Per-site atomic compact checkpoint and records. STOP checked between sites and after repair batches; abrupt interruption repeats at most one site or partial epoch. Repair saves full optimizer/scaler/RNG per completed epoch. Final epoch3, no selection.

One discovery seed only: no significance, generalization, quaternion superiority or methodological novelty claim. Identical fractions and same masks within a family, not identical masks across families. Q units group four components while real units are scalar; report real parameters and baseline quality. The final real classifier and residual outputs/shortcuts are unchanged by pruning. 3ep supervised repair uses all surviving parameters; no test. Previous He2017ChannelPruning, Kim2020, VieiraValle2022ELM and Singh2026 remain prior art. No automatic replication or PHM run.
