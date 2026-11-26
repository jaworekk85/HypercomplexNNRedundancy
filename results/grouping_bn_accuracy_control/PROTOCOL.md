# BN-only accuracy control

## Hypothesis

Recalibrating only BN running statistics may reduce the actual full-network accuracy penalty of mixing quaternion groups, without changing any trainable parameter.

## Scope

Reconstruct all15predetermined compact CIFAR10 Q ResNet20 models from frozen source parents and saved physical blocks. Add3uncompressed parent controls. No regrouping search, new supervised training, new algebra or official-test inference. All compact models have232266real trainables versus271594parents; no masks or runtime feature routers. Existing3epoch repair results are historical comparators, not training started from the newly recalibrated model.

## Data

Use1024unlabelled TRAIN images chosen with CPU generator seed340000, the same subset recipe as the completed statistics control. No augmentation; same subset/order for every source seed and model. Record absolute train indices and their hash. No labels are used in BN recalibration. Existing1000validation and500odd-calibration CHECK examples are evaluation-only, disjoint from BN images. Both splits have been inspected previously; this is an exploratory paired intervention, not fresh blind confirmation.

## Intervention

Reuse the frozen sequential eval-prefix BN population recalibration implementation. Every BN layer receives exact per-channel float64 population moments from the1024train images using the already recalibrated prefix; copy only running_mean/running_var/num_batches_tracked. Preserve all parameters including BN affine, all non-BN buffers and original source artifacts. No optimizer or feature standardization. Native assembled models must replay predecessor before-repair metrics and logits. Save and reload calibrated weights; logits must replay exactly.

## Primary

For each of4mixed cases and each source seed, compute the validation-accuracy gap original-minus-mixed before and after BN-only recalibration. Primary paired contrast in percentage points is native_gap minus bn_gap: positive means BN reduces the mixing penalty. Report all12comparisons and descriptive means across3source seeds, plus individual accuracy/CE for every model. No success threshold, significance/equivalence claim or pooling images as model seeds.

## Secondary

Report corresponding CHECK accuracy gaps, validation/CHECK CE and raw BN gains, parent native-to-BN changes and historical final epoch3 repair accuracy/CE. Compare relative gaps after historical repair separately; it includes changes in weights and BN statistics. Do not attribute all historical repair gain to weights, or all remaining native-to-BN difference to intrinsic quaternion semantics. No hyperparameter or successful-case selection.

## Checks

Before source freeze: inherited frozen BN unit tests, new reconstruction/contrast/checkpoint checks, synthetic end-to-end run and completed resume, plus discarded-copy compact-GPU tracked-variable/all-quaternion-component gradient/update audit. After intervention: exact trainable and non-BN buffer immutability, native predecessor replay, saved-state/logit replay, final source/input and output hash audit. Record failures accurately and add new modules if a frozen scientific source needs repair.

## Resume

Hash-check completed model results/checkpoints/logits and reuse them. STOP checked between models, BN batches/layers and evaluation splits. Interrupted calibration repeats at most one model; outputs atomic. No further experiment is queued after these18models.

Prior art: He2016ResNet, Vecchi2020, Shomron2020BNRecalibration, Jordan2023REPAIR; canonical references.bib.
