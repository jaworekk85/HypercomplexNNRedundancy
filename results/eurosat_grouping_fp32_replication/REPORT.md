# EuroSAT physical grouping replication

Status: completed

## Hypothesis

Original quaternion grouping may retain better whole-network quality than predetermined role-preserving/free fully mixed assignments after matched physical compression and three-epoch repair on EuroSAT CNN6.

## Measured full-network results

| Seed | Case | Actual trainables | Native val | Final epoch3 val | Final val CE |
|---|---|---:|---:|---:|---:|
| 2026 | free_100_0 | 44794 | 34.00% | 83.80% | 0.459331 |
| 2026 | original | 44794 | 85.20% | 93.60% | 0.186678 |
| 2026 | role_100_0 | 44794 | 57.40% | 86.30% | 0.370882 |
| 2027 | free_100_0 | 44794 | 61.90% | 82.90% | 0.469785 |
| 2027 | original | 44794 | 79.40% | 92.60% | 0.200899 |
| 2027 | role_100_0 | 44794 | 72.10% | 85.40% | 0.406264 |
| 2028 | free_100_0 | 44794 | 55.30% | 84.80% | 0.481293 |
| 2028 | original | 44794 | 74.60% | 93.80% | 0.172489 |
| 2028 | role_100_0 | 44794 | 45.80% | 88.00% | 0.377938 |

## Primary paired gaps

Positive accuracy gap favors original grouping; CE delta is mixed minus original.

| Seed | Case | Native val gap pp | Final val gap pp | Final CE delta |
|---|---|---:|---:|---:|
| 2026 | role_100_0 | 27.80 | 7.30 | 0.184204 |
| 2026 | free_100_0 | 51.20 | 9.80 | 0.272653 |
| 2027 | role_100_0 | 7.30 | 7.20 | 0.205365 |
| 2027 | free_100_0 | 17.50 | 9.70 | 0.268886 |
| 2028 | role_100_0 | 28.80 | 5.80 | 0.205449 |
| 2028 | free_100_0 | 19.30 | 9.00 | 0.308804 |

## Measured local results

| Seed | Block | Case | Exact-feature check MSE (fresh FP32) | Physical check MSE | Producer projection SSE |
|---|---|---|---:|---:|---:|
| 2026 | 0 | free_100_0 | 0.342825 | 0.370718 | 23.1576 |
| 2026 | 0 | original | 0.185911 | 0.185911 | 0 |
| 2026 | 0 | role_100_0 | 0.320053 | 0.358615 | 18.46 |
| 2026 | 2 | free_100_0 | 0.770159 | 0.861882 | 165.819 |
| 2026 | 2 | original | 0.391427 | 0.391427 | 0 |
| 2026 | 2 | role_100_0 | 0.744092 | 0.821225 | 155.744 |
| 2027 | 0 | free_100_0 | 0.414998 | 0.336063 | 33.5236 |
| 2027 | 0 | original | 0.207807 | 0.207807 | 0 |
| 2027 | 0 | role_100_0 | 0.360038 | 0.31471 | 21.9473 |
| 2027 | 2 | free_100_0 | 0.800055 | 1.00973 | 170.846 |
| 2027 | 2 | original | 0.347869 | 0.347869 | 0 |
| 2027 | 2 | role_100_0 | 0.770862 | 0.900329 | 169.656 |
| 2028 | 0 | free_100_0 | 0.327059 | 0.274656 | 23.415 |
| 2028 | 0 | original | 0.189192 | 0.189192 | 0 |
| 2028 | 0 | role_100_0 | 0.322659 | 0.349269 | 20.1193 |
| 2028 | 2 | free_100_0 | 0.747723 | 0.893315 | 169.395 |
| 2028 | 2 | original | 0.339408 | 0.339408 | 0 |
| 2028 | 2 | role_100_0 | 0.729423 | 0.769698 | 157.698 |

## Interpretation and limits

Fresh deterministic FP32 references and18new exact-feature fits, followed by the original18physical pair fits/nine-model design. Three existing four-band EuroSAT quaternion CNN6 parents, three conditions,9actual compact models and27repair epochs. Halve only the hidden quaternion widths between the two convolutions of blocks0/2, leaving block outputs, pooling/dropout, other block and real readout unchanged. All models have44794real trainables versus73898parents, about39.4percent whole-model savings; this is not the14.5percent CIFAR budget. No new parent, alternative algebra/dimension, group search, BN sweep or official-test evaluation.

Same existing TRAIN/validation/FIT/CHECK split and nine-model design as the base protocol, no test inference. Copy only the frozen original/role_100_0/free_100_0 assignments/kept scalar sets from the completed statistics control; these masks were originally derived with its archived precision. Collect fresh native features and targets for ALL conditions in deterministic FP32, plus fresh exact-feature fits, before physical fitting. The masks are not reselected. Previously explored validation/calibration data remain exploratory.

Combine both compact pairs into each whole model, measure validation/CHECK accuracy and CE before repair, then repair all surviving parameters with frozen AdamW/AMP augmentation recipe for final epoch3. Same order/augmentation seeds within each source seed; record dropout/RNG and successful/attempted updates through the existing repair function.19600train images give154batches per epoch and462attempted updates per model, unlike1125in CIFAR; equal epochs/recipe across datasets are not equal update/compute budgets. Save native/final models and per-example logits/labels, then verify reload predictions and metrics. No best-epoch or condition selection.

This is practical replication on a second dataset AND architecture/input encoding. It does not isolate a dataset effect, prove semantic necessity, global optimality, equivalence, method novelty or conference readiness. Earlier failed adaptation/reassembly/exchange gates remain failed. Do not tune this bounded study or add repairs after observing outcomes.

## Precision correction

The first registered implementation stopped before any supervised repair after3early-site physical fits: late archived EuroSAT activations failed bitwise replay. A four-mode diagnostic shows archived features/targets replay exactly only with cuDNN TF32 enabled, regardless of deterministic-algorithm setting. Preserve the failed output/source/config and its logs. This new version uses a separate output and new modules, retains all old masks/predetermined draws, and recomputes common FP32 targets/ridge scales for all conditions. No outcome-dependent method/group/budget selection.

Exact-feature rows were newly fitted once in this study and then reused by the physical-fit stage. The base local-row field ideal_reused describes that within-study reuse, not reuse of historical TF32 errors.
