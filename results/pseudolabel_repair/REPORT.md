# Low-label POST-compression repair

## Hypotheses and scope

Does compensation improve recovery when post-compression repair labels are scarce? Can current-student pseudo-labels help, or reinforce compression errors? No novel algorithm is proposed.

The original teacher was fully supervised and validation-selected. Labels per class count ONLY labels exposed to repair updates. Diagnostic validation/check labels and inherited normalization/BN statistics are outside that budget. This is not end-to-end few-shot or unsupervised training. Primary budgets: 5/20; zero is a boundary control. Same-distribution CIFAR simulation; domain shift and an actual deployment are untested.

Same frozen victims and physical 42,034-parameter architecture, with/without compensation, no fitting again. Online self-pseudo-labels come from the current compact model; the frozen full teacher is an additional-resource control. Confidence filtering is established (Sohn2020FixMatch), but crop/flip-only views are not a FixMatch reproduction or the student's exact pipeline.

Nested balanced labeled subsets and one disjoint 4096-image unlabeled pool per seed; three paired seeds, not many independent dataset replications. Fixed 1000 opportunities, final-step outcome, snapshots 0/200/1000; AdamW 1e-4 cosine to 1e-6, confidence .95. BN buffers and dropout fixed; BN affine remains trainable. Label/unlabeled batches and augmentation streams paired across policies. A batch with no usable target produces no optimizer step (including no weight decay). Teacher inference is an extra cost, not included in the count of update opportunities.

These repair settings differ from earlier full-label three-epoch train-mode fine-tuning; no direct label-efficiency claim follows from that cross-study comparison.

## Measured results

State: completed; completed runs: 54. Smoke checks are infrastructure checks only.

| Repair labels/class | Method | Initialization | Validation mean | Per-seed actual updates |
| ---: | --- | --- | ---: | --- |
| 0 | supervised | greedy_pruned | 23.47% | [0, 0, 0] |
| 0 | supervised | greedy_comp | 59.67% | [0, 0, 0] |
| 0 | self_pseudo | greedy_pruned | 10.00% | [1000, 1000, 924] |
| 0 | self_pseudo | greedy_comp | 10.00% | [1000, 1000, 1000] |
| 0 | teacher_pseudo | greedy_pruned | 63.90% | [1000, 1000, 1000] |
| 0 | teacher_pseudo | greedy_comp | 67.30% | [1000, 1000, 1000] |
| 5 | supervised | greedy_pruned | 51.17% | [1000, 1000, 1000] |
| 5 | supervised | greedy_comp | 55.60% | [1000, 1000, 1000] |
| 5 | self_pseudo | greedy_pruned | 49.10% | [1000, 1000, 1000] |
| 5 | self_pseudo | greedy_comp | 49.40% | [1000, 1000, 1000] |
| 5 | teacher_pseudo | greedy_pruned | 61.60% | [1000, 1000, 1000] |
| 5 | teacher_pseudo | greedy_comp | 64.70% | [1000, 1000, 1000] |
| 20 | supervised | greedy_pruned | 62.43% | [1000, 1000, 1000] |
| 20 | supervised | greedy_comp | 64.50% | [1000, 1000, 1000] |
| 20 | self_pseudo | greedy_pruned | 62.73% | [1000, 1000, 1000] |
| 20 | self_pseudo | greedy_comp | 64.63% | [1000, 1000, 1000] |
| 20 | teacher_pseudo | greedy_pruned | 66.03% | [1000, 1000, 1000] |
| 20 | teacher_pseudo | greedy_comp | 66.43% | [1000, 1000, 1000] |

## Interpretation limits

Keep all seeds, unfavorable effects and zero-acceptance/no-update runs. Compare final accuracy and CE with each initialization's step-zero values. Paired differences are descriptive with three seeds; no claim of superiority or deployment value follows merely from running this protocol. Full per-seed histories, pseudo-label coverage/precision and prediction files accompany the summary. No new official-test inference.
