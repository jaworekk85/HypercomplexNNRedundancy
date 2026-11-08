# ResNet20 internal-channel compression

Status: completed; replication seed2027.

## Hypothesis

At similar compact size, direct consumer-output fitting may retain more quality than shared-map folding after removing whole Q/scalar internal channels.

## Measured results

| Family | Fraction | Policy | Real trainable parameters | Validation before (%) | Validation after 3ep (%) |
|---|---:|---|---:|---:|---:|
| real | 0.50 | mapfold | 135754 | 40.50 | 81.90 |
| real | 0.50 | pruned | 135754 | 19.50 | 81.00 |
| real | 0.50 | direct | 135754 | 65.20 | 82.70 |
| quaternion | 0.50 | mapfold | 137290 | 35.70 | 81.10 |
| quaternion | 0.50 | pruned | 137290 | 19.70 | 81.10 |
| quaternion | 0.50 | direct | 137290 | 56.20 | 81.50 |

## Interpretation and limits

Prespecified focused replication at 50% removal: both real and quaternion families, mapfold/pruned/direct policies, seeds 2027 and 2028, same fit/check/validation split and recipe as discovery. Report 2026 separately as discovery and both new seeds individually. This is not an independent data-split or dataset replication. No official-test evaluation, no post-outcome selection or hyperparameter sweep.

Per-site atomic compact checkpoint and records. STOP checked between sites and after repair batches; abrupt interruption repeats at most one site or partial epoch. Repair saves full optimizer/scaler/RNG per completed epoch. Final epoch3, no selection.

One independent replication seed in this per-seed report: no significance, generalization, quaternion superiority or methodological novelty claim. Identical fractions and same masks within a family, not identical masks across families. Q units group four components while real units are scalar; report real parameters and baseline quality. The final real classifier and residual outputs/shortcuts are unchanged by pruning. 3ep supervised repair uses all surviving parameters; no test. Previous He2017ChannelPruning, Kim2020, VieiraValle2022ELM and Singh2026 remain prior art. No automatic replication or PHM run.
