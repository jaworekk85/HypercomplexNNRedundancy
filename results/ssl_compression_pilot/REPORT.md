# Low-label SSL and structure-preserving compression: bounded pilot

## Hypotheses and registered scope

Does compensation preserve the pseudo-label signal after compression, and improve subsequent SSL recovery? Do QNN and one matched circulant control show the same pattern? This is a SINGLE-SEED exploratory pilot, not replicated evidence, a new algorithm or a full FixMatch/SoftMatch reproduction (Sohn2020FixMatch, Chen2023SoftMatch, Ding2017CirCNN).

Fresh models: only 50 training labels/class from the FIRST update, source sibling soft pseudo-targets/epoch-lagged confidence weights, weak/strong views, joint train-mode BN. No supervised teacher/pretraining. Two baselines, 20x383 SGD steps, last checkpoint; no tuning or official-test inference. Input information/width/parameter budgets match; the encoder differs from the sibling's original 3-convolution/embedding CNN, so old test scores are not directly comparable.

ADAPTATION: reserve 500 fit and 500 check images from the previously unlabeled pool, never weight training. At the full scope: L=500, U=47,500, validation=1000, fit=500, check=500. Provider labels construct balanced L/validation; only L labels enter weight updates. Validation/check truth is diagnostic only. Constant source mean/std and separate sampler/augmentation streams; fixed diagnostic weak views.

Uniform 25% unit removal at all six sites leaves 42,034 actual real parameters vs 73,898. Same victims with/without compensation; no mask-only savings or extra inference maps. Uncompressed control receives the same continuation. Repair: 1000 opportunities, fresh SGD .003 and paired restarting sampler streams, inherit parent soft confidence statistics for SSL. CE-U control receives the SAME labeled/unlabeled views and joint BN, but zero pseudo-label loss; it is not a labeled-only baseline. No EMA/contrastive/triplet additions.

## Measured results

State: completed. Baselines complete: 2; continuations complete: 12. Tiny CPU checks are NOT scientific results.

| Family | Initialization | Continuation | Val before | Val after | Weak pseudo precision before/after (threshold .95) |
| --- | --- | --- | ---: | ---: | --- |
| quaternion | uncompressed | soft_ssl | 51.30% | 51.80% | 81.74% / 82.76% |
| quaternion | uncompressed | ce_u_control | 51.30% | 51.50% | 81.74% / 82.31% |
| quaternion | greedy_comp | soft_ssl | 43.30% | 50.10% | 68.97% / 94.92% |
| quaternion | greedy_comp | ce_u_control | 43.30% | 49.30% | 68.97% / 88.04% |
| quaternion | greedy_pruned | soft_ssl | 28.80% | 50.20% | 56.25% / 98.08% |
| quaternion | greedy_pruned | ce_u_control | 28.80% | 50.40% | 56.25% / 95.65% |
| circulant | uncompressed | soft_ssl | 53.70% | 53.30% | 85.37% / 83.74% |
| circulant | uncompressed | ce_u_control | 53.70% | 53.20% | 85.37% / 84.83% |
| circulant | greedy_comp | soft_ssl | 44.30% | 51.70% | 65.00% / 88.24% |
| circulant | greedy_comp | ce_u_control | 44.30% | 51.60% | 65.00% / 81.44% |
| circulant | greedy_pruned | soft_ssl | 32.70% | 51.50% | 68.42% / 89.66% |
| circulant | greedy_pruned | ce_u_control | 32.70% | 51.00% | 68.42% / 85.71% |

## Interpretation and decision limits

Inspect classification CE/accuracy together with pseudo-label coverage, precision, confidence weights and agreement/KL relative to the same family's SSL parent. Uncompressed continuing controls separate ordinary extra learning from recovery after damage. Report negative effects and collapse, not just the best condition. One seed, one label budget and one compression level cannot establish equivalence, robustness or a quaternion-specific mechanism. A small gain does not establish a new method.

This genuine low-label-from-initialization study is distinct from results/pseudolabel_repair, whose teacher was fully supervised. No transfer of its negative conclusion to the sibling's method. Decide whether SSL contributes a coherent empirical question to the paper only after all registered conditions finish; no draft commitment/title/email is automatically sent.
