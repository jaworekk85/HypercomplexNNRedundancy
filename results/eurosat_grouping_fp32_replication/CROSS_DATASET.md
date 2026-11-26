# Grouping replication across two datasets/architectures

Descriptive means across three source seeds; all individual values remain in per-study reports.

| Setting | Condition | Native accuracy | After 3 epochs | Final CE |
|---|---|---:|---:|---:|
| CIFAR-10 / ResNet-20 | original | 87.83% | 90.13% | 0.298386 |
| CIFAR-10 / ResNet-20 | role_100_0 | 79.63% | 88.60% | 0.345838 |
| CIFAR-10 / ResNet-20 | free_100_0 | 77.93% | 87.87% | 0.351334 |
| EuroSAT / CNN6 | original | 79.73% | 93.33% | 0.186688 |
| EuroSAT / CNN6 | role_100_0 | 58.43% | 86.57% | 0.385028 |
| EuroSAT / CNN6 | free_100_0 | 50.40% | 83.83% | 0.470136 |

## Actual budgets

- CIFAR-10 / ResNet-20: 232266compact/271594parent real trainables, 14.48%whole-model savings; attempted updates/model [1125]; successful updates [1125, 1125].
- EuroSAT / CNN6: 44794compact/73898parent real trainables, 39.38%whole-model savings; attempted updates/model [462]; successful updates [459, 461].

## Interpretation limits

Same retained scalar sets and compact sizes within each source seed/setting. Dataset, architecture, input encoding, whole-model savings and total repair update budgets differ across settings. Differences in gap magnitude do not isolate a dataset or quaternion-semantic effect. No pooled significance, equivalence, method novelty or blind-validation claim. AMP skips are retained; equal attempted updates need not mean equal successful updates.

[Figure](figures/grouping_replications.png). No official-test inference.
