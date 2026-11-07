# Multi-source compensation with and without the Hamilton restriction

Status: completed pilot. No fine-tuning, BN refit or official-test inference. Configuration: `configs/multisource_compensation.json`. Unconstrained multi-source ridge reconstruction folded into the next layer is prior art (GRAIL2026; also Asiaee2026CMR, Kim2020). Under study is only the right-Hamilton restriction q_v ~= sum_k q_k*a_k, which folds exactly as w_k <- w_k + a_k*w_v and keeps a quaternion consumer.

## Protocol

Block 2 output, three training seeds per architecture. Fit on 500 even calibration images, check on 500 odd ones; validation is the primary held-out split. Ridge alpha fixed a priori at 0.001 (lambda = alpha * mean Gram diagonal). Quaternion selected victims are the frozen nested pairwise victims from results/internal_replication, so multi_q and quaternion_exact remove identical units. Random rows: ten victim sets per seed, averaged within seed before the seed SD. multi_*_random applies the multi-source fit to those same random victim sets. multi_full_substitution is a functional oracle (unconstrained GRAIL-style substitution); folding it would break the Hamilton consumer, so it has no compact-quaternion budget. Real CNN6 control: 8/32 and 16/32 block-2 channels, victims by the scalar analogue of the pairwise rule.

## Validation results

| Architecture | Removed | Policy | Real parameters | Validation mean ± seed SD | Mean CE | Check mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| cnn6 | 4 | baseline | 73898 | 77.93% ± 1.10 pp | 0.6377 | 79.00% |
| cnn6 | 4 | random | 66954 | 73.22% ± 0.65 pp | 0.7577 | 75.13% |
| cnn6 | 4 | paired_pruned | 66954 | 71.40% ± 2.63 pp | 0.8286 | 71.40% |
| cnn6 | 4 | quaternion_exact | 66954 | 74.73% ± 0.83 pp | 0.7199 | 75.07% |
| cnn6 | 4 | multi_q_random | 66954 | 75.99% ± 1.10 pp | 0.6869 | 76.73% |
| cnn6 | 4 | multi_full_projected | 66954 | 74.70% ± 0.61 pp | 0.7263 | 76.07% |
| cnn6 | 4 | multi_q | 66954 | 76.63% ± 1.02 pp | 0.6775 | 77.00% |
| cnn6 | 4 | multi_full_substitution | functional oracle | 77.07% ± 1.10 pp | 0.6692 | 77.20% |
| cnn6 | 8 | baseline | 73898 | 77.93% ± 1.10 pp | 0.6377 | 79.00% |
| cnn6 | 8 | random | 60010 | 53.97% ± 2.60 pp | 1.3068 | 56.36% |
| cnn6 | 8 | paired_pruned | 60010 | 44.53% ± 6.86 pp | 1.6944 | 48.67% |
| cnn6 | 8 | quaternion_exact | 60010 | 65.20% ± 2.41 pp | 0.9942 | 67.13% |
| cnn6 | 8 | multi_q_random | 60010 | 71.86% ± 0.53 pp | 0.8155 | 71.63% |
| cnn6 | 8 | multi_full_projected | 60010 | 69.40% ± 1.78 pp | 0.8905 | 69.27% |
| cnn6 | 8 | multi_q | 60010 | 73.17% ± 0.85 pp | 0.7915 | 73.13% |
| cnn6 | 8 | multi_full_substitution | functional oracle | 74.33% ± 0.71 pp | 0.7434 | 74.33% |
| cnn8 | 4 | baseline | 118058 | 80.30% ± 0.96 pp | 0.5805 | 82.87% |
| cnn8 | 4 | random | 112266 | 77.00% ± 1.19 pp | 0.6595 | 79.52% |
| cnn8 | 4 | paired_pruned | 112266 | 75.30% ± 0.36 pp | 0.6960 | 79.33% |
| cnn8 | 4 | quaternion_exact | 112266 | 78.27% ± 1.45 pp | 0.6287 | 81.20% |
| cnn8 | 4 | multi_q_random | 112266 | 79.05% ± 0.57 pp | 0.6130 | 80.77% |
| cnn8 | 4 | multi_full_projected | 112266 | 78.50% ± 1.25 pp | 0.6256 | 80.87% |
| cnn8 | 4 | multi_q | 112266 | 79.53% ± 0.87 pp | 0.6078 | 81.13% |
| cnn8 | 4 | multi_full_substitution | functional oracle | 79.80% ± 0.40 pp | 0.5984 | 82.00% |
| cnn8 | 8 | baseline | 118058 | 80.30% ± 0.96 pp | 0.5805 | 82.87% |
| cnn8 | 8 | random | 106474 | 65.84% ± 4.85 pp | 0.9748 | 68.55% |
| cnn8 | 8 | paired_pruned | 106474 | 61.67% ± 6.56 pp | 1.1321 | 63.20% |
| cnn8 | 8 | quaternion_exact | 106474 | 73.43% ± 1.19 pp | 0.7778 | 75.07% |
| cnn8 | 8 | multi_q_random | 106474 | 76.14% ± 1.12 pp | 0.6993 | 76.91% |
| cnn8 | 8 | multi_full_projected | 106474 | 74.00% ± 1.11 pp | 0.7454 | 76.13% |
| cnn8 | 8 | multi_q | 106474 | 77.13% ± 0.72 pp | 0.6759 | 78.00% |
| cnn8 | 8 | multi_full_substitution | functional oracle | 78.23% ± 1.16 pp | 0.6403 | 79.47% |
| real_cnn6 | 8 | baseline | 72954 | 75.53% ± 1.27 pp | 0.6928 | 76.00% |
| real_cnn6 | 8 | random | 66026 | 70.60% ± 1.71 pp | 0.8147 | 71.49% |
| real_cnn6 | 8 | paired_pruned | 66026 | 71.67% ± 1.21 pp | 0.7923 | 72.00% |
| real_cnn6 | 8 | pair_merge | 66026 | 73.70% ± 1.51 pp | 0.7433 | 73.13% |
| real_cnn6 | 8 | multi_real_random | 66026 | 73.27% ± 0.78 pp | 0.7575 | 73.07% |
| real_cnn6 | 8 | multi_real | 66026 | 74.37% ± 0.97 pp | 0.7302 | 74.00% |
| real_cnn6 | 16 | baseline | 72954 | 75.53% ± 1.27 pp | 0.6928 | 76.00% |
| real_cnn6 | 16 | random | 59098 | 57.14% ± 2.87 pp | 1.1954 | 58.25% |
| real_cnn6 | 16 | paired_pruned | 59098 | 59.93% ± 1.97 pp | 1.1093 | 61.33% |
| real_cnn6 | 16 | pair_merge | 59098 | 67.47% ± 4.01 pp | 0.9189 | 66.67% |
| real_cnn6 | 16 | multi_real_random | 59098 | 66.80% ± 0.90 pp | 0.9390 | 66.39% |
| real_cnn6 | 16 | multi_real | 59098 | 70.37% ± 0.55 pp | 0.8524 | 69.00% |

## Paired effects (left minus right), all seeds retained

| Architecture | Removed | Comparison | Validation per seed (pp) | Mean (pp) | Check mean (pp) | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| cnn6 | 4 | multi_q - quaternion_exact | +1.20, +0.50, +4.00 | +1.90 | +1.93 | 3/3 |
| cnn6 | 4 | multi_q - multi_full_projected | +1.80, +1.60, +2.40 | +1.93 | +0.93 | 3/3 |
| cnn6 | 4 | multi_full_substitution - multi_q | +0.50, +0.30, +0.50 | +0.43 | +0.20 | 3/3 |
| cnn6 | 4 | multi_q - random | +2.25, +2.89, +5.10 | +3.41 | +1.87 | 3/3 |
| cnn6 | 4 | multi_q_random - random | +3.00, +1.78, +3.53 | +2.77 | +1.60 | 3/3 |
| cnn6 | 4 | multi_q - multi_q_random | -0.75, +1.11, +1.57 | +0.64 | +0.27 | 2/3 |
| cnn6 | 4 | quaternion_exact - random | +1.05, +2.39, +1.10 | +1.51 | -0.07 | 3/3 |
| cnn6 | 8 | multi_q - quaternion_exact | +8.10, +6.30, +9.50 | +7.97 | +6.00 | 3/3 |
| cnn6 | 8 | multi_q - multi_full_projected | +2.70, +6.40, +2.20 | +3.77 | +3.87 | 3/3 |
| cnn6 | 8 | multi_full_substitution - multi_q | +1.60, +0.40, +1.50 | +1.17 | +1.20 | 3/3 |
| cnn6 | 8 | multi_q - random | +16.54, +21.15, +19.91 | +19.20 | +16.77 | 3/3 |
| cnn6 | 8 | multi_q_random - random | +15.42, +19.23, +19.03 | +17.89 | +15.27 | 3/3 |
| cnn6 | 8 | multi_q - multi_q_random | +1.12, +1.92, +0.88 | +1.31 | +1.51 | 3/3 |
| cnn6 | 8 | quaternion_exact - random | +8.44, +14.85, +10.41 | +11.23 | +10.77 | 3/3 |
| cnn8 | 4 | multi_q - quaternion_exact | +0.80, +2.00, +1.00 | +1.27 | -0.07 | 3/3 |
| cnn8 | 4 | multi_q - multi_full_projected | +0.70, +1.50, +0.90 | +1.03 | +0.27 | 3/3 |
| cnn8 | 4 | multi_full_substitution - multi_q | -0.30, +1.00, +0.10 | +0.27 | +0.87 | 2/3 |
| cnn8 | 4 | multi_q - random | +2.24, +1.95, +3.40 | +2.53 | +1.61 | 3/3 |
| cnn8 | 4 | multi_q_random - random | +1.44, +1.96, +2.74 | +2.05 | +1.25 | 3/3 |
| cnn8 | 4 | multi_q - multi_q_random | +0.80, -0.01, +0.66 | +0.48 | +0.37 | 2/3 |
| cnn8 | 4 | quaternion_exact - random | +1.44, -0.05, +2.40 | +1.26 | +1.68 | 2/3 |
| cnn8 | 8 | multi_q - quaternion_exact | +3.20, +4.20, +3.70 | +3.70 | +2.93 | 3/3 |
| cnn8 | 8 | multi_q - multi_full_projected | +4.80, +1.30, +3.30 | +3.13 | +1.87 | 3/3 |
| cnn8 | 8 | multi_full_substitution - multi_q | +1.20, +2.70, -0.60 | +1.10 | +1.47 | 2/3 |
| cnn8 | 8 | multi_q - random | +9.20, +7.42, +17.25 | +11.29 | +9.45 | 3/3 |
| cnn8 | 8 | multi_q_random - random | +9.03, +6.63, +15.23 | +10.30 | +8.36 | 3/3 |
| cnn8 | 8 | multi_q - multi_q_random | +0.17, +0.79, +2.02 | +0.99 | +1.09 | 3/3 |
| cnn8 | 8 | quaternion_exact - random | +6.00, +3.22, +13.55 | +7.59 | +6.52 | 3/3 |
| real_cnn6 | 8 | multi_real - pair_merge | +1.20, -0.50, +1.30 | +0.67 | +0.87 | 2/3 |
| real_cnn6 | 8 | multi_real - random | +4.35, +2.24, +4.72 | +3.77 | +2.51 | 3/3 |
| real_cnn6 | 8 | multi_real_random - random | +3.46, +1.59, +2.96 | +2.67 | +1.58 | 3/3 |
| real_cnn6 | 8 | multi_real - multi_real_random | +0.89, +0.65, +1.76 | +1.10 | +0.93 | 3/3 |
| real_cnn6 | 8 | pair_merge - random | +3.15, +2.74, +3.42 | +3.10 | +1.64 | 3/3 |
| real_cnn6 | 16 | multi_real - pair_merge | +5.90, -1.40, +4.20 | +2.90 | +2.33 | 2/3 |
| real_cnn6 | 16 | multi_real - random | +15.46, +10.33, +13.88 | +13.22 | +10.75 | 3/3 |
| real_cnn6 | 16 | multi_real_random - random | +11.44, +7.40, +10.13 | +9.66 | +8.14 | 3/3 |
| real_cnn6 | 16 | multi_real - multi_real_random | +4.02, +2.93, +3.75 | +3.57 | +2.61 | 3/3 |
| real_cnn6 | 16 | pair_merge - random | +9.56, +11.73, +9.68 | +10.32 | +8.41 | 3/3 |

## Structural diagnostics

| Architecture | Removed | Seed | Discarded energy of unconstrained additions after projection |
| --- | ---: | ---: | ---: |
| cnn6 | 4 | 2026 | 49.20% |
| cnn6 | 8 | 2026 | 55.07% |
| cnn6 | 4 | 2027 | 58.30% |
| cnn6 | 8 | 2027 | 55.20% |
| cnn6 | 4 | 2028 | 53.45% |
| cnn6 | 8 | 2028 | 52.12% |
| cnn8 | 4 | 2026 | 43.27% |
| cnn8 | 8 | 2026 | 48.03% |
| cnn8 | 4 | 2027 | 47.08% |
| cnn8 | 8 | 2027 | 45.89% |
| cnn8 | 4 | 2028 | 48.25% |
| cnn8 | 8 | 2028 | 49.39% |

## Matched-savings comparison across families (CNN6 only)

Quaternion 4/16 and real 8/32 removals save almost the same number of real parameters (likewise 8/16 vs 16/32). The families start from different baselines, widths and compute, so only drops from each family's own baseline are compared. This is descriptive, not a causal claim about quaternion algebra.

| Removal pair | Policy pair | Quaternion drop (pp) | Real drop (pp) |
| --- | --- | ---: | ---: |
| 4/16 vs 8/32 (66954 vs 66026 params) | random vs random | -4.71 | -4.94 |
| 4/16 vs 8/32 (66954 vs 66026 params) | paired_pruned vs paired_pruned | -6.53 | -3.87 |
| 4/16 vs 8/32 (66954 vs 66026 params) | quaternion_exact vs pair_merge | -3.20 | -1.83 |
| 4/16 vs 8/32 (66954 vs 66026 params) | multi_q vs multi_real | -1.30 | -1.17 |
| 4/16 vs 8/32 (66954 vs 66026 params) | multi_q_random vs multi_real_random | -1.94 | -2.27 |
| 8/16 vs 16/32 (60010 vs 59098 params) | random vs random | -23.97 | -18.39 |
| 8/16 vs 16/32 (60010 vs 59098 params) | paired_pruned vs paired_pruned | -33.40 | -15.60 |
| 8/16 vs 16/32 (60010 vs 59098 params) | quaternion_exact vs pair_merge | -12.73 | -8.07 |
| 8/16 vs 16/32 (60010 vs 59098 params) | multi_q vs multi_real | -4.77 | -5.17 |
| 8/16 vs 16/32 (60010 vs 59098 params) | multi_q_random vs multi_real_random | -6.07 | -8.73 |

## Verification

Exact-fold equivalence checks (physical compact model vs functional substitution, check half): 216 cases, maximum logit error 8.58e-06 (tolerance 0.0001). Every quaternion compact model keeps QuaternionConv2d consumers, so the Hamilton structure is retained by construction.

## Limits

One block, two removal levels, small sequential CNNs on CIFAR-10, three seeds. Validation also selected baseline checkpoints. Ridge alpha was not tuned. No fine-tuning; a separate equal-budget fine-tuning study is required before practical claims. Weighted projection of the unconstrained ridge solution is not run here. No novelty or general superiority claim.
