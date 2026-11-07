# EuroSAT replication: multi-source compensation and equal-budget fine-tuning

Second dataset: EuroSAT_MS (Zenodo 7711810), four bands q = NIR + R i + G j + B k (NIR in the real part is a choice). Quaternion and real CNN6 baselines were trained with the unchanged CIFAR trainer and budget (results/eurosat_cnn). All compression and fine-tuning rules were copied unchanged from the CIFAR studies and fixed before any EuroSAT compression outcome. Official test split (5,400 images) held out and not evaluated. Validation also selected baseline checkpoints. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. No novelty claim.

## Phase A: compression without fine-tuning

| Model | Removed | Policy | Real parameters | Validation mean ± seed SD | Mean CE | Check mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| quaternion | 4 | baseline | 73898 | 96.67% ± 0.29 pp | 0.0979 | 96.87% |
| quaternion | 4 | random | 66954 | 93.11% ± 0.48 pp | 0.1899 | 93.64% |
| quaternion | 4 | paired_pruned | 66954 | 92.87% ± 1.12 pp | 0.1873 | 93.73% |
| quaternion | 4 | quaternion_exact | 66954 | 94.53% ± 0.51 pp | 0.1489 | 94.80% |
| quaternion | 4 | multi_q_random | 66954 | 95.81% ± 0.03 pp | 0.1220 | 96.31% |
| quaternion | 4 | multi_full_projected | 66954 | 95.70% ± 0.10 pp | 0.1159 | 96.73% |
| quaternion | 4 | multi_q | 66954 | 96.17% ± 0.12 pp | 0.1154 | 96.40% |
| quaternion | 4 | multi_full_substitution | functional oracle | 96.17% ± 0.15 pp | 0.1107 | 96.67% |
| quaternion | 8 | baseline | 73898 | 96.67% ± 0.29 pp | 0.0979 | 96.87% |
| quaternion | 8 | random | 60010 | 74.85% ± 2.69 pp | 0.7135 | 75.86% |
| quaternion | 8 | paired_pruned | 60010 | 79.10% ± 2.49 pp | 0.5764 | 80.33% |
| quaternion | 8 | quaternion_exact | 60010 | 89.20% ± 3.91 pp | 0.3044 | 90.47% |
| quaternion | 8 | multi_q_random | 60010 | 94.22% ± 0.18 pp | 0.1758 | 94.85% |
| quaternion | 8 | multi_full_projected | 60010 | 92.90% ± 1.41 pp | 0.1906 | 93.40% |
| quaternion | 8 | multi_q | 60010 | 94.33% ± 1.22 pp | 0.1719 | 95.20% |
| quaternion | 8 | multi_full_substitution | functional oracle | 94.77% ± 0.21 pp | 0.1512 | 95.40% |
| real | 8 | baseline | 73098 | 96.23% ± 0.38 pp | 0.1058 | 96.40% |
| real | 8 | random | 66170 | 93.80% ± 0.57 pp | 0.1730 | 93.63% |
| real | 8 | paired_pruned | 66170 | 93.33% ± 1.25 pp | 0.1779 | 93.40% |
| real | 8 | pair_merge | 66170 | 95.20% ± 0.61 pp | 0.1368 | 95.67% |
| real | 8 | multi_real_random | 66170 | 95.71% ± 0.69 pp | 0.1304 | 95.72% |
| real | 8 | multi_real | 66170 | 96.23% ± 0.55 pp | 0.1181 | 96.33% |
| real | 16 | baseline | 73098 | 96.23% ± 0.38 pp | 0.1058 | 96.40% |
| real | 16 | random | 59242 | 80.84% ± 2.52 pp | 0.5322 | 81.26% |
| real | 16 | paired_pruned | 59242 | 85.30% ± 1.67 pp | 0.3928 | 86.07% |
| real | 16 | pair_merge | 59242 | 92.60% ± 0.62 pp | 0.2299 | 92.67% |
| real | 16 | multi_real_random | 59242 | 92.84% ± 2.40 pp | 0.2118 | 93.00% |
| real | 16 | multi_real | 59242 | 95.20% ± 0.89 pp | 0.1710 | 95.13% |

| Model | Removed | Comparison | Validation per seed (pp) | Mean (pp) | Check mean (pp) | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| quaternion | 4 | multi_q - quaternion_exact | +1.00, +2.00, +1.90 | +1.63 | +1.60 | 3/3 |
| quaternion | 4 | multi_q - multi_full_projected | +0.50, +0.40, +0.50 | +0.47 | -0.33 | 3/3 |
| quaternion | 4 | multi_full_substitution - multi_q | +0.20, +0.10, -0.30 | +0.00 | +0.27 | 2/3 |
| quaternion | 4 | multi_q - random | +2.46, +3.39, +3.31 | +3.05 | +2.76 | 3/3 |
| quaternion | 4 | multi_q_random - random | +2.14, +3.13, +2.82 | +2.70 | +2.67 | 3/3 |
| quaternion | 4 | multi_q - multi_q_random | +0.32, +0.26, +0.49 | +0.36 | +0.09 | 3/3 |
| quaternion | 4 | quaternion_exact - random | +1.46, +1.39, +1.41 | +1.42 | +1.16 | 3/3 |
| quaternion | 4 | multi_q - paired_pruned | +2.00, +4.20, +3.70 | +3.30 | +2.67 | 3/3 |
| quaternion | 8 | multi_q - quaternion_exact | +3.50, +3.60, +8.30 | +5.13 | +4.73 | 3/3 |
| quaternion | 8 | multi_q - multi_full_projected | +3.00, +1.00, +0.30 | +1.43 | +1.80 | 3/3 |
| quaternion | 8 | multi_full_substitution - multi_q | +0.10, -0.40, +1.60 | +0.43 | +0.20 | 2/3 |
| quaternion | 8 | multi_q - random | +18.13, +23.66, +16.66 | +19.48 | +19.34 | 3/3 |
| quaternion | 8 | multi_q_random - random | +17.58, +22.67, +17.85 | +19.37 | +18.99 | 3/3 |
| quaternion | 8 | multi_q - multi_q_random | +0.55, +0.99, -1.19 | +0.12 | +0.35 | 2/3 |
| quaternion | 8 | quaternion_exact - random | +14.63, +20.06, +8.36 | +14.35 | +14.61 | 3/3 |
| quaternion | 8 | multi_q - paired_pruned | +13.30, +15.80, +16.60 | +15.23 | +14.87 | 3/3 |
| real | 8 | multi_real - pair_merge | +0.60, +0.70, +1.80 | +1.03 | +0.67 | 3/3 |
| real | 8 | multi_real - random | +2.22, +1.66, +3.43 | +2.44 | +2.71 | 3/3 |
| real | 8 | multi_real_random - random | +1.75, +0.98, +3.01 | +1.91 | +2.09 | 3/3 |
| real | 8 | multi_real - multi_real_random | +0.47, +0.68, +0.42 | +0.52 | +0.61 | 3/3 |
| real | 8 | pair_merge - random | +1.62, +0.96, +1.63 | +1.40 | +2.04 | 3/3 |
| real | 8 | multi_real - paired_pruned | +4.40, +1.00, +3.30 | +2.90 | +2.93 | 3/3 |
| real | 16 | multi_real - pair_merge | +2.60, +2.10, +3.10 | +2.60 | +2.47 | 3/3 |
| real | 16 | multi_real - random | +14.49, +11.17, +17.41 | +14.36 | +13.87 | 3/3 |
| real | 16 | multi_real_random - random | +12.69, +7.04, +16.26 | +12.00 | +11.74 | 3/3 |
| real | 16 | multi_real - multi_real_random | +1.80, +4.13, +1.15 | +2.36 | +2.13 | 3/3 |
| real | 16 | pair_merge - random | +11.89, +9.07, +14.31 | +11.76 | +11.41 | 3/3 |
| real | 16 | multi_real - paired_pruned | +12.40, +7.40, +9.90 | +9.90 | +9.07 | 3/3 |

Exact-fold checks: 144 cases, maximum logit error 1.14e-05.

## Phase B: after 3 fine-tuning epochs (final epoch, no selection; first 3 random sets)

| Model | Removed | Policy | Real parameters | Validation mean ± seed SD | Mean CE | Check mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| quaternion | 4 | baseline | 73898 | 96.57% ± 0.40 pp | 0.1016 | 96.80% |
| quaternion | 4 | random | 66954 | 96.03% ± 0.51 pp | 0.1131 | 96.24% |
| quaternion | 4 | paired_pruned | 66954 | 96.30% ± 0.46 pp | 0.1096 | 96.60% |
| quaternion | 4 | quaternion_exact | 66954 | 96.33% ± 0.50 pp | 0.1095 | 96.53% |
| quaternion | 4 | multi_q_random | 66954 | 96.00% ± 0.51 pp | 0.1127 | 96.27% |
| quaternion | 4 | multi_full_projected | 66954 | 96.33% ± 0.81 pp | 0.1086 | 96.27% |
| quaternion | 4 | multi_q | 66954 | 96.13% ± 0.60 pp | 0.1098 | 96.27% |
| quaternion | 8 | baseline | 73898 | 96.57% ± 0.40 pp | 0.1016 | 96.80% |
| quaternion | 8 | random | 60010 | 95.68% ± 0.45 pp | 0.1309 | 95.96% |
| quaternion | 8 | paired_pruned | 60010 | 95.77% ± 0.59 pp | 0.1282 | 96.13% |
| quaternion | 8 | quaternion_exact | 60010 | 95.60% ± 0.66 pp | 0.1307 | 95.87% |
| quaternion | 8 | multi_q_random | 60010 | 95.58% ± 0.54 pp | 0.1290 | 95.80% |
| quaternion | 8 | multi_full_projected | 60010 | 95.80% ± 0.44 pp | 0.1246 | 96.13% |
| quaternion | 8 | multi_q | 60010 | 95.80% ± 0.61 pp | 0.1266 | 96.00% |
| real | 8 | baseline | 73098 | 96.40% ± 0.70 pp | 0.1117 | 96.33% |
| real | 8 | random | 66170 | 96.11% ± 0.46 pp | 0.1200 | 96.33% |
| real | 8 | paired_pruned | 66170 | 96.10% ± 0.56 pp | 0.1166 | 96.07% |
| real | 8 | pair_merge | 66170 | 96.27% ± 0.32 pp | 0.1173 | 95.93% |
| real | 8 | multi_real_random | 66170 | 96.14% ± 0.47 pp | 0.1198 | 96.33% |
| real | 8 | multi_real | 66170 | 96.17% ± 0.49 pp | 0.1165 | 96.13% |
| real | 16 | baseline | 73098 | 96.40% ± 0.70 pp | 0.1117 | 96.33% |
| real | 16 | random | 59242 | 95.30% ± 1.19 pp | 0.1497 | 95.40% |
| real | 16 | paired_pruned | 59242 | 95.60% ± 0.35 pp | 0.1414 | 95.73% |
| real | 16 | pair_merge | 59242 | 95.70% ± 0.56 pp | 0.1407 | 95.53% |
| real | 16 | multi_real_random | 59242 | 95.60% ± 0.76 pp | 0.1420 | 95.64% |
| real | 16 | multi_real | 59242 | 95.87% ± 0.59 pp | 0.1353 | 95.80% |

| Model | Removed | Comparison | Validation per seed (pp) | Mean (pp) | Check mean (pp) | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| quaternion | 4 | multi_q - quaternion_exact | -0.20, -0.30, -0.10 | -0.20 | -0.27 | 0/3 |
| quaternion | 4 | multi_q - multi_full_projected | -0.50, +0.10, -0.20 | -0.20 | +0.00 | 1/3 |
| quaternion | 4 | multi_q - random | +0.03, +0.03, +0.23 | +0.10 | +0.02 | 3/3 |
| quaternion | 4 | multi_q_random - random | -0.03, -0.03, -0.03 | -0.03 | +0.02 | 0/3 |
| quaternion | 4 | multi_q - multi_q_random | +0.07, +0.07, +0.27 | +0.13 | -0.00 | 3/3 |
| quaternion | 4 | quaternion_exact - random | +0.23, +0.33, +0.33 | +0.30 | +0.29 | 3/3 |
| quaternion | 4 | multi_q - paired_pruned | -0.20, -0.30, +0.00 | -0.17 | -0.33 | 0/3 |
| quaternion | 8 | multi_q - quaternion_exact | -0.10, +0.20, +0.50 | +0.20 | +0.13 | 2/3 |
| quaternion | 8 | multi_q - multi_full_projected | +0.10, -0.20, +0.10 | +0.00 | -0.13 | 2/3 |
| quaternion | 8 | multi_q - random | +0.37, -0.10, +0.10 | +0.12 | +0.04 | 2/3 |
| quaternion | 8 | multi_q_random - random | -0.07, -0.20, -0.03 | -0.10 | -0.16 | 0/3 |
| quaternion | 8 | multi_q - multi_q_random | +0.43, +0.10, +0.13 | +0.22 | +0.20 | 3/3 |
| quaternion | 8 | quaternion_exact - random | +0.47, -0.30, -0.40 | -0.08 | -0.09 | 1/3 |
| quaternion | 8 | multi_q - paired_pruned | -0.10, +0.00, +0.20 | +0.03 | -0.13 | 1/3 |
| real | 8 | multi_real - pair_merge | +0.00, -0.30, +0.00 | -0.10 | +0.20 | 0/3 |
| real | 8 | multi_real - random | -0.10, +0.00, +0.27 | +0.06 | -0.20 | 1/3 |
| real | 8 | multi_real_random - random | -0.03, +0.00, +0.13 | +0.03 | +0.00 | 1/3 |
| real | 8 | multi_real - multi_real_random | -0.07, +0.00, +0.13 | +0.02 | -0.20 | 1/3 |
| real | 8 | pair_merge - random | -0.10, +0.30, +0.27 | +0.16 | -0.40 | 2/3 |
| real | 8 | multi_real - paired_pruned | +0.20, +0.10, -0.10 | +0.07 | +0.07 | 2/3 |
| real | 16 | multi_real - pair_merge | +0.30, +0.10, +0.10 | +0.17 | +0.27 | 3/3 |
| real | 16 | multi_real - random | +0.03, +1.27, +0.40 | +0.57 | +0.40 | 3/3 |
| real | 16 | multi_real_random - random | +0.20, +0.83, -0.13 | +0.30 | +0.24 | 2/3 |
| real | 16 | multi_real - multi_real_random | -0.17, +0.43, +0.53 | +0.27 | +0.16 | 2/3 |
| real | 16 | pair_merge - random | -0.27, +1.17, +0.30 | +0.40 | +0.13 | 2/3 |
| real | 16 | multi_real - paired_pruned | +0.30, +0.00, +0.50 | +0.27 | +0.07 | 2/3 |

120 fine-tuning runs; parameter and quaternion-layer counts unchanged in every run.

## Limits

One block, two removal levels, one small CNN family, three seeds, fixed fine-tuning schedule. Descriptive cross-family comparisons only.
