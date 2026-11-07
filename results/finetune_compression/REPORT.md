# Equal-budget fine-tuning of compact policies

Status: completed. Every compact model from results/multisource_compensation (first 3 random victim sets only) and the uncompressed baseline were fine-tuned for 3 epochs on the 48,000 training images: AdamW, LR 0.0001 cosine to 1e-06 per step, weight decay 0.0001, batch 128, original augmentation, BN in train mode. All policies of one training seed see identical batch order and augmentations. Primary outcome: final-epoch validation accuracy, no checkpoint selection. No official-test inference. Hyperparameters were fixed a priori, not tuned.

## Validation accuracy before and after fine-tuning

Before-values use the same random victim sets as the fine-tuned rows.

| Architecture | Removed | Policy | Real parameters | Before | After (mean ± seed SD) | Change (pp) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| cnn6 | 4 | baseline | 73898 | 77.93% | 77.93% ± 0.93 pp | +0.00 |
| cnn6 | 4 | random | 66954 | 72.82% | 76.47% ± 1.16 pp | +3.64 |
| cnn6 | 4 | paired_pruned | 66954 | 71.40% | 76.57% ± 0.65 pp | +5.17 |
| cnn6 | 4 | quaternion_exact | 66954 | 74.73% | 76.73% ± 0.83 pp | +2.00 |
| cnn6 | 4 | multi_q_random | 66954 | 75.96% | 76.59% ± 0.57 pp | +0.63 |
| cnn6 | 4 | multi_full_projected | 66954 | 74.70% | 76.60% ± 0.56 pp | +1.90 |
| cnn6 | 4 | multi_q | 66954 | 76.63% | 77.17% ± 0.32 pp | +0.53 |
| cnn6 | 8 | baseline | 73898 | 77.93% | 77.93% ± 0.93 pp | +0.00 |
| cnn6 | 8 | random | 60010 | 55.33% | 74.70% ± 0.58 pp | +19.37 |
| cnn6 | 8 | paired_pruned | 60010 | 44.53% | 74.97% ± 0.31 pp | +30.43 |
| cnn6 | 8 | quaternion_exact | 60010 | 65.20% | 74.87% ± 0.71 pp | +9.67 |
| cnn6 | 8 | multi_q_random | 60010 | 72.29% | 74.69% ± 0.70 pp | +2.40 |
| cnn6 | 8 | multi_full_projected | 60010 | 69.40% | 75.20% ± 0.62 pp | +5.80 |
| cnn6 | 8 | multi_q | 60010 | 73.17% | 75.10% ± 0.78 pp | +1.93 |
| cnn8 | 4 | baseline | 118058 | 80.30% | 80.50% ± 1.01 pp | +0.20 |
| cnn8 | 4 | random | 112266 | 76.94% | 79.33% ± 0.54 pp | +2.39 |
| cnn8 | 4 | paired_pruned | 112266 | 75.30% | 79.40% ± 0.46 pp | +4.10 |
| cnn8 | 4 | quaternion_exact | 112266 | 78.27% | 79.73% ± 0.50 pp | +1.47 |
| cnn8 | 4 | multi_q_random | 112266 | 78.93% | 79.69% ± 0.90 pp | +0.76 |
| cnn8 | 4 | multi_full_projected | 112266 | 78.50% | 79.97% ± 0.67 pp | +1.47 |
| cnn8 | 4 | multi_q | 112266 | 79.53% | 79.73% ± 0.49 pp | +0.20 |
| cnn8 | 8 | baseline | 118058 | 80.30% | 80.50% ± 1.01 pp | +0.20 |
| cnn8 | 8 | random | 106474 | 66.22% | 77.44% ± 0.57 pp | +11.22 |
| cnn8 | 8 | paired_pruned | 106474 | 61.67% | 78.23% ± 0.35 pp | +16.57 |
| cnn8 | 8 | quaternion_exact | 106474 | 73.43% | 78.20% ± 0.92 pp | +4.77 |
| cnn8 | 8 | multi_q_random | 106474 | 75.66% | 77.68% ± 0.41 pp | +2.02 |
| cnn8 | 8 | multi_full_projected | 106474 | 74.00% | 78.73% ± 0.46 pp | +4.73 |
| cnn8 | 8 | multi_q | 106474 | 77.13% | 78.20% ± 0.26 pp | +1.07 |
| real_cnn6 | 8 | baseline | 72954 | 75.53% | 75.63% ± 0.80 pp | +0.10 |
| real_cnn6 | 8 | random | 66026 | 71.09% | 74.07% ± 1.00 pp | +2.98 |
| real_cnn6 | 8 | paired_pruned | 66026 | 71.67% | 74.40% ± 0.92 pp | +2.73 |
| real_cnn6 | 8 | pair_merge | 66026 | 73.70% | 74.87% ± 1.05 pp | +1.17 |
| real_cnn6 | 8 | multi_real_random | 66026 | 73.43% | 74.43% ± 0.70 pp | +1.00 |
| real_cnn6 | 8 | multi_real | 66026 | 74.37% | 74.80% ± 1.35 pp | +0.43 |
| real_cnn6 | 16 | baseline | 72954 | 75.53% | 75.63% ± 0.80 pp | +0.10 |
| real_cnn6 | 16 | random | 59098 | 56.29% | 70.60% ± 1.56 pp | +14.31 |
| real_cnn6 | 16 | paired_pruned | 59098 | 59.93% | 72.20% ± 1.90 pp | +12.27 |
| real_cnn6 | 16 | pair_merge | 59098 | 67.47% | 72.47% ± 1.79 pp | +5.00 |
| real_cnn6 | 16 | multi_real_random | 59098 | 66.22% | 71.77% ± 1.08 pp | +5.54 |
| real_cnn6 | 16 | multi_real | 59098 | 70.37% | 72.93% ± 1.63 pp | +2.57 |

## Paired effects after fine-tuning (left minus right)

| Architecture | Removed | Comparison | Validation per seed (pp) | Mean (pp) | Check mean (pp) | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| cnn6 | 4 | multi_q - quaternion_exact | +1.00, +0.40, -0.10 | +0.43 | +0.13 | 2/3 |
| cnn6 | 4 | multi_q - multi_full_projected | +0.70, +0.90, +0.10 | +0.57 | -0.27 | 3/3 |
| cnn6 | 4 | multi_q - random | -0.37, +2.27, +0.20 | +0.70 | +0.36 | 2/3 |
| cnn6 | 4 | multi_q_random - random | -0.07, +0.83, -0.40 | +0.12 | +0.13 | 1/3 |
| cnn6 | 4 | multi_q - multi_q_random | -0.30, +1.43, +0.60 | +0.58 | +0.22 | 2/3 |
| cnn6 | 4 | quaternion_exact - random | -1.37, +1.87, +0.30 | +0.27 | +0.22 | 2/3 |
| cnn6 | 4 | multi_q - paired_pruned | +0.90, +0.80, +0.10 | +0.60 | -0.20 | 3/3 |
| cnn6 | 8 | multi_q - quaternion_exact | +0.10, +1.40, -0.80 | +0.23 | -0.33 | 2/3 |
| cnn6 | 8 | multi_q - multi_full_projected | -0.10, +0.10, -0.30 | -0.10 | +0.07 | 1/3 |
| cnn6 | 8 | multi_q - random | +0.40, +1.43, -0.63 | +0.40 | +0.58 | 2/3 |
| cnn6 | 8 | multi_q_random - random | +0.30, +0.27, -0.60 | -0.01 | +0.47 | 2/3 |
| cnn6 | 8 | multi_q - multi_q_random | +0.10, +1.17, -0.03 | +0.41 | +0.11 | 2/3 |
| cnn6 | 8 | quaternion_exact - random | +0.30, +0.03, +0.17 | +0.17 | +0.91 | 3/3 |
| cnn6 | 8 | multi_q - paired_pruned | +0.30, +0.60, -0.50 | +0.13 | +0.53 | 2/3 |
| cnn8 | 4 | multi_q - quaternion_exact | +0.10, -0.40, +0.30 | +0.00 | +0.33 | 2/3 |
| cnn8 | 4 | multi_q - multi_full_projected | -0.40, -0.40, +0.10 | -0.23 | +0.33 | 1/3 |
| cnn8 | 4 | multi_q - random | +0.40, +0.13, +0.67 | +0.40 | +0.62 | 3/3 |
| cnn8 | 4 | multi_q_random - random | +0.83, -0.10, +0.33 | +0.36 | -0.09 | 2/3 |
| cnn8 | 4 | multi_q - multi_q_random | -0.43, +0.23, +0.33 | +0.04 | +0.71 | 2/3 |
| cnn8 | 4 | quaternion_exact - random | +0.30, +0.53, +0.37 | +0.40 | +0.29 | 3/3 |
| cnn8 | 4 | multi_q - paired_pruned | +0.50, -0.10, +0.60 | +0.33 | +0.60 | 2/3 |
| cnn8 | 8 | multi_q - quaternion_exact | -1.20, +0.50, +0.70 | +0.00 | -0.47 | 2/3 |
| cnn8 | 8 | multi_q - multi_full_projected | -1.00, -0.50, -0.10 | -0.53 | -0.40 | 0/3 |
| cnn8 | 8 | multi_q - random | -0.10, +1.33, +1.03 | +0.76 | +0.16 | 2/3 |
| cnn8 | 8 | multi_q_random - random | +0.03, +0.40, +0.27 | +0.23 | +0.16 | 3/3 |
| cnn8 | 8 | multi_q - multi_q_random | -0.13, +0.93, +0.77 | +0.52 | +0.00 | 2/3 |
| cnn8 | 8 | quaternion_exact - random | +1.10, +0.83, +0.33 | +0.76 | +0.62 | 3/3 |
| cnn8 | 8 | multi_q - paired_pruned | -0.20, -0.10, +0.20 | -0.03 | +0.07 | 1/3 |
| real_cnn6 | 8 | multi_real - pair_merge | -0.30, +0.30, -0.20 | -0.07 | +0.20 | 1/3 |
| real_cnn6 | 8 | multi_real - random | -0.17, +1.00, +1.37 | +0.73 | +1.18 | 2/3 |
| real_cnn6 | 8 | multi_real_random - random | +0.10, -0.03, +1.03 | +0.37 | +0.60 | 2/3 |
| real_cnn6 | 8 | multi_real - multi_real_random | -0.27, +1.03, +0.33 | +0.37 | +0.58 | 2/3 |
| real_cnn6 | 8 | pair_merge - random | +0.13, +0.70, +1.57 | +0.80 | +0.98 | 3/3 |
| real_cnn6 | 8 | multi_real - paired_pruned | -0.10, +0.80, +0.50 | +0.40 | -0.20 | 2/3 |
| real_cnn6 | 16 | multi_real - pair_merge | +0.60, +0.20, +0.60 | +0.47 | -0.07 | 3/3 |
| real_cnn6 | 16 | multi_real - random | +1.93, +1.93, +3.13 | +2.33 | +1.82 | 3/3 |
| real_cnn6 | 16 | multi_real_random - random | +1.63, +0.67, +1.20 | +1.17 | +0.98 | 3/3 |
| real_cnn6 | 16 | multi_real - multi_real_random | +0.30, +1.27, +1.93 | +1.17 | +0.84 | 3/3 |
| real_cnn6 | 16 | pair_merge - random | +1.33, +1.73, +2.53 | +1.87 | +1.89 | 3/3 |
| real_cnn6 | 16 | multi_real - paired_pruned | +1.00, +0.40, +0.80 | +0.73 | +0.40 | 3/3 |

## Matched-savings comparison after fine-tuning (CNN6)

Drops are from each family's own fine-tuned baseline; descriptive only.

| Removal pair | Policy pair | Quaternion drop (pp) | Real drop (pp) |
| --- | --- | ---: | ---: |
| 4/16 vs 8/32 | random vs random | -1.47 | -1.57 |
| 4/16 vs 8/32 | paired_pruned vs paired_pruned | -1.37 | -1.23 |
| 4/16 vs 8/32 | quaternion_exact vs pair_merge | -1.20 | -0.77 |
| 4/16 vs 8/32 | multi_q vs multi_real | -0.77 | -0.83 |
| 4/16 vs 8/32 | multi_q_random vs multi_real_random | -1.34 | -1.20 |
| 8/16 vs 16/32 | random vs random | -3.23 | -5.03 |
| 8/16 vs 16/32 | paired_pruned vs paired_pruned | -2.97 | -3.43 |
| 8/16 vs 16/32 | quaternion_exact vs pair_merge | -3.07 | -3.17 |
| 8/16 vs 16/32 | multi_q vs multi_real | -2.83 | -2.70 |
| 8/16 vs 16/32 | multi_q_random vs multi_real_random | -3.24 | -3.87 |

## Verification

183 fine-tuning runs. Rebuilt compact models matched the frozen multi-source maps; parameter counts and QuaternionConv2d layer counts are unchanged by fine-tuning in every run.

## Limits

Three epochs at one fixed LR schedule; longer or tuned fine-tuning could change the ranking. Three random victim sets per seed. Validation also selected baseline checkpoints. One block, small CNNs, CIFAR-10. No novelty or general superiority claim.
