# Quality-size curve: uniform multi-site compression

Status: completed. Dataset cifar; architectures cnn6, real_cnn6; seeds [2026, 2027, 2028]. The same fraction [0.375, 0.625] of units is removed at every convolution output, shallow to deep, so each compact model is the standard architecture with narrower widths (checked by reloading it into that architecture). Compensation folds a ridge reconstruction (alpha 0.001) from the kept units of the same site into the consumer: right-Hamilton maps for quaternion models, scalar maps for real models. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. Fine-tuning: 3 epochs, final epoch, no selection. Scratch: same compact architecture, baseline trainer and budget, checkpoint selected by validation cross-entropy as for the baselines. No official-test inference. No novelty claim.

## Validation accuracy (mean ± seed SD)

| Architecture | Removed fraction | Policy | Real parameters | Without fine-tuning | After fine-tuning / from scratch |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn6 | 0.0 | baseline | 73898 | 77.93% ± 1.10 (n=3) | 77.93% ± 0.93 (n=3) |
| cnn6 | 0.375 | random | 29450 | 14.60% ± 2.34 (n=3) | 57.67% ± 0.38 (n=3) |
| cnn6 | 0.375 | bn_pruned | 29450 | 15.37% ± 5.39 (n=3) | 60.90% ± 2.05 (n=3) |
| cnn6 | 0.375 | greedy_pruned | 29450 | 15.10% ± 2.88 (n=3) | — |
| cnn6 | 0.375 | random_comp | 29450 | 23.91% ± 1.37 (n=3) | 59.50% ± 1.07 (n=3) |
| cnn6 | 0.375 | greedy_full_projected | 29450 | 24.33% ± 12.82 (n=3) | — |
| cnn6 | 0.375 | greedy_comp | 29450 | 34.03% ± 5.49 (n=3) | 63.53% ± 1.20 (n=3) |
| cnn6 | 0.375 | scratch | 29450 | — | 68.33% ± 0.81 (n=3) |
| cnn6 | 0.625 | random | 10978 | 10.27% ± 0.31 (n=3) | 29.00% ± 1.09 (n=3) |
| cnn6 | 0.625 | bn_pruned | 10978 | 9.97% ± 0.15 (n=3) | 32.20% ± 1.25 (n=3) |
| cnn6 | 0.625 | greedy_pruned | 10978 | 10.00% ± 0.00 (n=3) | — |
| cnn6 | 0.625 | random_comp | 10978 | 11.01% ± 1.11 (n=3) | 29.00% ± 1.97 (n=3) |
| cnn6 | 0.625 | greedy_full_projected | 10978 | 10.03% ± 0.15 (n=3) | — |
| cnn6 | 0.625 | greedy_comp | 10978 | 12.47% ± 1.75 (n=3) | 36.00% ± 3.03 (n=3) |
| cnn6 | 0.625 | scratch | 10978 | — | 58.23% ± 0.93 (n=3) |
| real_cnn6 | 0.0 | baseline | 72954 | 75.53% ± 1.27 (n=3) | 75.63% ± 0.80 (n=3) |
| real_cnn6 | 0.375 | random | 28860 | 12.63% ± 1.57 (n=3) | 56.17% ± 0.71 (n=3) |
| real_cnn6 | 0.375 | bn_pruned | 28860 | 12.17% ± 3.67 (n=3) | 55.37% ± 1.15 (n=3) |
| real_cnn6 | 0.375 | greedy_pruned | 28860 | 22.40% ± 2.65 (n=3) | — |
| real_cnn6 | 0.375 | random_comp | 28860 | 30.45% ± 2.88 (n=3) | 59.69% ± 1.87 (n=3) |
| real_cnn6 | 0.375 | greedy_comp | 28860 | 49.43% ± 2.53 (n=3) | 66.00% ± 2.07 (n=3) |
| real_cnn6 | 0.375 | scratch | 28860 | — | 64.50% ± 0.61 (n=3) |
| real_cnn6 | 0.625 | random | 10624 | 10.13% ± 0.14 (n=3) | 26.42% ± 0.45 (n=3) |
| real_cnn6 | 0.625 | bn_pruned | 10624 | 10.27% ± 0.46 (n=3) | 29.33% ± 0.59 (n=3) |
| real_cnn6 | 0.625 | greedy_pruned | 10624 | 9.27% ± 1.27 (n=3) | — |
| real_cnn6 | 0.625 | random_comp | 10624 | 14.53% ± 0.69 (n=3) | 28.73% ± 3.16 (n=3) |
| real_cnn6 | 0.625 | greedy_comp | 10624 | 18.13% ± 5.40 (n=3) | 41.40% ± 4.60 (n=3) |
| real_cnn6 | 0.625 | scratch | 10624 | — | 51.30% ± 1.80 (n=3) |

## Paired effects (left minus right, validation pp, all seeds retained)

| Architecture | Removed fraction | Comparison | Per seed | Mean | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: |
| cnn6 | 0.375 | greedy_comp@before - random@before | +23.70, +22.84, +11.76 | +19.43 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - bn_pruned@before | +17.60, +21.60, +16.80 | +18.67 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - greedy_pruned@before | +26.60, +16.60, +13.60 | +18.93 | 3/3 |
| cnn6 | 0.375 | random_comp@before - random@before | +8.82, +13.14, +5.98 | +9.31 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - random_comp@before | +14.88, +9.70, +5.78 | +10.12 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - greedy_full_projected@before | +0.70, +13.50, +14.90 | +9.70 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - random@after | +6.60, +4.87, +6.13 | +5.87 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - bn_pruned@after | +5.90, +1.30, +0.70 | +2.63 | 3/3 |
| cnn6 | 0.375 | random_comp@after - random@after | +2.63, +1.50, +1.37 | +1.83 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - random_comp@after | +3.97, +3.37, +4.77 | +4.03 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - scratch@scratch | -2.70, -6.60, -5.10 | -4.80 | 0/3 |
| cnn6 | 0.375 | bn_pruned@after - scratch@scratch | -8.60, -7.90, -5.80 | -7.43 | 0/3 |
| cnn6 | 0.375 | random@after - scratch@scratch | -9.30, -11.47, -11.23 | -10.67 | 0/3 |
| cnn6 | 0.625 | greedy_comp@before - random@before | +1.38, +4.34, +0.88 | +2.20 | 3/3 |
| cnn6 | 0.625 | greedy_comp@before - bn_pruned@before | +1.90, +4.60, +1.00 | +2.50 | 3/3 |
| cnn6 | 0.625 | greedy_comp@before - greedy_pruned@before | +2.00, +4.40, +1.00 | +2.47 | 3/3 |
| cnn6 | 0.625 | random_comp@before - random@before | -0.36, +2.22, +0.36 | +0.74 | 2/3 |
| cnn6 | 0.625 | greedy_comp@before - random_comp@before | +1.74, +2.12, +0.52 | +1.46 | 3/3 |
| cnn6 | 0.625 | greedy_comp@before - greedy_full_projected@before | +2.10, +4.20, +1.00 | +2.43 | 3/3 |
| cnn6 | 0.625 | greedy_comp@after - random@after | +8.70, +7.77, +4.53 | +7.00 | 3/3 |
| cnn6 | 0.625 | greedy_comp@after - bn_pruned@after | +5.30, +4.50, +1.60 | +3.80 | 3/3 |
| cnn6 | 0.625 | random_comp@after - random@after | +2.03, -1.07, -0.97 | -0.00 | 1/3 |
| cnn6 | 0.625 | greedy_comp@after - random_comp@after | +6.67, +8.83, +5.50 | +7.00 | 3/3 |
| cnn6 | 0.625 | greedy_comp@after - scratch@scratch | -20.90, -19.30, -26.50 | -22.23 | 0/3 |
| cnn6 | 0.625 | bn_pruned@after - scratch@scratch | -26.20, -23.80, -28.10 | -26.03 | 0/3 |
| cnn6 | 0.625 | random@after - scratch@scratch | -29.60, -27.07, -31.03 | -29.23 | 0/3 |
| real_cnn6 | 0.375 | greedy_comp@before - random@before | +38.54, +36.66, +35.22 | +36.81 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@before - bn_pruned@before | +35.90, +37.40, +38.50 | +37.27 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@before - greedy_pruned@before | +32.50, +22.40, +26.20 | +27.03 | 3/3 |
| real_cnn6 | 0.375 | random_comp@before - random@before | +17.12, +16.54, +19.82 | +17.83 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@before - random_comp@before | +21.42, +20.12, +15.40 | +18.98 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - random@after | +10.00, +12.00, +7.50 | +9.83 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - bn_pruned@after | +10.00, +14.10, +7.80 | +10.63 | 3/3 |
| real_cnn6 | 0.375 | random_comp@after - random@after | +2.20, +3.97, +4.40 | +3.52 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - random_comp@after | +7.80, +8.03, +3.10 | +6.31 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - scratch@scratch | +0.50, +3.50, +0.50 | +1.50 | 3/3 |
| real_cnn6 | 0.375 | bn_pruned@after - scratch@scratch | -9.50, -10.60, -7.30 | -9.13 | 0/3 |
| real_cnn6 | 0.375 | random@after - scratch@scratch | -9.50, -8.50, -7.00 | -8.33 | 0/3 |
| real_cnn6 | 0.625 | greedy_comp@before - random@before | +3.22, +7.14, +13.64 | +8.00 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@before - bn_pruned@before | +2.40, +7.30, +13.90 | +7.87 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@before - greedy_pruned@before | +3.20, +9.50, +13.90 | +8.87 | 3/3 |
| real_cnn6 | 0.625 | random_comp@before - random@before | +4.12, +5.16, +3.90 | +4.39 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@before - random_comp@before | -0.90, +1.98, +9.74 | +3.61 | 2/3 |
| real_cnn6 | 0.625 | greedy_comp@after - random@after | +10.80, +19.87, +14.27 | +14.98 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@after - bn_pruned@after | +6.90, +17.00, +12.30 | +12.07 | 3/3 |
| real_cnn6 | 0.625 | random_comp@after - random@after | +5.10, +3.60, -1.77 | +2.31 | 2/3 |
| real_cnn6 | 0.625 | greedy_comp@after - random_comp@after | +5.70, +16.27, +16.03 | +12.67 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@after - scratch@scratch | -12.40, -5.70, -11.60 | -9.90 | 0/3 |
| real_cnn6 | 0.625 | bn_pruned@after - scratch@scratch | -19.30, -22.70, -23.90 | -21.97 | 0/3 |
| real_cnn6 | 0.625 | random@after - scratch@scratch | -23.20, -25.57, -25.87 | -24.88 | 0/3 |

## Quaternion vs real at the same removed fraction (near-equal real parameters)

Descriptive: the families start from different baselines. Drop = accuracy minus the family's own baseline of the same stage (scratch is compared with the original baseline).

| Fraction | Policy and stage | Quaternion params | Real params | Quaternion acc | Real acc | Quaternion drop | Real drop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.375 | random@before | 29450 | 28860 | 14.60% | 12.63% | -63.33 | -62.91 |
| 0.375 | greedy_comp@before | 29450 | 28860 | 34.03% | 49.43% | -43.90 | -26.10 |
| 0.375 | random@after | 29450 | 28860 | 57.67% | 56.17% | -20.27 | -19.47 |
| 0.375 | bn_pruned@after | 29450 | 28860 | 60.90% | 55.37% | -17.03 | -20.27 |
| 0.375 | greedy_comp@after | 29450 | 28860 | 63.53% | 66.00% | -14.40 | -9.63 |
| 0.375 | scratch@scratch | 29450 | 28860 | 68.33% | 64.50% | -9.60 | -11.03 |
| 0.625 | random@before | 10978 | 10624 | 10.27% | 10.13% | -67.67 | -65.40 |
| 0.625 | greedy_comp@before | 10978 | 10624 | 12.47% | 18.13% | -65.47 | -57.40 |
| 0.625 | random@after | 10978 | 10624 | 29.00% | 26.42% | -48.93 | -49.21 |
| 0.625 | bn_pruned@after | 10978 | 10624 | 32.20% | 29.33% | -45.73 | -46.30 |
| 0.625 | greedy_comp@after | 10978 | 10624 | 36.00% | 41.40% | -41.93 | -34.23 |
| 0.625 | scratch@scratch | 10978 | 10624 | 58.23% | 51.30% | -19.70 | -24.23 |

## Verification and diagnostics

Exact-fold checks (physical compact model vs functional substitution, per site, 200 check images): 432 checks, maximum logit error 1.05e-05 (tolerance 0.001). Every compact model was reloaded into the standard narrower architecture with identical logits.

Energy of the unconstrained (greedy_full_projected) additions discarded by the Hamilton projection, mean over seeds, per site:

| Architecture | Fraction | Per site (shallow to deep) |
| --- | ---: | --- |
| cnn6 | 0.375 | 66.9%, 62.7%, 54.3%, 61.9%, 64.5%, 0.0% |
| cnn6 | 0.625 | 68.0%, 61.0%, 59.8%, 65.4%, 68.8%, 0.0% |

Figure: `quality_size_curve.png` (solid quaternion, dashed real).

## Limits

Small sequential CNNs, uniform fractions, one fixed ridge alpha and fine-tuning schedule, few seeds. Validation also selected the baseline and scratch checkpoints. Random rows average their repetitions within seed first.
