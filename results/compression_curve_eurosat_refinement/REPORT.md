# Quality-size curve: uniform multi-site compression

Status: completed. Dataset eurosat; architectures cnn6, real_cnn6; seeds [2026, 2027, 2028]. The same fraction [0.375, 0.625] of units is removed at every convolution output, shallow to deep, so each compact model is the standard architecture with narrower widths (checked by reloading it into that architecture). Compensation folds a ridge reconstruction (alpha 0.001) from the kept units of the same site into the consumer: right-Hamilton maps for quaternion models, scalar maps for real models. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. Fine-tuning: 3 epochs, final epoch, no selection. Scratch: same compact architecture, baseline trainer and budget, checkpoint selected by validation cross-entropy as for the baselines. No official-test inference. No novelty claim.

## Validation accuracy (mean ± seed SD)

| Architecture | Removed fraction | Policy | Real parameters | Without fine-tuning | After fine-tuning / from scratch |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn6 | 0.0 | baseline | 73898 | 96.67% ± 0.29 (n=3) | 96.57% ± 0.40 (n=3) |
| cnn6 | 0.375 | random | 29450 | 15.11% ± 1.30 (n=3) | 83.76% ± 1.53 (n=3) |
| cnn6 | 0.375 | bn_pruned | 29450 | 11.67% ± 2.28 (n=3) | 86.17% ± 1.20 (n=3) |
| cnn6 | 0.375 | greedy_pruned | 29450 | 22.63% ± 9.09 (n=3) | — |
| cnn6 | 0.375 | random_comp | 29450 | 40.67% ± 2.96 (n=3) | 87.13% ± 1.51 (n=3) |
| cnn6 | 0.375 | greedy_full_projected | 29450 | 41.77% ± 5.62 (n=3) | — |
| cnn6 | 0.375 | greedy_comp | 29450 | 57.10% ± 6.63 (n=3) | 90.47% ± 0.80 (n=3) |
| cnn6 | 0.375 | scratch | 29450 | — | 94.33% ± 0.59 (n=3) |
| cnn6 | 0.625 | random | 10978 | 10.77% ± 0.11 (n=3) | 59.92% ± 2.32 (n=3) |
| cnn6 | 0.625 | bn_pruned | 10978 | 10.80% ± 0.36 (n=3) | 61.30% ± 2.00 (n=3) |
| cnn6 | 0.625 | greedy_pruned | 10978 | 15.83% ± 6.56 (n=3) | — |
| cnn6 | 0.625 | random_comp | 10978 | 18.57% ± 4.22 (n=3) | 66.11% ± 2.91 (n=3) |
| cnn6 | 0.625 | greedy_full_projected | 10978 | 15.33% ± 2.24 (n=3) | — |
| cnn6 | 0.625 | greedy_comp | 10978 | 25.43% ± 3.79 (n=3) | 72.80% ± 3.32 (n=3) |
| cnn6 | 0.625 | scratch | 10978 | — | 90.33% ± 1.15 (n=3) |
| real_cnn6 | 0.0 | baseline | 73098 | 96.23% ± 0.38 (n=3) | 96.40% ± 0.70 (n=3) |
| real_cnn6 | 0.375 | random | 28950 | 14.95% ± 2.56 (n=3) | 82.28% ± 0.60 (n=3) |
| real_cnn6 | 0.375 | bn_pruned | 28950 | 12.77% ± 3.61 (n=3) | 82.63% ± 2.78 (n=3) |
| real_cnn6 | 0.375 | greedy_pruned | 28950 | 27.87% ± 9.47 (n=3) | — |
| real_cnn6 | 0.375 | random_comp | 28950 | 43.65% ± 7.92 (n=3) | 86.42% ± 0.69 (n=3) |
| real_cnn6 | 0.375 | greedy_comp | 28950 | 77.93% ± 6.43 (n=3) | 91.63% ± 1.93 (n=3) |
| real_cnn6 | 0.375 | scratch | 28950 | — | 92.73% ± 0.42 (n=3) |
| real_cnn6 | 0.625 | random | 10678 | 10.17% ± 0.98 (n=3) | 55.89% ± 3.25 (n=3) |
| real_cnn6 | 0.625 | bn_pruned | 10678 | 10.30% ± 1.73 (n=3) | 32.67% ± 4.16 (n=3) |
| real_cnn6 | 0.625 | greedy_pruned | 10678 | 14.00% ± 4.85 (n=3) | — |
| real_cnn6 | 0.625 | random_comp | 10678 | 24.01% ± 4.70 (n=3) | 60.61% ± 3.11 (n=3) |
| real_cnn6 | 0.625 | greedy_comp | 10678 | 24.90% ± 2.16 (n=3) | 74.13% ± 5.15 (n=3) |
| real_cnn6 | 0.625 | scratch | 10678 | — | 87.13% ± 0.67 (n=3) |

## Paired effects (left minus right, validation pp, all seeds retained)

| Architecture | Removed fraction | Comparison | Per seed | Mean | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: |
| cnn6 | 0.375 | greedy_comp@before - random@before | +38.60, +51.04, +36.32 | +41.99 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - bn_pruned@before | +42.00, +51.00, +43.30 | +45.43 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - greedy_pruned@before | +27.20, +35.90, +40.30 | +34.47 | 3/3 |
| cnn6 | 0.375 | random_comp@before - random@before | +22.20, +29.96, +24.50 | +25.55 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - random_comp@before | +16.40, +21.08, +11.82 | +16.43 | 3/3 |
| cnn6 | 0.375 | greedy_comp@before - greedy_full_projected@before | +8.60, +29.40, +8.00 | +15.33 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - random@after | +7.90, +6.40, +5.83 | +6.71 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - bn_pruned@after | +4.30, +2.30, +6.30 | +4.30 | 3/3 |
| cnn6 | 0.375 | random_comp@after - random@after | +3.47, +3.30, +3.37 | +3.38 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - random_comp@after | +4.43, +3.10, +2.47 | +3.33 | 3/3 |
| cnn6 | 0.375 | greedy_comp@after - scratch@scratch | -3.70, -4.20, -3.70 | -3.87 | 0/3 |
| cnn6 | 0.375 | bn_pruned@after - scratch@scratch | -8.00, -6.50, -10.00 | -8.17 | 0/3 |
| cnn6 | 0.375 | random@after - scratch@scratch | -11.60, -10.60, -9.53 | -10.58 | 0/3 |
| cnn6 | 0.625 | greedy_comp@before - random@before | +16.36, +17.20, +10.42 | +14.66 | 3/3 |
| cnn6 | 0.625 | greedy_comp@before - bn_pruned@before | +16.20, +17.00, +10.70 | +14.63 | 3/3 |
| cnn6 | 0.625 | greedy_comp@before - greedy_pruned@before | +18.00, +11.90, -1.10 | +9.60 | 2/3 |
| cnn6 | 0.625 | random_comp@before - random@before | +11.28, +2.96, +9.16 | +7.80 | 3/3 |
| cnn6 | 0.625 | greedy_comp@before - random_comp@before | +5.08, +14.24, +1.26 | +6.86 | 3/3 |
| cnn6 | 0.625 | greedy_comp@before - greedy_full_projected@before | +12.80, +14.30, +3.20 | +10.10 | 3/3 |
| cnn6 | 0.625 | greedy_comp@after - random@after | +7.17, +15.70, +15.77 | +12.88 | 3/3 |
| cnn6 | 0.625 | greedy_comp@after - bn_pruned@after | +8.50, +13.60, +12.40 | +11.50 | 3/3 |
| cnn6 | 0.625 | random_comp@after - random@after | +4.83, +5.37, +8.37 | +6.19 | 3/3 |
| cnn6 | 0.625 | greedy_comp@after - random_comp@after | +2.33, +10.33, +7.40 | +6.69 | 3/3 |
| cnn6 | 0.625 | greedy_comp@after - scratch@scratch | -21.70, -17.80, -13.10 | -17.53 | 0/3 |
| cnn6 | 0.625 | bn_pruned@after - scratch@scratch | -30.20, -31.40, -25.50 | -29.03 | 0/3 |
| cnn6 | 0.625 | random@after - scratch@scratch | -28.87, -33.50, -28.87 | -30.41 | 0/3 |
| real_cnn6 | 0.375 | greedy_comp@before - random@before | +55.42, +60.82, +72.70 | +62.98 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@before - bn_pruned@before | +62.60, +64.70, +68.20 | +65.17 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@before - greedy_pruned@before | +49.80, +38.40, +62.00 | +50.07 | 3/3 |
| real_cnn6 | 0.375 | random_comp@before - random@before | +18.72, +28.48, +38.90 | +28.70 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@before - random_comp@before | +36.70, +32.34, +33.80 | +34.28 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - random@after | +9.83, +7.40, +10.83 | +9.36 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - bn_pruned@after | +12.20, +7.90, +6.90 | +9.00 | 3/3 |
| real_cnn6 | 0.375 | random_comp@after - random@after | +4.17, +3.77, +4.50 | +4.14 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - random_comp@after | +5.67, +3.63, +6.33 | +5.21 | 3/3 |
| real_cnn6 | 0.375 | greedy_comp@after - scratch@scratch | +0.40, -3.80, +0.10 | -1.10 | 2/3 |
| real_cnn6 | 0.375 | bn_pruned@after - scratch@scratch | -11.80, -11.70, -6.80 | -10.10 | 0/3 |
| real_cnn6 | 0.375 | random@after - scratch@scratch | -9.43, -11.20, -10.73 | -10.46 | 0/3 |
| real_cnn6 | 0.625 | greedy_comp@before - random@before | +15.02, +13.10, +16.08 | +14.73 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@before - bn_pruned@before | +15.00, +10.80, +18.00 | +14.60 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@before - greedy_pruned@before | +13.00, +12.00, +7.70 | +10.90 | 3/3 |
| real_cnn6 | 0.625 | random_comp@before - random@before | +11.66, +11.66, +18.20 | +13.84 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@before - random_comp@before | +3.36, +1.44, -2.12 | +0.89 | 2/3 |
| real_cnn6 | 0.625 | greedy_comp@after - random@after | +26.37, +13.63, +14.73 | +18.24 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@after - bn_pruned@after | +42.30, +36.70, +45.40 | +41.47 | 3/3 |
| real_cnn6 | 0.625 | random_comp@after - random@after | +4.47, +5.47, +4.23 | +4.72 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@after - random_comp@after | +21.90, +8.17, +10.50 | +13.52 | 3/3 |
| real_cnn6 | 0.625 | greedy_comp@after - scratch@scratch | -8.00, -18.70, -12.30 | -13.00 | 0/3 |
| real_cnn6 | 0.625 | bn_pruned@after - scratch@scratch | -50.30, -55.40, -57.70 | -54.47 | 0/3 |
| real_cnn6 | 0.625 | random@after - scratch@scratch | -34.37, -32.33, -27.03 | -31.24 | 0/3 |

## Quaternion vs real at the same removed fraction (near-equal real parameters)

Descriptive: the families start from different baselines. Drop = accuracy minus the family's own baseline of the same stage (scratch is compared with the original baseline).

| Fraction | Policy and stage | Quaternion params | Real params | Quaternion acc | Real acc | Quaternion drop | Real drop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.375 | random@before | 29450 | 28950 | 15.11% | 14.95% | -81.55 | -81.28 |
| 0.375 | greedy_comp@before | 29450 | 28950 | 57.10% | 77.93% | -39.57 | -18.30 |
| 0.375 | random@after | 29450 | 28950 | 83.76% | 82.28% | -12.81 | -14.12 |
| 0.375 | bn_pruned@after | 29450 | 28950 | 86.17% | 82.63% | -10.40 | -13.77 |
| 0.375 | greedy_comp@after | 29450 | 28950 | 90.47% | 91.63% | -6.10 | -4.77 |
| 0.375 | scratch@scratch | 29450 | 28950 | 94.33% | 92.73% | -2.33 | -3.50 |
| 0.625 | random@before | 10978 | 10678 | 10.77% | 10.17% | -85.89 | -86.07 |
| 0.625 | greedy_comp@before | 10978 | 10678 | 25.43% | 24.90% | -71.23 | -71.33 |
| 0.625 | random@after | 10978 | 10678 | 59.92% | 55.89% | -36.64 | -40.51 |
| 0.625 | bn_pruned@after | 10978 | 10678 | 61.30% | 32.67% | -35.27 | -63.73 |
| 0.625 | greedy_comp@after | 10978 | 10678 | 72.80% | 74.13% | -23.77 | -22.27 |
| 0.625 | scratch@scratch | 10978 | 10678 | 90.33% | 87.13% | -6.33 | -9.10 |

## Verification and diagnostics

Exact-fold checks (physical compact model vs functional substitution, per site, 200 check images): 432 checks, maximum logit error 1.34e-05 (tolerance 0.001). Every compact model was reloaded into the standard narrower architecture with identical logits.

Energy of the unconstrained (greedy_full_projected) additions discarded by the Hamilton projection, mean over seeds, per site:

| Architecture | Fraction | Per site (shallow to deep) |
| --- | ---: | --- |
| cnn6 | 0.375 | 66.1%, 61.1%, 55.9%, 53.6%, 58.4%, 0.0% |
| cnn6 | 0.625 | 62.7%, 61.9%, 57.6%, 60.1%, 63.1%, 0.0% |

Figure: `quality_size_curve.png` (solid quaternion, dashed real).

## Limits

Small sequential CNNs, uniform fractions, one fixed ridge alpha and fine-tuning schedule, few seeds. Validation also selected the baseline and scratch checkpoints. Random rows average their repetitions within seed first.
