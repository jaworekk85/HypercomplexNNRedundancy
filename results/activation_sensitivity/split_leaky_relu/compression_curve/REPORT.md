# Quality-size curve: uniform multi-site compression

Status: completed. Dataset cifar; architectures cnn6; seeds [2026, 2027, 2028]. The same fraction [0.25, 0.5] of units is removed at every convolution output, shallow to deep, so each compact model is the standard architecture with narrower widths (checked by reloading it into that architecture). Compensation folds a ridge reconstruction (alpha 0.001) from the kept units of the same site into the consumer: right-Hamilton maps for quaternion models, scalar maps for real models. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. Fine-tuning: 3 epochs, final epoch, no selection. Scratch: same compact architecture, baseline trainer and budget, checkpoint selected by validation cross-entropy as for the baselines. No official-test inference. No novelty claim.

## Validation accuracy (mean ± seed SD)

| Architecture | Removed fraction | Policy | Real parameters | Without fine-tuning | After fine-tuning / from scratch |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn6 | 0.0 | baseline | 73898 | 78.80% ± 0.92 (n=3) | 78.40% ± 1.21 (n=3) |
| cnn6 | 0.25 | random | 42034 | 20.69% ± 3.87 (n=3) | 66.86% ± 0.40 (n=3) |
| cnn6 | 0.25 | bn_pruned | 42034 | 20.93% ± 3.30 (n=3) | 68.47% ± 1.00 (n=3) |
| cnn6 | 0.25 | greedy_pruned | 42034 | 29.17% ± 3.32 (n=3) | — |
| cnn6 | 0.25 | random_comp | 42034 | 45.09% ± 6.88 (n=3) | 68.60% ± 0.64 (n=3) |
| cnn6 | 0.25 | greedy_full_projected | 42034 | 41.83% ± 9.52 (n=3) | — |
| cnn6 | 0.25 | greedy_comp | 42034 | 60.03% ± 4.01 (n=3) | 72.43% ± 0.57 (n=3) |
| cnn6 | 0.5 | random | 19098 | 11.21% ± 0.80 (n=3) | 44.39% ± 0.78 (n=3) |
| cnn6 | 0.5 | bn_pruned | 19098 | 15.03% ± 2.75 (n=3) | 49.20% ± 0.46 (n=3) |
| cnn6 | 0.5 | greedy_pruned | 19098 | 15.00% ± 2.10 (n=3) | — |
| cnn6 | 0.5 | random_comp | 19098 | 15.98% ± 1.23 (n=3) | 46.53% ± 1.33 (n=3) |
| cnn6 | 0.5 | greedy_full_projected | 19098 | 11.90% ± 1.87 (n=3) | — |
| cnn6 | 0.5 | greedy_comp | 19098 | 25.47% ± 1.45 (n=3) | 54.23% ± 2.48 (n=3) |

## Paired effects (left minus right, validation pp, all seeds retained)

| Architecture | Removed fraction | Comparison | Per seed | Mean | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: |
| cnn6 | 0.25 | greedy_comp@before - random@before | +37.16, +35.16, +45.72 | +39.35 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - bn_pruned@before | +35.10, +36.00, +46.20 | +39.10 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_pruned@before | +29.80, +31.60, +31.20 | +30.87 | 3/3 |
| cnn6 | 0.25 | random_comp@before - random@before | +31.26, +22.90, +19.04 | +24.40 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - random_comp@before | +5.90, +12.26, +26.68 | +14.95 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_full_projected@before | +25.00, +14.00, +15.60 | +18.20 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random@after | +5.40, +5.73, +5.60 | +5.58 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - bn_pruned@after | +3.70, +3.30, +4.90 | +3.97 | 3/3 |
| cnn6 | 0.25 | random_comp@after - random@after | +2.93, +1.00, +1.30 | +1.74 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random_comp@after | +2.47, +4.73, +4.30 | +3.83 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - random@before | +14.96, +15.92, +11.90 | +14.26 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - bn_pruned@before | +7.30, +13.60, +10.40 | +10.43 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_pruned@before | +12.60, +11.90, +6.90 | +10.47 | 3/3 |
| cnn6 | 0.5 | random_comp@before - random@before | +5.94, +5.90, +2.48 | +4.77 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - random_comp@before | +9.02, +10.02, +9.42 | +9.49 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_full_projected@before | +11.50, +16.50, +12.70 | +13.57 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - random@after | +10.20, +6.83, +12.50 | +9.84 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - bn_pruned@after | +4.20, +2.70, +8.20 | +5.03 | 3/3 |
| cnn6 | 0.5 | random_comp@after - random@after | +3.27, +2.50, +0.67 | +2.14 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - random_comp@after | +6.93, +4.33, +11.83 | +7.70 | 3/3 |

## Verification and diagnostics

Exact-fold checks (physical compact model vs functional substitution, per site, 200 check images): 216 checks, maximum logit error 7.63e-06 (tolerance 0.001). Every compact model was reloaded into the standard narrower architecture with identical logits.

Energy of the unconstrained (greedy_full_projected) additions discarded by the Hamilton projection, mean over seeds, per site:

| Architecture | Fraction | Per site (shallow to deep) |
| --- | ---: | --- |
| cnn6 | 0.25 | 64.8%, 64.9%, 55.1%, 58.0%, 60.0%, 0.0% |
| cnn6 | 0.5 | 67.6%, 65.5%, 56.5%, 63.2%, 59.5%, 0.0% |

Figure: `quality_size_curve.png` (solid quaternion, dashed real).

## Limits

Small sequential CNNs, uniform fractions, one fixed ridge alpha and fine-tuning schedule, few seeds. Validation also selected the baseline and scratch checkpoints. Random rows average their repetitions within seed first.
