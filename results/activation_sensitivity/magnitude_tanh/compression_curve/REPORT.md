# Quality-size curve: uniform multi-site compression

Status: completed. Dataset cifar; architectures cnn6; seeds [2026, 2027, 2028]. The same fraction [0.25, 0.5] of units is removed at every convolution output, shallow to deep, so each compact model is the standard architecture with narrower widths (checked by reloading it into that architecture). Compensation folds a ridge reconstruction (alpha 0.001) from the kept units of the same site into the consumer: right-Hamilton maps for quaternion models, scalar maps for real models. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. Fine-tuning: 3 epochs, final epoch, no selection. Scratch: same compact architecture, baseline trainer and budget, checkpoint selected by validation cross-entropy as for the baselines. No official-test inference. No novelty claim.

## Validation accuracy (mean ± seed SD)

| Architecture | Removed fraction | Policy | Real parameters | Without fine-tuning | After fine-tuning / from scratch |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn6 | 0.0 | baseline | 73898 | 74.63% ± 0.85 (n=3) | 74.20% ± 0.69 (n=3) |
| cnn6 | 0.25 | random | 42034 | 13.51% ± 2.30 (n=3) | 65.03% ± 0.41 (n=3) |
| cnn6 | 0.25 | bn_pruned | 42034 | 9.50% ± 3.04 (n=3) | 65.10% ± 2.29 (n=3) |
| cnn6 | 0.25 | greedy_pruned | 42034 | 19.43% ± 5.27 (n=3) | — |
| cnn6 | 0.25 | random_comp | 42034 | 41.43% ± 6.42 (n=3) | 67.12% ± 0.18 (n=3) |
| cnn6 | 0.25 | greedy_full_projected | 42034 | 26.47% ± 4.61 (n=3) | — |
| cnn6 | 0.25 | greedy_comp | 42034 | 53.87% ± 4.45 (n=3) | 68.50% ± 0.52 (n=3) |
| cnn6 | 0.5 | random | 19098 | 9.52% ± 0.55 (n=3) | 47.50% ± 0.44 (n=3) |
| cnn6 | 0.5 | bn_pruned | 19098 | 9.57% ± 1.21 (n=3) | 45.63% ± 1.86 (n=3) |
| cnn6 | 0.5 | greedy_pruned | 19098 | 11.03% ± 1.62 (n=3) | — |
| cnn6 | 0.5 | random_comp | 19098 | 16.29% ± 2.96 (n=3) | 51.01% ± 0.91 (n=3) |
| cnn6 | 0.5 | greedy_full_projected | 19098 | 10.30% ± 0.35 (n=3) | — |
| cnn6 | 0.5 | greedy_comp | 19098 | 25.00% ± 2.82 (n=3) | 53.47% ± 1.35 (n=3) |

## Paired effects (left minus right, validation pp, all seeds retained)

| Architecture | Removed fraction | Comparison | Per seed | Mean | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: |
| cnn6 | 0.25 | greedy_comp@before - random@before | +38.74, +46.64, +35.68 | +40.35 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - bn_pruned@before | +42.40, +47.80, +42.90 | +44.37 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_pruned@before | +29.30, +40.00, +34.00 | +34.43 | 3/3 |
| cnn6 | 0.25 | random_comp@before - random@before | +29.12, +34.02, +20.60 | +27.91 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - random_comp@before | +9.62, +12.62, +15.08 | +12.44 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_full_projected@before | +27.30, +27.30, +27.60 | +27.40 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random@after | +4.33, +2.70, +3.37 | +3.47 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - bn_pruned@after | +3.10, +1.40, +5.70 | +3.40 | 3/3 |
| cnn6 | 0.25 | random_comp@after - random@after | +2.23, +1.83, +2.20 | +2.09 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random_comp@after | +2.10, +0.87, +1.17 | +1.38 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - random@before | +18.30, +15.70, +12.44 | +15.48 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - bn_pruned@before | +18.00, +16.40, +11.90 | +15.43 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_pruned@before | +18.00, +11.70, +12.20 | +13.97 | 3/3 |
| cnn6 | 0.5 | random_comp@before - random@before | +5.12, +5.46, +9.74 | +6.77 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - random_comp@before | +13.18, +10.24, +2.70 | +8.71 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_full_projected@before | +17.90, +13.90, +12.30 | +14.70 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - random@after | +4.43, +6.50, +6.97 | +5.97 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - bn_pruned@after | +6.30, +6.10, +11.10 | +7.83 | 3/3 |
| cnn6 | 0.5 | random_comp@after - random@after | +2.30, +4.63, +3.60 | +3.51 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - random_comp@after | +2.13, +1.87, +3.37 | +2.46 | 3/3 |

## Verification and diagnostics

Exact-fold checks (physical compact model vs functional substitution, per site, 200 check images): 216 checks, maximum logit error 6.68e-06 (tolerance 0.001). Every compact model was reloaded into the standard narrower architecture with identical logits.

Energy of the unconstrained (greedy_full_projected) additions discarded by the Hamilton projection, mean over seeds, per site:

| Architecture | Fraction | Per site (shallow to deep) |
| --- | ---: | --- |
| cnn6 | 0.25 | 65.9%, 64.0%, 43.9%, 39.5%, 40.0%, 0.0% |
| cnn6 | 0.5 | 65.3%, 63.3%, 46.1%, 45.8%, 45.5%, 0.0% |

Figure: `quality_size_curve.png` (solid quaternion, dashed real).

## Limits

Small sequential CNNs, uniform fractions, one fixed ridge alpha and fine-tuning schedule, few seeds. Validation also selected the baseline and scratch checkpoints. Random rows average their repetitions within seed first.
