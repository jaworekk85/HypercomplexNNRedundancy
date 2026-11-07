# Quality-size curve: uniform multi-site compression

Status: completed. Dataset eurosat; architectures cnn6, real_cnn6; seeds [2026, 2027, 2028]. The same fraction [0.125, 0.25, 0.5, 0.75, 0.875] of units is removed at every convolution output, shallow to deep, so each compact model is the standard architecture with narrower widths (checked by reloading it into that architecture). Compensation folds a ridge reconstruction (alpha 0.001) from the kept units of the same site into the consumer: right-Hamilton maps for quaternion models, scalar maps for real models. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. Fine-tuning: 3 epochs, final epoch, no selection. Scratch: same compact architecture, baseline trainer and budget, checkpoint selected by validation cross-entropy as for the baselines. No official-test inference. No novelty claim.

## Validation accuracy (mean ± seed SD)

| Architecture | Removed fraction | Policy | Real parameters | Without fine-tuning | After fine-tuning / from scratch |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn6 | 0.0 | baseline | 73898 | 96.67% ± 0.29 (n=3) | 96.57% ± 0.40 (n=3) |
| cnn6 | 0.125 | random | 56850 | 49.91% ± 7.34 (n=3) | 94.70% ± 0.64 (n=3) |
| cnn6 | 0.125 | bn_pruned | 56850 | 53.93% ± 9.97 (n=3) | 95.03% ± 0.15 (n=3) |
| cnn6 | 0.125 | greedy_pruned | 56850 | 54.53% ± 5.83 (n=3) | — |
| cnn6 | 0.125 | random_comp | 56850 | 90.05% ± 0.95 (n=3) | 94.92% ± 0.46 (n=3) |
| cnn6 | 0.125 | greedy_full_projected | 56850 | 81.70% ± 4.93 (n=3) | — |
| cnn6 | 0.125 | greedy_comp | 56850 | 92.83% ± 1.55 (n=3) | 95.30% ± 0.30 (n=3) |
| cnn6 | 0.25 | random | 42034 | 22.00% ± 0.90 (n=3) | 90.70% ± 0.47 (n=3) |
| cnn6 | 0.25 | bn_pruned | 42034 | 21.03% ± 1.94 (n=3) | 91.80% ± 1.14 (n=3) |
| cnn6 | 0.25 | greedy_pruned | 42034 | 32.03% ± 6.51 (n=3) | — |
| cnn6 | 0.25 | random_comp | 42034 | 71.86% ± 1.37 (n=3) | 92.29% ± 0.88 (n=3) |
| cnn6 | 0.25 | greedy_full_projected | 42034 | 57.63% ± 16.37 (n=3) | — |
| cnn6 | 0.25 | greedy_comp | 42034 | 78.13% ± 5.02 (n=3) | 93.73% ± 0.38 (n=3) |
| cnn6 | 0.25 | scratch | 42034 | — | 95.50% ± 0.10 (n=3) |
| cnn6 | 0.5 | random | 19098 | 12.53% ± 0.72 (n=3) | 76.37% ± 1.90 (n=3) |
| cnn6 | 0.5 | bn_pruned | 19098 | 14.93% ± 5.24 (n=3) | 79.17% ± 1.99 (n=3) |
| cnn6 | 0.5 | greedy_pruned | 19098 | 16.20% ± 4.52 (n=3) | — |
| cnn6 | 0.5 | random_comp | 19098 | 30.05% ± 0.85 (n=3) | 79.03% ± 1.20 (n=3) |
| cnn6 | 0.5 | greedy_full_projected | 19098 | 20.10% ± 5.30 (n=3) | — |
| cnn6 | 0.5 | greedy_comp | 19098 | 32.20% ± 0.72 (n=3) | 85.33% ± 2.85 (n=3) |
| cnn6 | 0.5 | scratch | 19098 | — | 92.47% ± 0.21 (n=3) |
| cnn6 | 0.75 | random | 5090 | 10.57% ± 1.08 (n=3) | 50.07% ± 4.58 (n=3) |
| cnn6 | 0.75 | bn_pruned | 5090 | 10.50% ± 1.04 (n=3) | 44.07% ± 4.21 (n=3) |
| cnn6 | 0.75 | greedy_pruned | 5090 | 12.73% ± 4.48 (n=3) | — |
| cnn6 | 0.75 | random_comp | 5090 | 14.02% ± 1.97 (n=3) | 48.39% ± 3.27 (n=3) |
| cnn6 | 0.75 | greedy_full_projected | 5090 | 13.23% ± 2.32 (n=3) | — |
| cnn6 | 0.75 | greedy_comp | 5090 | 18.43% ± 5.23 (n=3) | 50.37% ± 5.77 (n=3) |
| cnn6 | 0.75 | scratch | 5090 | — | 85.27% ± 2.48 (n=3) |
| cnn6 | 0.875 | random | 1434 | 10.47% ± 0.58 (n=3) | 30.20% ± 3.21 (n=3) |
| cnn6 | 0.875 | bn_pruned | 1434 | 10.50% ± 1.04 (n=3) | 18.67% ± 3.97 (n=3) |
| cnn6 | 0.875 | greedy_pruned | 1434 | 11.10% ± 0.00 (n=3) | — |
| cnn6 | 0.875 | random_comp | 1434 | 11.26% ± 1.27 (n=3) | 17.53% ± 2.30 (n=3) |
| cnn6 | 0.875 | greedy_full_projected | 1434 | 11.00% ± 0.17 (n=3) | — |
| cnn6 | 0.875 | greedy_comp | 1434 | 12.40% ± 2.25 (n=3) | 18.73% ± 3.15 (n=3) |
| cnn6 | 0.875 | scratch | 1434 | — | 66.47% ± 4.12 (n=3) |
| real_cnn6 | 0.0 | baseline | 73098 | 96.23% ± 0.38 (n=3) | 96.40% ± 0.70 (n=3) |
| real_cnn6 | 0.125 | random | 56150 | 55.60% ± 10.64 (n=3) | 94.20% ± 0.56 (n=3) |
| real_cnn6 | 0.125 | bn_pruned | 56150 | 60.40% ± 3.97 (n=3) | 95.13% ± 0.67 (n=3) |
| real_cnn6 | 0.125 | greedy_pruned | 56150 | 68.10% ± 10.28 (n=3) | — |
| real_cnn6 | 0.125 | random_comp | 56150 | 88.87% ± 0.49 (n=3) | 95.00% ± 0.39 (n=3) |
| real_cnn6 | 0.125 | greedy_comp | 56150 | 94.70% ± 0.70 (n=3) | 95.50% ± 0.66 (n=3) |
| real_cnn6 | 0.25 | random | 41434 | 25.44% ± 1.93 (n=3) | 89.77% ± 0.48 (n=3) |
| real_cnn6 | 0.25 | bn_pruned | 41434 | 30.00% ± 1.32 (n=3) | 91.60% ± 1.47 (n=3) |
| real_cnn6 | 0.25 | greedy_pruned | 41434 | 32.50% ± 9.04 (n=3) | — |
| real_cnn6 | 0.25 | random_comp | 41434 | 66.25% ± 2.40 (n=3) | 91.78% ± 0.39 (n=3) |
| real_cnn6 | 0.25 | greedy_comp | 41434 | 89.87% ± 2.25 (n=3) | 94.10% ± 1.48 (n=3) |
| real_cnn6 | 0.25 | scratch | 41434 | — | 93.63% ± 0.49 (n=3) |
| real_cnn6 | 0.5 | random | 18698 | 12.57% ± 0.51 (n=3) | 72.20% ± 4.04 (n=3) |
| real_cnn6 | 0.5 | bn_pruned | 18698 | 11.90% ± 2.76 (n=3) | 52.30% ± 6.96 (n=3) |
| real_cnn6 | 0.5 | greedy_pruned | 18698 | 18.00% ± 4.22 (n=3) | — |
| real_cnn6 | 0.5 | random_comp | 18698 | 31.27% ± 6.69 (n=3) | 76.84% ± 1.90 (n=3) |
| real_cnn6 | 0.5 | greedy_comp | 18698 | 54.33% ± 11.12 (n=3) | 85.23% ± 4.80 (n=3) |
| real_cnn6 | 0.5 | scratch | 18698 | — | 91.10% ± 0.78 (n=3) |
| real_cnn6 | 0.75 | random | 4890 | 10.84% ± 1.47 (n=3) | 43.81% ± 2.61 (n=3) |
| real_cnn6 | 0.75 | bn_pruned | 4890 | 11.73% ± 3.00 (n=3) | 21.73% ± 8.16 (n=3) |
| real_cnn6 | 0.75 | greedy_pruned | 4890 | 11.10% ± 0.00 (n=3) | — |
| real_cnn6 | 0.75 | random_comp | 4890 | 15.38% ± 2.53 (n=3) | 41.43% ± 6.28 (n=3) |
| real_cnn6 | 0.75 | greedy_comp | 4890 | 18.70% ± 6.95 (n=3) | 46.73% ± 9.18 (n=3) |
| real_cnn6 | 0.75 | scratch | 4890 | — | 79.50% ± 3.90 (n=3) |
| real_cnn6 | 0.875 | random | 1334 | 10.25% ± 0.76 (n=3) | 21.49% ± 2.70 (n=3) |
| real_cnn6 | 0.875 | bn_pruned | 1334 | 11.90% ± 3.08 (n=3) | 20.10% ± 4.88 (n=3) |
| real_cnn6 | 0.875 | greedy_pruned | 1334 | 11.10% ± 0.00 (n=3) | — |
| real_cnn6 | 0.875 | random_comp | 1334 | 11.65% ± 0.38 (n=3) | 13.41% ± 2.56 (n=3) |
| real_cnn6 | 0.875 | greedy_comp | 1334 | 11.37% ± 0.70 (n=3) | 22.23% ± 5.45 (n=3) |
| real_cnn6 | 0.875 | scratch | 1334 | — | 59.33% ± 3.41 (n=3) |

## Paired effects (left minus right, validation pp, all seeds retained)

| Architecture | Removed fraction | Comparison | Per seed | Mean | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: |
| cnn6 | 0.125 | greedy_comp@before - random@before | +33.72, +50.60, +44.44 | +42.92 | 3/3 |
| cnn6 | 0.125 | greedy_comp@before - bn_pruned@before | +48.20, +36.70, +31.80 | +38.90 | 3/3 |
| cnn6 | 0.125 | greedy_comp@before - greedy_pruned@before | +33.10, +35.50, +46.30 | +38.30 | 3/3 |
| cnn6 | 0.125 | random_comp@before - random@before | +33.76, +46.76, +39.88 | +40.13 | 3/3 |
| cnn6 | 0.125 | greedy_comp@before - random_comp@before | -0.04, +3.84, +4.56 | +2.79 | 2/3 |
| cnn6 | 0.125 | greedy_comp@before - greedy_full_projected@before | +7.50, +7.90, +18.00 | +11.13 | 3/3 |
| cnn6 | 0.125 | greedy_comp@after - random@after | +0.63, +1.03, +0.13 | +0.60 | 3/3 |
| cnn6 | 0.125 | greedy_comp@after - bn_pruned@after | +0.60, +0.10, +0.10 | +0.27 | 3/3 |
| cnn6 | 0.125 | random_comp@after - random@after | +0.03, +0.47, +0.17 | +0.22 | 3/3 |
| cnn6 | 0.125 | greedy_comp@after - random_comp@after | +0.60, +0.57, -0.03 | +0.38 | 2/3 |
| cnn6 | 0.25 | greedy_comp@before - random@before | +56.02, +61.94, +50.44 | +56.13 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - bn_pruned@before | +59.80, +60.90, +50.60 | +57.10 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_pruned@before | +46.90, +44.20, +47.20 | +46.10 | 3/3 |
| cnn6 | 0.25 | random_comp@before - random@before | +48.14, +50.52, +50.92 | +49.86 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - random_comp@before | +7.88, +11.42, -0.48 | +6.27 | 2/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_full_projected@before | +26.70, +6.80, +28.00 | +20.50 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random@after | +3.03, +2.80, +3.27 | +3.03 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - bn_pruned@after | +2.30, +2.70, +0.80 | +1.93 | 3/3 |
| cnn6 | 0.25 | random_comp@after - random@after | +1.57, +0.53, +2.67 | +1.59 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random_comp@after | +1.47, +2.27, +0.60 | +1.44 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - scratch@scratch | -2.20, -1.60, -1.50 | -1.77 | 0/3 |
| cnn6 | 0.25 | bn_pruned@after - scratch@scratch | -4.50, -4.30, -2.30 | -3.70 | 0/3 |
| cnn6 | 0.25 | random@after - scratch@scratch | -5.23, -4.40, -4.77 | -4.80 | 0/3 |
| cnn6 | 0.5 | greedy_comp@before - random@before | +20.56, +18.30, +20.14 | +19.67 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - bn_pruned@before | +21.90, +10.70, +19.20 | +17.27 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_pruned@before | +13.30, +13.80, +20.90 | +16.00 | 3/3 |
| cnn6 | 0.5 | random_comp@before - random@before | +16.86, +17.68, +18.02 | +17.52 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - random_comp@before | +3.70, +0.62, +2.12 | +2.15 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_full_projected@before | +13.10, +16.70, +6.50 | +12.10 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - random@after | +13.30, +4.37, +9.23 | +8.97 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - bn_pruned@after | +6.90, +5.20, +6.40 | +6.17 | 3/3 |
| cnn6 | 0.5 | random_comp@after - random@after | +3.67, +2.53, +1.80 | +2.67 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - random_comp@after | +9.63, +1.83, +7.43 | +6.30 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - scratch@scratch | -4.90, -10.60, -5.90 | -7.13 | 0/3 |
| cnn6 | 0.5 | bn_pruned@after - scratch@scratch | -11.80, -15.80, -12.30 | -13.30 | 0/3 |
| cnn6 | 0.5 | random@after - scratch@scratch | -18.20, -14.97, -15.13 | -16.10 | 0/3 |
| cnn6 | 0.75 | greedy_comp@before - random@before | +11.08, +0.68, +11.82 | +7.86 | 3/3 |
| cnn6 | 0.75 | greedy_comp@before - bn_pruned@before | +10.40, +1.30, +12.10 | +7.93 | 3/3 |
| cnn6 | 0.75 | greedy_comp@before - greedy_pruned@before | +12.20, +1.30, +3.60 | +5.70 | 3/3 |
| cnn6 | 0.75 | random_comp@before - random@before | +1.86, +1.90, +6.58 | +3.45 | 3/3 |
| cnn6 | 0.75 | greedy_comp@before - random_comp@before | +9.22, -1.22, +5.24 | +4.41 | 2/3 |
| cnn6 | 0.75 | greedy_comp@before - greedy_full_projected@before | +8.60, +1.30, +5.70 | +5.20 | 3/3 |
| cnn6 | 0.75 | greedy_comp@after - random@after | -1.17, -9.20, +11.27 | +0.30 | 1/3 |
| cnn6 | 0.75 | greedy_comp@after - bn_pruned@after | +9.30, -2.70, +12.30 | +6.30 | 2/3 |
| cnn6 | 0.75 | random_comp@after - random@after | -1.87, -2.90, -0.27 | -1.68 | 0/3 |
| cnn6 | 0.75 | greedy_comp@after - random_comp@after | +0.70, -6.30, +11.53 | +1.98 | 2/3 |
| cnn6 | 0.75 | greedy_comp@after - scratch@scratch | -33.40, -41.20, -30.10 | -34.90 | 0/3 |
| cnn6 | 0.75 | bn_pruned@after - scratch@scratch | -42.70, -38.50, -42.40 | -41.20 | 0/3 |
| cnn6 | 0.75 | random@after - scratch@scratch | -32.23, -32.00, -41.37 | -35.20 | 0/3 |
| cnn6 | 0.875 | greedy_comp@before - random@before | +0.74, +3.90, +1.14 | +1.93 | 3/3 |
| cnn6 | 0.875 | greedy_comp@before - bn_pruned@before | +0.00, +3.90, +1.80 | +1.90 | 2/3 |
| cnn6 | 0.875 | greedy_comp@before - greedy_pruned@before | +0.00, +3.90, +0.00 | +1.30 | 1/3 |
| cnn6 | 0.875 | random_comp@before - random@before | +0.58, +1.56, +0.22 | +0.79 | 3/3 |
| cnn6 | 0.875 | greedy_comp@before - random_comp@before | +0.16, +2.34, +0.92 | +1.14 | 3/3 |
| cnn6 | 0.875 | greedy_comp@before - greedy_full_projected@before | +0.00, +3.90, +0.30 | +1.40 | 2/3 |
| cnn6 | 0.875 | greedy_comp@after - random@after | -12.93, -10.47, -11.00 | -11.47 | 0/3 |
| cnn6 | 0.875 | greedy_comp@after - bn_pruned@after | -4.10, +4.00, +0.30 | +0.07 | 2/3 |
| cnn6 | 0.875 | random_comp@after - random@after | -14.37, -17.00, -6.63 | -12.67 | 0/3 |
| cnn6 | 0.875 | greedy_comp@after - random_comp@after | +1.43, +6.53, -4.37 | +1.20 | 2/3 |
| cnn6 | 0.875 | greedy_comp@after - scratch@scratch | -50.70, -40.00, -52.50 | -47.73 | 0/3 |
| cnn6 | 0.875 | bn_pruned@after - scratch@scratch | -46.60, -44.00, -52.80 | -47.80 | 0/3 |
| cnn6 | 0.875 | random@after - scratch@scratch | -37.77, -29.53, -41.50 | -36.27 | 0/3 |
| real_cnn6 | 0.125 | greedy_comp@before - random@before | +27.36, +40.12, +49.82 | +39.10 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@before - bn_pruned@before | +29.40, +35.10, +38.40 | +34.30 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@before - greedy_pruned@before | +14.50, +34.30, +31.00 | +26.60 | 3/3 |
| real_cnn6 | 0.125 | random_comp@before - random@before | +22.36, +34.82, +42.62 | +33.27 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@before - random_comp@before | +5.00, +5.30, +7.20 | +5.83 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@after - random@after | +0.77, +1.03, +2.10 | +1.30 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@after - bn_pruned@after | +0.30, +0.40, +0.40 | +0.37 | 3/3 |
| real_cnn6 | 0.125 | random_comp@after - random@after | +0.27, +0.80, +1.33 | +0.80 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@after - random_comp@after | +0.50, +0.23, +0.77 | +0.50 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - random@before | +63.44, +62.78, +67.06 | +64.43 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - bn_pruned@before | +56.90, +61.90, +60.80 | +59.87 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - greedy_pruned@before | +45.40, +66.40, +60.30 | +57.37 | 3/3 |
| real_cnn6 | 0.25 | random_comp@before - random@before | +39.62, +39.32, +43.50 | +40.81 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - random_comp@before | +23.82, +23.46, +23.56 | +23.61 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - random@after | +5.37, +2.13, +5.50 | +4.33 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - bn_pruned@after | +2.60, +2.50, +2.40 | +2.50 | 3/3 |
| real_cnn6 | 0.25 | random_comp@after - random@after | +2.33, +1.07, +2.63 | +2.01 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - random_comp@after | +3.03, +1.07, +2.87 | +2.32 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - scratch@scratch | +0.90, -1.00, +1.50 | +0.47 | 2/3 |
| real_cnn6 | 0.25 | bn_pruned@after - scratch@scratch | -1.70, -3.50, -0.90 | -2.03 | 0/3 |
| real_cnn6 | 0.25 | random@after - scratch@scratch | -4.47, -3.13, -4.00 | -3.87 | 0/3 |
| real_cnn6 | 0.5 | greedy_comp@before - random@before | +32.44, +39.02, +53.82 | +41.76 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@before - bn_pruned@before | +35.20, +40.50, +51.60 | +42.43 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@before - greedy_pruned@before | +21.90, +37.80, +49.30 | +36.33 | 3/3 |
| real_cnn6 | 0.5 | random_comp@before - random@before | +13.62, +16.36, +26.10 | +18.69 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@before - random_comp@before | +18.82, +22.66, +27.72 | +23.07 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - random@after | +14.63, +11.93, +12.53 | +13.03 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - bn_pruned@after | +43.50, +22.70, +32.60 | +32.93 | 3/3 |
| real_cnn6 | 0.5 | random_comp@after - random@after | +2.07, +8.60, +3.27 | +4.64 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - random_comp@after | +12.57, +3.33, +9.27 | +8.39 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - scratch@scratch | -2.80, -11.00, -3.80 | -5.87 | 0/3 |
| real_cnn6 | 0.5 | bn_pruned@after - scratch@scratch | -46.30, -33.70, -36.40 | -38.80 | 0/3 |
| real_cnn6 | 0.5 | random@after - scratch@scratch | -17.43, -22.93, -16.33 | -18.90 | 0/3 |
| real_cnn6 | 0.75 | greedy_comp@before - random@before | +12.14, -1.84, +13.28 | +7.86 | 2/3 |
| real_cnn6 | 0.75 | greedy_comp@before - bn_pruned@before | +7.10, -0.40, +14.20 | +6.97 | 2/3 |
| real_cnn6 | 0.75 | greedy_comp@before - greedy_pruned@before | +11.00, -0.40, +12.20 | +7.60 | 2/3 |
| real_cnn6 | 0.75 | random_comp@before - random@before | +2.92, +5.40, +5.30 | +4.54 | 3/3 |
| real_cnn6 | 0.75 | greedy_comp@before - random_comp@before | +9.22, -7.24, +7.98 | +3.32 | 2/3 |
| real_cnn6 | 0.75 | greedy_comp@after - random@after | -7.37, +0.63, +15.50 | +2.92 | 2/3 |
| real_cnn6 | 0.75 | greedy_comp@after - bn_pruned@after | +7.90, +32.10, +35.00 | +25.00 | 3/3 |
| real_cnn6 | 0.75 | random_comp@after - random@after | -11.13, -1.00, +5.00 | -2.38 | 1/3 |
| real_cnn6 | 0.75 | greedy_comp@after - random_comp@after | +3.77, +1.63, +10.50 | +5.30 | 3/3 |
| real_cnn6 | 0.75 | greedy_comp@after - scratch@scratch | -42.50, -29.30, -26.50 | -32.77 | 0/3 |
| real_cnn6 | 0.75 | bn_pruned@after - scratch@scratch | -50.40, -61.40, -61.50 | -57.77 | 0/3 |
| real_cnn6 | 0.75 | random@after - scratch@scratch | -35.13, -29.93, -42.00 | -35.69 | 0/3 |
| real_cnn6 | 0.875 | greedy_comp@before - random@before | +2.10, -0.40, +1.66 | +1.12 | 2/3 |
| real_cnn6 | 0.875 | greedy_comp@before - bn_pruned@before | -3.20, -0.40, +2.00 | -0.53 | 1/3 |
| real_cnn6 | 0.875 | greedy_comp@before - greedy_pruned@before | +1.00, -0.40, +0.20 | +0.27 | 2/3 |
| real_cnn6 | 0.875 | random_comp@before - random@before | +1.22, +0.66, +2.32 | +1.40 | 3/3 |
| real_cnn6 | 0.875 | greedy_comp@before - random_comp@before | +0.88, -1.06, -0.66 | -0.28 | 1/3 |
| real_cnn6 | 0.875 | greedy_comp@after - random@after | -0.80, -2.17, +5.20 | +0.74 | 1/3 |
| real_cnn6 | 0.875 | greedy_comp@after - bn_pruned@after | -1.30, -4.70, +12.40 | +2.13 | 1/3 |
| real_cnn6 | 0.875 | random_comp@after - random@after | -11.27, -7.23, -5.73 | -8.08 | 0/3 |
| real_cnn6 | 0.875 | greedy_comp@after - random_comp@after | +10.47, +5.07, +10.93 | +8.82 | 3/3 |
| real_cnn6 | 0.875 | greedy_comp@after - scratch@scratch | -38.00, -45.10, -28.20 | -37.10 | 0/3 |
| real_cnn6 | 0.875 | bn_pruned@after - scratch@scratch | -36.70, -40.40, -40.60 | -39.23 | 0/3 |
| real_cnn6 | 0.875 | random@after - scratch@scratch | -37.20, -42.93, -33.40 | -37.84 | 0/3 |

## Quaternion vs real at the same removed fraction (near-equal real parameters)

Descriptive: the families start from different baselines. Drop = accuracy minus the family's own baseline of the same stage (scratch is compared with the original baseline).

| Fraction | Policy and stage | Quaternion params | Real params | Quaternion acc | Real acc | Quaternion drop | Real drop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.125 | random@before | 56850 | 56150 | 49.91% | 55.60% | -46.75 | -40.63 |
| 0.125 | greedy_comp@before | 56850 | 56150 | 92.83% | 94.70% | -3.83 | -1.53 |
| 0.125 | random@after | 56850 | 56150 | 94.70% | 94.20% | -1.87 | -2.20 |
| 0.125 | bn_pruned@after | 56850 | 56150 | 95.03% | 95.13% | -1.53 | -1.27 |
| 0.125 | greedy_comp@after | 56850 | 56150 | 95.30% | 95.50% | -1.27 | -0.90 |
| 0.25 | random@before | 42034 | 41434 | 22.00% | 25.44% | -74.67 | -70.79 |
| 0.25 | greedy_comp@before | 42034 | 41434 | 78.13% | 89.87% | -18.53 | -6.37 |
| 0.25 | random@after | 42034 | 41434 | 90.70% | 89.77% | -5.87 | -6.63 |
| 0.25 | bn_pruned@after | 42034 | 41434 | 91.80% | 91.60% | -4.77 | -4.80 |
| 0.25 | greedy_comp@after | 42034 | 41434 | 93.73% | 94.10% | -2.83 | -2.30 |
| 0.25 | scratch@scratch | 42034 | 41434 | 95.50% | 93.63% | -1.17 | -2.60 |
| 0.5 | random@before | 19098 | 18698 | 12.53% | 12.57% | -84.13 | -83.66 |
| 0.5 | greedy_comp@before | 19098 | 18698 | 32.20% | 54.33% | -64.47 | -41.90 |
| 0.5 | random@after | 19098 | 18698 | 76.37% | 72.20% | -20.20 | -24.20 |
| 0.5 | bn_pruned@after | 19098 | 18698 | 79.17% | 52.30% | -17.40 | -44.10 |
| 0.5 | greedy_comp@after | 19098 | 18698 | 85.33% | 85.23% | -11.23 | -11.17 |
| 0.5 | scratch@scratch | 19098 | 18698 | 92.47% | 91.10% | -4.20 | -5.13 |
| 0.75 | random@before | 5090 | 4890 | 10.57% | 10.84% | -86.09 | -85.39 |
| 0.75 | greedy_comp@before | 5090 | 4890 | 18.43% | 18.70% | -78.23 | -77.53 |
| 0.75 | random@after | 5090 | 4890 | 50.07% | 43.81% | -46.50 | -52.59 |
| 0.75 | bn_pruned@after | 5090 | 4890 | 44.07% | 21.73% | -52.50 | -74.67 |
| 0.75 | greedy_comp@after | 5090 | 4890 | 50.37% | 46.73% | -46.20 | -49.67 |
| 0.75 | scratch@scratch | 5090 | 4890 | 85.27% | 79.50% | -11.40 | -16.73 |
| 0.875 | random@before | 1434 | 1334 | 10.47% | 10.25% | -86.19 | -85.99 |
| 0.875 | greedy_comp@before | 1434 | 1334 | 12.40% | 11.37% | -84.27 | -84.87 |
| 0.875 | random@after | 1434 | 1334 | 30.20% | 21.49% | -66.37 | -74.91 |
| 0.875 | bn_pruned@after | 1434 | 1334 | 18.67% | 20.10% | -77.90 | -76.30 |
| 0.875 | greedy_comp@after | 1434 | 1334 | 18.73% | 22.23% | -77.83 | -74.17 |
| 0.875 | scratch@scratch | 1434 | 1334 | 66.47% | 59.33% | -30.20 | -36.90 |

## Verification and diagnostics

Exact-fold checks (physical compact model vs functional substitution, per site, 200 check images): 1080 checks, maximum logit error 1.72e-05 (tolerance 0.001). Every compact model was reloaded into the standard narrower architecture with identical logits.

Energy of the unconstrained (greedy_full_projected) additions discarded by the Hamilton projection, mean over seeds, per site:

| Architecture | Fraction | Per site (shallow to deep) |
| --- | ---: | --- |
| cnn6 | 0.125 | 64.2%, 66.3%, 51.5%, 46.5%, 58.0%, 0.0% |
| cnn6 | 0.25 | 61.7%, 67.3%, 53.5%, 51.0%, 59.5%, 0.0% |
| cnn6 | 0.5 | 63.4%, 60.9%, 55.6%, 55.7%, 62.4%, 0.0% |
| cnn6 | 0.75 | 60.5%, 61.5%, 61.0%, 58.9%, 63.3%, 0.0% |
| cnn6 | 0.875 | 68.2%, 58.8%, 53.5%, 61.6%, 68.5%, 0.0% |

Figure: `quality_size_curve.png` (solid quaternion, dashed real).

## Limits

Small sequential CNNs, uniform fractions, one fixed ridge alpha and fine-tuning schedule, few seeds. Validation also selected the baseline and scratch checkpoints. Random rows average their repetitions within seed first.
