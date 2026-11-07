# Quality-size curve: uniform multi-site compression

Status: completed. Dataset cifar; architectures cnn6, real_cnn6; seeds [2026, 2027, 2028]. The same fraction [0.125, 0.25, 0.5, 0.75, 0.875] of units is removed at every convolution output, shallow to deep, so each compact model is the standard architecture with narrower widths (checked by reloading it into that architecture). Compensation folds a ridge reconstruction (alpha 0.001) from the kept units of the same site into the consumer: right-Hamilton maps for quaternion models, scalar maps for real models. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. Fine-tuning: 3 epochs, final epoch, no selection. Scratch: same compact architecture, baseline trainer and budget, checkpoint selected by validation cross-entropy as for the baselines. No official-test inference. No novelty claim.

## Validation accuracy (mean ± seed SD)

| Architecture | Removed fraction | Policy | Real parameters | Without fine-tuning | After fine-tuning / from scratch |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn6 | 0.0 | baseline | 73898 | 77.93% ± 1.10 (n=3) | 77.93% ± 0.93 (n=3) |
| cnn6 | 0.125 | random | 56850 | 40.31% ± 3.34 (n=3) | 73.30% ± 1.27 (n=3) |
| cnn6 | 0.125 | bn_pruned | 56850 | 43.57% ± 6.57 (n=3) | 74.97% ± 1.20 (n=3) |
| cnn6 | 0.125 | greedy_pruned | 56850 | 49.23% ± 8.64 (n=3) | — |
| cnn6 | 0.125 | random_comp | 56850 | 67.50% ± 2.50 (n=3) | 73.86% ± 1.29 (n=3) |
| cnn6 | 0.125 | greedy_full_projected | 56850 | 60.83% ± 8.81 (n=3) | — |
| cnn6 | 0.125 | greedy_comp | 56850 | 72.20% ± 0.44 (n=3) | 75.43% ± 1.46 (n=3) |
| cnn6 | 0.25 | random | 42034 | 21.45% ± 3.23 (n=3) | 67.49% ± 0.49 (n=3) |
| cnn6 | 0.25 | bn_pruned | 42034 | 29.50% ± 6.32 (n=3) | 69.27% ± 0.15 (n=3) |
| cnn6 | 0.25 | greedy_pruned | 42034 | 23.47% ± 2.75 (n=3) | — |
| cnn6 | 0.25 | random_comp | 42034 | 46.19% ± 6.28 (n=3) | 68.97% ± 1.22 (n=3) |
| cnn6 | 0.25 | greedy_full_projected | 42034 | 33.73% ± 16.03 (n=3) | — |
| cnn6 | 0.25 | greedy_comp | 42034 | 59.67% ± 3.21 (n=3) | 71.50% ± 1.20 (n=3) |
| cnn6 | 0.25 | scratch | 42034 | — | 71.57% ± 0.72 (n=3) |
| cnn6 | 0.5 | random | 19098 | 12.19% ± 0.24 (n=3) | 47.00% ± 2.69 (n=3) |
| cnn6 | 0.5 | bn_pruned | 19098 | 11.33% ± 0.55 (n=3) | 48.23% ± 0.51 (n=3) |
| cnn6 | 0.5 | greedy_pruned | 19098 | 13.43% ± 1.55 (n=3) | — |
| cnn6 | 0.5 | random_comp | 19098 | 15.15% ± 3.29 (n=3) | 47.12% ± 1.19 (n=3) |
| cnn6 | 0.5 | greedy_full_projected | 19098 | 15.63% ± 2.56 (n=3) | — |
| cnn6 | 0.5 | greedy_comp | 19098 | 21.97% ± 4.12 (n=3) | 53.57% ± 0.65 (n=3) |
| cnn6 | 0.5 | scratch | 19098 | — | 63.90% ± 1.08 (n=3) |
| cnn6 | 0.75 | random | 5090 | 9.93% ± 0.21 (n=3) | 20.57% ± 2.49 (n=3) |
| cnn6 | 0.75 | bn_pruned | 5090 | 10.03% ± 0.06 (n=3) | 21.77% ± 1.45 (n=3) |
| cnn6 | 0.75 | greedy_pruned | 5090 | 10.00% ± 0.00 (n=3) | — |
| cnn6 | 0.75 | random_comp | 5090 | 10.12% ± 0.19 (n=3) | 16.52% ± 1.61 (n=3) |
| cnn6 | 0.75 | greedy_full_projected | 5090 | 9.80% ± 0.53 (n=3) | — |
| cnn6 | 0.75 | greedy_comp | 5090 | 10.03% ± 0.06 (n=3) | 18.50% ± 4.14 (n=3) |
| cnn6 | 0.75 | scratch | 5090 | — | 46.17% ± 1.48 (n=3) |
| cnn6 | 0.875 | random | 1434 | 10.00% ± 0.00 (n=3) | 13.91% ± 0.50 (n=3) |
| cnn6 | 0.875 | bn_pruned | 1434 | 10.00% ± 0.00 (n=3) | 13.60% ± 1.95 (n=3) |
| cnn6 | 0.875 | greedy_pruned | 1434 | 10.00% ± 0.00 (n=3) | — |
| cnn6 | 0.875 | random_comp | 1434 | 9.97% ± 0.22 (n=3) | 12.29% ± 0.60 (n=3) |
| cnn6 | 0.875 | greedy_full_projected | 1434 | 10.00% ± 0.00 (n=3) | — |
| cnn6 | 0.875 | greedy_comp | 1434 | 10.17% ± 0.29 (n=3) | 13.57% ± 1.65 (n=3) |
| cnn6 | 0.875 | scratch | 1434 | — | 31.93% ± 3.62 (n=3) |
| real_cnn6 | 0.0 | baseline | 72954 | 75.53% ± 1.27 (n=3) | 75.63% ± 0.80 (n=3) |
| real_cnn6 | 0.125 | random | 56024 | 41.25% ± 3.51 (n=3) | 70.99% ± 0.48 (n=3) |
| real_cnn6 | 0.125 | bn_pruned | 56024 | 43.93% ± 8.42 (n=3) | 72.30% ± 0.44 (n=3) |
| real_cnn6 | 0.125 | greedy_pruned | 56024 | 57.63% ± 8.60 (n=3) | — |
| real_cnn6 | 0.125 | random_comp | 56024 | 66.46% ± 0.22 (n=3) | 71.70% ± 0.38 (n=3) |
| real_cnn6 | 0.125 | greedy_comp | 56024 | 71.27% ± 1.53 (n=3) | 73.73% ± 1.23 (n=3) |
| real_cnn6 | 0.25 | random | 41326 | 20.31% ± 5.51 (n=3) | 63.58% ± 2.91 (n=3) |
| real_cnn6 | 0.25 | bn_pruned | 41326 | 23.13% ± 9.28 (n=3) | 66.37% ± 1.91 (n=3) |
| real_cnn6 | 0.25 | greedy_pruned | 41326 | 34.27% ± 10.05 (n=3) | — |
| real_cnn6 | 0.25 | random_comp | 41326 | 49.39% ± 1.96 (n=3) | 66.12% ± 1.21 (n=3) |
| real_cnn6 | 0.25 | greedy_comp | 41326 | 62.03% ± 1.99 (n=3) | 70.40% ± 1.31 (n=3) |
| real_cnn6 | 0.25 | scratch | 41326 | — | 69.60% ± 1.11 (n=3) |
| real_cnn6 | 0.5 | random | 18626 | 11.41% ± 1.68 (n=3) | 40.39% ± 2.69 (n=3) |
| real_cnn6 | 0.5 | bn_pruned | 18626 | 11.40% ± 2.09 (n=3) | 39.33% ± 2.26 (n=3) |
| real_cnn6 | 0.5 | greedy_pruned | 18626 | 15.30% ± 3.95 (n=3) | — |
| real_cnn6 | 0.5 | random_comp | 18626 | 17.31% ± 2.07 (n=3) | 45.03% ± 3.25 (n=3) |
| real_cnn6 | 0.5 | greedy_comp | 18626 | 32.03% ± 2.54 (n=3) | 56.70% ± 0.26 (n=3) |
| real_cnn6 | 0.5 | scratch | 18626 | — | 59.53% ± 0.47 (n=3) |
| real_cnn6 | 0.75 | random | 4854 | 10.03% ± 0.65 (n=3) | 19.53% ± 2.35 (n=3) |
| real_cnn6 | 0.75 | bn_pruned | 4854 | 10.67% ± 1.24 (n=3) | 19.33% ± 1.52 (n=3) |
| real_cnn6 | 0.75 | greedy_pruned | 4854 | 9.87% ± 0.23 (n=3) | — |
| real_cnn6 | 0.75 | random_comp | 4854 | 10.77% ± 0.54 (n=3) | 17.09% ± 1.63 (n=3) |
| real_cnn6 | 0.75 | greedy_comp | 4854 | 11.97% ± 2.90 (n=3) | 24.27% ± 3.68 (n=3) |
| real_cnn6 | 0.75 | scratch | 4854 | — | 39.50% ± 1.55 (n=3) |
| real_cnn6 | 0.875 | random | 1316 | 10.00% ± 0.00 (n=3) | 12.72% ± 1.01 (n=3) |
| real_cnn6 | 0.875 | bn_pruned | 1316 | 10.93% ± 1.62 (n=3) | 13.37% ± 2.97 (n=3) |
| real_cnn6 | 0.875 | greedy_pruned | 1316 | 10.00% ± 0.00 (n=3) | — |
| real_cnn6 | 0.875 | random_comp | 1316 | 10.14% ± 0.28 (n=3) | 11.17% ± 0.50 (n=3) |
| real_cnn6 | 0.875 | greedy_comp | 1316 | 11.30% ± 2.25 (n=3) | 15.40% ± 0.36 (n=3) |
| real_cnn6 | 0.875 | scratch | 1316 | — | 25.63% ± 3.29 (n=3) |

## Paired effects (left minus right, validation pp, all seeds retained)

| Architecture | Removed fraction | Comparison | Per seed | Mean | Positive seeds |
| --- | ---: | --- | --- | ---: | ---: |
| cnn6 | 0.125 | greedy_comp@before - random@before | +29.66, +30.46, +35.54 | +31.89 | 3/3 |
| cnn6 | 0.125 | greedy_comp@before - bn_pruned@before | +32.90, +32.10, +20.90 | +28.63 | 3/3 |
| cnn6 | 0.125 | greedy_comp@before - greedy_pruned@before | +23.60, +14.40, +30.90 | +22.97 | 3/3 |
| cnn6 | 0.125 | random_comp@before - random@before | +28.02, +23.16, +30.38 | +27.19 | 3/3 |
| cnn6 | 0.125 | greedy_comp@before - random_comp@before | +1.64, +7.30, +5.16 | +4.70 | 3/3 |
| cnn6 | 0.125 | greedy_comp@before - greedy_full_projected@before | +5.20, +7.60, +21.30 | +11.37 | 3/3 |
| cnn6 | 0.125 | greedy_comp@after - random@after | +2.27, +1.77, +2.37 | +2.13 | 3/3 |
| cnn6 | 0.125 | greedy_comp@after - bn_pruned@after | +2.10, +0.30, -1.00 | +0.47 | 2/3 |
| cnn6 | 0.125 | random_comp@after - random@after | +0.47, +0.30, +0.90 | +0.56 | 3/3 |
| cnn6 | 0.125 | greedy_comp@after - random_comp@after | +1.80, +1.47, +1.47 | +1.58 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - random@before | +34.22, +43.46, +36.98 | +38.22 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - bn_pruned@before | +29.30, +40.10, +21.10 | +30.17 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_pruned@before | +35.90, +36.80, +35.90 | +36.20 | 3/3 |
| cnn6 | 0.25 | random_comp@before - random@before | +25.80, +28.98, +19.46 | +24.75 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - random_comp@before | +8.42, +14.48, +17.52 | +13.47 | 3/3 |
| cnn6 | 0.25 | greedy_comp@before - greedy_full_projected@before | +15.00, +21.50, +41.30 | +25.93 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random@after | +3.67, +5.77, +2.60 | +4.01 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - bn_pruned@after | +2.20, +3.60, +0.90 | +2.23 | 3/3 |
| cnn6 | 0.25 | random_comp@after - random@after | +2.53, +1.23, +0.67 | +1.48 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - random_comp@after | +1.13, +4.53, +1.93 | +2.53 | 3/3 |
| cnn6 | 0.25 | greedy_comp@after - scratch@scratch | -0.90, +1.60, -0.90 | -0.07 | 1/3 |
| cnn6 | 0.25 | bn_pruned@after - scratch@scratch | -3.10, -2.00, -1.80 | -2.30 | 0/3 |
| cnn6 | 0.25 | random@after - scratch@scratch | -4.57, -4.17, -3.50 | -4.08 | 0/3 |
| cnn6 | 0.5 | greedy_comp@before - random@before | +6.40, +14.68, +8.24 | +9.77 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - bn_pruned@before | +6.80, +15.80, +9.30 | +10.63 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_pruned@before | +6.80, +11.60, +7.20 | +8.53 | 3/3 |
| cnn6 | 0.5 | random_comp@before - random@before | +2.08, +6.84, -0.04 | +2.96 | 2/3 |
| cnn6 | 0.5 | greedy_comp@before - random_comp@before | +4.32, +7.84, +8.28 | +6.81 | 3/3 |
| cnn6 | 0.5 | greedy_comp@before - greedy_full_projected@before | +3.30, +13.40, +2.30 | +6.33 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - random@after | +10.30, +4.40, +5.00 | +6.57 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - bn_pruned@after | +6.10, +5.10, +4.80 | +5.33 | 3/3 |
| cnn6 | 0.5 | random_comp@after - random@after | +3.37, -0.27, -2.73 | +0.12 | 1/3 |
| cnn6 | 0.5 | greedy_comp@after - random_comp@after | +6.93, +4.67, +7.73 | +6.44 | 3/3 |
| cnn6 | 0.5 | greedy_comp@after - scratch@scratch | -8.80, -12.20, -10.00 | -10.33 | 0/3 |
| cnn6 | 0.5 | bn_pruned@after - scratch@scratch | -14.90, -17.30, -14.80 | -15.67 | 0/3 |
| cnn6 | 0.5 | random@after - scratch@scratch | -19.10, -16.60, -15.00 | -16.90 | 0/3 |
| cnn6 | 0.75 | greedy_comp@before - random@before | +0.40, +0.00, -0.10 | +0.10 | 1/3 |
| cnn6 | 0.75 | greedy_comp@before - bn_pruned@before | +0.10, -0.10, +0.00 | +0.00 | 1/3 |
| cnn6 | 0.75 | greedy_comp@before - greedy_pruned@before | +0.10, +0.00, +0.00 | +0.03 | 1/3 |
| cnn6 | 0.75 | random_comp@before - random@before | +0.32, +0.34, -0.10 | +0.19 | 2/3 |
| cnn6 | 0.75 | greedy_comp@before - random_comp@before | +0.08, -0.34, +0.00 | -0.09 | 1/3 |
| cnn6 | 0.75 | greedy_comp@before - greedy_full_projected@before | -0.10, +0.80, +0.00 | +0.23 | 1/3 |
| cnn6 | 0.75 | greedy_comp@after - random@after | +3.40, -3.07, -6.53 | -2.07 | 1/3 |
| cnn6 | 0.75 | greedy_comp@after - bn_pruned@after | -0.20, -3.10, -6.50 | -3.27 | 0/3 |
| cnn6 | 0.75 | random_comp@after - random@after | -2.77, -4.80, -4.57 | -4.04 | 0/3 |
| cnn6 | 0.75 | greedy_comp@after - random_comp@after | +6.17, +1.73, -1.97 | +1.98 | 2/3 |
| cnn6 | 0.75 | greedy_comp@after - scratch@scratch | -23.30, -25.70, -34.00 | -27.67 | 0/3 |
| cnn6 | 0.75 | bn_pruned@after - scratch@scratch | -23.10, -22.60, -27.50 | -24.40 | 0/3 |
| cnn6 | 0.75 | random@after - scratch@scratch | -26.70, -22.63, -27.47 | -25.60 | 0/3 |
| cnn6 | 0.875 | greedy_comp@before - random@before | +0.00, +0.50, +0.00 | +0.17 | 1/3 |
| cnn6 | 0.875 | greedy_comp@before - bn_pruned@before | +0.00, +0.50, +0.00 | +0.17 | 1/3 |
| cnn6 | 0.875 | greedy_comp@before - greedy_pruned@before | +0.00, +0.50, +0.00 | +0.17 | 1/3 |
| cnn6 | 0.875 | random_comp@before - random@before | +0.18, -0.26, +0.00 | -0.03 | 1/3 |
| cnn6 | 0.875 | greedy_comp@before - random_comp@before | -0.18, +0.76, +0.00 | +0.19 | 1/3 |
| cnn6 | 0.875 | greedy_comp@before - greedy_full_projected@before | +0.00, +0.50, +0.00 | +0.17 | 1/3 |
| cnn6 | 0.875 | greedy_comp@after - random@after | +1.47, -1.30, -1.20 | -0.34 | 1/3 |
| cnn6 | 0.875 | greedy_comp@after - bn_pruned@after | +2.50, -2.70, +0.10 | -0.03 | 2/3 |
| cnn6 | 0.875 | random_comp@after - random@after | -1.13, -2.77, -0.97 | -1.62 | 0/3 |
| cnn6 | 0.875 | greedy_comp@after - random_comp@after | +2.60, +1.47, -0.23 | +1.28 | 2/3 |
| cnn6 | 0.875 | greedy_comp@after - scratch@scratch | -14.70, -16.50, -23.90 | -18.37 | 0/3 |
| cnn6 | 0.875 | bn_pruned@after - scratch@scratch | -17.20, -13.80, -24.00 | -18.33 | 0/3 |
| cnn6 | 0.875 | random@after - scratch@scratch | -16.17, -15.20, -22.70 | -18.02 | 0/3 |
| real_cnn6 | 0.125 | greedy_comp@before - random@before | +25.16, +35.04, +29.86 | +30.02 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@before - bn_pruned@before | +18.90, +38.30, +24.80 | +27.33 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@before - greedy_pruned@before | +7.30, +10.60, +23.00 | +13.63 | 3/3 |
| real_cnn6 | 0.125 | random_comp@before - random@before | +21.74, +28.50, +25.40 | +25.21 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@before - random_comp@before | +3.42, +6.54, +4.46 | +4.81 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@after - random@after | +1.70, +3.63, +2.90 | +2.74 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@after - bn_pruned@after | +0.60, +3.10, +0.60 | +1.43 | 3/3 |
| real_cnn6 | 0.125 | random_comp@after - random@after | +0.33, +0.63, +1.17 | +0.71 | 3/3 |
| real_cnn6 | 0.125 | greedy_comp@after - random_comp@after | +1.37, +3.00, +1.73 | +2.03 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - random@before | +37.42, +38.52, +49.22 | +41.72 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - bn_pruned@before | +28.90, +42.90, +44.90 | +38.90 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - greedy_pruned@before | +20.10, +22.70, +40.50 | +27.77 | 3/3 |
| real_cnn6 | 0.25 | random_comp@before - random@before | +26.38, +27.02, +33.84 | +29.08 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@before - random_comp@before | +11.04, +11.50, +15.38 | +12.64 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - random@after | +4.07, +6.03, +10.37 | +6.82 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - bn_pruned@after | +4.80, +4.50, +2.80 | +4.03 | 3/3 |
| real_cnn6 | 0.25 | random_comp@after - random@after | +1.97, +1.17, +4.50 | +2.54 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - random_comp@after | +2.10, +4.87, +5.87 | +4.28 | 3/3 |
| real_cnn6 | 0.25 | greedy_comp@after - scratch@scratch | -1.80, +3.00, +1.20 | +0.80 | 2/3 |
| real_cnn6 | 0.25 | bn_pruned@after - scratch@scratch | -6.60, -1.50, -1.60 | -3.23 | 0/3 |
| real_cnn6 | 0.25 | random@after - scratch@scratch | -5.87, -3.03, -9.17 | -6.02 | 0/3 |
| real_cnn6 | 0.5 | greedy_comp@before - random@before | +22.66, +19.08, +20.12 | +20.62 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@before - bn_pruned@before | +19.80, +18.70, +23.40 | +20.63 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@before - greedy_pruned@before | +22.30, +13.70, +14.20 | +16.73 | 3/3 |
| real_cnn6 | 0.5 | random_comp@before - random@before | +8.70, +6.62, +2.38 | +5.90 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@before - random_comp@before | +13.96, +12.46, +17.74 | +14.72 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - random@after | +13.87, +15.47, +19.60 | +16.31 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - bn_pruned@after | +17.40, +14.90, +19.80 | +17.37 | 3/3 |
| real_cnn6 | 0.5 | random_comp@after - random@after | +5.33, +4.47, +4.13 | +4.64 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - random_comp@after | +8.53, +11.00, +15.47 | +11.67 | 3/3 |
| real_cnn6 | 0.5 | greedy_comp@after - scratch@scratch | -3.20, -3.30, -2.00 | -2.83 | 0/3 |
| real_cnn6 | 0.5 | bn_pruned@after - scratch@scratch | -20.60, -18.20, -21.80 | -20.20 | 0/3 |
| real_cnn6 | 0.5 | random@after - scratch@scratch | -17.07, -18.77, -21.60 | -19.14 | 0/3 |
| real_cnn6 | 0.75 | greedy_comp@before - random@before | +0.60, -0.70, +5.90 | +1.93 | 2/3 |
| real_cnn6 | 0.75 | greedy_comp@before - bn_pruned@before | -1.50, +0.10, +5.30 | +1.30 | 2/3 |
| real_cnn6 | 0.75 | greedy_comp@before - greedy_pruned@before | +0.60, +0.40, +5.30 | +2.10 | 3/3 |
| real_cnn6 | 0.75 | random_comp@before - random@before | +0.16, +0.26, +1.80 | +0.74 | 3/3 |
| real_cnn6 | 0.75 | greedy_comp@before - random_comp@before | +0.44, -0.96, +4.10 | +1.19 | 2/3 |
| real_cnn6 | 0.75 | greedy_comp@after - random@after | +6.63, +0.83, +6.73 | +4.73 | 3/3 |
| real_cnn6 | 0.75 | greedy_comp@after - bn_pruned@after | +7.50, +1.30, +6.00 | +4.93 | 3/3 |
| real_cnn6 | 0.75 | random_comp@after - random@after | -2.67, -3.43, -1.23 | -2.44 | 0/3 |
| real_cnn6 | 0.75 | greedy_comp@after - random_comp@after | +9.30, +4.27, +7.97 | +7.18 | 3/3 |
| real_cnn6 | 0.75 | greedy_comp@after - scratch@scratch | -9.80, -20.20, -15.70 | -15.23 | 0/3 |
| real_cnn6 | 0.75 | bn_pruned@after - scratch@scratch | -17.30, -21.50, -21.70 | -20.17 | 0/3 |
| real_cnn6 | 0.75 | random@after - scratch@scratch | -16.43, -21.03, -22.43 | -19.97 | 0/3 |
| real_cnn6 | 0.875 | greedy_comp@before - random@before | +3.90, +0.00, +0.00 | +1.30 | 1/3 |
| real_cnn6 | 0.875 | greedy_comp@before - bn_pruned@before | +3.90, +0.00, -2.80 | +0.37 | 1/3 |
| real_cnn6 | 0.875 | greedy_comp@before - greedy_pruned@before | +3.90, +0.00, +0.00 | +1.30 | 1/3 |
| real_cnn6 | 0.875 | random_comp@before - random@before | -0.04, +0.46, +0.00 | +0.14 | 1/3 |
| real_cnn6 | 0.875 | greedy_comp@before - random_comp@before | +3.94, -0.46, +0.00 | +1.16 | 1/3 |
| real_cnn6 | 0.875 | greedy_comp@after - random@after | +1.80, +2.30, +3.93 | +2.68 | 3/3 |
| real_cnn6 | 0.875 | greedy_comp@after - bn_pruned@after | +3.40, -1.10, +3.80 | +2.03 | 2/3 |
| real_cnn6 | 0.875 | random_comp@after - random@after | -1.83, -2.80, -0.03 | -1.56 | 0/3 |
| real_cnn6 | 0.875 | greedy_comp@after - random_comp@after | +3.63, +5.10, +3.97 | +4.23 | 3/3 |
| real_cnn6 | 0.875 | greedy_comp@after - scratch@scratch | -6.90, -12.40, -11.40 | -10.23 | 0/3 |
| real_cnn6 | 0.875 | bn_pruned@after - scratch@scratch | -10.30, -11.30, -15.20 | -12.27 | 0/3 |
| real_cnn6 | 0.875 | random@after - scratch@scratch | -8.70, -14.70, -15.33 | -12.91 | 0/3 |

## Quaternion vs real at the same removed fraction (near-equal real parameters)

Descriptive: the families start from different baselines. Drop = accuracy minus the family's own baseline of the same stage (scratch is compared with the original baseline).

| Fraction | Policy and stage | Quaternion params | Real params | Quaternion acc | Real acc | Quaternion drop | Real drop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.125 | random@before | 56850 | 56024 | 40.31% | 41.25% | -37.62 | -34.29 |
| 0.125 | greedy_comp@before | 56850 | 56024 | 72.20% | 71.27% | -5.73 | -4.27 |
| 0.125 | random@after | 56850 | 56024 | 73.30% | 70.99% | -4.63 | -4.64 |
| 0.125 | bn_pruned@after | 56850 | 56024 | 74.97% | 72.30% | -2.97 | -3.33 |
| 0.125 | greedy_comp@after | 56850 | 56024 | 75.43% | 73.73% | -2.50 | -1.90 |
| 0.25 | random@before | 42034 | 41326 | 21.45% | 20.31% | -56.49 | -55.22 |
| 0.25 | greedy_comp@before | 42034 | 41326 | 59.67% | 62.03% | -18.27 | -13.50 |
| 0.25 | random@after | 42034 | 41326 | 67.49% | 63.58% | -10.44 | -12.06 |
| 0.25 | bn_pruned@after | 42034 | 41326 | 69.27% | 66.37% | -8.67 | -9.27 |
| 0.25 | greedy_comp@after | 42034 | 41326 | 71.50% | 70.40% | -6.43 | -5.23 |
| 0.25 | scratch@scratch | 42034 | 41326 | 71.57% | 69.60% | -6.37 | -5.93 |
| 0.5 | random@before | 19098 | 18626 | 12.19% | 11.41% | -65.74 | -64.12 |
| 0.5 | greedy_comp@before | 19098 | 18626 | 21.97% | 32.03% | -55.97 | -43.50 |
| 0.5 | random@after | 19098 | 18626 | 47.00% | 40.39% | -30.93 | -35.24 |
| 0.5 | bn_pruned@after | 19098 | 18626 | 48.23% | 39.33% | -29.70 | -36.30 |
| 0.5 | greedy_comp@after | 19098 | 18626 | 53.57% | 56.70% | -24.37 | -18.93 |
| 0.5 | scratch@scratch | 19098 | 18626 | 63.90% | 59.53% | -14.03 | -16.00 |
| 0.75 | random@before | 5090 | 4854 | 9.93% | 10.03% | -68.00 | -65.50 |
| 0.75 | greedy_comp@before | 5090 | 4854 | 10.03% | 11.97% | -67.90 | -63.57 |
| 0.75 | random@after | 5090 | 4854 | 20.57% | 19.53% | -57.37 | -56.10 |
| 0.75 | bn_pruned@after | 5090 | 4854 | 21.77% | 19.33% | -56.17 | -56.30 |
| 0.75 | greedy_comp@after | 5090 | 4854 | 18.50% | 24.27% | -59.43 | -51.37 |
| 0.75 | scratch@scratch | 5090 | 4854 | 46.17% | 39.50% | -31.77 | -36.03 |
| 0.875 | random@before | 1434 | 1316 | 10.00% | 10.00% | -67.93 | -65.53 |
| 0.875 | greedy_comp@before | 1434 | 1316 | 10.17% | 11.30% | -67.77 | -64.23 |
| 0.875 | random@after | 1434 | 1316 | 13.91% | 12.72% | -64.02 | -62.91 |
| 0.875 | bn_pruned@after | 1434 | 1316 | 13.60% | 13.37% | -64.33 | -62.27 |
| 0.875 | greedy_comp@after | 1434 | 1316 | 13.57% | 15.40% | -64.37 | -60.23 |
| 0.875 | scratch@scratch | 1434 | 1316 | 31.93% | 25.63% | -46.00 | -49.90 |

## Verification and diagnostics

Exact-fold checks (physical compact model vs functional substitution, per site, 200 check images): 1080 checks, maximum logit error 1.03e-05 (tolerance 0.001). Every compact model was reloaded into the standard narrower architecture with identical logits.

Energy of the unconstrained (greedy_full_projected) additions discarded by the Hamilton projection, mean over seeds, per site:

| Architecture | Fraction | Per site (shallow to deep) |
| --- | ---: | --- |
| cnn6 | 0.125 | 69.7%, 58.3%, 56.0%, 51.7%, 60.2%, 0.0% |
| cnn6 | 0.25 | 65.3%, 64.3%, 52.9%, 54.2%, 62.0%, 0.0% |
| cnn6 | 0.5 | 68.2%, 61.2%, 58.3%, 59.1%, 66.1%, 0.0% |
| cnn6 | 0.75 | 70.7%, 64.5%, 63.3%, 66.5%, 70.5%, 0.0% |
| cnn6 | 0.875 | 70.5%, 57.5%, 68.5%, 70.4%, 72.2%, 0.0% |

Figure: `quality_size_curve.png` (solid quaternion, dashed real).

## Limits

Small sequential CNNs, uniform fractions, one fixed ridge alpha and fine-tuning schedule, few seeds. Validation also selected the baseline and scratch checkpoints. Random rows average their repetitions within seed first.
