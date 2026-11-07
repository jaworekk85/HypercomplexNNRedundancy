# Activation sensitivity: split LeakyReLU and coupled MagnitudeTanh

## Hypothesis and fixed scope

Test whether the earlier compression conclusions depend on activation choice, within quaternion CNN6 on CIFAR. Three seeds, unchanged baseline 50-epoch trainer and parameter count; remove 25%/50% at every convolution output. Fit/folding policies and three-epoch fine-tuning match the existing ReLU reference, including the random victim streams. No activation-specific scratch runs or new official-test inference. MagnitudeTanh is prior art (PoppelbaumSchwung2024Activations, Eq. 21); no new activation/method is proposed.

## Measured results

| Activation | Fraction | Policy | Stage | Real params | Val mean ± seed SD | Mean CE | Change from own baseline (pp) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| split_relu | 0.0 | baseline | after | 73898 | 77.93% ± 0.93 | 0.638908 | +0.00 |
| split_relu | 0.0 | baseline | before | 73898 | 77.93% ± 1.10 | 0.637730 | +0.00 |
| split_relu | 0.25 | bn_pruned | after | 42034 | 69.27% ± 0.15 | 0.884337 | -8.67 |
| split_relu | 0.25 | bn_pruned | before | 42034 | 29.50% ± 6.32 | 2.243554 | -48.43 |
| split_relu | 0.25 | greedy_comp | after | 42034 | 71.50% ± 1.20 | 0.826508 | -6.43 |
| split_relu | 0.25 | greedy_comp | before | 42034 | 59.67% ± 3.21 | 1.154582 | -18.27 |
| split_relu | 0.25 | greedy_full_projected | before | 42034 | 33.73% ± 16.03 | 2.530583 | -44.20 |
| split_relu | 0.25 | greedy_pruned | before | 42034 | 23.47% ± 2.75 | 3.087088 | -54.47 |
| split_relu | 0.25 | random | after | 42034 | 67.49% ± 0.49 | 0.933978 | -10.44 |
| split_relu | 0.25 | random | before | 42034 | 21.45% ± 3.23 | 2.814841 | -56.49 |
| split_relu | 0.25 | random_comp | after | 42034 | 68.97% ± 1.22 | 0.888999 | -8.97 |
| split_relu | 0.25 | random_comp | before | 42034 | 46.19% ± 6.28 | 1.524227 | -31.74 |
| split_relu | 0.5 | bn_pruned | after | 19098 | 48.23% ± 0.51 | 1.500992 | -29.70 |
| split_relu | 0.5 | bn_pruned | before | 19098 | 11.33% ± 0.55 | 2.655971 | -66.60 |
| split_relu | 0.5 | greedy_comp | after | 19098 | 53.57% ± 0.65 | 1.316825 | -24.37 |
| split_relu | 0.5 | greedy_comp | before | 19098 | 21.97% ± 4.12 | 2.333431 | -55.97 |
| split_relu | 0.5 | greedy_full_projected | before | 19098 | 15.63% ± 2.56 | 4.152065 | -62.30 |
| split_relu | 0.5 | greedy_pruned | before | 19098 | 13.43% ± 1.55 | 3.296321 | -64.50 |
| split_relu | 0.5 | random | after | 19098 | 47.00% ± 2.69 | 1.529748 | -30.93 |
| split_relu | 0.5 | random | before | 19098 | 12.19% ± 0.24 | 2.426101 | -65.74 |
| split_relu | 0.5 | random_comp | after | 19098 | 47.12% ± 1.19 | 1.467548 | -30.81 |
| split_relu | 0.5 | random_comp | before | 19098 | 15.15% ± 3.29 | 2.645905 | -62.78 |
| split_leaky_relu | 0.0 | baseline | after | 73898 | 78.40% ± 1.21 | 0.628385 | +0.00 |
| split_leaky_relu | 0.0 | baseline | before | 73898 | 78.80% ± 0.92 | 0.623359 | +0.00 |
| split_leaky_relu | 0.25 | bn_pruned | after | 42034 | 68.47% ± 1.00 | 0.900818 | -9.93 |
| split_leaky_relu | 0.25 | bn_pruned | before | 42034 | 20.93% ± 3.30 | 3.036035 | -57.87 |
| split_leaky_relu | 0.25 | greedy_comp | after | 42034 | 72.43% ± 0.57 | 0.811649 | -5.97 |
| split_leaky_relu | 0.25 | greedy_comp | before | 42034 | 60.03% ± 4.01 | 1.154737 | -18.77 |
| split_leaky_relu | 0.25 | greedy_full_projected | before | 42034 | 41.83% ± 9.52 | 2.127334 | -36.97 |
| split_leaky_relu | 0.25 | greedy_pruned | before | 42034 | 29.17% ± 3.32 | 2.123148 | -49.63 |
| split_leaky_relu | 0.25 | random | after | 42034 | 66.86% ± 0.40 | 0.952432 | -11.54 |
| split_leaky_relu | 0.25 | random | before | 42034 | 20.69% ± 3.87 | 2.956896 | -58.11 |
| split_leaky_relu | 0.25 | random_comp | after | 42034 | 68.60% ± 0.64 | 0.896469 | -9.80 |
| split_leaky_relu | 0.25 | random_comp | before | 42034 | 45.09% ± 6.88 | 1.633116 | -33.71 |
| split_leaky_relu | 0.5 | bn_pruned | after | 19098 | 49.20% ± 0.46 | 1.468621 | -29.20 |
| split_leaky_relu | 0.5 | bn_pruned | before | 19098 | 15.03% ± 2.75 | 2.586316 | -63.77 |
| split_leaky_relu | 0.5 | greedy_comp | after | 19098 | 54.23% ± 2.48 | 1.303233 | -24.17 |
| split_leaky_relu | 0.5 | greedy_comp | before | 19098 | 25.47% ± 1.45 | 2.141417 | -53.33 |
| split_leaky_relu | 0.5 | greedy_full_projected | before | 19098 | 11.90% ± 1.87 | 7.217224 | -66.90 |
| split_leaky_relu | 0.5 | greedy_pruned | before | 19098 | 15.00% ± 2.10 | 2.632946 | -63.80 |
| split_leaky_relu | 0.5 | random | after | 19098 | 44.39% ± 0.78 | 1.550959 | -34.01 |
| split_leaky_relu | 0.5 | random | before | 19098 | 11.21% ± 0.80 | 2.572204 | -67.59 |
| split_leaky_relu | 0.5 | random_comp | after | 19098 | 46.53% ± 1.33 | 1.488739 | -31.87 |
| split_leaky_relu | 0.5 | random_comp | before | 19098 | 15.98% ± 1.23 | 2.660481 | -62.82 |
| magnitude_tanh | 0.0 | baseline | after | 73898 | 74.20% ± 0.69 | 0.723423 | +0.00 |
| magnitude_tanh | 0.0 | baseline | before | 73898 | 74.63% ± 0.85 | 0.720667 | +0.00 |
| magnitude_tanh | 0.25 | bn_pruned | after | 42034 | 65.10% ± 2.29 | 1.021044 | -9.10 |
| magnitude_tanh | 0.25 | bn_pruned | before | 42034 | 9.50% ± 3.04 | 5.344545 | -65.13 |
| magnitude_tanh | 0.25 | greedy_comp | after | 42034 | 68.50% ± 0.52 | 0.900888 | -5.70 |
| magnitude_tanh | 0.25 | greedy_comp | before | 42034 | 53.87% ± 4.45 | 1.343434 | -20.77 |
| magnitude_tanh | 0.25 | greedy_full_projected | before | 42034 | 26.47% ± 4.61 | 2.906907 | -48.17 |
| magnitude_tanh | 0.25 | greedy_pruned | before | 42034 | 19.43% ± 5.27 | 3.654354 | -55.20 |
| magnitude_tanh | 0.25 | random | after | 42034 | 65.03% ± 0.41 | 1.006000 | -9.17 |
| magnitude_tanh | 0.25 | random | before | 42034 | 13.51% ± 2.30 | 5.339046 | -61.12 |
| magnitude_tanh | 0.25 | random_comp | after | 42034 | 67.12% ± 0.18 | 0.945546 | -7.08 |
| magnitude_tanh | 0.25 | random_comp | before | 42034 | 41.43% ± 6.42 | 1.915149 | -33.21 |
| magnitude_tanh | 0.5 | bn_pruned | after | 19098 | 45.63% ± 1.86 | 1.517893 | -28.57 |
| magnitude_tanh | 0.5 | bn_pruned | before | 19098 | 9.57% ± 1.21 | 4.948267 | -65.07 |
| magnitude_tanh | 0.5 | greedy_comp | after | 19098 | 53.47% ± 1.35 | 1.306989 | -20.73 |
| magnitude_tanh | 0.5 | greedy_comp | before | 19098 | 25.00% ± 2.82 | 2.417678 | -49.63 |
| magnitude_tanh | 0.5 | greedy_full_projected | before | 19098 | 10.30% ± 0.35 | 6.189647 | -64.33 |
| magnitude_tanh | 0.5 | greedy_pruned | before | 19098 | 11.03% ± 1.62 | 6.032935 | -63.60 |
| magnitude_tanh | 0.5 | random | after | 19098 | 47.50% ± 0.44 | 1.471993 | -26.70 |
| magnitude_tanh | 0.5 | random | before | 19098 | 9.52% ± 0.55 | 4.445184 | -65.11 |
| magnitude_tanh | 0.5 | random_comp | after | 19098 | 51.01% ± 0.91 | 1.370832 | -23.19 |
| magnitude_tanh | 0.5 | random_comp | before | 19098 | 16.29% ± 2.96 | 3.398400 | -58.34 |

## Paired differences from ReLU, including unfavorable seeds

| Activation | Fraction | Policy | Stage | Per-seed difference (pp) | Mean (pp) |
| --- | ---: | --- | --- | --- | ---: |
| split_leaky_relu | 0.0 | baseline | after | +1.30, +1.70, -1.60 | +0.47 |
| split_leaky_relu | 0.0 | baseline | before | +1.30, +2.30, -1.00 | +0.87 |
| split_leaky_relu | 0.25 | bn_pruned | after | -1.20, +0.50, -1.70 | -0.80 |
| split_leaky_relu | 0.25 | bn_pruned | before | -9.30, +1.40, -17.80 | -8.57 |
| split_leaky_relu | 0.25 | greedy_comp | after | +0.30, +0.20, +2.30 | +0.93 |
| split_leaky_relu | 0.25 | greedy_comp | before | -3.50, -2.70, +7.30 | +0.37 |
| split_leaky_relu | 0.25 | greedy_full_projected | before | -13.50, +4.80, +33.00 | +8.10 |
| split_leaky_relu | 0.25 | greedy_pruned | before | +2.60, +2.50, +12.00 | +5.70 |
| split_leaky_relu | 0.25 | random | after | -1.43, +0.23, -0.70 | -0.63 |
| split_leaky_relu | 0.25 | random | before | -6.44, +5.60, -1.44 | -0.76 |
| split_leaky_relu | 0.25 | random_comp | after | -1.03, +0.00, -0.07 | -0.37 |
| split_leaky_relu | 0.25 | random_comp | before | -0.98, -0.48, -1.86 | -1.11 |
| split_leaky_relu | 0.5 | bn_pruned | after | +1.50, +1.50, -0.10 | +0.97 |
| split_leaky_relu | 0.5 | bn_pruned | before | +6.30, +2.50, +2.30 | +3.70 |
| split_leaky_relu | 0.5 | greedy_comp | after | -0.40, -0.90, +3.30 | +0.67 |
| split_leaky_relu | 0.5 | greedy_comp | before | +6.80, +0.30, +3.40 | +3.50 |
| split_leaky_relu | 0.5 | greedy_full_projected | before | -1.40, -2.80, -7.00 | -3.73 |
| split_leaky_relu | 0.5 | greedy_pruned | before | +1.00, +0.00, +3.70 | +1.57 |
| split_leaky_relu | 0.5 | random | after | -0.30, -3.33, -4.20 | -2.61 |
| split_leaky_relu | 0.5 | random | before | -1.76, -0.94, -0.26 | -0.99 |
| split_leaky_relu | 0.5 | random_comp | after | -0.40, -0.57, -0.80 | -0.59 |
| split_leaky_relu | 0.5 | random_comp | before | +2.10, -1.88, +2.26 | +0.83 |
| magnitude_tanh | 0.0 | baseline | after | -3.20, -3.10, -4.90 | -3.73 |
| magnitude_tanh | 0.0 | baseline | before | -2.80, -2.10, -5.00 | -3.30 |
| magnitude_tanh | 0.25 | bn_pruned | after | -3.30, -2.30, -6.90 | -4.17 |
| magnitude_tanh | 0.25 | bn_pruned | before | -17.90, -12.80, -29.30 | -20.00 |
| magnitude_tanh | 0.25 | greedy_comp | after | -2.40, -4.50, -2.10 | -3.00 |
| magnitude_tanh | 0.25 | greedy_comp | before | -4.80, -5.10, -7.50 | -5.80 |
| magnitude_tanh | 0.25 | greedy_full_projected | before | -17.10, -10.90, +6.20 | -7.27 |
| magnitude_tanh | 0.25 | greedy_pruned | before | +1.80, -8.30, -5.60 | -4.03 |
| magnitude_tanh | 0.25 | random | after | -3.07, -1.43, -2.87 | -2.46 |
| magnitude_tanh | 0.25 | random | before | -9.32, -8.28, -6.20 | -7.93 |
| magnitude_tanh | 0.25 | random_comp | after | -3.37, -0.83, -1.33 | -1.84 |
| magnitude_tanh | 0.25 | random_comp | before | -6.00, -3.24, -5.06 | -4.77 |
| magnitude_tanh | 0.5 | bn_pruned | after | -2.30, -0.40, -5.10 | -2.60 |
| magnitude_tanh | 0.5 | bn_pruned | before | -1.90, -2.60, -0.80 | -1.77 |
| magnitude_tanh | 0.5 | greedy_comp | after | -2.10, +0.60, +1.20 | -0.10 |
| magnitude_tanh | 0.5 | greedy_comp | before | +9.30, -2.00, +1.80 | +3.03 |
| magnitude_tanh | 0.5 | greedy_full_projected | before | -5.30, -2.50, -8.20 | -5.33 |
| magnitude_tanh | 0.5 | greedy_pruned | before | -1.90, -2.10, -3.20 | -2.40 |
| magnitude_tanh | 0.5 | random | after | +3.77, -1.50, -0.77 | +0.50 |
| magnitude_tanh | 0.5 | random | before | -2.60, -3.02, -2.40 | -2.67 |
| magnitude_tanh | 0.5 | random_comp | after | +2.70, +3.40, +5.57 | +3.89 |
| magnitude_tanh | 0.5 | random_comp | before | +0.44, -4.40, +7.38 | +1.14 |

## Interpretation limits

Activation changes baseline learning as well as compression behavior. Compare both absolute accuracy and loss from each activation's own baseline; preserve unfavorable seeds. The coupled Tanh also changes boundedness/saturation, so its contrast with ReLU does not isolate component coupling. There is no matched split-Tanh or real-valued coupled control in this bounded pilot. BN and pooling remain component-wise; no full-model equivariance is implied. One dataset/two removal levels and one fixed slope/epsilon do not establish general superiority. All per-policy artifacts and feature-fold checks remain in the two compression_curve subdirectories.
