# Internal merging: replication across CNN depths and compression levels

## Hypotheses and prior outcome

The cost of restricting full-map compensation to Hamilton blocks may persist across removal levels and CNN depths. Direct constrained fitting and data-weighted projection are equivalent; their agreement is a strong control, not a new algorithm. The earlier CNN6/four-removal cell is reused exploratory evidence, not independent replication.

## Prespecified protocol

CNN6: three existing 50-epoch baselines, widths 8/16/32. CNN8: three new 50-epoch baselines, four blocks with widths 8/16/24/32. Both are sequential BN/ReLU/max-pooling CNNs; architecture diversity is limited to depth/width within that family. Seeds 2026/2027/2028, same CIFAR split, normalization, augmentation, AdamW and batch/epoch budget. Choose baseline checkpoint by validation CE. No official-test inference; validation was used for checkpoint selection and is not a new blind benchmark.

Architectures differ in parameter count and compute; equal epoch budgets do not make them matched-cost baselines. Compression comparisons are paired within each architecture and training seed.

Fit zero-intercept maps on 500 even calibration images, check on 500 odd images. Use nested prefixes of four/eight common disjoint fit-only pairs, selected by full-map residual times outgoing kernel energy. Freeze all manifests before compressed-model validation. Compare paired pruning, direct-Q folding, unweighted projection, data-moment projection, and ten nested random removals per condition. No compression optimizer updates or BN refit. Full substitution is a functional oracle without a compact-architecture claim. Zero-offset Q/full fits use four/sixteen real calibration coefficients per pair; these maps are folded into the compact network and are not extra inference parameters. Report actual budgets per architecture, not just removed-unit fractions.

## Completed new baseline trainings

| Architecture | Seed | Epochs | Selected epoch | Validation accuracy | Validation CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| cnn8 | 2026 | 50 | 50 | 81.40% | 0.591847 |
| cnn8 | 2027 | 50 | 44 | 79.90% | 0.570401 |
| cnn8 | 2028 | 50 | 49 | 79.60% | 0.579250 |

## Measured validation results

| Architecture | Removed units | Policy | Real parameters | Accuracy mean ± seed SD | Mean CE |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn6 | 4 | baseline | 73898 | 77.93% ± 1.10 pp | 0.637730 |
| cnn6 | 4 | random | 66954 | 73.22% ± 0.65 pp | 0.757720 |
| cnn6 | 4 | paired_pruned | 66954 | 71.40% ± 2.63 pp | 0.828642 |
| cnn6 | 4 | quaternion_exact | 66954 | 74.73% ± 0.83 pp | 0.719896 |
| cnn6 | 4 | full_projected | 66954 | 73.63% ± 0.65 pp | 0.738425 |
| cnn6 | 4 | weighted_projected | 66954 | 74.73% ± 0.83 pp | 0.719896 |
| cnn6 | 4 | full_substitution | functional oracle | 76.30% ± 0.26 pp | 0.683890 |
| cnn6 | 8 | baseline | 73898 | 77.93% ± 1.10 pp | 0.637730 |
| cnn6 | 8 | random | 60010 | 53.97% ± 2.60 pp | 1.306785 |
| cnn6 | 8 | paired_pruned | 60010 | 44.53% ± 6.86 pp | 1.694390 |
| cnn6 | 8 | quaternion_exact | 60010 | 65.20% ± 2.41 pp | 0.994173 |
| cnn6 | 8 | full_projected | 60010 | 61.57% ± 3.13 pp | 1.097703 |
| cnn6 | 8 | weighted_projected | 60010 | 65.20% ± 2.41 pp | 0.994173 |
| cnn6 | 8 | full_substitution | functional oracle | 70.77% ± 1.77 pp | 0.854888 |
| cnn8 | 4 | baseline | 118058 | 80.30% ± 0.96 pp | 0.580500 |
| cnn8 | 4 | random | 112266 | 77.00% ± 1.19 pp | 0.659548 |
| cnn8 | 4 | paired_pruned | 112266 | 75.30% ± 0.36 pp | 0.696013 |
| cnn8 | 4 | quaternion_exact | 112266 | 78.27% ± 1.45 pp | 0.628693 |
| cnn8 | 4 | full_projected | 112266 | 77.20% ± 1.92 pp | 0.649362 |
| cnn8 | 4 | weighted_projected | 112266 | 78.27% ± 1.45 pp | 0.628693 |
| cnn8 | 4 | full_substitution | functional oracle | 79.03% ± 1.32 pp | 0.612283 |
| cnn8 | 8 | baseline | 118058 | 80.30% ± 0.96 pp | 0.580500 |
| cnn8 | 8 | random | 106474 | 65.84% ± 4.85 pp | 0.974803 |
| cnn8 | 8 | paired_pruned | 106474 | 61.67% ± 6.56 pp | 1.132116 |
| cnn8 | 8 | quaternion_exact | 106474 | 73.43% ± 1.19 pp | 0.777802 |
| cnn8 | 8 | full_projected | 106474 | 69.40% ± 1.23 pp | 0.874735 |
| cnn8 | 8 | weighted_projected | 106474 | 73.43% ± 1.19 pp | 0.777802 |
| cnn8 | 8 | full_substitution | functional oracle | 75.63% ± 1.01 pp | 0.712802 |

Random repetitions are averaged within training seed before SD across three seeds.

## Secondary classification check on 500 held-out calibration images

These images are disjoint from the 500 fit images and from validation. Check labels did not select the maps or compact policies. Structural consumer errors below also use this half.

| Architecture | Removed | Policy | Check accuracy mean ± seed SD | Check mean CE |
| --- | ---: | --- | ---: | ---: |
| cnn6 | 4 | baseline | 79.00% ± 1.40 pp | 0.626882 |
| cnn6 | 4 | random | 75.13% ± 1.31 pp | 0.734119 |
| cnn6 | 4 | paired_pruned | 71.40% ± 1.06 pp | 0.819309 |
| cnn6 | 4 | quaternion_exact | 75.07% ± 1.10 pp | 0.720623 |
| cnn6 | 4 | full_projected | 74.87% ± 1.92 pp | 0.736594 |
| cnn6 | 4 | weighted_projected | 75.07% ± 1.10 pp | 0.720623 |
| cnn6 | 4 | full_substitution | 76.87% ± 2.14 pp | 0.682379 |
| cnn6 | 8 | baseline | 79.00% ± 1.40 pp | 0.626882 |
| cnn6 | 8 | random | 56.36% ± 2.20 pp | 1.260102 |
| cnn6 | 8 | paired_pruned | 48.67% ± 6.96 pp | 1.653327 |
| cnn6 | 8 | quaternion_exact | 67.13% ± 1.15 pp | 0.986455 |
| cnn6 | 8 | full_projected | 63.27% ± 1.79 pp | 1.080318 |
| cnn6 | 8 | weighted_projected | 67.13% ± 1.15 pp | 0.986455 |
| cnn6 | 8 | full_substitution | 71.40% ± 0.92 pp | 0.866157 |
| cnn8 | 4 | baseline | 82.87% ± 0.92 pp | 0.532465 |
| cnn8 | 4 | random | 79.52% ± 1.12 pp | 0.614157 |
| cnn8 | 4 | paired_pruned | 79.33% ± 0.99 pp | 0.631670 |
| cnn8 | 4 | quaternion_exact | 81.20% ± 0.72 pp | 0.581989 |
| cnn8 | 4 | full_projected | 80.60% ± 0.20 pp | 0.595643 |
| cnn8 | 4 | weighted_projected | 81.20% ± 0.72 pp | 0.581989 |
| cnn8 | 4 | full_substitution | 81.80% ± 1.00 pp | 0.562284 |
| cnn8 | 8 | baseline | 82.87% ± 0.92 pp | 0.532465 |
| cnn8 | 8 | random | 68.55% ± 4.63 pp | 0.921737 |
| cnn8 | 8 | paired_pruned | 63.20% ± 6.08 pp | 1.059195 |
| cnn8 | 8 | quaternion_exact | 75.07% ± 0.31 pp | 0.752281 |
| cnn8 | 8 | full_projected | 72.20% ± 2.16 pp | 0.823952 |
| cnn8 | 8 | weighted_projected | 75.07% ± 0.31 pp | 0.752281 |
| cnn8 | 8 | full_substitution | 77.07% ± 1.81 pp | 0.688308 |

## Structural diagnostics and strong control

| Architecture | Removed | Seed | Discarded added-block energy | Unweighted/oracle consumer MSE | Direct-Q/oracle consumer MSE | Weighted/direct map error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cnn6 | 4 | 2026 | 55.66% | 1.2815 | 1.2113 | 5.05e-15 |
| cnn6 | 8 | 2026 | 59.46% | 1.3257 | 1.2346 | 7.33e-15 |
| cnn6 | 4 | 2027 | 57.49% | 1.3474 | 1.2932 | 5.38e-15 |
| cnn6 | 8 | 2027 | 55.23% | 1.3077 | 1.2284 | 6.61e-15 |
| cnn6 | 4 | 2028 | 49.38% | 1.2604 | 1.1966 | 6.44e-15 |
| cnn6 | 8 | 2028 | 59.97% | 1.3865 | 1.2603 | 6.49e-15 |
| cnn8 | 4 | 2026 | 48.83% | 1.4473 | 1.1990 | 8.88e-15 |
| cnn8 | 8 | 2026 | 49.50% | 1.4559 | 1.2097 | 1.05e-14 |
| cnn8 | 4 | 2027 | 56.95% | 1.3523 | 1.2297 | 9.44e-15 |
| cnn8 | 8 | 2027 | 49.73% | 1.4104 | 1.2586 | 1.09e-14 |
| cnn8 | 4 | 2028 | 50.08% | 1.3081 | 1.1847 | 1.04e-14 |
| cnn8 | 8 | 2028 | 54.80% | 1.4433 | 1.2525 | 1.04e-14 |

## Paired effects, including unfavorable seeds

| Architecture | Removed | Comparison (left minus right) | Accuracy per seed (pp) | Mean (pp) | CE per seed |
| --- | ---: | --- | --- | ---: | --- |
| cnn6 | 4 | quaternion_exact - full_projected | +0.70, +1.80, +0.80 | +1.10 | -0.022207, -0.026804, -0.006575 |
| cnn6 | 4 | quaternion_exact - random | +1.05, +2.39, +1.10 | +1.51 | -0.019810, -0.050724, -0.042937 |
| cnn6 | 4 | quaternion_exact - paired_pruned | +1.70, +2.90, +5.40 | +3.33 | -0.078026, -0.097665, -0.150546 |
| cnn6 | 4 | weighted_projected - quaternion_exact | +0.00, +0.00, +0.00 | +0.00 | +0.000000, +0.000000, +0.000000 |
| cnn6 | 4 | full_substitution - full_projected | +1.90, +3.00, +3.10 | +2.67 | -0.050405, -0.068077, -0.045123 |
| cnn6 | 8 | quaternion_exact - full_projected | +3.00, +3.30, +4.60 | +3.63 | -0.033803, -0.103524, -0.173265 |
| cnn6 | 8 | quaternion_exact - random | +8.44, +14.85, +10.41 | +11.23 | -0.171285, -0.405699, -0.360852 |
| cnn6 | 8 | quaternion_exact - paired_pruned | +15.90, +20.10, +26.00 | +20.67 | -0.431104, -0.600452, -1.069095 |
| cnn6 | 8 | weighted_projected - quaternion_exact | +0.00, +0.00, +0.00 | +0.00 | +0.000000, +0.000000, +0.000000 |
| cnn6 | 8 | full_substitution - full_projected | +7.50, +8.60, +11.50 | +9.20 | -0.157033, -0.240047, -0.331365 |
| cnn8 | 4 | quaternion_exact - full_projected | +1.20, +1.80, +0.20 | +1.07 | -0.023529, -0.018294, -0.020184 |
| cnn8 | 4 | quaternion_exact - random | +1.44, -0.05, +2.40 | +1.26 | -0.024924, -0.009149, -0.058491 |
| cnn8 | 4 | quaternion_exact - paired_pruned | +4.10, +1.90, +2.90 | +2.97 | -0.087854, -0.041526, -0.072580 |
| cnn8 | 4 | weighted_projected - quaternion_exact | +0.00, +0.00, +0.00 | +0.00 | +0.000000, +0.000000, +0.000000 |
| cnn8 | 4 | full_substitution - full_projected | +1.70, +2.60, +1.20 | +1.83 | -0.038651, -0.036470, -0.036118 |
| cnn8 | 8 | quaternion_exact - full_projected | +5.90, +1.30, +4.90 | +4.03 | -0.102300, -0.047266, -0.141234 |
| cnn8 | 8 | quaternion_exact - random | +6.00, +3.22, +13.55 | +7.59 | -0.128469, -0.087707, -0.374828 |
| cnn8 | 8 | quaternion_exact - paired_pruned | +15.20, +3.00, +17.10 | +11.77 | -0.404365, -0.110713, -0.547864 |
| cnn8 | 8 | weighted_projected - quaternion_exact | +0.00, +0.00, +0.00 | +0.00 | +0.000000, +0.000000, +0.000000 |
| cnn8 | 8 | full_substitution - full_projected | +8.30, +4.20, +6.20 | +6.23 | -0.158790, -0.127452, -0.199555 |

## Interpretation and limits

Projection energy is the squared Frobenius norm of ADDED compensation blocks, not complete network weights. Consumer MSE, CE and classification accuracy are separate outcomes. The zero-intercept restriction avoids invalid constant-bias folding through zero padding. No map is commuted through ReLU. Exact-Q folding means equality with the graph after the chosen feature substitution, not equality with the original network: the fitted substitution itself is approximate. The functional oracle is a flexible feature-fit reference, not a task-accuracy upper bound. Observed consumer-error differences are not lower bounds on the best possible constrained fit: pairwise feature fitting does not optimize the joint consumer convolution. All positive and negative seed effects are retained. Data-weighted projection should reproduce direct-Q fits and predictions; any difference beyond tolerance is an integrity problem, not a gain. No new method, broad quaternion superiority or publication readiness is claimed. More datasets or a different architecture family would still be needed for broader generalization. Prior context: Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002 in references.bib.

## Focused prior-art positioning

Next-convolution compensation and feature preservation are established in [Kim2020, sections 3.1-3.3](https://proceedings.neurips.cc/paper_files/paper/2020/file/0678ca2eae02d542cc931e81b74de122-Paper.pdf). Our maps act after a complete block, so their use does not assert ReLU commutation.

Weighted affine fits and ordinary affine-consumer folding are explicit in [Asiaee2026CMR, Appendix A.5](https://arxiv.org/html/2602.24266v2). Its discussion of cross-unit coupling reinforces that our pair-ranking proxy is not exact joint consumer-error minimization. Retaining Hamilton blocks is a constraint of this experiment; this limited review does not establish a literature gap. See LITERATURE.md.

## Verified empirical interpretation

CNN6, 4/16 removal: direct-Q minus full_projected accuracy is +0.70, +1.80, +0.80 pp across seeds (mean +1.10, seed SD 0.61 pp); mean CE difference -0.018529. Negative effects and inconsistent signs are retained.

CNN6, 4/16 removal: direct-Q minus random accuracy is +1.05, +2.39, +1.10 pp across seeds (mean +1.51, seed SD 0.76 pp); mean CE difference -0.037824. Negative effects and inconsistent signs are retained.

CNN6, 8/16 removal: direct-Q minus full_projected accuracy is +3.00, +3.30, +4.60 pp across seeds (mean +3.63, seed SD 0.85 pp); mean CE difference -0.103531. Negative effects and inconsistent signs are retained.

CNN6, 8/16 removal: direct-Q minus random accuracy is +8.44, +14.85, +10.41 pp across seeds (mean +11.23, seed SD 3.28 pp); mean CE difference -0.312612. Negative effects and inconsistent signs are retained.

CNN8, 4/16 removal: direct-Q minus full_projected accuracy is +1.20, +1.80, +0.20 pp across seeds (mean +1.07, seed SD 0.81 pp); mean CE difference -0.020669. Negative effects and inconsistent signs are retained.

CNN8, 4/16 removal: direct-Q minus random accuracy is +1.44, -0.05, +2.40 pp across seeds (mean +1.26, seed SD 1.23 pp); mean CE difference -0.030855. Negative effects and inconsistent signs are retained.

CNN8, 8/16 removal: direct-Q minus full_projected accuracy is +5.90, +1.30, +4.90 pp across seeds (mean +4.03, seed SD 2.42 pp); mean CE difference -0.096933. Negative effects and inconsistent signs are retained.

CNN8, 8/16 removal: direct-Q minus random accuracy is +6.00, +3.22, +13.55 pp across seeds (mean +7.59, seed SD 5.35 pp); mean CE difference -0.197001. Negative effects and inconsistent signs are retained.

Across the twelve cells, unweighted projection discards 48.83-59.97% of the squared energy of ADDED compensation blocks. Its pre-BN consumer MSE change relative to full substitution ranges from +26.04% to +45.59%, with an increase in 12/12 cells. These diagnostics have different denominators and are not classification-accuracy effects.

Direct-Q (equivalently, data-weighted) fitting has an observed consumer-MSE change of +18.47% to +29.32% relative to full substitution, increasing it in 12/12 cells. These are measured effects of the specified feature-fit procedure, not an optimality lower bound for every structure-preserving compensation. The full functional reference is not a task-accuracy upper bound.

The strong data-moment control matches direct-Q maps within 1.09e-14 and saved predictions within 0. Direct constrained fitting and data-weighted projection are equivalent here. A gain against unweighted projection cannot establish a new algorithm or a unique quaternion optimization benefit.

Functional full-map substitution uses the original graph and is not a compact quaternion architecture at the reported parameter budget. All physical policies at a given architecture/removal level have identical parameter and buffer budgets. A quarter/half of block-2 units is not a quarter/half of total network parameters; no masked-sparsity or latency claim is made.

Five targeted implementation tests and 4118 artifact checks pass. Checks replay every compact checkpoint and functional oracle, recompute held-out predictions/metrics, verify fit-only selection and least-squares equations, independently expand/fold Hamilton blocks, and check all three 50-epoch training histories, first updates, checkpoint selection and resume states. One root launcher remains: WZNOW_BADANIA.cmd.

This extends the empirical structural trade-off to a second compression level and CNN depth. It does not establish a novel method, general quaternion superiority or paper readiness. One dataset, two related architectures, three seeds and shared validation limit generalization. Focused prior-art positioning (Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002) and broader validation remain necessary. See results/internal_replication/REPORT.md, comparison.png and structural_cost.png.
