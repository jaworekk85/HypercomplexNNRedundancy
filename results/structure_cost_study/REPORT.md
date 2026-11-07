# Structural cost: all-site Q versus physically compiled unrestricted compensation

## Hypotheses

- At fixed removed groups, the quality cost of retaining the Hamilton family can be measured against unrestricted physically compiled consumers, with their different size explicitly reported.
- free_full_fit vs free_q_init isolates the compensation initialization at the same dense architecture and repair budget. free_q_init initially equals q_comp; their post-repair difference captures release of weight constraints, not a better initial compensation map.
- The one-seed no-BN control asks whether compensation is useful when BN statistics cannot be stale. Its baseline/selection differ from BN; it does not causally isolate all effects of normalization.

## Measured results

Status: completed; 14/14 dataset/normalization/seed/fraction groups. Validation means; all seed values in summary.json. New study chosen after earlier outcomes; no test inference.

Q-pruned and Q-comp remove identical groups. Free-Q-init starts with exactly Q-comp's predictions and releases five consumer convolutions to dense real weights. Free-full-fit folds unrestricted maps without projection. First conv remains Q; readout was already real. Same width is not matched parameters/compute. Negative savings mean MORE parameters than the full source QNN.

| Dataset | Norm | Removed/site | Case | Stage | Real params | Saving vs full Q | n | Val mean (%) | SD (pp) |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cifar | bn | 0.25 | free_full_fit | after | 162562 | -119.98% | 3 | 73.90 | 0.00 |
| cifar | bn | 0.25 | free_full_fit | before | 162562 | -119.98% | 3 | 69.03 | 0.42 |
| cifar | bn | 0.25 | free_q_init | after | 162562 | -119.98% | 3 | 72.93 | 0.87 |
| cifar | bn | 0.25 | free_q_init | before | 162562 | -119.98% | 3 | 59.67 | 3.21 |
| cifar | bn | 0.25 | q_comp | after | 42034 | 43.12% | 3 | 71.50 | 1.20 |
| cifar | bn | 0.25 | q_comp | before | 42034 | 43.12% | 3 | 59.67 | 3.21 |
| cifar | bn | 0.25 | q_pruned | after | 42034 | 43.12% | 3 | 70.00 | 0.36 |
| cifar | bn | 0.25 | q_pruned | before | 42034 | 43.12% | 3 | 23.47 | 2.75 |
| cifar | bn | 0.5 | free_full_fit | after | 72666 | 1.67% | 3 | 63.17 | 0.21 |
| cifar | bn | 0.5 | free_full_fit | before | 72666 | 1.67% | 3 | 35.07 | 4.60 |
| cifar | bn | 0.5 | free_q_init | after | 72666 | 1.67% | 3 | 57.90 | 0.69 |
| cifar | bn | 0.5 | free_q_init | before | 72666 | 1.67% | 3 | 21.97 | 4.12 |
| cifar | bn | 0.5 | q_comp | after | 19098 | 74.16% | 3 | 53.57 | 0.65 |
| cifar | bn | 0.5 | q_comp | before | 19098 | 74.16% | 3 | 21.97 | 4.12 |
| cifar | bn | 0.5 | q_pruned | after | 19098 | 74.16% | 3 | 49.67 | 2.14 |
| cifar | bn | 0.5 | q_pruned | before | 19098 | 74.16% | 3 | 13.43 | 1.55 |
| cifar | none | 0.25 | free_full_fit | after | 161890 | -121.76% | 1 | 73.00 | — |
| cifar | none | 0.25 | free_full_fit | before | 161890 | -121.76% | 1 | 67.30 | — |
| cifar | none | 0.25 | free_q_init | after | 161890 | -121.76% | 1 | 70.20 | — |
| cifar | none | 0.25 | free_q_init | before | 161890 | -121.76% | 1 | 56.70 | — |
| cifar | none | 0.25 | q_comp | after | 41362 | 43.34% | 1 | 70.10 | — |
| cifar | none | 0.25 | q_comp | before | 41362 | 43.34% | 1 | 56.70 | — |
| cifar | none | 0.25 | q_pruned | after | 41362 | 43.34% | 1 | 68.40 | — |
| cifar | none | 0.25 | q_pruned | before | 41362 | 43.34% | 1 | 32.40 | — |
| cifar | none | 0.5 | free_full_fit | after | 72218 | 1.07% | 1 | 57.30 | — |
| cifar | none | 0.5 | free_full_fit | before | 72218 | 1.07% | 1 | 33.00 | — |
| cifar | none | 0.5 | free_q_init | after | 72218 | 1.07% | 1 | 51.90 | — |
| cifar | none | 0.5 | free_q_init | before | 72218 | 1.07% | 1 | 22.20 | — |
| cifar | none | 0.5 | q_comp | after | 18650 | 74.45% | 1 | 45.70 | — |
| cifar | none | 0.5 | q_comp | before | 18650 | 74.45% | 1 | 22.20 | — |
| cifar | none | 0.5 | q_pruned | after | 18650 | 74.45% | 1 | 38.70 | — |
| cifar | none | 0.5 | q_pruned | before | 18650 | 74.45% | 1 | 10.40 | — |
| eurosat | bn | 0.25 | free_full_fit | after | 162562 | -119.98% | 3 | 95.03 | 0.49 |
| eurosat | bn | 0.25 | free_full_fit | before | 162562 | -119.98% | 3 | 92.27 | 1.15 |
| eurosat | bn | 0.25 | free_q_init | after | 162562 | -119.98% | 3 | 94.47 | 0.32 |
| eurosat | bn | 0.25 | free_q_init | before | 162562 | -119.98% | 3 | 78.13 | 5.02 |
| eurosat | bn | 0.25 | q_comp | after | 42034 | 43.12% | 3 | 93.73 | 0.38 |
| eurosat | bn | 0.25 | q_comp | before | 42034 | 43.12% | 3 | 78.13 | 5.02 |
| eurosat | bn | 0.25 | q_pruned | after | 42034 | 43.12% | 3 | 92.47 | 0.47 |
| eurosat | bn | 0.25 | q_pruned | before | 42034 | 43.12% | 3 | 32.03 | 6.51 |
| eurosat | bn | 0.5 | free_full_fit | after | 72666 | 1.67% | 3 | 90.17 | 1.33 |
| eurosat | bn | 0.5 | free_full_fit | before | 72666 | 1.67% | 3 | 61.00 | 7.98 |
| eurosat | bn | 0.5 | free_q_init | after | 72666 | 1.67% | 3 | 88.23 | 1.17 |
| eurosat | bn | 0.5 | free_q_init | before | 72666 | 1.67% | 3 | 32.20 | 0.72 |
| eurosat | bn | 0.5 | q_comp | after | 19098 | 74.16% | 3 | 85.33 | 2.85 |
| eurosat | bn | 0.5 | q_comp | before | 19098 | 74.16% | 3 | 32.20 | 0.72 |
| eurosat | bn | 0.5 | q_pruned | after | 19098 | 74.16% | 3 | 77.47 | 5.72 |
| eurosat | bn | 0.5 | q_pruned | before | 19098 | 74.16% | 3 | 16.20 | 4.52 |

### Paired accuracy effects

| Dataset | Norm | Removed/site | Stage | Left minus right | Mean (pp) | Seeds (pp) |
| --- | --- | ---: | --- | --- | ---: | --- |
| cifar | bn | 0.25 | after | q_comp - q_pruned | +1.50 | +1.90, +2.40, +0.20 |
| cifar | bn | 0.25 | after | free_full_fit - q_comp | +2.40 | +2.40, +1.20, +3.60 |
| cifar | bn | 0.25 | after | free_q_init - q_comp | +1.43 | +1.20, +1.20, +1.90 |
| cifar | bn | 0.25 | after | free_full_fit - free_q_init | +0.97 | +1.20, +0.00, +1.70 |
| cifar | bn | 0.25 | before | q_comp - q_pruned | +36.20 | +35.90, +36.80, +35.90 |
| cifar | bn | 0.25 | before | free_full_fit - q_comp | +9.37 | +9.30, +5.90, +12.90 |
| cifar | bn | 0.25 | before | free_q_init - q_comp | +0.00 | +0.00, +0.00, +0.00 |
| cifar | bn | 0.25 | before | free_full_fit - free_q_init | +9.37 | +9.30, +5.90, +12.90 |
| cifar | bn | 0.5 | after | q_comp - q_pruned | +3.90 | +5.40, +0.80, +5.50 |
| cifar | bn | 0.5 | after | free_full_fit - q_comp | +9.60 | +8.90, +10.10, +9.80 |
| cifar | bn | 0.5 | after | free_q_init - q_comp | +4.33 | +4.50, +4.60, +3.90 |
| cifar | bn | 0.5 | after | free_full_fit - free_q_init | +5.27 | +4.40, +5.50, +5.90 |
| cifar | bn | 0.5 | before | q_comp - q_pruned | +8.53 | +6.80, +11.60, +7.20 |
| cifar | bn | 0.5 | before | free_full_fit - q_comp | +13.10 | +21.50, +7.10, +10.70 |
| cifar | bn | 0.5 | before | free_q_init - q_comp | +0.00 | +0.00, +0.00, +0.00 |
| cifar | bn | 0.5 | before | free_full_fit - free_q_init | +13.10 | +21.50, +7.10, +10.70 |
| cifar | none | 0.25 | after | q_comp - q_pruned | +1.70 | +1.70 |
| cifar | none | 0.25 | after | free_full_fit - q_comp | +2.90 | +2.90 |
| cifar | none | 0.25 | after | free_q_init - q_comp | +0.10 | +0.10 |
| cifar | none | 0.25 | after | free_full_fit - free_q_init | +2.80 | +2.80 |
| cifar | none | 0.25 | before | q_comp - q_pruned | +24.30 | +24.30 |
| cifar | none | 0.25 | before | free_full_fit - q_comp | +10.60 | +10.60 |
| cifar | none | 0.25 | before | free_q_init - q_comp | +0.00 | +0.00 |
| cifar | none | 0.25 | before | free_full_fit - free_q_init | +10.60 | +10.60 |
| cifar | none | 0.5 | after | q_comp - q_pruned | +7.00 | +7.00 |
| cifar | none | 0.5 | after | free_full_fit - q_comp | +11.60 | +11.60 |
| cifar | none | 0.5 | after | free_q_init - q_comp | +6.20 | +6.20 |
| cifar | none | 0.5 | after | free_full_fit - free_q_init | +5.40 | +5.40 |
| cifar | none | 0.5 | before | q_comp - q_pruned | +11.80 | +11.80 |
| cifar | none | 0.5 | before | free_full_fit - q_comp | +10.80 | +10.80 |
| cifar | none | 0.5 | before | free_q_init - q_comp | +0.00 | +0.00 |
| cifar | none | 0.5 | before | free_full_fit - free_q_init | +10.80 | +10.80 |
| eurosat | bn | 0.25 | after | q_comp - q_pruned | +1.27 | +1.00, +1.90, +0.90 |
| eurosat | bn | 0.25 | after | free_full_fit - q_comp | +1.30 | +2.30, +0.70, +0.90 |
| eurosat | bn | 0.25 | after | free_q_init - q_comp | +0.73 | +1.40, +0.10, +0.70 |
| eurosat | bn | 0.25 | after | free_full_fit - free_q_init | +0.57 | +0.90, +0.60, +0.20 |
| eurosat | bn | 0.25 | before | q_comp - q_pruned | +46.10 | +46.90, +44.20, +47.20 |
| eurosat | bn | 0.25 | before | free_full_fit - q_comp | +14.13 | +13.70, +10.50, +18.20 |
| eurosat | bn | 0.25 | before | free_q_init - q_comp | +0.00 | +0.00, +0.00, +0.00 |
| eurosat | bn | 0.25 | before | free_full_fit - free_q_init | +14.13 | +13.70, +10.50, +18.20 |
| eurosat | bn | 0.5 | after | q_comp - q_pruned | +7.87 | +7.40, +11.20, +5.00 |
| eurosat | bn | 0.5 | after | free_full_fit - q_comp | +4.83 | +3.00, +6.60, +4.90 |
| eurosat | bn | 0.5 | after | free_q_init - q_comp | +2.90 | +1.60, +4.80, +2.30 |
| eurosat | bn | 0.5 | after | free_full_fit - free_q_init | +1.93 | +1.40, +1.80, +2.60 |
| eurosat | bn | 0.5 | before | q_comp - q_pruned | +16.00 | +13.30, +13.80, +20.90 |
| eurosat | bn | 0.5 | before | free_full_fit - q_comp | +28.80 | +18.80, +34.40, +33.20 |
| eurosat | bn | 0.5 | before | free_q_init - q_comp | +0.00 | +0.00, +0.00, +0.00 |
| eurosat | bn | 0.5 | before | free_full_fit - free_q_init | +28.80 | +18.80, +34.40, +33.20 |

## Interpretation limits and decision rules

- Do not claim a new ridge/folding algorithm. The candidate contribution is a quantified structural-cost and capacity decomposition, not literature priority.
- If full-fit advantage is explained by free_q_init after repair, attribute that part to extra optimization freedom; do not sell it as a better compensation method.
- If Q has similar observed quality with fewer parameters, report the paired effects and uncertainty; no equivalence claim from n=3 or a small mean gap.
- If unrestricted compiled models exceed the original QNN parameter budget, show negative savings. Equal width is not equal parameters or compute.
- Report all outcomes and unfavorable seeds. No new architecture/activation sweep or test inference. Stop this registered study after the fixed two fractions; extra experiments need a separate scientific reason.

BN original masks/results are reused with provenance, not new training. No-BN uses its own fitted masks and only one seed. Full-vs-Q after repair combines map initialization and parameter-family freedom; free-Q-init provides the capacity control. Three epochs, same attempted batch/augmentation schedules, not audited equal successful AMP steps. No latency claim or new-method claim (GRAIL2026/Kim2020). Checkpoint bytes and reconstruction/fold checks are saved per group. No external correspondence.

## Independently reviewed interpretation

Review status: completed. Completed paired groups: 14/14. All entries below retain the registered outcomes; no new model selection or test inference.

| Dataset | Norm | Removed/site | n | Full minus Q before (pp) | Full minus Q after (pp) | Full minus dense-Q-init after (pp) | Dense-Q-init minus Q after (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar | bn | 0.25 | 3 | +9.37 | +2.40 | +0.97 | +1.43 |
| cifar | bn | 0.5 | 3 | +13.10 | +9.60 | +5.27 | +4.33 |
| eurosat | bn | 0.25 | 3 | +14.13 | +1.30 | +0.57 | +0.73 |
| eurosat | bn | 0.5 | 3 | +28.80 | +4.83 | +1.93 | +2.90 |
| cifar | none | 0.25 | 1 | +10.60 | +2.90 | +2.80 | +0.10 |
| cifar | none | 0.5 | 1 | +10.80 | +11.60 | +5.40 | +6.20 |

The final two columns sum to total Full-minus-Q after repair. This is an arithmetic decomposition under two controls, not evidence of independent/additive causal mechanisms. Per-seed differences are in decomposition.json.

Unrestricted half-width models have roughly the parameter count of the original full QNN; at .25 removal they are larger than that source. This follows approximately P_free(f) ≈ 4(1-f)^2 P_Q(0), versus P_Q(f) ≈ (1-f)^2 P_Q(0), with first-layer/BN/readout corrections. These are equal-width structural-cost comparisons, not a parameter-matched compression frontier.

Figures: [BN recovery](bn_recovery.pdf), [no-BN recovery](no_bn_recovery.pdf), [actual parameter trade-off](parameter_tradeoff.pdf). Recovery shading and parameter-plot error bars are ±1 sample SD of seed accuracy, not confidence intervals; no-BN has one seed and no estimated seed uncertainty. A source-model marker is included so a larger, less accurate reconstructed model is not presented as a useful compression improvement.

The observed Q/free gap concerns these specified compensation procedures. It is NOT an irreducible lower bound imposed by quaternion architectures. In particular, fitting one shared feature-reconstruction map and then folding it is more restrictive than directly optimizing every surviving quaternion consumer weight against its desired outputs. No such direct consumer fit is claimed or silently substituted here.

A full-fit advantage over dense-Q-init uses the SAME dense architecture and parameter count, so that comparison can support an initialization effect under the fixed repair recipe. Dense-Q-init minus Q-comp begins from the same function and measures the intervention of releasing five weight constraints, including changed optimization geometry. Neither comparison establishes a quaternion-specific mechanism or a novel ridge method.

Local Q/full map errors were evaluated on the SAME current full-path inputs at each site, using an untouched check-calibration half. These diagnostics are distinct from comparing activations along two already-diverged complete paths. Spatial observations are not claimed to be independent samples. No-BN n=1 is exploratory; the existing baseline, mask choice and parameter count differ from BN.

The fixed study ends at these two fractions. A scientifically distinct follow-up, if a remaining gap matters, would test whether directly fitting a quaternion consumer (same compact architecture) closes it. That would test a weakness of the current compensation parameterization; it would still require a closest-prior-work review before any novelty claim. It is NOT a newly queued experiment.
