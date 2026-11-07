# Physical last-block merging with readout compensation

## Hypotheses

Held-out reconstructible units may be removed while folding their contribution into retained readout weights/bias. Quaternion-restricted compensation may or may not retain quality as well as an unrestricted real 4x4 fit. No novel method or special quaternion advantage is assumed.

## Protocol

The diagnostic gate passed for both map families in all three existing baseline seeds. Fit/choose pairs using only 500 even-position calibration images; gate on the 500 disjoint odd-position images. Freeze every pair set and control manifest before new validation inference. Physically remove eight of 32 final-block quaternion units (73,898 to 64,298 real parameters, 12.99% saving). Evaluate affine compensation, pruning the exact same victims, and mean-only correction for each family. Reuse ten existing random-pruning selections/checkpoints per seed and the standardized baseline checkpoints. No CNN optimizer updates, BN refit, validation-based policy/size tuning or official-test inference. All 1,000 validation images are disjoint from fitting and the diagnostic gate, but served earlier checkpoint selection, so this is not a new blind benchmark.

## Measured validation results

| Policy | Real parameters | Mean accuracy ± seed SD | Mean CE |
| --- | ---: | ---: | ---: |
| baseline | 73898 | 77.93% ± 1.10 pp | 0.637730 |
| random | 64298 | 76.37% ± 0.99 pp | 0.687761 |
| quaternion_right_pruned | 64298 | 75.17% ± 0.83 pp | 0.707504 |
| quaternion_right_mean_only | 64298 | 76.77% ± 1.55 pp | 0.676093 |
| quaternion_right_affine | 64298 | 76.63% ± 1.10 pp | 0.667735 |
| full_real_pruned | 64298 | 76.30% ± 0.82 pp | 0.692571 |
| full_real_mean_only | 64298 | 76.50% ± 1.25 pp | 0.672888 |
| full_real_affine | 64298 | 76.67% ± 1.53 pp | 0.654740 |

Random repetitions are averaged within each training seed before computing the sample SD across three seeds. Parameters count actual compact tensors; no masks or extra inference mapping layers remain.

| Map | Comparison | Paired seed accuracy differences (pp) | Mean difference (pp) |
| --- | --- | --- | ---: |
| quaternion_right | random | +0.37, +0.13, +0.30 | +0.27 |
| quaternion_right | same_victims_pruned | +2.10, +0.90, +1.40 | +1.47 |
| quaternion_right | same_victims_mean_only | -0.40, +0.40, -0.40 | -0.13 |
| quaternion_right | unpruned | -1.30, -1.30, -1.30 | -1.30 |
| full_real | random | +0.87, -0.37, +0.40 | +0.30 |
| full_real | same_victims_pruned | +1.00, -0.50, +0.60 | +0.37 |
| full_real | same_victims_mean_only | +0.00, -0.20, +0.70 | +0.17 |
| full_real | unpruned | -0.80, -1.80, -1.20 | -1.27 |

## Interpretation and limits

Accuracy effects against the same-victim pruning and mean-only controls separate compensation from selection and trivial average-offset correction. Between-family comparisons still combine each family's own pair choices and fitted map; the diagnostic also records same-pair comparisons. Diagonal fitting is a coefficient-matched diagnostic, not another physical compression policy. Affine fit coefficients (8 vs 20 per pair, including offsets) are calibration estimates, not new retained neural weights: both final compact methods have exactly the same parameter budget. A better reconstruction ratio alone is not proof of a better classifier.

This is one architecture, dataset, selected block and removal level, with three existing training seeds and no fine-tuning. Report negative results and worse CE alongside accuracy. No general compression superiority, latency/FLOP gain, novelty or publication readiness follows automatically. Prior art keys: Kim2020, Ebid2025, Vecchi2020, Singh2026 in references.bib. No faithful full-method reproduction is claimed.

## Artifacts

DIAGNOSTIC_REPORT.md, diagnostic_protocol.json, all-pair fits, calibration partitions, gate_frozen.json, physical_protocol.json, selection_frozen.json, per-case manifests/checkpoints/predictions/metrics, aggregated.json, paired_effects.json and comparison.png.

## Verified measured interpretation

Right-Hamilton compensation improves accuracy over pruning the exact same victims by +2.10, +0.90 and +1.40 pp (mean +1.47 pp), but its mean accuracy is 0.13 pp below the same-victim mean-only correction. Full-real compensation improves over its own mean-only correction by 0.00, -0.20 and +0.70 pp. Thus a repeatable accuracy advantage beyond the cheap mean-only control is not demonstrated.

Validation accuracy with right-Hamilton/full-real merging is 76.63%/76.67%, versus 77.93% unpruned and 76.37% random-pruned (ten random repetitions averaged within each training seed). Paired right-Hamilton-minus-random differences are +0.37, +0.13 and +0.30 pp. These are small three-seed pilot effects, not a general superiority claim.

Retain the favorable CE evidence too: full-real merging has mean CE 0.654740, versus 0.672888 for its mean-only control and 0.687761 for random pruning; the unpruned mean is 0.637730. Right-Hamilton merging has CE 0.667735 versus 0.676093 for its mean-only control. Lower held-out logit MSE and better CE do not automatically imply better argmax accuracy. No special quaternion advantage is established.

Nine meaningful implementation tests and 922 independent artifact checks pass. Maximum cached-to-physical logit error is below 5e-6 (frozen tolerance 1e-4). Checks cover fit-only selection, affine normal equations, whole-image held-out splits, no hidden map parameters, exact kernel/BN slicing, compensation formulas, actual parameter/buffer/file counts and checkpoint reload predictions. All original numerical sources and model/data artifacts are preserved.

This is a bounded pilot at one layer and removal level. Map fitting/merging is established background (Kim2020, Ebid2025); quaternion-specific pruning also has prior art (Vecchi2020, Singh2026). The new evidence supports an empirical distinction between reconstruction/CE and classification accuracy, not a novel-method or paper-readiness claim. See results/quaternion_merging/DIAGNOSTIC_REPORT.md and REPORT.md.
