# Affine quaternion-unit reconstruction and merging diagnostic

## Hypotheses

A unit with weak fixed-coordinate similarity might still be reconstructible from another unit. A held-out affine relation may preserve classifier logits when the victim is removed and its contribution is folded into the keeper/readout bias. Whether quaternion constraints help is unconfirmed.

## Protocol

Three existing seeds, final block only. Average the 16 spatial positions to obtain one four-component vector per unit/image. Fit on even calibration positions (500 images); evaluate on odd positions (500). These are relations of GAP features, not an asserted spatial or ReLU equivariance. Fit right-Hamilton affine (4+4), diagonal affine (4+4, matched coefficient count), and full-real affine (16+4) maps in float64. All 992 directed pairs per family/seed are retained. Choose eight disjoint pairs using fit-only class-centered single-pair logit residual MSE. Check the combined eight-pair residual independently, including pair interactions. Compare to zeroing the same victims and to folding only their FIT-half means into the readout bias. No labels, new CNN training, validation inference or official-test inference are used in this diagnostic.

The gate requires aggregate held-out logit-MSE ratio <=0.90 vs mean-only and <=0.75 vs pruning in at least two of three seeds. If either map passes, physically evaluate both maps and all controls. This is a gate for a small compression test, not evidence of accuracy gains.

## Measured diagnostic results

| Seed | Selection/map | Check MSE vs mean-only | Check MSE vs pruning | Signal gate |
| ---: | --- | ---: | ---: | --- |
| 2026 | quaternion_right | 0.4055 | 0.3379 | True |
| 2026 | full_real | 0.2247 | 0.1605 | True |
| 2027 | quaternion_right | 0.4575 | 0.3060 | True |
| 2027 | full_real | 0.2501 | 0.2018 | True |
| 2028 | quaternion_right | 0.4698 | 0.3234 | True |
| 2028 | full_real | 0.2252 | 0.1558 | True |

| Seed | Map | Median check feature NMSE vs fit-half mean, all directed pairs |
| ---: | --- | ---: |
| 2026 | quaternion_right | 0.8387 |
| 2026 | diagonal | 0.8957 |
| 2026 | full_real | 0.6116 |
| 2027 | quaternion_right | 0.8338 |
| 2027 | diagonal | 0.9062 |
| 2027 | full_real | 0.6252 |
| 2028 | quaternion_right | 0.8178 |
| 2028 | diagonal | 0.8967 |
| 2028 | full_real | 0.6024 |

Physical compression supported by the prespecified diagnostic gate: **True**.

## Interpretation and limits

Each method selects its own pairs; between-policy effects include selection and map family. The common-full-selected-pair diagnostic additionally compares all three map families on the same pairs, but that selection favors the full fit. The diagonal diagnostic has the same number of fitted coefficients as the quaternion fit. No unique quaternion benefit follows from reconstruction alone. The gate uses held-out class-centered logit error, not labels or accuracy; a positive gate still needs physical evaluation on disjoint validation images. A simple quaternion least-squares fit is not a methodological novelty (PROJECT.md; prior context Kim2020, Ebid2025 in references.bib).

Reconstruction coefficients are calibration estimates. After folding into the existing readout they are not additional inference parameters. If physical evaluation proceeds, removing 8/32 final-block units retains 64,298 of 73,898 real parameters (about 13% saving), with no optimizer updates or BN refit.
