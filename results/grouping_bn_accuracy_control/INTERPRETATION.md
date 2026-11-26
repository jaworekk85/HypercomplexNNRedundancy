# BN-only accuracy intervention: interpretation

Completed 2026-11-26 at 01:43:51 Europe/Warsaw. See [protocol](PROTOCOL.md),
[all measured results](REPORT.md), [summary](summary.json),
[completion audit](completion_audit.json) and [figure](figures/validation_accuracy.png).

## Hypothesis

Correcting only BN running statistics might recover the full-network accuracy
lost after mixing quaternion groups. This tests actual classification in the
saved compact architectures, unlike the preceding modified-teacher local
reconstruction diagnostic. It is not a test of every possible BN correction.

## Measured results

All 15 compact models and three uncompressed parent controls completed. The
native compact models exactly replayed predecessor metrics and per-image logits.
Every BN-only model preserved all trainable parameters, including BN affine,
and all non-BN buffers. The same fixed 1,024 unlabelled training images supplied
the sequential eval-prefix population recalibration for every model.

Validation accuracies below are descriptive means across source seeds
2026/2027/2028. The mixed case is the predetermined free-role 100% draw zero;
all other cases and individual seeds remain in REPORT.md.

| Condition | Native accuracy | BN-only accuracy | Historical 3-epoch repair |
|---|---:|---:|---:|
| Original grouping | 87.83% | 86.67% | 90.13% |
| Free-role 100% mixed | 77.93% | 74.87% | 87.87% |
| Original-minus-mixed gap | 9.90 pp | 11.80 pp | 2.27 pp |

The registered primary endpoint is native gap minus BN-only gap, positive when
BN reduces the penalty. The validation gap decreased in 3/12 seed/case pairs
and increased in 9/12. Mean gap reductions across three seeds were negative
for every mixed condition:

| Mixed condition | Native gap | BN-only gap | BN gap reduction |
|---|---:|---:|---:|
| Role-preserving 50% | 2.83 pp | 4.63 pp | -1.80 pp |
| Free-role 50% | 4.40 pp | 5.20 pp | -0.80 pp |
| Role-preserving 100% | 8.20 pp | 8.73 pp | -0.53 pp |
| Free-role 100% | 9.90 pp | 11.80 pp | -1.90 pp |

The corresponding mean CHECK gaps also increased in all four mixed conditions.
BN-only validation accuracy improved in 4/18 individual models and declined in
14/18. Parent controls changed by -0.7/-2.7/+0.4 pp, mean -1.0 pp; degradation
was therefore not specific to mixed or compact models. All CE and CHECK results
are saved, including unfavorable outcomes.

## Interpretation

The tested BN-only procedure does not recover the grouping-related accuracy
loss. Original grouping remains better under this intervention. The existing
three-epoch supervised repair recovers considerably more accuracy, while still
leaving mixed models behind their matched original-group controls.

This rejects the practical claim that this particular recalibration alone is
sufficient. It does not prove that normalization plays no role or that the
remaining gap identifies learned quaternion semantics. The parent degradation
also limits an interpretation specific to grouping. Finite-subset population
recalibration and historical training-time running statistics are different
procedures. No additional recipe, augmentation or subset-size search was run.

Historical repair changes weights and BN statistics and did not start from
the newly recalibrated model. Its gain cannot be assigned entirely to weight
updates by this comparison. Likewise, the preceding local statistics control
and this full-network intervention have different endpoints; neither result
overrides the other's measured outcome.

These are exploratory comparisons on previously inspected validation/CHECK
splits, with three source seeds, one fixed mixed draw per repair condition and
two compressed internal sites. No official test was evaluated. There is no
new compression method, gain, equivalence, significance or novelty claim.
Compact models retain 232,266 real trainables against 271,594 parents, about
14.5% whole-model reduction. The older single-seed reassembly pilot's failed
continuation gate is unchanged.

## Verification and closure

All 72 saved artifact hashes and 565 registered source/input hashes passed the
completion audit. Calibrated checkpoint/logit replay and native predecessor
replay passed for every compact model. The pre-registration incorrect-helper
signature failure, corrected ten-test run, synthetic two-model run/resume and
discarded-copy compact GPU gradient/update audit are retained in DEVELOPMENT.md
and development_checks.json.

This bounded control is complete and no further computation is queued. The
BN-only recovery explanation has failed under the registered recipe. Any later
method or named-algebra extension needs a separate question and protocol; this
outcome does not automatically justify another round of controls.

Canonical background keys: `Shomron2020BNRecalibration`, `Jordan2023REPAIR`,
`Vecchi2020`, `He2016ResNet` in references.bib. The intervention is not an
implementation of the distinct REPAIR interpolation method.
