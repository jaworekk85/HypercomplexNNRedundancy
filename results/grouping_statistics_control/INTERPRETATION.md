# Statistics controls: completed interpretation

Completed 2026-11-26 at 00:19:51 Europe/Warsaw. This interpretation follows the
registered primary endpoint; it introduces no new fitting or selected endpoint.
See [measured report](REPORT.md), [descriptive audit](descriptive_summary.json),
[protocol](PROTOCOL.md), and [figure](figures/training_increment.png).

## Hypothesis

The previous initialization control found a larger original-group advantage
after training in all 12 CIFAR seed/site/policy comparisons. Does this contrast
persist after BN recalibration and marginal feature standardization, and on a
second dataset/architecture? A grouping advantage by itself can arise from
Hamilton weight sharing before any learning; it does not identify learned
quaternion semantics.

## Measured results

All 1,512 rows completed: 1,260 new ridge fits and 252 reused CIFAR native rows.
CIFAR-10 Q ResNet-20 and EuroSAT four-band Q CNN6 each use source seeds
2026/2027/2028 and early/late consumers. Each policy averages ten fixed random
fully mixed assignments against the original assignment, holding retained
scalar sets fixed within each dataset/seed/site. The random draws are not
independent model-training seeds.

The primary quantity is the increase, trained minus initial, in the
mixed-minus-original reconstruction error, divided by CHECK target energy
centered using FIT target-channel means. Its standardized condition uses
sequential BN recalibration followed by FIT-only scalar feature standardization.
Positive values mean training increases the mixing penalty. These values are
not classification accuracy or percentage points.

| Dataset / architecture | Consumer | Roles preserved | Roles free |
|---|---|---:|---:|
| CIFAR-10 / ResNet-20 | Early | -0.098577 | -0.139252 |
| CIFAR-10 / ResNet-20 | Late | +0.017036 | +0.012717 |
| EuroSAT / CNN6 | Early | +0.025840 | +0.001697 |
| EuroSAT / CNN6 | Late | -0.073040 | -0.079261 |

Table entries are descriptive means across three paired source seeds, not
significance estimates. All individual seeds are in REPORT.md and summary.json.

| Condition (same centered-energy metric) | CIFAR positive increments / 12 | EuroSAT positive increments / 12 |
|---|---:|---:|
| Native | 12 | 11 |
| BN recalibrated | 4 | 4 |
| BN recalibrated + standardized (primary) | 6 | 4 |

After the full control the original grouping still has lower mean error than
either mixed policy in all 24 dataset/seed/site/policy comparisons, both at
initialization and after training. The *increase due to training* is not uniform.
For example, both early CIFAR penalties increase for seed 2027 but decrease for
seeds 2026 and 2028; both late EuroSAT penalties decrease for every seed.

The centered denominator does not add an intercept or refit the model. Raw-energy
results, native conditions and BN-only conditions are retained, including
opposite and small effects. No classification inference or official-test or
validation evaluation was performed in this study.

## Interpretation

The earlier universal training-amplification claim does not survive these
controls. Original grouping remains favored, but that observation is compatible
with architectural constraints already present at initialization. BN/statistical
state materially changes the diagnostic comparison. These results do not support
using the previous 12/12 increases as evidence of learned quaternion entities.

This is not a proof that BN alone causes the original effect. Recalibration
changes intermediate network behavior, and standardization recomputes targets
with the unchanged full Hamilton consumer, defining a modified diagnostic
teacher. It is not function-preserving graph surgery. Marginal means/scales are
controlled; covariance, sparsity and higher moments are not matched. Initial
weights are regenerated from the pinned constructor/seed recipe rather than an
independently archived epoch-zero checkpoint. EuroSAT also changes architecture
and input encoding, so differences cannot be assigned to dataset alone.

There is no demonstration here that quaternion grouping is irrelevant, that
mixed models are equivalent, or that a new compression method improves accuracy.
No new compact architectures were trained. The earlier 90.8% pilot and its failed
0.5 pp continuation gate remain unchanged. This study is useful evidence against
an overstrong interpretation, not by itself an established IJCNN contribution.

## Verification and next decision

Completion audit passed. Postprocessing verified all 676 registered source/input
hashes, 72 coefficient-bank hashes, and 12 recalibrated-model hashes. Trainable
parameters, including BN affine parameters, remained bitwise unchanged in every
BN audit. Pre-registration development checks and the corrected smoke-resume
failure are preserved in DEVELOPMENT.md/development_checks.json.

The descriptive figure shows three source seeds separately and reports no
confidence intervals or significance. Reproduce it with
`python -B -m scripts.plot_grouping_statistics`; the descriptive hash audit is
`python -B -m scripts.summarize_grouping_statistics` in the registered environment.

No further computation is queued. Before another run, choose a narrower
falsifiable claim that distinguishes representation effects from the expected
structural regrouping penalty. Named 4D algebras remain a possible separate
extension, requiring separately trained matched-budget parents and audited
multiplication/solvers; none was tested here. See the
[verified Valle literature note](../../docs/literature/OTHER_FOUR_DIMENSIONAL_ALGEBRAS_2026-11-26.md)
and canonical keys `VieiraNetoValle2025VEfficientNets`, `Valle2024VNets`.
