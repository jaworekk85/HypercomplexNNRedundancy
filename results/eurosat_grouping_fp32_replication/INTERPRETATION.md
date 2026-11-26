# EuroSAT full-network grouping replication: interpretation

Completed2026-11-26 at03:32:03Europe/Warsaw. [Report](REPORT.md),
[protocol](PROTOCOL.md), [audit](completion_audit.json),
[descriptive checks](descriptive_summary.json), [cross-dataset table](CROSS_DATASET.md),
[figure](figures/grouping_replications.png).

## Hypothesis

Does the practical advantage of preserving original groups recur on EuroSAT
CNN6 after fixed-source-scalar-set compression and three-epoch repair? Dataset,
architecture/input encoding, compression and total update budget all change
from the CIFAR setting; this does not isolate a dataset effect.

## Measured results

All nine compact models completed. The same retained source scalar identities
are used within each seed/site. A mixed projected producer only approximates
those features; actual produced activations are not assumed identical. Only
hidden widths inside pairs0/2 are halved; outer dimensions remain unchanged.

| Seed | Original final accuracy | Role-preserving mixed | Free-role mixed |
|---|---:|---:|---:|
|2026|93.6%|86.3%|83.8%|
|2027|92.6%|85.4%|82.9%|
|2028|93.8%|88.0%|84.8%|
|Mean|93.33%|86.57%|83.83%|

These are fixed final-epoch3 validation values, not best epochs. Every mixed
model trails its matched original,6/6comparisons. Mean role/free gaps fall from
21.30/29.33ppbefore repair to6.77/9.50ppafter repair. Mean final validation CE:
0.186688original,0.385028role,0.470136free. Mean CHECK gaps:6.73ppand10.33pp.
All individual/native/CE values remain in REPORT.md.

Each compact model has44,794real trainables versus73,898parents,39.38%whole-model
savings. Repair budget is three epochs/462attempts per model. AMP skipped17steps:
4,141successful of4,158attempts total,459--461successful/model. The frozen
scheduler advances on attempts; its warning after an AMP skip is retained.
Successful update counts are therefore not exactly matched. No compensating
steps were added after observing outcomes.

CIFAR's comparable100%mixed conditions have final means90.13%original,
88.60%role,87.87%free, at14.48%savings and1,125successful/attempted updates/model.
The direction recurs across settings. Larger EuroSAT gaps cannot be assigned to
dataset alone given different architecture, input, compression and repair cost.

## Interpretation

The practical grouping finding now spans two datasets/architectures under their
registered budgets. Repair helps mixed models but does not close their gaps.
This supports a scoped empirical analysis of regrouping, physical realization
and bounded recovery, not a superior new compression algorithm.

A Hamilton teacher structurally favors compatible grouping before learning.
The initialization/statistical controls still preclude a learned-semantic or
universal-indispensability claim. The result establishes neither global optimality,
inevitable failure of other groupings/optimizers nor an asymptotic accuracy barrier.
No alternate algebra or rotation-equivariance conclusion follows.

One fixed draw per mixed condition, two compressed sites, three source parents;
nine models are not nine independently trained parents. Masks came from earlier
FIT-based native energy selection. Validation/CHECK were previously inspected.
No official test, pooled significance test, equivalence or acceptance guarantee.

## Precision failure and checks

The first frozen version stopped after three early-site fits and before repair
when archived late-layer activations failed exact replay. The preserved four-mode
diagnostic reproduced that batch only with cuDNN TF32 enabled; baseline/repair
disable TF32. No tolerance or frozen source changed. This separate version
regenerated all FIT/CHECK native features/targets, common ridge scales and18
exact-feature fits in deterministic FP32, with assignments unchanged. Old TF32
errors were not used as new references. See DEVELOPMENT.md and the preserved
../eurosat_grouping_replication failure.

Eight CPU tests, synthetic six-pair/three-model full repair/replay/resume, and
discarded compact CUDA tracking/four-component gradient/update checks passed
before freeze. All18physical fits and nine native/final model predictions
replayed. Final audit passed84output and729source/input hashes. Earlier failed
adaptation/reassembly/exchange gates remain failed.

## Paper use and next action

The planned practical replication is complete, with shared tables/figure and
explicit budget differences. The candidate empirical contribution remains
incremental; scope and differentiation from prior art must be argued carefully.
Canonical background: `Helber2019EuroSAT`, `Vecchi2020`, `He2017ChannelPruning`,
`VieiraValle2022ELM`, `Singh2026`, `Comminiello2024InductiveBiases`.

Next: consolidate the paper argument and specify a locked final-evaluation
protocol, auditing prior split access. No further numerical study, algebra/
dimension search, repair sweep or official test is queued by this protocol.
