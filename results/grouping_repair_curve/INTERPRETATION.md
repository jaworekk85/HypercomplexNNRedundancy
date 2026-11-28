# Matched repair curves: completed interpretation

Completed2026-11-26at20:52:27Europe/Warsaw(19:52:27UTC):18trajectories,
90requested snapshots. Completion audit passed798frozen source/input hashes
and180model/logit artifacts; an independent post-completion recheck passed the
same hashes. Worker exited. [Report](REPORT.md), [summary](summary.json),
[protocol](PROTOCOL.md), [completion audit](completion_audit.json).

The full saved epoch history is plotted in[the repair curve](figures/repair_curve.png)
and[PDF](figures/repair_curve.pdf), generated2026-11-28by
`scripts/plot_grouping_repair_curve.py`. Top panels:validation accuracy;
bottom panels:paired original-minus-mixed gaps. Thin lines are individual source
seeds, thick lines descriptive means; markers identify0/1/3/10/30snapshots.
This reuses saved outcomes without any new model inference. Source/figure hashes
and numerical curves are saved in`figures/repair_curve.json`.

## Hypothesis

Does original grouping maintain a classification advantage after longer matched
repair, or mainly recover faster? The experiment holds compact initial weights,
retained scalar masks and physical parameter counts fixed across repair budgets.
Each assignment follows one newly frozen30epoch cosine schedule from its
pre-repair initialization. Original, role-preserving and free-role assignments
share the prescribed recipe within source seed, not their initial weights.

## Measured results

Final epoch30validation accuracy, descriptive percentages; source seeds2026/27/28:

| Setting | Seed | Original | Role-preserving mixed | Free-role mixed |
|---|---:|---:|---:|---:|
| CIFAR-10 / ResNet-20 |2026|90.4|90.3|90.1|
| CIFAR-10 / ResNet-20 |2027|92.0|91.7|91.9|
| CIFAR-10 / ResNet-20 |2028|91.6|91.0|91.0|
| CIFAR-10 mean |3parents|91.33|91.00|91.00|
| EuroSAT / CNN6 |2026|94.3|92.8|92.2|
| EuroSAT / CNN6 |2027|95.1|92.1|92.7|
| EuroSAT / CNN6 |2028|95.0|92.4|93.1|
| EuroSAT mean |3parents|94.80|92.43|92.67|

Original-minus-mixed validation accuracy gaps(pp), means over the three parents:

| Epoch in the common30epoch schedule | CIFAR role | CIFAR free | EuroSAT role | EuroSAT free |
|---:|---:|---:|---:|---:|
|0|8.20|9.90|21.30|29.33|
|1|1.90|2.93|8.57|12.67|
|3|0.90|1.63|4.27|6.00|
|10|0.33|0.63|3.27|3.10|
|30|0.33|0.33|2.37|2.13|

Epoch30mixed-minus-original mean validation CE is-0.001577/-0.004383for
CIFAR role/free, respectively: mixed models have slightly lower mean CE despite
their lower mean accuracy. EuroSAT CE differences are+0.064835/+0.073779,
both favoring original grouping. Retain this metric-dependent outcome; do not
describe the CIFAR original condition as uniformly better classification quality.
All seed-level metrics, reconstruction-check results and epoch histories remain
in the report/summary; none was selected as a best epoch.

Total142787successful/142830attempted updates;43AMP skips. CIFAR cases each
attempt11250updates and achieve11248--11249; EuroSAT each attempts4620and
achieves4616--4617. The frozen scheduler advances on attempted updates. The
recorded PyTorch scheduler-order warning is retained in the stderr log and is
consistent with skipped AMP optimizer steps. No schedule was modified or extra
updates added after outcomes. Equal attempted budgets do not imply identical
successful updates.

## Interpretation

Longer repair substantially narrows the grouping accuracy gaps. On CIFAR, the
final descriptive gap is0.33ppfor both policies, with mean CE slightly favoring
mixed models. On EuroSAT, residual accuracy gaps of2.13--2.37ppand CE gaps
remain after30epochs. Thus the strong short-budget degradation is partly a
recovery-budget effect; a large universal persistent grouping advantage is not
supported. A residual cost persists in the tested EuroSAT setting/horizon.

This does not prove practical equivalence on CIFAR or a permanent advantage on
EuroSAT. Three source seeds, one predetermined full-network mixed draw/policy,
previously inspected1000-image validation and500-image reconstruction-check
splits remain limitations. Source-seed variability and paired image uncertainty
are distinct; no significance/equivalence test was introduced after outcomes.
Architecture, input encoding, compression fraction and updates/epoch differ
between settings, so different residual gaps cannot be assigned to dataset alone.

The old three-epoch experiment used a cosine schedule reaching its minimum at
epoch3. The new epoch3snapshot belongs to a30epoch schedule and is not that
historical point. Compare1/3/10/30within this new trajectory for budget dynamics;
keep old results separately labelled. No test-set inference or new compression
gain, learned-semantics or method-novelty claim follows from this study.

## Next required question

The user's required next study remains component-level selection plus regrouping
versus whole-unit pruning, at matched actual compact architecture/real parameter
budget and matched repair, across at least three independent source parents,
with the old selector's candidate/completion bias removed and intact answers
allowed. This completed fixed-retained-set study does not answer that practical
selection question. See[the research plan](../../docs/PLAN_IJCNN2027.md#required-next-component-selection-versus-whole-unit-pruning).
The manuscript has not yet been rewritten to include these30epoch results.
