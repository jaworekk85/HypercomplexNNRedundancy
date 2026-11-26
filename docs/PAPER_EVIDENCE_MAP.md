# Paper working title and evidence map

Updated: 2026-11-26. This records the paper-positioning discussion after the
completed exchange pilot. It is not an article draft, registered numerical
protocol or claim of sufficient novelty. [Current state](CURRENT.md),
[research plan](PLAN_IJCNN2027.md), [report index](RESEARCH_STATUS.md).
The user authorized an initial article draft on2026-11-26: [LaTeX](../paper/main.tex),
[PDF](../paper/draft.pdf), [build and review notes](../paper/README.md). This evidence
map remains the editorial record; drafting is not a claim of submission readiness.
The separately authorized nine-model EuroSAT replication is now
completed under its own [corrected FP32 protocol](../results/eurosat_grouping_fp32_replication/PROTOCOL.md).
The first version's archived-TF32 replay failure is preserved; all new numerical
references are collected in deterministic FP32 with the same frozen assignments.

## Working title

**Quaternion Channel Grouping under Structural Compression: Reconstruction,
Realizability, and Repair**

Proposed working title, not a final submission title. Candidate empirical
contribution: separate the effects of fixed retained scalar information,
Hamilton-constrained consumer maps, physical producer realization and bounded
repair. No new algebra, best compression algorithm, learned semantic entities
or publication-ready novelty established. See the
[contribution audit](literature/PAPER_CONTRIBUTION_AUDIT_2026-11-26.md).

## Hypothesis

How much observed damage from breaking quaternion groups remains when scalar
retention, actual architecture size and repair budget are controlled, and what
do initialization/statistics controls permit us to infer about that damage?
The paper should answer the scoped question rather than assume groups must
always be preserved or promise that they can be improved by reassembly.

## Existing measured evidence and proposed role

| Study family | Proposed paper use | Scientific boundary |
|---|---|---|
| [Real/Q ResNet baselines](../results/resnet20_baselines/REPORT.md), [compression](../results/resnet20_compression/REPORT.md), [replication2027](../results/resnet20_key_replication/seed_2027/REPORT.md), [replication2028](../results/resnet20_key_replication/seed_2028/REPORT.md) | Baseline quality, actual compact parameter counts, reconstruction versus direct removal/folding, repeated source seeds. | Nine-site whole-unit pruning differs from two-site regrouping; no matched-mask or matched-width real/Q equivalence. Compare within the registered scope. |
| Earlier redundancy, [CNN compression](../results/cnn_compression/REPORT.md), [compression curves](../results/compression_curve/REPORT.md), [activation sensitivity](../results/activation_sensitivity/REPORT.md) | Motivation, selected background results or additional repository evidence. | Different models/protocols; activation experiments do not directly establish robustness of the later grouping result. These are not independent confirmations of every later claim. |
| [Synthetic reassembly](../results/component_reassembly_diagnostic/REPORT.md) | Small explanatory example and elementary producer compatibility/projection mathematics. | Deliberately constructed favorable/less-compatible examples; not real-data gain or a new theorem/selector. |
| [Reassembly pilot](../results/resnet_component_reassembly_pilot/REPORT.md) | Transparent origin of the hypothesis, optionally a compact negative-result table. |90.8%versus90.4%is one seed, changed selections and a biased completion mechanism; the+0.5ppgate failed. Do not mix those masks with later controls. |
| [Grouping mechanism study](../results/grouping_mechanism_study/REPORT.md) | Core current experiment: same retained scalar sets; exact-feature versus physical realization; before/after bounded repair; three source seeds. | Two CIFAR ResNet sites, fixed assignments and three-epoch repair. Not universal optimality or semantic necessity. Producer-only fitting improves feature MSE12/12but worsens consumer MSE4/12. |
| [Initialization](../results/grouping_initialization_control/INTERPRETATION.md), [statistics/EuroSAT local diagnostics](../results/grouping_statistics_control/INTERPRETATION.md), [BN-only accuracy](../results/grouping_bn_accuracy_control/INTERPRETATION.md) | Key interpretation controls in the main analysis, with detailed grids retained in the repository. | Initial grouping advantage exists; uniform training amplification is not robust to statistical interventions. Standardization changes the diagnostic teacher. One failed BN recipe does not eliminate all normalization explanations. These local EuroSAT diagnostics are distinct from the subsequently completed full-network FP32 replication below. |
| [Fixed-set adaptive exchange](../results/grouping_exchange_pilot/INTERPRETATION.md) | Negative adaptive comparator, addressing the objection that only random mixing was tested. |298local fits selected original groups in every source seed. Identical initializations reuse historical repairs; no new independent training replication, equivalence or global-optimum claim. |
| [Structure catalogue](../results/resnet_structure_catalogue/REPORT.md), [PHM basis adaptation](../results/phm_basis_compression_pilot/REPORT.md), PHM baselines/initialization | Selected negative findings or supporting repository material if they help the final question. | Related but separate hypotheses and budgets; failed gates stay failed. Do not introduce a second main-method story or compare incompatible masks/budgets as one ablation. |
| [Phase0](../results/phase0/FEASIBILITY.md), gradient/update audits, source/data/checkpoint checks | Implementation provenance and reproducibility, with brief paper coverage and full repository records. | Engineering verification is not performance evidence or an independent scientific contribution. Preserve the actual upstream compatibility failure and credit canonical framework citations. |

These are editorial priorities, not permission to hide negative outcomes.
Experiments that affect a stated claim must be disclosed even if their complete
tables live in the repository. Final supplementary-file allowances must be
verified against the eventual venue instructions; availability is not assumed.

## Completed additions and remaining work

1. **Completed: full-network EuroSAT grouping replication.**
   original, role-preserving100%mixed and free-role100%mixed conditions, three
   existing quaternion parent seeds, nine actual compact models. Hold retained
   scalar sets, physical size and repair recipe fixed within each source seed.
   Reuse frozen local mask/assignment definitions when compatible, with the
   random draw fixed before outcomes. Record parameter counts, before/after
   accuracy/CE and producer realization. Fit-only fitting; CHECK/validation do
   not select groups or final epoch. This changes dataset AND architecture/input
   encoding, so any difference cannot be assigned to dataset alone.
   All9models completed; final means93.33%original,86.57%role,83.83%free.
   All6paired final comparisons favor original. See the
   [interpretation](../results/eurosat_grouping_fp32_replication/INTERPRETATION.md).
2. **Consolidate evidence (shared grouping overview completed).** The
   [cross-dataset table/figure](../results/eurosat_grouping_fp32_replication/CROSS_DATASET.md)
   now links physical grouping and repaired classification. Use it with the
   earlier reconstruction studies to assemble the final paper presentation,
   reconstruction, physical realization and repaired classification; distinguish
   discovery from replication, fractions of hidden width from whole-model
   savings, source seeds from random draws, and reused from newly trained models.
   Show unfavorable results, seed-level values and protocol differences.
3. **Lock and evaluate the final procedure.** Freeze selected methods,
   hyperparameters and reporting choices before final test evaluation. Audit
   prior access to each proposed test split and record it accurately; do not
   relabel explored validation as blind confirmation. Final test evaluation
   assesses a locked claim and does not create method novelty.

The EuroSAT item is now completed under a separately registered
protocol, following new compact-CNN realization, gradient/update, replay and
source/data checks. It has44794real trainables versus73898parents and462attempted
updates per model (459--461successful); these differ from the CIFAR count. Preserve
existing frozen sources and report this budget difference. No other dimension/algebra search, new parent training, new
grouping optimizer or further normalization grid is queued.

## Interpretation to preserve

Current evidence supports a bounded study of architectural constraints and
compression behavior. It supports neither a superior new method nor the claim
that quaternion structure is universally indispensable. The completed EuroSAT
replication repeats the practical direction in a second setting with different
architecture/input/compression/repair budgets; its larger gap is not an isolated
dataset effect. Conference acceptance and sufficient novelty remain open.
