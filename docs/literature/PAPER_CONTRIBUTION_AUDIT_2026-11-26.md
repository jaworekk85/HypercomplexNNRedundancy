# Contribution audit after the completed grouping controls

Date: 2026-11-26. This is a research decision record, not an article draft or a
systematic priority review. Canonical citation database: [references.bib](../../references.bib).
This audit predates the separately authorized
[completed fixed-set exchange pilot](../../results/grouping_exchange_pilot/INTERPRETATION.md).
That bounded adaptive comparison found no gain and selected original groups
for all three source seeds. Read the proposed missing comparison below as the
pre-experiment decision; stronger algorithms and broader utility remain unproven.
Earlier metadata/export checks remain in
[the targeted review](GROUPING_AND_TRANSFER_2026-11-25.md); current source
inspection and search limits are recorded in
[the audit manifest](PAPER_CONTRIBUTION_AUDIT_2026-11-26.sources.json).
[Verification](PAPER_CONTRIBUTION_AUDIT_2026-11-26.verification.json) checks
canonical citation keys, metadata consistency, local evidence and frozen inputs.

## Decision

Continue developing a narrowly scoped empirical contribution about the limits
of component regrouping in Hamilton-network compression. Current results do
not establish a better pruning method, learned quaternion semantics, or a
publication-ready novelty claim. This is an assessment of the evidence, not a
prediction of IJCNN acceptance. The number of completed fits is not itself a
contribution.

The strongest candidate distinction is the joint experimental separation of
retained scalar information, admissible consumer maps, realizable compact
producers and subsequent repair, with initialization and statistical controls.
The checked sources address neighboring parts; an exact predecessor for this
whole design was not identified in this bounded review. That is not proof of
priority or sufficient contribution strength.

The main remaining objection is substantive: random mixing is a perturbation
control. Beating it does not establish useful compression or that a sensible
alternative grouping cannot work. Addressing this objection has higher priority
than adding another normalization recipe or another algebra.

## Hypothesis and the narrow claim supported now

Research question: when original quaternion groups are broken during actual
structural compression, which measured costs arise from representation
constraints, producer realization and bounded repair?

Defensible present statement:

> In the tested fixed-retained-set Hamilton networks, original groups outperform
> the evaluated mixed assignments. A grouping penalty is already measurable at
> initialization; its training-associated amplification depends on the statistical
> intervention. In the compact CIFAR models, three-epoch repair substantially
> reduces the accuracy gap, while the tested BN-only recalibration does not.

This statement deliberately limits the architecture, assignments and repair
budget. It does not say original groups must always survive compression.

## Closest literature and the actual remaining distinction

The descriptions below concern the specifically inspected versions/sections.
Absence of a feature from an inspected passage is not evidence of its absence
from every version or related paper.

| Canonical key / primary source | Already established there | Our possible distinction and its limit |
|---|---|---|
| `Vecchi2020` ([published PDF](https://iris.uniroma1.it/retrieve/e3835329-db76-15e8-e053-a505fe0a3de9/Vecchi_Compressing_2020.pdf), sections 3/4) | Quaternion-targeted sparsity regularization, including quaternion-weight norms and BN-based neuron removal; comparison with component-wise penalties. | Post-training regrouping of retained hidden scalar features and physical producer/consumer compilation is a different intervention. Neither quaternion pruning nor component-versus-whole sparsification is new. |
| `Chaudhary2024` ([official abstract](https://www.isca-archive.org/interspeech_2024/chaudhary24_interspeech.html)) and `Singh2026` ([inspected preprint v1](https://arxiv.org/html/2510.21388v1), sections III/IV/IX-B) | Quaternion CNN compression, filter pruning, fine-tuning and output-level distillation in audio models. | We diagnose fixed-information hidden grouping and realizability. Generic QNN compression, fine-tuning or matching logits cannot be our headline novelty. Final journal/preprint text identity was not verified. |
| `PoppelbaumSchwung2025TimeCompression` ([preprint v2](https://arxiv.org/html/2403.11722v2), VI-B3/VII) | Component-wise versus whole-quaternion magnitude pooling, with components potentially selected from different temporal positions. | Our changes are hidden-channel membership/roles and smaller Hamilton layers, rather than temporal pooling. The broad idea of operating on separate quaternion components already exists. |
| `Wang2019FLGC` ([author abstract](https://arxiv.org/abs/1904.00346)) and `Zhang2021StructuredShuffle` ([proceedings PDF](https://proceedings.mlr.press/v161/zhang21b/zhang21b.pdf), section 3) | Learned channel groups and learned permutations for structured real-CNN compression. | A Hamilton-constrained grouping intervention is more specific. Learning groups or shuffling channels alone is not new; random shuffle is not a substitute for an adaptive comparator. |
| `He2017ChannelPruning` ([author abstract](https://arxiv.org/abs/1707.06168)); `VieiraValle2022ELM` (previously checked canonical record) | Selection followed by least-squares reconstruction in real CNNs; hypercomplex least squares via real embeddings. | Our solver is an implementation tool under Hamilton constraints, not a new least-squares algorithm or theorem. No new full-method check of the ELM paper was performed in this audit. |
| `Comminiello2024InductiveBiases` ([preprint v1](https://arxiv.org/html/2405.07024v1), sections 3/4); `ShahadatMaida2024CrossChannel` (previously inspected abstracts) | Hypercomplex inductive biases and cross-channel sharing already motivate sensitivity to component organization. | Measuring a specific post-compression intervention can add evidence. The existence of structured sharing, or damage after violating it, is not by itself surprising. |
| `Shomron2020BNRecalibration` ([author abstract](https://arxiv.org/abs/2010.05625)); `Jordan2023REPAIR` ([author abstract](https://arxiv.org/abs/2211.08403)) | BN/statistics recalibration and activation rescaling after model transformations have prior art. REPAIR concerns interpolated model merging. | Our negative BN-only control qualifies the compression interpretation; neither recalibration nor generic supervised repair is a new method. Failure of one recipe does not exclude all normalization explanations. |
| `Kang2026QuaternionCovarianceEEG` ([official conference abstract](https://www.apcmbe2026.org/index.php?ACT=View_Abs&GP=program/daily&SH=NON&num=O-T11-0419), abstract only) | Comparison of consecutive, nearby and random raw-EEG channel groupings for quaternion covariance representations. | Different from post-training hidden-layer pruning and Hamilton compilation. Nevertheless, a broad first-to-study-quaternion-grouping claim is unsafe. This is an official listed abstract, not a verified full journal article. |

The inspected V-EfficientNets comparison (`VieiraNetoValle2025VEfficientNets`)
already evaluates named 4D products. Another accuracy ranking would not answer
the missing adaptive-regrouping question. See the
[separate algebra note](OTHER_FOUR_DIMENSIONAL_ALGEBRAS_2026-11-26.md).

## Measured evidence, including results against the attractive story

| Evidence | Measurement | Claim allowed / blocked |
|---|---|---|
| [Reassembly pilot](../../results/resnet_component_reassembly_pilot/REPORT.md) | One seed: free-role 90.8%, whole-unit 90.4%, same compact size. Required +0.5 pp gate failed; most selected groups remained intact with a completion-reserve bias. | Hypothesis-generating example only. No replicated advantage or established method improvement. Keep the failed gate. |
| [Grouping mechanisms](../../results/grouping_mechanism_study/REPORT.md) | Three source seeds, two compressed sites, 15 compact models. After three repair epochs: original 90.13%, role-100% 88.60%, free-100% 87.87% on validation, means across seeds. Every mixed repaired model trails its matched original. | Evidence for a practical cost under these choices; not proof that an optimized alternative grouping must fail, or that more repair cannot close the gap. |
| Same study: producer-only fitting | Held-out producer-feature MSE improves in 12/12 local variants; consumer-output MSE worsens in 4/12. | Producer similarity alone need not select better functional reconstruction. This is secondary supporting evidence; reconstruction-aware objectives already have prior art. |
| [Initialization control](../../results/grouping_initialization_control/REPORT.md) | Penalty exists before training in all 12 seed/site/policy means; native trained penalty is larger in all 12. | Architectural baseline supported. Training-associated contrast requires the subsequent statistics controls. No learned-entity inference. |
| [Statistics control](../../results/grouping_statistics_control/INTERPRETATION.md) | After BN/marginal control, positive trained-minus-initial increments in 6/12 CIFAR and 4/12 EuroSAT comparisons, versus 12/12 and 11/12 native. | Uniform training amplification is not robust. EuroSAT provides local diagnostic replication on a second dataset/architecture, not compact-network accuracy replication. |
| [BN-only accuracy control](../../results/grouping_bn_accuracy_control/INTERPRETATION.md) | Mean free-100% accuracy gap: native 9.90 pp, BN-only 11.80 pp, historical repair 2.27 pp. BN closes 3/12 paired gaps; mean gaps widen in all four mixed conditions. Parents also decline by 1.0 pp on average. | This BN-only procedure does not recover the loss. It does not isolate a grouping-specific causal mechanism or make historical repair weights-only. |

Compact CIFAR models have 232,266 real trainables against 271,594 parents:
about 14.5% whole-network reduction. Removing half the hidden units at two
internal sites is not a 50% whole-network parameter saving. Masked sparsity,
quaternion-unit count and physical compact storage must remain separate.

## What is mathematically expected

Let X contain the fixed retained scalar features, Y the full consumer targets,
P a scalar permutation, and H the real matrix subspace representing the chosen
Hamilton consumer architecture. The local diagnostic solves the standard
constrained regression problem

    E(P) = min over A in H: ||Y - A P X||_F^2 + lambda ||A||_F^2.

For an unrestricted dense consumer, right multiplication by the orthogonal
permutation P can be absorbed into A without changing the optimum, including
this Frobenius ridge penalty. For Hamilton-constrained A, the class H P generally
differs from H. Whole-unit permutations preserve the class; arbitrary scalar
mixing need not. A reconstruction penalty can therefore exist before learning.
This is a standard linear-algebra explanation, not a novelty claim. With omitted
features and full-width targets, the original grouping need not have zero error.

The physical compiler also changes the feature-producing function. Its measured
error is not just E(P) plus a necessarily nonnegative producer-error term.
Exact-teacher-feature fitting is a diagnostic, not a universal lower bound on
all physically compiled models. Repair subsequently changes both weights and
statistics. Distinguishing these experiments is the candidate empirical value;
the separate effects have not all been causally identified.

## Reviewer objections and the smallest useful missing comparison

| Likely objection | What the completed work resolves | What remains |
|---|---|---|
| Randomly breaking weight sharing obviously hurts. | Initialization and dense inverse-permutation controls acknowledge the architectural baseline. | A useful alternative to the original assignment must be chosen from data, not merely sampled destructively. |
| The retained information or model size changed. | Fixed scalar sets within each seed/site and identical physical counts across conditions. | Pilot selection and mechanism-study masks differ; do not combine their accuracies as matched comparisons. |
| BN or fine-tuning explains everything. | Local statistical sensitivity measured; one practical BN-only recovery procedure failed. | No exhaustive normalization study, no isolation of weights from BN in historical repair, only a three-epoch repair horizon. |
| Is this more than a small CIFAR case study? | Three parent seeds; a second dataset/architecture for local diagnostics. | Whole-network regrouping accuracy is still CIFAR/two-site only; one mixed draw per full-model condition. |
| Does it beat existing compression? | Actual compact models and negative outcomes are available. | No demonstrated gain over a competitive adaptive regrouping/selection method. |

The highest-priority missing comparison is therefore **data-chosen regrouping
of exactly the same retained scalar set**, versus the original assignment at
the same compact architecture and repair budget. A prospective, bounded design
would use the existing CIFAR parents and two sites, allow the original groups
inside the search, and score the actual compiled producer/consumer on FIT data.
CHECK/validation must not choose the grouping or repair epoch. Report the search
cost and degree of mixing; give the intact comparator the same continuous
refitting opportunities. Random assignments remain diagnostic controls rather
than the sole practical competitor. The earlier pilot does not fill this gap:
its selection, completion bias and masks differ and its continuation gate failed.

If the search retains original groups or loses after repair, report that outcome;
it does not prove global optimality of intact groups or that every search fails.
If it improves reconstruction but not accuracy, the practical hypothesis has
failed under that budget. A later gain would require paired replication, not
selecting a favorable seed. This comparison is a proposal, not an authorized
run or completed result. Do not resurrect the failed pilot's automatic gate.

Second-dataset compact replication, another compression budget, longer repair
and final evaluation after locking the protocol would improve the scope of a
supported claim. They do not repair an unclear contribution by themselves and
are not an automatic queue. The already inspected validation split is exploratory;
an untouched test can assess a locked procedure but cannot create novelty.

## Paper positioning decision

Candidate working title: *Quaternion Grouping under Structural Compression:
Architectural Constraints, Realizability, and Repair*.

Use the fixed-information intervention and actual compact models as the main
experimental contribution; present initialization/statistics/BN outcomes as
limitations of a semantic interpretation. The producer-objective mismatch is a
secondary observation. Preserve the failed free-role gate and all contrary
results. Avoid claiming a novel solver, best accuracy, first component operation,
first grouping study, semantic indispensability, or general algebra invariance.

Current assessment: a coherent empirical-paper candidate, with an incremental
and still incompletely differentiated contribution. A submission headed only
by "random mixing hurts" would leave the strongest objection unanswered. Focus
the next decision on whether to address the adaptive-regrouping comparison or
deliberately keep a narrower case-study claim. No additional computation was
launched for this audit.

## Citation and search limits

Existing inspected primary citations already have canonical entries and earlier
raw exports/verification records; they were not duplicated. The one new EEG
abstract has a clearly labelled manually curated entry: the official page had
no visible BibTeX export, so no downloaded publisher BibTeX is claimed. No DOI,
proceedings pages, full-paper review or verified presentation is invented.

Targeted searches covered quaternion pruning/component regrouping/permutation
and channel grouping. They surfaced adjacent input-grouping and general
compression work, not a verified exact post-pruning predecessor. Search
omissions remain possible. Repository-only Q-SNet claims were screened but not
treated as a verified publication; a concatenated-matrix SVD paper was outside
the Hamilton-reassembly scope. Neither is used to support the candidate claims.
