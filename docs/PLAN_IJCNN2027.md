# IJCNN 2027 research plan

Updated: 2026-11-26. [CURRENT.md](CURRENT.md) records current execution and measured
state; the [study index](RESEARCH_STATUS.md) links the evidence. This file records
future decisions. The [JSON agenda](../configs/ijcnn_research_queue.json) contains
historical execution notes and is not an active scheduler.

## Required next: component selection versus whole-unit pruning

Execution was explicitly authorized2026-11-28. The bounded18-model comparison
is now prepared/frozen under[its own protocol](../results/component_selection_study/PROTOCOL.md):
both datasets/architectures, three existing seeds, energy whole-unit baseline,
optimized whole-unit selection and complete-assignment component selection.
The old completion reserve is absent; legal scalar replacement and role-swap
neighbors have reserved evaluation quotas alongside whole-unit substitutions.
Three starts and equal rounds/proposal caps bound each policy; this is not
exhaustive global search. All methods receive matched actual size and30epoch
repair. See[CURRENT.md](CURRENT.md)and live state for launch/progress.

**User decision,2026-11-26: this is the key next experiment after the current
repair curve, not an optional extension if time remains.** It addresses the
reviewers' practical "so what?": does preserving original quaternion units
help compression, or prevent retaining useful individual components?

- Compare selection of individual scalar components followed by regrouping and
  physical Hamilton compilation against selection/removal of whole quaternion units.
- Match actual compact architecture, real trainable parameter count/storage
  accounting and repair recipe/budget across methods. Report attempted/successful
  updates and any AMP skips. Masked sparsity alone does not satisfy the budget.
- Use several independent source-parent seeds(at least3), paired across methods.
  Retained scalar sets may differ: this study tests selection plus regrouping,
  unlike the current fixed-set grouping intervention.
- Audit and remove the earlier selector's candidate-coverage/completion bias.
  Mixed candidates must have a fair opportunity; original intact groups remain
  legitimate answers. Do not force reassembly to look beneficial or harmful.
- Freeze selection, fitting, repair and reporting choices under a separate
  protocol before new computation. Use training/calibration for development;
  final method/reporting lock and prior-split-access audit precede test evaluation.

The existing one-seed90.8%versus90.4%pilot is context, not completion of this
task. Report unfavorable results accurately. The next study is required for the
planned practical argument; improvement or novelty is not assumed. Exact-architecture
scratch, additional compression ratios, more draws and other suggested extensions
remain separate decisions. Do not modify frozen sources of the running study.

## External review response and next priority

The user supplied GPT and Claude manuscript feedback on2026-11-26.
[PAPER_REVIEW_ACTIONS.md](PAPER_REVIEW_ACTIONS.md) checks suggestions against
existing studies and separates editorial changes, the next proposed study and
conditional extensions. The draft now has a shorter abstract, contribution list,
conceptual grouping diagram and already measured uncompressed-parent accuracy.
No additional model was trained or test split evaluated.

Subsequent user authorization: commit/push the draft first, then execute the
matched repair study. The initial commits are pushed; new source/config/tests
and[protocol](../results/grouping_repair_curve/PROTOCOL.md)are now prepared/frozen.
Read[CURRENT.md](CURRENT.md)and the live state before reporting execution.
All18initial-model replays and both compact CUDA update audits passed. Only this
repair study ran under its separate frozen protocol and completed18/18models on
2026-11-26at20:52Europe/Warsaw; no automatic execution of every suggestion in the
reviews. [Completed interpretation](../results/grouping_repair_curve/INTERPRETATION.md)
reports small final CIFAR accuracy gaps(0.33pp)but persistent EuroSAT gaps
(2.13--2.37pp), with CIFAR CE slightly favoring mixed conditions. Integration into
the manuscript is pending; the required component-selection comparison above
remains next. Code/protocol commitf89a14bwas pushed before
launch2026-11-26at18:11Europe/Warsaw. Check live state rather than this plan for
progress or completion.

Highest-priority numerical proposal:18matched compact-model trajectories
(two settings, three source seeds, three assignments), a single30epoch repair
schedule from pre-repair initializations, snapshots0/1/3/10/30. Existing3epoch
cosine scheduling reaches its minimum at3; the new curve needs a separately
frozen schedule and cannot be presented as an unchanged continuation of that run.
The proposed endpoint, checks and exact-architecture scratch/selection gaps are
documented in the review response. New runs are not launched by this plan update.
Lock final methods/reporting and audit split access before test evaluation;
do not test now and continue choosing methods on those outcomes. Compression
grids, more parents/draws and new theory are conditional, not an automatic queue.

## Scientific objective

Study structured compression of fixed and learned hypercomplex representations.
The current paper direction is an empirical analysis of quaternion channel
grouping: retained information, constrained reconstruction, physically smaller
producer/consumer layers and bounded repair. Tested reassembly/adaptation methods
have not established a useful new-method gain or novelty. The earlier learned-
basis adaptation branch and bounded exchange pilot remain closed under their
failed continuation screens.

The real/Q ResNet-20 baselines, discovery compression, two-seed replication,
bounded structure catalogue, initialization diagnostic and from-scratch PHM4
baseline are complete. Do not restart their finished queues.

## Working title, evidence map and remaining work

**Quaternion Channel Grouping under Structural Compression: Reconstruction,
Realizability, and Repair** is the proposed working title.
[PAPER_EVIDENCE_MAP.md](PAPER_EVIDENCE_MAP.md) records the roles of earlier
baselines/compression/replication, synthetic realizability examples, grouping
studies, initialization/statistical controls and related PHM negative findings.
This is an editorial map, not an article draft or novelty certificate.

The user subsequently authorized the first manuscript draft, now in
[paper/main.tex](../paper/main.tex) and [paper/draft.pdf](../paper/draft.pdf).
The six-page IEEE conference draft records the completed evidence, including
negative controls, a generated bibliography/table subset and paired-seed figure.
Next review the scientific argument and limitations, then freeze final evaluation
and reporting choices. No new exploratory run or test inference was part of drafting.

The user has now authorized the next numerical study: original versus role-preserving100%mixed versus
free-role100%mixed on three existing EuroSAT Q parents, nine compact models,
fixed retained scalar sets/size and equal predetermined repair within seed.
This fills a full-network grouping-accuracy gap; existing EuroSAT work already
contains local diagnostics and other compression protocols. It also changes
architecture/input encoding, not dataset alone. The nine-model replication is
completed under a new checked/frozen protocol below. Do not alter frozen studies.

After that, consolidate a small set of cross-study tables/figures with seed-level
values, accurate model/parameter/repair budgets and negative results. Lock the
procedure before any final test evaluation and audit previous split access.
No new algebra/dimension grid, optimizer search or normalization sweep queued.

## Paper positioning after targeted literature review

The post-completion [2026-11-26 contribution audit](literature/PAPER_CONTRIBUTION_AUDIT_2026-11-26.md)
supersedes provisional interpretations: candidate empirical contribution is the
separation of retained information, Hamilton map constraints, producer realization
and repair. It remains incremental and incompletely differentiated. The primary
objection was the absence of a data-chosen fixed-set regrouping comparator;
the bounded exchange pilot below now evaluates one without improvement.
Random mixing alone does not establish useful compression. Its comparison used FIT-only selection,
actual compact compilation, original groups allowed, equal continuous-refit
opportunities and matched repair. The audit itself authorized no run; the user
separately authorized the completed pilot. Broader
dataset/repair/algebra grids cannot substitute for a clear contribution.

The [2026-11-25 review](literature/GROUPING_AND_TRANSFER_2026-11-25.md) finds
published precedents for substantive real-to-QNN adaptation and empirical
analysis. Candidate contribution: measure when original channel groups must
survive compression, separating retained information, producer realization and
repair. This is conditional on final evidence, not a declared novel method.
Component-vs-whole pooling already has prior art; shared-weight/grouping and
post-repair accuracy must not be overinterpreted. Keep the prior failed gate.
The official session explicitly includes hypercomplex generalizations of
real-valued architectures/algorithms. The current call lists six pages and
31January2027 regular-paper submission; recheck instructions before submission.
Verified citations/inspection limits and potential paper structure are in the
review; no additional numerical experiment is authorized by this literature work.

## Completed: EuroSAT full-network grouping replication

Completed2026-11-26 at03:32Europe/Warsaw:9models,18physical pairs,18new FP32
reference fits;84output and729source/input hashes passed.
[Interpretation](../results/eurosat_grouping_fp32_replication/INTERPRETATION.md).
Final means93.33%original,86.57%role,83.83%free; all6paired gaps favor original.
The direction repeats CIFAR, with different architecture/input encoding,
39.38%versus14.48%savings and462versus1125attempts. AMP skipped17/4158attempts.
No significance/semantic claim. Shared tables/figure are saved; next consolidate
the argument and locked final-evaluation protocol. No further run queued.

Authorized2026-11-26; corrected version launched03:22Europe/Warsaw after the
first version's archived-TF32/current-FP32 replay mismatch stopped it before SGD.
The failure is preserved; no frozen source or tolerance edited.
[Corrected protocol](../results/eurosat_grouping_fp32_replication/PROTOCOL.md): original and
role/free100%mixing, three source parents, nine physical compact CNN6 models.
Frozen retained sets/draw0, pairs0/2,44794vs73898real trainables; about39.4%
whole-model savings. All500FIT positions, fixed zero-centered Q consumer ridge,
matched three-epoch repair and462attempted updates/model. Report every seed,
before/after accuracy and CE, CHECK outcomes and all negative/reversed gaps.
No group search, new algebra/dimension/parent, official test or automatic sequel.

Assessment for the venue: the empirical scope fits the official
[hypercomplex session](https://ijcnn.org/2027/authors/special-sessions)
(rechecked2026-11-26). Contribution strength remains incremental/unproved;
replication improves scope but does not turn expected architectural sensitivity
into a novel method. Candidate value is the controlled separation of information,
realizability and repair, with contrary evidence explicitly reported. No acceptance
probability or guarantee is implied by topic fit.

## Completed: fixed-set exchange search and matched repair

Completed2026-11-26 at02:39:37Europe/Warsaw;538source/input and27artifact
hashes passed.298local fits,12network combinations. All3seeds selected the
original groups at both sites, yielding bitwise-identical initializations;
historical repair reused,0new repair runs. Mean gain0pp/CEdelta0, practical
screen failed. [Interpretation](../results/grouping_exchange_pilot/INTERPRETATION.md).
This bounded method is closed; no global-optimum claim or automatic further run.

Explicitly authorized2026-11-26 after the contribution audit.
[Protocol](../results/grouping_exchange_pilot/PROTOCOL.md): one bounded d4
experiment, existing three parents/two sites and same retained scalars. Two
starts, two rounds,16proposals; disjoint64inner/64outer FIT images. Score real
compiled block outputs then teacher KL for whole-model finalists; same full-FIT
refit and paired repair as intact. Reuse controls and bitwise-identical outcomes.
No other algebra/dimension or further automatic run. Prior art includes PQF
two-entry permutation search; no search/solver/KD novelty asserted. Read the
[targeted check](literature/GROUPING_EXCHANGE_PRIOR_ART_2026-11-26.md).

## Completed: BN-only full-model accuracy control

Completed2026-11-26 at01:43:51Europe/Warsaw;18models and audit passed.
[Interpretation](../results/grouping_bn_accuracy_control/INTERPRETATION.md):
BN-only recalibration did not recover the accuracy loss. Gap reduced3/12pairs,
widened9/12; mean gap widened in all4mixed conditions. Free100% gap9.90pp
native,11.80ppBN-only,2.27ppafter historical repair. Parent accuracy also
decreased1.0ppon average; no grouping-specific causal or semantic claim.
This bounded control is closed. No further computation queued.

User authorized2026-11-26. [Protocol](../results/grouping_bn_accuracy_control/PROTOCOL.md).
All15fixed compact cases plus3parents; same1024TRAIN calibration images,
sequential BN population recalibration only. No optimizer or feature scaling.
Primary: reduction in original-minus-mixed validation-accuracy gap, separately
for each of12seed/case pairs. All native/BN-only accuracy and CE, CHECK outcomes,
parent controls and historical epoch3repair retained. Historical repair is not
an intervention starting from the newly calibrated state. Previously explored
validation/CHECK splits, no official test, significance or novelty claim.
No automatic extra control, supervised repair or algebra study queued.

## Completed: BN/statistics controls and second-dataset check

Completed2026-11-26 at00:19:51Europe/Warsaw; audit passed.
[Interpretation](../results/grouping_statistics_control/INTERPRETATION.md):
the earlier uniform training-associated amplification is not robust to these
controls. Primary positive increments:6/12CIFAR and4/12EuroSAT, compared with
12/12and11/12native. Original-group advantage persists before and after training,
but does not establish learned quaternion semantics. Standardization recomputes
the diagnostic teacher; this is not a pure causal estimate for BN alone.
No new method, accuracy gain or paper readiness established. Next decision:
choose a narrow falsifiable contribution from the completed evidence before
authorizing more computation. No further run queued.

The user authorized the remaining statistics control and second-dataset check.
[Protocol](../results/grouping_statistics_control/PROTOCOL.md): common sequential
BN recalibration on1024unlabelled TRAIN images, then separate FIT-only marginal
standardization. CIFAR10 Q ResNet20 and already trained EuroSAT Q CNN6,3seeds,
early/late sites, initial/trained states.1260new ridge fits plus252CIFAR native
rows reused; no new supervised baselines. Primary endpoint and all native/BN-only
controls specified before fitting. Check [CURRENT](CURRENT.md) for execution.

Named4D algebras are a potential later extension. Valle's IJCNN2025 V-EfficientNet
study compares H/coquaternions/tessarines/hyperbolic quaternions. An extension
should ask whether grouping sensitivity depends on the multiplication, rather
than merely rank classification accuracies again. Each algebra needs its own
trained parent, equal scalar parameter budget and initialization/BN controls;
swapping the table inside a trained Hamilton model is not a fair substitute.
No multi-algebra run is queued and no algebra-independent claim is established.

## Completed: initialization control of the grouping penalty

The user authorized a paired step0/trained control on2026-11-25. The
[registered diagnostic](../results/grouping_initialization_control/PROTOCOL.md)
holds masks/group assignments fixed and normalizes reconstruction errors by
target energy. Crossed initial/trained feature and consumer states distinguish
architectural compatibility from changes associated with learning.504local
ridge fits,126predecessor replays, no new supervised training. Do not interpret
the earlier random-mixing penalty as learned quaternion semantics by itself.
No continuation threshold, significance claim or further automatic queue.

Completed at23:25Europe/Warsaw. The penalty is present at initialization and
larger after training in all12seed/site/policy means. Crossed controls associate
the late-block increase mainly with feature-state changes. The comparison still
includes BN/mean/covariance differences; no pure semantic effect is established.
The completed statistics control above weakens a universal training-amplification
interpretation; do not cite the initialization study alone for that claim.
See [interpretation](../results/grouping_initialization_control/INTERPRETATION.md).

## Completed: fixed-feature grouping mechanisms

The user authorized all three proposed follow-up experiments on2026-11-25.
The [registered grouping study](../results/grouping_mechanism_study/PROTOCOL.md)
separates exact-feature consumer grouping, physical producer error, and equal
repair. Same retained scalars within a seed/site; three existing parents and
two sites.138local cases (including repeated predetermined random assignments),
12additional producer-only fits,15matched3epoch repaired compact models.
Completed2026-11-25 at23:07Europe/Warsaw. Every mixed full-model condition
scored below its matched original control across the three source seeds.
See [CURRENT](CURRENT.md) and the [report](../results/grouping_mechanism_study/REPORT.md)
for measurements. Subsequent initialization/statistics controls above qualify
its interpretation. No novel-method, significance or
equivalence claim; original pilot's gate remains failed. No extra study queued.

The earlier selector imposed a reserve of untouched original units to ensure
completion, and its compiler represented original groups exactly but mixed
groups approximately. Thus18/20intact groups is not a controlled measure of
the inherent value of quaternion grouping. This follow-up avoids the reserve
and holds the retained scalar set fixed. It does not test random algebras.

## Completed candidate pilot: component selection and reassembly

A [technical diagnostic](../results/component_reassembly_diagnostic/REPORT.md)
and its [frozen configuration](../configs/component_reassembly_diagnostic.json)
verify the producer compatibility projection, physical smaller pair, gradients
and artifact reloads. Two constructed synthetic cases demonstrate possible
reassembly benefits and the gap between ideal selected teacher features and
features realizable by the smaller producer. All scalar selectors choose the
same winning quartet; no advantage over scalar-energy selection is established.

The [bounded real-data pilot](../results/resnet_component_reassembly_pilot/PROTOCOL.md)
was registered and launched on2026-11-25 at21:53Europe/Warsaw. It uses the existing
Q parent, disjoint fit/check data, whole-unit and scalar-energy controls, actual
producer/consumer reconstruction and equal3epoch repair for five models at
232266actual real trainables. Source-guided assignment
changes initialization; the exported architecture remains a standard smaller
Hamilton network. BN/ReLU and data covariance must be accounted for; the raw
Frobenius compatibility cost is not a post-activation optimum. Compare this exact
mechanism to Zhang2021StructuredShuffle, Wang2019FLGC, quaternion compression and
reconstruction prior art. Completed at22:01Europe/Warsaw: free_roles90.8%,
whole_joint90.4%, legacy89.8%, energy88.6%, role_fixed90.4%. Free-role CE was
lower, but the registered0.5pp accuracy gate FAILED (observed0.4pp).
No follow-up is queued; reassess the mechanism before allocating more compute.
Only block0 used mixed quartets, and its held-out local reconstruction error
was worse than the whole-unit control. This is not evidence of a general gain.
The continuation gate requires free-role selection to beat the best legacy,
whole-unit or energy control by0.5pp validation accuracy with CE no worse than
the lowest control CE. Report unfavorable outcomes and role-fixed ablations.

## Completed bounded pilot: continuation gate failed

Completed 2026-11-25. Adapt versus preserve reached 90.5% versus 90.2% validation
accuracy after equal repair, with slightly worse CE for adapt. The full-budget
compact scratch reference reached 90.4% at its CE-selected epoch. Local held-out
MSE improved by a mean 3.30%, but the registered classification/CE continuation
criteria failed. This single-seed result does not justify automatic replication
or establish equivalence. Reassess the scientific question before expanding the
branch. [Measured results](../results/phm_basis_compression_pilot/REPORT.md).

The [learned-basis compression protocol](../results/phm_basis_compression_pilot/protocol.json)
was registered before scientific fitting on 2026-11-25. Its
[configuration](../configs/phm_basis_compression_pilot.json) and
[runner](../scripts/phm_basis_compression_pilot.py) implement the following design:

1. Physically remove the same internal channels for all compared variants.
   Blocks 0/8 each lose 50% of their internal groups; residual outputs stay fixed.
2. Compare preserving, readapting and reinitializing learned bases. Specify
   initialization and what changes in each condition. Count frozen stored
   coefficients as well as trainable parameters; match actual architecture
   size and repair opportunities.
3. Include controls that distinguish compression-specific adaptation from
   ordinary further optimization and assess a compact model trained from
   scratch. Choose the exact bounded design in the registered protocol.
4. Use fit data for fitting/selection, a disjoint calibration check, and held-out
   evaluation with a fixed checkpoint-selection rule. Keep official test access
   separate from exploration. Report all failed checks and unfavorable cases.
5. Prespecify the effect/insight required to continue. If the pilot provides
   neither a useful quality-size trade-off nor a repeatable explanatory result,
   record the negative and reconsider the thesis before expanding compute.

The run contains six local 300-step fits on 500 images with 500 held-out checks,
three compact 3-epoch repairs, two uncompressed 3-epoch controls and one full-budget
200-epoch scratch model. All compact conditions store 233418 real parameters.
Preserve/adapt differ by 128 trainable basis coefficients; this is explicitly
reported rather than described as equal trainable freedom. Only the two chosen
conv2 bases can adapt during repair. The independent local fits use source-parent
targets, so their errors do not guarantee ordering in the combined network.

The actual frozen protocol/configuration, not this plan, defines initialization,
data splits, metrics and continuation thresholds. No follow-up is automatically
attached. Read [CURRENT.md](CURRENT.md) for live-state navigation.

## Conditional extensions and paper decision

After a useful pilot signal, replicate the principal result across independent
initialization seeds, then assess another compression level and CIFAR-100 under
separate protocols. Measure practical inference costs where relevant; parameter
savings alone do not establish speedups. Finer-width activation-fitted catalogues,
variable-n selection and allocation across all nine consumers require their own
justification and implementation. More datasets or a larger model alone do not
resolve the novelty question.

Retain the existing real/Q and simple compression controls. The negative Q/real
hybrid catalogue, failed PHM random start, and unsuccessful tail/logit, trainable-map
and centered-map pilots constrain the interpretation; do not rerun them without
a new scientific reason.

The paper needs a concrete finding, an explanation and evidence separating it
from prior work. Acceptance and worldwide priority cannot be promised. Prior
QNN pruning, reconstruction, learned PHM, structure allocation and adaptive
hypercomplex representations are indexed in [LITERATURE.md](../LITERATURE.md),
with full checks in the [archived literature audit](archive/LITERATURE_2026-11-07.md).

## Paper provenance and communication

Use keys from [references.bib](../references.bib), including He2017ChannelPruning,
Kim2020, VieiraValle2022ELM, Singh2026, ZhangEtAl2021, Grassucci2024,
IbnAhad2026HypercomplexQuantization, Liebenwein2021ALDS and Frantar2022SPDY
where their contributions are discussed.

Cite NiemczynowiczKycia2025 (KHNN, SoftwareX) for hypercomplex software and initial
implementation provenance. The tested Keras configuration failed parameter
tracking; current experiments use local PyTorch layers. Preserve that distinction
and the version-specific limitation in the [Phase 0 report](../results/phase0/FEASIBILITY.md).

The [draft to Marcos](MARCOS_DRAFT_2026-11-07.md) remains unsent; the author list is
open. No external message or submission follows automatically from this plan.
