# IJCNN 2027 research plan

Updated: 2026-11-25. [CURRENT.md](CURRENT.md) records current execution and measured
state; the [study index](RESEARCH_STATUS.md) links the evidence. This file records
future decisions. The [JSON agenda](../configs/ijcnn_research_queue.json) contains
historical execution notes and is not an active scheduler.

## Scientific objective

Study structured compression of fixed and learned hypercomplex representations.
The current user-proposed direction selects individual scalar components,
reassembles them into new quaternions and fits genuinely smaller Hamilton
producer/consumer layers. The earlier learned-basis adaptation branch is closed
under its failed continuation gate. Neither direction has established novelty.

The real/Q ResNet-20 baselines, discovery compression, two-seed replication,
bounded structure catalogue, initialization diagnostic and from-scratch PHM4
baseline are complete. Do not restart their finished queues.

## Paper positioning after targeted literature review

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
for measurements. Next action: analyze exact-feature versus compiled-producer
errors and repair effects before proposing more computation. No novel-method, significance or
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
