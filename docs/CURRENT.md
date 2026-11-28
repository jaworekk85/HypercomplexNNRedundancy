# Current research state

Updated: 2026-11-28. Read the relevant `state.json`, recent log tail and process
state before reporting activity. This page is a handoff, not a process monitor.

## Running: component selection versus whole-unit pruning

The user explicitly authorized execution2026-11-28. A separate new
[protocol](../results/component_selection_study/PROTOCOL.md),
[readiness](../results/component_selection_study/readiness.json)and
[state](../results/component_selection_study/state.json)are prepared/frozen.
11tests, a six-model two-architecture synthetic selection/refit/repair/completed-
resume smoke,6parent validation replays,12native reference/input batch replays
and both full compact CUDA gradient/update audits passed. The initial development
checksum-wrapper failure was fixed before freeze and is documented in
[DEVELOPMENT.md](../results/component_selection_study/DEVELOPMENT.md).
Code/protocol were committed/pushed as05df080before launch2026-11-28at
03:40Europe/Warsaw. Windows launcher PID23836, worker PID25164.
Logs:`results/component_selection_study/runner_20261128_024005.log`and`.err`.
Live check at03:41confirmed candidate fitting for CIFAR seed2026block0,
including whole-unit and component proposals, with no stderr output. This is
launch verification, not a measured gain. Read current state/log/process before
reporting progress. STOP:`results/component_selection_study/STOP`; resume after
removing STOP with`.venv-cuda/Scripts/python.exe -B -u -m scripts.component_selection_study --run`.

18models:both settings,3source seeds, unit_energy/unit_search/component_search.
All methods have identical actual compact architecture/parameter counts and the
same30epoch repair recipe. Selection uses inner64/outer64FIT only; retained
scalar sets can change. Complete assignments, whole-unit replacement moves for
both policies, scalar replacement/role swaps for component search, three starts
and equal bounded rounds/proposal caps. There is no partial-quartet completion
reserve. Whole-energy baseline is a legal network-selection answer; no useful
gain, novel solver or global optimum is assumed. Native/full refits use500FIT,
CHECK/validation report outcomes, no test inference. Scientific selection and
repair are now running under this frozen protocol; no automatic follow-up.

## Required next experiment after the repair curve

User decision2026-11-26: **component-level selection + regrouping versus
whole-quaternion-unit pruning is a key required next study, not an optional
"if time remains" extension.** See[the research plan](PLAN_IJCNN2027.md#required-next-component-selection-versus-whole-unit-pruning).
Match actual compact architecture/real parameter budget and repair protocol
across methods; use several independent source seeds(at least3). Allow retained
scalar sets to differ, since selection itself is the intervention. Audit/remove
the old selector's candidate/completion bias without forcing mixed groups or
excluding the intact answer. The existing90.8%/90.4%single-seed pilot does not
satisfy this requirement. Prepare a separate frozen protocol after the current
repair study, before final method/reporting lock and test evaluation. No claim
of gain/novelty or automatic launch follows from this planning note.

## Completed: matched repair trajectories

Completed2026-11-26at20:52:27Europe/Warsaw:18/18models,90snapshots,
142787successful/142830attempted updates(43AMP skips). Audit and independent
recheck passed798frozen hashes and180snapshot artifacts. Worker exited.
[Interpretation](../results/grouping_repair_curve/INTERPRETATION.md),
[report](../results/grouping_repair_curve/REPORT.md).
[Full repair curve](../results/grouping_repair_curve/figures/repair_curve.png),
[PDF](../results/grouping_repair_curve/figures/repair_curve.pdf), generated from
all saved epoch histories on2026-11-28(no new inference).
Epoch30validation means(original/role/free):CIFAR91.33/91.00/91.00%,
EuroSAT94.80/92.43/92.67%. Final gaps0.33/0.33ppand2.37/2.13pp;
CIFAR mean CE slightly favors mixed models. No equivalence/significance claim.
The new30epoch cosine trajectory differs from the old3epoch recipe; compare
its own1/3/10/30points. Manuscript integration remains pending. The required
component-selection comparison above is next; no further training is active.

The user explicitly authorized the repair experiment after requesting commits/push
on2026-11-26. Manuscript/review commits7293f53and8da0a8bwere pushed first.
New independent source/config: [protocol](../results/grouping_repair_curve/PROTOCOL.md),
[readiness](../results/grouping_repair_curve/readiness.json),
[state](../results/grouping_repair_curve/state.json),
[development checks](../results/grouping_repair_curve/DEVELOPMENT.md).
Preparation passed5CPU tests, a full synthetic trajectory/completed resume,
all18bitwise initial-model/logit replays, and discarded CUDA tracked-variable/
all-four-component gradient/update audits in both settings. New sources are frozen;
no predecessor was edited. The checked experiment was committed/pushed asf89a14b,
then launched2026-11-26at18:11Europe/Warsaw. Windows launcher PID14980;
worker PID23888from the live state. Logs:
`results/grouping_repair_curve/runner_20261126_171118.log`and`.err`.
Read state/recent log/process before reporting progress; completion is recorded above.
Launch verified at18:12Europe/Warsaw:first CIFAR seed2026original epoch completed,
375/375successful/attempted updates, epoch1model/logit snapshot and reload passed.
This is a launch check, not the final comparison between grouping conditions.

Scope:18trajectories(two settings, three existing parent seeds, three frozen
assignments), snapshots0/1/3/10/30, same pre-repair compact weights/masks/counts.
One30epoch cosine schedule per trajectory; historical3epoch scheduling is a
different recipe. Primary final epoch30paired validation gaps/CE, with reversals
retained and no best-epoch selection. Actual attempted/successful AMP counts and
per-example validation/reconstruction-check logits are stored. Test not evaluated.
No scratch, new masks/draws, compression ratios or parent/algebra grid is queued.
STOP:`results/grouping_repair_curve/STOP`; resume after removing STOP with
`.venv-cuda/Scripts/python.exe -B -u -m scripts.grouping_repair_curve --run`.

## Manuscript draft: first complete LaTeX version

The user authorized drafting on2026-11-26. [Manuscript source](../paper/main.tex),
[reading PDF](../paper/draft.pdf), [build/evidence/review notes](../paper/README.md).
Initial IEEE conference draft:6Letter pages, anonymous authors, actual Times-compatible
fonts,5printed tables and2figures (six table sources are generated; full seed values
also remain in the unused seed-table source and plotted points). The expansion adds all-draw local exact/physical
reconstruction means and two explicitly constructed synthetic reassembly examples;
these use existing results, not new fits or classification. Synthetic selection changes
the retained scalar set and all scalar selectors tie, unlike fixed-mask comparisons.
Tables/citation subset regenerate from canonical saved evidence;24model rows,
126local rows,8synthetic rows and23citation keys checked. Six existing canonical
sources now expand related work/statistical interpretation; two raw BibTeX exports
were newly downloaded, and Kim2020pages/publisher verified. Short AI-assistance
citations remain per affected section under the explicit IJCNN instruction, while
the acknowledgment now states the actual assistance concisely; review reminders
stay in the manuscript README. [Reference/policy verification](../paper/literature_verification.json).
Scientific and
human author review remain pending; this is not submission-ready or final-test
evidence. The current initial IJCNN2027 instructions were checked online and
their required AI-drafting disclosure is included. The user has now supplied external
GPT and Claude model reviews; these are assessed in[PAPER_REVIEW_ACTIONS.md](PAPER_REVIEW_ACTIONS.md).
They are not independent artifact verification or calibrated acceptance predictions.
The editorial revision shortens the abstract, adds explicit empirical contributions,
defines producer/consumer, adds a same-component grouping diagram and existing parent
accuracy(91.03%CIFAR,96.67%EuroSATmeans). Implementation failure details remain in the
repository; validation-only status and failed scientific comparisons remain explicit.
Unchanged IEEEtran class/style were obtained from CTAN after direct IEEE ZIP
requests returned empty payloads; vendor/tool provenance is recorded in`paper/vendor`.
Numerical protocols/sources were not edited and no new model was evaluated.
The previously proposed repair study is now separately authorized and prepared
above. Final procedure/reporting lock and prior-split-access audit precede test
evaluation. The current article's three-epoch results remain unchanged.

The user requested a concise literature review during the grouping study.
The [2026-11-25 adaptation/grouping review](literature/GROUPING_AND_TRANSFER_2026-11-25.md)
is complete:10new canonical entries, verified journal/conference-status upgrades
for Valle/V-EfficientNets, three existing compression citations rechecked,
15raw BibTeX exports including the2026-11-26 algebra follow-up.
[Verification](literature/exports/2026-11-25/verification.json) discloses the
manually checked REPAIR entry and failed export requests. The official IJCNN2027
session includes generalization of real-valued methods to hypercomplex networks;
this supports topic fit, not guaranteed acceptance. Numerical protocols unchanged.

## Latest completed: EuroSAT full-network grouping replication

Completed2026-11-26 at03:32:03Europe/Warsaw:9models,18physical pairs and18fresh
FP32reference fits. Audit passed84output and729source/input hashes.
[Interpretation](../results/eurosat_grouping_fp32_replication/INTERPRETATION.md),
[report](../results/eurosat_grouping_fp32_replication/REPORT.md),
[cross-dataset table](../results/eurosat_grouping_fp32_replication/CROSS_DATASET.md),
[figure](../results/eurosat_grouping_fp32_replication/figures/grouping_replications.png).
Final validation means93.33%original,86.57%role,83.83%free; original wins6/6pairs.
Mean role/free gaps shrink21.30/29.33ppto6.77/9.50pp.44794vs73898real parameters,
39.38%savings. AMP4141successful/4158attempts,17skips,459--461successful/model.
This repeats the practical direction in a second dataset/architecture, not an
isolated dataset effect, learned-semantics or novel-method claim. No run active.

Explicitly authorized2026-11-26. Corrected FP32 version launched at03:22:19
Europe/Warsaw after new checks/source freeze.
[State](../results/eurosat_grouping_fp32_replication/state.json),
[launch](../results/eurosat_grouping_fp32_replication/launch.json),
[protocol](../results/eurosat_grouping_fp32_replication/PROTOCOL.md).
The first version stopped before SGD at03:16: archived late EuroSAT features
replay only with cuDNN TF32 enabled in the tested batch, while the baseline
uses TF32 disabled. [Failed record](../results/eurosat_grouping_replication/state.json),
[precision diagnostic](../results/eurosat_grouping_replication/precision_diagnostic.json).
No frozen source/tolerance changed. New version regenerates native features,
targets/ridge scales and18exact-feature fits in deterministic FP32, keeping all
inherited scalar masks and fixed draw0assignments unchanged.
Nine actual compact CNN6 models: original, role-preserving100%mixed, free100%mixed,
three existing source seeds. Frozen scalar sets/draw0from the local EuroSAT
diagnostic; halve hidden widths inside pairs0/2 only.44794real trainables versus
73898parents (about39.4%whole-model saving, not CIFAR's14.5%). All500FIT examples
fit each physical consumer; separate CHECK/validation report outcomes. Equal
three-epoch repair within seed,462attempted updates per model. No official test,
new parent/algebra/dimension/search or automatic further study. Reversals retained.

Eight corrected-version CPU checks, six-pair/three-model synthetic end-to-end
with completed resume, and discarded compact CUDA gradient/update audit passed.
Initial legacy-metadata and first scientific replay failures are retained, with
the correction documented in DEVELOPMENT.md. STOP:
`results/eurosat_grouping_fp32_replication/STOP`; remove and resume with
`.venv-cuda/Scripts/python.exe -B -u -m scripts.eurosat_grouping_fp32_replication --run`.

## Paper direction and remaining work

The latest discussion is recorded in [PAPER_EVIDENCE_MAP.md](PAPER_EVIDENCE_MAP.md):
working title **Quaternion Channel Grouping under Structural Compression:
Reconstruction, Realizability, and Repair**, the role of earlier experiments,
and the distinction between measured findings and candidate contribution.
The working direction is an empirical paper, not a demonstrated better method.

The previously recommended numerical study is now completed above: full-network EuroSAT grouping replication,
original/role-preserving100%mixed/free-role100%mixed, three existing parents,
nine actual compact models with fixed retained sets and equal size/repair within
seed. Local and physical grouping evidence now both exist. Shared cross-dataset
tables/figures are saved above. Next consolidate the paper argument and lock the final procedure
before test evaluation, accurately recording prior split access. No new algebra,
dimension, parent, normalization grid or optimizer search currently queued.

## Latest completed: bounded fixed-set grouping exchange experiment

Completed2026-11-26 at02:39:37Europe/Warsaw; final audit passed538source/input
hashes and27output artifacts. [Interpretation](../results/grouping_exchange_pilot/INTERPRETATION.md),
[report](../results/grouping_exchange_pilot/REPORT.md),
[descriptive evidence](../results/grouping_exchange_pilot/descriptive_summary.json).
298unique local fits,12whole-network finalist combinations; all3seeds selected
original groups at both sites. Mixed starts improved their selection MSE by
6.59--26.68%, but did not win the network selection. Original starts stayed put.
All final initializations bitwise equal to original; verified historical repairs
were reused,0new SGD runs. Accuracy89.8/91.0/89.6%,gain0pp,CEdelta0; the new
prospective+0.5pp screen failed. This is checkpoint reuse, not independent
replication or equivalence evidence. Bounded search does not prove global
optimality or rule out other grouping algorithms. Pilot closed; no further run
queued and no dimension/algebra extension started.

User explicitly authorized the literature check and experiment2026-11-26.
Prepared/frozen after seven tests, full synthetic search/refit/repair/resume and
discarded-copy compact CUDA gradient/update audit. Launched at02:36Europe/Warsaw.
[State](../results/grouping_exchange_pilot/state.json),
[launch](../results/grouping_exchange_pilot/launch.json),
[protocol](../results/grouping_exchange_pilot/PROTOCOL.md).
Three existing CIFAR parents, sites0/8, SAME retained scalar sets and232266
real trainables. Two warm starts, at most2rounds of16swap proposals/site/start;
inner64FIT images, disjoint outer64FIT images. Actual block-output scores then
teacher-KL whole-network selection. Final full500FIT consumer refits match the
intact control and must bitwise replay it. At most3new three-epoch repairs;
bitwise-identical original initializations reuse their verified existing repair.
No dimension/algebra/parent grid or test evaluation. All outcomes reported.
Practical screen: mean paired accuracy gain>=0.5ppand mean CE not increased;
no significance/novelty claim or automatic continuation, prior failed gate kept.
STOP `results/grouping_exchange_pilot/STOP`; remove and resume with
`.venv-cuda/Scripts/python.exe -B -u -m scripts.grouping_exchange_pilot --run`.

[Specific literature check](literature/GROUPING_EXCHANGE_PRIOR_ART_2026-11-26.md)
adds PQF's published two-entry permutation search and preserves the distinction
between known search ingredients and the specific Hamilton compression test.
Author-provided BibTeX extracted/checked; direct download/access failures disclosed.

## Earlier contribution audit, completed before the exchange pilot

The user authorized a literature-versus-results assessment, completed2026-11-26.
[Contribution audit](literature/PAPER_CONTRIBUTION_AUDIT_2026-11-26.md) maps
the closest sources to measured claims and reviewer objections. Candidate:
an empirical study separating fixed retained information, Hamilton consumer
constraints, physical producer realization and bounded repair. No demonstrated
better compression method, learned semantics, priority or paper readiness.

The audit's strongest unresolved objection was random mixing as the sole practical
alternative. Its proposed highest-priority comparison was data-chosen regrouping with
the SAME retained scalar set and actual compact architecture, original groups
allowed, FIT-only search and matched repair/refitting opportunities. The earlier
biased/failed single-seed pilot did not settle it. The separately authorized
bounded experiment above now evaluates that comparison without finding a gain;
broader search/utility remains unproven. No automatic normalization/algebra queue.
The audit also adds a clearly labelled official EEG-grouping conference abstract
to references.bib; metadata manually checked, no publisher BibTeX download claimed.

User follow-up on smarter grouping and2/3/8components is recorded in the
[grouping/dimension note](literature/GROUP_SELECTION_AND_DIMENSION_2026-11-26.md).
The note proposed fixed-set exchanges from the intact assignment, scored on the
actual compiled block/network; separate later dimension choice at equal actual
storage. Arbitrary-n PHM/adaptive dimensions have prior art and earlier negative
PHM results remain negative. Its d4exchange follow-up above completed; no other
dimension was launched.

## Latest completed: BN-only full-network accuracy control

Completed2026-11-26 at01:43:51Europe/Warsaw. All18models,72output artifact
hashes and565source/input hashes passed; native compact metrics/logits and BN
saved-state/logits replayed exactly, all weights/BN affine/non-BN buffers fixed.
[Interpretation](../results/grouping_bn_accuracy_control/INTERPRETATION.md),
[measured report](../results/grouping_bn_accuracy_control/REPORT.md),
[figure](../results/grouping_bn_accuracy_control/figures/validation_accuracy.png).

BN-only calibration did NOT recover accuracy: the original-minus-mixed gap
closed in3/12pairs and widened in9/12, with a larger mean gap for all4mixed
cases. Free100% mean gap: native9.90pp, BN-only11.80pp, historical repair2.27pp.
Parent controls also degraded on average1.0pp, so this is not specific to mixed
models. This particular BN-only recovery recipe failed; do not infer that BN
never matters, that historical repair is weights-only, or learned Q semantics.
No further computation queued. This bounded control is closed; any new method
or algebra extension requires a separate falsifiable question/protocol.

Separately authorized2026-11-26 and launched at01:42:33Europe/Warsaw.
[State](../results/grouping_bn_accuracy_control/state.json),
[launch](../results/grouping_bn_accuracy_control/launch.json),
[protocol](../results/grouping_bn_accuracy_control/PROTOCOL.md).
Reconstruct the15saved compact grouping-study models and3uncompressed parents.
Recalibrate only BN running statistics on the same fixed1024TRAIN images;
measure validation/CHECK accuracy and CE. No feature standardization or SGD.
Native assembled models must replay historical metrics/logits, and every
trainable parameter/BN affine and non-BN buffer must remain bitwise unchanged.
Primary: original-minus-mixed validation-accuracy gap before/after BN, all12
paired seed/case contrasts; positive gap reduction means BN reduces the penalty.
Historical3epoch repair is a comparator, not training from the new BN state.
The already inspected validation/CHECK splits make this exploratory, not blind
confirmation. No test or extra dataset/algebra/repair run.

Ten CPU checks, two-model synthetic end-to-end with completed resume, and a
discarded-copy compact-GPU full gradient/update audit passed before source
freeze. Initial reconstruction test failed from an incorrect helper signature;
fixed before freeze, failure retained in development_checks.json.
STOP: `results/grouping_bn_accuracy_control/STOP`. Resume after removing STOP:
`.venv-cuda/Scripts/python.exe -B -u -m scripts.grouping_bn_accuracy_control --run`.
No automatic continuation after these completed18models.

## Completed: BN/marginal statistics and EuroSAT replication

Completed2026-11-26 at00:19:51Europe/Warsaw: all1512rows,1260new fits;
completion audit passed. Postprocessing verified676source/input hashes,
72coefficient banks and12BN model hashes; all BN trainable parameters unchanged.
[Interpretation](../results/grouping_statistics_control/INTERPRETATION.md),
[measured report](../results/grouping_statistics_control/REPORT.md),
[figure](../results/grouping_statistics_control/figures/training_increment.png).

The earlier universal training-amplification interpretation does NOT survive
the statistics controls. With centered-energy normalization, positive training
increments go from12/12native to6/12standardized on CIFAR, and11/12to4/12on
EuroSAT. Original grouping still beats mean mixed grouping in all24primary
seed/site/policy comparisons at both initialization and trained states.
That advantage alone is compatible with an architectural baseline; learned
quaternion semantics are not established. These are reconstruction diagnostics,
not accuracy gains. BN recalibration changes intermediate behavior and the
standardized condition recomputes teacher targets; it is not a pure causal
isolation of BN. The separately authorized full-network follow-up above tests
actual accuracy after a BN-only intervention, rather than this modified teacher.

Authorized after the initialization control, prepared before source freeze and
launched2026-11-26 at00:07Europe/Warsaw. Check live
[state](../results/grouping_statistics_control/state.json),
[launch](../results/grouping_statistics_control/launch.json),
[plan](../results/grouping_statistics_control/PROTOCOL.md),
[protocol](../results/grouping_statistics_control/protocol.json).

CIFAR10 Q ResNet20 and existing EuroSAT four-band Q CNN6 parents, three seeds
each, early/late consumers. Same retained scalar sets and assignments within
dataset/seed/site. Native, sequential-BN-recalibrated and recalibrated-plus-FIT
marginal-standardized features; initial/trained weights.1512reported rows:
252CIFAR native predecessor rows reused,1260new ridge fits. No supervised
training, architecture/algebra change, classification or test/validation inference.

BN population moments use1024unlabelled TRAIN images disjoint from500FIT/500CHECK;
trainable parameters including BN affine must remain bitwise unchanged. Feature
standardization uses only FIT means/scales. Its targets are recomputed using
the unchanged full consumer: this is a modified diagnostic teacher, not a
function-preserving compact network. Primary endpoint is standardized-condition
mixed-minus-original error normalized by target energy centered using FIT means.
Raw-energy and native/BN-only results remain reported. Covariance, sparsity and
higher moments are not matched; no pure semantic inference or significance claim.

Seven CPU tests,36fit end-to-end smoke with completed-case resume and discarded
four-band GPU gradient/update audit passed before registration. Initial smoke
resume exposed tuple-vs-JSON-list assignment comparison; normalized before freeze,
failed attempt retained in development_checks.json. STOP/resume:
`results/grouping_statistics_control/STOP`, then remove and rerun
`.venv-cuda/Scripts/python.exe -B -u -m scripts.grouping_statistics_control --run`.
No study beyond this completed bounded authorization queued.

The user also asked about other4D algebras. Verified primary author preprint
`VieiraNetoValle2025VEfficientNets` compares quaternions, coquaternions,
tessarines and hyperbolic quaternions. Publisher-deposited conference DOI/BibTeX
now verified; same canonical key upgraded. A later named-algebra comparison
would need separately trained parents and correct multiplication/solver audits.
It has not been launched; no Hamilton-vs-random-algebra return.

## Latest completed initialization control

Completed2026-11-25 at23:25:09Europe/Warsaw:504ridge fits, all126predecessor
rows replayed,24coefficient banks reloaded exactly. Worker exited and stderr
empty. [Interpretation](../results/grouping_initialization_control/INTERPRETATION.md),
[report](../results/grouping_initialization_control/REPORT.md),
[descriptive audit](../results/grouping_initialization_control/descriptive_summary.json).

Original-group advantage exists at initialization in all12seed/site/policy
means; it grows after training in all12. Mean free-role normalized excess MSE:
block0initial0.064235/trained0.164739; block8initial0.017048/trained0.284265.
These are reconstruction deltas, NOT accuracy percentages. In the late block,
crossed controls associate most increase with feature-state changes, not trained
consumer coefficients alone. Unadapted versus trained BN/means/covariances
remain confounds; no learned semantic entities, novelty or significance proved.
The labelled post-hoc target-statistics check adds context, not a new endpoint.
No further computation is queued.

Authorized2026-11-25 after the user questioned whether random mixing creates
a straw-man comparison. Prepared and launched around23:22Europe/Warsaw.
[State](../results/grouping_initialization_control/state.json),
[launch](../results/grouping_initialization_control/launch.json),
[plan](../results/grouping_initialization_control/PROTOCOL.md),
[frozen protocol](../results/grouping_initialization_control/protocol.json).

Three parent seeds, blocks0/8, identical retained units and original/10role/10free
fully mixed assignments from the completed study. Four crossed cells of
initial/trained features and initial/trained consumer weights,504ridge fits;
no new network training, producer compilation, accuracy or test/validation
inference. Initial networks reproduce the frozen constructor/seed recipe, not
an independently archived epoch0checkpoint. Initial BN state is unadapted.

Primary metric is held-out SSE/target squared energy and its mixed-minus-original
penalty, compared before/after training. Crossed cells help separate feature
and consumer changes; BN and feature covariance still differ between states.
This diagnoses an architectural baseline rather than proving learned semantics.
Trained/trained126fits must replay previous results. Original-unit reordering,
dense permutation, target/input scaling and saved-coefficient replay checks.

Eight new tests, a12fit end-to-end CPU smoke with completed-case resume and a
discarded-copy full-GPU gradient/update audit passed before source registration.
Create `results/grouping_initialization_control/STOP` to pause; remove it and run
`.venv-cuda/Scripts/python.exe -B -u -m scripts.grouping_initialization_control --run`
to resume interrupted runs. Reinvocation verifies/reuses completed cells.

## Latest completed study: fixed-feature grouping mechanisms

Completed2026-11-25 at23:07:01Europe/Warsaw:138local cases,12additional
producer-only fits and15full models, each3repair epochs. Final artifact audit
passed180hashes; all models logged1125successful/1125attempted updates, no stderr.
[Measured report](../results/grouping_mechanism_study/REPORT.md),
[artifact audit](../results/grouping_mechanism_study/completion_audit.json).

Validation accuracy after repair, seeds2026/2027/2028:

| Condition | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| Original groups | 89.8% | 91.0% | 89.6% |
| Role-preserving50% mixed | 89.1% | 89.7% | 89.0% |
| Free-role50% mixed | 88.7% | 90.1% | 88.8% |
| Role-preserving100% mixed | 88.2% | 88.5% | 89.1% |
| Free-role100% mixed | 87.4% | 88.3% | 87.9% |

All mixed full-model conditions scored below original grouping within each
seed. This is an observation for the registered masks, random draws, compiler,
two sites and3epoch repair; it does not isolate a universal algebraic necessity.
The new activation-energy masks differ from the earlier joint-selector masks.
Interpret exact-feature and physical local comparisons separately using the
complete report. No additional computation is queued.

Authorized2026-11-25 after the user asked to distinguish original-quartet value,
producer compilation error and repair. The user asked for concise updates.
Prepared and launched around22:45Europe/Warsaw; read live
[state](../results/grouping_mechanism_study/state.json) and
[launch metadata](../results/grouping_mechanism_study/launch.json) before reporting.
[Registered plan](../results/grouping_mechanism_study/PROTOCOL.md),
[protocol](../results/grouping_mechanism_study/protocol.json),
[readiness](../results/grouping_mechanism_study/readiness.json).

Three existing Q parents (2026/2027/2028), blocks0/8. Within each seed/site ALL
conditions retain exactly the same scalar channels: the half of whole units
ranked highest by summed fit activation energy. This differs from the old joint
mask; original is a new matched control, not the historical90.4percent result.
Original grouping, role-preserving mixing, role-free mixing. Mixing at100percent
has10predetermined random draws per policy;50percent has one draw. Total138local
cases. No catalogue or original-unit completion reserve biases these draws.

1. Fit Hamilton consumers to exact retained teacher features: ALL500fit images,
   separate500check, zero-centered ridge and one fixed lambda per seed/site.
2. Compare physically projected producers on those same groups. Also fit ONLY
   producer kernels to selected-feature targets for100steps on the first fully
   mixed draw of each policy; fixed BN, then identical consumer fit. These extra
   producer fits are local diagnostics, never substituted into full models.
3. Five projected compact models per seed (original/role50/free50/role100/free100,
   draw0 fixed beforehand),15models total, each232266actual real trainables and
   equal3epoch repair. Before/after accuracy, CE and paired logits retained.

Twenty-seven tests, ten-local/five-model CPU end-to-end smoke with artifact
replay/completed resume, discarded-copy GPU full-model updates and producer-only
gradient/update checks passed before source registration. Check/validation never
select grouping draws or endpoints. No official-test inference; explored
validation is not fresh confirmation. No equivalence/significance/novelty claim.

The runner's STOP and completed-epoch resume paths remain available for
interrupted studies. Reinvocation verifies/reuses this completed run and does
not launch a further experiment.
Earlier pilot's failed continuation gate stays failed; this study asks a new,
explicitly exploratory mechanism question rather than lowering that gate.

## Latest completed pilot: component reassembly

Completed2026-11-25 at22:01:35 Europe/Warsaw:10local cases,5models, all3repair
epochs each. Worker exited, GPU idle and stderr empty. All452 registered
source/input/final-artifact hashes passed the
[completion audit](../results/resnet_component_reassembly_pilot/completion_audit.json).
Each model made1125successful optimizer updates out of1125attempts.

Final validation accuracy: legacy89.8%, whole_joint90.4%, scalar_energy88.6%,
role_fixed90.4%, free_roles90.8%. CE for whole/free:0.297067/0.295178. The
registered continuation gate FAILED: free improved accuracy by0.4pp rather
than0.5pp, although CE improved. No replication is queued.
[Full measured report](../results/resnet_component_reassembly_pilot/REPORT.md).

Free-role selection made two mixed quartets with two changed scalar roles in
block0; block8 retained intact quartets. Its block0 local held-out MSE was worse
than whole-unit selection (0.408960 versus0.395997). Role-fixed selection chose
only intact quartets at both sites. Thus the classification difference is an
exploratory single-seed observation; no local-error advantage, significance,
novelty or reliable general gain is established. Reassess before more compute.

Registered and launched 2026-11-25 at 21:53 Europe/Warsaw. Worker PID at
launch: 21244 (launcher 15784). Check the live
[state](../results/resnet_component_reassembly_pilot/state.json) before reporting
progress. Logs: `results/resnet_component_reassembly_pilot/runner_20261125_215328.log`
and `.err`. [Plan](../results/resnet_component_reassembly_pilot/PROTOCOL.md),
[frozen protocol](../results/resnet_component_reassembly_pilot/protocol.json),
[readiness](../results/resnet_component_reassembly_pilot/readiness.json).

One selected Q parent (seed2026), blocks0/8, five equal-sized compact models:
legacy direct reconstruction, joint whole-unit selection, scalar-energy
selection, role-fixed reassembly, free-role reassembly. Each has232266 actual
real trainables versus parent271594 (about14.5% full-model savings). Selection
uses actual projected-producer BN/ReLU features on500fit images; separate500
check images and1000validation. All five receive exactly3repair epochs, with
AMP attempted/successful updates reported. No official test inference.

Sixteen tests, a ten-local/five-model synthetic smoke including completed-case
resume, and discarded-copy FP32/AMP GPU checks passed before registration.
GPU AMP preflight:2successful updates of3attempts. Initial development tests
exposed a replacement-BN train/eval-state bug; it was fixed before registration
and identity/reload tests now pass. Development evidence is retained.

Primary comparison: free-role versus whole-unit selection. Continue only if
free-role validation accuracy beats the best legacy/whole/energy control by at
least0.5pp, with CE no worse than the lowest control CE. This is a practical
single-seed gate, not significance or novelty evidence. No automatic next study.

The runner reuses hash-checked completed artifacts when invoked again; it does
not queue further experiments. Its STOP/completed-epoch resume path remains
available for interrupted runs.

## Latest completed experiment

Component reassembly direction, 2026-11-25: the user proposes selecting individual
scalar components and assigning new quaternion roles, followed by fitting truly
smaller Hamilton producer/consumer layers. A bounded CPU technical diagnostic is
complete: [report](../results/component_reassembly_diagnostic/REPORT.md),
[protocol](../results/component_reassembly_diagnostic/protocol.json),
[config](../configs/component_reassembly_diagnostic.json). Seven tests passed;
3360 ordered candidate quartets were enumerated over two deliberately constructed
synthetic teachers, with 128 fit and 1024 separate check inputs. All eight selected
artifacts reload exactly. Each student contains only a smaller Q producer,
BN/ReLU and smaller Q consumer: 16 real parameters versus 32 for the toy teacher.
There is no full teacher, gather/router or dense adapter in the student forward.

Derived compatibility cost: pull the four selected producer rows back through
their assigned Hamilton roles. The common coefficient vector minimizing
Frobenius error is their mean; the residual is their summed squared deviation,
equivalently one quarter of all pairwise squared distances. This is elementary
projection, not a new theorem. Original intact quartets have zero cost; arbitrary
reassembly generally requires approximation. The final small architecture family
is unchanged; this direction concerns source-guided compression initialization.

Synthetic held-out MSE for best-whole versus reassembled: 0.425194/0.001013 in
the aligned-producer example and 0.521865/0.286526 in the unrelated-producer
example. Ideal full-teacher selected-feature error is near zero in both; actual
compiled producer error differs substantially. Scalar-energy, consumer-only and
realizability-aware selectors choose the SAME winning quartet in both cases;
no selector superiority has been established. These are mechanism checks on
constructed data, not CIFAR accuracy or evidence of novelty. The real-data pilot
above now tests this mechanism with whole-unit/scalar-energy controls and
matched actual parameter budgets. Its measured results are summarized above.

The older PHM basis-adaptation pilot below remains closed with its failed gate.

The learned-basis compression pilot finished on 2026-11-25 at 18:06:23
Europe/Warsaw: six local fits and six models completed, no stderr output,
worker exited, GPU idle. All 12 final local/full model hashes passed a
[post-completion artifact check](../results/phm_basis_compression_pilot/completion_audit.json).
Model metric reload replays are recorded in individual result files.

| Compact model (233418 stored reals) | Validation accuracy | Validation CE |
|---|---:|---:|
| Preserve bases, 3-epoch repair | 90.2% | 0.303050 |
| Adapt bases, 3-epoch repair | 90.5% | 0.303820 |
| Reinitialize bases, 3-epoch repair | 90.0% | 0.304644 |
| Scratch, 200 epochs, CE-selected epoch 100 | 90.4% | 0.294454 |

Scratch final epoch: 90.6%, CE 0.416760; the CE-selected checkpoint remains
epoch 100. Scratch uses a full training budget, not the repair update budget.
Adaptation reduced local held-out MSE by 4.78%/1.81% (mean 3.30%), but the
registered continuation gate FAILED: accuracy gain was 0.3pp rather than 0.5pp,
CE was slightly worse and the compression-specific CE criterion did not pass.
Uncompressed preserve/adapt controls both reached 91.6% after repair. One
parent seed/two sites/one split; no equivalence, significance, novelty or gain
claim follows. No further study is queued. [Full report](../results/phm_basis_compression_pilot/REPORT.md).

The learned-basis compression pilot was registered and launched on 2026-11-25
at 16:38 Europe/Warsaw. Check its [state](../results/phm_basis_compression_pilot/state.json)
and recent log tail before reporting progress. Logs:
`results/phm_basis_compression_pilot/runner_20261125_163810.log` and `.err`.
Worker PID at launch: 7552 (Windows launcher PID 11824).

Scope: independent 50% internal-group removal in blocks 0 and 8, six local
300-step reconstruction fits (preserve/adapt/reinitialize), three matched
3-epoch compressed repairs, two 3-epoch uncompressed controls, and one 200-epoch
compact scratch reference. The stored compact count is **233418** versus 272746
for the parent (about 14.4% full-model savings, not 50%). Preserve trains 232266
reals; adapt/reinitialize train 232394. Only the two target consumer bases can
adapt during repair; all other PHM bases are frozen. Scratch learns all bases
from random initialization and uses a larger training budget than repair.

Ten CPU checks, a six-model synthetic end-to-end smoke and discarded-copy GPU
checks passed before scientific fitting. GPU AMP preflight recorded 2 successful
updates out of 3 attempts for each policy; skips and earlier development-check
failures remain in [the check record](../results/phm_basis_compression_pilot/development_checks.json).
Registered [protocol](../results/phm_basis_compression_pilot/protocol.json),
[configuration](../configs/phm_basis_compression_pilot.json),
[readiness](../results/phm_basis_compression_pilot/readiness.json).

Create `results/phm_basis_compression_pilot/STOP` to pause, then remove it and
resume with `python -B -u -m scripts.phm_basis_compression_pilot --run`.
Local Adam state is checkpointed every 25 steps; repair and scratch preserve
completed-epoch optimizer/scaler/RNG state. No later study is automatically queued.

## Completed parent baseline

The Q-stem/PHM4 ResNet-20 baseline completed **200/200 epochs** on 2026-11-25 at
16:19:55 Europe/Warsaw. The worker exited and no follow-up was launched.

- Selected by minimum validation CE: epoch 101, accuracy **91.5%**, CE 0.2691867676.
- Final epoch: accuracy **92.3%**, CE 0.3627398891. It does not replace the CE-selected model.
- Actual real trainables: **272746**, versus 271594 for the earlier Q baseline.
- Selected-model reload/validation replay and checkpoint hash checks passed.
- Seed 2026 only; no official test inference or demonstrated superiority.
- Optimizer: 75000 attempted, 74971 successful updates; 29 AMP-skipped updates are retained in the record.

[Report](../results/resnet20_qstem_phm4_baseline/REPORT.md),
[state](../results/resnet20_qstem_phm4_baseline/state.json),
[metrics](../results/resnet20_qstem_phm4_baseline/qstem_phm4/metrics.json).
The selected model is `results/resnet20_qstem_phm4_baseline/qstem_phm4/model.pt`;
the final resumable checkpoint is `last.pt` in the same directory.

The model was trained from scratch with RGB `(0,R,G,B)`, one Hamilton stem and
18 independently learned PHM residual convolutions at fixed n=rank=4. It does
not learn discrete n, inherit a pretrained Q model, or enforce algebra axioms.
Definitions: [config](../configs/resnet20_qstem_phm4_baseline.json),
[runner](../scripts/resnet20_qstem_phm4_baseline.py),
[model](../src/models/cifar_phm_resnet.py).

## Next action

Review the completed negative pilot above and retain every case, including failures.
Its prespecified continuation gate requires local held-out improvement, at least
0.5 percentage-point validation accuracy gain for adapt over preserve, lower CE,
and a CE gain at least as large as in the uncompressed further-training control.
These are practical pilot thresholds, not significance tests or proof of novelty.
The gate failed, so do not automatically extend this branch to more seeds or
datasets. Reassess the scientific question and prior-art distinction before
registering a new experiment; the scratch reference is now complete.

See the [research plan](PLAN_IJCNN2027.md). Variable-n selection and full-network
structure allocation remain conditional extensions. CIFAR-100 is a later
generalization branch. The separate SSL/few-shot project is outside this agenda.

## Completed evidence

| Study | Measured state and limits | Record |
|---|---|---|
| ResNet-20 baselines and core compression | Real/Q, discovery seed 2026; 12 compression cases at 25%/50% internal-channel removal | [Baseline](../results/resnet20_baselines/REPORT.md), [compression](../results/resnet20_compression/REPORT.md) |
| Focused 50% replication | Seeds 2027/2028; real/Q, pruning/mapfold/direct, equal 3-epoch repair; same data split | [2027](../results/resnet20_key_replication/seed_2027/REPORT.md), [2028](../results/resnet20_key_replication/seed_2028/REPORT.md) |
| Two-block structure catalogue | Q/Q selected by all-Q, Q/real and variable-n policies: 89.3% validation; no demonstrated hybrid benefit | [Report](../results/resnet_structure_catalogue/REPORT.md), [audit](../results/structure_catalogue_audit/REPORT.md) |
| PHM initialization diagnostic | Activation fitting improved local held-out MSE; one random start failed and PHM stores extra basis coefficients | [Report](../results/phm_initialization_pilot/REPORT.md), [interpretation](../results/phm_initialization_pilot/INTERPRETATION.md) |

The catalogue audit reproduced the optimum over 159 feasible pairs and checked
145 frozen source hashes. Its original PHM SVD fitting and coarse width grid
limit interpretation: it does not establish quaternion superiority. The later
local PHM diagnostic does not establish equal-budget classification gains.
Earlier studies, failed checks and negative pilots are indexed in
[RESEARCH_STATUS.md](RESEARCH_STATUS.md); full records remain under `results/`.

## Environment and operational records

The relocated Windows venv was repaired on 2026-11-25 without changing registered
Python/PyTorch/NumPy versions or the scientific recipe. Six CPU tests, all 430
registered source/input hashes, two checkpoint metric replays and 15/15 synthetic
GPU AMP updates passed. Cross-machine bitwise equality is not claimed.
[Migration and resume report](../results/phm_resume_20261125/REPORT.md).

Logs: `results/resnet20_qstem_phm4_baseline/runner_20261125_151729.log` and `.err`.
The epoch-45 checkpoint backup is in `results/phm_resume_20261125/`.
The November 8 timed-pause state is historical; no timed pause is scheduled.
The [JSON agenda](../configs/ijcnn_research_queue.json) and old queue-launch notes
can contain obsolete execution statuses; they do not authorize duplicate runs.

## Working rules and background

Follow [AGENTS.md](../AGENTS.md); before numerical work read [PROJECT.md](../PROJECT.md)
and [LITERATURE.md](../LITERATURE.md). Citation keys are canonical in
[references.bib](../references.bib). Frozen numerical sources/configs, datasets,
checkpoints, split indices and result paths must remain intact.

Keep hypotheses, measurements and interpretations separate; preserve failed
checks and negative results. Distinguish quaternion units, real trainables,
stored coefficients, masked sparsity and physical architecture savings. Seed
replication on the same split is not dataset replication. Do not assert novelty,
significance or acceptance from the current evidence.

The [draft for Marcos](MARCOS_DRAFT_2026-11-07.md) has not been sent; authors are
unconfirmed. The [full literature audit](archive/LITERATURE_2026-11-07.md) retains
prior-art checks. Its relative paths were written for the repository root.
Superseded narrative handoffs/plans are available through Git history.
