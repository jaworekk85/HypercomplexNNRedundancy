# Current research state

Updated: 2026-11-25. Read the relevant `state.json`, recent log tail and process
state before reporting activity. This page is a handoff, not a process monitor.

The user requested a concise literature review during the grouping study.
The [2026-11-25 adaptation/grouping review](literature/GROUPING_AND_TRANSFER_2026-11-25.md)
is complete:10new canonical entries, one verified journal-status upgrade for
Valle, three existing compression citations rechecked,14raw BibTeX exports.
[Verification](literature/exports/2026-11-25/verification.json) discloses the
manually checked REPAIR entry and failed export requests. The official IJCNN2027
session includes generalization of real-valued methods to hypercomplex networks;
this supports topic fit, not guaranteed acceptance. Numerical protocols unchanged.

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
