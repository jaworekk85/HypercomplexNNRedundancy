# Plan toward IJCNN 2027 (special session: Complex- and Hypercomplex-valued NNs)

## Consolidated authorized research queue (2026-11-07)

LATEST: ResNet20 compression COMPLETE,12/12 cases; no current worker.
Direct vs mapfold after3ep: real-.1/+2.0pp and Q-.3/+1.1pp at25%/50%; one
seed only. Full table and independent audit in results/resnet20_compression/.
Next: implement focused independent-seed replication, then hybrid/PHM. No
automatic replication/PHM yet. Current handoff: docs/HANDOFF.md top.

Core ResNet compression stage is now IMPLEMENTED and RUNNING: see
configs/resnet20_compression.json and results/resnet20_compression/state.json.
ONE seed, both families, fractions.25/.5, mapfold/pruned/direct, up to12 equal
3ep repairs, nine residual-internal sites. Same victims within family/fraction.
Two CPU tests and six-case/nine-site synthetic end-to-end smoke passed; frozen
sources/protocol and resumable per-site/per-epoch checkpoints. No test/automatic
replication/PHM. Earlier "core unimplemented" statements below are historical.

Implementation update: the bounded combined Q/real/PHM catalogue now has a
checked executable runner/config (resnet_structure_catalogue). Six CPU tests,
all5policy synthetic end-to-end repairs and mixed-graph gradient/reload/budget
checks passed. Numerical files frozen/readiness saved; no scientific result.
It implements TWO-block hard type/n/width selection and controls, not weighted
ALS or nine-layer allocation. Main core-compression/key-replication predecessor
workers remain unimplemented; the new --run explicitly waits for their completed
outputs and completed baselines. Current GPU process/launchers unchanged.
Hybrid/PHM queue entries share the same pilot, not two duplicate executions.

Subsequent exact novelty audit of the VARIABLE-n EXTENSION to the Q/real hybrid
is complete in LITERATURE.md (20queries, new canonical ALDS/SPDY/Numerion/SOHN
references). No exact full combined predecessor located, NOT a priority proof.
ALDS occupies automated layerwise group/rank allocation; SPDY general budgeted
profile DP; Numerion adaptive fusion across dimensions1/2/4/8/16; Beyond Real
Weights post-hoc selective PHM. Strong risk of incrementality remains. Narrow
candidate is hard per-layer Q/dense/variable-n PHM AFTER pruning/output repair,
with physical budgets and useful measured quality-size gains. Explicitly compare
Q/real-only hybrid, fixed-n PHM and a simple same-catalogue selector. No new
NAS/DP/SVD theorem or first-adaptive-algebra claim. Queue stage metadata updated;
running baseline worker/priority unchanged, no new numerical job.

User explicitly asked to add ALL discussed work, including the Q/real hybrid.
Canonical agenda: configs/ijcnn_research_queue.json (statuses/dependencies/scope).
Order: current ResNet baseline pair -> core real/Q compression -> key independent
seed replication -> bounded Q/real hybrid -> local PHM diagnostic -> only with
signal, weighted PHM/type allocation. CIFAR100 is a deferred task-generalization
branch after the core replication, not contingent on PHM/hybrid success.

Q/real hybrid is a DISTINCT stage, not dropped into the PHM shorthand: first
two conv2 consumers/one seed/.5, all-Q reference and selected/fixed-assignment
hybrid controls under ONE actual global budget. Dense release at fixed scalar
width costs4x the Q consumer coefficients; that is only a diagnostic. Matched
budget hybrids must also allocate internal widths and physically narrow the
connected producer/BN/consumer, preserving residual outer dimensions. Report
all actual counts/slack. Fit-only local Q/dense output-reconstruction error at
the allocated widths/costs may guide selection; separate held-out check images
and fixed validation evaluation, no architecture tuning there. The old62-91%
shared-map result is not the Q/dense selection proxy. No exhaustive search or
broad novelty claim; larger allocation requires a useful bounded result.

Only the baseline stage currently has a checked executable runner. This agenda
is NOT an automatic worker implementation; pending stages need new frozen
numerical modules/checks. Current process/launchers stay on resnet20_baselines.
Keep updates short to conserve the user's weekly token budget.

## Queue extension requested with Claude discussion (2026-11-07)

Main priority remains the stronger-backbone comparison. Baseline real selected
validation90.9%/final91.8%; Q running and has reached91% at the CE-selected
checkpoint (not a final quality result). Next implement/check internal-channel
pruning, map folding and direct consumer-output fitting for BOTH families at
.25/.5, one seed first, then replicate the key comparisons across three seeds.
Set aside unsuccessful center/tail/redundant-map pilots; keep their negatives.

Add conditional final stage configs/phm_postcompression_diagnostic_plan.json:
TWO consumer layers/ONE seed/ONE fraction, fixed-Q versus learned-PHM candidate
fits to a compact dense output-reconstruction reference. Compatible scalar
channel dimensions only, n2/4/8 initially, rank r recorded separately from n.
Do not change ResNet widths/ongoing protocol to admit n3/6. Correct Kronecker
SVD/rearrangement and actual local output replay first; factor storage includes
learned basis coefficients. Frobenius and activation-weighted errors distinct.
PHM4 of an already-Q kernel is exactly representable and not a useful gain test.
No full hybrid model/ALS/global-budget architecture selection without a signal.

Important correction to the proposed motivation:62.4-90.6% releases shared-map
ties WITHIN the fixed Hamilton consumer family. It is not measured Q-vs-dense
architecture cost. Also PHM is NOT confined to training from scratch:
IbnAhad2026HypercomplexQuantization already uses pretrained dense-to-PHM
conversion, reconstruction and per-layer basis-count allocation under budget.
Variable dimension/learned bases in pruned CNNs may differ, but priority and
sufficient novelty remain unverified. See updated LITERATURE.md.

This is a conditional research queue entry, not an executable worker stage.
Current runner remains ONLY resnet20_baselines; no automatic PHM job added to
the running process. Remaining compression/replication/diagnostic stages need
separate numerical implementations and checks. Respect remaining token budget.

## Next proposed validation: stronger residual backbone (2026-11-06)

Subsequently AUTHORIZED and baseline implementation started: user "to dzialamy".
Registered configs/resnet20_baselines.json, live results/resnet20_baselines/.
Two sequential200ep baselines, one seed2026, actual269722real/271594Q trainables;
four CPU checks and full-size GPU gradient/update/reload/memory preflight passed.
Common SGD momentum.9, lr.1 with5epoch warmup.01->.1, decay1e-4, drops100/150,
AMP, same existing split/augmentation. Best validation CE selection; final also
reported, no test. Atomic epoch resume/STOP and root launchers are ready. No
baseline quality result yet at registration. Older "not started" statements
below describe the earlier proposal and are superseded by this authorization.
Pruning/compensation stage remains pending baseline quality/cost review.

Latest user steering: set aside unsuccessful a/mu/tail variants and conserve
the remaining weekly token budget. Keep their negative findings documented;
do not expand/replicate them. No numerical experiment launched by this plan.

Recommended first target: CIFAR-specific ResNet20 on CIFAR-10, ordinary real
and fixed-Hamilton Q counterparts at approximately matched real parameter
budgets. He2016ResNet Table6 reports real ResNet20 test error8.75% (accuracy
91.25%); this motivates a roughly90% baseline target, not a guarantee for our
split/implementation or quaternion model. ResNet support is not yet implemented
here; the existing CNN6 surgery cannot be applied unchanged to residual sums.

Start with one seed per family to check training/gradients/checkpoints, quality,
memory and measured epoch time. Then compare the main existing approaches:
same-victim pruning without compensation, shared-map folding, and direct
consumer-output refit, in BOTH Q and real networks. Initial removal fractions
.25/.5. Preserve residual connections by narrowing only internal channels
between the two convolutions of each basic block; keep block outputs/skip
dimensions fixed. This is a declared restricted pruning scope, not the previous
CNN6 all-output graph. Report actual compact parameters, own-baseline loss,
before/after identical repair, and unequal baseline qualities/budgets. Fix
selection/splits/schedules before outcomes; do not claim masks are identical
across architectures or that all model-size budgets are exactly matched.

After a working pair and compression implementation, replicate key comparisons
over three seeds. CIFAR-100 is a second-stage extension, after validating the
larger-backbone protocol on existing CIFAR-10 data. It is a harder task, not a
way to reach90% more easily. No CIFAR-100 download/run, new baseline training,
ResNet implementation or long job has started yet. No automatic large sweep.
Larger-architecture validation strengthens generality, not methodological novelty
by itself. Prefer short logs/status reads; avoid repetitive literature searches
and new speculative pilot branches while token budget is constrained.

Formatting checked against official sources on 2026-11-04:

- [Initial Author Instructions](https://ijcnn.org/2027/authors/initial-author-instructions):
  six standard pages INCLUDING figures, tables, references and appendices; up to four extra
  pages (ten total) at USD 100 per additional page. Initial submission: IEEE conference template,
  Letter paper, two columns, single spacing, 10-point Times Roman; anonymized PDF.
- [Official CFP PDF](https://confcats-siteplex.s3.amazonaws.com/ijcnn/files/ijcnn27-cfp_web-19.pdf)
  also states six pages and IEEE two-column proceedings format.
- A separate [Full Paper Authors Instructions](https://ijcnn.org/2027/full-paper-authors-instructions)
  page states seven pages including title/references and A4. It also contains unfinished date/ID
  placeholders and EDAS instructions, whereas initial instructions name CMT. This discrepancy
  is unresolved; do not assert that seven is the unambiguous maximum or infer a special-session
  exception. Use six pages as the working manuscript budget, and confirm the applicable final
  limit/page size with organizers before submission. No contact has been sent.

This is a working plan, not a results document: measured outcomes live in docs/RESEARCH_STATUS.md
and the per-study REPORT.md files.

## Approved scientific focus (2026-11-05)

The user approved BOTH questions, with the first as the paper's main axis:

1. **Compression of an existing QNN family:** characterize preservation of Hamilton structure,
   actual compact size, accuracy cost, repair and failure regimes. Intrinsic quaternion
   superiority and a new method are not prerequisites or claims. The read-only synthesis
   `results/qnn_compression_review/REPORT.md` combines completed all-site CIFAR/EuroSAT curves
   and separately identifies the earlier single-site unrestricted functional oracle. Keep
   unequal scratch/repair budgets, held-out-data history and negative effects explicit.
2. **Role of the weight-sharing pattern:** one established 4x4 circulant-block control with
   exactly matched widths and real trainable parameter counts (Ding2017CirCNN context).
   Three new CIFAR CNN6 baselines, 50 epochs each; f=.25/.5, compensated greedy selection,
   same-victim pruning and BN-magnitude, before/after equal three-epoch repair. Reuse Q/real
   data and add only the six missing Q same-victim pruning repairs. Both full and compressed
   Circulant/Q budgets match exactly (73,898 / 42,034 / 19,098 real trainable parameters).
   Circulant groups are not quaternion units. Compare absolute accuracy/CE, own-baseline
   losses and paired compensation effects; retain weak baselines and all unfavorable seeds.

The control is exploratory after existing Q/real outcomes. It does not isolate a particular
quaternion property or prove a generic structure-only mechanism. No random algebra, new
multiplication rule, FFT acceleration claim, Circulant EuroSAT/scratch sweep or architecture
search. BN/ReLU/dropout, input information, initialization scale/tensors, seeds and training
budgets match. Parameter matching does not imply equal starting accuracy or measured runtime.
Pseudo-label repair is a supplementary question; it does not establish quaternion specificity.

Execution preserves the current jobs: curves/refinements -> activations -> pseudo-label repairs
-> structure control. New outer worker `scripts/run_structure_queue.py` and config
`configs/structure_control.json`; state `results/structure_control_queue.json`. Existing root
launchers resume the whole chain. No additional parallel GPU training.

## Candidate two-paper strategy (user suggestion, 2026-11-05)

The user clarified that a doctoral publication must contribute to the student's low-label
SSL dissertation; a pure compression/BN assignment does not meet that objective. The user
then suggested two papers. Latest steering: compression goes to Marcus's session; the
doctoral paper should use FewshotLearning's ordinary SSL study, WITHOUT adding quaternion,
circulant or compression experiments merely to connect the projects. Treat this as a proposed
division to develop, not two finished papers or authorization to declare both committed/send
an email. It supersedes the earlier candidate for a weight-sharing-centered SSL paper.

| | Paper A: existing compression evidence | Paper B: prospective SSL study |
| --- | --- | --- |
| Tentative title | Structure-Preserving Compression of Quaternion Neural Networks: Benefits and Limits | Do Metric Losses Help Low-Label Semi-Supervised Learning? (candidate question, not settled) |
| Main question | When does structure-preserving compensation help, what does it cost, and when does repair erase its advantage? | What do metric losses add beyond a working pseudo-label SSL baseline, in classification and representation quality? |
| Role of structure | QNN is the primary family; circulant and real models provide controls | Ordinary encoders from FewshotLearning; no hypercomplex or compression requirement |
| Core evidence | Existing multi-seed CIFAR/EuroSAT compression, actual compact budgets, repair/scratch limits | Existing CIFAR/Fashion-MNIST low-label SSL runs, metric/contrastive variants, compute/BN controls and representation audits; targeted replication still needed |
| Current status | Substantial completed evidence; focused manuscript and final evaluation still needed | Real empirical base exists, but a sufficiently distinctive and reproducible paper contribution is not yet established |

Paper B is based on the sibling's from-initialization low-label SSL, with its own suitable CE
and pseudo-label controls. The completed compression-continuation pilot HERE is not evidence
for that paper's main question and does not refute the sibling's observed SSL gains. Sibling
documentation records 74 actual runs (58 CIFAR incl. early supervised, 16 Fashion-MNIST).
Three-seed CIFAR CE/hard-PL/soft-PL means: 48.84/57.39/59.27% test. Most metric additions are
single-seed and small: e.g. soft PL seed0 60.12%, mixed triplets 60.57%, contrastive+pseudo
pairs 60.55%. Hard targets with soft weights give 60.06%, so that one seed does not establish
that soft TARGETS themselves drive the gain. No statistical/novelty claim from these deltas.
Longer CE40 49.34% vs CE20 49.25%, a joint-U forward control 48.05%, and normalization
ablations already exist; check exact cost/protocol assumptions rather than propose them as
unperformed experiments. Representation diagnostics can support a focused question but t-SNE
appearance is not a measured contribution. Fashion-MNIST currently has only seed0.

Next analysis should select ONE defensible question from that evidence, review closest prior
work, replicate the few key comparisons across seeds/budgets, and determine whether a properly
reproduced literature SSL reference is needed. Avoid adding an assortment of losses, AL, VAE,
architectures or datasets without a clear purpose. Known pseudo-label/contrastive/triplet
components are not a new method by themselves. Each paper needs its own question, evidence
and interpretation. No new training or sibling code changes were made in this planning update.

Paper A naturally fits Marcus's complex/hypercomplex session. Paper B targets ordinary IJCNN
submission or an appropriate SSL session. Do not force Paper B into Marcus's session by
adding a quaternion experiment. Marcus's committed-paper list concerns Paper A; Paper B's
research/publication planning proceeds separately.

Official registration information checked 2026-11-05 states that a full non-student member/
non-member registrant may present up to two papers, subject to attendance/presentation:
https://ijcnn.org/2027/registration-information. This is a presentation/registration provision,
not evidence of a numerical submission cap or a guarantee about final fees; the page still
contains placeholders. Initial submission instructions require original work not simultaneously
under review elsewhere: https://ijcnn.org/2027/authors/initial-author-instructions.

For Marcus's November 11 list, distinguish a contribution we can responsibly commit to from
a second topic still under investigation. Tentative titles above are proposals, not final
author commitments; author lists remain to be agreed. No external message has been sent.

## Immediate milestone: committed-paper declaration by 11 November 2026

SSL collaboration clarification, 2026-11-05: the newly shared workspace sibling
`../FewshotLearning` contains an actual low-training-label SSL pilot, distinct from the
completed post-compression repair study here. Source/metric inspection is documented in
docs/HANDOFF.md. The user authorized simulations BEFORE deciding the tentative title and
message to Marcus. The bounded genuine-SSL study is completed: configs/ssl_compression_pilot.json,
one seed, two fresh QNN/circulant soft-SSL baselines (50 labels/class, 20x383 steps), then
12 continuations at a single compression level .25. Compare same-victim compensation/pruning
and uncompressed continuation under soft SSL versus CE-U with identical views/joint BN.
Held-out fit/check images reduce U to 47,500; the CNN6 adapter adds no parameters. All learning
respects the small training-label budget from initialization. No test inference or teacher
trained with full labels. Restarted repair schedule .003/1000 steps, inherited soft-confidence
statistics; snapshots preserve pseudo-label quality and recovery trajectories.

Hypothesis: compensation can reduce disruption of the pseudo-label signal after compression.
Measured results: pilot COMPLETED 2026-11-05 18:44 CET, two baselines/all 12 continuations.
Compensated/pruned QNN starts 43.3/28.8%, reaches 50.1/50.2% after 1000 soft-SSL steps;
circulant starts 44.3/32.7%, reaches 51.7/51.5%. CE-U continuation gives QNN 49.3/50.4%,
circulant 51.6/51.0%. Immediate preservation does not become a consistent final advantage.
At 100 steps QNN pruning has already caught up (50.9% vs compensated 49.0%); do not claim
consistent faster recovery. All values are single-seed validation results, not replicated gains.
Interpretation: this pilot supports keeping compression/structural cost/limits primary; there
is no strong observed interaction to justify making SSL central. The result does not evaluate
SSL versus supervised learning from scratch: both parents were SSL-trained, and CE-U is a
continuation control. Keep negative outcomes and separate the old full-supervision repair study.
No title/author commitment, additional sweep or email is sent automatically.
Status: results/ssl_compression_pilot_runner.json; report: results/ssl_compression_pilot/REPORT.md.

On 2026-11-05 the user supplied the invitation from Marcus Valle. It explicitly requests
a tentative title, abstract and author list by 11 November for inclusion in the special-session
proposal. It separately invites participation as an author or potential reviewer and says that
organizers need a list of potential reviewers. No abstract word limit, full manuscript or
attachments are requested in the supplied message. The immediate goal is the contribution
declaration; the user explicitly said full-paper writing can wait.

The [official special-session call](https://ijcnn.org/2027/authors/call-for-special-sessions),
checked 2026-11-05, explicitly asks organizers for a list of committed papers with a tentative
title, author list and short abstract for each. It lists 15 November 2026 as the session-proposal
deadline and 31 January 2027 as the special-session PAPER deadline. This supports interpreting
11 November as Marcus's collection deadline ahead of the session proposal, now confirmed
by his actual invitation. Any additional abstract-length or format restriction is unspecified.

Prepare by the internal deadline: a tentative title, confirmed author list and brief abstract
describing the question, controlled design and measured evidence available then. Clearly label
unfinished experiments; no priority or unsupported performance claim. This is a declaration of
intended submission, not an accepted paper or a completed manuscript. Continue the queued
experiments to inform that declaration. Full-paper writing can wait until the main story is settled.

## Compression-paper readiness: current assessment (2026-11-05)

### Novelty decision checkpoint (2026-11-06; authorized replication)

Latest authorization: centered_maps_pilot, ONE late-consumer CIFAR BN2026 f=.5.
Compare oldfold, A-only output calibration, mu-only, joint A/mu then matched
three-epoch repair; same masks and final6Q/19098 reals. Per-victim Q centers,
not an independent center for every kernel coefficient. Common local output
objective/ridge, nonzero old A initialization. Joint linear A,d solution must
be recoverable as d=A*mu with accurate original-factor folding; otherwise
record failure. No new optimizer/linear-algebra novelty claim. Review/scope in
LITERATURE.md and HANDOFF.md; current state results/centered_maps_pilot/state.json.
No automatic replication, test inference, new baseline or extra hyperparameters.

Centered pilot COMPLETE: validation before/after3ep mapfold33.2/56.7%,
A-only34.2/56.3%,mu-only32.8/56.6%,joint34.5/56.3%. Joint check reconstruction
MSE improves .18802409->.18760790 versus A-only, without an after-repair accuracy
gain. Joint versus control -.4pp after repair, validation CE also slightly worse.
No convincing benefit, n=1; no follow-up sweep queued. CPU independent audit
passes134 frozen sources/299 inputs,16 classification and8 local-MSE replays.
Retain as a bounded negative result; no methodological novelty inferred.

Latest user idea (analysis ONLY, not a queued simulation): train quaternion
reconstruction maps A jointly with compact weights during ordinary labeled
fine-tuning, then fold them away for inference. Effective U=B+sum a*C with
frozen removed-input C preserves Hamilton structure and compact deployed size,
while adding temporary trainable map variables and C storage. Joint A+B is
a redundant parameterization of the same Q family; A-only is restricted.
See STRUCTURE_HYPOTHESIS.md and LITERATURE.md for the plain-SGD induced metric
and exact assumptions. New close prior art Ding2021ResRep/Zhang2026RCP already
learns temporary operators and folds them for ordinary-CNN inference. A Q
adaptation and potential optimization benefit require evidence, not automatic
novelty. Subsequent authorization registers the ONE late-consumer, two-arm
trainable_maps_pilot: two new three-epoch supervised repairs, identical direct-Q
initialization, zero map corrections, frozen C and fixed masks. Deployed size
19098 reals in both; training 19098/20122. Specific-Q review is documented in
LITERATURE.md (36 targeted queries): exact predecessor not located, priority
unconfirmed; Compacter and RepOptimizers also constrain broad novelty claims.
CPU technical checks passed; consult results/trainable_maps_pilot/state.json
for live execution. No automatic replication or additional scientific scope.

This pilot is now COMPLETE: frozen_maps56.7%, learned_maps56.9% validation,
only+.2pp n=1; CE1.2127349243 versus1.2127392883 slightly worse. No convincing
advantage demonstrated. Independent CPU replay and 130-source/296-input hash
audit passed; final six Q convolutions/19098 reals in both. No new experiment
queued. Keep this limited/negative result rather than a new-method gain claim.

Subsequent user authorization: TRY the tail/logit-sensitive proposal. Bounded
pilot COMPLETE: configs/tail_compensation_pilot.json. One late Q consumer only,
CIFAR BN seed2026 f=.5, reused direct-Q checkpoints and same masks/19098 final
parameters. Two new 200-step unlabeled calibrations: tail-logit MSE versus
equal-step local pre-BN MSE control, followed by two matched 3-epoch repairs.
No new baseline, seed sweep or official test. No best-checkpoint selection.
Reference teacher is the already-partly-compressed pre-site4 model, not the
pristine full network. Frozen BN/head during calibration; same old final head
folding treatment after calibration. Detailed scope/status in HANDOFF.md.

Results and independent CPU replay complete: local_extra 34.9->56.8%,
tail_logits 36.8->56.6%, versus reused direct-Q 34.8->56.7%. Tail's check
logit MSE decreases but its local-output MSE increases. It gains +1.9 pp
before supervised repair versus equal-step local calibration and -.2 pp after;
no after-repair benefit established in this n=1 late-consumer pilot. Local
control's fit MSE slightly worsens after its fixed Adam steps; retain this
negative and do not call it a new exact optimum. Two cases closed, no new
replication queued. Report: results/tail_compensation_pilot/REPORT.md.

Do not claim first quaternion distillation: Singh2026 v1 IX-B already studies
pruned QCNN14/QResNet38 with CE+KL KD. Park2020Lookahead is ordinary-CNN
multi-layer-distortion prior art. Exact constrained local procedure may differ,
but its priority remains unconfirmed. This pilot tests an empirical hypothesis,
not a claim that distillation becomes novel when applied to quaternion weights.

LATEST priority check: see LITERATURE.md, "Priority audit of direct-Q
post-pruning reconstruction". No exact predecessor located in the targeted
review, but no unqualified firstness confirmation. `VieiraValle2022ELM` already
gives the hypercomplex LS/real-embedding principle; `Liang2021TwoStep` Eq.8
already fits next-layer filters after feature-based selection; REAP and ThiNet
are additional reconstruction baselines. These constrain the method claim,
not just the list of related work. Candidate wording: formulate/evaluate
Hamilton-constrained reconstruction in physically compact Q CNNs and quantify
extra shared-map restrictions at matched architectures. Do not claim the
solver or the generic feature-vs-output distinction as new. Exact joint Q
study not located; comparing our two procedures alone is not sufficient
evidence of non-obviousness. Singh2026 accepted PDF could not be retrieved;
the checked arXiv v1 must not silently stand in for the final method version.
No new simulation follows from this review.

After the positive pilot, user AUTHORIZED deeper replication (2026-11-06).
`configs/direct_consumer_replication.json`: BN CIFAR/EuroSAT, three existing
seeds, f=.25/.5; secondary existing no-BN CIFAR2026 only, both fractions.
14 jobs, 13 new three-epoch repairs, discovery pilot reused. No new baselines.
Same fixed solver/alpha/masks/fit-check halves and unchanged compact parameter
counts. Discovery flagged/excluded in a separate new-seed table; all signs
and paired CE/accuracy changes retained. No new test inference or tuning.
Smoke passed on both new adapter paths; full protocol frozen before outcomes.
State/log/report in results/direct_consumer_replication; root launchers resume it.

COMPLETE and independently audited: 14/14 jobs, 13 new repairs plus reused
discovery. BN after-repair mean gains at .25/.5: CIFAR +1.27/+3.33 pp,
EuroSAT +.23/+1.37 pp. Two new CIFAR .5 seeds: +4.2/+3.3 pp (mean +3.75),
apart from discovery +2.5. Negative accuracy points: EuroSAT2026 .5 -.2 pp,
no-BN2026 .25 -.1 pp; CE improves in every job. No-BN .5 gains +4.3 pp (n=1).
All 70 held-out consumer MSEs improve; independent hashes/checkpoint counts/
layout/update/solve checks pass. Validation predictions were not independently
replayed for all 14 jobs. No new test evaluation. See REPORT.md and paired plot.

Interpretation for the paper: distinguish the extra ties of shared-map
compensation from the constraints of the compact Q architecture. Same-budget
direct fits improve the observed outcome without releasing Q weight sharing.
Avoid attributing the full old Q/free gap to an unavoidable quaternion cost.
Gains shrink after repair and scratch50 remains stronger at most operating
points with a different training budget. Empirical replication strengthens the
story, but reconstruction LS itself is prior art, not a claimed novel method.
Close this experiment now; the next task is a focused contribution/figure and
nearest-prior-art comparison using these results, rather than an automatic
larger sweep. No additional training is queued.

The accompanying cached-moment diagnostic is complete, without new network
training/inference: results/direct_consumer_pilot/gap_diagnostic/REPORT.md.
It compares the old folded point, reoptimized shared maps and direct-Q weights
under one centered consumer-output ridge objective. This distinguishes remaining
shared-map ties from the Q layer's own allowable weights at a LOCAL objective
level. Discovery-only: releasing those ties accounts for 62.4-90.6% of total
local objective improvement across five sites; both stages also improve the
saved held-out MSE. This does not decompose classification gains or establish
an unavoidable architecture bound. The nested LS fact itself is known. No new
trained map-output-fit control is included in this replication.

Subsequently AUTHORIZED and COMPLETE: the one-source direct-Q pilot.
At CIFAR BN seed2026 f=.5, same victims/19098 real parameters, before/after3ep
validation = 34.80/56.70%, versus old map-fold 18.70/54.20%. All five local
check-calibration consumer MSEs improve. A fixed centered ridge (.001) fits
Q kernels to pre-BN consumer outputs; final real readout stays map-fold-only.
See results/direct_consumer_pilot/REPORT.md and independent_audit.json.
This is an exploratory positive signal for the specified procedure, not a
replicated result, novel LS algorithm or an isolated feasible-space effect:
objective and parameterization change together. No automatic extension.

The requested deeper PHM review is complete in LITERATURE.md; independent
code counterexamples are cached under .cache/literature. They do not remove
that paper's prior-art relevance or validate our priority by exclusion.

User asks whether the current novelty is sufficient for IJCNN and what could
strengthen it. Assessment: a defensible empirical paper candidate, but weak/
borderline methodological novelty and substantial reviewer risk. More completed
controls are not proof of novelty; no acceptance probability is asserted.

Focused follow-up found IbnAhad2026HypercomplexQuantization (WACV2026 official
CVF entry, method inspected in arXiv v1): selective PHM capacity, budgeted
allocation, residual transition and reconstruction/distillation already exist
for dense-to-PHM compression. Do not sell generic hybrid weights, selective
capacity or residual additions as a new principle. See LITERATURE.md.

The bounded diagnostic was subsequently authorized and completed above: one CIFAR source,
f=.5, same frozen victims and compact Q architecture, compare map-then-fold
with direct fitting of the quaternion consumer against its pre-pruning outputs,
using 500 fit/500 check images and the same repair budget if needed. No extra
parameters or fresh baseline training. The question is whether the observed
gap comes from the reconstruction parameterization or from the target weight
family. Both are distinct from post-repair release of weight constraints.

Mathematical framing uses nested feasible update spaces U_map ⊆ U_Q ⊆ U_real;
ordinary least-squares projection identities are background, not a new theorem.
A useful contribution would be a demonstrated separation of these two gaps
and a fixed-budget remedy, supported on held-out data. Direct consumer fitting
alone is prior art (He2017ChannelPruning); a quaternion specialization alone
does not establish originality. No result is predicted or claimed before running.

Only extend beyond the pilot if it provides a distinct explanation or consistent
fixed-size improvement. If it does neither, adding more seeds/datasets/activations
does not solve the novelty problem. A later minimal non-Hamilton residual under
a fixed budget is a separate hypothesis requiring a more specific prior-art
comparison, not a newly queued experiment. All previous runs remain complete.

### Authorized extension, 2026-11-06: structural cost and capacity control

No-BN baseline completed (CIFAR seed2026): selected CE checkpoint 76.60% val
vs BN 78.30%, 73002 vs 73898 real parameters. End-of-training no-BN accuracy
77.00% is a different checkpoint and is not substituted for frozen CE selection.

User authorized whole-network Q/unrestricted compensation. Registered protocol:
`configs/structure_cost_study.json`; state/log/report in `results/structure_cost_study`.
Two fractions .25/.5, BN CIFAR+EuroSAT three seeds; no-BN CIFAR seed2026 only.
Same victims; q_pruned / q_comp / free_q_init / free_full_fit before and after
3 repair epochs. Free-Q-init initially equals Q-comp but has five independently
trainable real consumer convolutions. This control separates a better initial
map from merely releasing parameter constraints during repair. Full maps are
physically folded without projection. Actual parameters/storage are reported;
equal removed width is NOT equal budget, and unrestricted models may exceed
full-QNN size. There are 38 new short repairs and 18 reused prior repairs;
14 groups/112 before-after observations. No new test inference.

Prespecified interpretation: a full-fit advantage explained by free-Q-init
after repair is attributed to optimization capacity, not a better compensation
method. A small mean Q/free gap is not proof of equivalence with three seeds.
No-BN is exploratory n=1. These are tests of an empirical structural-cost
question, not a claimed new algorithm. Fixed study stops after these points;
further work needs a specific reason beyond increasing simulation count.

COMPLETED 2026-11-06: all 14 groups/112 rows; independent structural/provenance
audit passed (not a separate full-data prediction replay). No new training queued.
At f=.5 after 3 epochs, BN CIFAR Q-pruned/Q-comp/free-Q-init/free-full-fit means
= 49.67/53.57/57.90/63.17%; EuroSAT = 77.47/85.33/88.23/90.17% (3 seeds).
Q/free parameters 19098/72666; original QNN 73898. No-BN seed2026 gives
38.70/45.70/51.90/57.30%, Q/free 18650/72218. At f=.25, free BN has 162562
parameters and exceeds the original QNN. Thus higher dense quality is not an
equal-budget compression gain. Full-fit initialization contributes beyond
merely releasing weights: at .5, full-fit minus free-Q-init after repair is
+5.27 pp CIFAR and +1.93 pp EuroSAT, positive in every seed. At .25 it is
+.97/+.57 pp with one CIFAR tie. Same-family compensation initialization and
constraint release are distinct observed effects. All outcomes are validation
and check-calibration only; no new test evaluation. See REPORT.md, decomposition.json,
independent_audit.json and three PDF figures in results/structure_cost_study.
Recorded AMP/scheduler warning means equal attempted schedules, not audited
equal successful updates; retain this optimization-budget limitation.

Interpretation guardrail: Q-versus-free differences quantify the SPECIFIED
compensation procedures, not a mathematical lower bound on the quality of
every compact QNN. A shared reconstruction h_v≈sum h_k a_kv followed by
folding may restrict the consumer update more than directly fitting all
surviving quaternion consumer kernels to the original consumer outputs.
The latter keeps the same target architecture and contains the current
folded solution as a feasible candidate (for the corresponding unregularized
local output-error objective). It is not automatically a new method.
This is a concrete, conditional follow-up hypothesis if the current gap needs
explanation; NOT an added job. Review closest output-reconstruction literature
before claiming originality or implementing another experiment. Finish and
interpret the current registered comparison first.

Historical context: the earlier single-block functional oracle cost was
.43/1.17 pp on CIFAR and .00/.43 pp on EuroSAT for 4/16, 8/16 units removed.
The completed whole-network comparison above is a different experiment, with
physical dense consumers and explicit unequal budgets. No-BN pilot and all-site
extension are now complete; no additional job follows from this assessment.

The user returned to PAPER A for Marcus, explicitly setting aside SSL. Main numerical series
are COMPLETE: both datasets, three seeds, all seven removal fractions, physical compaction,
before/after three-epoch repair, compact scratch references at the six registered fractions
(.25/.375/.5/.625/.75/.875), matched real controls, a separate matched circulant study and
activation sensitivity. Fraction .125 has no registered scratch run; it is not a pending job.
See results/qnn_compression_review/REPORT.md and results/structure_control/REPORT.md.

Measured story: constrained compensation can preserve quality before repair and retain a
paired advantage at moderate all-site compression after repair. At .5 removed/site (74.16%
real-parameter saving), CIFAR comp/BN-pruning = 53.57/48.23%, EuroSAT = 85.33/79.17%, with
positive paired effects in all three seeds on both datasets. At stronger compression the
advantage can disappear/reverse (CIFAR .75: -3.27 pp mean, negative in all three seeds).
Fifty-epoch compact scratch training often wins, at a different student-training budget.
Circulant controls show compensation is not exclusive to Hamilton blocks; do not claim
universal structural equivalence. Single-site unrestricted functional oracles measure a
different contrast from all-site fit-then-project and are not compact QNN baselines.

Current readiness: enough empirical material to begin a focused manuscript and tentative
commitment package, NOT guaranteed acceptance or a fully closed experimental case. Main
scientific risk is incremental contribution relative to known compensation/pruning work.
Frame the contribution as empirical structural cost, operating regimes and limitations,
not a new ridge algorithm, unique quaternion advantage or measured inference speedup.

Follow-up authorized and executed after the literature review (2026-11-05):

- Focused review added to `LITERATURE.md`; GRAIL2026 publication confirmed (CPAL,
  PMLR 328:881–895). It already compares compensation against BN REPAIR/fine-tuning.
  Neither ridge compensation nor adding that comparison is a new contribution.
- Recovery analysis COMPLETE: `results/paper_a_evidence/RECOVERY.md`, `recovery.json`
  and `recovery.pdf/png`; 3216 saved observations from 730 artifacts, no new training.
  At CIFAR f=.25, comp after epochs 1/2/3 = 71.40/71.43/71.50%; same-victim pruning
  = 69.23/70.03/70.00%. At f=.5 comp = 50.20/53.37/53.57%, same-pruned
  = 44.70/48.63/49.67%. This does not establish runtime savings. Missing EuroSAT
  same-victim FT histories are explicitly marked not measured.
- BN-only control COMPLETE: `configs/paper_a_bn.json`, `scripts/paper_a_controls.py`,
  `results/paper_a_bn/REPORT.md`. Both datasets, QNN CNN6, seeds 2026/27/28,
  fractions .25/.5, comp / identical-victim pruning / BN-magnitude pruning plus
  full baselines = 42 initial models, each with BOTH BN estimators. Each uses the
  same 2048 unaugmented training images/seed, no labels, gradients or weight updates.
  Cumulative simultaneous BN takes one pass; sequential global evaluation-mode BN
  takes six. No winner selection. All 84 parameter/buffer invariance audits pass;
  original phase-A metrics reproduced before intervention. The separate 500-image
  compression-fit resource and unequal three-epoch labeled FT budget are disclosed.
  Global BN-only comp/same-pruned mean validation accuracy: CIFAR .25 64.70/52.60%,
  .5 32.67/21.47%; EuroSAT .25 90.23/60.70%, .5 58.70/23.77%. These means do not
  support the hypothesis that the compensation effect is only stale BN statistics.
  They do not prove a mechanism or quaternion exclusivity.
- Final test selection FROZEN before inference: `configs/paper_a_final.json`,
  `results/paper_a_final/protocol.json`. 333 existing checkpoints, no training.
  Primary Q/real both datasets at .25/.5/.75: before/after comp and BN magnitude,
  full baseline, 50-epoch selected scratch; both BN estimators and all existing
  circulant/same-victim controls. Seven-point curves remain validation evidence;
  three test levels were selected from known validation outcomes, not test results.
  EuroSAT is the final held-out tile test, not geographic transfer; CIFAR test was
  historically seen. See `results/paper_a_final/state.json` for live completion.
  Every loaded checkpoint must reproduce its frozen validation result before test.
  COMPLETED: all 333 cases; independent saved-logit/target/metric replay passed for
  all 333 (`results/paper_a_evidence/final_prediction_audit.json`). No jobs remain.
  Final QNN test comp/BN-magnitude/scratch: CIFAR .25 72.30/70.63/72.62%,
  .5 54.72/48.98/64.53%, .75 19.75/23.11/47.30%; EuroSAT .25
  93.85/92.34/95.07%, .5 85.68/79.78/93.04%, .75 51.02/44.28/86.01%.
  At .5, comp-minus-BN is positive in every seed on both datasets. At .75,
  CIFAR mean is negative with mixed individual signs; EuroSAT also has one
  unfavorable seed. Do not carry over validation-only all-positive claims.
  At .5, CIFAR comp-minus-SAME-victim after FT is +3.47 pp for Q and +4.43 pp
  for circulant, all seeds positive. This is not proof of equal robustness.
  Final evidence package and three vector figure files:
  `results/paper_a_evidence/REPORT.md`. Numerical checks are closed; prioritize
  selecting results/writing over more training. Full-paper writing remains a
  separate editorial step; author list and email to Marcus are not finalized.

Earlier prioritized checklist (current completion above supersedes these proposals):

1. **BN-only repair:** a small fixed subset of existing operating points; shared unlabeled
   training images, fixed affine weights and dropout disabled. Compare unchanged compact
   models, BN recalibration and gradient-based repair. Retain a negative result if this
   cheapest repair erases the apparent compensation advantage.
2. **Recovery budget:** first extract existing epoch 1/2/3 histories and same-victim controls.
   Add short-step runs only if the practical value of the initial advantage remains unclear.
   Do not claim optimization/time savings before measuring them.
3. **Final evidence:** COMPLETED as documented above: frozen choices, final EuroSAT test,
   historical CIFAR test and independent prediction replay. No post-test tuning.
4. **Prior work and writing:** sharpen comparison to GRAIL2026, Kim2020, Asiaee2026CMR and
   quaternion pruning; choose readable figures/tables and report paired uncertainty. No broad
   new architecture/dataset/activation sweep is needed merely to increase experiment count.

Marcus's November 11 request is tentative title/authors/short abstract, not the full paper.
Current evidence supports an honest conditional contribution statement while focused checks
proceed. Paper B stays in the sibling SSL project and is outside this paper's main argument.

## Candidate story (conditional findings; final checks remain)

Question: WHEN does structure-preserving compensation of removed quaternion units matter?

- Without fine-tuning: Hamilton-restricted multi-source compensation (q_v ~= sum_k q_k*a_k, folded
  exactly as w_k <- w_k + a_k*w_v) retains far more accuracy than pruning or random removal, costs
  ~1 pp versus the unconstrained (non-quaternion) oracle, and beats fit-then-unweighted-projection.
- After three equal-budget fine-tuning epochs, policy differences nearly vanish in the earlier
  single-block multi-source experiments. The completed ALL-SITE curves retain an advantage
  over BN-magnitude at moderate compression, with failures/reversals at stronger compression.
  Keep these regimes distinct and retain unfavorable seeds.
- Mechanism: exact folding works because Hamilton blocks are CLOSED under multiplication
  (R(w)R(a) = R(a*w), for the implemented right multiplication); this is a property of the algebra, not of "quaternion meaning". The
  four-channel grouping itself is an architectural constraint (cf. Valle2024VNets).

Guardrails: unconstrained multi-source ridge folding is prior art (GRAIL2026; also Kim2020,
Asiaee2026CMR). The direct-Q fit equals data-weighted projection, so no new algorithm is claimed.
Candidate contribution = the measured cost of preserving a known structural constraint and
the operating-regime map. The constraint itself and elementary closure are not new methods.

## Focused article outline within six pages including references

Working title: *When Does Structure-Preserving Compensation Help Compress Quaternion CNNs?*
The central question is when constrained compensation preserves useful quality, what restricting
the consumer costs, and how fine-tuning changes that trade-off. The final thesis must reflect
the completed curves and remaining focused controls; proposed experiments are not findings.

| Part | Approximate pages | Content |
| --- | ---: | --- |
| Title, abstract | 0.50 | Research question, controlled design, final measured conditional outcome |
| Introduction and focused related work | 1.00 | Structured consumers; prior merging/compensation (Kim2020, GRAIL2026, Asiaee2026CMR), QNN compression (Vecchi2020, Singh2026), V-net framing (Valle2024VNets) |
| Structure-preserving compensation | 0.75 | Shared-map reconstruction/folding versus direct Hamilton-constrained output fit, with correct multiplication order and actual compact budgets; cite VieiraValle2022ELM for the hypercomplex LS solver and Valle2024VNets for the structural view |
| Experimental setup | 0.75 | CIFAR RGB encoding and four-band EuroSAT; Q/real baselines, seed/split rules, actual parameter budgets, fixed fit/fine-tuning/scratch protocols |
| Results | 1.75 | Quality-size curves before/after fine-tuning, compact scratch control, matched-size real control, structural cost; means/seed variability and unfavorable conditions |
| Discussion and conclusion | 0.50 | Conditions where compensation helps or its advantage disappears; BN repair and budget controls where completed; scope and calibration sensitivity |
| References | 0.75 | Focused canonical citations from references.bib |

Total target: six IEEE two-column pages. If a seventh page is explicitly allowed under the
applicable rules, use it for a decisive control or clearer uncertainty reporting. Page allocations
are approximate and must be checked in the actual IEEE template.

Marcos citation placement is explicitly recorded in LITERATURE.md, "Valle group".
Core citations: `VieiraValle2022ELM` in related work and at the LS formulation;
`Valle2024VNets` in the introduction/weight-sharing interpretation. Keep
`VieiraNetoValle2025VEfficientNets` for architecture-by-design comparisons and
`VieiraNetoValle2026Ghost` for real-to-hypercomplex transfer if those topics fit.
Four canonical entries retained; no invented publication metadata or unrelated
courtesy citations. Formulas checked against frozen code are now in
docs/STRUCTURE_HYPOTHESIS.md, "The two implemented compensation procedures".
Any tail-sensitive proposal there is unimplemented and NOT a queued experiment.

Suggested compact evidence package:

1. One main multi-panel quality-size figure (two datasets; before and after fine-tuning), using
   actual retained real parameters. Include scratch controls where measured. Use a readable subset
   of curves and place complete policy/seed values in the repository.
2. A recovery-budget figure or compact BN-recalibration table, conditional on those controls being
   completed. This tests whether compensation has value beyond a cheap distribution repair or
   a short optimization head start.
3. A small matched-budget/structural-cost table: admissible Q compensation, unconstrained functional
   substitution and real-valued control. Never give the functional oracle a compact-Q size claim.

The earlier one-block result that fine-tuning nearly erases method differences belongs in the main
argument. The new all-layer result is a separate, harder regime, not a contradiction to be hidden.
Three-seed uncertainty must replace reliance on the first encouraging seed. Source hashes, full
configs, tests, all per-seed outcomes and implementation history can be supplied through the code
artifact; the article should substantiate a single coherent empirical argument.

Before final claims: complete both datasets and queued fractions, test label-free BN repair,
compare with scratch, and freeze the final evaluation choices. EuroSAT's 5,400-image test split
remains untouched; CIFAR's official test has already been seen in earlier pilots and must not be
described as a fresh blind benchmark. The elementary closure/folding argument and least-squares
fit do not establish a novel method or priority.

## Status

| Item | Status |
| --- | --- |
| Pairwise merging; replication on CNN6/CNN8 and 4/8 removed units | done (results/internal_replication) |
| Multi-source compensation + real CNN6 control (CIFAR) | done (results/multisource_compensation) |
| Equal-budget fine-tuning, 3 epochs (CIFAR) | done (results/finetune_compression): policy differences < 1 pp after fine-tuning |
| Second dataset EuroSAT four-band (q = NIR + R i + G j + B k): train, compress, fine-tune | done: replicates CIFAR (no fine-tuning 8/16: multi_q 94.3% vs random 74.9%; after fine-tuning all within ~0.5 pp) |
| Quality-size curve, all layers, f = 0.125-0.875, with scratch control | DONE on both CIFAR and EuroSAT; results/compression_curve and results/compression_curve_eurosat |
| Additional curve points f = 0.375 and 0.625 | DONE on CIFAR and EuroSAT, including repair and scratch; configs/compression_curve_queue.json |
| Exactly matched 4x4 circulant structured-weight control | DONE, three CIFAR seeds; configs/structure_control.json |
| Activation sensitivity | DONE, split LeakyReLU and MagnitudeTanh; results/activation_sensitivity |
| BN-only recalibration and final frozen test evaluation | DONE: 42 initial states, both BN estimators; 333 final test checkpoints and independent prediction replay |
| Focused literature review | DONE as a targeted claim/decision review in LITERATURE.md; not exhaustive and does not establish priority |

## Next experiments, in priority order

1. Fine-tuning budget curve. Steps 0, 50, 200, 1000, 1 epoch, 3 epochs for the same compact
   policies (same batch stream per seed). Output: how many fine-tuning steps compensation saves.
   Per-epoch values from the running study already give epochs 1-3.
2. Label-free BN recalibration. After compression, refit BN running statistics on unlabeled
   training images (no weight updates), then evaluate all policies. Tests whether the advantage
   survives the cheapest label-free repair. Minutes of compute.
3. Quality-size curve: COMPLETE including refinement on both datasets. It identifies stronger
   all-site failure regimes beyond the earlier single-block result. See the current readiness
   assessment above for remaining work; do not restart these completed studies.
4. Calibration-data size: 32 / 128 / 500 fit images (GRAIL uses ~128). No training.
5. Optional, only if time remains: complex-valued extension ([[a,-b],[b,a]] blocks are also
   closed) to show the effect follows closure rather than quaternions specifically.
6. Activation-sensitivity study (authorized and QUEUED on 2026-11-04 after all existing curve
   runs): CIFAR quaternion CNN6, three seeds; train split LeakyReLU (slope 0.1) and coupled
   MagnitudeTanh (Eq.21 of PoppelbaumSchwung2024Activations) for the same 50 epochs as the ReLU
   reference. Six new baselines, unchanged widths/data/budgets and no extra trainable parameters.
   Reuse the frozen compression/fine-tuning implementation at f=0.25 and 0.5; align random masks
   to the original fraction indices and report own-baseline accuracy loss as well as absolute
   accuracy/CE. See configs/activation_sensitivity.json and docs/HANDOFF.md. This first pilot
   has no activation-specific scratch, EuroSAT or matched real coupled/split-Tanh control;
   it tests within-Q sensitivity rather than isolated coupling or quaternion superiority.
   MagnitudeTanh changes saturation/boundedness too. New modules preserve frozen numerical sources.

7. Low-label **post-compression repair** (authorized 2026-11-05, appended AFTER activations).
   CIFAR ReLU quaternion CNN6, f=0.25 at every site (42,034 actual real parameters), three seeds.
   Reuse the original pre-fine-tuning greedy-compensated model and reconstruct its same-victim
   pruning control; no new victim selection. For each start, compare supervised-only repair,
   online current-student hard pseudo-labels and a frozen full-teacher pseudo-label control.
   This is 54 short repairs: 2 starts x 3 modes x 3 label budgets x 3 seeds.

   **Hypothesis:** compensation may improve recovery with few repair labels; current-student
   pseudo-labels may help or instead reinforce compression errors. No improvement/novelty assumed.
   **Measured results:** none from scientific repair runs yet; readiness checks are not results.
   **Interpretation limits:** primary budgets are 5/20 labels per class (50/200 total); zero is
   a boundary control, meaning no human labels exposed to repair updates. Original model training
   used full supervision and labeled validation selection; evaluation labels and inherited BN/
   normalization statistics are outside this repair-only budget. Do not call this an end-to-end
   few-shot/unsupervised result. Realistic motivation: adapting a smaller version of an existing
   classifier when repair images are available but their annotations are inaccessible/expensive.
   The current same-distribution CIFAR study simulates annotation withholding; it does not test
   new geographic/seasonal conditions or actual deployment.

   Nested balanced labels, shared disjoint 4096-image U pool, paired streams per seed. Fixed
   1000 update opportunities with snapshots 0/200/1000; confidence .95, crop/flip, AdamW 1e-4
   cosine to 1e-6. BN statistics fixed and dropout disabled; affine BN parameters remain trainable.
   No accepted target means no optimizer step, including no weight decay. Record accepted-label
   precision/coverage on held-out check data for diagnosis only, actual updates and all seeds.
   Frozen full teacher is an extra-resource control. No domain adaptation, new method, exact
   student-method reproduction or FixMatch reproduction is claimed (Sohn2020FixMatch).
   Existing full-label three-epoch repair uses different BN/dropout settings and is not a matched
   label-efficiency reference. Config: configs/pseudolabel_repair.json; queue state:
   results/pseudolabel_repair_queue.json; report: results/pseudolabel_repair/REPORT.md when running.

## Specification: quality-size curve (item 3)

Code: `src/compression/multilayer.py` (library, 7 tests in `tests_torch/test_multilayer.py`),
`scripts/compression_curve.py`, configs `configs/compression_curve.json` (CIFAR, run first),
`configs/compression_curve_eurosat.json` (EuroSAT, same rules), `configs/compression_curve_smoke.json`
(CPU pipeline check, ~2 min, output `.cache/smoke`, not a result).

Design (all fixed in the config before outcomes):
- The same fraction f of units is removed at EVERY convolution output (6 sites in CNN6),
  sequentially shallow to deep; f in {0.125, 0.25, 0.5, 0.75, 0.875}. Each compact model is exactly
  the standard architecture with narrower widths (verified by reloading). Parameters scale ~(1-f)^2:
  CNN6 quaternion 73.9k -> ~19.1k at f=0.5 -> ~1.4k at f=0.875; the real CNN6 matches within ~3%.
- Sites are fitted on the even calibration half of the already partly compressed model. Compensation
  folds a ridge reconstruction from the kept units of the same site into the consumer (right-Hamilton
  maps for quaternion, scalar for real; at the last site the consumer is the linear readout, where any
  4x4 map folds exactly). Every exact fold is checked per site against functional substitution.
- Policies: random; bn_pruned (BN-folded magnitude, standard reference); random_comp (same random
  victims + compensation); greedy_comp (greedy backward elimination by reconstruction residual times
  outgoing energy + compensation; main); greedy_pruned (same victims, no compensation; isolates
  compensation from selection); greedy_full_projected (quaternion, no fine-tuning only: cost of fitting
  without the structure); scratch (same compact architecture trained from scratch, baseline trainer and
  full 50-epoch budget, as the "is compression worth it at all" control reviewers will ask for).
- Phases: A no fine-tuning (minutes); B 3-epoch fine-tuning of baseline, random (3 sets), bn_pruned,
  random_comp (3 sets), greedy_comp; C scratch training for f in {0.25, 0.5, 0.75, 0.875}.
  Outputs: `results/compression_curve/{REPORT.md, summary.json, quality_size_curve.png}`; REPORT and
  figure are rewritten after each phase, so partial results are readable.
- Questions: where is the knee for each family/policy; does compensation keep an advantage after
  fine-tuning at strong compression; is compression + 3 epochs better than scratch training; does
  the quaternion family degrade more gracefully at equal removed fraction.

Disclosed deviation: the 0.125 level was added after a pre-fine-tuning diagnostic on seed 2026
(validation, CNN6): errors compound over six sites (greedy_comp leaves ~59% at f=0.25 and ~19% at
f=0.5 without fine-tuning; real CNN6 ~63% and ~34%). Single-site sanity check reproduced the earlier
study (block-2 output, 8/16: 74.6% compensated vs 50.6% pruned). Nothing else was changed.

Estimated GPU time (RTX 3050 Ti), CIFAR, 3 seeds: phase A ~15 min, phase B ~2.5 h (~200 runs),
phase C ~4-5 h (24 trainings). EuroSAT is roughly similar. Set "scratch.enabled": false to skip
phase C, or run it later (all phases resume).

Possible follow-ups once the curve exists: BN recalibration on unlabeled images (item 2) applied to
the same compact models, which may repair much of the compounding without labels; longer fine-tuning
at f >= 0.75.

Registered curve refinement (user request, 2026-11-04): add f = 0.375 and 0.625 after the ongoing
CIFAR AND EuroSAT curves. These points were chosen after inspecting coarse-grid outcomes, and must
be identified as follow-up measurements. Freeze them before their own outcomes; retain every parent
policy, architecture, seed, split, ridge parameter and training budget. Include scratch training at
both added fractions. Separate configs and outputs preserve the original protocol. Execution order
and the live queue are recorded in docs/HANDOFF.md and results/compression_curve_queue.json.

Rules for every new study: prespecify policies, levels and hyperparameters in configs/ before
outcomes; no official-test inference until a final, frozen evaluation; paired seed effects with
all unfavorable seeds retained; physical compact models with actual parameter counts.

## Order of work

1. All existing numerical queues are completed. For Paper A prioritize BN-only repair, recovery
   analysis from saved histories and the focused literature comparison. Proposed extra controls
   are not automatically queued by this status assessment.
2. Use the completed QNN synthesis/structural controls as the primary story. Prepare the November
   11 tentative commitment fields when author participation is agreed. Full-paper writing was
   deferred by the user; later freeze final comparisons/test evaluation and assemble the manuscript.
3. Submit with a margin before the deadline.

## Open items

- Helber2019EuroSAT volume/pages/DOI verified against the IEEE publication record, 2026-11-04 (LITERATURE.md).
- Decide whether Valle (IJCNN 2025 special-session co-organizer) work needs more than citation.
- Consider deleting .cache/eurosat/extracted (~2.5 GB; reproducible from the verified zip).
