# Research status: completed three-seed CNN and physical-compression pilot

## ResNet20 core compression (2026-11-07)

Status: completed; 12/12 cases, one seed2026. Both families, internal channels of nine residual blocks, same victims per family/fraction,25%/50%, three matched3ep repairs. No new test. Discovery only.

| Family | Removed | Policy | Real parameters | Validation before (%) | After3ep (%) |
|---|---:|---|---:|---:|---:|
| real | 0.25 | mapfold | 202738 | 86.20 | 89.20 |
| real | 0.25 | pruned | 202738 | 72.30 | 88.00 |
| real | 0.25 | direct | 202738 | 87.60 | 89.10 |
| real | 0.50 | mapfold | 135754 | 51.40 | 82.90 |
| real | 0.50 | pruned | 135754 | 25.10 | 83.00 |
| real | 0.50 | direct | 135754 | 70.70 | 84.90 |
| quaternion | 0.25 | mapfold | 204442 | 76.50 | 88.60 |
| quaternion | 0.25 | pruned | 204442 | 46.50 | 87.40 |
| quaternion | 0.25 | direct | 204442 | 84.10 | 88.30 |
| quaternion | 0.50 | mapfold | 137290 | 27.70 | 81.80 |
| quaternion | 0.50 | pruned | 137290 | 17.10 | 81.00 |
| quaternion | 0.50 | direct | 137290 | 53.30 | 82.90 |

Direct versus mapfold after repair: real -.1/+2.0pp and Q -.3/+1.1pp at25/50%. One seed: no equivalence, significance, generalization or Q-superiority claim. Independent audit passes146 source/308 input hashes,24 physical layouts,108 fold records and36 local fits; no independent full prediction replay. Runner itself replayed final validation. All cases1125/1125 successful/attempted. No current job; replication worker unimplemented, PHM waits. Detailed report: results/resnet20_compression/REPORT.md.

## Current stronger-backbone stage: ResNet20 baselines (2026-11-06)

Status at synthesis: completed; see results/resnet20_baselines/state.json for live progress. Sequential real/Q CIFAR10 ResNet20, one seed2026,200epochs each; actual real trainables269722/271594. Four CPU checks plus full-size GPU discarded-copy gradient/update/reload/memory preflight passed; technical checks are not scientific quality results.

Common SGD momentum.9, decay1e-4, lr.1 with5epoch warmup and drops100/150, AMP, crop4/flip; existing48000/1000/1000 split and RGB normalization. Option-A shortcuts preserve component grouping. Best by validation CE, final also reported, no test inference. Approximately90percent is a target, not guaranteed. Checkpoint each completed epoch includes optimizer/scaler/all RNG/best model; STOP pauses after batch, partial epoch repeats. Root launchers target this stage. No ResNet pruning/compensation, CIFAR100 or new-seed sweep automatically started; quality gate first. Older no-running-job/launcher statements below refer to earlier completed studies.

Completed real: selected validation 90.90% at epoch100; final 91.80%.
Completed quaternion: selected validation 91.00% at epoch101; final 91.10%.
## Current centered-Q-map calibration pilot (2026-11-06)

Status: completed; 4/4 fixed cases. ONE late-consumer CIFAR BN2026 f=.5, reuse earlier direct sites0-3. New hybrid mapfold control, output-fitted A, mu-only, joint A/mu; same masks, output reconstruction objective and effective-kernel ridge. A0 is nonzero original feature-ridge maps; one Q mu per removed input shared across outputs/offsets. Joint fit solves linear deltaA,d then recovers mu with verified A*mu=d, preserving the original bilinear formula. No new solver/priority claim.

| Case | Fit MSE | Check MSE | Validation before (%) | Validation after3ep (%) |
|---|---:|---:|---:|---:|
| mapfold | 0.20097541 | 0.20386748 | 33.20 | 56.70 |
| a_only | 0.18470907 | 0.18802409 | 34.20 | 56.30 |
| mu_only | 0.19942219 | 0.20252192 | 32.80 | 56.60 |
| a_mu | 0.18428406 | 0.18760790 | 34.50 | 56.30 |

Joint versus current control after repair: -0.40pp; versus A-only: +0.00pp. No after-repair benefit demonstrated, n=1. Validation CE control/joint: 1.2202836456/1.2227811584. Joint reconstruction improves slightly beyond A-only but does not improve recovered accuracy. No replication queued.

Independent CPU audit passed: 134 frozen sources/299 inputs, 16 full classification split replays and 8 local-error replays. Exact A*mu recovery, unchanged C, actual Q layouts/counts, same masks/head folds and all checkpoints/budgets checked.

Calibration adjustable reals A1024/mu64/joint1088 versus18432 full local Q coefficients. All final models have six Q convs/19098 reals; factor variables/constants discarded before identical3ep supervised repairs. Fit500/check500 separate; no test inference, baseline training, hyperparameter search or automatic replication. Root launchers now target this pilot; earlier shortcut descriptions below are historical. Detailed method/results: results/centered_maps_pilot/REPORT.md.

## Current trainable-Q-map pilot (2026-11-06)

Status: completed; 2/2 cases completed. ONE late consumer, CIFAR BN2026 f=.5. Two new three-epoch CE repairs from identical direct-Q compact weights and fixed masks. Training reals 19098 versus 20122; both deploy as six Q convolutions /19098 reals. Corrections start at zero; removed-input C is frozen. No test or automatic replication.

Technical CPU checks passed: literal quaternion multiplication, exact folding, tracking/gradients/updates, interrupted-epoch replay including BN/RNG/optimizer, and exact agreement with the frozen supervised repair recipe. Actual-model discarded-copy update audits also passed. These checks are not scientific gains. Epoch checkpoints preserve resume state; a partial epoch is repeated. Successful and attempted AMP steps are distinguished.

| Case | Validation before (%) | Validation epoch3 (%) |
|---|---:|---:|
| frozen_maps | 34.80 | 56.70 |
| learned_maps | 34.80 | 56.90 |

Observed validation difference +0.20 pp, n=1. Validation CE control/learned: 1.2127349243/1.2127392883. No convincing advantage demonstrated. Independent CPU audit passes 130 source/296 input hashes, six full split replays, actual compact layouts/counts, fixed C and learned-map component checks; see independent_audit.json. Both cases record 1125 attempted/1124 successful AMP steps. No new training queued.

Single previously studied discovery source; no replication, significance or priority claim. Q-specific literature review located no exact predecessor but does not prove firstness. ResRep/RCP already train and fold ordinary-CNN compensators; Compacter learns hypercomplex adapters; RepOptimizers establishes related optimization context. Bibliography and 36-query audit in LITERATURE.md. Current root launchers target this pilot; earlier launcher descriptions below are historical. Report: results/trainable_maps_pilot/REPORT.md.

## Current late-Q consumer logit-sensitive pilot (2026-11-06)

Status: completed; 2/2 fixed cases. One CIFAR BN seed2026 f=.5 late consumer, initialized at existing site4 direct-Q fit. Two 200-step FP32 Adam calibrations on 500 unlabeled images, same order, separate 500 check; local pre-BN MSE versus logit MSE through frozen tail. ONLY blocks.2.3.kernel changes during calibration; all four Q components audited. Teacher is current pre-site4 reference, already partly compressed, not the full pristine source. Final same-victim head map-fold and matched three-epoch supervised repair follow. Same six Q convs /19098 real parameters.

| Case | Validation before supervised repair (%) | Validation after 3 epochs (%) |
| --- | ---: | ---: |
| direct-Q, reused without extra steps | 34.80 | 56.70 |
| local_extra | 34.90 | 56.80 |
| tail_logits | 36.80 | 56.60 |

Tail-logit refinement improves pre-repair accuracy here, but no after-repair benefit is demonstrated. One exploratory seed/consumer: no superiority/equivalence claim from small differences. Local-extra is an optimizer-step control, not a converged optimum; its fit MSE slightly worsens after fixed steps. Tail reduces check logit MSE while increasing local-output MSE. All unfavorable results retained. Equal steps do not establish equal FLOPs/runtime; cached stems and measured calibration times recorded. No test evaluation or new baseline. Report: results/tail_compensation_pilot/REPORT.md. Independent CPU audit replays all before/after validation/check classification metrics and full-forward calibration errors, plus parameter/mask/buffer invariants. Frozen sources/inputs unchanged.

Prior art: Park2020Lookahead concerns ordinary CNNs; Singh2026 v1 IX-B ALREADY applies KD to pruned quaternion CNNs using CE+KL. Our label-free single-consumer MSE fit is a distinct limited protocol, not a reproduced published recipe or proof of novelty. Both root launchers resume this completed bounded pilot; no extra experiments queued. Earlier launcher descriptions below are historical. Marcos citation-placement table is in LITERATURE.md (four canonical entries); exact two-method equations are in docs/STRUCTURE_HYPOTHESIS.md.

## Current direct-Q replication (2026-11-06)

Status: completed; 14/14 jobs, 13 new three-epoch repairs, 1 reused discovery pilot. No new baseline training. BN CIFAR/EuroSAT three existing seeds, .25/.5 unit removal at all sites; secondary CIFAR no-BN seed2026 only. Same masks, compact Q architecture and actual real parameter counts as old map-fold. Unchanged five Q-consumer fits; final already-real head remains map-fold-only.

| Dataset | Norm | Removed/site | n | Direct-Q after 3ep (%) | Old map-fold after 3ep (%) | Paired mean gain (pp) | Individual gains (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cifar | bn | 0.25 | 3 | 72.77 | 71.50 | +1.27 | 2027: +0.90, 2028: +1.80, 2026: +1.10 |
| cifar | bn | 0.5 | 3 | 56.90 | 53.57 | +3.33 | 2027: +4.20, 2028: +3.30, 2026: +2.50 discovery |
| cifar | none | 0.25 | 1 | 70.00 | 70.10 | -0.10 | 2026: -0.10 |
| cifar | none | 0.5 | 1 | 50.00 | 45.70 | +4.30 | 2026: +4.30 |
| eurosat | bn | 0.25 | 3 | 93.97 | 93.73 | +0.23 | 2027: +0.10, 2028: +0.30, 2026: +0.30 |
| eurosat | bn | 0.5 | 3 | 86.70 | 85.33 | +1.37 | 2027: +2.60, 2028: +1.70, 2026: -0.20 |

CIFAR .5 excluding discovery: 2 new seeds, paired mean gain +3.75 pp. Three-seed means include the discovery seed and are not three independent confirmations.

Independent audit: passed; all frozen sources/inputs, checkpoint parameter counts/layout/hashes, matched masks, first-update evidence and 70 Torch-vs-NumPy local solves/check-MSE replays. Held-out local MSE improved 70/70 times. This is not independent full-dataset validation prediction replay.

All new metrics use validation/check only; no new test inference. EuroSAT is exploratory after prior data/test inspection, no-BN n=1. Retain negative accuracy changes even when CE improves. BN compact counts are 42034/19098 real parameters at .25/.5, versus full 73898: unit removal fractions are not parameter removal fractions. Dense references have larger budgets. Scratch uses 50 epochs versus 3 repair epochs plus a pretrained full model; no scratch dominance or measured latency saving is claimed.

Separate discovery-only cached-moment diagnostic: results/direct_consumer_pilot/gap_diagnostic/REPORT.md. Under the SAME local centered output-ridge objective, optimizing shared maps recovers less than allowing all compact Q coefficients to move: releasing map ties accounts for 62.4-90.6% of total local objective improvement at five sites. This is not a classification-accuracy decomposition, independent replication or architecture lower bound. It uses saved moments and no additional network training/inference. Numerical sources/config frozen separately.

Current report/paired plot/audit: results/direct_consumer_replication/. Both root launchers target this registered study; a completed run does not start more training. No architecture/alpha search, extra fraction or additional experiment is automatically queued. Methodological novelty remains to be argued against reconstruction/folding prior art (He2017ChannelPruning, GRAIL2026, Kim2020, Asiaee2026CMR); these results support an empirical distinction between shared-map compensation cost and Q-architecture cost, not a new LS principle.

Priority review completed 2026-11-06: no exact predecessor located by the targeted review, but worldwide firstness NOT confirmed. Newly documented VieiraValle2022ELM already gives hypercomplex LS via real embeddings; Liang2021TwoStep Eq.8 refits the next layer after feature-based selection; KammaWada2021REAP and Luo2017ThiNet are further reconstruction precedents. Candidate contribution concerns the post-pruning Q procedure and the measured extra shared-map restrictions, not inventing the solver. First comparison to our own restricted baseline is not by itself a novelty argument. Version gap: Singh2026 accepted manuscript is blocked by Cloudflare/403; inspected preprint is not assumed identical. Full claim matrix, primary links and limitations: LITERATURE.md, Priority audit of direct-Q post-pruning reconstruction. No new experiment or external contact.

## Latest direct-Q consumer pilot (2026-11-06)

Status: completed. CIFAR BN CNN6 seed2026, f=.5, SAME 19098 real parameters/masks. Before repair validation 34.80% versus map-fold 18.70%.

After three repair epochs: direct 56.70% vs map-fold 54.20%. All five Q consumer check-MSEs decrease; final already-real readout remains map-fold-only. Fixed centered ridge on 500 fit images, separate 500 check images, no labels in regression, no test inference. One seed/point, no automatic replication or new-method claim.
New repair successful/attempted steps: 1124/1125; reused references were not instrumented. Independent audit includes moment/solver checks and CPU prediction replay of before/after validation/check checkpoints. Objective and compensation parameterization change together; no pure feasible-space or irreducible architecture-cost attribution.

Report: results/direct_consumer_pilot/REPORT.md. Deeper Beyond Real Weights paper/sample-code review: LITERATURE.md; local independent code counterexample .cache/literature/phm_sample_review.json. Prior art is not negated by a sample-code issue. This historical pilot was followed by the separately registered replication described above.

## Current structural-cost extension (2026-11-06)

No-BN CIFAR seed2026 baseline COMPLETE: selected validation 76.60% vs original BN 78.30%, after the same 50-epoch recipe; 73002 vs 73898 real parameters. One seed, no new test inference.

User-authorized all-site Q/free study: completed, 14/14 groups, 38/38 new three-epoch repairs, 18/18 reused repairs. CIFAR/EuroSAT BN three seeds, no-BN CIFAR seed2026 only; fractions .25/.5, same victims. Cases: q_pruned, q_comp, free_q_init (same initial function, five conv weight constraints released), free_full_fit (full maps physically folded into dense consumers, no projection).

Equal widths are NOT equal parameter budgets. Dense variants share a budget with each other and can exceed the FULL QNN parameter count. Their initialization/capacity controls are needed before attributing a post-repair advantage to compensation. Observed gaps concern these procedures, not an irreducible lower bound on all quaternion models. Sources/config/inputs frozen before outcomes; no new-method or priority claim. New study selected after earlier validation/test outcomes; evaluation here uses validation/check-calibration only.

Protocol/results: results/structure_cost_study/{protocol.json,REPORT.md,summary.json,state.json}; resume with the existing URUCHOM_KRZYWA.cmd or python -m scripts.structure_cost_study. Independent review/figures: python -m scripts.review_structure_cost_study after completion. Fixed study ends at these two fractions; no further sweep is automatically queued.

## Current Paper A controls (2026-11-05)

User authorized focused literature review, BN-only repair, saved recovery analysis and frozen final evaluation; SSL is aside. LITERATURE.md records the decision: known compensation/QNN pruning, empirical structural-cost study only. GRAIL2026 is published at CPAL/PMLR and already compares BN REPAIR and fine-tuning.

BN-only status: completed, 42 original QNN states, two estimators, 2048 unlabeled training images per seed. No gradients or parameter updates. All invariance checks passed in completed cases. Recovery: results/paper_a_evidence/RECOVERY.md (3216 saved observations from 730 artifacts; no new training).

| Dataset | Removed/site | Comp + global BN | Same victims + global BN |
| --- | ---: | ---: | ---: |
| cifar | 0.25 | 64.70% | 52.60% |
| cifar | 0.5 | 32.67% | 21.47% |
| eurosat | 0.25 | 90.23% | 60.70% |
| eurosat | 0.5 | 58.70% | 23.77% |

Final test status: completed, 333/333 frozen checkpoints. EuroSAT final held-out tile test IS NOW EVALUATED; historical sections below describe earlier experiments only. CIFAR test was historically seen. No post-test tuning. Results: results/paper_a_final/REPORT.md, predictions and protocol hashes. All seven original fractions remain validation evidence; final test uses the declared .25/.5/.75 subset and all BN/circulant controls.

Measured final test: cifar, f=0.25, comp minus BN magnitude after 3 epochs = +1.67 pp; individual seeds are retained in the final summary.
Measured final test: cifar, f=0.5, comp minus BN magnitude after 3 epochs = +5.74 pp; individual seeds are retained in the final summary.
Measured final test: cifar, f=0.75, comp minus BN magnitude after 3 epochs = -3.36 pp; individual seeds are retained in the final summary.
Measured final test: eurosat, f=0.25, comp minus BN magnitude after 3 epochs = +1.51 pp; individual seeds are retained in the final summary.
Measured final test: eurosat, f=0.5, comp minus BN magnitude after 3 epochs = +5.90 pp; individual seeds are retained in the final summary.
Measured final test: eurosat, f=0.75, comp minus BN magnitude after 3 epochs = +6.74 pp; individual seeds are retained in the final summary.

Interpretation: BN-only mean results do not erase compensation's advantage at these operating points. This does not establish causality, uniqueness to quaternions, a new method, latency savings or acceptance. Scratch uses a different training budget. Evidence and figures: results/paper_a_evidence/; current plan: docs/PLAN_IJCNN2027.md.

Source of scope: PROJECT.md. All cited keys refer to references.bib. Reports: results/cnn_multiseed/REPORT.md, results/cnn_compression/block_analysis/REPORT.md, results/cnn_compression/REPORT.md.

## Hypotheses

H1-H3 concern whole-quaternion representation redundancy, measurement choice and useful structural compression. No novel method, first-in-literature result or general quaternion performance advantage is established. This pilot does not directly compare grouped and scalar-derived pruning policies (H2).

## Measured results: baseline quality

| Seed | Quaternion test | Real test | Difference (pp) |
| ---: | ---: | ---: | ---: |
| 2026 | 79.00% | 76.24% | +2.76 |
| 2027 | 78.77% | 75.83% | +2.94 |
| 2028 | 79.20% | 75.76% | +3.44 |

quaternion: mean 78.99%, sample SD 0.22 pp across three training seeds.

real: mean 75.94%, sample SD 0.26 pp across three training seeds.

73,898 versus 72,954 real trainable parameters. Widths and compute differ. Same train/validation/calibration split and normalization; frozen 50-epoch protocol, changed seeds only. Epoch choice uses validation CE. All four new trainings finished before the new final-test evaluations. The official test set was used by earlier pilots and is not a new blind benchmark.

## Measured results: activation diagnostics

| Seed | Block-3 individual low-impact removals | Maximum grouped cosine | Effective scalar rank / 128 | Similarity vs negative loss impact (Spearman) |
| ---: | ---: | ---: | ---: | ---: |
| 2026 | 26/32 | 0.5264 | 33.07 | 0.223 |
| 2027 | 24/32 | 0.5985 | 32.82 | 0.440 |
| 2028 | 20/32 | 0.5713 | 30.97 | 0.317 |

Individual low impact: <=0.01 validation CE increase and <=0.5 pp accuracy decrease, prespecified. Whole images are separated before image-half stability measurements. All blocks and both model families, including weak/negative predictors, are retained in the diagnostic report. Effective rank is not a count of removable units; independent seeds do not align neuron identities.

## Measured results: physical compression

| Block-3 removal | Policy | Retained real parameters | Total parameter saving | Test mean +/- training-seed SD |
| ---: | --- | ---: | ---: | ---: |
| 0% | baseline | 73898 | 0.00% | 78.99 +/- 0.22% |
| 25% | random | 64298 | 12.99% | 77.50 +/- 0.34% |
| 25% | incoming_norm | 64298 | 12.99% | 76.70 +/- 0.90% |
| 25% | activation_similarity | 64298 | 12.99% | 77.48 +/- 0.29% |
| 50% | random | 54698 | 25.98% | 74.08 +/- 0.32% |
| 50% | incoming_norm | 54698 | 25.98% | 72.25 +/- 3.16% |
| 50% | activation_similarity | 54698 | 25.98% | 74.12 +/- 1.31% |

Random: ten removal repetitions within each of three training seeds, averaged within seed before training-seed SD. No fine-tuning, BN re-calibration, weight compensation or test-based policy/size selection. All manifests and validation measurements were frozen before compact-model test evaluation.

At each retained parameter count, the following paired effects describe this pilot only (activation-similarity minus reference test accuracy, in percentage points):

- 25% removal vs incoming_norm: paired values +0.85, +1.87, -0.37; mean +0.78, sample SD 1.12 pp.
- 25% removal vs random: paired values +0.42, +0.26, -0.74; mean -0.02, sample SD 0.63 pp.
- 50% removal vs incoming_norm: paired values +2.26, +5.57, -2.23; mean +1.87, sample SD 3.91 pp.
- 50% removal vs random: paired values -0.60, +1.36, -0.65; mean +0.04, sample SD 1.15 pp.

## Measured decision

At 25% block-unit removal (12.99% total-parameter saving), similarity gives 77.48% test accuracy, a 1.51-pp drop. Its mean paired difference from random is -0.02 pp; the three seed effects do not have a consistent sign.
At 50% block-unit removal (25.98% total-parameter saving), similarity gives 74.12% test accuracy, a 4.87-pp drop. Its mean paired difference from random is +0.04 pp; the three seed effects do not have a consistent sign.
The fixed-coordinate similarity policy did not demonstrate an advantage over random removal. Mean performance against raw incoming norm is better, but reverses in seed 2028 at both sizes; raw norms also ignore BN scaling. Preserve this negative finding instead of claiming a pruning-method gain.
Physical compaction and a modest quality-size trade-off are demonstrated. Gate C's activation-policy advantage remains unconfirmed; Gate D is not satisfied by this pilot.

## Verification

Core PyTorch suite: 22 passed. Compact artifact verification: 1285 checks passed. Each new baseline passes the existing 35-check artifact verifier. Physical checks include zero-removal equivalence, compact-vs-masked equivalence, actual state parameter/buffer counts, exact checkpoint reload and recomputation from saved test logits. Source/configuration hashes and full epoch-resume states are preserved.
Threshold erratum: the initial seed-2026 report counted 24/32 low-impact block-3 removals. Two exact 0.5-pp boundary cases were excluded by floating-point subtraction; inclusive tolerance 1e-12 gives 26/32. No measured loss/accuracy, ordering, threshold value or selected block changed. The original artifacts are preserved and corrected labels are explicit in the multi-seed diagnostic output.

## Interpretation and limits

Infrastructure Gate A passes. Gate B has multi-seed diagnostic evidence for individually dispensable channels, not proof of near-duplicate whole units or a special quaternion effect. A positive compression result requires an actual quality-size trade-off against controls; the tables retain every outcome. Gate D (paper novelty/substance) remains open.
Simple positive-coordinate centered similarity can be a weak predictor; negative coefficients must not be replaced by a tuned score. Individually dispensable channels can be harmful when removed together. Raw kernel norms ignore BN scaling, so a difference against this reference does not establish superiority over strong weight-based pruning. Policy context: Ayinde2019, Vecchi2020, Singh2026; no faithful reproduction of their complete methods is claimed.
This study covers one small architecture, one dataset, three training seeds and a single selected block, without a direct scalar-policy control, fine-tuning, matched pre-pruning accuracy, MAC/latency measurements or a systematic novelty review. It supports a limited empirical pilot, not a general method claim.

## Next research decision

At the close of the original pilot, the next questions were a fixed BN-aware weight reference and H2: whether grouped diagnostics predict whole-unit removal harm better than ordinary scalar correlations. Completed follow-ups are recorded below when available. Only a validated signal would justify more architectures or a separate, equally budgeted fine-tuning study. The original similarity policy did not earn a method-superiority claim.

## Completed BN-aware weight-reference follow-up

One fixed additional magnitude criterion was authorized and frozen before computing its outcomes. It uses the joint norm of all four output rows in the Conv+BN-folded expanded real Hamilton matrix. It excludes folded bias and downstream effects; no training or BN re-calibration. Reference results above are reused unchanged. Source of the folding equation: PyTorchFusion260 in references.bib.

| Block-unit removal | BN-aware test mean +/- seed SD | Random test mean | BN minus random (pp) |
| ---: | ---: | ---: | ---: |
| 25% | 76.10 +/- 0.41% | 77.50% | -1.40 |
| 50% | 73.30 +/- 0.74% | 74.08% | -0.78 |

At 25% block-unit removal, BN-aware magnitude reaches 76.10% versus random 77.50% (mean paired BN-minus-random effect -1.40 pp). BN-aware is worse in each of the three training seeds at this size.
At 50% block-unit removal, BN-aware magnitude reaches 73.30% versus random 74.08% (mean paired BN-minus-random effect -0.78 pp). BN-aware is worse in each of the three training seeds at this size.
At 25% removal the BN-aware reference is also worse than raw magnitude in all three seeds. At 50% it improves the mean over raw magnitude, but reverses in seed 2028. Accounting exactly for BN's multiplicative scaling did not produce a reliably better importance criterion.
All 25 implementation tests and 130 BN artifact checks passed, including independent official folding, exact original-weight slicing, policy reconstruction and recomputed test metrics. Existing numerical sources and all original comparison artifacts were preserved.
This follow-up does not show that BN never matters. It shows this particular folded-weight norm remains a weak predictor in the fixed pilot. No bias-inclusive or alternative score was tried after seeing the outcome. The activation policy still has no demonstrated advantage over random. Next remains the prespecified grouped-versus-scalar diagnostic (H2), rather than tuning the weight criterion to these test results.

## Completed grouped-versus-scalar diagnostic

One prespecified scalar reference: for each component, maximum signed Pearson to a channel OUTSIDE its whole quaternion unit; average four maxima. Grouped reference: maximum existing centered whole-unit cosine. Both predict the same single-whole-unit validation-ablation target using calibration scores. Primary block 3; blocks 1/2 secondary. No new model inference, test access or training.

| Seed | Grouped Spearman | Scalar Spearman | Group-minus-scalar |
| ---: | ---: | ---: | ---: |
| 2026 | 0.223 | 0.487 | -0.264 |
| 2027 | 0.440 | 0.493 | -0.053 |
| 2028 | 0.317 | 0.157 | +0.160 |

Scalar Pearson predicts individual loss harm better in 2/3 primary seeds; grouped cosine in 1/3. Mean paired grouped-minus-scalar Spearman difference -0.052, sample SD 0.212. The prespecified repeatable-grouped-advantage rule returns False.
In the primary block, both rankings select a quarter of units whose mean separate individual-removal harm is lower than the uniform random expectation in all three seeds. This is a diagnostic signal, not simultaneous quarter-removal performance; the earlier greedy-pruning test accuracies belong to a different policy. Neither a weak positive Spearman nor individual dispensability establishes useful joint compression.
This particular grouped cosine has no demonstrated repeatable advantage over the fixed independent-scalar reference. H2 is not confirmed for these measurements. Close this candidate advantage rather than tune the score or select a favorable secondary block. This does not rule out all other measures of activation redundancy.
The user's broader structural-constraint hypothesis remains open. Each same-width quaternion connection uses four real coefficients to generate sixteen signed/tied real matrix entries. This factual parameter sharing does not prove that arbitrary alternative constraints would behave the same. Our controls do not isolate generic sharing from its specific pattern, width, compute or optimization. See docs/STRUCTURE_HYPOTHESIS.md; context: ZhangEtAl2021 in references.bib.
All 28 implementation tests and 200 diagnostic artifact checks passed. Initial failed spatial-shape assertion and its correction are preserved in results/group_scalar/FAILED_CHECKS.md and failed_run_protocol.json. Frozen numerical inputs and earlier outcomes remain unchanged.

## Completed learned-routing pilot (one seed)

Resume/final artifact review: 2026-11-03. Both new trainings completed 50 epochs before this review. Identity is reused from the equivalent original checkpoint. This is one seed (2026), not three replicated routing trainings.

| Routing condition | Selected validation | Final test |
| --- | ---: | ---: |
| identity (reused) | 78.30% | 79.00% |
| random | 78.00% | 78.14% |
| learned | 74.40% | 77.61% |

Epoch-40 soft-to-hard validation change: 79.20% to 56.00%, a 23.20-pp drop. The ten-epoch hard phase recovered part of this loss; the selected checkpoint still failed the prespecified validation gate.
Export changes 2/32 scalar output positions and creates 2/8 new unordered member groups. This is actual membership change, not only relabeling. It did not yield a quality gain in this pilot.
The soft assignment retained broad mixing (mean row entropy about 1.99 nats and about 30.4% mean mass on the eventual hard assignment). This is consistent with relying on a dense soft mixture rather than a near-discrete assignment. The immediate loss also includes changed downstream activation/BN distributions; this experiment does not isolate those effects. Ten late-schedule hard epochs and one temperature/init configuration do not rule out other training procedures.
The compression gate failed on validation before final test; the conditional stage records not_run_gate_failed, with no new compact models or compact-model test inference. Gate checks and serialized artifacts pass (64 routing checks; 7 conditional-stage checks). No result is claimed as a novel compression method. See results/learned_grouping_pilot/REPORT.md and docs/LEARNED_GROUPING_NOVELTY_AUDIT.md.

## Completed fixed-weight BN intervention at the routing boundary

Seed 2026, epoch 40 replay of the unchanged pilot. Both full soft/hard boundary states are now retained. The numerical epoch trace and boundary metrics match the original within 1e-8; unavailable original epoch-40 weights prevent a bitwise comparison. No optimizer updates after the boundary and no new official-test inference.

Refit only the four BNs downstream of the router, in forward order, on all 48,000 training images. No augmentation or dropout; global input moments with float64 accumulation and unbiased variance. Every model parameter, upstream statistics, counters and routing remain unchanged by refitting. Soft refitting controls for the procedure itself.

| Condition | Validation accuracy | Validation CE | Calibration accuracy |
| --- | ---: | ---: | ---: |
| soft_original | 79.20% | 0.609989 | 79.20% |
| hard_original | 56.00% | 1.636415 | 55.20% |
| soft_recalibrated | 79.00% | 0.609001 | 80.30% |
| hard_recalibrated | 66.30% | 0.930627 | 66.30% |

Hard validation recovery from BN refit: +10.30 pp; soft-control change: -0.20 pp. Original soft-minus-hard gap: 23.20 pp; gap with both refitted: 12.70 pp.
These values describe what one fixed-weight BN-statistic intervention repairs. Residual loss is not explained away by this procedure; it does not establish an unavoidable limitation of discrete grouping or isolate all remaining causes. One seed, architecture, boundary and estimator; no compression savings or novel-method claim. See results/routing_bn_diagnostic/REPORT.md and configs/routing_bn_diagnostic.json.

BN-only refitting repairs 10.30 pp of hard validation accuracy, while the soft control changes by -0.20 pp. The remaining refitted soft-minus-hard gap is 12.70 pp. Thus BN-buffer refitting is helpful but insufficient to restore soft-model quality in this intervention.
The secondary held-out calibration set agrees in direction: hard accuracy rises from 55.20% to 66.30%; refitted soft accuracy is 80.30%, leaving a 14.00-pp gap. No official test inference or weight updates were used in this diagnostic. Five implementation tests and 111 independent artifact checks pass; no scientific or implementation check failed in this stage.
Close the hypothesis that this particular downstream-BN refit alone restores the original soft-model quality. Do not call the remaining loss a proven inherent limitation or assign a uniquely causal percentage to BN. The next candidate experiment is a prespecified discrete-forward grouping-training comparison against paired fixed assignments, with extra variables/compute reported and equivalent final evaluation/refitting. Do not silently reopen the original failed compression gate. No further training is started by this diagnostic.

## Completed discrete-forward grouping pilot

One new seed-2026 50-epoch training with exact discrete assignment in every forward, Sinkhorn straight-through surrogate in backward, 40 adaptive epochs and 10 with assignment fixed. The paired fixed-random 50-epoch control is reused. Selection is minimum original-BN validation CE at epochs 40-50 for both; identical training-only downstream-BN refit after selection. No new official-test inference.

| Condition | Validation accuracy | Calibration accuracy |
| --- | ---: | ---: |
| random_original | 78.00% | 76.80% |
| learned_original | 78.10% | 76.70% |
| random_recalibrated | 78.10% | 77.20% |
| learned_recalibrated | 77.80% | 77.30% |

The selected exported assignment changes 6/32 scalar positions and creates 4/8 new unordered groups. Its BN-refitted learned-minus-fixed effect is -0.30 pp on validation and +0.10 pp on calibration.
The prespecified replication gate returns False: calibration gain False, validation quality True, actual membership change True. A positive gate only motivates replication; a negative gate preserves this unsuccessful configuration. Neither is a general method result or a novelty claim.
In this completed pilot, the primary calibration difference is one additional correct image out of 1,000, while validation has three fewer correct images. CE is worse for learned grouping on both held-out splits, before and after BN refitting. This does not demonstrate a useful learned-grouping advantage or establish equivalence; preserve the near-tied accuracy and unfavorable CE rather than claim a gain. The gate does not justify automatically launching replication or compression of this configuration.
The freeze has zero immediate prediction shift because routing was already discrete. This verifies the intended mechanism; it is not evidence of a learned-grouping advantage. Training adds 1,024 logit parameters and matching/surrogate compute; both fixed-index exports have 73,898 real parameters with no removed backbone channels.
Six implementation tests and 131 independent artifact checks pass. Numerical sources and paired reference artifacts are preserved. Prior context: Bengio2013STE, Mena2018, Wang2023PSTE, Wang2019FLGC and Zhang2021StructuredShuffle. See results/hard_forward_grouping/REPORT.md and configs/hard_forward_grouping.json.

## Completed affine-reconstruction and last-block merging pilot

Three existing quaternion CNN seeds, block 3, no new CNN training. Fit maps and choose eight disjoint pairs using 500 even calibration images; independently check combined reconstruction on 500 odd images. Right-Hamilton affine 4+4, diagonal affine 4+4, full-real affine 16+4 calibration coefficients. No labels in fitting. Both primary families passed the held-out gate in all three seeds; the diagonal family remains a coefficient-matched diagnostic.

Physically remove 8/32 final-block units and fold compensation into the existing GAP/real-linear readout. Every compact condition has 64,298 real parameters (12.99% fewer than 73,898), 838 buffer elements and a 269,712-byte standardized state file. No extra mapping layer remains. All manifests precede new validation inference; no BN refit, fine-tuning or official-test inference.

| Policy | Validation mean ± seed SD | Mean CE |
| --- | ---: | ---: |
| baseline | 77.93% ± 1.10 pp | 0.637730 |
| random | 76.37% ± 0.99 pp | 0.687761 |
| quaternion_right_pruned | 75.17% ± 0.83 pp | 0.707504 |
| quaternion_right_mean_only | 76.77% ± 1.55 pp | 0.676093 |
| quaternion_right_affine | 76.63% ± 1.10 pp | 0.667735 |
| full_real_pruned | 76.30% ± 0.82 pp | 0.692571 |
| full_real_mean_only | 76.50% ± 1.25 pp | 0.672888 |
| full_real_affine | 76.67% ± 1.53 pp | 0.654740 |

Right-Hamilton compensation improves accuracy over pruning the exact same victims by +2.10, +0.90 and +1.40 pp (mean +1.47 pp), but its mean accuracy is 0.13 pp below the same-victim mean-only correction. Full-real compensation improves over its own mean-only correction by 0.00, -0.20 and +0.70 pp. Thus a repeatable accuracy advantage beyond the cheap mean-only control is not demonstrated.
Validation accuracy with right-Hamilton/full-real merging is 76.63%/76.67%, versus 77.93% unpruned and 76.37% random-pruned (ten random repetitions averaged within each training seed). Paired right-Hamilton-minus-random differences are +0.37, +0.13 and +0.30 pp. These are small three-seed pilot effects, not a general superiority claim.
Retain the favorable CE evidence too: full-real merging has mean CE 0.654740, versus 0.672888 for its mean-only control and 0.687761 for random pruning; the unpruned mean is 0.637730. Right-Hamilton merging has CE 0.667735 versus 0.676093 for its mean-only control. Lower held-out logit MSE and better CE do not automatically imply better argmax accuracy. No special quaternion advantage is established.
Nine meaningful implementation tests and 922 independent artifact checks pass. Maximum cached-to-physical logit error is below 5e-6 (frozen tolerance 1e-4). Checks cover fit-only selection, affine normal equations, whole-image held-out splits, no hidden map parameters, exact kernel/BN slicing, compensation formulas, actual parameter/buffer/file counts and checkpoint reload predictions. All original numerical sources and model/data artifacts are preserved.
This is a bounded pilot at one layer and removal level. Map fitting/merging is established background (Kim2020, Ebid2025); quaternion-specific pruning also has prior art (Vecchi2020, Singh2026). The new evidence supports an empirical distinction between reconstruction/CE and classification accuracy, not a novel-method or paper-readiness claim. See results/quaternion_merging/DIAGNOSTIC_REPORT.md and REPORT.md.

## Completed structure-preserving intermediate merging pilot

Block 2, three existing CIFAR CNN seeds, one removal level, no new training/BN refit/test inference. Fit zero-offset maps on 500 calibration images and all 64 spatial positions; check on 500 separate calibration images. Use common full-fit-selected disjoint pairs for both methods, paired pruning and functional oracles. Freeze all choices before new validation inference.

Ordinary right-Hamilton associativity permits folding q_v=q_k*a into the next convolution as w_k <- w_k+a*w_v. A general matrix requires projecting R(w_v)A into the four-dimensional admissible block space. The statement concerns linear maps between post-activation feature maps and the next convolution, not commuting through ReLU. Affine offsets with zero padding are outside this pilot. No new theorem/method is claimed.

| Policy | Real parameters | Validation mean ± seed SD | Mean CE |
| --- | ---: | ---: | ---: |
| baseline | 73898 | 77.93% ± 1.10 pp | 0.637730 |
| random | 66954 | 73.22% ± 0.65 pp | 0.757720 |
| paired_pruned | 66954 | 71.40% ± 2.63 pp | 0.828642 |
| quaternion_exact | 66954 | 74.73% ± 0.83 pp | 0.719896 |
| full_projected | 66954 | 73.63% ± 0.65 pp | 0.738425 |
| full_substitution | functional oracle | 76.30% ± 0.26 pp | 0.683890 |

Projection discards 49.38-57.49% of the squared Frobenius energy of ADDED compensation blocks R(w_v)A, not of the complete layer. On held-out features this raises the consumer's pre-BN reconstruction MSE by 26.04-34.74% relative to full substitution in all three seeds.
Direct admissible fitting exceeds full-fit-then-project accuracy by +0.70, +1.80, +0.80 pp; mean +1.10 pp, sample SD 0.61 pp. CE and consumer MSE also favor direct quaternion fitting in every seed. Against ten random removals averaged within each seed, paired gains are +1.05, +2.39, +1.10 pp, mean +1.51 pp.
All actual compact methods have 66,954 real parameters (9.40% fewer), 870 buffer elements and 280,464-byte standardized state files. Direct quaternion fitting gives 74.73% mean validation accuracy versus 77.93% before compression, so meaningful quality loss remains. The 76.30% full-substitution result is a functional comparison and is not a compact quaternion model at this budget.
Five meaningful implementation tests and 853 independent artifact checks pass. Exact-Q composition and graph surgery agree within 1e-4 (observed consumer maximum below 4e-6). Verification covers all pair fits/normal equations, fit-only common selection, coefficient energy/projection, tensor slicing, actual budgets, source cache alignment and checkpoint/oracle predictions.
This supplies a repeatable pilot signal for the cost of enforcing an intermediate layer's structure and for fitting within the constraint from the start. It does not establish a new algorithm or broad quaternion superiority. One architecture/dataset/block/removal level and a common selection favoring the full fit limit interpretation. A second compression level and an independent architecture/dataset are needed before positioning a paper; a focused literature review is still required. Prior context: Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002.
Important POST-HOC analytic control: projecting the full fit using the keeper activation second moment reproduces the direct quaternion least-squares map across all 720 directed pairs (maximum matrix difference 1.02e-14). No new inference or validation selection was performed. Thus the +1.10-pp comparison is against unweighted Frobenius projection; the direct-Q fit is equivalent to standard data-weighted projection in the admissible subspace. It is not a new algorithm or evidence of an intrinsically quaternion-specific optimization benefit. The measured cost of restricting a full map and the need to retain actual model structure remain valid. The article hypothesis must concern that structural trade-off, with stronger controls, rather than claim novelty from direct fitting alone. See data_weighted_projection_audit.json and scripts/audit_internal_projection.py.

## Completed extension: two CNN depths and two compression levels

Three NEW CNN8 baselines completed 50 epochs each, using the original frozen trainer, split, normalization, optimizer and augmentation. Eight convolutions/four blocks, widths 8/16/24/32, 118,058 real parameters. Three existing CNN6 baselines are reused (73,898 parameters). Architectures differ in depth, widths, parameter count and compute; compare compression policies within each model. This covers two plain sequential CNNs, not independent architecture families.

Remove 4/16 or 8/16 block-2 quaternion units; no compression updates, BN refit or new official-test inference. Use 500 even calibration images for zero-offset fits and nested common disjoint pair selection, 500 odd images for feature/consumer checks. All compressed choices precede their new validation inference. The original CNN6/four-removal cell remains exploratory/reused. Twelve compression cells share six trained models and one dataset; they are not twelve independent training replications. Validation also selected the baseline checkpoint, so it is not a new blind benchmark.

| Architecture | Removed | Compact real parameters | Baseline | Direct Q | Unweighted projection | Data-weighted projection | Random | Full functional oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnn6 | 4 | 66954 | 77.93% | 74.73% | 73.63% | 74.73% | 73.22% | 76.30% |
| cnn6 | 8 | 60010 | 77.93% | 65.20% | 61.57% | 65.20% | 53.97% | 70.77% |
| cnn8 | 4 | 112266 | 80.30% | 78.27% | 77.20% | 78.27% | 77.00% | 79.03% |
| cnn8 | 8 | 106474 | 80.30% | 73.43% | 69.40% | 73.43% | 65.84% | 75.63% |

Accuracy values are seed means; random repetitions are first averaged within training seed. Full seed SDs, CE and all paired differences appear in the study report.

CNN6, 4/16 removal: direct-Q minus full_projected accuracy is +0.70, +1.80, +0.80 pp across seeds (mean +1.10, seed SD 0.61 pp); mean CE difference -0.018529. Negative effects and inconsistent signs are retained.
CNN6, 4/16 removal: direct-Q minus random accuracy is +1.05, +2.39, +1.10 pp across seeds (mean +1.51, seed SD 0.76 pp); mean CE difference -0.037824. Negative effects and inconsistent signs are retained.
CNN6, 8/16 removal: direct-Q minus full_projected accuracy is +3.00, +3.30, +4.60 pp across seeds (mean +3.63, seed SD 0.85 pp); mean CE difference -0.103531. Negative effects and inconsistent signs are retained.
CNN6, 8/16 removal: direct-Q minus random accuracy is +8.44, +14.85, +10.41 pp across seeds (mean +11.23, seed SD 3.28 pp); mean CE difference -0.312612. Negative effects and inconsistent signs are retained.
CNN8, 4/16 removal: direct-Q minus full_projected accuracy is +1.20, +1.80, +0.20 pp across seeds (mean +1.07, seed SD 0.81 pp); mean CE difference -0.020669. Negative effects and inconsistent signs are retained.
CNN8, 4/16 removal: direct-Q minus random accuracy is +1.44, -0.05, +2.40 pp across seeds (mean +1.26, seed SD 1.23 pp); mean CE difference -0.030855. Negative effects and inconsistent signs are retained.
CNN8, 8/16 removal: direct-Q minus full_projected accuracy is +5.90, +1.30, +4.90 pp across seeds (mean +4.03, seed SD 2.42 pp); mean CE difference -0.096933. Negative effects and inconsistent signs are retained.
CNN8, 8/16 removal: direct-Q minus random accuracy is +6.00, +3.22, +13.55 pp across seeds (mean +7.59, seed SD 5.35 pp); mean CE difference -0.197001. Negative effects and inconsistent signs are retained.
Across the twelve cells, unweighted projection discards 48.83-59.97% of the squared energy of ADDED compensation blocks. Its pre-BN consumer MSE change relative to full substitution ranges from +26.04% to +45.59%, with an increase in 12/12 cells. These diagnostics have different denominators and are not classification-accuracy effects.
Direct-Q (equivalently, data-weighted) fitting has an observed consumer-MSE change of +18.47% to +29.32% relative to full substitution, increasing it in 12/12 cells. These are measured effects of the specified feature-fit procedure, not an optimality lower bound for every structure-preserving compensation. The full functional reference is not a task-accuracy upper bound.
The strong data-moment control matches direct-Q maps within 1.09e-14 and saved predictions within 0. Direct constrained fitting and data-weighted projection are equivalent here. A gain against unweighted projection cannot establish a new algorithm or a unique quaternion optimization benefit.
Functional full-map substitution uses the original graph and is not a compact quaternion architecture at the reported parameter budget. All physical policies at a given architecture/removal level have identical parameter and buffer budgets. A quarter/half of block-2 units is not a quarter/half of total network parameters; no masked-sparsity or latency claim is made.
Five targeted implementation tests and 4118 artifact checks pass. Checks replay every compact checkpoint and functional oracle, recompute held-out predictions/metrics, verify fit-only selection and least-squares equations, independently expand/fold Hamilton blocks, and check all three 50-epoch training histories, first updates, checkpoint selection and resume states. One root launcher remains: WZNOW_BADANIA.cmd.
This extends the empirical structural trade-off to a second compression level and CNN depth. It does not establish a novel method, general quaternion superiority or paper readiness. One dataset, two related architectures, three seeds and shared validation limit generalization. Focused prior-art positioning (Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002) and broader validation remain necessary. See results/internal_replication/REPORT.md, comparison.png and structural_cost.png.

## Completed multi-source compensation pilot (Hamilton-restricted vs unconstrained, real control)

Same six quaternion models and frozen victims as the extension above, plus the three existing real CNN6 baselines. Each removed unit is reconstructed from ALL kept units by ridge regression (alpha fixed a priori at 1e-3) on 500 even calibration images and folded into the next convolution. Unconstrained multi-source ridge folding is prior art (GRAIL2026); only the right-Hamilton restriction q_v ~= sum_k q_k*a_k, which keeps a quaternion consumer, is under study. No fine-tuning, BN refit or official-test inference.

| Architecture | Removed | Real parameters | Baseline | Random | Pairwise fold | Multi-source (random victims) | Multi-source projected | Multi-source Hamilton | Unconstrained oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnn6 | 4 | 66954 | 77.93% | 73.22% | 74.73% | 75.99% | 74.70% | 76.63% | 77.07% |
| cnn6 | 8 | 60010 | 77.93% | 53.97% | 65.20% | 71.86% | 69.40% | 73.17% | 74.33% |
| cnn8 | 4 | 112266 | 80.30% | 77.00% | 78.27% | 79.05% | 78.50% | 79.53% | 79.80% |
| cnn8 | 8 | 106474 | 80.30% | 65.84% | 73.43% | 76.14% | 74.00% | 77.13% | 78.23% |
| real_cnn6 | 8 | 66026 | 75.53% | 70.60% | 73.70% | 73.27% | n/a | 74.37% | n/a |
| real_cnn6 | 16 | 59098 | 75.53% | 57.14% | 67.47% | 66.80% | n/a | 70.37% | n/a |

Validation seed means; random rows averaged within seed first. Full SDs, CE, check-half results and the matched-savings table are in results/multisource_compensation/REPORT.md.

cnn6, 4 removed: multi_q minus quaternion_exact is +1.20, +0.50, +4.00 pp (mean +1.90; 3/3 seeds positive).
cnn6, 4 removed: multi_q minus multi_full_projected is +1.80, +1.60, +2.40 pp (mean +1.93; 3/3 seeds positive).
cnn6, 4 removed: multi_full_substitution minus multi_q is +0.50, +0.30, +0.50 pp (mean +0.43; 3/3 seeds positive).
cnn6, 8 removed: multi_q minus quaternion_exact is +8.10, +6.30, +9.50 pp (mean +7.97; 3/3 seeds positive).
cnn6, 8 removed: multi_q minus multi_full_projected is +2.70, +6.40, +2.20 pp (mean +3.77; 3/3 seeds positive).
cnn6, 8 removed: multi_full_substitution minus multi_q is +1.60, +0.40, +1.50 pp (mean +1.17; 3/3 seeds positive).
cnn8, 4 removed: multi_q minus quaternion_exact is +0.80, +2.00, +1.00 pp (mean +1.27; 3/3 seeds positive).
cnn8, 4 removed: multi_q minus multi_full_projected is +0.70, +1.50, +0.90 pp (mean +1.03; 3/3 seeds positive).
cnn8, 4 removed: multi_full_substitution minus multi_q is -0.30, +1.00, +0.10 pp (mean +0.27; 2/3 seeds positive).
cnn8, 8 removed: multi_q minus quaternion_exact is +3.20, +4.20, +3.70 pp (mean +3.70; 3/3 seeds positive).
cnn8, 8 removed: multi_q minus multi_full_projected is +4.80, +1.30, +3.30 pp (mean +3.13; 3/3 seeds positive).
cnn8, 8 removed: multi_full_substitution minus multi_q is +1.20, +2.70, -0.60 pp (mean +1.10; 2/3 seeds positive).
real_cnn6, 8 removed: multi_real minus pair_merge is +1.20, -0.50, +1.30 pp (mean +0.67; 2/3 seeds positive).
real_cnn6, 16 removed: multi_real minus pair_merge is +5.90, -1.40, +4.20 pp (mean +2.90; 2/3 seeds positive).

Exact-fold checks: 216 physical compact models equal their functional substitution within 8.58e-06. Baseline, random and pairwise rows reproduce the extension above exactly.
Measured pattern (pre-fine-tuning): fitting all sources inside the Hamilton family recovers most of the gap to the unconstrained functional oracle, while unweighted projection of the unconstrained fit is consistently worse. The projected comparison remains a weak reference (the weighted equivalent is not run for the ridge multi-source fit). At matched parameter savings in CNN6, multi-source Hamilton and multi-source real compensation lose similar accuracy from their own baselines; uncompensated whole-unit removal is more damaging in the quaternion model. These are pilot observations on one block, small CNNs and CIFAR-10, without fine-tuning; no novelty or general superiority claim. Next: equal-budget fine-tuning of all compact policies.

## Completed equal-budget fine-tuning of the compact policies

183 runs: every compact model of the multi-source study (first 3 random victim sets) and the uncompressed baseline, 3 epochs, AdamW LR 0.0001 cosine to 1e-06, identical batch/augmentation stream per training seed. Final-epoch validation accuracy, no checkpoint selection, no official-test inference.

| Architecture | Removed | Policy | Before | After fine-tuning |
| --- | ---: | --- | ---: | ---: |
| cnn6 | 4 | baseline | 77.93% | 77.93% ± 0.93 pp |
| cnn6 | 4 | multi_full_projected | 74.70% | 76.60% ± 0.56 pp |
| cnn6 | 4 | multi_q | 76.63% | 77.17% ± 0.32 pp |
| cnn6 | 4 | paired_pruned | 71.40% | 76.57% ± 0.65 pp |
| cnn6 | 4 | quaternion_exact | 74.73% | 76.73% ± 0.83 pp |
| cnn6 | 4 | random | 72.82% | 76.47% ± 1.16 pp |
| cnn6 | 8 | baseline | 77.93% | 77.93% ± 0.93 pp |
| cnn6 | 8 | multi_full_projected | 69.40% | 75.20% ± 0.62 pp |
| cnn6 | 8 | multi_q | 73.17% | 75.10% ± 0.78 pp |
| cnn6 | 8 | paired_pruned | 44.53% | 74.97% ± 0.31 pp |
| cnn6 | 8 | quaternion_exact | 65.20% | 74.87% ± 0.71 pp |
| cnn6 | 8 | random | 55.33% | 74.70% ± 0.58 pp |
| cnn8 | 4 | baseline | 80.30% | 80.50% ± 1.01 pp |
| cnn8 | 4 | multi_full_projected | 78.50% | 79.97% ± 0.67 pp |
| cnn8 | 4 | multi_q | 79.53% | 79.73% ± 0.49 pp |
| cnn8 | 4 | paired_pruned | 75.30% | 79.40% ± 0.46 pp |
| cnn8 | 4 | quaternion_exact | 78.27% | 79.73% ± 0.50 pp |
| cnn8 | 4 | random | 76.94% | 79.33% ± 0.54 pp |
| cnn8 | 8 | baseline | 80.30% | 80.50% ± 1.01 pp |
| cnn8 | 8 | multi_full_projected | 74.00% | 78.73% ± 0.46 pp |
| cnn8 | 8 | multi_q | 77.13% | 78.20% ± 0.26 pp |
| cnn8 | 8 | paired_pruned | 61.67% | 78.23% ± 0.35 pp |
| cnn8 | 8 | quaternion_exact | 73.43% | 78.20% ± 0.92 pp |
| cnn8 | 8 | random | 66.22% | 77.44% ± 0.57 pp |
| real_cnn6 | 8 | baseline | 75.53% | 75.63% ± 0.80 pp |
| real_cnn6 | 8 | multi_real | 74.37% | 74.80% ± 1.35 pp |
| real_cnn6 | 8 | pair_merge | 73.70% | 74.87% ± 1.05 pp |
| real_cnn6 | 8 | paired_pruned | 71.67% | 74.40% ± 0.92 pp |
| real_cnn6 | 8 | random | 71.09% | 74.07% ± 1.00 pp |
| real_cnn6 | 16 | baseline | 75.53% | 75.63% ± 0.80 pp |
| real_cnn6 | 16 | multi_real | 70.37% | 72.93% ± 1.63 pp |
| real_cnn6 | 16 | pair_merge | 67.47% | 72.47% ± 1.79 pp |
| real_cnn6 | 16 | paired_pruned | 59.93% | 72.20% ± 1.90 pp |
| real_cnn6 | 16 | random | 56.29% | 70.60% ± 1.56 pp |

After fine-tuning, cnn6, 4 removed: multi_q minus quaternion_exact is +1.00, +0.40, -0.10 pp (mean +0.43; 2/3 seeds positive).
After fine-tuning, cnn6, 4 removed: multi_q minus multi_full_projected is +0.70, +0.90, +0.10 pp (mean +0.57; 3/3 seeds positive).
After fine-tuning, cnn6, 4 removed: multi_q minus random is -0.37, +2.27, +0.20 pp (mean +0.70; 2/3 seeds positive).
After fine-tuning, cnn6, 8 removed: multi_q minus quaternion_exact is +0.10, +1.40, -0.80 pp (mean +0.23; 2/3 seeds positive).
After fine-tuning, cnn6, 8 removed: multi_q minus multi_full_projected is -0.10, +0.10, -0.30 pp (mean -0.10; 1/3 seeds positive).
After fine-tuning, cnn6, 8 removed: multi_q minus random is +0.40, +1.43, -0.63 pp (mean +0.40; 2/3 seeds positive).
After fine-tuning, cnn8, 4 removed: multi_q minus quaternion_exact is +0.10, -0.40, +0.30 pp (mean +0.00; 2/3 seeds positive).
After fine-tuning, cnn8, 4 removed: multi_q minus multi_full_projected is -0.40, -0.40, +0.10 pp (mean -0.23; 1/3 seeds positive).
After fine-tuning, cnn8, 4 removed: multi_q minus random is +0.40, +0.13, +0.67 pp (mean +0.40; 3/3 seeds positive).
After fine-tuning, cnn8, 8 removed: multi_q minus quaternion_exact is -1.20, +0.50, +0.70 pp (mean +0.00; 2/3 seeds positive).
After fine-tuning, cnn8, 8 removed: multi_q minus multi_full_projected is -1.00, -0.50, -0.10 pp (mean -0.53; 0/3 seeds positive).
After fine-tuning, cnn8, 8 removed: multi_q minus random is -0.10, +1.33, +1.03 pp (mean +0.76; 2/3 seeds positive).
After fine-tuning, real_cnn6, 8 removed: multi_real minus random is -0.17, +1.00, +1.37 pp (mean +0.73; 2/3 seeds positive).
After fine-tuning, real_cnn6, 16 removed: multi_real minus random is +1.93, +1.93, +3.13 pp (mean +2.33; 3/3 seeds positive).

Three epochs at one fixed schedule; longer or tuned fine-tuning could change rankings. See results/finetune_compression/REPORT.md for all policies, check-half values and the matched-savings table.

## Completed second-dataset replication: EuroSAT four-band (NIR, R, G, B)

EuroSAT_MS (Zenodo 7711810), q = NIR + R i + G j + B k, no zero real part. Quaternion and real CNN6 baselines (3 seeds each) trained with the unchanged CIFAR trainer and budget; the 5,400-image test split is held out. All compression and fine-tuning rules were copied from the CIFAR studies before any EuroSAT compression outcome.

Seed 2026: quaternion validation 96.50% (73898 parameters, epoch 42), real validation 96.40% (73098 parameters, epoch 44).
Seed 2027: quaternion validation 96.50% (73898 parameters, epoch 45), real validation 95.80% (73098 parameters, epoch 47).
Seed 2028: quaternion validation 97.00% (73898 parameters, epoch 38), real validation 96.50% (73098 parameters, epoch 38).

EuroSAT, without fine-tuning:
- cnn6, 4 removed: multi_q minus quaternion_exact is +1.00, +2.00, +1.90 pp (mean +1.63; 3/3 seeds positive).
- cnn6, 4 removed: multi_q minus multi_full_projected is +0.50, +0.40, +0.50 pp (mean +0.47; 3/3 seeds positive).
- cnn6, 4 removed: multi_full_substitution minus multi_q is +0.20, +0.10, -0.30 pp (mean +0.00; 2/3 seeds positive).
- cnn6, 4 removed: multi_q minus random is +2.46, +3.39, +3.31 pp (mean +3.05; 3/3 seeds positive).
- cnn6, 8 removed: multi_q minus quaternion_exact is +3.50, +3.60, +8.30 pp (mean +5.13; 3/3 seeds positive).
- cnn6, 8 removed: multi_q minus multi_full_projected is +3.00, +1.00, +0.30 pp (mean +1.43; 3/3 seeds positive).
- cnn6, 8 removed: multi_full_substitution minus multi_q is +0.10, -0.40, +1.60 pp (mean +0.43; 2/3 seeds positive).
- cnn6, 8 removed: multi_q minus random is +18.13, +23.66, +16.66 pp (mean +19.48; 3/3 seeds positive).
- real_cnn6, 8 removed: multi_real minus random is +2.22, +1.66, +3.43 pp (mean +2.44; 3/3 seeds positive).
- real_cnn6, 16 removed: multi_real minus random is +14.49, +11.17, +17.41 pp (mean +14.36; 3/3 seeds positive).

EuroSAT, after equal-budget fine-tuning:
- cnn6, 4 removed: multi_q minus quaternion_exact is -0.20, -0.30, -0.10 pp (mean -0.20; 0/3 seeds positive).
- cnn6, 4 removed: multi_q minus multi_full_projected is -0.50, +0.10, -0.20 pp (mean -0.20; 1/3 seeds positive).
- cnn6, 4 removed: multi_q minus random is +0.03, +0.03, +0.23 pp (mean +0.10; 3/3 seeds positive).
- cnn6, 8 removed: multi_q minus quaternion_exact is -0.10, +0.20, +0.50 pp (mean +0.20; 2/3 seeds positive).
- cnn6, 8 removed: multi_q minus multi_full_projected is +0.10, -0.20, +0.10 pp (mean +0.00; 2/3 seeds positive).
- cnn6, 8 removed: multi_q minus random is +0.37, -0.10, +0.10 pp (mean +0.12; 2/3 seeds positive).
- real_cnn6, 8 removed: multi_real minus random is -0.10, +0.00, +0.27 pp (mean +0.06; 1/3 seeds positive).
- real_cnn6, 16 removed: multi_real minus random is +0.03, +1.27, +0.40 pp (mean +0.57; 3/3 seeds positive).

See results/eurosat_compression/REPORT.md for all policies and absolute accuracies.

## Quality-size curve: all-layer compression (pre-repair table below)

Started 2026-11-04 at 15:11 CET, using the prepared Claude Code protocol. Current published summary status: completed. The separate runner state is results/compression_curve_runner.json; progress between phase summaries is in results/compression_curve_run.log. EuroSAT runs automatically after all CIFAR phases. The older one-block and fine-tuning results above remain separate studies.

All six convolution outputs are narrowed by the same unit fraction, shallow to deep; compensation is fitted on the current partly compressed model. Three seeds, five fractions, paired random and greedy-victim controls, BN-folded magnitude reference. Main fractions/settings were copied from the handoff; the pre-existing disclosure that 0.125 was added after a seed-2026 diagnostic is retained in configs/compression_curve.json and docs/PLAN_IJCNN2027.md. This low-end point is not newly blind evidence.

| Architecture | Unit fraction removed at each site | Actual compact real parameters | Random mean | BN-magnitude mean | Greedy compensation mean ± seed SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| cnn6 | 0.125 | 56850 | 40.31% | 43.57% | 72.20% ± 0.44 pp |
| cnn6 | 0.25 | 42034 | 21.45% | 29.50% | 59.67% ± 3.21 pp |
| cnn6 | 0.5 | 19098 | 12.19% | 11.33% | 21.97% ± 4.12 pp |
| cnn6 | 0.75 | 5090 | 9.93% | 10.03% | 10.03% ± 0.06 pp |
| cnn6 | 0.875 | 1434 | 10.00% | 10.00% | 10.17% ± 0.29 pp |
| real_cnn6 | 0.125 | 56024 | 41.25% | 43.93% | 71.27% ± 1.53 pp |
| real_cnn6 | 0.25 | 41326 | 20.31% | 23.13% | 62.03% ± 1.99 pp |
| real_cnn6 | 0.5 | 18626 | 11.41% | 11.40% | 32.03% ± 2.54 pp |
| real_cnn6 | 0.75 | 4854 | 10.03% | 10.67% | 11.97% ± 2.90 pp |
| real_cnn6 | 0.875 | 1316 | 10.00% | 10.93% | 11.30% ± 2.25 pp |

These are validation results WITHOUT fine-tuning, averaged across three seeds; random repetitions are averaged within seed. Half the units at every site leaves about 19k parameters (roughly a quarter of the baseline parameter count), not half the model's parameters. At strong compression, the accumulation of errors makes even compensated networks approach chance. A favorable comparison to pruning does not make these low absolute accuracies useful.

The 411 phase-A outcomes include 1080 site-fold comparisons (maximum discrepancy 1.03e-05). The independent phase-A audit checks the 246 saved compact checkpoints, their actual shapes/parameter/buffer budgets, matched victims, finite metrics, seed aggregation, disjoint image splits and protected input hashes. It does not claim independent prediction replay for all outcomes or verification of unfinished training.

Current CIFAR summary status: completed. The table above remains the pre-repair subset. Completed CIFAR/EuroSAT repair and scratch outcomes are synthesized in results/qnn_compression_review/REPORT.md; compare their budgets explicitly. Refinement and queued study status are reported below. No new official-test inference. Prior context remains GRAIL2026, Kim2020 and Asiaee2026CMR. Original report and figure: results/compression_curve/REPORT.md and quality_size_curve.png.

## Registered follow-up: denser quality-size curve

Registered 2026-11-04T16:13:28.462650+00:00; current queue status: completed. After BOTH main CIFAR/EuroSAT curves finish, run fractions 0.375 and 0.625 on CIFAR, then EuroSAT. Each extension includes compression without fine-tuning, identical three-epoch fine-tuning controls and full-budget scratch training, with unchanged architectures, seeds, repetitions and data splits. Outputs: results/compression_curve_refinement and results/compression_curve_eurosat_refinement.

These points were selected after inspecting the coarse-grid outcomes. They are exploratory follow-up measurements using the same baseline models, not independent/blind replication. Preserve the original results, deduplicate baselines when combining the seven-point curves, and retain unfavorable seeds. No outcome is claimed from being registered or queued. Execution definition: configs/compression_curve_queue.json; live state: results/compression_curve_queue.json. Both root launchers now start/resume the whole queue.

## Registered activation-sensitivity study

Registered 2026-11-04T17:50:37.618012+00:00; queue state: completed. After all existing coarse/refined curves: CIFAR quaternion CNN6 only, three seeds, six fresh 50-epoch baselines. Alternatives to the existing split ReLU are split LeakyReLU (slope 0.1) and coupled MagnitudeTanh (PoppelbaumSchwung2024Activations Eq.21). No extra trainable activation parameters. Compression at fractions 0.25/0.5 and three-epoch fine-tuning reuse the frozen implementations, with random victim streams aligned to the original ReLU reference.

Compare absolute accuracy/CE AND loss from each activation's own full baseline; retain all paired seed effects. This bounded pilot has no activation-specific scratch or EuroSAT runs, and no matched real coupled or split-Tanh control. MagnitudeTanh also changes boundedness/saturation, so it does not isolate coupling. No activation gain is inferred from registration, implementation tests or CPU smoke checks. Output: results/activation_sensitivity; live state: results/activation_sensitivity_queue.json.

## Registered low-label POST-compression repair

Queue state: completed. Appended after all curves and activation sensitivity. CIFAR ReLU quaternion CNN6, 25% unit removal at all six sites, three seeds; same victims with/without compensation. 54 fixed-budget repairs: 0/5/20 repair labels per class, supervised-only, current-student online pseudo-labels, and a frozen full-teacher pseudo-label control. Primary low-label budgets are 5/20; zero is a boundary control.

The original model was fully supervised and validation-selected. Label counts constrain repair weight updates only; labeled diagnostics and inherited normalization/BN statistics are outside this budget. This is not end-to-end few-shot or unsupervised training. Same-distribution CIFAR simulates unavailable repair annotations; no actual deployment or domain shift is tested. Fixed 1000 opportunities, snapshots 0/200/1000, confidence .95, fixed BN statistics/dropout, paired streams and disjoint U pool. No accepted target means no AdamW update, including no decay-only drift. Frozen teacher has an additional resource/inference cost. Confidence filtering has prior art (Sohn2020FixMatch); crop/flip-only views are not FixMatch reproduction. No gain follows from registration or readiness checks.

Config: configs/pseudolabel_repair.json; worker: scripts/run_pseudolabel_queue.py; output: results/pseudolabel_repair. Preserve unfavorable seeds, pseudo-label precision/coverage and actual optimizer updates. Earlier full-label three-epoch fine-tuning uses different BN/dropout settings and is not a matched label-efficiency control.

## Approved focus and bounded structural control

Primary question: compression of an existing QNN family, preservation of structure, actual size, quality/repair cost and failure regimes. A new method or intrinsic quaternion superiority is not assumed. Read results/qnn_compression_review/REPORT.md for the completed CIFAR/EuroSAT all-site evidence and the separately labeled single-site functional oracle. Keep unequal scratch/repair budgets and negative effects explicit. Pseudo-label repair remains supplementary.

Structure-control queue state: completed. Appended after all existing jobs. Three new CIFAR CNN6 baselines with known 4x4 circulant weight-sharing (Ding2017CirCNN context), 50 epochs, fractions .25/.5; greedy compensation, same-victim pruning and BN-magnitude, before/after equal three-epoch repair. Reuse original Q/real results and add six missing Q same-victim pruning repairs. Same scalar widths, input information, seeded coefficient tensors and exact real parameter counts (73,898 / 42,034 / 19,098). Circulant groups are not quaternion units; ordinary expanded convolution is not a CirCNN FFT reproduction.

This alternative was chosen after existing Q/real outcomes. Report absolute accuracy/CE, own-baseline losses, baseline-quality gaps and all paired seeds. One matched pattern does not isolate a quaternion-specific mechanism or prove that any structure behaves alike. No random algebra, new multiplication rule, runtime claim, extra Circulant scratch/EuroSAT sweep or gain inferred from smoke checks. Source/config: scripts/structure_control.py and configs/structure_control.json; state: results/structure_control_queue.json. Both root launchers resume the full chain.

## Genuine low-label SSL compression pilot: before choosing the commitment title

User authorized simulations first, then a decision on the title/abstract and reply to Marcus. Runner state: completed. This NEW pilot starts learning with only 50 training labels/class; it differs from the completed repair-only study with a fully supervised parent. One seed, QNN and exactly matched circulant control, two fresh 20x383-step soft-SSL baselines and 12 continuations. Reuse the shared FewshotLearning view/loss code (Sohn2020FixMatch / Chen2023SoftMatch context, not full recipe reproduction).

Same 500 labeled/1000 validation IDs as the sibling seed0 reference. Reserve 500 fit and 500 check images from U, never weight training, leaving 47,500 unlabeled images; this is a disclosed adaptation. Adapter exposes existing GAP features/logits and adds no parameters. Uniform .25 unit removal, 73,898 to 42,034 actual real parameters; compensate/prune the SAME victims and keep an uncompressed continuing control. 1000 repair steps, paired restarting SGD/sampler streams, inherited soft-confidence state. CE-U control sees the same weak/strong L/U views and joint BN but no pseudo-label loss; it is not labeled-only training. No EMA/contrastive/triplet additions or test inference.

Fixed diagnostics measure classification, pseudo-label coverage/precision, soft-weighted correctness and agreement/KL with parent. Preserve all negative conditions, compare uncompressed continuation and same-victim removal. One seed cannot establish robustness, equivalence or a new mechanism. No title/author list/email is automatically finalized. Technical checks and tiny CPU/GPU preflight values are not scientific results. Config: configs/ssl_compression_pilot.json; log: results/ssl_compression_pilot_run.log; output: results/ssl_compression_pilot/REPORT.md. Both root launchers resume this stage; epoch checkpoints include full optimizer/RNG/loader/soft-confidence state.
