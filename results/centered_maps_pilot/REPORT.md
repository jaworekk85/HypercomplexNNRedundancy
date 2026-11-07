# Centered quaternion compensation pilot

Status: completed. One CIFAR BN2026 f=.5 late consumer.

## Hypothesis

Calibrating a per-removed-channel Q center together with shared Q maps may recover useful outputs at unchanged physical compact size.

## Measured results

| Case | Calibrated reals | Fit MSE | Check MSE | Validation before (%) | Validation after3ep (%) |
|---|---:|---:|---:|---:|---:|
| mapfold | 0 | 0.20097541 | 0.20386748 | 33.20 | 56.70 |
| a_only | 1024 | 0.18470907 | 0.18802409 | 34.20 | 56.30 |
| mu_only | 64 | 0.19942219 | 0.20252192 | 32.80 | 56.60 |
| a_mu | 1088 | 0.18428406 | 0.18760790 | 34.50 | 56.30 |
| direct-Q, REUSED | 18432 | context | context | 34.80 | 56.70 |

## Method and interpretation limits

Outcome: joint A/mu versus the present hybrid mapfold control gains1.3pp
before repair but loses.4pp afterward. Versus output-fitted A alone, joint
adds.3pp before and0pp after. No after-repair benefit demonstrated in this
single source/consumer. Validation CE control/A-only/mu-only/joint:
1.2202836456/1.2224404907/1.2222688904/1.2227811584; all calibrated arms are
slightly worse than control. Separate calibration-check accuracy after repair:
55.6/55.0/56.4/55.4%; check CE1.2699202576/1.2681970825/1.2715195313/1.2686616211.
Keep all signs and do not infer equivalence/significance from tiny differences.

Joint linear system retained rank1088/1088; recovered A-to-mu system rank64/64,
condition94.419, maximum residual8.02e-17. Exact factor recovery succeeded.
Four repairs each attempted1125/successful1124 AMP steps. Joint-system solve
time1.86sec excludes collection/verification, not an equal-cost benchmark.

Independent CPU audit PASSED:134 source/299 input hashes unchanged,16 full
before/after classification split evaluations,8 full local-MSE replays, original
factor-to-kernel identity, fixed C/source, fixed masks and recorded head folds,
actual standard6Q/19098-real architectures and checkpoint/resume budgets.
See independent_audit.json and scripts/review_centered_maps_pilot.py.

Reuse direct-pilot sites0-3 and fixed masks. The teacher is current pre-site4, already partly compressed, not the pristine full baseline. Four cases start at the same old rounded site4 map-fold; this hybrid control differs from the old all-sites map-fold reference. One mu[v] per removed channel shared over all outputs/offsets. U=U0+(A-A0)C-A*mu, fixed C and anchored B; initialization roundoff is measured. Fit500/check500 held-out calibration images; cached moments agree with actual forwards. All objective comparisons use identical output SSE plus lambda||U-U0||^2, not penalties on A/mu themselves. Joint linear (deltaA,d) solution is accepted only after recovering mu such that A*mu=d and checking the original product formula. Linear algebra is established, not a novel theorem. Matrix rank, condition, residuals, factor norms and any failures are retained.

Final same-victim real-head fold is recomputed per case, then ALL ordinary compact weights/BN train for the SAME3ep CE recipe. No A/mu factors remain during repair or inference. Six Q convolutions/19098 trainable reals in every final model. Local calibration uses18432 coefficient slots, A1024/mu64/joint1088 adjustable real values; no masks-as-compression claim. Reused direct-Q reference has the same initialization stem/support/repair recipe but different calibration space/cost. Equal epoch counts do not establish equal total runtime/compute. One discovery source only; no significance, robust superiority or held-out-test claim. Negative/tiny gains retained. Priors: VieiraValle2022ELM, Kim2020, Ding2021ResRep, Zhang2026RCP, Singh2026. Exact novelty/priority unconfirmed. No sweep, baseline retraining, new test or automatic extension.

Atomic epoch resume captures optimizer/scheduler/scaler/RNG states; partial epoch is repeated. STOP requests pause after a batch. Final epoch3 endpoint, no validation selection. Successful/attempted AMP updates recorded separately.
