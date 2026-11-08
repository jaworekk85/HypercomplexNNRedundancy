# Two-block Q/dense/PHM structure selection

Status: completed; seed2026, two local internal-channel pruning sites.

## Hypothesis

Hard variable-n PHM selection may improve quality per stored parameter beyond a Q/real-only hybrid; not established novelty or gain.

## Measured results

| Policy | Actual trainable/stored reals | Budget slack | Validation before (%) | Validation after (%) |
|---|---:|---:|---:|---:|
| all_q | 232266 | 0 | 86.50 | 89.30 |
| q_real | 232266 | 0 | 86.50 | 89.30 |
| fixed_phm4 | 213898 | 18368 | 62.70 | 86.70 |
| variable_n | 232266 | 0 | 86.50 | 89.30 |
| weight_only_selector | 218906 | 13360 | 69.10 | 83.50 |

## Interpretation and limits

ONE seed, TWO internal basic-block consumers. Standalone post-pruning diagnostic from the selected Q ResNet baseline, not reuse of an all-nine-sites compressed network. Remaining seven blocks fixed during fitting, all parameters train during identical3ep repair. Reference budget is actual full-model count with only these two hidden widths halved; unchanged residual outputs. Per-layer width/type selected jointly under <= that exact budget, slack reported. Controls: Q-only, Q/real-only, uniform learned PHM4, variable n and a same-catalogue weight-only-error selector. Infeasible policy recorded as skipped, never silently increases budget. All candidate fitting/selection uses500 fit images; separate500 calibration check and1000 validation do not tune choices. Same one-shot catalogue for all controls.

Prespecified BN-folded producer magnitude ranks retained Q units, nested kept sets for widths. Both direct-Q and dense output reconstruction use common dense-kernel Frobenius ridge; Q coefficient ridge scaled4 to match expanded-kernel norm. PHM candidates use Kronecker SVD of the output-fitted dense oracle, then ACTUAL float32 activation-output fit MSE selects structures. SVD itself minimizes weight error only; weighted ALS not implemented. One hard type/n per layer, no soft ensemble. Basis parameters remain stored/trainable and count in deployment. rank r=n only in this first catalogue; n4 learned PHM distinct from fixed Hamilton. No new solver/theorem/novelty claim.

Local independent candidate errors do not guarantee global classification ordering; earlier replacements alter later inputs. No validation/check-based type tuning. Infeasible policies remain explicit. <=common budget is not exact parameter equality; report slack and actual counts. Generic SVD/DP/ridge are established; no firstness claim. Layerwise PHM factors remain deployed and counted. Same3ep CE repairs; selected final epoch, no test. Epoch checkpoints and STOP support existing checked resume semantics.
