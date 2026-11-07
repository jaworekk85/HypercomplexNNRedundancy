# BatchNorm-aware magnitude follow-up

## Hypothesis

Conventional BN-folded weight magnitude may be a more meaningful reference than raw quaternion kernel norm. This is one fixed follow-up, not a novel method or a guaranteed stronger result.

## Fixed protocol

Same three selected models, block 3 and physical 25/50% removal as before. No training, BN re-calibration, compensation or new policy/size selection. All six new manifests and validation evaluations were frozen before test. Existing reference results are reused unchanged. The test set was already used in preceding pilots; this is exploratory.
Score: joint Frobenius norm of all four output rows in the expanded Hamilton convolution after per-row gamma/sqrt(running_var+eps) scaling. CPU float64 ranking, lowest score removed first, original-index tie break. Folded bias and downstream effects are excluded. Folding is used to score only: component-wise BN generally does not preserve the Hamilton kernel structure. Source: PyTorchFusion260; quaternion pruning context: Vecchi2020, Singh2026 (references.bib).

## Measured results

| Removed block units | Policy | Retained real parameters | State bytes | Validation mean | Test mean +/- training-seed SD |
| ---: | --- | ---: | ---: | ---: | ---: |
| 25% | random | 64298 | 269712 | 76.37% | 77.50 +/- 0.34% |
| 25% | incoming_norm | 64298 | 269712 | 76.63% | 76.70 +/- 0.90% |
| 25% | activation_similarity | 64298 | 269712 | 76.37% | 77.48 +/- 0.29% |
| 25% | bn_folded_group_l2 | 64298 | 269712 | 75.67% | 76.10 +/- 0.41% |
| 50% | random | 54698 | 231056 | 73.09% | 74.08 +/- 0.32% |
| 50% | incoming_norm | 54698 | 231056 | 71.80% | 72.25 +/- 3.16% |
| 50% | activation_similarity | 54698 | 231056 | 73.80% | 74.12 +/- 1.31% |
| 50% | bn_folded_group_l2 | 54698 | 231056 | 73.43% | 73.30 +/- 0.74% |

Random is averaged across ten removals within each training seed before calculating training-seed SD. Three seeds are not enough for a broad superiority/significance claim.

Paired BN-aware minus reference test accuracy (pp; seeds 2026, 2027, 2028):

- 25% vs incoming_norm: -0.46, -0.09, -1.24; mean -0.60, sample SD 0.59.
- 25% vs random: -0.89, -1.70, -1.61; mean -1.40, sample SD 0.44.
- 25% vs activation_similarity: -1.31, -1.96, -0.87; mean -1.38, sample SD 0.55.
- 50% vs incoming_norm: +2.56, +2.49, -1.89; mean +1.05, sample SD 2.55.
- 50% vs random: -0.30, -1.72, -0.31; mean -0.78, sample SD 0.82.
- 50% vs activation_similarity: +0.30, -3.08, +0.34; mean -0.81, sample SD 1.96.

## Interpretation

Changing BN scaling can change magnitude rankings; usefulness must be judged by held-out performance rather than the score's motivation. Better, unchanged and worse results are all retained. The baseline ignores folded offsets, ReLU and outgoing weights, so even an exact folded-kernel norm is not full functional importance. No score variant was selected to fix a negative result.
Actual compact parameter counts match the frozen controls: 64,298 and 54,698 (12.99% and 25.98% total-model savings). A 25/50% block-unit removal fraction is not the whole-model saving. No new MAC or latency measurement is claimed.

## Verified measured interpretation

At 25% block-unit removal, BN-aware magnitude reaches 76.10% versus random 77.50% (mean paired BN-minus-random effect -1.40 pp). BN-aware is worse in each of the three training seeds at this size.

At 50% block-unit removal, BN-aware magnitude reaches 73.30% versus random 74.08% (mean paired BN-minus-random effect -0.78 pp). BN-aware is worse in each of the three training seeds at this size.

At 25% removal the BN-aware reference is also worse than raw magnitude in all three seeds. At 50% it improves the mean over raw magnitude, but reverses in seed 2028. Accounting exactly for BN's multiplicative scaling did not produce a reliably better importance criterion.

All 25 implementation tests and 130 BN artifact checks passed, including independent official folding, exact original-weight slicing, policy reconstruction and recomputed test metrics. Existing numerical sources and all original comparison artifacts were preserved.

This follow-up does not show that BN never matters. It shows this particular folded-weight norm remains a weak predictor in the fixed pilot. No bias-inclusive or alternative score was tried after seeing the outcome. The activation policy still has no demonstrated advantage over random. Next remains the prespecified grouped-versus-scalar diagnostic (H2), rather than tuning the weight criterion to these test results.
