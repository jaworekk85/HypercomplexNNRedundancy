# Whole-quaternion physical compression pilot

## Hypotheses

A conventional activation-similarity policy may preserve quality better than raw incoming magnitude and random removal. No new algorithm, general superiority or success is assumed.

## Fixed protocol

Block 3 was selected on seed-2026 calibration/validation before new training outcomes. Three seeds; 0/25/50% whole-unit removal; no fine-tuning, BN re-calibration or weight compensation. Scores use calibration; all cases were frozen and validated before final test. The official test set was previously used in the pilots. See PROTOCOL.md and protocol.json for exact choices and hashes.
Physical removal slices the second convolution output, matching BN channels and readout inputs. The block's first convolution remains unchanged. Unit-removal percentage therefore differs from total-model parameter savings. State files use the same serialization envelope for every case; manifests are separate. No measured latency or FLOP savings are claimed.

## Measured results

| Removal | Policy | Real trainable parameters | State bytes | Test accuracy mean +/- training-seed SD | Validation usefulness passes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0% | baseline | 73898 | 308368 | 78.99 +/- 0.22% | 0/3 |
| 25% | random | 64298 | 269712 | 77.50 +/- 0.34% | 22/30 |
| 25% | incoming_norm | 64298 | 269712 | 76.70 +/- 0.90% | 3/3 |
| 25% | activation_similarity | 64298 | 269712 | 77.48 +/- 0.29% | 2/3 |
| 50% | random | 54698 | 231056 | 74.08 +/- 0.32% | 3/30 |
| 50% | incoming_norm | 54698 | 231056 | 72.25 +/- 3.16% | 0/3 |
| 50% | activation_similarity | 54698 | 231056 | 74.12 +/- 1.31% | 1/3 |

Error bars: sample SD across three training-seed means; random first averages ten removal repetitions within each seed. Random repetitions are not 30 independent training seeds. All individual cases and within-seed random SD are retained in cases.csv / aggregated.json.
Validation usefulness was prespecified as <=2 pp accuracy drop with >=10% total real-parameter reduction. Every policy and size is reported regardless of this threshold. These practical thresholds are not evidence of novelty.

## Interpretation

Single-unit dispensability need not survive simultaneous removal. Weak or negative similarity results and worse-than-random policies are valid outcomes. Raw kernel norms ignore BN scaling; this limits conclusions against stronger weight-based methods. The similarity policy has a norm tie/member rule and is not a faithful full reproduction of Ayinde2019. Weight-baseline context: Vecchi2020 and Singh2026 (references.bib).
H2's direct grouped-vs-scalar policy comparison, fine-tuning, matched pre-pruning quality, broader architectures/datasets, compute and latency remain untested. Three seeds in one small architecture support only a limited empirical conclusion.

## Measured interpretation and decision

At 25% block-unit removal (12.99% total-parameter saving), similarity gives 77.48% test accuracy, a 1.51-pp drop. Its mean paired difference from random is -0.02 pp; the three seed effects do not have a consistent sign.

At 50% block-unit removal (25.98% total-parameter saving), similarity gives 74.12% test accuracy, a 4.87-pp drop. Its mean paired difference from random is +0.04 pp; the three seed effects do not have a consistent sign.

The activation-similarity policy did not demonstrate an advantage over random removal. Mean gains against raw kernel norm are not consistent across seeds and do not establish superiority over a strong BN-aware weight baseline. Physical compaction works; Gate C's policy value and Gate D remain unconfirmed. The next small question is the direct grouped-vs-scalar diagnostic (H2), using existing calibration features and validation ablations under a separately fixed comparison.
