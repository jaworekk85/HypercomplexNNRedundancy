# Compression of an existing QNN family: evidence and limits

## Hypotheses

Characterize when Hamilton-structure-preserving compensation retains useful quality, its actual compact size and failure regimes. No novelty or intrinsic quaternion superiority is assumed. A separate matched circulant control tests one alternative weight-sharing pattern; pseudo-label repair is supplementary.

## Measured results

This report reads completed studies only and performs no new training or inference. Three seeds per dataset. Uniform removal at every convolution output; physical narrower networks. Random trials are averaged within seed before aggregation. Refinement points were selected after coarse outcomes and retain that follow-up status.

| Dataset | Removed/site | Real params | Parameter saving | Comp before | Comp +3 ep | BN +3 ep | Scratch 50 ep | Real comp +3 ep |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar | 0.125 | 56850 | 23.07% | 72.20% | 75.43% | 74.97% | pending | 73.73% |
| cifar | 0.25 | 42034 | 43.12% | 59.67% | 71.50% | 69.27% | 71.57% | 70.40% |
| cifar | 0.375 | 29450 | 60.15% | 34.03% | 63.53% | 60.90% | 68.33% | 66.00% |
| cifar | 0.5 | 19098 | 74.16% | 21.97% | 53.57% | 48.23% | 63.90% | 56.70% |
| cifar | 0.625 | 10978 | 85.14% | 12.47% | 36.00% | 32.20% | 58.23% | 41.40% |
| cifar | 0.75 | 5090 | 93.11% | 10.03% | 18.50% | 21.77% | 46.17% | 24.27% |
| cifar | 0.875 | 1434 | 98.06% | 10.17% | 13.57% | 13.60% | 31.93% | 15.40% |
| eurosat | 0.125 | 56850 | 23.07% | 92.83% | 95.30% | 95.03% | pending | 95.50% |
| eurosat | 0.25 | 42034 | 43.12% | 78.13% | 93.73% | 91.80% | 95.50% | 94.10% |
| eurosat | 0.375 | 29450 | 60.15% | 57.10% | 90.47% | 86.17% | 94.33% | 91.63% |
| eurosat | 0.5 | 19098 | 74.16% | 32.20% | 85.33% | 79.17% | 92.47% | 85.23% |
| eurosat | 0.625 | 10978 | 85.14% | 25.43% | 72.80% | 61.30% | 90.33% | 74.13% |
| eurosat | 0.75 | 5090 | 93.11% | 18.43% | 50.37% | 44.07% | 85.27% | 46.73% |
| eurosat | 0.875 | 1434 | 98.06% | 12.40% | 18.73% | 18.67% | 66.47% | 22.23% |

Paired compensation minus BN after equal three-epoch repair (validation pp; unfavorable seeds retained):

- cifar, fraction 0.125: +2.10, +0.30, -1.00; mean +0.47 pp.
- cifar, fraction 0.25: +2.20, +3.60, +0.90; mean +2.23 pp.
- cifar, fraction 0.5: +6.10, +5.10, +4.80; mean +5.33 pp.
- cifar, fraction 0.75: -0.20, -3.10, -6.50; mean -3.27 pp.
- cifar, fraction 0.875: +2.50, -2.70, +0.10; mean -0.03 pp.
- cifar, fraction 0.375: +5.90, +1.30, +0.70; mean +2.63 pp.
- cifar, fraction 0.625: +5.30, +4.50, +1.60; mean +3.80 pp.
- eurosat, fraction 0.125: +0.60, +0.10, +0.10; mean +0.27 pp.
- eurosat, fraction 0.25: +2.30, +2.70, +0.80; mean +1.93 pp.
- eurosat, fraction 0.5: +6.90, +5.20, +6.40; mean +6.17 pp.
- eurosat, fraction 0.75: +9.30, -2.70, +12.30; mean +6.30 pp.
- eurosat, fraction 0.875: -4.10, +4.00, +0.30; mean +0.07 pp.
- eurosat, fraction 0.375: +4.30, +2.30, +6.30; mean +4.30 pp.
- eurosat, fraction 0.625: +8.50, +13.60, +12.40; mean +11.50 pp.

Earlier single-site comparison: compact structure-preserving QNN versus unrestricted functional substitution, without repair:

| Dataset | Removed at block-2 output | Compact Q accuracy | Functional unrestricted accuracy | Mean Q minus unrestricted (pp) |
| --- | ---: | ---: | ---: | ---: |
| cifar | 4/16 | 76.63% | 77.07% | -0.43 |
| cifar | 8/16 | 73.17% | 74.33% | -1.17 |
| eurosat | 4/16 | 96.17% | 96.17% | +0.00 |
| eurosat | 8/16 | 94.33% | 94.77% | -0.43 |

The unrestricted functional reference retains extra computation and is NOT a compact QNN at the same budget. It is a representation diagnostic; this table is a different experiment from all-site compression above.

## Interpretation and limits

Half the units per site is about three quarters of parameters removed, not half the parameters. Parameter sharing and actual compact architecture savings are distinct; a mask alone is not compression. Stored checkpoint bytes depend on serialization and are not latency or MAC measurements.

Scratch models get 50 student-training epochs; repair gets three in addition to a previously trained teacher. Their comparison describes this resource allocation, not equal-compute superiority. Fully trained Q and real baselines may differ in quality; compare their own-baseline losses as well as absolute scores.

Directly constrained fitting versus full fit followed by projection compares fitting strategies at the same compact architecture. It is not the cost relative to an unconstrained functional substitution oracle; such an oracle is not a compact QNN. Report the earlier block-2 unrestricted substitution study separately (results/multisource_compensation/REPORT.md, results/eurosat_compression/REPORT.md). Prior context: GRAIL2026, Kim2020, Asiaee2026CMR; the four-coefficient restriction is not a new algorithm.

The strong-compression failure regimes and any negative paired effects above are evidence, not runs to discard. A benefit over pruning does not establish useful absolute accuracy, a universal advantage after repair, or a general advantage of quaternions. Three seeds and reused validation data constrain generalization claims. CIFAR test was used in early pilots; no new test inference here. EuroSAT test remains held out.

Pending/incomplete curves are omitted rather than treated as findings: none

Original plots/reports: results/compression_curve and results/compression_curve_eurosat. New structural control: configs/structure_control.json. Exact per-seed effects and source hashes: summary.json.
