# QNN compression and one matched structured control

## Hypotheses

Primary: characterize structure-preserving compression of an existing QNN family, including quality/size, repair and failure regimes. Secondary: does one established non-quaternion sharing pattern show comparable behavior? No new algorithm, algebra or quaternion superiority is claimed.

Circulant 4x4 blocks (Ding2017CirCNN context) have four trainable coefficients, like Hamilton blocks. Widths, input information (0,R,G,B), coefficient initialization, BN/ReLU/dropout and parameter counts match exactly. Circulant groups are not quaternion units. Both structured spaces permit exact linear folding; our expanded Conv2d does not implement FFT acceleration or reproduce CirCNN. Selected after Q/real outcomes; exploratory follow-up.

CIFAR CNN6 only, 3 seeds, 50-epoch baselines selected by validation CE; f=.25/.5 at all six sites. Identical 3-epoch repair and same-victim controls. Reuse original Q/real outcomes and add the previously missing Q same-victim pruning repair. No new official-test inference, latency claim, Circulant scratch or EuroSAT sweep.

## Measured results

State: completed. Scientific interpretation requires completed scientific runs; CPU smoke values are not results.

| Family | Fraction removed | Policy | Stage | Actual real params | Validation mean | Seed SD (pp) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| circulant_cnn6 | 0.0 | baseline | after | 73898 | 78.73% | 0.58 |
| circulant_cnn6 | 0.0 | baseline | before | 73898 | 78.67% | 0.40 |
| circulant_cnn6 | 0.25 | bn_pruned | after | 42034 | 66.33% | 2.52 |
| circulant_cnn6 | 0.25 | bn_pruned | before | 42034 | 17.53% | 6.01 |
| circulant_cnn6 | 0.25 | greedy_comp | after | 42034 | 72.20% | 1.41 |
| circulant_cnn6 | 0.25 | greedy_comp | before | 42034 | 57.23% | 0.35 |
| circulant_cnn6 | 0.25 | greedy_pruned | after | 42034 | 70.43% | 1.92 |
| circulant_cnn6 | 0.25 | greedy_pruned | before | 42034 | 28.77% | 11.57 |
| circulant_cnn6 | 0.5 | bn_pruned | after | 19098 | 46.03% | 1.20 |
| circulant_cnn6 | 0.5 | bn_pruned | before | 19098 | 11.53% | 2.32 |
| circulant_cnn6 | 0.5 | greedy_comp | after | 19098 | 52.07% | 4.55 |
| circulant_cnn6 | 0.5 | greedy_comp | before | 19098 | 19.53% | 1.40 |
| circulant_cnn6 | 0.5 | greedy_pruned | after | 19098 | 48.07% | 4.59 |
| circulant_cnn6 | 0.5 | greedy_pruned | before | 19098 | 12.10% | 3.64 |
| cnn6 | 0.0 | baseline | after | 73898 | 77.93% | 0.93 |
| cnn6 | 0.0 | baseline | before | 73898 | 77.93% | 1.10 |
| cnn6 | 0.25 | bn_pruned | after | 42034 | 69.27% | 0.15 |
| cnn6 | 0.25 | bn_pruned | before | 42034 | 29.50% | 6.32 |
| cnn6 | 0.25 | greedy_comp | after | 42034 | 71.50% | 1.20 |
| cnn6 | 0.25 | greedy_comp | before | 42034 | 59.67% | 3.21 |
| cnn6 | 0.25 | greedy_pruned | after | 42034 | 70.00% | 0.36 |
| cnn6 | 0.25 | greedy_pruned | before | 42034 | 23.47% | 2.75 |
| cnn6 | 0.5 | bn_pruned | after | 19098 | 48.23% | 0.51 |
| cnn6 | 0.5 | bn_pruned | before | 19098 | 11.33% | 0.55 |
| cnn6 | 0.5 | greedy_comp | after | 19098 | 53.57% | 0.65 |
| cnn6 | 0.5 | greedy_comp | before | 19098 | 21.97% | 4.12 |
| cnn6 | 0.5 | greedy_pruned | after | 19098 | 49.67% | 2.14 |
| cnn6 | 0.5 | greedy_pruned | before | 19098 | 13.43% | 1.55 |
| real_cnn6 | 0.0 | baseline | after | 72954 | 75.63% | 0.80 |
| real_cnn6 | 0.0 | baseline | before | 72954 | 75.53% | 1.27 |
| real_cnn6 | 0.25 | bn_pruned | after | 41326 | 66.37% | 1.91 |
| real_cnn6 | 0.25 | bn_pruned | before | 41326 | 23.13% | 9.28 |
| real_cnn6 | 0.25 | greedy_comp | after | 41326 | 70.40% | 1.31 |
| real_cnn6 | 0.25 | greedy_comp | before | 41326 | 62.03% | 1.99 |
| real_cnn6 | 0.25 | greedy_pruned | before | 41326 | 34.27% | 10.05 |
| real_cnn6 | 0.5 | bn_pruned | after | 18626 | 39.33% | 2.26 |
| real_cnn6 | 0.5 | bn_pruned | before | 18626 | 11.40% | 2.09 |
| real_cnn6 | 0.5 | greedy_comp | after | 18626 | 56.70% | 0.26 |
| real_cnn6 | 0.5 | greedy_comp | before | 18626 | 32.03% | 2.54 |
| real_cnn6 | 0.5 | greedy_pruned | before | 18626 | 15.30% | 3.95 |

## Interpretation limits

Report absolute accuracy/CE, own-baseline losses and every paired seed, including unfavorable outcomes. Baseline quality may differ despite identical parameter budgets; own-baseline normalization does not eliminate this confound. This single contrast cannot isolate one quaternion property or prove a generic structure-only explanation. No hyperparameter retuning or dropping a weak control after inspecting results.

The original paper evidence includes Q-vs-real quality-size curves on CIFAR/EuroSAT and matched compact scratch controls. Those 50-epoch scratch runs have a different student-training budget from 3-epoch repairs. Unrestricted functional substitution is not a compact quaternion architecture. Fit-then-project versus directly constrained fitting is a different contrast from unrestricted substitution.

Full per-seed rows, own-baseline drops, baseline gaps, paired compensation effects and exact-fold diagnostics are in summary.json. A negative result is retained. Pseudo-label repair remains a separate supplementary question.
