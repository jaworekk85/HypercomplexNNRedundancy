# Frozen CNN protocol: three training seeds

## Hypotheses

This confirms training variability without tuning to test results. It does not establish a general quaternion advantage or novelty.

## Protocol

Reuse seed 2026; change only the training seed to 2027 and 2028. Fixed architecture, optimizer, 50 epochs, split, normalization and validation-loss checkpoint selection. Complete all four new trainings before evaluating the new selected models on test. The official test set was previously used in the pilots; this is not a new blind benchmark.

## Measured results

| Seed | Quaternion test accuracy | Real test accuracy | Paired difference (pp) |
| ---: | ---: | ---: | ---: |
| 2026 | 79.00% | 76.24% | +2.76 |
| 2027 | 78.77% | 75.83% | +2.94 |
| 2028 | 79.20% | 75.76% | +3.44 |

quaternion: mean 78.99%, sample SD 0.22 percentage points (n=3 training seeds).

real: mean 75.94%, sample SD 0.26 percentage points (n=3 training seeds).

Paired difference: +3.05 pp, sample SD 0.35 pp.

## Interpretation

Three seeds characterize variability under this one architecture/training protocol. Parameter budgets are close (73,898 vs 72,954) but scalar widths, compute and initial quality differ. No significance claim or broad superiority claim follows from three seeds. Effective rank and ablation diagnostics do not by themselves prove useful physical compression.
