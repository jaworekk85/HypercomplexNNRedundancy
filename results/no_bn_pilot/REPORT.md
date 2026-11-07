# One-seed no-BN feasibility pilot

## Hypothesis

The existing six-convolution QNN may train without BN under the original settings. This is a controlled removal ablation, not an optimized normalization-free method.

## Measured result

Seed 2026; same train/validation/calibration split, 50 epochs, validation-CE checkpoint selection; no test inference.

| Model | Real parameters | Selected val accuracy (%) | Best epoch |
| --- | ---: | ---: | ---: |
| Original BN | 73898 | 78.30 | 49 |
| No BN | 73002 | 76.60 | 46 |

## Interpretation limits

One seed, no learning-rate/initialization search, 896 fewer affine parameters, bias-free convolutions retained. Remaining initialization/batch/augmentation settings unchanged. Poor results would not prove normalization is necessary. Good baseline quality would justify separately planning compression without BN; no such compression has been performed here. Prior context: Brock2021NFNets (not reproduced).
