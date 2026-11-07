# CIFAR ResNet20 baseline quality gate

Status: completed. One seed2026 per family; sequential200ep SGD.

## Hypothesis

A stronger residual backbone may reach approximately90percent validation accuracy and support meaningful Q/real compression validation. This is a target, not a measured guarantee or novelty claim.

## Measured results

| Family | Real parameters | Selected epoch | Selected validation (%) | Final validation (%) |
|---|---:|---:|---:|---:|
| real | 269722 | 100 | 90.90 | 91.80 |
| quaternion | 271594 | 101 | 91.00 | 91.10 |

## Interpretation and limits

SGD momentum.9, decay1e-4, crop4/flip, scalar BN, split ReLU, no dropout, real10-class readout, option-A zero-pad/downsample shortcuts. Adapted CIFAR ResNet20 recipe (He2016ResNet), NOT an exact reproduction: existing48000/1000/1000 train/validation/calibration split and RGB normalization,200epoch schedule, common5epoch warmup and AMP. Q input(0,R,G,B) has the same RGB information. Stage Q units8/16/32 give scalar widths32/64/128 versus real16/32/64: close real parameter budgets, NOT matched width/FLOPs/starting quality. Count actual weights/buffers. Same seeded batch/augmentation streams. Four Q components must pass discarded-copy update audit before training.

Minimum validation cross-entropy over the fixed200 epochs; selected and final validation both reported. No test inference.

First ONE seed per family, sequential GPU baselines only. Roughly90percent is a target, not a promise for either model. No architecture/LR search, failed-pilot replication, CIFAR100 download, test inference or automatic compression/extra seeds. Quality/cost gate before implementing residual internal-channel pruning. Negative results retained.

Atomic completed-epoch checkpoint with model/SGD/AMP, best model, Python/NumPy/torch/CUDA and dedicated batch/augmentation RNG. STOP checked after batches; interrupted partial epoch restarts from last completed epoch. Resume both models sequentially, reuse completed model after hash verification.

Actual counts distinguish real trainables, Q units and BN buffers. Approximate parameter matching is not equal width/compute or an isolated quaternion-algebra experiment. No pruning or compressed-model savings have been measured in this baseline stage. Same train/validation/calibration split reused; no calibration/test inference, validation selection disclosed. He2016ResNet original real ResNet20 test91.25percent is background, not our outcome. Literature: references.bib. GPU preflight uses synthetic discarded models and is not scientific evidence.
