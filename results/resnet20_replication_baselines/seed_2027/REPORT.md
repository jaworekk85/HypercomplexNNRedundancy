# CIFAR ResNet20 baseline quality gate

Status: running. One seed2027 per family; sequential200ep SGD.

## Hypothesis

A stronger residual backbone may reach approximately90percent validation accuracy and support meaningful Q/real compression validation. This is a target, not a measured guarantee or novelty claim.

## Measured results

| Family | Real parameters | Selected epoch | Selected validation (%) | Final validation (%) |
|---|---:|---:|---:|---:|
| real | 269722 | 102 | 91.50 | 91.40 |

Current family quaternion, completed epoch 54/200, current validation 84.10%; current epoch seconds 60.1.

## Interpretation and limits

SGD momentum.9, decay1e-4, crop4/flip, scalar BN, split ReLU, no dropout, real10-class readout, option-A zero-pad/downsample shortcuts. Adapted CIFAR ResNet20 recipe (He2016ResNet), NOT an exact reproduction: existing48000/1000/1000 train/validation/calibration split and RGB normalization,200epoch schedule, common5epoch warmup and AMP. Q input(0,R,G,B) has the same RGB information. Stage Q units8/16/32 give scalar widths32/64/128 versus real16/32/64: close real parameter budgets, NOT matched width/FLOPs/starting quality. Count actual weights/buffers. Same seeded batch/augmentation streams. Four Q components must pass discarded-copy update audit before training.

Minimum validation cross-entropy over the fixed200 epochs; selected and final validation both reported. No test inference.

Independent initialization and data-order/augmentation seeds for both real and quaternion CIFAR-10 ResNet-20 baselines, using the discovery-stage recipe and fixed split. Two seeds supplement discovery seed 2026; no test inference or hyperparameter selection on replication outcomes. These are baseline models, not yet a compression replication. The subsequent compression protocol must be registered before examining its outcomes.

Create STOP in the active seed output to pause after a batch; resume with the same command after removing it. Completion is checkpointed each epoch.

Actual counts distinguish real trainables, Q units and BN buffers. Approximate parameter matching is not equal width/compute or an isolated quaternion-algebra experiment. No pruning or compressed-model savings have been measured in this baseline stage. Same train/validation/calibration split reused; no calibration/test inference, validation selection disclosed. He2016ResNet original real ResNet20 test91.25percent is background, not our outcome. Literature: references.bib. GPU preflight uses synthetic discarded models and is not scientific evidence.
