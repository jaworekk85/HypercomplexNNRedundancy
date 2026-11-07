# One-boundary learned assignment pilot

Frozen before CIFAR training. Configuration: configs/learned_grouping.json.
Prototype synthetic/GPU gate passed 33 tests and 9 integrity checks.

## Hypotheses and scope

A learned fixed channel assignment may preserve quality better than a fixed
random assignment in the same constrained CNN. Whether it helps compression
is a separate, conditional question. One seed cannot establish a stable
advantage, novelty or generic structural-constraint causality.

## Fixed intervention

Use 32 scalar channels after block 1 and before block 2's first Hamilton
convolution. A deterministic log-domain Sinkhorn relaxation (30 iterations)
acts as a 32x32 mixing matrix during 40 training epochs. Initial logits use
strength 4 around the same seeded random permutation as the random control,
plus 0.01 Gaussian noise from a private generator. Temperature follows a
geometric schedule 1 to 0.25 over those epochs. No Gumbel sampling or new
multiplication rule is introduced. This is an application of Mena2018's
existing relaxation idea, not its full stochastic method.

After epoch 40, compute an exact maximum-weight one-to-one assignment from
the logits, freeze routing and train backbone weights for ten further epochs.
Record validation immediately before/after hardening, without intervening
optimization. Fixed controls receive the same 50-epoch backbone budget and
hard-checkpoint eligibility window (epochs 40-50). No temperature/LR variant
is selected from test results.

## Controls, data and accounting

Reuse the original seed-2026 identity checkpoint (selected epoch 49) only
after verifying exact full-model forward equivalence with identity routing.
Train fixed random and learned assignment from the same backbone initialization,
same dedicated sample-order/augmentation streams and seed 2026. Random
assignment uses a separate generator seeded 62026. New modes use the same
48,000 training images, 1000 validation and 1000 calibration examples and
training-only normalization as results/cifar_cnn. All expected quaternion
components and routing parameters must receive gradients/updates at the first
actual optimization batch. Test data never select weights or routing.

Backbone: 73,898 real parameters. Learned soft routing adds 1,024 training
variables (total 74,922) and extra mixing compute. These training budgets are
not parameter/compute matched. Route LR 0.01, no route weight decay; backbone
AdamW LR 0.001 and decay 0.0001, cosine to 1e-5 over 50 epochs. Same train
augmentation, mixed precision and FP32 first-update audit as the frozen pilot.

At export retain indices of a hard permutation, no learned dense matrix.
All exported modes have 73,898 trainable real weights plus routing buffers;
this is not a claim of compact-channel savings. Soft-model performance alone
cannot establish a learned fixed-grouping benefit or isolate added capacity.

## Selection, gate and integrity

Select minimum validation CE from hard checkpoints at epochs 40-50. Save full
epoch checkpoints with optimizer/scheduler/scaler and RNG state, including
routing temperature and hard/soft status. Verify exact serialized export and
preserve original numerical source files/artifacts.

Freeze selected new checkpoints and the compression decision before evaluating
their final test predictions. The official test set has already been used in
prior pilots. Report the identity reference as reused, not independently
retrained. The pilot is exploratory, not a new blind benchmark.

Compression may proceed only if the selected learned hard model is within
0.5 validation percentage points of the better fixed control and actually
changes at least one routing position. This is a prespecified practical gate,
not statistical evidence or a demand that compression must succeed. A failed
gate is retained and stops the compression extension for this configuration.

Closest-context keys include Mena2018, Wang2019FLGC, Vecchi2020, Singh2026;
the novelty audit adds further learned-shuffle/sparsification work separately.
No first-in-literature or paper-readiness claim is authorized by this pilot.
