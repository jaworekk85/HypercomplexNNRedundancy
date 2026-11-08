# Quaternion input/stem with learned PHM ResNet-20

Status: paused.

## Hypothesis

A from-scratch learned-PHM backbone can support a later compression study comparing preservation versus readaptation of learned bases. Quality is unmeasured until training completes.

## Measured results

| Model | Real parameters | Selected epoch | Selected validation (%) | Final validation (%) |
|---|---:|---:|---:|---:|

## Interpretation and limits

From-scratch random initialization. RGB is encoded as (0,R,G,B); stem uses fixed Hamilton Q multiplication. All18 residual convolutions are PHM with n=rank=4 and independent learned basis/core factors, no copied pretrained Q weights or fixed Hamilton bases. Scalar widths32/64/128, ordinary BN/ReLU, existing option-A shortcut layout and real10class readout. Random cores rescaled by sqrt(n) for expected effective-kernel He variance. Same200ep SGD recipe/split/augmentation/checkpoint selection as the existing baselines.272746 real trainables vs Q271594; not exact parameter equality.

First ONE seed2026 baseline only. User keeps quaternion input/stem and permits learned PHM structure deeper. Fixed n4 is a bounded first step; this does not learn discrete n or guarantee algebra axioms. No test, hyperparameter sweep, automatic compression or fixed-vs-relearned-basis followup yet. Report failures and actual updates. Q and real prior baselines are controls, not initialization sources.

Minimum validation cross-entropy over the fixed200 epochs; selected and final validation both reported. No test inference.

Existing verified full-epoch model/SGD/scaler/best/RNG checkpoint. STOP in output pauses after a batch; incomplete epoch is repeated. Restart the same command after removing STOP.

One Hamilton stem,18learned PHM bases, fixed n4, one seed. Discrete n is not trained. Learned bases are not guaranteed to satisfy algebra axioms. Actual model size differs from previous baselines by1152basis coefficients. No compression or test result here. Prior art: He2016ResNet, ZhangEtAl2021, Grassucci2024. Citation database: references.bib.
