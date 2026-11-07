# Physical compression protocol

Prespecified on 2026-11-02 during the first new-seed training, before observing
any multi-channel removal outcomes. Config: `configs/cnn_compression.json`.

## Hypotheses

Existing positive-coordinate centered whole-quaternion cosine may identify
removals that preserve more quality than raw incoming magnitude or random
removal. No new method or success is presumed. H2's comparison against a
scalar-derived selection policy remains outside this three-policy pilot.

## Gate and fixed choices

Use block 3 selected by seed-2026 diagnostics. Confirm the same descriptive
gate (>=2 individually low-impact removals, image-half pair stability >=0.9)
at that block for seeds 2027 and 2028. If it fails, report it and do not force
a compression-success conclusion. No changing blocks using new test results.

Three decision policies only:

1. Random permutation of whole units, 10 repetitions per training seed;
   RNG seed = 100000 + training_seed*100 + repetition.
2. Increasing raw incoming quaternion kernel L2 norm, index tie-break.
   This ignores BN scaling and is explicitly a simple reference baseline,
   not a reproduction of targeted regularization (Vecchi2020, Singh2026).
3. Greedy removal from the most positively similar remaining pair using
   centered four-component calibration cosine; remove its smaller raw-norm
   member, index tie-break. No re-fitting, absolute cosine, weight compensation
   or score tuning. General similarity-pruning context: Ayinde2019, not a
   faithful reproduction of that full method or a claimed novel algorithm.

Use 0%, 25% and 50% removal. Reuse the same removal ordering across the two
nonzero sizes, including random repetitions. No fine-tuning or re-calibration
of BatchNorm. Physical surgery slices the second convolution's output units,
all four BN component channels and the readout's matching input columns.
First convolution of the block stays at its original width.

## Verification and evaluation

Require exact zero-removal equivalence; compare each nonzero compact model
against full-model output masking on a validation probe (numerical tolerance
1e-4). Check source model remains unchanged. Save retained original unit
indices as a reconstruction manifest; serialize compact state identically
for every policy and size. Count actual real trainable parameters, BN buffer
elements and state checkpoint bytes. No FLOP/latency claims without benchmarks.

Complete every manifest and validation measurement before new test inference.
Validation useful-compression threshold: <=2 percentage-point accuracy loss
and >=10% reduction in total real trainable parameters. This is a practical
pilot threshold, not a universal scientific criterion. Report all curves even
if they fail. Evaluate all frozen policies/sizes on test once; no winner or
size is chosen from test results. Random SD measures removal variability
within a training seed; training-seed SD is separate. Use three paired seeds,
preserve weak/negative outcomes and avoid statistical significance claims.

The official test set was used in earlier pilots. This is a small exploratory
study, not a new blind benchmark or a proof of architectural superiority.

Numerical correction before the three-seed analysis/compression run: inclusive
thresholds use tolerance 1e-12 to include exact 0.5-pp / 2-pp boundary drops
despite floating-point subtraction. The old seed-2026 count becomes 26/32
instead of 24/32. Threshold values, numerical observations, policy rankings,
selected block and architecture choices are unchanged; retain old artifacts.
