# Learned-grouping novelty audit

Status: preliminary literature audit plus an implemented feasibility pilot;
no first-in-literature or publication-readiness claim. Review date 2026-11-02.
The scientific scope remains activation redundancy and physical compression.

Membership adaptation permutes scalar channels before a constrained
quaternion convolution. Whole-unit permutations alone merely relabel units;
a permutation only before an unrestricted real readout can be absorbed into
its columns. Classification training alone does not require greater
compressibility. The original protocol remains in
`results/learned_grouping_pilot/PROTOCOL.md`; completed stage descriptions
and outcomes are in their result reports and configs.

## Direct assessment

Learning channel groups, differentiable permutations, learned channel shuffle,
and using learned shuffle in structured compression all have prior art.
A Sinkhorn router inserted into a quaternion CNN is a combination of known
mechanisms. The word "quaternion" alone does not establish enough novelty.
Current evidence does not justify calling this a novel compression method.

## Closest primary-source context

| Canonical key | Established mechanism | Consequence for this project |
| --- | --- | --- |
| Mena2018 | Differentiable relaxation of latent permutations | Sinkhorn assignment is not a new learning mechanism |
| Wang2019FLGC | Learned input/filter group membership | Learning which channels belong together is established |
| Zhang2021StructuredShuffle | Joint structured sparsification and learned channel shuffle | Combining adaptive channel order and compression is established |
| Gong2023DynamicShuffle | Input-dependent channel permutations and static/dynamic mixing | Adaptive channel shuffling and mixing have close prior art |
| ZhangEtAl2021, Grassucci2024 | Parameterized hypercomplex weight sharing | Learning structural parameterizations is established background |
| Vecchi2020, Singh2026 | Quaternion structural compression | Quaternion pruning is not a novelty claim |

The 2021 sparsification paper learns permutations via alternating optimization
of a weight-norm objective and structured regularization. The 2023 shuffle
paper uses input-dependent routing. Our pilot instead uses one global
assignment in a fixed Hamilton backbone. These differences describe the
implementation; they do not by themselves establish a publishable gap.
Primary abstract/method passages were inspected, not every cited work
reproduced. Existing methods' published gain claims are not results here.

## Potential contribution to test, not a finding

A controlled empirical question could be: does adapting membership in a fixed
weight-sharing pattern change activation redundancy and retained quality in
physically compact models? A useful finding would distinguish effects of
soft mixing, hard exported assignment, backbone quality and actual size.
Even a negative result needs sufficient controls and repeatability; a single
small CNN/seed is a feasibility pilot, not enough evidence for a broad claim.

Minimum evidence before a paper-strength claim includes replicated seeds,
same final-size comparisons, an account of extra training variables/compute,
quality before/after hardening, fixed-random controls, and evidence that any
gain is not explained by added soft mixing. Accuracy improvement by itself
does not show improved compressibility. A positive pilot would justify a
stronger mixing control and further validation, not an immediate novelty claim.

## Current implementation and prespecified boundaries

`src/models/channel_routing.py` implements deterministic Sinkhorn relaxation,
an exact one-to-one assignment solver and export to fixed channel indices.
It does not implement new multiplication rules or stochastic Gumbel sampling.
The GPU integrity check passes; the completed one-seed result is recorded
below. Its experiment and protocol are in `results/learned_grouping_pilot/`.
Conditional compression choices were
recorded before its outcomes in configs/learned_grouping_compression.json.

Measure actual membership as well as changed positions: a permutation of
whole units can merely rename groups, and component-position changes can
leave unordered group membership unchanged. Neither should be presented as
proof that useful new memberships were learned.

## Search scope and uncertainty

Queries included learned quaternion grouping, quaternion learnable channel
shuffle, hypercomplex learnable permutation, and model-agnostic structured
sparsification with learnable shuffle. Primary records and selected method
passages were used. This was not an exhaustive systematic review; lack of an
exact quaternion/Sinkhorn match in these searches is not evidence of priority.
No claim of "first" is supported. Conference acceptance cannot be predicted
from the idea alone; a clear, validated empirical insight is still required.

## Completed pilot, reviewed 2026-11-03

The learned hard model reached 77.61% test accuracy, versus 78.14% for fixed
random grouping and 79.00% for the reused original identity reference. Two
scalar positions and two unordered groups changed. Soft-to-hard validation
accuracy dropped from 79.20% to 56.00%; subsequent hard training recovered
part of the loss but did not pass the validation gate. Conditional compression
was therefore skipped under the frozen protocol.

This is a concrete negative feasibility result for one deterministic soft
relaxation, initialization and hardening schedule. It does not demonstrate a
new effective compression method, settle the broader structural-constraint
hypothesis, or rule out all training approaches to learned assignment. The
soft matrix's diffuseness, downstream BN adaptation and late learning-rate
schedule are not separately controlled in this pilot. Gate D remains open;
the current implementation/result is not sufficient grounds to claim novelty
or promise a publication.

## Follow-up BN intervention, completed 2026-11-03

The original epoch-40 numerical trace and boundary measurements were
reproduced exactly; full boundary states are now retained separately.
At fixed weights, training-only downstream-BN refitting changed hard
validation accuracy from 56.00% to 66.30%. The paired soft control changed
from 79.20% to 79.00%, leaving a 12.70-pp gap. Calibration agrees in direction
(hard 55.20% to 66.30%; refitted soft 80.30%). No official test inference.

BN-buffer refitting helps but does not resolve the quality loss under this
estimator. This narrows the interpretation of the pilot without establishing
a new grouping/compression method or a uniquely identified causal share of
the loss. See `results/routing_bn_diagnostic/REPORT.md`. Controlled
discrete-forward training is a possible next experiment, not a demonstrated
gain or a novelty claim. The original failed compression gate is preserved.

## Discrete-forward follow-up, protocol frozen 2026-11-03

The next one-seed pilot uses exact hard matching during every forward and a
Sinkhorn straight-through surrogate for routing logits. It reuses the paired
fixed-random 50-epoch control and applies the same final BN refit to both.
Protocol: `results/hard_forward_grouping/REPORT.md` and
`configs/hard_forward_grouping.json`. Completed results are recorded below.

Straight-through estimators are established (Bengio2013STE), including direct
permutation-learning precedent (Wang2023PSTE). Both entries were added to the
canonical bibliography after inspecting the primary records/abstracts.
Avoiding soft-to-hard mismatch by using hard forwards is a design property,
not a novel or sufficient scientific finding. Actual learned membership and
held-out quality must be checked separately.

## Completed discrete-forward outcome, 2026-11-03

The new 50-epoch seed-2026 training selected epoch 46. Final exported
membership changes 6/32 positions and creates 4/8 new unordered groups.
Freeze has zero immediate logit shift. After the same BN refit, learned
versus fixed-random accuracy is 77.80% vs 78.10% on validation and 77.30%
vs 77.20% on calibration. CE is worse for learned grouping on both splits,
before and after refitting. No new official-test inference or compression.

This demonstrates the intended mechanism but not a useful quality advantage.
The +0.10-pp primary calibration difference is one image out of 1,000 and
does not pass the prespecified +0.50-pp replication threshold. Other gate
conditions pass. Do not claim equivalence, novelty or improved compression,
and do not automatically expand this configuration into a larger study.
One seed/surrogate/schedule cannot rule out all grouping approaches.
Report and 131-check verification: `results/hard_forward_grouping/`.
