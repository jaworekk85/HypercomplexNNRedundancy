# BatchNorm-aware magnitude follow-up

Prespecified after the user authorized the BN check and before computing the
new ranking/outcomes. This is one additional fixed weight reference, not a
new method or a rerun/tuning of the original three-policy pilot.

## Hypothesis

Raw convolution norms can be misleading after channel-wise BN. A conventional
folded-weight magnitude reference may predict removable whole quaternion
units better, or may fail. Its quality relative to raw norms, random removal
and the frozen activation policy is an empirical question.

## Fixed criterion

For output component c of unit u, use d[c,u] = gamma[c,u] /
sqrt(running_variance[c,u] + epsilon), from evaluation-mode BN. The expanded
Hamilton convolution has four scalar output rows per quaternion unit. Rank
by their joint L2 norm after multiplying each row by d[c,u]. Equivalently:

    score[u] = raw_quaternion_kernel_L2[u] * sqrt(sum_c d[c,u]^2)

Remove smallest scores first, ties by original index. Use CPU float64 for
the ranking. With all d=1, this equals twice the raw quaternion kernel norm
and preserves its ordering. The expanded real matrix is derived and is not
counted as additional trainable weights. Independent per-component scaling
generally breaks its Hamilton structure; folding is only a reference for
scoring/checking, not a replacement quaternion architecture.

Do not add BN beta, folded bias, outgoing weights, labels or activation scores
to the criterion. The folded bias and ReLU/downstream effects can matter;
this norm is not guaranteed functional importance. Standard evaluation-mode
Conv+BN folding follows PyTorchFusion260 in references.bib; quaternion
compression context remains Vecchi2020 and Singh2026. No faithful reproduction
of those complete methods or first-in-literature claim is made.

## Unchanged experimental setting

Use selected quaternion checkpoints for seeds 2026/2027/2028; block 3's second
convolution only; 25% and 50% whole-unit removal. No training, fine-tuning or
BN re-calibration. Physically slice the same convolution, BN and readout via
the tested builder. Existing reference cases/metrics are reused unchanged,
including ten random-removal repetitions within each training seed.

Verify formula against official PyTorch folding and explicit expanded-row
norms, preserve source state, check zero-removal equivalence and compact vs
mask equivalence, and test serialized reconstruction. Evaluate validation and
freeze all six manifests/architectures before new test inference. Do not
select one score variant, size or seed from the results. Test was already
seen in earlier pilots; this is exploratory, not a new blind benchmark.

Report every seed, mean and sample SD, paired effects and actual retained
real parameters/state bytes. Preserve negative results. Diagnostic Spearman
against existing validation individual-ablation loss harm is descriptive;
it does not select the policy. No FLOP, latency, novelty or general superiority
claims follow from this small follow-up.
