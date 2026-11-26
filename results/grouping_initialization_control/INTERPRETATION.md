# What the initialization control resolves

Completed2026-11-25.504fixed-feature ridge fits,3source seeds,2sites and4crossed
feature/consumer states. All126trained/trained rows replay the predecessor;
coefficient banks reload exactly. No network training or classification test.

## Hypothesis

The penalty for random regrouping could be explained by architecture alone:
arbitrary scalar permutations generally change the Hamilton-constrained linear
map class. Therefore original groups can have an advantage even at step0.

## Measured results

Delta = normalized held-out MSE after full mixing minus normalized MSE for the
original groups. Normalized MSE is SSE divided by total squared target energy.
This is not classification error or an accuracy percentage. Each source-seed
penalty averages10fixed random assignments; the table averages3source seeds.

| Block | Mixing policy | Initial delta | Trained delta | Trained minus initial |
|---|---|---:|---:|---:|
| 0 | Preserve roles | 0.053923 | 0.134248 | +0.080325 |
| 0 | Free roles | 0.064235 | 0.164739 | +0.100504 |
| 8 | Preserve roles | 0.016045 | 0.267812 | +0.251768 |
| 8 | Free roles | 0.017048 | 0.284265 | +0.267217 |

All12seed/site/policy mean penalties are positive at initialization. All12
increase after training. These are12paired descriptive measurements from3source
models, not12independent training seeds and not a significance result.

Crossed controls retain the full source consumer to construct targets, while
the regression sees the same retained scalar features. In block8, replacing
initial features with trained features while keeping initial consumer weights
raises delta by0.233519(role)/0.255285(free), averaged across seeds. Replacing
only consumer weights on initial features changes it by-0.010877/-0.011604.
The2x2interactions are+0.029125/+0.023535. Thus the large observed late-block
increase is associated mainly with feature-state changes in this diagnostic;
the trained consumer coefficients alone do not reproduce it.

In block0 the feature-only increments are+0.029576/+0.047001, the consumer-only
increments+0.000084/+0.003348, with interactions+0.050666/+0.050155. The early
block's contrast has a substantial interaction component.

## Interpretation

The user's architectural objection is supported: harmful mixing is already
present without learning. The previous grouped-versus-random comparison alone
would not establish a learned quaternion-group effect.

The effect is nevertheless not numerically constant across initialization and
the trained states: all measured penalties increase, especially at the late
site. The crossed comparison points to changes in activation geometry/statistics
and their interaction with consumers. This supports a narrower statement about
training-associated grouping sensitivity under the registered protocol, not a
claim that quartets have a unique semantic or physical meaning.

The diagnostic still favors reconstruction of the source Hamilton mapping;
an advantage over random mixing is not a novel compression algorithm or proof
that no alternative grouping/compressor could perform as well. Prior free-role
pilot's failed continuation gate remains failed.

## Limits and additional descriptive check

Initial networks use unadapted BN; trained states include learned weights and
running statistics. Energy normalization removes a simple target-amplitude
scale, not differences in means, covariance, conditioning or diversity.

A separately labelled, post-hoc
[target-distribution check](target_distribution_check.json) uses saved check
targets, without any refits or endpoint replacement. In block8, the fraction
of output energy due to channel-centered variance is0.393--0.464initially and
0.901--0.911after training. Consequently these before/after states are not
distribution-matched, even after dividing by total output energy. Do not
interpret the measured increase as a pure causal effect of learned semantic
grouping. The extra statistics are context, not a newly registered primary test.

Regenerated initial states follow the pinned baseline constructor/seed recipe;
no archived epoch0checkpoint independently certifies their historical bitwise
identity. The same trained-fit-selected masks are held fixed for all cells.
One dataset, two sites, three seeds; no equivalence or broad generalization claim.

Data/protocol: [REPORT](REPORT.md), [protocol](protocol.json),
[descriptive summary](descriptive_summary.json), [completion audit](completion_audit.json).
No further numerical study is queued.
