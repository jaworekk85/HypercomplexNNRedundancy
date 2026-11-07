# Grouped versus independent scalar diagnostics

Frozen before computing the new scalar criterion/outcomes. Existing grouped
outcomes and checkpoints have already been explored; this is a descriptive
follow-up on reused calibration/validation, not a new blind confirmation.

## Hypothesis

Treating four channels jointly may help predict the harm of removing one
whole quaternion unit. No special property of quaternion multiplication,
novel score or compression advantage is assumed. The user's broader
structural-constraint hypothesis is recorded in docs/STRUCTURE_HYPOTHESIS.md.

## Fixed comparison

For both criteria, use trained post-block calibration activations only:

1. Grouped: existing centered four-component cosine to each other whole
   unit; the unit's score is the maximum signed similarity.
2. Scalar: existing scalar Pearson matrix. For each of a unit's four scalar
   components, find maximum signed Pearson to any component outside that
   unit; average the four maxima. Exclude all within-unit matches because
   those components disappear together in the target ablation.

The scalar matrix is unit-major (r,i,j,k for unit 0, then unit 1...), not the
component-blocked convolution layout. No absolute correlations, score fitting,
alternative aggregation or label-based ranking. Grouped candidates share one
partner unit and corresponding component coordinates; independent scalar
matches may use different partners and coordinates. Scalar Pearson also
standardizes individual component scale whereas grouped cosine retains it.
These are transparent reference measurements, not a clean causal isolation
of algebra, matching constraint, coordinate alignment or scale effects.

If a criterion is undefined for a unit (constant components / no external
valid partner), exclude it from BOTH criteria's paired comparison and report
the count. Do not silently replace undefined scores with zero.

## Target and primary analysis

Reuse validation single-whole-quaternion ablations, always all four channels.
Primary: block 3 (already selected in the earlier pilot); three training seeds.
Metric: within-seed Spearman(score, negative validation CE increase). A larger
score is intended to predict smaller harm. Report paired group-minus-scalar
effects, their mean and sample SD across training seeds. Do not pool neurons
or layers as independent training runs; do not align neuron identities.

Secondary: for the highest-scoring quarter (floor(n/4), index tie-break),
report mean INDIVIDUAL loss harm and low-impact fraction, using the existing
inclusive <=0.01 CE / <=0.5-pp labels. The random expected mean individual
harm is exactly the mean over all eligible units, and expected low-impact
fraction is their overall fraction. These are diagnostics, not simultaneous
removal or physical compression outcomes. Earlier greedy-pruning test results
must not be relabeled as performance of these nearest-neighbor rankings.

Descriptive go: grouped Spearman greater AND top-quarter mean individual loss
harm no larger in every primary-block seed. Otherwise record no repeatable
grouped advantage. This is a pilot consistency rule, not a significance test.
Blocks 1 and 2 are secondary only and cannot replace a failed primary block.

## Integrity and stability

Read existing matrices/snapshots and corrected ablation summaries. Check
source hashes, disjoint calibration/validation indices, quaternion-layer
gradient/update checks from the original run, cached-vs-recomputed matrices
and exact agreement with previously recorded grouped scores/Spearman.
Split calibration by whole images before recomputing scores for each half;
report within-model half-to-half score-rank stability for both criteria.

No new data download, model inference, test-data access, training, graph
surgery, parameter savings or latency claims. Record versions, seeds, source
and input-artifact hashes. Keep hypotheses, measurements and interpretations
separate. Context keys from references.bib: Ayinde2019, Vecchi2020,
ZhangEtAl2021; neither measurement is claimed as a novel method.
