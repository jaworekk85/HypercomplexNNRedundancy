# Grouped versus independent scalar activation diagnostics

## Hypotheses

Whole-quaternion grouping may help predict individual whole-unit removal harm. The user's broader hypothesis is that structural weight constraints may explain benefits without any unique need for quaternion algebra. This within-model diagnostic cannot identify that causal explanation; see docs/STRUCTURE_HYPOTHESIS.md. No new method or general advantage is assumed.

## Fixed protocol

Saved calibration matrices and trained activations only; existing single-whole-unit validation ablations. Three seeds; block 3 is primary, blocks 1/2 secondary. No new inference, training or test access. The calibration/validation data were already explored; this is a descriptive follow-up, not an independent blind confirmation. See PROTOCOL.md for exact definitions and input hashes in summary.json.
Grouped score = maximum signed centered four-component cosine to another unit. Scalar score = mean of four maximum signed scalar Pearson correlations outside the entire current unit. Scalar features are unit-major in saved matrices. Both are judged on the identical four-channel ablation targets and shared valid-unit set. Scalar matching has more candidate partners, independent coordinates and per-component scale normalization. A difference does not isolate the algebra, matching restriction or scale effects.

## Measured results

| Seed | Block | Units | Grouped Spearman | Scalar Spearman | Group-minus-scalar | Grouped / scalar / random expected mean individual CE harm, top quarter |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | 1 | 8/8 | 0.000 | -0.667 | +0.667 | 0.08597 / 0.08597 / 0.05827 |
| 2026 | 2 | 16/16 | -0.199 | -0.147 | -0.052 | 0.02133 / 0.01877 / 0.01622 |
| 2026 | 3 | 32/32 | 0.223 | 0.487 | -0.264 | 0.00217 / 0.00116 / 0.00340 |
| 2027 | 1 | 8/8 | 0.108 | -0.095 | +0.204 | 0.04009 / 0.03935 / 0.04304 |
| 2027 | 2 | 16/16 | 0.018 | -0.156 | +0.174 | 0.01459 / 0.01466 / 0.01404 |
| 2027 | 3 | 32/32 | 0.440 | 0.493 | -0.053 | 0.00023 / 0.00046 / 0.00263 |
| 2028 | 1 | 8/8 | -0.108 | 0.095 | -0.203 | 0.07279 / 0.05636 / 0.06770 |
| 2028 | 2 | 16/16 | 0.351 | -0.156 | +0.507 | 0.01691 / 0.02301 / 0.01850 |
| 2028 | 3 | 32/32 | 0.317 | 0.157 | +0.160 | 0.00158 / 0.00093 / 0.00310 |

Primary block: paired Spearman differences [-0.26354163831040006, -0.05297267253411014, 0.15968966077004393]; mean -0.052, sample SD 0.212 across three training seeds.
Prespecified descriptive grouped-advantage rule passes: **False**. It requires higher grouped Spearman and no higher top-quarter individual loss harm in every primary seed. It is not a significance test.

| Seed | Block | Grouped image-half score-rank stability | Scalar image-half score-rank stability |
| ---: | ---: | ---: | ---: |
| 2026 | 1 | 0.921 | 1.000 |
| 2026 | 2 | 0.959 | 1.000 |
| 2026 | 3 | 0.986 | 0.968 |
| 2027 | 1 | 1.000 | 0.976 |
| 2027 | 2 | 0.979 | 0.982 |
| 2027 | 3 | 0.969 | 0.962 |
| 2028 | 1 | 1.000 | 0.976 |
| 2028 | 2 | 0.911 | 0.953 |
| 2028 | 3 | 0.908 | 0.978 |

## Interpretation and scope

A positive within-model rank association need not produce a better simultaneous-removal policy. The top-quarter diagnostic averages separate single-unit removals; it is not an observed quarter-size compact-model loss. The previous greedy-pruning policy includes a norm member rule and differs from either nearest-neighbor ranking here. No old test accuracies are relabeled as new policy results.
Undefined scores are excluded from both criteria together. Whole images, not pooled positions, are separated for stability. Neurons are not aligned across independently trained seeds. Secondary blocks cannot rescue a failed primary criterion; no score variant is selected from the outcomes.
No Hamilton products are evaluated in either similarity score: grouped cosine and Pearson are ordinary real-vector statistics. Even a grouped advantage would not establish that quaternion algebra itself caused the training advantage. Literature context: Ayinde2019, Vecchi2020, ZhangEtAl2021 (references.bib). No novelty or broad causal claim follows.

## Verified interpretation and decision

Scalar Pearson predicts individual loss harm better in 2/3 primary seeds; grouped cosine in 1/3. Mean paired grouped-minus-scalar Spearman difference -0.052, sample SD 0.212. The prespecified repeatable-grouped-advantage rule returns False.

In the primary block, both rankings select a quarter of units whose mean separate individual-removal harm is lower than the uniform random expectation in all three seeds. This is a diagnostic signal, not simultaneous quarter-removal performance; the earlier greedy-pruning test accuracies belong to a different policy. Neither a weak positive Spearman nor individual dispensability establishes useful joint compression.

This particular grouped cosine has no demonstrated repeatable advantage over the fixed independent-scalar reference. H2 is not confirmed for these measurements. Close this candidate advantage rather than tune the score or select a favorable secondary block. This does not rule out all other measures of activation redundancy.

The user's broader structural-constraint hypothesis remains open. Each same-width quaternion connection uses four real coefficients to generate sixteen signed/tied real matrix entries. This factual parameter sharing does not prove that arbitrary alternative constraints would behave the same. Our controls do not isolate generic sharing from its specific pattern, width, compute or optimization. See docs/STRUCTURE_HYPOTHESIS.md; context: ZhangEtAl2021 in references.bib.

All 28 implementation tests and 200 diagnostic artifact checks passed. Initial failed spatial-shape assertion and its correction are preserved in results/group_scalar/FAILED_CHECKS.md and failed_run_protocol.json. Frozen numerical inputs and earlier outcomes remain unchanged.
