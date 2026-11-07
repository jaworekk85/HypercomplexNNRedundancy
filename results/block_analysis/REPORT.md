# All-block activation and ablation pilot

> Numerical-threshold erratum: the original table below is preserved. Inclusive
> comparison with machine-rounding tolerance 1e-12 changes the seed-2026
> block-3 quaternion low-impact count from 24/32 to **26/32**, real block 2
> from 16/32 to 18/32 and real block 3 from 60/64 to 63/64. For example,
> 0.783 - 0.778 is stored as 0.0050000000000000044 but belongs at the
> prespecified <=0.005 boundary. All measured loss/accuracy, correlations,
> similarity matrices, thresholds and selected block (3) are unchanged.
> Corrected labels and original changed-unit IDs are recorded in
> `results/cnn_compression/block_analysis/`; the frozen original analysis
> code and JSON/CSV remain available for audit. The regression suite passes
> 22 tests, including the boundary check.

## Hypotheses

Calibration similarity may identify individually removable quaternion channels. This is a diagnostic hypothesis, not a new method or proof of useful compression.

## Fixed protocol

Frozen selected checkpoints only. No optimizer updates, data augmentation or test inference. Initial activations are reconstructed using the checkpoint's recorded initialization seed. Capture outputs after each complete block (including pooling, with dropout disabled).
Scores use 1000 calibration images. Functional ablations use the disjoint 1000 validation images. Stability splits calibration by whole images before pooling positions. Centered group cosine, scalar Pearson and spectral rank reuse the existing definitions.
An individually low-impact removal has validation cross-entropy increase at most 0.01 and accuracy decrease at most 0.5 percentage points. This threshold is prespecified, descriptive and allows removal to improve validation scores.
A candidate block has at least two such individual quaternion removals and calibration pair-value stability at least 0.9. Select the largest individual low-impact fraction; ties favor more units, then the earlier block. This gate motivates an experiment; it does not establish safe simultaneous removal.

## Measured results

| Model | Block | Units | Maximum similarity | Effective scalar rank | Individual low-impact removals | Similarity vs negative loss impact (Spearman) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quaternion | 1 | 8 | 0.517 | 19.11 | 0/8 | 0.000 |
| quaternion | 2 | 16 | 0.313 | 35.89 | 2/16 | -0.199 |
| quaternion | 3 | 32 | 0.526 | 33.07 | 24/32 | 0.223 |
| real | 1 | 16 | 0.596 | 12.28 | 4/16 | 0.531 |
| real | 2 | 32 | 0.595 | 21.01 | 16/32 | -0.015 |
| real | 3 | 64 | 0.899 | 18.77 | 60/64 | 0.246 |

## Interpretation and decision

Candidate block for the next controlled experiment: **3**.
Positive Spearman means greater nearest-neighbor similarity is associated with smaller loss harm. A small or negative coefficient is a failure of that predictor in this model/layer and must be retained. Individual units can be individually dispensable while their simultaneous removal is harmful.
Temporary activation zeroing measures functional effects. No architecture was made smaller; parameter savings are zero. Real ablations remove one scalar channel, whereas quaternion ablations remove all four components; counts are not directly comparable across these granularities.
Weight norms describe raw incoming kernels and ignore BatchNorm re-scaling. These are reference measurements, not an optimized importance criterion. Threshold 0.9 for near-duplicate pairs is not a universal gate for all forms of redundancy.
Only seed 2026 is available in this report. Three-seed confirmation and physically compact 0/25/50% removal remain separate stages. Literature context uses existing references.bib keys Ayinde2019, Vecchi2020, Singh2026; no faithful reproduction or novel pruning algorithm is claimed.
