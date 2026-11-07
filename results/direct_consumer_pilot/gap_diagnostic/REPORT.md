# Cached-moment shared-map constraint diagnostic

## Hypothesis

Direct-Q improvement can arise from refitting shared maps for consumer outputs and from removing their shared-map parameterization ties. Compare these on identical cached inputs/targets, with the same centered ridge objective.

## Measured results

One discovery source/point, five local consumers; no new network inference/training or data selection. Direct float64 solution is compared before model-dtype rounding. Eigenvalue cutoff prespecified; numerical ranks recorded.

| Site | Shared-map / all-Q coefficients | Objective: folded | Objective: output-fit maps | Objective: direct Q | Tie-release share of total gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0, 0] | 64/1152 | 0.606886 | 0.575076 | 0.399732 | 84.6% |
| [0, 1] | 64/2304 | 0.30238 | 0.297083 | 0.24601 | 90.6% |
| [1, 0] | 256/4608 | 0.378797 | 0.350034 | 0.266691 | 74.3% |
| [1, 1] | 256/9216 | 0.199775 | 0.193313 | 0.157172 | 84.8% |
| [2, 0] | 1024/18432 | 0.200975 | 0.184725 | 0.157764 | 62.4% |

## Interpretation limits

Nested objective ordering is a least-squares sanity check, not a novel theorem. The first gain includes changing the objective/regularization used to choose the original map point. The second compares nested update spaces under identical objective/center and measures the extra ties shared across consumer outputs and kernel offsets. These local fit-objective gaps do not decompose classification accuracy, prove held-out ordering or quantify an unavoidable quaternion architecture cost. Check MSE is separately retained in summary.json. No output-fitted-map network was trained or added to the registered replication.
