# Direct quaternion consumer-output reconstruction pilot

## Hypothesis

Part of the observed Q/full gap may arise from shared feature reconstruction followed by folding, rather than from the compact quaternion architecture alone. Direct consumer-output regression may improve quality at exactly the same size.

## Measured results

Status: completed. CIFAR BN CNN6 seed2026, .5 removed/site, same victims and 500 fit/500 check images; no test inference.

Centered ridge fits five Q consumers, keeping all other parameters and BN buffers fixed; final real readout uses original map-fold only. Previous reference outcomes are reused. Both objective and compensation parameterization change, so this is not a pure feasible-space attribution.

| Case | Real parameters | Before val (%) | After 3 epochs val (%) |
| --- | ---: | ---: | ---: |
| q_pruned (reused) | 19098 | 11.90 | 48.80 |
| q_comp (reused) | 19098 | 18.70 | 54.20 |
| free_q_init (reused) | 72666 | 18.70 | 58.70 |
| free_full_fit (reused) | 72666 | 40.20 | 63.10 |
| direct_q_consumer | 19098 | 34.80 | 56.70 |

### Local pre-BN consumer reconstruction

Each pair uses identical current-reference inputs/targets; different sites follow the direct-fit path, not the old complete Q path. MSE averages over all output components and spatial positions.

| Site | Fit folded MSE | Fit direct MSE | Check folded MSE | Check direct MSE |
| --- | ---: | ---: | ---: | ---: |
| [0, 0] | 0.606886 | 0.399255 | 0.668767 | 0.44648 |
| [0, 1] | 0.30238 | 0.24591 | 0.319884 | 0.26053 |
| [1, 0] | 0.378797 | 0.266466 | 0.404904 | 0.285159 |
| [1, 1] | 0.199775 | 0.157099 | 0.203314 | 0.161577 |
| [2, 0] | 0.200975 | 0.157696 | 0.203867 | 0.161911 |

## Interpretation limits

One seed/operating point, post hoc exploratory selection; no novelty or generality claim. Local fit-objective improvement is a numerical sanity check, not evidence of held-out classification improvement. The prior is the folded kernel, not zero; this is a fixed centered-ridge objective, not the unregularized optimum. Reference and new chains diverge after the first intervention. Training labels are used only in the optional matched three-epoch repair, not regression. No additional parameters, alpha search, new baseline or test evaluation. Consumer-output LS is established prior art (He2017ChannelPruning). Attempted schedules match; successful AMP steps are counted for the new repair only, not retrospectively known for reused references.
