# PHM4 initialization diagnostic

Status: completed.

## Hypothesis

Different initializations and activation fitting may explain the original PHM deficit.

## Measured results

| Block | Initialization | Seed | Consumer real parameters | Initial fit MSE | Best artifact fit MSE | Check MSE |
|---|---|---:|---:|---:|---:|---:|
| 0 | q_control | 2026 | 1152 | 0.724016 | 0.723974 | 0.75675 |
| 0 | from_q | 2026 | 1216 | 0.724016 | 0.674014 | 0.707439 |
| 0 | from_svd | 2026 | 1216 | 1.63588 | 0.674233 | 0.707589 |
| 0 | random | 31 | 1216 | 4.5788 | 0.675798 | 0.709385 |
| 0 | random | 32 | 1216 | 4.56926 | 0.817405 | 0.8575 |
| 0 | random | 33 | 1216 | 4.58772 | 0.675542 | 0.708763 |
| 8 | q_control | 2026 | 18432 | 0.181356 | 0.181356 | 0.186038 |
| 8 | from_q | 2026 | 18496 | 0.181356 | 0.180067 | 0.185021 |
| 8 | from_svd | 2026 | 18496 | 0.303789 | 0.180067 | 0.185021 |
| 8 | random | 31 | 18496 | 0.884756 | 0.180067 | 0.185021 |
| 8 | random | 32 | 18496 | 0.888157 | 0.180067 | 0.185021 |
| 8 | random | 33 | 18496 | 0.892886 | 0.180067 | 0.185021 |

## Interpretation and limits

One discovery Q parent, two layers at the frozen all-Q-selected widths. Compare PHM4 init from exact Q embedding, saved dense-oracle SVD and three random seeds. All PHM starts have identical real parameter counts; each has 64 additional basis parameters over Q. This is a local optimizer/initialization diagnostic, not a matched-original-budget quality gain or full-network fine-tuning. No new n search, test or classification evaluation.

Full-batch normalized pre-BN output MSE on the same 500 fit calibration images, no labels, no ridge and no weight decay. This differs from the original ridge solution; a Q-only Adam control uses the same steps/objective.

Lowest FIT objective including initialization, recorded independently per run; report final and best-fit losses. Evaluate the selected artifact on 500 separate check images only after optimization. No initialization or checkpoint selection on check/validation/test; all random starts reported.

PHM4 contains Q but has 64 extra stored real basis coefficients per layer. This is not a classification result or an equal-budget improvement over Q. Three random initializations do not prove an optimizer optimum. Retain failures and all starts. Prior art: ZhangEtAl2021, Grassucci2024, VanLoanPitsianis1993, VieiraValle2022ELM. See references.bib.
