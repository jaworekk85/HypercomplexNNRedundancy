# Interpretation of the completed initialization diagnostic

## Hypothesis

The original PHM deficit may partly arise from Frobenius-only initialization rather than the attainable activation reconstruction within PHM4.

## Measurements

All 12 preregistered cases completed on CPU: two blocks, Q-only control and PHM4 from Q, saved SVD, and random seeds 31/32/33. Each used 300 full-batch Adam steps on the same 500 fit images, the same normalized MSE objective and learning rate. Best checkpoints were selected solely on fit, including step zero. Independent artifact replay confirmed 24 fit/check metrics and all source/checkpoint hashes. Two numerical tests passed, including exact completed-step optimizer resumption.

Held-out output MSE on the separate 500 check images (lower is better):

| Initialization | Block 0 | Block 8 |
|---|---:|---:|
| Q-only control | 0.756750 | 0.186038 |
| PHM4 from Q | 0.707439 | 0.185021 |
| PHM4 from SVD | 0.707589 | 0.185021 |
| PHM4 random 31 | 0.709385 | 0.185021 |
| PHM4 random 32 | 0.857500 | 0.185021 |
| PHM4 random 33 | 0.708763 | 0.185021 |

PHM4 from Q and from SVD reduce held-out error by approximately 6.5% in block0 and 0.55% in block8 relative to the equally optimized Q-only control. The block0 random-32 case is a negative outcome: error remains about 13.3% above Q after the fixed 300 steps. All starts and final/best-fit losses remain in summary.json.

## Interpretation and limits

Activation fitting largely removes the original SVD-initialized PHM4 deficit at these widths. The favorable results from some random starts show that initializing from Q is not necessary for a useful local fit, but one failed random start demonstrates finite-budget optimizer sensitivity. Three starts and one trained parent cannot establish an initialization-independent optimum.

Every PHM4 start has the same deployed consumer count: 1,216 reals in block0 and 18,496 in block8. Q controls have 1,152 and 18,432 respectively. This comparison deliberately measures initialization at fixed widths; it spends 64 extra basis coefficients per PHM layer. It therefore does not establish superiority at the old exact global budget.

No classification accuracy, full-network repair, learned n selection, validation/test tuning or official-test inference was performed. Neither the original Q/Q selection nor this local PHM improvement establishes general algebra superiority. A subsequent, separately registered catalogue could combine activation-fitted candidates with finer widths and actual global budgets; that experiment has not been launched.
