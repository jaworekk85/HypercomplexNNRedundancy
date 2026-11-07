# All-block diagnostics across three seeds

## Hypothesis

Similarity may predict low-impact whole-unit removal. All negative correlations are retained.

## Measured results

| Seed | Model | Block | Low-impact units | Similarity vs negative loss impact (Spearman) | Image-half stability |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2026 | quaternion | 1 | 0/8 | 0.0 | 0.9985 |
| 2026 | quaternion | 2 | 2/16 | -0.19911591056614425 | 0.9980 |
| 2026 | quaternion | 3 | 26/32 | 0.22326188075118347 | 0.9923 |
| 2026 | real | 1 | 4/16 | 0.5313689313240572 | 0.9979 |
| 2026 | real | 2 | 18/32 | -0.01485693723410694 | 0.9975 |
| 2026 | real | 3 | 63/64 | 0.24621620081341264 | 0.9939 |
| 2027 | quaternion | 1 | 0/8 | 0.10844160586017297 | 0.9983 |
| 2027 | quaternion | 2 | 4/16 | 0.017699192050323935 | 0.9975 |
| 2027 | quaternion | 3 | 24/32 | 0.44006251808172553 | 0.9947 |
| 2027 | real | 1 | 0/16 | 0.4723279389547174 | 0.9966 |
| 2027 | real | 2 | 15/32 | 0.2578397602747305 | 0.9968 |
| 2027 | real | 3 | 55/64 | 0.31198122848280424 | 0.9917 |
| 2028 | quaternion | 1 | 0/8 | -0.1077863635721933 | 0.9974 |
| 2028 | quaternion | 2 | 0/16 | 0.3512939045975711 | 0.9975 |
| 2028 | quaternion | 3 | 20/32 | 0.3165811563712169 | 0.9940 |
| 2028 | real | 1 | 3/16 | -0.41328694658537785 | 0.9970 |
| 2028 | real | 2 | 16/32 | 0.05099515632626914 | 0.9976 |
| 2028 | real | 3 | 64/64 | 0.041398576310857436 | 0.9916 |

## Interpretation

All three seeds pass the fixed block-3 diagnostic gate: True.
The gate uses individual ablations and within-model image-half stability. It does not prove cumulative safe removal or neuron identity across seeds. Neurons are not aligned between independent trainings. Quaternion and scalar real ablations have different granularity. No novel score or established method superiority is claimed.
