# BN-only full-network accuracy control

Status: completed

## Hypothesis

BN-only recalibration may reduce the accuracy cost of mixing groups.

## Measured results

All accuracies below are percentages. Repair is the historical final epoch3.

| Seed | Model | Native val | BN-only val | Repair val | BN-only val CE |
|---|---|---:|---:|---:|---:|
| 2026 | free_100_0 | 78.00 | 75.70 | 87.40 | 0.727937 |
| 2026 | free_50_0 | 82.50 | 83.00 | 88.70 | 0.522699 |
| 2026 | original | 88.10 | 87.20 | 89.80 | 0.357517 |
| 2026 | parent | 91.00 | 90.30 | - | 0.299155 |
| 2026 | role_100_0 | 78.60 | 76.80 | 88.20 | 0.688275 |
| 2026 | role_50_0 | 85.70 | 82.10 | 89.10 | 0.497107 |
| 2027 | free_100_0 | 80.70 | 75.60 | 88.30 | 0.788314 |
| 2027 | free_50_0 | 85.00 | 82.50 | 90.10 | 0.568619 |
| 2027 | original | 88.50 | 85.50 | 91.00 | 0.455314 |
| 2027 | parent | 91.40 | 88.70 | - | 0.356262 |
| 2027 | role_100_0 | 81.50 | 77.60 | 88.50 | 0.711714 |
| 2027 | role_50_0 | 86.90 | 82.70 | 89.70 | 0.560424 |
| 2028 | free_100_0 | 75.10 | 73.30 | 87.90 | 0.802252 |
| 2028 | free_50_0 | 82.80 | 78.90 | 88.80 | 0.654130 |
| 2028 | original | 86.90 | 87.30 | 89.60 | 0.408845 |
| 2028 | parent | 90.70 | 91.10 | - | 0.284044 |
| 2028 | role_100_0 | 78.80 | 79.40 | 89.10 | 0.628352 |
| 2028 | role_50_0 | 82.40 | 81.30 | 89.00 | 0.585652 |

## Paired primary contrasts

Gap = original accuracy minus mixed accuracy, in percentage points. Positive reduction means BN closes the gap.

| Seed | Mixed case | Native gap | BN-only gap | BN gap reduction | Historical repair gap |
|---|---|---:|---:|---:|---:|
| 2026 | free_100_0 | 10.10 | 11.50 | -1.40 | 2.40 |
| 2026 | free_50_0 | 5.60 | 4.20 | 1.40 | 1.10 |
| 2026 | role_100_0 | 9.50 | 10.40 | -0.90 | 1.60 |
| 2026 | role_50_0 | 2.40 | 5.10 | -2.70 | 0.70 |
| 2027 | free_100_0 | 7.80 | 9.90 | -2.10 | 2.70 |
| 2027 | free_50_0 | 3.50 | 3.00 | 0.50 | 0.90 |
| 2027 | role_100_0 | 7.00 | 7.90 | -0.90 | 2.50 |
| 2027 | role_50_0 | 1.60 | 2.80 | -1.20 | 1.30 |
| 2028 | free_100_0 | 11.80 | 14.00 | -2.20 | 1.70 |
| 2028 | free_50_0 | 4.10 | 8.40 | -4.30 | 0.80 |
| 2028 | role_100_0 | 8.10 | 7.90 | 0.20 | 0.50 |
| 2028 | role_50_0 | 4.50 | 6.00 | -1.50 | 0.60 |

## Interpretation limits

Exploratory paired intervention on previously inspected validation/CHECK splits. Three model seeds; no significance, equivalence, gain or novelty claim. BN-only changes running statistics, never weights/affine. Historical repair also changes BN and weights and did not start from this intervention. No official test, new supervised training or further automatic study.
