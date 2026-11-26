# BN/statistics control and EuroSAT replication

Status: completed.

## Hypothesis

The training-associated grouping penalty may reflect BN state and marginal feature statistics. Test whether the contrast persists after common BN recalibration and additional feature marginal standardization, then measure the same question on an existing second dataset.

## Measured primary endpoint

Mean10draws excess held-out error normalized by target energy centered using FIT channel means.
Standardized condition follows sequential BN recalibration and per-scalar FIT marginal standardization.

| Dataset | Seed | Site | Policy | Initial delta | Trained delta | Increment |
|---|---:|---:|---|---:|---:|---:|
| cifar10 | 2026 | 0 | role | 0.257325 | 0.156971 | -0.100354 |
| cifar10 | 2026 | 0 | free | 0.364009 | 0.214457 | -0.149551 |
| cifar10 | 2026 | 1 | role | 0.319475 | 0.350963 | 0.0314885 |
| cifar10 | 2026 | 1 | free | 0.344555 | 0.369373 | 0.0248179 |
| cifar10 | 2027 | 0 | role | 0.198062 | 0.226513 | 0.0284512 |
| cifar10 | 2027 | 0 | free | 0.27135 | 0.284463 | 0.0131126 |
| cifar10 | 2027 | 1 | role | 0.326074 | 0.349264 | 0.0231909 |
| cifar10 | 2027 | 1 | free | 0.353846 | 0.377652 | 0.023806 |
| cifar10 | 2028 | 0 | role | 0.227936 | 0.00410905 | -0.223827 |
| cifar10 | 2028 | 0 | free | 0.285535 | 0.00421775 | -0.281318 |
| cifar10 | 2028 | 1 | role | 0.31844 | 0.314869 | -0.00357119 |
| cifar10 | 2028 | 1 | free | 0.345885 | 0.335412 | -0.0104736 |
| eurosat | 2026 | 0 | role | 0.119883 | 0.206099 | 0.0862162 |
| eurosat | 2026 | 0 | free | 0.172251 | 0.207522 | 0.0352706 |
| eurosat | 2026 | 1 | role | 0.245742 | 0.158904 | -0.0868379 |
| eurosat | 2026 | 1 | free | 0.261303 | 0.173625 | -0.0876775 |
| eurosat | 2027 | 0 | role | 0.10607 | 0.0761141 | -0.0299563 |
| eurosat | 2027 | 0 | free | 0.148149 | 0.116336 | -0.0318135 |
| eurosat | 2027 | 1 | role | 0.226385 | 0.170172 | -0.0562134 |
| eurosat | 2027 | 1 | free | 0.238249 | 0.174215 | -0.0640345 |
| eurosat | 2028 | 0 | role | 0.108909 | 0.130168 | 0.0212587 |
| eurosat | 2028 | 0 | free | 0.15258 | 0.154214 | 0.00163409 |
| eurosat | 2028 | 1 | role | 0.231627 | 0.155558 | -0.0760687 |
| eurosat | 2028 | 1 | free | 0.249882 | 0.163812 | -0.0860704 |

## Interpretation and limits

Two existing datasets and architectures,3seeds each: CIFAR10 Q ResNet20, EuroSAT four-band Q CNN6. Initial versus trained weights; native, BN-recalibrated, BN-recalibrated-plus-standardized feature states. Two early/late consumers per model.21groupings per site (original plus10role/10free fully mixed draws).1512reported fits:252CIFAR native rows reused from predecessor and1260new float64 ridge fits. No supervised training, new algebra, architecture search, classification prediction or official-test/validation inference. EuroSAT is a second dataset AND architecture/input encoding, not an isolated dataset-factor experiment.

From BN-recalibrated FIT activations estimate each full scalar channel's mean/population variance; transform fit/check using ONLY those fit statistics, subtract mean and divide by std clamped at1e-6. Count constant channels. Recompute full-width consumer targets from these transformed features and unchanged source Hamilton weights. This defines a modified exact-feature diagnostic teacher, NOT function-preserving graph surgery or a compact deployment model. Means/scales are controlled, covariance/sparsity/higher moments are not. Native and BN-only targets use actual full consumer outputs before BN2.

Report native/BN/standardized contrasts and all seeds/sites; no pooling random draws as independent training seeds. Opposite or vanished effects must be reported. This addresses stated marginal/BN confounds, not full distribution matching or semantic identifiability. No equivalence/significance/novelty claim, continuation threshold, choice of successful condition or automatic further experiment. Previous failed reassembly gate remains failed.

All native/BN-only/raw-energy contrasts and individual draws are retained in summary.json/per-cell results.
No significance, equivalence, semantic identifiability or method novelty claimed.
