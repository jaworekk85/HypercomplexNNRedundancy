# Paper A: BN-only control

Status: completed. Validation results; test not evaluated.

2048 unaugmented training images, identical within seed across policies. No labels/gradients/weight updates.
Cumulative: one forward pass, simultaneous train-mode BNs. Global: six passes, sequential eval-mode moments.
These differ in estimator and cost; report both. This is not a reproduction of EagleEye or REPAIR.

| Dataset | Removed/site | Policy | Repair | n seeds | Val accuracy mean (%) | SD (pp) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| cifar | 0.0 | baseline | before | 3 | 77.93 | 1.10 |
| cifar | 0.0 | baseline | cumulative | 3 | 76.63 | 1.04 |
| cifar | 0.0 | baseline | layerwise_global | 3 | 76.67 | 1.00 |
| cifar | 0.25 | bn_pruned | before | 3 | 29.50 | 6.32 |
| cifar | 0.25 | bn_pruned | cumulative | 3 | 45.03 | 3.21 |
| cifar | 0.25 | bn_pruned | layerwise_global | 3 | 44.93 | 3.16 |
| cifar | 0.25 | greedy_comp | before | 3 | 59.67 | 3.21 |
| cifar | 0.25 | greedy_comp | cumulative | 3 | 64.77 | 0.32 |
| cifar | 0.25 | greedy_comp | layerwise_global | 3 | 64.70 | 0.35 |
| cifar | 0.25 | greedy_pruned | before | 3 | 23.47 | 2.75 |
| cifar | 0.25 | greedy_pruned | cumulative | 3 | 52.60 | 4.53 |
| cifar | 0.25 | greedy_pruned | layerwise_global | 3 | 52.60 | 4.53 |
| cifar | 0.5 | bn_pruned | before | 3 | 11.33 | 0.55 |
| cifar | 0.5 | bn_pruned | cumulative | 3 | 18.83 | 5.52 |
| cifar | 0.5 | bn_pruned | layerwise_global | 3 | 18.90 | 5.53 |
| cifar | 0.5 | greedy_comp | before | 3 | 21.97 | 4.12 |
| cifar | 0.5 | greedy_comp | cumulative | 3 | 32.67 | 2.61 |
| cifar | 0.5 | greedy_comp | layerwise_global | 3 | 32.67 | 2.61 |
| cifar | 0.5 | greedy_pruned | before | 3 | 13.43 | 1.55 |
| cifar | 0.5 | greedy_pruned | cumulative | 3 | 21.43 | 4.70 |
| cifar | 0.5 | greedy_pruned | layerwise_global | 3 | 21.47 | 4.75 |
| eurosat | 0.0 | baseline | before | 3 | 96.67 | 0.29 |
| eurosat | 0.0 | baseline | cumulative | 3 | 95.93 | 0.35 |
| eurosat | 0.0 | baseline | layerwise_global | 3 | 96.00 | 0.17 |
| eurosat | 0.25 | bn_pruned | before | 3 | 21.03 | 1.94 |
| eurosat | 0.25 | bn_pruned | cumulative | 3 | 57.43 | 10.52 |
| eurosat | 0.25 | bn_pruned | layerwise_global | 3 | 57.03 | 10.07 |
| eurosat | 0.25 | greedy_comp | before | 3 | 78.13 | 5.02 |
| eurosat | 0.25 | greedy_comp | cumulative | 3 | 90.33 | 1.80 |
| eurosat | 0.25 | greedy_comp | layerwise_global | 3 | 90.23 | 1.89 |
| eurosat | 0.25 | greedy_pruned | before | 3 | 32.03 | 6.51 |
| eurosat | 0.25 | greedy_pruned | cumulative | 3 | 61.33 | 4.37 |
| eurosat | 0.25 | greedy_pruned | layerwise_global | 3 | 60.70 | 4.74 |
| eurosat | 0.5 | bn_pruned | before | 3 | 14.93 | 5.24 |
| eurosat | 0.5 | bn_pruned | cumulative | 3 | 24.87 | 13.38 |
| eurosat | 0.5 | bn_pruned | layerwise_global | 3 | 24.83 | 12.98 |
| eurosat | 0.5 | greedy_comp | before | 3 | 32.20 | 0.72 |
| eurosat | 0.5 | greedy_comp | cumulative | 3 | 58.67 | 4.54 |
| eurosat | 0.5 | greedy_comp | layerwise_global | 3 | 58.70 | 4.65 |
| eurosat | 0.5 | greedy_pruned | before | 3 | 16.20 | 4.52 |
| eurosat | 0.5 | greedy_pruned | cumulative | 3 | 23.97 | 9.86 |
| eurosat | 0.5 | greedy_pruned | layerwise_global | 3 | 23.77 | 9.86 |
