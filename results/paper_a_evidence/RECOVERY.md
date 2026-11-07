# Paper A: saved recovery histories

No new training or test inference. Means of seed means; random replicates average within seed.
Equal epochs/attempted updates do not establish equal successful AMP updates or runtime savings.

| Dataset | Removed/site | Policy | Before | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| cifar | 0.25 | greedy_comp | 59.67 | 71.40 | 71.43 | 71.50 |
| cifar | 0.25 | greedy_pruned | 23.47 | 69.23 | 70.03 | 70.00 |
| cifar | 0.25 | bn_pruned | 29.50 | 68.47 | 68.90 | 69.27 |
| cifar | 0.5 | greedy_comp | 21.97 | 50.20 | 53.37 | 53.57 |
| cifar | 0.5 | greedy_pruned | 13.43 | 44.70 | 48.63 | 49.67 |
| cifar | 0.5 | bn_pruned | 11.33 | 43.37 | 47.90 | 48.23 |
| cifar | 0.75 | greedy_comp | 10.03 | 17.67 | 18.60 | 18.50 |
| cifar | 0.75 | greedy_pruned | 10.00 | not measured | not measured | not measured |
| cifar | 0.75 | bn_pruned | 10.03 | 16.87 | 21.70 | 21.77 |
| eurosat | 0.25 | greedy_comp | 78.13 | 93.37 | 93.70 | 93.73 |
| eurosat | 0.25 | greedy_pruned | 32.03 | not measured | not measured | not measured |
| eurosat | 0.25 | bn_pruned | 21.03 | 90.47 | 91.97 | 91.80 |
| eurosat | 0.5 | greedy_comp | 32.20 | 82.23 | 84.37 | 85.33 |
| eurosat | 0.5 | greedy_pruned | 16.20 | not measured | not measured | not measured |
| eurosat | 0.5 | bn_pruned | 14.93 | 72.77 | 78.37 | 79.17 |
| eurosat | 0.75 | greedy_comp | 18.43 | 45.23 | 50.00 | 50.37 |
| eurosat | 0.75 | greedy_pruned | 12.73 | not measured | not measured | not measured |
| eurosat | 0.75 | bn_pruned | 10.50 | 34.83 | 42.80 | 44.07 |
