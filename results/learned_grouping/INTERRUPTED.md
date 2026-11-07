# Initial technical run retained

The first random-control run was intentionally interrupted after six saved
epochs to fix a discovered final-epoch resume edge case in the new driver.
That driver attempted to delete a local checkpoint variable that would not
exist when resuming after the last epoch; the final CSV also needed rebuilding
from checkpoint history if its write had been interrupted.

No scientific result is claimed from these six epochs. The checkpoint,
protocol, CSV and exact original source (`interrupted_driver.py`) are retained
here. Numerical model, optimizer, grouping criterion and training settings
were not tuned. The corrected comparison starts afresh in
`results/learned_grouping_pilot`; no source hash in the interrupted checkpoint
is silently rewritten. Regression tests inject interruption at both a soft
epoch and the final hard epoch, then compare with uninterrupted training.
