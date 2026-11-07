# Two-block hybrid and variable-n PHM: readiness

Status: technically ready, awaiting predecessor studies. No scientific run.

## Hypothesis

A hard Q/real/learned-PHM choice after local pruning may improve quality per
stored parameter compared with a Q/real-only hybrid and fixed-n PHM. This is
not established novelty or a measured gain.

## Measured technical checks

Six CPU tests passed: PHM n2/4/8 storage, Kronecker-SVD Frobenius optimum,
independent spatial/block reconstruction, gradients/updates of every PHM factor,
exact Hamilton inclusion, invalid partitions, knapsack versus exhaustive small
enumeration, infeasible-budget rejection, actual convolution-error replay,
physical Q producer/BN narrowing, residual shapes, mixed manifest reload and
source-model immutability. The six tests contain several invariants each.

End-to-end smoke completed five policies on32 SYNTHETIC CPU images, one tiny
repair epoch each. It checked candidate construction, actual counts under a
shared budget, selected graphs, repair/checkpoint/reload and result reporting.
These accuracies are NOT experimental evidence. First smoke correctly refused
the active shared GPU-study lock; the CPU-only test harness then used its own
lock, while the scientific runner retains the exclusive GPU-study lock.

Additional forced mixed PHM/dense ResNet update passed every parameter and all
remaining Q component checks. Source hashes and smoke results are in readiness.json;
numerical sources/config are HASH-FROZEN in protocol.json.

## Scope and execution

Runner: `python -B -u -m scripts.resnet_structure_catalogue --run`.
Config: configs/resnet_structure_catalogue.json. Controls: all-Q, Q/real,
fixed learned PHM4, variable-n PHM, same-catalogue weight-only selector.
Two consumers0/8, seed2026, local reference hidden widths halved. Other seven
blocks stay at baseline widths. This is not the all-nine-sites core experiment.
All policies use fit-only selection, separate check/validation, the same3ep
repair, actual common full-model parameter cap and explicit unused slack.
PHM basis storage is counted. Frobenius SVD is not an activation-weighted optimum;
independent local surrogate errors do not guarantee global classification gains.

Run requires completed main ResNet compression and key-replication summaries
plus completed baseline pair and matching checkpoint hash. The main compression
worker is now implemented and running; the key-replication worker remains
unimplemented. This pilot does not create it or silently skip its priority.
No additional scientific computation, ALS or full-nine-layer search started.

## Interpretation

Ready code is not a positive research finding. Novelty remains unconfirmed;
ALDS/SPDY, PHM/PHNNs, Numerion and post-training PHM work remain prior art.
