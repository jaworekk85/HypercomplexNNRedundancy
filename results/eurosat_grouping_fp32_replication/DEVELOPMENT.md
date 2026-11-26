# Corrected FP32 replication: development and provenance

The first registered implementation remains preserved in
../eurosat_grouping_replication. It completed three early-site physical fits,
then stopped before any supervised repair because a late-site archived
activation batch failed exact replay. No outcome-dependent tuning/group selection.

../eurosat_grouping_replication/precision_diagnostic.json compares four GPU
backend modes on the first16FIT examples of seed2026/site1. Archived features
and targets replay bitwise with cuDNN TF32 enabled, for either deterministic
setting. With TF32 disabled, maximum feature/target differences are0.00162363
and0.00330162. Parent baseline replay disables TF32. The diagnostic helper's
initial repository-import-path failure was corrected; no scientific data changed.

No tolerance or frozen source/config was edited. New modules/configuration and
a separate output retain masks, predetermined draw0assignments, model budgets
and repair settings. They regenerate all native FIT/CHECK features/targets,
ridge scales and18exact-feature fits in explicit deterministic FP32. Old TF32
errors/coefficients are not used as new physical reconstruction references.
The base local record ideal_reused means reuse of fresh within-study rows by
physical fitting; the corrected summary/report explicitly counts18new references.

Eight CPU tests passed: six base compact-CNN/shape/count/replay/hook controls,
plus precision enforcement and fresh-reference/mask/source/resume controls.
The base synthetic six-pair/three-model full repair/replay/resume passed again.
A discarded full-size compact four-band CUDA model passed tracked-variable,
finite-gradient and all-four-component gradient/update checks. New readiness
and hashes were recorded before --run. Source checkpoints/data remain fixed.
