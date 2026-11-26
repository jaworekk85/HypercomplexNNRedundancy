# Pre-registration development checks

Prepared2026-11-26. The first preparation passed six unit tests, the synthetic
local-fit/three-model repair/reload/completed-resume check and a discarded compact
CUDA gradient/update audit, then failed before registration: the old EuroSAT
metrics schema has no checkpoint_sha256 field. The new registration code was
corrected to validate each parent against the checksum already frozen in the
statistics-control protocol. No predecessor source, metadata or checkpoint changed.
The failed attempt remains in development_checks.json.

The corrected preparation passed all checks and registered the final source/config
hashes before scientific fitting. Six CPU tests cover pair extraction versus the
actual sequential prefix, retained-feature reproduction and source immutability,
fixed scalar-set rejection/actual scientific parameter counts, mixed compact
checkpoint roundtrip, source-prefix input capture/STOP hook cleanup, and dense
inverse-permutation preservation.

The synthetic end-to-end study fits six physical pairs, repairs all three model
conditions for one epoch, checks native-source/physical-block/model/logit replay,
and verifies that a completed-case resume preserves results. Synthetic accuracy
is not a scientific performance result. The full-size compact four-band CUDA
audit checks tracking, finite gradients and actual updates for every variable
and all four quaternion kernel components. Its audited copy is discarded.

Environment, detailed check output and frozen hashes are in readiness.json and
protocol.json. The scientific run starts only with --run after these checks.

## Scientific stop and separately versioned correction

The registered run stopped2026-11-26 at03:16Europe/Warsaw after three early-site
fits and before repair. Late archived features failed bitwise replay.
precision_diagnostic.json identifies cuDNN TF32 as the source on the tested
seed2026/site1batch: archived packs replay with TF32 enabled; baseline/current
FP32 collection disables it. No tolerance or registered source changed. This
failed output remains intact. ../eurosat_grouping_fp32_replication recomputes
common FP32 references for the same nine-model experiment under a new protocol.
