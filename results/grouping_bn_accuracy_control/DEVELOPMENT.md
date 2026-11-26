# Pre-registration checks

Prepared 2026-11-26. No scientific accuracy intervention ran before registration.

The initial test invocation had nine passes and one failure: the synthetic
reconstruction test called the frozen `original_quartets(width)` helper with
two arguments. Fixed the new test and new synthetic-smoke caller to select the
desired tuple from the one-argument helper. No predecessor source was edited.
The failed attempt is retained in development_checks.json.

The corrected invocation passed all ten tests: seven inherited sequential-BN,
immutability, hook cleanup and fit-statistics tests, plus three new tests for
saved physical-block reconstruction/source immutability/hash rejection,
parameter/non-BN-buffer immutability enforcement, and signed primary contrasts.

Two-model synthetic full-network run passed, including native metrics/logit
replay, calibrated checkpoint/logit replay and a completed-case resume with
identical result rows. A discarded compact GPU ResNet passed tracked-variable,
finite-gradient, all-four-quaternion-component gradient and actual-update checks.
These are development checks, not scientific accuracy results.

Readiness and registered numerical source/configuration hashes were written
before the separate `--run` launch. Scientific sources must now remain frozen.
