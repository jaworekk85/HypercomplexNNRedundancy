# Pre-registration checks

All checks passed before the numerical sources/configuration were frozen and
before the scientific search launch on2026-11-26. No failed development check
occurred in this preparation; development_checks.json retains the invocation.

Seven CPU tests check exact fixed-scalar-set swaps, canonical whole-unit order
without role erasure, reproducible proposal generation, producer-projection
cost equality to the actual compiler, retaining the original answer, accepting
a genuinely better nonoriginal answer, disjoint reproducible inner/outer FIT
subsets, compact Hamilton inner fitting/source immutability, and KL identity
and temperature validation. These are grouped into seven test functions.

A synthetic end-to-end run performed ten inner fits across two sites, global
teacher-KL selection, final refits, bitwise replay of the original reference,
one new one-epoch repair, saved model/logit replay, and a completed-case resume
with identical scientific result rows. The smoke explicitly forces its new
repair branch even when assignments remain original; scientific runs instead
reuse a verified bitwise-identical historical control to avoid redundant SGD.

A discarded-copy compact full-size CUDA ResNet passed all tracked-variable,
finite-gradient, four-quaternion-component gradient and actual-update checks.
The audited copy is never used for scientific results. Versions and hashes
are in readiness.json and protocol.json. No predecessor numerical file changed.
