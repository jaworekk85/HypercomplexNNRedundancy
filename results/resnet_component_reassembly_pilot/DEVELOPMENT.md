# Development checks before registration

The first targeted CPU run returned two failures and seven passes. Both failures
concerned identical-forward or standard-factory reload checks: the replacement
BatchNorm was in training mode after insertion into an otherwise evaluated
block. The new block compiler now calls eval after replacing its modules.
No scientific fit had started and no protocol for this pilot was frozen when
the correction was made. The follow-up preparation passed all16 tests (nine
ResNet-reassembly checks plus seven existing technical-diagnostic checks).

The candidate producer bank deliberately contains overlapping source quartets.
It uses an explicit bank-only overlap option; final compact models retain the
strict injective source-component check. A dedicated test compares overlapping
bank features with independently compiled single-quartet features and verifies
that the final compiler rejects overlap. This distinction was corrected during
implementation, before the initial targeted test and registration.

The end-to-end synthetic check covers ten local fits, five combined compact
models, training audits, model/metric reload and reuse of all completed cases.
It uses deliberately tiny random data and is not a scientific accuracy result.
GPU discarded-copy checks verified all trainable variables and all four
quaternion coefficient components had gradients and changed. AMP preflight
recorded two successful updates out of three attempts; one overflow skip was
retained in the readiness record. Scientific repair logs attempted/successful
updates using the existing frozen training recipe.

Machine-readable preparation output, versions, source hashes and synthetic
artifact locations are retained in development_checks.json, readiness.json and
protocol.json. Older registered numerical files were not changed.
