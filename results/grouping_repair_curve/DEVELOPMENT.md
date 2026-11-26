# Matched repair-curve implementation checks

Authorized2026-11-26 after committing/pushing the reviewed manuscript and plan.
The numerical question is recovery speed versus an advantage persisting through
a larger fixed repair budget, not a new grouping optimizer.

New config, wrapper and snapshot module reuse the frozen30epoch-capable trainer
without editing any predecessor. All runs restart from inherited **pre-repair**
models. Historical3epoch cosine results are not relabelled as new epoch3results.

Five CPU checks passed before registration: snapshot generation/reloading keeps
model state/mode and Python/NumPy/torch RNG unchanged; dropout training resumes
bitwise after an interrupted partial epoch; a saved completed epoch recovers a
missing requested snapshot; changed recipes/artifact hashes are rejected; paired
contrasts preserve reversals and do not pool settings. The full synthetic
two-epoch trajectory also verifies completed-case reuse. Detailed outcomes and
any failures are retained in`development_checks.json`.

Preparation validates predecessor hashes, then all18initial compact models must
replay previously saved validation/reconstruction-check logits exactly. Discarded
compact copies in each setting undergo tracked-variable/all-four-quaternion-
component gradient and update checks on CUDA. These checks introduce no new
scientific repair SGD and make no test-set forward pass. New sources are
registered only after these preparation checks pass; source hashes are rechecked
before launch and after completion. Per-case update checks are repeated on
discarded copies during execution.

Snapshots at0/1/3/10/30store physical state, held-out per-example logits/labels,
actual successful/attempted updates and checksums. Constructor/reload RNG is
restored so observation does not change dropout training. Missing snapshots are
recovered from a completed-epoch checkpoint before resuming the frozen trainer.
STOP is checked each batch; a partial epoch restarts from its previous completed
checkpoint. The final artifact audit covers all requested snapshot files.

No longer-repair outcome is claimed here. The primary endpoint is final epoch30;
earlier snapshots describe dynamics, without best-epoch selection. Existing
validation/calibration are exploratory. Final-test auditing/evaluation remains a
separate locked procedure after the research decisions.
