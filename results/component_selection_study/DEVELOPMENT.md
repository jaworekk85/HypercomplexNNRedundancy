# Component selection versus whole-unit pruning: development checks

User explicitly authorized the next study2026-11-28, after identifying it as a
required response to the practical "so what?" review objection. New files only;
no frozen predecessor is edited.

The previous pilot built partial quartet catalogues with a whole-unit completion
mechanism. This study instead maintains complete fixed-budget injective
assignments at every step. Legal whole-unit moves remain available to both
policies. Component moves can replace selected scalars with any omitted scalar
or swap selected roles, including within a quartet. Stratified proposal quotas
ensure component replacements/swaps are actually evaluated even when intact
unit moves have zero producer projection cost. This removes that old structural
completion bias; it is still a bounded heuristic search, not exhaustive global
optimization or a methodological novelty claim.

An initial manual CPU run passed five tests but failed its end-to-end check:
the frozen `save_block` helper returns a checksum dictionary, and the new wrapper
incorrectly nested that dictionary under another`sha256`key. The wrapper was
corrected before source registration or scientific selection. The next manual
run passed all six new tests, including synthetic selection, physical refitting,
two-epoch repair and exact completed-case resume for both architectures. The
formal prepare command repeats those checks plus the five established repair-
snapshot/resume tests, with output/failures retained in`development_checks.json`.

Pre-registration checks also replay all six source-parent validation metrics and
native source input/feature/target batches at both sites in deterministic FP32.
Discarded full-size compact CUDA copies must pass variable tracking, finite
gradients and updates for every quaternion component. Candidate source/input
hashes are captured and rechecked before the protocol is frozen. Technical smoke
models use synthetic data and are not scientific results.

The paired scientific comparison holds physical architecture/real parameter
budget and30epoch repair recipe fixed. It allows the retained scalar set to
change. Selection uses only the inner/outer FIT images; final refits use all FIT
images. CHECK/validation report outcomes only, and remain previously explored
splits. Per-image snapshots, actual AMP update counts, selected-set overlaps,
mixed quartets, role changes, search coverage and runtime are retained. No test
forward pass, scratch training, new parent or compression-ratio grid is included.
