# Fixed-set adaptive grouping: completed interpretation

Completed2026-11-26 at02:39:37Europe/Warsaw; final audit at02:39:44.
[Protocol](PROTOCOL.md), [measured report](REPORT.md),
[descriptive evidence](descriptive_summary.json),
[completion audit](completion_audit.json),
[specific prior-art check](../../docs/literature/GROUPING_EXCHANGE_PRIOR_ART_2026-11-26.md).

## Hypothesis

A bounded data-chosen grouping search might improve compact Hamilton networks
beyond preserving their original groups, when the retained scalar information,
physical architecture, final consumer estimator and repair budget are fixed.
The original assignment is a legitimate possible answer, not a forbidden case.

## Measured results

Three existing CIFAR QResNet20 parents, source seeds2026/2027/2028, sites0/8.
The search fitted298unique local assignments across six site/seed problems and
scored12whole-network finalist combinations. It used64inner FIT and64disjoint
outer FIT images, no CHECK/validation for selection. Per-site final fits:

| Seed | Block0 local fits | Block8 local fits | Selected whole-network assignment |
|---|---:|---:|---|
|2026|50|50|Original at both sites|
|2027|49|50|Original at both sites|
|2028|49|50|Original at both sites|

Each site had an intact start and the predetermined free-role50% mixed start.
Every intact start stayed unchanged. Each mixed start's outer block MSE fell,
by6.59--26.68% across the six sites; these are improvements on the reused
selection objective, not held-out accuracy or independently confirmed gains.
None of those mixed-start finalists won the whole-network teacher-KL selection.

Final full500FIT refits reproduced the frozen original compact blocks bitwise
at all six sites. Reconstructed original full-model metrics and logits also
replayed exactly. Hence all three guided initializations were bitwise identical
to their historical original-group controls. The final results deliberately
reuse the verified original repaired checkpoints rather than waste three
identical supervised repair runs:

| Seed | Original final validation accuracy | Guided final validation accuracy | New SGD updates |
|---|---:|---:|---:|
|2026|89.8%|89.8%|0|
|2027|91.0%|91.0%|0|
|2028|89.6%|89.6%|0|

Mean accuracy gain0.00pp, mean CE delta0. The prospective practical+0.5pp screen
failed. The equality follows identical models and explicit checkpoint reuse;
it is not an equivalence test or three new independent training replications.
The earlier single-seed reassembly pilot's failed gate is unchanged.

## Interpretation

This particular adaptive grouping procedure did not improve over intact groups.
It improves some deliberately perturbed starts, but its best deployed choice is
the existing original assignment under the registered search and teacher score.
There is no new compression gain or demonstrated method novelty.

The result narrows the objection that only destructive random mixing was ever
considered: this pilot actually chose groups using fit data and allowed the
intact answer. It does not establish that intact groups are globally optimal or
that all intelligently chosen alternatives fail. Two starts, two rounds,
16shortlisted swaps per round and a64image selection subset are a small search.
The original teacher was itself trained in the Hamilton architecture, so that
assignment has a structural advantage; preserving it is not evidence of learned
semantic entities. Fixed masks were previously chosen from all500FIT images;
inner/outer separation is conditional on those inherited masks.

Projection cost shortlisting, local block MSE and full-network teacher KL may
each miss candidates useful for classification after repair. Candidates receive
an inner64image consumer fit before selection and a final500image refit after
selection; larger search or a different selection criterion was not tested.
No alternate dimension, algebra, architecture, extra repair or extra seed was
silently added. Existing CHECK/validation have been inspected before and are
exploratory. No official-test evaluation.

## Verification and closure

Seven CPU tests, a full synthetic search/refit/one-epoch repair/resume and a
discarded-copy compact GPU tracked-variable/quaternion gradient/update audit
passed before source freeze. The final audit passed538registered source/input
hashes and27output artifact hashes. All compact models have232266real
trainables against271594parents, approximately14.5%whole-model reduction.

The new experiment consists of grouping search and model selection; historical
repair is reused transparently because selected initializations are identical.
Its scientific failure is retained, not replaced by the favorable local-MSE
changes. This bounded pilot is closed with no automatic continuation.

## Literature implication

`Martinez2021PermuteQuantizeFineTune` already uses stochastic two-entry
permutation refinement for compression; `Zhang2021StructuredShuffle` learns
groups/shuffles, and `Singh2026` includes quaternion pruning with fine-tuning/KD.
The checked sources do not establish priority for our specific combined setup,
and this negative result supplies no basis for a new-method headline.
The narrow empirical-paper interpretation remains the appropriate candidate;
its contribution strength and broader scope are still not demonstrated by this
pilot alone. Canonical references and checked BibTeX are in the literature note.
