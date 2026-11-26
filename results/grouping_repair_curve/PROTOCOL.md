# Matched grouping repair trajectories

## Hypothesis

Original grouping may maintain a classification advantage after longer matched supervised repair, or mainly accelerate recovery. Both directions and reversals are reportable.

## Scope

18 trajectories: two existing settings, three existing parents each, three frozen assignments. Retained scalar masks, physical architectures and initial fitted weights are inherited unchanged. No new parent, mask, draw, component selection, scratch model, compression ratio, algebra or dimension. Actual real trainables232266CIFAR/44794EuroSAT. Previously inspected validation and reconstruction-check splits remain exploratory.

## Schedule

Restart each case from its verified pre-repair compact initialization, not the previous repaired model. One30epoch cosine schedule from1e-4to1e-6per trajectory, same AdamW, augmentation, order streams and AMP recipe within setting/seed. The historical3epoch schedule reached its minimum at3and remains a separately labelled comparator. New epoch3is not historical epoch3. No best-epoch selection or outcome-driven extension.

## Primary

Original-minus-role/free validation accuracy gaps in percentage points and mixed-minus-original CE at final epoch30, separately by setting and source seed. Report descriptive means across3source parents. Snapshot0/1/3/10/30metrics and reconstruction-check results describe recovery dynamics. Checkpoints, examples and policies are not independent training seeds. No significance, practical equivalence, learned-semantics, novelty or superiority claim follows automatically.

## Data

Reuse frozen TRAIN/validation/FIT/CHECK partitions and preprocessing. Only TRAIN drives supervised updates; no test forward pass or test metric. All18epoch0models must replay predecessor validation/CHECK logits bitwise at the same batch size. No classification draw or final epoch selected using validation/CHECK. FIT is not reused to optimize masks or weights here.

## Checks

Before freeze: CPU tests including dropout/RNG-preserving snapshots, exact interrupted epoch resume and checkpoint-with-missing-snapshot recovery; full synthetic trajectory/completed resume; discarded compact CUDA tracked-variable/all-four-component gradient/update audits. After freeze: all18initial-model/logit replays before any new SGD, per-case discarded update audits, every requested snapshot model/logit reload, source/input checksums and final artifact audit. Do not edit frozen predecessors; record all failed checks.

## Resume

Reuse frozen epoch-resumable trainer unchanged. Snapshots save compact states, per-example validation/CHECK logits/labels, metrics, actual attempted/successful updates and hashes. Snapshot/reload work restores Python/NumPy/CPU/CUDA RNG and model mode. If a completed-epoch checkpoint exists without its requested snapshot, recover that snapshot before resuming. STOP checked at batches; interrupted partial epoch restarts from its last completed checkpoint. Atomic JSON/model writes and completed-case hash verification. No automatic follow-up after18runs.

Prior art: He2017ChannelPruning, Vecchi2020, Singh2026, Comminiello2024InductiveBiases, Lakens2017Equivalence; canonical references.bib.
