# Current research handoff

Snapshot: 2026-11-07. This is a navigation aid, not a live process monitor. Read the relevant `state.json` and recent log tail, and check the process before reporting current activity or ETA. Do not assume a stage has resumed automatically.

## Research question and measured state

We study structured post-training compression of quaternion CNNs, including how compensation and structure affect quality and actual parameter count, with real CNN controls. Phase 0 and many earlier studies are complete. The current paper focus is compression; the separate SSL/few-shot project is not part of this queue.

ResNet-20 CIFAR-10 baselines and all 12 core compression cases are complete for **one discovery seed (2026)**. Selected validation accuracy was 90.9% for real and 91.0% for Q. Within each family and compression fraction, pruning-only, shared-map folding and direct consumer reconstruction use the same removed channels and three epochs of repair.

| Removed internal channels | Real: map fold / direct after repair | Q: map fold / direct after repair |
| --- | --- | --- |
| 25% | 89.2% / 89.1% | 88.6% / 88.3% |
| 50% | 82.9% / 84.9% | 81.8% / 82.9% |

These are validation results for one seed, not evidence of statistical significance or equivalence. The real and Q models have similar but not identical actual parameter counts. The independent structural/hash audit passed but did not independently replay full validation predictions; the scientific runner did replay validation for final checkpoints. No new test evaluation in this stage. See [core report](../results/resnet20_compression/REPORT.md), [state](../results/resnet20_compression/state.json), [audit](../results/resnet20_compression/independent_audit.json) and [baseline report](../results/resnet20_baselines/REPORT.md).

Earlier three-seed compact CNN results and negative pilot findings are in the relevant study reports and [generated research status](RESEARCH_STATUS.md). Do not infer that quaternion structure itself explains earlier shared-map results; see [literature index](../LITERATURE.md).

## Next authorized work

The [queue](../configs/ijcnn_research_queue.json) is a research agenda, not an automatic scheduler. First predeclare and implement a focused independent-seed ResNet replication with the necessary controls. The replication worker/config do not yet exist. Treat 2026 as discovery and retain all negative outcomes. Check trainability, splits, actual physical compactness, source hashes, checkpoint resume and parameter counts before scientific runs.

The bounded hybrid Q/real/PHM catalogue is implemented and synthetically tested, but has **no scientific CIFAR result**. Its shared runner requires completed baseline, core compression and `results/resnet20_key_replication/summary.json`; replication is not yet implemented. The two-block pilot has matched-budget controls; do not launch it twice for the two queue entries. Weighted PHM fitting and full nine-block allocation remain conditional. Details: [catalogue report](../results/resnet_structure_catalogue/REPORT.md), [plan](PLAN_IJCNN2027.md).

The draft response to Marcos is [here](MARCOS_DRAFT_2026-11-07.md); it has not been sent and the author list remains open. Do not assert firstness or superiority based on current evidence.

## Working efficiently and safely

- Follow [AGENTS.md](../AGENTS.md); before writing code read [PROJECT.md](../PROJECT.md) and [LITERATURE.md](../LITERATURE.md). `PROJECT.md` is the original project brief: where early-stage plans conflict with measured progress, consult this handoff and the queue for current status.
- Read only the relevant report, configuration or source. Use scoped `rg` and short log tails; do not read all results, archives or the full citation database on every task.
- Sources/configs registered in existing protocols are hash-frozen. Add new numerical files instead of editing those files. Do not move datasets, checkpoints, protocol files or result paths. Preserve negative results, seeds, versions and held-out splits.
- Git excludes datasets and many intermediate checkpoints; a clone is insufficient to resume or audit all past studies without local artifacts.
- [Historical handoff](archive/HANDOFF_2026-11-07.md), [historical plan](archive/PLAN_IJCNN2027_2026-11-07.md), [full literature audit](archive/LITERATURE_2026-11-07.md) and [previous README](archive/README_2026-11-07.md) are byte-for-byte snapshots. Their local relative links were written for their original locations. Read them selectively; historical status and next-step instructions can be superseded.
