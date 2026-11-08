# Current research handoff

Snapshot: 2026-11-08. This is a navigation aid, not a live process monitor. Read the relevant `state.json` and recent log tail, and check the process before reporting current activity or ETA.

The PHM baseline is PAUSED after45completed epochs. The timed pause worked at12:50Europe/Warsaw on2026-11-08 so the GPU is free for the user's13:00-19:00classes. Timer state: `results/qstem_phm4_timed_pause/state.json` (training_stopped). No automatic restart after19:00. On renewed authorization remove the local `results/resnet20_qstem_phm4_baseline/STOP` and resume the original baseline command; its last completed epoch/optimizer/RNG are checkpointed, and a partial epoch repeats. Best CE-selected checkpoint so far: epoch33,validation86.6%; latest epoch45validation82.3%. These are intermediate results.

Current authorized direction: quaternion RGB encoding and Hamilton stem, independently learned PHM structures in deeper layers, trained FROM SCRATCH rather than converted from a Q teacher. First bounded model: `configs/resnet20_qstem_phm4_baseline.json`, `scripts/resnet20_qstem_phm4_baseline.py`, `src/models/cifar_phm_resnet.py`. One seed2026, fixed n4 in18residual convolutions,200epochs with the previous split/recipe. GPU preflight and2CPU tests passed;272746real trainables (Q baseline271594). Scientific baseline training has been started; live state/log/STOP: `results/resnet20_qstem_phm4_baseline/`. Resume command: `python -B -u -m scripts.resnet20_qstem_phm4_baseline --run`. This model learns PHM bases, not discrete n. No copied pretrained Q weights. Future compression should compare preserving versus readapting these learned bases; per-layer n selection is a separate unimplemented extension. No automatic follow-up is attached to this training.

Latest: all independent-seed baselines, the focused 50% compression replication (12 new cases) and the combined five-policy hybrid/variable-n pilot are COMPLETE. The queues finished; no later stage was launched by them. Replication results: `results/resnet20_key_replication/summary.json`. Pilot: `results/resnet_structure_catalogue/summary.json`.

The pilot selected the same Q/Q model for all-Q, Q/real and variable-n policies (89.3% validation). CPU audit: `results/structure_catalogue_audit/REPORT.md` and `audit.json`, runnable as `python -B -u -m scripts.audit_structure_catalogue`. Independent enumeration reproduced the optimum across 159 feasible pairs out of 210, checkpoint/count/forward checks passed and 145 frozen source hashes matched. PHM4 exactly contains the chosen Q kernels, but the original dense-oracle SVD initializer has worse activation error; its extra 128 basis coefficients and the coarse width grid also handicap the same-width alternative. Do not interpret this pilot as proof of quaternion superiority.

The preregistered initialization diagnostic is also COMPLETE: `configs/phm_initialization_pilot.json`, `scripts/phm_initialization_pilot.py`, `results/phm_initialization_pilot/REPORT.md`. Two blocks, Q control plus PHM4 from Q/SVD/three random seeds, 300 full-batch Adam steps each on 500 fit images, 500 separate check images. No classification/test inference. PHM from Q and SVD reach nearly identical held-out MSE: block0 ~0.7074 versus Q-control 0.75675 (~6.5% lower), block8 ~0.18502 versus Q-control 0.18604 (~0.55% lower). Two of three random starts work well in block0 and all three in block8; one block0 random start is worse than Q. All PHM variants store 64 extra basis coefficients per layer. Twelve cases and 24 artifact metric replays passed an independent hash/count/moment audit. Original PHM deficit was partly a fitting-procedure limitation. These local errors do not establish equal-budget accuracy gains. Next possible work is finer-width/budget-matched structure selection using activation-fitted candidates; not launched.

The resume and queue-launch notes below describe the completed execution history.

Resume update 2026-11-08: the seed-2027 Q baseline stopped on CUDA out-of-memory on 2026-11-07 and was left stopped at the user's request. On renewed authorization, baseline training resumed from the verified complete epoch-41 checkpoint (optimizer/scaler/RNG present). Both waiting processes were restarted in dependency order. Current logs in all three runner directories are `runner_20261108.log` / `runner_20261108.err`; original logs preserve the failure. Check live state before reporting progress. The GPU had only 27 MiB occupied before restart; external notebook jobs do not honor this project's GPU lock.

Update 2026-11-07 evening: independent-seed **baseline training** for seeds 2027/2028 has been started with `python -B -u -m scripts.resnet20_replication_baselines --run`. Configuration: `configs/resnet20_replication_baselines.json`. The runner trains real then Q for each seed, with per-epoch checkpoints and per-seed `state.json` under `results/resnet20_replication_baselines/seed_<seed>/`; logs are in its parent directory. To pause, create `STOP` in the active seed directory; resume by removing it and rerunning the same command. Four original CPU tests and both-seed real/Q GPU discarded-copy preflights passed.

The focused 50% compression replication is implemented and preregistered in `results/resnet20_key_replication/protocol.json`: real/Q, mapfold/pruned/direct, both seeds; 2026 is explicitly discovery. Three CPU tests and six-case synthetic end-to-end smoke passed. A waiting process (`python -B -u -m scripts.queue_resnet20_replication`) now starts replication after both baseline seeds complete and the GPU lock is free. Live queue status/log: `results/resnet20_replication_queue/`; create its `STOP` to stop the waiting process before launch. Replication checkpoints per site and repair epoch; place `STOP` in the active `results/resnet20_key_replication/seed_<seed>/` to pause it.

The combined Q/real hybrid and variable-n PHM pilot is also queued: `python -B -u -m scripts.queue_resnet_structure_catalogue` waits for the replication queue and both scientific predecessor summaries to complete, then for the GPU lock to be free. Queue status/log: `results/resnet_structure_catalogue_queue/`; its `STOP` cancels waiting before launch. Once the pilot is running, use `results/resnet_structure_catalogue/STOP` to pause it. This is ONE combined five-policy pilot, not two jobs. Queue gates passed tests, and all 145 frozen pilot source hashes matched readiness. Neither waiting process automatically restarts after computer shutdown; restart both after resuming baselines. No scientific replication or hybrid results yet.

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

The [queue](../configs/ijcnn_research_queue.json) describes the agenda and may contain historical execution statuses. The baseline, replication and bounded catalogue stages are now complete. Treat 2026 as discovery and retain all negative outcomes. Seeds use the same held-out data split, so do not describe them as independent datasets. Check trainability, splits, actual physical compactness, source hashes, checkpoint resume and parameter counts before future scientific runs.

The bounded catalogue is complete and audited; see the latest update above. Do not repeat it under its two queue labels. Weighted PHM fitting and full nine-block allocation remain conditional. Details: [catalogue report](../results/resnet_structure_catalogue/REPORT.md), [audit](../results/structure_catalogue_audit/REPORT.md), [plan](PLAN_IJCNN2027.md).

The draft response to Marcos is [here](MARCOS_DRAFT_2026-11-07.md); it has not been sent and the author list remains open. Do not assert firstness or superiority based on current evidence.

## Working efficiently and safely

- Follow [AGENTS.md](../AGENTS.md); before writing code read [PROJECT.md](../PROJECT.md) and [LITERATURE.md](../LITERATURE.md). `PROJECT.md` is the original project brief: where early-stage plans conflict with measured progress, consult this handoff and the queue for current status.
- Read only the relevant report, configuration or source. Use scoped `rg` and short log tails; do not read all results, archives or the full citation database on every task.
- Sources/configs registered in existing protocols are hash-frozen. Add new numerical files instead of editing those files. Do not move datasets, checkpoints, protocol files or result paths. Preserve negative results, seeds, versions and held-out splits.
- Git excludes datasets and many intermediate checkpoints; a clone is insufficient to resume or audit all past studies without local artifacts.
- [Historical handoff](archive/HANDOFF_2026-11-07.md), [historical plan](archive/PLAN_IJCNN2027_2026-11-07.md), [full literature audit](archive/LITERATURE_2026-11-07.md) and [previous README](archive/README_2026-11-07.md) are byte-for-byte snapshots. Their local relative links were written for their original locations. Read them selectively; historical status and next-step instructions can be superseded.
