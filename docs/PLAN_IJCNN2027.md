# IJCNN 2027 research plan

The current snapshot and links to measured results are in [CURRENT.md](CURRENT.md). The machine-readable [research queue](../configs/ijcnn_research_queue.json) records the full authorized stages and dependencies; it is an agenda, not an automatic scheduler. Its recorded statuses may age, so check the relevant state files and running processes before acting.

1. **Completed:** paired real and quaternion ResNet-20 baselines and a one-seed core compression comparison on CIFAR-10.
2. **Next:** predeclare and implement a focused independent-seed replication of the important real/Q comparisons, including pruning, map folding and direct fitting controls. The replication worker is not implemented. Minimize the scope while preserving a meaningful comparison and report the discovery seed separately.
3. **Then:** bounded Q/real hybrid and PHM catalogue under the same actual full-model parameter budget. The implemented two-block catalogue shares one runner across these entries and requires the completed replication summary; it has synthetic checks but no scientific CIFAR result. Compare all-Q, Q/real, fixed PHM and variable-n controls, with equal repair and held-out evaluation.
4. **Conditional:** extend PHM fitting or layer allocation only if evidence warrants it. CIFAR-100 is a separate later generalization branch after the core replication, not dependent on hybrid success.

Do not repeat unsuccessful tail/logit, learned-map or centered-map pilots without a new reason. Preserve negative results and distinguish actual compact sizes from masks. Novelty remains unverified; see [LITERATURE.md](../LITERATURE.md) and its archived audit.

The tentative contribution for Marcos is an unsent [draft](MARCOS_DRAFT_2026-11-07.md); authors remain to be agreed. The full previous plan, including historical and superseded stages, is preserved byte for byte in [the archive](archive/PLAN_IJCNN2027_2026-11-07.md); resolve its original relative paths from `docs/`.
