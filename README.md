# HypercomplexNNRedundancy

Research on structured compression of quaternion and learned-PHM neural networks,
with real-valued controls. The intended venue is IJCNN 2027; novelty and useful
compression gains must be established by evidence.

- [Current state and next action](docs/CURRENT.md)
- [Research plan](docs/PLAN_IJCNN2027.md)
- [Working title and paper evidence map](docs/PAPER_EVIDENCE_MAP.md)
- [Contribution audit against the literature](docs/literature/PAPER_CONTRIBUTION_AUDIT_2026-11-26.md)
- [Index of experiment reports](docs/RESEARCH_STATUS.md)
- [Scientific scope](PROJECT.md), [literature](LITERATURE.md), and [canonical bibliography](references.bib)
- [Agent instructions](AGENTS.md)

As of 2026-11-26, the ResNet-20 real/Q baselines, core compression comparison,
independent-seed replication, two-block structure catalogue, PHM initialization
diagnostic and from-scratch Q-stem/PHM4 baseline are complete. A registered
two-block pilot comparing preserved, adapted and reinitialized PHM bases after
compression is also complete, including uncompressed controls and a compact
scratch reference. Its adaptation continuation gate failed. See
[the state guide](docs/CURRENT.md). The Q component-reassembly pilot is also
complete: five matched compact models, two blocks and three repair epochs each.
The [registered plan](results/resnet_component_reassembly_pilot/PROTOCOL.md)
compares whole-unit selection, scalar energy and scalar regrouping. Free-role
reassembly reached90.8% versus90.4% for whole-unit selection; the registered
0.5pp continuation gate failed; it triggered no automatic replication.

A separately authorized [grouping mechanism study](results/grouping_mechanism_study/PROTOCOL.md)
completed on three existing Q parents. It holds the retained scalar set
fixed and separates consumer grouping, physical producer error and repair.
It contains138local comparisons and15matched compact models. Every mixed
full-model condition scored below its original-group control within each seed;
see the [measured report](results/grouping_mechanism_study/REPORT.md) for scope
and limits. A separately authorized
[initialization control](results/grouping_initialization_control/PROTOCOL.md)
compared the grouping penalty before/after training with crossed
feature/consumer states. All504fits and126predecessor replays completed.
The penalty already exists before training and grows afterwards; BN/statistical
state differences limit semantic interpretations.

The [BN/statistics control and EuroSAT check](results/grouping_statistics_control/PROTOCOL.md)
completed on two datasets/architectures and three existing parent seeds each:
1512reported rows,1260new fits, audit passed. After BN/marginal controls the
training-associated increase is inconsistent (6/12CIFAR,4/12EuroSAT comparisons).
Original grouping remains favored at initialization and after training, but
learned quaternion semantics are not established. See the
[interpretation](results/grouping_statistics_control/INTERPRETATION.md) and
[figure](results/grouping_statistics_control/figures/training_increment.png).
These are reconstruction diagnostics, not new accuracy or compression gains.

A separately authorized [BN-only full-network accuracy control](results/grouping_bn_accuracy_control/PROTOCOL.md)
completed on the15saved compact grouping-study models plus3parent controls.
BN-only calibration reduced the original-minus-mixed accuracy gap in3/12pairs;
mean gaps widened for all4mixed conditions. Free100% mean gap:9.90ppnative,
11.80ppBN-only,2.27ppafter historical3epoch repair. This particular recalibration
does not recover grouping damage; parent controls also degraded on average.
[Interpretation](results/grouping_bn_accuracy_control/INTERPRETATION.md).
No new SGD or test run, and no further computation queued.

The [fixed-set adaptive grouping pilot](results/grouping_exchange_pilot/INTERPRETATION.md)
also completed:298local fits and12whole-network combinations across3seeds.
The bounded search chose original groups at both sites for every seed, giving
bitwise-identical initializations and0ppgain. Historical identical repair was
reused; no new repair training or test inference. Its practical screen failed.

The separately authorized [EuroSAT whole-network grouping replication](results/eurosat_grouping_fp32_replication/PROTOCOL.md)
completed: nine actual compact models, three existing parents and three
grouping conditions, with equal three-epoch repair within seed.44794real
trainables versus73898parents. No official-test inference or new parent training.
It uses fresh deterministic FP32 references after the first version's archived
TF32 activation replay failed; that failed run and diagnostic remain preserved.
Final means93.33%original,86.57%role-preserving mixing,83.83%free mixing; all
six paired final comparisons favor original. [Interpretation](results/eurosat_grouping_fp32_replication/INTERPRETATION.md)
and [cross-dataset table/figure](results/eurosat_grouping_fp32_replication/CROSS_DATASET.md)
state the different budgets, AMP skips and limits. No further run queued.

## Windows CUDA environment

The verified environment uses Python 3.10.11, PyTorch 2.6.0+cu124 and NumPy 2.2.6.
For a fresh environment with Python installed at its usual per-user location:

```powershell
$python310 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
& $python310 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv-cuda\Scripts\python.exe -B -m pytest -q tests_torch/test_cifar_phm_resnet.py tests_torch/test_cifar_resnet.py
```

Activation is optional; use the venv's interpreter explicitly. After moving a
copied venv to another account or computer, recreate its interpreter/activation
paths and console entry points, or rebuild it with the registered versions.
The [migration report](results/phm_resume_20261125/REPORT.md) records the completed
repair and checks. Phase 0's separate TensorFlow/Keras environment and its failed
compatibility check are documented in [the feasibility report](results/phase0/FEASIBILITY.md).

## Reproducibility and navigation

Study protocols freeze numerical source/configuration hashes. Add new modules
for new experiments; do not edit registered numerical files. The JSON queues
and root `.cmd` launchers describe historical completed stages and are not the
current research plan. Check [CURRENT.md](docs/CURRENT.md) before launching work.

Git includes code, configurations, textual records, selected baseline models
and split indices. Datasets and most checkpoints are excluded; a clone alone
cannot resume or audit every study. Scientific reports and negative results
remain under `results/`. Superseded README, plan and handoff snapshots were
removed from the working tree during documentation cleanup; previous versions
remain in Git history.
