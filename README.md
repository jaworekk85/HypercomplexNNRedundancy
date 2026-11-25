# HypercomplexNNRedundancy

Research on structured compression of quaternion and learned-PHM neural networks,
with real-valued controls. The intended venue is IJCNN 2027; novelty and useful
compression gains must be established by evidence.

- [Current state and next action](docs/CURRENT.md)
- [Research plan](docs/PLAN_IJCNN2027.md)
- [Index of experiment reports](docs/RESEARCH_STATUS.md)
- [Scientific scope](PROJECT.md), [literature](LITERATURE.md), and [canonical bibliography](references.bib)
- [Agent instructions](AGENTS.md)

As of 2026-11-25, the ResNet-20 real/Q baselines, core compression comparison,
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
and limits. No further computation is queued.

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
