# PHM baseline environment migration and resume

Date: 2026-11-25. This report records infrastructure checks and an authorized
resume, not a new compression experiment or a novelty/performance claim.
Read the baseline state and recent log tail for current activity.

## Hypothesis and scope

The original from-scratch quaternion-stem/PHM4 ResNet-20 baseline can resume on
the relocated Windows installation with its registered versions, recipe, data
split and complete checkpoint intact. Complete the existing seed-2026,
200-epoch baseline before designing the learned-basis compression pilot.

## Measured checks

- The copied venv initially failed because its interpreter home referenced
  `C:\Users\jawor\AppData\Local\Programs\Python\Python310`.
- The new account has Python 3.10.11. Recreated venv launchers and activation
  scripts in place, without clearing packages or downloading dependencies.
  Refreshed 20 console entry points with the bundled pip distlib utility.
  Direct `pip.exe` and `pytest.exe` calls now resolve this repository's venv.
- Environment matches the registered versions: Python 3.10.11, PyTorch
  2.6.0+cu124, NumPy 2.2.6, CUDA build 12.4. GPU: RTX 4070 Laptop, 8188 MiB;
  NVIDIA driver 610.91. `pip check` passed. Full package list is in preflight.json.
- Six CPU tests passed in test_cifar_phm_resnet.py and test_cifar_resnet.py:
  shapes, parameter tracking/gradients/updates, component layout, reloads,
  physical widths, and exact interrupted-training replay. The first test
  invocation additionally named a nonexistent test_phm_conv.py, so collected
  no tests; the corrected invocation above passed.
- All 430 frozen protocol source/input hashes matched. No registered numerical
  sources, scientific configuration, protocol, dataset or split was changed.
- Checkpoint contains 45 complete epochs, optimizer momentum, AMP scaler,
  best model, and Python/NumPy/torch/CUDA/order/augmentation RNG states.
  Historic optimizer counts: 16875 attempted, 16870 successful updates.
  The five historic AMP-skipped updates remain recorded.
- Validation replay: epoch 45 accuracy 82.3%, CE difference 1.53e-8;
  best-CE epoch 33 accuracy 86.6%, CE difference 5.34e-8 from saved metrics.
  Neither model was fitted to validation. No official test inference.
- A discarded fresh full-width model passed all-variable and quaternion-component
  gradient/update checks and 15/15 synthetic AMP updates at batch size 128.
  Warmed synthetic batch time was approximately 0.0242 seconds; this is not
  an epoch-time measurement. Peak allocated memory was 275027456 bytes during
  this synthetic check, not a full-training memory benchmark.
- Preserved the epoch-45 checkpoint as checkpoint_epoch45.pt before launch.
  SHA-256: 0811953119511189ca678d4ea2532697eceb5f1759ccab956cc679e94d367619.

## Execution and limits

Removed the local STOP flag and launched the original command at 15:17
Europe/Warsaw:

```text
.venv-cuda\Scripts\python.exe -B -u -m scripts.resnet20_qstem_phm4_baseline --run
```

Logs: ../resnet20_qstem_phm4_baseline/runner_20261125_151729.log and .err.
The launcher PID and actual worker PID can differ on Windows; state.json holds
the worker PID. Launch metadata is in launch.json. To pause, create
results/resnet20_qstem_phm4_baseline/STOP; the last complete epoch is retained
and an interrupted partial epoch repeats on resume.

Checkpoint metric replays agree within the frozen 1e-6 CE tolerance, but the
small measured CE differences mean cross-machine bitwise equality is not
claimed. This resume does not change n=4, learn discrete dimensions, establish
novelty or launch a compression/replication queue. Existing source provenance
and prior art (He2016ResNet, ZhangEtAl2021, Grassucci2024) remain unchanged.
The November 8 timed-pause state is historical; no new timed pause is scheduled.

## First verified resumed epochs

The worker continued with epoch 46 (not epoch 1) and wrote complete epoch
checkpoints. Epochs 46, 47 and 48 took 20.2, 19.4 and 19.4 seconds respectively;
their validation accuracies were 87.8%, 85.4% and 83.5%. Epoch 46 improved the
CE-selected best checkpoint to CE 0.4052475. These are intermediate results,
not the selected outcome of the complete 200-epoch training. The original
epoch-45 log measured 60.5 seconds, so these first resumed epochs are roughly
three times faster; this is an observed run comparison, not an isolated GPU
benchmark. No stderr output was present at this check.
