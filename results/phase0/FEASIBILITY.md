# Phase 0 technical feasibility

Date: 2026-11-02. Scope: the first actionable task in `PROJECT.md`.

## Hypotheses

Gate A asks whether a tiny quaternion classifier has correctly tracked weights,
finite gradients, and actual optimizer updates. H1-H3 about redundancy and
compression remain untested research hypotheses. A synthetic startup check
cannot establish a performance or compression advantage.

## Upstream inspection

The implementation candidate is HypercomplexKeras 1.0.2, associated with
`NiemczynowiczKycia2025` in `references.bib`.

- Package repository: <https://github.com/rkycia/HypercomplexKeras>, commit
  `566f887a7039637860089863b5abfa13b1948267`.
- KHNN: <https://github.com/rkycia/KHNN>, commit
  `d2c532dbca8fe6c18355dbbc1f5749e7e0e4af32`.
- SoftwareX archive: <https://github.com/ElsevierSoftwareX/SOFTX-D-25-00004>,
  same commit `d2c532dbca8fe6c18355dbbc1f5749e7e0e4af32`.
- License: Apache-2.0 (`LICENCE` in the package repository; `LICENSE` in KHNN).
  No upstream source is vendored or modified here.
- [Package metadata](https://github.com/rkycia/HypercomplexKeras/blob/566f887a7039637860089863b5abfa13b1948267/pyproject.toml)
  requires Python >=3.8 but declares no runtime dependencies or version bounds.
- [Dense example](https://github.com/rkycia/HypercomplexKeras/blob/566f887a7039637860089863b5abfa13b1948267/examples/ExampleHyperdense.ipynb)
  suggests TensorFlow 2.12.0 in a commented installation cell. Notebook metadata
  records Python 3.10.12. It trains a four-example binary classifier. This is
  historical example context, not a verified modern compatibility guarantee.
- [Dense source](https://github.com/rkycia/HypercomplexKeras/blob/566f887a7039637860089863b5abfa13b1948267/src/HypercomplexKeras/Hyperdense.py)
  creates the kernel with `tf.Variable` while the bias uses `add_weight`.
  This warrants an explicit Keras 3 tracking check. The default seeded initializer
  is also called four times, potentially yielding identical component kernels.
- The [official Keras 3 migration guide](https://keras.io/guides/migrating_to_keras_3/#tensorflow-variables-tracking)
  documents that assigning a raw `tf.Variable` as a layer attribute no longer
  automatically tracks it. It recommends `add_weight` or `keras.Variable`.
  This establishes a source-level incompatibility to verify experimentally.
- Upstream layout is component-blocked: `[all real, all i, all j, all k]`.
  Its contraction implements right multiplication `x * weight`.

## Measured results

The package installed successfully. `pip check` reported no broken requirements.
The unmodified upstream audit **failed Gate A** on TensorFlow 2.20.0 / Keras
3.11.3 / Python 3.10.11 / NumPy 2.2.6, CPU, seed 2026:

| Check | Observed |
| --- | --- |
| Output shape | `(16, 12)`, correct for 3 quaternion output units |
| Expected real trainable parameters | 36 (24 kernel + 12 bias) |
| Registered real trainable parameters | 12 (bias only) |
| Raw kernel gradient norm | 1.6065946817 (finite and nonzero) |
| Kernel maximum update after standard SGD step | **0.0** |
| Bias maximum update | 0.0011455100 |
| Default four component kernels identical | true |
| Installed Dense source matches inspected Git blob | true |

See `upstream_smoke.json` for full provenance and individual checks.

## Local implementation decision

The documented Keras 3 incompatibility justifies the permitted local fallback:
`src/models/quaternion_dense.py`, a quaternion-only Dense layer using
`add_weight`. No upstream files were patched, and the project does not rebuild
the general KHNN framework. The fallback uses grouped `(batch, units, 4)`
tensors and the right Hamilton product `x * weight`. The quaternion kernel is
initialized in one call rather than reusing the same seeded initializer on each
component. These are implementation choices, not a proposed novel method.

## Measured local fallback results

Configuration: `configs/phase0.json`. One seed (2026), CPU, 240 full-batch Adam
updates at learning rate 0.01, split component-wise `tanh`. Inputs are eight
Gaussian real scalars grouped consecutively into two quaternions, without
padding; labels are the argmax of a fixed random linear teacher. Separate RNG
streams produce 256 training, 128 calibration and 128 test examples. Neither
calibration nor test examples participate in optimizer updates, model selection,
or stopping; the synthetic test metrics are computed after the fixed training.

| Quantity | Observed |
| --- | --- |
| Hidden quaternion units | 6 (24 scalar activation channels) |
| Quaternion-layer real trainable parameters | 72 = 48 kernel + 24 bias |
| Ordinary real readout parameters (3 logits) | 75 |
| Total real trainable parameters | 147, all registered |
| Initial / final training cross-entropy | 1.1790731 / 0.0413821 |
| Training accuracy | 1.0 |
| Held-out synthetic test cross-entropy / accuracy | 0.1797259 / 0.9140625 |
| Kernel component gradient norms at first step | 0.368896, 0.322680, 0.257688, 0.252700 |
| Kernel component update norms at first step | 0.034639, 0.034621, 0.034632, 0.034637 |
| Post-activation calibration tensor | `(128, 6, 4)`, before and after training |
| Maximum checkpoint reload output difference | 0.0 |
| Checkpoint bytes (model configuration and weights) | 19,908 |
| Masked parameters / removed units / compact savings | 0 / 0 / 0 |

`local_smoke.json` contains every check, source hashes, counts and version
information. `synthetic_data.npz` preserves the actual inputs and teacher;
`environment.freeze.txt` preserves all installed versions. The saved model is
uncompiled and does not include the external training optimizer's state.

The test suite checks an independent NumPy Hamilton reference, all 16 basis
products, finite differences for input/kernel gradients, graph-mode updates of
every component, source forward equivalence with explicit layout conversion,
extra leading dimensions, checkpoint roundtrip, and degenerate similarity inputs.
All **16 tests passed**; see `pytest.xml` for the executed suite. Keras emits two `DeprecationWarning`s
about NumPy 2's `__array__(copy=...)` interface during model serialization;
the roundtrip test passes. Warnings are not suppressed.

Two process runs with the same configuration reproduced identical metrics,
first-step measurements and both activation arrays (maximum absolute difference
0.0); see `reproducibility.json`. This is repeatability on the same CPU/software
stack, not evidence of stability across seeds, devices or platforms.

## Measured activation diagnostic

`similarity_matrix.png` and `redundancy_histogram.png` use calibration data only.
For each unit, each of its four components is centered over calibration samples;
the resulting sample-by-component block is flattened into a vector and compared
by ordinary cosine. Diagonals and duplicate pairs are excluded from the
histogram. Constant units yield undefined rows/columns and are excluded rather
than treated as similar; none occurred here.

There are only 15 distinct unit pairs. Maximum signed off-diagonal similarity
was 0.642172 initially and 0.229345 after training; mean similarities were
-0.090238 and -0.129274.

## Interpretation and limits

The nonzero raw upstream kernel gradient alone is insufficient: standard
optimization over `layer.trainable_variables` omits the kernel. The failure
is preserved, not relabeled as a passing upstream test.

The similarity calculation is a transparent reference diagnostic, not a novel
quaternion metric or an invariant similarity measure.
This toy run does **not** demonstrate strongly positively
aligned duplicate units under the chosen diagnostic. It does not establish a
general negative result about QNN redundancy either. Similarity is not proof of
functional unimportance or safe removability.

Gate A passes for the local implementation. Gates B-D and H1-H3 remain open:
there is no image dataset result, matched real-valued control, multi-seed
scientific study, structured pruning, compute/latency measurement or demonstrated
compression gain.

## One proposed controlled experiment (not executed)

After a small Phase 1 MNIST baseline and a held-out calibration check show a
reason to proceed, compare **25% removal of whole quaternion hidden units**
using random selection, incoming-weight group norm, and established pairwise
activation similarity (`Ayinde2019`; quaternion weight-based context:
`Vecchi2020`, `Singh2026`). Use the same trained starting checkpoint for all
three policies within each of at least three seeds. Calibration is drawn from
the training portion and disjoint from training and final test examples. Define
the four-component image encoding before training and freeze all choices before
test evaluation. Physically rebuild narrower layers and the connected readout;
first check zero-removal equivalence. Compare pre-fine-tune loss/accuracy against
actual remaining real parameter counts and checkpoint bytes at the same size.
No fine-tuning is needed for this first comparison. This is a proposal to test
H3, not a claim of novelty, usefulness or guaranteed signal.

## Execution notes

The supplied directory initially contained only project documents and a
bibliography; it was not a Git repository. The global Python was CPython 3.10.11
without TensorFlow or Keras. A project-local `.venv` was created.

The initial pip attempt failed with Windows socket permission error 10013
because network access was sandboxed. Installation was retried with explicit
network permission. This environment failure is separate from layer correctness.
