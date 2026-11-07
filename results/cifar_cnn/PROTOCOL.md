# CNN quality gate: target 80% on CIFAR-10

Prepared before image-CNN training, 2026-11-02. Configuration:
`configs/cifar_cnn.json`. The target is unmeasured until the run completes.

## Hypotheses and scope

The dense pilot's approximately 38% accuracy does not establish that quaternion
models cannot learn CIFAR-10. This next quality check uses a conventional small
CNN and more training data. No novel architecture or performance advantage is
claimed. H1-H3 about redundancy and compression remain open.

## Architecture

Three spatial blocks, each containing two Conv3×3–BatchNorm–ReLU sequences,
MaxPool2×2 and Dropout2d with probabilities 0.1/0.2/0.3. Global average pooling
feeds a real ten-logit readout. Input images retain their 32×32 RGB layout.

- Quaternion channel widths: 8, 16, 32; scalar feature-map widths: 32, 64, 128.
  Six quaternion convolutions implement the standard right Hamilton product,
  using four real trainable kernel tensors per connection. RGB is represented
  as `(0,R,G,B)` per pixel, with component-blocked channels in the implementation.
- Real channel widths: 16, 32, 64, with RGB inputs and no unused zero-input weights.
- Real trainable parameter counts: quaternion **73,898**, real **72,954**.
  The real model has 944 / approximately 1.28% fewer parameters. Width and compute
  are not matched. The derived expanded quaternion convolution kernel is not
  counted as an independent trainable tensor.
- Ordinary scalar-component BatchNorm and Dropout2d are used in both families.
  No quaternion rotation invariance or special normalization theorem is claimed.

The convolutional formulation is contextualized by `NiemczynowiczKycia2026`;
this remains an existing network family, not a proposed new method. Dataset:
`Krizhevsky2009` in `references.bib`.

## Data and optimization

Reuse the dense pilot's 1000 validation and 1000 calibration indices. Train on
all remaining **48,000** official training images. Preserve all 10,000 official
test images for the end of this CNN run. That official test set was previously
evaluated by the dense pilot, so it is not a newly collected blind benchmark.

Fit channel means/standard deviations on the 48,000 training images only.
Training augmentation: individual random crop after four pixels of zero padding,
then horizontal flip. Validation, calibration and test are not augmented.
Alpha is added as exact zero after RGB normalization.

One seed, 2026. AdamW, learning rate 0.001, weight decay 0.0001, batch 128,
50 epochs; cosine decay to 0.00001. Dedicated batch and augmentation generators
share streams between model families. GPU mixed precision is enabled; the first
actual-batch parameter/gradient/update audit is float32. Select by minimum
validation cross-entropy, including initialization. Complete both model trainings
and checkpoint selection before test inference. No test-dependent tuning.

## Checks, outputs and limits

Before training: all 16 Hamilton basis products, independent NumPy spatial
convolution, numerical input/kernel gradients, component updates with zero alpha,
serialization, spatial activation grouping and interrupted-training resume.
CPU tests passed before installation finished; repeat on the actual GPU runtime
and measure GPU memory/time before committing to the full run.

Save every-epoch history and last checkpoints (model, optimizer, scheduler,
scaler, RNG streams); save the validation-selected model separately. Record
versions, driver/device, seed, source hashes, trainable real parameters and
BatchNorm buffers separately. Preserve failed checks rather than relabeling them.

Collect initial and selected post-activation spatial features on calibration only.
For stability, split by whole images before pooling spatial positions. Report
grouped cosine, scalar Pearson, effective rank and inactive-feature counts. No
pruning is included; compact-architecture savings remain zero.

This one-seed quality gate cannot establish a quaternion gain or seed stability.
Changes from the earlier dense pilot include architecture, data amount,
augmentation, optimization and framework; an accuracy difference is not a
controlled estimate of architecture alone. Once quality is adequate, a later
multi-seed study can examine compression under a frozen protocol.

## Environment decision and acquisition notes

Hardware inspection: RTX 3050 Ti Laptop GPU, 4096 MiB, NVIDIA driver 555.99.
Existing TensorFlow and global PyTorch installations use CPU. The new
`.venv-cuda` uses PyTorch 2.6.0 with bundled CUDA 12.4, selected from official
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)
and the installed driver's support. Old environments remain separate.
The CUDA build and actual GPU execution must be verified, not inferred from
successful installation. No driver or system CUDA installation is changed.

Initial pip access failed with sandbox socket error 10013 and was retried with
network approval. The resolver rejected two mirror metadata variants and used
alternate package entries. These acquisition messages are distinct from model
checks. Full package versions will be frozen after successful setup.

## Measured setup checks

PyTorch 2.6.0+cu124 reports CUDA 12.4 and the RTX 3050 Ti; an actual CUDA
matrix multiplication/backward pass produced finite gradients. `pip check`
reported no broken requirements. The ten mathematical/training/serialization/
resume tests passed in the new CUDA-capable runtime (CPU test tensors).
The separate actual GPU benchmark verified all kernel component gradients and
updates on CUDA tensors for both full model families.

For 20 synthetic training batches of 128 images, measured seconds per batch
were 0.0463 (quaternion) and 0.0142 (real). Peak PyTorch allocated memory was
approximately 145 and 108 MiB respectively. These measurements plan run cost;
they do not establish a latency advantage or the full process's VRAM usage.
The benchmark excludes image augmentation, validation and checkpoint overhead.
See `benchmark.json` and `pytest.xml`.
