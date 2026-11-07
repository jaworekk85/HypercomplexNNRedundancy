# Frozen RGB pilot protocol

Prepared before image-model training, 2026-11-02. Configuration:
`configs/rgb_pilot.json`. This extends the passing local Phase 0 implementation.

## Hypotheses and scope

Inspect trainability and held-out activations on small real RGB data before
considering compression. H1-H3 in PROJECT.md remain hypotheses. This is a small
dense pilot, not a full CIFAR-10 benchmark, a reproduction of the malaria CNN,
or evidence of novelty or compression savings.

## Fixed choices

- CIFAR-10 (`Krizhevsky2009`): original 32×32 RGB images, no resizing or augmentation.
- From the official 50,000 training images: disjoint stratified 5000 training,
  1000 validation and 1000 calibration images. Split seed 471, fixed for all models.
  Leave 43,000 images unused. Final test: all 10,000 official separate test images.
- Normalize each RGB channel using only selected training pixels. Alpha is zero
  after normalization. One pixel is one quaternion, in row-major spatial order.
- Quaternion ARGB `(0,R,G,B)` and ABGR `(0,B,G,R)`; the latter is the sensitivity
  check motivated by the actual authors' notebook. See INPUT_ENCODING_REVIEW.md
  and `NiemczynowiczKycia2025`, `NiemczynowiczKycia2026`.
- Quaternion model: 1024 input quaternion units, 32 hidden quaternion units
  (128 scalar post-ReLU features), ordinary real ten-logit readout. **132,490**
  real trainable parameters.
- Real model: flattened RGB, 43 hidden ReLU units, ten-logit readout. **132,579**
  trainable parameters, **89 / 0.0672% more**. No padded zero input columns,
  masks or dummy parameters. This matches parameter budget approximately,
  not hidden width or compute.
- Hidden weights use RandomNormal variance `2/(3*32*32)` for both families;
  biases zero; readout GlorotUniform. This is an explicit variance choice for
  three active channels, not a new initialization method.
- Model seeds 2026, 2027, 2028. Within each seed, batch ordering is identical
  for all variants. Adam learning rate 0.001, batch 128, 20 epochs, CPU;
  two intra-op threads and one inter-op thread. No hyperparameter search.
- Select the checkpoint with lowest validation cross-entropy, including epoch 0.
  Calibration affects neither optimization nor checkpoint selection.
  Finish every training before evaluating final test metrics. Do not use test
  outcomes to choose an encoding or alter training.

## Measurements and checks

Audit all variables/gradients/updates on the first actual image batch; audit
all four quaternion kernel components despite the zero real input channel.
Check checkpoint reload outputs. Save raw metrics, histories, split indices,
normalization, activations, test logits, versions and source hashes.

On calibration only, collect initial and selected post-activation features.
Compute scalar Pearson, centered whole-quaternion cosine, constant/zero feature
counts, and entropy effective rank of scalar-feature covariance. Report pair
fractions at a fixed descriptive threshold 0.9. Exclude diagonal/duplicate pairs
and undefined constant-feature pairs. Compare corresponding pair values across
even/odd calibration halves within each model. Neurons across seeds are not aligned.

Report per-seed results and means/sample standard deviations, with paired seed
differences. Three seeds on one split do not establish broad generalization.
Compare calibration distributions while retaining the distinction between
scalar channels and whole quaternion units. Similarity does not establish
removability. No pruning or fine-tuning is included.

## Environment and acquisition notes

The Phase 0 report preserves the upstream HypercomplexKeras/Keras 3 tracking
failure; this pilot uses the tested local fallback. The initial CIFAR download
was blocked by sandbox networking. After approval, the author server was slow
and its partial transfer was interrupted. The TensorFlow storage mirror then
completed; Keras verified the official SHA-256
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
Source probes for two other mirror filenames returned 404. These are acquisition
issues, not model or scientific results. A Windows CIM hardware query was denied;
Python reports 12 logical CPUs. CPU/GPU visibility is recorded by TensorFlow.
