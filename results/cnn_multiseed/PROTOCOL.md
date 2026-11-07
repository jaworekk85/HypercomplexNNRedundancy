# Three-seed confirmation protocol

Frozen before new training on 2026-11-02. Seed 2026 is reused, not retrained.
Train seeds 2027 and 2028, quaternion and real models, changing only `seed`
in the original `results/cifar_cnn/config.json`. Same 48,000 training,
1000 validation and 1000 calibration images, normalization and 50 epochs.

Complete all four new trainings and minimum-validation-loss checkpoint
choices before the new test evaluations. Report all three seeds and paired
differences, with sample SD; do not select a winning seed. The official
test set has already been used in prior pilots and is not a new blind test.

Epoch checkpoints retain optimizer/scheduler/scaler and RNG state; the
runner resumes automatically with identical protocol/source hashes.

The seed-2026 block-analysis gate selected block 3 (24/32 individually
low-impact ablations). Subsequent compression stays at this block for every
seed, regardless of later test results. Similarity was a weak predictor in
block 3 (Spearman about 0.22), negative in block 2; preserve these findings.
No new architecture/score or performance advantage is presumed.

Execution note: a separate `scripts/cnn_training_worker.py` real-2028 job
overlaps the main runner's other-model jobs on the same GPU. It calls the
unchanged `train_one` with the exact frozen seed/configuration, has its own
checkpoint folder and performs no test inference. The main runner reuses the
completed result. Its wrapper hash/configuration are preserved in `worker.json`.
This changes scheduling only; no numerical training setting or data split.

The overlap was stopped after real-2028 epoch 3 because quaternion epoch time
increased from about 19s to 42-46s. Its full checkpoint remains intact; the
main runner will resume real-2028 from epoch 4 sequentially. The incomplete
fourth epoch, if started, is discarded exactly as in the standard pause flow.
