# Frozen block analysis protocol

Recorded before running the analysis on 2026-11-02. Configuration:
`configs/block_analysis.json`. Source checkpoints: `results/cifar_cnn`, seed 2026.

## Hypotheses

Whole-quaternion activation similarity may predict individually low-impact
channel removals. This is not an assumed relationship or a new method.

## Measurements and checks

- Collect outputs after all three complete blocks, dropout disabled.
- Reconstruct initial models from the recorded seed; compare both initial and
  selected last-block activations against the original saved snapshots.
- Scores: centered grouped cosine, scalar Pearson, entropy effective rank,
  constant-feature counts. Calibration: the same 1000 held-out images.
- Split stability by whole calibration images before pooling spatial positions.
- Zero one complete quaternion unit (all four component channels) at a time;
  real control removes one scalar channel. Evaluate on the separate 1000-image
  validation set. These granularities are explicitly different.
- Compare zero-removal predictions with the unmodified full forward pass;
  verify exact parameter and BatchNorm-buffer preservation.
- Record nearest-neighbor similarity against negative loss increase (Spearman),
  and raw incoming norm against loss increase. No fitted score or p-value claim.

## Descriptive decision rule

Individual low impact means validation cross-entropy increase <=0.01 and
accuracy decrease <=0.005 (0.5 percentage points). These are practical,
prespecified thresholds, not literature-established constants.

A candidate quaternion block has at least two individual low-impact removals
and image-half pair-value stability >=0.9. Choose the largest low-impact
fraction, breaking ties by more quaternion units and then earlier block.
This licenses a controlled simultaneous-removal experiment, not a safe-pruning
claim. No candidates means report that result before broadening the study.

No test inference, optimizer updates, physical model-size savings or latency
claims. Preserve negative score-impact correlations. Further training seeds
and physical graph surgery are separate stages. Citation context uses existing
keys Ayinde2019, Vecchi2020 and Singh2026 in `references.bib`; the diagnostic
is not claimed as a faithful implementation of their full methods.
