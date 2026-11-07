# Paper A: evidence package after the focused literature review

## Hypotheses and fixed question

When does compensation of removed quaternion units help while preserving the original structured layer family and physical parameter budget? Does its advantage survive BN-only recalibration and brief labeled repair? No new compression method, quaternion exclusivity or literature priority is assumed.

## Measured results

The original seven-point CIFAR/EuroSAT curves, three seeds, QNN and real controls, circulant control and activation study are complete. This follow-up adds no new weight training. It extracts 3216 saved recovery observations (730 artifacts), recalibrates 42 original QNN states using both fixed BN estimators (84 interventions), and evaluates 333 frozen existing checkpoints on test data. Every final checkpoint reproduced its original validation metrics before test inference; saved test logits/labels were independently checked against the frozen targets and reported metrics.

BN control uses 2048 unaugmented training images/seed, identical across policies, no labels/gradients/weight updates. Cumulative BN: one pass, simultaneous train-mode BNs; global BN: six passes, sequential evaluation-mode moments. Both reported. Calibration is an extra resource beyond the 500 compression-fit images. All parameter/buffer invariance audits passed.

| Dataset | Units removed/site | Global BN-only: comp minus same-victim pruning (pp) | Three paired seeds (pp) |
| --- | ---: | ---: | --- |
| cifar | 0.25 | +12.10 | +17.10, +7.50, +11.70 |
| cifar | 0.5 | +11.20 | +13.10, +7.50, +13.00 |
| eurosat | 0.25 | +29.53 | +27.50, +32.80, +28.30 |
| eurosat | 0.5 | +34.93 | +19.50, +46.40, +38.90 |

### Frozen test results: QNN after three repair epochs

EuroSAT: 5400 final held-out tiles, with a random tile split (not geographic transfer). CIFAR: official 10000-image test, already seen in earlier pilots. Test levels .25/.5/.75 were chosen from validation outcomes and frozen before test; all seven fractions remain validation evidence. Scratch uses 50 epochs and frozen best-validation-CE selection; these are different student-training budgets and pretrained-parent cost is additional.

| Dataset | Units removed/site | Real parameters | Comp + 3 epochs (%) | BN magnitude + 3 epochs (%) | Scratch 50 epochs (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| cifar | 0.25 | 42,034 | 72.30 | 70.63 | 72.62 |
| cifar | 0.5 | 19,098 | 54.72 | 48.98 | 64.53 |
| cifar | 0.75 | 5,090 | 19.75 | 23.11 | 47.30 |
| eurosat | 0.25 | 42,034 | 93.85 | 92.34 | 95.07 |
| eurosat | 0.5 | 19,098 | 85.68 | 79.78 | 93.04 |
| eurosat | 0.75 | 5,090 | 51.02 | 44.28 | 86.01 |

### Paired test comparisons

| Dataset | Family | Removed/site | Comparison after 3 epochs | Mean (pp) | Individual seeds (pp) |
| --- | --- | ---: | --- | ---: | --- |
| cifar | circulant | 0.25 | greedy_comp - greedy_pruned | +0.92 | +0.89, +1.11, +0.76 |
| cifar | circulant | 0.25 | greedy_comp - bn_pruned | +4.15 | +3.28, +2.68, +6.48 |
| cifar | circulant | 0.5 | greedy_comp - greedy_pruned | +4.43 | +5.33, +4.49, +3.47 |
| cifar | circulant | 0.5 | greedy_comp - bn_pruned | +5.16 | +2.92, +3.74, +8.81 |
| cifar | quaternion | 0.25 | greedy_comp - greedy_pruned | +1.68 | +2.50, +1.83, +0.70 |
| cifar | quaternion | 0.25 | greedy_comp - bn_pruned | +1.67 | +2.10, +2.20, +0.70 |
| cifar | quaternion | 0.5 | greedy_comp - greedy_pruned | +3.47 | +4.33, +1.74, +4.35 |
| cifar | quaternion | 0.5 | greedy_comp - bn_pruned | +5.74 | +3.55, +5.98, +7.68 |
| cifar | quaternion | 0.75 | greedy_comp - bn_pruned | -3.36 | +0.39, -3.02, -7.46 |
| eurosat | quaternion | 0.25 | greedy_comp - bn_pruned | +1.51 | +1.81, +1.61, +1.09 |
| eurosat | quaternion | 0.5 | greedy_comp - bn_pruned | +5.90 | +6.02, +4.69, +6.98 |
| eurosat | quaternion | 0.75 | greedy_comp - bn_pruned | +6.74 | +8.94, -0.93, +12.20 |

## Interpretation and limits

1. In the registered moderate-compression subset, BN-only repair does not erase the paired compensation advantage. This rules out the simplest 'only stale BN' account for those outcomes, not every alternative explanation.
2. Most labeled recovery appears in the first stored epoch; the curves do not resolve the first few updates. No short-step or time-saving claim follows. At severe compression the CIFAR compensation advantage reverses on average, and compact scratch models often achieve much higher accuracy with their larger training budget.
3. Circulant compensation also helps in matched four-parameter blocks. This supports discussing an effect of constrained weight sharing; it neither proves equivalence of structures nor establishes a special quaternion benefit. Absolute full-model quality differs across families.
4. Three seeds, small CNNs, two image datasets and repeated validation use limit generality. Refinement fractions and new controls were chosen after earlier outcomes. No inference speedup/MAC advantage was benchmarked. Equal attempted training updates do not establish equal successful AMP steps.
5. Scientific contribution remains an empirical characterization of the cost, useful regimes and failures of structure-preserving QNN compression. Acceptance and novelty are not guaranteed. No further training or parameter choice is justified merely by these test outcomes.

## Literature decision and figure files

The focused review and claim matrix are in ../../LITERATURE.md. Canonical citations: GRAIL2026 (CPAL/PMLR; ridge reconstruction and BN REPAIR/FT comparisons already known), Kim2020/Asiaee2026CMR (merging/replacement/folding), Vecchi2020/Singh2026 (QNN compression), Li2020EagleEye (adaptive BN), Liu2019RethinkingPruning (scratch baseline), Valle2024VNets/Ding2017CirCNN (structured sharing). No full published-method reproduction was performed.

- [Seven-point quality–size figure](quality_size.pdf): Q/real, both datasets, validation means with ±1 sample SD across seeds. Logarithmic parameter axis. Full pretrained baseline marked separately; no scratch run at .125.
- [Recovery figure](recovery.pdf): epoch 0/1/2/3, measured means; missing same-victim continuations are not fabricated.
- [BN control figure](bn_control.pdf): paired advantages, error bars = sample SD of three seed differences, not confidence intervals.
- [All final test outcomes](../paper_a_final/REPORT.md), protocol and prediction hashes; [independent prediction audit](final_prediction_audit.json).

Next task is editorial: select a small set of these results and draft the compression paper if requested. Marcus's November 11 request requires a tentative title, authors and short abstract; authorship and correspondence are not automatically finalized.
