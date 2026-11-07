# Tail-sensitive quaternion compensation: one late-consumer pilot

## Hypothesis

At one prespecified late Q consumer, logit-sensitive unlabeled refinement may improve final compact accuracy relative to equal-step local reconstruction refinement and the reused direct-Q solution, while retaining Hamilton weights and physical parameter count.

## Measured results

Status: completed. Scientific results: True. CIFAR BN seed2026, .5 unit removal/site, validation only.
Reuse sites0-3 and the site4 direct-Q fit; ONE last Q consumer is refined. This is not a five-site implementation or a new baseline. Same victims and compact budget. Two new 200-step unlabeled calibrations, followed by two matched supervised three-epoch repairs. Labels are excluded from calibration; check/validation never select steps or hyperparameters.

| Case | Real parameters | Val before supervised repair (%) | Val after repair (%) |
| --- | ---: | ---: | ---: |
| map_fold (reused; no extra calibration) | 19098 | 18.70 | 54.20 |
| direct_q (reused; no extra calibration) | 19098 | 34.80 | 56.70 |
| local_extra | 19098 | 34.90 | 56.80 |
| tail_logits | 19098 | 36.80 | 56.60 |

### Calibration diagnostics BEFORE final head compaction

| Case | Fit logit MSE initial/final | Check logit MSE initial/final | Fit local MSE initial/final | Check local MSE initial/final | Seconds |
| --- | --- | --- | --- | --- | ---: |
| local_extra | 0.453405 / 0.453851 | 0.448586 / 0.449032 | 0.157696 / 0.157705 | 0.161911 / 0.161913 | 0.91 |
| tail_logits | 0.453405 / 0.284101 | 0.448586 / 0.306608 | 0.157696 / 0.181324 | 0.161911 / 0.185387 | 0.79 |

## Interpretation limits

One exploratory discovery seed/point/consumer; preserve unfavorable effects. A mean over minibatches or tests is not replication. Equal steps do not imply equal FLOPs/runtime: tail_logits differentiates the frozen BN/ReLU/pool/readout tail, local_extra only the consumer. The stem is cached in both. Ridge is centered at the same direct-Q kernel and is scaled by its squared norm; this differs from the earlier moment-centered direct solver. Final head uses the old map-fold treatment with the same victims, with maps recomputed for each path. Its weights may differ across paths, but it is never optimized during calibration.

Distillation of pruned quaternion CNNs already exists (Singh2026 sec.IX-B); broader multi-layer distortion is also known (Park2020Lookahead). No first-Q-distillation, novel-principle, superiority-to-published-recipe or acceptance claim. This is a bounded implementation of one proposed late-layer refinement, not a full new method established by evidence. All six convolutional layers remain Q; scalar BN and the real ten-output head are unchanged architecture conventions. No new official-test inference or automatic extension.

Sources/config/inputs frozen before scientific outcomes; per-case checkpoints resume completed calibration/repair. Two CPU checks and smoke outcomes are technical readiness, not scientific results. New repair records successful/attempted AMP steps; inherited scheduler advances on attempts, reused references lack successful-step instrumentation.

## Independent review

CPU replay passed for both cases: before/after validation and check classification, fit/check local and logit MSE, matched masks, actual compact architecture and only-target changes during calibration. Frozen sources/inputs intact. See independent_audit.json.

Measured late-consumer pilot: tail logits improve pre-repair validation by +1.9 pp over equal-step local refinement (36.8 vs 34.9%), but after three epochs give -0.2 pp (56.6 vs 56.8%). Against reused direct-Q: +2.0 pp before, -0.1 pp after. One seed/consumer; these small post-repair differences establish neither superiority nor equivalence. No automatic replication; the practical after-repair benefit was not demonstrated here.
