# BatchNorm intervention at the soft-to-hard boundary

## Hypotheses

The original one-seed pilot's hardening loss may partly reflect downstream BN running-statistic mismatch. A separate possibility is that the soft mixing operator provides behavior a permutation cannot reproduce. These hypotheses are not mutually exclusive.

## Protocol

Replay the frozen seed-2026 pilot through epoch 40; retain full pre/post-hardening checkpoints. Compare the same weights in four prespecified conditions. Refit only the four downstream BNs, in forward order, using all 48,000 training images without augmentation. All modules, including dropout, stay in evaluation mode. Moments accumulate in float64; variance uses N-1 observations including spatial positions. Earlier BNs use their newly fitted evaluation statistics. BN affine parameters, every model parameter, upstream BN buffers, counters, and routing remain unchanged by refitting. No optimizer steps or official-test inference are performed after the boundary.

## Measured results

| Condition | Validation accuracy | Validation CE | Calibration accuracy | Calibration CE |
| --- | ---: | ---: | ---: | ---: |
| soft_original | 79.20% | 0.609989 | 79.20% | 0.608316 |
| hard_original | 56.00% | 1.636415 | 55.20% | 1.596982 |
| soft_recalibrated | 79.00% | 0.609001 | 80.30% | 0.598084 |
| hard_recalibrated | 66.30% | 0.930627 | 66.30% | 0.942568 |

Validation hard-routing gain from BN refitting: **+10.30 pp**. Soft-routing control change: **-0.20 pp**. The soft-minus-hard gap is 23.20 pp with original BN and 12.70 pp with both BNs refitted.

Replay verification: numerical epoch metrics and boundary metrics match the original within 1.0e-08. The original boundary weights were not saved, so bitwise identity to those unavailable weights cannot be checked.

## Interpretation and limits

The paired soft refit is a control for the refitting procedure itself. The observed hard-model recovery measures what this training-only BN-buffer intervention repairs at fixed weights. Any remaining gap is not repaired by this procedure; it does not prove an unavoidable limitation of learned grouping. The result is one seed, one CNN, one boundary and one refitting estimator. No new method, general gain, compression saving or publication readiness is claimed. A hard model may still need weight adaptation; that is not tested here.

The retained soft router has 1,024 extra real parameters. Hard condition checkpoints retain those frozen logits for exact diagnosis; they are not compact exports. The pilot's existing fixed-router export mechanism can remove them, but rearrangement alone removes no backbone channels.

Prior context: Mena2018 (permutation relaxation); Wang2019FLGC and Zhang2021StructuredShuffle (learned grouping/shuffle), as recorded in references.bib. This diagnostic makes no priority claim.

## Artifacts

- replay/boundary_soft.pt and replay/boundary_hard.pt: full boundary states.
- intervention_protocol.json: choices, source hashes, split and input hashes before the intervention.
- */model.pt, predictions.npz, metrics.json: reproducible fixed-weight conditions.
- evaluation_targets.npz: held-out labels and original split indices.
- summary.json and comparison.png: numeric results and comparison.

## Verified measured interpretation and decision

BN-only refitting repairs 10.30 pp of hard validation accuracy, while the soft control changes by -0.20 pp. The remaining refitted soft-minus-hard gap is 12.70 pp. Thus BN-buffer refitting is helpful but insufficient to restore soft-model quality in this intervention.

The secondary held-out calibration set agrees in direction: hard accuracy rises from 55.20% to 66.30%; refitted soft accuracy is 80.30%, leaving a 14.00-pp gap. No official test inference or weight updates were used in this diagnostic. Five implementation tests and 111 independent artifact checks pass; no scientific or implementation check failed in this stage.

Close the hypothesis that this particular downstream-BN refit alone restores the original soft-model quality. Do not call the remaining loss a proven inherent limitation or assign a uniquely causal percentage to BN. The next candidate experiment is a prespecified discrete-forward grouping-training comparison against paired fixed assignments, with extra variables/compute reported and equivalent final evaluation/refitting. Do not silently reopen the original failed compression gate. No further training is started by this diagnostic.
