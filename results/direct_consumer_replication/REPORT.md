# Direct-Q consumer reconstruction: fixed replication

## Hypothesis

Direct pre-BN quaternion consumer-output regression can recover more classification quality than shared feature reconstruction followed by folding at exactly the same compact Q architecture, masks and real parameter count.

## Measured results

Status: completed; 14/14 jobs. Scientific results: True. Validation, no new test inference.
Discovery CIFAR BN seed2026 f=.5 is reused and marked; aggregates containing it are not independent confirmation. Separate new-seed effects below. No-BN remains one seed. Same compact Q size/masks; dense references have different budgets.

| Dataset | Norm | Removed/site | Stage | Case | Parameters | n | Val mean (%) | SD (pp) |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cifar | bn | 0.25 | before | q_pruned | 42034 | 3 | 23.47 | 2.75 |
| cifar | bn | 0.25 | before | q_comp | 42034 | 3 | 59.67 | 3.21 |
| cifar | bn | 0.25 | before | direct_q | 42034 | 3 | 66.87 | 0.40 |
| cifar | bn | 0.25 | before | free_q_init | 162562 | 3 | 59.67 | 3.21 |
| cifar | bn | 0.25 | before | free_full_fit | 162562 | 3 | 69.03 | 0.42 |
| cifar | bn | 0.25 | after | q_pruned | 42034 | 3 | 70.00 | 0.36 |
| cifar | bn | 0.25 | after | q_comp | 42034 | 3 | 71.50 | 1.20 |
| cifar | bn | 0.25 | after | direct_q | 42034 | 3 | 72.77 | 0.76 |
| cifar | bn | 0.25 | after | free_q_init | 162562 | 3 | 72.93 | 0.87 |
| cifar | bn | 0.25 | after | free_full_fit | 162562 | 3 | 73.90 | 0.00 |
| cifar | bn | 0.5 | before | q_pruned | 19098 | 3 | 13.43 | 1.55 |
| cifar | bn | 0.5 | before | q_comp | 19098 | 3 | 21.97 | 4.12 |
| cifar | bn | 0.5 | before | direct_q | 19098 | 3 | 32.03 | 2.44 |
| cifar | bn | 0.5 | before | free_q_init | 72666 | 3 | 21.97 | 4.12 |
| cifar | bn | 0.5 | before | free_full_fit | 72666 | 3 | 35.07 | 4.60 |
| cifar | bn | 0.5 | after | q_pruned | 19098 | 3 | 49.67 | 2.14 |
| cifar | bn | 0.5 | after | q_comp | 19098 | 3 | 53.57 | 0.65 |
| cifar | bn | 0.5 | after | direct_q | 19098 | 3 | 56.90 | 0.20 |
| cifar | bn | 0.5 | after | free_q_init | 72666 | 3 | 57.90 | 0.69 |
| cifar | bn | 0.5 | after | free_full_fit | 72666 | 3 | 63.17 | 0.21 |
| cifar | none | 0.25 | before | q_pruned | 41362 | 1 | 32.40 | — |
| cifar | none | 0.25 | before | q_comp | 41362 | 1 | 56.70 | — |
| cifar | none | 0.25 | before | direct_q | 41362 | 1 | 62.80 | — |
| cifar | none | 0.25 | before | free_q_init | 161890 | 1 | 56.70 | — |
| cifar | none | 0.25 | before | free_full_fit | 161890 | 1 | 67.30 | — |
| cifar | none | 0.25 | after | q_pruned | 41362 | 1 | 68.40 | — |
| cifar | none | 0.25 | after | q_comp | 41362 | 1 | 70.10 | — |
| cifar | none | 0.25 | after | direct_q | 41362 | 1 | 70.00 | — |
| cifar | none | 0.25 | after | free_q_init | 161890 | 1 | 70.20 | — |
| cifar | none | 0.25 | after | free_full_fit | 161890 | 1 | 73.00 | — |
| cifar | none | 0.5 | before | q_pruned | 18650 | 1 | 10.40 | — |
| cifar | none | 0.5 | before | q_comp | 18650 | 1 | 22.20 | — |
| cifar | none | 0.5 | before | direct_q | 18650 | 1 | 30.50 | — |
| cifar | none | 0.5 | before | free_q_init | 72218 | 1 | 22.20 | — |
| cifar | none | 0.5 | before | free_full_fit | 72218 | 1 | 33.00 | — |
| cifar | none | 0.5 | after | q_pruned | 18650 | 1 | 38.70 | — |
| cifar | none | 0.5 | after | q_comp | 18650 | 1 | 45.70 | — |
| cifar | none | 0.5 | after | direct_q | 18650 | 1 | 50.00 | — |
| cifar | none | 0.5 | after | free_q_init | 72218 | 1 | 51.90 | — |
| cifar | none | 0.5 | after | free_full_fit | 72218 | 1 | 57.30 | — |
| eurosat | bn | 0.25 | before | q_pruned | 42034 | 3 | 32.03 | 6.51 |
| eurosat | bn | 0.25 | before | q_comp | 42034 | 3 | 78.13 | 5.02 |
| eurosat | bn | 0.25 | before | direct_q | 42034 | 3 | 86.27 | 2.12 |
| eurosat | bn | 0.25 | before | free_q_init | 162562 | 3 | 78.13 | 5.02 |
| eurosat | bn | 0.25 | before | free_full_fit | 162562 | 3 | 92.27 | 1.15 |
| eurosat | bn | 0.25 | after | q_pruned | 42034 | 3 | 92.47 | 0.47 |
| eurosat | bn | 0.25 | after | q_comp | 42034 | 3 | 93.73 | 0.38 |
| eurosat | bn | 0.25 | after | direct_q | 42034 | 3 | 93.97 | 0.32 |
| eurosat | bn | 0.25 | after | free_q_init | 162562 | 3 | 94.47 | 0.32 |
| eurosat | bn | 0.25 | after | free_full_fit | 162562 | 3 | 95.03 | 0.49 |
| eurosat | bn | 0.5 | before | q_pruned | 19098 | 3 | 16.20 | 4.52 |
| eurosat | bn | 0.5 | before | q_comp | 19098 | 3 | 32.20 | 0.72 |
| eurosat | bn | 0.5 | before | direct_q | 19098 | 3 | 45.20 | 1.00 |
| eurosat | bn | 0.5 | before | free_q_init | 72666 | 3 | 32.20 | 0.72 |
| eurosat | bn | 0.5 | before | free_full_fit | 72666 | 3 | 61.00 | 7.98 |
| eurosat | bn | 0.5 | after | q_pruned | 19098 | 3 | 77.47 | 5.72 |
| eurosat | bn | 0.5 | after | q_comp | 19098 | 3 | 85.33 | 2.85 |
| eurosat | bn | 0.5 | after | direct_q | 19098 | 3 | 86.70 | 1.78 |
| eurosat | bn | 0.5 | after | free_q_init | 72666 | 3 | 88.23 | 1.17 |
| eurosat | bn | 0.5 | after | free_full_fit | 72666 | 3 | 90.17 | 1.33 |

### Paired direct minus map-fold effects

| Dataset | Norm | Removed/site | Stage | Exclude discovery | Mean (pp) | Seeds (pp) |
| --- | --- | ---: | --- | --- | ---: | --- |
| cifar | bn | 0.25 | before | False | +7.20 | 2027:+3.80, 2028:+9.90, 2026:+7.90 |
| cifar | bn | 0.25 | before | True | +7.20 | 2027:+3.80, 2028:+9.90, 2026:+7.90 |
| cifar | bn | 0.25 | after | False | +1.27 | 2027:+0.90, 2028:+1.80, 2026:+1.10 |
| cifar | bn | 0.25 | after | True | +1.27 | 2027:+0.90, 2028:+1.80, 2026:+1.10 |
| cifar | bn | 0.5 | before | False | +10.07 | 2027:+4.50, 2028:+9.60, 2026:+16.10 |
| cifar | bn | 0.5 | before | True | +7.05 | 2027:+4.50, 2028:+9.60 |
| cifar | bn | 0.5 | after | False | +3.33 | 2027:+4.20, 2028:+3.30, 2026:+2.50 |
| cifar | bn | 0.5 | after | True | +3.75 | 2027:+4.20, 2028:+3.30 |
| cifar | none | 0.25 | before | False | +6.10 | 2026:+6.10 |
| cifar | none | 0.25 | before | True | +6.10 | 2026:+6.10 |
| cifar | none | 0.25 | after | False | -0.10 | 2026:-0.10 |
| cifar | none | 0.25 | after | True | -0.10 | 2026:-0.10 |
| cifar | none | 0.5 | before | False | +8.30 | 2026:+8.30 |
| cifar | none | 0.5 | before | True | +8.30 | 2026:+8.30 |
| cifar | none | 0.5 | after | False | +4.30 | 2026:+4.30 |
| cifar | none | 0.5 | after | True | +4.30 | 2026:+4.30 |
| eurosat | bn | 0.25 | before | False | +8.13 | 2027:+5.80, 2028:+12.40, 2026:+6.20 |
| eurosat | bn | 0.25 | before | True | +8.13 | 2027:+5.80, 2028:+12.40, 2026:+6.20 |
| eurosat | bn | 0.25 | after | False | +0.23 | 2027:+0.10, 2028:+0.30, 2026:+0.30 |
| eurosat | bn | 0.25 | after | True | +0.23 | 2027:+0.10, 2028:+0.30, 2026:+0.30 |
| eurosat | bn | 0.5 | before | False | +13.00 | 2027:+14.60, 2028:+12.20, 2026:+12.20 |
| eurosat | bn | 0.5 | before | True | +13.00 | 2027:+14.60, 2028:+12.20, 2026:+12.20 |
| eurosat | bn | 0.5 | after | False | +1.37 | 2027:+2.60, 2028:+1.70, 2026:-0.20 |
| eurosat | bn | 0.5 | after | True | +1.37 | 2027:+2.60, 2028:+1.70, 2026:-0.20 |

## Interpretation limits

- Report CIFAR discovery seed2026 f=.5 separately from the two new-seed replications. Three-seed summaries include discovery and are not an independent confirmation. EuroSAT is new for this method but its data/test outcomes were used earlier; no fresh blind evaluation claim.
- Keep all unfavorable effects and per-seed signs, means and sample SD. One-seed no-BN remains exploratory. No significance/equivalence claim from a small mean or n=3.
- Objective, centered regularization and compensation parameterization differ from the old map-fold procedure together; this is not an isolated pure feasible-space effect, a new LS theorem or an irreducible Q architecture bound.
- Do not tune alpha, masks or checkpoint selection on validation/check outcomes. Final third repair epoch stays the endpoint. Record successful steps for new repairs; old references have only attempted schedules audited.
- Stop after these fixed jobs and assess repeatability/heterogeneity before planning another study. No automatic growth of the sweep or guaranteed novelty/acceptance.

Five quaternion consumers use direct pre-BN output ridge centered on their folded candidates; final real head remains map-fold-only. Fit and check halves are separate. Local fit improvements are partly guaranteed by the centered objective; local check/classification improvements are empirical and may be negative. Preserve all cases. No new LS algorithm or quaternion uniqueness claim (He2017ChannelPruning).

## Independently reviewed interpretation

Full provenance/solve/error audit passed. This is not an independent full-dataset inference replay. All new outcomes are validation/check only. Paired plot: [PDF](paired_effects.pdf), [PNG](paired_effects.png); error bars are sample SD of three paired effects, not confidence intervals. Black stars mark the reused discovery point.

| Dataset | Norm | Removed/site | Map-fold +3ep (%) | Direct-Q +3ep (%) | Paired mean gain (pp) | Compact scratch 50ep (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cifar | bn | 0.25 | 71.50 | 72.77 | +1.27 | 71.57 |
| cifar | bn | 0.5 | 53.57 | 56.90 | +3.33 | 63.90 |
| eurosat | bn | 0.25 | 93.73 | 93.97 | +0.23 | 95.50 |
| eurosat | bn | 0.5 | 85.33 | 86.70 | +1.37 | 92.47 |
| cifar | none | 0.25 | 70.10 | 70.00 | -0.10 | not measured |
| cifar | none | 0.5 | 45.70 | 50.00 | +4.30 | not measured |

### Paired accuracy and CE signs

| Dataset | Norm | Removed/site | Positive/zero/negative accuracy effects | Mean CE difference (direct minus map) | Seeds: accuracy pp / CE difference |
| --- | --- | ---: | --- | ---: | --- |
| cifar | bn | 0.25 | 3/0/0 | -0.02076 | 2027: +0.90 / -0.02391, 2028: +1.80 / -0.01709, 2026: +1.10 / -0.02129 |
| cifar | bn | 0.5 | 3/0/0 | -0.08933 | 2027: +4.20 / -0.08018, 2028: +3.30 / -0.09541, 2026: +2.50 / -0.09241 |
| cifar | none | 0.25 | 0/0/1 | -0.03142 | 2026: -0.10 / -0.03142 |
| cifar | none | 0.5 | 1/0/0 | -0.11688 | 2026: +4.30 / -0.11688 |
| eurosat | bn | 0.25 | 3/0/0 | -0.00735 | 2027: +0.10 / -0.00741, 2028: +0.30 / -0.01435, 2026: +0.30 / -0.00030 |
| eurosat | bn | 0.5 | 2/0/1 | -0.04544 | 2027: +2.60 / -0.07626, 2028: +1.70 / -0.04766, 2026: -0.20 / -0.01240 |

Scratch is a different training budget: 50 student epochs versus 3 repair epochs plus a pretrained full model. The new method is not claimed to dominate scratch or save measured runtime. No-BN is n=1; BN aggregates are n=3, with the CIFAR .5 aggregate including discovery.

New repairs record successful optimizer steps; AMP may skip nonfinite-gradient attempts. The inherited scheduler advances on attempted batches, including skipped optimizer steps, as in the frozen comparison recipe. Reused references lack successful-step instrumentation. Therefore the endpoint matches the attempted schedule, not a guarantee of identical successful-step counts. Per-job counts are retained in summary.json; the PyTorch scheduler warning is disclosed here rather than silently changing the frozen recipe.

Check-calibration reconstruction MSE improved at 70/70 local consumers; retain any failures above. Local objective improvement is partly guaranteed by the centered fit, whereas check/classification changes are measured.

The separate cached-moment diagnostic on the discovery pilot is in ../direct_consumer_pilot/gap_diagnostic/REPORT.md. It quantifies shared-map ties under a common local objective and does not decompose classification accuracy. It is not a replicated architecture lower bound or a new projection theorem.

This registered study is closed. Interpret paired signs/CE consistency and the separate two-new-seed CIFAR table before any next proposal. No tuning, new architecture, test inference or automatic additional training follows.
