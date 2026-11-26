# Fixed-set grouping exchange pilot

Status: completed

## Hypothesis

Data-chosen fixed-set scalar exchanges, assessed through actual compact Hamilton producers/consumers, may improve repaired accuracy over intact groups at identical physical size.

## Measured final epoch results

| Seed | Original val | Guided val | Gain pp | CE delta | Repair reused |
|---|---:|---:|---:|---:|---|
| 2026 | 89.80 | 89.80 | 0.00 | 0.000000 | True |
| 2027 | 91.00 | 91.00 | 0.00 | 0.000000 | True |
| 2028 | 89.60 | 89.60 | 0.00 | 0.000000 | True |

## Decision

{'status': 'evaluated', 'mean_accuracy_gain_pp': 0.0, 'mean_ce_delta': 0.0, 'practical_screen_passed': False, 'significance_claimed': False, 'automatic_followup': False}

## Interpretation limits

Three existing CIFAR10 QResNet20 parents, two internal sites0/8, same retained scalar sets as grouping_mechanism_study. Standard Hamilton d4 only,232266real trainables. No other dimension/algebra, new parent or test inference. At most three new3epoch repairs; reuse the verified historical original control and random-grouping comparators. A bitwise identical guided initialization reuses original repaired results rather than retraining it.

Use500even-calibration FIT examples only. CPU private permutation(seed+370000) chooses64inner-fit images then64disjoint outer-selection images, same split at both sites. Frozen masks were previously chosen from all500FIT images; this is conditional fixed-mask comparison, not independent mask discovery. CHECK/validation never choose groups, endpoints or epochs. Final refit uses all500FIT images after assignments are locked. These data and validation have been explored previously.

Rebuild/refit original blocks on all500FIT examples and require bitwise state equality to the frozen original local artifacts; native original full-model metrics/logits must replay. Thus original control has identical final continuous fitting and historical paired repair recipe; additional discrete search time is measured, not called equal compute. Compare guided versus original final epoch3 accuracy/CE per seed and mean. Historical mixed cases are diagnostic context, not the sole competitor. Expose independent site/group fractions, search counts and all unfavorable outcomes.

Bounded heuristic, no optimality, significance, equivalence, learned-semantics or novelty claim. Original groups are allowed; unchanged results must be reported, not forced into mixed groups. Previous pilot gate stays failed; no automatic further run.
