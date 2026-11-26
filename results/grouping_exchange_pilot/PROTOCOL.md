# Fixed-set grouping exchange pilot

## Hypothesis

Data-chosen fixed-set scalar exchanges, assessed through actual compact Hamilton producers/consumers, may improve repaired accuracy over intact groups at identical physical size.

## Scope

Three existing CIFAR10 QResNet20 parents, two internal sites0/8, same retained scalar sets as grouping_mechanism_study. Standard Hamilton d4 only,232266real trainables. No other dimension/algebra, new parent or test inference. At most three new3epoch repairs; reuse the verified historical original control and random-grouping comparators. A bitwise identical guided initialization reuses original repaired results rather than retraining it.

## Selection

Use500even-calibration FIT examples only. CPU private permutation(seed+370000) chooses64inner-fit images then64disjoint outer-selection images, same split at both sites. Frozen masks were previously chosen from all500FIT images; this is conditional fixed-mask comparison, not independent mask discovery. CHECK/validation never choose groups, endpoints or epochs. Final refit uses all500FIT images after assignments are locked. These data and validation have been explored previously.

## Search

At each site start from original groups and predecessor free_50_0. Canonically sort whole groups, never roles, to remove whole-unit order duplicates. Every proposal exchanges two scalar entries from distinct groups, preserving the exact scalar multiset and physical width. Enumerate unique swaps; shortlist8lowest Frobenius producer-projection costs plus8private random proposals from the remaining candidates. Two rounds per start, terminate a start when no strict outer-score improvement. Candidate inner consumers use zero-centered Hamilton ridge with predecessor lambda scaled64/500, no producer or BN updates. Rank by actual full-block output MSE on64outer FIT examples, with source block targets and copied BN/shortcut. Cache identical assignments. Keep the original and each start's final assignment, at most3per site. No completion reserve, forced mixing, global-optimum claim or check-dependent restart.

## Global_selection

Evaluate all at-most9combinations of the two sites' local finalists using their inner-fitted weights in the actual full compact network. Choose minimum teacher-to-student KL at fixedT2 on the same64outer FIT images; ties prefer original assignment, then deterministic order. Source teacher logits are label-free. After selection, independently refit each chosen physical consumer on all500FIT source-input examples with the same frozen zero-centered ridge/lambda as the original control. This teacher-input local refit does not assume compact-upstream activations equal source activations. Store pre-refit selection scores; no CHECK/validation enters selection.

## Controls

Rebuild/refit original blocks on all500FIT examples and require bitwise state equality to the frozen original local artifacts; native original full-model metrics/logits must replay. Thus original control has identical final continuous fitting and historical paired repair recipe; additional discrete search time is measured, not called equal compute. Compare guided versus original final epoch3 accuracy/CE per seed and mean. Historical mixed cases are diagnostic context, not the sole competitor. Expose independent site/group fractions, search counts and all unfavorable outcomes.

## Decision

Primary descriptive endpoint: mean paired guided-minus-original final epoch3 validation accuracy over the3fixed seeds, plus mean paired CE. Practical continuation screen requires accuracy gain>=0.5pp and mean CE no higher than original. This is a new prospective screen, not significance, novelty or permission to erase the older failed pilot gate. No automatic further run regardless of outcome.

## Checks

Before freeze: partition/proposal invariants, producer-cost ordering against actual compiler, outer-score selection/retaining original, separate FIT subsets, compiled consumer fit and full synthetic run/resume, discarded-copy compact GPU all-variable/quaternion-gradient/update audit. Save/reload every finalist/final block/model and final logits, verify frozen source/input/output hashes and identical original replay. STOP between proposals/batches/repair steps; completed local searches and repairs resume, interrupted local search repeats at most one site.

Prior art: Vecchi2020, He2017ChannelPruning, Wang2019FLGC, Zhang2021StructuredShuffle, Singh2026, VieiraValle2022ELM; canonical references.bib.
