# EuroSAT physical grouping replication

Status: failed

## Hypothesis

Original quaternion grouping may retain better whole-network quality than predetermined role-preserving/free fully mixed assignments after matched physical compression and three-epoch repair on EuroSAT CNN6.

## Measured full-network results

| Seed | Case | Actual trainables | Native val | Final epoch3 val | Final val CE |
|---|---|---:|---:|---:|---:|

## Primary paired gaps

Positive accuracy gap favors original grouping; CE delta is mixed minus original.

| Seed | Case | Native val gap pp | Final val gap pp | Final CE delta |
|---|---|---:|---:|---:|

## Measured local results

| Seed | Block | Case | Exact-feature check MSE (reused) | Physical check MSE | Producer projection SSE |
|---|---|---|---:|---:|---:|
| 2026 | 0 | free_100_0 | 0.342825 | 0.370718 | 23.1576 |
| 2026 | 0 | original | 0.185911 | 0.185911 | 0 |
| 2026 | 0 | role_100_0 | 0.320053 | 0.358615 | 18.46 |

## Interpretation and limits

Three existing four-band EuroSAT quaternion CNN6 parents, three conditions,9actual compact models and27repair epochs. Halve only the hidden quaternion widths between the two convolutions of blocks0/2, leaving block outputs, pooling/dropout, other block and real readout unchanged. All models have44794real trainables versus73898parents, about39.4percent whole-model savings; this is not the14.5percent CIFAR budget. No new parent, alternative algebra/dimension, group search, BN sweep or official-test evaluation.

Use the existing disjoint19600TRAIN/1000validation/1000calibration/5400test split and NIR,R,G,B preprocessing. FIT is the500even-calibration examples; CHECK is the500odd-calibration examples. Test is never inferred. Reuse native-trained local activation targets, retained scalar sets and original/role_100_0/free_100_0 assignments from the completed statistics control. Draw0is fixed before outcomes and never selected from the ten draws. Masks came from existing native-trained FIT activation energy; all three conditions retain exactly the SAME scalar multiset per seed/site. These calibration/validation splits were explored previously; this is not blind confirmation.

Combine both compact pairs into each whole model, measure validation/CHECK accuracy and CE before repair, then repair all surviving parameters with frozen AdamW/AMP augmentation recipe for final epoch3. Same order/augmentation seeds within each source seed; record dropout/RNG and successful/attempted updates through the existing repair function.19600train images give154batches per epoch and462attempted updates per model, unlike1125in CIFAR; equal epochs/recipe across datasets are not equal update/compute budgets. Save native/final models and per-example logits/labels, then verify reload predictions and metrics. No best-epoch or condition selection.

This is practical replication on a second dataset AND architecture/input encoding. It does not isolate a dataset effect, prove semantic necessity, global optimality, equivalence, method novelty or conference readiness. Earlier failed adaptation/reassembly/exchange gates remain failed. Do not tune this bounded study or add repairs after observing outcomes.
