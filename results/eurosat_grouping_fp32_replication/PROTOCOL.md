# EuroSAT grouping replication with fresh FP32 references

## Hypothesis

Original quaternion grouping may retain better whole-network quality than predetermined role-preserving/free fully mixed assignments after matched physical compression and three-epoch repair on EuroSAT CNN6.

## Correction

The first registered implementation stopped before any supervised repair after3early-site physical fits: late archived EuroSAT activations failed bitwise replay. A four-mode diagnostic shows archived features/targets replay exactly only with cuDNN TF32 enabled, regardless of deterministic-algorithm setting. Preserve the failed output/source/config and its logs. This new version uses a separate output and new modules, retains all old masks/predetermined draws, and recomputes common FP32 targets/ridge scales for all conditions. No outcome-dependent method/group/budget selection.

## Precision

TF32 disabled for both cuDNN convolution and CUDA matmul, deterministic algorithms enabled, benchmark disabled, exactly as parent baseline/repair. Regenerate ALL native FIT/CHECK features and targets and18exact-feature fits under these settings; do not relax replay tolerance or mix TF32 archives with fresh FP32 producer inputs.

## Scope

Fresh deterministic FP32 references and18new exact-feature fits, followed by the original18physical pair fits/nine-model design. Three existing four-band EuroSAT quaternion CNN6 parents, three conditions,9actual compact models and27repair epochs. Halve only the hidden quaternion widths between the two convolutions of blocks0/2, leaving block outputs, pooling/dropout, other block and real readout unchanged. All models have44794real trainables versus73898parents, about39.4percent whole-model savings; this is not the14.5percent CIFAR budget. No new parent, alternative algebra/dimension, group search, BN sweep or official-test evaluation.

## Data

Same existing TRAIN/validation/FIT/CHECK split and nine-model design as the base protocol, no test inference. Copy only the frozen original/role_100_0/free_100_0 assignments/kept scalar sets from the completed statistics control; these masks were originally derived with its archived precision. Collect fresh native features and targets for ALL conditions in deterministic FP32, plus fresh exact-feature fits, before physical fitting. The masks are not reselected. Previously explored validation/calibration data remain exploratory.

## Realization

Use the base sequential-pair compiler and physical fitter unchanged, in a new output. For each site collect current source-consumer features/targets on500FIT and500CHECK images with TF32 disabled, compute original retained-feature raw moments, set common lambda=0.001times original Hamilton Gram mean diagonal, and solve all three exact-feature grouping fits. Save/replay coefficients. The base physical fitting then reuses these within-study FP32 reference rows, not old TF32 numerical errors. Full producer inputs and all native activation batch checks use the same FP32 state. All18physical pairs and9repairs retain their original size/budget.

## Full_models

Combine both compact pairs into each whole model, measure validation/CHECK accuracy and CE before repair, then repair all surviving parameters with frozen AdamW/AMP augmentation recipe for final epoch3. Same order/augmentation seeds within each source seed; record dropout/RNG and successful/attempted updates through the existing repair function.19600train images give154batches per epoch and462attempted updates per model, unlike1125in CIFAR; equal epochs/recipe across datasets are not equal update/compute budgets. Save native/final models and per-example logits/labels, then verify reload predictions and metrics. No best-epoch or condition selection.

## Primary

Report original-minus-role and original-minus-free final epoch3validation accuracy gaps in percentage points separately for each source seed and descriptive means over3parents. Positive gap favors original grouping. Also report paired validation CE, raw before/after values and CHECK outcomes; unfavorable/reversed differences remain in the report. Compare to the same100percent mixed CIFAR cases as a cross-dataset/architecture description, not a pooled significance test. No continuation threshold or automatic further experiment.

## Checks

Before freeze: pair extraction/installation, exact original and dense inverse-permutation controls, saved compact state replay, fixed scalar-set enforcement and parameter counts, source-input hook cleanup, full synthetic local-fit/model-repair/completed-resume, plus discarded compact GPU tracked-variable/all-quaternion-component gradient/update audit. During science: frozen source/input hashes, parent validation replay, activation replay, per-case gradient audit on discarded copies, fitted-block prediction replay, final saved-model/logit replay and artifact hashes. STOP between data/model/local batches and existing repair updates; completed artifacts verified/reused, partial local fits repeat at most one case.

## Additional_checks

Test fixed precision enforcement and fresh-reference generation on CPU, then the base full synthetic six-pair/three-model repair/resume and discarded full-size compact CUDA variable/gradient/update audit. New protocol freezes the base sources/config plus this wrapper/new module/new tests, old assignments and the precision diagnostic before scientific collection. No changes to any frozen predecessor. Completion audit includes fresh activation/ideal-coefficient hashes, physical pairs, input packs and all saved native/final models/logits.

## Interpretation

This is practical replication on a second dataset AND architecture/input encoding. It does not isolate a dataset effect, prove semantic necessity, global optimality, equivalence, method novelty or conference readiness. Earlier failed adaptation/reassembly/exchange gates remain failed. Do not tune this bounded study or add repairs after observing outcomes.

Prior art: Helber2019EuroSAT, Vecchi2020, He2017ChannelPruning, VieiraValle2022ELM, Singh2026, Comminiello2024InductiveBiases; canonical references.bib.
