# Component-level selection versus whole-unit pruning

## Hypothesis

Selecting individual scalar components and grouping them into realizable smaller Hamilton layers may improve compression relative to whole-unit selection at matched actual compact size and repair, or may fail. Original intact groups remain legal answers.

## Scope

Two existing settings and three existing source parents each; three conditions per parent,18compact models. Halve only the hidden widths at the same two sites as earlier studies. Unit_energy uses a transparent inner-FIT energy mask. Unit_search and component_search choose different retained scalar sets if useful. Both have the same number of complete warm-start assignments, rounds/proposal cap, zero-centered physical Hamilton refitting, network selection opportunity and30epoch repair recipe. No new parent, algebra, dimension, scratch model or compression grid.

## Selection

Select only using fixed64inner and64disjoint outer images from the500FIT partition. Energy/saliency starts use inner images only. Whole-unit search starts: summed scalar activation energy, activation-energy times outgoing-column squared norm, private random unit subset. Component search starts: the same first two intact masks and top4Kscalar-energy packing. Every candidate is already Kcomplete ordered quartets containing4Kdistinct source scalar channels: no partial catalogue or whole-unit completion reserve. Whole-unit substitutions are available to both policies; component search additionally permits selected/unselected single-scalar replacements and all selected pair/role swaps. All legal local moves are enumerated before a bounded shortlist. Per active move category allocate an equal quota, half low projection cost and half private random proposals, so intact zero-cost moves cannot eliminate all scalar proposals. Three starts, three rounds,12proposals/round cap per policy; cache duplicate fits and report actual counts, candidates, source-set changes, role changes and time. This is bounded heuristic coverage, not global optimality or a novel algorithm.

## Scoring

Fit the consumer on all inner spatial positions using the actual projected producer and fixed zero-centered ridge. Lambda is0.001times mean diagonal of the original unit-energy inner Hamilton Gram, identical across all candidates at a source/site. Score actual block outputs after BN/nonlinearity/shortcut on outer FIT images. Each policy supplies at most the baseline and its best local assignment/site. Evaluate at most4two-site combinations per policy using whole-network teacher KL(T=2)on the same outer FIT images; original energy baseline is allowed. No labels, CHECK or validation choose assignments. Refit selected consumers on all500FIT images with lambda scaled by500/64, then measure CHECK. Selection subsets are exploratory; final fitting includes outer selection images and does not make them held out.

## Evaluation

Primary final-epoch30component_search versus unit_search paired validation accuracy and CE, separate settings/source seeds and descriptive means. Report unit_energy baseline, all source-set overlaps, mixed quartets/roles, compact counts, initial quality and snapshots0/1/3/10/30. All18conditions receive new matched repair trajectories even if selected models coincide; coincident initializations are explicitly identified and are not independent parents. Record actual AMP attempted/successful/skipped updates. No best-epoch selection or post-outcome extension. A negative result remains negative. No significance, equivalence, semantic, novelty or conference-acceptance claim is assumed.

## Data

Use the same TRAIN/validation/FIT/CHECK splits and deterministic FP32 reference collection as the completed studies. Previously explored validation/calibration remain exploratory. Read archived native activation/input packs only after hash and native-batch replay checks. EuroSAT uses fresh-FP32 native packs, never old TF32 references. New retained sets are not constrained to the old masks. Training uses TRAIN only; no test forward pass or test metric.

## Checks

Before freeze: exhaustive small-case complete-neighbor coverage/injection checks, unit-family membership, scalar-set-changing and role-changing moves, stratified shortlist coverage, full synthetic selection/compilation/refit/trajectory/completed resume, discarded full compact CUDA tracked-variable/all-component gradient-update audits. Validate all inherited hashes, parent validation metrics and native activation/input batch replay. During science: deterministic candidate fits, saved local/full-model replay, fixed physical counts, per-case discarded updates, snapshot/logit reload and final source/output hash audit. Do not edit frozen predecessors or registered new sources after freeze.

## Resume

Atomic hash-checked per-site search/finalists and selected pair artifacts. Stop checked between proposals/data batches and repair batches. An interrupted site search may repeat that bounded site; completed searches/fits/trajectories verified and reused. Frozen repair/snapshot code restores completed-epoch optimizer/scaler/RNG; requested missing snapshots recovered. Preserve failures. No automatic follow-up after this bounded experiment.

Prior art: He2017ChannelPruning, Vecchi2020, Singh2026, Wang2019FLGC, Zhang2021StructuredShuffle, Martinez2021PermuteQuantizeFineTune, VieiraValle2022ELM; canonical references.bib.
