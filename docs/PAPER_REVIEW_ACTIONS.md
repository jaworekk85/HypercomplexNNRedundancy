# Response to the two manuscript reviews

Received from the user on2026-11-26: one GPT review and one Claude review.
These are external model assessments, not conference reviews or independent
verification of the numerical artifacts. Their acceptance percentages are not
calibrated estimates and are not adopted as project evidence. The quoted2025
conference statistics are not used to predict acceptance.

## Assessment against existing evidence

| Review point | Repository evidence / response | Action |
|---|---|---|
| Three repair epochs cannot distinguish a persistent grouping cost from slower recovery. | Correct. All full-network grouping conditions stop at epoch3. | Highest-priority prospective experiment: a common30epoch trajectory with reports at0/1/3/10/30. |
| Test results are missing. | Correct for these grouping studies; explored validation remains exploratory. | Freeze methods, repair/reporting choices and prior-split-access audit before a final test pass. Do not open test during exploratory repair design. |
| Original versus random groups may be structurally predictable. | Already acknowledged by the initialization, statistics, dense inverse-permutation and adaptive-search controls. | State the contribution as separating constrained representation, physical realization and finite-budget recovery. Do not infer semantic entities or a new successful algorithm. |
| Component-level selection versus whole-unit retention has not been tried. | It exists in the earlier single-seed pilot: whole_joint90.4%, scalar_energy88.6%, role_fixed90.4%, free_roles90.8%, all232266real trainables. Different masks, completion bias, and failed+0.5ppscreen limit it. | Show the existing comparison clearly as exploratory history; a cleaner multi-seed selection comparison is a possible later study, not a new discovery or an already-solved question. |
| Uncompressed parent results are missing from the article. | Six parent validation replays already exist: CIFAR91.0/91.4/90.7%, EuroSAT96.5/96.5/97.0%. | Add their descriptive means to the accuracy table without new inference. |
| Compact scratch baseline is missing. | Correct for the exact fixed-Hamilton grouping architecture. The earlier PHM scratch has233418stored reals and a different parameterization, versus232266in this CIFAR study. Existing real/Q studies also use different masks/scopes. | Never relabel PHM scratch or nine-site compression as a matched grouping baseline. Consider a separately specified exact-architecture scratch control after the repair study. |
| More seeds, full-network draws and compression ratios. | Three parent seeds and one fixed full-network draw per mixed condition are limited. | Later strengthening, not a simultaneous automatic grid. Distinguish input information, mask changes, compression fraction and whole-model parameter reduction. |
| A0.5ppdifference is within the accuracy's roughly0.95ppstandard error. | An individual accuracy's binomial SE does not determine uncertainty of the difference between two models evaluated on the same images. | Use paired per-image predictions for uncertainty; source-seed variability remains a separate level. Do not infer significance or equivalence from either marginal SEs or three seed means. |
| New theory would strengthen the contribution. | Potentially useful, but a new theorem is not established. Simple projection/invariance/Lipschitz arguments are standard. | Precisely define input-only versus joint input/output permutations, unsigned versus signed transformations, and whether BN/ReLU are included. Verify proofs and prior art before a novelty claim. |
| Text reads as a protocol/lab log. | Fair. Repeated novelty disclaimers, implementation failures and private terms obscure the empirical argument. | Rewrite abstract/introduction/related work; add a conceptual grouping diagram and contributions; retain caveats/failed-result details in limitations and repository evidence. |
| Remove the KHNN failure because of possible reviewer identities. | Reviewer-identity speculation is not a scientific editing criterion. Version-specific failure remains accurately documented. | Shorten implementation provenance because it is peripheral; retain framework credit and all failure evidence in the repository. |
| AI disclosure can become smaller after rewriting. | Another AI rewrite remains AI-assisted drafting; it does not narrow actual use automatically. | Keep truthful scope and the conference-required section citations. Human editing may alter final wording, but disclosure must follow actual provenance. |

Relevant records: [evidence map](PAPER_EVIDENCE_MAP.md),
[component pilot](../results/resnet_component_reassembly_pilot/REPORT.md),
[CIFAR parent replays](../results/grouping_mechanism_study/parent_validation_replays.json),
[EuroSAT parent replays](../results/eurosat_grouping_fp32_replication/parent_replays.json),
[PHM scratch](../results/phm_basis_compression_pilot/REPORT.md).

## Next numerical study — subsequently authorized and prepared

The user explicitly authorized this next study after requesting manuscript
commits/push. Those commits were pushed first; the separately frozen
[repair protocol](../results/grouping_repair_curve/PROTOCOL.md)now instantiates
the design below. Five CPU tests, a full synthetic completed-resume check,
18bitwise initial-logit replays and compact CUDA update audits passed.
Read[CURRENT.md](CURRENT.md)and the result state for actual launch/progress.

Question: under a common repair schedule, does original grouping maintain a
classification advantage after a larger optimization budget, or mostly recover
faster? Both outcomes are reportable; neither is evidence for learned semantics
by itself.

Minimum design: existing source seeds2026/2027/2028, original and fixed fully
mixed role-preserving/free-role assignments, CIFAR and EuroSAT. Nine runs per
setting,18runs total. Reuse verified compact **pre-repair** initializations,
retained sets and parameter counts. No new parents, masks, draws, algebras or
compression-ratio sweep in this first study.

Use one30epoch trajectory per initialization, with prespecified snapshots at
0/1/3/10/30. Primary: paired original-minus-mixed validation accuracy and CE at
epoch30, separately by setting and source seed. Intermediate snapshots describe
recovery dynamics; no best-epoch selection, new equivalence margin after outcomes,
or significance claim from the checkpoint count. Log per-image logits and
successful/attempted AMP updates. A common training/augmentation stream is paired
within seed. Numerical failures remain failures, not reasons to select a favorable
checkpoint or add outcome-driven repair.

Important scheduler distinction: the frozen three-epoch implementation sets
cosine T_max from`config['epochs']`, so its learning rate is already at the floor
at epoch3. The old epoch3 point is not the third point of a30epoch schedule.
Restart from the saved initial compact model with a new common30epoch schedule;
do not silently extend or edit the frozen three-epoch protocol. Prespecify whether
the new schedule uses the existing1e-4initial/1e-6minimum rates and the same AdamW,
batch/augmentation/precision settings; do not tune them after validation outcomes.
The historical three-epoch recipe remains a separately labelled comparator.

Before execution: add new modules/config/protocol, verify source/input hashes,
checkpoint initialization replay, all-component gradients/updates and an exact
interrupted-resume check. Freeze all choices, disk/budget/STOP behavior and final
reporting before launch. No training was started by receiving these reviews.

Afterward, decide whether an exact-architecture scratch control or a cleaner
component-selection comparison is needed to support the final claim. A25/50/75%
compression grid, extra parents and more draws are conditional extensions, not an
automatic requirement to perform every proposed experiment.

## Theory scope to investigate

An invariance statement must specify what is held fixed. A channel permutation
absorbed into an unrestricted real consumer is not necessarily absorbable within
a Hamilton family. Simultaneously changing output roles or permitting signs is a
different problem. ReLU commutes with coordinate permutations but generally not
with arbitrary signed/rotational transformations. An algebraic symmetry need not
be a symmetry of the full BN/nonlinearity network.

A standard bound can be derived for a **fixed consumer**, but is not yet a
validated manuscript theorem. Let h=ReLU(DWx+b), h_hat=ReLU(DW_hat x+b), where D
is the copied eval-BN diagonal scale and M=E[xx^T] is an uncentered input moment.
For a fixed downstream real linear operator B, the1-Lipschitz ReLU property gives

    ||B(h_hat-h)||_{L2} <= ||B||_2 ||D||_2 sqrt(tr(DeltaW M DeltaW^T))
                       <= ||B||_2 ||D||_2 sqrt(lambda_max(M)) ||DeltaW||_F.

Apply the L2 triangle inequality with the original residual Y-Bh to bound the
new residual Y-Bh_hat. This relates producer mismatch and input statistics under
explicit assumptions. It is a standard perturbation inequality, not a novelty
claim, a held-out guarantee after fitting a different consumer, or an additive
nonnegative error decomposition. For convolutions, B/W must be the finite
convolution operators; stored-kernel projection SSE cannot be substituted for
their Frobenius norms without the appropriate spatial/overlap factors. BN in
training mode or changing BN statistics requires a separate argument. Any paper
use needs a checked derivation, relevant primary citations and an assessment of
whether the bound is informative on the actual networks.

## Draft revision and remaining status

The current editing pass adds a conceptual assignment diagram, a concise
contribution list and existing uncompressed-parent results; it removes lab-log
detail from the main narrative without deleting numerical failure records.
Validation-only status and finite repair scope remain explicit in limitations.
External GPT/Claude reviews have now been received; human author verification,
longer repair, final procedure/test audit and final test evaluation remain open.
