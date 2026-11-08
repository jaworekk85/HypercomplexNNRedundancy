# Hybrid/variable-n pilot audit (2026-11-08)

## Hypotheses

The all-Q selection might reflect a selector/counting bug, a favorable compression trade-off in this trained Q parent, or limitations of PHM fitting and the width catalogue. These explanations must be distinguished before interpreting algebra preference.

## Measured checks

The independent CPU audit is reproducible with:

`python -B -u -m scripts.audit_structure_catalogue`

- All 145 registered scientific source hashes match. Scientific files are unchanged.
- The two saved catalogues contain 14 and 15 candidates: 210 pairs, of which 159 satisfy the local 39,328-parameter budget.
- Independent exhaustive enumeration reproduces all five policy selections. The global variable-n optimum is Q/Q with normalized proxy error 0.374329634. Runner-up is real/Q at different widths, error 0.534510059. This verifies this finite local-proxy search, not a classification optimum.
- The dense candidate has no worse fit MSE than Q at the same retained channels. All candidate mutable parameter counts match physical modules.
- Five final checkpoint hashes match; physical model counts and finite forward passes are checked independently. All 1,125 attempted repair updates succeeded per policy.
- The all-Q and variable-n final checkpoint hashes are identical; their equal accuracies are not independent evidence.
- Replacing each selected Q consumer by a PHM4 factorization reconstructs its kernel to relative error below 4e-15. Maximum float32 output differences on 16 fixed held-out calibration images are below 3e-6. No official-test inference or fitting on these images.
- At the selected widths, saved Q fit MSEs are 0.724016 / 0.181356 for blocks 0 / 8. The original PHM4 initialization from a dense oracle has fit MSEs 1.635878 / 0.303789. These are two different initialization/fitting procedures.
- The exact PHM4 representation requires 64 extra stored basis coefficients per consumer. Both same-width PHM4 replacements exceed the local budget by 128 coefficients (39,456 versus 39,328). The frozen catalogue only provides widths 1/2/4 and 4/8/16 respectively, so it cannot spend the remaining budget through intermediate widths.
- Six original structure-catalogue CPU tests pass, including PHM gradients, Hamilton inclusion, independent kernel assembly and small exact selection.

## Interpretation and limitations

No selector, storage-counting or PHM implementation bug was found in these checks. This does not prove absence of every possible bug.

The scientifically supported observation is: for this Q-trained parent, two blocks, this discrete catalogue and this fitting recipe/budget, retaining Q consumers minimizes the local proxy and gives 89.3% validation after repair.

This does not establish general quaternion superiority. Candidate fitting is asymmetric: Q directly minimizes activation-output reconstruction with ridge, while PHM uses unweighted Frobenius SVD of a dense fitted kernel. PHM4 contains every Q kernel, yet that initializer can produce a worse output reconstruction. The extra basis storage is a genuine deployment cost, but the coarse width catalogue amplifies its impact: 128 extra coefficients exclude the same-width PHM4 pair and force a large width drop. Neither factor is a selector coding error; both limit the experiment's interpretation.

The weight-only selector's score is distance to each width's own dense oracle. Dense fits therefore have nearly zero scores even when their target-reconstruction errors are large. This is a weak diagnostic control, not a meaningful common oracle across widths; its poor validation result alone is not evidence that activation selection generally dominates strong weight-based baselines.

## Recommended bounded follow-up

Keep all pilot results and the frozen numerical sources. Before a larger run, add a separate diagnostic comparing PHM4 initialized exactly from Q with the original dense-to-PHM SVD, using the same activation objective and explicit counts. Check finer admissible widths or separately label a budget-slack sensitivity control (+128 coefficients); do not present the relaxed budget as exact parameter matching. Only after that check should a multi-seed or broader layer allocation be considered. No new scientific training was performed by this audit.

Canonical prior-art context: `ZhangEtAl2021`, `Grassucci2024`, `VanLoanPitsianis1993`, `Liebenwein2021ALDS`, `Frantar2022SPDY` in `references.bib`.
