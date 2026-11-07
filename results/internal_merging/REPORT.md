# Structure-preserving merging inside quaternion CNNs

## Hypotheses

H1: an unrestricted fitted 4x4 map has a substantial component that cannot be folded into the next right-Hamilton convolution. H2: enforcing that structure increases held-out consumer error. H3: directly fitting an admissible quaternion map retains quality better than fitting a full map and projecting its folded kernels. H2/H3 are empirical and can fail.

## Algebraic conditions and scope

Let R(a)x represent x*a with column vectors. If a removed feature satisfies q_v=q_k*a, its consumer contribution (q_k*a)*w_v can be folded as q_k*(a*w_v), by ordinary associativity. For a general matrix A the required block is R(w_v)A. The right-Hamilton blocks form a four-dimensional subspace with orthogonal basis of squared Frobenius norm four. Orthogonal projection coefficients are <R(e_c), M>/4. If at least one outgoing w_v is nonzero, it is invertible; exact folding into that block requires A itself to lie in this subspace. All-zero outgoing weights are a degenerate exception. These are elementary algebraic facts, not a claimed new theorem.

Maps are fitted AFTER block-2 BN/ReLU/pooling and BEFORE the next linear convolution. No map is moved through a nonlinearity. Intercepts are deliberately zero: a constant feature passed through zero-padded 3x3 convolution produces a spatially varying border term, generally not one absorbable bias. This pilot does not solve general affine offset propagation.

## Prespecified protocol

Three completed seed-2026/2027/2028 CNNs, block 2 only. Fit on 500 even calibration images, all 64 spatial positions; check on the 500 odd images. Whole images remain disjoint. Use zero-intercept least squares for a four-coefficient quaternion map and a sixteen-coefficient full-real map. Common four disjoint pairs are chosen using full-real fit residual MSE times outgoing kernel coefficient energy; this is a transparent proxy, not exact consumer error. This common selection favors the unrestricted fit. It is not evidence about all possible pair policies. Freeze every map, pair and random selection before validation inference. No training, BN refit or official-test inference. Validation images served earlier checkpoint selection, so are not a new blind set.

Physically remove 4/16 producer units and matching consumer inputs. Admissible and projected models both have 66,954 real parameters (9.40% fewer than 73,898), with no extra map layer. Paired pruning and ten fixed random selections have the same size. Full substitution keeps the original network and intervenes on feature maps; it is a FUNCTIONAL ORACLE, not a compact quaternion architecture or equal-budget deployment result.

## Measured structural and consumer diagnostics

| Seed | Discarded full-map kernel energy | Full projected/oracle consumer error | Exact-Q closure max error |
| ---: | ---: | ---: | ---: |
| 2026 | 55.66% | 1.2815 | 3.34e-06 |
| 2027 | 57.49% | 1.3474 | 3.81e-06 |
| 2028 | 49.38% | 1.2604 | 3.81e-06 |

## Measured validation results

| Policy | Real parameters | Accuracy mean ± seed SD | Mean CE |
| --- | ---: | ---: | ---: |
| baseline | 73898 | 77.93% ± 1.10 pp | 0.637730 |
| random | 66954 | 73.22% ± 0.65 pp | 0.757720 |
| paired_pruned | 66954 | 71.40% ± 2.63 pp | 0.828642 |
| quaternion_exact | 66954 | 74.73% ± 0.83 pp | 0.719896 |
| full_projected | 66954 | 73.63% ± 0.65 pp | 0.738425 |
| full_substitution | functional oracle | 76.30% ± 0.26 pp | 0.683890 |

Random repetitions are averaged within each seed before SD across the three training seeds. Both fitted maps use the SAME victims and keepers. Report squared-weight projection loss, actual consumer error, classification CE and accuracy separately; none alone establishes all the others.

## Interpretation and limits

One architecture, dataset, block, compression level and fit/selection rule. The linear/no-intercept restriction is material. Full substitution is not counted as a compact model. No generic quaternion advantage, new multiplication rule, novel-method status or publication readiness is assumed. Prior context: Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002 in references.bib. A focused literature review is still needed before any priority claim.

## Verified empirical interpretation

Projection discards 49.38-57.49% of the squared Frobenius energy of ADDED compensation blocks R(w_v)A, not of the complete layer. On held-out features this raises the consumer's pre-BN reconstruction MSE by 26.04-34.74% relative to full substitution in all three seeds.

Direct admissible fitting exceeds full-fit-then-project accuracy by +0.70, +1.80, +0.80 pp; mean +1.10 pp, sample SD 0.61 pp. CE and consumer MSE also favor direct quaternion fitting in every seed. Against ten random removals averaged within each seed, paired gains are +1.05, +2.39, +1.10 pp, mean +1.51 pp.

All actual compact methods have 66,954 real parameters (9.40% fewer), 870 buffer elements and 280,464-byte standardized state files. Direct quaternion fitting gives 74.73% mean validation accuracy versus 77.93% before compression, so meaningful quality loss remains. The 76.30% full-substitution result is a functional comparison and is not a compact quaternion model at this budget.

Five meaningful implementation tests and 853 independent artifact checks pass. Exact-Q composition and graph surgery agree within 1e-4 (observed consumer maximum below 4e-6). Verification covers all pair fits/normal equations, fit-only common selection, coefficient energy/projection, tensor slicing, actual budgets, source cache alignment and checkpoint/oracle predictions.

This supplies a repeatable pilot signal for the cost of enforcing an intermediate layer's structure and for fitting within the constraint from the start. It does not establish a new algorithm or broad quaternion superiority. One architecture/dataset/block/removal level and a common selection favoring the full fit limit interpretation. A second compression level and an independent architecture/dataset are needed before positioning a paper; a focused literature review is still required. Prior context: Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002.

Important POST-HOC analytic control: projecting the full fit using the keeper activation second moment reproduces the direct quaternion least-squares map across all 720 directed pairs (maximum matrix difference 1.02e-14). No new inference or validation selection was performed. Thus the +1.10-pp comparison is against unweighted Frobenius projection; the direct-Q fit is equivalent to standard data-weighted projection in the admissible subspace. It is not a new algorithm or evidence of an intrinsically quaternion-specific optimization benefit. The measured cost of restricting a full map and the need to retain actual model structure remain valid. The article hypothesis must concern that structural trade-off, with stronger controls, rather than claim novelty from direct fitting alone. See data_weighted_projection_audit.json and scripts/audit_internal_projection.py.
