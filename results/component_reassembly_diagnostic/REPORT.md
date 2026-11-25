# Component reassembly: synthetic realizability diagnostic

Status: completed.

## Hypothesis

Selecting individual scalar components and assigning them to new quaternion roles may give useful compact initializations when the selected producer rows are jointly Hamilton-realizable. A consumer-only criterion can overestimate this benefit by assuming access to full-teacher features.

## Measured results

| Synthetic case | Selection | Quartet indices | Fit MSE | Held-out MSE | Producer projection SSE | Stored reals |
|---|---|---|---:|---:|---:|---:|
| aligned_complementary | best_whole_unit | [0, 2, 4, 6] | 0.378487 | 0.425194 | 0 | 16 |
| aligned_complementary | scalar_energy | [0, 3, 4, 7] | 0.00101541 | 0.00101327 | 0.00578339 | 16 |
| aligned_complementary | consumer_only_selection | [0, 3, 4, 7] | 0.00101541 | 0.00101327 | 0.00578339 | 16 |
| aligned_complementary | realizable_selection | [0, 3, 4, 7] | 0.00101541 | 0.00101327 | 0.00578339 | 16 |
| unrelated_complementary | best_whole_unit | [1, 3, 5, 7] | 0.485314 | 0.521865 | 0 | 16 |
| unrelated_complementary | scalar_energy | [0, 3, 4, 7] | 0.240139 | 0.286526 | 1.79753 | 16 |
| unrelated_complementary | consumer_only_selection | [0, 3, 4, 7] | 0.240139 | 0.286526 | 1.79753 | 16 |
| unrelated_complementary | realizable_selection | [0, 3, 4, 7] | 0.240139 | 0.286526 | 1.79753 | 16 |

## Mathematical construction

For target scalar producer rows b_a assigned roles a=r/i/j/k, let T_a
be the signed permutation taking quaternion coefficients to output row a.
Pull them back as c_a=T_a^T b_a. The optimal shared coefficients are
their mean. Frobenius projection error is sum_a||c_a-mean(c)||^2,
equivalently one quarter of sum_{a<b}||c_a-c_b||^2. Original intact
quartets have zero error. Arbitrary regrouped quartets generally do not.
This identity is elementary projection algebra, not a novelty claim.

The consumer is fitted on actual compact-producer BN/ReLU outputs.
Full-teacher selected activations are an optimistic diagnostic only.
The final forward contains one smaller producer, its BN/ReLU and one
smaller consumer; no full teacher, gather, router or dense adapter.

## Interpretation and limits

Technical CPU diagnostic only. Two deliberately constructed synthetic teachers with complementary BN/ReLU activation patterns; no CIFAR, pretrained scientific checkpoint, classification labels, test predictions, fine-tuning or GPU training. Exhaust all8P4=1680ordered selections of4distinct scalar channels from8. One new producer and consumer quaternion unit; both old producer units are physically replaced, with no gather or full teacher in the deployed student.

Private seeded generators. Teacher producer kernels are nearby in one case and independent in the other. Consumer coefficients are identical across the two teacher hidden units. BN biases alternate by component: unit0 positive at r/j and negative at i/k, unit1 the opposite. This is a favorable mechanism demonstration plus a less compatible control, not representative performance evidence. Fit/check input draws are separate; all assignment selection is fit-only.

Select/order four pre-BN effective producer rows; orthogonally project onto the four-parameter Hamilton kernel space in Frobenius norm. Copy their BN affine/running values by role. The four pulled-back row coefficient vectors must agree for exact representability; their mean minimizes squared error, equivalently one quarter of the sum of pairwise squared distances. This is elementary projection algebra, not a novel theorem. Projection is not an optimum for post-ReLU or activation-weighted loss.

Best whole-unit chooses the lower fit-error original quartet. Scalar-energy retains the4largest fit activation second moments and assigns sorted scalar indices to roles. Consumer-only minimizes fit reconstruction using selected full-teacher features, then deploys the projected producer and refits its consumer. Realizable-selection minimizes actual compact-pair fit error over all1680ordered candidates. Ties lexicographic. Held-out synthetic check does not select candidates.

After routing is compiled out, all students share the same small Hamilton architecture. Reassembly changes source-guided initialization/selection, not the final architecture family or guaranteed optimum after unlimited training. No claim of methodological novelty, classification gain or IJCNN readiness; compare nearest learned-shuffle/compression work before any scientific extension. No automatic follow-up.

Teacher:32real parameters; all compiled students:16, including BN affine.
No labels, classifier accuracy, fine-tuning or CIFAR inference were used.
These deliberately constructed examples are not evidence of real-data gains.
Prior art: Zhang2021StructuredShuffle, Wang2019FLGC, Vecchi2020, VieiraValle2022ELM, He2017ChannelPruning, GRAIL2026 (references.bib).

## Post-run descriptive interpretation

All three scalar-selection methods chose [0,3,4,7] in both fixed cases.
Thus this diagnostic demonstrates a possible benefit of component reassembly
over intact-unit retention, but does not demonstrate a better selector than
scalar energy. The winning quartet preserves original component roles and
combines two original units; free role reassignment has no demonstrated added
benefit in these examples.

With the original full-teacher features available, the selected quartet's
held-out consumer MSE is approximately1e-16 in both cases. After physically
replacing the producer, MSE is0.001013 and0.286526 respectively. This exposes
the ideal-feature versus realizable-feature gap. Producer projection SSE is
0.005783 versus1.79753. It does not establish that the proposed selection
criterion improves generalization. Post-hoc top20ranking overlap between ideal
and realizable fit objectives is2/20 and3/20; winners nevertheless agree.
Details are retained in selection_analysis.json. No extra cases were generated.
