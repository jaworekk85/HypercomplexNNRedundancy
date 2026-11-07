# Annotated bibliography — hypercomplex activation redundancy and compression

**Updated:** 2026-11-06
**Working paper:** *Activation-Based Redundancy Analysis and Compression of Hypercomplex Neural Networks*  
**Citations:** `references.bib` (keys below match exactly).

This is a **curated, verified starting database**, not a systematic review proving priority. Descriptions below paraphrase publisher/proceedings summaries and explain why each article matters for *this* project. Read full text before writing strong claims or exact methodological comparisons. URLs point to publisher landing pages/DOIs or freely available author versions; `extra` links lead to code or preprints. A missing DOI is intentionally omitted (not guessed).

## Read-first map

**Direct prior art (P0):** `NiemczynowiczKycia2025`, `NiemczynowiczKycia2026`, `Vecchi2020`, `Singh2026`, `Ayinde2019`, `Kim2020`, `Ebid2025`, `Maroudis2026`.  
**Methodology and theory (P1):** `GanguliChong2024`, `HuangPao2025`, `Kobayashi2010`, `IqbalMishra2023`, `Chaudhary2024`.  
**Wider context (P2):** the remaining works.

### Literature-to-claim guardrails

- **Already known:** quaternion weight/structured pruning; activation-dependent pruning in real networks; similarity-based filter pruning; neuron merging and compensation; complex-valued structural reducibility.
- **Not established as novel:** using quaternion group norms, ordinary activation cosine, four-component feature correlation, or a quaternion least-squares fit by themselves.
- **Possible research gap to test:** systematic measurement of *activation-representation redundancy* in trained QNNs and its predictive/practical value for physically compact quaternion networks, under fair controls.
- **Possible failure mode:** the group-aware method may behave no better than a standard feature-based metric. Record that transparently.

## Priority audit of direct-Q post-pruning reconstruction (2026-11-06)

**Verdict:** no exact predecessor was located in this targeted review. This is
evidence for investigating a narrowly scoped contribution, NOT confirmation of
worldwide first-in-literature status. In particular, the solver principle is
already covered by `VieiraValle2022ELM`, and next-layer reconstruction after
channel removal is covered by `He2017ChannelPruning` and `Liang2021TwoStep`.
Combining known ideas may still yield a useful method/empirical paper; novelty
and non-obviousness do not follow from an unsuccessful exact-phrase search.

### Exact claim screened

Start with a trained Hamilton-weight-sharing quaternion CNN; remove whole Q
channels and physically shrink the model; fit the surviving next convolution's
Q coefficients directly to its pre-BN output targets on unlabeled calibration
images, with fixed masks and centered ridge; retain the Hamilton blocks and
compare to shared-channel-map folding at the SAME compact architecture and
real parameter count. Our implementation refits five Q consumers; the final
already-real readout keeps map-fold only. Three-epoch supervised repair is a
separate stage, not part of the unlabeled closed-form fit.

### Observations from primary sources

| Canonical key / inspected version | Relevant established content | What was not located in that inspected method |
| --- | --- | --- |
| `Vecchi2020`, arXiv:1907.11546v3, sec.3 | Whole-quaternion weight norm and BN-scale sparsity remove Q connections/neurons during learning | Post-pruning direct consumer-output Q regression |
| `IqbalMishra2023`, arXiv:2308.06780v1, sec.4 | Global iterative weight pruning and retraining from initial weights | Physical whole-channel compaction with direct output reconstruction |
| `Mukhopadhyay2024`, official PMLR PDF, sec.3.4.2 | L1 unstructured masks and reset/train/prune loops in large Q/real ResNets | The Q consumer reconstruction procedure above |
| `Chaudhary2024`, official ISCA PDF, sec.3.2.2 | Norm-based quaternion filter pruning followed by fine-tuning | Closed-form Hamilton-constrained output refit |
| `Singh2026`, arXiv:2510.21388v1, secs.III-IV and IX-B; journal metadata separately verified | L1/geometric-median/operator-norm filter removal, fine-tuning and output-logit KD | Direct local Q output regression in the preprint; accepted-PDF identity UNVERIFIED |
| `He2017ChannelPruning`, arXiv:1707.06168v2, sec.3.1(ii) | Refit surviving convolution weights by LS to recover next-layer outputs | Hamilton weight ties / whole-Q-group repair |
| `Luo2017ThiNet`, arXiv:1707.06342, sec.3.2.3, Eq.7 | Next-layer-aware selection and LS channel rescaling before fine-tuning | All-Q-coefficient Hamilton-constrained refit; its LS parameters are scalings, not arbitrary kernels |
| `KammaWada2021REAP`, official PDF, secs.3.1-3.5 | Multi-source feature reconstruction, folded outgoing-weight updates and im2col extension | Hamilton-constrained quaternion consumers |
| `Liang2021TwoStep`, publisher full text, sec.3 Eq.8 | Feature-map selection followed by direct LS refit of next-layer filters | Quaternion kernel restriction |
| `VieiraValle2022ELM`, arXiv:2101.06166v1, secs.3-4, Eqs.22-25/30-31; publisher 2022 record | Hypercomplex LS by real embeddings; fit hypercomplex output weights | A post-channel-pruning CNN study or map-fold/direct-Q comparison |
| `GRAIL2026`, official PMLR record and arXiv:2602.23795v2 secs.3.1-3.2 | Gram/ridge feature reconstruction absorbed into consumers; head/GQA structure explicitly considered | Hamilton-tied consumer-kernel refitting in the inspected text |
| `IbnAhad2026HypercomplexQuantization`, arXiv:2512.08524v1 sec.3/Algorithm1 | Dense-to-PHM replacement with operator reconstruction and distillation | Channel pruning of an already-Q teacher at fixed compact Hamilton family |

Primary links are in references.bib and the entries below. New particularly
important sources: [Vieira and Valle](https://www.sciencedirect.com/science/article/pii/S2772415822000062),
[inspected ELM preprint](https://arxiv.org/pdf/2101.06166),
[REAP full text](https://www.jstage.jst.go.jp/article/transinf/E104.D/1/E104.D_2020EDP7049/_pdf/-char/en),
[two-step reconstruction](https://link.springer.com/article/10.1007/s11760-021-01888-4),
[ThiNet preprint](https://arxiv.org/pdf/1707.06342).

### Interpretation: the solver is an existing special case

For fixed surviving input patches, write the Q convolution as D_Q theta, with
theta containing the four independent real coefficients of each Hamilton block.
Our problem is

    min_theta ||D_Q theta - y||^2 + lambda ||theta - theta_fold||^2.

Setting delta = theta - theta_fold reduces this to ordinary ridge regression
on residual targets y - D_Q theta_fold. Its normal-equation solution, real
embedding, positive-ridge uniqueness and feasibility ordering are established
linear algebra; `VieiraValle2022ELM` gives the directly relevant hypercomplex
LS framework. This equivalence is OUR mathematical comparison, not a claim
that Vieira/Valle describe our pruning protocol. It substantially limits any
claim to inventing a quaternion reconstruction solver. Spatial patch unfolding
and using a layer's old outputs as regression targets are also established.

The possible contribution is the explicit post-pruning procedure and its
controlled analysis in compact QNNs, including the additional restrictions of
shared-map folding. Do not claim that comparing current-feature reconstruction
with next-layer reconstruction is itself a new global principle: the two-step
paper already explicitly distinguishes those roles. Our discovery-only local
diagnostic compares nested update spaces under ONE objective; the replicated
network comparison changes the old objective/regularization and parameterization
together. Neither is an irreducible architecture lower bound.

### Wording supported now, and wording to avoid

**Later clarification for the authorized tail-sensitive pilot:**
`Park2020Lookahead` concerns ordinary CNNs, but logit distillation of pruned
QUATERNION CNNs is already explicit in `Singh2026` arXiv v1 sec.IX-B, using
the CE+KL loss of Eq.3 and evaluating Prune_KD versus Prune_FT. Consequently
"first distillation of compressed QNNs" is not a candidate claim. Our pilot
instead adjusts one Q consumer on 500 unlabeled calibration images using
logit MSE and frozen tail/head/BN, compared with equal-step local MSE. This is
not a reproduction of Singh's full supervised KD recipe. A different loss,
scope and teacher do not by themselves establish methodological novelty.
Full primary method inspected: https://arxiv.org/html/2510.21388v1#S9.SS2.

Preferred contribution statement (does not need a first claim):

> We formulate and evaluate Hamilton-constrained output reconstruction for
> physically compact quaternion CNNs, and quantify the additional restrictions
> of shared-map compensation at matched compact architectures.

If an explicit priority sentence is later retained, its defensible scope is
conditional: to our knowledge, no prior study jointly compares shared-map
folding and direct Hamilton-constrained consumer reconstruction after whole-Q
channel removal at matched compact architectures. This review LOCATED NO such
study; it does not prove there is none. Also, being first to compare against
our own restricted baseline is not by itself a strong contribution. Explain
what a reader learns beyond the combination of known pruning and Q regression.

Avoid "first QNN compression", "first structure-preserving Q pruning", "first
quaternion LS/output fit", "first hypercomplex reconstruction-based compression",
and generic "first structure-aware compensation". These broader claims overlap
the papers above. Do not imply outperforming published methods: our measured
gains compare our implementations, not reproduced full He/REAP/GRAIL recipes.

### Search scope, access gaps and reproducibility

Search date 2026-11-06. General web discovery with quaternion/hypercomplex,
channel/filter/neuron pruning, reconstruction/compensation/merging,
least-squares/ridge, Hamilton/structure-preserving and ELM combinations; some
queries restricted to arXiv/OpenReview. Followed primary references into
publisher/proceedings/author PDFs and inspected method sections. This was NOT
a complete Scopus/Web of Science/Google Scholar citation census or patent
clearance. No author contact, external issue or model reproduction.

Most false positives concerned rotation quaternions in Gaussian splatting,
attitude estimation or unrelated financial services. An unverified Lacuna
research-proposal hit about pruning algebra multiplication components was
screened out: full page inaccessible and no verified scholarly version;
search snippets describe changing multiplication components, not our fixed
Hamilton channel-pruning procedure. It is not evidence of published priority
either for or against us. Surveys and generated aggregators served only as
discovery leads, not evidence for method claims.

Important access gap: King's College London lists an accepted manuscript for
`Singh2026`, [TASLP_QCNN_Final.pdf](https://kclpure.kcl.ac.uk/portal/files/369520366/TASLP_QCNN_Final.pdf).
Both browser fetch and direct public-file retrieval failed (Cloudflare/403).
The accessible arXiv v1 and final journal abstract differ in reported summary
numbers; do not infer identical method versions. GRAIL published PDF still
returns an unsupported content type; the method comparison uses v2 preprint.
PHM comparison continues to use inspected arXiv v1, not an assumed identical
published WACV PDF. These gaps prevent an unqualified priority confirmation.

Raw retrieved excerpts/search batches and the query/claim log are cached in
`.cache/literature/priority_review_2026_11_06/`. The review added no experiments
and changed no hash-frozen numerical source, configuration, checkpoint or result.

## Learning compensation coefficients during fine-tuning (2026-11-06)

User suggested additionally learning the quaternion reconstruction coefficients
a during repair. The bounded two-arm pilot was subsequently authorized;
its registration is results/trainable_maps_pilot/protocol.json. The earlier
proposal below describes the general construction, not its measured success.
Current pipeline fits a by feature ridge, folds it away, then fine-tunes compact
weights U; it does NOT keep a as a separately optimized variable. The cached
shared-map diagnostic reoptimized maps under one local quadratic objective only,
not jointly with the model under a supervised task loss.

Use U_{ok,d}(A,B,C) = B_{ok,d} + sum_v a_{kv} * C_{ov,d}, where B is the
retained kernel and C contains removed-input outgoing weights kept as fixed
training-time constants. Initialize A from existing ridge, B/C from the source
so that the effective starting kernel exactly equals original map-fold. Optimize
ordinary classification CE through U(A,B,C), differentiating the four real
components of each a. No distillation is required. At deployment fold U and
discard A/C. Hamilton closure holds inside the consumer kernel, without moving
maps through BN/ReLU. Input feature maps can remain physically narrowed.

Important distinctions and hypotheses:

- If B/C and the rest of the model are frozen, learning only A restricts
  updates to the shared-map subspace. It changes the objective from feature
  reconstruction to classification, not the map family's dimension.
- If B and A both train and C is fixed, B is an unrestricted Q kernel over
  the retained channels. Every compact Q kernel remains representable, so
  the functional family is unchanged; parameterization and optimization differ.
  In real coordinates theta=b+T a, with T fixed by C. Under simultaneous plain
  SGD, no decay/regularization/momentum, induced update is
  delta_theta = -(eta_B I + eta_A T T^T) grad_theta L.
  This is our elementary chain-rule analysis, not a claim of a new optimizer
  theorem. Adam/regularizers/learned C require separate analysis.
- More training variables/resources do not imply more deployed parameters.
  Keeping all maps adds 4|K||V| trainable reals per site, plus frozen C storage,
  relative to learning B alone; final U has the ordinary compact Q count.
  If C also trains, its extra trainable count must also be reported.
- Re-running descent on the unchanged original ridge objective with fixed
  features cannot beat its exact optimum. A new question requires changed
  features, task objective, constraints, regularization or optimization dynamics.
- Natural small comparison: fixed K/V and initialization, same labeled data
  and repair schedule, ordinary compact-Q fine-tuning versus joint B+A learning
  with C frozen, final exact-fold check and actual compact parameter counts.
  An A-only arm is diagnostic if later needed. The separately authorized pilot
  trains A+B in one last consumer only; no A-only sweep is queued.

**New close primary literature:** `Ding2021ResRep` learns ordinary-CNN 1x1
compactors with a pruning-specific gradient rule and converts the trained
structure into narrower convolutions. `Zhang2026RCP` explicitly learns a
compensation operator jointly with task training under a fixed channel
partition, then consolidates branches at inference. These are strong prior
art for the BROAD idea of learning temporary compensation maps and folding
them away. RCP's indexed method places its projector within a linear block
before activation; our maps concern post-activation features folded into the
following consumer, so they are not asserted to be identical constructions.
Their reported experiments are ordinary CNNs; no exact Hamilton-constrained
variant was verified here. Adding quaternion coefficients alone is not
evidence of sufficient novelty. Do not claim first learnable-map compensation.

Sources: [ResRep official CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.html),
[RCP publisher](https://www.mdpi.com/2073-8994/18/4/585). Read indexed primary
method passages and metadata; direct ResRep PDF returned 403, RCP HTML 429 and
PDF inaccessible. This is targeted discovery/reading, not an exhaustive
priority review or a reproduction. Cached excerpts:
.cache/literature/trainable_maps_review_2026_11_06.txt.

## Targeted QNN review of task-trained compensation maps (2026-11-06)

**Verdict:** no exact predecessor located in the inspected sources; priority
and sufficient methodological novelty remain UNCONFIRMED. The previous
ordinary-CNN search did not adequately answer the user's QNN-specific question.
This follow-up contains 36 queries (34 logged query-array entries plus two
additional queries retained in qmap_primary_3.txt), primary-method reading and
explicit exclusions. It is not an exhaustive bibliographic census.

Exact target: start from an already-trained Hamilton QNN, fix removed/retained
whole-Q channels, task-train quaternion cross-channel maps under supervised CE,
and absorb them into the SAME physically compact Hamilton consumer family.
The registered pilot uses U=B+sum delta_a*C with zero-initialized corrections
around the existing direct-Q kernel, rather than absolute old ridge maps;
C remains frozen. Only the last Q convolution gets 1024 temporary trainables.
Both arms start identically and export to 19098 real parameters; joint repair
temporarily trains 20122 versus 19098 in the control. This tests optimization
within one functional family; it does not test a larger deployed capacity.

| Canonical key / inspected scope | Established content | Relation to the exact target |
|---|---|---|
| Vecchi2020, arXiv v3 sec.3 | Q-norm sparsity and trained scalar BN scales | BN gamma is not a cross-channel quaternion reconstruction map |
| Chaudhary2024, official PDF sec.3.2.2 | Q filter norm pruning and fine-tuning | No jointly learned K-to-V maps located |
| Singh2026, arXiv v1 III-IV/IX-B | Q filter pruning, CE/KL distillation and fine-tuning | No exact map-and-fold construction located; accepted-PDF identity unverified |
| IqbalMishra2023 / Mukhopadhyay2024, earlier method audit reused | Unstructured Q masking, reset/train/prune | Does not establish physically compact task-trained maps |
| IbnAhad2026HypercomplexQuantization, arXiv v1 sec.3 | Dense-to-PHM replacement, fixed 2x2 basis, learned cores, reconstruction/KD and residual fading | Not removal of channels from an existing Q teacher with learned Hamilton K-to-V maps |
| Mahabadi2021Compacter, arXiv:2106.04647 secs.2-3, Eqs.2/4/5 | Learned shared PHM factors and low-rank nonlinear adapters | Learning hypercomplex coefficients is established; adapter nonlinearity differs from exact consumer folding |
| Cao2024GDPL, arXiv:2405.08668v1 secs.5.2-5.4/Appendix B | Quaternion prompts and separately real-valued cross-modal LoRA factors | Not quaternion compression-map training; quaternion keywords alone do not establish equivalence |
| Ding2021ResRep, indexed official method/Algorithm1 | Trainable real-CNN compactors, gradient treatment and folding | Strong broad prior art; no exact Hamilton version verified |
| Zhang2026RCP, indexed publisher method | Task-trained compensation and exact inference consolidation in ordinary CNNs | Closest broad principle; placement differs, and Hamilton-specific procedure unverified |
| Ding2023RepOptimizers, primary paper/author metadata | Structural priors expressed via gradient reparameterization | Optimization through redundant parameterization is not a new general principle |

Primary sources: [Compacter](https://arxiv.org/pdf/2106.04647),
[GDPL](https://arxiv.org/html/2405.08668v1),
[RepOptimizers](https://arxiv.org/abs/2205.15242),
[Q pruning preprint](https://arxiv.org/html/2510.21388v1),
[PHM compression preprint](https://arxiv.org/html/2512.08524v1).
Citation records are canonical in references.bib. Query log and retrieved
excerpts: .cache/literature/qnn_trainable_maps_review_2026_11_06/.

Access limitations retained: Singh accepted PDF and PHM final CVF PDF blocked;
do not infer version identity. ResRep/RCP comparison uses substantial indexed
primary method passages where direct documents were blocked. Unstructured-Q
comparisons reuse explicitly documented previous method reading. Search false
positives included ordinary networks predicting rotation quaternions, Gaussian
splatting, complex-valued sparsification and metaheuristic quaternion DBN
hyperparameter search; these are not evidence for the exact QNN construction.
No author's unpublished knowledge was solicited and no priority claim is made.

## Per-victim quaternion centers with jointly calibrated maps (2026-11-06)

User authorized a bounded pilot of U=B+sum_v A[k,v]*(C[o,v,s]-mu[v]),
one Q center per removed channel, shared across outputs and offsets. Supplementary
three-query targeted search is cached at
.cache/literature/centered_maps_review_2026_11_06.txt. No exact predecessor located;
this small follow-up and the earlier36-query review do NOT prove priority.
Q quantization/codebook centroids are not automatically this compensation.
Singh2026 arXiv v1 was reopened; RCP publisher again returned429. Preserve earlier
version/access gaps and broad learned-map/folding prior art.

Our algebraic observation (not a literature novelty claim): d[k]=sum_v A[k,v]*mu[v]
gives U=B+A*C-d, linear in A,d. Solve the common output/kernel-centered-ridge
problem over A,d, then recover mu by the real representation of the left-Q
linear system A*mu=d. If recovery accurately reproduces d, the relaxed optimum
is feasible in the original bilinear family; otherwise it is merely a relaxation
and the registered pilot must report failure, not assert the centered solution.
Recovery rank/condition and actual factor-to-kernel error are recorded. This is
elementary linear algebra, not a new solver theorem (VieiraValle2022ELM context).

Registered scope: configs/centered_maps_pilot.json. One late CIFAR BN2026 f=.5
consumer, four cases old fold / output-calibrated A / calibrated mu / joint A,mu;
fixed B,C/masks during calibration, same held-out fit/check halves and three-epoch
supervised repair afterward. A0 is old nonzero feature-ridge maps, NOT zero
correction factors from the last CE pilot. Common effective-kernel ridge
SSE+lambda||U-U0||^2 avoids incomparable penalties on redundant factors.
Centered anchor U0+(A-A0)C-A*mu removes initialization roundoff differences.
No fully independent mu[o,v,s], bias across padding, new algebra, priority claim,
new test, hyperparameter sweep or automatic replication. Deployed size unchanged;
calibration variables A1024/mu64/joint1088 and temporary constants reported.

## Stronger-backbone validation reference (2026-11-06)

He2016ResNet, original arXiv:1512.03385 PDF sec.4.2/Table6, inspected directly:
CIFAR-specific6n+2 architecture, real ResNet20 with0.27M parameters and8.75%
test error (91.25% accuracy), ResNet32 error7.51%. This is published real-model
evidence motivating a stronger baseline, not our measured outcome or a promise
for quaternion ResNets. Residual pruning must account for channel compatibility
at additions; our proposed first scope is internal basic-block channels with
fixed outer/skip dimensions. Larger-scale validation alone is not novelty.
Primary: https://arxiv.org/pdf/1512.03385. Official CVF PDF returned403.

## Post-compression learned-PHM follow-up review (2026-11-07)

User supplied a Claude proposal for selecting per-layer structure after channel
pruning/compensation. This is feasible as a small diagnostic, NOT established
novelty. Primary material rechecked: Grassucci2024 author preprint, original
PHM ZhangEtAl2021 metadata, IbnAhad2026HypercomplexQuantization arXiv v1 secs3
and selective-capacity equations, VanLoanPitsianis1993 publisher chapter/author
PDF indexing. OpenReview required browser verification; direct VanLoan author
PDF403, publisher metadata/abstract accessible. Prior PHM final-version gaps
remain. Research plan: configs/phm_postcompression_diagnostic_plan.json.

Corrections/own mathematical analysis:
- Our62.4-90.6% diagnostic concerns SHARED-MAP ties within Q; it is not the
  measured cost of the Hamilton family versus unrestricted real consumers.
- PHM is not solely trained from scratch. IbnAhad2026HypercomplexQuantization
  converts pretrained dense weights to fixed2x2-basis PHM, uses reconstruction,
  removes a dense transition branch and assigns basis count2/3 per layer. Its
  grouping dimension is fixed2, not the proposed variable n. This distinction
  does not justify a broad first-post-hoc/first-hybrid-capacity claim.
- For n dividing scalar O/I, convolution W_s=sum_{j=1}^r A_j tensor S_{j,s},
  shared A_j in R^(n*n), requires r*n^2+r*O*I*kernel_area/n^2 stored reals.
  Standard PHM takes r=n; do not conflate rank and group dimension. Fixed-Q
  n=r=4 has no learned64-real basis overhead. Learned bases count for storage
  even if frozen as buffers. Equal n/width is not exact parameter matching.
- A rearrangement turns unconstrained Kronecker-rank approximation at a fixed
  partition into a matrix-rank problem; truncated SVD solves Frobenius error,
  not generic activation-covariance-weighted reconstruction. No new theorem
  or general global-optimum guarantee for alternating weighted fits.
- Already-Hamilton kernels are exactly representable at PHM4/r4; projecting
  those alone yields no informative advantage. Use a locally output-fitted
  dense reference, then compare Q/PHM/output error against actual size.
- Learning factor matrices does not by itself impose associativity, identity
  or division-algebra properties. Prior small grouping pilots do not establish
  universal irrelevance of grouping. Existing ResNet widths stay fixed; only
  compatible n2/4/8 considered, no padding/n3/6-driven architecture redesign.

Primary links: https://arxiv.org/pdf/2110.04176,
https://arxiv.org/html/2512.08524v1 (sec3, dense initialization and selective
capacity assignment), https://link.springer.com/chapter/10.1007/978-94-015-8196-7_17.
No experiment or current-worker change follows from this literature update.

## Novelty audit: automatic per-layer Q/real/variable-n PHM allocation (2026-11-07)

**User's exact target:** EXTEND the proposed post-pruning Q/real hybrid to
choose one deployed structure per consumer layer from fixed-Hamilton Q,
ordinary dense(n1), and learned-PHM candidates n2/4/8, potentially with
different internal widths, under one actual global parameter budget. Fit
candidates to pre-pruning consumer outputs on calibration inputs after channel
removal; retain residual output shapes and record exact compact storage.
This is discrete model selection, not merely fitting PHM basis matrices at
fixed n, nor a soft mixture that keeps every branch for deployment.

**Verdict:** no exact earlier instance of that full combination was located in
this targeted20-query review. Worldwide priority is NOT established. Broad
automatic selection of per-layer grouping/rank/compression and adaptive use of
different hypercomplex dimensions are already known. Merely adding PHM to a
standard budgeted selector is at clear risk of being incremental. A defensible
narrow contribution requires useful measured results/insight and controls, not
the assertion that no exact keyword match was found. No new method/performance
claim and no novel knapsack/DP/SVD/ALS theorem follows.

| Canonical source / inspected scope | Established content | Difference from exact candidate |
|---|---|---|
| ZhangEtAl2021, original PDF introduction/sec3/appendix | User-selected n; learned PHM factor matrices; experiments n2/4/8/16 | Not automatic hard per-layer n allocation |
| Grassucci2024, existing PHC review | Learned hypercomplex convolution factors, flexible n | Flexible dimension alone is not the new contribution |
| Liebenwein2021ALDS, NeurIPS PDF and arXiv secs2.1-2.3 | Post-hoc convolution compression, automatic per-layer channel subspace count and rank under a global parameter budget | Uses channel-sliced low-rank factors, not a Q/dense/PHM catalogue; major broad-principle prior art |
| Frantar2022SPDY, official ICML record/PDF introduction and layerwise optimization | General constrained layerwise profile selection using dynamic programming/error scores, instantiated for sparsity and inference speed | Not hypercomplex dimension selection; generic budgeted allocation solver is established |
| IbnAhad2026HypercomplexQuantization, already-read arXiv v1 sec3 | Pretrained dense-to-PHM conversion, output reconstruction, fixed2x2 bases, depth-based basis-count2/3 allocation | Fixed block dimension; not learned variable-n PHM or hard Q-vs-PHM selection after pruning, but very close application context |
| Cao2026Numerion, official ICLR2026 record and arXiv2510.03251v1 secs3.2.2-3.2.3/Eqs9-10 | Parallel real/complex/Q/octonion/sedenion branches n1/2/4/8/16 and learned adaptive soft fusion | Keeps parallel fixed Cayley-Dickson branches for forecasting; not post-hoc pruning, learned PHM bases, hard one-choice-per-layer or global storage allocation. Strong prior art against a broad first-adaptive-algebra/dimension claim |
| Sfikas2023SharedOperation, author PDF secs3-4.3 | Co-learned operation tensor shared across PHM layers and size-matched comparisons | Uses predefined dimensions16/32; sharing operations is not automatic per-layer dimension selection |

Candidate-specific question still worth a bounded test: does a hard structure
choice AFTER pruning/output compensation improve quality versus actual stored
size beyond a Q/real-only hybrid, fixed-n learned PHM and a simple common-budget
selector? The selection must respond to the reconstruction target/covariance,
not only a static weight-space SVD. An empirical result identifying when the
fixed Hamilton constraint is worth retaining is a possible contribution, not
yet an established finding. Failing to beat simpler alternatives ends this
branch; do not disguise a negative as methodological novelty.

Required comparisons before any strong claim: Q/real hybrid WITHOUT variable-n
PHM (direct ablation of the user's extension), uniform/fixed-n PHM at actual
matched global budgets, and a simple prespecified allocation baseline using
the same fitted catalogue/errors/costs. Attribute generic optimization to
ALDS/SPDY context; a head-to-head reproduction has NOT been run. Record n
separately from Kronecker rank r and fixed Hamilton versus learned n4 bases.
Equal widths/n do not imply equal parameter count; include all basis storage.

Review scope:20 targeted queries; abstracts/metadata are not all full-method
proofs. ALDS primary PDF confirms five authors despite four on landing page.
Numerion official record confirms ICLR2026; detailed method read in arXiv v1,
accepted PDF exceeds browser18MB limit, version identity unverified. Prior PHM
final-version gaps remain. Search noise included PHM=prognostics/health,
mathematical hypercomplex manifolds, patents and AI-generated proposal sites;
these were not treated as peer-reviewed proof of the exact method. No author
contact, external message, numerical source mutation or additional GPU job.
Query log/excerpts: .cache/literature/structure_selection_novelty_2026_11_07/.

Primary links: https://proceedings.neurips.cc/paper/2021/file/2adcfc3929e7c03fac3100d3ad51da26-Paper.pdf,
https://proceedings.mlr.press/v162/frantar22a.html,
https://arxiv.org/html/2510.03251v1,
https://proceedings.iclr.cc/paper_files/paper/2026/hash/5eca2e4fe7858cbbfef4e08573cfcb25-Abstract-Conference.html.

## Core papers — read first

### `NiemczynowiczKycia2025` — KHNN: Hypercomplex-valued neural networks computations via Keras using TensorFlow and PyTorch (2025)

**Venue:** SoftwareX.  
**Read:** https://doi.org/10.1016/j.softx.2025.102163  
**DOI:** `10.1016/j.softx.2025.102163`  
**What it does:** Presents KHNN software with dense and convolutional hypercomplex layers for TensorFlow/Keras and experimental PyTorch variants.  
**Why it matters:** Primary candidate implementation; verify currently working code, examples, license and version compatibility.

**Additional resources:** [code](https://github.com/ElsevierSoftwareX/SOFTX-D-25-00004) · [code](https://github.com/rkycia/KHNN) · [code](https://github.com/rkycia/HypercomplexKeras) · [package](https://pypi.org/project/HypercomplexKeras/)

### `NiemczynowiczKycia2026` — Fully tensorial approach to hypercomplex-valued neural networks (2026)

**Venue:** Information Sciences.  
**Read:** https://doi.org/10.1016/j.ins.2026.123796  
**DOI:** `10.1016/j.ins.2026.123796`  
**What it does:** Algebra multiplication is encoded as a rank-three tensor; gives a general formulation for hypercomplex dense and convolutional layers.  
**Why it matters:** Useful mathematical/implementation background; no need to implement variable-algebra experiments for this project.

**Additional resources:** [preprint](https://arxiv.org/abs/2407.00449)

### `Vecchi2020` — Compressing deep-quaternion neural networks with targeted regularisation (2020)

**Venue:** CAAI Transactions on Intelligence Technology.  
**Read:** https://doi.org/10.1049/trit.2020.0020  
**DOI:** `10.1049/trit.2020.0020`  
**What it does:** Quaternion-specific regularizers drive whole weights or quaternion neurons toward zero and permit structural reduction.  
**Why it matters:** Direct QNN compression prior art; appropriate weight/group baseline.

### `Singh2026` — Compressing Quaternion Convolutional Neural Networks for Audio Classification (2026)

**Venue:** IEEE Transactions on Audio, Speech and Language Processing.  
**Read:** https://doi.org/10.1109/TASLPRO.2026.3677690  
**DOI:** `10.1109/TASLPRO.2026.3677690`  
**What it does:** Investigates pruning and knowledge distillation of quaternion CNNs for audio tasks, with filter-based compression and efficiency comparisons.  
**Why it matters:** Closest contemporary QNN compression comparison; do not claim filter pruning is novel.

**Additional resources:** [preprint](https://arxiv.org/abs/2510.21388)

### `Ayinde2019` — Redundant feature pruning for accelerated inference in deep neural networks (2019)

**Venue:** Neural Networks.  
**Read:** https://doi.org/10.1016/j.neunet.2019.04.021  
**DOI:** `10.1016/j.neunet.2019.04.021`  
**What it does:** Detects redundant CNN features/filters with cosine distance in feature space and removes redundant filters for efficient inference.  
**Why it matters:** Central similarity-based structural pruning baseline; protect against rediscovering this concept.

### `Kim2020` — Neuron Merging: Compensating for Pruned Neurons (2020)

**Venue:** Advances in Neural Information Processing Systems (NeurIPS).  
**Read:** https://proceedings.neurips.cc/paper/2020/hash/0678ca2eae02d542cc931e81b74de122-Abstract.html  
**What it does:** Neuron/filter merging uses a decomposition and compensation mechanism to preserve post-pruning feature maps, including a data-free strategy.  
**Why it matters:** Key prior art for optional neuron-merging experiments; its method is NOT primarily an activation correlation score.

**Additional resources:** [code](https://github.com/friendshipkim/neuron-merging)

### `Ebid2025` — Correlation-based pruning algorithm with weight compensation for feedforward neural networks (2025)

**Venue:** Neural Computing and Applications.  
**Read:** https://doi.org/10.1007/s00521-024-10932-6  
**DOI:** `10.1007/s00521-024-10932-6`  
**What it does:** Identifies highly correlated neuron activations and removes units while compensating their contributions with weight updates.  
**Why it matters:** Critical prior art for activation-correlation-driven compression and compensation.

### `Maroudis2026` — Weight Importance Is Not Enough: Neural Expressiveness for Model Compression (2026)

**Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence.  
**Read:** https://doi.org/10.1109/TPAMI.2026.3734995  
**DOI:** `10.1109/TPAMI.2026.3734995`  
**What it does:** Introduces Neural Expressiveness (NEXP), an activation-overlap-based compression criterion that contrasts with conventional weight-importance measures.  
**Why it matters:** Very close to broad activation-redundancy motivation; must compare its conceptual contribution before claiming novelty.

## Direct methods and important theoretical context

### `GanguliChong2024` — Activation-Based Pruning of Neural Networks (2024)

**Venue:** Algorithms.  
**Read:** https://doi.org/10.3390/a17010048  
**DOI:** `10.3390/a17010048`  
**What it does:** Uses frequency of activation to prune fully connected networks and relates pruning to rank reduction.  
**Why it matters:** An activation-based baseline, but distinct from pairwise redundancy; useful for terminology and methodology.

**Additional resources:** [code](https://github.com/tusharganguli/PrunedNetwork)

### `HuangPao2025` — Interpretable deep model pruning (2025)

**Venue:** Neurocomputing.  
**Read:** https://doi.org/10.1016/j.neucom.2025.130485  
**DOI:** `10.1016/j.neucom.2025.130485`  
**What it does:** Uses mutual information and total correlation to estimate feature redundancy in learned representation layers for pruning.  
**Why it matters:** A more expressive redundancy measure beyond simple cosine/correlation; relevant to RQ2 if simpler baselines fail.

### `Kobayashi2010` — Exceptional reducibility of complex-valued neural networks (2010)

**Venue:** IEEE Transactions on Neural Networks.  
**Read:** https://doi.org/10.1109/TNN.2010.2048040  
**DOI:** `10.1109/TNN.2010.2048040`  
**What it does:** Establishes a form of exact reducibility specific to certain complex-valued neural networks and discusses neuron minimization.  
**Why it matters:** Theoretical prior art: do not claim hypercomplex-specific reducibility was never studied.

### `IqbalMishra2023` — Neural Networks at a Fraction with Pruned Quaternions (2023)

**Venue:** Proceedings of the 6th Joint International Conference on Data Science and Management of Data (CODS-COMAD).  
**Read:** https://arxiv.org/abs/2308.06780  
**What it does:** Studies pruning and lottery-ticket-style sparsity in real vs quaternion architectures, including severe sparsity in image models.  
**Why it matters:** Shows QNN pruning itself is established; relevant results and experimental baselines.

### `Chaudhary2024` — Efficient CNNs with Quaternion Transformations and Pruning for Audio Tagging (2024)

**Venue:** Proceedings of Interspeech 2024.  
**Read:** https://doi.org/10.21437/Interspeech.2024-1331  
**DOI:** `10.21437/Interspeech.2024-1331`  
**What it does:** Investigates quaternion CNN transformation and pruning in audio tagging; physically structured filter removal.  
**Why it matters:** Direct structured-QNN pruning prior art and predecessor to Singh2026.

**Additional resources:** [resource](https://www.isca-archive.org/interspeech_2024/chaudhary24_interspeech.pdf)

## Wider supporting literature

### `Mukhopadhyay2024` — Large Neural Networks at a Fraction (2024)

**Venue:** Proceedings of the 5th Northern Lights Deep Learning Conference (NLDL).  
**Read:** https://proceedings.mlr.press/v233/mukhopadhyay24a.html  
**What it does:** Extends quaternion-vs-real pruning experiments to larger ResNets and extreme sparsity.  
**Why it matters:** Establishes breadth of preexisting QNN compression comparisons; secondary background.

### `Grassucci2024` — PHNNs: Lightweight Neural Networks via Parameterized Hypercomplex Convolutions (2024)

**Venue:** IEEE Transactions on Neural Networks and Learning Systems.  
**Read:** https://doi.org/10.1109/TNNLS.2022.3226772  
**DOI:** `10.1109/TNNLS.2022.3226772`  
**What it does:** Parameterized hypercomplex convolution shares parameters while learning multiplication structure, often yielding lightweight models.  
**Why it matters:** Background on hypercomplex parameter sharing/controls. Not a reason to expand the present study into learned algebras.

**Additional resources:** [code](https://github.com/eleGAN23/HyperNets)

### `ZhangEtAl2021` — Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters (2021)

**Venue:** International Conference on Learning Representations (ICLR).  
**Read:** https://arxiv.org/abs/2102.08597  
**What it does:** Introduces parameterized hypercomplex multiplication, learning multiplication rules rather than fixing Hamilton product.  
**Why it matters:** Context only; do not recreate the previous Hamilton-vs-learned-algebra research tangent.

### `Nitta2002` — Redundancy of the parameters of the complex-valued neural network (2002)

**Venue:** Neurocomputing.  
**Read:** https://doi.org/10.1016/S0925-2312(02)00669-0  
**DOI:** `10.1016/S0925-2312(02)00669-0`  
**What it does:** Analyzes parameter redundancies and transformations in complex-valued neural networks.  
**Why it matters:** Historical algebraic redundancy context, conceptually different from post-training activation similarity.

### `DongHuang2019` — A Structural Optimization Algorithm for Complex-Valued Neural Networks (2019)

**Venue:** 2019 IEEE Symposium Series on Computational Intelligence (SSCI).  
**Read:** https://doi.org/10.1109/SSCI44817.2019.9003040  
**DOI:** `10.1109/SSCI44817.2019.9003040`  
**What it does:** Applies group-lasso regularization to remove superfluous complex-valued network neurons.  
**Why it matters:** Alternative structured complex-domain compression precedent.

### `SunEtAl2024` — A Simple and Effective Pruning Approach for Large Language Models (2024)

**Venue:** International Conference on Learning Representations (ICLR).  
**Read:** https://proceedings.iclr.cc/paper_files/paper/2024/hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html  
**What it does:** Wanda multiplies weight magnitude by input-activation norms to score LLM weights for one-shot pruning.  
**Why it matters:** Useful conceptual activation+weight baseline; distinct from structural neuron redundancy and QNN.

**Additional resources:** [code](https://github.com/locuslab/wanda)

### `KyciaNiemczynowicz2025` — 4D hypercomplex-valued neural network in multivariate time series forecasting (2025)

**Venue:** Scientific Reports.  
**Read:** https://doi.org/10.1038/s41598-025-08957-5  
**DOI:** `10.1038/s41598-025-08957-5`  
**What it does:** Evaluates multiple four-dimensional hypercomplex architectures in multivariate time-series forecasting.  
**Why it matters:** Optional examples of training on 4D data, not a compression method.

## Questions for an agent extending the literature

1. Are there newer or overlooked **activation redundancy** papers explicitly studying deep quaternion/complex networks? Search beyond titles: "feature similarity", "channel correlation", "neuron merging", "network reducibility", "quaternion pruning", "hypercomplex pruning".
2. Does NEXP (`Maroudis2026`) already cover the proposed grouped-activation score in a model-agnostic way? Read its full methodology before proposing a novelty claim.
3. Are there implementations of structured QNN pruning with released code suitable as baselines? Verify availability and licenses rather than guessing.
4. Do published QNN comparisons report truly reduced inference models, or merely masked weights? Distinguish both clearly.
5. For any new paper: confirm authors, publication year, published venue, DOI, abstract and link from an authoritative source; add a BibTeX entry and a short annotation, marking unpublished or unverified works.

## BibTeX usage

In LaTeX: `\bibliography{references}` (BibTeX) or `\addbibresource{references.bib}` (biblatex). Use citation keys in this file, e.g. `\cite{Vecchi2020,Singh2026,Maroudis2026}`.

## Dataset used in the small RGB pilot

### `Krizhevsky2009` — Learning Multiple Layers of Features from Tiny Images (2009)

**Type:** University of Toronto technical report, linked as the recommended dataset citation.
**Read:** https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
**Dataset and verified citation metadata:** https://www.cs.toronto.edu/~kriz/cifar.html
**Use here:** CIFAR-10 supplies 32×32 RGB images for a small dense-network pilot.
The official page specifies 50,000 training and 10,000 test images, ten classes.
This dataset reference does not support a claim about quaternion networks.
The dataset page and installed Keras loader were inspected; the report PDF timed out
and its detailed methodology has not been reviewed in this session.

## Version-pinned software formula used in the BN follow-up

### `PyTorchFusion260` — Conv+BatchNorm folding utilities, PyTorch v2.6.0

**Read:** https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/utils/fusion.py
**Use here:** `fuse_conv_bn_weights` supplies the existing evaluation-mode
folding reference: scale each output weight row by gamma/sqrt(running_var+eps)
and account for running mean/bias in the folded offset. The pinned upstream
source and installed torch 2.6.0+cu124 function were inspected. Tests compare
the grouped score to explicit expanded-row norms and verify forward equality.
The experiment scores weights only, deliberately excluding the folded offset;
it does not reproduce a complete pruning method or claim a new formula.

## Learned-grouping feasibility follow-up

### `Wang2019FLGC` — Fully Learnable Group Convolution for Acceleration of Deep Neural Networks (2019)

**Venue:** CVPR 2019 (confirmed by the primary arXiv record).
**Read:** https://arxiv.org/abs/1904.00346
**Relevance:** Learns channel/filter group membership using selection variables
within training. Method passages in sections 2.1-2.2 were inspected. This is
prior art for the user's learned-membership idea, not a direct implementation
of balanced quaternion groups of four. No reproduction or speed claim is made.

### `Mena2018` — Learning Latent Permutations with Gumbel-Sinkhorn Networks (2018)

**Venue:** ICLR 2018 (confirmed by the primary record and paper header).
**Read:** https://arxiv.org/abs/1802.08665
**Relevance:** Existing mechanism for differentiable permutation learning and
discrete matching. Relevant relaxation/hard-evaluation passages were inspected.
This supplies methodological context for a possible internal channel router;
it does not demonstrate better quaternion grouping or compression here.

The bounded design, accounting limits and gates are recorded in
`results/learned_grouping_pilot/PROTOCOL.md`. Neither reference establishes novelty of
the proposed application; no new training result is implied.

## Closest learned-shuffle/compression work found in the novelty audit

### `Zhang2021StructuredShuffle` — Structured Sparsification with Joint Optimization of Group Convolution and Channel Shuffle (2021)

**Venue:** UAI 2021, PMLR 161:440-450 (verified in official proceedings).
**Read:** https://proceedings.mlr.press/v161/zhang21b.html
**Relevance:** Jointly addresses structured compression and learned channel
permutations. Selected primary method passages describe a weight-norm
objective, alternating permutation optimization and structured regularization.
This is close prior art against claiming that learned shuffle plus compression
is new. It is not a claimed implementation or reproduced result here.

### `Gong2023DynamicShuffle` — Dynamic Shuffle: An Efficient Channel Mixture Method (2023 author preprint)

**Read:** https://arxiv.org/abs/2310.02776v1
**Relevance:** Input-dependent permutations and static/dynamic channel mixing
extend beyond predefined shuffles. Primary abstract/method passages were
inspected. Our proposed global fixed export differs from input-conditioned
routing, but that distinction alone does not establish novelty. This entry
does not claim to verify later publication status or reproduce reported gains.

The preliminary assessment, limitations and required evidence are recorded in
`docs/LEARNED_GROUPING_NOVELTY_AUDIT.md`. No exact-match absence or first-in-
literature claim is supported by this limited search.

## Hard-forward grouping follow-up

### `Bengio2013STE` — Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation (2013 preprint)

**Read:** https://arxiv.org/abs/1308.3432
**Relevance:** The primary abstract describes the straight-through estimator
as a heuristic gradient-copy mechanism. Hard-forward differentiable training
is established prior art. A surrogate gradient is not the true derivative
of a piecewise-constant assignment or an unbiased estimator thereof.

### `Wang2023PSTE` — Compacting Binary Neural Networks by Sparse Kernel Selection (CVPR 2023)

**Read:** https://arxiv.org/abs/2303.14470
**Relevance:** The primary abstract and CVPR identification were inspected.
It explicitly combines Gumbel-Sinkhorn selection with a permutation
straight-through estimator for binary kernel codebooks. This is direct
precedent against a novelty claim for straight-through permutation learning.
The present deterministic quaternion-channel application is not a faithful
reproduction of that paper's full BNN method or reported gains.

## Close replacement/compensation prior art, reviewed 2026-11-04

### `Berisha2025VBP` — Variance-Based Pruning for Accelerating and Compressing Trained Networks (ICCV 2025)

**Read:** https://arxiv.org/abs/2507.12988
**Relevance:** Activation statistics guide structured pruning, while removed
units' means are folded into the next affine layer's bias. Primary abstract,
publication identification and a primary CVF PDF method excerpt were
inspected. Our mean-only control is therefore established background, not
a discovered new compression mechanism. The complete method was not reproduced.

### `Asiaee2026CMR` — Causal Mechanism Reduction: Mechanism Replacement for Neural Network Pruning and Abstraction (2026 preprint, v2)

**Read:** https://arxiv.org/abs/2602.24266v2
**Relevance:** Unifies constant/affine replacement, folding, logit distortion
and risk; includes a margin-based fidelity certificate and invariance tests.
The primary abstract and selected theory/folding passages were inspected.
Thus logit reconstruction versus decision preservation is already studied;
that observation alone is not a new contribution. Preprint status only;
no later venue or reproduced performance claim is made.

A possible next question concerns which replacements can be absorbed into
an INTERMEDIATE constrained quaternion consumer while retaining its real
parameter budget. The existing final GAP/real-linear-head experiment does
not test that restriction. This is a research candidate, not an established
literature gap or a new theorem/method claim. A focused full-text review is
still required before positioning quaternion-specific admissibility results.

## Focused full-text follow-up: folding and restrictions (2026-11-04)

`Kim2020`: sections 3.1-3.3, Theorem 1, Corollary 1.1 and the BN discussion
were inspected in the [primary PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/0678ca2eae02d542cc931e81b74de122-Paper.pdf).
The paper already folds compensation into the next convolution and studies
feature-map preservation without fine-tuning. Its ReLU-commutation condition
requires a nonnegative map with at most one positive entry per relevant
column. Our post-block activation replacement does not commute its general
4x4 map through ReLU. This difference defines scope, not priority.

`Asiaee2026CMR`: section 3.2, Theorem 1, Appendix A.5 (affine WLS/folding)
and Appendix B.1 were inspected in the [versioned full text](https://arxiv.org/html/2602.24266v2).
Weighted affine fitting and exact compilation into an ordinary affine
consumer are explicit prior art. The reviewed derivation updates arbitrary
consumer columns; retaining signed/tied Hamilton blocks is an additional
constraint in our implementation. This is a reading of the inspected
derivation, not proof that the literature lacks constrained variants.
The paper's cross-unit coupling discussion also cautions against treating
our pairwise residual-times-kernel-energy ranking as exact joint convolution
or task-loss minimization. No published method was reproduced by this review.

## Multi-source linear compensation (reviewed 2026-11-04)

### `GRAIL2026` — GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks (CPAL 2026)

**Published record:** https://proceedings.mlr.press/v328/tang26a.html (PMLR 328:881–895).
**Inspected method version:** https://arxiv.org/html/2602.23795v2
**Relevance:** After structured pruning or folding, fits a closed-form ridge
map from ALL kept consumer-input channels to the removed ones on ~128
unlabeled calibration images (lambda = alpha * mean Gram diagonal, alpha in
[1e-4, 5e-3]) and folds it into the consumer (W' = W B). No labels, no
fine-tuning. Experiments: ResNet-18/CIFAR-10, ViT-B/32, CLIP, LLaMA-2-7B.
The inspected v2 explicitly considers head-structured attention and GQA in
section 3.2; no Hamilton-tied quaternion consumer-kernel fit was located.
Unconstrained multi-source ridge folding is therefore prior
art; `configs/multisource_compensation.json` studies only the right-Hamilton
restriction of that fit. This is a reading of the inspected text, not proof
that no constrained variant exists.

**Correction, 2026-11-05:** publication is now confirmed by the official proceedings;
the earlier unconfirmed-preprint description is obsolete. In v2, section 4.1 and
Figure 2 already compare compensation with BN REPAIR and fine-tuning, and also
combine compensation with REPAIR. Consequently neither comparison with BN-only
repair nor a recovery-budget plot is itself a new contribution. The published
PDF returned an unsupported content type; metadata and the inspected preprint
are distinguished rather than assuming identical versions. No GRAIL reproduction
or head-to-head performance comparison is claimed.

## Paper A decision review (2026-11-05)

### Search scope and evidence limits

Focused searches combined quaternion/hypercomplex, pruning/compression, neuron
merging, compensation/reconstruction and BN recalibration; primary proceedings,
author preprints and publisher records were preferred. This is a targeted review,
not a systematic or exhaustive priority search. Existing full-text notes above
remain applicable. Additional inspected passages: Singh2026 preprint sections
III–IV, GRAIL2026 v2 section 4.1/Figure 2 and Li2020EagleEye adaptive-BN section.
Singh's journal record and older preprint are separate versions; do not mix their
headline numerical results.

### Established work versus our evidence

| Prior work | What is already established | Consequence for Paper A |
| --- | --- | --- |
| Vecchi2020 | Training-time regularization of quaternion weights and entire neurons using BN scales | No first QNN compression, grouped pruning or BN-aware importance claim |
| Singh2026 | Quaternion filter pruning using L1, geometric-median and operator-norm criteria, followed by fine-tuning; distillation comparison in audio models | Our image CNNs are not a reproduction or a stronger benchmark result; structured QNN pruning is established |
| Kim2020; Asiaee2026CMR | Replacement/merging and absorption into an affine consumer; explicit nonlinearity and folding conditions | No new compensation or exact-folding principle; document the Hamilton multiplication convention and approximation error |
| GRAIL2026 | Unlabeled multi-source ridge reconstruction folded into consumer weights; comparison with BN REPAIR and fine-tuning | Restricting the consumer family and measuring its cost is the possible empirical contribution, not ridge reconstruction |
| Li2020EagleEye | Refit BN statistics on training examples without gradients to evaluate pruned networks | BN-only repair is a required control, not a new method |
| Liu2019RethinkingPruning | Compact scratch training can match or beat inherited pruned models | Keep our unfavorable scratch results and unequal training budgets explicit |
| Valle2024VNets; Ding2017CirCNN | Hypercomplex layers as constrained real layers; circulant weight sharing | Similar circulant behavior argues against exclusivity, not for equivalence of all structures |

**Li2020EagleEye:** https://arxiv.org/abs/2007.02491 and
https://arxiv.org/pdf/2007.02491 (adaptive-BN discussion, printed pp. 6–8).
**Shomron2020BNRecalibration:** https://arxiv.org/abs/2010.05625;
the abstract concerns altered arithmetic in NB-SMT, not pruning specifically.
**Liu2019RethinkingPruning:** https://arxiv.org/abs/1810.05270v2 (ICLR 2019).

### Decision and bounded remaining work

GO for an empirical study of an existing QNN family's structure-preserving
compression, operating regimes and failure cases. NO-GO for a claim of a new
compensation algorithm, uniquely quaternion benefit, or first BN repair study.
The inspected literature does not establish that our precise experiment has
already been done, and this review does not establish novelty by absence.

Complete the cheap BN-only control on frozen compact models, extract recovery
histories already recorded, and freeze the final test comparisons. Do not launch
new architecture, algebra or SSL sweeps. A useful result must survive comparison
with same-victim pruning and BN recalibration; report its disappearance if it
does not. The contribution remains incremental and acceptance is uncertain.

### Training without BN: user-requested feasibility check (2026-11-05)

`Brock2021NFNets`, [official ICML/PMLR record](https://proceedings.mlr.press/v139/brock21a.html),
establishes that high-performing image classifiers can be trained without BN
using a dedicated normalization-free recipe. It does not establish accuracy for
our shallow quaternion CNN after simply deleting its six BNs. Training without
normalization is therefore neither impossible nor a novelty claim here.

The user's question motivates a bounded ablation: one CIFAR QNN, seed 2026,
50 epochs, original splits/settings/remaining initialization, BN replaced by
Identity before training. No new test inference, new compression sweep, NFNet
recipe or hyperparameter search. Removing BN removes 896 learned affine
parameters; report this change. A failure under the fixed recipe cannot rule
out successful no-BN training under other settings. See configs/no_bn_pilot.json.

### Consumer-output fitting guardrail (2026-11-06)

`He2017ChannelPruning`, [official ICCV/CVF record](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html),
already alternates LASSO channel selection with least-squares reconstruction of
the following layer's outputs. Section 3.1 of the [author preprint](https://arxiv.org/pdf/1707.06168)
was inspected. Directly refitting surviving consumer weights is therefore
established background. Any future restriction of that objective to Hamilton
kernels must be positioned as a constrained case, not as inventing output
reconstruction. No exact quaternion variant or priority claim was established
by this limited follow-up search.

The current all-site study (`configs/structure_cost_study.json`) tests Q maps,
unrestricted maps compiled into dense consumers, and a lossless Q-to-dense
initialization control. It does not optimize all compact Q kernels against
consumer targets. Its measured gaps must not be called unavoidable architecture
costs. Releasing weight constraints, changing the compensation fit and keeping
equal real-parameter budgets are distinct experimental questions.

## Novelty assessment after the all-site comparison (2026-11-06)

Current evidence is an incremental empirical study; completing controls does
not establish sufficient novelty or acceptance. Main reviewer risk: known
reconstruction/folding applied to quaternion CNNs; unequal dense/Q budgets;
small architectures and a limited case for practical compression over scratch.
No numerical acceptance probability or guaranteed conference threshold is asserted.

`IbnAhad2026HypercomplexQuantization`: [author record](https://arxiv.org/abs/2512.08524),
[inspected v1 full text](https://arxiv.org/html/2512.08524v1),
[official CVF paper URL](https://openaccess.thecvf.com/content/WACV2026/papers/Ibn_Ahad_Beyond_Real_Weights_Hypercomplex_Representations_for_Stable_Quantization_WACV_2026_paper.pdf).
Sections 3.1–3.2 and Algorithm 1 describe dense-to-PHM compression with selective
basis capacity, a parameter-budget formulation, residual interpolation and
reconstruction/distillation in multimodal language models. Published-PDF access
failed; method claims here refer to v1. A generic proposal to add residual paths
or allocate hypercomplex capacity selectively is therefore not a safe novelty
claim. No new-algebra or MLLM experiment is proposed for this project.

This bounded diagnostic was subsequently authorized and completed (one CIFAR
seed/fraction; results/direct_consumer_pilot/REPORT.md): compare map-then-fold
compensation with direct reconstruction of consumer outputs under EXACTLY the
same compact quaternion architecture, masks and real-parameter budget. Use a
single CIFAR source/fraction and split fit/check data before any extension.
Unregularized local least-squares error cannot increase at its global optimum
because the existing folded kernel is feasible. Held-out error, classification,
regularized estimates and multi-site compounding have no such guarantee.

The potential research contribution would be distinguishing compensation-
parameterization loss from architecture-family loss, with a measured remedy
at fixed size. Neither a new theorem nor a new least-squares algorithm is
assumed: consumer-output reconstruction is established by He2017ChannelPruning.
If the pilot provides no new explanation or fixed-budget benefit, more seeds,
activations or datasets alone do not resolve novelty. Any selective non-Hamilton
residual extension needs a separate comparison to the prior art above.

## Closer reading and code check: Beyond Real Weights (2026-11-06)

**Paper evidence:** IbnAhad2026HypercomplexQuantization, arXiv v1 sections 3–4,
Algorithm 1 and ablations inspected. Its fixed 2x2 bases implement complex-like
B=2 or a larger B=3 subspace, not our 4x4 Hamilton kernels. Capacity follows
a top-K rule, rather than learned discrete allocation. Final deployment drops
the dense transition branch. Operator-output reconstruction is already present.

**Code evidence:** [pinned public sample](https://github.com/milab-nsu/PHM/blob/6982112ef33f0c81cc81aab900bfc63340be6565/Sample_Code.ipynb).
The repository contains README and a sample notebook; its default dataset is
Flickr8k, with placeholder model paths and custom METEOR/CIDEr-lite functions.
It is not a complete reproduction package for every reported benchmark.

Our independent CPU check extracts only ComplexPHMLinear. Projection uses an
interleaved reshape while forward uses blocked halves. Exactly representable
6x8 examples give Frobenius residuals 37.99 (B=2), 52.37 (B=3), versus <1.1e-14
for the matching block projection. The sample's two GEMMs per basis imply
matmul-FLOP ratios B/2 versus dense, excluding mixing overhead; this is not
a latency benchmark. Reproducer: scripts/check_phm_sample.py; evidence and
pinned notebook: .cache/literature/phm_sample_review.json.

These checks concern that sample snapshot and do not reproduce the paper's
trained models or invalidate its full results. They do not remove its
prior-art status. Our pilot asks a different operational question: refitting
an already quaternion consumer after channel removal, at unchanged Q size.

## Valle group: vector-/hypercomplex-valued networks (reviewed 2026-11-04; citation plan updated 2026-11-06)

Marcos Eduardo Valle (UNICAMP) co-organized the IJCNN 2025 special session on
complex- and hypercomplex-valued neural networks. The user later supplied his
invitation for the proposed 2027 session. None of the three papers listed in
the original 2026-11-04 review discusses pruning, removing/merging units, or
folding/compensating weights. The subsequently reviewed ELM paper below is
important prior art for the regression principle.

**User explicitly requested retaining Marcos's papers for appropriate article
citations (2026-11-06). All four keys below are in references.bib.**

| Canonical key | Intended manuscript location | Precise purpose |
| --- | --- | --- |
| `VieiraValle2022ELM` | Related work AND method, immediately after defining the real design matrix / hypercomplex least-squares problem | Credit the existing hypercomplex LS/real-embedding solution. Cite before the normal equations; do not present this solver as invented here. |
| `Valle2024VNets` | Introduction's structured-weight-sharing viewpoint; method's interpretation of Hamilton blocks | Explain hypercomplex layers as constrained vector-valued/real layers. This motivates asking about imposed structure rather than intrinsic quaternion superiority. |
| `VieiraNetoValle2025VEfficientNets` | Related-work paragraph distinguishing compact architectures by design from post-training channel compression, if that distinction appears | A relevant architectural alternative, not a reproduced pruning baseline or evidence for our compensation. |
| `VieiraNetoValle2026Ghost` | Discussion of transferring trained real models into hypercomplex representations, only if that topic remains in the six-page paper | Distinguish real-to-hypercomplex transfer from compressing an existing QNN. Do not claim that ghost features establish measured redundancy or use a courtesy citation without a relevant claim. |

For a short article, the first two are central; the other two remain available
and should be cited when their specific topic is actually discussed. Preserve
the verified source versions in references.bib rather than inventing final DOIs.

- `Valle2024VNets`: a hypercomplex (V-net) layer is a real layer with weights
  restricted so as to model intercorrelation between feature channels; the
  algebra reduces free parameters. Which real channels form a group is not
  derived from data. Directly relevant framing for our "structured weight
  sharing" interpretation (docs/STRUCTURE_HYPOTHESIS.md) and for the
  closure argument behind exact Hamilton folding.
- `VieiraNetoValle2025VEfficientNets` (IJCNN 2025): parameter reduction BY
  DESIGN (vector-valued EfficientNet, ~1/d parameters), not post-training
  compression. Channel-to-component assignment follows the input encoding.
- `VieiraNetoValle2026Ghost`: embeds a trained real layer as the real part of
  a hypercomplex layer; imaginary outputs ("ghost features") are described as
  "redundant yet rich"; no redundancy measurement or compression. Lighter
  models are mentioned only as future work.

For the three architecture/transfer papers, no matching removed-unit
compensation procedure was located; cite the V-net view as conceptual basis.
The ELM paper DOES overlap the underlying LS mathematics and must be credited
there. Neither observation establishes absence of an exact pruning predecessor
elsewhere. See the priority audit above for the current, narrower assessment.

## Second dataset: EuroSAT four-band (added 2026-11-04)

### `Helber2019EuroSAT`

**Data:** https://zenodo.org/records/7711810 (EuroSAT_MS.zip, MD5 091174add3c8e680a49244acf185b9f0 verified).
27,000 Sentinel-2 tiles, 64x64, 13 bands, 10 land-cover classes, MIT license
(Copernicus terms apply). We keep B08/B04/B03/B02 as q = NIR + R i + G j + B k,
which removes the zero real part used for CIFAR RGB. NIR in the real part is
an encoding choice. Stratified split (seed 20261104): 19,600 train, 1,000
validation, 1,000 calibration, 5,400 held-out test.

Journal metadata verified on 2026-11-04 against the
[IEEE publication record](https://ieeexplore.ieee.org/document/8736785):
IEEE JSTARS 12(7), July 2019, pages 2217-2226, DOI
`10.1109/JSTARS.2019.2918242` (online publication 14 June 2019).
The [authors' repository](https://github.com/phelber/EuroSAT) independently
confirms authors, journal and 2019 year; Zenodo confirms the dataset archive
and linked journal DOI. This verifies citation metadata, not a reproduction
of the paper's reported benchmark.

## Activation sensitivity: candidate follow-up (2026-11-04)

### `PoppelbaumSchwung2024Activations`

**Read:** https://arxiv.org/abs/2406.16481 (v1, 24 June 2024).
**Title:** Improving Quaternion Neural Networks with Quaternionic Activation Functions.
**Authors:** Johannes Pöppelbaum and Andreas Schwung.
The primary abstract, sections 2.1-2.3 and section 4.3.1 were inspected. The paper compares
split activations with coupled magnitude/phase alternatives on CIFAR-10 and
SVHN. Thus changing the quaternion activation or comparing split/coupled
forms has direct prior art. We have not reproduced its methods or results;
later publication status was not checked.

Section 4.3.1, Eq.21 defines MagnitudeTanh as `tanh(norm(q))*q/norm(q)`.
This is the parameter-free coupled alternative selected for our queued
CIFAR sensitivity pilot (configs/activation_sensitivity.json), alongside
split LeakyReLU. The zero-input extension is continuous, and our implementation
uses a small-norm Taylor branch for numerical stability. The pilot has no
activation results yet. BN and pooling remain scalar-channel-wise, so the
coupled nonlinearity alone does not establish full-model equivariance.

## Low-label repair background (2026-11-05)

### `Sohn2020FixMatch`

**Primary source:** https://arxiv.org/abs/2001.07685 (NeurIPS 2020; metadata and abstract inspected).
Confidence-filtered pseudo-labels from weakly augmented images supervise a stronger view.
This is direct prior art for pseudo-label consistency; no novelty is claimed for adding
online hard pseudo-labels to our repair loop. Our bounded pilot uses an unaugmented
prediction view and crop/flip training view, with fixed BN statistics, and does not
reproduce FixMatch's stronger augmentation pipeline or the doctoral student's unknown
implementation. The teacher-pseudo control uses a frozen full model instead of the
current student; it has an additional teacher resource and inference cost.

The 0/5/20 label budget concerns post-compression repair ONLY. Full supervision,
validation-based checkpoint selection and inherited normalization/BN statistics were
available when training the original model. This is not end-to-end few-shot learning.
Main low-label cases are 5/20 per class; zero is a boundary control. Same-distribution
CIFAR simulates withheld repair annotations, not an observed deployment/domain shift.

## Established structured-weight control (2026-11-05)

### `Ding2017CirCNN`

**Primary metadata:** https://arxiv.org/abs/1708.08917; related DOI 10.1145/3123939.3124552.
**Primary full text:** https://arxiv.org/html/1708.08917, sections 3.1/3.2 inspected.
The MICRO-50 paper studies block-circulant weight representations, including convolutional
layers, with FFT-based computation. It establishes that circulant weight sharing is known.
Our control uses 4x4 circulant blocks across component channels at each spatial kernel offset,
expanded for ordinary Conv2d. This is a deliberately small conventional structured-weight
control, not a reproduction of CirCNN's architecture, FFT implementation or measured speedups.

Four independent real coefficients specify each 4x4 circulant block, matching the degrees
of freedom of our Hamilton block. Widths, real parameter counts, input information and seeded
coefficient tensors can therefore be matched exactly. Both spaces are closed under products,
allowing structure-preserving linear compensation; closure is checked numerically and exact
folding is verified at each site. This does not make either restriction novel. Different full
baseline quality is a confound even after reporting own-baseline loss; one alternative cannot
establish a unique quaternion mechanism or a universal structure-only explanation. No random
algebra or learned multiplication rules are proposed.

## Actual shared low-label SSL pilot (2026-11-05)

### `Chen2023SoftMatch`

**Primary source:** https://arxiv.org/abs/2301.10921, v2; ICLR 2023 acceptance stated in
the primary metadata. Abstract/metadata inspected. Confidence-dependent truncated Gaussian
weighting and uniform alignment are prior art. The workspace sibling FewshotLearning uses
an epoch-lagged Gaussian weighting helper plus sharpened soft pseudo-targets, without the
full SoftMatch alignment recipe. Do not identify that implementation as full SoftMatch or
claim a novel SSL method.

The new configs/ssl_compression_pilot.json imports the sibling's exact view/loss helpers,
trains QNN/circulant models with only 50 training labels/class from initialization and then
compresses them. This differs from the older fully-supervised-parent repair pilot. Held-out
fit/check images are excluded from weight training (U=47,500, a disclosed adaptation),
and no official-test inference is planned. One seed is exploratory, not replicated evidence.
