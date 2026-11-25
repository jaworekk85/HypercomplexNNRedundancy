# Quaternion grouping, adaptation and empirical contributions

Reviewed2026-11-25. Canonical citation keys: [references.bib](../../references.bib).
This is a targeted review for paper positioning, not a systematic review or a
priority certificate. Hypotheses below are not measured findings. Numerical
protocols and the running study were not modified for this review.

## Main assessment

A defensible contribution can adapt an existing real-network idea to QNNs when
it identifies a consequential structural issue, solves or measures it, and
compares suitable controls. A new algebra or optimizer is not intrinsically
required. This assessment is an inference from the precedents below, not an
acceptance prediction. A substitution of layer names alone would provide much
less evidence than a controlled result about grouping and realizability.

The strongest current candidate question is:

> When must original quaternion channel groups survive structural compression,
> and how much observed damage comes from constrained producer realization or
> subsequent repair rather than from regrouping the retained information?

The prior reassembly pilot's failed continuation gate remains failed. Its
18/20 intact groups are not an unbiased test: the search reserved intact units
for completion, and the compiler reproduces intact producers exactly while
approximating mixed producers. The separately registered
[grouping study](../../results/grouping_mechanism_study/PROTOCOL.md) controls the
retained scalar set and separates exact-feature fitting, physical producer
realization, and equal repair. Outcomes are still to be interpreted from its
final report; nothing here establishes that quaternion structure is dispensable.

## Published examples of useful adaptation or analysis

| Key | Inspected evidence | Relevance and boundary |
|---|---|---|
| `Vecchi2020` | Published PDF, introduction and sections3/4; Crossref metadata/export | Extends real-network sparsity regularization using quaternion grouping and BN-related neuron sparsification. Explicitly compares component-wise and quaternion-aware regularization. A direct precedent for substantive adaptation; our post-training activation-grouping intervention is a different question. |
| `BillCox2024LossSurfaces` | Publisher full-text introduction/section3.3 and metadata | Adapts loss-surface visualization/curvature analysis to quaternion MLPs. Shows that a study explaining QNN behavior can constitute the core contribution. Their30 random surface projections are not30 independently trained model seeds; do not inherit a broad optimization claim. |
| `GaudetMaida2018DeepQuaternion` | Author abstract/version record; publisher-deposited citation | Published at IJCNN2018: quaternion convolutions, initialization and BN support deep-network experiments. Historical precedent for domain-specific architectural adaptation; it does not establish what reviewers will accept in2027. |
| `Tay2019QuaternionNLP` | ACL abstract, PDF title page, official BibTeX and registry | Quaternion attention/Transformer adaptations evaluated in NLP. Supports a concrete adaptation-plus-evaluation contribution. Their parameter-saving statements are architecture-dependent, not a guarantee for our entire ResNet or an exact grouping predecessor. |
| `Chaudhary2024` | ISCA published PDF/abstract and official citation | Combines quaternion CNN transformations with pruning for audio tagging. Direct compression prior art, so generic “we prune a quaternion CNN” is insufficient differentiation. Its reported architecture/application savings are not transferable numerical results for our models. |

Primary links: [Vecchi](https://doi.org/10.1049/trit.2020.0020),
[Bill/Cox](https://doi.org/10.1007/s00006-024-01313-2),
[Gaudet/Maida](https://doi.org/10.1109/IJCNN.2018.8489651),
[Tay](https://aclanthology.org/P19-1145/),
[Chaudhary](https://www.isca-archive.org/interspeech_2024/chaudhary24_interspeech.html).

## Close conceptual evidence: are components an indivisible object?

**`PoppelbaumSchwung2025TimeCompression`** is particularly useful. Inspected
author-preprintv2 sectionsVI-B3/VII compare component max-pooling, which may take
different quaternion components from different temporal positions, with
magnitude max-pooling, which selects an entire quaternion. The favorable pooling
choice depends on the evaluated learning setup. This makes the distinction
between treating components separately and preserving a whole quaternion an
existing experimental issue. It motivates testing rather than assuming the
importance of a grouping.

The boundary is essential: pooling occurs across temporal positions within a
channel, preserves component roles, and compresses input time series. Our
intervention changes hidden-channel group membership/possibly roles and compiles
smaller Hamilton producer/consumer layers. Neither the paper nor our current
pilot proves that arbitrary groups are interchangeable. Cite the final2025
journal record, while identifying the inspected preprint separately.
[Journal](https://doi.org/10.1016/j.neunet.2025.107465),
[inspected preprint](https://arxiv.org/html/2403.11722v2).

**`Comminiello2024InductiveBiases`** separates dimensionality, algebraic,
geometric and regularization biases; preprint sections3/4 were inspected.
This is a useful vocabulary for distinguishing a constraint on shared weights
from the semantic identity of a learned group. The framework does not imply
rotation equivariance for our scalar-BN/split-ReLU architecture, or predict the
outcome of a hidden-channel intervention.
[Journal](https://doi.org/10.1109/MSP.2024.3401622),
[preprint](https://arxiv.org/html/2405.07024v1).

**`ShahadatMaida2024CrossChannel`** examines cross-channel weight sharing and
fully hypercomplex architectures. Publisher and author-preprint abstracts were
inspected, not full methods. It provides close context for the hypothesis that
the sharing pattern matters; it cannot support a claim that particular learned
quartets do or do not matter. Its journal issue is2024, despite the2023 DOI
string/online/preprint dates.
[Journal](https://doi.org/10.1016/j.imavis.2023.104872).

**`Valle2024VNets`**, already in the bibliography, places real and hypercomplex
networks in a vector-valued framework. The earlier preprint-only citation has
been upgraded to the verified IEEE Signal Processing Magazine publication,
keeping the key and arXiv identifier. Abstract and previously inspected preprint
passages support discussion of intercorrelation/parameter sharing, not a theorem
that correlations identify removable channels.
[Journal](https://doi.org/10.1109/MSP.2024.3401621).

## Functional reconstruction and repair need careful interpretation

**`Bansal2021ModelStitching`** uses frozen networks and a learned connection to
investigate representations. The proceedings abstract was inspected. It
supports separating fixed-representation compatibility from full-network
retraining; it is not a matched-size quaternion compression baseline.
[Proceedings](https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html).

**`Smith2025FunctionalAlignment`** gives counterexamples to equating successful
functional alignment with informational similarity. The official ICML abstract
was inspected. This supports a limitation in our interpretation: recovery of
classification after repair does not establish identical representations or
that original grouping carried no useful structure. It does not invalidate
the practical utility of a recovered compact model.
[Proceedings](https://proceedings.mlr.press/v267/smith25a.html).

**`Jordan2023REPAIR`** studies activation rescaling after permutation-aligned
model interpolation; indexed accepted-paper title-page/intro and author arXiv
abstract were inspected. Distribution changes can matter after structural
operations. The method is different from our three-epoch supervised fine-tuning
function named `repair`; do not claim to implement REPAIR. This source motivates
possible future activation-statistics controls, not a new experiment added to
the current frozen study.
[Accepted-paper record](https://openreview.net/forum?id=gU5sJ6ZggcX),
[author preprint](https://arxiv.org/abs/2211.08403).

## Small effects and negative findings

**`Lakens2017Equivalence`** explains prospectively justified practical-equivalence
bounds and distinguishes evidence for a sufficiently small effect from failure
to detect a difference. Publisher abstract/export were inspected. We have not
registered or performed an equivalence test in the current study. Three source
seeds are an initial repeatability check; neither image count nor random-group
draw count automatically supplies independent training replications. A future
equivalence design needs a justified tolerance and power/uncertainty assessment.
[Publisher](https://doi.org/10.1177/1948550617697177).

## Candidate paper structure, conditional on final evidence

Working question/title: *Must Quaternion Channel Groups Survive Structural
Compression?* This is a proposed framing, not a finding or priority claim.

1. Exact retained scalar features: intact versus role-preserving/free groups;
   repeat predetermined assignments and keep the mask/parameter budget fixed.
2. Realizability: compare exact-feature fits with physically compiled producers
   and separately report local producer adaptation. Elementary Hamilton
   projection/ridge are implementation tools rather than new theorems.
3. Practical behavior: before/after equal repair across parent seeds; report
   actual compact counts, calibration/search compute and unfavorable outcomes.

A favorable result could support a useful adaptation procedure; a strong
grouping penalty could instead support a mechanistic reason to preserve groups.
A sufficiently precise small-effect result could support robustness, subject to
a prospectively appropriate design. Ambiguous results remain ambiguous.

Do not claim first quaternion pruning, first component-wise operation, first
learned grouping, a novel least-squares solver, or general irrelevance of
Hamilton structure. Discuss `Wang2019FLGC`, `Zhang2021StructuredShuffle`,
`He2017ChannelPruning`, `Frantar2022OBC`, `Singh2026` and the sources above.
Retain framework credit to `NiemczynowiczKycia2025`/`NiemczynowiczKycia2026`, while
describing the local tested PyTorch implementation accurately.

## Venue fit, checked2026-11-25

The official **Complex- and Hypercomplex-valued Neural Networks** session
description explicitly includes generalizing conventional real-valued
architectures and training algorithms to the hypercomplex setting. This is
evidence of topical fit, not an acceptance guarantee or exemption from review.
The public call lists model compression/pruning among conference topics,
six-page complete papers, and a regular-paper deadline of31January2027.
Recheck the current author instructions before submission; do not infer an
unpublished timezone or an extra reference-page allowance.
[Session](https://ijcnn.org/2027/authors/special-sessions),
[call](https://ijcnn.org/2027/authors/call-for-papers).

## Bibliography provenance and limits

Ten new canonical entries were added; `Valle2024VNets` was upgraded without a
key change. Existing `Vecchi2020`, `Chaudhary2024`, `Singh2026` were rechecked
against publisher-deposited Crossref metadata and downloaded exports.

Raw exports/registry responses and URLs/timestamps/SHA-256 are under
[exports/2026-11-25](exports/2026-11-25/sources.json). Canonical entries normalize
stable keys, LaTeX accents, page ranges and case-sensitive abbreviations.
Important export issues: Lakens contains HTML italic tags in its raw BibTeX,
its JSON title omits the subtitle; Bill/Cox raw BibTeX omits article19, which
publisher/registry metadata supply. Publication years distinguish the journal
from older preprints; a DOI's embedded year is not the publication year.

OpenReview BibTeX/API requests for REPAIR failed404/403; direct arXiv downloads
failed certificate validation. Its entry was manually normalized from the
primary metadata/title-page evidence above, not presented as a downloaded
publisher BibTeX. Crossref429 responses were retried; invalid Springer HTML
was rejected rather than saved as a citation. All attempts remain logged.

See [verification.json](exports/2026-11-25/verification.json) for field/structure
checks. This targeted audit is not a fresh verification of every older entry
in references.bib, nor a claim that every cited final journal PDF was read.
