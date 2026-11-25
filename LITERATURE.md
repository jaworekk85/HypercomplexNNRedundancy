# Literature: compact index

Citation keys and bibliographic metadata are canonical in [references.bib](references.bib). This index is a starting point, not a systematic review or proof of priority. Read the relevant primary sources before making detailed claims. The full previous annotated review and its targeted novelty audits are preserved in [docs/archive/LITERATURE_2026-11-07.md](docs/archive/LITERATURE_2026-11-07.md); its local relative links were written relative to the repository root.

## Grouping and substantive adaptation to QNNs: 2026-11-25 review

The [targeted review](docs/literature/GROUPING_AND_TRANSFER_2026-11-25.md)
reframes the contribution around what quaternion structure changes, rather
than requiring an entirely new optimizer/algebra. Published examples include
`Vecchi2020` (targeted quaternion regularization), `BillCox2024LossSurfaces`
(empirical loss-surface analysis), `GaudetMaida2018DeepQuaternion` (IJCNN2018),
and `Tay2019QuaternionNLP` (quaternion attention/Transformer adaptations).

Closest conceptual sources: `PoppelbaumSchwung2025TimeCompression` compares
component versus whole-quaternion magnitude pooling across temporal positions;
this is not hidden-channel reassembly. `Comminiello2024InductiveBiases`,
`ShahadatMaida2024CrossChannel` and the now journal-verified `Valle2024VNets`
provide context on sharing and inter-feature relations. Successful repair does
not establish informational equivalence: see `Bansal2021ModelStitching`,
`Smith2025FunctionalAlignment` and the distinct interpolation method
`Jordan2023REPAIR`. `Lakens2017Equivalence` supports planning small-effect tests,
not a claim that our current study has demonstrated equivalence.

Ten new entries and one publication-status upgrade have
[raw exports and verification records](docs/literature/exports/2026-11-25/sources.json).
OpenReview export failures and the manually verified REPAIR entry are disclosed.
The review records inspected sections/abstract-only limits and potential paper
uses. It does not establish firstness or guarantee acceptance. The official
[IJCNN2027 session description](https://ijcnn.org/2027/authors/special-sessions)
explicitly includes generalizing real-valued architectures/training algorithms
to hypercomplex networks; topical fit is separate from contribution quality.

## Hypercomplex frameworks and implementation provenance

- `NiemczynowiczKycia2025`: Agnieszka Niemczynowicz and Radosław A. Kycia, *KHNN: Hypercomplex-valued neural networks computations via Keras using TensorFlow and PyTorch*, SoftwareX 30 (2025), 102163, [DOI](https://doi.org/10.1016/j.softx.2025.102163). Include this work in the future paper's background/related work on hypercomplex software and in the implementation provenance discussion. The canonical BibTeX entry already exists in [references.bib](references.bib).
- KHNN/HypercomplexKeras was the initial implementation candidate; the tested HypercomplexKeras 1.0.2 configuration failed the Keras 3 parameter-tracking check, as documented in [Phase 0](results/phase0/FEASIBILITY.md). Current experiments use local PyTorch layers. Credit the framework while accurately describing the implementation used; the recorded compatibility failure concerns the tested versions and does not establish a general limitation of KHNN.

## Quaternion compression and reconstruction

- `Vecchi2020`, `IqbalMishra2023`, `Chaudhary2024`, `Singh2026`: existing QNN pruning and retraining; `Singh2026` also discusses distillation of pruned QNNs.
- `VieiraValle2022ELM`: hypercomplex least squares with real embeddings; relevant to the reconstruction solver, not a post-pruning CNN procedure. Cite Marcos Valle's work where the solver context is discussed.
- `He2017ChannelPruning`, `Luo2017ThiNet`, `Liang2021TwoStep`: established reconstruction or refitting after channel selection in real CNNs.
- `KammaWada2021REAP`, `GRAIL2026`: feature reconstruction and compensation in real networks.

## Learned structure and budget allocation

- `ZhangEtAl2021`, `Grassucci2024`: PHM/PHNN parameterizations and learned component matrices. Learned matrices do not automatically satisfy algebra axioms.
- `IbnAhad2026HypercomplexQuantization`: pretrained dense-to-PHM conversion and layerwise allocation; PHM is not restricted to training from scratch.
- `VanLoanPitsianis1993`: Kronecker approximation via rearrangement and SVD. The unweighted Frobenius result does not establish an optimum for activation-weighted reconstruction.
- `Liebenwein2021ALDS`, `Frantar2022SPDY`, `Cao2026Numerion`, `Sfikas2023SharedOperation`: relevant prior work on structure selection, allocation or adaptive hypercomplex representations. Consult the archived audit for the exact scope and limitations of each comparison.

## Claims and interpretation

For a possible next direction involving masks selected by the attainable
consumer reconstruction error, consider `He2017ChannelPruning`, `Frantar2022OBC`
and `Jeong2022QCQP`. Reconstruction-aware pruning, OBS compensation and coupled
resource-constrained channel selection have prior art. The 2026-11-25 OBC/QCQP
checks cover official metadata/abstracts, not full-method equivalence. A new
Hamilton/PHM implementation alone does not establish methodological novelty.
The earlier reconstruction comparisons fixed masks across fitting methods.
The completed [component-reassembly pilot](results/resnet_component_reassembly_pilot/REPORT.md)
now includes bounded joint selection using actual projected-producer features
and a Hamilton consumer ridge objective. Its single-seed free-role result
missed the registered continuation gate; it establishes neither a general
performance gain nor novelty relative to these methods and learned grouping
work (`Zhang2021StructuredShuffle`, `Wang2019FLGC`).

The separately authorized [grouping-mechanism study](results/grouping_mechanism_study/PROTOCOL.md)
holds retained scalar features fixed, compares role-preserving/free assignments
on exact features, separates physical producer projection/data-aware producer
fitting, then measures equal repair. Primary abstract pages for
`Zhang2021StructuredShuffle` and `Wang2019FLGC` were rechecked2026-11-25:
learned grouping and channel shuffle are established. This bounded check is not
a systematic search or evidence of firstness for the proposed diagnostic.
Neither the control design nor standard ridge/projection alone establishes a
novel method. The study has completed; its measured report must be interpreted
within its registered scope. The previous pilot gate stays failed.

Quaternion pruning, real-network activation pruning, neuron merging, next-layer reconstruction, hypercomplex least squares and learned PHM structures already exist. A targeted review did not locate an exact predecessor for the investigated combined post-pruning procedure; this does **not** establish firstness or sufficient novelty. The proposed contribution is conditional on careful controls and measured results.

The reported 62.4–90.6% local-objective improvement concerns releasing extra shared-map restrictions **within** quaternion consumers; it does not measure the cost of quaternion versus real architectures. Compare actual parameter budgets and separately report hypotheses, results and interpretation. For an earlier redundancy-focused reading map and detailed source-by-source qualifications, consult the archived review.
