# Literature: compact index

Citation keys and bibliographic metadata are canonical in [references.bib](references.bib). This index is a starting point, not a systematic review or proof of priority. Read the relevant primary sources before making detailed claims. The full previous annotated review and its targeted novelty audits are preserved in [docs/archive/LITERATURE_2026-11-07.md](docs/archive/LITERATURE_2026-11-07.md); its local relative links were written relative to the repository root.

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

Quaternion pruning, real-network activation pruning, neuron merging, next-layer reconstruction, hypercomplex least squares and learned PHM structures already exist. A targeted review did not locate an exact predecessor for the investigated combined post-pruning procedure; this does **not** establish firstness or sufficient novelty. The proposed contribution is conditional on careful controls and measured results.

The reported 62.4–90.6% local-objective improvement concerns releasing extra shared-map restrictions **within** quaternion consumers; it does not measure the cost of quaternion versus real architectures. Compare actual parameter budgets and separately report hypotheses, results and interpretation. For an earlier redundancy-focused reading map and detailed source-by-source qualifications, consult the archived review.
