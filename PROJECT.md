# Activation-Based Redundancy Analysis and Compression of Hypercomplex Neural Networks

**Research project brief for Codex / Claude Code**  
**Target venue:** IJCNN 2027 special session, *Complex- and Hypercomplex-valued Neural Networks*  
**Status:** research hypothesis and implementation plan — **no results claimed**  
**Prepared:** 2026-11-02  
**Bibliography:** [`LITERATURE.md`](LITERATURE.md), [`references.bib`](references.bib)

> **Agent instructions:** This document is the source of truth for the project scope. Read `LITERATURE.md` before suggesting a new method and use citations from `references.bib`. Do not assert novelty or performance without evidence. Start with a minimal working baseline and an exploratory analysis; avoid expanding the scope without an empirical reason.

## 1. Research motivation

Hypercomplex-valued networks (especially quaternion-valued networks, QNNs) use coupled multicomponent representations and structured weight sharing. Existing research investigates their parameter efficiency and pruning, mostly through weight-based or filter-level criteria [Vecchi2020; Singh2026; Chaudhary2024]. Separately, activation redundancy, feature similarity, neuron merging, and activation-dependent pruning are well studied in ordinary real-valued networks [Ayinde2019; Kim2020; Ebid2025; Maroudis2026].

**Research intersection:** Characterize *internal activation redundancy* in hypercomplex models and test whether this information helps with **structural compression**. The focus is on understanding representations, not designing a new hypercomplex architecture or inventing an algebra for its own sake.

**Potential contribution, NOT an established literature gap:** An empirical, controlled evaluation of activation redundancy in QNNs, the suitability of conventional vs quaternion-group-aware redundancy measures, and the practical value of this analysis for removal or merging of whole quaternion units. The nearest prior work must be discussed openly; this may turn out to be incremental if no distinctive behavior emerges.

## 2. Core research questions

- **RQ1 — Representation:** How much activation redundancy is present in trained quaternion-valued networks? How does it compare to parameter-matched real-valued models?
- **RQ2 — Measurement:** Do criteria that respect the four-component quaternion *unit* identify useful redundancies better than standard real-valued feature-similarity measures?
- **RQ3 — Compression:** Does an activation-redundancy-guided structured compression policy give a better quality–size trade-off than fair existing baselines?

Exploratory questions only if results justify them: across layers; before vs after training; effect of input channel relations; optional neuron merging with compensation.

### Working hypotheses (not findings)

- H1: Learned QNN activations contain redundancy at the level of quaternion units/filters.
- H2: Treating a quaternion feature as a coupled 4D object may change which units appear redundant compared with independently flattening its real components.
- H3: Redundancy-guided structural compression may outperform simple magnitude-based removal **at matched actual compact model size**; it may also fail, in which case report why.

## 3. Deliberate scope limits

**In scope first:** quaternion dense layers; possibly one small quaternion CNN after the dense proof of concept; one or two standard image datasets; activation collection; pairwise/covariance redundancy analysis; simple structured pruning and honest baselines.

**Out of scope initially:** complex RBMs; new quaternion multiplication rules; Hamilton-vs-random algebra experiments; inventing a new architecture; Transformers/LLMs; a large framework; compression across many algebras; an exhaustive benchmark suite; unsupported claims of invariance/equivariance. Complex-valued networks may be considered **only** as a later small extension.

Avoid the previous rabbit hole: this project is **not** about whether Hamilton multiplication itself is necessary for pruning robustness.

## 4. Experimental protocol (minimum viable research study)

### Phase 0 — Reproducible environment + library smoke test

1. Inspect the upstream KHNN / HypercomplexKeras repository, its license, examples, and dependency versions. Main paper: [NiemczynowiczKycia2025]. Further tensor formulation: [NiemczynowiczKycia2026].
2. Test installing the package and running **one forward pass, one backward pass, and one optimizer update** on a tiny hypercomplex model.
3. Verify all expected kernel parameters appear in trainable variables, receive non-`None` gradients, and actually change after an optimizer step. Be careful with TensorFlow vs Keras 3 behavior and `tf.Variable` tracking; do not assume compatibility from installation alone.
4. Record library commit hash, environment versions, random seeds, CPU/GPU device, and actual parameter counts.
5. If the package is incompatible, implement a minimal tested quaternion Dense layer **only after** documenting the issue; preserve a small, auditable codebase. Do not spend weeks rebuilding KHNN.

Official/author links:
- Article: https://doi.org/10.1016/j.softx.2025.102163
- SoftwareX archival repository: https://github.com/ElsevierSoftwareX/SOFTX-D-25-00004
- KHNN repository: https://github.com/rkycia/KHNN
- Keras extension: https://github.com/rkycia/HypercomplexKeras
- Package: https://pypi.org/project/HypercomplexKeras/

### Phase 1 — Baseline models, data, activations

- Start with **MNIST or Fashion-MNIST** to validate the entire pipeline; consider **CIFAR-10** for the main experiments only if the architecture/data representation makes scientific sense.
- Train (i) a small quaternion network and (ii) a carefully specified real-valued control. State whether the control matches **trainable real parameter count**, effective width, or compute — these are not interchangeable. Prefer matched parameter budgets plus matched pre-pruning quality where feasible.
- Quaternion input must be explicitly defined; **do not silently pad RGB with a zero channel without reporting it**. Consider grayscale grouping / a justified four-component encoding and at least one sensitivity check.
- Collect *post-activation* tensor snapshots on a held-out **calibration set**, never test data. Preserve the grouping of four real components into each quaternion unit.
- Dense activation layout: `(n_samples, n_quaternion_units, 4)`; CNN layout: `(n_samples, spatial_positions, n_quaternion_channels, 4)` or equivalently documented.

### Phase 2 — Measure redundancy BEFORE inventing a new score

Implement a few transparent reference measurements:

1. **Real baseline:** Pearson correlation/cosine similarity of flattened component-wise activations; clearly define whether the comparison uses individual scalar channels or grouped 4-component channels.
2. **Grouped baseline:** pairwise similarity of entire quaternion features represented as 4D real vectors, pooled over examples/spatial positions.
3. **Optional directional/alignment test:** fit a single quaternion coefficient relating two quaternion activation sequences and report held-out reconstruction error. This is a diagnostic, **not automatically a novel algorithm**.
4. **Global diagnostics:** effective rank / eigen-spectrum / pairwise redundancy distribution for each layer. Clearly separate these from actual removable units.

Checks: zero-variance activation handling; scale sensitivity; train–calibration vs calibration–test generalization; reproducibility across seeds; distinguish high similarity from low functional importance.

### Phase 3 — Structured compression baselines

Start with **at most three** decision policies to keep the experiment interpretable:

- Randomly remove whole quaternion units/channels (sanity check).
- Remove whole quaternion units/channels using a standard weight-based magnitude/group norm criterion (Vecchi2020; Singh2026).
- Remove whole quaternion units/channels using an established activation-redundancy criterion (pairwise similarity or nearest-neighbor coverage; Ayinde2019, Ebid2025). If justified, compare a group-aware variation *without assuming it is new*.

Optional only if promising: activation-guided **neuron merging with outgoing weight compensation** (Kim2020), taking care of multiplication order in quaternion layers and nonlinearity constraints. A simple quaternion least-squares fit alone is **not** a publishable methodological novelty.

**Critical:** zero masks demonstrate sparsity, not compact inference. For structural pruning, physically remove the selected channels and adjust connected layers; verify forward-pass equivalence before/after graph surgery when pruning 0 units; report trainable parameters, checkpoint bytes, MACs/FLOPs if measured consistently, and latency only when benchmarked.

### Phase 4 — Evaluation and decision

- Main plots: performance vs retained *real trainable parameters* and vs removed quaternion units. Include a pre-fine-tune curve; if used, report fine-tune results separately and equally for all approaches.
- Additional metrics: cross-entropy loss change, accuracy change, time and memory measurement where reliable, stability across >=3 seeds when feasible.
- A useful negative result is allowed **only if** supported by enough controls to rule out a weak baseline or software bug.
- Present an effect size / uncertainty interval where appropriate; do not rely on a one-seed accuracy difference.

## 5. Explicit stop/go gates

- **Gate A (infrastructure):** Can the quaternion network be trained correctly? If no, fix the environment or change implementation before claiming results.
- **Gate B (scientific signal):** Is measurable activation redundancy present, stable across held-out samples and seeds, and distinct from trivial inactive neurons?
- **Gate C (compression value):** Does redundancy-based channel removal retain useful accuracy at meaningful physical model-size reduction, with a fair baseline?
- **Gate D (paper):** Is the scientific story more than “we applied cosine similarity to QNN”? If no, do not dress it up as novelty. Either strengthen the empirical insight or abandon the paper hypothesis.

## 6. Suggested code structure

```text
hypercomplex-redundancy/
  PROJECT.md                 # project instructions (this document)
  LITERATURE.md              # annotated, linked research bibliography
  references.bib             # BibTeX citation records
  configs/                   # deterministic experimental configs
  src/
    models/                  # quaternion and real-valued control models
    data/                    # preprocessing with explicit 4D encoding
    activations/             # hook/collector and similarity metrics
    compression/             # compact architecture builders; baselines
    evaluation/              # parameters / accuracy / inference metrics
  tests/                     # shape, gradients, quaternion ops, pruning
  scripts/                   # train, collect, analyze, prune, compare
  results/                   # metrics, tables, provenance, graphs
```

Implement incrementally; don't create unused abstractions. Store configs and structured machine-readable results (CSV/JSON); use pytest for core invariants.

## 7. Agent's FIRST actionable task

> Inspect KHNN and write a *short technical feasibility report*, then implement and test a tiny quaternion Dense classifier. Confirm parameter tracking, gradients, optimizer updates, and correct output shapes. Do **not** start with a novel pruning algorithm. Once training works, collect grouped 4-component activations and plot a basic similarity matrix and a redundancy histogram. Only then propose one controlled pruning experiment.

Agent should record what ran, what failed, exact versions, and which findings remain hypotheses. Do not fabricate measurements or cite unverified publications.

## 8. Publication positioning

Working title: **Activation-Based Redundancy Analysis and Compression of Hypercomplex Neural Networks**.

A conference paper could contribute (i) an empirical characterization of redundancy in quaternion internal representations, (ii) a controlled comparison of activation-based redundancy measures, and (iii) a tested connection to physically structured compression. **Whether those are sufficiently novel cannot be settled from the title or from missing search matches.**

See `LITERATURE.md`: especially Maroudis2026, Ayinde2019, Kim2020, Ebid2025, Vecchi2020, Singh2026, and Kobayashi2010. These are prior art to take seriously, not footnotes to ignore.
