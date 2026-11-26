# IJCNN 2027 manuscript draft

Working title: **Quaternion Channel Grouping under Structural Compression:
Reconstruction, Realizability, and Repair**.

The user authorized the initial draft on 2026-11-26. The English manuscript is
[main.tex](main.tex); the built reading copy is [draft.pdf](draft.pdf). It is an
exploratory empirical paper draft requiring human scientific review. Final test
evaluation has not been performed for these grouping studies. No new numerical
experiment, source modification, submission, or acceptance claim follows from
writing the draft.

The revised version has six Letter pages, five printed tables, a conceptual
grouping figure, a paired-seed results figure and23citations. Six table sources
are generated; the full seed table remains available as a source but is not
duplicated in the printed paper. The shorter abstract and introduction state the
empirical question and three contributions; parent accuracy is included from
six existing replay records. It includes exact/physical local reconstruction
means over all fully mixed draws, and two constructed synthetic examples showing
feasible beneficial reassembly. These are existing results, not new experiments;
the synthetic examples allow scalar selection to change, and all three scalar
selectors choose the same quartet. [Build inspection](build_report.json) records embedded fonts,
no overfull boxes, no undefined references/citations and verified evidence/vendor
hashes. Underfull paragraph/bibliography warnings remain. The first XeTeX
preview used a fallback font; loading T1 encoding before IEEEtran restored the
class's Times-compatible Nimbus Roman family in the final PDF. This does not
constitute scientific review or IEEE PDF eXpress validation.

## Template and venue instructions

Checked 2026-11-26:

- [IJCNN 2027 initial author instructions](https://ijcnn.org/2027/authors/initial-author-instructions)
  specify IEEE conference style, Letter paper, two columns, 10-point type,
  anonymous authors and no page numbers. Six pages include references; up to
  four additional pages are listed at USD100per extra page. This was rechecked
  after the user's page-limit question; six is the standard allowance, not the
  absolute paid maximum. We target the standard six pages, including references.
- [IEEE author tools](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/)
  direct authors to IEEE conference templates. `IEEEtran.cls` v1.8b and the
  IEEEtran bibliography style are vendored unchanged from the maintainer's
  [CTAN distribution](https://ctan.org/pkg/ieeetran). This is the standard class,
  not a separately branded IJCNN class. Upstream notices remain in the files;
  see [vendor/template.json](vendor/template.json) for sources and hashes.
- The main IJCNN landing-page CFP says six pages. A separate
  `/2027/full-paper-authors-instructions` page contains conflicting A4/seven-page
  guidance and unfilled camera-ready placeholders. This draft follows the
  explicit **initial** instructions linked from the live Authors menu. Recheck
  the submission-stage instructions before uploading; this does not establish
  final camera-ready requirements.
- The initial instructions require disclosure of AI-generated text in the
  acknowledgments and citations in affected sections. This is stricter wording
  than the general IEEE author-center page, which asks the acknowledgment to
  identify affected sections and extent of use. We follow the conference's
  explicit instruction. It does not mandate a full repeated sentence or a
  particular display style. The draft now uses short section-opening
  `AI-assisted drafting [citation]` markers and a concise acknowledgment naming
  the tool, affected text, LaTeX and table/figure-code assistance. The previous
  longer acknowledgment contained internal review reminders, which have been
  removed from the manuscript; human scientific review and updating the final
  workflow disclosure remain pending tasks documented here. The draft includes both,
  using canonical key `OpenAI2026Codex`. The software entry is manually curated
  from the official product page, not a publisher BibTeX download. No human
  author review, model-version identifier or Claude contribution is invented.

The two user-supplied model reviews are assessed in
[PAPER_REVIEW_ACTIONS.md](../docs/PAPER_REVIEW_ACTIONS.md). They do not count as
independent human/experimental validation. Implementation-failure detail was moved
out of the manuscript because it is peripheral to the paper question; the full
KHNN/version-specific tracking result and EuroSAT precision failure remain in
their existing records. The working-draft banner was removed, while explored
validation and absence of a final test assessment remain explicit in limitations.
This editorial change does not make the paper submission-ready. Another Codex
rewrite remains AI-assisted; disclosure scope was not reduced.

## References expanded after the first draft

Six previously indexed canonical sources are now substantively discussed and
cited, increasing the printed bibliography from17to23entries (including the
AI-tool disclosure and the separately labelled conference abstract). This is not
six newly discovered publications or a new systematic literature review.

| Canonical key | Purpose in manuscript | Verification |
|---|---|---|
| `Chaudhary2024` | Quaternion transformations/pruning for audio tagging | Official ISCA abstract and inline citation rechecked; previously downloaded Crossref BibTeX/registry preserved. |
| `ShahadatMaida2024CrossChannel` | Cross-channel weight-sharing context | Publisher/author-preprint abstract and existing journal metadata; no new full-method inspection claimed. |
| `PoppelbaumSchwung2025TimeCompression` | Component versus magnitude temporal pooling, distinct from our hidden-channel compression | Author preprintv2sectionVI-B3 rechecked; journal metadata/BibTeX previously verified. |
| `Ayinde2019` | Redundancy/cosine-based pruning precedent | Publisher abstract checked and Crossref BibTeX downloaded2026-11-26; canonical metadata matches. |
| `Kim2020` | Neuron merging and conditions for absorption into the next layer | Official abstract and BibTeX checked; canonical entry gains verified pages585--595and publisher. |
| `Lakens2017Equivalence` | Explains why small observed differences do not establish equivalence | Publisher abstract and prior verified BibTeX; no equivalence test performed here. |

Source links, download hashes and access limits are recorded in
[literature_verification.json](literature_verification.json), with new raw exports
in`docs/literature/exports/2026-11-26/manuscript-expansion/`. Previously archived
exports remain unchanged. Shortening redundant prose kept the expanded paper
within six pages; class, font sizes and margins were not reduced.

## Build

From the repository root on this laptop:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File paper/build.ps1
```

The script first regenerates tables, the publication citation subset and a
paired-seed plot, then
uses Tectonic from `.cache/paper-tooling/tectonic/tectonic.exe` or PATH. It sets
`TECTONIC_CACHE_DIR` to the repository-local cache. Downloaded tools and TeX
resources are not versioned. Compiler used for this draft: Tectonic 0.17.0,
official Windows release. To use another installation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File paper/build.ps1 -Tectonic C:\path\to\tectonic.exe
```

On a conventional TeX installation, run `python paper/prepare.py`, then from
`paper/` use `latexmk -pdf -outdir=build main.tex` (or the normal
pdflatex/bibtex/pdflatex/pdflatex sequence). The source folder can also be
uploaded to Overleaf: select `main.tex` and compile normally. Supply the
generated `references.bib`, `tables/`, `IEEEtran.cls` and `IEEEtran.bst`.

[prepare.py](prepare.py) uses only the Python standard library.
[plot_overview.py](plot_overview.py) uses the existing NumPy/Matplotlib environment
and saved cross-setting summary, with provenance in `figures/manifest.json`.
[plot_grouping_concept.py](plot_grouping_concept.py) creates an illustrative
same-component assignment diagram and verifies identical eight-component
multisets across panels. It contains no measured data; definitions/hashes are in
`figures/grouping_concept.json`.
Neither script evaluates a model. The root
[references.bib](../references.bib) remains canonical. The copy in this folder
contains only cited entries and omits internal provenance `note` fields; do not
edit it manually. Numerical table cells are generated from saved seed-level
results, cross-checked against the shared summary, with input/output hashes in
[evidence.json](evidence.json). Regenerating tables does not train or evaluate
any model. Python checks 24 existing compact-model rows, six saved parent replays,
126local rows,8synthetic rows,
actual parameter counts,
the 12 producer-feature improvements and four consumer-output worsenings.

## Claims and evidence

| Draft content | Existing source | Boundary |
|---|---|---|
| Hamilton convention, projection and ridge | `src/models/quaternion_conv_torch.py`, `src/compression/component_reassembly.py`, `src/compression/grouping_mechanism.py` | Standard algebra/least squares, no new theorem or solver. Ridge penalizes independent coefficients; it is not the older centered-prior estimator. |
| CIFAR grouping and repair | [Report](../results/grouping_mechanism_study/REPORT.md), `summary.json` | Fixed masks, three parents, two sites, draw zero chosen in advance. Fifteen models include four mixed conditions. |
| All-draw local reconstruction | Same grouping `summary.json` | Newly tabulated descriptive means over all ten fully mixed draws/policy/site and three source seeds. Raw MSE scales differ by site; draws are not new training replications. |
| Constructed beneficial reassembly | [Report](../results/component_reassembly_diagnostic/REPORT.md), `summary.json` | Two favorable/less-compatible synthetic teachers, exhaustive FIT-only selection and independent CHECK inputs; changed retained sets, no selector advantage, no classification gain claim. |
| EuroSAT grouping and repair | [Interpretation](../results/eurosat_grouping_fp32_replication/INTERPRETATION.md), `summary.json` | Nine compact models; fresh deterministic FP32 references, AMP repair, 17 skipped updates. Failed predecessor retained. |
| Shared means and physical budgets | [Cross-setting report](../results/eurosat_grouping_fp32_replication/CROSS_DATASET.md), `cross_dataset_summary.json` | Dataset, architecture, encoding, savings and repair update budgets change together. |
| Initialization baseline | [Interpretation](../results/grouping_initialization_control/INTERPRETATION.md) | Regenerated initial weights; native contrasts alone do not identify learned semantics. |
| BN/statistics controls | [Interpretation](../results/grouping_statistics_control/INTERPRETATION.md) | Standardization changes the diagnostic teacher; counts are not independent source seeds. |
| BN-only classification | [Interpretation](../results/grouping_bn_accuracy_control/INTERPRETATION.md) | One failed recipe; historical repair changes weights and statistics and starts elsewhere. |
| Adaptive negative comparison | [Interpretation](../results/grouping_exchange_pilot/INTERPRETATION.md) | Bounded search, intact answer selected, identical historical repaired models explicitly reused. |
| Earlier 90.8% pilot | [Report](../results/resnet_component_reassembly_pilot/REPORT.md) | One seed, different masks, completion bias and failed practical screen. |

Earlier reconstruction, real/Q baselines and PHM studies remain documented in
[the evidence map](../docs/PAPER_EVIDENCE_MAP.md). The draft does not combine
their incompatible budgets into a single ablation or erase their failed gates.
The main text prioritizes the grouping question rather than listing every
experiment. See the literature audits for inspected source/version limits.

## Human review and remaining publication work

1. Review the central contribution: is the empirical separation useful and
   sufficiently different from the nearest quaternion pruning/grouping work?
2. Verify the mathematics against the right-Hamilton convention, descriptions
   against the frozen protocols, and each related-work statement against the
   inspected primary source. Bibliographic provenance remains in the canonical
   database; its removal from the printed copy does not remove review duties.
3. Check the figures/tables, paired seed interpretation, validation selection,
   precision correction, AMP skips and transparent reuse of repaired models.
4. Lock a final evaluation/reporting protocol and audit earlier access to any
   proposed test split before test inference. No such inference was authorized
   or performed by this drafting step.
5. Confirm the author list/affiliations and current venue requirements. Review
   the AI disclosure to describe the final actual workflow accurately. A review
   by external GPT and Claude models has now been supplied by the user; it is
   assessed in[the review-response record](../docs/PAPER_REVIEW_ACTIONS.md).
   Independent human scientific verification remains pending.

These are editorial and final-evaluation tasks, not an automatic queue of new
algebra, dimension, normalization or optimization experiments. A successful
LaTeX build does not establish submission readiness or IEEE PDF eXpress compliance.
