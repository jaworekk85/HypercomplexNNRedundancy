# Other four-dimensional products: relevance to the grouping study

Date:2026-11-26. Canonical citations: [references.bib](../../references.bib).
This is a bounded follow-up to the user's question, not a new algebra benchmark
or a priority claim. No non-Hamilton scientific run has been launched.

## What is established

Four real components specify a vector space, not its multiplication. A bilinear
product is additionally specified by structure constants, e.g.
`e_a * e_b = sum_c C[a,b,c] e_c`. Different choices can have different algebraic
properties. A learned PHM product should not be assumed to have Hamilton's
identities, a unit, associativity or division properties without checks.

`Valle2024VNets` presents vector-valued networks through restrictions on
real-valued operations and discusses hypercomplex, Clifford and Cayley-Dickson
examples. The inspected source is author-preprintv2, sectionsII--IV; the final
journal citation is IEEE Signal Processing Magazine41(3),49--58,
[DOI](https://doi.org/10.1109/MSP.2024.3401621).

`VieiraNetoValle2025VEfficientNets` compares quaternions, coquaternions,
tessarines and hyperbolic quaternions on ALL-IDB2. Inspected: author-preprintv1
sectionsII-B/V, TablesI/II. It includes RGB/HSV choices and describes hyperbolic
quaternions as non-associative; results do not establish a universal algebra
ranking.
[Inspected preprint](https://arxiv.org/html/2505.05659v1),
[conference DOI](https://doi.org/10.1109/IJCNN64981.2025.11227779).

The existing canonical key was preserved and upgraded from a preprint citation
to IJCNN2025, pages1--8. Publisher-deposited Crossref BibTeX and registry metadata
were downloaded and checked for title, author order, year, pages and DOI.
[Raw export](exports/2026-11-25/VieiraNetoValle2025VEfficientNets.bib),
[registry](exports/2026-11-25/VieiraNetoValle2025VEfficientNets.crossref.json).
The export directory is the continuing review's start date; retrieval timestamps
are recorded in sources.json. Published/preprint text identity is not assumed.

## The useful follow-up question

An extension could ask whether the advantage of preserving trained channel
groups depends on the multiplication, after controlling initialization, BN and
marginal feature statistics. This would generalize the present grouping question.
Classification ranking of these four algebras already has explicit prior art.
Repeating that ranking alone is not an established literature gap.

A small first extension would compare Hamilton with one named alternative,
such as coquaternions, on one existing dataset. It would require a separately
trained parent for the alternative product and the same real scalar coefficient
budget, split, input information and repair/training budget. Replacing the
multiplication table in an already trained Hamilton model confounds the question
with changing its represented function. The current BN and EuroSAT control
should be interpreted first; no extra baseline is silently queued.

## Implementation requirements before a run

- Fix the exact published table and multiplication side. The present local
  convolution implements right-Hamilton multiplication; a paper's left product
  cannot be substituted without checking signs and component ordering.
- Verify basis identities/properties claimed for each product and the rank of
  its real operator basis. Do not call all4Dproducts associative or division
  algebras.
- Confirm tracked variables, all coefficient gradients and actual updates,
  numerical scale and parameter counts before any experiment.
- Derive consumer normal equations from the selected basis. Existing
  `q_normal_equations`, Hamilton projection and compatibility-row formulas are
  Hamilton-specific, frozen sources; add new modules rather than reusing their
  assumptions for another product.
- Distinguish fixed multiplication, learned PHM coefficient sharing and a
  simple change of basis. Report actual compact storage, not just masking.

This proposal is an interpretation of the literature and current research
question. It is not evidence of superiority, novelty or a finished method.
