# Data-chosen groups and group dimension

Date: 2026-11-26. User-proposed extension, not a registered experiment. No
calculation or method-performance claim follows from this note. Current
[contribution audit](PAPER_CONTRIBUTION_AUDIT_2026-11-26.md) still applies.

## A concrete selection proposal

Keep exactly the retained scalar set from the original-group control and start
from its valid assignment. Propose exchanges between groups, initially at d=4.
Every proposed assignment must remain injective, use the entire fixed retained
set, and compile to the same actual compact architecture. There is no need for
the earlier pilot's reserve of untouched whole units to guarantee completion:
an exchange preserves the partition automatically.

For promising exchanges, compile the physical producer and refit the consumer
with a fixed budget. Rank candidates by the discrepancy in actual block outputs
or, for a small finalist set, whole-network teacher predictions after inserting
the compact block. A possible finalist score is mean teacher-to-student KL on
unlabelled FIT-only selection examples. Fix the temperature and any refitting
budget prospectively. Recompute the score for the full current partition;
independent pairwise correlations or per-group costs omit interactions.

Separate inner fitting from outer selection inside the existing FIT split;
CHECK and validation must not select assignments, restarts or repair epochs.
Allow the original partition to remain the answer. Monotonic improvement of a
selection score does not ensure better classification/generalization. Compare
the intact control with equal continuous-refitting opportunities and disclose
the additional discrete-search cost. No global optimum is promised.

This extends the previous catalogue pilot rather than inventing selection from
nothing: that pilot already scores projected-producer features with a refitted
Hamilton consumer, then performs greedy selection/local replacements. The
important proposed changes are a fixed retained set, exchanges without the
completion bias, and a composed block/network score for finalists. Standard
local search, teacher matching and reconstruction losses have prior art
(`Wang2019FLGC`, `Zhang2021StructuredShuffle`, `He2017ChannelPruning`,
`Singh2026`). None of these ingredients alone establishes method novelty.

## Group size is separate from multiplication

| Components d | Possible student family | Qualification |
|---|---|---|
| 2 | Fixed complex product, other specified 2D algebra, or PHM2 | A pair does not uniquely select multiplication. |
| 3 | Specified 3D bilinear algebra or PHM3 | Computationally valid; an unconstrained learned PHM does not automatically satisfy unit/associativity/division axioms. |
| 4 | Hamilton or another specified 4D product; PHM4 | The current tested family is Hamilton; learned PHM4 adaptation already has a failed gate. |
| 8 | Octonions, another specified 8D algebra, or PHM8 | Octonions are non-associative; PHM8 should not be called octonionic unless it implements that product. |

`Valle2024VNets` supplies the bilinear-algebra distinction
([author preprint v2, definitions/matrix embeddings](https://arxiv.org/html/2309.07716v2)).
`ZhangEtAl2021` already parameterizes arbitrary-dimensional multiplication
([author abstract](https://arxiv.org/abs/2102.08597)); `Grassucci2024` extends
this to convolution, including non-four-dimensional domains
([author preprint v2, introduction and section III](https://arxiv.org/html/2110.04176v2)).
`Cao2026Numerion` already uses multiple power-of-two dimensions and adaptive
fusion ([author preprint v1](https://arxiv.org/html/2510.03251v1), abstract;
earlier method inspection documented in the archived review).
These existing canonical entries were reused; no new bibliography entry or
downloaded export is claimed. The institutional PHNN PDF returned HTTP403;
the author HTML/abstract were accessible. Final journal/preprint identity was
not checked anew.

## Budget and interpretation

At fixed scalar input/output widths I/O and kernel area k, standard learned
PHM with d terms stores d^3 + O I k / d real coefficients when dimensions divide
I/O. This includes learned basis matrices. A specified fixed product with d
coefficient kernels has O I k / d trainables and no learned-basis overhead;
other buffers/storage must still be counted. Thus d=2 and d=8 at the same widths
do not have the same parameter budget. Dimension, Kronecker rank, scalar widths,
padding and group count are separate choices. The present scalar widths are
not divisible by three; PHM3 requires an explicit width/padding policy and its
real storage/compute cost, not a silent reshape.

One can fit differently structured compact students to the same existing
Hamilton teacher. This asks which representation works for compression of
that teacher; it does not compare equally trained parent algebras or establish
their intrinsic superiority. A parent-algebra comparison is a separate study
and requires its own trained parents, as explained in the algebra note.

The broad future optimization could jointly choose partitions and a deployed
dimension per layer under a measured whole-model budget. Flexible dimensions,
budgeted layer allocation and adaptive hypercomplex representations already
have prior art (`Grassucci2024`, `Liebenwein2021ALDS`, `Frantar2022SPDY`,
`Cao2026Numerion`). See the scoped 2026-11-07 archive, lines on variable-n
allocation; do not present this as a new idea by itself or restart closed
PHM adaptation studies without new authorization/evidence.

Small first question: can a physically scored fixed-set exchange search improve
the existing d=4 intact assignment beyond equal refitting? Changing d can then
be a separately justified extension. No adaptive search, new dimension or
training run has been launched for this brainstorming note.
