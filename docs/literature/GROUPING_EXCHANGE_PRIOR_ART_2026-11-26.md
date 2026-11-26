# Literature check for the bounded fixed-set exchange experiment

Date: 2026-11-26. The user authorized a targeted literature check and one
experiment, not a broad algebra/dimension benchmark. This note adds the closest
new permutation-search precedent to the existing contribution audit.

## What the literature already covers

`Martinez2021PermuteQuantizeFineTune`, CVPR2021, optimizes permutations before
vector quantization, then fine-tunes. Author-preprintv3 section3.2.1 describes
stochastic local search that exchanges two dimensions and accepts improvement
in a covariance-determinant criterion; the convolution appendix preserves spatial
kernel blocks. Thus two-entry exchange search for easier compression is not a
new algorithmic principle. The difference under investigation is compilation
into constrained Hamilton producers/consumers at fixed retained scalars and
actual model size, with actual block/network output scores.

Primary sources:
[official indexed CVF record](https://openaccess.thecvf.com/content/CVPR2021/html/Martinez_Permute_Quantize_and_Fine-Tune_Efficient_Compression_of_Neural_Networks_CVPR_2021_paper.html),
[author-provided citation](https://una-dinosauria.github.io/papers/bibtex/cvpr21.html),
[inspected preprintv3](https://arxiv.org/html/2010.15703v3).
The official record supplies six ordered authors,2021and pages15699--15708;
the author citation preserves the Barsan accent. No DOI was invented.

`Zhang2021StructuredShuffle` already jointly optimizes group convolution and
channel shuffle ([official proceedings](https://proceedings.mlr.press/v161/zhang21b.html)).
`Wang2019FLGC` already learns groups; `He2017ChannelPruning` already reconstructs
layer outputs after selection. `Singh2026` combines quaternion filter pruning,
fine-tuning and output-logit distillation in the inspected author-preprintv1
sectionsIV/IX-B ([preprint](https://arxiv.org/html/2510.21388v1)). These are
canonical existing entries, not new downloads. Our greedy proposal screen,
ridge, two-scalar swaps, local MSE and teacher KL individually have no novelty
claim. Bounded search differences do not establish priority for a combined method.

The official WANT@NeurIPS2023 LeanFlex-GKP poster was also screened: flexible
group counts and post-prune grouping selection are already explored in ordinary
CNNs. Only the poster was inspected here; no full-method comparison or new
canonical publication entry is claimed. It does not justify a first-adaptive-
dimension claim. Arbitrary-n PHM and Numerion precedents remain in the earlier
[dimension note](GROUP_SELECTION_AND_DIMENSION_2026-11-26.md).

## Why this experiment is bounded and useful

The [registered experiment](../../results/grouping_exchange_pilot/PROTOCOL.md)
keeps the same frozen scalar sets at sites0/8 in the three existing CIFAR Q
parents. It starts from intact and predetermined partially mixed assignments,
uses projection-cost and random proposals, scores actual full-block outputs,
then ranks at most nine whole-network finalist combinations by teacher KL.
All grouping selection uses disjoint inner/outer subsets inside FIT, and all
final consumers receive the same full-FIT refit as the intact control.
Previously explored CHECK/validation do not select the candidate. All seeds and
unchanged/original solutions must be reported.

This tests whether data-chosen groups improve practical compression beyond the
intact assignment under the bounded search. A negative result cannot establish
that every optimizer, larger search or longer repair fails. A positive result
would need to survive the three-seed comparison; it would not alone establish
method novelty. No new algebra/parent/dimension grid or automatic extra study.

## BibTeX provenance and access failures

The new canonical entry is checked against two primary metadata presentations
and the author preprint title page. The author-provided inline BibTeX was
retrieved through the web renderer and saved as
[extracted BibTeX](exports/2026-11-26/Martinez2021PermuteQuantizeFineTune.author-extracted.bib).
This file is an extracted source payload, not a byte-for-byte HTTP .bib response.
The direct .bib download returned cache miss; direct CVF page/PDF requests
returned403. The indexed CVF record and author HTML remained accessible.
No accepted-version/preprint identity is assumed. Verification metadata are
saved alongside the extracted entry; older export failures remain preserved.

Targeted queries: quaternion pruning/grouping/reconstruction; quaternion
channel permutation/compression; channel grouping/distillation/pruning;
exact PQF title; LeanFlex-GKP title. This is not a systematic priority search.
