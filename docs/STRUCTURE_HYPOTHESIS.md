# Structural constraints as a hypothesis

## Hypothesis

The user proposes that parameter sharing / restricted parameter freedom may
matter more than interpreting activations as quaternions. This is plausible
and not established by the present experiments. It does not imply that every
restriction is equivalent or that the chosen pattern cannot matter.

## Known implementation facts

For each quaternion input/output connection and each spatial kernel offset,
our convolution has four learned real coefficients. They generate a signed,
repeated 4x4 real block with 16 entries (`hamilton_kernel` in
src/models/quaternion_conv_torch.py). Thus the algebra imposes specific weight
sharing and sign relationships, not sixteen independent real parameters.
The expanded matrix is derived, not another set of trained parameters.

The approximately fourfold reduction is a same-width connection comparison,
not a fourfold saving against our narrower parameter-matched real control.
Our CNNs have 73,898 and 72,954 actual real trainable parameters, different
scalar widths and compute, and real component-wise BN/nonlinearities.

## What the early measurements did and did not establish (historical)

Three seeds show a quality difference under this particular CNN protocol.
They do not separate the effects of width, weight sharing, the specific
sharing pattern, optimization or regularization. There is no generic
structure-matched control establishing a causal explanation.

The early grouped-versus-scalar study compares activation diagnostics
WITHIN the same trained quaternion models, with the same whole-unit ablation
targets. It can test usefulness of group-aware measurement here; it cannot
prove that quaternion algebra is necessary, irrelevant or superior to all
other parameter-sharing structures. Either a positive or negative diagnostic
outcome leaves the broader causal hypothesis open.

## Literature and scope

ZhangEtAl2021 (references.bib) discusses parameterized hypercomplex
multiplications and parameter savings. Its primary-source abstract was
inspected for context, not as proof of the user's causal claim or an exact
compression-method comparison. General structural sharing is established
background; no novelty follows from naming it here.

The current work remains a bounded activation/compression study. No
Hamilton-versus-random-algebra trials, new multiplication rules or broad
architecture search are included in the diagnostic comparison.

## Intermediate folding: a structural fact, not a new theorem

Let `R(w)` be the implemented 4x4 right-Hamilton block, acting on column
coordinates as `x -> x*w`. Associativity gives `R(w) R(a) = R(a*w)`.
Thus replacing a post-block feature by `q_v ~= q_k*a` can be folded into the
next convolution without adding independent coefficients to a Hamilton block.
This equality concerns the graph AFTER substitution. Fitting `q_k*a` to
`q_v` is approximate and does not make the compressed network identical to
the original one.

For a nonzero outgoing weight `w`, `R(w)` is invertible and its inverse is
another right-Hamilton block. If a general replacement matrix `A` satisfies
`R(w) A = R(c)` for every possible feature input, then
`A = R(w)^(-1) R(c)` belongs to the same four-dimensional block family.
An all-zero outgoing edge is an exception; agreement on a restricted data
support is weaker than agreement for every input. Our implementation checks
the actual convention against the expanded convolution, rather than assuming
that left and right multiplication can be exchanged.

Closure under composition, rather than the name of the number system, is
the relevant structural property in this argument. It neither proves that
all parameter-sharing patterns are equivalent nor establishes a uniquely
quaternion performance benefit. General map fitting and consumer folding
have prior art (`Kim2020`, `Asiaee2026CMR`); no priority claim is made for
this elementary algebraic observation. The current experiment measures its
quality/size consequences within the implemented architecture family.

## The two implemented compensation procedures (2026-11-06)

These equations describe a single local step with a QUATERNION convolution
consumer. The actual study applies five such fits sequentially; its sixth
consumer is an already-real linear head and remains map-fold-only. At each
step the reference is the current model before that local removal, which may
already contain earlier compressed layers. BN buffers/affine parameters and
all other weights stay fixed during the closed-form regression; supervised
three-epoch repair is subsequent and separate.

Let K be retained Q input channels, V removed channels, S kernel offsets, n a
calibration image and p a spatial position (zero padding understood). The
source consumer output BEFORE BN/nonlinearity is

    z_o[n,p] = sum_{i in K union V, d in S} h_i[n,p+d] * w_{oi,d},

where * is Hamilton multiplication, on the RIGHT as in our frozen code.
Each w and h is a quaternion. The noncommuting multiplication order matters.

### Implemented method A: shared feature maps, then fold

For each removed channel v separately, fit quaternion coefficients a_{kv}:

    A_v = argmin_a sum_{n,p} |h_v[n,p] - sum_{k in K} h_k[n,p] * a_{kv}|^2
                    + lambda_A sum_k |a_{kv}|^2.

No affine intercept or feature centering. Coefficients are shared across
consumer output channels and kernel offsets. Associativity gives the folded
quaternion kernel

    w^A_{ok,d} = w_{ok,d} + sum_{v in V} a_{kv} * w_{ov,d}.

Folding exactly implements the APPROXIMATE substitution of h_v by its fitted
reconstruction; it does not make the compressed model equal to the reference.
No map is commuted through BN/ReLU and no offset is folded across padding.
Map coefficients are temporary fit variables, absorbed into existing weights;
they are not extra deployed parameters. Background: Kim2020, GRAIL2026,
KammaWada2021REAP. Actual sources: multilayer.SiteMoments.fit and fold,
multisource.py, internal_merging.outgoing_addition.

### Implemented method B: direct quaternion consumer-output fit

Allow every surviving consumer coefficient u_{ok,d} to be quaternion-valued,
without requiring the common-map factorization above:

    U^B = argmin_{U in H} sum_{n,p,o}
                 |z_o[n,p] - sum_{k in K,d in S} h_k[n,p+d] * u_{ok,d}|^2
                 + lambda_B sum_{o,k,d} |u_{ok,d} - w^A_{ok,d}|^2.

The ridge is centered on method A's folded kernel, not on zero. Patches and
the four Hamilton basis blocks give a real design D; for each output group
o, theta_o stacks its independent real Q coefficients, y_o stacks its four
output components. The implemented float64 solve is

    theta^B_o = theta^A_o + (D^T D + lambda_B I)^(-1)
                             D^T (y_o - D theta^A_o).

Numerical code uses a linear solve, not an explicitly formed inverse.
Both fits set lambda = alpha * mean(diag(Gram)) with alpha=.001, but their
Grams, objectives and ridge centers differ. The network comparison therefore
does not isolate parameterization alone. Unlabeled fit/check halves each have
500 images; check data never choose coefficients. Cite VieiraValle2022ELM
for the hypercomplex LS/real-embedding principle, He2017ChannelPruning and
Liang2021TwoStep for output reconstruction after channel selection. Sources:
src/compression/direct_consumer.py and frozen pilot/replication configs.

### What is preserved and what changes

Both deployed kernels have 4 * O * |K| * |S| independent real coefficients;
Hamilton 4x4 blocks are preserved. Method A explores updates linked by its
shared A; method B can adjust all allowed Q coefficients directly. Its
optimization has more adjustable degrees of freedom than the temporary map
fit, while the resulting MODEL has the SAME compact parameter count.

For two outputs and one surviving channel/offset, method A produces

    u_1 = w_11 + a*w_12,   u_2 = w_21 + a*w_22

with ONE common quaternion a. Method B can choose the two Q weights u_1,u_2
independently. Both resulting layers store two Q weights (eight reals);
method A's update is restricted to at most four real directions around the
stored original weights. With only one output/offset, a nonzero removed
weight can instead make this map surjective, so do not assert strict extra
restrictions for every possible architecture. The cached local diagnostic
tests the actual multi-output/spatial cases under a COMMON objective.

Full BN model 73898 reals -> compact models 42034/19098 at .25/.5 unit
removal. Thus equal counts refer to A-versus-B compact networks, not the
compact model versus its full parent. Initial fitted Q weights are checked
to retain the correct layout, gradients and subsequent optimizer updates.

## Possible tail-sensitive compensation (PROPOSAL ONLY, 2026-11-06)

Update: the simple exact-gradient ONE-late-consumer version has since been
implemented and measured in results/tail_compensation_pilot/REPORT.md. It
improved before-repair accuracy but demonstrated no after-repair benefit in
that seed. The optional Gauss-Newton formula below was not implemented.

Motivation/hypothesis: local output MSE treats all perturbation directions
equally, although later BN/ReLU/layers can amplify or suppress them. Existing
results do not establish that this is the cause of any accuracy difference.
Let g denote the fixed remaining network, including downstream normalization
and nonlinearities, ending in logits. With theta the Q consumer coefficients:

    min_theta sum_n ||g(z_n(theta)) - logits_ref_n||^2
                      + lambda ||theta-theta_B||^2.

This uses reference logits on unlabeled calibration data; it is a form of
functional distillation, not an automatically novel principle. Linearizing
f_n(theta)=g(z_n(theta)) around theta_B gives an optional ridge/Gauss-Newton
step with J_n = derivative of f_n w.r.t. theta and
r_n = logits_ref_n - f_n(theta_B):

    delta = (sum_n J_n^T J_n + lambda I)^(-1) sum_n J_n^T r_n.

All parameters remain Hamilton coefficients. The proposal requires derivatives
and may be costly; local linearization need not improve the nonlinear held-out
objective or classification. Assess cost versus label-free fit and supervised
repair separately. Lookahead/multi-layer objectives already exist
(Park2020Lookahead); replacement/logit-distortion context also exists
(Asiaee2026CMR). No priority claim for this formula, implementation, chosen
hyperparameters or new simulation. A genuinely distinctive method would need
a specific advantage over these established ideas, not just the word quaternion.

## Trainable shared maps during repair (original proposal, 2026-11-06)

Subsequently implemented as ONE-consumer correction pilot, completed in
results/trainable_maps_pilot/: control56.7%, learned56.9% validation n=1,
CE effectively unchanged. No convincing gain. Its zero-centered correction
initialization differs from the generic absolute-map proposal below.

Keep A as an explicit training variable and define effective Q weights

    U_{ok,d}(A,B,C) = B_{ok,d} + sum_v a_{kv} * C_{ov,d}.

With C fixed to removed-input outgoing weights, initialize A/B to reproduce
the old folded point exactly, and optimize supervised CE on labeled training
data. Differentiate through quaternion products using their four real
components; no teacher/distillation term is necessary. Fold A/C away after
training. Thus inference retains 4*O*|K|*|S| real coefficients, while training
adds 4*|K|*|V| map coefficients plus frozen-C storage. Do not count temporary
variables as permanent compact-architecture savings.

A-only with B/C fixed uses the restricted map update subspace. Joint A+B with
C fixed has the SAME function class as an unrestricted compact Q kernel B:
for any A and desired U, B can absorb their difference. In real coordinates
theta=b+T a, ordinary SGD (no regularizers/decay/momentum) gives

    theta_next = theta - (eta_B I + eta_A T T^T) grad_theta L.

The interesting question is optimization/regularization at fixed deployed
size, not additional representational capacity. No guaranteed improvement.
Ridge-optimal A cannot be improved on the identical fixed-feature ridge
objective just by running more gradients. Learning under classification loss
or jointly changing features is a different problem. Established ordinary-CNN
related work: Ding2021ResRep and Zhang2026RCP, see LITERATURE.md. No new
training/implementation or priority claim follows from this proposal.

## Joint calibration of shared maps and per-victim centers (2026-11-06)

User-authorized bounded pilot: results/centered_maps_pilot/, four cases in
configs/centered_maps_pilot.json. This is measured separately from the preceding
CE-only redundant-parameterization pilot. One Q mu_v is shared over all outputs
o and spatial offsets s. Freeze ordinary kernel bases during calibration:

    U_{ok,s} = B_{ok,s} + sum_v A_{kv} * (C_{ov,s} - mu_v)
             = B_{ok,s} + sum_v A_{kv} * C_{ov,s} - d_k,
    d_k = sum_v A_{kv} * mu_v.

This is a weight correction inside a convolution; zero padding is handled by
the original convolution, with no assumption of a spatially constant output
bias. Associativity/distributivity retain the fixed Hamilton family. An
independent mu_{ov,s} would be full removed-kernel learning and is not tested.

The actual anchor is the original rounded fold U0 with nonzero feature-ridge
A0: U=U0+(A-A0)C-A*mu. Equivalently B=U0-A0*C, differing from the original B
only by measured accumulated fold rounding. A=A0,mu=0 starts EXACTLY at U0.
This avoids the zero-mu-gradient problem at A=0. Calibration uses pre-BN
consumer-output SSE + lambda||U-U0||^2, fixed fit500/check500 image halves and
the original direct-fit lambda, not factor-specific regularization.

At fixed A, d=L(A)mu is a real-linear Q system. Jointly, deltaA,d enter U
linearly. Solve that relaxed linear problem, then recover mu from
L(A0+deltaA)mu=d. The relaxed minimizer is also an original-factor minimizer
ONLY if it is accurately recovered. Full row rank suffices for arbitrary d;
rank-deficient systems may still recover particular d. Record residual/rank/
condition, and verify the original Q product. Failure must not be hidden by
reporting an unrecovered relaxed result as joint A/mu. This elementary argument
is not a new linear-algebra theorem; credit VieiraValle2022ELM for the broader
hypercomplex LS/real-embedding framework.

Pilot cases: unchanged fold, A-only output refit, mu-only with fixed A0,
joint A/mu. Common kernel-centered objective permits nested fit-objective
checks, but not guaranteed accuracy or held-out ordering. Temporary adjustable
real coefficients: A1024, mu64, joint1088; unrestricted local Q kernel18432.
After calibration, fold factors away, recompute the SAME final-head treatment
and run matched3ep supervised repairs; all final models6Q/19098 reals.
Original direct-Q result reused as context. One source/consumer, no novel
method/priority/robust-gain claim, test inference, sweep or automatic extension.
