# Learned group-assignment pilot

## Hypotheses

Learning scalar-channel membership may help a fixed constrained CNN. One seed cannot establish a stable gain, novelty or compression value. Canonical context: Mena2018, Wang2019FLGC, Vecchi2020, Singh2026.

## Fixed protocol

One router between blocks 1 and 2; same 48k/1k/1k split and backbone initialization, dedicated sample-order/augmentation streams. Reuse the equivalent original identity seed-2026 reference. New random and learned runs use 50 epochs, 40 soft epochs then 10 hard epochs in the learned condition. Checkpoint choice uses minimum validation CE within epochs 40-50, never soft-model quality or test. All choices and the compression gate were frozen before the two new tests. The test set was used in earlier pilots.
Learned training adds 1024 logits and dense mixing computation. Export retains only permutation indices and the original backbone parameters. Training budgets are not parameter/compute matched; final routing does not itself remove channels. No Gumbel noise, new multiplication rules or latency gain is claimed.

## Measured results

| Mode | Training real parameters | Exported real parameters | Selected epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity (reused) | 73898 | 73898 | 49 | 78.30% | 79.00% |
| random | 73898 | 73898 | 49 | 78.00% | 78.14% |
| learned | 74922 | 73898 | 49 | 74.40% | 77.61% |

At hardening: soft validation 79.20%, immediate hard validation 56.00% (drop 23.20 pp). Selected exported assignment changes 2/32 output positions from the initial random assignment.
Validation-only compression gate: {'useful_hardened_quality': False, 'assignment_actually_changed': True, 'supports_compression_followup': False}.

## Interpretation and limitations

Retain any soft-to-hard collapse and failed gate. A soft model can exploit extra linear mixing, so soft quality does not prove a useful fixed grouping. A fixed hard model passing the gate would only justify a subsequent matched physical-compression comparison, not establish compression benefit. One seed and one boundary are exploratory; no first-in-literature or publishable novelty claim follows. A learned permutation is not inherently invariant to the chosen coordinate basis or uniquely quaternion-specific.

## Verified interpretation and decision — 2026-11-03

The resumed work verified a completed training, rather than starting new
epochs. Every new model completed 50 epochs. The checkpoint selected by the
fixed validation-CE rule is epoch 49 in all three conditions. Learned final
test accuracy is 77.61%, versus 78.14% for fixed random and 79.00% for the
reused original identity reference. These are one-seed observations, not a
statistical ranking of method families.

The learned assignment swaps scalar channels 1 and 9 between two groups:
`{1,7,14,15}` / `{9,18,23,24}` become `{7,9,14,15}` /
`{1,18,23,24}`. Six groups retain their original unordered membership.
Thus some actual membership was learned, but a different grouping did not
produce a useful hardened-quality gain under this protocol.

At epoch 40, validation accuracy drops 79.20% -> 56.00% immediately on hard
assignment. The last epoch reaches 74.90% validation accuracy; the selected
minimum-CE hard checkpoint has 74.40%. The selected hard model therefore
misses the previously frozen 77.80% validation gate.

The soft matrix remained diffuse: mean row entropy 1.9915 nats; mean mass
on the exported matching about 0.3041. Replacing a broad mixture with one
channel per position substantially changes the forward computation. This is
consistent with learning useful soft mixing without learning an equally
useful near-discrete assignment. Hardening also changes downstream activation
distributions and the BN statistics fitted to soft activations; this pilot
does not isolate those mechanisms. The limited late-schedule adaptation is
another limitation, not evidence that all learned grouping must fail.

No additional compact models were generated after the failed validation-only
gate. The conditional stage is explicitly recorded as `not_run_gate_failed`.
This preserves the negative result without choosing another schedule, score
or gate from test performance. No novelty or paper-readiness claim follows.

Verification: 36 implementation tests; GPU prototype integrity checks passed;
64 saved-routing checks passed; 7 conditional-stage checks passed. Separate
files retain the discarded six-epoch technical run and its resume-code fix.
The baseline and earlier results were preserved.
