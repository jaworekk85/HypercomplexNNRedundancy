# Trainable Q-map correction pilot

Status: completed. CIFAR BN seed2026, f=.5, ONE late consumer.

## Hypothesis

Temporary jointly learned quaternion corrections may change optimization at the same deployed compact size. Exact predecessor not located; no novelty or priority claim.

## Measured results

| Case | Training reals | Deployment reals | Validation before (%) | Validation epoch3 (%) |
|---|---:|---:|---:|---:|
| frozen_maps | 19098 | 19098 | 34.80 | 56.70 |
| learned_maps | 20122 | 19098 | 34.80 | 56.90 |

## Interpretation and limits

Observed validation difference: +0.2 percentage points (2/1000 images), n=1.
Validation CE: control 1.2127349243, learned 1.2127392883, slightly worse and
effectively unchanged. Separate calibration-check accuracy 55.8/56.4%, CE
1.2539669189/1.2522645264. No convincing benefit demonstrated; no statistical
or equivalence conclusion. Both arms: 1125 attempted /1124 successful AMP steps.

Independent CPU audit passed: 130 frozen sources/296 inputs unchanged, six
full split evaluations (initial validation/check once plus both final models),
correct actual compact parameter/layout counts, fixed C, zero control maps and
nonzero updates in all four learned components. Saved GPU export logit error
is zero in both arms. See independent_audit.json and
scripts/review_trainable_maps_pilot.py. These checks establish implementation
and metric consistency, not novelty or a robust accuracy gain.

Two new matched three-epoch CE repairs, fixed masks and identical direct-Q initialization. A corrections start at zero; B and all ordinary parameters train, C remains frozen. DeltaA has 1024 temporary real trainables; C stores 9216 frozen reals and the product basis 64. Control retains the same buffers with zero fixed DeltaA. Both export to standard six-Q-convolution models with 19098 trainable reals. BN/dropout follow the existing recipe. No teacher distillation, test evaluation, LR tuning or automatic replication.

Single previously studied discovery source: no independent confirmation, significance or superiority claim. Joint A+B changes parameterization within the same compact Q functional family. AdamW and correction-centered decay are specified choices; plain-SGD preconditioning analysis does not describe AdamW exactly. Broad learned-compensator/folding prior art: Ding2021ResRep, Zhang2026RCP; hypercomplex adapters: Mahabadi2021Compacter; optimizer reparameterization: Ding2023RepOptimizers.

Atomic completed-epoch model/optimizer/scheduler/scaler/RNG checkpoints. STOP checked after each batch; an interrupted partial epoch is replayed from its beginning. Remove STOP only when resuming. Schedule follows attempted AMP steps; successful steps recorded separately. Final epoch is fixed; validation never selects a checkpoint.
