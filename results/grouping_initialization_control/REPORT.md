# Grouping before and after training

Status: completed.

## Hypothesis

The original-group advantage may already exist in randomly initialized Hamilton layers. A change after training is exploratory evidence beyond this architectural baseline, not proof of semantic quaternion entities.

## Measured normalized grouping penalties

Delta = mean held-out normalized MSE of10fully mixed draws minus original-group normalized MSE.
Positive training increment means a larger penalty after training.

| Seed | Block | Policy | Initial delta | Trained delta | Training increment | Feature-only increment | Consumer-only increment | Interaction |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 2026 | 0 | role | 0.0420852 | 0.131758 | 0.0896731 | 0.0178933 | 0.00689473 | 0.0648851 |
| 2026 | 0 | free | 0.0673435 | 0.174351 | 0.107008 | 0.014438 | 0.00280275 | 0.0897673 |
| 2026 | 8 | role | 0.0182305 | 0.279588 | 0.261357 | 0.239511 | -0.012413 | 0.0342592 |
| 2026 | 8 | free | 0.0192345 | 0.290347 | 0.271113 | 0.255961 | -0.0130678 | 0.02822 |
| 2027 | 0 | role | 0.0565057 | 0.134308 | 0.0778025 | 0.0377079 | 0.0139859 | 0.0261086 |
| 2027 | 0 | free | 0.0654525 | 0.167669 | 0.102217 | 0.0672913 | 0.0130299 | 0.0218954 |
| 2027 | 8 | role | 0.0189222 | 0.258787 | 0.239865 | 0.225569 | -0.0132789 | 0.0275753 |
| 2027 | 8 | free | 0.0200112 | 0.27687 | 0.256859 | 0.246867 | -0.0141595 | 0.0241515 |
| 2028 | 0 | role | 0.0631791 | 0.136679 | 0.0734998 | 0.033126 | -0.0206293 | 0.0610031 |
| 2028 | 0 | free | 0.0599078 | 0.152195 | 0.0922877 | 0.0592729 | -0.00578811 | 0.0388028 |
| 2028 | 8 | role | 0.0109809 | 0.265062 | 0.254081 | 0.235479 | -0.00693873 | 0.0255406 |
| 2028 | 8 | free | 0.0118993 | 0.285577 | 0.273678 | 0.263028 | -0.00758435 | 0.0182339 |

## Interpretation and limits

Reuse the EXACT retained units and original/10role-preserving/10free-role fully mixed assignments from the completed study, identical across initialization/trained conditions. No50percent draws, producer compilation, producer optimization, supervised repair, accuracy measurement or new architecture/algebra. Fixed exact-feature consumer diagnostic only:3seeds x2sites x4crossed feature/consumer states x21groups =504ridge fits. Trained-feature/trained-consumer126fits must replay predecessor results.

Regenerate step0 using the frozen baseline seed_everything(seed) and CifarResNet20 constructor on CPU, then move to GPU; all parameters and BN state untrained. Save/hash these initial models before scientific fitting. This reconstructs the initialization recipe; no archived epoch0checkpoint exists to independently prove bitwise ancestral identity. No BN recalibration. Initial features are produced by the complete randomly initialized prefix on the same real calibration images; trained features come from predecessor hash-checked packs.

Within each seed/site/cell compute mean of10draws per policy, retaining all individual draws. Delta=mean normalized mixed error minus normalized original error. Primary training increment=Delta(trained,trained)-Delta(initial,initial); positive means larger penalty after training, negative means smaller. Secondary feature and consumer increments relative to initial/initial and descriptive2x2interaction reported. Report each seed/site; draw/image counts are not independent network seeds. No equivalence/significance/novelty/general necessity claim, no numerical continuation threshold, and no automatic further study. Existing failed pilot gate remains failed.

Initialization uses the same Hamilton architecture, not a random algebra. This is an exact-feature diagnostic;
it establishes no deployed compact architecture savings or random-model classification comparison.
Per-cell rows retain all draws, target energies, raw/normalized MSE, ridge penalties and replay controls.
