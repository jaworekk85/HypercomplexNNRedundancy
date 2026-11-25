# Q ResNet component-reassembly pilot

Status: completed.

## Hypothesis

Selecting scalar components, assigning new quaternion groups and compiling both producer and consumer may improve compressed quality beyond whole-unit selection at identical physical architecture size. Role-fixed selection separates component retention from free role reassignment. No novelty or gain is assumed.

## Measured results

| Method | Actual real trainables | Validation before (%) | Validation epoch3 (%) | CE after |
|---|---:|---:|---:|---:|
| legacy_direct | 232266 | 87.20 | 89.80 | 0.300787 |
| whole_joint | 232266 | 87.60 | 90.40 | 0.297067 |
| scalar_energy | 232266 | 80.20 | 88.60 | 0.345657 |
| role_fixed | 232266 | 87.60 | 90.40 | 0.297109 |
| free_roles | 232266 | 87.80 | 90.80 | 0.295178 |

| Block | Method | Mixed quartets | Changed roles | Fit MSE | Check MSE |
|---|---|---:|---:|---:|---:|
| 0 | legacy_direct | 0 | 0 | 0.365304 | 0.396004 |
| 0 | whole_joint | 0 | 0 | 0.365296 | 0.395997 |
| 0 | scalar_energy | 4 | 4 | 0.685193 | 0.733139 |
| 0 | role_fixed | 0 | 0 | 0.365296 | 0.395997 |
| 0 | free_roles | 2 | 2 | 0.375036 | 0.40896 |
| 8 | legacy_direct | 0 | 0 | 0.150042 | 0.153201 |
| 8 | whole_joint | 0 | 0 | 0.146781 | 0.150237 |
| 8 | scalar_energy | 16 | 12 | 0.378322 | 0.373626 |
| 8 | role_fixed | 0 | 0 | 0.146781 | 0.150237 |
| 8 | free_roles | 0 | 0 | 0.146781 | 0.150237 |

## Interpretation and limits

ONE Q ResNet20 CE-selected parent, seed2026, blocks0/8 independently reconstructed against source-parent targets,50% internal width removal at each. Five combined models, each232266real trainables versus parent271594; approximately14.5% full-model savings, not50%. Same final standard Hamilton architecture, BN/ReLU and residual outer dimensions; no router, adapter, full producer or learned PHM in the deployed compact network. No new baseline/scratch200epoch run, other seeds, all-nine-site compression or official-test inference; no automatic follow-up.

legacy_direct reproduces the frozen activation-reconstruction-times-outgoing-energy masks, mapfold and centered direct-Q fit on the two sites, keeping original alpha. It is a same-scope rerun, not a relabelled all-nine-site historical result. whole_joint uses the NEW joint selector/consumer fit but only original intact quartets. scalar_energy takes the4khighest uncentered fit activation second moments and assigns sorted source indices in component-blocked order. role_fixed searches new groups but each source scalar retains its original r/i/j/k role. free_roles can assign any source component to any new role. All scalar selections are injective; no source feature is duplicated.

Evaluate validation/check CE and accuracy before repair and after fixed final epoch3; no best-epoch selection. Same paired train-order/augmentation streams, AdamW/AMP recipe and attempted update budget; record successful/skipped updates and any scheduler warnings. All weights train; count232266actual trainables, buffers, checkpoint bytes and selection/fitting times separately. Additional search/calibration compute is NOT equalized by equal repair epochs. Per-model saved-weight forward replay and hash checks required. No official-test predictions.

Primary comparison free_roles versus whole_joint; report role_fixed, scalar_energy and legacy_direct even if unfavorable. Continuation gate requires free_roles final validation accuracy >= the BEST accuracy of legacy_direct/whole_joint/scalar_energy plus0.5pp and CE <= the LOWEST CE of those controls. Practical single-seed gate, not significance or novelty evidence. Local MSE and role_fixed outcomes are secondary and cannot silently replace this gate. No automatic replication or retraining follows.

Gate outcome: `{"status": "evaluated", "continue_to_replication": false, "accuracy_gain_vs_best_control": 0.0040000000000000036, "ce_gain_vs_best_control": 0.0018893928527832227, "primary_free_minus_whole_accuracy": 0.0040000000000000036, "primary_free_minus_whole_ce": -0.0018893928527832227, "statistical_significance_claimed": false, "automatic_followup": false}`.

Detailed mask/group assignments, selection traces, times and local metrics are in per-block records.
No local-error ordering, novelty, equivalence or general classification gain is assumed.
Prior art: He2016ResNet, VieiraValle2022ELM, He2017ChannelPruning, GRAIL2026, Frantar2022OBC, Jeong2022QCQP, Zhang2021StructuredShuffle, Wang2019FLGC, Vecchi2020, Singh2026 (references.bib).
