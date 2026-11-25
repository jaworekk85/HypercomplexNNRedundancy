# Experiment report index

Updated: 2026-11-25. Use [CURRENT.md](CURRENT.md) for current state and
[PLAN_IJCNN2027.md](PLAN_IJCNN2027.md) for next decisions. This index links
per-study evidence instead of duplicating metrics and old execution notes.

Reports, protocols, failed checks and interrupted-run records remain in their
original result directories. Each report states its own seeds, data access,
verification and limitations; a listed study is not a novelty or gain claim.

## ResNet and PHM studies

| Study | Documents |
|---|---|
| resnet_component_reassembly_pilot (completed; continuation gate failed) | [REPORT](../results/resnet_component_reassembly_pilot/REPORT.md), [Plan](../results/resnet_component_reassembly_pilot/PROTOCOL.md), [completion audit](../results/resnet_component_reassembly_pilot/completion_audit.json) |
| phm_basis_compression_pilot (completed; continuation gate failed) | [Protocol](../results/phm_basis_compression_pilot/protocol.json), [REPORT](../results/phm_basis_compression_pilot/REPORT.md) |
| phm_initialization_pilot | [INTERPRETATION](../results/phm_initialization_pilot/INTERPRETATION.md), [REPORT](../results/phm_initialization_pilot/REPORT.md) |
| phm_resume_20261125 | [REPORT](../results/phm_resume_20261125/REPORT.md) |
| resnet20_baselines | [REPORT](../results/resnet20_baselines/REPORT.md) |
| resnet20_compression | [REPORT](../results/resnet20_compression/REPORT.md) |
| resnet20_key_replication | [seed_2027/REPORT](../results/resnet20_key_replication/seed_2027/REPORT.md), [seed_2028/REPORT](../results/resnet20_key_replication/seed_2028/REPORT.md) |
| resnet20_qstem_phm4_baseline | [REPORT](../results/resnet20_qstem_phm4_baseline/REPORT.md) |
| resnet20_replication_baselines | [seed_2027/REPORT](../results/resnet20_replication_baselines/seed_2027/REPORT.md), [seed_2028/REPORT](../results/resnet20_replication_baselines/seed_2028/REPORT.md) |
| resnet_structure_catalogue | [REPORT](../results/resnet_structure_catalogue/REPORT.md) |
| structure_catalogue_audit | [REPORT](../results/structure_catalogue_audit/REPORT.md) |

## Earlier studies and diagnostics

| Study | Documents |
|---|---|
| activation_sensitivity | [REPORT](../results/activation_sensitivity/REPORT.md), [magnitude_tanh/compression_curve/REPORT](../results/activation_sensitivity/magnitude_tanh/compression_curve/REPORT.md), [split_leaky_relu/compression_curve/REPORT](../results/activation_sensitivity/split_leaky_relu/compression_curve/REPORT.md) |
| block_analysis | [PROTOCOL](../results/block_analysis/PROTOCOL.md), [REPORT](../results/block_analysis/REPORT.md) |
| bn_compression | [PROTOCOL](../results/bn_compression/PROTOCOL.md), [REPORT](../results/bn_compression/REPORT.md) |
| centered_maps_pilot | [REPORT](../results/centered_maps_pilot/REPORT.md) |
| component_reassembly_diagnostic (synthetic only) | [REPORT](../results/component_reassembly_diagnostic/REPORT.md), [protocol](../results/component_reassembly_diagnostic/protocol.json) |
| cifar_cnn | [PROTOCOL](../results/cifar_cnn/PROTOCOL.md), [REPORT](../results/cifar_cnn/REPORT.md) |
| cnn_compression | [PROTOCOL](../results/cnn_compression/PROTOCOL.md), [REPORT](../results/cnn_compression/REPORT.md), [block_analysis/REPORT](../results/cnn_compression/block_analysis/REPORT.md) |
| cnn_multiseed | [PROTOCOL](../results/cnn_multiseed/PROTOCOL.md), [REPORT](../results/cnn_multiseed/REPORT.md), [seed_2027/REPORT](../results/cnn_multiseed/seed_2027/REPORT.md), [seed_2028/REPORT](../results/cnn_multiseed/seed_2028/REPORT.md) |
| compression_curve | [REPORT](../results/compression_curve/REPORT.md) |
| compression_curve_eurosat | [REPORT](../results/compression_curve_eurosat/REPORT.md) |
| compression_curve_eurosat_refinement | [REPORT](../results/compression_curve_eurosat_refinement/REPORT.md) |
| compression_curve_refinement | [REPORT](../results/compression_curve_refinement/REPORT.md) |
| direct_consumer_pilot | [REPORT](../results/direct_consumer_pilot/REPORT.md), [gap_diagnostic/REPORT](../results/direct_consumer_pilot/gap_diagnostic/REPORT.md) |
| direct_consumer_replication | [REPORT](../results/direct_consumer_replication/REPORT.md) |
| eurosat_compression | [REPORT](../results/eurosat_compression/REPORT.md) |
| finetune_compression | [REPORT](../results/finetune_compression/REPORT.md) |
| group_scalar | [FAILED_CHECKS](../results/group_scalar/FAILED_CHECKS.md), [PROTOCOL](../results/group_scalar/PROTOCOL.md), [REPORT](../results/group_scalar/REPORT.md) |
| hard_forward_grouping | [REPORT](../results/hard_forward_grouping/REPORT.md) |
| internal_merging | [REPORT](../results/internal_merging/REPORT.md) |
| internal_replication | [REPORT](../results/internal_replication/REPORT.md) |
| learned_grouping | [INTERRUPTED](../results/learned_grouping/INTERRUPTED.md), [PROTOCOL](../results/learned_grouping/PROTOCOL.md) |
| learned_grouping_compression | [REPORT](../results/learned_grouping_compression/REPORT.md) |
| learned_grouping_pilot | [PROTOCOL](../results/learned_grouping_pilot/PROTOCOL.md), [REPORT](../results/learned_grouping_pilot/REPORT.md) |
| multisource_compensation | [REPORT](../results/multisource_compensation/REPORT.md) |
| no_bn_pilot | [REPORT](../results/no_bn_pilot/REPORT.md) |
| paper_a_bn | [REPORT](../results/paper_a_bn/REPORT.md) |
| paper_a_evidence | [RECOVERY](../results/paper_a_evidence/RECOVERY.md), [REPORT](../results/paper_a_evidence/REPORT.md) |
| paper_a_final | [REPORT](../results/paper_a_final/REPORT.md) |
| phase0 | [FEASIBILITY](../results/phase0/FEASIBILITY.md) |
| pseudolabel_repair | [REPORT](../results/pseudolabel_repair/REPORT.md) |
| qnn_compression_review | [REPORT](../results/qnn_compression_review/REPORT.md) |
| quaternion_merging | [DIAGNOSTIC_REPORT](../results/quaternion_merging/DIAGNOSTIC_REPORT.md), [REPORT](../results/quaternion_merging/REPORT.md) |
| rgb_pilot | [PROTOCOL](../results/rgb_pilot/PROTOCOL.md), [REPORT](../results/rgb_pilot/REPORT.md) |
| routing_bn_diagnostic | [REPORT](../results/routing_bn_diagnostic/REPORT.md) |
| ssl_compression_pilot | [REPORT](../results/ssl_compression_pilot/REPORT.md) |
| structure_control | [REPORT](../results/structure_control/REPORT.md) |
| structure_cost_study | [REPORT](../results/structure_cost_study/REPORT.md) |
| tail_compensation_pilot | [REPORT](../results/tail_compensation_pilot/REPORT.md) |
| trainable_maps_pilot | [REPORT](../results/trainable_maps_pilot/REPORT.md) |

## Method and literature context

- [Structural hypotheses and compensation equations](STRUCTURE_HYPOTHESIS.md)
- [Input encoding review](INPUT_ENCODING_REVIEW.md)
- [Learned-grouping novelty audit](LEARNED_GROUPING_NOVELTY_AUDIT.md)
- [Literature index](../LITERATURE.md) and [full prior-art audit](archive/LITERATURE_2026-11-07.md)

The frozen historical summarizer can regenerate an older narrative at this
path; it is not a live-state monitor. Previous narrative versions are in Git
history. Do not use their old queue/launcher descriptions as current instructions.
