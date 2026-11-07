"""Build a measured-status document only after the three-seed pilot is verified."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def run():
    training = read("results/cnn_multiseed/summary.json")
    compression = read("results/cnn_compression/summary.json")
    verification = read("results/cnn_compression/verification.json")
    if training["status"] != "completed" or compression["status"] != "completed" or verification["status"] != "passed":
        raise ValueError("Do not report an unverified or incomplete study as completed")
    groups = read("results/cnn_compression/aggregated.json")
    diagnostics = read("results/cnn_compression/block_analysis/summary.json")
    lines = ["# Research status: completed three-seed CNN and physical-compression pilot", "",
             "Source of scope: PROJECT.md. All cited keys refer to references.bib. Reports: results/cnn_multiseed/REPORT.md, results/cnn_compression/block_analysis/REPORT.md, results/cnn_compression/REPORT.md.", "",
             "## Hypotheses", "",
             "H1-H3 concern whole-quaternion representation redundancy, measurement choice and useful structural compression. No novel method, first-in-literature result or general quaternion performance advantage is established. This pilot does not directly compare grouped and scalar-derived pruning policies (H2).", "",
             "## Measured results: baseline quality", "", "| Seed | Quaternion test | Real test | Difference (pp) |", "| ---: | ---: | ---: | ---: |"]
    for run_record in training["runs"]:
        q, r = (next(r for r in run_record["models"] if r["kind"] == kind)["test"]["accuracy"] for kind in ("quaternion", "real"))
        lines.append(f"| {run_record['seed']} | {100*q:.2f}% | {100*r:.2f}% | {100*(q-r):+.2f} |")
    for kind in ("quaternion", "real"):
        value = training["statistics"][kind]
        lines.extend(["", f"{kind}: mean {100*value['mean']:.2f}%, sample SD {100*value['sample_sd']:.2f} pp across three training seeds."])
    lines.extend(["", "73,898 versus 72,954 real trainable parameters. Widths and compute differ. Same train/validation/calibration split and normalization; frozen 50-epoch protocol, changed seeds only. Epoch choice uses validation CE. All four new trainings finished before the new final-test evaluations. The official test set was used by earlier pilots and is not a new blind benchmark.", "",
                  "## Measured results: activation diagnostics", "", "| Seed | Block-3 individual low-impact removals | Maximum grouped cosine | Effective scalar rank / 128 | Similarity vs negative loss impact (Spearman) |", "| ---: | ---: | ---: | ---: | ---: |"])
    for record in diagnostics["records"]:
        if record["kind"] != "quaternion":
            continue
        block = record["blocks"][2]
        diag = block["trained"]
        lines.append(f"| {record['seed']} | {len(block['individual_low_impact_units'])}/32 | {diag['grouped_pairs']['maximum']:.4f} | {diag['rank']['effective_rank']:.2f} | {block['similarity_vs_negative_loss_impact_spearman']:.3f} |")
    lines.extend(["", "Individual low impact: <=0.01 validation CE increase and <=0.5 pp accuracy decrease, prespecified. Whole images are separated before image-half stability measurements. All blocks and both model families, including weak/negative predictors, are retained in the diagnostic report. Effective rank is not a count of removable units; independent seeds do not align neuron identities.", "",
                  "## Measured results: physical compression", "", "| Block-3 removal | Policy | Retained real parameters | Total parameter saving | Test mean +/- training-seed SD |", "| ---: | --- | ---: | ---: | ---: |"])
    for group in groups:
        params = group["real_trainable_parameters"]
        lines.append(f"| {100*group['removed_fraction']:.0f}% | {group['policy']} | {params} | {100*(1-params/73898):.2f}% | {100*group['mean_test_accuracy']:.2f} +/- {100*group['sample_sd_across_training_seed_means']:.2f}% |")
    lines.extend(["", "Random: ten removal repetitions within each of three training seeds, averaged within seed before training-seed SD. No fine-tuning, BN re-calibration, weight compensation or test-based policy/size selection. All manifests and validation measurements were frozen before compact-model test evaluation.", "",
                  "At each retained parameter count, the following paired effects describe this pilot only (activation-similarity minus reference test accuracy, in percentage points):", ""])
    for fraction in (0.25, 0.5):
        active = next(g for g in groups if g["policy"] == "activation_similarity" and g["removed_fraction"] == fraction)
        for policy in ("incoming_norm", "random"):
            reference = next(g for g in groups if g["policy"] == policy and g["removed_fraction"] == fraction)
            difference = 100*(np.array(active["training_seed_means"]) - np.array(reference["training_seed_means"]))
            lines.append(f"- {100*fraction:.0f}% removal vs {policy}: paired values {', '.join(f'{v:+.2f}' for v in difference)}; mean {difference.mean():+.2f}, sample SD {difference.std(ddof=1):.2f} pp.")
    decisions = []
    for fraction in (0.25, 0.5):
        active = next(g for g in groups if g["policy"] == "activation_similarity" and g["removed_fraction"] == fraction)
        random = next(g for g in groups if g["policy"] == "random" and g["removed_fraction"] == fraction)
        baseline = next(g for g in groups if g["policy"] == "baseline")
        decisions.append(f"At {100*fraction:.0f}% block-unit removal ({100*(1-active['real_trainable_parameters']/73898):.2f}% total-parameter saving), similarity gives {100*active['mean_test_accuracy']:.2f}% test accuracy, a {100*(baseline['mean_test_accuracy']-active['mean_test_accuracy']):.2f}-pp drop. Its mean paired difference from random is {100*(active['mean_test_accuracy']-random['mean_test_accuracy']):+.2f} pp; the three seed effects do not have a consistent sign.")
    lines.extend(["", "## Measured decision", "", *decisions,
                  "The fixed-coordinate similarity policy did not demonstrate an advantage over random removal. Mean performance against raw incoming norm is better, but reverses in seed 2028 at both sizes; raw norms also ignore BN scaling. Preserve this negative finding instead of claiming a pruning-method gain.",
                  "Physical compaction and a modest quality-size trade-off are demonstrated. Gate C's activation-policy advantage remains unconfirmed; Gate D is not satisfied by this pilot.", "", "## Verification", "",
                  f"Core PyTorch suite: 22 passed. Compact artifact verification: {len(verification['checks'])} checks passed. Each new baseline passes the existing 35-check artifact verifier. Physical checks include zero-removal equivalence, compact-vs-masked equivalence, actual state parameter/buffer counts, exact checkpoint reload and recomputation from saved test logits. Source/configuration hashes and full epoch-resume states are preserved.",
                  "Threshold erratum: the initial seed-2026 report counted 24/32 low-impact block-3 removals. Two exact 0.5-pp boundary cases were excluded by floating-point subtraction; inclusive tolerance 1e-12 gives 26/32. No measured loss/accuracy, ordering, threshold value or selected block changed. The original artifacts are preserved and corrected labels are explicit in the multi-seed diagnostic output.", "",
                  "## Interpretation and limits", "",
                  "Infrastructure Gate A passes. Gate B has multi-seed diagnostic evidence for individually dispensable channels, not proof of near-duplicate whole units or a special quaternion effect. A positive compression result requires an actual quality-size trade-off against controls; the tables retain every outcome. Gate D (paper novelty/substance) remains open.",
                  "Simple positive-coordinate centered similarity can be a weak predictor; negative coefficients must not be replaced by a tuned score. Individually dispensable channels can be harmful when removed together. Raw kernel norms ignore BN scaling, so a difference against this reference does not establish superiority over strong weight-based pruning. Policy context: Ayinde2019, Vecchi2020, Singh2026; no faithful reproduction of their complete methods is claimed.",
                  "This study covers one small architecture, one dataset, three training seeds and a single selected block, without a direct scalar-policy control, fine-tuning, matched pre-pruning accuracy, MAC/latency measurements or a systematic novelty review. It supports a limited empirical pilot, not a general method claim.", "",
                  "## Next research decision", "",
                  "At the close of the original pilot, the next questions were a fixed BN-aware weight reference and H2: whether grouped diagnostics predict whole-unit removal harm better than ordinary scalar correlations. Completed follow-ups are recorded below when available. Only a validated signal would justify more architectures or a separate, equally budgeted fine-tuning study. The original similarity policy did not earn a method-superiority claim."])
    bn_folder = ROOT / "results" / "bn_compression"
    if (bn_folder / "verification.json").exists() and json.loads((bn_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        comparison = json.loads((bn_folder / "comparison.json").read_text(encoding="utf-8"))
        bn_verify = json.loads((bn_folder / "verification.json").read_text(encoding="utf-8"))
        lines.extend(["", "## Completed BN-aware weight-reference follow-up", "",
                      "One fixed additional magnitude criterion was authorized and frozen before computing its outcomes. It uses the joint norm of all four output rows in the Conv+BN-folded expanded real Hamilton matrix. It excludes folded bias and downstream effects; no training or BN re-calibration. Reference results above are reused unchanged. Source of the folding equation: PyTorchFusion260 in references.bib.", "",
                      "| Block-unit removal | BN-aware test mean +/- seed SD | Random test mean | BN minus random (pp) |", "| ---: | ---: | ---: | ---: |"])
        bn_decisions = []
        for fraction in (.25, .5):
            bn = next(g for g in comparison["groups"] if g["removed_fraction"] == fraction and g["policy"] == "bn_folded_group_l2")
            random = next(g for g in comparison["groups"] if g["removed_fraction"] == fraction and g["policy"] == "random")
            effect = 100*(bn["test_accuracy_mean"] - random["test_accuracy_mean"])
            lines.append(f"| {100*fraction:.0f}% | {100*bn['test_accuracy_mean']:.2f} +/- {100*bn['test_accuracy_sample_sd_across_training_seed_means']:.2f}% | {100*random['test_accuracy_mean']:.2f}% | {effect:+.2f} |")
            bn_decisions.append(f"At {100*fraction:.0f}% block-unit removal, BN-aware magnitude reaches {100*bn['test_accuracy_mean']:.2f}% versus random {100*random['test_accuracy_mean']:.2f}% (mean paired BN-minus-random effect {effect:+.2f} pp). BN-aware is worse in each of the three training seeds at this size.")
        bn_decisions.extend(["At 25% removal the BN-aware reference is also worse than raw magnitude in all three seeds. At 50% it improves the mean over raw magnitude, but reverses in seed 2028. Accounting exactly for BN's multiplicative scaling did not produce a reliably better importance criterion.",
                             f"All 25 implementation tests and {len(bn_verify['checks'])} BN artifact checks passed, including independent official folding, exact original-weight slicing, policy reconstruction and recomputed test metrics. Existing numerical sources and all original comparison artifacts were preserved.",
                             "This follow-up does not show that BN never matters. It shows this particular folded-weight norm remains a weak predictor in the fixed pilot. No bias-inclusive or alternative score was tried after seeing the outcome. The activation policy still has no demonstrated advantage over random. Next remains the prespecified grouped-versus-scalar diagnostic (H2), rather than tuning the weight criterion to these test results."])
        lines.extend(["", *bn_decisions])
        bn_report = bn_folder / "REPORT.md"
        bn_marker = "\n## Verified measured interpretation\n"
        bn_report.write_text(bn_report.read_text(encoding="utf-8").split(bn_marker)[0] + bn_marker + "\n" + "\n\n".join(bn_decisions) + "\n", encoding="utf-8")
    diagnostic_folder = ROOT / "results" / "group_scalar"
    if (diagnostic_folder / "verification.json").exists() and json.loads((diagnostic_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        diagnostic = json.loads((diagnostic_folder / "summary.json").read_text(encoding="utf-8"))
        diagnostic_verify = json.loads((diagnostic_folder / "verification.json").read_text(encoding="utf-8"))
        primary = diagnostic["primary"]
        selected = [r for r in diagnostic["records"] if r["block"] == primary["block"]]
        group_key, scalar_key = "grouped_centered_cosine", "scalar_pearson_mean_external_nn"
        lines.extend(["", "## Completed grouped-versus-scalar diagnostic", "",
                      "One prespecified scalar reference: for each component, maximum signed Pearson to a channel OUTSIDE its whole quaternion unit; average four maxima. Grouped reference: maximum existing centered whole-unit cosine. Both predict the same single-whole-unit validation-ablation target using calibration scores. Primary block 3; blocks 1/2 secondary. No new model inference, test access or training.", "",
                      "| Seed | Grouped Spearman | Scalar Spearman | Group-minus-scalar |", "| ---: | ---: | ---: | ---: |"])
        for record in selected:
            lines.append(f"| {record['seed']} | {record['criteria'][group_key]['spearman_score_vs_negative_loss_harm']:.3f} | {record['criteria'][scalar_key]['spearman_score_vs_negative_loss_harm']:.3f} | {record['group_minus_scalar_spearman']:+.3f} |")
        scalar_wins = sum(r["group_minus_scalar_spearman"] < 0 for r in selected)
        grouped_wins = sum(r["group_minus_scalar_spearman"] > 0 for r in selected)
        diagnostic_decisions = [f"Scalar Pearson predicts individual loss harm better in {scalar_wins}/3 primary seeds; grouped cosine in {grouped_wins}/3. Mean paired grouped-minus-scalar Spearman difference {primary['mean_paired_spearman_difference']:+.3f}, sample SD {primary['sample_sd_paired_spearman_difference']:.3f}. The prespecified repeatable-grouped-advantage rule returns {primary['repeatable_grouped_advantage_under_prespecified_rule']}.",
                                "In the primary block, both rankings select a quarter of units whose mean separate individual-removal harm is lower than the uniform random expectation in all three seeds. This is a diagnostic signal, not simultaneous quarter-removal performance; the earlier greedy-pruning test accuracies belong to a different policy. Neither a weak positive Spearman nor individual dispensability establishes useful joint compression.",
                                "This particular grouped cosine has no demonstrated repeatable advantage over the fixed independent-scalar reference. H2 is not confirmed for these measurements. Close this candidate advantage rather than tune the score or select a favorable secondary block. This does not rule out all other measures of activation redundancy.",
                                "The user's broader structural-constraint hypothesis remains open. Each same-width quaternion connection uses four real coefficients to generate sixteen signed/tied real matrix entries. This factual parameter sharing does not prove that arbitrary alternative constraints would behave the same. Our controls do not isolate generic sharing from its specific pattern, width, compute or optimization. See docs/STRUCTURE_HYPOTHESIS.md; context: ZhangEtAl2021 in references.bib.",
                                f"All 28 implementation tests and {len(diagnostic_verify['checks'])} diagnostic artifact checks passed. Initial failed spatial-shape assertion and its correction are preserved in results/group_scalar/FAILED_CHECKS.md and failed_run_protocol.json. Frozen numerical inputs and earlier outcomes remain unchanged."]
        lines.extend(["", *diagnostic_decisions])
        diagnostic_report = diagnostic_folder / "REPORT.md"
        diagnostic_marker = "\n## Verified interpretation and decision\n"
        diagnostic_report.write_text(diagnostic_report.read_text(encoding="utf-8").split(diagnostic_marker)[0] + diagnostic_marker + "\n" + "\n\n".join(diagnostic_decisions) + "\n", encoding="utf-8")
    routing_folder = ROOT / "results" / "learned_grouping_pilot"
    if (routing_folder / "verification.json").exists() and json.loads((routing_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        routing = json.loads((routing_folder / "summary.json").read_text(encoding="utf-8"))
        membership = json.loads((routing_folder / "membership_analysis.json").read_text(encoding="utf-8"))
        learned = next(r for r in routing["models"] if r["mode"] == "learned")
        lines.extend(["", "## Completed learned-routing pilot (one seed)", "",
                      "Resume/final artifact review: 2026-11-03. Both new trainings completed 50 epochs before this review. Identity is reused from the equivalent original checkpoint. This is one seed (2026), not three replicated routing trainings.", "",
                      "| Routing condition | Selected validation | Final test |", "| --- | ---: | ---: |",
                      f"| identity (reused) | {100*routing['identity_reference']['selected_validation']['accuracy']:.2f}% | {100*routing['identity_reference']['test']['accuracy']:.2f}% |"])
        for record in routing["models"]:
            lines.append(f"| {record['mode']} | {100*record['selected_validation']['accuracy']:.2f}% | {100*record['test']['accuracy']:.2f}% |")
        hardening = learned["hardening"]
        lines.extend(["", f"Epoch-40 soft-to-hard validation change: {100*hardening['soft_validation']['accuracy']:.2f}% to {100*hardening['immediate_hard_validation']['accuracy']:.2f}%, a {100*hardening['accuracy_drop']:.2f}-pp drop. The ten-epoch hard phase recovered part of this loss; the selected checkpoint still failed the prespecified validation gate.",
                      f"Export changes {membership['changed_positions']}/32 scalar output positions and creates {membership['new_unordered_member_groups']}/8 new unordered member groups. This is actual membership change, not only relabeling. It did not yield a quality gain in this pilot.",
                      "The soft assignment retained broad mixing (mean row entropy about 1.99 nats and about 30.4% mean mass on the eventual hard assignment). This is consistent with relying on a dense soft mixture rather than a near-discrete assignment. The immediate loss also includes changed downstream activation/BN distributions; this experiment does not isolate those effects. Ten late-schedule hard epochs and one temperature/init configuration do not rule out other training procedures.",
                      "The compression gate failed on validation before final test; the conditional stage records not_run_gate_failed, with no new compact models or compact-model test inference. Gate checks and serialized artifacts pass (64 routing checks; 7 conditional-stage checks). No result is claimed as a novel compression method. See results/learned_grouping_pilot/REPORT.md and docs/LEARNED_GROUPING_NOVELTY_AUDIT.md."])
    bn_intervention_folder = ROOT / "results" / "routing_bn_diagnostic"
    if (bn_intervention_folder / "verification.json").exists() and json.loads((bn_intervention_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        intervention = json.loads((bn_intervention_folder / "summary.json").read_text(encoding="utf-8"))
        lines.extend(["", "## Completed fixed-weight BN intervention at the routing boundary", "",
                      "Seed 2026, epoch 40 replay of the unchanged pilot. Both full soft/hard boundary states are now retained. The numerical epoch trace and boundary metrics match the original within 1e-8; unavailable original epoch-40 weights prevent a bitwise comparison. No optimizer updates after the boundary and no new official-test inference.", "",
                      "Refit only the four BNs downstream of the router, in forward order, on all 48,000 training images. No augmentation or dropout; global input moments with float64 accumulation and unbiased variance. Every model parameter, upstream statistics, counters and routing remain unchanged by refitting. Soft refitting controls for the procedure itself.", "",
                      "| Condition | Validation accuracy | Validation CE | Calibration accuracy |", "| --- | ---: | ---: | ---: |"])
        for name, record in intervention["conditions"].items():
            validation, calibration = record["evaluation"]["validation"], record["evaluation"]["calibration"]
            lines.append(f"| {name} | {100*validation['accuracy']:.2f}% | {validation['cross_entropy']:.6f} | {100*calibration['accuracy']:.2f}% |")
        effect = intervention["contrasts"]["validation"]
        lines.extend(["", f"Hard validation recovery from BN refit: {effect['hard_refit_accuracy_gain_pp']:+.2f} pp; soft-control change: {effect['soft_refit_accuracy_gain_pp']:+.2f} pp. Original soft-minus-hard gap: {effect['original_soft_minus_hard_accuracy_pp']:.2f} pp; gap with both refitted: {effect['refitted_soft_minus_hard_accuracy_pp']:.2f} pp.",
                      "These values describe what one fixed-weight BN-statistic intervention repairs. Residual loss is not explained away by this procedure; it does not establish an unavoidable limitation of discrete grouping or isolate all remaining causes. One seed, architecture, boundary and estimator; no compression savings or novel-method claim. See results/routing_bn_diagnostic/REPORT.md and configs/routing_bn_diagnostic.json."])
        bn_decision = [
            f"BN-only refitting repairs {effect['hard_refit_accuracy_gain_pp']:.2f} pp of hard validation accuracy, while the soft control changes by {effect['soft_refit_accuracy_gain_pp']:+.2f} pp. The remaining refitted soft-minus-hard gap is {effect['refitted_soft_minus_hard_accuracy_pp']:.2f} pp. Thus BN-buffer refitting is helpful but insufficient to restore soft-model quality in this intervention.",
            "The secondary held-out calibration set agrees in direction: hard accuracy rises from 55.20% to 66.30%; refitted soft accuracy is 80.30%, leaving a 14.00-pp gap. No official test inference or weight updates were used in this diagnostic. Five implementation tests and 111 independent artifact checks pass; no scientific or implementation check failed in this stage.",
            "Close the hypothesis that this particular downstream-BN refit alone restores the original soft-model quality. Do not call the remaining loss a proven inherent limitation or assign a uniquely causal percentage to BN. The next candidate experiment is a prespecified discrete-forward grouping-training comparison against paired fixed assignments, with extra variables/compute reported and equivalent final evaluation/refitting. Do not silently reopen the original failed compression gate. No further training is started by this diagnostic."
        ]
        lines.extend(["", *bn_decision])
        intervention_report = bn_intervention_folder / "REPORT.md"
        intervention_marker = "\n## Verified measured interpretation and decision\n"
        intervention_report.write_text(intervention_report.read_text(encoding="utf-8").split(intervention_marker)[0] + intervention_marker + "\n" + "\n\n".join(bn_decision) + "\n", encoding="utf-8")
    discrete_folder = ROOT / "results" / "hard_forward_grouping"
    if (discrete_folder / "verification.json").exists() and json.loads((discrete_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        discrete = json.loads((discrete_folder / "summary.json").read_text(encoding="utf-8"))
        checks = json.loads((discrete_folder / "verification.json").read_text(encoding="utf-8"))
        membership, effect, gate = discrete["membership"], discrete["effect"], discrete["replication_gate"]
        lines.extend(["", "## Completed discrete-forward grouping pilot", "",
                      "One new seed-2026 50-epoch training with exact discrete assignment in every forward, Sinkhorn straight-through surrogate in backward, 40 adaptive epochs and 10 with assignment fixed. The paired fixed-random 50-epoch control is reused. Selection is minimum original-BN validation CE at epochs 40-50 for both; identical training-only downstream-BN refit after selection. No new official-test inference.", "",
                      "| Condition | Validation accuracy | Calibration accuracy |", "| --- | ---: | ---: |"])
        for name, record in discrete["conditions"].items():
            lines.append(f"| {name} | {100*record['evaluation']['validation']['accuracy']:.2f}% | {100*record['evaluation']['calibration']['accuracy']:.2f}% |")
        discrete_decision = [
            f"The selected exported assignment changes {membership['changed_positions']}/32 scalar positions and creates {membership['new_unordered_groups']}/8 new unordered groups. Its BN-refitted learned-minus-fixed effect is {effect['validation_accuracy_difference_pp']:+.2f} pp on validation and {effect['calibration_accuracy_difference_pp']:+.2f} pp on calibration.",
            f"The prespecified replication gate returns {gate['promising_for_replication']}: calibration gain {gate['calibration_gain']}, validation quality {gate['validation_quality']}, actual membership change {gate['membership_changed']}. A positive gate only motivates replication; a negative gate preserves this unsuccessful configuration. Neither is a general method result or a novelty claim.",
            "In this completed pilot, the primary calibration difference is one additional correct image out of 1,000, while validation has three fewer correct images. CE is worse for learned grouping on both held-out splits, before and after BN refitting. This does not demonstrate a useful learned-grouping advantage or establish equivalence; preserve the near-tied accuracy and unfavorable CE rather than claim a gain. The gate does not justify automatically launching replication or compression of this configuration.",
            "The freeze has zero immediate prediction shift because routing was already discrete. This verifies the intended mechanism; it is not evidence of a learned-grouping advantage. Training adds 1,024 logit parameters and matching/surrogate compute; both fixed-index exports have 73,898 real parameters with no removed backbone channels.",
            f"Six implementation tests and {checks['check_count']} independent artifact checks pass. Numerical sources and paired reference artifacts are preserved. Prior context: Bengio2013STE, Mena2018, Wang2023PSTE, Wang2019FLGC and Zhang2021StructuredShuffle. See results/hard_forward_grouping/REPORT.md and configs/hard_forward_grouping.json."
        ]
        lines.extend(["", *discrete_decision])
        discrete_report = discrete_folder / "REPORT.md"
        discrete_marker = "\n## Verified interpretation and decision\n"
        discrete_report.write_text(discrete_report.read_text(encoding="utf-8").split(discrete_marker)[0] + discrete_marker + "\n" + "\n\n".join(discrete_decision) + "\n", encoding="utf-8")
    merging_folder = ROOT / "results" / "quaternion_merging"
    if (merging_folder / "verification.json").exists() and json.loads((merging_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        merging = json.loads((merging_folder / "summary.json").read_text(encoding="utf-8"))
        merge_checks = json.loads((merging_folder / "verification.json").read_text(encoding="utf-8"))
        lines.extend(["", "## Completed affine-reconstruction and last-block merging pilot", "",
                      "Three existing quaternion CNN seeds, block 3, no new CNN training. Fit maps and choose eight disjoint pairs using 500 even calibration images; independently check combined reconstruction on 500 odd images. Right-Hamilton affine 4+4, diagonal affine 4+4, full-real affine 16+4 calibration coefficients. No labels in fitting. Both primary families passed the held-out gate in all three seeds; the diagonal family remains a coefficient-matched diagnostic."])
        if merging["status"] == "completed":
            merge_aggregates = json.loads((merging_folder / "aggregated.json").read_text(encoding="utf-8"))
            merge_effects = json.loads((merging_folder / "paired_effects.json").read_text(encoding="utf-8"))
            lines.extend(["", "Physically remove 8/32 final-block units and fold compensation into the existing GAP/real-linear readout. Every compact condition has 64,298 real parameters (12.99% fewer than 73,898), 838 buffer elements and a 269,712-byte standardized state file. No extra mapping layer remains. All manifests precede new validation inference; no BN refit, fine-tuning or official-test inference.", "",
                          "| Policy | Validation mean ± seed SD | Mean CE |", "| --- | ---: | ---: |"])
            for item in merge_aggregates:
                lines.append(f"| {item['policy']} | {100*item['mean_accuracy']:.2f}% ± {100*item['sample_sd_accuracy']:.2f} pp | {item['mean_cross_entropy']:.6f} |")
            merging_decisions = [
                "Right-Hamilton compensation improves accuracy over pruning the exact same victims by +2.10, +0.90 and +1.40 pp (mean +1.47 pp), but its mean accuracy is 0.13 pp below the same-victim mean-only correction. Full-real compensation improves over its own mean-only correction by 0.00, -0.20 and +0.70 pp. Thus a repeatable accuracy advantage beyond the cheap mean-only control is not demonstrated.",
                "Validation accuracy with right-Hamilton/full-real merging is 76.63%/76.67%, versus 77.93% unpruned and 76.37% random-pruned (ten random repetitions averaged within each training seed). Paired right-Hamilton-minus-random differences are +0.37, +0.13 and +0.30 pp. These are small three-seed pilot effects, not a general superiority claim.",
                "Retain the favorable CE evidence too: full-real merging has mean CE 0.654740, versus 0.672888 for its mean-only control and 0.687761 for random pruning; the unpruned mean is 0.637730. Right-Hamilton merging has CE 0.667735 versus 0.676093 for its mean-only control. Lower held-out logit MSE and better CE do not automatically imply better argmax accuracy. No special quaternion advantage is established.",
                f"Nine meaningful implementation tests and {merge_checks['check_count']} independent artifact checks pass. Maximum cached-to-physical logit error is below 5e-6 (frozen tolerance 1e-4). Checks cover fit-only selection, affine normal equations, whole-image held-out splits, no hidden map parameters, exact kernel/BN slicing, compensation formulas, actual parameter/buffer/file counts and checkpoint reload predictions. All original numerical sources and model/data artifacts are preserved.",
                "This is a bounded pilot at one layer and removal level. Map fitting/merging is established background (Kim2020, Ebid2025); quaternion-specific pruning also has prior art (Vecchi2020, Singh2026). The new evidence supports an empirical distinction between reconstruction/CE and classification accuracy, not a novel-method or paper-readiness claim. See results/quaternion_merging/DIAGNOSTIC_REPORT.md and REPORT.md."
            ]
            lines.extend(["", *merging_decisions])
            merge_report = merging_folder / "REPORT.md"
            merge_marker = "\n## Verified measured interpretation\n"
            merge_report.write_text(merge_report.read_text(encoding="utf-8").split(merge_marker)[0] + merge_marker + "\n" + "\n\n".join(merging_decisions) + "\n", encoding="utf-8")
        else:
            lines.extend(["", "Held-out reconstruction gate failed; physical comparison was skipped and no new validation/test inference or training was performed."])
    internal_folder = ROOT / "results" / "internal_merging"
    if (internal_folder / "verification.json").exists() and json.loads((internal_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        internal = json.loads((internal_folder / "summary.json").read_text(encoding="utf-8"))
        internal_checks = json.loads((internal_folder / "verification.json").read_text(encoding="utf-8"))
        internal_aggregates = json.loads((internal_folder / "aggregated.json").read_text(encoding="utf-8"))
        by_policy = {row["policy"]: row for row in internal_aggregates}
        effects = {}
        for name, left, right in (("quaternion_minus_projected", "quaternion_exact", "full_projected"),
                                  ("quaternion_minus_random", "quaternion_exact", "random"),
                                  ("quaternion_minus_paired_prune", "quaternion_exact", "paired_pruned"),
                                  ("full_oracle_minus_projected", "full_substitution", "full_projected")):
            differences = [100*(a["accuracy"]-b["accuracy"]) for a, b in zip(by_policy[left]["seed_values"], by_policy[right]["seed_values"])]
            effects[name] = {"seed_differences_pp": differences, "mean_pp": float(np.mean(differences)), "sample_sd_pp": float(np.std(differences, ddof=1))}
        (internal_folder / "paired_effects.json").write_text(json.dumps(effects, indent=2)+"\n", encoding="utf-8")
        q_vs_projected = effects["quaternion_minus_projected"]
        q_vs_random = effects["quaternion_minus_random"]
        diagnostic_values = list(internal["diagnostics"].values())
        fractions = [row["discarded_full_energy_fraction"] for row in diagnostic_values]
        penalties = [100*(row["consumer_check"]["full_projected_over_oracle_error_ratio"]-1) for row in diagnostic_values]
        lines.extend(["", "## Completed structure-preserving intermediate merging pilot", "",
                      "Block 2, three existing CIFAR CNN seeds, one removal level, no new training/BN refit/test inference. Fit zero-offset maps on 500 calibration images and all 64 spatial positions; check on 500 separate calibration images. Use common full-fit-selected disjoint pairs for both methods, paired pruning and functional oracles. Freeze all choices before new validation inference.", "",
                      "Ordinary right-Hamilton associativity permits folding q_v=q_k*a into the next convolution as w_k <- w_k+a*w_v. A general matrix requires projecting R(w_v)A into the four-dimensional admissible block space. The statement concerns linear maps between post-activation feature maps and the next convolution, not commuting through ReLU. Affine offsets with zero padding are outside this pilot. No new theorem/method is claimed.", "",
                      "| Policy | Real parameters | Validation mean ± seed SD | Mean CE |", "| --- | ---: | ---: | ---: |"])
        for item in internal_aggregates:
            params = str(item["real_parameters"]) if item["real_parameters"] is not None else "functional oracle"
            lines.append(f"| {item['policy']} | {params} | {100*item['mean_accuracy']:.2f}% ± {100*item['sample_sd_accuracy']:.2f} pp | {item['mean_cross_entropy']:.6f} |")
        internal_decisions = [
            f"Projection discards {100*min(fractions):.2f}-{100*max(fractions):.2f}% of the squared Frobenius energy of ADDED compensation blocks R(w_v)A, not of the complete layer. On held-out features this raises the consumer's pre-BN reconstruction MSE by {min(penalties):.2f}-{max(penalties):.2f}% relative to full substitution in all three seeds.",
            f"Direct admissible fitting exceeds full-fit-then-project accuracy by {', '.join(f'{x:+.2f}' for x in q_vs_projected['seed_differences_pp'])} pp; mean {q_vs_projected['mean_pp']:+.2f} pp, sample SD {q_vs_projected['sample_sd_pp']:.2f} pp. CE and consumer MSE also favor direct quaternion fitting in every seed. Against ten random removals averaged within each seed, paired gains are {', '.join(f'{x:+.2f}' for x in q_vs_random['seed_differences_pp'])} pp, mean {q_vs_random['mean_pp']:+.2f} pp.",
            "All actual compact methods have 66,954 real parameters (9.40% fewer), 870 buffer elements and 280,464-byte standardized state files. Direct quaternion fitting gives 74.73% mean validation accuracy versus 77.93% before compression, so meaningful quality loss remains. The 76.30% full-substitution result is a functional comparison and is not a compact quaternion model at this budget.",
            f"Five meaningful implementation tests and {internal_checks['check_count']} independent artifact checks pass. Exact-Q composition and graph surgery agree within 1e-4 (observed consumer maximum below 4e-6). Verification covers all pair fits/normal equations, fit-only common selection, coefficient energy/projection, tensor slicing, actual budgets, source cache alignment and checkpoint/oracle predictions.",
            "This supplies a repeatable pilot signal for the cost of enforcing an intermediate layer's structure and for fitting within the constraint from the start. It does not establish a new algorithm or broad quaternion superiority. One architecture/dataset/block/removal level and a common selection favoring the full fit limit interpretation. A second compression level and an independent architecture/dataset are needed before positioning a paper; a focused literature review is still required. Prior context: Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002."
        ]
        weighted_audit_path = internal_folder / "data_weighted_projection_audit.json"
        if weighted_audit_path.exists():
            weighted_audit = json.loads(weighted_audit_path.read_text(encoding="utf-8"))
            if weighted_audit["status"] != "passed":
                raise ValueError("Do not omit a failed data-weighted projection audit")
            maximum_error = max(row["maximum_weighted_vs_direct_matrix_error"] for row in weighted_audit["records"])
            internal_decisions.append(
                f"Important POST-HOC analytic control: projecting the full fit using the keeper activation second moment reproduces the direct quaternion least-squares map across all 720 directed pairs (maximum matrix difference {maximum_error:.3g}). No new inference or validation selection was performed. Thus the +1.10-pp comparison is against unweighted Frobenius projection; the direct-Q fit is equivalent to standard data-weighted projection in the admissible subspace. It is not a new algorithm or evidence of an intrinsically quaternion-specific optimization benefit. The measured cost of restricting a full map and the need to retain actual model structure remain valid. The article hypothesis must concern that structural trade-off, with stronger controls, rather than claim novelty from direct fitting alone. See data_weighted_projection_audit.json and scripts/audit_internal_projection.py."
            )
        lines.extend(["", *internal_decisions])
        internal_report = internal_folder / "REPORT.md"
        internal_marker = "\n## Verified empirical interpretation\n"
        internal_report.write_text(internal_report.read_text(encoding="utf-8").split(internal_marker)[0] + internal_marker + "\n" + "\n\n".join(internal_decisions) + "\n", encoding="utf-8")
    replication_folder = ROOT / "results" / "internal_replication"
    if (replication_folder / "verification.json").exists() and json.loads((replication_folder / "verification.json").read_text(encoding="utf-8"))["status"] == "passed":
        replication = json.loads((replication_folder / "summary.json").read_text(encoding="utf-8"))
        replication_checks = json.loads((replication_folder / "verification.json").read_text(encoding="utf-8"))
        replication_aggregates = json.loads((replication_folder / "aggregated.json").read_text(encoding="utf-8"))
        replication_effects = json.loads((replication_folder / "paired_effects.json").read_text(encoding="utf-8"))
        lines.extend(["", "## Completed extension: two CNN depths and two compression levels", "",
                      "Three NEW CNN8 baselines completed 50 epochs each, using the original frozen trainer, split, normalization, optimizer and augmentation. Eight convolutions/four blocks, widths 8/16/24/32, 118,058 real parameters. Three existing CNN6 baselines are reused (73,898 parameters). Architectures differ in depth, widths, parameter count and compute; compare compression policies within each model. This covers two plain sequential CNNs, not independent architecture families.", "",
                      "Remove 4/16 or 8/16 block-2 quaternion units; no compression updates, BN refit or new official-test inference. Use 500 even calibration images for zero-offset fits and nested common disjoint pair selection, 500 odd images for feature/consumer checks. All compressed choices precede their new validation inference. The original CNN6/four-removal cell remains exploratory/reused. Twelve compression cells share six trained models and one dataset; they are not twelve independent training replications. Validation also selected the baseline checkpoint, so it is not a new blind benchmark.", "",
                      "| Architecture | Removed | Compact real parameters | Baseline | Direct Q | Unweighted projection | Data-weighted projection | Random | Full functional oracle |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        extension_decisions=[]
        for architecture in ("cnn6", "cnn8"):
            for removed in (4, 8):
                lookup={r["policy"]:r for r in replication_aggregates if r["architecture"]==architecture and r["removed"]==removed}
                values=" | ".join(f"{100*lookup[policy]['mean_accuracy']:.2f}%" for policy in ("baseline", "quaternion_exact", "full_projected", "weighted_projected", "random", "full_substitution"))
                lines.append(f"| {architecture} | {removed} | {lookup['quaternion_exact']['real_parameters']} | {values} |")
                for reference in ("full_projected", "random"):
                    effect=next(r for r in replication_effects if (r["architecture"],r["removed"],r["left"],r["right"])==(architecture,removed,"quaternion_exact",reference))
                    signs=", ".join(f"{value:+.2f}" for value in effect["seed_differences_pp"])
                    extension_decisions.append(f"{architecture.upper()}, {removed}/16 removal: direct-Q minus {reference} accuracy is {signs} pp across seeds (mean {effect['mean_pp']:+.2f}, seed SD {effect['sample_sd_pp']:.2f} pp); mean CE difference {effect['mean_ce_difference']:+.6f}. Negative effects and inconsistent signs are retained.")
        diagnostics=replication["diagnostics"]
        fractions=[100*r["discarded_full_energy_fraction"] for r in diagnostics]
        penalties=[100*(r["consumer_check"]["full_projected_over_oracle_error_ratio"]-1) for r in diagnostics]
        direct_penalties=[100*(r["quaternion_over_oracle_error_ratio"]-1) for r in diagnostics]
        worse=sum(value>0 for value in penalties)
        map_error=max(r["weighted_map_max_error"] for r in diagnostics)
        logit_error=max(r[f"{split}_weighted_logit_error"] for r in diagnostics for split in ("check_calibration","validation"))
        extension_decisions.extend([
            f"Across the twelve cells, unweighted projection discards {min(fractions):.2f}-{max(fractions):.2f}% of the squared energy of ADDED compensation blocks. Its pre-BN consumer MSE change relative to full substitution ranges from {min(penalties):+.2f}% to {max(penalties):+.2f}%, with an increase in {worse}/12 cells. These diagnostics have different denominators and are not classification-accuracy effects.",
            f"Direct-Q (equivalently, data-weighted) fitting has an observed consumer-MSE change of {min(direct_penalties):+.2f}% to {max(direct_penalties):+.2f}% relative to full substitution, increasing it in {sum(value>0 for value in direct_penalties)}/12 cells. These are measured effects of the specified feature-fit procedure, not an optimality lower bound for every structure-preserving compensation. The full functional reference is not a task-accuracy upper bound.",
            f"The strong data-moment control matches direct-Q maps within {map_error:.3g} and saved predictions within {logit_error:.3g}. Direct constrained fitting and data-weighted projection are equivalent here. A gain against unweighted projection cannot establish a new algorithm or a unique quaternion optimization benefit.",
            "Functional full-map substitution uses the original graph and is not a compact quaternion architecture at the reported parameter budget. All physical policies at a given architecture/removal level have identical parameter and buffer budgets. A quarter/half of block-2 units is not a quarter/half of total network parameters; no masked-sparsity or latency claim is made.",
            f"Five targeted implementation tests and {replication_checks['check_count']} artifact checks pass. Checks replay every compact checkpoint and functional oracle, recompute held-out predictions/metrics, verify fit-only selection and least-squares equations, independently expand/fold Hamilton blocks, and check all three 50-epoch training histories, first updates, checkpoint selection and resume states. One root launcher remains: WZNOW_BADANIA.cmd.",
            "This extends the empirical structural trade-off to a second compression level and CNN depth. It does not establish a novel method, general quaternion superiority or paper readiness. One dataset, two related architectures, three seeds and shared validation limit generalization. Focused prior-art positioning (Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002) and broader validation remain necessary. See results/internal_replication/REPORT.md, comparison.png and structural_cost.png."
        ])
        lines.extend(["", "Accuracy values are seed means; random repetitions are first averaged within training seed. Full seed SDs, CE and all paired differences appear in the study report.", "", *extension_decisions])
        extension_report=replication_folder / "REPORT.md"
        extension_marker="\n## Verified empirical interpretation\n"
        extension_report.write_text(extension_report.read_text(encoding="utf-8").split(extension_marker)[0]+extension_marker+"\n"+"\n\n".join(extension_decisions)+"\n",encoding="utf-8")
    multisource_folder = ROOT / "results" / "multisource_compensation"
    if (multisource_folder / "summary.json").exists():
        multisource = json.loads((multisource_folder / "summary.json").read_text(encoding="utf-8"))
        lookup = {(r["architecture"], r["removed"], r["policy"]): r for r in multisource["aggregates"]}
        lines.extend(["", "## Completed multi-source compensation pilot (Hamilton-restricted vs unconstrained, real control)", "",
                      "Same six quaternion models and frozen victims as the extension above, plus the three existing real CNN6 baselines. Each removed unit is reconstructed from ALL kept units by ridge regression (alpha fixed a priori at 1e-3) on 500 even calibration images and folded into the next convolution. Unconstrained multi-source ridge folding is prior art (GRAIL2026); only the right-Hamilton restriction q_v ~= sum_k q_k*a_k, which keeps a quaternion consumer, is under study. No fine-tuning, BN refit or official-test inference.", "",
                      "| Architecture | Removed | Real parameters | Baseline | Random | Pairwise fold | Multi-source (random victims) | Multi-source projected | Multi-source Hamilton | Unconstrained oracle |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for architecture, removed, pairwise, multi_random, projected, multi in (("cnn6", 4, "quaternion_exact", "multi_q_random", "multi_full_projected", "multi_q"), ("cnn6", 8, "quaternion_exact", "multi_q_random", "multi_full_projected", "multi_q"),
                                                                             ("cnn8", 4, "quaternion_exact", "multi_q_random", "multi_full_projected", "multi_q"), ("cnn8", 8, "quaternion_exact", "multi_q_random", "multi_full_projected", "multi_q"),
                                                                             ("real_cnn6", 8, "pair_merge", "multi_real_random", None, "multi_real"), ("real_cnn6", 16, "pair_merge", "multi_real_random", None, "multi_real")):
            cells = [f"{100*lookup[(architecture, removed, policy)]['validation']['mean_accuracy']:.2f}%" if policy else "n/a" for policy in ("baseline", "random", pairwise, multi_random, projected, multi)]
            oracle = lookup.get((architecture, removed, "multi_full_substitution"))
            cells.append(f"{100*oracle['validation']['mean_accuracy']:.2f}%" if oracle else "n/a")
            lines.append(f"| {architecture} | {removed} | {lookup[(architecture, removed, multi)]['real_parameters']} | " + " | ".join(cells) + " |")
        effect_lines = []
        for item in multisource["effects"]:
            if (item["left"], item["right"]) in (("multi_q", "quaternion_exact"), ("multi_q", "multi_full_projected"), ("multi_full_substitution", "multi_q"), ("multi_real", "pair_merge")):
                effect_lines.append(f"{item['architecture']}, {item['removed']} removed: {item['left']} minus {item['right']} is {', '.join(f'{v:+.2f}' for v in item['validation_seed_pp'])} pp (mean {item['validation_mean_pp']:+.2f}; {item['positive_seeds']}/3 seeds positive).")
        lines.extend(["", "Validation seed means; random rows averaged within seed first. Full SDs, CE, check-half results and the matched-savings table are in results/multisource_compensation/REPORT.md.", "", *effect_lines, "",
                      f"Exact-fold checks: {multisource['equivalence']['count']} physical compact models equal their functional substitution within {multisource['equivalence']['maximum']:.2e}. Baseline, random and pairwise rows reproduce the extension above exactly.",
                      "Measured pattern (pre-fine-tuning): fitting all sources inside the Hamilton family recovers most of the gap to the unconstrained functional oracle, while unweighted projection of the unconstrained fit is consistently worse. The projected comparison remains a weak reference (the weighted equivalent is not run for the ridge multi-source fit). At matched parameter savings in CNN6, multi-source Hamilton and multi-source real compensation lose similar accuracy from their own baselines; uncompensated whole-unit removal is more damaging in the quaternion model. These are pilot observations on one block, small CNNs and CIFAR-10, without fine-tuning; no novelty or general superiority claim. Next: equal-budget fine-tuning of all compact policies."])
    finetune_folder = ROOT / "results" / "finetune_compression"
    if (finetune_folder / "summary.json").exists():
        tuned = json.loads((finetune_folder / "summary.json").read_text(encoding="utf-8"))
        after = {(r["architecture"], r["removed"], r["policy"]): r for r in tuned["aggregates"]}
        before = {(r["architecture"], r["removed"], r["policy"]): r for r in tuned["before_aggregates"]}
        config = tuned["protocol"]["config"]
        lines.extend(["", "## Completed equal-budget fine-tuning of the compact policies", "",
                      f"{tuned['run_count']} runs: every compact model of the multi-source study (first {config['random_repetitions_finetuned']} random victim sets) and the uncompressed baseline, {config['epochs']} epochs, AdamW LR {config['learning_rate']} cosine to {config['minimum_learning_rate']}, identical batch/augmentation stream per training seed. Final-epoch validation accuracy, no checkpoint selection, no official-test inference.", "",
                      "| Architecture | Removed | Policy | Before | After fine-tuning |", "| --- | ---: | --- | ---: | ---: |"])
        for key in sorted(after):
            if key[2] in ("baseline", "random", "paired_pruned", "quaternion_exact", "pair_merge", "multi_q", "multi_real", "multi_full_projected"):
                lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {100*before[key]['validation']['mean_accuracy']:.2f}% | {100*after[key]['validation']['mean_accuracy']:.2f}% ± {100*after[key]['validation']['sample_sd_accuracy']:.2f} pp |")
        lines.append("")
        for item in tuned["effects"]:
            if (item["left"], item["right"]) in (("multi_q", "random"), ("multi_q", "quaternion_exact"), ("multi_q", "multi_full_projected"), ("multi_real", "random")):
                lines.append(f"After fine-tuning, {item['architecture']}, {item['removed']} removed: {item['left']} minus {item['right']} is {', '.join(f'{v:+.2f}' for v in item['validation_seed_pp'])} pp (mean {item['validation_mean_pp']:+.2f}; {item['positive_seeds']}/3 seeds positive).")
        lines.extend(["", "Three epochs at one fixed schedule; longer or tuned fine-tuning could change rankings. See results/finetune_compression/REPORT.md for all policies, check-half values and the matched-savings table."])
    eurosat_folder = ROOT / "results" / "eurosat_compression"
    if (eurosat_folder / "summary.json").exists():
        eurosat = json.loads((eurosat_folder / "summary.json").read_text(encoding="utf-8"))
        baselines = json.loads((ROOT / "results" / "eurosat_cnn" / "summary.json").read_text(encoding="utf-8"))
        lines.extend(["", "## Completed second-dataset replication: EuroSAT four-band (NIR, R, G, B)", "",
                      "EuroSAT_MS (Zenodo 7711810), q = NIR + R i + G j + B k, no zero real part. Quaternion and real CNN6 baselines (3 seeds each) trained with the unchanged CIFAR trainer and budget; the 5,400-image test split is held out. All compression and fine-tuning rules were copied from the CIFAR studies before any EuroSAT compression outcome.", ""])
        for row in baselines["runs"]:
            models = json.loads((ROOT / row["path"] / "summary.json").read_text(encoding="utf-8"))["models"]
            lines.append(f"Seed {row['seed']}: " + ", ".join(f"{m['kind']} validation {100*m['selected_validation']['accuracy']:.2f}% ({m['real_trainable_parameters']} parameters, epoch {m['best_epoch']})" for m in models) + ".")
        for phase, label in (("phase_a", "without fine-tuning"), ("phase_b", "after equal-budget fine-tuning")):
            if eurosat.get(phase):
                lines.extend(["", f"EuroSAT, {label}:"])
                for item in eurosat[phase]["effects"]:
                    if (item["left"], item["right"]) in (("multi_q", "random"), ("multi_q", "quaternion_exact"), ("multi_q", "multi_full_projected"), ("multi_full_substitution", "multi_q"), ("multi_real", "random")):
                        lines.append(f"- {item['architecture']}, {item['removed']} removed: {item['left']} minus {item['right']} is {', '.join(f'{v:+.2f}' for v in item['validation_seed_pp'])} pp (mean {item['validation_mean_pp']:+.2f}; {item['positive_seeds']}/3 seeds positive).")
        lines.extend(["", "See results/eurosat_compression/REPORT.md for all policies and absolute accuracies."])
    curve_folder=ROOT / "results/compression_curve"
    if (curve_folder / "phase_a_verification.json").exists():
        curve_checks=json.loads((curve_folder / "phase_a_verification.json").read_text(encoding="utf-8"))
        if curve_checks["status"]!="passed":
            raise ValueError("Do not report a curve whose phase-A artifact audit failed")
        curve=json.loads((curve_folder / "summary.json").read_text(encoding="utf-8"))
        values={(r["architecture"],r["fraction"],r["policy"],r["stage"]):r for r in curve["aggregates"]}
        lines.extend(["", "## Quality-size curve: all-layer compression (pre-repair table below)", "",
                      f"Started 2026-11-04 at 15:11 CET, using the prepared Claude Code protocol. Current published summary status: {curve['status']}. The separate runner state is results/compression_curve_runner.json; progress between phase summaries is in results/compression_curve_run.log. EuroSAT runs automatically after all CIFAR phases. The older one-block and fine-tuning results above remain separate studies.", "",
                      "All six convolution outputs are narrowed by the same unit fraction, shallow to deep; compensation is fitted on the current partly compressed model. Three seeds, five fractions, paired random and greedy-victim controls, BN-folded magnitude reference. Main fractions/settings were copied from the handoff; the pre-existing disclosure that 0.125 was added after a seed-2026 diagnostic is retained in configs/compression_curve.json and docs/PLAN_IJCNN2027.md. This low-end point is not newly blind evidence.", "",
                      "| Architecture | Unit fraction removed at each site | Actual compact real parameters | Random mean | BN-magnitude mean | Greedy compensation mean ± seed SD |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for architecture in ("cnn6","real_cnn6"):
            for fraction in curve["protocol"]["config"]["fractions"]:
                main=values[(architecture,fraction,"greedy_comp","before")]
                random=values[(architecture,fraction,"random","before")]
                magnitude=values[(architecture,fraction,"bn_pruned","before")]
                accuracy=lambda row:100*row["validation"]["mean_accuracy"]
                lines.append(f"| {architecture} | {fraction} | {main['real_parameters']} | {accuracy(random):.2f}% | {accuracy(magnitude):.2f}% | {accuracy(main):.2f}% ± {100*main['validation']['sample_sd_accuracy']:.2f} pp |")
        lines.extend(["", "These are validation results WITHOUT fine-tuning, averaged across three seeds; random repetitions are averaged within seed. Half the units at every site leaves about 19k parameters (roughly a quarter of the baseline parameter count), not half the model's parameters. At strong compression, the accumulation of errors makes even compensated networks approach chance. A favorable comparison to pruning does not make these low absolute accuracies useful.", "",
                      f"The {len([r for r in curve['rows'] if r['stage']=='before'])} phase-A outcomes include {curve['diagnostics']['fold_checks']} site-fold comparisons (maximum discrepancy {curve['diagnostics']['max_fold_error']:.3g}). The independent phase-A audit checks the 246 saved compact checkpoints, their actual shapes/parameter/buffer budgets, matched victims, finite metrics, seed aggregation, disjoint image splits and protected input hashes. It does not claim independent prediction replay for all outcomes or verification of unfinished training.", "",
                      f"Current CIFAR summary status: {curve['status']}. The table above remains the pre-repair subset. Completed CIFAR/EuroSAT repair and scratch outcomes are synthesized in results/qnn_compression_review/REPORT.md; compare their budgets explicitly. Refinement and queued study status are reported below. No new official-test inference. Prior context remains GRAIL2026, Kim2020 and Asiaee2026CMR. Original report and figure: results/compression_curve/REPORT.md and quality_size_curve.png."])
    curve_queue_path=ROOT / "configs/compression_curve_queue.json"
    if curve_queue_path.exists():
        curve_queue=json.loads(curve_queue_path.read_text(encoding="utf-8"))
        state_path=ROOT / "results/compression_curve_queue.json"
        state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status":"registered"}
        lines.extend(["", "## Registered follow-up: denser quality-size curve", "",
                      f"Registered {curve_queue['registered_utc']}; current queue status: {state['status']}. After BOTH main CIFAR/EuroSAT curves finish, run fractions 0.375 and 0.625 on CIFAR, then EuroSAT. Each extension includes compression without fine-tuning, identical three-epoch fine-tuning controls and full-budget scratch training, with unchanged architectures, seeds, repetitions and data splits. Outputs: results/compression_curve_refinement and results/compression_curve_eurosat_refinement.", "",
                      "These points were selected after inspecting the coarse-grid outcomes. They are exploratory follow-up measurements using the same baseline models, not independent/blind replication. Preserve the original results, deduplicate baselines when combining the seven-point curves, and retain unfavorable seeds. No outcome is claimed from being registered or queued. Execution definition: configs/compression_curve_queue.json; live state: results/compression_curve_queue.json. Both root launchers now start/resume the whole queue."])
    activation_config=ROOT / "configs/activation_sensitivity.json"
    if activation_config.exists():
        act=json.loads(activation_config.read_text(encoding="utf-8"))
        state_path=ROOT / "results/activation_sensitivity_queue.json"
        state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status":"registered"}
        lines.extend(["", "## Registered activation-sensitivity study", "",
                      f"Registered {act['registered_utc']}; queue state: {state['status']}. After all existing coarse/refined curves: CIFAR quaternion CNN6 only, three seeds, six fresh 50-epoch baselines. Alternatives to the existing split ReLU are split LeakyReLU (slope 0.1) and coupled MagnitudeTanh (PoppelbaumSchwung2024Activations Eq.21). No extra trainable activation parameters. Compression at fractions 0.25/0.5 and three-epoch fine-tuning reuse the frozen implementations, with random victim streams aligned to the original ReLU reference.", "",
                      "Compare absolute accuracy/CE AND loss from each activation's own full baseline; retain all paired seed effects. This bounded pilot has no activation-specific scratch or EuroSAT runs, and no matched real coupled or split-Tanh control. MagnitudeTanh also changes boundedness/saturation, so it does not isolate coupling. No activation gain is inferred from registration, implementation tests or CPU smoke checks. Output: results/activation_sensitivity; live state: results/activation_sensitivity_queue.json."])
    repair_config=ROOT / "configs/pseudolabel_repair.json"
    if repair_config.exists():
        repair=json.loads(repair_config.read_text(encoding="utf-8"))
        state_path=ROOT / "results/pseudolabel_repair_queue.json"
        state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status":"registered"}
        lines.extend(["", "## Registered low-label POST-compression repair", "",
                      f"Queue state: {state['status']}. Appended after all curves and activation sensitivity. CIFAR ReLU quaternion CNN6, 25% unit removal at all six sites, three seeds; same victims with/without compensation. 54 fixed-budget repairs: 0/5/20 repair labels per class, supervised-only, current-student online pseudo-labels, and a frozen full-teacher pseudo-label control. Primary low-label budgets are 5/20; zero is a boundary control.", "",
                      "The original model was fully supervised and validation-selected. Label counts constrain repair weight updates only; labeled diagnostics and inherited normalization/BN statistics are outside this budget. This is not end-to-end few-shot or unsupervised training. Same-distribution CIFAR simulates unavailable repair annotations; no actual deployment or domain shift is tested. Fixed 1000 opportunities, snapshots 0/200/1000, confidence .95, fixed BN statistics/dropout, paired streams and disjoint U pool. No accepted target means no AdamW update, including no decay-only drift. Frozen teacher has an additional resource/inference cost. Confidence filtering has prior art (Sohn2020FixMatch); crop/flip-only views are not FixMatch reproduction. No gain follows from registration or readiness checks.", "",
                      "Config: configs/pseudolabel_repair.json; worker: scripts/run_pseudolabel_queue.py; output: results/pseudolabel_repair. Preserve unfavorable seeds, pseudo-label precision/coverage and actual optimizer updates. Earlier full-label three-epoch fine-tuning uses different BN/dropout settings and is not a matched label-efficiency control."])
    structure_config=ROOT / "configs/structure_control.json"
    if structure_config.exists():
        control=json.loads(structure_config.read_text(encoding="utf-8"))
        state_path=ROOT / "results/structure_control_queue.json"
        state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status":"registered"}
        lines.extend(["", "## Approved focus and bounded structural control", "",
                      "Primary question: compression of an existing QNN family, preservation of structure, actual size, quality/repair cost and failure regimes. A new method or intrinsic quaternion superiority is not assumed. Read results/qnn_compression_review/REPORT.md for the completed CIFAR/EuroSAT all-site evidence and the separately labeled single-site functional oracle. Keep unequal scratch/repair budgets and negative effects explicit. Pseudo-label repair remains supplementary.", "",
                      f"Structure-control queue state: {state['status']}. Appended after all existing jobs. Three new CIFAR CNN6 baselines with known 4x4 circulant weight-sharing (Ding2017CirCNN context), 50 epochs, fractions .25/.5; greedy compensation, same-victim pruning and BN-magnitude, before/after equal three-epoch repair. Reuse original Q/real results and add six missing Q same-victim pruning repairs. Same scalar widths, input information, seeded coefficient tensors and exact real parameter counts (73,898 / 42,034 / 19,098). Circulant groups are not quaternion units; ordinary expanded convolution is not a CirCNN FFT reproduction.", "",
                      "This alternative was chosen after existing Q/real outcomes. Report absolute accuracy/CE, own-baseline losses, baseline-quality gaps and all paired seeds. One matched pattern does not isolate a quaternion-specific mechanism or prove that any structure behaves alike. No random algebra, new multiplication rule, runtime claim, extra Circulant scratch/EuroSAT sweep or gain inferred from smoke checks. Source/config: scripts/structure_control.py and configs/structure_control.json; state: results/structure_control_queue.json. Both root launchers resume the full chain."])
    genuine_ssl_config=ROOT / "configs/ssl_compression_pilot.json"
    if genuine_ssl_config.exists():
        config=json.loads(genuine_ssl_config.read_text(encoding="utf-8"))
        state_path=ROOT / "results/ssl_compression_pilot_runner.json"
        state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status":"registered"}
        lines.extend(["", "## Genuine low-label SSL compression pilot: before choosing the commitment title", "",
                      f"User authorized simulations first, then a decision on the title/abstract and reply to Marcus. Runner state: {state['status']}. This NEW pilot starts learning with only 50 training labels/class; it differs from the completed repair-only study with a fully supervised parent. One seed, QNN and exactly matched circulant control, two fresh 20x383-step soft-SSL baselines and 12 continuations. Reuse the shared FewshotLearning view/loss code (Sohn2020FixMatch / Chen2023SoftMatch context, not full recipe reproduction).", "",
                      "Same 500 labeled/1000 validation IDs as the sibling seed0 reference. Reserve 500 fit and 500 check images from U, never weight training, leaving 47,500 unlabeled images; this is a disclosed adaptation. Adapter exposes existing GAP features/logits and adds no parameters. Uniform .25 unit removal, 73,898 to 42,034 actual real parameters; compensate/prune the SAME victims and keep an uncompressed continuing control. 1000 repair steps, paired restarting SGD/sampler streams, inherited soft-confidence state. CE-U control sees the same weak/strong L/U views and joint BN but no pseudo-label loss; it is not labeled-only training. No EMA/contrastive/triplet additions or test inference.", "",
                      "Fixed diagnostics measure classification, pseudo-label coverage/precision, soft-weighted correctness and agreement/KL with parent. Preserve all negative conditions, compare uncompressed continuation and same-victim removal. One seed cannot establish robustness, equivalence or a new mechanism. No title/author list/email is automatically finalized. Technical checks and tiny CPU/GPU preflight values are not scientific results. Config: configs/ssl_compression_pilot.json; log: results/ssl_compression_pilot_run.log; output: results/ssl_compression_pilot/REPORT.md. Both root launchers resume this stage; epoch checkpoints include full optimizer/RNG/loader/soft-confidence state."])
    # Current controls supersede historical statements that the EuroSAT test is
    # still untouched. Numerical studies themselves remain immutable.
    bn_path = ROOT / "results/paper_a_bn/summary.json"
    final_path = ROOT / "results/paper_a_final/summary.json"
    if bn_path.exists():
        bn = json.loads(bn_path.read_text(encoding="utf-8"))
        latest = ["## Current Paper A controls (2026-11-05)", "",
                  "User authorized focused literature review, BN-only repair, saved recovery analysis and frozen final evaluation; SSL is aside. LITERATURE.md records the decision: known compensation/QNN pruning, empirical structural-cost study only. GRAIL2026 is published at CPAL/PMLR and already compares BN REPAIR and fine-tuning.", "",
                  f"BN-only status: {bn['status']}, {len(bn['cases'])} original QNN states, two estimators, 2048 unlabeled training images per seed. No gradients or parameter updates. All invariance checks passed in completed cases. Recovery: results/paper_a_evidence/RECOVERY.md (3216 saved observations from 730 artifacts; no new training).", "",
                  "| Dataset | Removed/site | Comp + global BN | Same victims + global BN |", "| --- | ---: | ---: | ---: |"]
        lookup = {(r['dataset'], r['fraction'], r['policy'], r['method']): r for r in bn['aggregates']}
        for dataset in ('cifar', 'eurosat'):
            for fraction in (.25, .5):
                a = lookup.get((dataset, fraction, 'greedy_comp', 'layerwise_global'))
                b = lookup.get((dataset, fraction, 'greedy_pruned', 'layerwise_global'))
                if a and b:
                    latest.append(f"| {dataset} | {fraction} | {100*a['mean_accuracy']:.2f}% | {100*b['mean_accuracy']:.2f}% |")
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            latest.extend(["", f"Final test status: {final['status']}, {len(final['rows'])}/333 frozen checkpoints. EuroSAT final held-out tile test IS NOW EVALUATED; historical sections below describe earlier experiments only. CIFAR test was historically seen. No post-test tuning. Results: results/paper_a_final/REPORT.md, predictions and protocol hashes. All seven original fractions remain validation evidence; final test uses the declared .25/.5/.75 subset and all BN/circulant controls.", ""])
            for effect in final['paired_effects']:
                if effect['family'] == 'quaternion' and effect['stage'] == 'after' and effect['comparison'] == 'greedy_comp - bn_pruned':
                    latest.append(f"Measured final test: {effect['dataset']}, f={effect['fraction']}, comp minus BN magnitude after 3 epochs = {effect['mean_difference_pp']:+.2f} pp; individual seeds are retained in the final summary.")
        latest.extend(["", "Interpretation: BN-only mean results do not erase compensation's advantage at these operating points. This does not establish causality, uniqueness to quaternions, a new method, latency savings or acceptance. Scratch uses a different training budget. Evidence and figures: results/paper_a_evidence/; current plan: docs/PLAN_IJCNN2027.md.", ""])
        lines[2:2] = latest
    structural_path = ROOT / "results/structure_cost_study/summary.json"
    if structural_path.exists():
        structural = json.loads(structural_path.read_text(encoding="utf-8"))
        no_bn = json.loads((ROOT / "results/no_bn_pilot/summary.json").read_text(encoding="utf-8"))
        current = ["## Current structural-cost extension (2026-11-06)", "",
                   f"No-BN CIFAR seed2026 baseline COMPLETE: selected validation {100*no_bn['no_bn']['selected_validation']['accuracy']:.2f}% vs original BN {100*no_bn['existing_bn_reference']['selected_validation']['accuracy']:.2f}%, after the same 50-epoch recipe; 73002 vs 73898 real parameters. One seed, no new test inference.", "",
                   f"User-authorized all-site Q/free study: {structural['status']}, {structural['groups_done']}/14 groups, {structural['new_ft_cases']}/38 new three-epoch repairs, {structural['reused_ft_cases']}/18 reused repairs. CIFAR/EuroSAT BN three seeds, no-BN CIFAR seed2026 only; fractions .25/.5, same victims. Cases: q_pruned, q_comp, free_q_init (same initial function, five conv weight constraints released), free_full_fit (full maps physically folded into dense consumers, no projection).", "",
                   "Equal widths are NOT equal parameter budgets. Dense variants share a budget with each other and can exceed the FULL QNN parameter count. Their initialization/capacity controls are needed before attributing a post-repair advantage to compensation. Observed gaps concern these procedures, not an irreducible lower bound on all quaternion models. Sources/config/inputs frozen before outcomes; no new-method or priority claim. New study selected after earlier validation/test outcomes; evaluation here uses validation/check-calibration only.", "",
                   "Protocol/results: results/structure_cost_study/{protocol.json,REPORT.md,summary.json,state.json}; resume with the existing URUCHOM_KRZYWA.cmd or python -m scripts.structure_cost_study. Independent review/figures: python -m scripts.review_structure_cost_study after completion. Fixed study ends at these two fractions; no further sweep is automatically queued.", ""]
        lines[2:2] = current
    direct_path = ROOT / "results/direct_consumer_pilot/summary.json"
    if direct_path.exists():
        direct = json.loads(direct_path.read_text(encoding="utf-8"))
        old_q = direct['references']['q_comp']
        recent = ["## Latest direct-Q consumer pilot (2026-11-06)", "",
                  f"Status: {direct['status']}. CIFAR BN CNN6 seed2026, f=.5, SAME 19098 real parameters/masks. Before repair validation {100*direct['before']['validation']['accuracy']:.2f}% versus map-fold {100*old_q['before']['evaluation']['validation']['accuracy']:.2f}%.", ""]
        if direct.get('after'):
            recent.append(f"After three repair epochs: direct {100*direct['after']['validation']['accuracy']:.2f}% vs map-fold {100*old_q['after']['evaluation']['validation']['accuracy']:.2f}%. All five Q consumer check-MSEs decrease; final already-real readout remains map-fold-only. Fixed centered ridge on 500 fit images, separate 500 check images, no labels in regression, no test inference. One seed/point, no automatic replication or new-method claim.")
            recent.append(f"New repair successful/attempted steps: {direct['repair']['successful_optimizer_steps']}/{direct['repair']['attempted_optimizer_steps']}; reused references were not instrumented. Independent audit includes moment/solver checks and CPU prediction replay of before/after validation/check checkpoints. Objective and compensation parameterization change together; no pure feasible-space or irreducible architecture-cost attribution.")
        recent += ["", "Report: results/direct_consumer_pilot/REPORT.md. Deeper Beyond Real Weights paper/sample-code review: LITERATURE.md; local independent code counterexample .cache/literature/phm_sample_review.json. Prior art is not negated by a sample-code issue. This historical pilot was followed by the separately registered replication described above.", ""]
        lines[2:2] = recent
    replication_path = ROOT / "results/direct_consumer_replication/summary.json"
    if replication_path.exists():
        study = json.loads(replication_path.read_text(encoding="utf-8"))
        current = ["## Current direct-Q replication (2026-11-06)", "",
                   f"Status: {study['status']}; {study['completed_jobs']}/{study['planned_jobs']} jobs, {study['new_repairs_completed']} new three-epoch repairs, {study['discovery_reused']} reused discovery pilot. No new baseline training. BN CIFAR/EuroSAT three existing seeds, .25/.5 unit removal at all sites; secondary CIFAR no-BN seed2026 only. Same masks, compact Q architecture and actual real parameter counts as old map-fold. Unchanged five Q-consumer fits; final already-real head remains map-fold-only.", "",
                   "| Dataset | Norm | Removed/site | n | Direct-Q after 3ep (%) | Old map-fold after 3ep (%) | Paired mean gain (pp) | Individual gains (pp) |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        index = {(a['dataset'], a['normalization'], a['fraction'], a['case'], a['stage']): a for a in study['aggregates']}
        for e in study['effects']:
            if e['stage'] != 'after' or e['exclude_discovery']:
                continue
            key = (e['dataset'], e['normalization'], e['fraction'])
            d, q = [index[key + (c, 'after')]['mean_accuracy'] for c in ('direct_q', 'q_comp')]
            gains = ', '.join(f"{v['seed']}: {v['accuracy_difference_pp']:+.2f}" + (' discovery' if v['discovery'] else '') for v in e['seed_values'])
            current.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(e['seed_values'])} | {100*d:.2f} | {100*q:.2f} | {e['mean_accuracy_difference_pp']:+.2f} | {gains} |")
        for e in study['effects']:
            if (e['dataset'], e['normalization'], e['fraction'], e['stage'], e['exclude_discovery']) == ('cifar', 'bn', .5, 'after', True):
                current += ["", f"CIFAR .5 excluding discovery: {len(e['seed_values'])} new seeds, paired mean gain {e['mean_accuracy_difference_pp']:+.2f} pp. Three-seed means include the discovery seed and are not three independent confirmations."]
        audit_path = ROOT / 'results/direct_consumer_replication/independent_audit.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            current += ["", f"Independent audit: {audit['status']}; all frozen sources/inputs, checkpoint parameter counts/layout/hashes, matched masks, first-update evidence and 70 Torch-vs-NumPy local solves/check-MSE replays. Held-out local MSE improved {audit['check_mse_improved']}/70 times. This is not independent full-dataset validation prediction replay."]
        current += ["", "All new metrics use validation/check only; no new test inference. EuroSAT is exploratory after prior data/test inspection, no-BN n=1. Retain negative accuracy changes even when CE improves. BN compact counts are 42034/19098 real parameters at .25/.5, versus full 73898: unit removal fractions are not parameter removal fractions. Dense references have larger budgets. Scratch uses 50 epochs versus 3 repair epochs plus a pretrained full model; no scratch dominance or measured latency saving is claimed.", "",
                    "Separate discovery-only cached-moment diagnostic: results/direct_consumer_pilot/gap_diagnostic/REPORT.md. Under the SAME local centered output-ridge objective, optimizing shared maps recovers less than allowing all compact Q coefficients to move: releasing map ties accounts for 62.4-90.6% of total local objective improvement at five sites. This is not a classification-accuracy decomposition, independent replication or architecture lower bound. It uses saved moments and no additional network training/inference. Numerical sources/config frozen separately.", "",
                    "Current report/paired plot/audit: results/direct_consumer_replication/. Both root launchers target this registered study; a completed run does not start more training. No architecture/alpha search, extra fraction or additional experiment is automatically queued. Methodological novelty remains to be argued against reconstruction/folding prior art (He2017ChannelPruning, GRAIL2026, Kim2020, Asiaee2026CMR); these results support an empirical distinction between shared-map compensation cost and Q-architecture cost, not a new LS principle.", "",
                    "Priority review completed 2026-11-06: no exact predecessor located by the targeted review, but worldwide firstness NOT confirmed. Newly documented VieiraValle2022ELM already gives hypercomplex LS via real embeddings; Liang2021TwoStep Eq.8 refits the next layer after feature-based selection; KammaWada2021REAP and Luo2017ThiNet are further reconstruction precedents. Candidate contribution concerns the post-pruning Q procedure and the measured extra shared-map restrictions, not inventing the solver. First comparison to our own restricted baseline is not by itself a novelty argument. Version gap: Singh2026 accepted manuscript is blocked by Cloudflare/403; inspected preprint is not assumed identical. Full claim matrix, primary links and limitations: LITERATURE.md, Priority audit of direct-Q post-pruning reconstruction. No new experiment or external contact.", ""]
        lines[2:2] = current
    tail_path = ROOT / 'results/tail_compensation_pilot/summary.json'
    if tail_path.exists():
        tail = json.loads(tail_path.read_text(encoding='utf-8'))
        recent = ["## Current late-Q consumer logit-sensitive pilot (2026-11-06)", "",
                  f"Status: {tail['status']}; {len(tail['results'])}/2 fixed cases. One CIFAR BN seed2026 f=.5 late consumer, initialized at existing site4 direct-Q fit. Two 200-step FP32 Adam calibrations on 500 unlabeled images, same order, separate 500 check; local pre-BN MSE versus logit MSE through frozen tail. ONLY blocks.2.3.kernel changes during calibration; all four Q components audited. Teacher is current pre-site4 reference, already partly compressed, not the full pristine source. Final same-victim head map-fold and matched three-epoch supervised repair follow. Same six Q convs /19098 real parameters.", "",
                  "| Case | Validation before supervised repair (%) | Validation after 3 epochs (%) |",
                  "| --- | ---: | ---: |",
                  "| direct-Q, reused without extra steps | 34.80 | 56.70 |"]
        for r in tail['results']:
            recent.append(f"| {r['case']} | {100*r['before']['validation']['accuracy']:.2f} | {100*r['after']['validation']['accuracy']:.2f} |")
        recent += ["", "Tail-logit refinement improves pre-repair accuracy here, but no after-repair benefit is demonstrated. One exploratory seed/consumer: no superiority/equivalence claim from small differences. Local-extra is an optimizer-step control, not a converged optimum; its fit MSE slightly worsens after fixed steps. Tail reduces check logit MSE while increasing local-output MSE. All unfavorable results retained. Equal steps do not establish equal FLOPs/runtime; cached stems and measured calibration times recorded. No test evaluation or new baseline. Report: results/tail_compensation_pilot/REPORT.md. Independent CPU audit replays all before/after validation/check classification metrics and full-forward calibration errors, plus parameter/mask/buffer invariants. Frozen sources/inputs unchanged.", "",
                   "Prior art: Park2020Lookahead concerns ordinary CNNs; Singh2026 v1 IX-B ALREADY applies KD to pruned quaternion CNNs using CE+KL. Our label-free single-consumer MSE fit is a distinct limited protocol, not a reproduced published recipe or proof of novelty. Both root launchers resume this completed bounded pilot; no extra experiments queued. Earlier launcher descriptions below are historical. Marcos citation-placement table is in LITERATURE.md (four canonical entries); exact two-method equations are in docs/STRUCTURE_HYPOTHESIS.md.", ""]
        lines[2:2] = recent
    map_state_path = ROOT / "results/trainable_maps_pilot/state.json"
    if map_state_path.exists():
        map_state = json.loads(map_state_path.read_text(encoding="utf-8"))
        map_summary_path = ROOT / "results/trainable_maps_pilot/summary.json"
        map_rows = json.loads(map_summary_path.read_text(encoding="utf-8"))["cases"] if map_summary_path.exists() else []
        recent = ["## Current trainable-Q-map pilot (2026-11-06)", "",
                  f"Status: {map_state['status']}; {len(map_rows)}/2 cases completed. ONE late consumer, CIFAR BN2026 f=.5. Two new three-epoch CE repairs from identical direct-Q compact weights and fixed masks. Training reals 19098 versus 20122; both deploy as six Q convolutions /19098 reals. Corrections start at zero; removed-input C is frozen. No test or automatic replication.", "",
                  "Technical CPU checks passed: literal quaternion multiplication, exact folding, tracking/gradients/updates, interrupted-epoch replay including BN/RNG/optimizer, and exact agreement with the frozen supervised repair recipe. Actual-model discarded-copy update audits also passed. These checks are not scientific gains. Epoch checkpoints preserve resume state; a partial epoch is repeated. Successful and attempted AMP steps are distinguished.", "",
                  "| Case | Validation before (%) | Validation epoch3 (%) |", "|---|---:|---:|"]
        for row in map_rows:
            recent.append(f"| {row['case']} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
        if len(map_rows) == 2:
            control, learned = ({r["case"]: r for r in map_rows}[c] for c in ("frozen_maps", "learned_maps"))
            recent += ["", f"Observed validation difference {100*(learned['after']['validation']['accuracy']-control['after']['validation']['accuracy']):+.2f} pp, n=1. Validation CE control/learned: {control['after']['validation']['cross_entropy']:.10f}/{learned['after']['validation']['cross_entropy']:.10f}. No convincing advantage demonstrated. Independent CPU audit passes 130 source/296 input hashes, six full split replays, actual compact layouts/counts, fixed C and learned-map component checks; see independent_audit.json. Both cases record 1125 attempted/1124 successful AMP steps. No new training queued."]
        recent += ["", "Single previously studied discovery source; no replication, significance or priority claim. Q-specific literature review located no exact predecessor but does not prove firstness. ResRep/RCP already train and fold ordinary-CNN compensators; Compacter learns hypercomplex adapters; RepOptimizers establishes related optimization context. Bibliography and 36-query audit in LITERATURE.md. Current root launchers target this pilot; earlier launcher descriptions below are historical. Report: results/trainable_maps_pilot/REPORT.md.", ""]
        lines[2:2] = recent
    center_path = ROOT / "results/centered_maps_pilot/summary.json"
    if center_path.exists():
        center = json.loads(center_path.read_text(encoding="utf-8"))
        recent = ["## Current centered-Q-map calibration pilot (2026-11-06)", "",
                  f"Status: {center['status']}; {len(center['results'])}/4 fixed cases. ONE late-consumer CIFAR BN2026 f=.5, reuse earlier direct sites0-3. New hybrid mapfold control, output-fitted A, mu-only, joint A/mu; same masks, output reconstruction objective and effective-kernel ridge. A0 is nonzero original feature-ridge maps; one Q mu per removed input shared across outputs/offsets. Joint fit solves linear deltaA,d then recovers mu with verified A*mu=d, preserving the original bilinear formula. No new solver/priority claim.", "",
                  "| Case | Fit MSE | Check MSE | Validation before (%) | Validation after3ep (%) |", "|---|---:|---:|---:|---:|"]
        for row in center["results"]:
            local = center["calibration"]["cases"][row["case"]]
            recent.append(f"| {row['case']} | {local['fit_mse']:.8f} | {local['check_mse']:.8f} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
        if len(center["results"]) == 4:
            indexed = {row["case"]: row for row in center["results"]}
            joint, control, a_only = (indexed[c] for c in ("a_mu", "mapfold", "a_only"))
            recent += ["", f"Joint versus current control after repair: {100*(joint['after']['validation']['accuracy']-control['after']['validation']['accuracy']):+.2f}pp; versus A-only: {100*(joint['after']['validation']['accuracy']-a_only['after']['validation']['accuracy']):+.2f}pp. No after-repair benefit demonstrated, n=1. Validation CE control/joint: {control['after']['validation']['cross_entropy']:.10f}/{joint['after']['validation']['cross_entropy']:.10f}. Joint reconstruction improves slightly beyond A-only but does not improve recovered accuracy. No replication queued."]
        center_audit = ROOT / "results/centered_maps_pilot/independent_audit.json"
        if center_audit.exists():
            audit = json.loads(center_audit.read_text(encoding="utf-8"))
            recent += ["", f"Independent CPU audit {audit['status']}: {audit['source_count']} frozen sources/{audit['input_count']} inputs, {audit['classification_split_replays']} full classification split replays and {audit['local_fit_check_replays']} local-error replays. Exact A*mu recovery, unchanged C, actual Q layouts/counts, same masks/head folds and all checkpoints/budgets checked."]
        recent += ["", "Calibration adjustable reals A1024/mu64/joint1088 versus18432 full local Q coefficients. All final models have six Q convs/19098 reals; factor variables/constants discarded before identical3ep supervised repairs. Fit500/check500 separate; no test inference, baseline training, hyperparameter search or automatic replication. Root launchers now target this pilot; earlier shortcut descriptions below are historical. Detailed method/results: results/centered_maps_pilot/REPORT.md.", ""]
        lines[2:2] = recent
    resnet_state_path = ROOT / "results/resnet20_baselines/state.json"
    if resnet_state_path.exists():
        resnet_state = json.loads(resnet_state_path.read_text(encoding="utf-8"))
        recent = ["## Current stronger-backbone stage: ResNet20 baselines (2026-11-06)", "",
                  f"Status at synthesis: {resnet_state['status']}; see results/resnet20_baselines/state.json for live progress. Sequential real/Q CIFAR10 ResNet20, one seed2026,200epochs each; actual real trainables269722/271594. Four CPU checks plus full-size GPU discarded-copy gradient/update/reload/memory preflight passed; technical checks are not scientific quality results.", "",
                  "Common SGD momentum.9, decay1e-4, lr.1 with5epoch warmup and drops100/150, AMP, crop4/flip; existing48000/1000/1000 split and RGB normalization. Option-A shortcuts preserve component grouping. Best by validation CE, final also reported, no test inference. Approximately90percent is a target, not guaranteed. Checkpoint each completed epoch includes optimizer/scaler/all RNG/best model; STOP pauses after batch, partial epoch repeats. Root launchers target this stage. No ResNet pruning/compensation, CIFAR100 or new-seed sweep automatically started; quality gate first. Older no-running-job/launcher statements below refer to earlier completed studies.", ""]
        resnet_summary_path = ROOT / "results/resnet20_baselines/summary.json"
        if resnet_summary_path.exists():
            resnet = json.loads(resnet_summary_path.read_text(encoding="utf-8"))
            for row in resnet["results"]:
                recent.append(f"Completed {row['kind']}: selected validation {100*row['selected_validation']['accuracy']:.2f}% at epoch{row['selected_epoch']}; final {100*row['final_validation']['accuracy']:.2f}%.")
        lines[2:2] = recent
    resnet_core_path = ROOT / "results/resnet20_compression/summary.json"
    if resnet_core_path.exists():
        core = json.loads(resnet_core_path.read_text(encoding="utf-8"))
        recent = ["## ResNet20 core compression (2026-11-07)", "",
                  f"Status: {core['status']}; {len(core['results'])}/12 cases, one seed2026. Both families, internal channels of nine residual blocks, same victims per family/fraction,25%/50%, three matched3ep repairs. No new test. Discovery only.", "",
                  "| Family | Removed | Policy | Real parameters | Validation before (%) | After3ep (%) |", "|---|---:|---|---:|---:|---:|"]
        for row in core["results"]:
            recent.append(f"| {row['family']} | {row['fraction']:.2f} | {row['policy']} | {row['parameters']} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
        recent += ["", "Direct versus mapfold after repair: real -.1/+2.0pp and Q -.3/+1.1pp at25/50%. One seed: no equivalence, significance, generalization or Q-superiority claim. Independent audit passes146 source/308 input hashes,24 physical layouts,108 fold records and36 local fits; no independent full prediction replay. Runner itself replayed final validation. All cases1125/1125 successful/attempted. No current job; replication worker unimplemented, PHM waits. Detailed report: results/resnet20_compression/REPORT.md.", ""]
        lines[2:2] = recent
    (ROOT / "docs" / "RESEARCH_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_path = ROOT / "results" / "cnn_compression" / "REPORT.md"
    marker = "\n## Measured interpretation and decision\n"
    report = report_path.read_text(encoding="utf-8").split(marker)[0]
    report_path.write_text(report + marker + "\n" + "\n\n".join(decisions) + "\n\nThe activation-similarity policy did not demonstrate an advantage over random removal. Mean gains against raw kernel norm are not consistent across seeds and do not establish superiority over a strong BN-aware weight baseline. Physical compaction works; Gate C's policy value and Gate D remain unconfirmed. The next small question is the direct grouped-vs-scalar diagnostic (H2), using existing calibration features and validation ablations under a separately fixed comparison.\n", encoding="utf-8")
    print("Measured research status updated", flush=True)


if __name__ == "__main__":
    run()
