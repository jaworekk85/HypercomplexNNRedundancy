"""Read-only evidence figures and prediction replay checks; no model selection."""
from collections import defaultdict
import json

import numpy as np
import torch

from scripts.paper_a_controls import ROOT, CURVES, rel
from scripts.compression_curve import read
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 6})
    output = ROOT / "results/paper_a_evidence"
    output.mkdir(parents=True, exist_ok=True)
    combined = {}
    for name in CURVES:
        dataset = "eurosat" if "eurosat" in name else "cifar"
        for row in read(ROOT / name / "summary.json")["aggregates"]:
            key = dataset, row["architecture"], row["fraction"], row["policy"], row["stage"]
            if key in combined and combined[key] != row:
                raise ValueError("Conflicting duplicate curve aggregate")
            combined[key] = row
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    for i, dataset in enumerate(("cifar", "eurosat")):
        for j, arch in enumerate(("cnn6", "real_cnn6")):
            ax = axes[i, j]
            for policy, stage, label, style in [("greedy_comp", "before", "Compensated, no repair", ":"),
                                               ("greedy_comp", "after", "Compensated + 3 epochs", "-"),
                                               ("bn_pruned", "after", "BN magnitude + 3 epochs", "--"),
                                               ("scratch", "scratch", "Scratch, 50 epochs", "-.")]:
                group = sorted([r for (d, a, f, p, s), r in combined.items() if d == dataset and a == arch and p == policy and s == stage and f > 0], key=lambda r: r["real_parameters"])
                x = np.array([r["real_parameters"] / 1000 for r in group])
                y = np.array([100*r["validation"]["mean_accuracy"] for r in group])
                sd = np.array([100*r["validation"]["sample_sd_accuracy"] for r in group])
                line, = ax.plot(x, y, marker="o", ms=3, linestyle=style, label=label)
                ax.fill_between(x, y-sd, y+sd, color=line.get_color(), alpha=.10)
            baseline = combined[(dataset, arch, 0., "baseline", "before")]
            ax.plot(baseline["real_parameters"]/1000, 100*baseline["validation"]["mean_accuracy"], "k*", ms=8, label="Full pretrained model")
            ax.set_xscale("log")
            ax.set_ylim(0, 101)
            ax.grid(alpha=.2)
            ax.set_title(f"{dataset.upper()}: {'quaternion' if j == 0 else 'real'} CNN6")
            ax.set_xlabel("Real trainable parameters (thousands, log scale)")
            ax.set_ylabel("Validation accuracy (%)")
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"quality_size.{suffix}", dpi=200)
    plt.close(fig)
    bn = read(ROOT / "results/paper_a_bn/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    paired = []
    for i, dataset in enumerate(("cifar", "eurosat")):
        ax = axes[i]
        for fraction, marker in ((.25, "o"), (.5, "s")):
            for reference, linestyle in (("greedy_pruned", "-"), ("bn_pruned", "--")):
                means, errors = [], []
                for method in ("before", "cumulative", "layerwise_global"):
                    group = [r for r in bn["aggregates"] if r["dataset"] == dataset and r["fraction"] == fraction and r["method"] == method]
                    by_policy = {r["policy"]: {v["seed"]: v["accuracy"] for v in r["seed_values"]} for r in group}
                    a, b = by_policy["greedy_comp"], by_policy[reference]
                    if set(a) != set(b) or len(a) != 3: raise ValueError("Incomplete BN pairing")
                    differences = [100*(a[s]-b[s]) for s in sorted(a)]
                    paired.append({"dataset": dataset, "fraction": fraction, "method": method, "comparison": f"comp - {reference}",
                                   "seed_differences_pp": differences, "mean_pp": float(np.mean(differences))})
                    means.append(np.mean(differences)); errors.append(np.std(differences, ddof=1))
                ax.errorbar(range(3), means, yerr=errors, marker=marker, linestyle=linestyle, capsize=2,
                            label=f"{fraction:.0%}: vs {'same victims' if reference == 'greedy_pruned' else 'BN magnitude'}")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks(range(3), ["No repair", "Cumulative\nBN", "Layerwise\nglobal BN"])
        ax.set_title(dataset.upper())
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Paired compensation advantage (pp)")
    axes[1].legend()
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"bn_control.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output / "bn_paired_effects.json", {"status": "completed", "effects": paired,
               "uncertainty": "Sample SD of three paired seed differences, not confidence intervals"})

    final = ROOT / "results/paper_a_final"
    if (final / "state.json").exists() and read(final / "state.json")["status"] == "completed":
        summary, protocol = read(final / "summary.json"), read(final / "protocol.json")
        from src.data.cifar_cached import load_cached_cifar
        from scripts.eurosat_cnn import load_eurosat
        _, (_, cifar_targets) = load_cached_cifar(ROOT / ".cache/keras/datasets/cifar-10-batches-py-target_archive")
        _, euro_targets, euro_split, _ = load_eurosat(ROOT / ".cache/eurosat/prepared")
        expected_targets = {"cifar": cifar_targets, "eurosat": euro_targets[euro_split["test"]]}
        verified = []
        for row in summary["rows"]:
            path = final / "predictions" / row["id"] / "logits.npz"
            if file_sha256(path) != row["predictions_sha256"]:
                raise ValueError(f"Prediction hash mismatch: {path}")
            with np.load(path) as data:
                logits, targets = data["logits"], data["targets"]
            if logits.shape != (row["test_images"], 10) or len(targets) != len(logits) or not np.isfinite(logits).all():
                raise AssertionError("Invalid saved predictions")
            if not np.array_equal(targets, expected_targets[row["dataset"]]):
                raise AssertionError("Saved targets do not match the frozen test ordering")
            accuracy = float(np.mean(logits.argmax(1) == targets))
            cross_entropy = float(torch.nn.functional.cross_entropy(torch.from_numpy(logits).double(), torch.from_numpy(targets)))
            if accuracy != row["test"]["accuracy"] or abs(cross_entropy-row["test"]["cross_entropy"]) > 1e-4:
                raise AssertionError(f"Saved prediction/metric mismatch: {row['id']}")
            verified.append(row["id"])
        if len(verified) != len(protocol["cases"]) or any(len(r["seed_values"]) != 3 for r in summary["aggregates"]):
            raise AssertionError("Incomplete final cases or seeds")
        write_json(output / "final_prediction_audit.json", {"status": "passed", "verified_cases": len(verified),
                   "logits_and_target_metrics_replayed": True, "all_aggregates_three_seeds": True,
                   "protocol_sha256": file_sha256(final / "protocol.json"), "cases": verified})
        write_overview(output, summary, paired)
    print("Paper A evidence figures and paired BN effects written.")


def write_overview(output, final, bn_effects):
    lines = ["# Paper A: evidence package after the focused literature review", "",
             "## Hypotheses and fixed question", "",
             "When does compensation of removed quaternion units help while preserving the original structured layer family and physical parameter budget? Does its advantage survive BN-only recalibration and brief labeled repair? No new compression method, quaternion exclusivity or literature priority is assumed.", "",
             "## Measured results", "",
             "The original seven-point CIFAR/EuroSAT curves, three seeds, QNN and real controls, circulant control and activation study are complete. This follow-up adds no new weight training. It extracts 3216 saved recovery observations (730 artifacts), recalibrates 42 original QNN states using both fixed BN estimators (84 interventions), and evaluates 333 frozen existing checkpoints on test data. Every final checkpoint reproduced its original validation metrics before test inference; saved test logits/labels were independently checked against the frozen targets and reported metrics.", "",
             "BN control uses 2048 unaugmented training images/seed, identical across policies, no labels/gradients/weight updates. Cumulative BN: one pass, simultaneous train-mode BNs; global BN: six passes, sequential evaluation-mode moments. Both reported. Calibration is an extra resource beyond the 500 compression-fit images. All parameter/buffer invariance audits passed.", "",
             "| Dataset | Units removed/site | Global BN-only: comp minus same-victim pruning (pp) | Three paired seeds (pp) |",
             "| --- | ---: | ---: | --- |"]
    for r in bn_effects:
        if r["method"] == "layerwise_global" and r["comparison"] == "comp - greedy_pruned":
            lines.append(f"| {r['dataset']} | {r['fraction']} | {r['mean_pp']:+.2f} | " + ", ".join(f"{v:+.2f}" for v in r["seed_differences_pp"]) + " |")
    lines.extend(["", "### Frozen test results: QNN after three repair epochs", "",
                  "EuroSAT: 5400 final held-out tiles, with a random tile split (not geographic transfer). CIFAR: official 10000-image test, already seen in earlier pilots. Test levels .25/.5/.75 were chosen from validation outcomes and frozen before test; all seven fractions remain validation evidence. Scratch uses 50 epochs and frozen best-validation-CE selection; these are different student-training budgets and pretrained-parent cost is additional.", "",
                  "| Dataset | Units removed/site | Real parameters | Comp + 3 epochs (%) | BN magnitude + 3 epochs (%) | Scratch 50 epochs (%) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    lookup = {(r["dataset"], r["family"], r["fraction"], r["policy"], r["stage"]): r for r in final["aggregates"]}
    for dataset in ("cifar", "eurosat"):
        for fraction in (.25, .5, .75):
            keys = [(dataset, "quaternion", fraction, p, s) for p, s in (("greedy_comp", "after"), ("bn_pruned", "after"), ("scratch", "scratch"))]
            numbers = [100*lookup[k]["mean_accuracy"] for k in keys]
            parameters = next(r["real_parameters"] for r in final["rows"] if r["dataset"] == dataset and r["family"] == "quaternion" and r["fraction"] == fraction)
            lines.append(f"| {dataset} | {fraction} | {parameters:,} | " + " | ".join(f"{v:.2f}" for v in numbers) + " |")
    lines.extend(["", "### Paired test comparisons", "",
                  "| Dataset | Family | Removed/site | Comparison after 3 epochs | Mean (pp) | Individual seeds (pp) |",
                  "| --- | --- | ---: | --- | ---: | --- |"])
    for r in final["paired_effects"]:
        if r["stage"] == "after" and r["family"] in ("quaternion", "circulant"):
            differences = ", ".join(f"{v['accuracy_difference_pp']:+.2f}" for v in r["paired_seeds"])
            lines.append(f"| {r['dataset']} | {r['family']} | {r['fraction']} | {r['comparison']} | {r['mean_difference_pp']:+.2f} | {differences} |")
    lines.extend(["", "## Interpretation and limits", "",
                  "1. In the registered moderate-compression subset, BN-only repair does not erase the paired compensation advantage. This rules out the simplest 'only stale BN' account for those outcomes, not every alternative explanation.",
                  "2. Most labeled recovery appears in the first stored epoch; the curves do not resolve the first few updates. No short-step or time-saving claim follows. At severe compression the CIFAR compensation advantage reverses on average, and compact scratch models often achieve much higher accuracy with their larger training budget.",
                  "3. Circulant compensation also helps in matched four-parameter blocks. This supports discussing an effect of constrained weight sharing; it neither proves equivalence of structures nor establishes a special quaternion benefit. Absolute full-model quality differs across families.",
                  "4. Three seeds, small CNNs, two image datasets and repeated validation use limit generality. Refinement fractions and new controls were chosen after earlier outcomes. No inference speedup/MAC advantage was benchmarked. Equal attempted training updates do not establish equal successful AMP steps.",
                  "5. Scientific contribution remains an empirical characterization of the cost, useful regimes and failures of structure-preserving QNN compression. Acceptance and novelty are not guaranteed. No further training or parameter choice is justified merely by these test outcomes.", "",
                  "## Literature decision and figure files", "",
                  "The focused review and claim matrix are in ../../LITERATURE.md. Canonical citations: GRAIL2026 (CPAL/PMLR; ridge reconstruction and BN REPAIR/FT comparisons already known), Kim2020/Asiaee2026CMR (merging/replacement/folding), Vecchi2020/Singh2026 (QNN compression), Li2020EagleEye (adaptive BN), Liu2019RethinkingPruning (scratch baseline), Valle2024VNets/Ding2017CirCNN (structured sharing). No full published-method reproduction was performed.", "",
                  "- [Seven-point quality–size figure](quality_size.pdf): Q/real, both datasets, validation means with ±1 sample SD across seeds. Logarithmic parameter axis. Full pretrained baseline marked separately; no scratch run at .125.",
                  "- [Recovery figure](recovery.pdf): epoch 0/1/2/3, measured means; missing same-victim continuations are not fabricated.",
                  "- [BN control figure](bn_control.pdf): paired advantages, error bars = sample SD of three seed differences, not confidence intervals.",
                  "- [All final test outcomes](../paper_a_final/REPORT.md), protocol and prediction hashes; [independent prediction audit](final_prediction_audit.json).", "",
                  "Next task is editorial: select a small set of these results and draft the compression paper if requested. Marcus's November 11 request requires a tentative title, authors and short abstract; authorship and correspondence are not automatically finalized."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
