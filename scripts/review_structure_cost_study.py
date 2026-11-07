"""Read-only audit and interpretation of the registered structural-cost study.

Can run on partial output for figures. Complete-study checks only run when the
worker reports completion. No training, new model choices or test inference.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.compression_curve import read
from scripts.cifar_cnn import write_json
from src.compression.structure_cost import describe, make_model, release_consumers
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/structure_cost_study"
PAIRS = [("cifar", "bn"), ("eurosat", "bn"), ("cifar", "none")]


def run():
    summary, protocol = read(OUTPUT / "summary.json"), read(OUTPUT / "protocol.json")
    complete = summary["status"] == "completed"
    checks, groups, local = [], [], []
    rows = summary["rows"]
    phases = sorted(OUTPUT.glob("*/*/seed_*/f_*/phase_a.json"))
    for path in phases:
        phase = read(path)
        source, fraction = phase["source"], phase["fraction"]
        selected = [r for r in rows if (r["dataset"], r["normalization"], r["seed"], r["fraction"]) ==
                    (source["dataset"], source["normalization"], source["seed"], fraction)]
        by_case = {(r["case"], r["stage"]): r for r in selected}
        for case in protocol["config"]["cases"]:
            checkpoint = path.parent / f"{case}.pt"
            if file_sha256(checkpoint) != phase["checkpoint_hashes"][case]:
                raise AssertionError("Initial checkpoint hash changed")
            after = by_case.get((case, "after"))
            if after:
                model = path.parent / case / "model.pt"
                if file_sha256(model) != after["checkpoint_sha256"] or not all(after["discarded_copy_update_audit"]["checks"].values()):
                    raise AssertionError("Repaired checkpoint or update audit failed")
        before_q, before_e = by_case.get(("q_comp", "before")), by_case.get(("free_q_init", "before"))
        if before_q and before_e and before_q["evaluation"] != before_e["evaluation"]:
            raise AssertionError("Function-preserving embedding changed evaluated metrics")
        if complete:
            if len(by_case) != 8: raise AssertionError("Missing case/stage")
            config = dict(source["config"])
            config["quaternion_widths"] = [w-round(fraction*w) for w in config["quaternion_widths"]]
            q = make_model(config, source["dataset"], source["normalization"], "quaternion")
            q.load_state_dict(torch.load(path.parent / "q_comp.pt", map_location="cpu", weights_only=True)["model"])
            embedded = release_consumers(q)
            stored = torch.load(path.parent / "free_q_init.pt", map_location="cpu", weights_only=True)["model"]
            if any(not torch.equal(v, stored[n]) for n, v in embedded.state_dict().items()):
                raise AssertionError("Stored free-Q initialization is not the exact expanded Q state")
            if describe(embedded)["real_parameters"] != phase["architectures"]["free_full_fit"]["real_parameters"]:
                raise AssertionError("Dense initialization controls have unequal budgets")
            for a, b in zip(phase["full_site_records"], phase["frozen_victims"]):
                if a["victims"] != b: raise AssertionError("Full fit changed selected victims")
                if a["fold_max_abs_error"] > protocol["config"]["equivalence_tolerance"]:
                    raise AssertionError("Failed exact full-map fold")
            checks.append({"dataset": source["dataset"], "normalization": source["normalization"], "seed": source["seed"],
                           "fraction": fraction, "initial_embedding_bitwise_verified": True, "same_victims_verified": True,
                           "checkpoint_hashes_and_updates_verified": True})
        if all((case, "after") in by_case for case in ("q_comp", "free_q_init", "free_full_fit")):
            q, e, f = [by_case[(case, "after")]["evaluation"]["validation"]["accuracy"] for case in ("q_comp", "free_q_init", "free_full_fit")]
            total, initialization, release = 100*(f-q), 100*(f-e), 100*(e-q)
            if abs(total-initialization-release) > 1e-10: raise AssertionError("Arithmetic decomposition failed")
            groups.append({"dataset": source["dataset"], "normalization": source["normalization"], "seed": source["seed"], "fraction": fraction,
                           "full_minus_q_after_pp": total, "full_minus_embedded_after_pp": initialization,
                           "embedded_minus_q_after_pp": release,
                           "full_minus_q_before_pp": 100*(by_case[("free_full_fit", "before")]["evaluation"]["validation"]["accuracy"]-before_q["evaluation"]["validation"]["accuracy"])})
        for r in phase["full_site_records"]:
            diagnostics = r["local_reconstruction"]
            local.append({"dataset": source["dataset"], "normalization": source["normalization"], "seed": source["seed"], "fraction": fraction,
                          "site": r["site"], "check_full_mse": float(np.mean([d["check_full_mse"] for d in diagnostics])),
                          "check_q_mse": float(np.mean([d["check_q_mse"] for d in diagnostics])),
                          "fit_full_mse": float(np.mean([d["fit_full_mse"] for d in diagnostics])),
                          "fit_q_mse": float(np.mean([d["fit_q_mse"] for d in diagnostics])), "fold_max_abs_error": r["fold_max_abs_error"]})
    if complete:
        if len(checks) != 14 or len(rows) != 112 or summary["new_ft_cases"] != 38 or summary["reused_ft_cases"] != 18:
            raise AssertionError("Registered experiment incomplete")
        for n, h in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
            if file_sha256(ROOT / n) != h: raise AssertionError(f"Registered input/source drift: {n}")
        write_json(OUTPUT / "independent_audit.json", {"status": "passed", "groups": checks, "before_after_rows": len(rows),
                   "source_and_input_hashes_verified": True, "test_evaluated": False,
                   "scope": "Checkpoint hashes, exact Q-to-free stored-weight embedding, same victims, fold checks, parameter/update audits and complete case pairing. No independent replay of every model on all validation images."})
    write_json(OUTPUT / "decomposition.json", {"status": summary["status"], "groups": groups, "local_same_input_diagnostics": local,
               "meaning": "Arithmetic decomposition of observed accuracy differences under named controls, not an additive causal mechanism model."})
    figures(summary)
    interpretation(summary, groups, local)
    print(f"Reviewed {summary['status']} structural-cost study: {len(groups)} decompositions; {len(local)} site diagnostics.")


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "legend.fontsize": 6})
    histories = defaultdict(list)
    for row in summary["rows"]:
        key = row["dataset"], row["normalization"], row["fraction"], row["case"]
        if row["stage"] == "before":
            histories[(*key, 0)].append(row["evaluation"]["validation"]["accuracy"])
        else:
            for epoch in row["history"]:
                histories[(*key, epoch["epoch"])].append(epoch["validation"]["accuracy"])
    for norms, stem, size in [([("cifar", "bn"), ("eurosat", "bn")], "bn_recovery", (7.2, 4.9)),
                              ([("cifar", "none")], "no_bn_recovery", (7.2, 2.7))]:
        if len(norms) == 2:
            fig, axes = plt.subplots(2, 2, figsize=size, squeeze=False)
        else:
            fig, axes = plt.subplots(1, 2, figsize=size, squeeze=False)
        for i, (dataset, norm) in enumerate(norms):
            for j, fraction in enumerate((.25, .5)):
                ax = axes[i, j]
                for case, label, style in [("q_pruned", "Q, same victims pruned", ":"), ("q_comp", "Q compensation", "-"),
                                           ("free_q_init", "Dense, Q initialization", "--"), ("free_full_fit", "Dense, full-map initialization", "-.")]:
                    values = [(e, histories[(dataset, norm, fraction, case, e)]) for e in range(4) if histories.get((dataset, norm, fraction, case, e))]
                    if not values: continue
                    x = [e for e, _ in values]
                    y = np.array([100*np.mean(v) for _, v in values])
                    sd = np.array([100*np.std(v, ddof=1) if len(v) > 1 else 0. for _, v in values])
                    line, = ax.plot(x, y, marker="o", ms=3, linestyle=style, label=label)
                    ax.fill_between(x, y-sd, y+sd, color=line.get_color(), alpha=.12)
                ax.set_title(f"{dataset.upper()}, {norm}, {fraction:.0%} removed/site")
                ax.set_xlabel("Repair epochs")
                ax.set_ylabel("Validation accuracy (%)")
                ax.set_xticks(range(4)); ax.grid(alpha=.2)
        axes[0, 0].legend(loc="lower right")
        fig.tight_layout()
        for suffix in ("png", "pdf"): fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=180)
        plt.close(fig)
    # Make the unequal parameter cost visible beside the matched-width curves.
    baselines = {}
    for path in OUTPUT.glob("*/*/seed_*/f_*/phase_a.json"):
        source = read(path)["source"]
        metrics = read(ROOT / source["path"] / "quaternion/metrics.json")
        baselines[(source["dataset"], source["normalization"], source["seed"])] = metrics
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))
    for ax, (dataset, norm) in zip(axes, PAIRS):
        for case, label, style in [("q_pruned", "Q, pruned", ":"), ("q_comp", "Q compensation", "-"),
                                   ("free_q_init", "Dense, Q initialization", "--"), ("free_full_fit", "Dense, full-map initialization", "-.")]:
            values = sorted([r for r in summary["aggregates"] if r["dataset"] == dataset and r["normalization"] == norm
                             and r["stage"] == "after" and r["case"] == case], key=lambda r: r["real_parameters"])
            if not values: continue
            ax.errorbar([r["real_parameters"]/1000 for r in values], [100*r["mean_accuracy"] for r in values],
                        yerr=[100*(r["sample_sd_accuracy"] or 0.) for r in values], marker="o", ms=3, capsize=2, linestyle=style, label=label)
        base = [v for (d, n, _), v in baselines.items() if (d, n) == (dataset, norm)]
        if base:
            ax.plot(np.mean([v["real_trainable_parameters"] for v in base])/1000,
                    100*np.mean([v["selected_validation"]["accuracy"] for v in base]), "k*", ms=9, label="Full source QNN")
        ax.set_title(f"{dataset.upper()}, {norm}")
        ax.set_xlabel("Real trainable parameters (thousands)")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_xscale("log"); ax.set_ylim(0, 101); ax.grid(alpha=.2)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(OUTPUT / f"parameter_tradeoff.{suffix}", dpi=180)
    plt.close(fig)


def interpretation(summary, groups, local):
    lines = ["\n## Independently reviewed interpretation", "",
             f"Review status: {summary['status']}. Completed paired groups: {len(groups)}/14. All entries below retain the registered outcomes; no new model selection or test inference.", "",
             "| Dataset | Norm | Removed/site | n | Full minus Q before (pp) | Full minus Q after (pp) | Full minus dense-Q-init after (pp) | Dense-Q-init minus Q after (pp) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset, norm in PAIRS:
        for fraction in (.25, .5):
            selected = [r for r in groups if (r["dataset"], r["normalization"], r["fraction"]) == (dataset, norm, fraction)]
            if not selected: continue
            fields = ("full_minus_q_before_pp", "full_minus_q_after_pp", "full_minus_embedded_after_pp", "embedded_minus_q_after_pp")
            lines.append(f"| {dataset} | {norm} | {fraction} | {len(selected)} | " + " | ".join(f"{np.mean([r[k] for r in selected]):+.2f}" for k in fields) + " |")
    lines.extend(["", "The final two columns sum to total Full-minus-Q after repair. This is an arithmetic decomposition under two controls, not evidence of independent/additive causal mechanisms. Per-seed differences are in decomposition.json.", "",
                  "Unrestricted half-width models have roughly the parameter count of the original full QNN; at .25 removal they are larger than that source. This follows approximately P_free(f) ≈ 4(1-f)^2 P_Q(0), versus P_Q(f) ≈ (1-f)^2 P_Q(0), with first-layer/BN/readout corrections. These are equal-width structural-cost comparisons, not a parameter-matched compression frontier.", "",
                  "Figures: [BN recovery](bn_recovery.pdf), [no-BN recovery](no_bn_recovery.pdf), [actual parameter trade-off](parameter_tradeoff.pdf). Recovery shading and parameter-plot error bars are ±1 sample SD of seed accuracy, not confidence intervals; no-BN has one seed and no estimated seed uncertainty. A source-model marker is included so a larger, less accurate reconstructed model is not presented as a useful compression improvement.", "",
                  "The observed Q/free gap concerns these specified compensation procedures. It is NOT an irreducible lower bound imposed by quaternion architectures. In particular, fitting one shared feature-reconstruction map and then folding it is more restrictive than directly optimizing every surviving quaternion consumer weight against its desired outputs. No such direct consumer fit is claimed or silently substituted here.", "",
                  "A full-fit advantage over dense-Q-init uses the SAME dense architecture and parameter count, so that comparison can support an initialization effect under the fixed repair recipe. Dense-Q-init minus Q-comp begins from the same function and measures the intervention of releasing five weight constraints, including changed optimization geometry. Neither comparison establishes a quaternion-specific mechanism or a novel ridge method.", "",
                  "Local Q/full map errors were evaluated on the SAME current full-path inputs at each site, using an untouched check-calibration half. These diagnostics are distinct from comparing activations along two already-diverged complete paths. Spatial observations are not claimed to be independent samples. No-BN n=1 is exploratory; the existing baseline, mask choice and parameter count differ from BN.", "",
                  "The fixed study ends at these two fractions. A scientifically distinct follow-up, if a remaining gap matters, would test whether directly fitting a quaternion consumer (same compact architecture) closes it. That would test a weakness of the current compensation parameterization; it would still require a closest-prior-work review before any novelty claim. It is NOT a newly queued experiment."])
    marker = "\n## Independently reviewed interpretation\n"
    path = OUTPUT / "REPORT.md"
    original = path.read_text(encoding="utf-8").split(marker)[0]
    path.write_text(original + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(4)
    run()
