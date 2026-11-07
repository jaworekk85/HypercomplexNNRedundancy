"""Read-only audit, paired-effect figure and interpretation for direct-Q replication."""
from pathlib import Path

import numpy as np
import torch

from scripts.compression_curve import read
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/direct_consumer_replication"


def run():
    torch.set_num_threads(4)
    summary, protocol = read(OUTPUT / "summary.json"), read(OUTPUT / "protocol.json")
    if summary["status"] != "completed": raise ValueError("Complete the registered study first")
    if len(summary["jobs"]) != 14 or summary["new_repairs_completed"] != 13 or summary["discovery_reused"] != 1:
        raise AssertionError("Incomplete registered jobs")
    for name, expected in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
        if file_sha256(ROOT / name) != expected: raise ValueError(f"Frozen source/input changed: {name}")
    checks, local = [], []
    for row in summary["jobs"]:
        for name in ("before_checkpoint", "after_checkpoint"):
            if file_sha256(ROOT / row[name]) != row[name+"_sha256"]: raise AssertionError("Checkpoint mismatch")
            tensors = torch.load(ROOT / row[name], map_location="cpu", weights_only=False)["model"]
            parameters = {k: v for k, v in tensors.items() if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
            if sum(v.numel() for v in parameters.values()) != row["architecture"]["real_parameters"]:
                raise AssertionError("Checkpoint parameter count differs from recorded compact size")
            kernels = [v for k, v in parameters.items() if k.endswith(".kernel")]
            if len(kernels) != 6 or any(v.ndim != 5 or v.shape[0] != 4 for v in kernels):
                raise AssertionError("Quaternion checkpoint layout changed")
            if not all(bool(torch.isfinite(v).all()) for v in tensors.values()): raise AssertionError("Nonfinite checkpoint")
        if row["architecture"]["real_parameters"] != row["references"]["q_comp"]["before"]["real_parameters"]:
            raise AssertionError("Unmatched compact parameter count")
        if not all(row["repair"]["discarded_copy_update_audit"]["checks"].values()): raise AssertionError("Failed update audit")
        parent = ROOT / protocol["config"]["discovery_pilot"] if row["reuse_discovery"] else OUTPUT / row["id"]
        phase = read(ROOT / row["reference_group"] / "phase_a.json")
        for index, site in enumerate(row["sites"]):
            if site["victims"] != phase["frozen_victims"][index]: raise AssertionError("Victims differ from reference")
            if "direct_audit" not in site: continue
            folder = parent / "sites" / f"site_{index:02d}"
            if file_sha256(folder / "moments.npz") != site["moments_sha256"]: raise AssertionError("Moment mismatch")
            audit = site["direct_audit"]
            if audit["fit_images"] != 500 or audit["check_images"] != 500 or not audit["all_other_parameters_and_buffers_unchanged"] or audit["labels_used"]:
                raise AssertionError("Unmatched fit protocol")
            with np.load(folder / "moments.npz") as data:
                g, b = torch.from_numpy(data["fit_gram"]), torch.from_numpy(data["fit_rhs"])
                prior = torch.from_numpy(data["prior_theta"])
                solved = prior+torch.linalg.solve(g+audit["lambda"]*torch.eye(len(g), dtype=g.dtype), b-g@prior)
                error = float((solved-torch.from_numpy(data["direct_theta_float64"])).abs().max())
                if error > 1e-7: raise AssertionError("Independent ridge solve differs")
                values = data["stored_theta"]
                check_sse = float(data["check_target_squares"]-2*np.sum(values*data["check_rhs"])+np.sum(values*(data["check_gram"]@values)))
                check_mse = max(0., check_sse)/(audit["check_spatial_rows"]*4*audit["output_groups"])
                if abs(check_mse-audit["check_direct_mse"]) > 1e-9: raise AssertionError("Held-out error replay mismatch")
            local.append({"job": row["id"], "site": site["site"], "relative_check_mse": audit["check_direct_mse"]/max(1e-30, audit["check_prior_mse"]),
                          "independent_solve_max_error": error, "check_improved": audit["check_direct_mse"] < audit["check_prior_mse"]})
        checks.append({"id": row["id"], "checkpoint_and_moment_hashes": True, "same_masks_and_compact_size": True, "loaded_checkpoint_parameters_and_layout": True,
                       "update_audit": True, "discovery_reused": row["reuse_discovery"]})
    if len(local) != 70: raise AssertionError("Missing local consumer fits")
    write_json(OUTPUT / "independent_audit.json", {"status": "passed", "jobs": checks, "consumer_checks": local,
               "source_input_hashes_verified": True, "check_mse_improved": sum(r["check_improved"] for r in local),
               "scope": "Checkpoint/input hashes, parameter/update/mask invariants and independent Torch-vs-NumPy solves with held-out MSE replay. Full validation predictions are not independently replayed here.", "test_evaluated": False})
    figure(summary)
    interpretation(summary, local)
    print(f"Independent replication audit passed: {len(checks)} jobs, {len(local)} consumers; check error improved {sum(r['check_improved'] for r in local)}/{len(local)}.")


def figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, dataset in zip(axes, ("cifar", "eurosat")):
        for stage, label, style in (("before", "Before repair", "--"), ("after", "After 3 epochs", "-")):
            group = sorted([e for e in summary["effects"] if e["dataset"] == dataset and e["normalization"] == "bn" and e["stage"] == stage and not e["exclude_discovery"]], key=lambda e: e["fraction"])
            means = [e["mean_accuracy_difference_pp"] for e in group]
            sds = [np.std([v["accuracy_difference_pp"] for v in e["seed_values"]], ddof=1) for e in group]
            line = ax.errorbar([e["fraction"] for e in group], means, yerr=sds, marker="o", linestyle=style, capsize=3, label=label)
            color = line.lines[0].get_color()
            for e in group:
                for i, v in enumerate(sorted(e["seed_values"], key=lambda v: v["seed"])):
                    ax.scatter(e["fraction"]+(i-1)*.008, v["accuracy_difference_pp"], marker="*" if v["discovery"] else ".",
                               s=48 if v["discovery"] else 14, color="black" if v["discovery"] else color, zorder=4)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks([.25, .5], ["25%", "50%"])
        ax.set_xlabel("Units removed at every site")
        ax.set_title(dataset.upper()); ax.grid(alpha=.2)
    axes[0].set_ylabel("Direct-Q minus map-fold accuracy (pp)")
    axes[0].legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(OUTPUT / f"paired_effects.{suffix}", dpi=190)
    plt.close(fig)


def interpretation(summary, local):
    index = {(r["dataset"], r["normalization"], r["fraction"], r["case"], r["stage"]): r for r in summary["aggregates"]}
    lines = ["\n## Independently reviewed interpretation", "",
             "Full provenance/solve/error audit passed. This is not an independent full-dataset inference replay. All new outcomes are validation/check only. Paired plot: [PDF](paired_effects.pdf), [PNG](paired_effects.png); error bars are sample SD of three paired effects, not confidence intervals. Black stars mark the reused discovery point.", "",
             "| Dataset | Norm | Removed/site | Map-fold +3ep (%) | Direct-Q +3ep (%) | Paired mean gain (pp) | Compact scratch 50ep (%) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    context_hashes = {}
    for dataset, norm in (("cifar", "bn"), ("eurosat", "bn"), ("cifar", "none")):
        context = ROOT / ("results/compression_curve/summary.json" if dataset == "cifar" else "results/compression_curve_eurosat/summary.json")
        context_hashes[str(context.relative_to(ROOT)).replace("\\", "/")] = file_sha256(context)
        scratch = read(context)["aggregates"]
        for fraction in (.25, .5):
            q, d = [index[(dataset, norm, fraction, c, "after")]["mean_accuracy"] for c in ("q_comp", "direct_q")]
            baseline = next(r["validation"]["mean_accuracy"] for r in scratch if r["architecture"] == "cnn6" and r["fraction"] == fraction and r["policy"] == "scratch" and r["stage"] == "scratch") if norm == "bn" else None
            lines.append(f"| {dataset} | {norm} | {fraction} | {100*q:.2f} | {100*d:.2f} | {100*(d-q):+.2f} | {'not measured' if baseline is None else f'{100*baseline:.2f}'} |")
    lines += ["", "### Paired accuracy and CE signs", "",
              "| Dataset | Norm | Removed/site | Positive/zero/negative accuracy effects | Mean CE difference (direct minus map) | Seeds: accuracy pp / CE difference |",
              "| --- | --- | ---: | --- | ---: | --- |"]
    for e in summary["effects"]:
        if e["stage"] != "after" or e["exclude_discovery"]: continue
        values = e["seed_values"]
        signs = [sum(v["accuracy_difference_pp"] > 1e-9 for v in values), sum(abs(v["accuracy_difference_pp"]) <= 1e-9 for v in values), sum(v["accuracy_difference_pp"] < -1e-9 for v in values)]
        seeds = ", ".join(f"{v['seed']}: {v['accuracy_difference_pp']:+.2f} / {v['cross_entropy_difference']:+.5f}" for v in values)
        lines.append(f"| {e['dataset']} | {e['normalization']} | {e['fraction']} | {'/'.join(map(str, signs))} | {np.mean([v['cross_entropy_difference'] for v in values]):+.5f} | {seeds} |")
    lines += ["", "Scratch is a different training budget: 50 student epochs versus 3 repair epochs plus a pretrained full model. The new method is not claimed to dominate scratch or save measured runtime. No-BN is n=1; BN aggregates are n=3, with the CIFAR .5 aggregate including discovery.", "",
              "New repairs record successful optimizer steps; AMP may skip nonfinite-gradient attempts. The inherited scheduler advances on attempted batches, including skipped optimizer steps, as in the frozen comparison recipe. Reused references lack successful-step instrumentation. Therefore the endpoint matches the attempted schedule, not a guarantee of identical successful-step counts. Per-job counts are retained in summary.json; the PyTorch scheduler warning is disclosed here rather than silently changing the frozen recipe.", "",
              f"Check-calibration reconstruction MSE improved at {sum(r['check_improved'] for r in local)}/70 local consumers; retain any failures above. Local objective improvement is partly guaranteed by the centered fit, whereas check/classification changes are measured.", "",
              "The separate cached-moment diagnostic on the discovery pilot is in ../direct_consumer_pilot/gap_diagnostic/REPORT.md. It quantifies shared-map ties under a common local objective and does not decompose classification accuracy. It is not a replicated architecture lower bound or a new projection theorem.", "",
              "This registered study is closed. Interpret paired signs/CE consistency and the separate two-new-seed CIFAR table before any next proposal. No tuning, new architecture, test inference or automatic additional training follows."]
    marker = "\n## Independently reviewed interpretation\n"
    path = OUTPUT / "REPORT.md"
    path.write_text(path.read_text(encoding="utf-8").split(marker)[0]+"\n".join(lines)+"\n", encoding="utf-8")
    write_json(OUTPUT / "review_context.json", {"scratch_context_hashes": context_hashes, "new_model_selection": False, "test_evaluated": False})


if __name__ == "__main__":
    run()
