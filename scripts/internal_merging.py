"""One intermediate-block study of admissible right-Hamilton compensation."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import traceback

import numpy as np
import torch

from scripts.cifar_cnn import make_model, write_json
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.internal_merging import (fit_pairs, grouped_to_tensor, merge_internal,
    outgoing_addition, select_pairs, substitute_features)
from src.compression.quaternion_channels import compact_block, component_indices
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.block_ablation import capture_blocks, classification_metrics, predict_from_block

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
SOURCES = ["scripts/internal_merging.py", "src/compression/internal_merging.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cache_path(seed):
    base = ROOT / "results/block_analysis/quaternion" if seed == 2026 else ROOT / f"results/cnn_compression/block_analysis/seed_{seed}/quaternion"
    return base / "block_2/calibration_activations.npz"


def load_compact(manifest, device):
    model = compact_block(make_model(manifest["base_config"], "quaternion"), 1, manifest["kept_units"]).to(device).eval()
    model.load_state_dict(torch.load(ROOT / manifest["checkpoint_path"], map_location=device, weights_only=True)["model"])
    return model


@torch.no_grad()
def consumer_errors(source, models, features, pairs, batch_size):
    """Compare linear consumers on the same held-out cached feature maps."""
    device = next(source.parameters()).device
    totals, count, closure_error = {}, 0, 0.
    kept = sorted(set(range(16))-{row["victim"] for row in pairs})
    channels = component_indices(kept, 16)
    for start in range(0, len(features), batch_size):
        values = features[start:start+batch_size].to(device)
        baseline = source.blocks[2][0](values)
        oracle_q = source.blocks[2][0](substitute_features(values, pairs, "quaternion_right"))
        oracle_full = source.blocks[2][0](substitute_features(values, pairs, "full_real"))
        q = models["quaternion_exact"].blocks[2][0](values[:, channels])
        projected = models["full_projected"].blocks[2][0](values[:, channels])
        pruned = models["paired_pruned"].blocks[2][0](values[:, channels])
        differences = {"full_oracle_error": oracle_full-baseline, "quaternion_error": q-baseline,
                       "full_projected_error": projected-baseline, "paired_pruned_error": pruned-baseline,
                       "projection_vs_full_oracle": projected-oracle_full}
        for name, value in differences.items():
            totals[name] = totals.get(name, 0.)+float(value.square().sum(dtype=torch.float64))
        closure_error = max(closure_error, float((q-oracle_q).abs().max()))
        count += baseline.numel()
    result = {name: value/count for name, value in totals.items()}
    result["full_projected_over_oracle_error_ratio"] = result["full_projected_error"]/result["full_oracle_error"] if result["full_oracle_error"] > 1e-20 else None
    result["quaternion_closure_max_abs_error"] = closure_error
    return result


def render(output, summary):
    order = ["baseline", "random", "paired_pruned", "quaternion_exact", "full_projected", "full_substitution"]
    aggregate = []
    for policy in order:
        seeds = []
        for seed in summary["protocol"]["config"]["seeds"]:
            rows = [row for row in summary["rows"] if row["seed"] == seed and row["policy"] == policy]
            seeds.append({"seed": seed, "accuracy": float(np.mean([row["evaluation"]["validation"]["accuracy"] for row in rows])),
                          "cross_entropy": float(np.mean([row["evaluation"]["validation"]["cross_entropy"] for row in rows]))})
        aggregate.append({"policy": policy, "seed_values": seeds, "mean_accuracy": float(np.mean([row["accuracy"] for row in seeds])),
                          "sample_sd_accuracy": float(np.std([row["accuracy"] for row in seeds], ddof=1)),
                          "mean_cross_entropy": float(np.mean([row["cross_entropy"] for row in seeds])),
                          "real_parameters": 73898 if policy == "baseline" else None if policy == "full_substitution" else 66954})
    write_json(output / "aggregated.json", aggregate)
    rows = ["# Structure-preserving merging inside quaternion CNNs", "", "## Hypotheses", "",
            "H1: an unrestricted fitted 4x4 map has a substantial component that cannot be folded into "
            "the next right-Hamilton convolution. H2: enforcing that structure increases held-out consumer error. "
            "H3: directly fitting an admissible quaternion map retains quality better than fitting a full map "
            "and projecting its folded kernels. H2/H3 are empirical and can fail.", "",
            "## Algebraic conditions and scope", "",
            "Let R(a)x represent x*a with column vectors. If a removed feature satisfies q_v=q_k*a, its "
            "consumer contribution (q_k*a)*w_v can be folded as q_k*(a*w_v), by ordinary associativity. "
            "For a general matrix A the required block is R(w_v)A. The right-Hamilton blocks form a "
            "four-dimensional subspace with orthogonal basis of squared Frobenius norm four. "
            "Orthogonal projection coefficients are <R(e_c), M>/4. If at least one outgoing w_v is nonzero, "
            "it is invertible; exact folding into that block requires A itself to lie in this subspace. "
            "All-zero outgoing weights are a degenerate exception. These are elementary algebraic facts, "
            "not a claimed new theorem.", "",
            "Maps are fitted AFTER block-2 BN/ReLU/pooling and BEFORE the next linear convolution. "
            "No map is moved through a nonlinearity. Intercepts are deliberately zero: a constant feature "
            "passed through zero-padded 3x3 convolution produces a spatially varying border term, generally "
            "not one absorbable bias. This pilot does not solve general affine offset propagation.", "",
            "## Prespecified protocol", "",
            "Three completed seed-2026/2027/2028 CNNs, block 2 only. Fit on 500 even calibration images, "
            "all 64 spatial positions; check on the 500 odd images. Whole images remain disjoint. "
            "Use zero-intercept least squares for a four-coefficient quaternion map and a sixteen-coefficient "
            "full-real map. Common four disjoint pairs are chosen using full-real fit residual MSE times "
            "outgoing kernel coefficient energy; this is a transparent proxy, not exact consumer error. "
            "This common selection favors the unrestricted fit. It is not evidence about all possible pair policies. "
            "Freeze every map, pair and random selection before validation inference. No training, BN refit or "
            "official-test inference. Validation images served earlier checkpoint selection, so are not a new blind set.", "",
            "Physically remove 4/16 producer units and matching consumer inputs. Admissible and projected "
            "models both have 66,954 real parameters (9.40% fewer than 73,898), with no extra map layer. "
            "Paired pruning and ten fixed random selections have the same size. Full substitution keeps the "
            "original network and intervenes on feature maps; it is a FUNCTIONAL ORACLE, not a compact "
            "quaternion architecture or equal-budget deployment result.", "",
            "## Measured structural and consumer diagnostics", "",
            "| Seed | Discarded full-map kernel energy | Full projected/oracle consumer error | Exact-Q closure max error |",
            "| ---: | ---: | ---: | ---: |"]
    for seed, diagnostic in summary["diagnostics"].items():
        error = diagnostic["consumer_check"]
        rows.append(f"| {seed} | {100*diagnostic['discarded_full_energy_fraction']:.2f}% | {error['full_projected_over_oracle_error_ratio']:.4f} | {error['quaternion_closure_max_abs_error']:.3g} |")
    rows += ["", "## Measured validation results", "",
             "| Policy | Real parameters | Accuracy mean ± seed SD | Mean CE |", "| --- | ---: | ---: | ---: |"]
    for row in aggregate:
        params = str(row["real_parameters"]) if row["real_parameters"] is not None else "functional oracle"
        rows.append(f"| {row['policy']} | {params} | {100*row['mean_accuracy']:.2f}% ± {100*row['sample_sd_accuracy']:.2f} pp | {row['mean_cross_entropy']:.6f} |")
    rows += ["", "Random repetitions are averaged within each seed before SD across the three training seeds. "
             "Both fitted maps use the SAME victims and keepers. Report squared-weight projection loss, actual "
             "consumer error, classification CE and accuracy separately; none alone establishes all the others.", "",
             "## Interpretation and limits", "",
             "One architecture, dataset, block, compression level and fit/selection rule. The linear/no-intercept "
             "restriction is material. Full substitution is not counted as a compact model. No generic quaternion "
             "advantage, new multiplication rule, novel-method status or publication readiness is assumed. "
             "Prior context: Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002 in references.bib. "
             "A focused literature review is still needed before any priority claim.", ""]
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Full", "Random", "Paired prune", "Q fold", "Full→Q", "Full oracle"]
    axes[0].bar(labels, [100*row["mean_accuracy"] for row in aggregate], yerr=[100*row["sample_sd_accuracy"] for row in aggregate], capsize=3,
                color=["#777777", "#999999", "#aaaaaa", "#326ea8", "#429266", "#d9a65b"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Validation accuracy (%), mean ± seed SD")
    axes[0].tick_params(axis="x", labelrotation=20)
    seeds = list(summary["diagnostics"])
    axes[1].bar(seeds, [summary["diagnostics"][seed]["consumer_check"]["full_projected_over_oracle_error_ratio"] for seed in seeds], color="#429266")
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set_ylabel("Projected/full-oracle consumer MSE ratio")
    axes[1].set_xlabel("Training seed; 500 held-out calibration images")
    fig.suptitle("Intermediate block 2; equal 66,954-parameter compact models; functional oracle labeled separately")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    return aggregate


def run(config):
    output = ROOT / config["output"]
    if config["block"] != 2 or config["removed_units"] != 4 or config["intercept"] or config["optimizer_updates"] or config["bn_recalibration"] or config["official_test_evaluation"]:
        raise ValueError("Only the prespecified linear, block-2, four-unit study is implemented")
    runs = read(ROOT / config["source_multiseed"] / "summary.json")["runs"]
    if [row["seed"] for row in runs] != config["seeds"]:
        raise ValueError("Unexpected source seeds")
    previous = read(ROOT / "results/quaternion_merging/diagnostic_protocol.json")
    for name, digest in previous["source_hashes"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    paths = [ROOT / config["source_multiseed"] / "summary.json"]
    for row in runs:
        source = ROOT / row["path"]
        paths.extend(source / name for name in ("summary.json", "data.json", "split_indices.npz", "quaternion/model.pt", "quaternion/metrics.json"))
        paths.extend([cache_path(row["seed"]), ROOT / f"results/cnn_compression/seed_{row['seed']}/baseline_r00_removed00/model.pt"])
        if not all(next(item for item in row["models"] if item["kind"] == "quaternion")["first_update"]["checks"].values()):
            raise ValueError("Original tracked-variable/gradient/update gate failed")
    protected = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}
    hashes = {**previous["source_hashes"], **{name: file_sha256(ROOT / name) for name in SOURCES}}
    protocol = {"config": config, "source_hashes": hashes, "protected_inputs": protected,
                "environment": {"torch": str(torch.__version__), "numpy": np.__version__}, "labels_used_for_fitting": False,
                "optimizer_updates": 0, "official_test_evaluated": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists() and read(output / "protocol.json") != protocol:
        raise ValueError("Study protocol changed; preserve the run and use a new output")
    write_json(output / "protocol.json", protocol)
    if (output / "summary.json").exists():
        summary = read(output / "summary.json")
        render(output, summary)
        return summary
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    model_config = read(ROOT / runs[0]["path"] / "summary.json")["config"]
    (images, labels), unused_test = load_cached_cifar(ROOT / model_config["archive"])
    del unused_test
    stats = read(ROOT / runs[0]["path"] / "data.json")["statistics"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    manifests, pairs_by_seed = [], {}
    for row in runs:
        seed, source = row["seed"], ROOT / row["path"]
        seed_everything(seed)
        checkpoint = torch.load(source / "quaternion/model.pt", map_location="cpu", weights_only=True)
        model = make_model(checkpoint["config"], "quaternion").to(device).eval()
        model.load_state_dict(checkpoint["model"])
        with np.load(cache_path(seed)) as cached:
            values = cached["trained"]
        if values.shape != (1000, 64, 16, 4) or not np.isfinite(values).all():
            raise ValueError("Unexpected block-2 activation cache")
        consumer = model.blocks[2][0].kernel.detach().cpu().double().numpy()
        candidates = fit_pairs(values[::2], consumer)
        folder = output / f"seed_{seed}"
        write_json(folder / "all_pair_fits.json", candidates)
        pairs = deepcopy(select_pairs(candidates, config["removed_units"]))
        for pair in pairs:
            x, y = values[1::2, :, pair["keeper"], :].astype(np.float64), values[1::2, :, pair["victim"], :].astype(np.float64)
            for mode in config["maps"]:
                pair["maps"][mode]["check_feature_mse"] = float(np.square(y-x@np.asarray(pair["maps"][mode]["matrix"]).T).mean())
        write_json(folder / "selected_pairs.json", pairs)
        pairs_by_seed[str(seed)] = pairs
        with np.load(source / "split_indices.npz") as split:
            cal_ids = split["calibration"]
        np.savez_compressed(folder / "partition.npz", fit_positions=np.arange(0, 1000, 2), check_positions=np.arange(1, 1000, 2),
                            fit_original_ids=cal_ids[::2], check_original_ids=cal_ids[1::2])
        normalized = prepare_batch(torch.as_tensor(images[cal_ids[np.arange(0, 64, 2)]], device=device), mean, std)
        probe_features = capture_blocks(model, normalized, 32)[1]
        cached_probe = grouped_to_tensor(values[np.arange(0, 64, 2)])
        if float((probe_features-cached_probe).abs().max()) > config["equivalence_tolerance"]:
            raise AssertionError("Hidden activation cache does not match the selected model")
        specs = [("baseline", "baseline", None), ("quaternion_exact", "quaternion_exact", None),
                 ("full_projected", "full_projected", None), ("paired_pruned", "paired_pruned", None)]
        specs.extend((f"random_r{rep:02d}", "random", rep) for rep in range(config["random_repetitions"]))
        for case_name, policy, repetition in specs:
            case_folder = folder / case_name
            if policy == "baseline":
                compact, kept, projections = deepcopy(model), list(range(16)), []
                path = ROOT / f"results/cnn_compression/seed_{seed}/baseline_r00_removed00/model.pt"
            elif policy == "random":
                rng = np.random.default_rng(config["random_seed_offset"]+seed*100+repetition)
                removed = rng.permutation(16)[:config["removed_units"]].tolist()
                kept = sorted(set(range(16))-set(removed))
                compact, projections = compact_block(model, 1, kept).eval(), []
                path = case_folder / "model.pt"
            else:
                compact, kept, projections = merge_internal(model, 1, pairs, policy)
                path = case_folder / "model.pt"
            if policy != "baseline":
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": {name: tensor.detach().cpu() for name, tensor in compact.state_dict().items()}}, path)
            # Independent full-width reference for projection/pruning; direct
            # feature substitution validates the exact-Q folding separately.
            if policy == "quaternion_exact":
                expected = predict_from_block(model, 1, substitute_features(probe_features.to(device), pairs, "quaternion_right"), batch_size=32)
            else:
                reference = deepcopy(model)
                if policy == "full_projected":
                    for pair in pairs:
                        addition, _ = outgoing_addition(consumer, pair["victim"], pair["maps"]["full_real"]["matrix"])
                        target = reference.blocks[2][0].kernel[:, :, pair["keeper"]]
                        target.add_(torch.as_tensor(addition, dtype=target.dtype, device=device))
                expected = predict_from_block(reference, 1, probe_features, sorted(set(range(16))-set(kept)), 32)
            with torch.no_grad():
                actual = compact(normalized).cpu().numpy()
            error = float(np.max(np.abs(actual-expected)))
            if error > config["equivalence_tolerance"]:
                raise AssertionError(f"Intermediate graph/fold equivalence failed: {seed}/{case_name}: {error}")
            manifest = {"seed": seed, "case": case_name, "policy": policy, "base_config": checkpoint["config"], "kept_units": kept,
                        "removed_units": sorted(set(range(16))-set(kept)), "pairs": pairs if policy in ("quaternion_exact", "full_projected", "paired_pruned") else [],
                        "projections": projections, "probe_equivalence_max_abs_error": error,
                        "checkpoint_path": str(path.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": file_sha256(path), "random_repetition": repetition}
            write_json(case_folder / "manifest.json", manifest)
            manifests.append(manifest)
        print(f"Seed {seed}: common pairs fitted, 14 physical cases frozen", flush=True)
    write_json(output / "selection_frozen.json", {"manifests": manifests, "pairs_by_seed": pairs_by_seed,
               "all_choices_frozen_before_validation": True, "official_test_evaluated": False})
    rows, diagnostics = [], {}
    for row in runs:
        seed, source = row["seed"], ROOT / row["path"]
        checkpoint = torch.load(source / "quaternion/model.pt", map_location="cpu", weights_only=True)
        original = make_model(checkpoint["config"], "quaternion").to(device).eval()
        original.load_state_dict(checkpoint["model"])
        with np.load(source / "split_indices.npz") as split:
            ids = {"check_calibration": split["calibration"][1::2], "validation": split["validation"]}
        with np.load(cache_path(seed)) as cached:
            check_features = grouped_to_tensor(cached["trained"][1::2])
        val_normalized = prepare_batch(torch.as_tensor(images[ids["validation"]], device=device), mean, std)
        val_features = capture_blocks(original, val_normalized, config["batch_size"])[1]
        pairs = pairs_by_seed[str(seed)]
        models = {}
        for manifest in [item for item in manifests if item["seed"] == seed]:
            model = load_compact(manifest, device)
            policy, case_name = manifest["policy"], manifest["case"]
            metrics, predictions = {}, {}
            for split, split_ids in ids.items():
                metrics[split], predictions[f"{split}_logits"] = evaluate(model, torch.as_tensor(images[split_ids], device=device), torch.as_tensor(labels[split_ids], device=device), mean, std, config["batch_size"])
            folder = output / f"seed_{seed}" / case_name
            np.savez_compressed(folder / "predictions.npz", **predictions,
                                **{f"{split}_labels": labels[split_ids] for split, split_ids in ids.items()},
                                **{f"{split}_ids": split_ids for split, split_ids in ids.items()})
            result = {"seed": seed, "case": case_name, "policy": policy, "physical": True, "evaluation": metrics,
                      "real_parameters": sum(p.numel() for p in model.parameters()), "buffer_elements": sum(b.numel() for b in model.buffers()),
                      "checkpoint_bytes": (ROOT / manifest["checkpoint_path"]).stat().st_size,
                      "manifest_sha256": file_sha256(folder / "manifest.json"), "predictions_sha256": file_sha256(folder / "predictions.npz")}
            write_json(folder / "metrics.json", result)
            rows.append(result)
            if policy in ("quaternion_exact", "full_projected", "paired_pruned"):
                models[policy] = model
        errors = consumer_errors(original, models, check_features, pairs, config["batch_size"])
        if errors["quaternion_closure_max_abs_error"] > config["equivalence_tolerance"]:
            raise AssertionError("Exact-Q consumer composition failed")
        full_projections = [item for manifest in manifests if manifest["seed"] == seed and manifest["policy"] == "full_projected" for item in manifest["projections"]]
        diagnostic = {"consumer_check": errors,
                      "discarded_full_energy_fraction": sum(item["discarded_squared_norm"] for item in full_projections)/sum(item["full_squared_norm"] for item in full_projections)}
        diagnostics[str(seed)] = diagnostic
        write_json(output / f"seed_{seed}/consumer_diagnostics.json", diagnostic)
        for mode, policy in (("quaternion_right", "quaternion_substitution"), ("full_real", "full_substitution")):
            metrics, predictions = {}, {}
            for split, features in (("check_calibration", check_features), ("validation", val_features)):
                outputs = predict_from_block(original, 1, substitute_features(features.to(device), pairs, mode), batch_size=config["batch_size"])
                metrics[split] = classification_metrics(outputs, labels[ids[split]])
                predictions[f"{split}_logits"] = outputs
                predictions[f"{split}_labels"] = labels[ids[split]]
                predictions[f"{split}_ids"] = ids[split]
                if mode == "quaternion_right":
                    with np.load(output / f"seed_{seed}/quaternion_exact/predictions.npz") as saved:
                        if float(np.max(np.abs(outputs-saved[f"{split}_logits"]))) > config["equivalence_tolerance"]:
                            raise AssertionError("Exact-Q physical logits differ from functional substitution")
            folder = output / f"seed_{seed}" / policy
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "predictions.npz", **predictions)
            result = {"seed": seed, "case": policy, "policy": policy, "physical": False, "evaluation": metrics,
                      "real_parameters": None, "architecture_savings_claim": False, "predictions_sha256": file_sha256(folder / "predictions.npz")}
            write_json(folder / "metrics.json", result)
            rows.append(result)
        print(f"Seed {seed}: projected/oracle consumer MSE ratio={errors['full_projected_over_oracle_error_ratio']:.4f}; discarded energy={100*diagnostic['discarded_full_energy_fraction']:.2f}%", flush=True)
    if any(file_sha256(ROOT / name) != digest for name, digest in protected.items()):
        raise AssertionError("Original input artifacts changed")
    summary = {"status": "completed", "protocol": protocol, "rows": rows, "diagnostics": diagnostics,
               "selection_frozen_sha256": file_sha256(output / "selection_frozen.json"), "optimizer_updates": 0, "official_test_evaluated": False}
    write_json(output / "summary.json", summary)
    aggregate = render(output, summary)
    print(json.dumps({"validation_accuracy": {row['policy']: row['mean_accuracy'] for row in aggregate}, "diagnostics": diagnostics}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/internal_merging.json")
    args = parser.parse_args()
    config = read(ROOT / args.config)
    try:
        with torch.no_grad():
            run(config)
    except Exception:
        output = ROOT / config["output"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "FAILED_CHECKS.md").write_text("# Failed execution/check\n\n```text\n"+traceback.format_exc()+"\n```\nNo gain is claimed from this failure.\n", encoding="utf-8")
        raise
