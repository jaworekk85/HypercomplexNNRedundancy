"""Verify fit-only selection, algebraic folding, physical tensors and results."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import make_model, write_json
from scripts.internal_merging import cache_path, consumer_errors, load_compact
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.internal_merging import grouped_to_tensor, substitute_features
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.block_ablation import capture_blocks, classification_metrics, predict_from_block
from src.models.quaternion_conv_torch import hamilton_kernel

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mse_from_moments(covariance, keeper, victim, matrix):
    k, v = slice(4*keeper, 4*keeper+4), slice(4*victim, 4*victim+4)
    xx, xy, yy = covariance[k, k], covariance[k, v], covariance[v, v]
    return float((np.trace(yy)-2*np.trace(matrix@xy)+np.trace(matrix@xx@matrix.T))/4), xy-xx@matrix.T


def independently_folded_addition(original_kernel, victim, matrix, basis):
    """Extract 4x4 blocks from the frozen expanded convolution, not new helper."""
    expanded = hamilton_kernel(original_kernel.double()).numpy()
    out_units, in_units, size = original_kernel.shape[1], original_kernel.shape[2], original_kernel.shape[-1]
    blocks = expanded.reshape(4, out_units, 4, in_units, size, size)[:, :, :, victim].transpose(1, 3, 4, 0, 2)
    full = blocks@matrix
    coefficients = np.einsum("...oi,coi->...c", full, basis)/4
    projected = np.einsum("...c,coi->...oi", coefficients, basis)
    return coefficients.transpose(3, 0, 1, 2), float(np.square(full-projected).sum()), float(np.square(full).sum())


def verify(output):
    output = Path(output)
    summary, frozen = read(output / "summary.json"), read(output / "selection_frozen.json")
    protocol, checks = summary["protocol"], {}
    config = protocol["config"]

    def check(name, value):
        checks[name] = bool(value)

    for name, digest in {**protocol["source_hashes"], **protocol["protected_inputs"]}.items():
        check(f"unchanged:{name}", file_sha256(ROOT / name) == digest)
    check("selection_manifest_preserved", file_sha256(output / "selection_frozen.json") == summary["selection_frozen_sha256"])
    check("choices_frozen_before_validation", frozen["all_choices_frozen_before_validation"])
    check("no_training_or_test", summary["optimizer_updates"] == 0 and not summary["official_test_evaluated"] and not protocol["labels_used_for_fitting"])
    check("expected_case_counts", len(frozen["manifests"]) == 42 and len(summary["rows"]) == 48)
    basis = np.stack([hamilton_kernel(torch.tensor(row, dtype=torch.float64).reshape(4,1,1,1,1)).reshape(4,4).numpy() for row in np.eye(4)])
    runs = read(ROOT / config["source_multiseed"] / "summary.json")["runs"]
    sources = {}
    for run_record in runs:
        seed, source = run_record["seed"], ROOT / run_record["path"]
        checkpoint = torch.load(source / "quaternion/model.pt", map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        with np.load(cache_path(seed)) as saved:
            values = saved["trained"].astype(np.float64)
        flattened = {"fit": values[::2].reshape(-1, 64), "check": values[1::2].reshape(-1, 64)}
        moments = {name: array.T@array/len(array) for name, array in flattened.items()}
        with np.load(source / "split_indices.npz") as split:
            indices = {name: split[name] for name in ("train", "calibration", "validation")}
        with np.load(output / f"seed_{seed}/partition.npz") as partition:
            check(f"{seed}:whole_image_partition", np.array_equal(partition["fit_positions"], np.arange(0,1000,2)) and np.array_equal(partition["check_positions"], np.arange(1,1000,2)))
            check(f"{seed}:original_image_ids", np.array_equal(partition["fit_original_ids"], indices["calibration"][::2]) and np.array_equal(partition["check_original_ids"], indices["calibration"][1::2]))
        check(f"{seed}:disjoint_splits", len(np.unique(np.concatenate(list(indices.values())))) == 50000)
        candidates = read(output / f"seed_{seed}/all_pair_fits.json")
        check(f"{seed}:all_240_directed_pairs", {(row["keeper"],row["victim"]) for row in candidates} == {(a,b) for a in range(16) for b in range(16) if a != b})
        max_mse_error, max_stationarity, max_projection_error = 0., 0., 0.
        for row in candidates:
            for mode, fit in row["maps"].items():
                matrix = np.asarray(fit["matrix"])
                mse, normal = mse_from_moments(moments["fit"], row["keeper"], row["victim"], matrix)
                max_mse_error = max(max_mse_error, abs(mse-fit["fit_feature_mse"]))
                if mode == "quaternion_right":
                    coefficients = np.einsum("oi,coi->c", matrix, basis)/4
                    reconstructed = np.einsum("c,coi->oi", coefficients, basis)
                    check_normal = np.einsum("coi,io->c", basis, normal)
                    max_projection_error = max(max_projection_error, float(np.max(np.abs(matrix-reconstructed))))
                else:
                    check_normal = normal
                max_stationarity = max(max_stationarity, float(np.max(np.abs(check_normal))))
                _, discarded, energy = independently_folded_addition(state["blocks.2.0.kernel"], row["victim"], matrix, basis)
                max_projection_error = max(max_projection_error, abs(discarded-fit["projection"]["discarded_squared_norm"]), abs(energy-fit["projection"]["full_squared_norm"]))
            energy = float(state["blocks.2.0.kernel"][:, :, row["victim"]].double().square().sum())
            max_mse_error = max(max_mse_error, abs(row["fit_selection_score"]-row["maps"]["full_real"]["fit_feature_mse"]*energy))
        check(f"{seed}:fit_errors_and_selection_proxy", max_mse_error < 1e-8)
        check(f"{seed}:least_squares_stationarity", max_stationarity < 1e-8)
        check(f"{seed}:family_and_kernel_projection", max_projection_error < 1e-8)
        selected, used = [], set()
        for row in sorted(candidates, key=lambda row: (row["fit_selection_score"],row["keeper"],row["victim"])):
            if row["keeper"] not in used and row["victim"] not in used:
                selected.append(row)
                used.update((row["keeper"],row["victim"]))
                if len(selected) == 4:
                    break
        pairs = frozen["pairs_by_seed"][str(seed)]
        check(f"{seed}:common_fit_only_disjoint_selection", [(r["keeper"],r["victim"]) for r in selected] == [(r["keeper"],r["victim"]) for r in pairs] and len(used) == 8)
        check(f"{seed}:selected_file_preserved", read(output / f"seed_{seed}/selected_pairs.json") == pairs)
        for row in pairs:
            for mode, fit in row["maps"].items():
                mse, _ = mse_from_moments(moments["check"],row["keeper"],row["victim"],np.asarray(fit["matrix"]))
                max_mse_error = max(max_mse_error, abs(mse-fit["check_feature_mse"]))
        check(f"{seed}:heldout_feature_errors", max_mse_error < 1e-8)
        sources[seed] = {"source":source,"checkpoint":checkpoint,"values":values,"indices":indices,"pairs":pairs}
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seeds"][0])
    device = torch.device(config["device"])
    first = sources[config["seeds"][0]]
    (images, labels), unused_test = load_cached_cifar(ROOT / first["checkpoint"]["config"]["archive"])
    del unused_test
    stats = read(first["source"] / "data.json")["statistics"]
    mean = torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    std = torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    manifest_lookup = {(row["seed"],row["case"]):row for row in frozen["manifests"]}
    for seed, source in sources.items():
        state, pairs = source["checkpoint"]["model"], source["pairs"]
        original = make_model(source["checkpoint"]["config"],"quaternion").to(device).eval()
        original.load_state_dict(state)
        val_ids = source["indices"]["validation"]
        val_features = capture_blocks(original,prepare_batch(torch.as_tensor(images[val_ids],device=device),mean,std),config["batch_size"])[1]
        check_features = grouped_to_tensor(source["values"][1::2].astype(np.float32))
        check_ids = source["indices"]["calibration"][1::2]
        actual_features = capture_blocks(original,prepare_batch(torch.as_tensor(images[check_ids[:32]],device=device),mean,std),32)[1]
        check(f"{seed}:cached_hidden_features_match_source", float((actual_features-check_features[:32]).abs().max()) <= config["equivalence_tolerance"])
        diagnostic_models = {}
        for row in [item for item in summary["rows"] if item["seed"] == seed]:
            folder = output / f"seed_{seed}" / row["case"]
            name = f"{seed}:{row['case']}"
            check(f"{name}:metrics_record", read(folder / "metrics.json") == row)
            check(f"{name}:prediction_hash", file_sha256(folder / "predictions.npz") == row["predictions_sha256"])
            if row["physical"]:
                manifest = manifest_lookup[(seed,row["case"])]
                check(f"{name}:manifest", read(folder / "manifest.json") == manifest and file_sha256(folder / "manifest.json") == row["manifest_sha256"])
                check(f"{name}:checkpoint", file_sha256(ROOT / manifest["checkpoint_path"]) == manifest["checkpoint_sha256"])
                kept = manifest["kept_units"]
                channels = [component*16+unit for component in range(4) for unit in kept]
                current = torch.load(ROOT / manifest["checkpoint_path"],map_location="cpu",weights_only=True)["model"]
                changed_keys = {"blocks.1.3.kernel","blocks.1.4.weight","blocks.1.4.bias","blocks.1.4.running_mean","blocks.1.4.running_var","blocks.2.0.kernel"}
                check(f"{name}:unchanged_other_state", set(current)==set(state) and all(torch.equal(value,current[key]) for key,value in state.items() if key not in changed_keys))
                slices = torch.equal(current["blocks.1.3.kernel"],state["blocks.1.3.kernel"][:,kept])
                for suffix in ("weight","bias","running_mean","running_var"):
                    slices &= torch.equal(current[f"blocks.1.4.{suffix}"],state[f"blocks.1.4.{suffix}"][channels])
                check(f"{name}:exact_producer_surgery",slices)
                expected = state["blocks.2.0.kernel"][:,:,kept].double().numpy().copy()
                if row["policy"] in ("quaternion_exact","full_projected"):
                    mode = "quaternion_right" if row["policy"]=="quaternion_exact" else "full_real"
                    check(f"{name}:same_pairs",manifest["pairs"]==pairs)
                    for pair in pairs:
                        addition,_,_ = independently_folded_addition(state["blocks.2.0.kernel"],pair["victim"],np.asarray(pair["maps"][mode]["matrix"]),basis)
                        expected[:,:,kept.index(pair["keeper"])]+=addition
                check(f"{name}:consumer_fold_and_projection",np.allclose(current["blocks.2.0.kernel"].numpy(),expected,atol=1e-6,rtol=1e-5))
                if row["policy"]=="random":
                    rng=np.random.default_rng(config["random_seed_offset"]+seed*100+manifest["random_repetition"])
                    check(f"{name}:random_selection",set(manifest["removed_units"])==set(rng.permutation(16)[:4]))
                model=load_compact(manifest,device)
                baseline=row["policy"]=="baseline"
                check(f"{name}:actual_parameter_buffer_counts",sum(p.numel() for p in model.parameters())==row["real_parameters"]==(73898 if baseline else 66954) and sum(b.numel() for b in model.buffers())==row["buffer_elements"]==(902 if baseline else 870))
                check(f"{name}:real_compact_dimensions",model.blocks[1][3].out_channels==(16 if baseline else 12) and model.blocks[2][0].in_channels==(16 if baseline else 12))
                check(f"{name}:checkpoint_bytes",(ROOT / manifest["checkpoint_path"]).stat().st_size==row["checkpoint_bytes"])
                if row["policy"] in ("quaternion_exact","full_projected","paired_pruned"):
                    diagnostic_models[row["policy"]]=model
            else:
                check(f"{name}:oracle_not_claimed_as_architecture",row["real_parameters"] is None and not row["architecture_savings_claim"])
            with np.load(folder / "predictions.npz") as saved:
                for split, ids, features in (("check_calibration",check_ids,check_features),("validation",val_ids,val_features)):
                    check(f"{name}:{split}:ids_labels",np.array_equal(saved[f"{split}_ids"],ids) and np.array_equal(saved[f"{split}_labels"],labels[ids]))
                    if row["physical"]:
                        metrics,logits=evaluate(model,torch.as_tensor(images[ids],device=device),torch.as_tensor(labels[ids],device=device),mean,std,config["batch_size"])
                        tolerance=1e-10
                    else:
                        mode="quaternion_right" if row["policy"]=="quaternion_substitution" else "full_real"
                        logits=predict_from_block(original,1,substitute_features(features.to(device),pairs,mode),batch_size=config["batch_size"])
                        metrics=classification_metrics(logits,labels[ids])
                        tolerance=1e-10
                    check(f"{name}:{split}:reload_logits",np.array_equal(logits,saved[f"{split}_logits"]))
                    check(f"{name}:{split}:metrics",all(abs(metrics[key]-row["evaluation"][split][key])<=tolerance for key in metrics))
        consumer=consumer_errors(original,diagnostic_models,check_features,pairs,config["batch_size"])
        stored=summary["diagnostics"][str(seed)]["consumer_check"]
        check(f"{seed}:consumer_diagnostics_reproduced",all(abs(consumer[key]-stored[key])<=1e-10 for key in consumer))
        check(f"{seed}:quaternion_closure",consumer["quaternion_closure_max_abs_error"]<=config["equivalence_tolerance"])
        discarded,energy=0.,0.
        for pair in pairs:
            _,a,b=independently_folded_addition(state["blocks.2.0.kernel"],pair["victim"],np.asarray(pair["maps"]["full_real"]["matrix"]),basis)
            discarded+=a; energy+=b
        check(f"{seed}:aggregate_discarded_energy",abs(discarded/energy-summary["diagnostics"][str(seed)]["discarded_full_energy_fraction"])<1e-10)
    aggregates=read(output / "aggregated.json")
    for aggregate in aggregates:
        values=[]
        for seed in config["seeds"]:
            cases=[row for row in summary["rows"] if row["seed"]==seed and row["policy"]==aggregate["policy"]]
            values.append(float(np.mean([row["evaluation"]["validation"]["accuracy"] for row in cases])))
        check(f"aggregate:{aggregate['policy']}",abs(np.mean(values)-aggregate["mean_accuracy"])<1e-12 and abs(np.std(values,ddof=1)-aggregate["sample_sd_accuracy"])<1e-12)
    result={"status":"passed" if all(checks.values()) else "failed","checks":checks,"check_count":len(checks),
            "verifier_sha256":file_sha256(Path(__file__)),"official_test_evaluated":False}
    write_json(output / "verification.json",result)
    print(json.dumps({"status":result["status"],"checks":len(checks),"failed":[key for key,value in checks.items() if not value]},indent=2),flush=True)
    if result["status"]!="passed":
        raise AssertionError("Intermediate merging checks failed")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/internal_merging")
    args=parser.parse_args()
    with torch.no_grad():
        verify(ROOT / args.output)
