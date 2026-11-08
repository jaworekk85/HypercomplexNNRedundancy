"""Independent CPU audit of completed hybrid pilot; preserves scientific files."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.resnet20_baselines import make_model
from src.cnn_training import prepare_batch
from src.compression.structure_catalogue import collect, mutable_count, parameter_count, restore
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.phm_conv import PHMConv2d
from src.models.quaternion_conv_torch import hamilton_kernel

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/structure_catalogue_audit"


def enumerate_choices(catalogues, budget, metric="error"):
    choices = list(product(*catalogues))
    feasible = [items for items in choices if sum(x["cost"] for x in items) <= budget]
    if not feasible:
        raise ValueError("No feasible combinations")
    key = lambda items: (sum(x[metric] for x in items), sum(x["cost"] for x in items), [x["id"] for x in items])
    ranked = sorted(feasible, key=key)
    best = ranked[0]
    return {"total_combinations": len(choices), "feasible_combinations": len(feasible),
            "best_choices": [x["id"] for x in best], "best_error": key(best)[0],
            "best_cost": key(best)[1],
            "runner_up": {"choices": [x["id"] for x in ranked[1]], "error": key(ranked[1])[0]} if len(ranked) > 1 else None}


def main():
    torch.set_num_threads(2)
    folder = ROOT / "results/resnet_structure_catalogue"
    summary = curve.read(folder / "summary.json")
    assert summary["status"] == "completed"
    study = summary["config"]
    protocol = curve.read(folder / "protocol.json")
    frozen_before = {}
    for name, digest in protocol["source_hashes"].items():
        assert file_sha256(ROOT / name) == digest, name
        frozen_before[name] = digest
    catalogues, metas = [], []
    for index in study["blocks"]:
        meta = curve.read(folder / f"catalogue_block_{index}.json")
        path = folder / f"catalogue_block_{index}.pt"
        assert file_sha256(path) == meta["sha256"]
        rows = torch.load(path, map_location="cpu", weights_only=False)
        assert len({x["id"] for x in rows}) == len(rows)
        for row in rows:
            assert mutable_count(row["block"]) == row["cost"]
            assert all(row[k] >= 0 for k in ("fit_mse", "check_mse", "error"))
            assert row["kept"] == sorted(set(row["kept"]))
            dense = next(r for r in rows if r["family"] == "real" and r["kept"] == row["kept"])
            if row["family"] == "q":
                assert dense["fit_mse"] <= row["fit_mse"] + 1e-7
        catalogues.append(rows)
        metas.append(meta)
    budget = summary["selections"]["variable_n"]["budget"]
    exact = enumerate_choices(catalogues, budget)
    assert exact["best_choices"] == summary["selections"]["variable_n"]["choices"]
    assert abs(exact["best_error"] - summary["selections"]["variable_n"]["proxy_error"]) < 1e-12
    policies = {}
    for name, selection in summary["selections"].items():
        pools = [[r for r in rows if name in ("variable_n", "weight_only_selector") or
                  name == "all_q" and r["family"] == "q" or
                  name == "q_real" and r["family"] in ("q", "real") or
                  name == "fixed_phm4" and r["family"] == "phm" and r["n"] == 4] for rows in catalogues]
        check = enumerate_choices(pools, budget, "weight_error" if name == "weight_only_selector" else "error")
        assert check["best_choices"] == selection["choices"]
        policies[name] = check
    baseline = curve.read(ROOT / study["baseline_config"])
    source = make_model(baseline, "quaternion").eval()
    source.load_state_dict(torch.load(ROOT / study["baseline_checkpoint"], map_location="cpu", weights_only=True)["model"])
    (images, _), unused_test = load_cached_cifar(ROOT / baseline["archive"])
    del unused_test
    with np.load(ROOT / baseline["split_source"]) as split:
        ids = split["calibration"][1::2][:16]
    stats = curve.read(ROOT / baseline["data_statistics"])["statistics"]
    x = prepare_batch(torch.as_tensor(images[ids]), torch.tensor(stats["mean"]).reshape(1,3,1,1),
                      torch.tensor(stats["std"]).reshape(1,3,1,1))
    local = []
    for index, rows in zip(study["blocks"], catalogues):
        inputs, targets = collect(source, index, x, 8)
        width = source.blocks[index].conv1.out_channels
        selected_id = summary["selections"]["variable_n"]["choices"][len(local)]
        row = next(r for r in rows if r["id"] == selected_id)
        scalar = [c*width+k for c in range(4) for k in row["kept"]]
        q = row["block"].conv2
        expanded = hamilton_kernel(q.kernel.detach()).double()
        phm = PHMConv2d.from_kernel(expanded, 4)
        relative_error = float((phm.effective_kernel()-expanded).norm()/expanded.norm())
        assert relative_error < 1e-10
        with torch.no_grad():
            prediction = q(inputs[:,scalar])
            replacement = phm.float()(inputs[:,scalar])
            q_mse = float((prediction-targets).double().square().mean())
            replacement_mse = float((replacement-targets).double().square().mean())
            max_error = float((prediction-replacement).abs().max())
        assert max_error < 1e-4
        assert abs(q_mse-replacement_mse) < 1e-5
        phm_copy = deepcopy(row["block"])
        phm_copy.conv2 = phm
        extra = mutable_count(phm_copy)-row["cost"]
        assert extra == 64
        cached = next(r for r in rows if r["family"] == "phm" and r["n"] == 4 and r["kept"] == row["kept"])
        local.append({"block": index, "selected": selected_id, "q_fit_mse": row["fit_mse"],
                      "svd_dense_to_phm4_fit_mse": cached["fit_mse"],
                      "q_to_phm4_relative_kernel_error": relative_error,
                      "q_to_phm4_max_output_error_16_check_images": max_error,
                      "q_mse_16_check_images": q_mse, "q_to_phm4_mse_16_check_images": replacement_mse,
                      "phm4_basis_extra_parameters": extra})
    checkpoint_counts = {}
    for row in summary["results"]:
        path = folder / row["policy"] / "compact_after.pt"
        assert file_sha256(path) == row["checkpoint_sha256"]
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        model = restore(artifact["architecture"]).eval()
        model.load_state_dict(artifact["model"])
        assert parameter_count(model) == row["real_parameters"]
        with torch.no_grad():
            logits = model(x[:2])
            assert logits.shape == (2,10) and torch.isfinite(logits).all()
        checkpoint_counts[row["policy"]] = parameter_count(model)
        assert row["counts"]["attempted"] == row["counts"]["successful"]
        assert row["real_parameters"] <= summary["global_budget"]
    assert summary["results"][0]["checkpoint_sha256"] == next(r["checkpoint_sha256"] for r in summary["results"] if r["policy"] == "variable_n")
    for name, digest in frozen_before.items():
        assert file_sha256(ROOT / name) == digest
    record = {"status": "passed", "frozen_source_count": len(frozen_before),
              "catalogue_sizes": [len(rows) for rows in catalogues],
              "exhaustive": exact, "policies": policies, "local_replays": local,
              "actual_checkpoint_parameter_counts": checkpoint_counts,
              "same_width_phm4_extra_parameters": sum(r["phm4_basis_extra_parameters"] for r in local),
              "warning": "PHM4 contains the Q kernel, but dense-oracle Frobenius SVD is not activation-weighted fitting. A tight budget and coarse widths exclude the same-width learned-PHM4 pair (128 extra basis parameters). A Q-parent and this surrogate search do not establish quaternion superiority.",
              "official_test_evaluated": False, "scientific_training": False}
    write_json(OUTPUT / "audit.json", record)
    print(json.dumps({"status": record["status"], "sizes": record["catalogue_sizes"],
                      "enumeration": exact, "local": local}, indent=2))


if __name__ == "__main__":
    main()
