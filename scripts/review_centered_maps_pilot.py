"""Independent CPU factor/metric replay; never changes scientific checkpoints."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.tail_compensation_pilot import intermediate
from src.cnn_training import evaluate, prepare_batch
from src.compression import multilayer as old
from src.compression.centered_maps import CenteredKernel, left_operator
from src.compression.direct_consumer import theta, sse
from src.compression.structure_cost import make_model, describe
from src.compression.tail_compensation import capture
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def run():
    torch.set_num_threads(4)
    study = curve.read(ROOT/"configs/centered_maps_pilot.json")
    output = ROOT/study["output"]
    protocol = curve.read(output/"protocol.json")
    for name, digest in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest: raise AssertionError(f"Frozen source/input changed: {name}")
    summary = curve.read(output/"summary.json")
    if summary["status"] != "completed" or len(summary["results"]) != 4: raise AssertionError("Incomplete pilot")
    calibration = curve.read(output/"calibration.json")
    if file_sha256(output/"calibration.pt") != calibration["checkpoint_sha256"]: raise AssertionError("Calibration checkpoint changed")
    payload = torch.load(output/"calibration.pt", map_location="cpu", weights_only=False)
    data_study = curve.read(ROOT/"configs/compression_curve.json")
    images, labels, mean, std, sources = curve.load_data(data_study, torch.device("cpu"))
    source = next(s for s in sources if s["architecture"] == "cnn6" and s["seed"] == study["seed"])
    ids = curve.split_ids(source, None)
    victims = curve.read(ROOT/study["reference_group"]/"phase_a.json")["frozen_victims"]
    reference = intermediate(study, source, victims, 3, torch.device("cpu"))
    folded, _, _ = old.compress_site(reference, (2, 0), victims[4])
    with torch.no_grad(): folded.blocks[2][3].kernel.copy_(payload["prior_kernel"])
    if not torch.equal(payload["c"], reference.blocks[2][3].kernel.detach()[:, :, victims[4]].double()): raise AssertionError("C source mismatch")
    tensors = {s: torch.as_tensor(images[ids[s]]) for s in ("validation", "check_calibration")}
    targets = {s: torch.as_tensor(labels[ids[s]]) for s in tensors}
    fit = prepare_batch(torch.as_tensor(images[ids["fit"]]), mean, std)
    check = prepare_batch(tensors["check_calibration"], mean, std)
    packs = {"fit": capture(reference, folded, fit, 128), "check": capture(reference, folded, check, 128)}
    with np.load(ROOT/study["pilot"]/"sites/site_04/moments.npz") as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    prior = theta(payload["prior_kernel"].numpy())
    torch.testing.assert_close(torch.from_numpy(prior), torch.from_numpy(arrays["prior_theta"]), atol=1e-7, rtol=0)
    rows, objectives = [], {}
    for row in summary["results"]:
        case = row["case"]
        factors = payload["solutions"][case]
        model = CenteredKernel(payload["prior_kernel"], payload["c"], payload["a0"])
        with torch.no_grad(): model.a.copy_(factors["a"]); model.mu.copy_(factors["mu"])
        kernel = model().detach()
        shape = tuple(kernel.shape)
        actual = theta(kernel.numpy())
        torch.testing.assert_close(torch.from_numpy(actual), factors["theta"], atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(left_operator(factors["a"])@factors["mu"].reshape(-1), factors["d"].reshape(-1), atol=1e-8, rtol=1e-8)
        if case in ("mapfold", "mu_only") and not torch.equal(factors["a"], payload["a0"]): raise AssertionError("Fixed A changed")
        if case in ("mapfold", "a_only") and bool(factors["mu"].abs().max()): raise AssertionError("Fixed mu changed")
        local = deepcopy(folded)
        with torch.no_grad(): local.blocks[2][3].kernel.copy_(kernel.float())
        torch.testing.assert_close(local.blocks[2][3].kernel, factors["kernel"], atol=1e-7, rtol=0)
        local_errors = {}
        with torch.no_grad():
            for split, pack in packs.items():
                total, count = 0., 0
                for start in range(0, pack["images"], 128):
                    residual = local.blocks[2][3](pack["inputs"][start:start+128]).double()-pack["local"][start:start+128].double()
                    total += float(residual.square().sum()); count += residual.numel()
                local_errors[split] = total/count
                if abs(total/count-calibration["cases"][case][split+"_mse"]) > 1e-5: raise AssertionError("Full-forward local MSE mismatch")
        penalty = calibration["lambda"]*float(np.square(actual-prior).sum())
        numerator = sse(arrays["fit_gram"], arrays["fit_rhs"], float(arrays["fit_target_squares"]), actual)+penalty
        denominator = packs["fit"]["local"].numel()
        objectives[case] = numerator/denominator
        if abs(objectives[case]-calibration["cases"][case]["normalized_common_objective"]) > 1e-8: raise AssertionError("Objective mismatch")
        # Replay recorded final-head maps, without refitting on held-out check data.
        expected_before, _, _ = old.compress_site(local, (2, 1), victims[5], row["head_fold"]["maps"])
        replay = {}
        for stage in ("before", "after"):
            path = output/case/f"compact_{stage}.pt"
            if file_sha256(path) != row[stage+"_sha256"]: raise AssertionError("Scientific checkpoint changed")
            compact = make_model(curve.reduced_config(source["config"], "cnn6", .5), "cifar", "bn", "quaternion").eval()
            compact.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
            layout = describe(compact)
            if layout["real_parameters"] != 19098 or layout["quaternion_convolutions"] != 6 or layout["dense_real_convolutions"]: raise AssertionError("Wrong compact family/count")
            if stage == "before":
                for name, value in compact.state_dict().items():
                    torch.testing.assert_close(value, expected_before.state_dict()[name], atol=1e-6, rtol=1e-5)
            replay[stage] = {s: evaluate(compact, tensors[s], targets[s], mean, std, 128)[0] for s in tensors}
            for split, metrics in replay[stage].items():
                for metric, measured in metrics.items():
                    if abs(measured-row[stage][split][metric]) > (1e-12 if metric == "accuracy" else 1e-5): raise AssertionError(f"Metric mismatch {case}/{stage}/{split}/{metric}")
        saved_state = torch.load(output/case/"resume.pt", map_location="cpu", weights_only=False)
        if len(saved_state["history"]) != 3 or saved_state["count"] != row["counts"] or row["counts"]["attempted"] != 1125: raise AssertionError("Repair budget mismatch")
        rows.append({"case": case, "local_mse_replay": local_errors, "classification_replay": replay, "architecture": layout,
                     "factor_kernel_shape": list(shape), "A_mu_recovery_checked": True, "same_masks_head_fold_checked": True, "counts": row["counts"]})
    if objectives["a_mu"] > min(objectives["a_only"], objectives["mu_only"])+1e-7 or max(objectives["a_only"], objectives["mu_only"]) > objectives["mapfold"]+1e-7:
        raise AssertionError("Nested objective ordering failed")
    result = {"status": "passed", "source_count": len(protocol["source_hashes"]), "input_count": len(protocol["input_hashes"]),
              "classification_split_replays": 16, "local_fit_check_replays": 8, "rows": rows, "test_evaluated": False}
    write_json(output/"independent_audit.json", result)
    print(f"PASSED: {result['source_count']} sources/{result['input_count']} inputs,16 classification and8 local-error CPU replays", flush=True)


if __name__ == "__main__": run()
