"""Cached-moment decomposition of shared-map versus full-Q consumer updates."""
from pathlib import Path

import numpy as np
import torch

from scripts.compression_curve import read
from scripts.cifar_cnn import write_json
from src.compression import multilayer as old
from src.compression.direct_consumer import theta, sse
from src.compression.internal_merging import expand_right, outgoing_addition
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/direct_consumer_gap_diagnostic.json"


def map_operator(kernel, kept, victims):
    """T[o] maps a[c,kept,victim] to delta-w[c,kept,kernel-offset].

    delta w[o,k,s] = sum_v a[k,v] * w[o,v,s], with original right-Hamilton
    convention. Coefficients are shared across all outputs and spatial offsets.
    """
    coefficients = kernel[:, :, victims].transpose(1, 3, 4, 2, 0)
    outputs, kh, kw, removed, _ = coefficients.shape
    matrices = expand_right(coefficients.reshape(outputs, kh*kw, removed, 4))
    block = matrices.transpose(0, 3, 1, 4, 2)  # output, result-component, offset, a-component, victim
    count, spatial = len(kept), kh*kw
    operator = np.zeros((outputs, 4, count, spatial, 4, count, removed), dtype=np.float64)
    for k in range(count): operator[:, :, k, :, :, k, :] = block
    return operator.reshape(outputs, 4*count*spatial, 4*count*removed)


def verify_operator():
    rng = np.random.default_rng(19)
    kernel = rng.normal(size=(4, 3, 5, 3, 3))
    kept, victims = [0, 2, 4], [1, 3]
    coefficients = rng.normal(size=(4, len(kept), len(victims)))
    operator = map_operator(kernel, kept, victims)
    expected = np.zeros((4, 3, len(kept), 3, 3))
    for k in range(len(kept)):
        for v, victim in enumerate(victims):
            addition, _ = outgoing_addition(kernel, victim, expand_right(coefficients[:, k, v]))
            expected[:, :, k] += addition
    actual = np.einsum("opv,v->po", operator, coefficients.reshape(-1))
    error = float(np.max(np.abs(actual-theta(expected))))
    if error > 1e-12: raise AssertionError("Shared-map parameterization differs from exact physical folding")
    return error


def run():
    config = read(ROOT / CONFIG)
    torch.set_num_threads(config["cpu_threads"])
    pilot = ROOT / config["pilot"]
    summary = read(pilot / "summary.json")
    if summary["status"] != "completed": raise ValueError("Complete discovery pilot first")
    parent = read(pilot / "protocol.json")
    names = {CONFIG, "scripts/direct_consumer_gap_diagnostic.py", "src/compression/direct_consumer.py", "src/compression/internal_merging.py"}
    inputs = {str(pilot.relative_to(ROOT)).replace("\\", "/")+"/summary.json": file_sha256(pilot / "summary.json")}
    original = ROOT / parent["config"]["baseline_source"] / "quaternion/model.pt"
    rows = []
    for index in config["sites"]:
        folder = pilot / "sites" / f"site_{index:02d}"
        prior = original if index == 0 else pilot / "sites" / f"site_{index-1:02d}/model.pt"
        for path in (folder / "moments.npz", folder / "metrics.json", prior):
            inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    definition = {"config": config, "source_hashes": {n: file_sha256(ROOT / n) for n in sorted(names)}, "input_hashes": inputs}
    output = ROOT / config["output"]
    if (output / "protocol.json").exists() and read(output / "protocol.json") != definition:
        raise ValueError("Cached diagnostic source/input changed")
    write_json(output / "protocol.json", definition)
    operator_error = verify_operator()
    parameter_names = ["blocks.0.3.kernel", "blocks.1.0.kernel", "blocks.1.3.kernel", "blocks.2.0.kernel", "blocks.2.3.kernel"]
    for index in config["sites"]:
        folder = pilot / "sites" / f"site_{index:02d}"
        record = read(folder / "metrics.json")
        prior_path = original if index == 0 else pilot / "sites" / f"site_{index-1:02d}/model.pt"
        state = torch.load(prior_path, map_location="cpu", weights_only=True)["model"]
        kernel = state[parameter_names[index]].numpy()
        victims = record["victims"]
        kept = sorted(set(range(kernel.shape[2]))-set(victims))
        t = torch.from_numpy(map_operator(kernel, kept, victims))
        with np.load(folder / "moments.npz") as data:
            g, b = torch.from_numpy(data["fit_gram"]), torch.from_numpy(data["fit_rhs"])
            prior, direct = torch.from_numpy(data["prior_theta"]), torch.from_numpy(data["direct_theta_float64"])
            lam = record["direct_audit"]["lambda"]
            h = g+lam*torch.eye(len(g), dtype=g.dtype)
            residual_rhs = b-g@prior
            a_gram = torch.zeros((t.shape[-1], t.shape[-1]), dtype=torch.float64)
            a_rhs = torch.zeros(t.shape[-1], dtype=torch.float64)
            for o in range(len(t)):
                a_gram.add_(t[o].T@h@t[o]); a_rhs.add_(t[o].T@residual_rhs[:, o])
            eigenvalues, eigenvectors = torch.linalg.eigh((a_gram+a_gram.T)/2)
            cutoff = float(eigenvalues.max())*config["relative_eigen_cutoff"]
            if float(eigenvalues.min()) < -max(1e-8, cutoff): raise AssertionError("Unexpected indefinite shared-map Hessian")
            keep = eigenvalues > cutoff
            a = eigenvectors[:, keep]@((eigenvectors[:, keep].T@a_rhs)/eigenvalues[keep])
            delta = torch.einsum("opv,v->po", t, a)
            shared = prior+delta
            g_np, b_np = data["fit_gram"], data["fit_rhs"]
            target_squares = float(data["fit_target_squares"])
            check_g, check_b, check_yy = data["check_gram"], data["check_rhs"], float(data["check_target_squares"])
            fit_den = record["direct_audit"]["fit_spatial_rows"]*4*record["direct_audit"]["output_groups"]
            check_den = record["direct_audit"]["check_spatial_rows"]*4*record["direct_audit"]["output_groups"]
            values = {"folded": prior.numpy(), "output_fitted_shared_maps": shared.numpy(), "direct_q": direct.numpy()}
            objectives, fit_mse, check_mse = {}, {}, {}
            for name, coefficients in values.items():
                fit_sse = sse(g_np, b_np, target_squares, coefficients)
                fit_mse[name] = fit_sse/fit_den
                objectives[name] = (fit_sse+lam*float(np.square(coefficients-prior.numpy()).sum()))/fit_den
                check_mse[name] = sse(check_g, check_b, check_yy, coefficients)/check_den
            if objectives["direct_q"] > objectives["output_fitted_shared_maps"]+1e-7 or objectives["output_fitted_shared_maps"] > objectives["folded"]+1e-7:
                raise AssertionError("Nested centered-objective ordering failed")
            first = objectives["folded"]-objectives["output_fitted_shared_maps"]
            second = objectives["output_fitted_shared_maps"]-objectives["direct_q"]
            rows.append({"site": record["site"], "shared_map_coefficients": t.shape[-1], "all_q_coefficients": prior.numel(),
                         "retained_numerical_rank": int(keep.sum()), "eigenvalue_cutoff": cutoff,
                         "normalized_regularized_objectives": objectives, "fit_mse": fit_mse, "check_mse": check_mse,
                         "shared_map_refit_gain": first, "release_shared_map_ties_gain": second,
                         "ties_share_of_total_fit_objective_gain": second/(first+second) if first+second > 1e-12 else None})
        print(f"site {index}: objectives {objectives}; check MSE {check_mse}", flush=True)
    write_json(output / "summary.json", {"status": "completed", "source_config": config, "operator_verification_max_error": operator_error,
               "sites": rows, "network_training_performed": False, "test_evaluated": False})
    lines = ["# Cached-moment shared-map constraint diagnostic", "", "## Hypothesis", "",
             "Direct-Q improvement can arise from refitting shared maps for consumer outputs and from removing their shared-map parameterization ties. Compare these on identical cached inputs/targets, with the same centered ridge objective.", "",
             "## Measured results", "", "One discovery source/point, five local consumers; no new network inference/training or data selection. Direct float64 solution is compared before model-dtype rounding. Eigenvalue cutoff prespecified; numerical ranks recorded.", "",
             "| Site | Shared-map / all-Q coefficients | Objective: folded | Objective: output-fit maps | Objective: direct Q | Tie-release share of total gain |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        o = r["normalized_regularized_objectives"]
        lines.append(f"| {r['site']} | {r['shared_map_coefficients']}/{r['all_q_coefficients']} | {o['folded']:.6g} | {o['output_fitted_shared_maps']:.6g} | {o['direct_q']:.6g} | {100*r['ties_share_of_total_fit_objective_gain']:.1f}% |")
    lines += ["", "## Interpretation limits", "",
              "Nested objective ordering is a least-squares sanity check, not a novel theorem. The first gain includes changing the objective/regularization used to choose the original map point. The second compares nested update spaces under identical objective/center and measures the extra ties shared across consumer outputs and kernel offsets. These local fit-objective gaps do not decompose classification accuracy, prove held-out ordering or quantify an unavoidable quaternion architecture cost. Check MSE is separately retained in summary.json. No output-fitted-map network was trained or added to the registered replication."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    run()
