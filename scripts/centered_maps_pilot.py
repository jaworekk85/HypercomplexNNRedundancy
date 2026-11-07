"""One late-consumer centered-Q calibration pilot; no automatic extension."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel
from scripts.run_activation_queue import lock
from scripts.tail_compensation_pilot import intermediate, final_compaction, save_model
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.compression import multilayer as old
from src.compression.centered_maps import CenteredKernel, solve_cases
from src.compression.direct_consumer import theta, kernel_of, error, sse
from src.compression.internal_merging import project_right
from src.compression.structure_cost import describe, make_model
from src.compression.tail_compensation import capture
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save, repair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/centered_maps_pilot.json"
NEW = [CONFIG, "scripts/centered_maps_pilot.py", "src/compression/centered_maps.py", "tests_torch/test_centered_maps.py"]
SPLITS = ("validation", "check_calibration")


def register(study):
    parent = curve.read(ROOT/study["parent_protocol"])
    for name, digest in {**parent["source_hashes"], **parent["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest: raise ValueError(f"Frozen parent changed: {name}")
    pilot = ROOT/study["pilot"]
    inputs = [ROOT/study["parent_protocol"], ROOT/study["reference_group"]/"phase_a.json",
              pilot/"sites/site_04/moments.npz", pilot/"sites/site_04/metrics.json",
              pilot/"sites/site_03/model.pt", pilot/"direct_before.pt", pilot/"summary.json", pilot/"finetuned/metrics.json"]
    definition = {"config": study,
                  "source_hashes": {**parent["source_hashes"], **{p: file_sha256(ROOT/p) for p in NEW}},
                  "input_hashes": {**parent["input_hashes"], **{rel(p): file_sha256(p) for p in inputs}},
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
                  "test_evaluated": False}
    output = ROOT/study["output"]
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("Registered centered-map study changed")
    write_json(path, definition)
    return definition


def prepare(study):
    seed_everything(study["seed"])
    model = CenteredKernel(torch.randn(4, 3, 2, 3, 3, dtype=torch.float64),
                           torch.randn(4, 3, 2, 3, 3, dtype=torch.float64), torch.randn(4, 2, 2, dtype=torch.float64))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model().square().mean().backward()
    gradients = {n: p.grad.flatten(1).norm(dim=1).tolist() for n, p in model.named_parameters()}
    torch.optim.Adam(model.parameters(), lr=.001).step()
    updates = {n: (p-before[n]).flatten(1).norm(dim=1).tolist() for n, p in model.named_parameters()}
    if not all(v > 0 for values in [*gradients.values(), *updates.values()] for v in values): raise AssertionError("A/mu tracking audit failed")
    definition = register(study)
    output = ROOT/study["output"]
    write_json(output/"readiness.json", {"status": "passed", "source_hashes": definition["source_hashes"],
              "scientific_results": False, "synthetic_component_gradients": gradients, "synthetic_component_updates": updates,
              "tests": "2 CPU tests passed: independent product/folding/gradients and nested objective/exact joint mu recovery", "updated_utc": now()})
    if not (output/"state.json").exists(): write_json(output/"state.json", {"status": "ready_not_started", "updated_utc": now()})
    print("READY: 4 fixed cases; no scientific calibration or repair performed", flush=True)


@torch.no_grad()
def forward_mse(model, pack):
    total, count = 0., 0
    for start in range(0, pack["images"], 128):
        predicted = model.blocks[2][3](pack["inputs"][start:start+128])
        residual = predicted.double()-pack["local"][start:start+128].double()
        total += float(residual.square().sum())
        count += residual.numel()
    return total/count


def calibration(study, reference, folded, maps, kept, fit_images, check_images):
    output = ROOT/study["output"]
    path = output/"calibration.pt"
    if path.exists():
        metadata = curve.read(output/"calibration.json")
        if file_sha256(path) != metadata["checkpoint_sha256"]: raise ValueError("Saved calibration changed")
        return torch.load(path, map_location="cpu", weights_only=False), metadata
    source = reference.blocks[2][3].kernel.detach().cpu().double()
    victims = [row["victim"] for row in maps]
    c = source[:, :, victims]
    a0 = torch.stack([torch.from_numpy(project_right(np.asarray(row["matrices"]))[0].T) for row in maps], dim=-1)
    prior_kernel = folded.blocks[2][3].kernel.detach().cpu().double()
    prior = torch.from_numpy(theta(prior_kernel.numpy()))
    local = ROOT/study["pilot"]/"sites/site_04"
    record = curve.read(local/"metrics.json")["direct_audit"]
    with np.load(local/"moments.npz") as cached:
        arrays = {key: cached[key].copy() for key in cached.files}
    prior_error = float(np.max(np.abs(prior.numpy()-arrays["prior_theta"])))
    if prior_error > 1e-7: raise AssertionError("Cached moment prior disagrees with reconstructed fold")
    basis_model = CenteredKernel(prior_kernel, c, a0)
    with torch.no_grad():
        basis_model.a.zero_()
        anchor = basis_model()
    source_kept = source[:, :, kept]
    anchor_error = float((anchor-source_kept).abs().max())
    if anchor_error > 1e-6: raise AssertionError("Anchored B differs beyond old folding roundoff")
    fit, check = capture(reference, folded, fit_images, 128), capture(reference, folded, check_images, 128)
    # Discarded actual-shape A/mu update verifies both gradients at nonzero A0.
    probe = CenteredKernel(prior_kernel.to(fit_images.device).float(), c.to(fit_images.device).float(), a0.to(fit_images.device).float())
    from src.models.quaternion_conv_torch import hamilton_kernel
    pred = torch.nn.functional.conv2d(fit["inputs"][:8], hamilton_kernel(probe()), padding=1)
    loss = (pred-fit["local"][:8]).square().mean()
    loss.backward()
    gradient = {n: p.grad.flatten(1).norm(dim=1).tolist() for n, p in probe.named_parameters()}
    initial = {n: p.detach().clone() for n, p in probe.named_parameters()}
    torch.optim.Adam(probe.parameters(), lr=.001).step()
    update = {n: (p-initial[n]).detach().flatten(1).norm(dim=1).tolist() for n, p in probe.named_parameters()}
    if not all(v > 0 for values in [*gradient.values(), *update.values()] for v in values): raise AssertionError("Actual A/mu update failed")
    start = time.perf_counter()
    print("CALIBRATION: common output-moment system, solving A, mu and joint A/mu", flush=True)
    solutions = solve_cases(torch.from_numpy(arrays["fit_gram"]), torch.from_numpy(arrays["fit_rhs"]), prior, c, a0,
                            record["lambda"], study["relative_eigen_cutoff"])
    seconds = time.perf_counter()-start
    fit_data = {"gram": arrays["fit_gram"], "rhs": arrays["fit_rhs"], "target_squares": float(arrays["fit_target_squares"]), "rows": record["fit_spatial_rows"], "outputs": record["output_groups"]}
    check_data = {"gram": arrays["check_gram"], "rhs": arrays["check_rhs"], "target_squares": float(arrays["check_target_squares"]), "rows": record["check_spatial_rows"], "outputs": record["output_groups"]}
    metrics = {}
    for case, values in solutions.items():
        candidate = deepcopy(folded)
        with torch.no_grad(): candidate.blocks[2][3].kernel.copy_(torch.from_numpy(kernel_of(values["theta"].numpy(), tuple(prior_kernel.shape))).to(candidate.blocks[2][3].kernel))
        actual = theta(candidate.blocks[2][3].kernel.detach().cpu().numpy())
        fitted = values["theta"].numpy()
        objective = (sse(fit_data["gram"], fit_data["rhs"], fit_data["target_squares"], fitted)+record["lambda"]*float(np.square(fitted-prior.numpy()).sum()))/(fit_data["rows"]*4*fit_data["outputs"])
        fit_mse, check_mse = error(fit_data, actual), error(check_data, actual)
        replay_fit, replay_check = forward_mse(candidate, fit), forward_mse(candidate, check)
        if abs(fit_mse-replay_fit) > 1e-5 or abs(check_mse-replay_check) > 1e-5: raise AssertionError("Moment/full-consumer reconstruction disagrees")
        # The saved factors must reproduce the actual user's centered formula.
        reconstruction = CenteredKernel(prior_kernel, c, a0)
        with torch.no_grad(): reconstruction.a.copy_(values["a"]); reconstruction.mu.copy_(values["mu"])
        factor_error = float((reconstruction()-torch.from_numpy(kernel_of(fitted, tuple(prior_kernel.shape)))).abs().max())
        if factor_error > 1e-8: raise AssertionError("A/mu formula does not match solved kernel")
        values["kernel"] = candidate.blocks[2][3].kernel.detach().cpu().clone()
        metrics[case] = {"fit_mse": fit_mse, "check_mse": check_mse, "normalized_common_objective": objective,
                         "full_forward_fit_mse": replay_fit, "full_forward_check_mse": replay_check,
                         "factor_fold_max_abs_error": factor_error, "A_components": values["a"].flatten(1).norm(dim=1).tolist(),
                         "mu_components": values["mu"].flatten(1).norm(dim=1).tolist(), "solver": values["audit"]}
        print(f"{case}: fit MSE {fit_mse:.8f}; check MSE {check_mse:.8f}", flush=True)
    objectives = {case: row["normalized_common_objective"] for case, row in metrics.items()}
    direct = arrays["direct_theta_float64"]
    direct_objective = (sse(fit_data["gram"], fit_data["rhs"], fit_data["target_squares"], direct)+record["lambda"]*float(np.square(direct-prior.numpy()).sum()))/(fit_data["rows"]*4*fit_data["outputs"])
    if max(objectives["a_only"], objectives["mu_only"]) > objectives["mapfold"]+1e-7 or objectives["a_mu"] > min(objectives["a_only"], objectives["mu_only"])+1e-7 or direct_objective > objectives["a_mu"]+1e-7:
        raise AssertionError("Nested common-objective ordering failed")
    payload = {"solutions": solutions, "prior_kernel": prior_kernel, "a0": a0, "c": c}
    atomic_save(payload, path)
    metadata = {"cases": metrics, "lambda": record["lambda"], "joint_system_seconds": seconds,
                "cached_prior_max_abs_error": prior_error, "rounding_anchor_max_abs_error": anchor_error,
                "actual_discarded_update": {"gradient_component_norms": gradient, "update_component_norms": update},
                "direct_Q_reused_local_objective": direct_objective, "fit_images": len(fit_images), "check_images": len(check_images),
                "calibration_labels_used": False, "all_other_parameters_buffers_frozen": True,
                "checkpoint_sha256": file_sha256(path)}
    write_json(output/"calibration.json", metadata)
    return payload, metadata


def report(study, rows, status, calibrated):
    output = ROOT/study["output"]
    direct = curve.read(ROOT/study["pilot"]/"summary.json")
    write_json(output/"summary.json", {"status": status, "config": study, "results": rows,
               "calibration": calibrated, "reused_direct_Q": {"before": direct["before"], "after": direct["after"]}, "test_evaluated": False})
    lines = ["# Centered quaternion compensation pilot", "", f"Status: {status}. One CIFAR BN2026 f=.5 late consumer.", "",
             "## Hypothesis", "", "Calibrating a per-removed-channel Q center together with shared Q maps may recover useful outputs at unchanged physical compact size.", "",
             "## Measured results", "", "| Case | Calibrated reals | Fit MSE | Check MSE | Validation before (%) | Validation after3ep (%) |", "|---|---:|---:|---:|---:|---:|"]
    counts = {"mapfold": 0, "a_only": 1024, "mu_only": 64, "a_mu": 1088}
    for row in rows:
        local = calibrated["cases"][row["case"]]
        lines.append(f"| {row['case']} | {counts[row['case']]} | {local['fit_mse']:.8f} | {local['check_mse']:.8f} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
    lines += [f"| direct-Q, REUSED | 18432 | context | context | {100*direct['before']['validation']['accuracy']:.2f} | {100*direct['after']['validation']['accuracy']:.2f} |", "",
              "## Method and interpretation limits", "",
              "Reuse direct-pilot sites0-3 and fixed masks. The teacher is current pre-site4, already partly compressed, not the pristine full baseline. Four cases start at the same old rounded site4 map-fold; this hybrid control differs from the old all-sites map-fold reference. One mu[v] per removed channel shared over all outputs/offsets. U=U0+(A-A0)C-A*mu, fixed C and anchored B; initialization roundoff is measured. Fit500/check500 held-out calibration images; cached moments agree with actual forwards. All objective comparisons use identical output SSE plus lambda||U-U0||^2, not penalties on A/mu themselves. Joint linear (deltaA,d) solution is accepted only after recovering mu such that A*mu=d and checking the original product formula. Linear algebra is established, not a novel theorem. Matrix rank, condition, residuals, factor norms and any failures are retained.", "",
              "Final same-victim real-head fold is recomputed per case, then ALL ordinary compact weights/BN train for the SAME3ep CE recipe. No A/mu factors remain during repair or inference. Six Q convolutions/19098 trainable reals in every final model. Local calibration uses18432 coefficient slots, A1024/mu64/joint1088 adjustable real values; no masks-as-compression claim. Reused direct-Q reference has the same initialization stem/support/repair recipe but different calibration space/cost. Equal epoch counts do not establish equal total runtime/compute. One discovery source only; no significance, robust superiority or held-out-test claim. Negative/tiny gains retained. Priors: VieiraValle2022ELM, Kim2020, Ding2021ResRep, Zhang2026RCP, Singh2026. Exact novelty/priority unconfirmed. No sweep, baseline retraining, new test or automatic extension.", "",
              "Atomic epoch resume captures optimizer/scheduler/scaler/RNG states; partial epoch is repeated. STOP requests pause after a batch. Final epoch3 endpoint, no validation selection. Successful/attempted AMP updates recorded separately."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(study):
    output = ROOT/study["output"]
    definition = register(study)
    ready = curve.read(output/"readiness.json")
    if ready["status"] != "passed" or ready["source_hashes"] != definition["source_hashes"]: raise ValueError("Readiness changed")
    with lock("compression_curve_runner.lock"):
        rows = []
        write_json(output/"state.json", {"status": "running", "stage": "initialization", "pid": os.getpid(), "updated_utc": now()})
        try:
            seed_everything(study["seed"])
            device = torch.device(study["device"])
            images, labels, mean, std, sources = curve.load_data(curve.read(ROOT/"configs/compression_curve.json"), device)
            source = next(s for s in sources if s["architecture"] == "cnn6" and s["seed"] == study["seed"])
            ids = curve.split_ids(source, None)
            groups = [set(ids[s]) for s in ("train", "validation", "fit", "check_calibration")]
            if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]): raise AssertionError("Splits overlap")
            fit = prepare_batch(torch.as_tensor(images[ids["fit"]], device=device), mean, std)
            check = prepare_batch(torch.as_tensor(images[ids["check_calibration"]], device=device), mean, std)
            victims = curve.read(ROOT/study["reference_group"]/"phase_a.json")["frozen_victims"]
            reference = intermediate(study, source, victims, 3, device)
            moments = old.SiteMoments(*old.site_gram(reference, (2, 0), fit, 128), True)
            maps = old.site_maps(moments, victims[4], study["ridge_alpha"], "quaternion_right")
            folded, kept, _ = old.compress_site(reference, (2, 0), victims[4], maps)
            payload, calibrated = calibration(study, reference, folded, maps, kept, fit, check)
            report(study, rows, "running", calibrated)
            tensors = {s: torch.as_tensor(images[ids[s]], device=device) for s in ("train", *SPLITS)}
            targets = {s: torch.as_tensor(labels[ids[s]], device=device) for s in tensors}
            original_state = {n: value.clone() for n, value in folded.state_dict().items()}
            for case in study["cases"]:
                folder = output/case
                folder.mkdir(parents=True, exist_ok=True)
                if (folder/"result.json").exists():
                    row = curve.read(folder/"result.json")
                    if file_sha256(folder/"compact_after.pt") != row["after_sha256"]: raise ValueError("Completed export changed")
                    rows.append(row)
                    continue
                if (output/"STOP").exists():
                    report(study, rows, "paused", calibrated)
                    write_json(output/"state.json", {"status": "paused", "case": case, "updated_utc": now()})
                    return
                current = deepcopy(folded)
                with torch.no_grad(): current.blocks[2][3].kernel.copy_(payload["solutions"][case]["kernel"].to(device))
                for name, value in current.state_dict().items():
                    if name != "blocks.2.3.kernel" and not torch.equal(value, original_state[name]): raise AssertionError("Calibration changed another state")
                current, head = final_compaction(current, victims, fit, check, study)
                architecture = describe(current)
                if architecture["real_parameters"] != 19098 or architecture["quaternion_convolutions"] != 6 or architecture["dense_real_convolutions"]: raise AssertionError("Compact size/family changed")
                before = {s: evaluate(current, tensors[s], targets[s], mean, std, 128)[0] for s in SPLITS}
                before_hash = save_model(current, folder/"compact_before.pt")
                probe = deepcopy(current)
                update = first_update_audit(probe, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32], torch.optim.AdamW(probe.parameters(), lr=.0001))
                del probe
                def epoch_done(row, counts):
                    write_json(output/"state.json", {"status": "running", "stage": "supervised_repair", "case": case, "completed_epoch": row["epoch"], "counts": counts, "pid": os.getpid(), "updated_utc": now()})
                    print(f"{case} epoch{row['epoch']}/3 val={100*row['validation']['accuracy']:.2f}% time={row['epoch_seconds']:.1f}s", flush=True)
                print(f"START {case}: val before={100*before['validation']['accuracy']:.2f}%, 19098 real parameters", flush=True)
                write_json(output/"state.json", {"status": "running", "stage": "supervised_repair", "case": case, "pid": os.getpid(), "updated_utc": now()})
                current, history, counts, done = repair(current, tensors, targets, mean, std, study["finetune"], study["seed"], folder/"resume.pt", output/"STOP", epoch_done)
                if not done:
                    report(study, rows, "paused", calibrated)
                    write_json(output/"state.json", {"status": "paused", "case": case, "completed_epoch": len(history), "updated_utc": now()})
                    return
                after_hash = save_model(current, folder/"compact_after.pt")
                row = {"case": case, "architecture": architecture, "before": before, "after": {s: history[-1][s] for s in SPLITS},
                       "history": history, "counts": counts, "head_fold": head, "discarded_copy_update_audit": update,
                       "before_sha256": before_hash, "after_sha256": after_hash, "test_evaluated": False}
                write_json(folder/"result.json", row)
                rows.append(row)
                report(study, rows, "running", calibrated)
                print(f"DONE {case}: {100*before['validation']['accuracy']:.2f}->{100*row['after']['validation']['accuracy']:.2f}%", flush=True)
                del current
            register(study)
            report(study, rows, "completed", calibrated)
            write_json(output/"state.json", {"status": "completed", "completed_cases": len(rows), "completed_utc": now(), "test_evaluated": False})
        except BaseException as error_value:
            write_json(output/"state.json", {"status": "failed", "error": f"{type(error_value).__name__}: {error_value}", "updated_utc": now()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    study = curve.read(ROOT/CONFIG)
    torch.set_num_threads(study["cpu_threads"])
    if args.run: run(study)
    else: prepare(study)
