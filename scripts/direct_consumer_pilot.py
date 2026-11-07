"""One fixed-size direct-Q consumer pilot, independently resumable per site."""
from copy import deepcopy
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel
from scripts.run_activation_queue import lock
from src.cnn_training import first_update_audit, prepare_batch, seed_everything
from src.compression import multilayer as old
from src.compression.direct_consumer import refit_consumer
from src.compression.structure_cost import describe, make_model
from src.data.cifar_cached import file_sha256
from src.finetuning import finetune

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/direct_consumer_pilot.json"
NEW = [CONFIG, "scripts/direct_consumer_pilot.py", "src/compression/direct_consumer.py", "tests_torch/test_direct_consumer.py"]


def freeze(study):
    prior = curve.read(ROOT / "results/structure_cost_study/protocol.json")
    for name, expected in {**prior["source_hashes"], **prior["input_hashes"]}.items():
        if file_sha256(ROOT / name) != expected: raise ValueError(f"Frozen prior source/input changed: {name}")
    hashes = {**prior["source_hashes"], **{n: file_sha256(ROOT / n) for n in NEW}}
    folder = ROOT / study["reference_group"]
    names = [folder / "phase_a.json"]
    for case in ("q_pruned", "q_comp", "free_q_init", "free_full_fit"):
        names += [folder / f"{case}.pt", folder / case / "before.json", folder / case / "metrics.json", folder / case / "model.pt"]
    inputs = {**prior["input_hashes"], **{rel(p): file_sha256(p) for p in names}}
    definition = {"config": study, "source_hashes": hashes, "input_hashes": inputs,
                  "environment": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}}
    path = ROOT / study["output"] / "protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("Direct-Q pilot protocol changed")
    write_json(path, definition)
    return definition


def evaluate(model, images, labels, ids, mean, std, study, device):
    return curve.metrics(model, images, labels, ids, mean, std, study, device)


def report(study, result):
    output = ROOT / study["output"]
    references = {}
    for case in ("q_pruned", "q_comp", "free_q_init", "free_full_fit"):
        folder = ROOT / study["reference_group"] / case
        references[case] = {"before": curve.read(folder / "before.json"), "after": curve.read(folder / "metrics.json")}
    summary = {**result, "references": references, "test_evaluated": False}
    write_json(output / "summary.json", summary)
    lines = ["# Direct quaternion consumer-output reconstruction pilot", "", "## Hypothesis", "", study["hypothesis"], "",
             "## Measured results", "", f"Status: {result['status']}. CIFAR BN CNN6 seed2026, .5 removed/site, same victims and 500 fit/500 check images; no test inference.", "",
             "Centered ridge fits five Q consumers, keeping all other parameters and BN buffers fixed; final real readout uses original map-fold only. Previous reference outcomes are reused. Both objective and compensation parameterization change, so this is not a pure feasible-space attribution.", "",
             "| Case | Real parameters | Before val (%) | After 3 epochs val (%) |", "| --- | ---: | ---: | ---: |"]
    for case, data in references.items():
        lines.append(f"| {case} (reused) | {data['before']['real_parameters']} | {100*data['before']['evaluation']['validation']['accuracy']:.2f} | {100*data['after']['evaluation']['validation']['accuracy']:.2f} |")
    before, after = result.get("before"), result.get("after")
    if before:
        value = "pending" if after is None else f"{100*after['validation']['accuracy']:.2f}"
        lines.append(f"| direct_q_consumer | {result['architecture']['real_parameters']} | {100*before['validation']['accuracy']:.2f} | {value} |")
    lines += ["", "### Local pre-BN consumer reconstruction", "",
              "Each pair uses identical current-reference inputs/targets; different sites follow the direct-fit path, not the old complete Q path. MSE averages over all output components and spatial positions.", "",
              "| Site | Fit folded MSE | Fit direct MSE | Check folded MSE | Check direct MSE |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["sites"]:
        if "direct_audit" not in row: continue
        a = row["direct_audit"]
        lines.append(f"| {row['site']} | {a['fit_prior_mse']:.6g} | {a['fit_direct_mse']:.6g} | {a['check_prior_mse']:.6g} | {a['check_direct_mse']:.6g} |")
    lines += ["", "## Interpretation limits", "",
              "One seed/operating point, post hoc exploratory selection; no novelty or generality claim. Local fit-objective improvement is a numerical sanity check, not evidence of held-out classification improvement. The prior is the folded kernel, not zero; this is a fixed centered-ridge objective, not the unregularized optimum. Reference and new chains diverge after the first intervention. Training labels are used only in the optional matched three-epoch repair, not regression. No additional parameters, alpha search, new baseline or test evaluation. Consumer-output LS is established prior art (He2017ChannelPruning). Attempted schedules match; successful AMP steps are counted for the new repair only, not retrospectively known for reused references."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    study = curve.read(ROOT / CONFIG)
    output = ROOT / study["output"]
    with lock("compression_curve_runner.lock"):
        protocol = freeze(study)
        torch.set_num_threads(study["cpu_threads"])
        device = torch.device(study["device"])
        seed_everything(study["seed"])
        data_study = {**curve.read(ROOT / "configs/compression_curve.json"), "architectures": ["cnn6"], "seeds": [study["seed"]]}
        images, labels, mean, std, sources = curve.load_data(data_study, device)
        source = next(s for s in sources if s["seed"] == study["seed"])
        ids = curve.split_ids(source, None)
        fit = prepare_batch(torch.as_tensor(images[ids["fit"]], device=device), mean, std)
        check = prepare_batch(torch.as_tensor(images[ids["check_calibration"]], device=device), mean, std)
        phase = curve.read(ROOT / study["reference_group"] / "phase_a.json")
        frozen = phase["frozen_victims"]
        current = curve.load_source(data_study, source, device)
        site_rows = []
        write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "started_utc": now(), "test_evaluated": False})
        try:
            for index, site in enumerate(old.sites(current)):
                folder = output / "sites" / f"site_{index:02d}"
                checkpoint, record_path = folder / "model.pt", folder / "metrics.json"
                if record_path.exists():
                    row = curve.read(record_path)
                    if file_sha256(checkpoint) != row["checkpoint_sha256"]: raise ValueError("Site checkpoint changed")
                    current, _, _ = old.compress_site(current, site, frozen[index])
                    current.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
                    site_rows.append(row)
                    continue
                moments = old.SiteMoments(*old.site_gram(current, site, fit, study["batch_size"]), True)
                maps = old.site_maps(moments, frozen[index], study["ridge_alpha"], "quaternion_right")
                folded, kept, lost = old.compress_site(current, site, frozen[index], maps)
                row = {"site": list(site), "victims": frozen[index], "fold_projection_discarded_energy": lost}
                with torch.no_grad():
                    error = float((folded(check[:32])-old.substituted_logits(current, site, maps, check[:32], study["batch_size"])).abs().max())
                if error > .001: raise AssertionError("Map-fold prefit consistency failed")
                row["fold_max_abs_error"] = error
                if index < 5:
                    current, audit, arrays = refit_consumer(current, folded, site, kept, fit, check, study["ridge_alpha"], study["batch_size"])
                    row["direct_audit"] = audit
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / "moments.npz", **arrays)
                    row["moments_sha256"] = file_sha256(folder / "moments.npz")
                    print(f"Site {site}: fit {audit['fit_prior_mse']:.6g}->{audit['fit_direct_mse']:.6g}; check {audit['check_prior_mse']:.6g}->{audit['check_direct_mse']:.6g}", flush=True)
                else:
                    current = folded
                    print(f"Site {site}: real readout, map-fold only", flush=True)
                torch.save({"model": {n: v.detach().cpu() for n, v in current.state_dict().items()}}, checkpoint)
                row["checkpoint_sha256"] = file_sha256(checkpoint)
                write_json(record_path, row)
                site_rows.append(row)
                write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "sites_done": len(site_rows), "updated_utc": now(), "test_evaluated": False})
            narrow = curve.reduced_config(source["config"], "cnn6", study["fraction"])
            reloaded = make_model(narrow, "cifar", "bn", "quaternion").to(device).eval()
            reloaded.load_state_dict(current.state_dict())
            with torch.no_grad():
                if float((reloaded(check[:32])-current(check[:32])).abs().max()) > 1e-6: raise AssertionError("Standard compact architecture reload failed")
            structure = describe(current)
            if structure["real_parameters"] != 19098 or structure["quaternion_convolutions"] != 6 or structure["dense_real_convolutions"] != 0:
                raise AssertionError("Pilot changed model size/family")
            before = evaluate(current, images, labels, ids, mean, std, data_study, device)
            torch.save({"model": {n: v.detach().cpu() for n, v in current.state_dict().items()}}, output / "direct_before.pt")
            result = {"status": "running", "sites": site_rows, "before": before, "architecture": structure,
                      "initial_checkpoint_sha256": file_sha256(output / "direct_before.pt")}
            report(study, result)
            print(f"Direct Q before repair: val {100*before['validation']['accuracy']:.2f}% ({structure['real_parameters']} params)", flush=True)
            final_path = output / "finetuned/metrics.json"
            if final_path.exists():
                final = curve.read(final_path)
                if file_sha256(final_path.parent / "model.pt") != final["checkpoint_sha256"]: raise ValueError("Repair checkpoint changed")
            else:
                tensors = {name: torch.as_tensor(images[ids[name]], device=device) for name in ("train", "validation", "check_calibration")}
                targets = {name: torch.as_tensor(labels[ids[name]], device=device) for name in tensors}
                audited = deepcopy(current)
                audit = first_update_audit(audited, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32], torch.optim.AdamW(audited.parameters(), lr=.0001))
                del audited
                counter = {"successful_optimizer_steps": 0}
                original_adamw = torch.optim.AdamW
                class CountedAdamW(original_adamw):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.register_step_post_hook(lambda optimizer, args, kwargs: counter.__setitem__("successful_optimizer_steps", counter["successful_optimizer_steps"]+1))
                with patch.object(torch.optim, "AdamW", CountedAdamW):
                    tuned, history = finetune(current, tensors, targets, mean, std, study["finetune"], study["seed"], ("validation", "check_calibration"))
                if describe(tuned) != structure: raise AssertionError("Fine-tuning changed structure")
                final_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": {n: v.detach().cpu() for n, v in tuned.state_dict().items()}}, final_path.parent / "model.pt")
                final = {"history": history, "final": {s: history[-1][s] for s in ("validation", "check_calibration")},
                         "discarded_copy_update_audit": audit, **counter,
                         "attempted_optimizer_steps": study["finetune"]["epochs"]*((len(ids["train"])+127)//128),
                         "checkpoint_sha256": file_sha256(final_path.parent / "model.pt"), "test_evaluated": False}
                write_json(final_path, final)
            result.update(status="completed", after=final["final"], repair=final)
            freeze(study)
            report(study, result)
            write_json(output / "state.json", {"status": "completed", "completed_utc": now(), "test_evaluated": False})
            print(f"Direct Q after repair: val {100*final['final']['validation']['accuracy']:.2f}%", flush=True)
        except BaseException as error:
            write_json(output / "state.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "updated_utc": now(),
                       "error": f"{type(error).__name__}: {error}", "sites_done": len(site_rows), "test_evaluated": False})
            raise


if __name__ == "__main__":
    run()
