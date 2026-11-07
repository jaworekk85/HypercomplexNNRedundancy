"""Read-only CPU replay of completed map-pilot checkpoints and invariants."""
from pathlib import Path

import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.trainable_maps_pilot import CONFIG, starting_model
from src.cnn_training import evaluate, prepare_batch
from src.compression.structure_cost import make_model, describe
from src.compression.trainable_maps import attach
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def run():
    torch.set_num_threads(4)
    study = curve.read(ROOT/CONFIG)
    output = ROOT/study["output"]
    protocol = curve.read(output/"protocol.json")
    for name, expected in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
        if file_sha256(ROOT/name) != expected: raise AssertionError(f"Frozen input/source changed: {name}")
    summary = curve.read(output/"summary.json")
    if summary["status"] != "completed" or len(summary["cases"]) != 2: raise AssertionError("Incomplete pilot")
    cpu = torch.device("cpu")
    initial, c, source, _ = starting_model(study, cpu)
    images, labels, mean, std, _ = curve.load_data(curve.read(ROOT/"configs/compression_curve.json"), cpu)
    ids = curve.split_ids(source, None)
    tensors = {s: torch.as_tensor(images[ids[s]]) for s in ("validation", "check_calibration")}
    targets = {s: torch.as_tensor(labels[ids[s]]) for s in tensors}
    before = {s: evaluate(initial, tensors[s], targets[s], mean, std, 128)[0] for s in tensors}
    results = []
    for row in summary["cases"]:
        case = row["case"]
        saved = torch.load(output/case/"resume.pt", map_location="cpu", weights_only=False)
        if len(saved["history"]) != 3 or saved["count"]["attempted"] != 1125: raise AssertionError("Unexpected repair budget")
        model = attach(initial, c, case == "learned_maps").eval()
        model.load_state_dict(saved["model"])
        layer = model.blocks[2][3]
        if not torch.equal(layer.removed_kernel, c): raise AssertionError("Frozen C changed")
        delta = layer.map_delta.detach().flatten(1).norm(dim=1)
        if case == "learned_maps" and not bool((delta > 0).all()): raise AssertionError("Missing learned component")
        if case == "frozen_maps" and bool((delta != 0).any()): raise AssertionError("Control maps changed")
        if sum(p.numel() for p in model.parameters()) != row["training_reals"]: raise AssertionError("Training count mismatch")
        compact = make_model(curve.reduced_config(source["config"], "cnn6", .5), "cifar", "bn", "quaternion").eval()
        if file_sha256(ROOT/row["checkpoint"]) != row["checkpoint_sha256"]: raise AssertionError("Export checkpoint changed")
        compact.load_state_dict(torch.load(ROOT/row["checkpoint"], map_location="cpu", weights_only=True)["model"])
        architecture = describe(compact)
        if architecture["real_parameters"] != 19098 or architecture["quaternion_convolutions"] != 6 or architecture["dense_real_convolutions"]:
            raise AssertionError("Unexpected deployed family")
        if any("map_delta" in name or "removed_kernel" in name for name in compact.state_dict()): raise AssertionError("Temporary storage deployed")
        torch.testing.assert_close(layer.effective_kernel(), compact.blocks[2][3].kernel, atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            x = prepare_batch(tensors["check_calibration"][:32], mean, std)
            fold_error = float((model(x)-compact(x)).abs().max())
        if fold_error > .0001: raise AssertionError("CPU export replay failed")
        after = {s: evaluate(compact, tensors[s], targets[s], mean, std, 128)[0] for s in tensors}
        for stage, values in (("before", before), ("after", after)):
            for split in values:
                for metric, measured in values[split].items():
                    if abs(measured-row[stage][split][metric]) > (1e-12 if metric == "accuracy" else 1e-5):
                        raise AssertionError(f"Metric mismatch: {case}/{stage}/{split}/{metric}")
        results.append({"case": case, "before": before, "after": after, "map_component_norms": delta.tolist(),
                        "frozen_C_equal": True, "cpu_fold_error": fold_error, "architecture": architecture, "counts": saved["count"]})
    audit = {"status": "passed", "frozen_sources": len(protocol["source_hashes"]), "frozen_inputs": len(protocol["input_hashes"]),
             "full_cpu_split_evaluations": 6, "results": results, "test_evaluated": False}
    write_json(output/"independent_audit.json", audit)
    print(f"PASSED: {audit['frozen_sources']} frozen sources, {audit['frozen_inputs']} inputs, 6 CPU split replays", flush=True)


if __name__ == "__main__": run()
