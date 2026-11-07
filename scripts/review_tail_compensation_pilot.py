"""Read-only CPU checkpoint/prediction audit for the bounded late-consumer pilot."""
from pathlib import Path

import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.tail_compensation_pilot import intermediate
from src.cnn_training import prepare_batch
from src.compression import multilayer as old
from src.compression.structure_cost import describe, make_model
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/tail_compensation_pilot"


@torch.no_grad()
def full_forward_errors(reference, student, inputs, batch_size=32):
    values = {}
    hooks = [old.consumer(reference, (2, 0)).register_forward_hook(lambda m, x, y: values.__setitem__("teacher", y)),
             old.consumer(student, (2, 0)).register_forward_hook(lambda m, x, y: values.__setitem__("student", y))]
    local = logits = 0.
    local_count = logits_count = 0
    try:
        for x in inputs.split(batch_size):
            expected, actual = reference(x), student(x)
            a, b = values["teacher"], values["student"]
            local += float((a-b).double().square().sum()); local_count += a.numel()
            logits += float((expected-actual).double().square().sum()); logits_count += expected.numel()
    finally:
        for handle in hooks: handle.remove()
    return {"local_mse": local/local_count, "logit_mse": logits/logits_count}


def run():
    torch.set_num_threads(4)
    summary = curve.read(OUTPUT / "summary.json")
    protocol = curve.read(OUTPUT / "protocol.json")
    if summary["status"] != "completed" or len(summary["results"]) != 2: raise ValueError("Incomplete pilot")
    for name, h in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
        if file_sha256(ROOT / name) != h: raise ValueError(f"Frozen source/input changed: {name}")
    study = summary["config"]
    phase = curve.read(ROOT / study["reference_group"] / "phase_a.json")
    victims = phase["frozen_victims"]
    data = curve.read(ROOT / "configs/compression_curve.json")
    images, labels, mean, std, sources = curve.load_data(data, torch.device("cpu"))
    source = next(s for s in sources if s["seed"] == 2026 and s["architecture"] == "cnn6")
    ids = curve.split_ids(source, None)
    teacher = intermediate(study, source, victims, 3, "cpu")
    original = intermediate(study, source, victims, 4, "cpu")
    normal = {name: prepare_batch(torch.as_tensor(images[ids[name]]), mean, std) for name in ("fit", "check_calibration")}
    rows = []
    for result in summary["results"]:
        for key in ("intermediate_checkpoint", "before_checkpoint", "after_checkpoint"):
            if file_sha256(ROOT / result[key]) != result[key+"_sha256"]: raise AssertionError("Checkpoint hash mismatch")
        calibrated = intermediate(study, source, victims, 4, "cpu")
        calibrated.load_state_dict(torch.load(ROOT / result["intermediate_checkpoint"], map_location="cpu", weights_only=True)["model"])
        old_state = original.state_dict()
        changed = [k for k, v in calibrated.state_dict().items() if not torch.equal(v, old_state[k])]
        if changed != ["blocks.2.3.kernel"]: raise AssertionError(f"Unexpected calibrated state changes: {changed}")
        if not all(v > 0 for v in result["calibration"]["first_update"]["gradient_component_norms"]+result["calibration"]["first_update"]["update_component_norms"]):
            raise AssertionError("Missing component update")
        if result["calibration"]["labels_used"] or result["calibration"]["steps"] != 200: raise AssertionError("Calibration protocol mismatch")
        if result["final_head"]["victims"] != victims[5]: raise AssertionError("Final mask changed")
        checked_calibration = {}
        for split, stored in (("fit", "fit"), ("check_calibration", "check")):
            values = full_forward_errors(teacher, calibrated, normal[split])
            for name, value in values.items():
                if abs(value-result["calibration"]["final"][stored][name]) > 2e-5: raise AssertionError("CPU calibration replay differs")
            checked_calibration[split] = values
        replay = {}
        for stage in ("before", "after"):
            model = make_model(curve.reduced_config(source["config"], "cnn6", .5), "cifar", "bn", "quaternion").eval()
            model.load_state_dict(torch.load(ROOT / result[stage+"_checkpoint"], map_location="cpu", weights_only=True)["model"])
            shape = describe(model)
            if shape != result["architecture"] or shape["real_parameters"] != 19098 or shape["quaternion_convolutions"] != 6 or shape["dense_real_convolutions"]:
                raise AssertionError("Actual checkpoint architecture changed")
            if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()): raise AssertionError("Nonfinite checkpoint")
            values = curve.metrics(model, images, labels, ids, mean, std, data, "cpu")
            for split in ("validation", "check_calibration"):
                if values[split]["accuracy"] != result[stage][split]["accuracy"] or abs(values[split]["cross_entropy"]-result[stage][split]["cross_entropy"]) > 2e-5:
                    raise AssertionError("Independent CPU classification replay differs")
            replay[stage] = values
        rows.append({"case": result["case"], "only_target_Q_kernel_changed_before_final_fold": True,
                     "same_masks": True, "same_actual_compact_architecture": True, "classification_replay": replay, "calibration_full_forward_replay": checked_calibration})
    audit = {"status": "passed", "source_input_hashes_verified": True, "rows": rows, "test_evaluated": False,
             "scope": "CPU-loaded checkpoints; full validation/check accuracy and CE replay before/after repair, full-forward fit/check calibration-MSE replay, actual structure/mask/buffer invariants. No test inference or new training."}
    write_json(OUTPUT / "independent_audit.json", audit)
    path = OUTPUT / "REPORT.md"
    marker = "\n## Independent review\n"
    text = path.read_text(encoding="utf-8").split(marker)[0]
    text += marker+"\nCPU replay passed for both cases: before/after validation and check classification, fit/check local and logit MSE, matched masks, actual compact architecture and only-target changes during calibration. Frozen sources/inputs intact. See independent_audit.json.\n\n"
    text += "Measured late-consumer pilot: tail logits improve pre-repair validation by +1.9 pp over equal-step local refinement (36.8 vs 34.9%), but after three epochs give -0.2 pp (56.6 vs 56.8%). Against reused direct-Q: +2.0 pp before, -0.1 pp after. One seed/consumer; these small post-repair differences establish neither superiority nor equivalence. No automatic replication; the practical after-repair benefit was not demonstrated here.\n"
    path.write_text(text, encoding="utf-8")
    print("Independent CPU audit passed: 2 cases, 4 compact checkpoints, 8 classification evaluations and 4 full-forward calibration replays.")


if __name__ == "__main__":
    run()
