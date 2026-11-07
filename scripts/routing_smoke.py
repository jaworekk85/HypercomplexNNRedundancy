"""GPU training-integrity and hard-export gate for one learned channel router."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from src.cnn_training import first_update_audit, seed_everything
from src.data.cifar_cached import file_sha256
from src.models.channel_routing import RoutedCifarCNN

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    torch.set_num_threads(4)
    seed_everything(2026)
    folder = ROOT / "results" / "learned_grouping"
    model = RoutedCifarCNN("learned").cuda()
    inputs = torch.randn(16, 3, 32, 32, device="cuda")
    labels = torch.arange(16, device="cuda") % 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    audit = first_update_audit(model, inputs, labels, optimizer)
    training_parameters = sum(p.numel() for p in model.parameters())
    diagnostics = model.router.diagnostics()
    model.eval()
    with torch.no_grad():
        soft_logits = model(inputs)
    model.router.harden()
    exported = model.export_hard()
    with torch.no_grad():
        error = float((model(inputs)-exported(inputs)).abs().max())
        hardening_shift = float((soft_logits-exported(inputs)).abs().max())
    checkpoint = {name:value.detach().cpu() for name,value in exported.state_dict().items()}
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"model": checkpoint}, folder / "smoke_model.pt")
    restored = RoutedCifarCNN("identity").cuda().eval()
    restored.load_state_dict(torch.load(folder / "smoke_model.pt", weights_only=True, map_location="cuda")["model"])
    with torch.no_grad():
        reload_error = float((restored(inputs)-exported(inputs)).abs().max())
    checks = {**audit["checks"], "correct_training_real_parameter_count": training_parameters == 74922,
              "correct_exported_real_parameter_count": sum(p.numel() for p in exported.parameters()) == 73898,
              "valid_exported_permutation": sorted(exported.router.indices.cpu().tolist()) == list(range(32)),
              "hard_export_equivalence": error == 0, "serialized_hard_model_equivalence": reload_error == 0,
              "balanced_soft_row_marginals": diagnostics["row_sum_max_error"] < 1e-4,
              "balanced_soft_column_marginals": diagnostics["column_sum_max_error"] < 1e-4}
    result = {"status": "passed" if all(checks.values()) else "failed", "seed": 2026, "device": torch.cuda.get_device_name(),
              "torch": str(torch.__version__), "checks": checks, "first_update": audit, "routing": diagnostics,
              "training_real_parameters": training_parameters, "exported_real_parameters": sum(p.numel() for p in exported.parameters()),
              "hard_export_error": error, "reload_error": reload_error, "soft_to_hard_max_logit_shift": hardening_shift,
              "scope": "Synthetic integrity check only; no CIFAR quality or compression gain.",
              "source_hashes": {name:file_sha256(ROOT / name) for name in ("scripts/routing_smoke.py", "src/models/channel_routing.py")}}
    write_json(folder / "smoke.json", result)
    print(f"Routing smoke: {result['status']}; {len(checks)} checks; real parameters {training_parameters} -> {result['exported_real_parameters']}", flush=True)
    if not all(checks.values()):
        raise AssertionError([key for key, passed in checks.items() if not passed])
