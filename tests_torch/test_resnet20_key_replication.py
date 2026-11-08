from copy import deepcopy

import pytest

from scripts import run_resnet20_key_replication as runner
from src.data.cifar_cached import file_sha256


def test_validation_rejects_different_masks_and_tampered_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    study = {"families": ["real"], "fractions": [0.5], "policies": ["mapfold", "pruned", "direct"], "output": "out"}
    rows = []
    for policy in study["policies"]:
        folder = tmp_path / "out" / "real" / "f_500" / policy
        folder.mkdir(parents=True)
        (folder / "after.pt").write_bytes(policy.encode())
        rows.append({"family": "real", "fraction": 0.5, "policy": policy, "parameters": 12,
                     "hidden_widths": [2] * 9, "site_records": [{"victims": [1]} for _ in range(9)],
                     "counts": {"successful": 3, "attempted": 3}, "test_evaluated": False,
                     "after_sha256": file_sha256(folder / "after.pt")})
    summary = {"status": "completed", "results": rows}
    assert len(runner.validate_results(study, summary)) == 3
    changed = deepcopy(summary)
    changed["results"][1]["site_records"][0]["victims"] = [0]
    with pytest.raises(ValueError, match="victim"):
        runner.validate_results(study, changed)
    (tmp_path / "out" / "real" / "f_500" / "direct" / "after.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_results(study, summary)
