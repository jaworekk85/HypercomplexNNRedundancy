import json

import pytest

from scripts import resnet_structure_catalogue as catalogue
from src.data.cifar_cached import file_sha256


def test_hybrid_accepts_completed_replication_and_rejects_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogue, "ROOT", tmp_path)
    monkeypatch.setattr(catalogue, "rel", lambda path: str(path.relative_to(tmp_path)).replace("\\", "/"))
    def save(path, data):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding="utf-8")
    checkpoint = tmp_path / "baseline" / "quaternion" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    save("baseline/quaternion/metrics.json", {"checkpoint_sha256": file_sha256(checkpoint)})
    save("baseline/summary.json", {"status": "completed"})
    save("core/summary.json", {"status": "completed"})
    save("replication/summary.json", {"status": "running"})
    study = {"required_completed_studies": ["core/summary.json", "replication/summary.json"],
             "baseline_checkpoint": "baseline/quaternion/model.pt",
             "baseline_metrics": "baseline/quaternion/metrics.json", "output": "hybrid"}
    with pytest.raises(RuntimeError, match="Incomplete prerequisite"):
        catalogue.check_dependencies(study)
    save("replication/summary.json", {"status": "completed", "studies": []})
    catalogue.check_dependencies(study)
    assert (tmp_path / "hybrid/scientific_inputs.json").exists()
