import json

from scripts import queue_resnet20_replication as queue


def test_baseline_gate_requires_both_completed_and_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "BASELINES", tmp_path)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    a, b = tmp_path / "seed_2027", tmp_path / "seed_2028"
    write(a / "state.json", {"status": "completed"})
    write(b / "state.json", {"status": "running"})
    write(tmp_path / "completion.json", {"status": "completed", "seeds": [2027, 2028]})
    assert queue.baseline_state() == "waiting"
    write(b / "state.json", {"status": "completed"})
    assert queue.baseline_state() == "completed"
    (b / "STOP").touch()
    assert queue.baseline_state() == "baseline_paused"
    (b / "STOP").unlink()
    write(b / "state.json", {"status": "failed"})
    assert queue.baseline_state() == "baseline_failed"
    (b / "state.json").write_text("{", encoding="utf-8")
    assert queue.baseline_state() == "waiting"
