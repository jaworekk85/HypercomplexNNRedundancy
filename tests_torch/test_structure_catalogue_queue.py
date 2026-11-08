import json

from scripts import queue_resnet_structure_catalogue as queue


def test_pilot_requires_queue_and_all_completed_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    def write(path, data):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding="utf-8")
    write("configs/resnet_structure_catalogue.json",
          {"required_completed_studies": ["core.json", "replication.json"]})
    write("core.json", {"status": "completed"})
    write("replication.json", {"status": "completed"})
    state = "results/resnet20_replication_queue/state.json"
    write(state, {"status": "running_replication"})
    assert queue.predecessor_state() == "waiting"
    write(state, {"status": "completed"})
    assert queue.predecessor_state() == "completed"
    write("replication.json", {"status": "running"})
    assert queue.predecessor_state() == "waiting"
    write(state, {"status": "failed"})
    assert queue.predecessor_state() == "failed"


def test_queue_does_not_launch_completed_pilot(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "OUTPUT", tmp_path / "queue")
    monkeypatch.setattr(queue, "lock", lambda name: nullcontext())
    monkeypatch.setattr(queue, "predecessor_state", lambda: "completed")
    monkeypatch.setattr(queue, "busy", lambda name: False)
    def unexpected_launch(*args, **kwargs):
        raise AssertionError("Completed pilot must not be launched again")
    monkeypatch.setattr(queue.subprocess, "run", unexpected_launch)
    summary = tmp_path / "results/resnet_structure_catalogue/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"status": "completed"}', encoding="utf-8")
    queue.main()
    assert json.loads((tmp_path / "queue/state.json").read_text())["status"] == "completed"
