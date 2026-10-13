"""A prepared G4 restart cannot activate an unbound or out-of-scope fit adapter."""

import json
from contextlib import contextmanager

import pytest

from tools import prepared_batch, recover_g4_batching


def _recipe(tmp_path, *, step="G4"):
    manifest = tmp_path / "fits.json"
    manifest.write_text("{}")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({
        "scientific_step": step,
        "g4_batched_restart": prepared_batch.binding(manifest),
    }))
    return path, manifest


def test_changed_fit_binding_prevents_worker_start(tmp_path, monkeypatch):
    path, manifest = _recipe(tmp_path)
    manifest.write_text('{"modified": true}')
    started = []
    monkeypatch.setattr(prepared_batch, "_run_recipe", lambda _: started.append(True))
    with pytest.raises(ValueError, match="Prepared binding changed"):
        prepared_batch.run_recipe(path)
    assert started == []


def test_fit_restart_cannot_be_applied_to_another_experiment(tmp_path, monkeypatch):
    path, _ = _recipe(tmp_path, step="E17")
    monkeypatch.setattr(prepared_batch, "_run_recipe", lambda _: pytest.fail("worker started"))
    with pytest.raises(ValueError, match="scoped to G4"):
        prepared_batch.run_recipe(path)


def test_worker_failure_releases_fit_adapter(tmp_path, monkeypatch):
    path, manifest = _recipe(tmp_path)
    active = []

    @contextmanager
    def adapter(bound):
        assert bound == manifest
        active.append(bound)
        try:
            yield
        finally:
            active.clear()

    def failing_worker(_):
        assert active == [manifest]
        raise RuntimeError("interrupted worker")

    monkeypatch.setattr(recover_g4_batching, "cell_configs", adapter)
    monkeypatch.setattr(prepared_batch, "_run_recipe", failing_worker)
    with pytest.raises(RuntimeError, match="interrupted worker"):
        prepared_batch.run_recipe(path)
    assert active == []


def test_other_prepared_work_does_not_install_fit_adapter(tmp_path, monkeypatch):
    path = tmp_path / "recipe.json"
    path.write_text('{"scientific_step": "E25"}')
    monkeypatch.setattr(recover_g4_batching, "cell_configs", lambda _: pytest.fail("adapter installed"))
    monkeypatch.setattr(prepared_batch, "_run_recipe", lambda _: "unchanged worker")
    assert prepared_batch.run_recipe(path) == "unchanged worker"
