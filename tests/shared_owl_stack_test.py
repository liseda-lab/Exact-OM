from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pyowl_core
import pytest

from exact.ontology import OwlOntologySource, load_ontology

FIXTURE = Path(__file__).parent / "fixtures" / "ontologies" / "mini_src.owl"


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: FIXTURE, id="path"),
        pytest.param(lambda: FIXTURE.read_bytes(), id="bytes"),
        pytest.param(lambda: BytesIO(FIXTURE.read_bytes()), id="stream"),
    ],
)
def test_acquisition_loads_one_snapshot_and_preserves_provider_identity(monkeypatch, factory):
    calls = 0
    actual = pyowl_core.load_snapshot

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(pyowl_core, "load_snapshot", counted)
    source = load_ontology(factory())

    assert calls == 1
    assert source.owl_snapshot() is pyowl_core.coerce_snapshot(source)


def test_existing_snapshot_is_never_reloaded_and_projector_sees_same_object(monkeypatch):
    snapshot = pyowl_core.load_snapshot(FIXTURE)

    def unexpected(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("existing snapshot must not be parsed again")

    monkeypatch.setattr(pyowl_core, "load_snapshot", unexpected)
    source = OwlOntologySource(snapshot)
    before = (
        snapshot.structural_fingerprint,
        snapshot.logical_fingerprint,
        snapshot.signature_fingerprint,
        tuple(snapshot.iter_axioms()),
    )

    assert source.owl_snapshot() is snapshot
    source.projection_edges()
    assert source.projector.last_view is snapshot
    assert before == (
        snapshot.structural_fingerprint,
        snapshot.logical_fingerprint,
        snapshot.signature_fingerprint,
        tuple(snapshot.iter_axioms()),
    )
