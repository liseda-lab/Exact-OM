from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pyowl_core
import pytest

from exact.ontology import OwlOntologySource, load_ontology
from exact.ontology.projection import (
    ProjectorSettings,
    cache_key,
    encoded_contract_identity,
    projector_cache_identity,
)
from exact.ontology.reasoning import reasoner_cache_identity

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


def test_projection_only_source_does_not_eagerly_build_exact_feature_indexes():
    source = load_ontology(FIXTURE)
    feature_indexes = {
        "_signature",
        "_annotations",
        "_axioms",
        "_class_view",
        "_property_view",
        "_domain_range",
        "_class_features",
        "_property_hierarchy",
        "_individual_features",
        "_data_values",
        "_annotation_property_features",
        "_excluded",
    }

    assert feature_indexes.isdisjoint(source.__dict__)
    source.configure_projector(backend="python")
    assert source.projection_edges()
    assert feature_indexes.isdisjoint(source.__dict__)

    assert source.entities()
    assert "_signature" in source.__dict__
    assert "_class_features" not in source.__dict__


def test_projection_cache_key_covers_shared_semantic_versions():
    source = load_ontology(FIXTURE)
    source.configure_projector(backend="python", profile="mowl-d993536-v1")

    assert len(source.projection_edges()) == 42
    assert len(source.projection_edges()) == 42
    keys = source._projection.cache_keys

    assert len(keys) == 1
    assert keys[0] == cache_key(
        source.owl_snapshot(),
        ProjectorSettings(backend="python"),
        method="owl2vecstar",
        include_literals=False,
    )
    assert keys[0].core_model_schema_version == pyowl_core.MODEL_SCHEMA_VERSION
    assert keys[0].core_wire_format_version == pyowl_core.WIRE_FORMAT_VERSION
    assert keys[0].projector_compiler_cache_schema
    contract = keys[0].encoded_contract
    assert contract == encoded_contract_identity()
    assert contract.core_descriptor_sha256 == (
        pyowl_core.ENCODED_STRUCTURAL_DESCRIPTOR_SHA256_V1.hex()
    )
    assert projector_cache_identity(ProjectorSettings())["encoded_contract"] == (contract.as_dict())


def test_projection_cache_key_invalidates_on_public_descriptor_change(monkeypatch):
    source = load_ontology(FIXTURE)
    settings = ProjectorSettings(backend="python")
    before = cache_key(
        source.owl_snapshot(),
        settings,
        method="owl2vecstar",
        include_literals=False,
    )

    monkeypatch.setattr(
        pyowl_core,
        "ENCODED_STRUCTURAL_DESCRIPTOR_SHA256_V1",
        b"\x00" * 32,
    )
    after = cache_key(
        source.owl_snapshot(),
        settings,
        method="owl2vecstar",
        include_literals=False,
    )

    assert before != after
    assert before.structural_fingerprint == after.structural_fingerprint


def test_reasoner_cache_identity_versions_the_mmap_worker_contract():
    identity = reasoner_cache_identity("asserted")

    assert identity["worker_schema_version"] == 2
    assert identity["encoded_contract"] == encoded_contract_identity().as_dict()["core"]
    assert identity["consumer_compiler"] == {
        "compiler_cache_schema_version": None,
        "ir_schema_version": None,
        "native_abi_version": None,
        "compatibility_id": None,
    }


def test_reasoner_cache_identity_uses_public_consumer_compiler_schemas(monkeypatch):
    import pyelk.indexing
    import pyhermit

    elk = reasoner_cache_identity("elk")
    hermit = reasoner_cache_identity("hermit")

    assert elk["consumer_compiler"] == {
        "compiler_cache_schema_version": pyelk.indexing.COMPILER_SCHEMA_VERSION,
        "ir_schema_version": None,
        "native_abi_version": None,
        "compatibility_id": pyelk.indexing.ELK_COMPATIBILITY_ID,
    }
    assert hermit["consumer_compiler"] == {
        "compiler_cache_schema_version": pyhermit.COMPILER_CACHE_SCHEMA_VERSION,
        "ir_schema_version": pyhermit.COMPILED_IR_SCHEMA_VERSION,
        "native_abi_version": pyhermit.NATIVE_ABI_VERSION,
        "compatibility_id": None,
    }

    monkeypatch.setattr(
        pyhermit,
        "COMPILER_CACHE_SCHEMA_VERSION",
        pyhermit.COMPILER_CACHE_SCHEMA_VERSION + 1,
    )
    assert reasoner_cache_identity("hermit") != hermit

    monkeypatch.setattr(pyhermit, "NATIVE_ABI_VERSION", True)
    with pytest.raises(TypeError, match="NATIVE_ABI_VERSION"):
        reasoner_cache_identity("hermit")


def test_reasoner_cache_identity_keeps_missing_extras_import_free(monkeypatch):
    import importlib.metadata

    import exact.ontology.reasoning as reasoning

    def missing_distribution(_name):
        raise importlib.metadata.PackageNotFoundError

    def unexpected_import(name):  # pragma: no cover - assertion helper
        raise AssertionError(f"optional module {name} must not be imported")

    monkeypatch.setattr(reasoning, "version", missing_distribution)
    monkeypatch.setattr(reasoning, "import_module", unexpected_import)

    identity = reasoning.reasoner_cache_identity("hermit")

    assert identity["package_version"] == "not-installed"
    assert identity["consumer_compiler"] == {
        "compiler_cache_schema_version": None,
        "ir_schema_version": None,
        "native_abi_version": None,
        "compatibility_id": None,
    }
