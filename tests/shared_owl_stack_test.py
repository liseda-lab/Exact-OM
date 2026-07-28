from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pyowl2vec_star_projector as shared_projector
import pyowl_core
import pytest

import exact.ontology.versions as versions_module
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


def test_overlay_and_composite_views_retain_identity_through_exact_consumers():
    base = pyowl_core.load_snapshot(FIXTURE)
    target = pyowl_core.load_snapshot(FIXTURE.with_name("mini_tgt.owl"))
    overlay = pyowl_core.apply_delta(
        base,
        pyowl_core.OntologyDelta(
            add_axioms={
                pyowl_core.Declaration(pyowl_core.Class(pyowl_core.IRI("urn:exact:overlay")))
            }
        ),
    )
    composite = pyowl_core.compose_views(overlay, target, roles=("source", "target"))

    for view, owner_kind in ((overlay, "overlay"), (composite, "composite")):
        source = load_ontology(view)
        source.configure_projector(backend="python")

        assert source.owl_snapshot() is view
        assert source.reasoner.ontology is view
        assert source.projection_edges()
        assert source.projector.last_view is view
        provenance = source.ontology_stack_provenance()
        assert provenance["consumer_handoff"]["core"]["owner_kind"] == owner_kind
        closure = provenance["core"]["closure"]
        assert closure["view_kind"] == owner_kind
        assert closure["complete"] is view.is_complete
        assert closure["manifest_available"] is (owner_kind == "overlay")
        provenance_digest = (
            "overlay_provenance_sha256"
            if owner_kind == "overlay"
            else "composition_provenance_sha256"
        )
        assert closure[provenance_digest]
        if owner_kind == "composite":
            assert set(closure["member_roles"].values()) == {"source", "target"}


def test_layered_view_construction_does_not_materialize_lazy_core_state():
    base = pyowl_core.load_snapshot(FIXTURE)
    target = pyowl_core.load_snapshot(FIXTURE.with_name("mini_tgt.owl"))
    overlay = pyowl_core.apply_delta(base, pyowl_core.OntologyDelta())
    composite = pyowl_core.compose_views(overlay, target, roles=("source", "target"))
    cache_fields = {
        overlay: (
            "_structural_context_cache",
            "_fingerprint_cache",
            "_report_cache",
            "_origin_cache",
        ),
        composite: (
            "_context_cache",
            "_fingerprint_cache",
            "_report_cache",
            "_origin_cache",
            "_axiom_count_cache",
        ),
    }

    for view, fields in cache_fields.items():
        before = tuple(getattr(view, name) for name in fields)
        assert load_ontology(view).owl_snapshot() is view
        after = tuple(getattr(view, name) for name in fields)
        assert all(current is retained for current, retained in zip(after, before, strict=True))


def test_projector_encoded_native_parity_preserves_layered_view_identity():
    import pyowl2vec_star_projector as projector_package

    try:
        projector_package.select_backend("native")
    except projector_package.NativeBackendUnavailableError:
        pytest.skip("native projector is unavailable")
    base = pyowl_core.load_snapshot(FIXTURE)
    target = pyowl_core.load_snapshot(FIXTURE.with_name("mini_tgt.owl"))
    overlay = pyowl_core.apply_delta(base, pyowl_core.OntologyDelta())
    composite = pyowl_core.compose_views(overlay, target, roles=("source", "target"))

    for view in (base, overlay, composite):
        python = load_ontology(view)
        native = load_ontology(view)
        python.configure_projector(backend="python")
        native.configure_projector(backend="native")

        assert native.projection_edges() == python.projection_edges()
        assert native.projector.last_view is view
        assert python.projector.last_view is view
        native_report = native.projector.last_report.to_dict()["provenance"]
        python_report = python.projector.last_report.to_dict()["provenance"]
        assert native_report["counts"] == python_report["counts"]
        assert native_report["diagnostics_digest"] == python_report["diagnostics_digest"]
        ingestion = native_report["ingestion"]
        assert ingestion["path"] == "encoded-native"
        assert ingestion["encoded_schema_name"] == "pyowl-core/structural-columns"
        assert ingestion["encoded_schema_version"] == 1
        assert ingestion["encoded_descriptor_sha256"] == (
            pyowl_core.ENCODED_STRUCTURAL_DESCRIPTOR_SHA256_V1.hex()
        )
        counters = ingestion["counters"]
        assert counters["encoded_segment_count"] > 0
        assert counters["encoded_buffer_count"] > 0
        assert counters["encoded_compiler_gil_released"] is True
        assert counters["encoded_zero_copy_buffers"] == counters["encoded_buffer_count"]
        for name in (
            "base_flattening_bytes",
            "encoded_staging_copy_bytes",
            "materialized_scalar_rows",
            "parser_calls",
            "per_row_ffi_calls",
            "resolver_calls",
            "scalar_axiom_materializations",
            "scalar_term_materializations",
            "structural_copy_bytes",
            "wire_decoder_calls",
            "wire_encoder_calls",
        ):
            assert counters[name] == 0
        assert python_report["ingestion"]["path"] == "scalar-python"


def test_snapshot_provider_is_retained_without_loading(monkeypatch):
    snapshot = pyowl_core.load_snapshot(FIXTURE)

    class Provider:
        def owl_snapshot(self):
            return snapshot

    def unexpected(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("provider input must not be parsed")

    monkeypatch.setattr(pyowl_core, "load_snapshot", unexpected)

    source = load_ontology(Provider())

    assert source.owl_snapshot() is snapshot


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


@pytest.mark.parametrize(
    "distribution",
    ["pyowl-core", "pyowl2vec-star-projector"],
)
def test_projection_cache_identity_rejects_module_distribution_version_drift(
    monkeypatch,
    distribution,
):
    module_versions = {
        "pyowl-core": pyowl_core.__version__,
        "pyowl2vec-star-projector": shared_projector.__version__,
    }

    def installed_version(name):
        return "0.1.1" if name == distribution else module_versions[name]

    monkeypatch.setattr(versions_module, "version", installed_version)

    with pytest.raises(RuntimeError, match="module/distribution version mismatch"):
        projector_cache_identity(ProjectorSettings())


def test_reasoner_cache_identity_versions_the_mmap_worker_contract():
    identity = reasoner_cache_identity("asserted")

    assert identity["worker_schema_version"] == 3
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
