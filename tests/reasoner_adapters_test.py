from __future__ import annotations

from types import SimpleNamespace

import pyowl_core
import pytest

import exact.ontology.reasoning as reasoning_module
import exact.ontology.versions as versions_module
from exact.ontology import load_ontology
from exact.ontology.reasoning import (
    AssertedHierarchyReasoner,
    ElkHierarchyReasoner,
    HermitHierarchyReasoner,
    ReasonerSettings,
    ReasonerUnavailableError,
    load_reasoner,
)

_ONTOLOGY = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Ontology rdf:about="urn:exact:test:reasoning"/>
  <owl:Class rdf:about="urn:exact:test:A">
    <rdfs:subClassOf rdf:resource="urn:exact:test:B"/>
  </owl:Class>
  <owl:Class rdf:about="urn:exact:test:B">
    <rdfs:subClassOf rdf:resource="urn:exact:test:C"/>
  </owl:Class>
  <owl:Class rdf:about="urn:exact:test:C"/>
</rdf:RDF>
"""


def _compiler_handoff() -> dict[str, object]:
    encoded_view = pyowl_core.EncodedStructuralView
    return {
        "schema_name": encoded_view.SCHEMA_NAME,
        "schema_version": encoded_view.SCHEMA_VERSION,
        "model_schema": pyowl_core.MODEL_SCHEMA_VERSION,
        "descriptor_sha256": encoded_view.DESCRIPTOR_SHA256.hex(),
        "buffer_widths": {
            "field_kinds": 1,
            "field_lengths": 8,
            "field_values": 8,
            "item_kinds": 1,
            "item_lengths": 8,
            "item_values": 8,
            "node_field_offsets": 8,
            "node_tags": 2,
            "root_ids": 4,
            "root_kinds": 1,
            "scalar_bytes": 1,
        },
    }


@pytest.fixture(scope="module")
def reasoning_source():
    return load_ontology(_ONTOLOGY)


def _assert_chain(reasoner) -> None:
    assert reasoner.direct_parents("urn:exact:test:A") == ["urn:exact:test:B"]
    assert reasoner.direct_children("urn:exact:test:C") == ["urn:exact:test:B"]
    assert reasoner.ancestors("urn:exact:test:A") == {
        "urn:exact:test:B",
        "urn:exact:test:C",
    }
    assert reasoner.descendants("urn:exact:test:C") == {
        "urn:exact:test:A",
        "urn:exact:test:B",
    }


def test_asserted_mode_keeps_exact_snapshot_and_core_provenance(reasoning_source):
    reasoner = load_reasoner("asserted", reasoning_source)
    assert isinstance(reasoner, AssertedHierarchyReasoner)
    assert reasoner.ontology is reasoning_source.owl_snapshot()
    _assert_chain(reasoner)
    provenance = reasoner.provenance
    assert provenance["selection"]["effective"] == "asserted"
    assert provenance["fingerprints"]["structural"].endswith(
        reasoning_source.owl_snapshot().structural_fingerprint.hex
    )
    assert provenance["core"]["wire_format_version"] == list(pyowl_core.WIRE_FORMAT_VERSION)
    assert provenance["mmap_verified"] is False
    assert provenance["owl_parse_count"] is None


def test_asserted_mode_never_imports_optional_reasoner_distributions(
    reasoning_source,
    monkeypatch,
):
    def unexpected_import(name):
        raise AssertionError(f"asserted mode imported optional module {name}")

    monkeypatch.setattr(reasoning_module, "import_module", unexpected_import)

    reasoner = load_reasoner("asserted", reasoning_source)

    assert isinstance(reasoner, AssertedHierarchyReasoner)
    assert reasoner.ontology is reasoning_source.owl_snapshot()
    _assert_chain(reasoner)


@pytest.mark.parametrize("reasoner_name", ["elk", "hermit"])
def test_missing_optional_reasoner_distribution_fails_actionably(
    reasoning_source,
    monkeypatch,
    reasoner_name,
):
    def missing_module(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(reasoning_module, "import_module", missing_module)

    with pytest.raises(ReasonerUnavailableError, match="reasoning.*extra"):
        load_reasoner(reasoner_name, reasoning_source, backend="auto")


def test_elk_adapter_uses_public_facade_and_exact_snapshot(reasoning_source):
    reasoner = load_reasoner("elk", reasoning_source, backend="auto")
    assert isinstance(reasoner, ElkHierarchyReasoner)
    assert reasoner.ontology is reasoning_source.owl_snapshot()
    assert reasoner.shared_reasoner.ontology is reasoning_source.owl_snapshot()
    try:
        _assert_chain(reasoner)
        provenance = reasoner.provenance
        assert provenance["backend"]["effective"] == "rust"
        handoff = provenance["consumer_handoff"]
        assert handoff["ingestion_path"] == "encoded-native"
        assert len(handoff["compiler_digest"]) == 64
        assert handoff["consumer_compile_seconds"] >= 0.0
        assert handoff["counters"]["materialized_scalar_rows"] == 0
        assert handoff["counters"]["native_result_validation"] is True
        assert handoff["counters"]["encoded_staging_copy_bytes"] == 0
    finally:
        reasoner.close()


def test_hermit_adapter_preserves_identity_timeout_and_narrow_results(reasoning_source):
    import pyhermit

    reasoner = load_reasoner(
        "hermit",
        reasoning_source,
        settings=ReasonerSettings(backend="auto", timeout_seconds=30),
    )
    assert isinstance(reasoner, HermitHierarchyReasoner)
    assert reasoner.ontology is reasoning_source.owl_snapshot()
    assert reasoner.shared_reasoner.ontology is reasoning_source.owl_snapshot()
    try:
        _assert_chain(reasoner)
        provenance = reasoner.provenance
        assert provenance["options"]["timeout_seconds"] == 30.0
        assert provenance["timed_out"] is False
        handoff = provenance["consumer_handoff"]
        assert handoff["compiler_cache_schema_version"] == pyhermit.COMPILER_CACHE_SCHEMA_VERSION
        assert handoff["ir_schema_version"] == pyhermit.COMPILED_IR_SCHEMA_VERSION
        assert handoff["implementation_version"] == provenance["backend"]["implementation_version"]
        assert handoff["native_abi_version"] == pyhermit.NATIVE_ABI_VERSION
        assert len(handoff["compiler_digest"]) == 64
        assert set(handoff["compiler_digest"]) <= set("0123456789abcdef")
        assert handoff["consumer_compile_seconds"] >= 0.0
        assert handoff["counters"]["native_result_validation"] is True
    finally:
        reasoner.close()


def test_hermit_compiler_digest_is_backend_independent(reasoning_source):
    import pyhermit

    if not pyhermit.backend_info().native.available:
        pytest.skip("pyHermiT native backend is unavailable")
    diagnostics = {}
    for backend in ("python", "native", "verify"):
        # The upstream default APIs remain the small scalar/native oracle.
        reasoner = pyhermit.Reasoner(
            reasoning_source.owl_snapshot(),
            config=pyhermit.ReasonerConfig(backend=backend, timeout=30),
        )
        try:
            assert reasoning_module._hermit_query(
                reasoner, "urn:exact:test:A", upward=True, direct=True
            ) == {"urn:exact:test:B"}
            diagnostics[backend] = reasoning_module._consumer_handoff(reasoner).as_dict()
        finally:
            reasoner.dispose()

    assert {values["compiler_digest"] for values in diagnostics.values()} == {
        diagnostics["python"]["compiler_digest"]
    }
    assert diagnostics["python"]["ingestion_path"] == "scalar-python"
    assert all(value in {0, False} for value in diagnostics["python"]["counters"].values())
    for backend in ("native", "verify"):
        handoff = diagnostics[backend]
        assert handoff["ingestion_path"] == "encoded-native"
        assert handoff["native_abi_version"] == pyhermit.NATIVE_ABI_VERSION
        assert handoff["encoded_schema"] == _compiler_handoff()
        counters = handoff["counters"]
        assert counters["encoded_buffer_bytes"] > 0
        assert counters["encoded_buffer_count"] > 0
        assert counters["encoded_compiler_gil_released"] is True
        assert (
            counters["encoded_detached_buffer_count"]
            == counters["encoded_buffer_count"]
            == counters["encoded_zero_copy_buffers"]
        )
        assert counters["encoded_segment_count"] > 0
        assert counters["encoded_staging_copy_bytes"] == 0


@pytest.mark.parametrize(
    ("reasoner_name", "module_name", "cleanup_name"),
    [
        ("elk", "pyelk", "close"),
        ("hermit", "pyhermit", "dispose"),
    ],
)
def test_rejected_consumer_handoff_releases_session_and_retry_is_clean(
    reasoning_source,
    monkeypatch,
    reasoner_name,
    module_name,
    cleanup_name,
):
    module = __import__(module_name)
    cleanup = getattr(module.Reasoner, cleanup_name)
    released = []

    def tracked_cleanup(reasoner):
        released.append(reasoner)
        cleanup(reasoner)

    original_handoff = reasoning_module._consumer_handoff
    attempts = 0

    def reject_once(reasoner):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("incompatible native compiler handoff")
        return original_handoff(reasoner)

    monkeypatch.setattr(module.Reasoner, cleanup_name, tracked_cleanup)
    monkeypatch.setattr(reasoning_module, "_consumer_handoff", reject_once)

    with pytest.raises(ValueError, match="incompatible native compiler handoff"):
        load_reasoner(reasoner_name, reasoning_source, backend="auto")
    assert len(released) == 1

    retry = load_reasoner(reasoner_name, reasoning_source, backend="auto")
    try:
        _assert_chain(retry)
    finally:
        retry.close()
    assert attempts >= 2
    assert len(released) == 2


def test_reasoner_module_and_distribution_version_drift_fails_closed(
    reasoning_source,
    monkeypatch,
):
    import pyhermit

    cleanup = pyhermit.Reasoner.dispose
    released = []

    def tracked_cleanup(reasoner):
        released.append(reasoner)
        cleanup(reasoner)

    installed_version = versions_module.version
    monkeypatch.setattr(pyhermit.Reasoner, "dispose", tracked_cleanup)
    monkeypatch.setattr(
        versions_module,
        "version",
        lambda distribution: (
            "0.1.1" if distribution == "pyHermiT" else installed_version(distribution)
        ),
    )

    with pytest.raises(RuntimeError, match="module/distribution version mismatch"):
        load_reasoner("hermit", reasoning_source, backend="auto")
    assert len(released) == 1


def test_encoded_reasoner_handoff_records_exact_public_core_schema() -> None:
    schema = _compiler_handoff()
    reasoner = SimpleNamespace(
        backend=SimpleNamespace(compiler_handoff=schema),
        diagnostics=lambda: {
            "ingestion_path": "encoded-native",
            "compiler_digest": "0" * 64,
        },
    )

    handoff = reasoning_module._consumer_handoff(reasoner)

    assert handoff is not None
    assert handoff.as_dict()["encoded_schema"] == schema
    assert reasoning_module._consumer_handoff_from_record(handoff.as_dict()) == handoff


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {"schema_name": "pyowl-core/structural-columns"},
        {**_compiler_handoff(), "schema_version": True},
        {
            **_compiler_handoff(),
            "buffer_widths": {
                **_compiler_handoff()["buffer_widths"],
                "root_ids": 8,
            },
        },
    ],
)
def test_encoded_reasoner_handoff_rejects_missing_or_incompatible_schema(
    schema: object,
) -> None:
    backend = SimpleNamespace() if schema is None else SimpleNamespace(compiler_handoff=schema)
    reasoner = SimpleNamespace(
        backend=backend,
        diagnostics=lambda: {
            "ingestion_path": "encoded-native",
            "compiler_digest": None,
        },
    )

    with pytest.raises((TypeError, ValueError), match="compiler_handoff|width|schema_version"):
        reasoning_module._consumer_handoff(reasoner)


@pytest.mark.parametrize("reasoner_name", ["elk", "hermit"])
def test_strict_reasoners_reject_unsupported_overlay_and_composite_views(
    reasoning_source, reasoner_name
):
    import pyowl_core

    base = reasoning_source.owl_snapshot()
    overlay = pyowl_core.apply_delta(
        base,
        pyowl_core.OntologyDelta(
            add_axioms={
                pyowl_core.Declaration(pyowl_core.Class(pyowl_core.IRI("urn:exact:test:Overlay")))
            }
        ),
    )
    second = pyowl_core.load_snapshot(_ONTOLOGY, document_iri="urn:exact:test:second")
    composite = pyowl_core.compose_views(overlay, second, roles=("source", "target"))

    for view in (overlay, composite):
        source = load_ontology(view)
        with pytest.raises((pyowl_core.AdapterCompatibilityError, pyowl_core.BackendProtocolError)):
            load_reasoner(reasoner_name, source)


@pytest.mark.parametrize("reasoner_name", ["elk", "hermit"])
def test_unready_wire_worker_is_rejected_before_serialization(
    reasoning_source, monkeypatch, reasoner_name
):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported wire worker serialized the ontology")

    monkeypatch.setattr(pyowl_core, "encode_snapshot", forbidden)
    with pytest.raises(ReasonerUnavailableError, match="verified-wire"):
        load_reasoner(reasoner_name, reasoning_source, worker_wire=True, timeout=30)


def test_source_selection_routes_class_queries_and_records_provenance(reasoning_source):
    reasoning_source.configure_reasoner("elk", backend="auto")
    try:
        assert reasoning_source.direct_parents("urn:exact:test:A") == ["urn:exact:test:B"]
        assert reasoning_source.reasoner_provenance["selection"]["effective"] == "elk"
    finally:
        reasoning_source.configure_reasoner("asserted")


def test_explicit_timeout_fallback_is_visible_in_provenance(reasoning_source, monkeypatch):
    reasoner = load_reasoner(
        "elk",
        reasoning_source,
        settings=ReasonerSettings(backend="auto", fallback="asserted"),
    )

    def timeout(*_args, **_kwargs):
        raise TimeoutError("/tmp/private/ontology.pyocore test deadline")

    monkeypatch.setattr(reasoner, "_query", timeout)
    try:
        assert reasoner.direct_parents("urn:exact:test:A") == ["urn:exact:test:B"]
        provenance = reasoner.provenance
        assert provenance["selection"]["effective"] == "asserted"
        assert provenance["options"]["fallback"] == "asserted"
        assert provenance["timed_out"] is True
        assert "test deadline" in provenance["fallback_reason"]
        assert "/tmp/private" not in provenance["fallback_reason"]
    finally:
        reasoner.close()


@pytest.mark.parametrize("reasoner_name", ["elk", "hermit"])
def test_exact_rejects_python_reasoning_before_construction(
    reasoning_source, monkeypatch, reasoner_name
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Python reasoner was constructed")

    monkeypatch.setattr(reasoning_module, "_optional_module", forbidden)
    with pytest.raises(ValueError, match="backend must be"):
        load_reasoner(reasoner_name, reasoning_source, backend="python")


def test_failed_native_result_attestation_does_not_publish(reasoning_source, monkeypatch):
    reasoner = load_reasoner("elk", reasoning_source)
    cls = type(reasoner.shared_reasoner)
    diagnostics = cls.diagnostics

    def unverified(self):
        return {**diagnostics(self), "native_result_validation": False}

    monkeypatch.setattr(cls, "diagnostics", unverified)
    try:
        with pytest.raises(RuntimeError, match="verify the required native pipeline"):
            reasoner.direct_parents("urn:exact:test:A")
    finally:
        reasoner.close()
