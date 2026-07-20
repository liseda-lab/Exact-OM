from __future__ import annotations

import os

import pytest

from exact.ontology import load_ontology
from exact.ontology.reasoning import (
    AssertedHierarchyReasoner,
    ElkHierarchyReasoner,
    HermitHierarchyReasoner,
    ReasonerSettings,
    WorkerWireHierarchyReasoner,
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
    assert provenance["core"]["wire_format_version"] == [1, 1]
    assert provenance["mmap_verified"] is False
    assert provenance["owl_parse_count"] is None


def test_elk_adapter_uses_public_facade_and_exact_snapshot(reasoning_source):
    reasoner = load_reasoner("elk", reasoning_source, backend="python")
    assert isinstance(reasoner, ElkHierarchyReasoner)
    assert reasoner.ontology is reasoning_source.owl_snapshot()
    assert reasoner.shared_reasoner.ontology is reasoning_source.owl_snapshot()
    try:
        _assert_chain(reasoner)
        provenance = reasoner.provenance
        assert provenance["backend"]["effective"] == "python"
        handoff = provenance["consumer_handoff"]
        assert handoff["ingestion_path"] == "scalar-python"
        assert handoff["compiler_digest"] is None
        assert handoff["counters"] == {}
    finally:
        reasoner.close()


def test_hermit_adapter_preserves_identity_timeout_and_narrow_results(reasoning_source):
    import pyhermit

    reasoner = load_reasoner(
        "hermit",
        reasoning_source,
        settings=ReasonerSettings(backend="python", timeout_seconds=30),
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
        assert "native_abi_version" not in handoff
        assert len(handoff["compiler_digest"]) == 64
        assert set(handoff["compiler_digest"]) <= set("0123456789abcdef")
    finally:
        reasoner.close()


@pytest.mark.parametrize("reasoner_name", ["elk", "hermit"])
def test_verified_wire_worker_matches_in_process_without_parser_calls(
    reasoning_source, tmp_path, monkeypatch, reasoner_name
):
    import pyowl_core

    encode_snapshot = pyowl_core.encode_snapshot
    encode_calls = 0

    def counted_encode(snapshot):
        nonlocal encode_calls
        encode_calls += 1
        return encode_snapshot(snapshot)

    monkeypatch.setattr(pyowl_core, "encode_snapshot", counted_encode)
    guard = tmp_path / "sitecustomize.py"
    guard.write_text(
        "import pyowl_core\n"
        "import pyowl_core.api as core_api\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('worker parser call is forbidden')\n"
        "pyowl_core.load_snapshot = forbidden\n"
        "pyowl_core.parse_document = forbidden\n"
        "core_api.load_snapshot = forbidden\n"
        "core_api.parse_document = forbidden\n",
        encoding="utf-8",
    )
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.fspath(tmp_path) if not existing else os.fspath(tmp_path) + os.pathsep + existing,
    )
    reasoner = load_reasoner(
        reasoner_name, reasoning_source, backend="python", worker_wire=True, timeout=30
    )
    assert isinstance(reasoner, WorkerWireHierarchyReasoner)
    assert reasoner.provenance["verified_wire"] is False
    assert reasoner.provenance["mmap_verified"] is False
    try:
        _assert_chain(reasoner)
        provenance = reasoner.provenance
        assert provenance["verified_wire"] is True
        assert provenance["mmap_verified"] is True
        assert provenance["owl_parse_count"] == 0
        assert provenance["options"]["worker_wire"] is True
        assert provenance["consumer_handoff"]["ingestion_path"] == "scalar-python"
        if reasoner_name == "hermit":
            assert provenance["consumer_handoff"]["compiler_cache_schema_version"] == 1
            assert provenance["consumer_handoff"]["ir_schema_version"] == 1
        assert encode_calls == 1
    finally:
        reasoner.close()


def test_source_selection_routes_class_queries_and_records_provenance(reasoning_source):
    reasoning_source.configure_reasoner("elk", backend="python")
    try:
        assert reasoning_source.direct_parents("urn:exact:test:A") == ["urn:exact:test:B"]
        assert reasoning_source.reasoner_provenance["selection"]["effective"] == "elk"
    finally:
        reasoning_source.configure_reasoner("asserted")


def test_explicit_timeout_fallback_is_visible_in_provenance(reasoning_source, monkeypatch):
    reasoner = load_reasoner(
        "elk",
        reasoning_source,
        settings=ReasonerSettings(backend="python", fallback="asserted"),
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
