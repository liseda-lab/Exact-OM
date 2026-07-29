"""Executable final-stack parity over every public pyowl-core owner kind."""

from __future__ import annotations

import gc
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import pyowl2vec_star_projector
import pyowl_core
import pyowl_core.api as core_api
import pytest

from exact.ontology import load_ontology
from exact.ontology.provenance import ontology_stack_provenance
from exact.ontology.reasoning import load_reasoner

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "release" / "core-compatibility.json").read_text(encoding="utf-8"))
PROJECTOR_COUNTERS = frozenset(CONTRACT["projector_handoff"]["required_counters"])
PROJECTOR_SCALAR_COUNTERS = frozenset(CONTRACT["projector_handoff"]["scalar_counter_ledger"])
REASONER_COUNTERS = frozenset(CONTRACT["reasoner_handoff"]["required_counters"])
FORBIDDEN_NATIVE_WORK = frozenset(CONTRACT["reasoner_handoff"]["required_zero_counters"])
OWNER_KINDS = ("direct", "decoded", "mmap", "overlay", "composite")

ONTOLOGY = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Ontology rdf:about="urn:exact:test:public-handoff"/>
  <owl:Class rdf:about="urn:exact:test:A">
    <rdfs:subClassOf rdf:resource="urn:exact:test:B"/>
  </owl:Class>
  <owl:Class rdf:about="urn:exact:test:B">
    <rdfs:subClassOf rdf:resource="urn:exact:test:C"/>
  </owl:Class>
  <owl:Class rdf:about="urn:exact:test:C"/>
</rdf:RDF>
"""


@contextmanager
def _public_owners(
    directory: Path,
) -> Iterator[dict[str, pyowl_core.OntologyView]]:
    direct = pyowl_core.load_snapshot(
        ONTOLOGY,
        document_iri="urn:exact:test:public-handoff",
    )
    wire = pyowl_core.encode_snapshot(direct)
    decoded = pyowl_core.decode_snapshot(wire, verify=True)
    wire_path = directory / "public-owner.pyocore"
    wire_path.write_bytes(wire)
    mapped = pyowl_core.open_snapshot(wire_path, mmap=True, verify=True)
    overlay = pyowl_core.apply_delta(direct, pyowl_core.OntologyDelta())
    composite = pyowl_core.compose_views(
        direct,
        decoded,
        roles=("primary", "equivalent"),
    )
    owners: dict[str, pyowl_core.OntologyView] = {
        "direct": direct,
        "decoded": decoded,
        "mmap": mapped,
        "overlay": overlay,
        "composite": composite,
    }
    try:
        yield owners
    finally:
        gc.collect()
        active_error = sys.exc_info()[0] is not None
        try:
            mapped.close()
        except pyowl_core.SnapshotInUseError:
            if not active_error:
                raise


@contextmanager
def _forbid_exact_conversion() -> Iterator[None]:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Exact crossed a parse, wire, or materialization boundary")

    with ExitStack() as stack:
        for module in (pyowl_core, core_api):
            for name in (
                "decode_snapshot",
                "encode_snapshot",
                "load_snapshot",
                "open_snapshot",
                "parse_document",
            ):
                if hasattr(module, name):
                    stack.enter_context(mock.patch.object(module, name, side_effect=forbidden))
        stack.enter_context(
            mock.patch.object(
                pyowl_core.OntologyOverlay,
                "materialize",
                side_effect=forbidden,
            )
        )
        stack.enter_context(
            mock.patch.object(
                pyowl_core.OntologyComposite,
                "materialize",
                side_effect=forbidden,
            )
        )
        yield


def _assert_advertised_ledger(
    counters: Mapping[str, object],
    expected: frozenset[str],
) -> None:
    assert set(counters) == expected
    for name, value in counters.items():
        if name == "encoded_compiler_gil_released":
            # This closure verifies the public claim's type, not a performance result.
            assert type(value) is bool
        else:
            assert type(value) is int
            assert value >= 0
    assert {name: counters[name] for name in FORBIDDEN_NATIVE_WORK} == {
        name: 0 for name in FORBIDDEN_NATIVE_WORK
    }
    buffer_count = counters["encoded_buffer_count"]
    assert buffer_count > 0
    assert counters["encoded_buffer_bytes"] > 0
    assert counters["encoded_segment_count"] > 0
    assert counters["encoded_zero_copy_buffers"] == buffer_count
    assert counters["encoded_referenced_view_count"] <= counters["encoded_segment_count"]


def _assert_scalar_encoded_resources_are_empty(counters: Mapping[str, object]) -> None:
    for name, value in counters.items():
        if not name.startswith("encoded_"):
            continue
        expected: int | bool = False if name == "encoded_compiler_gil_released" else 0
        assert value == expected
        assert type(value) is type(expected)


def _project(
    view: pyowl_core.OntologyView,
    backend: str,
) -> tuple[tuple[tuple[str, str, str], ...], dict[str, object]]:
    source = load_ontology(view)
    source.configure_projector(backend=backend)
    edges = tuple(edge.astuple() for edge in source.projection_edges())
    assert source.owl_snapshot() is view
    assert source.projector.last_view is view
    return edges, source.ontology_stack_provenance()


def _reason(
    view: pyowl_core.OntologyView,
    reasoner_name: str,
    backend: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source = load_ontology(view)
    options = {"timeout": 30} if reasoner_name == "hermit" else {}
    reasoner = load_reasoner(reasoner_name, source, backend=backend, **options)
    assert reasoner.ontology is view
    assert reasoner.shared_reasoner.ontology is view
    try:
        result = {
            "direct_parents": reasoner.direct_parents("urn:exact:test:A"),
            "direct_children": reasoner.direct_children("urn:exact:test:C"),
            "ancestors": sorted(reasoner.ancestors("urn:exact:test:A")),
            "descendants": sorted(reasoner.descendants("urn:exact:test:C")),
        }
        reasoner_provenance = reasoner.provenance
        stack = ontology_stack_provenance(
            view,
            projector_settings=source.projector_settings,
            projector=source.projector,
            reasoner=reasoner_provenance,
        )
        return result, stack
    finally:
        reasoner.close()


def test_projector_public_owner_matrix_preserves_identity_results_and_ledgers(
    tmp_path: Path,
) -> None:
    try:
        pyowl2vec_star_projector.select_backend("native")
    except pyowl2vec_star_projector.NativeBackendUnavailableError:
        pytest.skip("advertised native projector is unavailable")

    expected_edges: tuple[tuple[str, str, str], ...] | None = None
    semantic_fingerprints: set[tuple[str, str]] = set()
    with _public_owners(tmp_path) as owners:
        assert tuple(owners) == OWNER_KINDS
        for owner_kind, view in owners.items():
            with _forbid_exact_conversion():
                scalar_edges, scalar = _project(view, "python")
                native_edges, native = _project(view, "native")
            expected_edges = scalar_edges if expected_edges is None else expected_edges
            assert scalar_edges == native_edges == expected_edges
            assert scalar["consumer_handoff"]["core"]["owner_kind"] == owner_kind
            assert native["consumer_handoff"]["core"]["owner_kind"] == owner_kind

            scalar_handoff = scalar["consumer_handoff"]["projector"]
            native_handoff = native["consumer_handoff"]["projector"]
            assert scalar_handoff["ingestion_path"] == "scalar-python"
            _assert_scalar_encoded_resources_are_empty(scalar_handoff["counters"])
            expected_path = CONTRACT["projector_handoff"]["tested_owner_ingestion_paths"][
                owner_kind
            ]
            assert native_handoff["ingestion_path"] == expected_path
            if expected_path == "encoded-native":
                _assert_advertised_ledger(native_handoff["counters"], PROJECTOR_COUNTERS)
                assert native_handoff["schema_name"] == CONTRACT["encoded_contract"]["schema_name"]
                assert (
                    native_handoff["schema_version"]
                    == CONTRACT["encoded_contract"]["schema_version"]
                )
                assert (
                    native_handoff["descriptor_sha256"]
                    == CONTRACT["encoded_contract"]["descriptor_sha256"]
                )
            else:
                assert owner_kind == "mmap"
                assert set(native_handoff["counters"]) == PROJECTOR_SCALAR_COUNTERS
                _assert_scalar_encoded_resources_are_empty(native_handoff["counters"])
                fallback_reason = native["projector"]["last_projection"]["provenance"]["ingestion"][
                    "reason"
                ]
                assert "zero-copy retention requires" in fallback_reason
            semantic_fingerprints.add(
                (
                    native["core"]["fingerprints"]["logical"],
                    native["core"]["fingerprints"]["signature"],
                )
            )
            gc.collect()
    assert expected_edges
    assert len(semantic_fingerprints) == 1


@pytest.mark.parametrize(
    ("reasoner_name", "scalar_backend", "native_backend"),
    [
        pytest.param("elk", "python", "rust", id="pyelk"),
        pytest.param("hermit", "python", "native", id="pyhermit"),
    ],
)
def test_reasoner_public_owner_matrix_preserves_identity_results_and_ledgers(
    tmp_path: Path,
    reasoner_name: str,
    scalar_backend: str,
    native_backend: str,
) -> None:
    if reasoner_name == "elk":
        import pyelk

        if pyelk.backend_report().rust.available is not True:
            pytest.skip("advertised pyELK native backend is unavailable")
    else:
        import pyhermit

        if not pyhermit.backend_info().native.available:
            pytest.skip("advertised pyHermiT native backend is unavailable")

    expected = {
        "direct_parents": ["urn:exact:test:B"],
        "direct_children": ["urn:exact:test:B"],
        "ancestors": ["urn:exact:test:B", "urn:exact:test:C"],
        "descendants": ["urn:exact:test:A", "urn:exact:test:B"],
    }
    semantic_fingerprints: set[tuple[str, str]] = set()
    with _public_owners(tmp_path) as owners:
        assert tuple(owners) == OWNER_KINDS
        for owner_kind, view in owners.items():
            with _forbid_exact_conversion():
                scalar_result, scalar = _reason(view, reasoner_name, scalar_backend)
                native_result, native = _reason(view, reasoner_name, native_backend)
            assert scalar_result == native_result == expected
            assert scalar["consumer_handoff"]["core"]["owner_kind"] == owner_kind
            assert native["consumer_handoff"]["core"]["owner_kind"] == owner_kind

            scalar_handoff = scalar["consumer_handoff"]["reasoner"]
            native_handoff = native["consumer_handoff"]["reasoner"]
            assert scalar_handoff["ingestion_path"] == "scalar-python"
            _assert_scalar_encoded_resources_are_empty(scalar_handoff["counters"])
            assert native_handoff["ingestion_path"] == "encoded-native"
            _assert_advertised_ledger(native_handoff["counters"], REASONER_COUNTERS)
            assert native_handoff["encoded_schema"] == {
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
                "descriptor_sha256": CONTRACT["encoded_contract"]["descriptor_sha256"],
                "model_schema": CONTRACT["encoded_contract"]["model_schema"],
                "schema_name": CONTRACT["encoded_contract"]["schema_name"],
                "schema_version": CONTRACT["encoded_contract"]["schema_version"],
            }
            semantic_fingerprints.add(
                (
                    native["core"]["fingerprints"]["logical"],
                    native["core"]["fingerprints"]["signature"],
                )
            )
            gc.collect()
    assert len(semantic_fingerprints) == 1
