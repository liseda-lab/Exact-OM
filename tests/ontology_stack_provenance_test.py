from __future__ import annotations

import json
from pathlib import Path

from exact.ontology import load_ontology
from exact.ontology.provenance import ontology_stack_provenance

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


def test_ontology_stack_provenance_is_complete_and_path_free() -> None:
    path = FIXTURES / "mini_src.owl"
    source = load_ontology(path)

    provenance = source.ontology_stack_provenance()

    assert provenance["kind"] == "owl"
    core = provenance["core"]
    assert core["shared_snapshot"] is True
    assert set(core["fingerprints"]) == {"structural", "logical", "signature"}
    assert len(core["closure"]["import_manifest_sha256"]) == 64
    assert len(core["closure"]["resolver_configuration_sha256"]) == 64
    assert core["closure"]["source_documents"]
    assert provenance["projector"]["profile"] == "mowl-d993536-v1"
    assert provenance["projector"]["selection"]["effective"] == "python"
    assert provenance["reasoner"]["selection"]["effective"] == "asserted"

    encoded = json.dumps(provenance, sort_keys=True)
    assert str(path.resolve()) not in encoded
    assert "object at 0x" not in encoded
    assert "file://" not in encoded


def test_ontology_stack_provenance_redacts_consumer_diagnostics() -> None:
    source = load_ontology(FIXTURES / "mini_src.owl")

    provenance = ontology_stack_provenance(
        source.owl_snapshot(),
        projector_settings=source.projector_settings,
        projector=source.projector,
        reasoner={
            "failure_reason": (
                "/private/tmp/exact-reasoner/ontology.pyocore "
                "https://user:secret@example.org/private object at 0x1234abcd"
            )
        },
    )

    encoded = json.dumps(provenance, sort_keys=True)
    assert "/private/tmp" not in encoded
    assert "user:secret" not in encoded
    assert "0x1234abcd" not in encoded
