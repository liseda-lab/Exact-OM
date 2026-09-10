"""Configured OWL imports retain strict offline closure and pinned input bytes."""

from hashlib import sha256

import pytest
from pyowl_core.exceptions import ImportResolutionError

from exact.io.sources import SourceOptionsError, resolve


def documents(tmp_path):
    root = tmp_path / "root.ofn"
    imported = tmp_path / "imported.ofn"
    root.write_text(
        "Ontology(<urn:root> Import(<urn:imported>) Declaration(Class(<urn:child>)) "
        "SubClassOf(<urn:child> <urn:parent>))"
    )
    imported.write_text(
        "Ontology(<urn:imported> Declaration(Class(<urn:parent>)) "
        "AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> "
        '<urn:parent> "Imported parent"))'
    )
    options = {
        "imports": {
            "urn:imported": {
                "path": imported.name,
                "sha256": sha256(imported.read_bytes()).hexdigest(),
            }
        }
    }
    return root, imported, options


def test_imported_labels_and_hierarchy_survive_configured_source_loading(tmp_path):
    root, _, options = documents(tmp_path)
    source = resolve(root, options=options)
    assert source.labels("urn:parent") == ["Imported parent"]
    assert source.direct_parents("urn:child") == ["urn:parent"]
    assert set(source.entities()) == {"urn:child", "urn:parent"}


def test_changed_import_bytes_fail_before_they_enter_the_snapshot(tmp_path):
    root, imported, options = documents(tmp_path)
    imported.write_text("Ontology(<urn:imported> Declaration(Class(<urn:changed>)))")
    with pytest.raises(SourceOptionsError, match="checksum mismatch"):
        resolve(root, options=options)


def test_unbound_transitive_import_fails_without_network_or_partial_closure(tmp_path):
    root, imported, options = documents(tmp_path)
    imported.write_text("Ontology(<urn:imported> Import(<urn:missing>))")
    options["imports"]["urn:imported"]["sha256"] = sha256(imported.read_bytes()).hexdigest()
    with pytest.raises(ImportResolutionError):
        resolve(root, options=options)


@pytest.mark.parametrize("bindings", [None, [], {"urn:imported": "unhashed.ofn"}])
def test_import_bindings_require_explicit_identity(tmp_path, bindings):
    root, _, _ = documents(tmp_path)
    with pytest.raises(SourceOptionsError):
        resolve(root, options={"imports": bindings})
