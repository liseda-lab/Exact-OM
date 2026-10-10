"""Minimal offline reproduction of declaration-dependent native RDF admission.

Compare a declaration supplied by an import with the same declaration explicitly
present in the root. This diagnostic creates no admitted context or publication.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

from exact_inspect.contracts import file_hash


def probe(directory):
    import pyowl_core as core

    root = Path(directory) / "root.owl"
    imports = Path(directory) / "annotations.owl"
    options = core.LoadOptions(
        format=core.DocumentFormat.RDF_XML,
        backend=core.BackendPreference.NATIVE,
        imports=core.ImportPolicy.RESOLVE_STRICT,
        offline=True,
    )
    resolver = core.MappingResolver({"urn:annotations": imports})
    results = []
    for name, data in (
        ("declaration_in_import", root.read_bytes()),
        (
            "declaration_also_in_root",
            root.read_bytes().replace(
                b"</rdf:RDF>",
                b'<owl:AnnotationProperty rdf:about="urn:example:definition"/></rdf:RDF>',
            ),
        ),
    ):
        try:
            snapshot = core.load_snapshot(data, options=options, resolver=resolver)
            results.append(
                {
                    "case": name,
                    "status": "parsed",
                    "complete": snapshot.is_complete,
                    "axioms": len(list(snapshot.iter_axioms())),
                }
            )
        except (core.ParseError, core.UnsupportedSyntaxError) as error:
            results.append(
                {
                    "case": name,
                    "status": "rejected",
                    "type": type(error).__name__,
                    "code": getattr(error, "code", None),
                    "message": str(error),
                }
            )
    return {
        "schema": "exact-review-import-mapping-reproduction/1",
        "version": importlib.metadata.version("pyowl-core"),
        "backend": "native",
        "root_sha256": file_hash(root),
        "import_sha256": file_hash(imports),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.directory), indent=2))
