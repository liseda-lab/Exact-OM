# Changelog

Exact-OM follows semantic versioning and keeps the authoritative changelog in the repository.

Exact-OM 2.1.0 migrates ontology loading, projection, and optional reasoning to the released
pyOWL 0.2 package family. It binds provenance and cache identities to core API 0.2, model and
encoded schema 2, the current public wire writer, and
`EncodedStructuralView.DESCRIPTOR_SHA256`. Ontology-derived schema-1 caches are rejected and
rebuilt without conversion, while completed immutable run artifacts remain readable. The
release retains one public core owner across the source facade and consumers, remains
Java-free, and expands clean-install wheel/sdist checks across Python 3.10–3.12 for base,
visualization, and reasoning environments.

The historical fixture timing comparison is diagnostic, and NCIT–DOID is the sole
external-data correctness gate. Exact-OM 2.1.0 sets `performance_claim: false`.

Exact-OM 2.0 introduced config schema v2, versioned dataset tracks, ontology/KG format
registries, property/instance matching, decomposed action/model boundaries, accurate timing,
the indexed run-artifact store, and `exact-inspect`.

Read the complete
[CHANGELOG.md](https://github.com/liseda-lab/Exact-OM/blob/main/CHANGELOG.md)
for compatibility shims and release-by-release details.
