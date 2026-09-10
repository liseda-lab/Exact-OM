# Datasets and tracks

Track providers turn a named task into resolved source, target, reference, and candidate paths.
They pin immutable revisions and hashes in `datasets.lock.json`, so a run records exactly which
upstream materialization it used.

## Built-in workflow

```console
exact data list
exact data pull bioml_hf/ncit-doid --root data
exact data verify bioml_hf/ncit-doid --root data
exact data status bioml_hf/ncit-doid --root data
```

`pull` will not silently move an existing pin. Pass `--update` to accept and record a changed
revision. Verification reports one of four states:

| State | Meaning |
| --- | --- |
| `ok` | Local files and pinned upstream metadata match. |
| `local-drift` | A local file differs from its lock checksum. |
| `upstream-moved` | A mutable upstream reference no longer resolves to the locked revision. |
| `not-materialized` | The requested task is absent. |

The alignment action resolves a track task and the `data` paths, then applies explicit CLI path
overrides. Resolved track provenance and file fingerprints are written to run statistics; the
manifest indexes that statistics artifact and the resolved configuration.

## Declarative providers

A YAML descriptor names the provider, engine (`http` or `huggingface`), revision, files,
checksums, safe unpack rules, and task-layout mappings. Inspect descriptors under
`exact/tracks/builtin/` for complete examples. Use one without installing it:

```console
exact data pull my-track/my-task --descriptor track.yaml --root data
```

Archive extraction rejects absolute paths, parent traversal, and escaping links. Licensed
providers describe expected local files but never download material that requires acceptance.
BioKG supports version-pinned local CSV inputs through explicit, hashed experiment
`CaseBinding` paths and `io.input_format: csv-kg`. Its automatic built-in provider remains
unimplemented. Bind only the selected release's public train/validation labels; keep final
submission inputs reference-free. A CSV graph descriptor may set `entities_file: entities.csv`
with `entity,kind` columns to preserve isolated classes without introducing graph edges.

Third-party Python providers implement `TrackProvider` and register in the `exact.tracks`
entry-point group. They must expose deterministic task names, verification, status, pinned
revisions, and a `TaskLayout` with provenance.

## Offline OWL imports

OWL inputs use strict parsing and local import resolution. Bind each declared import, including
transitive imports, through `io.source_options.imports` or `io.target_options.imports`:

```yaml
io:
  target_options:
    imports:
      "http://example.org/imported.owl":
        path: imports/imported.owl
        sha256: "<SHA256 of the pinned file>"
```

Relative paths resolve against the importing root ontology's directory. Each binding is read
and checksum-verified before the public `pyowl_core.MappingResolver` receives its bytes.
Changed, unresolved, or malformed imports fail loading; the adapter does not fetch network
resources or enable partial RDF mapping. Use files from the declared ontology release, and
record that release's provenance alongside the bindings.
