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
The BioKG provider is intentionally unavailable until its upstream repository is published;
its descriptor already records the expected CSV-KG layout.

Third-party Python providers implement `TrackProvider` and register in the `exact.tracks`
entry-point group. They must expose deterministic task names, verification, status, pinned
revisions, and a `TaskLayout` with provenance.
