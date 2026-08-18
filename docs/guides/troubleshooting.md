# Troubleshooting

## Shared OWL packages are missing or incompatible

The base distribution requires `pyowl-core>=0.2,<0.3` and
`pyowl2vec-star-projector>=0.2,<0.3`; released `pyowl-core==0.2.0` is supported. If importing
Exact reports either package as missing or incompatible, confirm that the installer used the
same Python 3.10–3.12 environment as `exact`, then run:

```console
python -m pip check
python -m pip show exact-om pyowl-core pyowl2vec-star-projector
```

Do not install `py-horned-owl`, mOWL, JPype, DeepOnto, or a JDK as a fallback. The exact
published package set used for a release is recorded in
`release/core-compatibility.json`.

## An optional reasoner is unavailable

`asserted` is included in the base install. For `elk` or `hermit`, install:

```console
python -m pip install "exact-om[reasoning]"
```

This selects compatible `pyelk-reasoner>=0.2,<0.3` and `pyhermit>=0.2,<0.3` releases.
Selecting an unavailable or incompatible reasoner is an error by default. Programmatic callers
may explicitly choose an asserted fallback policy; the requested/effective reasoner and reason
are then recorded in `ontology_stack`.

## A native backend was not selected

The core and projector expose complete Python paths. `backend: auto` may report that a
platform-native wheel is unavailable. Use `backend: python` for an explicit portable choice.
Use `native` only when compatible published upstream wheels are installed; Exact never runs
Cargo during installation.

Exact negotiates encoded ingestion from public capabilities, not package-version guesses. An
advertised schema or descriptor mismatch is a hard compatibility error. Do not work around it
by retrying with an OWL path or importing a private native module.

## An ontology cache is rebuilt after upgrading

This is required for ontology-derived schema-1 caches. Core model schema 2 changes structural
identities, including anonymous-individual/component scoping, so Exact rejects the old metadata
before unpickling or interpreting IDs and rebuilds from the original OWL source. A matching
path, timestamp, or source digest does not make the old cache reusable.

There is no cache converter. Completed immutable run outputs remain readable, but a resumable
workflow cannot continue through schema-1 ontology cache state. Delete stale files only to
reclaim disk space; do not rename them to bypass the compatibility check.

## Provenance appears incomplete

Check `stats/run_stats.json` and `run_manifest.json` after successful finalization. Generic
RDF/CSV sources have `kind: generic`; OWL sources have `kind: owl` plus public core,
projector, reasoner, cache, and handoff records. A killed run may have run statistics but no
refreshed manifest; rerun or finalize it rather than copying private paths into provenance
manually.

## Legacy JVM flags still appear in a script

Compatibility flags are accepted only to issue a deprecation warning and have no runtime
effect. Remove them. `exact.init_jvm` is an error-only migration shim and does not initialize
Java.
