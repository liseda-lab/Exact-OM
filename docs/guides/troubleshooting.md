# Troubleshooting

## Shared OWL packages are missing

The base distribution requires compatible `pyowl-core` and
`pyowl2vec-star-projector` 0.1 releases. If importing Exact reports either package as missing,
confirm that the package installer used the same Python 3.10–3.12 environment as `exact` and
run `python -m pip check`. Do not install `py-horned-owl`, mOWL, JPype, or a JDK as a fallback.

## An optional reasoner is unavailable

`asserted` is included in the base install. For `elk` or `hermit`, install:

```console
python -m pip install "exact-om[reasoning]"
```

Selecting an unavailable reasoner is an error by default. Programmatic callers may explicitly
choose an asserted fallback policy; the requested/effective reasoner and reason are then
recorded in `ontology_stack`.

## A native backend was not selected

Both the core parser and projector have complete Python implementations. `backend: auto` may
report that an optional native accelerator is unavailable or not yet preferred. Use
`backend: python` for an explicit portable choice. Use `native` only when a compatible
upstream wheel is installed; Exact never runs Cargo during installation.

## A dataset cache is rebuilt after upgrade

This is expected for pre-2.1 ontology caches. Their metadata cannot prove shared-snapshot,
projector, or reasoner compatibility. Exact rebuilds from the input bytes and never unpickles
a legacy ontology graph. Delete `dataset/dataset.csv` and `dataset/dataset.meta.json` only if
you want to reclaim the stale files before rerunning.

## Provenance appears incomplete

Check `stats/run_stats.json` and `run_manifest.json` after successful finalization. Generic
RDF/CSV sources have `kind: generic`; OWL sources have `kind: owl` plus core, projector, and
reasoner records. A killed run may have run statistics but no refreshed manifest; rerun or
finalize the run rather than copying private paths into provenance manually.

## Legacy JVM flags still appear in a script

Compatibility flags are accepted only to issue a deprecation warning and have no runtime
effect. Remove them. `exact.init_jvm` is an error-only migration shim and does not initialize
Java.
