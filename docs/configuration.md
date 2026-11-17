# Configuration and inputs

```console
exact config default --output defaults.yaml
exact config profile native --output native.yaml
exact config profile local --output local.yaml
exact config profile biokg --output biokg.yaml
exact config migrate old.yaml --output migrated.yaml
```

The versioned configuration separates run, dataset, candidates, matching,
selector, inference, output and evaluator settings. Public profiles are bundled
inside the wheel. `native` and `local` retain explicit selected numerical
settings and immutable encoder revisions. Their LLM generation is disabled:
the separate decision stage owns Elephant requests and their cache.

Use the local profile for supplied benchmark pools. Global scoring rejects
supplied pools and generates its own candidates. For scoped global matching,
specify owning source and target namespaces with repeated `--source-prefix`
and `--target-prefix` flags. RDF/XML declarations, including relative IRIs and
`rdf:ID`, are resolved against `xml:base`. Scope is never inferred from a
reference file.

A configuration's `matching.channels.*.enabled=false` can disable an enhanced
variant without removing its baseline evidence channel. Use the documented
profile rather than interpreting enhancement flags as channel ablations.

The general `exact align` command accepts OWL/RDF and CSV-KG sources via
`--input-format`, source/target options, and configuration. Use
`exact data list` to inspect built-in tracks, `exact data pull TRACK/TASK`
to materialize permitted data, and `exact data verify TRACK/TASK` to check
recorded identities. Dataset access and licenses remain the user's responsibility.

Do not place provider keys in configuration committed to source control. The
generic hosted adapters use their documented environment credentials; the final
Elephant workflow performs local inference and has no provider fallback.
