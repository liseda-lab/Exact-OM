# Preparing independent ontology context

Context preparation is an explicit CPU task. It uses the installed `pyowl-core`
parser, verifies the pinned root/import bytes, and publishes an immutable SQLite
package. Serving reads the package without initializing the parser or a model.
The package can be copied with the original source directory unavailable.

For the declared NCIT–DOID operational inputs:

```sh
.venv/bin/python -m tools.validate_context_runtime NCIT
.venv/bin/python -m tools.validate_context_runtime DOID --verify-original-doid
.venv/bin/python -m tools.validate_context_runtime DOID
```

These commands prepare context and measure bounded queries; they do not run
matching experiments. Outputs and checkpoints live in
`data/explanation-framework/context/`. `*.operational.json` records source/runtime
identity, preparation resources and verification. `*.queries.json` records 100
mixed reads and four concurrent readers in a separate process. An interrupted
attempt is retained as `*.attempt-<timestamp>.json`.

The original pinned DOID document is tested separately with strict native mapping.
The installed 0.2.1 runtime rejects it. The DOID preparation command therefore uses
the explicitly recorded derivative containing two annotation-property declarations;
its distinct source hash and derivation receipt remain in the manifest. This does
not count as successful loading of the unmodified original. Both commands use
root-only scope, with the unresolved DOID import reported explicitly.

## Interruption and recovery

Only a directory containing a verified `manifest.json` is a published package.
The adjacent `.NAME.building` directory holds private SQLite transactions,
`attempts.jsonl`, and `checkpoints.jsonl`. At most one local writer owns its OS lock;
an exited process releases ownership. Completed chunks retain the exact input and
preparation identity plus the last original axiom digest. SQLite rolls back the
uncommitted chunk after abrupt termination. Resume verifies SQLite integrity and
the original axiom boundary before continuing. The input still needs to be parsed
for an incomplete build; completed packages can be reused directly.

Rerun the same command to continue. To continue on another machine, copy the
private `.NAME.building` directory beside the same destination name, provide the
same pinned input bytes, and rerun preparation. Do not serve a staging directory.
An input/version/coverage mismatch fails closed; choose a new destination when
preparing changed semantics or eligibility.

The builder defaults to a bounded 256 MiB SQLite page cache; the full-data validation
command uses 2 GiB via `--sqlite-cache-mib`. Preparation commits every 10,000 axioms,
and creates secondary indexes after all source records have been written. This
avoids maintaining random secondary indexes through repeated network filesystem
writes. Checkpoint records expose processing time, commit time and peak RSS.

On a network filesystem, add `--work-directory /tmp/exact-context-work` to perform
database writes on local storage. The builder copies the last durable checkpoint
into a private working directory, commits locally every 10,000 axioms, and copies
an atomic durable checkpoint back every 500,000 axioms, on graceful interruption,
and before publication. Abrupt machine loss can require repeating work since the
last durable copy. The checkpoint log distinguishes local and durable counts and
records copy time. Keep the destination on persistent storage; local workspaces
are disposable.

## Policy-filtered resources for external ontology inspection

```sh
.venv/bin/python -m exact_inspect.context_resources \
  --context data/explanation-framework/context/doid \
  --policy policy.json \
  --output resources/doid.ofn
```

The exporter uses the installed canonical decoder and OWL Functional Syntax
renderer to create an import-free ontology for Protégé. The policy must preserve
all logical axiom categories. Disallowed annotation assertions and nested mapping
qualifiers are removed before bytes are written. Source/attribution and synonym
metadata remain explicit. The adjacent `.receipt.json` binds the bytes, source
context, ontology identity and immutable policy; study publication verifies this
trusted preparation receipt and the complete resource hash without parsing OWL.
Participant uploads cannot establish their own admission authority.

For a native round-trip check of an operational resource:

```sh
.venv/bin/python -m tools.verify_context_resource DOID \
  --context data/explanation-framework/context/doid \
  --resource resources/doid.ofn --output resources/doid.parity.json
```

This compares the exported logical axioms against the verified original context,
then checks every exported annotation predicate and nested qualifier against the
policy. It parses the Functional Syntax with the installed native runtime. Omitting
`--context` reparses the pinned original source as an additional independent check.
The report distinguishes these two sources of expected logical axioms.

Fact and summary reads have byte budgets as well as row limits. Oversized values
return a typed axiom reference with explicit partial status; the faithful original
is retained. Inline axiom detail is capped at 2 MiB. Offline/local consumers can use
`OntologyContext.axiom_chunks()` to stream larger exact JSON artifacts. Policy
scoped streams require a physically filtered context package whose recorded policy
matches the request; the raw package cannot serve a participant stream.
