# Prepared explanation backend

The backend separates ontology context, recorded matcher decisions, and generated
interpretations. `exact-inspect` serves immutable prepared resources on a CPU.
The exploration and study frontends are a separate implementation assignment.

Install the repository with the `viz` extra for exploration, or `study` for the
PostgreSQL study service. `poetry.lock` includes the optional study dependencies.
The core ontology implementation uses the installed native pyowl-core stack;
serving prepared packages does not import its parser or a model runtime.

## Bind and prepare

Create a JSON execution template. Paths are relative to the template file; optional
preexisting SHA-256 and byte-size pins are checked rather than replaced.

```json
{
  "design_revision": "my-development-inputs/1",
  "ontologies": [
    {"name": "source", "root": {"path": "source.owl"}, "scope": "root"},
    {"name": "target", "root": {"path": "target.owl"}, "scope": "root"}
  ]
}
```

```sh
exact-inspect bind-lock --template bindings.json --output execution.json
exact-inspect prepare --lock execution.json --output-root prepared --dry-run
exact-inspect prepare --lock execution.json --output-root prepared
```

Binding reads and hashes every input, records installed distribution versions and
RECORD hashes, and produces an immutable lock. It does not parse, score or generate
text. The design blueprint in `specs/explanation-framework/protocol/development.json`
can be inspected with `prepare --plan ... --dry-run`; it is not an executable lock.
An unresolved future run or generation profile does not block ontology preparation.

For closure scope, provide an `imports` mapping from exact ontology import IRI to
file binding. Root scope deliberately ignores imports and reports that boundary.
An explicit source derivative belongs in `source_derivation`; preserve its original
input hash, changes and reconstruction provenance. Original source failures remain
separate evidence.

A verified context can be reused with `prepared_context: {"path": "context/manifest.json"}`
on an ontology binding. Preparation checks source hash, scope, resolved imports,
parser version, derivative provenance and the full index checksum before copying it.
Ontology contents and parser policies are never inferred from filenames.

The remaining lock fields are described by
`exact_inspect.preparation.ExecutionLock.model_json_schema()`:

- `run`: manifest binding, complete file inventory, source/target ontology names
  and root hashes. `bind-lock` can inventory the manifest directory automatically.
- `profile`: public OpenRouter model, explicit provider routing, revision and
  bounded request parameters. Secrets are environment configuration only.
- `entities`: stable worklist IDs, ontology names, IRIs and kinds; `pairs` refers
  to their worklist IDs. Profiles do not receive counterpart or score inputs.
- `policy`: server-owned ontology/category visibility policy. It is applied before
  facts, generation requests, cache identities and participant publication.
- `study_definition`: frozen validated study definition with content-bound resource
  paths. `study-export` verifies and copies explanations, ontologies and their
  policy-filtering receipts. Answer keys are a separate researcher publication input.

`--stage` selects explicit stages: `acquire-verify`, `context-index`, `run-import`,
`profiles`, `comparisons`, `study-export`, `portable-export`. Without it, all ready
stages run in dependency order. Matching is never an implicit preparation stage.
Provider generation requires an explicit bound profile and uses the existing
OpenRouter transport and persistent attempt/cost ledger.

## Recovery and selective repair

```sh
exact-inspect prepare --lock execution.json --output-root prepared --resume
exact-inspect prepare --lock revised.json --output-root repaired \
  --resume-from prepared --reuse-plan-only --repair-plan reuse-plan.json
exact-inspect prepare --lock revised.json --output-root repaired --resume-from prepared
```

Create `prepared/STOP` for a cooperative stop at the next safe boundary; remove it
before resuming. SQLite context checkpoints and provider response checkpoints survive
process exit. A copied output directory retains its completed stage inventories and
generation ledger. A prompt/model/policy change rebuilds the affected text; unrelated
matcher results and compatible context indexes remain reusable. Unknown provider
delivery is retained as ambiguous rather than silently charged again.

Execution locks, stage outputs and explanation identities are immutable. Use a new
lock path for changed inputs. Corrupt completed artifacts fail verification; inspect
and quarantine the corrupt artifact before rerunning the affected stage. Preserve
ledgers when repairing text to avoid unnecessary repeat requests.

## Serve, import and verify

```sh
exact-inspect serve --package prepared/stages/STAGE/package.json --profile local_app
exact-inspect verify-backend --package prepared/stages/STAGE/package.json --output verification
exact-inspect export --package prepared/stages/STAGE/package.json --output development.zip
```

The API includes `/api/v1/ontologies`, `/entities`, `/entity-context`, `/entity-facts`,
`/hierarchy`, explicit `/axioms/{id}`, run sources/candidates/pair/evidence, prepared
`/explanations/{id}`, and job status. FastAPI publishes `/openapi.json`.
`exact_inspect.client.InspectClient` is the typed backend fixture client.
Collection cursors bind ontology, policy, query and context revision; stale cursors
return 409. Summary/list responses are bounded to 2 MiB. Large original axioms remain
available through explicit context streaming or the prepared ontology resource.

The local profile supports streamed ZIP import with a persistent import job, status,
cancellation and atomic library selection. Archives contain inert data only; traversal,
symlinks, conflicting members, excess expansion and digest mismatches are rejected.
The default expanded-file budget is 32 GiB with bounded streaming memory; the JSON
cache is limited to 128 entries and 32 MiB. Uploaded artifacts are never executed.

The `public_demo` profile requires a package explicitly marked `development_demo`;
set `audience: "development_demo"` in its execution template before export.
It exposes no import, study or researcher mutation routes. The isolated study
application and its durable database use a separate factory, deployment and asset
universe. See [the study runbook](explanation-study-service.md).

## Lean CPU deployment of prepared packages

The `viz` extra retains Exact's training dependencies because it belongs to the
full distribution. A prepared package can instead use the dedicated serving image:

```sh
docker build -f deploy/render/exact_inspect_prepared.Dockerfile -t exact-prepared .
docker run --rm -p 127.0.0.1:10000:10000 \
  -v "$PWD/demo-package:/data/package:ro" exact-prepared
```

This image installs the pinned Python 3.12 dependencies in
`deploy/render/exact_inspect_prepared_requirements.txt` and copies only
`exact_inspect/`. It contains no `exact` matcher package, PyTorch, Transformers or
ontology parser. It runs the prepared CLI/API without a frontend build. The
mounted package must have `audience: "development_demo"` for its default hosted
profile. Health is `/api/v1/health`.

For a local import library, create a writable directory and override the CLI
arguments; the published host port remains loopback-only:

```sh
mkdir -p local-library
docker run --rm -p 127.0.0.1:10000:10000 \
  -v "$PWD/local-library:/data/library" exact-prepared \
  --profile local_app --library-dir /data/library --host 0.0.0.0 --port 10000
```

Without Docker, create an isolated Python 3.12 virtual environment, install that
requirements file, copy the release's `exact_inspect/` directory into a clean
working directory, and run `python -m exact_inspect.cli serve` there with the same
package/profile arguments. Copy the directory in full so the structured OWL schema
and study resource models are present. This serving-only environment cannot run
preparation, legacy raw-run adaptation or provider generation; those remain offline
steps in the full Exact environment. The existing Render Dockerfile and `viz`
extra remain available for the historical viewer.

## Interpretation and grounding

Original facts preserve exact literals, datatypes, language tags, axiom IDs and
source origins. Structured OWL expressions retain their faithful typed AST and
canonical original representation. Literal asserted edges, structural navigation,
inference and projected matcher features have distinct interpretation fields.
Unavailable history never becomes a fabricated stage decision or inferred provenance.
Saved numerical values retain their meaning; calibration is not presumed.
Alignment eligibility is `null` when no matcher eligibility set was bound, with
an explicit availability status; declared sets distinguish inclusion and exclusion.

Generated output is admitted only when its citations belong to the frozen fact
packet and its claims are exact supported excerpts or conservative comparison
templates. Templates describe recorded wording and asymmetric information without
asserting equivalence or incompatibility. Unsupported paraphrases, relation claims
and invented limitations require review and fail automatic admission. Provider
failure produces original-fact fallback text with the failure state retained.
Generation inputs are bounded by fact count and bytes.

`exact_inspect.study.builder.build_explanation_resource` adapts the shared indexed
context, saved request packets, grounded outputs and saved evidence into strict study
panels. It checks packet facts against the authoritative original index. Ontology
resources use `context_resources.export_ontology_resource`, which removes prohibited
annotations using the installed structural renderer and emits a frozen admission
receipt. Serving validates bytes/receipts without reparsing OWL.

## Bounded operational evidence

`tools.validate_context_runtime` prepares the pinned full ontologies and records
native/runtime provenance, exact context parity and resource measurements.
`tools.validate_explanation_export` scores at most twelve supplied development pairs
on a cached CPU encoder; it performs no training or benchmark evaluation.
`tools.prepare_explanation_demo` consumes those existing contexts and run artifacts:

```sh
python -m tools.prepare_explanation_demo \
  --contexts data/explanation-framework/context \
  --source SOURCE.owl --target DECLARED_TARGET.owl \
  --run data/explanation-framework/matcher/current-12-root \
  --output data/explanation-framework/prepared --generate
```

Only `--generate` enables the bound OpenRouter preparation. Use the normal authorized
OpenRouter environment/key configuration; never put credentials in a lock or package.
The command retains a claim audit and prints the package and execution-lock paths.
`--additional-pairs` accepts at most four explicit public-data coverage pairs without
scoring them. `--resume-from` reuses the frozen generation ledger when preparing
in a new directory; a local SSD output directory avoids random network-disk writes.
Run `verify-backend` in a fresh process to measure serving memory separately from
preparation. Actual evidence and remaining gate limitations are recorded in the
implementation status ledger; successful unit tests alone do not admit the frontend.
Full experiment campaigns, public deployment and participant enrollment are separate
steps beyond this backend assignment.
