# Backend API and package service — B3

Extend `exact_inspect` using RunReader and independently prepared context packages. Separate local exploration, fixed public demo and participant study serving per 10. Read-only ontology/explanation routes remain separate from local import and transactional study mutations in 12. No request-path matching, ontology parsing, embedding/model loading, GPU initialization or LLM calls. Preparation is an explicit job/CLI operation with durable state; requests return prepared data or clear job/unavailable status.

## Required v1 resource surface

Implement equivalent typed routes under `/api/v1/` with generated OpenAPI and a committed fixture client. Exact URL spelling below is the target; encode opaque IDs safely, pass full IRIs as query values rather than splitting them into URL path segments.

| Route | Contract |
|---|---|
| `GET /health` | Service/package/schema readiness; public form excludes filesystem paths/secrets |
| `GET /ontologies` | Paged allowed ontology versions/capabilities |
| `GET /entities` | Search by ontology version, kind, term, language; paged stable results |
| `GET /entity-context` | Independent entity identity/context summary; no run or candidate required |
| `GET /entity-facts` | Category/basis-filtered facts and references with counts and paging |
| `GET /hierarchy` | Parents/children or bounded ancestors, basis and limits explicit |
| `GET /axioms/{id}` | Typed and original axiom/expression; source/derivation metadata |
| `GET /runs/{id}/sources` | Paged eligible source summaries |
| `GET /runs/{id}/candidates` | One source's paged candidate summaries; no graphs/rationale bulk expansion |
| `GET /runs/{id}/pair` | Requested pair's decision trace and linked evidence summary |
| `GET /runs/{id}/pair-evidence` | Bounded evidence graph/list representation for that pair |
| `GET /explanations/{id}` | Prepared profile/comparison, generation and grounding status |
| `GET /jobs/{id}` | Read-only generation/preparation status; policy-appropriate metadata |
| Local-only `/bundles/import` and import-job/library routes | Explicit validated upload/import and bundle selection; absent in demo/study profiles |
| Study routes in [12](12-study-service-and-render.md) | Session/assignment, forms, drafts/submissions, consultation, events and researcher exports |

Study mode exposes only routes/resources allowed by its pack and session; do not mount unrestricted research routes beside redacted study routes. It must be impossible to recover an answer by switching a client mode, guessing a pair ID, searching a withheld entity or downloading a raw unfiltered bundle. The same study participant receives two server-assigned conditions. A shared study service must enforce access per assigned case; standalone packages alone do not supply participant allocation/resumption. No enterprise account platform is required, but the boundary and researcher authentication must be real and tested.

## Bounds and defaults

Initial defaults: ordinary fact/candidate/search page 20, maximum 100; hierarchy child page 50, maximum 200; ancestor subgraph budget 500 nodes, maximum 1,000. Evidence graph default at most 150 nodes/300 edges, with omitted counts/reasons and a complete fact-list route. These are display/query bounds, not ontology extraction limits. Never silently erase edges to existing nodes or same-label entities.

Return summaries first, load pair detail and axiom expressions on demand. Oversized expressions are lossless references available through detail/streamed artifact access, not malformed truncated ASTs. Apply response-size and query-work budgets with explicit limits/errors. All pagination/counts bind the visibility policy; invalid or stale cursors fail clearly.

Bound process caches by bytes and entries with documented eviction; cache key includes ontology/run artifact, schema, query/basis/language and visibility policy. The current source endpoint constructs every candidate graph and retains caches without these bounds: replace this for v1. Do not preload all pair records, ontology objects or all generated text at server startup. Target 64 GB indexing baseline but a substantially smaller serving process over read-only ontology indexes, with study transactions handled separately.

Use immutable snapshots/packages; manifest publication swaps atomically so concurrent reads see one consistent version. Cancellation/timeouts close resources. Report recoverable corruption/unavailable package without falling back to live expensive recomputation. B5 requires no runtime loading/use of training models, accelerators or training libraries to serve prepared JSON/indexes; enforce lazy imports. The current distribution still declares Torch and related packages as installation dependencies. A separate lightweight serving distribution is deferred and is not a B5 prerequisite; do not claim that lazy imports remove those installation dependencies.

## Preparation command surface to implement

These commands are **planned**, not existing capabilities:

```sh
exact-inspect prepare --plan /path/development.json --output-root /path/prepared --dry-run
exact-inspect prepare --lock /path/execution.lock.json --output-root /path/prepared --resume
exact-inspect prepare --lock /path/execution.lock.json --output-root /path/repaired \
  --resume-from /path/old-prepared --repair-plan /path/repair.json --reuse-plan-only
exact-inspect verify-backend --package /path/prepared/package.json --output /path/verification
exact-inspect serve --package /path/prepared/package.json --profile local_app
exact-inspect serve --package /path/demo/package.json --profile public_demo
exact-inspect serve --package /path/study/pack.json --profile study
```

Preparation stages can be selected explicitly: acquire/verify, context-index, run-import, profiles, comparisons, study-export. A preparation command does not secretly launch a new matcher. Generate fresh matching artifacts with the existing Exact entrypoint and import them. Preserve existing `serve/open --run-dir` behavior through an explicit legacy adapter; legacy output cannot claim the v1 readiness gate.

Locks are stage-scoped and immutable once used. Context preparation requires verified ontology/import/runtime bindings; run-import additionally requires the actual completed Exact run identity; each generation stage requires its own resolved OpenRouter profile and policy-filtered packet. Bind the challenger only before challenger dispatch. Publish a new lock revision as later artifacts become available, preserving completed-stage identities. Dry-run reports ready and blocked stages separately; an unresolved future run does not block independent indexing.

## Acceptance

An ontology-only package serves entities without a run. Listing 100 candidates does not construct 100 graphs or call a model. Query paging yields each allowed item once under stable ordering; duplicate labels remain distinct; hidden facts/counts stay hidden. Required API shapes, failure states, lazy loading, bounded cache eviction, package relocation and concurrent immutable reads have meaningful tests. Verify response latency and memory per 07 on real data, not only a tiny fake bundle.

B3 also delivers the bounded study service and local import surfaces in 10/12. Study database operations are explicit writes; the no-work-on-view rule means no matching/parsing/generation, not a prohibition on saving answers. Serving remains CPU-only. `--profile` and any compatibility mapping from old `--mode` options must be explicit, server-owned and validated. Study database credentials are deployment secrets, not package contents or command-line arguments.
