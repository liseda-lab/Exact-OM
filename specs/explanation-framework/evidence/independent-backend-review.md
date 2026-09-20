# Independent explanation-backend review

19 September 2026 · Read-only review of Exact-OM `e81865aceed0cd1257580097bc788329cd9084ae`

**The proposed split is sound. Implement the shared contract, Exact adapter and context backend first; involve the specialized frontend agent only for full implementation after the handoff gate.** A working NCIT–DOID class-pair service is a sufficient first gate. KGA, complete reasoning, exhaustive property matching and the final human study should not delay it.

This review checks the existing implementation against the earlier [feasibility report](feasibility-report.md). It adds concrete requirements for the specifications, without repeating the dataset measurements. No production code was changed and no new benchmark or LLM generation was run.

**1. First establish that the proposed parser/runtime and dataset actually run together.**

The small successful capability probe imported sibling pyOWLCore source directly and used its Python backend. The probe is useful evidence of capability, but not proof of a supported installed native runtime. Current implementation must use the [published stack contract](../../native-stack.md), with `pyowl-core >=0.2.1,<0.3` and `0.2.1` in the lock. The current Bio-ML descriptor still has the old checksum-manifest name, user-supplied NCIT/DOID/FMA declarations, and obsolete repaired candidate paths. These are implementation prerequisites, not frontend issues.

Require a pinned environment and pinned NCIT–DOID root/import manifest, a native load/index build on the target node, and a fresh bounded Exact run using that same identity. Compare selected raw definitions and expressions against the resulting API. Record effective parser/backend, package versions, options, imports, warnings and peak RAM. A parser fallback must be reported rather than silently treated as equivalent. Do not make a complete new matching campaign a prerequisite.

Sources: [runtime requirement](../../../pyproject.toml), [descriptor](../../../exact/tracks/builtin/bioml_hf.yaml), [probe scope](capability-probe-scope.md).

**2. Context must be independent of a selected graph and must report partial caches honestly.**

The existing node-information route requires source, target and graph-node IDs. It cannot serve a hierarchy browser independently. More subtly, `annotation_payload` returns a cached payload whenever *any* cached field is nonempty. A cache containing a label but no definition therefore prevents lookup of richer native annotations. Cached parent/child rows similarly suppress the native path, even if the cache was truncated. Current substring-based definition detection misses opaque NCIT P97/P325 and DOID IAO_0000115 predicates.

The new contract must provide entity lookup by ontology version, IRI and kind without needing a run or candidate. Use a versioned annotation-predicate registry, retain language/datatype/citation metadata, and distinguish explicit absence within a loaded scope from unexported, truncated, unresolved-import, unsupported and failed context. Empty lists and `false` are not substitutes for unknown status. Avoid on-demand merging with a newer remote ontology; a richer context version must be identified explicitly.

Sources: [current graph-bound route](../../../exact_inspect/app.py), [partial-cache behavior and predicate heuristic](../../../exact_inspect/bundles.py), [truncated export](../../../exact_inspect/bundles.py).

**3. Separate small collection endpoints from expensive detail payloads.**

The current source endpoint constructs every candidate graph, returns all of them, and retains source/pair payloads in dictionaries without eviction. This is reasonable for a small fixed demonstration but a poor basis for browsing whole NCIT runs. Lazy explanation-shard reading already exists and should be retained.

Require a lightweight paginated candidate list, separate per-pair evidence/trace requests, and paginated hierarchy, axiom and search resources. Cursors must bind to the immutable index version, query and stable ordering, with identity as a tie-breaker. Counts may be unknown or lower bounds where expensive; the contract must say which. Bound response size, ancestor depth, cache size and background-job concurrency. Test duplicate labels, cycles, multiple inheritance, high degree and invalid/stale cursors. Never enumerate every root path or run a full-ontology scan on a click.

Use one offline index builder and a read-only indexed serving path initially. Measure total process RAM, including duplicated worker snapshots; increasing HTTP worker count must not accidentally load a full ontology per worker. The 64 GB profile is the baseline, with a documented bounded configuration for 128 GB where helpful. Serving entity context and explanations must not require a GPU, instantiate matching models, or make LLM calls in response to reads.

Sources: [current eager payload construction](../../../exact_inspect/bundles.py), [payload caches](../../../exact_inspect/bundles.py), [lazy run opening](../../../exact_inspect/bundles.py).

**4. Keep the Exact changes narrow, but do not let the backend manufacture missing provenance.**

Existing records and overlays already expose useful scores, selection outcomes and saved-alignment membership. Reuse them. The genuine additions are preserved candidate origin/ranks, stable evidence identity, original-axiom/projection references, and stage outcomes explaining threshold/cardinality/extraction effects. Current evidence IDs hash display text; extraction diagnostics are aggregate counts. A historical rejected candidate cannot reliably acquire a causal story from these alone.

For the first implementation, define a finite stage-outcome schema rather than a general event-sourcing platform. Each stage declares whether it ran, its outcome, relevant score/rule/config and provenance, and competing mappings where the algorithm provides them. Keep `not_run`, `not_recorded`, `failed`, `abstained` and `rejected` distinct. Preserve raw relation direction, reference relation and reviewer judgment independently. A reconstructed final-file membership may be useful, but must be marked reconstructed. Historical incomplete runs remain readable with explicit limits; complete fresh NCIT–DOID fixtures are required for handoff.

Ontology service ownership: original facts/expressions, context scope and navigation. Exact ownership: selected/projected features, contributions and decisions. Backend ownership: typed joins and readable serialization. LLM ownership: explicitly generated interpretations. Do not conflate these origins.

Sources: [overlay fields](../../../exact/impl/trainer/overlays.py), [display-derived IDs](../../../exact/impl/models/pair_adaptive_evidence.py), [aggregate extraction diagnostics](../../../exact/impl/extraction.py).

**5. Study visibility is an input policy, not just an output-field filter.**

The current endpoint includes `ground_truth`. Removing that property is necessary but insufficient: answers or prohibited advice can survive through cached prose, mapping xrefs, annotations, filenames, scores, ranks or another unredacted route. A condition should explicitly state which of those categories is permitted; not every annotation or ranking is necessarily prohibited.

Apply the condition policy before building generation inputs and before serving every participant resource. Include the policy/version hash in explanation/cache identity. Separate researcher artifacts from participant exports, and prevent a participant URL parameter from selecting a privileged condition. Inspect actual participant response bodies, downloadable bundles and generated-text inputs; hiding frontend controls is not a test of blinding. Fair baseline access to allowed ontology facts must be documented separately.

Sources: [answer field in API payload](../../../exact_inspect/bundles.py), [answer fields in selected records](../../../exact/analysis/user_study/export.py).

**6. Versioned text generation must sit outside the matcher and the request path.**

The current rationale path receives selector advice, and backfill skips any existing nonempty rationale. It is not an appropriate implementation for independently grounded entity profiles or a score-blind comparison. Introduce separate offline jobs using the existing OpenRouter routing infrastructure; do not load an entire matching model just to generate prose.

An entity profile uses permitted ontology context without candidate identity or score. A comparison uses permitted source/target context and explicitly distinguishes agreement, scope, explicit incompatibility and missing information. Original facts and readable expressions must remain accessible if text generation fails. Claim-level citations must resolve to input fact IDs; citation existence alone is not evidence that a claim is entailed, so a small independently reviewed quality set is still needed. No generation should fabricate source definitions or equate missing assertions with disagreement.

Cache identity must cover the exact context/fact IDs, ontology scope, visibility policy, task/schema, full prompt/template version, model/provider and decoding settings, language and truncation policy. Record the effective provider/model where returned, token usage, attempts and terminal errors. Preserve completed results under their old identities; regenerate only affected dependents after a prompt or context fix. An uncertain hosted response cannot always guarantee exactly-once billing, so state the retry policy honestly and never promise that it can.

Sources: [verdict-bearing rationale context](../../../exact/impl/models/semantic_llm.py), [nonempty rationale skip](../../../exact/analysis/user_study/export.py), [existing cache keys](../../../exact/impl/models/scorer_common.py).

**7. Resume guarantees must cover the new artifacts, not merely the existing run store.**

The explanation store already has atomic index commits and recovery of uncommitted tails. Reuse its design and reader; do not reimplement matching checkpoints. This does not automatically make the new index, text jobs and exports resumable. The current study JSON writer is a direct overwrite.

Give acquisition/indexing, optional inference, Exact adaptation, entity profiles, comparisons and publication separate content-derived identities and completion manifests. Use transactions or temporary artifacts plus atomic publication; designate a single writer or enforce locking. Record a dependency graph and compatibility checks, with no reuse merely because a filename exists. Support reopening a copied/moved artifact directory via relative package references or an explicit root remap. Changed ontology/import/normalization inputs invalidate context dependents; changed prompts invalidate only affected text; a renderer-only change should not rerun matching.

Required recovery checks: interruption mid-index and mid-export; restart after some hosted jobs complete; prompt-only regeneration; context-fix invalidation; opening a relocated package; and rejection of a corrupted/incompatible artifact while preserving other valid work. Optional inferred artifacts must not block asserted browsing.

Sources: [existing transaction/recovery contract](../../../exact/runs/store.py), [direct JSON overwrite](../../../exact/analysis/user_study/export.py).

**Practical implementation order and frontend handoff.**

1. Repair/pin acquisition and supported runtime; define the class-first versioned contract and real/synthetic fixtures.
2. Implement the ontology index and Exact producer/adapter work in parallel against that contract. Preserve typed expressions; use lossless original-expression fallbacks for unsupported readable templates.
3. Assemble a bounded fresh NCIT–DOID case through context, candidates, evidence, decision trace and API. Establish measured 64 GB build/serve limits and bounded pagination/caches.
4. Add independent OpenRouter profile/comparison jobs, condition-aware publication, portable manifests and targeted resume/invalidation checks.
5. Freeze the handoff package: machine-readable schemas/OpenAPI, fixture server or runnable API, deterministic examples, command/runbook, test evidence and a precise implemented/missing capability matrix. Then assign full frontend design/implementation to the specialist.

The handoff must demonstrate both a rich and a sparse-definition real case, multiple parents, predicate descriptions/import status, a nonselected alternative, an incomplete historical trace, and unavailable/failed generated text. Synthetic cases cover logical constructors and identity collisions not exercised by the real pair. The frontend must be able to display honest missingness from the API without parsing OWL, guessing scores, generating prose or consulting the original agent.

Do not block this gate on full OWL reasoning, proof justifications, a generic job-orchestration platform, a universal graph database, exhaustive ontology verbalization, KGA adapters or a confirmatory human study. Those can follow a usable, semantically faithful class-pair service. In particular, the specialized frontend agent has no early review or approval responsibility.


## Final specification review and resolution

The draft was independently checked after this review. Its earlier findings are incorporated.
Three specification issues were corrected before publication:

- Stage-scoped locks avoid requiring a future Exact run before independent context indexing;
  a challenger model is required only before its own dispatch.
- Missing historical stage records use `not_recorded`; corresponding resource availability uses
  `not_exported`. Unrecorded execution is not assumed to have occurred.
- B5 requires no runtime loading/use of training dependencies. The current distribution still
  installs them; splitting a lightweight serving distribution is explicitly deferred.

See [backend API](../04-backend-api.md), [Exact integration](../02-exact-integration.md) and
[protocol](../protocol/README.md). This review is specification evidence, not completed runtime
acceptance.
