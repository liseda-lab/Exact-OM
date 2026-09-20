# Durable artifacts, selective recovery and study isolation

Reuse Exact's run readers/stores and the principles in [experiment recovery](../experiments/CHECKPOINT-RECOVERY.md). Do not require a whole matching rerun when a context exporter, prompt or interface changes. Scientific design identity, semantic artifact compatibility and execution-attempt identity are different concepts.

## Dependency graph

`verified input bytes + import lock + parser/options -> ontology snapshot -> context index -> policy-filtered packets -> entity profiles -> pair comparisons -> study package`.

Separately: `frozen Exact run + candidate/selection artifacts -> run import + decision/evidence adapter -> linked evidence -> pair comparison inputs where allowed + study package`.

Reference/adjudication keys are not ancestors of participant generation inputs. A pair comparison using only ontology facts need not depend on unrelated numerical model artifacts. Link only genuine dependencies; a whole repository commit hash is provenance, not a blanket invalidation key.

Each stage manifest includes schema/design revision, semantic implementation fingerprint, input/output hashes and sizes, state, successful record inventory, failed/ambiguous records, attempts and lineage, verification result, resource/cost totals, and relocatable artifact locators. Status transitions and writer ownership are explicit. Immutable completed artifacts publish by atomic rename/transaction; incomplete staging paths never look complete to readers.

Content-addressed ontology packages are reusable across runs; entity profiles across candidate pairs; model outputs across identical compatible requests. One stage writer owns publication. Locks/leases include stale-owner recovery; concurrent readers see a consistent published revision. Index build may checkpoint partitioned records or resume by verified stage/chunk; partial facts are not silently served as full coverage.

## Stop, copy, repair

Graceful STOP stops admitting new work, saves completed records and returns a continuation command. Abrupt termination can lose only in-flight local work, not validated published records. A copied old run/package must resume in a new directory or host with source paths unavailable where portability was promised. Resolve locators through manifest roots or explicit rebindings and reverify content, never by matching filename alone.

Before bug repair, emit a dependency-based reuse plan: cause, affected implementation/version/fields/cases, artifacts reusable as-is, artifacts to rebuild, scientific comparability consequences and expected cost. Retain the old attempts and current-result pointer separately. The execution can apply a safe, authorized plan without asking approval for every routine resume.

| Change | Reuse | Rebuild |
|---|---|---|
| Frontend layout/font | All compatible backend artifacts | Frontend build only |
| Profile prompt/model | Ontology, matcher and context | Affected profiles, dependent comparisons/packages |
| Pair prompt only | Profiles and upstream data | Comparisons/packages |
| Annotation classifier/index bug | Raw snapshots, unaffected Exact run | Affected context and dependent text/export |
| Visibility policy | Raw authorized source storage | Filtered packets, policy-dependent text/search counts/packages |
| Ontology/import content | Unrelated snapshots/runs | Affected context; matcher outputs only if the matching input changed |
| Evidence-export/provenance bug | Verified original scoring/mappings | Adapter/export and dependencies; instrumented matching replay only if missing facts cannot be recovered |
| Actual scoring defect | Ontology/context independent of score | Affected matching/downstream products under experiment repair rules |

Never silently recompute historical selected evidence and label it original. Runtime repairs need a new execution revision; changes to case selection, experimental condition or target question need a new design revision and fresh validity assessment. Record prior participant exposure if a study-affecting repair occurs.

## Study policy and answer separation

Define server-owned immutable `visibility_policy` with ID/hash, allowed ontology/fact categories, xref/mapping handling, permitted matcher advice, candidate presentation order, capabilities and generation routes. An allowlist policy is preferable to stripping a few known answer keys. Apply it to facts, search, counts, axiom access, generation packets, caches, text responses, exports and error messages. Policy hash participates in every filtered artifact key.

Reference membership/relations, gold target IDs, hidden-test artifacts and independent adjudication answers are stored outside participant-serving packages. Same candidate/ontology facts may be permitted in different advice conditions, but prohibited advice must not leak through prose, labels, ranks, links or raw artifact downloads. Test existing `ground_truth` exposure, alternate client modes, route enumeration, cross-condition cache reuse and predictable artifact IDs. Restricted facts may remain in an access-controlled preparation store, never mounted into the study service.

The complete study product in 12 requires a durable session spanning two assigned conditions, not only separate static condition packages. Server-bound opaque session/case/condition assignment enforces access; do not build an unrelated participant account platform. Browser-supplied condition flags cannot elevate scope. Exploration of permitted source/target ontologies remains available; the policy determines allowable universe rather than a UI-only hide rule. Treat SNOMED/license packaging constraints as input capabilities, not an invitation to rehost everything.

Study records bind the pseudonymous session, study/assignment/case/presentation/package versions, partial ranking or explicit none/insufficient evidence, per-case ontology consultation, form answers, timing and unique idempotency keys. Semantics follow 11–13; an exploration-app review action is a separate resource. Commit submission and progression atomically; retries cannot duplicate or replace a response. First submission is primary by default, with any permitted revisions append-only. Resume through the original invitation restores the same allocation, current step, drafts and saved responses. No participant account/name/email is required or collected. Do not contact anyone through this implementation assignment.

## Acceptance

Interrupt index and text preparation, kill after response-save/before-validation, restart from copied paths, and repair a prompt and a context bug. Show record counts/hashes and cost ledgers demonstrating correct reuse and selective rebuild. Corrupt/missing artifacts fail closed with actionable status; unknown compatibility does not authorize reuse. Redaction tests include adversarial requests and cached prose produced under a broader policy. Published study data contains no answer keys or forbidden answer-bearing derivatives.

## Participant-state recovery

Artifact resumption and participant resumption are separate guarantees. Immutable context/text packages follow the dependency graph above; live study sessions require PostgreSQL transactions, versioned migrations and tested backup/restore per 12. An invitation is reusable after cookie loss or server restart. Opening or previewing a link does not assign a new schedule or start a case timer. No study revision change may silently replace already exposed case content, answers or instructions. Preserve timing gaps and telemetry-loss indicators; do not manufacture uninterrupted active time.
