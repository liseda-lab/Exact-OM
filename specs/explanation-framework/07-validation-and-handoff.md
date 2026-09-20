# Verification and frontend admission — B5

The specialized frontend agent starts only after this backend gate passes. No early frontend-agent consultation is required. Required evidence is a working backend, not a written implementation claim. Do not wait for final human-study results, all optional providers or whole-ontology reasoning to begin F1.

## Gates

| Gate | Required evidence |
|---|---|
| G0: contract/input lock | Versioned models/OpenAPI, validated shared primitives, manifest/policy and dataset locks, readable actual inputs, installed runtime compatibility |
| G1: independent context | Full pinned NCIT–DOID indexed; ontology-only lookup; original definitions/restrictions/multiple-parent navigation; explicit sparse/import/unsupported states |
| G2: Exact integration | Small fresh current-schema development run imported; real score/evidence/fact links and final membership; synthetic tests for rejected/conflicting/unsupported branches; unchanged matching outputs |
| G3: service and text | Bounded lazy APIs, local import and fixed-demo enforcement; real prepared OpenRouter examples/fallbacks; complete study flow APIs with synthetic mixed cases, no work triggered by viewing |
| G4: recovery/isolation/resources | Artifact kill/copy/selective repair; PostgreSQL invitation/session/response restart and restore; retry/conflict/event/timing checks; redaction/profile isolation; full-data resource measures |
| G5: handoff publication | All required gates pass; reproducible launch, contracts, fixtures, artifacts, commands, limitations and frontend task brief committed/exported |

Readiness ledger records pass/fail/not-run per check with exact command, runtime/data hashes and result artifact. `not_run` is not pass. Optional features have separate statuses. No production score, provider response or user result may be fabricated to fill a gate.

## Test matrix

- Contract semantics: typed identities/punning, duplicate labels, preserved literal language/datatype, all status distinctions, stale cursor rejection and policy-sensitive counts.
- Ontology semantics: literal asserted versus structural/inferred hierarchy; equivalent intersections, cycles, existential/universal/negated/inverse/cardinality expressions; annotation citations versus definitions; complex domain/range; unlabelled imported predicates. Fallback retains original expression.
- Real-data context: known available and missing definitions; recover definitions using exact opaque predicates; known-partial cache does not mask compatible richer context; graph detail does not bound ontology navigation.
- Matcher fidelity: final alignment reconciliation, real enabled-stage trace, candidate rank semantics, competing mapping facts, optional NIL fields and namespaced legacy incompleteness. Defaults remain behavior-preserving.
- Service behavior: candidate listing builds no graphs, entity-only lookup requires no run, bounded cache eviction, immutable concurrent readers, no model/GPU/provider activity on GET or cold serve.
- Generation: candidate-blind profiles, score-blind comparison, supported citations, no missing-as-contradiction, provider failure fallback, policy hash isolation and actual request ledger.
- Operational recovery: scenarios in 06, corruption, partial publication, new output directory, saved request ambiguity and no unnecessary recomputation.
- Study isolation: direct endpoints, client mode changes, cached text, search/total counts and ontology downloads cannot reveal withheld answers or baseline explanations; local import is disabled server-side on both hosted profiles.
- Complete study lifecycle: original invitation after cookie loss/server restart, background/setup/practice, mixed-case allocation, partial/none/uncertain responses, consultation, final form, completion and researcher-only exports. Test repeated links, lost acknowledgements, stale drafts/two-tab writes and event retry independently.
- Study correctness: positive MRR and separate negative metrics, unfinished versus empty response, simulated schedule balance and no rerandomization; first response retained.
- Timing: case-ready versus preload, browser-hidden external work, explicit breaks, unknown gaps, reload page-instance IDs and post-case consultation-questionnaire time excluded from task duration; actual external ontology inspection remains timed.

Extend existing `tests/exact_inspect_test.py`, ontology, run-store, user-study and rationale suites where their ownership fits. Unit/CI tests use synthetic local data and fake providers. Mark full-data/native/provider checks separately and retain their evidence. Do not add shallow tests that merely mirror the implementation; these checks protect actual semantic and recovery risks.

## Resource and usability-of-API measurements

Baseline: one 64 GB node with RTX 5090; alternate 128 GB declared explicitly. Serving is CPU-only. Initial operational limits: preparation peak RSS <=48 GiB on the 64 GB profile (<=96 GiB alternate), server peak RSS <=4 GiB for the tested class-pair package; if these fail, implement streaming/index/lazy-loading improvements and remeasure. Do not label a 128 GB-only result ready for 64 GB.

Initial performance targets on local SSD after preparation: warm p95 <=500 ms for entity summary/search/one hierarchy page; <=1 s for a bounded pair detail; cold service-to-health <=10 s without parser/model initialization; <=2 s cold first bounded entity query. Measure at least 100 fixed mixed queries and modest four-reader concurrency, with versions and hardware recorded. These are engineering targets, not measured promises; a justified revision must be documented before declaring readiness, with the user-visible impact and replacement limit. No silent threshold relaxation.

Record index build wall time, peak RSS, artifact sizes, cache bounds and response sizes for full NCIT–DOID, plus repeat/cold-open behavior. Default JSON response budget is 2 MiB for summary/list routes; larger exact axioms use explicit detail/streamed access. Provider latency is recorded separately, never charged to read API latency. Count LLM calls/costs, retries, wasted interrupted work and avoided repeated generation. The single-node matching campaign's 336–504 hours remains separately governed; avoid concurrently competing heavy jobs.

## Handoff bundle

Publish `backend-handoff.json` with contract/schema versions, passed-gate evidence refs, exact runtime launch/build commands, package/ontology/run/policy hashes, API/OpenAPI and typed client artifacts, fixture inventory, supported/unsupported capabilities, measured resource profile, continuation/repair commands and remaining optional work. Secrets and answer keys are excluded.

Include local-import/demo/study deployment profiles and a synthetic full-study fixture client, study schemas, response/scoring examples, database migrations/restore proof, invitation issuance/resume commands and a Render runbook. PostgreSQL behavior must be exercised against a real test database, not only mocked. B5 does not require recruitment or a public deployment; real Render restart/restore and end-to-end frontend checks are mandatory before the later study launch.

Include a self-contained class-pair demonstration with real current Exact artifacts; good, sparse, ambiguous and error fixtures; restricted study condition examples; screenshots are not a substitute for working API responses. The frontend can browse entities and inspect a selected pair against this package without the original training environment or LLM access. Provide a real explanation generation example as a frozen artifact, not a required live call.

Acceptance owner signs G5 by machine-readable ledger plus concise report. F1 may request justified additive API changes during implementation; version them and rerun affected contract checks. The delayed specialist does not require the backend to predict every final layout choice.

## Post-frontend evaluation

F1 implements both products, then formative usability and a duration pilot precede the frozen ranking study in [11](11-study-design-and-ranking.md) and [human validation](../experiments/HUMAN-VALIDATION.md). Participants install Protégé and the ontologies before scored tasks. Compare explanation-assisted ranking with candidates/scores plus access to external inspection; record actual tools used after each case. Measure positive-case MRR/Top-1 and negative-case false endorsement/explicit-none separately. Counterbalance cases and order, account for participant/case clustering and expertise, and keep final cases separate from development. Pass a Render staging session/redeploy/database-restore rehearsal, version/data isolation checks and final launch readiness before enrollment. Participant numbers, recruitment and public launch remain separately authorized and scheduled.
