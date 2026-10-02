# Corrective review evidence and specification audit

**2026-10-02. Reviewed source baseline: `57e501a`.** This record supports
[14](../14-corrective-programme.md), [15](../15-frontend-corrections.md) and
[16](../16-backend-corrections.md). It distinguishes observed defects, source findings,
new protocol decisions, prior fixes and checks that have not been repeated.

No product implementation is changed by this specification task. No human participant
was contacted, no provider call made and no deployment performed. The existing untracked
`exact-repair-first-tests-20260929.tar.gz` is unrelated and was left untouched.

## 1. Verification of the user's follow-up

The Protégé requirement is not just a stray sentence. It exists in all these layers:

- [SetupStage and SETUP_CHECKS](../../../explanations_visualizer/src/components/study/Stages.tsx):
  installation and opening both files in Protégé, all required for readiness.
- [CaseView](../../../explanations_visualizer/src/components/study/CaseView.tsx): phone
  notice says a desktop with Protégé open; baseline instructions also prescribe it.
- [Setup model](../../../exact_inspect/study/models.py) and
  [setup state transition](../../../exact_inspect/study/store.py): a false installation
  assertion prevents progression, regardless of access through another tool.
- The pre-amendment 07/08/11/13, human-validation specification and design blueprint
  required installation. These normative texts are corrected in this change, while
  historical reports and implemented v1 runtime snapshots remain unchanged.

A fresh isolated SQLite probe reused `publication`, `invite`, `write` and
`background_answers` from [the study tests](../../../tests/explanation_study_test.py).
It published synthetic resources into a temporary directory, accepted synthetic consent,
and varied the setup installation field with all other readiness fields true. Results:

```text
protege_installed=False -> stage setup
protege_installed=True  -> stage background
background submitted, completed_tutorial_steps=[0,1,2,3],
without any practice-action or assessment submission -> stage case
Consultation(methods=[plain_files, other_resource], consulted_external_ontologies=True)
-> accepted
```

The first probe invocation had a harness constructor-argument error; it did not test the
behavior. The corrected invocation used `StudyStore(database_url, assets_dir=…, allow_test_sqlite=True)`
and produced the results above. Temporary probe resources were removed automatically.
No product data or active study sessions were mutated by this follow-up probe.

The method report is already per case and multi-select in the backend model, form and
frontend. It does not lock a participant to a study-wide method. The correction clarifies
that a case is one source and its five candidates, permits changes and adds durable drafts.
It does not introduce five candidate-level interruptions or force installation of another
tool instead. Freedom of method remains distinct from the study's frozen information scope.

Further source findings:

- `PracticeStage` keeps current practice, response and finished IDs in local component state;
  its final mutation sends tutorial indices, not the actual lesson/assessment evidence.
- `ConsultationStage` initializes fields empty, holds them locally and only sends the final
  save. An unfinished consultation is not covered by the otherwise persisted mutation queue.
- [ParticipantApp](../../../explanations_visualizer/src/components/study/ParticipantApp.tsx)
  decides whether setup changes stage using the same old installation fields; this must
  change together with models, queue routes and timer transition handling.
- [Telemetry](../../../explanations_visualizer/src/study/telemetry.ts) currently emits only
  case/presentation-bound events, so it cannot serve as durable tutorial completion state.
- [StudyExplanation](../../../explanations_visualizer/src/components/study/StudyExplanation.tsx)
  emits `hierarchy_expand` when merely opening the Hierarchy tab. Its resource normalization
  drops capabilities, uses hardcoded original interpretation for graph facts and can omit
  unsupported nonliteral evidence readings. These are source-confirmed problems, not a
  claim that every such data shape was reproduced in the small fixture.

## 2. Earlier live review in this conversation

Built current sources with a clean temporary dependency installation. Used real NCIT/DOID
excerpts, synthetic matcher values and an offline exact-excerpt generation stub from
[the fixture builder](../../../tools/build_explanation_ui_fixture.py). The full historical
backend-release package was not present locally. The study harness used explicit test-only
SQLite and loopback Origin rewriting; this is not production TLS/PostgreSQL evidence.

Verified a complete synthetic journey through setup, background, controls practice, three
scored cases in both conditions, consultation, final form and completion. Checked answer
restoration, pause/resume, explicit none/insufficient answers, ontology search including a
typed object property, original axioms, evidence graph/list and the study narrow-screen
notice. At 375 px the inspected study page did not overflow horizontally. The setup
checkboxes were synthetic test self-reports, not actual Protégé installation or human consent.

Observed findings, now assigned in 14:

| IDs | Concrete observation / source |
|---|---|
| C01, C02 | Study hierarchy renders only direct parents; graph expansion is disabled. Current builder prepares first pages with `limit=20`. Main Browse provides independent navigation. |
| C03 | Main Trisomy/Patau view had nine citation anchors; five had no destination element, including repeated missing label references and a hidden duplicate synonym. Study profiles/comparisons rendered no citation links. |
| C04 | Main Dravet Evidence offered original-axiom actions; the study Evidence tab did not. This does not mean all study cards lack an axiom button: restriction cards do have one. |
| C05 | Final helpfulness order: Did not use, Moderately, Not, Slightly, Very. Effort: Cannot judge, High, Low, Moderate, Very high, Very low. Sorted canonical dictionary keys reached `Object.entries` rendering. |
| C06 | At 1280×720 on a long case the ranking panel began around y=1176. A separate scrolling explanation column buried comparison beneath the candidate card. |
| C07 | On Dravet, zoom twice then add rank 1: same pair and Graph tab, but view resets to fit. New bundle object causes the graph creation effect to run again. |
| C08 | Cytoscape rejected the requested `double` line style; graph bridges rendered solid while the legend showed two strokes. |
| C09 | Study tabs all had tabIndex=0; Arrow Right did not move tab focus. Main detail tabs implement keyboard behavior. |
| C10 | Four authored shape lessons exercised response controls, without interactive use of the ontology/explanation workspace. |
| C17 | Portable context export failed with a URI-default-off SQLite build; the same attach operation worked with URI interpretation explicitly enabled. Source output connection omits that flag. |

Successful fresh checks in that live review: production frontend build and TypeScript;
15 frontend unit tests; 67 backend tests passed, one skipped across frontend-serving,
study-resource and study suites. The skip was the real PostgreSQL recovery check. Browser
work was manual automation of a synthetic fixture, not a newly run full existing e2e suite.
Some mouse actions had tooling coordinate discrepancies; keyboard activation worked.
Those discrepancies are not reported as product defects.

## 3. Historical findings reconciled rather than reopened indiscriminately

The [original handoff](../../../docs/verification/explanation-frontend-handoff.md) is
followed by [subsequent integration verification](../../../docs/verification/explanation-frontend-e2e.md).
The latter records real-package/PostgreSQL/HTTPS checks and fixes after that handoff.
Those historical results are not relabeled as freshly reproduced here.

| Prior issue | Current corrective disposition |
|---|---|
| Shared-axiom subject collision | Later builder keys facts by fact and typed subject. Retain regression under C18/C27; do not claim the old collision still exists. |
| Missing referenced labels | Later builder admits bounded referenced labels. C18 tests coverage/policy/unresolved status; small excerpt omissions are not proof this fix regressed. |
| Null ontology names | Later preparation revision supplies names. Retain versioned preparation regression; do not rewrite old locks. |
| Missing runs/labels/explanations discovery and typed clients | Implemented and subsequently typed. Keep explicit errors and bounded/policy-scoped tests. |
| Pair-ID paging versus rank / 500 cap | Still an explicit limitation; C20 requires honest scope and usable continuation. |
| Empty evidence display fields; base64 canonical originals | Documented representations, not automatically invalid data. C04/C18 require typed/original fallback and no silent omission. |
| Hardcoded provider provenance wording | Later generation revision uses neutral wording. Retain actual manifest/provider identity and offline-stub labeling. |
| Asset role/name and current-condition metadata | C19 addresses download metadata/access; current condition can still be safely sourced from cases/current. A missing convenience state field alone is not a correctness bug. |
| No admin study list | C21 is a workflow improvement/verification item, not evidence of broken authentication or export. |
| Browser-local review decisions / library delete | Disclose local retention; C22 provides library-copy deletion and metadata without inventing server review storage. |
| Practice placeholder | Later four synthetic control lessons exist. C10/C11/C14 address remaining actual-tool training, assessment and recovery. |
| Timing-stage race and outbox/session recovery | Later fixes exist. C16/C23 test new transitions, late telemetry and honest gaps; no claim that the previous happy-path race still occurs. |
| macOS RSS units | Previously fixed; retain verification regression. |
| Legacy/global-only files, search kinds, stale responses | Later fixes exist. C20/C27 retain them; full real-run legacy behavior was not retested in this review. |
| Node/venv stalls, preview restrictions, temporary fixture availability | Review-environment constraints; not product defects. Use reproducible isolated dependencies rather than misclassify them. |
| HTTPS cookies/origin and invitation trailing slash | Security/deployment behavior, not a request to weaken it. Earlier redirect preserved the invitation fragment. Keep real-origin testing and no production Origin rewrite. |
| G1/G5 parser/data-admission and launch readiness | No new evidence passes these gates; C25 retains them separately from UI implementation. |

## 4. Final specification audit

Audit the authored specs for coverage and consistency, not just spelling:

- All nine numbered findings in the preceding review map to C01–C11/C17, with current
  user concerns and newly source-confirmed recovery/telemetry gaps assigned through C27.
- Every C01–C27 has an owner, evidence classification, desired behavior and explicit
  acceptance scenario in frontend and/or backend assignments. Non-frontend runtime issues
  are not left to the frontend agent to bypass.
- Method reporting stays per source-plus-candidate-set case, multiple and changeable.
  Setup cannot force use; optional resource-scope answers stay optional/unknown if omitted.
- Tutorial event evidence does not masquerade as proof of reading. Server-evaluated
  assessment and acknowledged completion gate allocation, while retries/help remain available.
- Shared ontology navigation remains permitted without exposing other scored-case metadata.
  Baseline cannot gain explanation content via tutorial scopes, caches or guessed URLs.
- Legacy forms/sessions remain immutable; all proposed routes and v2 schemas are explicitly
  planned. Source runtime schemas are not falsely regenerated from unwritten code.
- Consultation drafts do not advance, modify rankings or become default No responses.
  Case timing includes external work and excludes post-submit consultation/preparation.
- Former backend-first/bootstrap instructions are explicitly superseded for this iteration;
  protocol/data/privacy and separate participant-launch gates are retained.

The final review also found a leftover `external_setup_required: [protege, …]` in the
development blueprint and a Protégé setup-pack requirement in 09. Both are amended;
the validator now checks development and study blueprints together. It also caught the
need to distinguish synthetic tutorial feedback from the ban on scored-case feedback,
and to define partial assessment drafts and new timing transition semantics explicitly.

Final checks completed successfully:

| Check | Result |
|---|---|
| `python specs/explanation-framework/protocol/validate_specs.py` | Passed: 24 JSON files, 13 schema fixtures, 129 local links, 8 study schedules, 9 scoring oracles; no errors. |
| Issue-to-acceptance coverage | C01–C27 enumerated exactly once in the programme register; all 27 appear in frontend/backend acceptance tables. |
| Protocol negative checks | Five deliberately contradictory in-memory mutations were rejected: mandatory Protégé, locked method choice, study-wide reporting, non-durable training and preparation included in case time. Unchanged protocol accepted. |
| `git diff --check` | Passed. |
| Source/runtime scope check | Changes restricted to specifications, blueprints, their validator and this evidence record. Product source, runtime OpenAPI snapshots and databases unchanged by this task. |

The Python checks used the isolated review dependencies via `PYTHONPATH`, Python 3.12
with `-S` and a temporary bytecode cache because the repository environment previously
stalled. The logical validator command above is reproducible in a working project
environment with jsonschema. These checks validate specifications, not implementation
of the corrections. The live-review build/test results in section 2 are not new tests of
an implemented v2 service.

### Checks still required before claiming the corrected tool is ready

Fresh real-package comparison, signed-in admin interaction, full import/corruption/cancel
and two-bundle switching, legacy real-run opening, PostgreSQL restart/concurrency/restore,
Docker image execution, Render rehearsal, manual screen-reader use, formative usability
and duration pilot were not completed by this specification task. Existing receipts are
useful historical evidence, not substitutes for the new acceptance matrix. No LLM-output
quality or human benefit was established by the synthetic review. These limits prevent
an honest claim that every possible issue in the tool has been ruled out.
