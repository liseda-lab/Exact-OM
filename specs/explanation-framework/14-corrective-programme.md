# Corrective programme: shared exploration and study readiness

**Date: 2026-10-02. Status: approved corrective direction, implementation pending.**
This amendment follows the user's review of the implemented tool and the bounded
source/browser review at `57e501a`. It is an implementation assignment, not a report
that the corrections already work. Read the [verification record](evidence/corrective-review-20261002.md),
[frontend assignment](15-frontend-corrections.md) and [backend assignment](16-backend-corrections.md).

## Authority and scope

This amendment supersedes mandatory Protégé installation/opening in 07–09, 11–13,
the development/study blueprints and the human-validation brief. It supersedes the initial B0–B5-before-F1 sequence
**for this corrective iteration only**: both implementations now exist. The original
backend gates, information policy, matching contracts and launch gates remain in force.
Historical verification reports and published study revisions remain immutable evidence;
do not edit them to imply that they followed the new protocol.

The explanation condition must expose the same permitted ontology/explanation workspace
as the main app. The study shell adds consent, preparation, task training, assignment,
ranking, consultation and progress. It must not maintain simplified alternative entity,
comparison, hierarchy and evidence implementations. Baseline intentionally has no
integrated explanation tools. Reuse does not authorize unrestricted exploration APIs,
other scored cases, upload, researcher controls or answer-key access in the study.

No matching changes, new biomedical conclusions, runtime generation, participant contact,
deployment or recruitment are authorized by these specification changes. Preserve
unrelated working-tree files. This is a complete assignment for the **identified** issues
and regression risks, not a claim that all possible defects have been discovered.

## Decisions on external inspection and reporting

1. **Protégé is a recommendation, never an eligibility requirement.** Participants may
   use any ontology inspection method or combination: an ontology editor, a browser/viewer,
   file inspection, queries/scripts, a reasoner, another method, or no external method.
   Do not require installing software, having windows open or asserting use that did not occur.
2. Make both frozen ontology resources accessible in both conditions. Identify their roles,
   versions and relevant information restrictions. Method freedom does not silently change
   the supplied information universe: an online viewer may expose a different release or
   prohibited mapping annotations. Explain that distinction neutrally; permit reporting an
   outside/different/unknown resource without changing the ranking or silently excluding it.
   Freeze any analysis treatment of such reports before collection. Do not prescribe the
   tool a participant must use to satisfy the information policy.
3. A **case** is one source and its five candidate mappings, with one submitted ranking.
   Ask about actual external inspection **after each case**, in both conditions. This is
   not one report for an entire NCIT–DOID ontology pair, nor five mandatory questionnaires
   interrupting the evaluation of the five candidates.
4. Permit multiple methods and changes within and between cases. A background question
   about familiar tools is optional descriptive context, never a promise to keep using them.
   No study-wide method lock. A previous-case answer may be offered for explicit reuse,
   but never silently preselected, submitted or inferred from downloads/visibility.
5. Required reporting: explicit Yes/No; Yes requires at least one method; No requires an
   empty method list. Include an optional resource-scope report (supplied files,
   different/additional version/resource, unsure); unanswered stays unknown. Optional short
   tool names are not URLs, browsing histories, paths or identifying details.
6. Do not collect per-candidate method attribution in this revision. A case-level method
   set does not establish which candidate used which method or how long any tool was used.
   If a later research question requires that, amend the protocol and burden pilot first.
7. Internal workspace use is distinct from external consultation. Case time includes
   external inspection before submission; switching tabs/windows does not pause it.
   Post-submission reporting and tutorial/assessment time are excluded from case-task time.
   Report tool-use associations descriptively; do not infer a causal benefit of self-selected
   methods or adjust the primary condition comparison for a mediator affected by condition.

Suggested participant wording, to freeze in the new publication:

> You may inspect the supplied ontologies using any method or combination of methods.
> Protégé is one option; installing or using it is not required. You may also complete a
> case without external tools. Use the supplied ontology versions and follow the information
> guidance below. You can change methods between cases. After each ranking, we will ask
> which methods, if any, you used for that source and its candidates.

> For the case you just completed, did you inspect ontology information outside this
> study interface? Select all methods you actually used. If you used several methods
> for different candidates, include all of them.

These are task instructions, not invented ethics approval or replacement consent text.

## Interactive tutorial and short assessment

Keep the useful controls practice, but extend it into the actual shared workspace with
small, explicitly synthetic ontology resources, facts and prepared explanations. All
participants receive identical pre-assignment exposure to both condition workflows.
Practice identities, downloads, explanations and answer material must be disjoint from
scored and public-demo cases; visual design and interactions are shared with the product.

| Lesson | Required interaction and observable result |
|---|---|
| Identity and candidate inspection | Select a candidate, read source and target identity/meaning, return to a prior candidate without changing the answer. Explain initial position versus participant rank. |
| Context | Find a synthetic entity, navigate a parent and child, return to the compared entity. Show multiple inheritance and explicit missing/truncated information. |
| Evidence | Open a generated claim's citation, inspect its original axiom and origin, then locate the same evidence in the accessible list. |
| Optional graph | Inspect an edge, identify its type, zoom/fit and retain the view while editing a practice rank. Offer the equivalent list interaction to keyboard/screen-reader users. |
| Answer controls | Add, move, remove, undo, submit a partial ranking, explicitly keep initial order, and distinguish None of these from Insufficient information. Include a complex qualified example. |
| Baseline and external resources | Use the restricted baseline practice view, find/copy an IRI and locate the two downloads. No installation or actual external use is mandatory. Practice a multiple-method report and a No report. |

Follow this with **five short comprehension items**, using synthetic examples only:
(1) a matching score is advice, not a probability or answer key; (2) equivalence differs
from relatedness/broader/narrower meaning; (3) explicit none differs from insufficient
information and an untouched draft; (4) original facts, projected matcher evidence and
generated prose have different origins, and missing information is not contradiction;
(5) external methods are optional/combinable, may change per case and are reported per case.
Items 3 and 4 may use two-part responses while remaining one displayed item each.

Initial assessment content for implementation and the burden pilot follows. Freeze stable
question/option codes, explicit display order and these answer predicates with the tutorial;
wording changes require a new tutorial/assessment version. This is synthetic training
answer material, never a research case key.

| ID | Prompt and responses | Pass predicate / corrective feedback |
|---|---|---|
| Q1 `score_meaning` | “A candidate is first in the initial order and has the highest matching score. What does this tell you?” Options: `advice` — the matcher suggests it; inspect the evidence; `certain` — it must be equivalent; `probability` — the score is necessarily a probability of correctness. | `advice`. Explain that rank and score are recorded system advice; neither proves equivalence or calibration. |
| Q2 `equivalence_scope` | “In this practice ontology, the source means red circles. The candidate means circles of any colour and explicitly includes blue circles. Are these the same concept?” Options: `same` — yes; `broader` — no, the candidate is broader; `unrelated` — no, they are unrelated. | `broader`. Explain that overlap or a broader category is not equivalent meaning. Link back to the two practice definitions. |
| Q3 `response_states` | Match three situations to `none_of_these`, `insufficient_evidence`, `unanswered`: “You judged that none of the displayed candidates is equivalent”; “You cannot make a justified judgment”; “You have not chosen or submitted an answer.” | All three matched in that order. Explain that None is an explicit judgment, Insufficient information expresses uncertainty, and an empty draft is not a submitted response. |
| Q4 `evidence_origin` | Part A: match “statement in the ontology,” “feature selected/projected by Exact,” and “prepared generated summary” to `original`, `matcher`, `generated`. Part B: “The candidate has no definition in the loaded scope. Does that alone establish a contradiction with the source?” Yes/No. | All origin matches correct and Part B `no`. Explain how to inspect an original citation and why absent/filtered/unprepared information does not prove incompatibility. Provide an accessible return to the fact-inspection lesson. |
| Q5 `external_methods` | “Which statements are true?” Select all: `optional` — external inspection is optional; `combine_change` — I may combine methods and change them between cases; `per_case` — I report actual methods after each source-and-candidates case; `protege_required` — I must use Protégé; `locked` — I must keep one method for the whole study. | Exactly `optional`, `combine_change`, `per_case`. Explain optional use, changing combinations, and that the report covers the completed case rather than each individual candidate. |

Question order Q1–Q5 is fixed for this draft. Response controls use stable codes rather
than answer-position checks; Q3/Q4 require accessible matching controls, not drag-only
interaction. Keep partial assessment responses as drafts separately from submitted attempts.
Pilot readability and duration; do not add biomedical knowledge questions to make the
assessment look more rigorous.

Every core item must eventually be answered correctly before scored allocation. Give
immediate, specific corrective feedback and an accessible route back to the relevant
lesson. Allow unlimited retries, help, pause and later resumption; no speed threshold,
one-shot screening or silent participant exclusion. This is a task-understanding check,
not a validated expertise measure. Author and pilot the exact questions, explanations
and pass predicates before publication. Keep any unresolved completion as incomplete,
with a reason, not an incorrect scored response. Backend-acknowledged progress is required;
a list of checked tutorial indices is insufficient. Client action reports are not proof
that a participant read or understood content.

Tutorial help can be reopened during cases with synthetic content and without losing a
draft or stopping task time automatically. A synthetic help request must never retrieve
another scored case or unlock baseline explanations. Record the help opening separately.

## Issue register and ownership

Priority P1 means required before a participant pilot; P2 means required for the corrected
product/release acceptance unless explicitly marked a conditional usability enhancement.
Evidence classes: **observed** = reproduced during this review; **source** = established
by current code; **design** = required improvement; **regression** = retain/prove a previous
fix; **unverified** = a check remains, not a newly proven bug. Frontend and backend sections
use these IDs as acceptance-test IDs; none may disappear without a recorded disposition.

| ID | Priority / evidence | Issue and required result | Owner |
|---|---|---|---|
| C01 | P1 / source, design | Replace the reduced study engine with the shared permitted workspace; preserve baseline restrictions. | Frontend + backend |
| C02 | P1 / source | Bounded focal study resources cannot support full search/navigation. Supply policy-scoped, paged context and original-fact resolution. | Backend + frontend adapter |
| C03 | P1 / observed | Main-app citations have missing targets; study prose omits citation controls. Every admitted citation opens the correct fact, including hidden/deduplicated/paged facts. | Frontend + backend resolver |
| C04 | P1 / source, observed | Study Evidence lacks original-axiom inspection, drops unsupported nonliteral readings and hardcodes provenance; adapter drops capabilities. Preserve typed origin, missingness, original fallback and counts. | Both |
| C05 | P1 / observed | Questionnaire options/matrix columns are reordered by canonical JSON keys. Freeze explicit order independently of object serialization. | Backend + frontend |
| C06 | P1 / observed | Ranking and comparison are buried beneath long cards with competing scroll regions. Keep selection, answer status and comparison usable together. | Frontend |
| C07 | P1 / observed | Editing a ranking recreates the graph and resets zoom. Preserve view/selection across unrelated renders and saving. | Frontend |
| C08 | P2 / observed | Unsupported graph double-line style disagrees with legend. Use supported, accessible visual encodings. | Frontend |
| C09 | P1 / observed | Study tabs lack main-app keyboard behavior and tab/panel relationships. Share accessible primitives. | Frontend |
| C10 | P1 / design, source | Existing shape practice does not exercise explanation tools. Add synthetic interactive lessons in the real workspace. | Both |
| C11 | P1 / source, probe | Tutorial gating accepts indices without practice/assessment evidence. Add a versioned, server-validated comprehension gate. | Backend + frontend |
| C12 | P1 / source, probe | UI, server, setup model and older protocol mandate Protégé. Make inspection tool-neutral throughout. | Both + protocol |
| C13 | P1 / verified existing + amendment | Preserve multi-method per-case reporting; clarify granularity, freedom to switch and information scope. No study-wide method lock or automatic answers. | Both + protocol |
| C14 | P1 / source | Practice selections/completions are local component state and disappear on remount. Persist acknowledged progress and attempts across devices/reloads. | Both |
| C15 | P1 / source | Consultation fields are local until final save; an unsubmitted report is lost on reload. Add draft/submit separation and recovery. | Both |
| C16 | P1 / source, regression | Opening Hierarchy emits hierarchy_expand without an expansion; practice is outside current event scope. Align events with actions and preserve honest timing gaps/retries. | Both |
| C17 | P2 / observed | SQLite portable export relies on implicit URI support. Explicitly enable URI handling and test both runtime behaviors. | Backend |
| C18 | P1 / source, regression | Preserve qualifiers, subject-scoped shared axioms, referenced labels and availability states; never imply absence/contradiction from truncation or unsupported rendering. | Both |
| C19 | P1 / source, design | Downloads use asset IDs without role/name; access differs between case layouts. Provide equal, persistent access and resource metadata in both conditions. | Both |
| C20 | P2 / existing limitation | Candidate paging is by pair ID and the UI caps loads at 500. Preserve original rank, expose partial coverage and make continuation usable. | Both |
| C21 | P2 / existing limitation, unverified | Researcher revision selection is manual; review publish/progress/invitation/export/close ergonomics. Add an authenticated revision selector if needed; never expose credentials or keys. | Both |
| C22 | P2 / existing limitation | Library has limited metadata/no deletion; reviews are browser-local. Make these limits explicit; add safe library-copy deletion and useful metadata for spec 10 acceptance. Durable review storage is a separately declared capability, not implied. | Both |
| C23 | P1 / regression | Preserve session-bound outbox, idempotency, conflicts, stage transitions, profile isolation, revocation and resource-failure recovery across new routes. | Both |
| C24 | P1 / observed gap, unverified | Accessibility audits must include actual cases, graph/list, assessment and consultation; verify reflow, touch, keyboard, screen reader and both themes. | Frontend |
| C25 | P1 / unverified | Re-run real-package integration, PostgreSQL and deployment/restore gates. Prior receipts and synthetic review do not establish current release readiness. | Both |
| C26 | P1 / design | Setup/tutorial/consultation and form-order changes require versioned publication, migration and export semantics. Do not rewrite frozen sessions. | Backend + frontend compatibility |
| C27 | P1 / regression | Retain subsequent fixes and distinguish historical defects from current ones; publish evidence and updated handoff without unsupported completion claims. | Both |

## Implementation order and handoffs

**Frontend first for workflow design and the contract inventory, not frontend-only
implementation against imaginary endpoints.** Suggested ordered work:

1. Frontend reads all three corrective specs and reviews existing code. Deliver the shared
   workspace composition, case layout, interactive tutorial/assessment flow, method-report
   wording, capability matrix and an exact list of data/route needs. Demonstrate them with
   clearly labeled synthetic fixtures. Include loading, failure, partial and recovery states.
2. Jointly freeze the corrective contract described in 16: typed entities/facts, safe scope,
   tutorial evidence, setup, consultation drafts and ordered forms. Record representative
   success/error fixtures and compatibility decisions in the backend handoff. Backend can
   start known isolated fixes (C05, C12 design, C17) immediately; frontend can fix C03,
   C06–C09 and shared primitives in parallel without weakening current server gates.
3. Backend implements/adopts the frozen contract and proves boundaries/recovery. Frontend
   integrates it; remove temporary fixture adapters from production paths. Do not claim
   setup is fixed merely by hiding the checkbox while the backend still requires it.
4. Run the joint acceptance matrix, inspect the same real case in both products, complete
   accessibility and formative usability sessions, then re-estimate tutorial/case burden.
   The duration pilot and launch gates remain separate from implementation acceptance.

Readiness requires every P1 closed with evidence; P2 has an implemented result or explicit
scoped disposition that does not contradict existing required functionality. A discovery
outside this register must be added, classified and assigned, not silently ignored to
finish the checklist. See the exact deliverables and tests in 15 and 16.
