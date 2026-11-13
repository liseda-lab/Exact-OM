# Final Exact 2.0 study preparation — 11 November 2026

**Selection update, 13 November:** the owner identified overly obvious cases.
The [case-quality correction](explanation-final-study-selection-quality-20261113.md)
records replacement proposals and reference-scope concerns. The preparation and
admission checks below remain historical evidence; they do not approve the original
40-case pool for research use.

## Delivery and publication boundary

This change prepares the final NCIT–DOID study for the owner's review. It does not publish
its answer pool to the live study or change existing participant sessions. The owner explicitly
requested inspection of the selected sources, candidates and their information before deployment.
The private catalogue includes a welcome/consent preview and local notes with JSON export.
Private catalogues, answer keys, provider responses, reviews and test credentials are outside Git
and outside the served asset root. Do not upload the owner catalogue to the public website.

The working branch is `codex/final-study-review`, based on `c0193ec` of `study-deploy`.
The canonical checkout has unrelated work; it has not been reset, cleaned or committed here.
The active checkout for this change is `/private/tmp/exact-study-nov10`.

## Participant changes

- Independently reviewed generated prose is labelled **Generated** in the UI. Review receipts
  remain available internally; changing the badge does not weaken admission.
- Ranking has a top progress bar and previous/next candidate buttons. Progress counts completed
  cases, not candidate selections. The condition changes after five cases in this protocol.
- Study participants can select the colour theme; the preference survives reload.
- Setup explains source concepts, five proposed matches, plausible candidates, ordering,
  exclusion, no-match and insufficient-information answers with a concrete disease/subtype example.
- An incomplete tutorial can be skipped only when enabled in the publication, through the
  assessment/question-mark section and an explicit confirmation dialog. Cancellation returns
  to the tutorial. Skipping records `skipped_at`, preserves outstanding work and attempts, and
  never fabricates a passed assessment or `completed_at`. Setup/background remain prerequisites.
- Inspection methods are requested after each **no-tool** case only. An explanation case goes
  directly to the next case. Exports distinguish `not_requested`, `unanswered` and `submitted`;
  omission is never encoded as a false no-use response.
- A session code appears at the foot of study pages for withdrawal requests without sharing
  the private invitation link.

Legacy publications retain their old schedules, consultation behaviour and publication hashes.
The added fields are opt-in and default values are omitted from the historical canonical form.

## Sampling and final-run identity

The user approved 40 sources with an unequal overall pool after the available rank-5 stratum
proved smaller than eight. The selected pool contains **9, 9, 9, 8 and 5 sources** at reference
ranks 1–5. Every participant receives ten distinct sources: one from every rank in the
explanation block and one from every rank in the no-tool block. Blocks alternate order within
separate test and participant allocation sequences; source selection/order are random and
persisted transactionally. Reload and mutation replay cannot reallocate a participant.

Each displayed set retains exactly the final matcher's five candidates, their order and scores.
Selection uses the final `queries.candidates` records, verified against the final submission,
not a reranking of historical pair scores. Every set has one known reference answer among its
five. Two rank-5 sources also have other reference answers outside those five; these are flagged
in the owner catalogue. This does not establish that every non-reference candidate is false.

Final run on `compute-lis5`:

- `/mnt/datasets/pgcotovio/exact-om-retrieval-20261024/final-bioml-20261110-v1/H0`
- final local submission `local-apply/ncit-doid.tsv`, SHA-256
  `9af4541dc44b35b8e32531214ecc03913f9e936352037812eff98a2ac50a8577`
- reference alignment SHA-256
  `1a2ca025dcdf9d5f34562ed90c9975c29731ee01d311c45d9a71dad3810e543b`
- frozen input manifest SHA-256
  `dbfd4b6bdc1bdb68e8fc1db3c232b203afaa9c00ce027a267a84ec4589cfa89b`

There are 224 distinct concepts across 40 sources and 200 pairs. Original definitions cover
all six concepts in 29 cases, five in nine cases and four in two cases. Selection prioritises
original definitions and usable hierarchy/restrictions/synonyms. The rank-5 stratum contains
C131422, C34880, C5105, C3434 and C84927; C34880 and C5105 have additional reference targets
outside their displayed five. Earlier inspection found only three strictly single-reference
sources; the accepted displayed-set criterion yields five.

The pool is purposive, not representative of full-run matcher accuracy. Rare ranks will repeat
more across participants. Related sources and repeated target candidates may produce learning
or carryover effects; the catalogue shows repeated candidates for owner review. Do not silently
replace sources or repair candidates to make answers easier. Any change needs a new selection
identity and revalidation of both blocks' rank balance.

## Information and consent

Prepared from the owner's supplied details: Pedro Giesteira Cotovio,
pgcotovio@ciencias.ulisboa.pt, one-month collection period, and one-year retention after
collection. The month is not presented as session completion time. Withdrawal wording proposes
emailing the session code for exclusion and deletion while identifiable session data is retained;
already published aggregate results cannot be removed individually. This requires an operational
researcher deletion process, including any restricted exports/backups covered by the final policy.

The draft explicitly leaves the session-duration estimate and institutional review requirement
unconfirmed. It does not claim ethics approval or exemption. Recruitment/publication is not
approved by the technical tests. The owner can inspect the complete wording in the catalogue's
**Welcome and consent** tab. `tools/final_study_copy.py` is the editable source.

## Data preparation and generated prose

Heavy work runs on the servers. Existing immutable, policy-filtered ontology contexts and
partial ELK results are reused, with their hashes and exact roots checked against the final run.
There is no new matcher run, classifier run, HermiT job or physical full ontology bundle.

The final run retains 7,414 selected matcher features: 3,201 resolve to original facts and
4,213 remain explicitly unresolved. 199 of 200 pairs have original-fact links; the remaining
pair is a protected exact match with no selected evidence. No saved graph projections or
cross-ontology bridge records exist in this run. The frontend can derive visible ontology edges
from admitted original ASTs; never fabricate matcher bridges or pretend missing provenance was
recorded. The final real-data graph still requires visual inspection in the participant app.

Luna preparation used `openai/gpt-6-luna`, OpenAI provider, 424 bounded requests, four workers,
no automatic retries/fallback model, and a USD 5 ceiling. The frozen conservative maximum was
USD 2.17005775; this is not a claim about the final billed cost. Inputs contain admitted facts
and independent typed profiles, not reference answers, original ranks or scores.

- 421 reviewable outputs, two contract failures, one abstention.
- The Codex assistant separately read all 461 candidate claims against their cited literals
  and typed axioms: **435 approved, 26 rejected**.
- 400 generated artifacts retain approved claims. The other 24 retain typed original-fact
  descriptions. In partly approved outputs, rejected clauses are absent.
- Receipts bind exact source, response, packet, policy, manifest and ordered claim hashes.
  Raw generation and rejections remain in the private archive. Admission cannot infer approvals.
- This is independent grounding review of saved Luna output, not external biomedical validation.
  Comparison inputs remain the original independent typed profiles used for generation; the
  later reviewed Luna profiles are not retroactively represented as generation inputs.

Rejected examples include ambiguous OR/causal wording, a claim referring to the wrong feature
list, unsupported claims about missing information, treating synonymous cause/material-basis
wording as a substantive difference, unhelpfully generic descriptions and inconsistent original
enzyme names. Original ontology issues remain visible: the H6PD phosphatase/dehydrogenase
inconsistency, atopic-dermatitis classification, FGFR1 AND/OR wording and dated terminology.
Do not silently correct these using outside knowledge in the study's frozen original data.

Private server locations:

- compute-lis5: `/mnt/datasets/pgcotovio/exact-study-preparation-20261111-01/`
  (`luna-final-v1/plan.json`, generation, review requests, independent reviews, admitted results).
- AppVM: `/home/pcotovio/exact-study/final-preparation-20261111/private/`
  (catalogues, reference keys, reviewed results, owner page and review publication).
- AppVM generated assets: `/home/pcotovio/exact-study/assets/final-pool-owner-review-20261111-v4/`
  and small workspace indices under the existing `assets/workspaces/` root.

The final private publication keeps purpose `development_review`, unresolved scoring keys and
correctness metrics disabled; real reference keys are retained separately. Promotion to research
requires a new publication with the reviewed protocol and actual private keys, not relabelling
test responses as research observations.

## Corrected preparation defects

1. Referenced-label budgeting omitted JSON array punctuation; count the actual serialized list.
2. Frozen-resource reconstruction inflated packet labels with transport-only/null fields,
   incorrectly rejecting valid 48 KiB packets. Reconstruct the exact admitted packet projection.
3. Combining all eleven descriptions exceeded an existing resource label bound. Partition into
   six entity-profile and five pair-comparison resources while presenting one case/workspace.
   Existing byte/count limits are preserved and cited labels are not dropped.
4. Changing the tutorial's compatible-build list requires a new complete tutorial hash.
5. Every review publication needs an actual timezone-aware closing date, not null.
6. Preparation dependencies are isolated on AppVM; the live service environment is unchanged.

Failed preparation revisions are retained as diagnostic evidence and are not live publications.

## Verification and operational notes

Completed before this handoff: TypeScript checking, production export, 142 frontend unit tests,
and two full ten-case Chrome journeys in opposite condition orders. The journeys check tutorial
skip/cancel, honest skip state, theme persistence, progress, previous/next navigation, reload,
ten distinct cases, baseline explanation denial, selective methods and arrival at the final form.
The initial 45-test backend run passed with two opt-in skips. Later focused regression results
and real-publication verification are recorded in the completion addendum below.

The owner catalogue was opened in the in-app browser; the rank-5 filter shows five sources,
prepared generated text opens, and the welcome/consent tab renders with the pending items.

Two early browser test failures were test-harness issues: waiting for `stage=case` accepted the
previous case before save acknowledgement; rebuilding the static export underneath a running
server left its static/CSP bindings stale. Waiting for completed-case advancement and restarting
the owned test server resolved those runs. Use versioned builds and restart when switching them.
This does not establish the cause of the historical unreproduced live black-screen incident.

No new screen-reader audit, participant usability study, full final-data graph visual audit or
PostgreSQL concurrency campaign was completed in this change. Existing unrelated historical
issues are not declared fixed. The public live service and existing database are preserved.

Shared local development usage was approximately 51.9 GB before the final small review export,
below the combined 100 GB limit. The guard includes worktrees, datasets, dependencies and caches.
The owner HTML is about 31 MB; complete ontology databases remain remote/reused.

## Reproduction

Use the installed development Python environment and run local storage-growing operations via
`tools.review_storage_budget` in the canonical checkout. New tools support `--help`:

1. `tools.prepare_final_study_catalogue`: verify final inputs, select unchanged five-candidate sets.
2. `tools.prepare_final_study_resources --inputs-only`: reuse exact contexts, retain private inputs.
3. `tools.final_study_prose prepare/generate`: frozen bounded generation; do not re-run completed calls.
4. `tools.admit_final_study_prose`: require complete separately authored claim decisions and bind receipts.
5. `tools.prepare_final_study_resources --reviewed-results ... --revision NEW`: assemble a new private
   review publication; the command does not publish into the live database.
6. `tools.render_final_study_catalogue ... --reviewed-results ...`: self-contained owner HTML,
   outside public assets. It contains reference answers and must remain private.

Backend checks: `pytest -q tests/explanation_study_sampling_test.py
 tests/explanation_final_study_catalogue_test.py tests/explanation_label_budget_test.py
 tests/explanation_luna_review_test.py tests/explanation_study_v2_test.py
 tests/explanation_study_test.py`. Browser checks: `final-study-protocol.spec.ts` using the explicit
synthetic ten-case harness, a separate test database and the production export. Never point these
mutating test journeys at real participant sessions.

## Completion addendum

The final focused backend run passed **53 tests**, with **two existing opt-in skips**,
in 145.57 seconds. `git diff --check` passed; changed Python files were formatted with Black
and isort. The complete 40-case/200-pair revision assembled successfully on AppVM and passed
its strict publication model, preserving all 7,414 selected-evidence records.

The generation plan SHA-256 is
`1d719497f29787ee464554462aa05abdf49fe34d5e7329d137a859ad22179bc7`;
the separately authored claim-decision file SHA-256 is
`f2e2330b3ad65185d70743b3dafd91523a44119bc3d68c7404d160506207a5ef`.
The reviewed revision's generated resource files total about 47 MB; the private preparation
folder on AppVM is about 183 MB. These are small additions over reused ontology data.

The portable owner page is retained locally at
`/Users/pedrocotovio/.codex/visualizations/2026/09/25/01a0d99e-ecf4-7430-ad31-3805420dcd7a/final-study-review.html`.
A local-only preview is at `http://127.0.0.1:19712/` while its review server is running.
The page contains private reference answers. Review notes remain in that browser until exported.

The full real publication then passed `StudyStore.publish` in an isolated SQLite test database
on AppVM (131.8 seconds), including original-fact/resource/workspace admission. Two real-data
service journeys each reached the final questionnaire with ten distinct sources and one rank
1–5 per condition; source-context reads succeeded for explanation cases, baseline workspaces
and explanation references remained absent, and methods were collected only after baseline.
The combined journey check took 50.3 seconds after publication. These were service-level checks,
not a substitute for the separately completed synthetic browser journeys or real graph visual QA.

The private verification helper initially used obsolete client/setup request versions; the
service correctly refused them. It was rerun with `study-review/1`, `setup/3` and
`consultation/3`, matching the current browser. The rejected attempts affected only the separate
test database. Logs are `verify-real.log` and `verify-journeys-v3.log` under the AppVM preparation
folder; no test was pointed at the live database.

Final review publication SHA-256:
`bef48ac4b723fe142f64f3fc39a0c74ea853f15d6150b156602593a8d501ed92`.
Final combined local usage: **52,064,839,271 bytes (52.06 GB)**, with approximately 89.27 GB
filesystem free. The synthetic browser server was stopped; the local owner-catalogue server
remains available for review. All implementation changes are committed on and pushed to
`codex/final-study-review`; the live `study-deploy` branch/service has not been switched to this
new protocol or pool.
