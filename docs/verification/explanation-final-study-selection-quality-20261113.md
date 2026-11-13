# Final-study case quality correction — 13 November 2026

## Decision

The first 40-case catalogue is a preparation result, **not an approved study
selection**. The owner found cases answerable by a disease/disorder name change.
The original selector prioritized counts of definitions, hierarchy, restrictions
and synonyms. It did not evaluate whether the alternatives were plausible or
whether those facts helped distinguish them. This favoured information-rich but
obvious category matches.

Use the following criteria for the replacement selection. Do not retain an easy
or questionable case solely to fill a quota. This correction concerns study data
selection; it does not require another frontend redesign, matcher run, full bundle
copy, or HermiT classification.

## Selection criteria

1. **Information quality.** Prefer original definitions for the source and all five
   candidates. A missing definition is acceptable only after a useful description
   has been prepared from admitted original facts and checked. Counts of axioms
   are insufficient: the displayed information must describe the concepts and
   contain distinctions relevant to this particular comparison. Verify linked
   evidence, hierarchy and generated descriptions actually available in the study.
2. **Useful challenge.** Exclude direct or near-direct name matches where the
   other choices provide little competition. Screen disease/disorder substitutions,
   capitalization, punctuation, reordered words, abbreviations and familiar
   synonyms. Preserve clinical qualifiers and do not normalize away numbers,
   negation, inheritance, anatomical sites or subtypes. A lexical screen is triage,
   not an equivalence or difficulty test.
3. **Plausible alternatives.** Identify at least two credible alternatives in the
   unchanged five, and explain why someone might consider them. Broad parent,
   related subtype, shared mechanism or similar presentation can provide genuine
   competition. Character overlap alone cannot establish it.
4. **Distinguishing information.** Record what an examiner can learn from the
   definitions, original facts or hierarchy that resolves the comparison: for
   example inheritance, gene, tissue, scope, cellular features or disease course.
   Retain original fact identities supporting that rationale. Do not count an
   obscure synonym alone as substantive difficulty. A set may be moderate rather
   than hard; actual difficulty depends on participants' expertise and needs pilot
   observation.
5. **Reference validity is independent of difficulty.** Compare source and
   reference definitions and scope, and inspect whether another shown candidate
   could also be equivalent. Hold wider/narrower, conflicting, obsolete or
   insufficiently supported mappings for adjudication. One known benchmark answer
   among five does not prove that answer is equivalent or that the other four are
   wrong. Do not silently repair the source records or change the frozen gold.
6. **Study integrity.** Preserve each original final-run five-candidate set, ranks,
   scores and source identity. Keep reference keys and screening rationales private;
   they must not enter provider packets, generated comparisons or participant
   assets. Keep the approved design of five cases per condition with one reference
   rank 1–5 in each, distinct source cases within a participant and counterbalanced
   block order. Pool-level quotas may be unequal. At least two suitable distinct
   sources per rank are needed; do not pad scarce ranks with invalid cases.
7. **Diversity and transfer.** Check shared candidate concepts, close source
   families and repeated reference targets across the ten assigned cases. Do not
   make difficulty, information richness, domain, or reference uncertainty
   systematically different between conditions. Record purposive selection and
   avoid interpreting the study pool as overall matcher accuracy.

## Work performed

- Re-read the frozen final run, its receipts/shards, reference and ontology context
  bindings on `compute-lis5`. Inventory: 1,773 eligible query sets, with distinct
  source capacities by reference rank of 1,656 / 65 / 20 / 10 / 5.
- Added `tools/audit_final_study_candidates.py`. Its private lexical screen catches
  exact/reordered labels after disease/disorder normalization and retains numbers
  and qualifiers. It reports definition coverage and lexical competition separately;
  neither is an admission decision. Same-named **distractors** are not incorrectly
  classified as obvious reference answers. The full inventory remains private.
- Inspected original definitions for the current pool's source/reference pairs and
  25 additional sets, with bounded hierarchy/restriction/synonym excerpts available
  in the private review. This is an evidence-based screening pass, not biomedical
  adjudication or proof that every candidate has been fully validated.
- Flagged 10 original cases for replacement, 13 for reference-scope review, and
  two for inconsistent original wording. Two further cases need attention because
  their reference has no original definition. These categories do not prove that
  the remaining cases are suitable.
- Prepared 12 replacement proposals: nine at rank 1 and three at rank 2. Eleven
  have original definitions for all six entities; one has five. Each has a specific
  rationale and any known information/scope limitation. Thirteen other inspected
  alternatives are shown separately as reserves or holds, with reasons.
- Updated the private catalogue renderer to support named review groups, selection
  notes, accurate “not yet extracted” evidence wording and safe initial note carryover
  for explicitly unchanged predecessor cases.

## Private review and preservation

The original review remains at `http://127.0.0.1:19712/` with its existing HTML and
browser-storage identity unchanged. The quality review is a separate
`http://127.0.0.1:19712/challenge-review.html` page. It opens on the 12 proposals;
the group selector also exposes the original 40 with quality flags and the 13
held/reserve alternatives. The original candidate lists and scores are unchanged.

The new page has its own notes key. On its first opening in the same browser/origin,
it can copy existing notes only for the unchanged original case identities; it
never writes the predecessor key. Subsequent notes on the two pages are independent.
Export remains the portable backup. No user notes were read or modified by the
agent's browser checks.

Private files are deliberately outside Git and public participant assets:

- Remote inventory and fact excerpts:
  `/mnt/datasets/pgcotovio/exact-study-preparation-20261111-01/challenge-review-20261113/`.
- Local inventory, excerpts, per-case selection record and reconstruction script:
  `/private/tmp/exact-review-work/challenge-*` and
  `/private/tmp/exact-review-work/build-challenge-review.py`.
- Durable HTML: `challenge-review.html` beside the existing `final-study-review.html`
  in the registered Codex visualizations directory.

The local private record binds the input files, predecessor HTML/catalogue and
new output hashes. Existing admitted entity descriptions are reused where available;
new candidate-pair comparisons, pair evidence and missing descriptions have **not**
been generated/admitted. Do not present this shortlist as a fully prepared new study
revision or deploy it to replace the current study.

## Remaining work

1. Resolve or replace the held references, especially scarce ranks 3–5. The 12
   proposals replace weak material but do not establish a suitable final pool of
   40. Use an explicit private selection record with case identity, reason,
   plausible alternatives, supporting fact identities, reference decision and
   preparation status; do not rerun the old coverage-only selector and call that
   quality approval.
2. Prepare and independently check only new/missing descriptions and comparisons,
   and admit original-fact-linked evidence for retained new pairs. Reuse unchanged
   outputs and partial ELK; never feed the selection rationale or gold to generation.
3. Validate distinct-source sampling, one case per rank in each block, overlap and
   information-quality balance on the resulting pool. Re-run strict publication
   admission and real ten-case journeys after those data changes.
4. Record expert reference adjudication and pilot difficulty separately from code
   and data checks. These are research judgments, not frontend/backend test results.
   Continue all agent-verifiable preparation without making owner feedback a generic
   stop condition. The owner's required review before live publication still applies.

No live deployment, participant data changes, inference job or provider calls were
performed in this correction.

## Verification

- `tests/explanation_study_challenge_test.py` and
  `tests/explanation_final_study_catalogue_test.py`: **7 passed**. Disease/disorder
  normalization, reordered words, preserved clinical qualifiers, same-named
  distractors, immutable candidate order and distinct-source selection are covered.
- Headless Chrome, isolated temporary browser profile: **65 of 65 case pages
  rendered**, zero page errors. All original source/target/rank/score/reference
  identities matched the verified inventory. The original 40 also matched the
  predecessor page independently.
- Verified the 12/40/13 review groups, label search, reference highlighting,
  initial predecessor-note copy, independent autosave and reload persistence.
  The test used synthetic notes in its own profile; the user's browser was untouched.
- Desktop (1440 px) and phone (375 px) screenshots inspected; no horizontal phone
  overflow. These checks concern the private review page, not a new study UI release.
- Predecessor HTML SHA-256 unchanged. New HTML and private selection-record input
  hashes verified. Private per-case records retain original definition fact IDs.
- Shared storage guard measured approximately **52.10 GB**, below the user's
  combined 100 GB limit. Only compact inventory/excerpts and one additional private
  HTML page were created; no ontology database or full bundle was copied.
- Black, isort and Git whitespace checks pass. The live service, participant
  database and frozen matcher artifacts were not changed.
