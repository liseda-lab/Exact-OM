# Forty-source pool review — 13 November 2026

## Result and approval boundary

The owner approved all twelve replacement proposals in the preceding
[case-quality review](explanation-final-study-selection-quality-20261113.md).
The new private catalogue retains those twelve, adds 23 cases selected by content
review, and presents five remaining choices with alternatives. It proposes **40
distinct sources**, each with its original five candidates. The default counts at
reference ranks 1–5 are **17 / 12 / 5 / 3 / 3**.

This is a concrete selection proposal, not a published study or a claim that all
40 cases have been approved. The five choices remain pending. No live service,
participant session, provider generation, ontology database or matcher output was
changed. Case identities, gold keys, choice rationales and private review files
are not committed to the repository or placed in participant/provider payloads.

## Content review

The review used the frozen final Exact 2.0 run and its existing inventory of 1,773
eligible query sets. It considered original definitions, inheritance, genes,
anatomical scope, cell types, mechanisms, synonyms and the plausibility of the
unchanged alternatives. Simple disease/disorder substitutions were not treated as
interesting comparisons. An identical-label distractor was not mistaken for an
obvious correct answer when the source definition specified a different subtype.

The 40 proposed cases have this original-definition coverage, counting the source
and five candidates separately:

| Defined entities per case | Cases |
| --- | ---: |
| 6 of 6 | 32 |
| 5 of 6 | 5 |
| 4 of 6 | 3 |

Each case has a private rationale and original definition fact identities. Existing
reviewed descriptions and citations are reused unchanged. These are suitability
judgments grounded in the frozen records, not independent biomedical adjudication
or measured difficulty estimates. The eight cases with missing definitions still
need useful descriptions from admitted original facts before publication.

The five decisions concern an inconsistent enzyme term, incomplete candidate
descriptions, a reference with only a generic description, broad target-ontology
classification wording, and an eponym whose difficulty depends on expertise. Four
alternative source sets are supplied with their own limitations. Some alternatives
are shared between decisions and cannot be chosen twice. One alternative changes
the pool quota while preserving the required two distinct sources per rank.

The default pool has two pairs of cases sharing a distractor; one alternative adds
another such overlap. Those overlaps are flagged in the private rationale. Block
counterbalancing does not eliminate possible transfer effects. Final preparation
must inspect the assignments and retain this limitation in the study analysis.

## Private review and preservation

The new page is served only on loopback:

`http://127.0.0.1:19712/pool40-review.html`

It opens on the five decisions. Other views expose the proposed forty, the twelve
approved cases, four alternatives, and all 44 inspected sets. Inspecting an
alternative does not select it. Choices and comments save in the browser and can
be exported. The page flags repeated sources, reused alternatives, selected
exclusions and insufficient sources at a rank.

The old `final-study-review.html` and `challenge-review.html` remain byte-identical,
with their storage keys unchanged. On first opening, the new page copies notes
only for unchanged case identities into its own key. The latest explicit approval
sets the twelve approved cases to Keep in that new copy, retaining their text and
earlier decision. Tests used an isolated browser profile, not the owner's notes.

Private local working files are under `/private/tmp/exact-review-work/`:

- `pool40-catalogue.json`: immutable selection input, facts, preserved prose,
  matcher evidence and five decision definitions.
- `pool40-review-record.json`: hashes, counts and context identities.
- `pool40-retained-evidence.json`: original records and receipt/role provenance.
- `pool40-validation.json`, browser captures and verification logs.
- `build-pool40-review.py`: private composition and rationale record.

The durable HTML is in the existing Codex visualization directory
`/Users/pedrocotovio/.codex/visualizations/2026/09/25/01a0d99e-ecf4-7430-ad31-3805420dcd7a/`.
The adjacent private `pool40-private-review-20261113.tar.gz` preserves the catalogue,
input excerpts, inventory, evidence, composition script and validation evidence.
Its manifest records entry hashes. It excludes synthetic browser choices; those
test files must never be interpreted as owner approval.

## Artifact identity and evidence correction

Frozen run root on `compute-lis5`:
`/mnt/datasets/pgcotovio/exact-om-retrieval-20261024/final-bioml-20261110-v1/H0`.
Native result SHA-256:
`d5b5b54e19ca04de29f5843e376bceb921fd4d9f5bd1e9e277b307c0e6581459`.
Catalogue SHA-256:
`c22bb3aa2b44b996ef814de5116fc4eec7077513099fafdac0e1cb8ca0c8a7cb`.

The data server was reachable and retained evidence was read there. All native
receipt and compressed shard hashes were verified. All **220 pairs** across the
44 inspected sets have retained records and selected features. This does not mean
all 220 have admitted study graph evidence.

Two pairs have both a feature-bearing scored row and a protected exact-match
anchor. The earlier coverage selector overwrote the row with the anchor, losing
the displayed features for one previously prepared pair. The new extractor keeps
both originals and their roles. Binding combines them only when the pair identity,
explicit protected-exact flag and final score agree. The anchor supplies the final
score; the scored row retains its features. Unexplained variants remain unresolved.
Resource preparation now retains the exact-match trace event even when that pair
also has selected evidence. The affected pair still needs evidence re-admission.

For additional ontology excerpts, existing filtered local contexts were reused.
Their full database hashes were verified, their root/ontology versions matched the
frozen run, and their physical visibility policy was respected. Context overrides
cannot substitute a different ontology version. Actual manifest/database/policy
identities are recorded in the private receipt. No full bundle was copied, shared
inode unlocked or reasoner restarted.

## Implementation and checks

- `audit_final_study_candidates.py`: verified context relocation/reuse.
- `extract_final_study_evidence.py`: hash-bound retained record extraction and
  explicit scored-row/exact-anchor binding.
- `final_study_selection.py`: frozen case verification, choice validation, and
  separate selection receipt/catalogue output. It always leaves publication false.
- `render_final_study_catalogue.py`: five-choice workflow, note migration,
  persistence/export, dynamic quotas, compact prose/citation preservation and
  accurate evidence variants.
- `prepare_final_study_resources.py`: preserve the protected-exact trace event.

Verification completed:

- All 44 case signatures match the original inventory: query, source, reference,
  ordered targets, original ranks and scores.
- All 243 offered combinations checked: 91 valid and 152 rejected. The 91 valid
  combinations were sampled with 100 seeds each: **9,100 assignments**, all with
  ten distinct sources and ranks 1–5 represented once in each condition.
- The real service allocator already shuffles case presentation order after
  stratum selection. A deterministic regression checks that actual participant
  order. Sampling runtime code is unchanged; no ordering defect was found.
- An earlier focused run passed 18 tests including the real allocator flow. After
  the final extraction, rendering and selection changes, 14 focused tests passed.
- Isolated Chrome checked all 44 rendered cases, five choices, alternative return
  navigation, unchanged predecessor notes, approval carryover, save/reload, text
  edits, duplicate rejection, exclusions and export. No page errors or horizontal
  overflow at 375 px. Desktop and phone captures were inspected.
- The CLI consumed a synthetic exported review and produced a validated selection
  receipt and compatible preparation catalogue. These test choices are clearly
  named synthetic and are not owner approval.
- Reviewed claims/citations and both predecessor HTML hashes were checked unchanged.
  Python formatting and `git diff --check` passed.

The shared storage guard covered growing operations. Combined local usage was
**52,053,220,287 bytes (52.05 GB)**, below the owner's **100 GB combined limit**;
the measurement is preserved beside the HTML in `pool40-storage-final.json`. This work added bounded
excerpts and private review artifacts, not another physical bundle.

## After the five choices

Export the real owner notes from the new page and validate them against the exact
catalogue and inventory. For example, from this checkout with the existing Python
environment, using new private output paths:

```sh
python -m tools.final_study_selection \
  --catalogue /private/tmp/exact-review-work/pool40-catalogue.json \
  --inventory /private/tmp/exact-review-work/challenge-inventory.json \
  --review-notes /path/to/actual-owner-export.json \
  --output /private/path/owner-selection.json \
  --selected-catalogue /private/path/owner-selected-catalogue.json
```

The selected catalogue supplies `options.forty_unequal` and its actual quotas for
the existing resource-preparation tool. Do not rerun the old coverage-only selector
or use a synthetic test selection. Carry the resulting private selection receipt
into preparation without sending reference answers or review rationales to a
provider or participant.

Prepare only missing/changed descriptions, comparisons and evidence resources,
reusing unchanged original contexts, partial ELK and independently reviewed prose.
Re-admit recovered exact-match evidence, inspect the resulting ten-case assignments
and information quality, then run the existing strict publication and participant
browser checks. The owner requested a selection review before deployment; the new
pool has not been deployed. Human suitability choices remain distinct from the
agents' responsibility to finish and verify preparation.
