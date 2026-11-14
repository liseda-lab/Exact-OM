# Ontology-only descriptions and score display — 14 November 2026

## Release scope

Runtime implementation: `60ada8f` on `study-deploy`. Live study revision:
`exact2-ncit-doid-production-20261114-r2`. The owner completed the sudo-authenticated activation. The new data and frontend
are live on AppVM, the service is active, and public HTTPS readiness passes.
The publication receipt binds frozen hash
`d3ad076662848f6104e0c127c2f4ac59dd904b23dff96a163d6714c9d70751a4`.
The public browser check passed welcome, consent, tutorial-to-ranking, score
formatting against raw API values, profile boundaries and strict CSP, with no
browser errors (14 November 2026, 16:53 UTC).

The owner withdrew permission to use external descriptions. No external text,
URLs or citations were added. The unfinished external-description implementation
was removed before this release. Descriptions remain grounded in the supplied
original ontology facts and available only through the tool condition.

The score display now distinguishes nonzero values rounded to zero (`≈0.00`) and
values rounded to one (`≈1.00`) from genuine endpoints (`0.00`, `1.00`). This applies
to study candidates, the tutorial, exploration candidates and score/trace panels.
The underlying scores and ordering are unchanged. In the reported neuroaxonal
case, the five scores display as `≈1.00`, followed by four `≈0.00` values.

## Description review

The valid project API key successfully generated 36 reviewable drafts containing
39 claims; one of the 37 attempts abstained. The assistant reviewed each claim
against its actual cited literal or typed axiom, independently of the generating
model. This is ontology grounding review, not biomedical adjudication.

Decisions: **32 approved claims in 29 profiles; seven drafts rejected**. Rejections
covered parent-only summaries that omit available details, omitted anatomical or
cell-origin distinctions, ambiguous source wording silently rewritten, and preferred
labels replaced with synonyms without explanation. The seven rejected drafts and
the abstention retain their previously admitted material or typed fallback; no
rejected draft was substituted into the publication.

A proposed additional eight-request generation pass was rejected by automatic
approval review because it would send private study packets to the external
provider without sufficiently explicit authorization for that additional transfer.
It was not run or bypassed. Deployment uses only already-generated, reviewed drafts.
The failed expired-key attempt and successful valid-key attempt remain archived.

The release updates **29 profile resources across 22 cases**. Five previously
fallback-only concepts gain approved generated descriptions, giving **232 generated
entity profiles and 196 generated comparisons** across the unchanged 40-source pool.
Five entity profiles still use typed fallbacks:

- `DOID_0060160`: childhood spinal muscular atrophy.
- `DOID_12960`: acrocephalosyndactylia.
- `DOID_3565`: meningioma.
- `DOID_3649`: pyruvate decarboxylase deficiency.
- `DOID_2367`: neuroaxonal dystrophy.

The ontology supplies no original definition for neuroaxonal dystrophy or
meningothelial meningioma. Their parent classification does not support inventing a
more distinguishing clinical definition. The latter's existing limited generated
profile remains; the former retains its typed classification. Other revised
profiles preserve specific genes, inheritance, onset, anatomical sites or symptoms
where those details occur in the frozen facts. Existing malformed ontology wording
is not silently corrected using external knowledge.

## Preservation and verification

The preparation changes only the revision identifier, 29 profile-resource entries
and the 22 affected workspace indexes. The other **416 asset entries are identical**,
including ontology downloads and tutorial assets. All other study-definition fields,
all candidate scores, positions, identities, reference keys and the sampling protocol
are identical to the preceding live publication. Workspace table comparison checks
unchanged pair comparisons, evidence, matcher metadata and entity/task membership;
new profile IDs are expected. The full ontology databases retain their identity.
No full bundle, matcher run or classifier run was created.

Verified on AppVM:

- Strict publication admission in isolated PostgreSQL.
- Two complete ten-case backend journeys, starting in opposite conditions; each
  participant receives ranks 1–5 once in each condition, and test sessions remain
  excluded from ordinary exports.
- All 144 frontend unit tests and TypeScript checking pass; the production static
  export built successfully.
- Browser acceptance checks cover two ten-case journeys, score rendering from raw
  API values in both conditions, absence of generated blocks in baseline, tutorial
  skip, candidate navigation, progress, themes, evidence graph, final welcome text,
  responsive layout and stalled-load recovery. **All five browser tests passed** (no failures or skips).

Publication SHA-256:
`25340d36fab81b7415d5c210c74320a892ce343807b9c2176c4b4b05906487da`.
Combined reviewed-artifact SHA-256:
`c2483a7d38e61a964a20c5404550b4f7d77db49f9909568d6d4f55bd65c9e2c6`.
Independent decision-file SHA-256:
`3b169ee0063d50cf4bf6e032115ffd9b1d02fc08bae10d60db02358164940d68`.

## Activation and retained evidence

On AppVM:

- Frontend release: `~/exact-study/releases/description-scores-20261114`.
- New publication, validation, review decisions, content audit and database backup:
  `~/exact-study/descriptions-20261114/private`.
- Preparation, unit, type and browser logs: `~/exact-study/descriptions-20261114`.
- Activation command from the owner's machine:
  `ssh -t AppVM '~/exact-study/activate-description-scores-20261114.sh'`.

The activation script authenticates sudo before any live changes, verifies the
preparation and acceptance receipts, switches the versioned frontend, restarts the
service, checks readiness, publishes the new immutable revision, creates private
test invitations and checks the public website. Existing hashed assets are retained
for already-open tabs. A failed readiness check restores the previous frontend.

Existing publications, invitations and responses remain frozen. To review the new
descriptions, select `exact2-ncit-doid-production-20261114-r2` in Admin and issue a
new test invitation; an old invitation intentionally retains its original data.
The score display is application-wide after activation.

Compute-server generation and admission evidence remains in
`/mnt/datasets/pgcotovio/exact-study-descriptions-20261114-v2`, including
`prose-valid-key`, `independent-decisions.json` and
`deployment-reviewed-results.json`. No credential is copied into the repository.
Local transfers were limited to small review files and checked using the shared
100 GB storage guard. Shared contexts and original frozen resources were reused.

## Live verification after activation

Confirmed on the public website at 17:10 UTC on 14 November 2026, using a dedicated
test session on the new revision: all ten scored cases were traversed, **19 approximate
score displays** matched their raw API values, **five revised generated profiles**
were visible with their approved text, and **all five no-tool cases** had neither
workspace/explanation access nor generated-description elements. There were no
browser errors. The baseline DOM assertion now checks the actual `.generated`
class as well as `.generated-block` rather than relying on the latter alone.

The private receipt is `public-content-verification.json` alongside the publication
and earlier public acceptance receipts. The isolated test server and PostgreSQL
instance were stopped after acceptance; the live service remains healthy. Final
combined local storage was 52.03 GB against the owner's 100 GB limit.
