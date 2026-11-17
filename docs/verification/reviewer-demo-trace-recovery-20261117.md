# Reviewer demo: decision recovery and feature audit

## Status

The corrected release is live at <https://exact-explain.lasige.di.fc.ul.pt/> from
`~/exact-demo/releases/reviewer-traces-20261117`, code revision `5daff82`. Public
verification completed on 17 November 2026 at 12:38 UTC. The owner renewed the
scoped deployment permission; it was revoked again after verification, and
passwordless activation was independently confirmed unavailable. Neither the
active study nor its publication has been modified by this recovery.

## Defect and correction

The initial demo adapter copied the latest study's descriptions and selected
evidence but filled every decision stage with `not_recorded`. That was a demo
preparation omission, not proof that the original matcher retained no decisions.
The original native result, applied query outputs, model-response bank and final
ranked TSV are still available on `compute-lis5`.

`tools/recover_reviewer_decisions.py` reads those originals on the data server and
verifies their hashes, compressed-shard receipts and uncompressed hashes. It binds
the selected source/query/candidate identities to the original 100-candidate input,
output order and saved TSV. It checks applied model signals against the response
bank. Only a 269 KB numeric/identity receipt was transferred; no raw model prompt,
reference answer, full ontology, provider key or matcher rerun was needed.

The original run is **local ranking**, not global alignment. Its TSV contains
`SrcEntity` and `TgtCandidates`; it does not assert that any ranked pair is accepted.
The corrected adapter and frontend therefore show original rank, native component
scores, applied pair score, query utility, actual model-signal usage and saved
ranking. Threshold, cardinality, relation typing and repair are marked as not used
by this original execution path. Stage summaries are explicitly derived from saved
artifacts, not presented as a contemporaneous event log. The saved application-code
hash supports this execution-mode interpretation.

All 50 pairs have native scores; nine used a recorded local model response and 41
did not apply one. None is a protected exact-match anchor. A score near or equal to
one is never used to infer exact matching. The contract rejects a ranked output
that simultaneously asserts accepted alignment membership or a typed relation.

The selected ten sources, their five displayed candidates, final scores, order,
latest r2 descriptions and policy-filtered evidence are preserved. Stable run and
pair identifiers preserve existing reviewer bookmarks across package revisions.

## Active study

The active `exact2-ncit-doid-production-20261114-r2` publication admits exactly:
original context, entity descriptions, hierarchy, evidence table, evidence graph
and pair comparison. Its workspace indexes have no trace component, and the case
shell adds no trace tab. This follows
[spec 15, F1](../../specs/explanation-framework/15-frontend-corrections.md), which
explicitly keeps traces optional in the main app unless separately admitted into
the study. The no-tool condition still shows identities, scores and ranking only.

Thus **participants do not currently see decision traces**. Adding them would be a
study-protocol/publication change, not this demo repair. No invitation, participant
record, frozen workspace, study HTML, study credential or study service is changed.

## Full-tool audit

| Capability | Result |
| --- | --- |
| Ten-source/fifty-pair selection | Same latest deployed r2 cases, two per reference-rank position; no reference answers served. |
| Decision trace | Restored for all 50 pairs from verified original records. |
| Component scores and model signals | Restored; native pair score, post-signal score and final query-ranking score remain distinct. |
| Accepted mappings, relation typing, repair, NIL | Not outputs of this local-ranking run. No false alignment or missing-history claims. |
| Candidate retrieval | Original provided 100-candidate query verified; no retrieval rank or channel invented from input-array order. The demo intentionally displays its first five. |
| Descriptions and comparisons | All 110 reads checked against latest frozen workspace claims and support packets. |
| Original context and citations | Reuses full prepared, policy-filtered ontology indexes; original-record opening remains available. |
| Hierarchy | Asserted and structural views available; existing ELK remains partial. No reasoner rerun or stronger completeness claim. |
| Selected evidence | 2,090 retained items across 50 pairs; 1,084 have resolved original-fact links and 1,006 remain unresolved in the prepared policy scope. No selected evidence omitted by this demo export. |
| Graph | Graph interface is present. Originals have no cross-side correspondence records for these pairs; all retained feature graph-projection statuses are `not_recorded`. Do not fabricate matcher bridges from channel scores. Original ontology facts and the admitted evidence can still be inspected. |
| Imports and library | Intentionally unavailable publicly. The full local application retains its library/import flow. |
| Review annotations | Remain browser-local; no server review store is implied. |
| Study/admin isolation | Separate upstreams and unchanged study assets/authentication responses; active study PID remains 230372. |

The selected pool does not demonstrate every possible matcher path: it contains no
protected exact anchors and is not a global-alignment/repair run. These are example
coverage limits, not missing tabs that should be populated with invented data.
Unresolved evidence, partial ELK and absent cross-side records also remain limits
of the source artifacts. The demo can showcase the full interface without proving
that every available feature has complete data in this run.

## Identities and checks

- Original root: `/mnt/datasets/pgcotovio/exact-om-retrieval-20261024/final-bioml-20261110-v1/H0`.
- Native test result: `d5b5b54e19ca04de29f5843e376bceb921fd4d9f5bd1e9e277b307c0e6581459`.
- Saved ranking: `9af4541dc44b35b8e32531214ecc03913f9e936352037812eff98a2ac50a8577`.
- Applied worker: `4bedf786fef14316db6036392e0804e1d8ac7c43f69ffab496385c9aa1c186cb`.
- Recovery receipt: `ab0913e735072d74deacf4bff4f62c5e1b40c696dd951fd34922a20d8176bc17`.
- Corrected package: `sha256:9ee65838f6a10ad4501642f6481b6b5002ce156cc4153ef7b6ec4aa0f7a7879e`.
- Local combined development storage: 51.49 GB before recovery, below 100 GB.
- Focused backend checks: 33 passed, including mismatch rejection, ranking/accepted
  alignment separation, protected-anchor handling, model-signal semantics, artifact
  tampering, serving boundaries, preparation and integration contracts.
- Frontend: 145 unit checks passed; TypeScript and production export passed.
- All 50 traces/scores and 110 description reads passed HTTP comparison against
  the frozen study and recovery receipt. Imports and reference-answer boundaries
  passed. Original context inode, size, mtime and ctime remained unchanged.
- Five browser checks passed through the production-equivalent loopback router
  (20.0 seconds): trace content and reveal/focus, retained bookmarks, mobile layout
  including technical hashes, original citations, graph view persistence, themes,
  comparison and browse. Study/admin HTML and referenced assets remained identical
  to the active study upstream, with unchanged authentication responses.
- The same five browser checks passed on public HTTPS after activation (20.0
  seconds), and all 50 traces plus 110 description reads passed against the live
  demo process. Public package identity matched the tested package. Study PID
  230372 and publication SHA-256
  `25340d36fab81b7415d5c210c74320a892ce343807b9c2176c4b4b05906487da`
  remained unchanged; both services were active with zero automatic restarts.
  Temporary loopback test services were stopped before activation.

Private preparation and acceptance receipts are under
`~/exact-demo/private/traces-20261117`, including `activation-verification.json`,
`live-api-verification.json`, `public-browser-verification.json` and
`public-routing-verification.json`. The trace screenshot was visually inspected.
Docker runtime,
screen-reader use and a concurrent-load benchmark are not claimed by these checks.

## Maintenance

No deployment grant remains active. For later maintenance the owner can renew only
demo activation/rollback/revocation after the five browser
checks and routing/API receipts are present:

```sh
ssh -t AppVM 'sudo /usr/bin/python3 ~/exact-demo/releases/reviewer-traces-20261117/deploy/appvm/install.py --hours 4 --evidence-dir /home/pcotovio/exact-demo/private/traces-20261117'
```

The agent then switches the demo's release link, activates the separate demo service,
checks public traces and unchanged study routes, and revokes the grant. The helper
cannot restart the study. Its rollback restores the original study-only routing
and disables the demo; it does not republish or change participant data.
