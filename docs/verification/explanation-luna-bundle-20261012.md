# Luna prose revision — prepared batch and publication handoff

> Historical preparation record. The subsequent full-batch authorization was granted, and
> generation and review are complete. See the [Luna publication handoff](explanation-luna-publication-20261012.md)
> for the delivered revision, actual cost and verification outcomes. Statements below about
> a pending approval describe the earlier preparation checkpoint.

12 October 2026. The complete 260-request plan is frozen and the local preparation/publication
path is implemented. **The new provider batch has not run, no Luna revision has been published,
and the current study is unchanged.** Automatic approval review rejected dispatch because it
recognized the prior external-transfer authorization as limited to 74 entity profiles and
12 comparisons. A specific authorization request for 140 profiles and 120 comparisons is
pending. This is a preparation handoff, not a claim of generated-content acceptance.

Starting checkout: `90f929a8b48b6ba51caa5b74a91a540db19b9f55`. The implementation commit is the
commit adding this report; retrieve it with `git log -1 --format=%H --
docs/verification/explanation-luna-bundle-20261012.md`. Unrelated frontend, specification,
review, storage and cluster work is excluded. Nothing was pushed or deployed.

## Storage and frozen inputs

The full dataset is reused directly from:

```text
/Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011/combined
```

The plan is stored at:

```text
/Users/pedrocotovio/Library/Caches/exact-om-review/luna-bundle-20261012/plan.json
```

Plan identity: `sha256:7f58bd7c31223de889baf0cb5f53ddc27c8c91dc868fd4049024122f1c95c155`.
The plan is 9,709,847 bytes and freezes exact requests, source artifacts, ontology identities,
original facts, readable reference labels, profile dependencies, model catalogue/prices,
runner hash and the original files' inode/protection/timestamp identities.

The preserved publication hash is
`sha256:5eff10207af3ccecbd541840088b022f074df962fd35d3630854c8aa1e743b2a`;
the preserved package file hash is
`sha256:1e160105df9ab9fcccbc449716066904e03eb93711c57189e896cc728d9a5e70`.
Its package ID remains
`sha256:b4027297f62b41066cca3e4c529c0a28a9984c945df444491ca9a94790a9a7fb`.
Original publication/composition hashes and all indexed artifacts' file identities were checked
again after preparation. No existing protected hardlink was changed, unlocked or relinked.

Measured combined usage was 49,384,672,494 bytes before this work and 49,748,487,197 bytes at the
later check, below the 100 GB limit. These include concurrent frontend work and are not a
measurement of this task's growth alone. The batch's conservative additional peak reservation
is **1 GB**, including requests/responses, ledger, reviews, revised resources and small indexes.
The plan itself currently accounts for about 10 MB. Actual completed-batch growth is unmeasured
because generation has not run. Guard records and test details are linked in the
[machine-readable receipt](explanation-luna-bundle-20261012/receipt.json).

No complete physical bundle, ontology database, matcher database, runtime, or ontology export
will be copied by this path. The intended new manifest is
`combined/package/package-luna-20261012.json`: its parent remains the existing package root,
so all large immutable artifact locators stay valid. Small new files use distinct paths.
Historical manifests remain self-contained valid manifests; the standard archive exporter can
export the new manifest under the standard `package.json` name. Only a tiny fixture export was
created to verify this property; no full real-data archive was allocated.

## Generation and admission contracts

The exact requested model is `openai/gpt-6-luna`, medium reasoning, through OpenRouter with
OpenAI as the only provider, no provider fallback, 4,000 output tokens, at most two claims and
a 128 KiB request bound. The frozen ordinary/cache-price estimate is **$1.052581125** for
260 calls, with a separate $5 absolute configuration ceiling. This is an estimate, not a
charge. Zero generation requests have been sent in this batch. Requests are score-blind:
admitted NCIT/DOID facts, original citations and frozen profile descriptions are included;
participant answers, matcher scores, ranks and answer keys are excluded.

`tools/review_luna_bundle.py` provides explicit prepare/generate phases. A durable provider
ledger and saved responses support exact resume; repairs, retries and unknown-delivery retries
are disabled. The runner must match the frozen plan. Four bounded workers do not multiply the
request cap. Provider outputs remain unapproved drafts until a separate claim-by-claim review.
Comparisons intentionally use the predecessor's already frozen profile dependencies; newly
unreviewed descriptions cannot silently become their inputs.

The current V5 prompt and diagnostic input corrections are retained. Instructions explicitly
preserve causal/association/material-basis distinctions and uncertainty, reject synonym-only
restatement, prohibit moving modifiers between anatomical features, and allow abstention when
a definition is malformed. These instructions reduce known risks; they do not prove fidelity.
The earlier [12-case comparison](explanation-model-comparison-20261012.md) remains diagnostic
evidence, not acceptance of these 260 future results.

Independent review now accepts validated original labels of exact typed references in the
support packet. Previously those IDs could be admitted as ontology facts but were excluded
from the review's citation allowlist. The change uses the existing bounded `typed-readable/3`
label contract. Main-service and scoped-workspace admission still reread originals, and frozen
study resources now compare these packet labels to their actual admitted values as well.
No new wire fields or runtime schema shapes were added; historical packets without this
renderer revision keep their prior behavior.

`tools/publish_luna_review.py` requires complete separately recorded review decisions for every
reviewable draft. Approved claims receive new content-bound artifact identities. All-rejected,
abstained or invalid outputs retain the predecessor's approved prose or typed fallback. The
new manifest retains old description dependencies and all ontology, matcher, trace, inference
and original-fact artifacts. Case resources and scope indexes preserve exact old evidence and
pair metadata. Public provenance gains a separate Luna layer, while historical generation
counts and the combined evidence identity remain recorded separately.

Interrupted case-index installation can resume only when the retained small index is identical
to a freshly assembled private index; a mismatch is refused and retained for inspection.
Completed case checkpoints and final publication receipts are immutable and content-bound.
New study revisions do not rewrite the existing publication or participant sessions.

## Verification and limitations

**136 distinct tests passed.** The three overlapping local test runs recorded 34,
86, and 40 passed respectively. The last initially skipped a PostgreSQL process check;
that check was then run against the retained native PostgreSQL runtime and passed separately.
The separate instance used port 19583, exercised restart preservation plus authenticated
closure/republication, and stopped its own processes. Existing study databases were untouched.
The XML receipts preserve the initial skip and its successful explicit follow-up.

The full publisher fixture caught and corrected assignment to a frozen manifest and a
pre-existing access-time false alarm in `AdmissionHashes`. The hasher now ignores changes
caused solely by reading while still binding device/inode, size, modification/change times,
permissions, link count and protection flags. A negative regression changes actual content
during hashing and still requires rejection. Early fixture setup failures were corrected to
use the current v2 review contract and accurately declare its profile-only case resources.
The real publisher retains every predecessor case's full component declaration.

The checks cover exact typed label citations at the main API, frozen resources and workspace
admission; wrong typed identities and modified originals; independent-review requirements;
immutable predecessor files; alternate-manifest export/relocation; approved and all-rejected
publication paths; actual study publication HTTP admission/idempotent replay; and interrupted
small-index resume. Synthetic test review receipts do not approve real model-generated prose.

The real-data plan was prepared against the existing validated package. Final real-data output
parity cannot be checked until the batch and semantic reviews finish. The new Luna revision
has not been exercised against the retained PostgreSQL service. No compiled-frontend browser journey, accessibility/pilot review, deployment, ontology admission,
study-owner approval or launch gate was exercised or closed. Existing partial-ELK limitations,
missing review-data coverage and DOID import-scope limitations remain exactly as documented in
the [backend corrections handoff](explanation-postmigration-backend-corrections.md).
No reasoner or matcher experiment was restarted.

## Exact continuation

Use `/Users/pedrocotovio/Developer/Repos/OM/Exact-OM` and the retained Python
`/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python`. Do not use the former Documents
checkout. No installation is needed.

1. Obtain the pending explicit authorization for the complete 260-request transfer. Do not
   treat the earlier 86-request authorization or this preparation report as that approval.
2. Under `tools.review_storage_budget run --growth-gb 1 --`, run that Python with
   `-m tools.review_luna_bundle generate --root
   /Users/pedrocotovio/Library/Caches/exact-om-review/luna-bundle-20261012 --key-path
   /Users/pedrocotovio/Library/Caches/exact-om-review/s1-20261010/credentials/openrouter-api-key`.
   The credential must stay local and must not appear in logs or committed files. Reuse the
   exact frozen runner and root; do not prepare a replacement plan over it.
3. Review every generated claim against its exact cited originals and the full packet. Save
   complete `exact-description-review/1` receipts under `reviews/<job-digest>.json`, using
   the corresponding `review-requests` bindings and ordered inventory. Separate the reviewer
   from the API generator; record limitations of an AI review. Do not infer approval merely
   from a valid schema, valid citation IDs, or the earlier diagnostic's pass rate.
4. Under the same storage guard, run `-m tools.publish_luna_review --root` with the same root.
   Intended outputs, which **do not yet exist as a delivered revision**, are the new manifest
   above, `combined/publication-luna-20261012.json`, `combined/catalogue-luna-20261012.json`,
   `combined/luna-20261012/resources/`, and new small indexes under `combined/workspaces/`.
   The assembly receipt is `luna-bundle-20261012/publication-receipt.json`.
5. Independently reconcile all 260 selected artifacts against source/review receipts, and
   compare main API, frozen resources and scoped workspaces on actual data. Verify matcher
   database identity, all retained fact links, partial inference and historical sessions.
   Record provider usage/cost, actual storage growth, fallbacks and rejected claims honestly.
6. Prepare a separate review environment referencing these new files and the existing compiled
   frontend/runtime. Preserve the current frontend review environment and database. Update
   the environment's backend identity after the code commit; do not change its files in place.
   Run the affected integration checks before handing the new revision to frontend S2.

Frontend S2 requires no new JSON field to render this revision. It must use each selected
artifact's new ID, existing presentation/support status and exact citation IDs; reference-label
citations must remain resolvable. Clear any artifact cache when switching publication identity.
Do not retarget existing sessions to the new revision. No frontend implementation is included.
