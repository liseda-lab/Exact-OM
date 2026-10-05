# Hosted prompt and spending protection

User-approved amendment, 2026-10-05. This supersedes notification-only aggregate spending
for prospective paid workers; it does not rewrite historical accounts or running source.

## Request construction

Every final outgoing prompt, including appended training examples, passes offline admission:
12,288 estimated input tokens, 32,768 UTF-8 input bytes, and 1,024 requested output tokens.
Existing smaller output bounds remain. The selected tokenizer is pinned and locally available;
its count plus a 128-token framing reserve is an estimate, not exact provider billing. Unsupported
input structures fail closed. Both chat and completion endpoints, all generative roles and every
retry are covered. There is no silent truncation, abstention or fallback on admission failure.
Exact saved responses remain reusable without new payment, including historical oversized ones.

E21's compact whole-fact exemplars have a distinct fitting/rendering identity. Keep all selected
source/candidate identities, gold labels and aliases; select complete semantic facts in a fixed
order and count omissions. The suffix is limited to 2,500 tokenizer tokens and 12,000 bytes.
Completed original E21 results are preserved. No compact-arm quality result or prediction
equivalence is inferred from offline reconstruction. No automatic paid rerun is authorized.

## Approved campaign and experiment limits

- Campaign: notify at each new 10,000,000-token interval; pause new paid calls at 200,000,000.
- Each stable experiment family: warn at 10,000,000; pause new paid calls at 25,000,000.
- Count all past usage, retries, teacher/setup requests and conservative unknown-send exposure.
  A recovery ID, new process, cloned request ledger or compact variant does not reset an allowance.
- Pause before transmission, preserve checkpoints, and allow independent/local work to continue.
  A raised allowance requires explicit user approval and a newly bound policy. Automatic repair
  must not clear a spending pause or increase its cap. Per-family exceptions must not raise every
  other family's allowance.

The historical bootstrap is 96,190,067 accounted tokens. E21 accounts for 72,193,688, so no
further paid E21 work is admitted under the default cap. Its completed local final stage is
unaffected. Older unmatched preparation remains included in campaign totals as unattributed;
it is never discarded. Initial historical thresholds are acknowledged by this user-approved
bootstrap, avoiding a burst of retrospective emails. Future warnings include reported provider
cost and identify incomplete cost records.

A single shared admission store is authoritative across copied response ledgers. New attempts
reserve campaign and experiment exposure atomically before transmission. In-flight/unknown
reservations remain counted until valid usage can settle them; a missing store fails closed.
Response-cache identity remains independent of operational limits. Actual provider usage remains
in the durable request ledger; failures and uncertain delivery cannot be reclassified as free.

## Deployment and verification

Prepared launch receipts bind reviewed prompt/spending code, the immutable worker and recipe,
the policy and successor environment. The supervisor rejects stale or unbound paid launches
before Slurm submission. Explicitly reviewed native or cached-only workers may retain frozen
code under a bound no-new-hosted-calls exemption. Keep E14 candidate-first prerequisites,
storage guards and the retained interactive allocation intact.

Evidence is recorded in `data/experiments-v2/hosted-prompt-safety-20261005-01/`:
`recorded-prompt-audit.json` accepts all 1,000 original normal router requests and rejects all
1,000 oversized exemplar requests; `compact-preflight.json` admits all 1,000 reconstructed
compact requests with unchanged base prompts and all source/candidate/gold identities retained.
Maximum compact request: 11,916 estimated input tokens and 32,114 bytes. These checks made no
paid calls. Test receipts, source commits and deployment status belong in the handoff; these
fixtures and offline checks do not establish compact quality or release readiness.

The final combined regression suite passed 409 tests. Separately, operational backports passed
139 E12, 131 E25 and 161 E04 tests. They preserve the workers' frozen scientific implementation;
only prospective admission and family propagation changed. A crash after reserving centrally
but before recording the local attempt can leave conservative exposure that requires explicit
reconciliation; it never silently frees uncertain spending. Warning delivery still depends on
the existing email transport; the durable outbox and deterministic limits are independent.
