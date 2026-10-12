# Brief comparison of generated ontology descriptions — 12 October 2026

GPT-6 Luna performed substantially better on this deliberately difficult sample: **9/12
complete responses were usable without correction, versus 1/12 for GPT-4o mini**. This supports
trying Luna for the next reviewed generation batch. It does not justify automatic publication:
Luna still failed three cases, including one unsupported medical comparison.

The frozen study publication, service configuration, matcher evidence, existing sessions and
approved descriptions were not changed. These outputs are diagnostic drafts only.

| Observed result | GPT-4o mini | GPT-6 Luna |
|---|---:|---:|
| Complete usable responses | 1/12 | 9/12 |
| Without an original definition | 0/4 | 3/4 |
| With an original definition | 1/4 | 3/4 |
| Pair comparisons | 0/4 | 3/4 |
| Claims meeting support, citation and usefulness criteria | 1/21 | 13/16 |
| Claims semantically supported somewhere in the complete packet | 11/21 | 15/16 |
| Claims fully supported by their actual citations | 3/21 | 15/16 |
| Valid JSON/schema and citation-ID membership | 12/12 | 12/12 |
| Provider-reported cost for all 12 calls | $0.00997995 | $0.010148725 |
| Median measured request time | 2.46 s | 4.90 s |
| Input / output tokens | 57,009 / 2,381 | 56,901 / 6,074 |

Luna's output-token count includes 3,004 reasoning tokens. Total reported cost was **$0.020128675**.
All 24 actual provider calls returned HTTP 200, with zero repairs, retries or provider fallbacks.
Neither model abstained. These measurements describe this run, including its provider routing
and cache accounting; they are not general latency or price benchmarks.

## Method and review boundary

The plan was frozen before dispatch. From attempt 7's 22 artifacts with no independently
approved claims, selection took the first four profiles without definitions and first four
with definitions in original-ID order, plus all four rejected comparisons. This is a
purposive diagnostic sample, not a representative corpus evaluation. There was one response
per model per case, with no output-based selection or retries.

Exact requested models were `openai/gpt-4o-mini-2024-07-18` and `openai/gpt-6-luna`, routed
through OpenRouter with OpenAI as the only allowed provider. Returned identities were checked.
Mini used temperature 0; Luna used medium reasoning and its supported default sampling.
Both received identical prompts, JSON schema and corrected admitted facts, with a 4,000-token
output ceiling, at most two claims and 160 words. This compares those configurations, not
models under identical internal computation.

The experiment reused each selected corrected publication packet and its frozen profile
dependencies. Original focal facts and typed identities were checked against their predecessor.
The diagnostic wire projection explicitly includes verified `referenced_labels`; the historical
provider projection omitted that additional field. Both models therefore received the same
readable typed parent/property labels. V5 prose instructions were retained and supplemented
with the same explicit rules for necessary conditions, malformed definitions, qualifiers and
scope. No scores, ranks, reference answers or answer keys were found in the transmitted data.
Improvement over the old batch cannot be attributed solely to a model change because its inputs
and instructions differed; the paired comparison here controls those differences.

Per-case A/B assignment and call order were randomized. The root Codex agent reviewed all 37
claims against the exact cited facts and complete packets, then froze the review before reading
model assignments, usage or latency. This reviewer was separate from both API generators but
also designed the experiment. **It is one blinded AI review, not external independent acceptance,
medical validation or a human audit.** No generator approved its own content.

The primary outcome requires a nonempty response with every claim grounded, correctly cited and
useful. Alias repetition and copying a malformed definition do not qualify as useful prose.
Correct citation IDs alone do not prove semantic support. Mini's failures were not all invented
medicine: 13 claims omitted required readable-name citations, sometimes despite otherwise
faithful content. The table separates semantic support from citation completeness to make that
distinction visible. No statistical significance or corpus-wide failure rate is claimed.

## What still failed

Luna's three unsuccessful complete responses were:

- **Jejunal cancer:** repeated the recorded malignant-tumor synonym without using the supplied
  hierarchy to add a useful independent description.
- **Hyperostosis:** copied the malformed source wording “abnormal growth of located in bone.”
  It avoided inventing the missing object but did not produce readable prose or choose the
  valid hierarchy-based alternative.
- **Blau syndrome comparison:** claimed both descriptions involved granulomatous arthritis.
  The source says granulomatous dermatitis and separately arthritis with synovitis; only the
  target says granulomatous arthritis. Luna's second claim correctly preserved the source's
  usual onset qualifier, but the first claim prevents accepting the complete response.

Luna handled readable parent classification, the Nestor–Guillermo material-basis relation,
the congenital hypoventilation genetic-frequency qualifiers, and the neonatal progeroid
association/characterization distinction better in this sample. Mini still introduced retinal
inflammation from names, supplied the missing object in the malformed hyperostosis definition,
and strengthened a genetic material-basis relation into causation.

Keep semantic review, exact citations and conservative fallbacks with either model. Fixing only
the citation formatting would not remove the observed unsupported claims. Luna is the better
candidate for a subsequent reviewed batch; this study does not authorize replacing the existing
publication or treating all newly generated content as accepted.

## Evidence, checks and reproduction

Started from commit `0f38a5f4e8c969b3520c1f0ab59d24cbdf2552ad`. The implementation and report
are committed separately from ongoing frontend and specification edits; use the commit
containing this report as the delivery revision. No push or deployment was performed.

The [frozen plan](explanation-model-comparison-20261012/plan.json) contains all exact input
packets, prompts, schema bindings, model catalogue entries, selected identities and limitations.
The [blind review](explanation-model-comparison-20261012/blind-review.json) records each claim,
its citation IDs and the specific acceptance/rejection reasons. The
[machine-readable receipt](explanation-model-comparison-20261012/receipt.json) includes per-call
request/response hashes, provider IDs, costs, latency, scoring and verification. The saved
[assignment](explanation-model-comparison-20261012/assignment.json) permits auditing the totals.

- Plan identity: `sha256:5a979093ad1b1b28face3cf30725d0e6cbf5e11a43c92f48ee0ca64fa4b98d19`.
- Review identity: `sha256:3d05ba98f02f9921d7aa7920af20fdac2ddbd0d0b349745f2119224c6248ec6f`.
- Preserved publication: `5eff10207af3ccecbd541840088b022f074df962fd35d3630854c8aa1e743b2a`.
- Preserved package file: `sha256:1e160105df9ab9fcccbc449716066904e03eb93711c57189e896cc728d9a5e70`.

Raw responses and the durable provider ledger remain under
`/Users/pedrocotovio/Library/Caches/exact-om-review/luna-comparison-20261012`. Only small
diagnostic artifacts were created; no ontology, dataset bundle, runtime or database was copied.
The runner is `tools/review_model_comparison.py`, using the existing provider client and ledger.
It enforces 24 calls, finite request sizes, a $0.50 study ceiling and no implicit retries.
The frozen plan's ordinary-token estimate was $0.09870815; the final audit included the higher
cache-write price and bounded the planned batch at $0.103302925, still below the ceiling.
Actual costs above are taken from provider receipts, not estimated from the plan.

Verification completed:

- **45 tests passed**, including the diagnostic's negative boundary tests and the existing
  transport/ledger suites. One upstream deprecation warning; see
  [test results](explanation-model-comparison-20261012/regressions.xml).
- **24 exact responses replayed from the real durable ledger**, with networking and key
  resolution replaced by fail-fast sentinels. An altered uncached request was rejected.
  The ledger retained exactly the same 24 attempts before and after replay.
- Frozen package and every selected source-artifact hash remained unchanged after generation
  and after the final verification. Request payloads were reconciled with the ledger.

On this host, use the retained Python at
`/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python`. From the migrated checkout,
repeat the three test files named in the receipt under `tools.review_storage_budget run`.
An existing-run resume is available through the runner's `run` phase with the root above;
set `EXACT_HOSTED_CACHE_ONLY=1` to prohibit new provider calls. A fresh paid study requires a new
output root and a newly frozen plan; do not overwrite this plan or its saved reviews.

The [storage record](explanation-model-comparison-20261012/storage.json) preserves guard
measurements. Combined usage was 49.250 GB before this work and 49.294 GB after evidence export,
below the 100 GB combined limit; the peak additional reservation was 0.2 GB. These combined
measurements include concurrent unrelated work. The local diagnostic itself was about 1.27 MB
before its result receipt, and the repository evidence about 0.62 MB. Shared protected inodes
were never modified. No frontend S2 implementation, browser acceptance, deployment or launch
gate was exercised or closed by this diagnostic.
