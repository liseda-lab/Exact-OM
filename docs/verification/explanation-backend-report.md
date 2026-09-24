# Explanation backend verification report

Backend implementation B0–B5 is complete and committed through
`f4b411760e50da1a58587362cec64cc6f9cec608`. The [status ledger](explanation-framework-status.json)
and [operational receipt](explanation-backend-operational.json) bind the actual
commands, inputs, outputs and measurements. The public
[handoff](../../data/explanation-framework/backend-release/handoff/backend-handoff.json) is published, with contracts, API responses, real
study resources, ontology downloads, deployment instructions and the frontend brief.
**Acceptance remains blocked by the original DOID input; F1 is not admitted.**

| Gate | Disposition |
|---|---|
| G0 | passed |
| G1 | blocked_input |
| G2 | passed |
| G3 | passed |
| G4 | passed |
| G5 | blocked_input |

The unmodified pinned DOID (`611355c445537fcf4bae2c519f1b3598af5a8fea793274316e35525b7d05e945`)
fails strict native `pyowl-core==0.2.1` loading. The installed version was also the
latest published version at verification. A declared derivative adds two
annotation-property declarations, deletes no source bytes and passes its own
checks. The portable manifest and ontology-list API preserve that distinction.
Its unresolved import is reported explicitly; full import closure and successful
parsing of the original are not claimed. Resolving G1 is required before G5/F1 admission.

The implemented backend includes independently indexed ontology context, original
axiom provenance, physically filtered portable bundles, lazy bounded APIs,
local import and fixed-demo isolation, grounded generation and factual fallback,
resumable preparation and provider ledgers, and the transactional PostgreSQL study
lifecycle. The frontend and full experiment campaigns were excluded from this work.

Full NCIT contains 3,506,377 axioms and 212,055 entities; the declared DOID root
contains 177,912 axioms and 16,395 entities. NCIT preparation peaked at 11.1 GiB
on the declared 128 GiB node. Its last build resumed at 430,000 axioms; retained
receipts distinguish that attempt from an uninterrupted build. Exact definitions,
alternate definitions, restrictions, multiple parents and sparse cases were checked.
Both exported Functional Syntax resources passed native structural parity.

The saved current-schema 12-pair matcher run retains its scores, candidate/evidence
order and final mapping: 1,974 shared numerical values agree with the baseline.
The final imported package has independent provenance and API verification.
The [development coverage receipt](explanation-development-coverage.json) records
12 saved pairs plus three explicit unscored alternatives. Generation produced
25 entity profiles and 15 comparisons: 32 validated generations and 8
factual fallbacks. All 40 active outputs passed the separate grounding audit.
The retained provider ledger records 79 completed wire
attempts costing $0.06680205, including earlier attempts;
52 have measured HTTP timing and 27 historical attempts remain explicitly unmeasured.

Fresh-process loopback HTTP on local SSD, with four readers and 100 fixed mixed
queries, measured health readiness at 9.469 s, first entity
at most 0.067 s, context p95 0.106 s and pair p95
0.058 s. Peak serving memory was 58.7
MiB. All original resource limits passed, including the 2 MiB response budget.
The serving environment excludes the matcher, OWL parser, model and provider
packages. OS page caches were not flushed; these are cold-process measurements,
not cold-storage measurements or proof on a physical 64 GiB node. The full DOID
package additionally passed archive/import/select/restart/demo response parity;
the combined package was independently persisted and hash-verified. The minimal
Python deployment was exercised; a Docker image build and public deployment were not run.

The [regression receipt](explanation-backend-regression.json) records 1,508 passing
hermetic tests and one unchanged interruption test that timed out under filesystem
contention, then passed its isolated rerun. Final metadata, provenance, handoff,
provider-timing and recovery changes have focused passing checks. Type checks,
architecture contracts, specification validation and wheel/source builds passed.
[42 real PostgreSQL tests](explanation-study-postgres.json) passed with zero skips,
including server restart and backup/restore into a separate database with identical
immutable exports. These use synthetic participants and do not constitute a live study.

From the repository root, the durable package can be served without generation:

```sh
.venv/bin/python -m exact_inspect.cli serve \
  --package data/explanation-framework/backend-release/package/package.json \
  --profile local_app
```

The measured performance target assumes local SSD. On a network-mounted checkout,
copy `data/explanation-framework/backend-release/package` to local storage and use
that copy for serving. The sibling `backend-release/handoff` directory holds the
published public assets; its package locator is `../package/package.json`.
The [backend guide](../guides/explanation-backend.md),
[context guide](../guides/ontology-context-preparation.md) and
[study guide](../guides/explanation-study-service.md) provide API, build, import,
publication, invitation, migration and recovery commands. The study guide describes
required environment settings; no credentials or private adjudication are in the handoff.

The completed generation ledger and verified preparation stages are durable under
`data/explanation-framework/prepared-combined`. An unchanged preparation resumes with:

```sh
.venv/bin/python -m exact_inspect.cli prepare \
  --lock data/explanation-framework/prepared-combined/execution-47388b7e7a83f4afd4d7a4d649d7faf769a7f84271242211da531f5b604075f1.json \
  --output-root data/explanation-framework/prepared-combined --resume
```

For a deliberate repair, bind a revised lock and use a new output directory with
`--resume-from data/explanation-framework/prepared-combined`; the guide explains
previewing selective reuse. Keep the frozen provider ledger to avoid duplicate requests.
The 48-pair model-selection exercise, frontend implementation/usability checks,
public Render and staging rehearsal, recruitment and full campaigns remain deferred.
The handoff ID is `sha256:5857ed6002526582f2cc069c95bad51a5aa474372ff6360e7d49c485027ce4b9`; its blocked G1/G5 dispositions are intentional.
