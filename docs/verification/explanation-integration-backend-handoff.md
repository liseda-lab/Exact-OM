# Study integration backend handoff — 2026-10-03

> Historical S1 receipt. The [2026-10-04 final backend review](explanation-integration-backend-final-review.md)
> records the current frontend integration, R14/R15 corrections and unresolved R16.
> Its [receipt](explanation-integration-backend-final-receipt.json) supersedes the pending
> frontend status below within its explicitly verified scope; full acceptance remains open.

**S1 backend verified; frontend integration and joint acceptance pending.** This implements
specifications 17/18 from `b5d8b64`. It does not close the UI findings or reuse the earlier
browser receipt as evidence of their correction. The implementation commit is
`96b76fe74afab6d2fdb6437e2ff0b1c28d9b5a3f`; this handoff is recorded in a subsequent documentation commit.

The [machine-readable receipt](explanation-integration-backend-receipt.json) binds commits,
contracts, dependencies, tests and fixture hashes. The [actual HTTP examples](explanation-integration-backend-examples.json)
record the real response projection and explicit injected failure. The active
[backend contract guide](explanation-study-v2-backend-contract.md) gives patch, compatibility
and analysis rules. Earlier dated receipts remain unchanged.

## Acceptance boundary

| Finding | Backend result and evidence | Next owner |
|---|---|---|
| R01 | `backend_verified`: strict current-case scope survives HTTP projection; capabilities agree with the frozen scope; baseline has null/no explanation refs; malformed/missing storage returns 503. Native synthetic navigation exercises non-focal search, multiple parents, 65 children, paged facts and exact original fact/axiom identity. Session, stage, stale scope and cursor denials remain enforced. | Frontend scoped adapter and J01–J03 browser assertions pending. |
| R02 | `backend_verified` for backend support: genuine terminal absence differs from 503/authorization failures; per-page readiness gates timing; duplicate/new-page ready events retain the earliest server start. | Frontend six-entity required-content readiness, delays/retry and J04/J05 pending. |
| R03 | `backend_verified`: typed complete positions, position-only writes, omission preservation, strict null/ID rejection, legacy alias compatibility, historical receipt replay, conflict/atomicity and process/PostgreSQL recovery. | Frontend acknowledged navigation, focus restoration and J06/J07/J10 pending. |
| R04 | `backend_verified`: explicit analysis/CSV 3, every timing oracle, separate tutorial/consultation summaries, unknown coverage, frozen source metadata, old export compatibility and JSON/CSV equality. | Admin explicit selector and browser consumer regression pending. |

J01/J02/J06/J08/J09 pass at the API/database boundary. J03 data and J04/J05 failure inputs
are available. These are not `joint_verified` results: no fresh browser journey, manual
screen-reader test, viewport/reflow check or frontend build is claimed by S1.

## Wire changes and compatibility

Fresh authenticated v2 state advertises `integration_contract: "study-integration/1"`.
Publication protocol stays `exact-study/2.0`; v1 state has no extension field. Frozen
publication hashes, assignment and original receipts remain unchanged. Old receipts may
omit new fields on identical replay; refresh state to obtain the current extension.

The explanation case descriptor is only `workspace: {"scope_id":"…"}`. It is a locator,
not authorization. URL-encode the scope in the existing same-origin route family. Encoded
slash identifiers are supported. Paused case discovery retains its existing permission,
but paused workspace reads remain forbidden. Consultation retains current-scope access;
advancing and opening synthetic help never grant baseline scored access. No filesystem
paths or arbitrary service URLs are returned.

Position is either a real lesson with null question or assessment with null lesson and
optional real question. Omitted position preserves it; explicit null returns 422. The old
lesson alias updates only when non-null and must agree if supplied alongside position.
Action `lesson_id` does not navigate. Historical missing positions are normalized without
writing on read, then persisted by the next valid mutation. Saved old mutation receipts
and request hashes are preserved, including final consultation responses.

Admin create-export accepts `analysis_schema=exact-study-analysis/3`. Omission retains the
frozen default; old v2 default/canonicalization remains 2, v1 only supports 1. Corrected
fixture CLI defaults explicitly to 3. New derivations preserve source `protocol_versions`
and add manifest `source_protocol_versions`; `manifest.schema` names the derivation.
Saved exports are fetched unchanged and CSV dispatches their saved schema.

Analysis 3 sums eligible intervals as **page-seconds**, independently within page/stage/case.
For one page, several pages or no observations, unique elapsed coverage and unobserved
elapsed seconds remain null. Reasons distinguish those three cases. Two 60-second pages
over a 100-second server interval export 120 page-seconds, with no subtraction or clamping.
Aliases retain the same raw sum. Tutorial and consultation summaries remain separate;
unavailable intervals keep their reasons and contribute nothing. Historical analysis-2
coverage arithmetic must not be used as corrected elapsed-coverage evidence. Dictionaries
in JSON and CSV explain units, eligibility, aliases, clock scope and null cells.

## Reproduce the synthetic package and examples

Run from the repository root with its study and native preparation dependencies installed.
The verified interpreter was Python 3.12.14 with FastAPI 0.116.2, Pydantic 2.13.5,
httpx 0.28.1, pytest 9.1.1, psycopg 3.3.6, pyowl-core 0.2.1 and uvicorn 0.35.0.

```sh
python -m tools.build_explanation_v2_fixture /tmp/exact-study-integration-new --navigation-size 65
python -m tools.record_explanation_integration_examples \
  --fixture /tmp/exact-study-integration-new \
  --output /tmp/exact-study-integration-new-examples.json
python -m tools.freeze_explanation_contracts --check
python specs/explanation-framework/protocol/validate_specs.py
```

The builder requires an empty destination and never overwrites a publication. The helper
retains `export_version="exact-study-analysis/2"` for old fixture tests; CLI callers can
explicitly select `--export-version exact-study-analysis/2` for compatibility testing.
`--navigation-size 65` adds searchable non-focal classes, multiple inheritance and enough
context/children to require continuation. Full filtered indexes contain these data even
though the prepared prose packet is bounded. No provider or matching job is dispatched.

The recorded package is retained locally at `/tmp/exact-study-integration-20261003`.
Its 59-file [inventory](explanation-integration-fixture-inventory.json) has canonical SHA-256
`54889b496b889efa1288360efe2101c1bf7f66ba3993965f55c002482163d096`; the publication file
hash is `500963709434c1c9ee45ce3d64153691d20b19be7406f5a39a4b4fb0c5996704` and admitted
publication hash is `27ef646d370c22e0fb92db6e5ff020589f93d1b647a0c2206a976afe2ee1f243`.
Rebuilding creates new synthetic revision IDs/timestamps and therefore new hashes. Verify
the newly built package rather than treating a regenerated publication as identical bytes.

The recorder uses TestClient through the actual router. Backend test helpers complete
setup/training; that is explicitly not the UI acceptance journey. It stores no request
cookies, authorization headers, invitation URLs, private researcher keys or human records.
It captures 12 examples: lesson/focused assessment acknowledgements, omitted and null
position behavior, explanation/baseline cases, capabilities, cross-query 409, stale-scope
403, terminal absence, injected missing-scope 503 and a complete analysis-3 export.
Receipt timestamps for the 100/120 timing example are deliberately controlled synthetic
inputs. Success cases use the real service response without replacement payloads.

For an HTTPS/PostgreSQL frontend development harness, prepare a dedicated synthetic
database and a local TLS certificate/key, then run the existing launcher:

```sh
python -m tools.serve_explanation_v2_e2e \
  --fixture /tmp/exact-study-integration-new \
  --database-url postgresql://127.0.0.1:18973/exact_integration \
  --tls-cert /tmp/exact-integration-local/cert.pem \
  --tls-key /tmp/exact-integration-local/key.pem \
  --config /tmp/exact-integration-local/private-config.json \
  --port 18974 --frontend-dir explanations_visualizer/out
```

The launcher writes fresh harness credentials to a mode-0600 local config. Keep it outside
Git. Start a dedicated PostgreSQL service before launching; this verification's synthetic
cluster was stopped after the checks, with its data and recovery evidence preserved.
Build the frontend in S2 before using this combined harness for S3. Fault tests can
delay/fail actual scoped endpoints; keep the captured explicit absence separate from the
503 error fixture. S1 exercised API responses, not rendered-content readiness.

## Verification and review

`tests/explanation_integration_contract_test.py`, `…_position_test.py` and `…_timing_test.py`
contain the new executable regressions. Three initial regressions failed at the starting
source: absent workspace descriptor, rejected explicit position, and omitted-position
reset. Timing tests additionally preserve the old analysis-2 overcount result beside the
corrected analysis-3 unknown-coverage result, so version differences remain observable.

The final SQLite run passed 153 tests with two PostgreSQL-only skips; the PostgreSQL run
passed all 64 tests with no skips, including those recovery checks. The receipt lists exact
suite arguments/results. SQLite includes legacy/v2 study,
workspace policy, telemetry, export policy, actual HTTP runtime and copied-source image
imports. PostgreSQL 16.2 covers concurrency, process crash before/after commit, actual
server restart, and `pg_dump`/`pg_restore` into separate databases. V2 restore preserves
focused assessment position, completed tutorial, consultation draft, saved analysis-2/3
JSON and byte-identical analysis-3 CSV. The retained recovery dump is private and ignored.

Independent review checked position/hash compatibility, scoped authorization and export
semantics. Findings fixed before delivery include final-consultation receipt projection,
encoded slash routing, frozen-resource failure classification, and saved CSV row ordering.
Regression runs also corrected the crash fixture's omitted-field serialization and the
revision-inventory test's pagination assumption. The final receipt distinguishes these
resolved failures from passing final checks. One upstream Starlette/AnyIO deprecation
warning remains; it is not an application/test failure.

## Remaining gates and next owner

The frontend specialist should consume this exact commit and contract hashes, implement
specification 19, produce S2's handoff, then run S3's fresh complete browser journey and
joint matrix. In particular, API-completed training cannot replace that journey.

Real-release integration remains unverified: the designated `backend-release/package/package.json`
is absent locally. The Docker daemon is unavailable, so copied-source import tests do not
establish a built-image or deployment result. Render/redeploy/restore, historical ontology
admission, owner consent/information approval, real case adjudication, accessibility/usability
and duration-pilot gates remain open. No deployment, push, recruitment or participant contact
was performed or authorized by this handoff.
