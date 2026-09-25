# Explanation frontend integration verification

Date: 2026-09-25. Follow-up to the [specialist handoff](explanation-frontend-handoff.md).
This is bounded implementation verification, not a usability study or experiment.

The frontend was exercised against the retained full NCIT/DOID package and an isolated
PostgreSQL study over actual HTTPS. The prepared package was read without regenerating
its outputs. Study journeys used synthetic cases, practice text, self-reports and test
invitations; no participant was contacted and no provider was called.

## Scope and fixes

- Discovery now has typed API/client contracts. Real exported run counts include a string
  scope alongside numeric counts; their validation no longer prevents the app from opening.
  Failed discovery remains an explicit retriable error instead of appearing to be an empty run.
- Property search and hierarchy navigation preserve entity kind through the URL, reload and
  full-context reads. Stale search responses cannot replace a newer query.
- Selection changes cannot briefly expose the previous entity's details. Candidate lists
  capped at 500 disclose incomplete coverage; their displayed positions refer to the loaded
  subset. Backend keyset order and original matcher ranks/scores are preserved.
- New context preparations carry ontology names under implementation revision `context/3`.
  Existing locks and contexts remain unchanged. Generation revision `/5` uses neutral
  provenance wording and can revalidate saved responses without another provider dispatch.
- Whole-case study resources preserve a shared original axiom under every typed subject.
  Comparison packet subjects remain explicit and scoped. Bounded original labels for
  referenced properties/fillers are policy-filtered separately from the focal case entities.
- Study edits enter a bounded persisted outbox immediately. Coalescing, replay, conflicts,
  invitation changes and pause transitions preserve pending edits. All participant writes
  assert their session identity so a cookie changed by another tab cannot redirect an answer.
- Case timing starts when required content is usable and flushes before progression. Failed
  explanation loading disables ranking until a successful retry. A pending submission is
  distinguished from a server-acknowledged answer.
- Four optional frozen synthetic practice lessons cover simple, complex, partial-ranking
  and none-of-these actions outside the scored cases. Older publications use an explicitly
  synthetic controls tutorial. This does not supply study-owner consent or adjudication.
- Detail tabs support arrow/Home/End focus with explicit activation. Dialog focus returns
  correctly; long labels/headings wrap at 200% text on narrow screens. The legacy viewer
  opens global-only saved alignment files and prefers its own serving origin.
- Researcher screens clear private links and stale progress on sign-out/revision changes.
  Next.js remains on the 15.5 branch with compatible patched dependencies. The obsolete
  `next lint` script is replaced by explicit type checking and test commands.

Verification results and file hashes are recorded in the [receipt](explanation-frontend-e2e.json).

| Check | Result |
|---|---|
| Chromium browser suite, real package plus synthetic PostgreSQL/HTTPS study | 38 passed, 0 skipped; includes full participant/admin journeys, offline reload, cross-tab rejection, failed-resource recovery, typed property navigation and legacy canvas |
| Browser timing export | All three cases contain positive observed segments; pause/resume yields at least four case segments; no rejected timing requests |
| Automated axe audit | 20 route/theme/viewport cases without reported violations or horizontal overflow, plus two keyboard focus/navigation checks; consent-region follow-up recorded separately |
| Focused frontend state/timing/practice tests | 15 passed |
| Backend explanation suite on UTF-8 PostgreSQL 16.15 | 236 passed, 1 intentionally skipped cluster-restart test |
| Legacy backend regression after global-only fix | 12 passed, including both layout versions and local/global mapping files |
| TypeScript, production static build, changed Python static checks, spec validator | Passed |
| Dependency audit | 0 known vulnerabilities reported by npm at verification time |
| Isolated study Dockerfile source set and minimal Python requirements | Publication/resources, migrations/readiness, static pages and profile isolation passed without matcher/parser/model dependencies; this used the explicit SQLite test adapter and is separate from the PostgreSQL/browser evidence |

The axe reports retain items it could not conclusively check, including contrast inside
clipped content. These results do not establish accessibility conformance or replace a
screen-reader audit.

The full package is `sha256:3007654bfaf24a6fd24fe375aea1f69f99533e67c3d7e9820af735b703d1acd6`:
12 saved pairs, 40 prepared outputs and both full filtered ontology indexes. The synthetic
study uses the small NCIT/DOID excerpt fixture with five candidates per case; it is not
presented as an additional production matcher result.

## Reproduce the browser checks

Use a disposable UTF-8 PostgreSQL database and Node.js 24 for the native TypeScript unit
runner. The browser tests use Playwright Chromium. `EXACT_CHROMIUM_PATH` may point to an
already installed compatible Chromium; otherwise install it with `npx playwright install chromium`.

```bash
.venv/bin/python -m tools.build_explanation_ui_fixture --output /tmp/exact-ui-fixture
(cd explanations_visualizer && npm ci && npm run typecheck && npm run test:unit && npm run build)
```

The fixture command prints its portable `package.json`. Serve the full retained package on
18765 and the small fixture on 18768 using `exact-inspect serve --profile local_app`, each
with its own `--library-dir` and `--frontend-dir explanations_visualizer/out`. Serve the
small fixture with `--profile public_demo` on 18767. Export the small ZIP with
`exact-inspect export --package <fixture package.json> --output /tmp/exact-ui-fixture.zip`.
An optional legacy server on 18769 uses `exact-inspect serve --run-dir <saved run>`.

Create a local certificate and start the study with the real origin check:

```bash
mkdir -p /tmp/exact-ui-tls
openssl req -x509 -newkey rsa:2048 -nodes -days 2 \
  -keyout /tmp/exact-ui-tls/key.pem -out /tmp/exact-ui-tls/cert.pem \
  -subj /CN=localhost -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1'

# Create this disposable database with UTF8, including when the cluster uses locale C.
.venv/bin/python -m tools.serve_explanation_ui_e2e \
  --fixture /tmp/exact-ui-fixture \
  --database-url postgresql://localhost/explanation_ui_e2e \
  --tls-cert /tmp/exact-ui-tls/cert.pem --tls-key /tmp/exact-ui-tls/key.pem \
  --config /tmp/exact-ui-tls/study-private.json
```

The harness freezes its publication once and writes generated credentials to a mode-0600
scratch file. It never prints tokens or rewrites origins. Use a database created with, for
example, `createdb --template template0 --encoding UTF8 --lc-collate C --lc-ctype C
explanation_ui_e2e`; psycopg returns byte strings from a SQL_ASCII database.

```bash
cd explanations_visualizer
EXACT_E2E_URL=http://127.0.0.1:18765 \
EXACT_E2E_FIXTURE_URL=http://127.0.0.1:18768 \
EXACT_E2E_BUNDLE_ZIP=/tmp/exact-ui-fixture.zip \
EXACT_E2E_DEMO_URL=http://127.0.0.1:18767 \
EXACT_E2E_LEGACY_URL=http://127.0.0.1:18769 \
EXACT_E2E_STUDY_CONFIG=/tmp/exact-ui-tls/study-private.json \
npm run test:e2e
```

Optional service-dependent tests explicitly skip when their environment variables are
missing. The temporary verification services and PostgreSQL cluster were stopped after the checks.
Study screenshots, videos and traces are disabled to keep invitation/token
material out of retained browser artifacts. Never commit the private configuration or
study database. Stop the local verification servers after use.

## Limits

Docker image execution and Render deployment/recovery were not performed: this node's
Docker socket is inaccessible and passwordless sudo is unavailable. A browser test of a
static export does not prove a deployed image. No manual VoiceOver/NVDA conformance,
formative usability iteration, duration pilot, recruitment or frozen evaluation is claimed.
The backend suite deliberately does not restart the PostgreSQL cluster shared by the live
browser harness; its existing separate recovery evidence is not relabelled as a new run.

The handoff’s intentional API choices remain: source facts provide evidence readings, typed
ASTs provide original-axiom rendering, study asset IDs supply display names, the current-case
endpoint supplies condition, researchers select revisions explicitly, and exploration reviews
are labelled as browser-local. Unsupported graph forms remain available in the evidence list.

The original pinned DOID strict-native parser blocker remains. Operational verification
of the declared derivative does not pass G1/G5 or authorize a live study. Existing release
artifacts and their historical receipts are retained; this report is a separate follow-up.
