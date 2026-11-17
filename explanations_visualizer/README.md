# Exact Explain frontend

The local app has three pages: Compare (`/`), Browse (`/browse/`) and Library
(`/library/`). The Python service serves their static export and the read/import
API from one origin. There is no Node.js server in an installed deployment.

For installation and bundle preparation, see [the local guide](../docs/local-tool.md).

## Development

Use Node.js 24. From this directory:

```console
npm ci
npm run dev
```

The development server proxies `/api` to `http://127.0.0.1:8000`. Set
`EXACT_DEV_BACKEND` to change that address. Start the Python service separately
with a prepared bundle. Run `python scripts/build_frontend.py` from the repository
root once before using the normal Python entrypoint in a source checkout.

## Checks and release build

```console
npm run typecheck
npm run test:unit
npx playwright install chromium
npm run test:e2e
```

Browser tests start and stop their own loopback server with a small offline
ontology/run fixture and a temporary library. Build the frontend first. The
active Python environment needs the package's `viz` and test dependencies; set
`EXACT_UI_PYTHON` to its Python executable if necessary. Set `EXACT_UI_PORT` to
change the default port (19818), or `EXACT_UI_BROWSER=chrome` to use installed
Chrome instead of Playwright Chromium. Tests never contact a hosted study.

From the repository root, `python scripts/build_frontend.py` installs the lockfile,
exports the pages and copies them into the Python package. Use `--skip-install`
only after `npm ci`. Wheel and source-distribution checks require the built assets
and reject study/admin/legacy pages. The source distribution includes the source
and lockfile so users can rebuild the app. System fonts avoid build-time font
requests and work offline.
