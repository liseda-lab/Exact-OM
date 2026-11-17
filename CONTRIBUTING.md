# Contributing

Install development tools with `pip install -e '.[dev,docs,reasoning]'`. Keep experiment
orchestration, private data, generated results and unfinished products outside
the production package. Do not put credentials or machine-local paths in
profiles or documentation.

The full test suite also exercises the optional reasoner adapters, so the
contributor installation includes the `reasoning` extra.

Run the offline production checks:

```console
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q \
  -m 'not requires_data and not slow and not requires_llm and not requires_cuda and not requires_openrouter'
ruff check exact scripts
mkdocs build --strict
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist
```

Install the wheel in another virtual environment and run
`python scripts/smoke_installed.py --help` for the installed-command smoke suite.
Run it outside the checkout and verify imports come from site-packages.
Repeat for the sdist. The smoke suite uses local fixture services and cached
decision answers; it does not make model/provider calls.

Changes to numerical scoring, feature order, precision, membership, anchors,
tie rules or acceptance require behavioral parity checks and an explicit
version/identity change where applicable. Source/target holdout and unknown
references must remain distinct from confirmed negative labels.

GitHub Actions templates are provided in `scripts/workflows/`. They are inactive
until a maintainer with workflow-write permission copies them into
`.github/workflows/`. The CI template covers Python 3.10–3.12; the docs template
builds the documentation; the release template builds and archives distributions.

A release consists of tested wheel/sdist artifacts. The release template does not
upload them to PyPI. After the release checks pass, a maintainer can publish using
`python -m twine upload dist/*` with their authenticated publishing setup.
This repository does not embed upload tokens.
