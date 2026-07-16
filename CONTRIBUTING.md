# Contributing to Exact-OM

Thank you for improving Exact-OM. Changes should preserve the public contracts in `specs/`
and keep the default class-only OWL pipeline behavior stable unless a proposal explicitly
changes it.

## Set up a development environment

```console
git clone https://github.com/liseda-lab/Exact-OM.git
cd Exact-OM
poetry install --with prebuild --extras "docs viz"
poetry run exact --help
```

Use Python 3.10 or 3.12. CPU-only development is supported; mark tests that genuinely need an
accelerator, a hosted service, or downloaded benchmark data instead of making the hermetic
suite depend on them.

## Repository boundaries

The dependency direction is intentional:

```text
delivery -> core contracts/actions -> implementation
analysis -> public library seams
ontology and io -> core KnowledgeSource contracts
exact_inspect -> exact (never the reverse)
```

`poetry run lint-imports` enforces the concrete rules in `.importlinter`: core cannot depend
on implementation or delivery layers, ontology/I/O cannot depend on delivery, and optional
parser dependencies stay isolated in their backend modules. Add a small protocol or registry
seam instead of importing across a forbidden boundary.

Run-artifact paths belong in `exact.runs.RunLayout`; consumers use `RunReader`. Code outside
trainer checkpointing and `exact.runs` must not write checkpoint files.

## Registries and extension points

Built-in components use `ComponentRegistry` and keep stable registry names. Import-time
bootstrap is centralized; avoid registration side effects in unrelated package imports.

To add an extension:

- **Model, trainer, or dataset:** implement the matching core contract, register the stable
  component name, and add a bootstrap/import test.
- **Evaluator:** implement the evaluator protocol and register the backend without importing
  it from core.
- **Knowledge source:** implement `KnowledgeSource`, add a factory to the built-in source
  registry, or publish an `exact.sources` entry point. Run the conformance tests.
- **Alignment writer:** implement `AlignmentWriter`, register it in `exact.io.writers`, or
  publish an `exact.writers` entry point. Validate filenames and deterministic ordering.
- **Reasoner:** publish an `exact.reasoners` entry point whose factory returns the reasoner
  protocol. Pipeline code must continue to depend only on `KnowledgeSource`.
- **Dataset track:** prefer a declarative descriptor. Third-party providers use the
  `exact.tracks` entry-point group and must pin revisions and verify checksums.

Optional integrations must be lazy and produce an actionable `pip install "exact-om[extra]"`
message when their dependency is absent.

## Tests and quality gates

Place deterministic tests under `tests/`; fixture data belongs under `tests/fixtures/` and
should be small enough to review. The standard local gates are:

```console
poetry run black --check exact exact_inspect tests tools study_visualizer_runtime
poetry run isort --check-only exact exact_inspect tests tools study_visualizer_runtime
poetry run flake8 exact exact_inspect tests tools study_visualizer_runtime
poetry run mypy
poetry run lint-imports
poetry run pytest \
  -m "not requires_data and not slow and not requires_cuda and not requires_openrouter"
poetry run mkdocs build --strict
poetry run python benchmarks/bench.py --repeat 7 --check-reference
```

The repository may retain Black/isort/Flake8 compatibility gates during the 2.0 transition;
run the commands configured in `.github/workflows/ci.yml` before opening a pull request.

Use markers deliberately:

| Marker | Use |
| --- | --- |
| `requires_data` | Needs a materialized external dataset. |
| `requires_cuda` | Cannot run correctly on CPU. |
| `requires_openrouter` | Makes a hosted OpenRouter request. |
| `slow` | Unsuitable for the normal hermetic PR suite. |

Every behavior change needs a focused regression. Performance-sensitive work should add a
benchmark and a reference budget. Preserve byte-level golden outputs when a work package says
the default behavior is unchanged.

## Configuration and documentation

Configuration defaults and descriptions live on the Pydantic models. Do not hand-edit a
second default in consumer code. Regenerate `exact/default_config.yaml` with
`exact config default`, and confirm the generated configuration reference builds.

Public functions and protocols use Google-style docstrings. The priority public packages are
checked by `interrogate`; examples should be executable and avoid machine-specific paths.
Published pages must not contain `TODO` or `FIXME` placeholders.

## Commits and pull requests

- Keep commits scoped to one component and use an imperative subject.
- Do not mix formatting-only churn with unrelated behavior when it can be avoided.
- Update compatibility notes for renamed public symbols or artifact formats.
- Record deliberate deviations in the applicable work-package spec.
- Never commit credentials, hosted-model keys, downloaded benchmark corpora, caches, run
  artifacts, or generated frontend builds.

The work-package suite in `specs/README.md` is retained as design history. Read the shared
contracts before changing a public seam, and link the relevant package in the pull request.
