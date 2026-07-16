# Contributing

Changes must preserve the shared contracts and enforced dependency direction. In particular,
artifact paths belong in `RunLayout`, consumers use `RunReader`, optional integrations stay
lazy, and core does not import implementation or delivery layers.

Before opening a pull request, run the relevant focused tests plus the repository gates:

```console
poetry run black --check exact exact_inspect study_visualizer_runtime tests tools
poetry run isort --check-only exact exact_inspect study_visualizer_runtime tests tools
poetry run flake8 exact exact_inspect study_visualizer_runtime tests tools
poetry run mypy
poetry run lint-imports
poetry run pytest \
  -m "not requires_data and not slow and not requires_cuda and not requires_openrouter"
poetry run mkdocs build --strict
poetry run python benchmarks/bench.py --repeat 7 --check-reference
```

Defaults and descriptions live in the Pydantic config models; regenerate the committed default
YAML instead of editing two sources. Public APIs use Google-style docstrings. Add small reviewable
fixtures, mark external-data/accelerator/hosted tests accurately, and keep commits scoped by
component.

See the complete
[CONTRIBUTING.md](https://github.com/liseda-lab/Exact-OM/blob/main/CONTRIBUTING.md)
for registry/plugin patterns, markers, performance gates, and commit guidance.
