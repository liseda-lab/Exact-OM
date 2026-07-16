# WP-H — Documentation Overhaul

**Depends on**: scaffold after WP-A (can run in wave 1); content after all WPs land.
**Size**: M. Two phases: **H-scaffold** (tooling, wave 1) and **H-content** (wave 3).
**Status**: Done (2026-07-16).

## Context

Docs today: an accurate but monolithic `README.md`; a hand-written static GitHub Pages site
(`docs/index.html` + `styles.css`, published at liseda-lab.github.io/Exact-OM);
`docs/semantic_scorer_defaults.md` (417 lines, thorough but named after the legacy scorer and
containing a hardcoded author-machine path `/home/pgcotovio/...` at lines 4 and 415);
`docs/pair_adaptive_scoring_slides.md`; uneven docstrings (strong in DeepOnto-derived modules,
thin in `candidate_generation.py` and the selector's ~100 helpers). No API reference, no build.
Config documentation drifts because it's maintained by hand next to a 408-line
`default_config.yaml`.

## H-scaffold (wave 1, small PR)

1. **MkDocs Material** + `mkdocstrings[python]` + `mkdocs-gen-files` + `mkdocs-literate-nav`
   as a `docs` dependency group. `mkdocs.yml` at repo root; docs sources under `docs/` (the
   current hand-written `index.html` site is replaced — port its content/images; keep
   `docs/assets/*`).
2. **CI**: docs job builds strictly (`mkdocs build --strict`) on PRs; deploy job publishes to
   GitHub Pages via `mkdocs gh-deploy` (or actions/deploy-pages) on pushes to `main`, replacing
   the `.nojekyll` static site at the same URL.
3. **Generated config reference**: `docs/_scripts/gen_config_reference.py` (run by
   mkdocs-gen-files) walks the pydantic models in `exact/core/entities/configs/config.py` and
   emits one page per section (field, type, default from `default_config.yaml`, docstring).
   Field descriptions come from pydantic `Field(description=...)` — seeding those descriptions
   (initially from `semantic_scorer_defaults.md` content) is part of H-content. This kills the
   drift: the reference is rebuilt from code on every deploy.
4. **Generated CLI reference**: render each argparse parser's options table (small script over
   the parsers in `delivery/cli/*`).

## H-content (wave 3)

### Site structure (nav)

```
Home                       # what Exact-OM is, pipeline diagram (port assets/pipeline.png), citation
Getting started
  Installation             # pip/poetry, CPU vs CUDA torch note, extras ([bioml-eval]) — NO Java
  Quickstart               # fixture-sized end-to-end run + expected outputs walk-through
Guides
  Datasets & tracks                   # `exact data pull/verify/status`, HF revisions, lockfile,
                                      #   adding a dataset via YAML descriptor (WP-I)
  Ontology matching (OAEI Bio-ML)     # track-based configs, eval
  KG matching (BioKG-Align)           # WP-G draft page, polished
  Property & instance matching        # matching.entity_kinds guide (WP-F)
  Evaluation                          # builtin vs bioml backends, metrics glossary, timing/OAEI reporting
  Runs, caching & resume              # output-dir layout, checkpoints, timings.json semantics (WP-C)
  Inspecting alignments               # exact-inspect open/serve/bundle, Render deployment (WP-K)
  LLM configuration                   # profiles/routing/keys (port README §LLM + llm-debug CLI)
  Tuning & Slurm                      # tools/hparam_tuner.py, run_*_job.py, sbatch
  User study                          # analysis CLIs + study workflow
Concepts
  Architecture             # port specs/01 (current-state sections updated to "as built")
  Scoring model            # pair-adaptive channels, fusion, LLM arbitration (port pair_adaptive_scoring_slides.md)
  Extending Exact-OM       # registry, KnowledgeSource, writers, exact.reasoners plugin how-to
Reference
  Configuration            # generated
  CLI                      # generated
  API                      # mkdocstrings: exact.delivery.api, exact.ontology, exact.io, core contracts
  Output files             # every artifact a run writes: run_manifest.json, layout v2,
                           #   explanation store + `exact run export/info/clean`, timings.json
Project
  Changelog | Migration (1.x→2.0) | Contributing
```

### Deliverables

1. **README.md rewrite**: short — value proposition, 10-line quickstart, install (incl. CUDA
   note), links into the site, citation, license. Everything else moves to the site. Delete the
   Java prerequisite and `-m` flag documentation (WP-B removed them).
2. **MIGRATION.md** (also a site page): 1.x → 2.0 — Java/mowl removal (`init_jvm`, heap flags),
   `EvaluationRunner` rename, `utils.paths` → `utils.graph_search`, `times.txt` → `timings.json`,
   **config v1 → v2** (`exact config migrate`, render WP-J's key map), `study_visualizer_runtime`
   → `exact_inspect`/`exact-inspect`, `data/get_data.py` → `exact data`, extras table
   (contracts §13), cache/checkpoint compatibility notes — i.e. every shim in contracts §11.
3. **CONTRIBUTING.md**: layout & layering rules (import-linter contracts), registry patterns,
   how to add a model/dataset/evaluator/source/writer/reasoner-plugin, test markers, style
   gates, spec-suite pointer.
4. **Docstring pass** (priority order): `exact/delivery/api/*`, `exact/ontology/*`,
   `exact/io/*`, `exact/core/contracts/*`, `exact/utils/{timing,candidate_generation,
   graph_search}.py` — Google style, examples on the API entry points. Gate: `interrogate`
   ≥ 80% on these packages (add to CI as non-blocking report first, blocking once met).
5. **Retire** `docs/semantic_scorer_defaults.md` (content absorbed into generated reference +
   Scoring-model page; file replaced by a redirect stub) and `docs/index.html` site.
6. `explanations_visualizer/README.md` and the `exact_inspect` service docs folded into the
   Inspecting-alignments guide; deploy/render instructions verified against the render.yaml
   service (post-WP-K names).
7. **specs/ upkeep**: mark each WP spec Done/Deviations; `specs/README.md` gets a final status
   table (the suite stays in-repo as design history).

## Acceptance criteria

1. `mkdocs build --strict` green in CI; site deployed to the existing Pages URL with no broken
   links (strict mode + `linkchecker` step).
2. Config/CLI reference pages regenerate from code — touching a pydantic field description
   changes the built page without manual doc edits.
3. README ≤ ~120 lines, no stale flags, quickstart executes verbatim on a fresh clone (CPU).
4. MIGRATION.md covers every deprecation shim listed in contracts §11.
5. `interrogate` report ≥ 80% on the priority packages; no `TODO`/`FIXME` in published pages.
6. No references to Java/JVM anywhere in docs except the migration page.

## Deviations

The documentation toolchain is a published `docs` wheel extra rather than only a Poetry
dependency group, satisfying the optional-extra contract for both pip and Poetry installs.
