"""Wrap the synthetic v2 study fixture's frozen contexts as a main-app package (J03).

The study and the exploration app must read the same frozen information where the study's
scope permits it (spec 17 J03). This copies the fixture's source/target context packages
unchanged, binds them under the publication's own visibility policy with the library's
``publish_bundle`` and prints the package path. Nothing is parsed, generated or matched.

    python explanations_visualizer/e2e/prepare_main_app_package.py FIXTURE_DIR OUTPUT_DIR
    exact-inspect serve --package OUTPUT_DIR/package.json --profile local_app --port 18985

``--extra-category`` widens the copied policy for main-app-only checks (for example
``alternate_definitions`` to page alternate definitions, which the study policy withholds);
leave it out for the J03 parity comparison, which must use the study's own policy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from exact_inspect.artifacts import publish_bundle
from exact_inspect.contracts import VisibilityPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra-category", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("The output directory must be empty")
    definition = json.loads((args.fixture / "publication.json").read_text())["definition"]
    if not definition["synthetic"]:
        parser.error("Only synthetic fixtures are accepted")
    args.output.mkdir(parents=True, exist_ok=True)
    ontologies = {}
    for name in ("source", "target"):
        manifest = json.loads((args.fixture / f"{name}-v2-context" / "manifest.json").read_text())
        shutil.copytree(args.fixture / f"{name}-v2-context", args.output / f"{name}-context")
        ontologies[manifest["ontology_version_id"]] = f"{name}-context"
    policy = VisibilityPolicy.model_validate(definition["visibility_policy"])
    if args.extra_category:
        policy = policy.model_copy(
            update={"categories": tuple(dict.fromkeys([*policy.categories, *args.extra_category]))}
        )
    path = publish_bundle(
        args.output,
        ontologies=ontologies,
        policy=policy,
        capabilities={"contexts": "available"},
        provenance={"note": "Synthetic study fixture contexts for J03 parity; not an Exact result"},
    )
    print(path)


if __name__ == "__main__":
    main()
