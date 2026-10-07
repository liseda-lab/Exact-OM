"""Regenerate the implemented study schemas without models, OWL or provider access."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from exact_inspect.study.api import create_study_app
from exact_inspect.study.models import Publish
from exact_inspect.study.resources import ExplanationResource
from exact_inspect.study.v2_models import TutorialDefinition


def snapshots():
    with tempfile.TemporaryDirectory(prefix="exact-contracts-") as temporary:
        app = create_study_app(
            f"sqlite:///{temporary}/schema.sqlite",
            "schema-only-signing-secret-000000000000",
            "schema-only-researcher-secret-000000000",
            "https://schema.invalid",
            allow_test_sqlite=True,
        )
        return {
            "study.runtime-openapi.json": app.openapi(),
            "study-resource.runtime-schema.json": ExplanationResource.model_json_schema(),
            "study-publication.runtime-schema.json": Publish.model_json_schema(),
            "study-tutorial.runtime-schema.json": TutorialDefinition.model_json_schema(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("specs/explanation-framework/protocol"))
    args = parser.parse_args()
    for name, value in snapshots().items():
        target = args.output / name
        if args.check:
            if not target.is_file() or json.loads(target.read_text()) != value:
                raise SystemExit(f"Schema differs from implemented models: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(
        "Verified implemented study schemas" if args.check else "Updated implemented study schemas"
    )


if __name__ == "__main__":
    main()
