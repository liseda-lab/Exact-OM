#!/usr/bin/env python3
"""Generate an SPDX 2.3 SBOM for an installed Exact dependency graph."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


def _spdx_id(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9.-]", "-", canonicalize_name(name))
    return f"SPDXRef-Package-{normalized}"


def _license(distribution: metadata.Distribution) -> str:
    value = distribution.metadata.get("License-Expression")
    return value or "NOASSERTION"


def installed_graph(root: str) -> tuple[dict[str, metadata.Distribution], set[tuple[str, str]]]:
    """Resolve installed, active base requirements from ``root``."""

    distributions: dict[str, metadata.Distribution] = {}
    relationships: set[tuple[str, str]] = set()
    pending = [canonicalize_name(root)]
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    while pending:
        name = pending.pop()
        if name in distributions:
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as error:
            raise SystemExit(f"installed dependency metadata is missing for {name}") from error
        distributions[name] = distribution
        for raw in distribution.requires or ():
            try:
                requirement = Requirement(raw)
            except InvalidRequirement as error:
                raise SystemExit(f"invalid requirement in {name}: {raw}") from error
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            dependency = canonicalize_name(requirement.name)
            relationships.add((name, dependency))
            pending.append(dependency)
    return distributions, relationships


def spdx_document(root: str) -> dict[str, object]:
    """Build a deterministic package graph with a time-stamped creation record."""

    distributions, relationships = installed_graph(root)
    components = [
        f"{name}=={distribution.version}" for name, distribution in sorted(distributions.items())
    ]
    namespace_key = "\n".join(components)
    root_name = canonicalize_name(root)
    packages = []
    for name, distribution in sorted(distributions.items()):
        packages.append(
            {
                "name": distribution.metadata.get("Name", name),
                "SPDXID": _spdx_id(name),
                "versionInfo": distribution.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": _license(distribution),
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{name}@{distribution.version}",
                    }
                ],
            }
        )
    relation_rows = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": _spdx_id(root_name),
        }
    ]
    relation_rows.extend(
        {
            "spdxElementId": _spdx_id(parent),
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": _spdx_id(dependency),
        }
        for parent, dependency in sorted(relationships)
    )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{root_name}-installed-dependency-graph",
        "documentNamespace": (
            "https://liseda-lab.github.io/Exact-OM/sbom/"
            + str(uuid.uuid5(uuid.NAMESPACE_URL, namespace_key))
        ),
        "creationInfo": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "creators": ["Tool: Exact-OM tools/generate_sbom.py"],
        },
        "packages": packages,
        "relationships": relation_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="exact-om", help="Installed root distribution name.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = spdx_document(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
