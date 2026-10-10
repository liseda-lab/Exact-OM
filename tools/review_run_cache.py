"""Reuse only fully verified, context-bound portable decision exports."""

import json
from pathlib import Path

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import DomainError, canonical_hash, file_hash
from exact_inspect.decisions import DecisionStore
from exact_inspect.decisions_export import POLICY_EXPORTER_REVISION


def prepared_policy_run(
    source, destination, *, contexts, policy, checkpoint, adopt_existing, exporter
):
    destination, checkpoint = Path(destination), Path(checkpoint)
    if any(
        context.manifest.get("policy_filter", {}).get("policy_hash") != policy.policy_hash
        for context in contexts.values()
    ):
        raise ValueError("Prepared export contexts must match the visibility policy")
    revision = source.manifest()["revision"]
    binding = {
        "schema": "exact-review-policy-export/1",
        "source_revision": revision,
        "policy_hash": policy.policy_hash,
        "exporter": POLICY_EXPORTER_REVISION,
        "context_hashes": {
            key: file_hash(context.database) for key, context in sorted(contexts.items())
        },
    }
    if any(
        binding["context_hashes"][key] != context.manifest["artifacts"]["context.sqlite"]
        for key, context in contexts.items()
    ):
        raise ValueError("Prepared export context bytes differ from their manifest")
    expected_revision = canonical_hash(
        {key: binding[key] for key in ("source_revision", "policy_hash", "exporter")}
    )
    try:
        cached_binding = json.loads(checkpoint.read_bytes()) if checkpoint.is_file() else None
    except (OSError, ValueError):
        cached_binding = {}
    if cached_binding == binding or (cached_binding is None and adopt_existing):
        try:
            existing = DecisionStore(destination)
            manifest = existing.manifest()
            if (
                manifest.get("policy_filter")
                == {"policy_hash": policy.policy_hash, "source_revision": revision}
                and manifest.get("revision") == expected_revision
            ):
                if cached_binding is None:
                    atomic_json(checkpoint, binding)
                return existing
        except (OSError, ValueError, DomainError):
            # A missing/partial/corrupt unpublished export is rebuilt explicitly.
            pass
    result = exporter(source, destination, contexts=contexts, policy=policy)
    if result.manifest().get("revision") != expected_revision or result.manifest().get(
        "policy_filter"
    ) != {"policy_hash": policy.policy_hash, "source_revision": revision}:
        raise ValueError("Prepared policy exporter returned an unexpected revision")
    atomic_json(checkpoint, binding)
    return result
