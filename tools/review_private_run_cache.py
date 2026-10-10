"""Version private imports by adapter and resolver inputs; never rewrite old indexes."""

import json
import uuid
from pathlib import Path

from exact.runs.reader import RunReader
from exact_inspect import decisions
from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import DomainError, canonical_hash, file_hash


def prepared_private_run(root, run, *, contexts, source_version, target_version, selected_entities):
    root, run = Path(root), Path(run).resolve()
    selected = sorted({canonical_hash(entity) for entity in selected_entities})
    resolver = {
        "implementation": "review-selected-original-facts/2",
        "contexts": {
            version: file_hash(context.database) for version, context in sorted(contexts.items())
        },
        "selected_typed_entities": selected,
    }
    if any(
        resolver["contexts"][version] != context.manifest["artifacts"]["context.sqlite"]
        for version, context in contexts.items()
    ):
        raise ValueError("Private import context bytes differ from their manifest")
    binding = {
        "schema": "exact-review-private-import/1",
        "adapter": decisions.ADAPTER_REVISION,
        "run_manifest": (
            file_hash(run / "run_manifest.json")
            if (run / "run_manifest.json").is_file()
            else canonical_hash(RunReader.open(run).manifest())
        ),
        "source_decisions": (
            file_hash(run / "source_decisions.json")
            if (run / "source_decisions.json").is_file()
            else None
        ),
        "source": source_version,
        "target": target_version,
        "root_hashes": {
            version: context.manifest["identity"]["root_sha256"]
            for version, context in sorted(contexts.items())
        },
        "resolver_revision": canonical_hash(resolver),
    }
    directory = root / "private-decision-imports" / canonical_hash(binding)[7:]
    directory.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with directory.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = None
        if (directory / "decisions.manifest.json").is_file():
            try:
                saved = decisions.DecisionStore(directory)
            except (ValueError, OSError, DomainError):
                pass
        if saved is not None:
            metadata = saved.manifest()
            if (
                metadata["implementation"] != binding["adapter"]
                or metadata["context_revision"] != binding["resolver_revision"]
                or not (directory / "review-import-inputs.json").is_file()
                or json.loads((directory / "review-import-inputs.json").read_bytes()) != binding
            ):
                raise ValueError(
                    "Private index no longer agrees with its immutable preparation binding"
                )
            for relative, expected in metadata["dependencies"].items():
                path = (run / relative).resolve()
                if (
                    not path.is_relative_to(run)
                    or not path.is_file()
                    or file_hash(path) != expected
                ):
                    raise ValueError("Consumed matcher artifact changed; use a new review root")
            return saved
        if directory.exists():
            # Preserve interrupted/corrupt staging for diagnosis before preparing
            # this exact identity anew. Previously completed indexes are untouched.
            directory.rename(
                directory.with_name(directory.name + ".incomplete-" + uuid.uuid4().hex)
            )
        decisions.import_run(
            run,
            directory,
            source_ontology_version_id=source_version,
            target_ontology_version_id=target_version,
            source_root_sha256=binding["root_hashes"][source_version],
            target_root_sha256=binding["root_hashes"][target_version],
            context_revision=binding["resolver_revision"],
            evidence_resolver=lambda item, entity: (
                contexts[entity["ontology_version_id"]].resolve_feature(item, entity)
                if canonical_hash(entity) in selected
                else {}
            ),
        )
        atomic_json(directory / "review-import-inputs.json", binding)
        return decisions.DecisionStore(directory)
