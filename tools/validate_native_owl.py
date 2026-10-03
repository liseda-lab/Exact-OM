#!/usr/bin/env python3
"""Strict native admission of recorded OWL inputs, without alignment references."""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path


def run_diagnostic(recipe_path):
    import pyhermit
    import pyowl_core as core

    from exact.ontology.versions import ontology_execution_identity
    from exact.utils.provenance import sha256_file
    from tools.run_directional_diagnostic import digest, verified, write

    recipe = json.loads(Path(recipe_path).read_text())
    if recipe.get("kind") != "e14_native_admission" or set(recipe["inputs"]) != {
        "source",
        "target",
    }:
        raise ValueError("Native admission accepts only original OWL inputs")
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=True)
    reasoning_inputs, reasoning_imports = recipe["inputs"], recipe["imports"]
    if recipe.get("reasoning_preparation"):
        from tools.prepare_bridge_ontology import verify_preparation

        reasoning_inputs, reasoning_imports = verify_preparation(
            recipe["reasoning_preparation"], recipe["inputs"], recipe["imports"]
        )
    paths = {side: verified(value) for side, value in reasoning_inputs.items()}
    imports = {iri: verified(value) for iri, value in reasoning_imports.items()}
    result = {
        "status": "running",
        "selection_eligible": False,
        "installed_code": ontology_execution_identity("hermit"),
        "inputs": recipe["inputs"],
        "imports": recipe["imports"],
        "ontologies": {},
    }
    if recipe.get("reasoning_preparation"):
        result.update(
            reasoning_preparation=recipe["reasoning_preparation"],
            reasoning_inputs=reasoning_inputs,
            reasoning_imports=reasoning_imports,
            admission_scope="full_original_logical_content_with_recorded_metadata_exclusions",
        )
    result["identity_sha256"] = digest(
        {
            "recipe": {key: value for key, value in recipe.items() if key != "output"},
            "installed_code": result["installed_code"],
            "validator": sha256_file(Path(__file__)),
        }
    )
    retained = output / "admission.json"
    if retained.exists():
        previous = json.loads(retained.read_text())
        if previous.get("identity_sha256") != result["identity_sha256"]:
            raise ValueError("Native admission identity changed; use a new prepared attempt")
        result["ontologies"] = previous["ontologies"]
    result["reused_ontologies"] = []
    began = time.monotonic()
    reasoner = None
    try:
        for side, path in paths.items():
            if result["ontologies"].get(side, {}).get("status") == "passed":
                result["reused_ontologies"].append(side)
                continue
            report = {"stage": "native_load", "status": "running"}
            result["ontologies"][side] = report
            write(output / "admission.json", result)
            started = time.monotonic()
            view = core.load_snapshot(
                path,
                options=core.LoadOptions(backend=core.BackendPreference.NATIVE),
                resolver=core.MappingResolver(imports),
            )
            report.update(load_seconds=time.monotonic() - started, stage="native_compile")
            write(output / "admission.json", result)
            options = {
                "backend": "native",
                "require_native_pipeline": True,
                "timeout": None,
                "workers": 1,
                "max_memory_bytes": recipe["max_memory_bytes"],
            }
            if recipe.get("max_compile_work") is not None:
                options["max_compile_work"] = recipe["max_compile_work"]
            started = time.monotonic()
            reasoner = pyhermit.Reasoner(view, config=pyhermit.ReasonerConfig(**options))
            report.update(compile_seconds=time.monotonic() - started, stage="native_queries")
            write(output / "admission.json", result)
            if not reasoner.is_consistent():
                raise ValueError("Admitted ontology is inconsistent: " + side)
            entity = core.Class(core.IRI(recipe["query_classes"][side]))
            if not reasoner.is_defined(entity) or not reasoner.entails(
                core.SubClassOf(entity, entity)
            ):
                raise ValueError("Known native class admission query failed: " + side)
            report.update(
                status="passed", stage="complete", diagnostics=dict(reasoner.diagnostics())
            )
            reasoner.dispose()
            reasoner = None
            del view
            write(output / "admission.json", result)
        for value in [*recipe["inputs"].values(), *recipe["imports"].values()]:
            verified(value)
        if recipe.get("reasoning_preparation"):
            verify_preparation(recipe["reasoning_preparation"], recipe["inputs"], recipe["imports"])
        result["status"] = "passed"
    except Exception as error:
        result.update(
            status="failed",
            error_type=type(error).__name__,
            message=str(error),
            error_code=getattr(error, "code", None),
            error_context=dict(getattr(error, "context", {}) or {}),
        )
        raise
    finally:
        if reasoner is not None:
            reasoner.dispose()
        result.update(
            elapsed_seconds=time.monotonic() - began,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write(output / "admission.json", result)
    from tools.prepared_batch import binding

    write(
        output / "completion.json",
        {
            "status": "complete",
            "selection_eligible": False,
            "diagnostic": binding(output / "admission.json"),
        },
    )
    return result
