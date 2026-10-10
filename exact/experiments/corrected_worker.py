"""Small execution adapter for the corrected design, outside historical E17 locks."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

from exact.experiments.mixed_scale import binding, read_binding, verified
from exact.utils.fitted_artifacts import freeze_json


def validate_cohort(cohort_binding, *, selection_freeze):
    """Recheck original membership and queries; execution seeds never select sources."""
    from exact.experiments.mixed_scale import DESIGN, validate_selection_freeze
    from exact.experiments.public_inference import validate_population
    from exact.experiments.submission import _pools

    validate_selection_freeze(selection_freeze)
    cohort = read_binding(cohort_binding)
    case = cohort.get("case")
    if (cohort.get("kind") != "fixed_public_diagnostic_cohort"
            or case not in {"D0_E03", "D1", "D_H2_valid"}
            or cohort.get("selection_eligible") is not False
            or cohort.get("official_submission") is not False
            or cohort.get("reference_labels_used") is not False):
        raise ValueError("Expected a fixed report-only public development cohort")
    if case == "D_H2_valid":
        validate_selection_freeze(selection_freeze)
        descriptor = read_binding(cohort["public_validation_descriptor"])
        if (cohort.get("selection_freeze") != selection_freeze
                or descriptor.get("kind") != "public_validation_descriptor"
                or descriptor.get("case") != case or descriptor.get("role") != "valid"
                or descriptor.get("reference_access") != "public_development"
                or descriptor.get("source_population_basis") != "eligible_ontology_entities_independent_of_reference_positives"
                or any(descriptor.get(field) != cohort[field] for field in (
                    "source_universe", "public_queries", "target_population"))):
            raise ValueError("H2 cohort must retain its post-freeze public validation binding")
        verified(descriptor["source_ontology"])
    universe = verified(cohort["source_universe"]).read_text().splitlines()
    if not universe or len(universe) != len(set(universe)):
        raise ValueError("Invalid independent development source universe")
    if case == "D_H2_valid":
        prefix = DESIGN + "/H2-public-valid/"
        expected = sorted(universe, key=lambda iri: (hashlib.sha256((prefix + iri).encode()).hexdigest(), iri))[:1000]
    else:
        expected = verified(cohort["frozen_membership"]).read_text().splitlines()
        if len(expected) != (619 if case == "D0_E03" else 1000):
            raise ValueError("Changed existing G4 diagnostic cohort count")
    queries = [q for q in _pools(verified(cohort["public_queries"]), "bioml-local") if q["source"] in set(expected)]
    target, _ = validate_population(verified(cohort["target_population"]))
    if (len(expected) != len(set(expected)) or not set(expected) <= set(universe)
            or cohort["source_ids"] != expected or cohort["source_count"] != len(expected)
            or cohort["source_ids_sha256"] != hashlib.sha256(('\n'.join(expected) + '\n').encode()).hexdigest()
            or cohort["original_queries"] != queries or cohort["local_query_count"] != len(queries)
            or cohort["full_target_count"] != target["count"]):
        raise ValueError("Cohort changed its frozen sources, original queries or full target population")
    return cohort


def prepare_cohort_inference(config, destination, *, cohort, selection_freeze,
                             source, target, mode, fitted_artifacts=None):
    """Prepare frozen inference on fixed sources; retain every original local pool."""
    from exact.core.entities.configs.config import ConfigModel
    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments.public_inference import prepare_public_inference
    from exact.utils.frozen_inference import freeze_inference_manifest

    receipt = validate_cohort(cohort, selection_freeze=selection_freeze)
    if mode not in {"global_alignment", "local_ranking"}:
        raise ValueError("Unknown corrected execution mode")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    queries_path = destination / "cohort.queries.tsv"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(["SrcEntity", "TgtCandidates"])
    writer.writerows((q["source"], repr(q["candidates"])) for q in receipt["original_queries"])
    queries_path.write_text(stream.getvalue())
    path = prepare_public_inference(Path(config), destination / "prepared", source=Path(source),
        target=Path(target), track="bioml-global" if mode == "global_alignment" else "bioml-local",
        public_candidates=queries_path if mode == "local_ranking" else None,
        fitted_artifacts=Path(fitted_artifacts) if fitted_artifacts else None)
    manifest = json.loads(path.read_text())
    manifest.update(cohort=cohort, selection_freeze=selection_freeze,
                    original_query_ids=[q["qid"] for q in receipt["original_queries"]],
                    report_only=True, official_submission=False)
    if mode == "global_alignment":
        population = destination / "cohort.sources.txt"
        population.write_text('\n'.join(receipt["source_ids"]) + '\n')
        run = manifest["runs"][0]
        resolved = ConfigModel.load_config(verified(run["config"]))
        resolved.data.source_universe = population
        resolved.supervision.inference_artifact = None
        artifact = destination / "cohort.frozen.json"
        freeze_inference_manifest(resolved, artifact)
        resolved.supervision.inference_artifact = artifact
        output = destination / "cohort.yaml"
        output.write_text(dump_yaml_document(resolved.model_dump(mode="json", by_alias=True)))
        run.update(config=binding(output), inputs={"source_universe": binding(population)})
    result = destination / "inference.json"
    freeze_json(result, manifest)
    return result


def cohort_case(manifest, cell, full_case, selection_freeze):
    """Construct expected bounded query identity without treating it as global retrieval."""
    from exact.experiments.mixed_scale import _population_identity
    from exact.experiments.public_inference import validate_population
    from exact.experiments.submission import _pools

    cohort = validate_cohort(manifest["cohort"], selection_freeze=selection_freeze)
    if (cohort["case"] != cell["case"] or manifest.get("report_only") is not True
            or manifest.get("official_submission") is not False
            or manifest.get("selection_freeze") != selection_freeze):
        raise ValueError("Bounded inference belongs to a different frozen report-only cohort")
    target, _ = validate_population(verified(full_case["target_population"]))
    actual, _ = validate_population(verified(cohort["target_population"]))
    source, _ = validate_population(verified(full_case["source_population"]))
    eligible = set(verified(source["population"]).read_text().splitlines())
    if (_population_identity(target) != _population_identity(actual)
            or not set(cohort["source_ids"]) <= eligible):
        raise ValueError("Bounded inference changed its ontology population")
    if cell["case"] == "D_H2_valid":
        descriptor = read_binding(cohort["public_validation_descriptor"])
        if descriptor["source_ontology"]["sha256"] != source["ontology"]["sha256"]:
            raise ValueError("H2 validation source ontology differs from its public case")
    expected = [{**q, "qid": index} for index, q in enumerate(cohort["original_queries"])]
    if manifest.get("original_query_ids") != [q["qid"] for q in cohort["original_queries"]]:
        raise ValueError("Bounded inference lost original query identifiers")
    if cell["mode"] == "local_ranking":
        if _pools(verified(manifest["public_candidates"]), manifest["track"]) != expected:
            raise ValueError("Bounded local inference changed original queries or candidates")
    return cohort


def compile_worker_descriptor(destination, *, cell, inference, fitting_recipe,
                              selection_freeze, public_case, public_inputs, source_revision):
    """Write a concrete executable recipe, never a live launch declaration."""
    manifest = read_binding(inference)
    payload = dict(schema_version=1, kind="corrected_cell_worker", cell=cell,
        inference=inference, fitting_recipe=fitting_recipe, selection_freeze=selection_freeze,
        public_case=public_case, public_inputs=public_inputs, source_revision=source_revision,
        executor="pinned_published_matcher" if cell["section"] == "published" else "exact_frozen_inference",
        run_eval=False, refit=False, launchable=False,
        requires="reviewed_corrected_worker_rollout_admission", runs=manifest["runs"])
    path = Path(destination) / "workers" / (cell["id"].replace("/", "--") + ".json")
    freeze_json(path, payload)
    return binding(path)


def runtime_record(descriptor, entry, config, *, runtime, code):
    """Use the real recovery/cache identity contract, including fitted dependencies."""
    from importlib.metadata import PackageNotFoundError, version
    from exact.experiments.recovery import stage_identity
    from exact.experiments.runtime import _code_identity
    from exact.utils.frozen_inference import validate_inference_config

    fitted = validate_inference_config(config)
    if fitted is None:
        raise ValueError("Corrected runtime requires verified frozen inference")
    manifest = read_binding(descriptor["inference"])
    inputs = {name: binding(getattr(config.data, name))["sha256"]
              for name in ("source", "target", "source_universe", "candidates")
              if getattr(config.data, name) is not None}
    inputs.update({"artifact/" + name: item["sha256"] for name, item in fitted["artifacts"].items()})
    inputs["fitting_recipe"] = descriptor["fitting_recipe"]["sha256"]
    inputs["selection_freeze"] = descriptor["selection_freeze"]["sha256"]
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "numpy", "scipy", "pyowl-core",
                 "pyowl2vec-star-projector", "sentence-transformers", "pandas"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    packages["ontology_artifacts"] = {side: read_binding(descriptor["public_case"][side + "_population"])["ontology_core"]
                                       for side in ("source", "target")}
    role = "valid" if descriptor["cell"]["section"] in {"bounded", "components"} else "test"
    identity = stage_identity("extraction", parameters=config.model_dump(mode="json", by_alias=True),
        inputs=inputs, implementation=_code_identity(Path(code), evaluation=False), dependencies=packages,
        role=role, entity_kind=",".join(sorted(config.matching.entity_kinds)), seed=descriptor["cell"]["seed"])
    return {"root": str(runtime), "identity": identity, "stop_after_checkpoint": False,
            "original_query_indices": entry["query_indices"], "inference": descriptor["inference"],
            "inputs": {name: manifest[name] for name in ("source", "target")}}


def checked_write_roots(descriptor, admission, policy, environment):
    """All durable outputs and cache/scratch files must be inside measured envelopes."""
    from exact.core.entities.configs.config import ConfigModel
    guard = policy.get("storage_guard")
    if not guard:
        raise ValueError("Corrected workers require an approved storage guard covering every write root")
    allowed = [Path(guard["usage_root"]).resolve()] + [Path(row["usage_root"]).resolve()
        for row in guard.get("additional_roots", [])]
    env = {**os.environ, **environment}
    if env.get("TMPDIR"):
        temporary = Path(env["TMPDIR"])
        if not temporary.is_absolute() or not temporary.is_dir() or not os.access(temporary, os.W_OK | os.X_OK):
            raise ValueError("Reviewed TMPDIR must already exist, be absolute and writable/searchable; fallback is not admitted")
    base_cache = env.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    roots = [admission["root"], *[row["run_dir"] for row in descriptor["runs"]],
             env.get("HF_HOME", str(Path(base_cache) / "huggingface")),
             env.get("TORCH_HOME", str(Path(base_cache) / "torch")), env.get("TMPDIR") or "/tmp"]
    if env.get("MPLCONFIGDIR"):
        roots.append(env["MPLCONFIGDIR"])
    else:
        roots += [str(Path(base_cache) / "matplotlib"),
                  str(Path(env.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "matplotlib")]
    if descriptor["executor"] == "pinned_published_matcher":
        import shlex
        # JAVA_TOOL_OPTIONS is passed directly to the JVM; its last definition
        # replaces the Java default, independently of Python's TMPDIR.
        java_tmp = [item.split("=", 1)[1] for item in shlex.split(env.get("JAVA_TOOL_OPTIONS", ""))
                    if item.startswith("-Djava.io.tmpdir=")]
        selected_tmp = java_tmp[-1] if java_tmp else "/tmp"
        if not selected_tmp or not Path(selected_tmp).is_absolute():
            raise ValueError("Java temporary directory must be an absolute covered write root")
        roots.append(selected_tmp)
        # Other launcher/JVM injection mechanisms may override the option above.
        # Conservatively require coverage for those explicit locations as well.
        for name in ("JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"):
            roots += [item.split("=", 1)[1] for item in shlex.split(env.get(name, ""))
                      if item.startswith("-Djava.io.tmpdir=")]
    roots += [value for name, value in env.items() if value and (
        name.endswith(("CACHE_DIR", "CACHE_ROOT")) or name in {"HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_EXTENSIONS_DIR", "EXACT_EXTRACTION_SQLITE_DIR",
        "SENTENCE_TRANSFORMERS_HOME", "PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
        "EXACT_DATASET_CACHE_LOCAL_DIR", "EXACT_EXPERIMENT_SHARED_CACHE_ROOT",
        "EXACT_NUMERICAL_CACHE_ROOT", "TEMP", "TMP"})]
    def configured(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"cache_dir", "cache_root", "cache_path", "output_dir", "checkpoint_dir",
                           "log_dir", "scratch_dir", "scratch_root", "temporary_dir"} and isinstance(item, str) and item:
                    roots.append(item)
                else:
                    configured(item)
        elif isinstance(value, list):
            for item in value:
                configured(item)
    for row in descriptor["runs"]:
        configured(ConfigModel.load_config(verified(row["config"])).model_dump(mode="json", by_alias=True))
    # Consumers differ: transformers expands '~', while matplotlib treats it
    # literally. Both possible write locations must be covered.
    resolved = sorted({str((Path(admission["code_root"]) / candidate).resolve())
                       for path in roots for candidate in (Path(path), Path(path).expanduser())})
    outside = [path for path in resolved if not any(Path(path).is_relative_to(root) for root in allowed)]
    if outside:
        raise ValueError("Uncovered corrected worker write roots: " + repr(outside))
    return resolved
