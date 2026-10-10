import copy
import json
from pathlib import Path

import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.experiments import mixed_scale as mixed
from exact.experiments.corrected_worker import prepare_cohort_inference
from exact.experiments.inputs import prepare_pool
from exact.experiments.public_inference import prepare_population, prepare_public_inference
from exact.utils.provenance import sha256_path
from tests.mixed_scale_test import saved, selection_freeze


def test_full_corrected_matrix_compiles_concrete_workers_with_fixed_bounded_queries(tmp_path, monkeypatch):
    source, target = tmp_path / "source.ofn", tmp_path / "target.ofn"
    sources = [f"urn:s{i:04}" for i in range(1001)]
    source.write_text("Ontology(" + " ".join(f"Declaration(Class(<{iri}>))" for iri in sources) + ")")
    target.write_text("Ontology(Declaration(Class(<urn:t1>)) Declaration(Class(<urn:t2>)))")
    case = {}
    for side, ontology in (("source", source), ("target", target)):
        path = tmp_path / (side + ".txt")
        prepare_population(ontology, path, entity_kinds=["class"])
        case[side + "_population"] = mixed.binding(path.with_suffix(".txt.manifest.json"))
    public = tmp_path / "public.tsv"
    public.write_text("SrcEntity\tTgtCandidates\nurn:s0000\t['urn:t1']\nurn:s0000\t['urn:t2', 'urn:t1']\nurn:s0001\t[]\n")
    prepare_pool(public, tmp_path / "queries", role="test", expose_labels=False)
    case["local_queries"] = mixed.binding(tmp_path / "queries/test.inputs.json")
    case["query_count"] = 3
    inputs = saved(tmp_path / "inputs.json", {name: case for name in ("H0", "H1", "H2")})
    registry = tmp_path / "registry.json"
    registry.write_text('{"runs": [], "pending_batches": [{"id": "E17-run-once-followup"}, {"id": "unrelated"}]}')
    configs = {}
    for seed in mixed.SEEDS:
        config = ConfigModel.from_mapping({"config_version": 2, "run": {"seed": seed},
            "supervision": {"mode": "label_free"}, "dataset": {"filter_ignored_alignment_classes": False}})
        configs[seed] = tmp_path / f"selected-{seed}.json"
        configs[seed].write_text(json.dumps(config.model_dump(mode="json", by_alias=True)))
    bundle_dir = tmp_path / "logmap"
    bundle_dir.mkdir()
    (bundle_dir / "parameters.txt").write_text("fixture")
    (bundle_dir / "matcher.jar").write_text("fixture; never executed")
    matcher = {"matcher": "logmap", "bundle": {"path": str(bundle_dir), "sha256": sha256_path(bundle_dir)},
        "jar": "matcher.jar", "timeout_seconds": 60, "java_heap_gb": 2, "java_threads": 1}
    frozen = mixed.read_binding(selection_freeze(tmp_path / "selection", monkeypatch))
    recipes = {}
    for index, cell in enumerate(frozen["design"]["cells"]):
        recipe = {"kind": "frozen_published_recipe" if cell["section"] == "published" else "frozen_final_fitting_recipe",
            "cell_id": cell["id"], "selected_config": mixed.binding(configs[cell["seed"]]),
            "runtime_fitted_artifacts": None, "artifacts": {}}
        if cell["section"] == "published":
            recipe["published_matcher"] = matcher
        recipes[cell["id"]] = saved(tmp_path / f"recipe-{index}.json", recipe)
    frozen["fitting_recipes"] = recipes
    frozen.pop("identity")
    frozen["identity"] = mixed.fingerprint(frozen)
    frozen_binding = saved(tmp_path / "freeze.json", frozen)
    cohorts = {}
    for name, count in (("D0_E03", 619), ("D1", 1000), ("D_H2_valid", 1000)):
        membership = tmp_path / (name + ".txt")
        membership.write_text("\n".join(sources[:count]) + "\n")
        kwargs = {"frozen_membership": mixed.binding(membership)}
        if name == "D_H2_valid":
            kwargs = {"public_validation_descriptor": saved(tmp_path / "h2-public.json", {
                "kind": "public_validation_descriptor", "case": name, "role": "valid",
                "reference_access": "public_development", "source_population_basis": "eligible_ontology_entities_independent_of_reference_positives",
                "source_universe": mixed.binding(tmp_path / "source.txt"), "public_queries": mixed.binding(public),
                "target_population": case["target_population"], "source_ontology": mixed.binding(source),
                "dataset_revision": "a" * 40})}
        mixed.bind_diagnostic_cohort(tmp_path / name, case=name, source_universe=mixed.binding(tmp_path / "source.txt"),
            public_queries=mixed.binding(public), target_population=case["target_population"],
            selection_freeze=frozen_binding, **kwargs)
        cohorts[name] = mixed.binding(tmp_path / name / "cohort.json")
    deployments = {}
    for index, cell in enumerate(frozen["design"]["cells"]):
        output = tmp_path / f"inference-{index}"
        if cell["section"] == "bounded":
            manifest = prepare_cohort_inference(configs[cell["seed"]], output, cohort=cohorts[cell["case"]],
                selection_freeze=frozen_binding, source=source, target=target, mode=cell["mode"])
        else:
            manifest = prepare_public_inference(configs[cell["seed"]], output, source=source, target=target,
                track="bioml-global" if cell["mode"] == "global_alignment" else "bioml-local",
                public_candidates=public if cell["mode"] == "local_ranking" else None)
            if cell["section"] == "published":
                record = json.loads(manifest.read_text())
                record["published_matcher"] = matcher
                manifest = Path(saved(output / "published.json", record)["path"])
        deployments[cell["id"]] = mixed.binding(manifest)
    before = registry.read_bytes()
    result = mixed.prepare_successor_bundle(tmp_path / "successor", registry=registry, public_inputs=inputs["path"],
        selection_freeze=frozen_binding, deployments=deployments, source_revision="a" * 40)
    assert registry.read_bytes() == before and result["launchable"] is False
    from tools.prepare_corrected_queue import prepare
    proposal = prepare(tmp_path / "successor/successor-bundle.json", tmp_path / "queue")
    assert len(proposal["proposed_pending_batches"]) == 71
    assert all(not row["enabled"] and row["needs_user"] and "launch" not in row
               for row in proposal["proposed_pending_batches"])
    assert registry.read_bytes() == before
    (tmp_path / "dispatch-state.json").write_text('{"E17-run-once-followup": {"status": "uncertain"}}')
    with pytest.raises(ValueError, match="started or uncertain"):
        prepare(tmp_path / "successor/successor-bundle.json", tmp_path / "uncertain")
    (tmp_path / "dispatch-state.json").unlink()
    assert result["worker_descriptors"] == 71
    workers = [mixed.read_binding(row["worker"]) for row in result["logical_to_physical"]]
    for section, expected in (("primary", 12), ("supervision", 2), ("published", 3)):
        selected = [worker for worker in workers if worker["cell"]["section"] == section]
        assert len(selected) == expected and {worker["cell"]["seed"] for worker in selected} == {17}
    bounded = [worker for worker in workers if worker["cell"]["section"] == "bounded"]
    assert len(bounded) == 54 and {worker["cell"]["seed"] for worker in bounded} == {17, 29, 43}
    for worker in bounded:
        cell, manifest = worker["cell"], mixed.read_binding(worker["inference"])
        cohort = mixed.read_binding(cohorts[cell["case"]])
        assert cell["report_only"] and not cell["official_submission"]
        if cell["mode"] == "global_alignment":
            assert len(worker["runs"]) == 1
            config = ConfigModel.load_config(mixed.verified(worker["runs"][0]["config"]))
            assert Path(config.data.source_universe).read_text().splitlines() == cohort["source_ids"]
        else:
            assert [(q["source"], q["candidates"]) for q in manifest["queries"]] == [
                (q["source"], q["candidates"]) for q in cohort["original_queries"]]
            assert manifest["query_count"] == cohort["local_query_count"]
            assert all(len(run["query_indices"]) == 1 for run in worker["runs"])
        for run in worker["runs"]:
            config = ConfigModel.load_config(mixed.verified(run["config"]))
            assert config.seed == cell["seed"] and not config.data.refs
    bad = copy.deepcopy(mixed.read_binding(cohorts["D0_E03"]))
    bad["original_queries"].pop()
    bad_binding = saved(tmp_path / "changed-cohort.json", bad)
    with pytest.raises(ValueError, match="original queries"):
        prepare_cohort_inference(configs[17], tmp_path / "bad", cohort=bad_binding,
            selection_freeze=frozen_binding, source=source, target=target, mode="local_ranking")


def test_worker_requires_real_supervisor_policy_and_preserves_hard_pause(tmp_path):
    from tools.run_corrected_cell import admission_environment, run
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text('{"kind": "corrected_cell_worker"}')
    missing = tmp_path / "missing-admission.json"
    missing.write_text('{}')
    with pytest.raises(ValueError, match="rollout admission"):
        run(descriptor, missing)
    environment = saved(tmp_path / "environment.json", {"EXACT_OPENROUTER_RETRY_UNKNOWN": "1"})
    spending = saved(tmp_path / "policy.json", {"schema_version": 2, "kind": "exact_om_hosted_spending_policy",
        "mode": "hard_pause", "authorization": "fixture only", "notification_tokens": 100,
        "campaign_id": "campaign", "campaign_tokens_cap": 500, "experiment_tokens_cap": 200,
        "experiment_warning_tokens": 100, "admission_store": str(tmp_path / "admission.sqlite"),
        "bootstrap": {"path": str(tmp_path / "history.json"), "sha256": "a" * 64}})
    admission = {"environment": environment, "spending_policy": spending}
    result, loaded = admission_environment(admission, {"hosted_spending_policy": spending})
    assert result["EXACT_OPENROUTER_RETRY_UNKNOWN"] == "0"
    assert result["EXACT_HOSTED_EXPERIMENT_ID"] == "E17"
    assert loaded["policy"]["experiment_tokens_cap"] == 200
    with pytest.raises(ValueError, match="unchanged supervisor"):
        admission_environment({**admission, "spending_policy": {}}, {"hosted_spending_policy": spending})


def test_registered_worker_copies_history_and_accounts_reference_free_commands(tmp_path, monkeypatch):
    import os
    from exact.llm.ledger import RequestLedger
    from tests.mixed_scale_test import native_case, deployment_selection
    from tools import run_corrected_cell as worker
    from tools.prepared_batch import write
    from tools.prepare_corrected_queue import prepare

    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    monkeypatch.setenv("SLURM_STEP_ID", "999")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-never-used")
    case, source, target, queries = native_case(tmp_path)
    config = ConfigModel.from_mapping({"config_version": 2, "run": {"seed": 17},
        "supervision": {"mode": "label_free"}, "dataset": {"filter_ignored_alignment_classes": False}})
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(config.model_dump(mode="json", by_alias=True)))
    manifest = prepare_public_inference(selected, tmp_path / "inference", source=source, target=target,
        track="bioml-local", public_candidates=queries)
    cell_id = "primary/H0/stack_all/local_ranking/seed-17"
    frozen = deployment_selection(tmp_path / "frozen", monkeypatch, manifest, cell_id)
    supervisor, root = tmp_path / "supervisor", tmp_path / "worker"
    supervisor.mkdir()
    (supervisor / "supervisor-step-id").write_text("14372.42")
    limits = {"envelopes_hours": {"final": 54}, "node_hours_cap": 336,
              "requests_cap": 50000, "tokens_cap": 32000000}
    history = {"schema_version": 2, "limits": limits, "work": {"historical/closed": {
        "group": "final", "status": "complete", "start": 1, "end": 2, "seconds": 1,
        "requests": 4, "tokens": 25, "actual_usd": 0.1}}, "intervals": [[1, 2]]}
    parent = tmp_path / "parent/runtime/budget.json"
    write(parent, history)
    RequestLedger(parent.parent / "openrouter").summary()
    previous_status = tmp_path / "parent/status.json"
    write(previous_status, {"cumulative_budget": str(parent)})
    registry = {"runs": [{"id": "prior", "status_path": str(previous_status)}],
                "pending_batches": [{"id": "E17-run-once-followup", "depends_on": ["prior"]}]}
    write(supervisor / "registry.json", registry)
    inputs = saved(tmp_path / "inputs.json", {name: case for name in ("H0", "H1", "H2")})
    result = mixed.prepare_successor_bundle(tmp_path / "bundle", registry=supervisor / "registry.json",
        public_inputs=inputs["path"], selection_freeze=frozen, deployments={cell_id: mixed.binding(manifest)},
        source_revision="a" * 40)
    descriptor = next(row["worker"] for row in result["logical_to_physical"] if row["id"] == cell_id)
    environment = saved(tmp_path / "env.json", {"EXACT_OPENROUTER_RETRY_UNKNOWN": "1",
        "HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch"),
        "TMPDIR": str(tmp_path / "scratch"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "MPLCONFIGDIR": str(tmp_path / "mpl"), "JAVA_TOOL_OPTIONS": ""})
    (tmp_path / "scratch").mkdir()
    spending = saved(tmp_path / "spending.json", {"schema_version": 2, "kind": "exact_om_hosted_spending_policy",
        "mode": "hard_pause", "authorization": "fixture", "notification_tokens": 100,
        "campaign_id": "campaign", "campaign_tokens_cap": 500, "experiment_tokens_cap": 200,
        "experiment_warning_tokens": 100, "admission_store": str(tmp_path / "central.sqlite"),
        "bootstrap": {"path": str(parent), "sha256": mixed.binding(parent)["sha256"]}})
    code_root = Path(worker.__file__).resolve().parent.parent
    policy = saved(supervisor / "policy.json", {"allocation": "14372", "repository": str(tmp_path),
        "hosted_spending_policy": spending, "storage_guard": {"usage_root": str(tmp_path),
            "python": str(Path(os.sys.executable)), "source": mixed.binding(code_root / "tools/storage_guard.py"),
            "min_free_bytes": 1, "max_used_bytes": 1000000000, "growth_reserve_bytes": 1000}})
    run_id = "corrected-" + cell_id.replace("/", "--")
    admission = saved(tmp_path / "admission.json", {"kind": "corrected_worker_rollout_admission",
        "rollout_authorized": True, "descriptor": descriptor, "supervisor": str(supervisor), "root": str(root),
        "code_root": str(Path(worker.__file__).resolve().parent.parent), "supervisor_policy": policy,
        "environment": environment, "spending_policy": spending, "run_id": run_id,
        "forecast": {"seconds": 1, "requests": 0, "tokens": 0, "usd": 0},
        "commit": "a" * 40, "scientific_step": "E17", "python": str(Path(os.sys.executable)),
        "resources": {"cpus": 1, "gpus": 0, "memory_mb": 100}, "gres": "none"})
    from exact.experiments.corrected_worker import checked_write_roots
    for field in ("output", "cache"):
        changed = mixed.read_binding(descriptor)
        env = mixed.read_binding(environment)
        if field == "output":
            changed["runs"][0]["run_dir"] = str(tmp_path.parent / "uncovered-output")
        else:
            env["EXACT_NUMERICAL_CACHE_ROOT"] = str(tmp_path.parent / "uncovered-cache")
        with pytest.raises(ValueError, match="Uncovered corrected worker write roots"):
            checked_write_roots(changed, mixed.read_binding(admission), mixed.read_binding(policy), env)
    proposal = prepare(tmp_path / "bundle/successor-bundle.json", tmp_path / "queue",
                       admissions={cell_id: admission})
    declared = next(row for row in proposal["proposed_pending_batches"] if row["logical_cell"] == cell_id)
    assert not declared["enabled"] and declared["launch"]["argv"][0] == "/usr/bin/srun"
    assert declared["launch"]["run"]["hosted_scope"]["experiment_id"] == "E17"
    worker.subprocess.run(["/bin/bash", "-n", str(root / "worker-entry.sh")], check=True)
    nonce = declared["launch"]["nonce"]
    write(root / "step.json", {"step_id": "14372.999", "dispatch_nonce": nonce})
    registry["runs"].append({"id": run_id, "status_path": str(root / "status.json"),
                             "step_id": "14372.999", "dispatch_nonce": nonce})
    write(supervisor / "registry.json", registry)
    monkeypatch.setattr("tools.resume_e19_once.check_owner", lambda *args: None)
    monkeypatch.setattr(worker.subprocess, "check_output", lambda args, **kwargs: "a" * 40 if args[1] == "rev-parse" else "")
    calls = []
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            import pandas as pd
            import torch
            from types import SimpleNamespace
            from exact.experiments.runtime import runtime_checkpoint
            from exact.experiments.recovery import ArtifactStore
            from tests.numerical_cache_test import Probe
            calls.append(command)
            assert os.environ["EXACT_OPENROUTER_RETRY_UNKNOWN"] == "0"
            assert os.environ["EXACT_HOSTED_EXPERIMENT_ID"] == "E17"
            assert Path(os.environ["EXACT_OPENROUTER_LEDGER_DIR"]) == root / "runtime/openrouter"
            runtime_path = Path(os.environ["EXACT_EXPERIMENT_RUNTIME"])
            runtime = json.loads(runtime_path.read_text())
            assert runtime["identity"]["inputs"]["source"] == mixed.binding(source)["sha256"]
            assert runtime["identity"]["dependencies"]["torch"]
            model = Probe()
            model.forward(torch.tensor([0.2]))
            model.forward(torch.tensor([0.2]))
            assert model.forward_calls == 1  # Real numerical-cache scope consumes the generated mapping.
            checkpoint = runtime_path.parent / "checkpoints/fixture.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint fixture")
            frame = pd.DataFrame({"Src": ["urn:s"], "Tgt": ["urn:t1"]})
            runner = SimpleNamespace(dataset=SimpleNamespace(_active_dataframe=lambda: frame))
            runtime_checkpoint(runner, checkpoint, 1)
            restored = ArtifactStore(root / "runtime").latest_checkpoint(runtime["identity"]["artifact_id"])
            assert restored["cursor"] == {"next_pair": 1, "dataset_rows": 1}
            assert restored["completed_ids"] == ['["urn:s", "urn:t1"]']
        def poll(self):
            return 0
    monkeypatch.setattr(worker.subprocess, "Popen", Process)
    result = worker.run(Path(descriptor["path"]), Path(admission["path"]))
    assert result["status"] == "complete" and len(calls) == 2
    assert result["dispatch_nonce"] == nonce and result["exit_code"] == 0
    assert all(not {"-e", "-r", "-f"} & set(command) for command in calls)
    account = json.loads((root / "runtime/budget.json").read_text())
    assert account["limits"] == limits and account["work"]["historical/closed"] == history["work"]["historical/closed"]
    assert account["work"][cell_id + "/999"]["requests"] == 0
    assert parent.read_text() == json.dumps(history, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValueError, match="explicit recovery"):
        worker.run(Path(descriptor["path"]), Path(admission["path"]))
    for field in ("seed", "executor", "run_eval"):
        changed = mixed.read_binding(descriptor)
        if field == "seed":
            changed["cell"]["seed"] = 29
        else:
            changed[field] = "pinned_published_matcher" if field == "executor" else True
        changed_descriptor = saved(tmp_path / (field + ".json"), changed)
        changed_admission = mixed.read_binding(admission)
        changed_admission["descriptor"] = changed_descriptor
        changed_admission = saved(tmp_path / (field + "-admission.json"), changed_admission)
        # Java requires /tmp before scientific reconstruction; this fixture only
        # admits its isolated Python scratch directory. Both gates precede execution.
        expected = (r"Uncovered corrected worker write roots:.*'/tmp'"
                    if field == "executor" else "reconstructed corrected scientific descriptor")
        with pytest.raises(ValueError, match=expected):
            worker.run(Path(changed_descriptor["path"]), Path(changed_admission["path"]))
        assert len(calls) == 2
    for changed_path in (Path(mixed.read_binding(descriptor)["runs"][0]["config"]["path"]),
                         Path(inputs["path"]), supervisor / "policy.json"):
        original = changed_path.read_bytes()
        changed_path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="Changed final-study input binding"):
            worker.run(Path(descriptor["path"]), Path(admission["path"]))
        changed_path.write_bytes(original)


def test_early_worker_failure_has_reconcilable_terminal_receipt(tmp_path, monkeypatch):
    from tools.run_corrected_cell import record_exit
    from exact.experiments.dispatch import _step
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    monkeypatch.setenv("SLURM_STEP_ID", "999")
    admission = saved(tmp_path / "admission.json", {"root": str(tmp_path)})
    saved(tmp_path / "step.json", {"step_id": "14372.999", "dispatch_nonce": "fixture-nonce-123"})
    (tmp_path / "exit-code").write_text("1\n")
    record_exit(Path(admission["path"]), 1)
    launch = {"step_path": str(tmp_path / "step.json"), "nonce": "fixture-nonce-123",
              "run": {"completion_path": str(tmp_path / "completion.json"), "exit_path": str(tmp_path / "exit-code")}}
    assert _step(launch, "14372", set()) == "14372.999"
    assert json.loads((tmp_path / "completion.json").read_text())["status"] == "failed"


def test_java_temporary_root_uses_last_explicit_override(tmp_path, monkeypatch):
    import os
    from exact.experiments.corrected_worker import checked_write_roots
    monkeypatch.setattr(os, "environ", {})
    descriptor = {"executor": "pinned_published_matcher", "runs": []}
    admission = {"root": str(tmp_path / "run"), "code_root": str(tmp_path / "code")}
    policy = {"storage_guard": {"usage_root": str(tmp_path)}}
    environment = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch"),
                   "TMPDIR": str(tmp_path / "python-tmp"), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    (tmp_path / "python-tmp").mkdir()
    with pytest.raises(ValueError, match=r"Uncovered corrected worker write roots:.*'/tmp'"):
        checked_write_roots(descriptor, admission, policy, environment)
    java_tmp = tmp_path / "java tmp"
    environment["JAVA_TOOL_OPTIONS"] = f'-Djava.io.tmpdir=/uncovered/earlier -Djava.io.tmpdir="{java_tmp}"'
    roots = checked_write_roots(descriptor, admission, policy, environment)
    assert str(java_tmp) in roots and "/tmp" not in roots and "/uncovered/earlier" not in roots
    environment["JAVA_TOOL_OPTIONS"] = "-Djava.io.tmpdir=/uncovered/last"
    with pytest.raises(ValueError, match="Uncovered corrected worker write roots"):
        checked_write_roots(descriptor, admission, policy, environment)
    for value in ("", "relative"):
        environment["JAVA_TOOL_OPTIONS"] = "-Djava.io.tmpdir=" + value
        with pytest.raises(ValueError, match="absolute covered write root"):
            checked_write_roots(descriptor, admission, policy, environment)


@pytest.mark.parametrize("field", ["EXACT_EXTRACTION_SQLITE_DIR", "TORCH_EXTENSIONS_DIR", "HF_DATASETS_CACHE", "MPLCONFIGDIR",
    "SENTENCE_TRANSFORMERS_HOME", "PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE"])
def test_additional_cache_and_extraction_roots_require_storage_coverage(tmp_path, monkeypatch, field):
    import os
    from exact.experiments.corrected_worker import checked_write_roots
    monkeypatch.setattr(os, "environ", {})
    descriptor = {"executor": "exact_frozen_inference", "runs": []}
    admission = {"root": str(tmp_path / "run"), "code_root": str(tmp_path / "code")}
    policy = {"storage_guard": {"usage_root": str(tmp_path)}}
    environment = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch"),
                   "TMPDIR": str(tmp_path / "tmp"), "MPLCONFIGDIR": str(tmp_path / "mpl"),
                   field: str(tmp_path.parent / "uncovered")}
    (tmp_path / "tmp").mkdir()
    with pytest.raises(ValueError, match="Uncovered corrected worker write roots"):
        checked_write_roots(descriptor, admission, policy, environment)
    environment[field] = str(tmp_path / field)
    assert environment[field] in checked_write_roots(descriptor, admission, policy, environment)


def test_tilde_cache_path_is_checked_after_real_home_expansion(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from exact.experiments.corrected_worker import checked_write_roots
    monkeypatch.setattr(os, "environ", {"HOME": str(tmp_path.parent)})
    config = saved(tmp_path / "config.json", {})
    monkeypatch.setattr(ConfigModel, "load_config", lambda _: SimpleNamespace(
        model_dump=lambda **kwargs: {"pipeline": [{"params": {"cache_dir": "~/uncovered"}}]}))
    descriptor = {"executor": "exact_frozen_inference", "runs": [{"config": config, "run_dir": str(tmp_path / "output")}]}
    admission = {"root": str(tmp_path / "run"), "code_root": str(tmp_path / "code")}
    policy = {"storage_guard": {"usage_root": str(tmp_path)}}
    environment = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch"),
                   "TMPDIR": str(tmp_path / "tmp"), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    (tmp_path / "tmp").mkdir()
    with pytest.raises(ValueError, match="Uncovered corrected worker write roots"):
        checked_write_roots(descriptor, admission, policy, environment)
    descriptor["runs"] = []
    environment["HF_HOME"] = "~/uncovered"
    with pytest.raises(ValueError, match="Uncovered corrected worker write roots"):
        checked_write_roots(descriptor, admission, policy, environment)


def test_missing_temporary_directory_cannot_fall_back_outside_guard(tmp_path, monkeypatch):
    import os
    from exact.experiments.corrected_worker import checked_write_roots
    monkeypatch.setattr(os, "environ", {})
    descriptor = {"executor": "exact_frozen_inference", "runs": []}
    admission = {"root": str(tmp_path / "run"), "code_root": str(tmp_path / "code")}
    policy = {"storage_guard": {"usage_root": str(tmp_path)}}
    environment = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch"),
                   "TMPDIR": str(tmp_path / "tmp"), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    with pytest.raises(ValueError, match="TMPDIR must already exist"):
        checked_write_roots(descriptor, admission, policy, environment)
    (tmp_path / "tmp").mkdir()
    assert environment["TMPDIR"] in checked_write_roots(descriptor, admission, policy, environment)
    environment["TMPDIR"] = ""
    with pytest.raises(ValueError, match=r"Uncovered corrected worker write roots:.*'/tmp'"):
        checked_write_roots(descriptor, admission, policy, environment)
    monkeypatch.chdir(tmp_path)
    environment["TMPDIR"] = "tmp"
    with pytest.raises(ValueError, match="TMPDIR must already exist"):
        checked_write_roots(descriptor, admission, policy, environment)
    environment["TMPDIR"] = str(tmp_path / "tmp")
    monkeypatch.setattr(os, "access", lambda *args: False)
    with pytest.raises(ValueError, match="TMPDIR must already exist"):
        checked_write_roots(descriptor, admission, policy, environment)


def test_matplotlib_default_config_and_cache_roots_are_both_guarded(tmp_path, monkeypatch):
    import os
    from exact.experiments.corrected_worker import checked_write_roots
    monkeypatch.setattr(os, "environ", {})
    descriptor = {"executor": "exact_frozen_inference", "runs": []}
    admission = {"root": str(tmp_path / "run"), "code_root": str(tmp_path / "code")}
    policy = {"storage_guard": {"usage_root": str(tmp_path)}}
    environment = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch"),
                   "TMPDIR": str(tmp_path / "tmp"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
                   "XDG_CONFIG_HOME": str(tmp_path.parent / "uncovered-config")}
    (tmp_path / "tmp").mkdir()
    with pytest.raises(ValueError, match="Uncovered corrected worker write roots"):
        checked_write_roots(descriptor, admission, policy, environment)
    environment["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    roots = checked_write_roots(descriptor, admission, policy, environment)
    assert str(tmp_path / "cache/matplotlib") in roots and str(tmp_path / "config/matplotlib") in roots
