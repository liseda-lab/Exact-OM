"""Resume preserves original cold measurements and verifies saved bytes before reuse."""

import hashlib
import json
import sqlite3
from copy import deepcopy

import pytest
import yaml

from exact.experiments.harness import hash_payload
from exact.experiments.recovery import ArtifactStore, stage_identity
from exact.llm.ledger import RequestLedger
from exact.utils.provenance import sha256_file
from tools.validation_resume import DATASET_FILES, adopt_cold_probe, seed_warm_dataset


@pytest.fixture
def saved(tmp_path):
    previous = tmp_path / "previous"
    run = previous / "cold64/run"
    (run / "_inputs").mkdir(parents=True)
    config = {
        "run": {"source_cap": 64, "seed": 17},
        "data": {"execution_mode": "global_alignment", "refs": {}, "train_candidates": None},
        "dataset": {"verbalization_mode": "deterministic"},
    }
    (run / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
    campaign = "c" * 64
    limits = {
        "seconds": 41400,
        "soft_seconds": 39600,
        "requests": 2000,
        "tokens": 3200000,
        "ram_gb": 56,
    }
    ledger = RequestLedger(previous / "shared/openrouter")
    key = ledger.plan({"role": "decision", "payload": {"max_tokens": 2}})
    attempt = ledger.sent(key)
    ledger.received(key, attempt, b'{"response":"cached"}', 200)
    ledger.usage(key, attempt, {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.01})
    usage = {
        "attempts": 1,
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "reported_cost_usd": 0.01,
        "unknown": 0,
        "unpriced_attempts": 0,
    }
    vectors_path = previous / "shared/embeddings/vectors.sqlite3"
    vectors_path.parent.mkdir()
    vectors = sqlite3.connect(vectors_path)
    vectors.execute("PRAGMA journal_mode=WAL")
    vectors.execute("CREATE TABLE vectors (key TEXT PRIMARY KEY, raw BLOB)")
    vectors.execute("INSERT INTO vectors VALUES (?,?)", ("cached-vector", b"values"))
    vectors.commit()
    report = {
        "status": "failed",
        "reason": "ValueError: A completed artifact must have durable outputs",
        "stages": [],
        "campaign_sha256": campaign,
        "limits": limits,
        "elapsed_seconds": 26003.846,
        "hosted_usage": usage,
    }
    (previous / "report.json").write_text(json.dumps(report))
    worker = {"return_code": 0, "wall_seconds": 25927.177, "scorer_encoded_texts": 18844}
    (run / "validation-worker.json").write_text(json.dumps(worker))
    manifest = {
        "experiment_id": "G0",
        "source_cap": 64,
        "resolved_config_hash": hash_payload(config),
        "experiment_config_hash": campaign,
        "status": "running",
    }
    (run / "experiment_manifest.json").write_text(json.dumps(manifest))
    store = ArtifactStore(run.parent)
    kwargs = {
        "parameters": {},
        "inputs": {"source": hashlib.sha256(b"source").hexdigest()},
        "role": "development",
        "entity_kind": "all",
        "implementation": {"sha256": "original-implementation"},
        "dependencies": {},
        "seed": 17,
    }
    inputs = stage_identity("inputs", **kwargs)
    store.publish(inputs, {"_locked_inputs/source": b"source"})
    extraction = stage_identity("extraction", parents=[inputs["artifact_id"]], **kwargs)
    outputs = {
        name: b"original-" + name.encode()
        for name in [
            *DATASET_FILES,
            "alignment/maps_global.tsv",
            "source_decisions.json",
            "timings.json",
        ]
    }
    for name, data in outputs.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    store.publish(extraction, outputs)
    (run / "recovery-runtime.json").write_text(
        json.dumps({"identity": extraction, "root": str(store.root)})
    )
    yield previous, run, config, campaign, limits, key, store, extraction
    vectors.close()


def adopt(saved, output, **kwargs):
    previous, _, config, campaign, limits, *_ = saved
    return adopt_cold_probe(
        previous, output, campaign_sha256=campaign, limits=limits, expected_config=config, **kwargs
    )


def test_prepare_then_adopt_retains_original_wall_identity_usage_and_old_bytes(saved, tmp_path):
    previous, run, _, _, _, key, store, extraction = saved
    # SQLite read-only backup updates transient WAL read marks, never DB/WAL contents.
    before = {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }
    destination = tmp_path / "resumed"
    planned, elapsed = adopt(saved, destination, materialize=False)
    assert not destination.exists()
    row, actual_elapsed = adopt(saved, destination)
    assert elapsed == actual_elapsed == 26003.846
    assert row["wall_seconds"] == planned["wall_seconds"] == 25927.177
    assert row["new_worker_calls"] == 0 and not any(row["new_usage"].values())
    assert row["output_dir"] == str(run)
    assert row["recovery"]["original_identity"] == extraction
    assert row["recovery"]["parents"] == extraction["parents"]
    assert RequestLedger(destination / "shared/openrouter").cached(key) == b'{"response":"cached"}'
    with sqlite3.connect(destination / "shared/embeddings/vectors.sqlite3") as db:
        assert db.execute("SELECT raw FROM vectors").fetchone()[0] == b"values"
    assert {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    } == before
    with pytest.raises(ValueError, match="already has shared"):
        adopt(saved, destination)


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "limits",
        "campaign",
        "failure",
        "stages",
        "worker",
        "missing_worker",
        "nan_wall",
        "elapsed",
    ],
)
def test_rejects_changed_scope_missing_success_and_invalid_timings(saved, tmp_path, change):
    previous, run, config, campaign, limits, *_ = saved
    config, limits = deepcopy(config), dict(limits)
    if change == "config":
        config["run"]["seed"] = 99
    elif change == "limits":
        limits["seconds"] += 1
    elif change == "campaign":
        campaign = "d" * 64
    elif change in {"failure", "stages", "elapsed"}:
        path = previous / "report.json"
        data = json.loads(path.read_text())
        data[{"failure": "reason", "stages": "stages", "elapsed": "elapsed_seconds"}[change]] = {
            "failure": "different failure",
            "stages": [{"id": "warm64"}],
            "elapsed": 1,
        }[change]
        path.write_text(json.dumps(data))
    else:
        path = run / "validation-worker.json"
        if change == "missing_worker":
            path.unlink()
        else:
            worker = json.loads(path.read_text())
            worker["return_code" if change == "worker" else "wall_seconds"] = (
                1 if change == "worker" else float("nan")
            )
            path.write_text(json.dumps(worker))
    with pytest.raises((ValueError, FileNotFoundError)):
        adopt_cold_probe(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            limits=limits,
            expected_config=config,
        )
    assert not (tmp_path / "new/shared").exists()


@pytest.mark.parametrize("where", ["output", "blob", "parent"])
def test_rejects_tampered_durable_outputs_or_parent(saved, tmp_path, where):
    _, run, _, _, _, _, store, extraction = saved
    payload = store.verify(extraction["artifact_id"])
    if where == "output":
        (run / "alignment/maps_global.tsv").write_bytes(b"tampered")
    elif where == "blob":
        (store.root / payload["outputs"]["alignment/maps_global.tsv"]["path"]).write_bytes(
            b"tampered"
        )
    else:
        (store.directory / "stages" / f"{extraction['parents'][0]}.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        adopt(saved, tmp_path / "new")


def test_warm_seed_copies_only_verified_dataset_with_identical_config(saved, tmp_path):
    _, run, config, *_ = saved
    output = tmp_path / "warm/run"
    (output / "_inputs").mkdir(parents=True)
    (output / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
    copied = seed_warm_dataset(run, output)
    assert set(copied) == set(DATASET_FILES)
    assert {
        str(p.relative_to(output))
        for p in output.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    } == {
        *DATASET_FILES,
        "_inputs/resolved.config.yaml",
    }
    assert copied == seed_warm_dataset(run, output)
    config["run"]["source_cap"] = 300
    (output / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        seed_warm_dataset(run, output)


@pytest.fixture
def completed(saved, tmp_path, monkeypatch):
    from tools import validation_resume

    previous, _, original, campaign, limits, key, _, _ = saved
    native = {"pyowl-core": "a" * 64, "pyowl2vec-star-projector": "b" * 64}
    monkeypatch.setattr(validation_resume, "ontology_execution_identity", lambda _: native)
    expected = {}
    rows = []
    for index, name in enumerate(("cold64", "warm64", "fit64")):
        run = previous / name / "run"
        (run / "_inputs").mkdir(parents=True, exist_ok=True)
        config = deepcopy(original)
        config["data"]["reference_role"] = None
        config["dataset"]["reasoner"] = "asserted"
        config["pipeline"] = [
            {
                "name": "PairAdaptiveSemanticScorer",
                "params": {"use_llm": False, "generate_llm_rationales": False},
            }
        ]
        if name == "fit64":
            inputs = previous / "inputs"
            inputs.mkdir(exist_ok=True)
            for filename in ("train.candidates.tsv", "train.reference.tsv"):
                (inputs / filename).write_text("frozen training bytes")
            config["data"].update(
                train_candidates=str(inputs / "train.candidates.tsv"),
                refs={"train": str(inputs / "train.reference.tsv")},
            )
        (run / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
        expected[name] = deepcopy(config)
        outputs = {
            name: b"frozen output"
            for name in (
                *DATASET_FILES,
                "alignment/maps_global.tsv",
                "source_decisions.json",
                "timings.json",
            )
        }
        if name == "fit64":
            outputs["fitting/labels/training_units.json"] = b'{"groups":64}'
        for path, data in outputs.items():
            target = run / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        store = ArtifactStore(run.parent)
        identity = stage_identity(
            "extraction",
            parameters={},
            inputs={},
            role="development",
            entity_kind="all",
            implementation={"sha256": f"original-{name}"},
            dependencies={"ontology_artifacts": native},
            seed=17,
        )
        store.publish(identity, outputs)
        (run / "recovery-runtime.json").write_text(json.dumps({"identity": identity}))
        recovery = {"artifacts": {"extraction": identity["artifact_id"]}}
        manifest = {
            "status": "complete",
            "experiment_id": "G0",
            "source_cap": 64,
            "resolved_config_hash": hash_payload(config),
            "experiment_config_hash": campaign,
            "recovery": recovery,
        }
        (run / "experiment_manifest.json").write_text(json.dumps(manifest))
        worker = {
            "return_code": 0,
            "wall_seconds": 50 + index,
            "scorer_encoded_texts": 0 if name == "warm64" else 10,
        }
        if name == "warm64":
            worker["dataset_cache_hits"] = 2
        (run / "validation-worker.json").write_text(json.dumps(worker))
        row = {
            "id": name,
            "status": "complete",
            "source_cap": 64,
            "hosted": False,
            "evaluate": False,
            "new_worker_calls": 1,
            "new_usage": dict.fromkeys(validation_resume.USAGE_KEYS, 0),
            "output_dir": str(run),
            "manifest": str(run / "experiment_manifest.json"),
            "wall_seconds": 100 + index,
            "worker_measurement": worker,
            "recovery": recovery,
        }
        rows.append(row)
        (previous / f"{name}.measurement.json").write_text(json.dumps(row))
    report = json.loads((previous / "report.json").read_text())
    report.update(
        status="blocked_budget",
        reason="measured cap exceeded",
        elapsed_seconds=500,
        stages=rows + [{"id": "hosted20", "status": "complete", "hosted": True}],
    )
    (previous / "report.json").write_text(json.dumps(report))
    return previous, expected, campaign, key


def test_completed_probe_adoption_retains_measurements_and_charges_with_cold_hosted_cache(
    completed, tmp_path
):
    from tools.validation_resume import adopt_completed_probes

    previous, expected, campaign, key = completed
    before = {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }
    output = tmp_path / "resumed"
    planned, elapsed = adopt_completed_probes(
        previous, output, campaign_sha256=campaign, expected_configs=expected, materialize=False
    )
    assert not output.exists() and elapsed == 500
    rows, elapsed = adopt_completed_probes(
        previous, output, campaign_sha256=campaign, expected_configs=expected
    )
    assert set(rows) == {"cold64", "warm64", "fit64"}
    for index, name in enumerate(("cold64", "warm64", "fit64")):
        row = rows[name]
        assert row["wall_seconds"] == planned[name]["wall_seconds"] == 100 + index
        assert row["new_worker_calls"] == 0 and not any(row["new_usage"].values())
        assert row["adoption_evidence"]["prior_hosted_usage"]["attempts"] == 1
        assert row["adoption_evidence"]["hosted_cache_reused"] is False
        assert (
            row["recovery"]["original_identity"]["artifact_id"]
            == row["recovery"]["artifacts"]["extraction"]
        )
    ledger = RequestLedger(output / "shared/openrouter")
    assert ledger.cached(key) is None and ledger.summary()["roles"] == {}
    with sqlite3.connect(output / "shared/embeddings/vectors.sqlite3") as db:
        assert db.execute("SELECT raw FROM vectors").fetchone()[0] == b"values"
    assert before == {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }


def test_completed_fitting_allows_only_content_equal_training_relocation(completed, tmp_path):
    from tools.validation_resume import adopt_completed_probes

    previous, configs, campaign, _ = completed
    fit = configs["fit64"]["data"]
    relocated = tmp_path / "relocated.tsv"
    relocated.write_text("frozen training bytes")
    fit["train_candidates"] = str(relocated)
    fit["refs"]["train"] = str(relocated)
    adopt_completed_probes(
        previous,
        tmp_path / "new",
        campaign_sha256=campaign,
        expected_configs=configs,
        materialize=False,
    )
    relocated.write_text("changed training bytes")
    with pytest.raises(ValueError, match="training bytes"):
        adopt_completed_probes(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            expected_configs=configs,
            materialize=False,
        )


@pytest.mark.parametrize(
    "change",
    ["native", "worker", "measurement", "output", "config", "charges", "status", "warm_cache"],
)
def test_completed_probe_adoption_rejects_changed_evidence(
    completed, tmp_path, monkeypatch, change
):
    from tools import validation_resume

    previous, configs, campaign, _ = completed
    if change == "native":
        monkeypatch.setattr(
            validation_resume, "ontology_execution_identity", lambda _: {"pyowl-core": "changed"}
        )
    elif change == "config":
        configs["cold64"]["run"]["seed"] = 99
    elif change in {"charges", "status"}:
        p = previous / "report.json"
        report = json.loads(p.read_text())
        if change == "charges":
            report["hosted_usage"]["attempts"] = 0
        else:
            report["status"] = "failed"
        p.write_text(json.dumps(report))
    elif change == "output":
        (previous / "fit64/run/fitting/labels/training_units.json").write_text("tampered")
    elif change == "warm_cache":
        worker_path = previous / "warm64/run/validation-worker.json"
        worker = json.loads(worker_path.read_text())
        worker["dataset_cache_hits"] = 0
        worker_path.write_text(json.dumps(worker))
        path = previous / "warm64.measurement.json"
        row = json.loads(path.read_text())
        row["worker_measurement"] = worker
        path.write_text(json.dumps(row))
        path = previous / "report.json"
        report = json.loads(path.read_text())
        report["stages"][1] = row
        path.write_text(json.dumps(report))
    else:
        p = previous / (
            "warm64/run/validation-worker.json" if change == "worker" else "warm64.measurement.json"
        )
        d = json.loads(p.read_text())
        d["wall_seconds"] = 0.001
        p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        validation_resume.adopt_completed_probes(
            previous, tmp_path / "new", campaign_sha256=campaign, expected_configs=configs
        )
    assert not (tmp_path / "new/shared").exists()


@pytest.fixture
def failed_validation(completed, tmp_path, monkeypatch):
    from tools import validation_resume as resume

    origin, configs, campaign, _ = completed
    previous = tmp_path / "failed"
    adopted, _ = resume.adopt_completed_probes(
        origin, previous, campaign_sha256=campaign, expected_configs=configs
    )
    origin_report = json.loads((origin / "report.json").read_text())
    rows = list(adopted.values())
    for row in rows:
        (previous / f"{row['id']}.measurement.json").write_text(json.dumps(row))
    ledger = RequestLedger(previous / "shared/openrouter")
    key = ledger.plan({"role": "decision", "payload": {"max_tokens": 3}})
    attempt = ledger.sent(key)
    ledger.received(key, attempt, b'{"response":"rationale-free"}', 200)
    ledger.usage(key, attempt, {"prompt_tokens": 7, "completion_tokens": 1, "cost": 0.002})
    local_usage = resume._ledger_usage(previous)
    native = resume.ontology_execution_identity("asserted")
    repository = tmp_path / "repository"
    audit = repository / resume.AUDIT_FILE
    audit.parent.mkdir(parents=True)
    old_code = b"class AuditIOMixin:\n def _write_source_decisions(self):\n  return 'old'\n def untouched(self):\n  return 42\n"
    audit.write_bytes(old_code)
    monkeypatch.setattr(resume, "REPOSITORY", repository)
    monkeypatch.setattr(resume, "_source_at_revision", lambda *_: old_code)
    old_implementation = resume._code_identity(repository, evaluation=False)
    audit.write_bytes(old_code.replace(b"'old'", b"'fixed'"))
    expected = deepcopy(configs)
    for name, cap in (("hosted20", 20), ("global300", 300)):
        run = previous / name / "run"
        (run / "_inputs").mkdir(parents=True)
        config = deepcopy(configs["cold64"])
        config["run"]["source_cap"] = cap
        config["pipeline"][0]["params"]["use_llm"] = True
        expected[name] = config
        (run / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
        outputs = {name: b"saved bytes" for name in resume.DATASET_FILES}
        outputs.update(
            {
                "alignment/maps_global.tsv": b"mapping",
                "source_decisions.json": b"{}",
                "timings.json": b"[]",
            }
        )
        complete = name == "hosted20"
        store = ArtifactStore(run.parent)
        identity = stage_identity(
            "extraction",
            parameters={"configuration": config},
            inputs={},
            role="development",
            entity_kind="all",
            implementation=old_implementation,
            dependencies={"ontology_artifacts": native},
            seed=17,
        )
        if complete:
            store.publish(identity, outputs)
        else:
            outputs = {
                name: data for name, data in outputs.items() if not name.startswith("alignment/")
            }
            outputs.update(
                {
                    "checkpoints/inference_1.json": b'{"processed_examples":2}',
                    "checkpoints/inference_additional_models_1.json": json.dumps(
                        {"complete": True, "candidate_records_count": 2}
                    ).encode(),
                    "checkpoints/inference_additional_models_1.jsonl.zst": b"two saved rows",
                }
            )
            store.checkpoint(
                identity,
                completed_ids=["pair1", "pair2"],
                cursor={"next_pair": 2, "dataset_rows": 2},
                outputs=outputs,
            )
            (run / "experiment.stderr.log").write_text(
                "audit_io.py\nInvalidIndexError: Reindexing only valid with uniquely valued Index objects"
            )
        for relative, data in outputs.items():
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (run / "recovery-runtime.json").write_text(json.dumps({"identity": identity}))
        worker = {"return_code": 0 if complete else 1, "wall_seconds": 15}
        (run / "validation-worker.json").write_text(json.dumps(worker))
        recovery = {"artifacts": {"extraction": identity["artifact_id"]} if complete else {}}
        manifest = {
            "status": "complete" if complete else "failed",
            "experiment_id": "G0",
            "source_cap": cap,
            "resolved_config_hash": hash_payload(config),
            "experiment_config_hash": campaign,
            "recovery": recovery,
            "git": {"commit": "a" * 40},
        }
        (run / "experiment_manifest.json").write_text(json.dumps(manifest))
        row = {
            "id": name,
            "status": manifest["status"],
            "source_cap": cap,
            "hosted": True,
            "evaluate": not complete,
            "new_worker_calls": 1,
            "new_usage": local_usage if complete else dict.fromkeys(resume.USAGE_KEYS, 0),
            "output_dir": str(run),
            "manifest": str(run / "experiment_manifest.json"),
            "wall_seconds": 20,
            "worker_measurement": worker,
            "recovery": recovery,
        }
        rows.append(row)
        (previous / f"{name}.measurement.json").write_text(json.dumps(row))
    report = {
        "status": "failed",
        "reason": "RuntimeError: global300: worker failed",
        "campaign_sha256": campaign,
        "elapsed_seconds": 700,
        "stages": rows,
        "hosted_usage": {
            k: local_usage[k] + origin_report["hosted_usage"][k] for k in resume.USAGE_KEYS
        },
        "resume": {
            "from": str(origin),
            "previous_elapsed_seconds": origin_report["elapsed_seconds"],
            "previous_hosted_usage": origin_report["hosted_usage"],
        },
        "limits": {"seconds": None, "requests": 100000, "tokens": 32000000},
    }
    (previous / "report.json").write_text(json.dumps(report))
    return previous, expected, campaign, key


def test_failed_adoption_preserves_inherited_measurements_and_copied_ledger_charges(
    failed_validation, tmp_path
):
    from tools import validation_resume as resume

    previous, expected, campaign, key = failed_validation
    before = {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }
    output = tmp_path / "repaired"
    rows, elapsed, repair = resume.adopt_failed_validation(
        previous, output, campaign_sha256=campaign, expected_configs=expected, materialize=False
    )
    assert not output.exists() and elapsed == 700
    rows, elapsed, repair = resume.adopt_failed_validation(
        previous, output, campaign_sha256=campaign, expected_configs=expected
    )
    assert set(rows) == {"cold64", "warm64", "fit64", "hosted20"}
    assert rows["hosted20"]["measurement_usage"]["attempts"] == 1
    for row in rows.values():
        assert row["new_worker_calls"] == 0 and not any(row["new_usage"].values())
        assert row["adoption_evidence"]["prior_hosted_usage"]["attempts"] == 2
        assert row["adoption_evidence"]["prior_ledger_usage"]["attempts"] == 1
    assert (
        RequestLedger(output / "shared/openrouter").cached(key) == b'{"response":"rationale-free"}'
    )
    assert repair["repair_record"]["affected_stages"] == ["extraction"]
    assert before == {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }


def _prepared_repair(fixture, tmp_path):
    from tools import validation_resume as resume

    previous, expected, campaign, _ = fixture
    _, _, repair = resume.adopt_failed_validation(
        previous,
        tmp_path / "new",
        campaign_sha256=campaign,
        expected_configs=expected,
        materialize=False,
    )
    old = json.loads((previous / "global300/run/recovery-runtime.json").read_text())["identity"]
    current = stage_identity(
        **{
            k: v
            for k, v in old.items()
            if k not in {"artifact_id", "schema_version", "implementation"}
        },
        implementation=resume._code_identity(resume.REPOSITORY, evaluation=False),
    )
    output = tmp_path / "worker"
    output.mkdir()
    (output / "recovery-runtime.json").write_text(json.dumps({"identity": current}))
    (output / "_locked_inputs").mkdir()
    (output / "_locked_inputs/current").write_text("current verified input")
    return repair, output, old, current


def test_output_repair_restores_exact_completed_rows_without_relabeling_runtime(
    failed_validation, tmp_path
):
    from tools.validation_resume import seed_repaired_checkpoint

    repair, output, old, current = _prepared_repair(failed_validation, tmp_path)
    evidence = seed_repaired_checkpoint(repair, output)
    assert evidence["original_identity"] == old and evidence["current_identity"] == current
    assert json.loads((output / "recovery-runtime.json").read_text())["identity"] == current
    assert (output / "_locked_inputs/current").read_text() == "current verified input"
    assert (
        output / "checkpoints/inference_additional_models_1.jsonl.zst"
    ).read_bytes() == b"two saved rows"
    assert any((output / "checkpoints").iterdir())  # Harness sees a continuation before launch.
    assert evidence["new_inference_measurement"] is False


@pytest.mark.parametrize(
    "change", ["parameters", "native", "other_file", "other_method", "blob", "checkpoint"]
)
def test_output_repair_rejects_scope_and_checkpoint_changes(failed_validation, tmp_path, change):
    from tools import validation_resume as resume

    repair, output, _, _ = _prepared_repair(failed_validation, tmp_path)
    path = output / "recovery-runtime.json"
    runtime = json.loads(path.read_text())
    if change in {"parameters", "native"}:
        runtime["identity"]["parameters" if change == "parameters" else "dependencies"][
            "changed"
        ] = True
        path.write_text(json.dumps(runtime))
    elif change in {"other_file", "other_method"}:
        audit = resume.REPOSITORY / resume.AUDIT_FILE
        if change == "other_file":
            (audit.parent / "scorer.py").write_text("new_scoring = True\n")
        else:
            audit.write_text(audit.read_text().replace("return 42", "return 43"))
        runtime["identity"]["implementation"] = resume._code_identity(
            resume.REPOSITORY, evaluation=False
        )
        path.write_text(json.dumps(runtime))
    else:
        store, checkpoint = resume._repair_checkpoint(repair["checkpoint_repair"])
        path = (
            store.root / next(iter(checkpoint["outputs"].values()))["path"]
            if change == "blob"
            else store.directory
            / "checkpoints"
            / checkpoint["identity"]["artifact_id"]
            / "00000001.json"
        )
        path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        resume.seed_repaired_checkpoint(repair, output)
    assert not (output / "checkpoints").exists()


@pytest.mark.parametrize("change", ["rationale", "charges", "imported_measurement", "failed_kind"])
def test_failed_adoption_rejects_changed_policy_or_history(failed_validation, tmp_path, change):
    from tools import validation_resume as resume

    previous, configs, campaign, _ = failed_validation
    path = previous / "report.json"
    report = json.loads(path.read_text())
    if change == "rationale":
        configs["hosted20"]["pipeline"][0]["params"]["generate_llm_rationales"] = True
    elif change == "charges":
        report["hosted_usage"]["attempts"] += 1
    elif change == "imported_measurement":
        report["stages"][0]["wall_seconds"] += 1
        (previous / "cold64.measurement.json").write_text(json.dumps(report["stages"][0]))
    else:
        (previous / "global300/run/experiment.stderr.log").write_text("a different worker failure")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        resume.adopt_failed_validation(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            expected_configs=configs,
            materialize=False,
        )
