"""Resume preserves original cold measurements and verifies saved bytes before reuse."""

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

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


@pytest.fixture
def posthoc_validation(failed_validation, tmp_path):
    from tools import validation_resume as resume

    previous, configs, campaign, key = failed_validation
    current = tmp_path / "posthoc"
    probes, prior_elapsed, _ = resume.adopt_failed_validation(
        previous, current, campaign_sha256=campaign, expected_configs=configs
    )
    for name, row in probes.items():
        (current / f"{name}.measurement.json").write_text(json.dumps(row))
    run = current / "global300/run"
    (run / "_inputs").mkdir(parents=True)
    config = configs["global300"]
    (run / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
    identity = stage_identity(
        "extraction",
        parameters={"configuration": config},
        inputs={},
        role="development",
        entity_kind="all",
        implementation=resume._code_identity(resume.REPOSITORY, evaluation=False),
        dependencies={"ontology_artifacts": resume.ontology_execution_identity("asserted")},
        seed=17,
    )
    outputs = {
        name: b"finished extraction"
        for name in (
            *resume.DATASET_FILES,
            "alignment/maps_global.tsv",
            "source_decisions.json",
            "timings.json",
        )
    }
    for name, data in outputs.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    ArtifactStore(run.parent).publish(identity, outputs)
    (run / "recovery-runtime.json").write_text(json.dumps({"identity": identity}))
    worker = {"return_code": 0, "wall_seconds": 3, "scorer_encoded_texts": 0}
    (run / "validation-worker.json").write_text(json.dumps(worker))
    recovery = {"artifacts": {"extraction": identity["artifact_id"]}}
    manifest = {
        "experiment_id": "G0",
        "status": "failed",
        "return_code": 0,
        "source_cap": 300,
        "extraction_complete": True,
        "resolved_config_hash": hash_payload(config),
        "experiment_config_hash": campaign,
        "recovery": recovery,
        "failure": {
            "type": "ValueError",
            "message": "E00 reference rows require source, target, and a canonical relation",
        },
    }
    (run / "experiment_manifest.json").write_text(json.dumps(manifest))
    row = {
        "id": "global300",
        "status": "failed",
        "source_cap": 300,
        "hosted": True,
        "evaluate": True,
        "new_worker_calls": 1,
        "new_usage": dict.fromkeys(resume.USAGE_KEYS, 0),
        "output_dir": str(run),
        "manifest": str(run / "experiment_manifest.json"),
        "wall_seconds": 5,
        "worker_measurement": worker,
        "recovery": recovery,
    }
    (current / "global300.measurement.json").write_text(json.dumps(row))
    previous_report = json.loads((previous / "report.json").read_text())
    report = {
        "status": "failed",
        "reason": "RuntimeError: global300: E00 posthoc failure",
        "campaign_sha256": campaign,
        "elapsed_seconds": prior_elapsed + 10,
        "stages": [*probes.values(), row],
        "hosted_usage": previous_report["hosted_usage"],
        "resume": {
            "from": str(previous),
            "previous_elapsed_seconds": prior_elapsed,
            "previous_hosted_usage": previous_report["hosted_usage"],
            "copied_hosted_usage": resume._ledger_usage(previous),
        },
        "limits": {"seconds": None, "requests": 100000, "tokens": 32000000},
    }
    (current / "report.json").write_text(json.dumps(report))
    return current, configs, campaign, key


def test_posthoc_failure_reuses_extraction_and_preserves_multi_attempt_charges(
    posthoc_validation, tmp_path, monkeypatch
):
    from exact.experiments.recovery import build_reuse_plan
    from tools import validation_resume as resume

    previous, configs, campaign, key = posthoc_validation
    # A historical output repair must not be reclassified against a later evaluator fix.
    monkeypatch.setattr(
        resume, "_audit_repair_compatibility", lambda *_: pytest.fail("old repair revalidated")
    )
    evaluator = resume.REPOSITORY / "exact/experiments/error_attribution.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("evaluation_fix = True\n")
    output = tmp_path / "final"
    rows, elapsed, metadata = resume.adopt_failed_validation(
        previous, output, campaign_sha256=campaign, expected_configs=configs
    )
    assert elapsed == 710
    assert "checkpoint_repair" not in metadata
    assert metadata["repair_record"]["affected_stages"] == ["evaluation"]
    assert rows["hosted20"]["measurement_usage"]["attempts"] == 1
    assert rows["cold64"]["wall_seconds"] == 100
    for row in rows.values():
        assert row["adoption_evidence"]["prior_hosted_usage"]["attempts"] == 2
        assert row["adoption_evidence"]["prior_ledger_usage"]["attempts"] == 1
        assert row["new_worker_calls"] == 0 and not any(row["new_usage"].values())
    assert (
        RequestLedger(output / "shared/openrouter").cached(key) == b'{"response":"rationale-free"}'
    )
    # The normal recovery dependency graph, with no checkpoint seeding, reuses extraction.
    identity = json.loads((previous / "global300/run/recovery-runtime.json").read_text())[
        "identity"
    ]
    target = ArtifactStore(output / "global300")
    target.import_artifact(previous / "global300", identity["artifact_id"])
    evaluation = stage_identity(
        "evaluation",
        parameters={},
        inputs={},
        role="development",
        entity_kind="all",
        implementation={"fixed": True},
        dependencies={},
        parents=[identity["artifact_id"]],
    )
    plan = build_reuse_plan(
        target,
        {"extraction": identity, "evaluation": evaluation},
        {"extraction": identity["artifact_id"]},
        changed_stages=["evaluation"],
    )
    assert [(r["stage"], r["action"]) for r in plan["stages"]] == [
        ("extraction", "reuse"),
        ("evaluation", "recompute"),
    ]


@pytest.mark.parametrize(
    "change",
    ["copied_charges", "elapsed", "extraction", "implementation", "imported_probe", "new_usage"],
)
def test_posthoc_failure_rejects_incomplete_extraction_and_changed_chain(
    posthoc_validation, tmp_path, change
):
    from tools import validation_resume as resume

    previous, configs, campaign, _ = posthoc_validation
    path = previous / "report.json"
    report = json.loads(path.read_text())
    if change == "copied_charges":
        report["resume"]["copied_hosted_usage"]["attempts"] += 1
    elif change == "elapsed":
        report["elapsed_seconds"] = report["resume"]["previous_elapsed_seconds"]
    elif change == "extraction":
        (previous / "global300/run/source_decisions.json").write_bytes(b"changed")
    elif change == "implementation":
        audit = resume.REPOSITORY / resume.AUDIT_FILE
        audit.write_text(audit.read_text().replace("'fixed'", "'new scoring'"))
    elif change == "imported_probe":
        report["stages"][3]["measurement_usage"]["attempts"] += 1
        (previous / "hosted20.measurement.json").write_text(json.dumps(report["stages"][3]))
    else:
        report["stages"][-1]["new_usage"]["attempts"] = 1
        (previous / "global300.measurement.json").write_text(json.dumps(report["stages"][-1]))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        resume.adopt_failed_validation(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            expected_configs=configs,
            materialize=False,
        )


@pytest.fixture
def completed_globals(posthoc_validation, tmp_path, request):
    from tools import validation_resume as resume

    previous, configs, campaign, key = posthoc_validation
    current = tmp_path / "completed-globals"
    probes, prior_elapsed, _ = resume.adopt_failed_validation(
        previous, current, campaign_sha256=campaign, expected_configs=configs
    )
    options = getattr(request, "param", {})
    rows = list(probes.values())
    config = configs["global300"]
    for name in ("global300", "global300-stop", "global300-resume", "global300-replay"):
        run = current / name / "run"
        (run / "_inputs").mkdir(parents=True)
        (run / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
        identity = stage_identity(
            "extraction",
            parameters={"configuration": config},
            inputs={},
            role="development",
            entity_kind="all",
            implementation=resume._code_identity(resume.REPOSITORY, evaluation=False),
            dependencies={"ontology_artifacts": resume.ontology_execution_identity("asserted")},
            seed=17,
        )
        interrupted = name.endswith("-stop")
        worker = name.endswith(("-stop", "-resume"))
        outputs = {name: b"frozen" for name in resume.DATASET_FILES}
        score = 0.8 + (
            options.get("score_delta", 8e-8)
            if name in {"global300-resume", "global300-replay"}
            else 0
        )
        if name == "global300-replay":
            score += options.get("cache_score_delta", 0)
        target = options.get("target", "t") if name != "global300" else "t"
        outputs.update(
            {
                "alignment/maps_global.tsv": f"SrcEntity\tTgtEntity\tScore\tRelation\ns\t{target}\t{score}\t=\n".encode(),
                "source_decisions.json": b"{}",
                "timings.json": b"[]",
                "stats/run_stats.json": json.dumps(
                    {
                        "observed_execution": {
                            "device_type": "cuda",
                            "device": (
                                options.get("device", "cuda:0") if name != "global300" else "cuda:0"
                            ),
                        }
                    }
                ).encode(),
            }
        )
        store = ArtifactStore(run.parent)
        if interrupted:
            outputs = {k: v for k, v in outputs.items() if k.startswith("dataset/")}
            outputs["checkpoints/inference_1.json"] = b'{"processed_examples":1}'
            initial_timing = {
                "schema_version": 1,
                "sessions": [
                    {
                        "run_id": "stop",
                        "ended_at": None,
                        "stages": [{"stage": "Dataset", "cache_status": "fresh", "seconds": 0.2}],
                    }
                ],
            }
            outputs["timings.json"] = json.dumps(initial_timing).encode()
            store.checkpoint(
                identity,
                completed_ids=["pair1"],
                cursor={"next_pair": 1, "dataset_rows": 2},
                outputs=outputs,
            )
            (run / "interrupted.json").write_text(
                json.dumps({"status": "interrupted", "completed_pairs": 1})
            )
            artifacts = {}
        else:
            store.publish(identity, outputs)
            evaluation = stage_identity(
                "evaluation",
                parameters={},
                inputs={},
                role="development",
                entity_kind="all",
                implementation=resume._code_identity(resume.REPOSITORY, evaluation=True),
                dependencies={},
                parents=[identity["artifact_id"]],
            )
            metric = 0.8 + (options.get("metric_delta", 0) if name != "global300" else 0)
            evaluation_outputs = {
                "evaluation/evaluation_results.json": json.dumps(
                    {
                        "builtin": {"P": metric, "R": metric, "F1": metric},
                        "meta": {"refs": {"full_reference": {"path": "fixture"}}},
                    }
                ).encode()
            }
            store.publish(evaluation, evaluation_outputs)
            outputs.update(evaluation_outputs)
            artifacts = {
                "extraction": identity["artifact_id"],
                "evaluation": evaluation["artifact_id"],
            }
        for relative, data in outputs.items():
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if name == "global300":
            stats_path = run / "stats/run_stats.json"
            stats = json.loads(stats_path.read_text())
            stats["evaluation_inputs"] = {"full_reference": {"path": "fixture"}}
            stats_path.write_text(json.dumps(stats))
        (run / "recovery-runtime.json").write_text(
            json.dumps({"identity": identity, "stop_after_checkpoint": interrupted})
        )
        recovery = {"artifacts": artifacts}
        manifest = {
            "status": "interrupted" if interrupted else "complete",
            "return_code": 130 if interrupted else 0,
            "extraction_complete": not interrupted,
            "resolved_config_hash": hash_payload(config),
            "experiment_config_hash": campaign,
            "recovery": recovery,
        }
        if interrupted:
            finalized_timing = deepcopy(initial_timing)
            finalized_timing["sessions"][0]["stages"].extend(
                [
                    {"stage": "Alignment", "cache_status": "fresh", "seconds": 0.3},
                    {"stage": "Total", "cache_status": "fresh", "seconds": 0.6},
                ]
            )
            (run / "timings.json").write_text(json.dumps(finalized_timing))
            manifest["timing_ledger"] = finalized_timing
        (run / "experiment_manifest.json").write_text(json.dumps(manifest))
        row = {
            "id": name,
            "status": manifest["status"],
            "source_cap": 300,
            "hosted": True,
            "evaluate": True,
            "new_worker_calls": int(worker),
            "new_usage": dict.fromkeys(resume.USAGE_KEYS, 0),
            "output_dir": str(run),
            "manifest": str(run / "experiment_manifest.json"),
            "wall_seconds": 2,
            "recovery": recovery,
        }
        if interrupted:
            row["timing_ledger"] = finalized_timing
        if worker:
            measurement = {"return_code": manifest["return_code"], "wall_seconds": 1}
            row["worker_measurement"] = measurement
            (run / "validation-worker.json").write_text(json.dumps(measurement))
        rows.append(row)
    for row in rows:
        (current / f"{row['id']}.measurement.json").write_text(json.dumps(row))
    old = json.loads((previous / "report.json").read_text())
    report = {
        "status": "failed",
        "reason": "ValueError: Global interruption/relocation replay changed mappings",
        "campaign_sha256": campaign,
        "elapsed_seconds": prior_elapsed + 10,
        "stages": rows,
        "hosted_usage": old["hosted_usage"],
        "resume": {
            "from": str(previous),
            "previous_elapsed_seconds": prior_elapsed,
            "previous_hosted_usage": old["hosted_usage"],
            "copied_hosted_usage": resume._ledger_usage(previous),
        },
    }
    (current / "report.json").write_text(json.dumps(report))
    return current, configs, campaign, key


def test_adopts_completed_globals_using_declared_tolerance_and_exact_cache(
    completed_globals, tmp_path
):
    from tools import validation_resume as resume

    previous, configs, campaign, key = completed_globals
    before = {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }
    output = tmp_path / "local-continuation"
    rows, elapsed, repair = resume.adopt_failed_validation(
        previous, output, campaign_sha256=campaign, expected_configs=configs
    )
    assert len(rows) == 8 and elapsed == 720 and repair == {}
    assert rows["global300-stop"]["status"] == "interrupted"
    assert rows["hosted20"]["measurement_usage"]["attempts"] == 1
    for row in rows.values():
        assert row["new_worker_calls"] == 0 and not any(row["new_usage"].values())
        assert row["adoption_evidence"]["prior_hosted_usage"]["attempts"] == 2
        assert row["adoption_evidence"]["prior_ledger_usage"]["attempts"] == 1
    evidence = json.loads((output / "global300.replay-validation.json").read_text())
    assert 0 < evidence["interruption"]["max_score_delta"] < 1e-5
    assert evidence["completed_cache"]["completed_cache_bytes_verified"] is True
    stop_timing = evidence["verified_stages"]["global300-stop"]["timing_finalization"]
    assert stop_timing["original_sha256"] != stop_timing["final_sha256"]
    stats = evidence["verified_stages"]["global300"]["statistics_provenance"]
    assert stats["original_sha256"] != stats["current_sha256"]
    assert (
        stats["verified_by_evaluation"]
        == evidence["verified_stages"]["global300"]["evaluation_artifact_id"]
    )
    assert rows["global300-replay"]["adoption_evidence"]["replay_validation"] == evidence
    assert (
        RequestLedger(output / "shared/openrouter").cached(key) == b'{"response":"rationale-free"}'
    )
    assert before == {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }


@pytest.mark.parametrize(
    "completed_globals",
    [
        {"score_delta": 0.001},
        {"target": "changed"},
        {"cache_score_delta": 1e-9},
        {"metric_delta": 0.001},
        {"device": "cpu"},
    ],
    indirect=True,
)
def test_completed_global_adoption_rejects_real_replay_changes(completed_globals, tmp_path):
    from tools import validation_resume as resume

    previous, configs, campaign, _ = completed_globals
    with pytest.raises(ValueError):
        resume.adopt_failed_validation(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            expected_configs=configs,
            materialize=False,
        )


@pytest.mark.parametrize(
    "change",
    [
        "checkpoint",
        "evaluation",
        "worker",
        "usage",
        "implementation",
        "stats_refs",
        "stats_measurement",
    ],
)
def test_completed_global_adoption_rejects_changed_evidence(completed_globals, tmp_path, change):
    from tools import validation_resume as resume

    previous, configs, campaign, _ = completed_globals
    if change == "checkpoint":
        (previous / "global300-stop/run/interrupted.json").write_text(
            '{"status":"interrupted","completed_pairs":2}'
        )
    elif change == "evaluation":
        (previous / "global300-resume/run/evaluation/evaluation_results.json").write_text("{}")
    elif change == "worker":
        (previous / "global300-replay/run/validation-worker.json").write_text('{"return_code":0}')
    elif change == "implementation":
        (resume.REPOSITORY / "exact/scorer.py").write_text("changed = True\n")
    elif change.startswith("stats_"):
        path = previous / "global300/run/stats/run_stats.json"
        stats = json.loads(path.read_text())
        if change == "stats_refs":
            stats["evaluation_inputs"]["full_reference"]["path"] = "unverified"
        else:
            stats["wall_seconds"] = 0
        path.write_text(json.dumps(stats))
    else:
        path = previous / "report.json"
        report = json.loads(path.read_text())
        report["hosted_usage"]["attempts"] += 1
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        resume.adopt_failed_validation(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            expected_configs=configs,
            materialize=False,
        )


@pytest.mark.parametrize("change", ["prefix", "total", "unbound"])
def test_interrupted_timing_finalization_cannot_replace_prior_measurements(
    completed_globals, tmp_path, change
):
    from tools import validation_resume as resume

    previous, configs, campaign, _ = completed_globals
    run = previous / "global300-stop/run"
    timing = json.loads((run / "timings.json").read_text())
    timing["sessions"][0]["stages"][0 if change == "prefix" else -1]["seconds"] = (
        0.1 if change == "prefix" else 100
    )
    (run / "timings.json").write_text(json.dumps(timing))
    if change != "unbound":
        manifest_path = run / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["timing_ledger"] = timing
        manifest_path.write_text(json.dumps(manifest))
        report_path = previous / "report.json"
        report = json.loads(report_path.read_text())
        row = next(row for row in report["stages"] if row["id"] == "global300-stop")
        row["timing_ledger"] = timing
        report_path.write_text(json.dumps(report))
        (previous / "global300-stop.measurement.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="timing finalization"):
        resume.adopt_failed_validation(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            expected_configs=configs,
            materialize=False,
        )


@pytest.mark.parametrize(
    "change", [None, "metric", "reference_hash", "reference_path", "file_bytes"]
)
def test_evaluator_relocation_preserves_metrics_and_verified_reference_content(tmp_path, change):
    from tools.validation_resume import _verify_run_outputs

    old = tmp_path / "old/run/alignment/maps_global.tsv"
    new = tmp_path / "new/run/alignment/maps_global.tsv"
    for path in (old, new):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"unchanged mapping rows")
    original = {
        "builtin": {"F1": 0.8},
        "meta": {
            "refs": {
                "alignment": {
                    "path": str(old),
                    "sha256": sha256_file(old),
                    "bytes": old.stat().st_size,
                    "rows": 1,
                }
            }
        },
    }
    store = ArtifactStore(new.parents[2])
    identity = stage_identity(
        "evaluation",
        parameters={},
        inputs={},
        role="development",
        entity_kind="all",
        implementation={"fixture": True},
        dependencies={},
    )
    payload = store.publish(
        identity, {"evaluation/evaluation_results.json": json.dumps(original).encode()}
    )
    current = deepcopy(original)
    current["meta"]["refs"]["alignment"]["path"] = str(new)
    if change == "metric":
        current["builtin"]["F1"] += 0.001
    elif change == "reference_hash":
        current["meta"]["refs"]["alignment"]["sha256"] = "0" * 64
    elif change == "reference_path":
        current["meta"]["refs"]["alignment"]["path"] = str(tmp_path / "unrelated")
    elif change == "file_bytes":
        new.write_bytes(b"changed mapping rows")
    report = new.parents[1] / "evaluation/evaluation_results.json"
    report.parent.mkdir()
    report.write_text(json.dumps(current))
    verified = store.verify(payload["identity"]["artifact_id"])
    if change is None:
        _verify_run_outputs(new.parents[1], verified, store=store)
    else:
        with pytest.raises(ValueError, match="differs from verified artifact"):
            _verify_run_outputs(new.parents[1], verified, store=store)


@pytest.fixture
def interrupted_validation(completed_globals, tmp_path):
    from tools import validation_resume as resume

    origin, configs, campaign, key = completed_globals
    current = tmp_path / "reportless"
    inherited, prior_elapsed, _ = resume.adopt_failed_validation(
        origin, current, campaign_sha256=campaign, expected_configs=configs
    )
    expected = deepcopy(configs)
    config = deepcopy(configs["global300"])
    config["data"]["execution_mode"] = "local_ranking"
    expected["local300"] = config
    old_usage = resume._ledger_usage(current)
    ledger = RequestLedger(current / "shared/openrouter")
    local_key = ledger.plan({"role": "decision", "payload": {"max_tokens": 4}})
    attempt = ledger.sent(local_key)
    ledger.received(local_key, attempt, b'{"response":"local-scored"}', 200)
    ledger.usage(local_key, attempt, {"prompt_tokens": 11, "completion_tokens": 1, "cost": 0.003})
    usage = resume._ledger_usage(current)
    run = current / "local300/run"
    (run / "_inputs").mkdir(parents=True)
    (run / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
    store = ArtifactStore(run.parent)
    inputs = stage_identity(
        "inputs",
        parameters={},
        inputs={},
        role="development",
        entity_kind="all",
        implementation={"fixture": True},
        dependencies={},
    )
    store.publish(inputs, {"locked_input": b"safe"})
    identity = stage_identity(
        "extraction",
        parameters={"configuration": config},
        inputs={},
        role="development",
        entity_kind="all",
        implementation=resume._code_identity(resume.REPOSITORY, evaluation=False),
        dependencies={"ontology_artifacts": resume.ontology_execution_identity("asserted")},
        seed=17,
        parents=[inputs["artifact_id"]],
    )
    outputs = {
        name: b"local scoring complete"
        for name in (
            *resume.DATASET_FILES,
            "alignment/maps_local.tsv",
            "alignment/paper.maps_global.tsv",
            "source_decisions.json",
            "timings.json",
        )
    }
    for relative, data in outputs.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    store.publish(identity, outputs)
    evaluation = stage_identity(
        "evaluation",
        parameters={},
        inputs={},
        role="development",
        entity_kind="all",
        implementation=resume._code_identity(resume.REPOSITORY, evaluation=True),
        dependencies={},
        parents=[identity["artifact_id"]],
    )
    evaluation_outputs = {"evaluation/evaluation_results.json": b'{"builtin":{"MRR":0.8}}'}
    store.publish(evaluation, evaluation_outputs)
    for relative, data in evaluation_outputs.items():
        path = run / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    (run / "recovery-runtime.json").write_text(json.dumps({"identity": identity}))
    worker = {"return_code": 0, "wall_seconds": 10}
    (run / "validation-worker.json").write_text(json.dumps(worker))
    ids = {
        "inputs": inputs["artifact_id"],
        "extraction": identity["artifact_id"],
        "evaluation": evaluation["artifact_id"],
    }
    manifest = {
        "status": "complete",
        "return_code": 0,
        "extraction_complete": True,
        "resolved_config_hash": hash_payload(config),
        "experiment_config_hash": campaign,
        "recovery": {"artifacts": ids},
    }
    (run / "experiment_manifest.json").write_text(json.dumps(manifest))
    local = {
        "id": "local300",
        "status": "complete",
        "source_cap": 300,
        "hosted": True,
        "evaluate": True,
        "new_worker_calls": 1,
        "new_usage": {k: usage[k] - old_usage[k] for k in resume.USAGE_KEYS},
        "output_dir": str(run),
        "manifest": str(run / "experiment_manifest.json"),
        "wall_seconds": 12,
        "worker_measurement": worker,
        "recovery": {"artifacts": ids},
    }
    rows = [*inherited.values(), local]
    for row in rows:
        (current / f"{row['id']}.measurement.json").write_text(json.dumps(row))
    old = json.loads((origin / "report.json").read_text())
    plan = {
        "status": "prepared",
        "campaign_sha256": campaign,
        "resume": {
            "from": str(origin),
            "previous_elapsed_seconds": prior_elapsed,
            "previous_hosted_usage": old["hosted_usage"],
            "copied_hosted_usage": old_usage,
        },
        "limits": {"seconds": None},
    }
    (current / "plan.json").write_text(json.dumps(plan))
    (current / "status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "phase": "local300",
                "elapsed_seconds": prior_elapsed + 14,
                "completed_stages": rows,
            }
        )
    )
    pending = current / "local300-replay/run"
    (pending / "_inputs").mkdir(parents=True)
    (pending / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
    (pending / "recovery-runtime.json").write_text(json.dumps({"identity": identity}))
    (pending / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "status": "running",
                "started_at": "2026-09-18T12:16:40+00:00",
                "resolved_config_hash": hash_payload(config),
                "experiment_config_hash": campaign,
            }
        )
    )
    (pending / "reuse-plan.json").write_text(
        json.dumps(
            {"stages": [{"stage": k, "artifact_id": v, "action": "reuse"} for k, v in ids.items()]}
        )
    )
    return current, expected, campaign, local_key


def test_reportless_adoption_preserves_nine_stages_and_unknown_elapsed_tail(
    interrupted_validation, tmp_path
):
    from tools import validation_resume as resume

    previous, configs, campaign, local_key = interrupted_validation
    before = {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }
    output = tmp_path / "final-replay"
    rows, elapsed, repair = resume.adopt_interrupted_validation(
        previous, output, campaign_sha256=campaign, expected_configs=configs, materialize=False
    )
    assert not output.exists()
    assert len(rows) == 9 and "local300-replay" not in rows and repair == {}
    rows, actual_elapsed, _ = resume.adopt_interrupted_validation(
        previous, output, campaign_sha256=campaign, expected_configs=configs
    )
    assert actual_elapsed == elapsed == 734
    assert (
        rows["local300"]["wall_seconds"] == 12
        and rows["local300"]["measurement_usage"]["attempts"] == 1
    )
    assert rows["global300-stop"]["status"] == "interrupted"
    for row in rows.values():
        evidence = row["adoption_evidence"]
        assert evidence["prior_hosted_usage"]["attempts"] == 3
        assert evidence["prior_ledger_usage"]["attempts"] == 2
        assert evidence["elapsed_accounting"]["status"] == "lower_bound"
        assert evidence["elapsed_accounting"]["final_elapsed_seconds"] is None
        assert evidence["elapsed_accounting"]["unrecorded_interval_seconds"] is None
        assert row["new_worker_calls"] == 0 and not any(row["new_usage"].values())
    assert (
        RequestLedger(output / "shared/openrouter").cached(local_key)
        == b'{"response":"local-scored"}'
    )
    assert not (previous / "report.json").exists()
    after = {
        str(p): sha256_file(p)
        for p in previous.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }
    assert all(after.get(path) == digest for path, digest in before.items())
    # SQLite may create empty WAL bookkeeping even for a read-only backup.
    assert all(
        path.endswith("-wal") and Path(path).stat().st_size == 0
        for path in after.keys() - before.keys()
    )


@pytest.mark.parametrize(
    "change",
    [
        "status_row",
        "local_mapping",
        "replay_plan",
        "replay_identity",
        "replay_worker",
        "elapsed",
        "charges",
        "extra_stage",
        "final_report",
    ],
)
def test_reportless_adoption_rejects_missing_or_changed_evidence(
    interrupted_validation, tmp_path, change
):
    from tools import validation_resume as resume

    previous, configs, campaign, _ = interrupted_validation
    status_path = previous / "status.json"
    status = json.loads(status_path.read_text())
    if change == "status_row":
        status["completed_stages"][-1]["wall_seconds"] += 1
    elif change == "local_mapping":
        (previous / "local300/run/alignment/maps_local.tsv").write_bytes(b"changed")
    elif change == "replay_plan":
        path = previous / "local300-replay/run/reuse-plan.json"
        plan = json.loads(path.read_text())
        plan["stages"][1]["action"] = "recompute"
        path.write_text(json.dumps(plan))
    elif change == "replay_identity":
        path = previous / "local300-replay/run/recovery-runtime.json"
        runtime = json.loads(path.read_text())
        runtime["identity"]["parameters"]["changed"] = True
        path.write_text(json.dumps(runtime))
    elif change == "replay_worker":
        (previous / "local300-replay/run/validation-worker.json").write_text('{"return_code":0}')
    elif change == "elapsed":
        status["elapsed_seconds"] = 720
    elif change == "charges":
        status["completed_stages"][-1]["new_usage"]["attempts"] += 1
        (previous / "local300.measurement.json").write_text(
            json.dumps(status["completed_stages"][-1])
        )
    elif change == "extra_stage":
        status["completed_stages"].append({"id": "local300-replay", "status": "complete"})
    else:
        (previous / "report.json").write_text('{"status":"passed"}')
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError):
        resume.adopt_interrupted_validation(
            previous,
            tmp_path / "new",
            campaign_sha256=campaign,
            expected_configs=configs,
            materialize=False,
        )
