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
