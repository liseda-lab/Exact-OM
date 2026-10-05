"""Manifest recovery binds originals, keeps unknowns and never replays spent rows."""
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import common_inventory as science, common_inventory_payload_recovery as recovery
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound
from tools.repair.expanded_profile import checkpoint
from tools.repair.scaling_payload_recovery import payload_manifest


def fixture(tmp_path, monkeypatch):
    import exact.repair.study
    import tools.repair.batch
    monkeypatch.setattr(exact.repair.study, "runtime_manifest", lambda: {})
    monkeypatch.setattr(science, "validate_schedule", lambda s: None)
    source = str(Path(science.__file__).resolve().parents[2])
    rows = [dict(id=str(i), case_index=0, arm_id="uniform", seconds=300, cpu_seconds=600,
        memory_mb=8192) for i in range(4)]
    schedule = tmp_path / "schedule.json"
    write_artifact(schedule, dict(rows=rows, cases=[dict(status="materialized")],
        arms=[dict(id="uniform",status="available")], followup="evaluation"))
    identity = canonical_hash((sha(schedule), sha(science.__file__), sha(science.fresh.__file__), {}))
    work = tmp_path / "work"; old = work / "inventory"
    argv = ["python", "-m", "tools.repair.common_inventory",str(schedule),str(old),"--start","0","--stop","4"]
    batch = tmp_path / "batch.json"
    write_artifact(batch, dict(code=source, python="python", jobs=[dict(id="job",commands=[argv])]))
    monkeypatch.setattr(tools.repair.batch, "checked_batch", lambda p: read(p))
    refs = []
    for i in range(2):
        path = old / "rows" / f"{i}.json"
        checkpoint(path, canonical_hash((identity, rows[i])), row=rows[i], status="timeout",
            result=None, cleanup_complete=True, payloads=[])
        refs.append(dict(index=i, receipt=binding(path)))
    guard = old / "inflight/2.json"
    checkpoint(guard, canonical_hash((identity, rows[2])), row=rows[2], started_epoch=1)
    result = old / "payloads/2/result.json"
    write_artifact(result, dict(row_id="2",schedule_hash=canonical_hash(read(schedule)),
        status="verification_timeout", logical_status="UNKNOWN", semantic_benefit=None,
        semantic_status="not_evaluated",study_kind="common_inventory_diagnostic",elapsed_seconds=235))
    debris = result.with_name(".search-ledger.json.abcdefgh"); debris.touch(mode=0)
    payloads, excluded = payload_manifest(result.parent)
    attempt = tmp_path / "attempt"; completion = attempt / "completion.json"
    write_artifact(completion,dict(status="failed",exit_code=1,job_id="job",batch=str(batch),
        work=str(work),step_id="14408.397",dispatch_nonce="nonce",
        error=dict(message="command 0 exited 1: PermissionError: [Errno 13] Permission denied: " + repr(str(debris)))))
    owner = attempt / "step.json"; write_artifact(owner,dict(step_id="14408.397",dispatch_nonce="nonce"))
    command = attempt / "command-0.json";write_artifact(command,dict(cwd=source,argv=argv))
    evidence = tmp_path / "evidence.json"
    write_artifact(evidence,dict(completion=binding(completion),ownership=binding(owner),command=binding(command),
        rows=refs,guard=binding(guard),result=binding(result),payloads=payloads,temporary=excluded[0]))
    output = work / "inventory-revision-002";plan = tmp_path / "plan.json"
    write_artifact(plan,dict(original_slice=[0,4],failed_index=2,only_unstarted_slice=[3,4],
        original_row_count=4,prior_costs_reset=False,original_step="14408.397",
        source_root=source,frozen_batch=binding(batch),schedule=binding(schedule),runtime={},
        scientific_identity=identity,runner=binding(recovery.__file__),evidence=binding(evidence),output=str(output)))
    return plan, output, old


def owner(step):
    return dict(step_id=step,surviving_processes=0,cgroup_absent=True,signals_sent=0)


@pytest.mark.parametrize("change", [None,"nonce","command","error","missing_prefix","guard","published",
    "spent_suffix","known_result","changed_schedule","temporary","row_receipt"])
def test_original_evidence_is_fail_closed(tmp_path, monkeypatch, change):
    plan_path, output, old = fixture(tmp_path, monkeypatch)
    plan = read(plan_path); evidence = bound(plan["evidence"])
    if change in ("nonce","command","error"):
        key = {"nonce":"ownership","command":"command","error":"completion"}[change]
        ref = evidence[key]; value = bound(ref)
        if change == "nonce": value["dispatch_nonce"] = "wrong"
        elif change == "command": value["argv"][-1] = "5"
        else: value["error"]["message"] = "other error"
        write_artifact(ref["path"],value); evidence[key] = binding(ref["path"])
    elif change == "missing_prefix": evidence["rows"].pop()
    elif change == "guard":
        write_artifact(evidence["guard"]["path"],dict(identity="changed",row={"id":"wrong"}));evidence["guard"]=binding(evidence["guard"]["path"])
    elif change == "published": write_artifact(old / "payloads/2/extra.json", {})
    elif change == "spent_suffix": (old / "payloads/3").mkdir()
    elif change in ("known_result","changed_schedule"):
        result = bound(evidence["result"])
        result["semantic_benefit" if change == "known_result" else "schedule_hash"] = 1
        write_artifact(evidence["result"]["path"],result);evidence["result"]=binding(evidence["result"]["path"])
    elif change == "temporary": Path(evidence["temporary"]["path"]).chmod(0o600)
    elif change == "row_receipt": write_artifact(old / "rows/2.json", {})
    if change:
        with pytest.raises(ValueError): recovery.validate_original(plan,evidence,plan["scientific_identity"])
    else:
        assert set(recovery.validate_original(plan,evidence,plan["scientific_identity"]))=={0,1}


def test_recovery_keeps_original_unknown_and_only_executes_untouched_suffix(tmp_path, monkeypatch):
    import exact.repair.workers
    from exact.experiments.science_health import inspect_science
    plan, output, old = fixture(tmp_path,monkeypatch)
    original = {p:p.read_bytes() for p in old.rglob("*") if p.is_file() and not p.name.startswith(".")}
    calls=[]
    def call(fn,schedule,row,directory,**limits):
        calls.append((fn,row["id"],limits))
        Path(directory).mkdir(parents=True)
        Path(directory,".search-ledger.json.abcdefgh").touch(mode=0)
        return CallResult("timeout",cleanup_complete=True)
    monkeypatch.setattr(exact.repair.workers,"bounded_call",call)
    first = recovery.run(plan,output,owner_check=owner)
    second = recovery.run(plan,output,owner_check=owner)
    assert first==second and first["recorded"]==first["scheduled"]==4
    assert calls==[(science.evaluate_row,"3",dict(timeout=300,cpu_seconds=600,memory_mb=8192))]
    retained = bound(first["rows"][2])
    assert bound(retained["result"])["logical_status"]=="UNKNOWN"
    assert retained["additional_scientific_seconds"]==0 and not retained["outer_telemetry_available"]
    assert first["scientific_rows_replayed"]==0 and not first["gates_passed"]
    assert (old/"inflight/2.json").exists() and all(p.read_bytes()==raw for p,raw in original.items())
    assert (output/"continuation/calls/3.json").exists()
    attempt = tmp_path/"new-attempt";attempt.mkdir()
    write_artifact(attempt/"outputs.json",{"inventory-revision-002/science-health.json":sha(output/"science-health.json")})
    run=dict(step_id="14408.999",dispatch_nonce="n",completion_path=str(attempt/"completion.json"),
        science_report_relative="inventory-revision-002/science-health.json")
    assert inspect_science(run,dict(status="complete",step_id="14408.999",dispatch_nonce="n",work=str(output.parent)))==dict(failures=[],errors=[])


@pytest.mark.parametrize("change",["live","cgroup","signals","step"])
def test_recovery_rejects_unresolved_owner(tmp_path,monkeypatch,change):
    plan,output,_=fixture(tmp_path,monkeypatch)
    def bad(step):
        value=owner(step)
        value[{"live":"surviving_processes","cgroup":"cgroup_absent","signals":"signals_sent","step":"step_id"}[change]] = {"live":1,"cgroup":False,"signals":1,"step":"14408.0"}[change]
        return value
    with pytest.raises(ValueError,match="ownership|owner cleanup"):
        recovery.run(plan,output,owner_check=bad)


def test_call_receipt_survives_published_artifact_failure_and_denies_replay(tmp_path,monkeypatch):
    import exact.repair.workers
    plan,output,_=fixture(tmp_path,monkeypatch)
    calls=[]
    def call(fn,schedule,row,directory,**limits):
        calls.append(row["id"]);Path(directory).mkdir(parents=True)
        Path(directory,"result.json").touch(mode=0)
        return CallResult("timeout",cleanup_complete=True)
    monkeypatch.setattr(exact.repair.workers,"bounded_call",call)
    with pytest.raises(PermissionError): recovery.run(plan,output,owner_check=owner)
    assert (output/"continuation/calls/3.json").exists() and (output/"continuation/inflight/3.json").exists()
    with pytest.raises(RuntimeError,match="reconciliation"): recovery.run(plan,output,owner_check=owner)
    assert calls==["3"]
