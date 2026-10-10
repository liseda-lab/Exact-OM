"""Real serialized transport, preserved validation, capability and budget admission."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from exact.repair import semantic_fidelity as fidelity
from exact.repair.annotation_controls import (
    response_format,
    controlled_completion,
    input_token_bound,
)
from exact.repair.records import canonical_hash, canonical_json, read_record
from tests.repair_semantic_fidelity_test import adapter, packet, judgment, run
from tests.repair_semantic_interface_test import revised
from tests.repair_october_learning_annotation_runner_test import manifest
from tools.repair import corrective_semantics as runner
from tools.repair.annotation_profile import request_profile

REAL_DISPATCH = fidelity.SemanticAnnotationAdapter._dispatch


def wire_client(tmp_path, monkeypatch, effort):
    client, calls = adapter(tmp_path, monkeypatch)
    for profile in client.router.profiles.values():
        profile.provider.update(allow_fallbacks=False, require_parameters=True)
    monkeypatch.setattr(fidelity.SemanticAnnotationAdapter, "_dispatch", REAL_DISPATCH)
    monkeypatch.setattr(fidelity, "OpenRouterClient", lambda: client.router.hosted)
    monkeypatch.setattr(client.router.hosted, "close", lambda: None)
    monkeypatch.setattr(
        fidelity,
        "bounded_call",
        lambda fn, *args, **kw: SimpleNamespace(status="complete", value=fn(*args)),
    )
    args = {f.name: getattr(run(), f.name) for f in dataclasses.fields(run())}
    args.update(
        max_output_tokens=6000,
        role_request_controls={
            fidelity.TEACHER: dict(
                **({"reasoning": effort} if effort is not None else {}),
                response_format=response_format(packet()),
            )
        },
    )
    args.update(
        evaluator_split="test",
        development_use_policy="independent_evaluation",
        selection_model_ids=(),
    )
    settings = fidelity.ControlledSelectionAnnotationRunV1(**args)
    return fidelity.SemanticAnnotationAdapter(client.router, settings, tmp_path), calls


@pytest.mark.parametrize("effort", [{"effort": "low"}, {"enabled": False}, None])
def test_actual_worker_serializes_controls_and_replays_without_payment(
    tmp_path, monkeypatch, effort
):
    client, calls = wire_client(tmp_path, monkeypatch, effort)
    label = client.annotate(packet(), role=fidelity.TEACHER)
    assert len(calls) == 1
    payload = json.loads(calls[0]["content"])
    assert payload.get("reasoning") == effort
    if effort is None:
        assert "reasoning" not in payload
    assert payload["response_format"] == response_format(packet())
    assert payload["max_tokens"] == 6000
    assert payload["provider"]["require_parameters"] is True
    wire = label.annotator["wire_receipt"]
    assert json.loads(canonical_json(wire["identity"]["payload"])) == payload
    assert label.annotator["request_parameters"].get("reasoning") == effort
    assert client.annotate(packet(), role=fidelity.TEACHER) == label
    assert len(calls) == 1
    with client.ledger._transaction() as db:
        row = db.execute("SELECT tokens FROM repair_annotation_reserves").fetchone()
    assert (
        row["tokens"]
        == input_token_bound(
            payload["messages"],
            {k: payload[k] for k in ("reasoning", "response_format") if k in payload},
        )
        + 6000
    )
    assert read_record(client.run.to_dict()) == client.run


def test_controls_affect_identity_and_receipts_cannot_drop_them(tmp_path, monkeypatch):
    client, calls = wire_client(tmp_path, monkeypatch, {"effort": "low"})
    first = client.annotate(packet(), role=fidelity.TEACHER)
    updated = dataclasses.replace(
        client.run,
        role_request_controls={
            fidelity.TEACHER: dict(
                reasoning={"enabled": False}, response_format=response_format(packet())
            )
        },
    )
    second = fidelity.SemanticAnnotationAdapter(client.router, updated, tmp_path).annotate(
        packet(), role=fidelity.TEACHER
    )
    assert len(calls) == 2
    assert first.annotator["parameters_hash"] != second.annotator["parameters_hash"]
    params = json.loads(canonical_json(second.annotator["request_parameters"]))
    del params["reasoning"]
    with pytest.raises(ValueError, match="differs"):
        fidelity._receipt_response(second.annotator["wire_receipt"], params)


def save(tmp_path, name, value):
    path = tmp_path / name
    runner.write_artifact(path, value)
    return dict(path=str(path), sha256=runner.sha(path))


def profile_fixture(tmp_path):
    settings = revised(manifest(tmp_path))
    settings["profiles"]["teacher"]["provider"].update(
        allow_fallbacks=False, require_parameters=True, max_price=dict(prompt=1, completion=2)
    )
    # A tiny local tokenizer is enough to exercise the pinned-tokenizer path.
    from tokenizers import Tokenizer, models

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    controls = dict(
        schema_version="semantic-fidelity-response/v1",
        max_output_tokens=6000,
        reasoning={"effort": "low"},
        endpoint=save(
            tmp_path,
            "endpoint.json",
            dict(
                data=dict(
                    id="vendor/teacher",
                    endpoints=[
                        dict(
                            provider_name="pinned",
                            status=0,
                            max_completion_tokens=10000,
                            pricing=dict(prompt="0.000001", completion="0.000002"),
                            supported_parameters=[
                                "reasoning",
                                "response_format",
                                "structured_outputs",
                                "max_tokens",
                                "temperature",
                            ],
                        )
                    ],
                )
            ),
        ),
        catalog=save(
            tmp_path,
            "catalog.json",
            dict(
                data=[
                    dict(
                        id="vendor/teacher",
                        hugging_face_id="test/model",
                        reasoning=dict(mandatory=True, supported_efforts=["low"]),
                    )
                ]
            ),
        ),
        tokenizer=dict(
            path=str(path), sha256=runner.sha(path), repository="test/model", revision="a" * 40
        ),
    )
    settings["request_profiles"] = {"teacher": controls}
    return settings


def test_entrypoint_accounts_6000_outputs_and_binds_schema(tmp_path, monkeypatch):
    import exact.llm.routing

    client, calls = wire_client(tmp_path / "ledger", monkeypatch, {"effort": "low"})
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = profile_fixture(tmp_path)
    # Panel policy has separate fail-closed tests; this test isolates the real
    # annotation entry point, HTTP serialization and both reservation ledgers.
    settings["teacher_panel"] = {"test_only": True}
    monkeypatch.setattr(runner, "_request_budget", lambda manifest: (runner.PHASE_LIMITS, None))
    result = runner.annotate_packet(packet(), settings, tmp_path / "out", slot_id="s", seconds=300)
    assert result is not None and len(calls) == 1
    payload = json.loads(calls[0]["content"])
    assert payload["max_tokens"] == 6000 and payload["reasoning"] == {"effort": "low"}
    state = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert next(iter(state["reservations"].values()))["reserved_cost_usd"] == pytest.approx(0.020)
    assert (
        runner.annotate_packet(packet(), settings, tmp_path / "out", slot_id="s", seconds=1)
        == result
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "defect", ["schema_support", "mandatory", "effort", "fallback", "price", "tamper"]
)
def test_unsupported_controls_rejected_without_any_network(tmp_path, defect):
    settings = profile_fixture(tmp_path)
    controls = settings["request_profiles"]["teacher"]
    if defect == "schema_support":
        data = runner.read(controls["endpoint"]["path"])
        data["data"]["endpoints"][0]["supported_parameters"].remove("structured_outputs")
        controls["endpoint"] = save(tmp_path, "endpoint.json", data)
    elif defect == "mandatory":
        controls["reasoning"] = {"enabled": False}
    elif defect == "effort":
        controls["reasoning"] = {"effort": "max"}
    elif defect == "fallback":
        settings["profiles"]["teacher"]["provider"]["allow_fallbacks"] = True
    elif defect == "price":
        settings["profiles"]["teacher"]["provider"]["max_price"]["completion"] = 99
    elif defect == "tamper":
        controls["tokenizer"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        request_profile(settings, "teacher")
    assert not (tmp_path / "ledger").exists()


def test_json_object_is_not_a_schema_and_validation_stays_strict(tmp_path, monkeypatch):
    client, _ = wire_client(tmp_path, monkeypatch, {"effort": "low"})
    with pytest.raises(ValueError, match="strict JSON schema"):
        controlled_completion(
            client.router.hosted,
            client.profiles[fidelity.TEACHER],
            [],
            2000,
            fidelity.TEACHER,
            {"reasoning": {"effort": "low"}, "response_format": {"type": "json_object"}},
        )
    value = judgment(packet())
    value["criteria"][0]["symbolic_claims"][0]["value"] = "false"
    with pytest.raises(ValueError, match="unsupported symbolic"):
        fidelity.validate_comparison(json.dumps(value), packet())


def panel_fixture(tmp_path, monkeypatch):
    from tests.repair_october_learning_annotation_runner_test import _approved_replacement_fixture
    from tools.repair import teacher_panel as panel

    old, original, _, _ = _approved_replacement_fixture(tmp_path, monkeypatch)
    state = {
        **original,
        "request_limits": runner.REALLOCATED_LIMITS,
        "request_budget_amendment": old["request_budget_amendment"],
        "reservations": dict(original["reservations"]),
    }
    for i in range(27):
        state["reservations"][f"later-{i}"] = dict(
            phase="calibration",
            slot=f"later-{i}",
            packet_hash="old",
            comparison_id=f"original-{i}",
            reserved_cost_usd=0.01,
            state="unresolved",
        )
    runner.write_artifact(tmp_path / "ledger/phase-reservations.json", state)
    started = runner.time.time() - 100
    stage = save(
        tmp_path,
        "stage.json",
        dict(
            stages={"calibration": dict(started_epoch=started)},
            stage_limits={"calibration": dict(elapsed_seconds=21600)},
        ),
    )
    packets = [save(tmp_path, f"p{i}.json", packet().to_dict()) for i in range(4)]
    settings = {
        **old,
        "profile": "teacher",
        "request_limits": dict(panel.LIMITS),
        "slots": [
            dict(
                id=f"new-{i}-{int(order)}",
                comparison_id=f"panel-{i}",
                profile="teacher",
                packet=p,
                swapped=order,
            )
            for i, p in enumerate(packets)
            for order in (False, True)
        ],
        "seconds": 900,
        "deadline_epoch": started + 21600,
    }
    proposal = dict(
        schema="exact-repair/teacher-panel-proposal/v1",
        request_limits=dict(panel.LIMITS),
        previous_phase_state=save(tmp_path, "panel-prior.json", state),
        previous_manifest=save(tmp_path, "panel-previous.json", old),
        prior_requests=59,
        prior_reserved_usd=sum(r["reserved_cost_usd"] for r in state["reservations"].values()),
        max_requests=24,
        max_reserved_usd=0.11376,
        calibration_comparison_identity_limit=48,
        packets=packets,
        contracts={"teacher": canonical_hash(panel.contract(settings))},
        stage=dict(
            ledger_path=stage["path"],
            started_epoch=started,
            original_elapsed_seconds=14400,
            proposed_elapsed_seconds=21600,
            deadline_epoch=started + 21600,
            latest_admission_epoch=started + 21600 - 900 / 0.7,
        ),
    )
    binding = save(tmp_path, "panel-proposal.json", proposal)
    approval = save(
        tmp_path,
        "panel-approval.json",
        dict(
            schema="exact-repair/teacher-panel-authorization/v1",
            status="approved",
            approval_text="approve exact fixture",
            proposal=binding,
        ),
    )
    settings["teacher_panel"] = dict(proposal=binding, authorization=approval)
    return settings, state


def test_panel_atomic_reallocation_preserves59_costs_and_resume(tmp_path, monkeypatch):
    settings, prior = panel_fixture(tmp_path, monkeypatch)
    assert (
        runner._reserve_phase(
            settings, "new-0-0", packet().content_hash, 0.0042, comparison_id="panel-0"
        )
        is None
    )
    after = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert after["request_limits"] == dict(calibration=83, train=333, development=96, test=192)
    assert len(after["reservations"]) == 60
    assert all(after["reservations"][k] == v for k, v in prior["reservations"].items())
    assert (
        runner._reserve_phase(
            settings, "new-0-0", packet().content_hash, 0.0042, comparison_id="panel-0"
        )
        is False
    )
    assert runner.read(tmp_path / "ledger/phase-reservations.json") == after


@pytest.mark.parametrize(
    "defect", [None, "unapproved", "clock_reset", "extra_time", "unbound", "ledger", "expired"]
)
def test_approved_panel_window_preserves_costs_and_original_clock(tmp_path, monkeypatch, defect):
    from tools.repair import teacher_panel as panel

    settings, prior = panel_fixture(tmp_path, monkeypatch)
    proposal = runner.read(settings["teacher_panel"]["proposal"]["path"])
    now = panel.time.time()
    started = now - 50000
    old = json.loads(json.dumps(proposal))
    old["stage"].update(
        started_epoch=started, deadline_epoch=started + 21600,
        latest_admission_epoch=started + 21600 - 900 / 0.7,
    )
    timing = dict(
        schema="exact-repair/teacher-panel-timing-authorization/v1",
        status="proposed" if defect == "unapproved" else "approved",
        approval_text="I authorise please run this experiments",
        previous_proposal=save(tmp_path, "expired-proposal.json", old),
        approved_epoch=now - 1,
        window_seconds=7201 if defect == "extra_time" else 7200,
    )
    deadline = timing["approved_epoch"] + timing["window_seconds"]
    proposal["stage"].update(
        started_epoch=started + (1 if defect == "clock_reset" else 0),
        deadline_epoch=deadline, proposed_elapsed_seconds=deadline - started,
        latest_admission_epoch=deadline - 900 / 0.7,
    )
    proposal["timing_authorization"] = save(tmp_path, "timing.json", timing)
    settings["deadline_epoch"] = deadline
    proposal["contracts"]["teacher"] = canonical_hash(panel.contract(settings))
    settings["teacher_panel"]["proposal"] = save(tmp_path, "successor-proposal.json", proposal)
    approval = runner.read(settings["teacher_panel"]["authorization"]["path"])
    approval.update(proposal=settings["teacher_panel"]["proposal"])
    if defect != "unbound":
        approval["timing_authorization"] = proposal["timing_authorization"]
    settings["teacher_panel"]["authorization"] = save(tmp_path, "successor-approval.json", approval)
    stage = runner.read(tmp_path / "stage.json")
    stage["stages"]["calibration"]["started_epoch"] = started
    stage["stage_limits"]["calibration"]["elapsed_seconds"] = (
        21600 if defect == "ledger" else deadline - started
    )
    runner.write_artifact(tmp_path / "stage.json", stage)
    if defect == "expired":
        monkeypatch.setattr(panel.time, "time", lambda: deadline)
    before = (tmp_path / "ledger/phase-reservations.json").read_bytes()
    if defect:
        with pytest.raises((ValueError, PermissionError)):
            runner._reserve_phase(
                settings, "new-0-0", packet().content_hash, 0.0042, comparison_id="panel-0"
            )
        assert (tmp_path / "ledger/phase-reservations.json").read_bytes() == before
        return
    runner._reserve_phase(settings, "new-0-0", packet().content_hash, 0.0042, comparison_id="panel-0")
    after = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert len(after["reservations"]) == 60
    assert all(after["reservations"][k] == v for k, v in prior["reservations"].items())
    assert runner.read(tmp_path / "stage.json") == stage
    assert runner._reserve_phase(
        settings, "new-0-0", packet().content_hash, 0.0042, comparison_id="panel-0"
    ) is False
    assert runner.read(tmp_path / "ledger/phase-reservations.json") == after


@pytest.mark.parametrize(
    "defect", ["unapproved", "quota", "contract", "reset", "stage", "late", "unscheduled"]
)
def test_panel_refuses_unapproved_changed_or_expired_work(tmp_path, monkeypatch, defect):
    from tools.repair import teacher_panel as panel

    settings, prior = panel_fixture(tmp_path, monkeypatch)
    slot = "new-0-0"
    if defect == "unapproved":
        ref = settings["teacher_panel"]["authorization"]
        approval = runner.read(ref["path"])
        approval["status"] = "proposed"
        settings["teacher_panel"]["authorization"] = save(tmp_path, "panel-approval.json", approval)
    elif defect == "quota":
        settings["request_limits"]["train"] += 1
    elif defect == "contract":
        settings["seconds"] = 901
    elif defect == "reset":
        changed = json.loads(json.dumps(prior))
        changed["reservations"].pop(next(iter(changed["reservations"])))
        runner.write_artifact(tmp_path / "ledger/phase-reservations.json", changed)
    elif defect == "stage":
        stage = runner.read(tmp_path / "stage.json")
        stage["stages"]["calibration"]["started_epoch"] += 1
        runner.write_artifact(tmp_path / "stage.json", stage)
    elif defect == "late":
        monkeypatch.setattr(
            panel.time,
            "time",
            lambda: runner.read(tmp_path / "panel-proposal.json")["stage"]["deadline_epoch"],
        )
    elif defect == "unscheduled":
        slot = "unscheduled"
    before = (tmp_path / "ledger/phase-reservations.json").read_bytes()
    with pytest.raises((ValueError, PermissionError)):
        runner._reserve_phase(
            settings, slot, packet().content_hash, 0.0042, comparison_id="panel-0"
        )
    assert (tmp_path / "ledger/phase-reservations.json").read_bytes() == before


def test_response_schema_enforces_structure_without_replacing_evidence_validation():
    import jsonschema

    form = response_format(packet())["json_schema"]["schema"]
    jsonschema.Draft202012Validator.check_schema(form)
    value = judgment(packet())
    jsonschema.validate(value, form)
    value["criteria"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, form)


def test_new_panel_denominator_and_deferred_selection_are_explicit(tmp_path):
    from tools.repair.corrective_calibration import summarize

    settings = manifest(tmp_path)
    settings.update(
        calibration_profiles=["teacher"], defer_teacher_selection=True, slots=[], gold={}
    )
    path = tmp_path / "cal.json"
    runner.write_artifact(path, settings)
    gate = summarize(path, tmp_path / "out")
    assert gate["selected_profile"] is None and gate["selection_deferred"]
    assert gate["metrics"]["teacher"] == dict(
        valid=0, correct=0, order_consistent=0, eligible=False
    )


@pytest.mark.parametrize(
    "defect", [None, "valid_replay", "changed_order", "second_retry", "raw_tamper"]
)
def test_glm_requires_source_bound_first_truncation(tmp_path, monkeypatch, defect):
    from tools.repair import teacher_panel as panel
    from exact.llm.ledger import request_identity
    import hashlib

    settings, _ = panel_fixture(tmp_path, monkeypatch)
    settings["profile"] = "glm"
    settings["profiles"]["glm"] = settings["profiles"]["teacher"]
    proposal = runner.read(settings["teacher_panel"]["proposal"]["path"])
    prior = runner.read(proposal["previous_phase_state"]["path"])
    lineage = {}
    for index, slot in enumerate(settings["slots"]):
        slot["profile"] = "glm"
        slot["comparison_id"] = f"original-{index}"
        old = prior["reservations"][canonical_hash(("calibration", f"original-{index}"))]
        old["comparison_id"] = slot["comparison_id"]
        context = dict(
            packet=dict(context=dict(packet_hash=packet().content_hash)), swapped=slot["swapped"]
        )
        payload = dict(
            model="vendor/teacher",
            provider=dict(only=["pinned"]),
            messages=[
                dict(role="system", content="old"),
                dict(role="user", content=json.dumps(context)),
            ],
        )
        identity = dict(payload=payload)
        raw = json.dumps(
            dict(model="vendor/teacher", provider="pinned", choices=[dict(finish_reason="length")])
        )
        wire = dict(
            identity=identity,
            request_id=request_identity(identity)[0],
            raw_response=raw,
            sha256=hashlib.sha256(raw.encode()).hexdigest(),
        )
        if index == 0 and defect == "valid_replay":
            wire["raw_response"] = raw.replace("length", "stop")
            wire["sha256"] = hashlib.sha256(wire["raw_response"].encode()).hexdigest()
        if index == 0 and defect == "raw_tamper":
            wire["sha256"] = "0" * 64
        lineage[slot["id"]] = dict(
            old_slot=f"original-{index}",
            same_cause_unsuccessful_attempts=2 if index == 0 and defect == "second_retry" else 1,
            receipt=save(tmp_path, f"r{index}.json", dict(status="annotation_unavailable")),
            wire=save(tmp_path, f"w{index}.json", wire),
        )
    if defect == "changed_order":
        settings["slots"][0]["swapped"] = True
    proposal["previous_phase_state"] = save(tmp_path, "panel-prior.json", prior)
    proposal["glm_recovery"] = lineage
    proposal["contracts"] = {"glm": canonical_hash(panel.contract(settings))}
    settings["teacher_panel"]["proposal"] = save(tmp_path, "panel-proposal.json", proposal)
    approval = runner.read(settings["teacher_panel"]["authorization"]["path"])
    approval["proposal"] = settings["teacher_panel"]["proposal"]
    settings["teacher_panel"]["authorization"] = save(tmp_path, "panel-approval.json", approval)
    if defect:
        with pytest.raises(ValueError):
            panel.panel_contract(settings)
    else:
        assert panel.panel_contract(settings)[0]["glm_recovery"] == lineage


def test_schema_only_identity_is_distinct_from_disabled_reasoning(tmp_path, monkeypatch):
    client, calls = wire_client(tmp_path, monkeypatch, None)
    first = client.annotate(packet(), role=fidelity.TEACHER)
    changed = dataclasses.replace(
        client.run,
        role_request_controls={
            fidelity.TEACHER: dict(
                reasoning={"enabled": False}, response_format=response_format(packet())
            )
        },
    )
    second = fidelity.SemanticAnnotationAdapter(client.router, changed, tmp_path).annotate(
        packet(), role=fidelity.TEACHER
    )
    assert (
        len(calls) == 2
        and first.annotator["parameters_hash"] != second.annotator["parameters_hash"]
    )
    assert "reasoning" not in json.loads(calls[0]["content"])
    assert json.loads(calls[1]["content"])["reasoning"] == {"enabled": False}


def test_nonreasoning_endpoint_rejects_reasoning_controls(tmp_path):
    settings = profile_fixture(tmp_path)
    controls = settings["request_profiles"]["teacher"]
    endpoint = runner.read(controls["endpoint"]["path"])
    endpoint["data"]["endpoints"][0]["supported_parameters"].remove("reasoning")
    controls["endpoint"] = save(tmp_path, "endpoint.json", endpoint)
    with pytest.raises(ValueError, match="lacks required"):
        request_profile(settings, "teacher")


def test_four_model_amendment_retains_total_and_prior_attempts(tmp_path, monkeypatch):
    from tools.repair import teacher_panel as panel

    settings, prior = panel_fixture(tmp_path, monkeypatch)
    settings["request_limits"] = dict(panel.FOUR_MODEL_LIMITS)
    proposal = runner.read(settings["teacher_panel"]["proposal"]["path"])
    proposal.update(
        request_limits=dict(panel.FOUR_MODEL_LIMITS),
        max_requests=32,
        max_reserved_usd=0.13296,
        calibration_comparison_identity_limit=56,
    )
    proposal["contracts"] = {"teacher": canonical_hash(panel.contract(settings))}
    settings["teacher_panel"]["proposal"] = save(tmp_path, "four-proposal.json", proposal)
    approval = runner.read(settings["teacher_panel"]["authorization"]["path"])
    approval["proposal"] = settings["teacher_panel"]["proposal"]
    settings["teacher_panel"]["authorization"] = save(tmp_path, "four-approval.json", approval)
    assert (
        runner._reserve_phase(
            settings, "new-0-0", packet().content_hash, 0.0024, comparison_id="panel-0"
        )
        is None
    )
    after = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert after["request_limits"] == dict(calibration=91, train=325, development=96, test=192)
    assert sum(after["request_limits"].values()) == 704 and all(
        after["reservations"][k] == v for k, v in prior["reservations"].items()
    )


def test_nonreasoning_profile_entrypoint_serializes_schema_only(tmp_path, monkeypatch):
    import exact.llm.routing
    from tests.repair_semantic_fidelity_test import response

    client, calls = wire_client(tmp_path / "ledger", monkeypatch, None)
    model = "openai/gpt-4o-mini-2024-07-18"
    client.router.profiles["teacher"].model = model

    def transport(**kwargs):
        calls.append(kwargs)
        return response(judgment(packet()), model=model)

    monkeypatch.setattr(client.router.hosted._client, "request", transport)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = profile_fixture(tmp_path)
    settings["profiles"]["teacher"]["model"] = model
    controls = settings["request_profiles"]["teacher"]
    controls.update(reasoning=None, max_output_tokens=2000)
    endpoint = runner.read(controls["endpoint"]["path"])
    endpoint["data"]["id"] = model
    endpoint["data"]["endpoints"][0]["supported_parameters"].remove("reasoning")
    controls["endpoint"] = save(tmp_path, "endpoint.json", endpoint)
    controls["catalog"] = save(tmp_path, "catalog.json", dict(data=[dict(id=model)]))
    controls["tokenizer"].update(
        kind="openai_o200k_base",
        repository="openai/tiktoken",
        encoding="o200k_base",
        source=controls["endpoint"],
        ordinary_token_ids_equal=True,
    )
    settings["teacher_panel"] = {"test_only": True}
    monkeypatch.setattr(runner, "_request_budget", lambda manifest: (runner.PHASE_LIMITS, None))
    result = runner.annotate_packet(packet(), settings, tmp_path / "out", slot_id="s", seconds=300)
    assert result is not None
    payload = json.loads(calls[0]["content"])
    assert (
        payload["model"] == model and payload["max_tokens"] == 2000 and "reasoning" not in payload
    )
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert (
        runner.annotate_packet(packet(), settings, tmp_path / "out", slot_id="s", seconds=1)
        == result
    )
    assert len(calls) == 1
