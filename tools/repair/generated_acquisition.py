"""Common generated TRAIN acquisition, independent of fitting and hosted teachers."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.learning import (
    RepairLabel,
    SemanticTargetSpec,
    TeacherCache,
    collect_sampled_repairs,
)
from exact.repair.records import canonical_hash, canonical_json, read_record
from exact.repair.workers import CallResult, bounded_call
from tools.repair.acquisition import record_call, raise_on_software_failure
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checkpoint, checked_checkpoint
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_from_dict, case_to_dict, publish_label_cache, read_label_cache

SCHEMA = "exact-repair/common-generated-acquisition/v1"
QUOTAS = dict(utility=4, proposal=4, diversity=4, quartet=4, uniform=0)


def validate(plan):
    if (
        plan.get("schema") != SCHEMA
        or plan.get("heldout_outcomes_opened") is not False
        or plan.get("model_updates") != 0
        or plan.get("round_id") not in ("0", "1")
        or plan.get("shared_conditions") != ["symbolic", "symbolic_plus_llm"]
        or plan.get("plan_quotas") != QUOTAS
    ):
        raise ValueError("Invalid common TRAIN acquisition contract")
    index = verify_binding(plan["inputs"])
    rows = index["training_rows"]
    for name, split, count in (("training", "train", 128), ("development", "development", 32)):
        declared = index[name + "_rows"]
        release = verify_binding(index[name + "_manifest"])
        if (
            len(declared) != count
            or len({r["case_id"] for r in declared}) != count
            or any(r["split"] != split for r in declared)
            or {canonical_hash(r) for r in declared} != {canonical_hash(r) for r in release["rows"]}
        ):
            raise ValueError("Frozen TRAIN/DEV denominator or split changed")
    if set(r["structural_parent"] for r in rows) & set(
        r["structural_parent"] for r in index["development_rows"]
    ):
        raise ValueError("TRAIN/DEV parent overlap")
    if len(set(plan["case_ids"])) != len(plan["case_ids"]) or not set(plan["case_ids"]) <= {
        r["case_id"] for r in rows
    }:
        raise ValueError("Acquisition may select only frozen TRAIN rows")
    protocol = verify_binding(plan["protocol"])
    model = plan["checkpoint"]
    if sha(model["path"]) != model["sha256"]:
        raise ValueError("Acquisition checkpoint changed")
    selected = []
    by_id = {r["case_id"]: r for r in rows}
    for key in plan["case_ids"]:
        row = by_id[key]
        record = verify_binding(row["evaluator"])
        case = case_from_dict(record)
        if (
            case.split != "train"
            or case.schema_revision != "v3"
            or case.case_id != key
            or case.structural_parent != row["structural_parent"]
            or record["hash"] != row["case_hash"]
            or case.problem.content_hash != row["input_hash"]
        ):
            raise ValueError("Acquisition case identity changed")
        selected.append((row, record))
    return selected, protocol


def initialize(protocol_path, output):
    """Publish protocol-authoritative CPU weights with zero optimizer updates."""
    import torch
    from exact.repair.graph_schema import declared_metadata, generic_graph_schema
    from exact.repair.model import RepairModel

    protocol = read(protocol_path)
    config = protocol["model"]
    schema = generic_graph_schema()
    if canonical_hash(schema) != canonical_hash(config["graph_schema"]):
        raise ValueError("Installed graph declaration differs from protocol")
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = RepairModel(
        declared_metadata(schema),
        graph_schema=schema,
        revision="v3",
        encoder=config["backbone"],
        hidden_dim=config["hidden_width"],
        layers=config["layers"],
        heads=config["attention_heads"],
        dropout=config["dropout"],
        pairwise=config["pair_benefit"],
        plan_risk=config["plan_risk"],
        support_enabled=config["support_enabled"],
        pair_factor_bound=protocol["objective"]["pair_factor_bound"],
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(
            dict(
                metadata=model.metadata,
                config=model.config,
                model_schema="exact-repair/model/v3",
                state_dict=model.state_dict(),
            ),
            stream,
        )
    result = dict(
        checkpoint=binding(output),
        protocol=binding(protocol_path),
        seed=13,
        optimizer_updates=0,
        device="cpu",
        node_types=len(model.metadata[0]),
        relations=len(model.metadata[1]),
        config=model.config,
    )
    write_artifact(output.with_suffix(".json"), result)
    return result


def _pairs(problem, options):
    from exact.repair.graph import EffectivePreparation
    from exact.repair.retrieval import retrieve_vocabulary

    keys = (
        "max_graph_nodes",
        "max_graph_edges",
        "max_explanations",
        "max_text_tokens",
        "pair_factor_limit_per_object",
        "pair_max_pairs",
        "pair_max_factors",
        "retrieval_config",
    )
    prep = EffectivePreparation(**{k: options[k] for k in keys}, revision="v3")
    graph = prep.graph(problem, retrieve_vocabulary(problem, config=prep.retrieval_config))
    return prep.pairs(problem, graph, enabled=True).pairs


def _assignment_payload(case, assignment, profile, *, output_directory, **options):
    """Publish full support evidence before returning a small, authenticated IPC receipt."""
    from tools.repair.train import _assignment_label

    started = time.monotonic()
    label = _assignment_label(case, assignment, profile, **options)
    cache = TeacherCache(
        tuple(len(obj.candidates) for obj in case.problem.objects),
        (label,),
        False,
        "single_assignment_transport",
        (("input", case.problem.content_hash),),
        time.monotonic() - started,
        schema=f"exact-repair/teacher-cache/{case.schema_revision}",
    )
    return publish_label_cache(cache, Path(output_directory))


def _read_assignment_payload(artifact, directory, case, assignment):
    cache = read_label_cache(artifact, directory, case)
    if len(cache.labels) != 1 or cache.labels[0].assignment != tuple(assignment):
        raise ValueError("Assignment artifact belongs to another scheduled assignment")
    return asdict(cache.labels[0])


def case_worker(record, plan, protocol, directory, identity, deadline_epoch):
    from exact.repair.pipeline import bounded_freeze_checkpoint
    from exact.repair.maxsat import solve_master
    from tools.repair.evaluate_campaign import generation_options
    from tools.repair.train import _verify_intended

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    finished = checked_checkpoint(result_path, identity)
    if finished is not None:
        return finished["result"]

    def finish(result):
        checkpoint(result_path, identity, result=result)
        return result

    state_path = directory / "state.json"
    state = checked_checkpoint(state_path, identity) or dict(
        case_deadline_epoch=min(time.time() + plan["resources"]["case_seconds"], deadline_epoch),
        started_epoch=time.time(),
        cpu_seconds=0.0,
        calls=[],
    )
    resources = plan["resources"]
    # Reconcile calls committed between the last state save and interruption.
    committed_calls = [
        checked_checkpoint(p, canonical_hash((identity, p.stem)))
        for p in (directory / "calls").glob("*.json")
        if not p.name.endswith(".started.json")
    ]
    state["cpu_seconds"] = max(
        state["cpu_seconds"],
        sum(r.get("resources", {}).get("cpu_seconds", 0.0) for r in committed_calls),
    )
    case = case_from_dict(record)
    limit = min(state["case_deadline_epoch"], deadline_epoch)

    def save():
        state["elapsed_seconds"] = max(
            state.get("elapsed_seconds", 0), time.time() - state["started_epoch"]
        )
        checkpoint(
            state_path,
            identity,
            **{k: v for k, v in state.items() if k not in ("identity", "content_hash")},
        )

    def left():
        return max(0.0, limit - time.time() - resources["cleanup_seconds"])

    def cpu_left():
        return max(0.0, resources["cpu_seconds"] - state["cpu_seconds"])

    def call(name, function, *args, cap, **kwargs):
        """A durable call marker prohibits replay after unknown delivery/interruption."""
        receipt = directory / "calls" / (name + ".json")
        expected = canonical_hash((identity, name))
        saved = checked_checkpoint(receipt, expected)
        if saved:
            if not saved["cleanup_complete"]:
                raise RuntimeError("Prior acquisition child cleanup incomplete")
            raise_on_software_failure(saved)
            result = CallResult(
                saved["status"],
                detail=saved.get("detail", ""),
                cleanup_complete=saved["cleanup_complete"],
            )
            if result.status == "complete":
                return saved["value"]
            return result
        marker = receipt.with_suffix(".started.json")
        seconds = min(cap, left())
        if marker.exists():
            prior_call = directory / "native" / name / "call.json"
            if prior_call.exists() and not read(prior_call)["cleanup_complete"]:
                raise RuntimeError("Prior acquisition child cleanup incomplete")
            # Do not retransmit an interrupted native/solver call. Charge reservation.
            old = read(marker)
            state["cpu_seconds"] += old["cpu_reservation"]
            result = CallResult(
                "interrupted",
                detail="Prior call has no final receipt",
                resource_usage=(("cpu_seconds", old["cpu_reservation"]),),
            )
        elif seconds <= 0 or cpu_left() <= 0:
            result = CallResult("budget_exhausted")
        else:
            write_artifact(marker, dict(started_epoch=time.time(), cpu_reservation=cpu_left()))
            if function is bounded_freeze_checkpoint:
                result = function(
                    *args,
                    seconds=seconds,
                    cpu_seconds=cpu_left(),
                    memory_mb=resources["memory_mb"],
                    **kwargs,
                )
            else:
                result = bounded_call(
                    function,
                    *args,
                    timeout=seconds,
                    cpu_seconds=cpu_left(),
                    memory_mb=resources["memory_mb"],
                    **kwargs,
                )
            state["cpu_seconds"] += dict(result.resource_usage).get("cpu_seconds", 0.0)
        record_call(
            directory / "native" / name,
            result,
            seconds,
            args[0] if name.startswith("label-") else case,
            args[1] if name.startswith("label-") else None,
        )
        value = None
        if result.status == "complete":
            if name == "generation":
                value = dict(
                    case=case_to_dict(replace(case, problem=result.value.problem)),
                    objective=result.value.objective.to_dict(),
                    reports=[dict(r) for r in result.value.proposal_reports],
                    model_hash=result.value.model_hash,
                )
            elif name.startswith("label-"):
                value = _read_assignment_payload(
                    result.value, directory / "label-artifacts" / name, args[0], args[1]
                )
            elif name.startswith("utility-"):
                value = dict(assignment=result.value.assignment)
            else:
                value = result.value
        value = json.loads(canonical_json(value))
        checkpoint(
            receipt,
            expected,
            status=result.status,
            detail=result.detail,
            cleanup_complete=result.cleanup_complete,
            value=value,
            resources=dict(result.resource_usage),
        )
        state["calls"].append(binding(receipt))
        save()
        return value if result.status == "complete" else result

    save()
    parent = call(
        "parent",
        _verify_intended,
        case,
        cap=resources["initial_parent_seconds"],
        evidence_directory=directory / "native" / "parent-checks",
    )
    if parent is not True:
        return finish(
            dict(
                status="unknown_intended_parent",
                requested=16,
                labels=0,
                attempts=[
                    dict(order=i, status="not_attempted_parent_unqualified") for i in range(16)
                ],
            )
        )
    options = generation_options(protocol, directory / "compiler-cache")
    options["seed"] = plan["seed"]
    options["compile_seconds"] = min(options["compile_seconds"], max(0.001, left()))
    generated = call(
        "generation",
        bounded_freeze_checkpoint,
        case.problem,
        plan["checkpoint"]["path"],
        checkpoint_sha256=plan["checkpoint"]["sha256"],
        cap=resources["generation_seconds"],
        final_candidate_removals=dict(case.final_candidate_removals),
        **options,
    )
    if isinstance(generated, CallResult):
        return finish(
            dict(
                status="generation_" + generated.status,
                requested=16,
                labels=0,
                attempts=[
                    dict(order=i, status="not_attempted_generation_unavailable") for i in range(16)
                ],
            )
        )
    generated_case = case_from_dict(generated["case"])
    generated_path = directory / "generated.json"
    if not generated_path.exists():
        write_artifact(generated_path, generated)
    elif read(generated_path) != generated:
        raise ValueError("Generated pool changed on resume")
    objective = read_record(generated["objective"])
    if "proposed" not in state:
        proposed = []
        for index in range(QUOTAS["utility"]):
            result = call(
                "utility-" + str(index),
                solve_master,
                objective,
                tuple(row[0] for row in proposed),
                cap=min(resources["compile_seconds"], left()),
            )
            if not isinstance(result, CallResult) and result["assignment"] is not None:
                proposed.append(
                    (result["assignment"], "maxsat_initial" if index == 0 else "maxsat_diverse")
                )
        for draw in range(QUOTAS["proposal"]):
            choices, probability = [], 1.0
            for obj, report in zip(generated_case.problem.objects, generated["reports"]):
                samples = report.get("samples", ())
                if draw >= len(samples):
                    break
                sample = samples[draw]
                # Canonical serialization retains the typed v3 record envelope.
                if sample.get("$record") == "ProposalRecordV3":
                    sample = sample["payload"]
                if sample.get("object_id", obj.object_id) != obj.object_id:
                    raise ValueError("Proposal sample belongs to another object")
                ids = [c.candidate_id for c in obj.candidates]
                if sample["candidate_id"] not in ids:
                    break
                choices.append(ids.index(sample["candidate_id"]))
                probability *= math.exp(sample["log_probability"])
            if len(choices) == len(generated_case.problem.objects):
                proposed.append((choices, "proposal", probability))
        state["proposed"] = proposed
        save()
    if "pairs" not in state:
        pairs = call(
            "pairs", _pairs, generated_case.problem, options, cap=resources["compile_seconds"]
        )
        state["pairs"] = [] if isinstance(pairs, CallResult) else pairs
        state["pair_status"] = pairs.status if isinstance(pairs, CallResult) else "complete"
        save()
    profile = tuple(sorted(protocol["objective"]["edit_weights"].items()))
    target = SemanticTargetSpec(
        canonical_hash(case.probes),
        protocol["teacher"]["family_weights"]["desired"],
        protocol["teacher"]["family_weights"]["unwanted"],
    )
    problem = generated_case.problem
    hashes = dict(
        input=problem.content_hash,
        patch=canonical_hash(problem.objects),
        policy=problem.policy.content_hash,
        query=canonical_hash(case.probes),
        inventory=canonical_hash(tuple(o.candidates for o in problem.objects)),
        backend=canonical_hash(
            ("qualified-auto/v1", version("pyhermit"), version("pyelk-reasoner"), "auto")
        ),
        profile=canonical_hash(profile),
        semantic_target=target.content_hash,
    )

    def label(assignment):
        name = "label-" + canonical_hash(assignment)
        result = call(
            name,
            _assignment_payload,
            generated_case,
            assignment,
            profile,
            semantic_target=target,
            output_directory=str(directory / "label-artifacts" / name),
            cap=resources["full_check_seconds"],
            evidence_directory=directory / "native" / (name + "-checks"),
        )
        if isinstance(result, CallResult):
            return RepairLabel(assignment, None, None, 0.0)
        # Use the collector's own deserializer to preserve semantic/support types.
        from exact.repair.learning import ProbeOutcome, SupportTarget

        return RepairLabel(
            tuple(result["assignment"]),
            result["feasible"],
            result["benefit"],
            result["cost"],
            tuple(ProbeOutcome(**r) for r in result["semantic_vector"]),
            tuple(
                SupportTarget(
                    **{
                        **r,
                        "assignment": tuple(r["assignment"]),
                        "occurrence_ids": tuple(tuple(x) for x in r["occurrence_ids"]),
                        "asserted_axioms": tuple(r["asserted_axioms"]),
                        "activation": tuple(r["activation"]),
                    }
                )
                for r in result.get("support_targets", ())
            ),
        )

    def progress(value):
        state["collection_state"] = copy.deepcopy(value)
        save()

    acquired = collect_sampled_repairs(
        tuple(len(o.candidates) for o in problem.objects),
        label,
        case_id=case.case_id,
        parent_group_id=case.structural_parent,
        split="train",
        hashes=hashes,
        model_hash=generated["model_hash"],
        round_id=plan["round_id"],
        max_assignments=16,
        deadline_seconds=left() if cpu_left() > 0 else 0,
        seed=plan["seed"],
        proposed=state["proposed"],
        plan_quotas=QUOTAS,
        eligible_pairs=state["pairs"],
        object_candidate_ids=tuple(
            (o.object_id, tuple(c.candidate_id for c in o.candidates)) for o in problem.objects
        ),
        resume_state=state.get("collection_state"),
        progress=progress,
    )
    artifact = publish_label_cache(acquired.cache, directory / "cache")
    read_label_cache(artifact, directory / "cache", generated_case)
    write_artifact(directory / "collection.json", asdict(acquired))
    save()
    return finish(
        dict(
            status="collected",
            requested=16,
            labels=len(acquired.cache.labels),
            usable_labels=sum(l.usable for l in acquired.cache.labels),
            stop_reason=acquired.stop_reason,
            cache=artifact,
            generated=binding(generated_path),
            collection=binding(directory / "collection.json"),
            pair_status=state["pair_status"],
            elapsed_seconds=state["elapsed_seconds"],
            cpu_seconds=state["cpu_seconds"],
        )
    )


def run(plan_path, output):
    plan = read(plan_path)
    rows, protocol = validate(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    deadline = min(
        plan["resources"]["deadline_epoch"],
        float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", "inf")),
    )
    from exact.repair.study import runtime_manifest

    identity = canonical_hash((sha(plan_path), sha(__file__), runtime_manifest()))
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        for row, record in rows:
            key = canonical_hash((identity, row))
            directory = output / "cases" / canonical_hash(row["case_id"])
            receipt = directory / "completion.json"
            saved = checked_checkpoint(receipt, key)
            if saved is None:
                state = checked_checkpoint(directory / "state.json", key)
                remaining = min(plan["resources"]["case_seconds"], deadline - time.time())
                if state:
                    remaining = min(remaining, state["case_deadline_epoch"] - time.time())
                # A prior expired case can finalize its retained collection without more native work.
                result = (
                    bounded_call(
                        case_worker,
                        record,
                        plan,
                        protocol,
                        str(directory),
                        key,
                        min(deadline, time.time() + max(0.0, remaining)),
                        timeout=max(plan["resources"]["cleanup_seconds"], remaining),
                        cpu_seconds=max(
                            0.01,
                            plan["resources"]["cpu_seconds"] - (state or {}).get("cpu_seconds", 0),
                        ),
                        memory_mb=plan["resources"]["memory_mb"],
                    )
                    if deadline > time.time() + 2
                    else CallResult("stage_deadline")
                )
                if not result.cleanup_complete:
                    raise RuntimeError("Acquisition outer child cleanup incomplete")
                saved = checkpoint(
                    receipt,
                    key,
                    case_id=row["case_id"],
                    family=row["family"],
                    control=row["control"],
                    parent=row["structural_parent"],
                    status=result.status,
                    detail=result.detail,
                    cleanup_complete=result.cleanup_complete,
                    result=result.value,
                    resources=dict(result.resource_usage),
                    scheduled_attempts=16,
                    partial_state=(
                        binding(directory / "state.json")
                        if (directory / "state.json").exists()
                        else None
                    ),
                    native_evidence=[
                        binding(p) for p in sorted((directory / "native").glob("**/*.json"))
                    ],
                )
            for item in saved["native_evidence"]:
                verify_binding(item)
            value = saved.get("result") or {}
            if value.get("cache"):
                generated = verify_binding(value["generated"])
                verify_binding(value["collection"])
                read_label_cache(
                    value["cache"], directory / "cache", case_from_dict(generated["case"])
                )
            raise_on_software_failure(saved)
            completed.append(binding(receipt))
            write_artifact(
                output / "progress.json",
                dict(completed_rows=len(completed), expected_rows=len(rows)),
            )
        return checkpoint(
            output / "report.json",
            identity,
            schema=SCHEMA,
            status="complete",
            plan=binding(plan_path),
            expected_rows=len(rows),
            rows=completed,
            denominator_preserved=True,
            training_denominator=128,
            development_denominator=32,
            heldout_outcomes_opened=False,
            hosted_calls=0,
            optimizer_updates=0,
            shared_conditions=plan["shared_conditions"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("protocol", type=Path)
    init.add_argument("output", type=Path)
    collect = commands.add_parser("run")
    collect.add_argument("plan", type=Path)
    collect.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize(args.protocol, args.output)
    else:
        run(args.plan, args.output)


if __name__ == "__main__":
    main()
