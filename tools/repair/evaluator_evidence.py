"""Theory-bound auxiliary DEV checks; never change training probes or repair scores."""

from dataclasses import replace
import json
from pathlib import Path
import time

from exact.repair.records import canonical_hash, canonical_json, read_record
from tools.repair.batch import read
from tools.repair.shared_release import immutable

REVISION = "exact-repair/intended-commitment-evidence/v1"
TASK = (
    "Assess retained meaning within the authored D1/D2/D3 setting using the complete "
    "theories and checked facts. A supported equivalence is a tie with grounded equal "
    "scores. Abstain only when necessary evidence is missing, unknown or ambiguous. "
    "A completed false entailment is evidence of non-retention, not missing evidence. "
    "Logical feasibility and semantic fidelity are separate."
)


def request_reallocation(manifest):
    """Transfer unused TRAIN attempts to DEV; never borrow from TEST or reset costs."""
    from collections import Counter
    from tools.repair.shared_release import bound

    amendment = bound(manifest["evaluator_request_amendment"])
    prior = bound(amendment["phase_before"])
    previous = bound(amendment["previous_manifest"])
    old, new = prior["request_limits"], manifest["request_limits"]
    spent = Counter(r["phase"] for r in prior["reservations"].values())
    if not (
        amendment.get("schema") == "exact-repair/dev-evaluator-request-reallocation/v1"
        and amendment.get("authorized") is True
        and amendment.get("user_instruction")
        and amendment.get("costs_reset") is False
        and amendment.get("old_limits") == old == previous["request_limits"]
        and amendment.get("new_limits") == new
        and set(new) == set(old) == {"calibration", "train", "development", "test"}
        and all(type(v) is int and v >= spent[k] for k, v in new.items())
        and new["calibration"] == old["calibration"]
        and new["test"] == old["test"]
        and new["train"] < old["train"]
        and new["development"] > old["development"]
        and sum(new.values()) == sum(old.values()) == 704
        and prior["lineage"] == previous["lineage_id"] == manifest["lineage_id"]
        and prior["cost_ceiling_usd"]
        == previous["cost_ceiling_usd"]
        == manifest["cost_ceiling_usd"]
        == 35
        and previous["ledger_directory"] == manifest["ledger_directory"]
        and manifest["phase"] in {"development", "test"}
    ):
        raise ValueError("Invalid cumulative DEV evaluator request reallocation")
    return amendment, prior, previous


def original_request_limits(manifest):
    """Authenticate each cumulative amendment back to the original phase quotas."""
    seen = set()
    while manifest.get("evaluator_request_amendment"):
        identity = canonical_hash(manifest["evaluator_request_amendment"])
        if identity in seen:
            raise ValueError("Cyclic evaluator quota amendments")
        seen.add(identity)
        _, _, manifest = request_reallocation(manifest)
    return manifest["request_limits"]


def intentions(case):
    import pyowl_core as owl

    fixed = set(case.problem.fixed_axioms)
    if (
        case.split != "development"
        or case.origin != "generated"
        or not fixed <= set(case.intended_theory)
    ):
        raise ValueError("Auxiliary intention evidence requires the frozen generated DEV case")
    queries = tuple(
        sorted(
            (
                a
                for a in set(case.intended_theory) - fixed
                if isinstance(a, owl.LOGICAL_AXIOM_TYPES)
            ),
            key=owl.canonical_bytes,
        )
    )
    if not queries:
        raise ValueError("Missing authored logical commitments")
    return queries


def _check(case_record, assignment, plan_record):
    """Run once inside a bounded worker and retain the complete native report."""
    from exact.repair.kernel import materialize
    from exact.repair.learning import TeacherProbe
    from exact.repair.owl import (
        OwlVerifier,
        _query_id,
        _declarations,
        snapshot_from_axioms,
        qualified_routes,
    )
    from tools.repair.prepare import case_from_dict
    from tools.repair.grounded_supervision import render

    case, plan = case_from_dict(case_record), read_record(plan_record)
    theory, active = materialize(case.problem, tuple(assignment))
    selected = tuple(
        (obj.object_id, obj.candidates[i].candidate_id)
        for obj, i in zip(case.problem.objects, assignment)
    )
    if (
        not plan.eligible
        or plan.complete_assignment != selected
        or plan.theory_hash != canonical_hash((theory, active))
        or plan.policy_hash != case.problem.policy.content_hash
        or plan.content[0] != "Complete reconstructed theory:\n" + render(theory)
    ):
        raise ValueError("Intention checks do not match the saved complete theory")
    queries = intentions(case)
    probes = tuple(TeacherProbe("intended-%02d" % i, q, "intended") for i, q in enumerate(queries))
    expressions = tuple(dict.fromkeys(e for p in probes for e in p.conditions()))
    # Deleted entities may occur only in an intended query. DL declarations are
    # semantically inert; retain their exact identities for the backend's query
    # signature check, without asserting any intended logical commitment.
    query_declarations = tuple(sorted(_declarations(queries), key=canonical_hash))
    snapshot = snapshot_from_axioms((*theory, *query_declarations))
    report = None
    for route in qualified_routes(snapshot, queries, expressions):
        report = OwlVerifier(route, backend="auto").check_theory(
            snapshot, (), required=queries, activated=expressions
        )
        # A decisive negative feasibility answer can legitimately be partial.
        # Auxiliary annotation facts instead require every query to be complete.
        if report.complete:
            break
    if report is None:
        raise ValueError("No qualified route for auxiliary evidence")
    checks = {(o.kind, o.query_id): o for o in report.obligations}
    if not report.complete or not checks[("consistency", "consistency")].verdict:
        raise ValueError("Auxiliary checks lack complete consistent native evidence")
    rows = []
    for probe in probes:
        result = checks[("required_entailment", _query_id(probe.axiom))]
        nonvacuity = [checks[("active_satisfiability", _query_id(e))] for e in probe.conditions()]
        if not result.complete or any(not r.complete for r in nonvacuity):
            raise ValueError("Auxiliary query coverage is incomplete")
        rows.append(
            dict(
                query_id=probe.probe_id,
                query=render((probe.axiom,)),
                entailed=result.verdict,
                nonvacuous=all(r.verdict for r in nonvacuity),
                nonvacuity_query_ids=[r.query_id for r in nonvacuity],
            )
        )
    return dict(
        schema=REVISION,
        plan_hash=plan.content_hash,
        theory_hash=plan.theory_hash,
        policy_hash=plan.policy_hash,
        intended_hash=canonical_hash(queries),
        query_signature_declarations=render(query_declarations),
        report=report.to_dict(),
        rows=rows,
    )


def augment(packet, case_record, assignments, directory, *, seconds, memory_mb=8192):
    """Cache only exact dependency matches. Auxiliary facts are never selection scores."""
    from exact.repair.workers import bounded_call
    from tools.repair.prepare import case_from_dict
    from tools.repair.grounded_supervision import render

    case = case_from_dict(case_record)
    queries = intentions(case)
    expected_d2 = (
        "Intended editable commitments, together with the unchanged full fixed context:\n"
        + render(a for a in case.intended_theory if a not in set(case.problem.fixed_axioms))
    )
    if packet.evidence["D2"]["text"] != expected_d2:
        raise ValueError("Authored evidence differs from the frozen intended construction")
    evidence = {}
    index = []
    for key, value in packet.evidence.items():
        item = dict(value)
        if item["kind"] in {"native_entailment", "native_nonvacuity"}:
            fact = json.loads(item["text"])
            prefix = "ENTAILS" if item["kind"] == "native_entailment" else "SATISFIABLE"
            key = prefix + "_" + key[1:]
            index.append(
                f"{key}: plan {fact['plan_id']}, {fact['query_id']}, {prefix}={item['symbolic_value']}; query: {fact['query']}"
            )
        evidence[key] = item
    started = time.monotonic()
    for plan, assignment in zip((packet.plan_a, packet.plan_b), assignments):
        context = canonical_hash(
            (REVISION, case_record, tuple(assignment), plan.content_hash, packet.evidence["D2"])
        )
        path = Path(directory) / (context + ".json")
        if path.exists():
            saved = read(path)
        else:
            left = seconds - (time.monotonic() - started) - 2
            if left <= 1:
                raise TimeoutError("Intention evidence preparation exhausted its worker slice")
            result = bounded_call(
                _check,
                case_record,
                tuple(assignment),
                plan.to_dict(),
                timeout=left,
                memory_mb=memory_mb,
            )
            saved = dict(
                context=context,
                status=result.status,
                cleanup_complete=result.cleanup_complete,
                detail=result.detail,
                evidence=result.value,
                evidence_hash=canonical_hash(result.value),
                resource_usage=dict(getattr(result, "resource_usage", ())),
            )
            immutable(path, saved)
        if saved["context"] != context or saved["evidence_hash"] != canonical_hash(
            saved["evidence"]
        ):
            raise ValueError("Auxiliary evidence checkpoint identity changed")
        if saved["status"] != "complete" or not saved["cleanup_complete"] or not saved["evidence"]:
            raise ValueError("Auxiliary native evidence unavailable: " + str(saved.get("detail")))
        proof = saved["evidence"]
        if (
            proof["schema"],
            proof["plan_hash"],
            proof["theory_hash"],
            proof["policy_hash"],
            proof["intended_hash"],
        ) != (
            REVISION,
            plan.content_hash,
            plan.theory_hash,
            plan.policy_hash,
            canonical_hash(queries),
        ) or len(
            proof["rows"]
        ) != len(
            queries
        ):
            raise ValueError("Auxiliary native proof does not cover this plan/intention basis")
        for row in proof["rows"]:
            for field, prefix in (
                ("entailed", "INTENDED_ENTAILS"),
                ("nonvacuous", "INTENDED_SATISFIABLE"),
            ):
                if type(row[field]) is not bool:
                    raise ValueError("Incomplete auxiliary fact")
                key = prefix + "_" + canonical_hash((context, row["query_id"]))[:16]
                truth = str(row[field]).lower()
                interpretation = (
                    "true: commitment follows; false: commitment does not follow, a completed negative result, not missing evidence or entailment of negation."
                    if field == "entailed"
                    else "true: tested expressions can have instances; false: at least one cannot. This does not establish entailment."
                )
                evidence[key] = dict(
                    source_id=saved["evidence_hash"],
                    release=REVISION,
                    kind="native_intended_" + field,
                    symbolic_value=truth,
                    text=canonical_json(
                        dict(
                            plan_id=plan.plan_id,
                            theory_hash=plan.theory_hash,
                            policy_hash=plan.policy_hash,
                            intended_basis_hash=proof["intended_hash"],
                            query=row["query"],
                            query_id=row["query_id"],
                            outcome=truth,
                            interpretation=interpretation,
                        )
                    ),
                )
                index.append(
                    f"{key}: plan {plan.plan_id}, {row['query_id']}, {prefix}={truth}; query: {row['query']}"
                )
    return replace(
        packet,
        task=TASK,
        evidence=evidence,
        local_context=(
            *packet.local_context,
            "Checked fact index. Plan IDs identify complete theories; ENTAILS and SATISFIABLE answer different questions.\n"
            + "\n".join(dict.fromkeys(index)),
        ),
    )
