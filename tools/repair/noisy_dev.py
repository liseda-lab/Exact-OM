"""Prospective, noise-aware DEV reporting at a fixed common training endpoint.

Reuses native evidence, projection, strict parsing and durable paid-call receipts.
Disagreement masks a comparison; it never discards a model or triggers resampling.
Historical qualification and checkpoint-selection results are left untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import statistics

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair import categorical_dev as legacy, scoped_judge as scoped
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, immutable

VERSION = "exact-repair/noise-aware-dev/v1"
ROLE = "repair-noise-aware-development/v1"
SCOPE = dict(
    meaning_retention="The intended editable commitments explicitly declared in D2, "
    "interpreted using D1/D3 and the complete fixed formal context.",
    assertion_fidelity="All currently selected editable assertions, including their direction, "
    "scope and qualifiers, justified against D1/D2/D3. Original observations are fallible.",
    collateral_fidelity="Other intended fixed/contextual relationships in D1/D2/D3. Do not "
    "count a focal omission or unsupported focal assertion twice; count a distinct downstream "
    "mismatch only when its own intended meaning is supported. Required unknown stays unknown.",
)


def rules():
    return dict(
        checkpoint_selection="fixed_common_training_endpoint",
        epoch=6,
        optimizer_updates=756,
        semantic_comparisons=32,
        logical_rows=192,
        audit="unchanged six order-audited slots; repeat unary assessments for those slots",
        orientation="alternate hash-ordered (seed,parent) groups; same order for both arms",
        reuse="exact wire identity within repetition only; shared receipts are not independent",
        weights={k: 1 / 3 for k in scoped.chat.CRITERIA},
        tie_tolerance=0.05,
        maximum_criterion_repeat_range=0.25,
        ordinal="all direct votes and every complete repeat-derived preference must agree; "
        "otherwise abstain, retaining every observation and reason",
        numeric="median per criterion, equal fixed weights; null if any required value is "
        "unknown, missing, invalid, range >0.25, or ordinal/numeric inconsistent",
        paired_analysis="both arms on the same (seed,parent,case), no model filtering; "
        "cluster at parent and retain unavailable pairs in the scheduled denominator",
        uncertainty="2000 deterministic parent bootstrap draws, seed 4713; descriptive "
        "percentile interval, not judge calibration or expert truth",
        sensitivity="all scheduled parent means; unavailable arm differences range [-1,1]",
        automatic_resampling=False,
        train_eligible=False,
        test_opened=False,
        scientific_gate="noise is a reported outcome, not a campaign execution failure",
    )


def cohort(prior):
    """Authenticate the exact old DEV outputs, without reusing old judge labels."""
    ancestor = bound(bound(prior["authorization"])["prior_development_plan"])
    manifest = bound(ancestor["annotation_manifest"])
    amendment = bound(manifest["dev_interface_amendment"])
    historical = bound(amendment["historical_review"])
    if (
        not historical["training_integrity_passed"]
        or not historical["full_dev_lifecycle_qualified"]
    ):
        raise ValueError("Training/native lifecycle is not qualified")
    expected = {
        (key, swap)
        for key in amendment["requests"]
        for swap in (
            [False, True] if manifest["frozen_annotation_slots"][key].get("swapped") else [False]
        )
    }
    observed = [(s["selection_key"], s["swapped"]) for s in prior["slots"]]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Original semantic denominator changed")
    admission = bound(prior["source_native_admission"])
    if (
        admission["manifest"] != ancestor["annotation_manifest"]
        or admission["truncated"]
        or {(r["slot"], r["swapped"]) for r in admission["rows"]} != expected
        or any(r["status"] != "eligible" for r in admission["rows"])
    ):
        raise ValueError("Native evidence admission changed")
    models = {(m["condition"], m["seed"]): m for m in prior["models"]}
    if len(models) != 6 or set(models) != {
        (c, s) for c in ("symbolic", "symbolic_plus_llm") for s in (13, 37, 73)
    }:
        raise ValueError("Six-model cohort changed")
    weights, rows = {}, {}
    for key, model in models.items():
        w = bound(model["weights"])
        authenticate(w["artifact"])
        if w["epoch"] != 6 or w["optimizer_updates"] != 756 or (w["condition"], w["seed"]) != key:
            raise ValueError("Common training endpoint changed")
        weights[key] = w
        report = bound(model["source_report"])
        source_manifest = bound(report["manifest"])
        rr = [bound(r) for r in report["rows"]]
        if len(rr) != 32 or {r["case_id"] for r in rr} != set(source_manifest["case_ids"]):
            raise ValueError("Logical DEV denominator changed")
        rows[key] = {canonical_hash(r["selection_slot"]): r for r in rr}
    cases = []
    for s in prior["slots"]:
        if s["swapped"]:
            continue
        key, selection = s["selection_key"], s["selection_slot"]
        member, request = amendment["requests"][key], bound(s["source_request"])
        model_key = (selection["supervision_condition"], selection["seed"])
        packet = legacy.packet_for(s)
        if (
            s["source_request"] != member["source_request"]
            or canonical_hash(request["selection_slot"]) != key
            or selection != request["selection_slot"]
            or packet.case_id != request["case"]["case"]["case_id"]
            or bound(s["packet_request"])["interface_origin"] != member["source_request"]
            or Path(s["packet"]["path"]).parent != Path(s["packet_request"]["path"]).parent
            or key not in rows[model_key]
        ):
            raise ValueError("Decoded DEV source/provenance changed")
        for ref in member["verified_plans"]:
            bound(ref)
        legacy.validate_decoded_identity(s, weights[model_key])
        payload = packet.judge_payload()
        if any(
            not payload["evidence"][k]["kind"].startswith("authored") for k in ("D1", "D2", "D3")
        ):
            raise ValueError("Expected authored meaning contract missing")
        cases.append(
            dict(
                id=key,
                payload=payload,
                scope=SCOPE,
                source_slot=s,
                parent=packet.parent_group_id,
                case_id=packet.case_id,
                seed=selection["seed"],
                condition=selection["supervision_condition"],
                family=rows[model_key][key]["family"],
                audited=(key, True) in expected,
            )
        )
    pairs = defaultdict(list)
    for c in cases:
        pairs[(c["seed"], c["parent"], c["case_id"])].append(c)
    if len(cases) != 32 or len(pairs) != 16 or sum(c["audited"] for c in cases) != 6:
        raise ValueError("Fixed semantic sample/audits changed")
    for i, key in enumerate(sorted(pairs, key=canonical_hash)):
        pp = pairs[key]
        if (
            sorted(c["condition"] for c in pp) != ["symbolic", "symbolic_plus_llm"]
            or {k: v for k, v in pp[0]["payload"]["plan_b"].items() if k != "report_id"}
            != {k: v for k, v in pp[1]["payload"]["plan_b"].items() if k != "report_id"}
            or pp[0]["audited"] != pp[1]["audited"]
        ):
            raise ValueError("Arm pairing or common comparator changed")
        for c in pp:
            c["swapped"] = bool(i % 2)
    # Native records contain tuples; authenticate their lossless JSON form on
    # preparation and replay alike, instead of rejecting tuple/list round trips.
    return json.loads(
        json.dumps(sorted(cases, key=lambda c: (c["seed"], c["parent"], c["condition"])))
    )


def packet(case, slot, models):
    wire = scoped.packet(case, slot, models)
    payload = json.loads(wire["messages"][1]["content"])
    payload["assessment_contract"]["version"] = VERSION
    wire["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    findings = {
        **scoped.FINDINGS,
        "mixed": "This criterion contains both supported and damaged content; "
        "damage may be lost intended content, unsupported commitments, or both.",
    }
    wire["messages"][0]["content"] += (
        "\nFinding definitions: "
        + json.dumps(findings)
        + "\nDirect preference definitions: "
        + json.dumps(scoped.PREFERENCE)
    )
    return wire


def schedule(cases, models):
    """Reuse only identical complete wires, without collapsing independent repeats."""
    slots, links = {}, {}
    for c in cases:
        link = dict(plans={s: [] for s in ("a", "b")}, pairs=[])
        for repeat in range(2 if c["audited"] else 1):
            requests = [dict(kind="plan", side=s, swapped=False) for s in ("a", "b")]
            requests.append(dict(kind="pair", swapped=c["swapped"] != bool(repeat)))
            for request in requests:
                slot = dict(case=c["id"], repeat=repeat, stage="chat", model="chat", **request)
                wire = packet(c, slot, models)
                identity = canonical_hash((wire, repeat, request["kind"]))
                slot["id"] = "noise-dev001-" + identity[:24]
                slots.setdefault(slot["id"], slot)
                if request["kind"] == "plan":
                    link["plans"][request["side"]].append(slot["id"])
                else:
                    link["pairs"].append(dict(slot=slot["id"], swapped=slot["swapped"]))
        links[c["id"]] = link
    return list(slots.values()), links


def parse(response, wire, stage, model):
    answer = scoped.parse(response, wire, stage, model)
    p = json.loads(wire["messages"][1]["content"])
    if "plan" in p and answer["meaning_retention"]["anchor"] == "full":
        if any(
            f["kind"] == "native_intended_entailed" and f.get("symbolic_value") == "false"
            for f in p["evidence"].values()
        ):
            raise ValueError("Perfect retention contradicts a supplied missing intended commitment")
    if (
        "plan_a" in p
        and p["plan_a"]["current_theory"] == p["plan_b"]["current_theory"]
        and answer["preference"] not in ("tie", "abstain")
    ):
        raise ValueError("Unequal preference for identical current theories")
    return answer


def validate(plan):
    if (
        plan["schema"] != VERSION
        or plan["semantic_role"] != ROLE
        or plan["rules"] != rules()
        or plan["budget_phase"] != "development"
        or plan["transport_role"] != ROLE
        or set(plan["models"]) != {"chat"}
        or plan["test_opened"]
        or plan["selection_rule_unchanged"]
        or plan["deadline_epoch"] > 1791997200
    ):
        raise ValueError("Prospective DEV protocol changed")
    auth, quota, phase = (
        bound(plan[k]) for k in ("authorization", "quota_amendment", "previous_phase_state")
    )
    if (
        auth["rules"] != rules()
        or auth["max_requests"] != plan["max_requests"]
        or phase["cost_ceiling_usd"] != 35
        or quota["cost_ceiling_usd"] != 35
        or quota["new_limits"] != phase["request_limits"]
        or quota["new_limits"]["test"] != 154
    ):
        raise ValueError("Budget or authorization changed")
    prior = bound(plan["prior_development_plan"])
    cases = cohort(prior)
    if cases != bound(plan["cases"]):
        raise ValueError("Saved evidence, scopes or checkpoint lineage changed")
    expected, links = schedule(cases, plan["models"])
    if len(expected) != plan["max_requests"] or len(expected) != len(plan["slots"]):
        raise ValueError("Incomplete request schedule")
    costs, by_case = {}, {c["id"]: c for c in cases}
    for e, s in zip(expected, plan["slots"]):
        if e != {k: v for k, v in s.items() if k != "packet"}:
            raise ValueError("Schedule or reuse changed")
        wire = bound(s["packet"])
        if wire != packet(by_case[s["case"]], s, plan["models"]):
            raise ValueError("Frozen wire changed")
        cost = scoped.chat.request_cost(plan["models"]["chat"], wire, plan["max_input_bytes"])
        route = next(
            r
            for r in bound(plan["models"]["chat"]["endpoint"])["data"]["endpoints"]
            if r["tag"] == plan["models"]["chat"]["route"]
        )
        prices = [route["pricing"]] + [
            p
            for p in route["pricing"].get("overrides", [])
            if p["min_prompt_tokens"] <= cost["input_token_upper_bound"]
        ]
        cost["reserved_cost_usd"] += (
            max(float(p.get("input_cache_write", 0)) for p in prices)
            * cost["input_token_upper_bound"]
        )
        costs[s["id"]] = cost
    if sum(c["reserved_cost_usd"] for c in costs.values()) > plan["reservation_usd"]:
        raise ValueError("Frozen reservation too small")
    return cases, links, costs


def aggregate(case, link, records):
    """Keep raw direct votes distinct from admitted ordinal and numeric labels."""
    reasons, assessments, numeric_stable = [], {}, True
    for side, ids in link["plans"].items():
        obs = [records.get(i, {}) for i in ids]
        if any(r.get("valid") is not True for r in obs):
            reasons.append("missing_or_invalid_assessment:" + side)
        values = {
            k: [
                scoped.chat.ANCHORS[r["answers"][k]["anchor"]] if r.get("valid") else None
                for r in obs
            ]
            for k in scoped.chat.CRITERIA
        }
        known = all(v is not None for vv in values.values() for v in vv)
        medians = {
            k: statistics.median(vv) if all(v is not None for v in vv) else None
            for k, vv in values.items()
        }
        ranges = {
            k: max(vv) - min(vv) if all(v is not None for v in vv) else None
            for k, vv in values.items()
        }
        stable = known and all(v <= 0.25 for v in ranges.values())
        numeric_stable &= stable
        if not known:
            reasons.append("required_criterion_unavailable:" + side)
        totals = [
            (
                sum(vv[i] for vv in values.values()) / 3
                if all(vv[i] is not None for vv in values.values())
                else None
            )
            for i in range(len(ids))
        ]
        assessments[side] = dict(
            observation_slots=ids,
            criteria=values,
            medians=medians,
            ranges=ranges,
            repeat_totals=totals,
            descriptive_total=sum(medians.values()) / 3 if known else None,
            numeric_stable=stable,
        )
    votes = []
    for p in link["pairs"]:
        r = records.get(p["slot"], {})
        v = r["answers"]["preference"] if r.get("valid") else "unavailable"
        if p["swapped"] and v in ("A", "B"):
            v = "B" if v == "A" else "A"
        votes.append(v)
    direct = votes[0] if len(set(votes)) == 1 and votes[0] in ("A", "B", "tie") else "abstain"
    if direct == "abstain":
        reasons.append("direct_abstention_or_disagreement")
    derived = [
        scoped.old.derived([a, b])
        for a, b in zip(assessments["a"]["repeat_totals"], assessments["b"]["repeat_totals"])
    ]
    median_preference = scoped.old.derived(
        [assessments[s]["descriptive_total"] for s in ("a", "b")]
    )
    if direct != "abstain" and (any(v != direct for v in derived) or median_preference != direct):
        reasons.append("direct_numeric_conflict")
    ordinal_ok = not reasons
    numeric_ok = ordinal_ok and numeric_stable
    if not numeric_stable:
        reasons.append("numeric_repeat_unstable_or_unavailable")
    return dict(
        id=case["id"],
        case_id=case["case_id"],
        parent=case["parent"],
        seed=case["seed"],
        condition=case["condition"],
        family=case["family"],
        audited=case["audited"],
        scheduled=True,
        assessments=assessments,
        canonical_votes=votes,
        direct_preference=direct,
        derived_preferences=derived,
        median_preference=median_preference,
        preference=direct if ordinal_ok else "abstain",
        ordinal_eligible=ordinal_ok,
        numeric_eligible=numeric_ok,
        score=assessments["a"]["descriptive_total"] if numeric_ok else None,
        reasons=reasons,
        training_eligible=False,
    )


def paired_summary(rows):
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["seed"], r["parent"], r["case_id"])][r["condition"]] = r
    groups, sensitivity, details = defaultdict(list), defaultdict(list), []
    for (seed, parent, case_id), arms in sorted(pairs.items()):
        if set(arms) != {"symbolic", "symbolic_plus_llm"}:
            raise ValueError("Unpaired semantic denominator")
        a, b = arms["symbolic"]["score"], arms["symbolic_plus_llm"]["score"]
        delta = b - a if a is not None and b is not None else None
        details.append(dict(seed=seed, parent=parent, case_id=case_id, delta=delta))
        sensitivity[parent].append((delta, delta) if delta is not None else (-1.0, 1.0))
        if delta is not None:
            groups[parent].append(delta)
    means = [statistics.mean(v) for v in groups.values()]
    rng = random.Random(4713)
    samples = (
        sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(2000))
        if means
        else []
    )
    interval = [samples[49], samples[1949]] if len(means) >= 2 else None
    bounds = [
        statistics.mean(statistics.mean(v[i] for v in vv) for vv in sensitivity.values())
        for i in (0, 1)
    ]
    return dict(
        scheduled_pairs=len(pairs),
        scheduled_parents=len(sensitivity),
        evaluable_pairs=sum(d["delta"] is not None for d in details),
        evaluable_parents=len(means),
        paired_rows=details,
        mean_difference=statistics.mean(means) if means else None,
        descriptive_parent_bootstrap_95=interval,
        full_denominator_missing_label_bounds=bounds,
        conclusion=(
            "directional_proxy_difference"
            if interval
            and ((interval[0] > 0 and bounds[0] > 0) or (interval[1] < 0 and bounds[1] < 0))
            else "inconclusive"
        ),
        claim="Authored DEV weak proxy, not independent TEST or expert truth",
    )


def summarize(plan, cases, links, records):
    by_id = {r["slot"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {s["id"] for s in plan["slots"]}:
        raise ValueError("Missing, duplicate or foreign observation slots")
    delivered = [r["request_id"] for r in records if r["status"] == "received"]
    if len(set(delivered)) != len(delivered):
        raise ValueError("A reused HTTP observation cannot count as independent")
    rows = [aggregate(c, links[c["id"]], by_id) for c in cases]
    models = []
    for m in bound(plan["prior_development_plan"])["models"]:
        own = [r for r in rows if (r["condition"], r["seed"]) == (m["condition"], m["seed"])]
        source = bound(m["source_report"])
        logical = [bound(r) for r in source["rows"]]
        models.append(
            dict(
                **m,
                selected_checkpoint=bound(m["weights"])["artifact"],
                selection_rule="fixed_common_training_endpoint",
                selection_criterion=None,
                logical_scheduled=len(logical),
                logical_verified=sum(
                    r["result"]["decoded"]["status"] == "verified" for r in logical
                ),
                semantic_scheduled=len(own),
                semantic_numeric_available=sum(r["numeric_eligible"] for r in own),
                weak_preferences=dict(Counter(r["preference"] for r in own)),
            )
        )
    return dict(
        schema=VERSION,
        plan_identity=canonical_hash(plan),
        status=(
            "complete_with_uncertainty"
            if any(not r["numeric_eligible"] for r in rows)
            else "complete"
        ),
        models=models,
        comparisons=rows,
        records=records,
        scheduled_comparisons=len(rows),
        logical_scheduled=sum(m["logical_scheduled"] for m in models),
        ordinal_available=sum(r["ordinal_eligible"] for r in rows),
        numeric_available=sum(r["numeric_eligible"] for r in rows),
        paired=paired_summary(rows),
        by_seed={str(s): paired_summary([r for r in rows if r["seed"] == s]) for s in (13, 37, 73)},
        by_family={
            f: paired_summary([r for r in rows if r["family"] == f])
            for f in sorted({r["family"] for r in rows})
        },
        selected_models_available=len(models),
        checkpoint_selection=rules()["checkpoint_selection"],
        dev_outcomes_selected_checkpoints=False,
        training_eligible=False,
        test_opened=False,
        semantic_truth_qualified=False,
        observation_integrity_checked=True,
        reported_cost_usd=sum(r.get("usage", {}).get("cost", 0) or 0 for r in records),
        cost_complete=all("cost" in r.get("usage", {}) for r in records),
        technical_unavailable=sum(r["status"] != "received" for r in records),
        technical_review_required=any(r["status"] != "received" for r in records),
    )


def run(path, output):
    if (
        os.getenv("EXACT_EXPERIMENT_MODE") == "1"
        or os.getenv("EXACT_OPENROUTER_RETRY_UNKNOWN") == "1"
    ):
        raise ValueError("Unapproved retry mode")
    plan, output = read(path), Path(output)
    cases, links, costs = validate(plan)
    scoped.chat.reserve(plan, costs)
    records = []

    # FIFO submission keeps paired seed/parent work adjacent; ordered results
    # make reports deterministic despite concurrent network I/O.
    def execute(slot):
        return scoped.old.one(plan, slot, output / "presentations", parser=parse)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(execute, plan["slots"]):
            records.append(record)
            write_artifact(
                output / "progress.json",
                dict(
                    status="running",
                    completed=len(records),
                    scheduled=len(plan["slots"]),
                    valid=sum(r["valid"] for r in records),
                ),
            )
    report = summarize(plan, cases, links, records)
    immutable(output / "report.json", report)
    immutable(
        output / "model-freeze.json",
        dict(
            schema=VERSION,
            plan=binding(path),
            report=binding(output / "report.json"),
            selection_rule=rules()["checkpoint_selection"],
            outcome_selection=False,
            models=report["models"],
            test_opened=False,
        ),
    )
    write_artifact(
        output / "progress.json",
        dict(
            status=report["status"],
            completed=len(records),
            scheduled=len(plan["slots"]),
            numeric_available=report["numeric_available"],
            selected_models_available=6,
            report=binding(output / "report.json"),
        ),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    r = run(args.plan, args.output)
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "status",
                    "scheduled_comparisons",
                    "numeric_available",
                    "selected_models_available",
                    "reported_cost_usd",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
