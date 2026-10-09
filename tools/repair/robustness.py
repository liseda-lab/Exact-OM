"""Predeclared paired evidence/omission robustness; no outcome-based selection.

All variants inherit their original structural parent and split. Removal targets
are evaluator-owned controls, never ordinary evidence or neural graph features.
Conflict-overlap construction and the final paired report remain separate stages.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
from pathlib import Path

import pyowl_core as owl

from exact.repair.records import canonical_hash, read_record
from tools.repair.expanded_corpus import binding, bound, immutable
from tools.repair import fresh_evaluation as fresh

CONDITIONS = (
    ("baseline", "none", 0.0),
    ("score_noise_020", "score_noise", 0.2),
    ("evidence_missing_050", "evidence_missing", 0.5),
    ("final_pool_050", "final_pool", 0.5),
    ("retrieval_symbols_025", "retrieval_symbols", 0.25),
)


def _seed(group, *parts):
    return int(canonical_hash((20261003, group, parts))[:16], 16)


def _sample(items, group, kind, fraction):
    items = sorted(set(items), key=lambda x: canonical_hash((20261003, group, kind, x)))
    return items[: math.ceil(len(items) * fraction)]


def intervention(problem, group, condition, protocol):
    """Use observable inputs only, with common deterministic perturbations by group."""
    name, kind, dose = condition
    manifest = dict(
        condition=name,
        kind=kind,
        dose=dose,
        group_id=group,
        original_input_hash=problem.content_hash,
        seed=20261003,
        final_candidate_removals={},
        omitted_generation_symbols=[],
    )
    if kind in {"score_noise", "evidence_missing"}:
        evidence, changed = [], []
        for owner, frozen_values in problem.evidence:
            values = copy.deepcopy(dict(frozen_values))
            if "channels" in values:
                values["channels"] = dict(values["channels"])
            slots = [("score",)] if isinstance(values.get("score"), (int, float)) else []
            slots += [
                ("channels", key)
                for key, val in values.get("channels", {}).items()
                if isinstance(val, (int, float)) and not isinstance(val, bool)
            ]
            chosen = slots if kind == "score_noise" else _sample(slots, group, (kind, owner), dose)
            for slot in chosen:
                container = values if len(slot) == 1 else values[slot[0]]
                key = slot[-1]
                if kind == "score_noise":
                    delta = random.Random(_seed(group, kind, owner, slot)).gauss(0.0, dose)
                    container[key] = min(1.0, max(0.0, container[key] + delta))
                else:
                    del container[key]
                changed.append([owner, *slot])
            evidence.append((owner, values))
        problem = dataclasses.replace(problem, evidence=tuple(evidence))
        manifest.update(changed_fields=changed, applicable=bool(changed))
    elif kind == "final_pool":
        removals = {}
        for obj in problem.objects:
            eligible = [
                c.candidate_id
                for c in obj.candidates
                if c.axioms and (set(c.axioms) != set(obj.original_axioms) or c.active_expressions)
            ]
            selected = _sample(eligible, group, (kind, obj.object_id), dose)
            if selected:
                removals[obj.object_id] = selected
        manifest.update(
            final_candidate_removals=removals,
            applicable=bool(removals),
            target_scope="supplied nonempty changed candidates; excludes unchanged and deletion",
        )
    elif kind == "retrieval_symbols":
        from exact.repair.retrieval import retrieve_vocabulary

        options = fresh.pilot.generation_options(protocol, Path("unused-robustness-cache"))
        retrieval = retrieve_vocabulary(problem, config=options["retrieval_config"])
        original = {
            str(entity.iri.value)
            for obj in problem.objects
            for axiom in obj.original_axioms
            for entity in owl.signature(axiom)
        }
        available = {
            str(e.iri.value)
            for menu in retrieval.menus
            for field in (
                "source_classes",
                "target_classes",
                "source_properties",
                "target_properties",
            )
            for e in getattr(menu, field)
        } - original
        selected = _sample(available, group, kind, dose)
        manifest.update(
            omitted_generation_symbols=selected,
            applicable=bool(selected),
            eligible_symbols=len(available),
            target_scope="declared output menus; asserted ontology and evidence remain visible",
        )
    elif kind == "none":
        manifest["applicable"] = True
    else:
        raise ValueError("Unknown robustness intervention")
    manifest["effective_input_hash"] = problem.content_hash
    return problem, manifest


def generation_overrides(item, problem):
    control = item.get("intervention")
    if not control:
        return {}
    if problem.content_hash != control["effective_input_hash"]:
        raise ValueError("Robustness effective observable changed")
    return dict(
        final_candidate_removals=control["final_candidate_removals"],
        omitted_generation_symbols=tuple(control["omitted_generation_symbols"]),
    )


def audit_final_pool(problem, item):
    """Assert exclusion after all deterministic, sampled, preserved and fallback producers."""
    control = item.get("intervention")
    if not control:
        return None
    removed = control["final_candidate_removals"]
    omitted = set(control["omitted_generation_symbols"])
    for obj in problem.objects:
        if set(removed.get(obj.object_id, ())) & {c.candidate_id for c in obj.candidates}:
            raise ValueError("Final candidate exclusion violated")
        original = {str(e.iri.value) for a in obj.original_axioms for e in owl.signature(a)}
        for candidate in obj.candidates:
            novel = {
                str(e.iri.value)
                for a in (*candidate.axioms, *candidate.active_expressions)
                for e in owl.signature(a)
            } - original
            if novel & omitted:
                raise ValueError("Omitted vocabulary reintroduced")
    return dict(
        condition=control["condition"],
        applicable=control["applicable"],
        final_exclusion_verified=True,
        removal_target_count=sum(map(len, removed.values())),
        omitted_symbol_count=len(omitted),
        control_hash=canonical_hash(control),
    )


def prepare(campaign, model_schedule, corpus_attempt, output):
    output = Path(output)
    base = fresh.prepare(campaign, model_schedule, corpus_attempt, output / "base-schedule.json")
    schedule = copy.deepcopy(base)
    schedule.update(
        study_kind="paired_robustness_evidence_and_omission",
        followup="xr21-expanded-robustness-001",
        base_schedule=binding(output / "base-schedule.json"),
    )
    protocol = bound(base["arms"][0]["protocol"])
    cases = []
    for original in base["cases"]:
        for condition in CONDITIONS:
            item = copy.deepcopy(original)
            item.update(
                base_case_id=original.get("case_id"),
                condition=condition[0],
                case_id=str(original.get("case_id")) + ":robustness:" + condition[0],
            )
            if original["status"] == "materialized":
                problem = read_record(bound(original["observable"]))
                transformed, control = intervention(
                    problem, original["group_id"], condition, protocol
                )
                key = canonical_hash((original, condition))
                item["observable"] = immutable(
                    output / "observables" / (key + ".json"), transformed.to_dict()
                )
                item["intervention"] = control
                item["input_hash"] = transformed.content_hash
                # Keep the evaluator sealed and unchanged; only frozen.problem replaces
                # its presented theory after selection in fresh.evaluate_row.
                item["evaluator"] = original["evaluator"]
            cases.append(item)
    schedule.update(
        cases=cases,
        rows=fresh.make_rows(cases, schedule["arms"]),
        planned_cases=len(cases),
        planned_base_cases=64,
        planned_parent_groups=32,
        planned_model_rows=len(cases) * 6,
        planned_control_rows=len(cases) * 3,
        scheduled_rows=len(cases) * 9,
        conditions=CONDITIONS,
        baseline_policy="New same-source cold-cache baseline, charged in this named robustness study; no prior primary rows overwritten or silently retried",
        pairing="base_case_id x arm; parents are the uncertainty unit; coherent/corrupted remain nested; variants never create independent parents",
        selection_policy="No primary or robustness test outcomes inspected to select doses, targets, models or settings",
        primary_measure="paired generated-pool verified quality, coverage and effort; all scheduled statuses retained",
        inference_policy="Exploratory paired effects, 2000 parent bootstrap resamples seed 20261003, 95% intervals; no uncorrected confirmatory significance claims",
        omission_policy="Evaluator-owned final removal IDs never enter ordinary evidence; vocabulary omission changes declared menus; empty target sets retained as explicit no-op coverage",
        additional_stages=[
            "overlapping_conflicts",
            "receipt_and_semantic_scope_audit",
            "paired_full_denominator_report",
        ],
        larger_trained_models="Not included; any addition requires a separately frozen schedule before its outcomes",
        study_complete=False,
    )
    immutable(output / "schedule.json", schedule)
    return schedule
