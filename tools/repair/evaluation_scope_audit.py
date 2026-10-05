"""Authenticate fresh intended qualification and report required-control gaps.

This reader makes no reasoner calls. Intended truth never becomes inference
information, new assignment labels, or a stronger baseline by implication.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import pyowl_core as owl

from exact.repair.api import write_artifact
from exact.repair.learning import OwlTeacherOracle, evaluate_teacher
from exact.repair.owl import CheckReport, ObligationResult, SupportReport, _class, _query_id, snapshot_from_axioms
from exact.repair.records import canonical_hash
from tools.repair.evaluation_intended import BUDGET, query_scope, validate_schedule
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.scaling_audit import attempt, checkpoint

SCHEMA = 'exact-repair/fresh-evaluation-scope-audit/v1'
FOLLOWUPS = ['observable_semantic_control_development_qualification',
             'separately_frozen_matched_fresh_control_evaluation',
             'failure_inclusive_parent_group_final_report']


def expected_queries(monitored=(), required=(), prohibited=(), activated=()):
    classes = sorted({_class(v) for v in monitored} - {owl.OWL_NOTHING}, key=lambda v: v.iri.value)
    queries = [('consistency', None, True)]
    queries += [('class_satisfiability', v, True) for v in classes]
    queries += [('required_entailment', v, True) for v in required]
    queries += [('prohibited_entailment', v, False) for v in prohibited]
    queries += [('active_satisfiability', v, True)
                for v in sorted(set(activated), key=lambda v: v.canonical_bytes())]
    return [(kind, _query_id(value), expected) for kind, value, expected in queries]


def native_report(evidence, payloads, directory, sequence, snapshot, expected):
    path = directory / 'native' / f'check-{sequence}.json'
    require(str(path) in payloads, 'Missing completed native report')
    value = evidence.read(payloads[str(path)])
    require(value['theory_hash'] == snapshot.logical_fingerprint.hex,
            'Native report checked a different intended theory')
    actual = [(o['kind'], o['query_id'], o['expected']) for o in value['obligations']]
    require(actual == expected, 'Native policy/query obligation identities differ')
    support = value['support']
    require((support['reasoner'], support['backend']) == ('hermit', 'python'), 'Native backend differs')
    require(support['query_support'] == [[o['kind'], o['query_id'], o['complete']]
                                       for o in value['obligations']], 'Native support coverage differs')
    starts = evidence.read(payloads[str(path.with_name(f'check-{sequence}.started.json'))])
    require((starts['reasoner'], starts['backend']) == ('hermit', 'python'), 'Native start differs')
    progress_path = path.with_name(f'check-{sequence}.obligations.jsonl')
    progress = ([json.loads(line) for line in evidence.verify(payloads[str(progress_path)]).read_text().splitlines()]
                if str(progress_path) in payloads else [])
    require([p['obligation'] for p in progress] == [o for o in value['obligations'] if o['complete']],
            'Fsynced native obligations differ from terminal report')
    require(all((p['support']['reasoner'], p['support']['backend']) == ('hermit', 'python')
                for p in progress), 'Progress backend differs')
    obligations = tuple(ObligationResult(**o) for o in value['obligations'])
    feasible = support['input_supported'] and all(o.satisfied is True for o in obligations)
    infeasible = support['input_supported'] and any(o.satisfied is False for o in obligations)
    require(value['logical_status'] == ('VERIFIED_FEASIBLE' if feasible else
            'VERIFIED_INFEASIBLE' if infeasible else 'UNKNOWN'), 'Native aggregate status differs')
    return CheckReport(value['logical_status'], value['verification_scope'], value['theory_hash'],
                       obligations, SupportReport(**support)), value


class RecordedOracle:
    """Expose already authenticated outcomes through the teacher's read interface."""
    def __init__(self, report):
        self.report = report

    def check_theory(self, *args, **kwargs):
        return self.report


def audit_row(evidence, ref, row, case, identity):
    saved = evidence.checkpoint(ref, canonical_hash((identity, row)))
    require((saved['case_id'], saved['parent'], saved['split'], saved['family'], saved['control']) ==
            (case.case_id, case.structural_parent, 'test', case.family, case.control),
            'Qualification case/split identity differs')
    require(saved['query_scope'] == query_scope(case) and saved['query_denominator'] == len(case.probes),
            'Qualification typed query denominator differs')
    require(saved['budget'] == BUDGET and saved['cleanup_complete'] and
            not saved['supervision_admitted'] and not saved['original_results_replaced'],
            'Qualification boundary or cleanup differs')
    directory = Path(ref['path']).parent
    payloads = {p['path']: p for p in saved['payloads']}
    require(len(payloads) == len(saved['payloads']), 'Duplicate native payload')
    for payload in payloads.values():
        evidence.verify(payload)
    call = evidence.read(payloads[str(directory / 'call.json')])
    require((call['status'], call['cleanup_complete'], call['budget'], call['detail'], call['resources']) ==
            (saved['outer_status'], saved['cleanup_complete'], saved['budget'], saved['detail'], saved['resources']),
            'Outer native call evidence differs')
    result = saved['result']
    native = []
    if result is None:
        require(saved['outer_status'] != 'complete' and saved['status'] == saved['outer_status'],
                'Missing completed native result')
    else:
        require(saved['outer_status'] == 'complete', 'Native result on unfinished outer call')
        snapshot = snapshot_from_axioms(case.intended_theory)
        policy, raw = native_report(evidence, payloads, directory, 0, snapshot,
            expected_queries(case.problem.policy.monitored_classes, case.problem.policy.required,
                             case.problem.policy.prohibited, case.intended_active))
        require(evidence.read(payloads[str(directory / 'intended-policy.json')]) == raw,
                'Published intended policy differs')
        native.append(raw)
        expected = dict(logical_status=policy.logical_status, query_complete=False,
            original_guard_passed=False, intended_target_satisfied=None, semantic=None,
            query_denominator=len(case.probes), query_scope=query_scope(case))
        if policy.logical_status == 'VERIFIED_FEASIBLE':
            active = {c for p in case.probes if p.desired for c in p.conditions()}
            query, raw = native_report(evidence, payloads, directory, 1, snapshot,
                expected_queries(required=tuple(p.axiom for p in case.probes), activated=active))
            native.append(raw)
            semantic = evaluate_teacher(OwlTeacherOracle(RecordedOracle(query), snapshot, case.probes), case.probes)
            expected.update(query_complete=semantic.complete, original_guard_passed=semantic.complete,
                intended_target_satisfied=(all(o.credit if o.desired else not o.entailed
                    for o in semantic.outcomes) if semantic.complete else None), semantic=asdict(semantic))
            require(canonical_hash(evidence.read(payloads[str(directory / 'intended-queries.json')])) ==
                    canonical_hash(expected), 'Published intended query result differs')
        require(canonical_hash(result) == canonical_hash(expected), 'Semantic result differs from native obligations')
        status = ('qualified' if expected['original_guard_passed'] and expected['intended_target_satisfied'] else
                  'intended_target_mismatch' if expected['original_guard_passed'] else 'native_unknown_or_infeasible')
        require(saved['status'] == status, 'Qualification outcome differs')
    return dict(case_id=case.case_id, group_id=case.structural_parent, family=case.family,
        family_exposure=row['family_exposure'], control=case.control, status=saved['status'],
        outer_status=saved['outer_status'], detail=saved['detail'], receipt=ref,
        query_denominator=len(case.probes), query_scope=query_scope(case),
        original_guard_passed=result['original_guard_passed'] if result else False,
        intended_target_satisfied=result['intended_target_satisfied'] if result else None,
        elapsed_seconds=saved['elapsed_seconds'], resources=saved['resources'],
        native_reports=[dict(logical_status=v['logical_status'], verification_scope=v['verification_scope'],
            theory_hash=v['theory_hash'], support=v['support'], obligations=v['obligations']) for v in native],
        native_payload_count=len(payloads), original_arm_results_replaced=False)


def control_review(evidence, manifest, primary):
    """Source-bound review; implementation availability is not scientific qualification."""
    review = evidence.read(manifest['control_review'])
    require(review['schema'] == 'exact-repair/fresh-control-review/v1' and
            review['test_outcomes_used_for_design'] is False and
            review['evaluator_inputs_allowed_at_inference'] is False and
            review['implementation_qualified'] is False, 'Control review boundary differs')
    for ref in review['requirements'] + review['source_files']:
        evidence.verify(ref)
    require(review['remaining_registered_obligations'] == FOLLOWUPS, 'Control followups lost')
    source = next(r for r in review['source_files'] if r['path'].endswith('/fresh_evaluation.py'))
    tree = ast.parse(evidence.verify(source).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'control_objective')
    require([a.arg for a in fn.args.args] == ['problem', 'arm', 'protocol'], 'Control data interface changed')
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    require('original_axioms' in attrs and 'axioms' in attrs, 'Historical retention control changed')
    require({a['id'] for a in primary['arms'] if a['kind'] == 'control'} ==
            {'symbolic_rich_action', 'uniform', 'deletion'}, 'Original control schedule changed')
    require('not strongest' in primary['control_definition']['symbolic_rich_action'],
            'Original semantic-control limitation missing')
    # Do not infer a qualified comparator from an AST or passing fixture tests.
    return dict(review=manifest['control_review'], status='reviewed_gaps_remain',
        original_controls=primary['control_definition'], required_comparisons=review['required_comparisons'],
        strongest_symbolic_comparison_qualified=False, implementation_qualified=False,
        evaluator_leakage_forbidden=True, remaining_registered_obligations=FOLLOWUPS)


def run(manifest_path, output):
    evidence = Evidence()
    manifest = evidence.read(binding(manifest_path))
    require(manifest['schema'] == SCHEMA and manifest['expected_cases'] == 64 and
            manifest['expected_parents'] == 32 and manifest['scientific_rows_replayed'] == 0 and
            manifest['supervision_admitted'] is False, 'Scope audit boundary differs')
    plan = evidence.read(manifest['intended_plan'])
    cases = validate_schedule(plan)
    primary = evidence.read(plan['primary_schedule'])
    ledger = evidence.read(manifest['ledger_snapshot'])
    registry = evidence.read(manifest['registry_snapshot'])
    registered = {r['id']: r for r in registry['runs']}
    require(len(manifest['attempts']) == len(manifest['active_run_ids']) == 9 and
            {i['run']['id'] for i in manifest['attempts']} == set(manifest['active_run_ids']),
            'Qualification attempt denominator differs')
    rows, attempts, seen = [], [], set()
    validation = None
    for item in manifest['attempts']:
        owner = item['run']
        require(registered[owner['id']] == owner and owner.get('enabled', True) and
                not owner.get('superseded_by'), 'Qualification lineage differs')
        result = attempt(evidence, item, ledger, True)
        batch = evidence.read(item['batch'])
        report = checkpoint(evidence, item['report']) if owner.get('row_slice') else evidence.read(item['report'])
        complete = result['completion']
        outputs = evidence.read(item['outputs'])
        relative = str(Path(item['report']['path']).relative_to(complete['work']))
        require(outputs.get(relative) == item['report']['sha256'], 'Qualification report absent from outputs')
        require(any(j['id'] == owner['logical_id'] for j in batch['jobs']), 'Frozen qualification job missing')
        attempts.append(dict(run_id=owner['id'], step_id=owner['step_id'], completion=item['completion'],
            report=item['report'], runtime=item['runtime'], charge=result['charge']))
        if not owner.get('row_slice'):
            require(validation is None and report['status'] == 'complete', 'Invalid native validation')
            checks = []
            for group in report['groups']:
                evidence.verify(group['log'])
                root = ET.parse(evidence.verify(group['junit'])).getroot()
                checks += [t for t in root.iter('testcase')]
            require(len(checks) == 38 and not any(any(t.find(k) is not None
                for k in ('error', 'failure', 'skipped')) for t in checks), 'Native fixture validation differs')
            validation = dict(report=item['report'], passed_tests=len(checks))
            continue
        start, stop = owner['row_slice']
        require(report['row_slice'] == [start, stop] and report['plan'] == manifest['intended_plan'] and
                report['status'] == 'complete' and report['scheduled_rows'] == report['recorded_rows'] ==
                len(report['rows']) == stop-start == 8, 'Qualification shard denominator differs')
        code = Path(batch['code'])
        identity = canonical_hash((manifest['intended_plan'],
            batch['frozen_files'][str(code/'tools/repair/evaluation_intended.py')],
            batch['frozen_files'][str(code/'tools/repair/development_intended.py')],
            batch['frozen_files'][str(code/'tools/repair/acquisition.py')], result['runtime']))
        require(report['identity'] == identity, 'Qualification source/runtime identity differs')
        shard = []
        for ref, (row, _, case) in zip(report['rows'], cases[start:stop]):
            require(case.case_id not in seen, 'Duplicate qualification case')
            seen.add(case.case_id)
            shard.append(audit_row(evidence, ref, row, case, identity))
        require(report['counts'] == dict(Counter(r['status'] for r in shard)) and
                report['query_denominator'] == sum(r['query_denominator'] for r in shard),
                'Qualification shard aggregates differ')
        rows += shard
    require(seen == {c.case_id for _, _, c in cases} and validation is not None, 'Missing qualification denominator')
    review = control_review(evidence, manifest, primary)
    report = dict(schema=SCHEMA, status='complete', manifest=binding(manifest_path),
        scheduled_rows=64, recorded_rows=len(rows), parent_groups=32, rows=rows,
        counts=dict(Counter(r['status'] for r in rows)), attempts=attempts, validation=validation,
        query_denominator=sum(len(c.probes) for _, _, c in cases),
        unwanted_queries=sum(not p.desired for _, _, c in cases for p in c.probes),
        typed_nonvacuity_conditions=sum(len(p.conditions()) for _, _, c in cases for p in c.probes if p.desired),
        control_review=review, native_payload_count=sum(r['native_payload_count'] for r in rows),
        gates_passed=False, study_complete=False, supervision_admitted=False,
        original_results_replaced=False, strongest_symbolic_comparison_qualified=False,
        scientific_rows_replayed=0, new_native_calls=0, remaining_registered_obligations=FOLLOWUPS,
        limitations=['Bound intended theories/policies/typed queries on recorded Python HermiT backend only.',
            'Unknowns and mismatches retain the full denominator; this does not relabel any arm result.',
            'No unwanted probes in the original basis; no whole-language or real-domain semantic claim.',
            'Required semantic-control implementation/development qualification and fresh comparison remain.',
            'Scope review is not G0-G2, learning efficiency, campaign completion or terminal accounting.'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_artifact(output/'verified-files.json', evidence.files)
    write_artifact(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == '__main__':
    main()
