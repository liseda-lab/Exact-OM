"""Audit frozen train/development acquisition without relabeling or opening test data."""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.learning import SemanticTargetSpec
from exact.repair.records import canonical_hash
from tools.repair.acquisition import validate_schedule
from tools.repair.batch import read, sha
from tools.repair.historical_audit import checked_content
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import read_label_cache


def attempt_report(item):
    """Authenticate terminal ownership, frozen source/runtime and report publication."""
    completion = verify_binding(item['completion'])
    if (completion.get('status'), completion.get('exit_code'), completion.get('step_id'),
            completion.get('dispatch_nonce')) != ('complete', 0, item['step_id'], item['nonce']):
        raise ValueError('Acquisition audit completion ownership mismatch')
    if completion['batch'] != item['batch']['path']:
        raise ValueError('Acquisition audit batch mismatch')
    batch = verify_binding(item['batch'])
    if sha(item['batch']['path']) != Path(item['batch']['path']).with_name('batch.sha256').read_text().strip():
        raise ValueError('Frozen acquisition manifest changed')
    for path, digest in batch['frozen_files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen acquisition dependency changed: ' + path)
    outputs = verify_binding(item['outputs'])
    relative = str(Path(item['report']['path']).relative_to(completion['work']))
    if outputs.get(relative) != item['report']['sha256']:
        raise ValueError('Acquisition report is not in terminal output manifest')
    report = verify_binding(item['report'])
    return report, read(Path(batch['code']).parent / 'runtime.json'), batch


def expected_hashes(case, settings, runtime):
    weights = (settings['desired_family_weight'], settings['false_positive_weight'])
    target = SemanticTargetSpec(canonical_hash(case.probes), *weights)
    return dict(input=case.problem.content_hash, patch=canonical_hash(case.problem.objects),
        policy=case.problem.policy.content_hash, query=canonical_hash(case.probes),
        inventory=canonical_hash(tuple(obj.candidates for obj in case.problem.objects)),
        backend=canonical_hash(('pyhermit', runtime['dependencies']['pyhermit'], 'python')),
        profile=canonical_hash(settings['profile']), teacher_weights=canonical_hash(weights),
        semantic_target=target.content_hash)


def label_masks(cache, case, settings):
    """Check full semantic denominators/scalars and count only supported loss targets."""
    probes = {p.probe_id: p for p in case.probes}
    counts = Counter(requested=math.prod(cache.candidate_counts), visited=len(cache.labels))
    for label in cache.labels:
        counts['risk'] += label.feasible is not None
        counts['unknown_policy'] += label.feasible is None
        counts['infeasible'] += label.feasible is False
        counts['unknown_queries'] += label.feasible is True and label.benefit is None
        vector = {out.probe_id: out for out in label.semantic_vector}
        if len(vector) != len(label.semantic_vector) or not set(vector) <= set(probes):
            raise ValueError('Semantic vector has duplicate or foreign queries')
        families = defaultdict(list)
        for key, out in vector.items():
            probe = probes[key]
            if (out.family, out.desired) != (probe.family, probe.desired):
                raise ValueError('Semantic vector target basis changed')
            credit = (None if out.entailed is None or out.nonvacuous is None else
                      out.entailed and out.nonvacuous) if out.desired else out.entailed
            if out.credit != credit:
                raise ValueError('Semantic vector credit changed')
            families[(out.desired, out.family)].append(out.credit)
        if label.usable:
            if set(vector) != set(probes) or any(out.credit is None for out in vector.values()):
                raise ValueError('Semantic scalar omits required unknown query denominator')
            benefit = sum((settings['desired_family_weight'] if desired else
                           -settings['false_positive_weight']) * sum(values) / len(values)
                          for (desired, _), values in families.items())
            if not math.isclose(benefit, label.benefit, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Semantic scalar differs from frozen target')
            counts['usable'] += 1
    for key in ('risk', 'unknown_policy', 'unknown_queries', 'infeasible', 'usable'):
        counts.setdefault(key, 0)
    counts['unvisited'] = counts['requested'] - counts['visited']
    counts['anchored_value_candidates'] = max(0, counts['usable'] - 1)
    usable = [label for label in cache.labels if label.usable]
    counts['unequal_rank_pairs'] = sum(a.benefit != b.benefit for i, a in enumerate(usable)
                                      for b in usable[i + 1:])
    counts['exact_proposal_case'] = int(cache.complete and bool(usable))
    counts['sample_conditioned_proposal_case'] = int(not cache.complete and bool(usable))
    return dict(counts)


def audit_row(ref, case, acquisition_plan, runtime):
    saved = verify_binding(ref)
    checked_content(ref['path'])
    if (saved['case_id'], saved['split'], saved['parent']) != (
            case.case_id, case.split, case.structural_parent) or not saved['cleanup_complete']:
        raise ValueError('Acquisition row ownership, split or cleanup mismatch')
    native_counts, details, starts, checks, obligations = (Counter() for _ in range(5))
    calls = []
    for native_ref in saved['native_evidence']:
        value = verify_binding(native_ref)
        path = Path(native_ref['path'])
        if value.get('schema') == 'exact-repair/acquisition-call/v1':
            if (value['case_id'], value['split'], value['parent'], value['input_hash'],
                    value['policy_hash'], value['query_hash']) != (
                    case.case_id, case.split, case.structural_parent, case.problem.content_hash,
                    case.problem.policy.content_hash, canonical_hash(case.probes)):
                raise ValueError('Native call dependency identity differs')
            # Reconciled cleanup failures remain original evidence, never upgraded labels.
            native_counts[value['status']] += 1
            details[value.get('detail', '')] += 1
            calls.append(dict(receipt=native_ref, status=value['status'],
                cleanup_complete=value['cleanup_complete'], deadline_seconds=value['deadline_seconds']))
        elif path.name.endswith('.started.json'):
            starts[str((value.get('reasoner'), value.get('backend')))] += 1
        elif 'logical_status' in value:
            checks[value['logical_status']] += 1
            for obligation in value.get('obligations', []):
                obligations[str((obligation['kind'], obligation.get('complete'),
                                 obligation.get('verdict')))] += 1
    cache_ref, hashes, masks = None, None, None
    if saved.get('result'):
        result = saved['result']
        cache_ref = result['artifact']
        cache = read_label_cache(cache_ref, Path(cache_ref['path']).parent, case)
        hashes = expected_hashes(case, acquisition_plan['teacher'], runtime)
        if dict(cache.hashes) != hashes:
            raise ValueError('Label cache source/target/profile dependency mismatch')
        if result['coverage'] != cache.coverage or saved['status'] != ('complete' if cache.complete else 'partial'):
            raise ValueError('Label cache coverage/status mismatch')
        masks = label_masks(cache, case, acquisition_plan['teacher'])
    elif saved['status'] in {'complete', 'partial'}:
        raise ValueError('Completed acquisition label cache is missing')
    return dict(case_id=case.case_id, split=case.split, parent=case.structural_parent,
        family=case.family, control=case.control, receipt=ref, status=saved['status'],
        detail=saved.get('detail'), cache=cache_ref, expected_hashes=hashes, loss_candidates=masks,
        requested_assignments=math.prod(len(obj.candidates) for obj in case.problem.objects),
        required_queries=len(case.probes), query_types=dict(Counter(type(p.axiom).__name__ for p in case.probes)),
        native_calls=calls, native_call_statuses=dict(native_counts), native_call_details=dict(details),
        backend_start_records=dict(starts), native_check_statuses=dict(checks), native_obligations=dict(obligations),
        supervision_admitted=False,
        scope='Authenticated fixed-inventory candidates only; protocol/gates/generated-pool review required')


def requirement_review(matrices, qualifications, source):
    """Bind requested tests and production files; passing fixtures never imply gates."""
    evidence = []
    for item, report, batch in qualifications:
        if report.get('schema') != 'exact-repair/native-qualification/v1':
            continue
        observed = {}
        for group in report['groups']:
            for ref in (group['junit'], group['log']):
                if sha(ref['path']) != ref['sha256']:
                    raise ValueError('Qualification evidence changed')
            for test in ET.parse(group['junit']['path']).getroot().iter('testcase'):
                name = test.get('classname', '') + '::' + test.get('name', '')
                passed = not any(test.find(tag) is not None for tag in ('failure', 'error', 'skipped'))
                observed[name] = observed.get(name, True) and passed
        evidence.append((item, batch, observed))
    rows = []
    for matrix_ref in matrices:
        matrix = verify_binding(matrix_ref)
        for requirement in matrix['requirements']:
            requested = list(requirement.get('tests', requirement.get('acceptance_tests', [])))
            requested += [test for case in requirement.get('acceptance_cases', []) for test in case['tests']]
            checks = []
            for test in requested:
                filename, separator, name = test.partition('::')
                module = filename[:-3].replace('/', '.')
                prefix = module + '::' + name
                sources = []
                for item, batch, observed in evidence:
                    code = Path(batch['code'])
                    files = [*requirement['implementation'], filename]
                    compatibility = {file: (code.joinpath(file).exists() and
                        sha(code/file) == sha(source/file)) for file in files}
                    # A file selector requests every observed case in that module,
                    # including class methods. Keep module boundaries exact and
                    # retain failed/skipped cases and parameterized test variants.
                    if separator:
                        matches = {key: value for key, value in observed.items()
                                   if key == prefix or key.startswith(prefix + '[')}
                    else:
                        matches = {key: value for key, value in observed.items()
                                   if key.startswith(module + '::') or key.startswith(module + '.')}
                    sources.append(dict(run_id=item['run_id'], report=item['report'],
                        source_compatibility=compatibility, observed=matches,
                        passed=bool(matches) and all(matches.values()) and all(compatibility.values())))
                checks.append(dict(test=test, source_compatible_pass=any(s['passed'] for s in sources), evidence=sources))
            rows.append(dict(id=requirement.get('id', requirement.get('requirement')),
                matrix=matrix_ref, tests=checks,
                complete_source_compatible_test_evidence=bool(checks) and all(c['source_compatible_pass'] for c in checks)))
    return dict(requirements=rows, required=len(rows),
        source_compatible=sum(row['complete_source_compatible_test_evidence'] for row in rows),
        claim='Mapped fixture evidence only; actual native/generation coverage and research gates remain separate')


def validate_boundary(plan):
    if (plan.get('schema') != 'exact-repair/training-label-audit-plan/v1'
            or set(plan.get('acquisition_plans', {})) != {'train', 'development'}
            or plan.get('expected_cases') != {'train': 128, 'development': 32}
            or plan.get('heldout_use') is not False or plan.get('model_fitting') is not False):
        raise ValueError('Training audit split boundary must exclude held-out payloads')


def run(plan_path, output):
    plan = read(plan_path)
    validate_boundary(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    acquisition = {split: verify_binding(ref) for split, ref in plan['acquisition_plans'].items()}
    expected = {}
    for split, acq in acquisition.items():
        for _, _, case in validate_schedule(acq):
            if case.case_id in expected or case.split != split:
                raise ValueError('Duplicate/cross-split acquisition case')
            expected[case.case_id] = case
    seen, rows = set(), []
    for item in plan['attempts']:
        report, runtime, _ = attempt_report(item)
        checked_content(item['report']['path'])
        if (report['plan'] != plan['acquisition_plans'][item['split']]
                or report['recorded_rows'] != report['scheduled_rows']
                or report['recorded_rows'] != len(report['rows'])):
            raise ValueError('Acquisition report denominator/plan mismatch')
        for ref in report['rows']:
            saved = verify_binding(ref)
            key = saved['case_id']
            if key in seen or key not in expected or expected[key].split != item['split']:
                raise ValueError('Duplicate, missing or cross-split report row')
            seen.add(key)
            row = audit_row(ref, expected[key], acquisition[item['split']], runtime)
            row['attempt'] = item['run_id']
            path = output / 'rows' / (canonical_hash(key) + '.json')
            write_artifact(path, row)
            rows.append(row)
    if seen != set(expected):
        raise ValueError('Training audit must retain all160 case outcomes')
    qualifications = []
    for item in plan['supporting_attempts']:
        report, _, batch = attempt_report(item)
        qualifications.append((item, report, batch))
    requirements = requirement_review(plan['requirement_matrices'], qualifications,
                                      Path(__file__).resolve().parents[2])
    write_artifact(output / 'requirements.json', requirements)
    # Explicit recovery chains include failed original attempts, without requiring a
    # nonexistent successful original report or discarding its resource charges.
    for chain in plan['lineages']:
        for entry in chain:
            receipt = verify_binding(entry['completion'])
            if (receipt['step_id'], receipt['dispatch_nonce']) != (entry['step_id'], entry['nonce']):
                raise ValueError('Recovery lineage receipt ownership mismatch')
    summaries = {}
    for split in acquisition:
        selected = [row for row in rows if row['split'] == split]
        summary = dict(scheduled_cases=len(selected), statuses=dict(Counter(row['status'] for row in selected)),
            cases_with_cache=sum(row['cache'] is not None for row in selected),
            cases_with_usable_semantics=sum(bool((row['loss_candidates'] or {}).get('usable')) for row in selected),
            by_family={family: dict(Counter(row['status'] for row in selected if row['family'] == family))
                       for family in sorted({row['family'] for row in selected})})
        for key in ('native_call_statuses', 'native_call_details', 'backend_start_records',
                    'native_check_statuses', 'native_obligations', 'loss_candidates'):
            counter = Counter()
            for row in selected:
                counter.update(row[key] or {})
            summary[key] = dict(counter)
        summary['all_case_requested_assignments'] = sum(row['requested_assignments'] for row in selected)
        summary['no_cache_requested_assignments'] = sum(row['requested_assignments'] for row in selected if not row['cache'])
        summaries[split] = summary
    limitations = [
        'Native started records establish attempted route, not completed query support; timeouts remain unknown.',
        'Cache candidates are dependency-checked but not automatically admitted for fitting.',
        'Fixed-inventory evidence does not establish generated-pool acquisition or decoded checkpoint selection.',
        'Full-schema GPU capacity, proposal-circuit backward and executable18-arm protocols remain required.',
        'Research gates require current requirement/source, backend/query and actual generation coverage review.',
        'No test outcomes, relabeling, budget increases, pilot weights or source artifacts were used or changed.',
    ]
    result = dict(schema='exact-repair/training-label-audit/v1', status='complete', plan=binding(plan_path),
        source_sha256=sha(__file__), summaries=summaries, rows=rows, scheduled_rows=160,
        requirement_review=binding(output / 'requirements.json'),
        heldout_cases_opened=False, fitting_started=False, supervision_admitted=False,
        gates={gate: 'not_established' for gate in ('G0', 'G1', 'G2')}, limitations=limitations,
        remaining_stage='xr21-expanded-training-001', acquisition_costs_recharged=False)
    write_artifact(output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == '__main__':
    main()
