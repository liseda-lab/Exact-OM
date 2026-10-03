"""Failure-inclusive, explicitly migrated historical generator engineering replay.

No teacher labels, model fitting or gate passage is inferred by this stage.
The registered audit stage owns backend/solver evidence and the final gate audit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, canonical_json, promote_input_v3
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checked_checkpoint, checkpoint
from tools.repair.prepare import case_from_dict, case_to_dict


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def verify_binding(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('Historical evidence changed: ' + item['path'])
    return read(item['path'])


def migrate(record):
    """Preserve case/split/intent and axiom/candidate identities; explicit new schema."""
    case = case_from_dict(record)
    if case.split != 'development' or case.schema_revision != 'v2':
        raise ValueError('Historical replay requires archived v2 development cases')
    problem = promote_input_v3(case.problem)
    migrated = dataclasses.replace(case, problem=problem, schema_revision='v3')
    return case_to_dict(migrated), dict(
        adapter='promote_input_v3', source_case_hash=record['hash'],
        source_input_hash=case.problem.content_hash, migrated_input_hash=problem.content_hash,
        public_policy_signature_before=len(case.problem.policy.monitored_classes),
        public_policy_signature_after=len(problem.policy.monitored_classes),
        changes=['v3 record types', 'complete public candidate policy signature', 'migration_source_hash'],
        unchanged=['case_id', 'parent', 'split', 'axioms', 'candidates', 'budgets', 'probes', 'intent'],
        missing_control='historical initial omission only; no final-pool omission claim',
    )


def make_plan(preparation_path, report_path, protocol_path, legacy_train_path, qualification_path):
    preparation, report, protocol = map(read, (preparation_path, report_path, protocol_path))
    history = next(row for row in report['history'] if row['epoch'] == report['selected_epoch'])
    records = [row for row in preparation['cases'] if row['case']['split'] == 'development']
    cases = {row['case']['case_id']: row for row in records}
    generated = history['generated']
    if len(records) != 90 or len(cases) != 90 or set(cases) != set(generated):
        raise ValueError('Historical 90-case denominator differs from selected-epoch generation')
    counts = Counter(row['status'] for row in generated.values())
    if counts != {'generation_error': 79, 'generated': 11}:
        raise ValueError('Historical 79-failure reference changed')
    settings = report['settings']
    if (settings['max_depth'], settings['max_constructors'], settings['compile_seconds'],
        settings['max_circuit_nodes'], settings['decode_seconds']) != (2, 2, 20, 1000000, 300):
        raise ValueError('Unexpected historical language or stage settings')
    if 'seconds=seconds * 0.8' not in Path(legacy_train_path).read_text():
        raise ValueError('Original generation deadline formula not found')
    retrieval = next(iter(report['retrieval'].values()))['config']
    if any(row['config'] != retrieval for row in report['retrieval'].values()):
        raise ValueError('Historical retrieval settings differ by case')
    rows = []
    for case_id in sorted(cases):
        migrated, receipt = migrate(cases[case_id])
        rows.append(dict(case_id=case_id, family=cases[case_id]['case']['family'],
                         parent=cases[case_id]['case']['structural_parent'],
                         original_case=cases[case_id], migrated_case=migrated,
                         migration=receipt, historical=generated[case_id]))
    return dict(
        schema='exact-repair/historical-generation-plan/v1',
        sources=[binding(p) for p in (preparation_path, report_path, protocol_path, legacy_train_path)],
        qualification=binding(qualification_path), expected_cases=90, expected_rows=180,
        historical_status_counts=dict(counts), rows=rows, modes=['cold', 'warm'], seed=13,
        per_case_seconds=settings['decode_seconds'] * 0.8,
        per_case_memory_mb=protocol['resources']['memory_mb'],
        generation=dict(retrieval_config=retrieval, draws_per_object=settings['development_draws_per_object'],
                        candidate_cap=settings['candidate_cap'], max_depth=settings['max_depth'],
                        max_constructors=settings['max_constructors'], compile_seconds=settings['compile_seconds'],
                        max_circuit_nodes=settings['max_circuit_nodes'],
                        max_graph_nodes=settings['max_graph_nodes'], max_graph_edges=settings['max_graph_edges'],
                        max_explanations=settings['max_explanations'], max_text_tokens=settings['max_text_tokens'],
                        pair_factor_limit_per_object=settings['pair_factor_limit_per_object'],
                        quantization_scale=settings['quantization_scale'],
                        contextual_filtering=False, factored=True, proposal_arm='grammar_uniform',
                        vtree_type='balanced'),
        model=dict(encoder='none', hidden_dim=8, heads=2, layers=0, dropout=0,
                   revision='v3', pairwise=False, plan_risk=False),
        scope='Production uniform generator engineering replay; no neural performance or semantic quality comparison',
        changes_from_archive=['explicit v3 migration', 'current factored compiler and production generator',
                              'uniform proposal control with disposable untrained scoring model',
                              'persistent per-case cold/warm cache', 'current protected elementary actions'],
        model_role='Uniform proposals ignore weights; generated objective is discarded, no fitting or checkpoints',
        scientific_limits='Original generation allowance 0.8*300=240s, 20s compiler aggregate per object, 1M nodes per family; no widening of grammar',
        next_stage='xr21-expanded-historical-audit-001', study_complete=False,
    )


def generate(record, settings, model_config, cache, seed):
    import torch
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    case = case_from_dict(record)
    if case.split != 'development' or case.schema_revision != 'v3':
        raise ValueError('Only explicitly migrated development cases may be replayed')
    options = dict(settings)
    retrieval = RetrievalConfig(**options.pop('retrieval_config'))
    preparation = EffectivePreparation(
        max_graph_nodes=options['max_graph_nodes'], max_graph_edges=options['max_graph_edges'],
        max_explanations=options['max_explanations'], max_text_tokens=options['max_text_tokens'],
        pair_factor_limit_per_object=options['pair_factor_limit_per_object'],
        retrieval_config=retrieval, revision='v3', pair_max_pairs=None, pair_max_factors=None)
    graph = preparation.graph(case.problem, retrieve_vocabulary(case.problem, config=retrieval))
    model = RepairModel(graph.metadata, **model_config)
    frozen = freeze_neural_round(case.problem, model, graph=graph, retrieval_config=retrieval,
                                 compiler_cache_directory=cache, seed=seed, **options)
    reports = [dict(row) for row in frozen.proposal_reports]
    # Preserve production telemetry and pool, while avoiding large duplicated sample records.
    for report in reports:
        samples = report.pop('samples', ())
        report['sampled_families'] = dict(Counter(sample['family'] for sample in samples))
        report['samples_hash'] = canonical_hash(samples)
    partial = any(row.get('distribution_scope') != 'declared_language' for row in reports)
    return json.loads(canonical_json(dict(
        status='partial' if partial else 'generated', reports=reports,
        generated_input=frozen.problem.to_dict(),
        scope='generated pool only; objective discarded and no semantic verification claimed',
        original_inventory_retained=[
            dict(object_id=old.object_id, original=len(old.candidates), generated=len(new.candidates),
                 retained=len({c.candidate_id for c in old.candidates} & {c.candidate_id for c in new.candidates}))
            for old, new in zip(case.problem.objects, frozen.problem.objects)
        ])))


def run(plan_path, output, start, stop):
    from exact.repair.study import runtime_manifest
    from exact.repair.workers import bounded_call

    plan_path, output = Path(plan_path), Path(output)
    plan = read(plan_path)
    for source in plan['sources']:
        # Legacy source code is a text binding, not a JSON document.
        if sha(source['path']) != source['sha256']:
            raise ValueError('Historical source binding changed')
    verify_binding(plan['qualification'])
    if not 0 <= start < stop <= len(plan['rows']):
        raise ValueError('Invalid frozen case shard')
    runtime = runtime_manifest()
    identity = canonical_hash((sha(plan_path), sha(__file__), runtime))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in plan['rows'][start:stop]:
        key = canonical_hash(item['case_id'])
        for mode in plan['modes']:
            receipt = output/'rows'/(key+'-'+mode+'.json')
            row_identity = canonical_hash((identity, item['migrated_case']['hash'], mode))
            row = checked_checkpoint(receipt, row_identity)
            if row is None:
                attempts = output/'cache'/key
                attempts.mkdir(parents=True, exist_ok=True)
                if mode == 'cold':
                    cache = attempts/('population-'+str(len(list(attempts.iterdir()))+1))
                    cache.mkdir()
                else:
                    cold = checked_checkpoint(output/'rows'/(key+'-cold.json'),
                        canonical_hash((identity, item['migrated_case']['hash'], 'cold')))
                    cache = Path(cold['cache_directory'])
                write_artifact(output/'progress.json', dict(case_id=item['case_id'], mode=mode,
                    completed_rows=len(rows), shard=[start, stop], stage='generation'))
                began = time.monotonic()
                result = bounded_call(generate, item['migrated_case'], plan['generation'], plan['model'],
                    str(cache), plan['seed'], timeout=plan['per_case_seconds'], memory_mb=plan['per_case_memory_mb'])
                row = checkpoint(receipt, row_identity, case_id=item['case_id'], family=item['family'],
                    parent=item['parent'], historical=item['historical'], migration=item['migration'],
                    mode=mode, cache_directory=str(cache), call_status=result.status, detail=result.detail,
                    result=result.value, elapsed_seconds=time.monotonic()-began,
                    resources=dict(result.resource_usage), cleanup_complete=result.cleanup_complete)
            if not row['cleanup_complete']:
                raise RuntimeError('Worker cleanup incomplete; ownership audit required')
            rows.append(dict(**binding(receipt), case_id=item['case_id'], family=item['family'], mode=mode,
                formerly_failed=item['historical']['status']=='generation_error',
                status=(row.get('result') or {}).get('status', row['call_status'])))
    report = checkpoint(output/f'report-{start:03d}-{stop:03d}.json', identity,
        schema='exact-repair/historical-generation-shard/v1', status='complete',
        scheduled_cases=stop-start, scheduled_rows=(stop-start)*len(plan['modes']), recorded_rows=len(rows),
        rows=rows, counts=dict(Counter(row['status'] for row in rows)),
        family_counts={family:dict(Counter(row['status'] for row in rows if row['family']==family))
                       for family in sorted({row['family'] for row in rows})},
        plan=binding(plan_path), runtime=runtime, study_complete=False,
        gates={g:'not_established_requires_registered_audit' for g in ('G0','G1','G2')},
        next_stage=plan['next_stage'])
    write_artifact(output/'progress.json', dict(stage='shard_complete', shard=[start,stop], recorded_rows=len(rows)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    args = parser.parse_args()
    run(args.plan, args.output, args.start, args.stop)


if __name__ == '__main__':
    main()
