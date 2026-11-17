"""Generate label-free native evidence with source-boundary resume."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin

from exact.utils.provenance import sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def primary_classes(path, prefixes):
    """Read explicitly declared primary-document classes, excluding disabled entities.

    This RDF/XML adapter deliberately retains imported entities only as context.
    Prefixes are supplied by the dataset owner, independently of reference labels.
    """
    rdf = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    owl = '{http://www.w3.org/2002/07/owl#}'
    entities, excluded = set(), set()
    depth, root = 0, None
    bases = [Path(path).resolve().as_uri()]
    for event, element in ET.iterparse(path, events=('start', 'end')):
        if event == 'start':
            depth += 1
            bases.append(urljoin(bases[-1], element.get('{http://www.w3.org/XML/1998/namespace}base', '')))
            if root is None:
                root = element
                if root.tag != rdf + 'RDF':
                    raise ValueError('Scoped global scoring requires RDF/XML ontology documents')
            continue
        if depth == 2:
            iri = element.get(rdf + 'about')
            if iri is None and element.get(rdf + 'ID'):
                iri = '#' + element.get(rdf + 'ID')
            if iri is not None:
                iri = urljoin(bases[-1], iri)
            if iri and any(iri.startswith(p) for p in prefixes):
                if element.tag == owl + 'Class' or any(
                    c.tag == rdf + 'type' and c.get(rdf + 'resource') == owl[1:-1] + 'Class'
                    for c in element
                ):
                    entities.add(iri)
                for child in element:
                    name = child.tag.rsplit('}', 1)[-1].lower()
                    value = (child.text or '').strip().lower()
                    if (child.tag == owl + 'deprecated' and value in {'1', 'true'}) or (
                        name in {'use_in_alignment', 'useinalignment'} and value in {'0', 'false'}
                    ):
                        excluded.add(iri)
            element.clear()
            root.remove(element)
        depth -= 1
        bases.pop()
    return entities - excluded


def _binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(Path(path)))


def _verify_outputs(root, outputs):
    for name, digest in outputs.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f'Checkpoint output changed: {name}')


def _encoder_bindings(config):
    """Pin remote models and bind every local model/tokenizer byte on resume."""
    params = config['pipeline'][0]['params']
    models = [(config['candidates']['encoder'], config['candidates'].get('encoder_revision'))]
    for kind in ('lexical', 'context'):
        if params.get('use_' + kind, True):
            models.append((params.get(kind + '_model_name'), params.get(kind + '_model_revision')))
    bindings = []
    for name, revision in models:
        if not name:
            raise ValueError('Enabled encoders need an explicit model name or local path')
        path = Path(name).expanduser()
        if path.is_dir():
            files = {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob('*'))
                     if p.is_file() and '.cache' not in p.parts}
            if not files:
                raise ValueError(f'Empty local encoder directory: {path}')
            bindings.append(dict(local_path=str(path.resolve()), files=files))
        else:
            if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
                raise ValueError(f'Native resumable scoring requires a 40-character revision for {name}')
            bindings.append(dict(model=name, revision=revision))
    return bindings


def score(source, target, config, output, *, candidates=None, mode='global', device='cpu',
          source_prefixes=(), target_prefixes=(), sources_per_shard=64):
    """Score original generated/local candidates without reference labels or decisions."""
    import torch
    from exact.core.entities.configs.config import ConfigModel
    from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
    from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
    from exact.impl.models.selector import CandidateSetSelector
    from exact.impl.trainer.fitting import source_batches
    from exact.native.evidence import (
        distinctive, primary_parameters, recompute_retrieval, resident_items, rows_from_output,
    )

    if sources_per_shard < 1:
        raise ValueError('sources_per_shard must be positive')
    if mode not in {'global', 'local'}:
        raise ValueError('mode must be global or local')
    if mode == 'global' and (candidates is not None or not source_prefixes or not target_prefixes):
        raise ValueError('Global scoring generates candidates and requires owning source/target prefixes')
    if mode == 'local' and candidates is None:
        raise ValueError('Local scoring requires the original public candidate TSV')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = ConfigModel.load_config(Path(config))
    effective = cfg.model_dump(mode='json', by_alias=True)
    if mode == 'global' and effective['matching']['entity_kinds'] != ['class']:
        raise ValueError('Scoped global scoring supports named classes; use exact align for other entity kinds')
    implementation = {str(p.relative_to(Path(__file__).parents[1])): sha256_file(p)
                      for p in sorted(Path(__file__).parents[1].rglob('*.py'))}
    from importlib.metadata import version
    identity = dict(source=_binding(source), target=_binding(target), config=_binding(config),
                    encoders=_encoder_bindings(effective),
                    dependencies={name: version(name) for name in ('transformers', 'numpy', 'pyowl-core', 'pyowl2vec-star-projector')},
                    candidates=_binding(candidates) if candidates else None, mode=mode,
                    source_prefixes=list(source_prefixes), target_prefixes=list(target_prefixes),
                    device=device, torch=torch.__version__, implementation=implementation,
                    sources_per_shard=sources_per_shard,
                    gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None)
    identity_path = root / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Inputs, code or execution settings changed; use a new output directory')
    write_json(identity_path, identity)
    if (root / 'complete.json').exists():
        receipt = json.loads((root / 'complete.json').read_text())
        _verify_outputs(root, receipt['outputs'])
        return receipt
    params = dict(effective['dataset'])
    params.update(output_path=root / 'dataset', verbalization_mode='deterministic',
                  device=torch.device(device), cardinality_1_to_many=mode == 'local',
                  entity_kinds=effective['matching']['entity_kinds'],
                  input_format=effective['io']['input_format'],
                  source_options=effective['io']['source_options'],
                  target_options=effective['io']['target_options'])
    ds = PairAdaptiveContextDataset(**params)
    ds.load_ontologies(Path(source), Path(target))
    if mode == 'global':
        sources = sorted(s for s, k in ds._source_entity_kind_index.items()
                         if k in ds.entity_kinds and s not in ds.source_ignored_alignment_classes)
    else:
        from exact.utils.data import read_table
        query_frame = read_table(Path(candidates))
        column = next((c for c in ('SrcEntity', 'Src', 'source') if c in query_frame), None)
        if column is None:
            raise ValueError('Candidate TSV requires a SrcEntity column')
        sources = sorted(set(query_frame[column].astype(str)))
    ds.freeze_source_universe(sources, cap=None, seed=cfg.seed)
    cp = root / 'candidate-checkpoint.json'
    if cp.exists():
        import pandas as pd
        saved = json.loads(cp.read_text())
        _verify_outputs(root, saved['outputs'])
        ds._candidates = pd.read_parquet(root / 'candidates.parquet')
        ds._exact_matches = pd.DataFrame(saved['anchors'])
        ds._candidate_pool_manifest = saved['manifest']
    else:
        options = dict(effective['candidates'])
        options['lexical_encoder_name'] = options.pop('encoder')
        options.pop('source_cap', None)
        ds.load_candidates(Path(candidates) if candidates else None, device=torch.device(device), **options)
        if mode == 'local':
            ds.restrict_benchmark_exact_matches()
        tmp = root / 'candidates.tmp.parquet'
        ds.candidates.to_parquet(tmp, index=False, compression='zstd')
        os.replace(tmp, root / 'candidates.parquet')
        write_json(cp, dict(anchors=[] if ds.exact_matches is None else ds.exact_matches.to_dict('records'),
                            manifest=ds.candidate_pool_manifest,
                            outputs={'candidates.parquet': sha256_file(root / 'candidates.parquet')}))
    frame = ds.candidates.reset_index(drop=True)
    generation_sources = sources
    anchors = [] if ds.exact_matches is None else ds.exact_matches.to_dict('records')
    anchor_pairs = {(str(a['Src']), str(a['Tgt'])) for a in anchors}
    if mode == 'global':
        owned_s = primary_classes(source, source_prefixes)
        owned_t = primary_classes(target, target_prefixes)
        sources = sorted(set(sources) & owned_s)
        # Score the original candidate groups first, then filter. Relative primitive
        # evidence and batch arithmetic must match the original unfiltered run.
        write_json(root / 'scope.json', dict(source=sources, target=sorted(owned_t),
                                            reference_used=False, filtering='after_original_retrieval'))
    if frame.duplicated(['Src', 'Tgt']).any():
        raise ValueError('Duplicate native candidate pairs')
    (root / 'sources.txt').write_text(''.join(s + '\n' for s in sources))
    anchors = [dict(a, S_final=1., S_pair_final=1.) for a in anchors
               if mode == 'local' or (a['Src'] in owned_s and a['Tgt'] in owned_t)]
    write_json(root / 'anchors.json', anchors)
    model_params = primary_parameters(effective, device=device)
    model_params.update(dataset_signature=ds.dataset_signature, cache_namespace='native-v2')
    model = PairAdaptiveSemanticScorer(**model_params)
    model.attach_dataset(ds)
    model.eval()
    selector = CandidateSetSelector()
    groups = frame.groupby('Src', sort=False).indices
    shards, pair_count = [], 0
    for start in range(0, len(generation_sources), sources_per_shard):
        group = generation_sources[start:start + sources_per_shard]
        shard = root / 'shards' / f'{start // sources_per_shard:07d}.jsonl'
        receipt_path = shard.with_suffix('.receipt.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['sources'] != group:
                raise ValueError('Checkpoint source membership changed')
            _verify_outputs(root, receipt['outputs'])
        else:
            indices = [int(i) for s in group for i in groups.get(s, [])]
            rows = []
            if indices:
                features = ds.get_features(frame.loc[indices]).reset_index(drop=True)
                for batch in source_batches(features, 64):
                    part = features.loc[batch]
                    with torch.no_grad():
                        values = model.forward(src_iris=part.Src.tolist(), tgt_iris=part.Tgt.tolist(),
                                               src_label_lists=part.SrcLabels.tolist(), tgt_label_lists=part.TgtLabels.tolist(),
                                               src_contexts=part.SrcCtx.tolist(), tgt_contexts=part.TgtCtx.tolist())
                    rows.extend(rows_from_output(part, values, facts=True))
            if mode == 'global':
                rows = [r for r in rows if r['Src'] in owned_s and r['Tgt'] in owned_t]
            by_source = {}
            for row in rows:
                row['selector_evidence_items'] = resident_items(row, selector)
                row['original_exact_anchor'] = (row['Src'], row['Tgt']) in anchor_pairs
                row['exact_anchor'] = bool(mode == 'local' and row['original_exact_anchor'])
                row.setdefault('S_pair_final', row['S_final'])
                by_source.setdefault(row['Src'], []).append(row)
            if mode == 'global':
                rows = []
                for source_rows in by_source.values():
                    scores = distinctive([r['selector_evidence_items'] for r in source_rows])
                    rows.extend(recompute_retrieval([dict(r, repair_distinctive=s) for r, s in zip(source_rows, scores)]))
            else:
                # Local exact anchors are native hard bypasses, not synthesized
                # scored evidence. Include them for the source ambiguity gate;
                # the local ranker removes them using the explicit sidecar.
                present = {(r['Src'], r['Tgt']) for r in rows}
                rows.extend(dict(a, S_base=1., U=0., exact_anchor=True,
                                 original_exact_anchor=True, selector_evidence_items=[])
                            for a in anchors if a['Src'] in group and (a['Src'], a['Tgt']) not in present)
                rows.sort(key=lambda r: (r['Src'], -r['S_base'], r['Tgt']))
            shard.parent.mkdir(parents=True, exist_ok=True)
            temp = shard.with_suffix('.tmp')
            with temp.open('w') as stream:
                for row in rows:
                    # These are the selected facts; retain both packets and selector evidence.
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, shard)
            receipt = dict(sources=group, rows=len(rows), outputs={str(shard.relative_to(root)): sha256_file(shard)})
            write_json(receipt_path, receipt)
        shards.append(shard)
        pair_count += receipt['rows']
        write_json(root / 'status.json', dict(status='running', sources_complete=start + len(group),
                                             sources_total=len(generation_sources), pairs_complete=pair_count))
    tmp = root / 'pairs.jsonl.tmp'
    with tmp.open('wb') as output_stream:
        import shutil
        for shard in shards:
            with shard.open('rb') as stream:
                shutil.copyfileobj(stream, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    os.replace(tmp, root / 'pairs.jsonl')
    outputs = {str(p.relative_to(root)): sha256_file(p) for p in [root / 'pairs.jsonl', root / 'sources.txt', root / 'anchors.json', *shards]}
    receipt = dict(status='complete', mode=mode, source_count=len(sources), candidate_count=pair_count,
                   outputs=outputs, reference_labels_loaded=False, provider_calls=0,
                   identity_sha256=sha256_file(identity_path))
    write_json(root / 'complete.json', receipt)
    write_json(root / 'status.json', receipt)
    return receipt
