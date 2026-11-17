"""Independent native CLI qualification with real, tiny offline encoders."""
import json
from pathlib import Path

import pytest

from exact.native.evidence import distinctive, recompute_retrieval
from exact.native.score import primary_classes


def test_primary_scope_excludes_context_deprecated_and_disabled(tmp_path):
    path = tmp_path / 'scope.owl'
    path.write_text('''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:x="urn:annotation:">
      <owl:Class rdf:about="urn:s:included"/>
      <owl:Class rdf:about="urn:context:excluded"/>
      <owl:Class rdf:about="urn:s:deprecated"><owl:deprecated>true</owl:deprecated></owl:Class>
      <owl:Class rdf:about="urn:s:disabled"><x:use_in_alignment>false</x:use_in_alignment></owl:Class>
      <rdf:Description rdf:about="urn:s:typed"><rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Class"/></rdf:Description>
    </rdf:RDF>''')
    assert primary_classes(path, ['urn:s:']) == {'urn:s:included', 'urn:s:typed'}


def test_primary_scope_resolves_rdf_id_and_base_relative_iris(tmp_path):
    path = tmp_path / 'relative.owl'
    path.write_text('''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:owl="http://www.w3.org/2002/07/owl#" xml:base="https://example.org/ontology">
      <owl:Class rdf:ID="Heart"/>
      <owl:Class rdf:about="#Lung"/>
    </rdf:RDF>''')
    assert primary_classes(path, ['https://example.org/ontology#']) == {
        'https://example.org/ontology#Heart', 'https://example.org/ontology#Lung'}


def test_removed_candidates_recompute_retrieval_and_distinctive_population():
    rows = [{'Src': 's', 'Tgt': 'a', 'cand_sim': .8}, {'Src': 's', 'Tgt': 'b', 'cand_sim': .2}]
    before = recompute_retrieval(rows)
    after = recompute_retrieval(rows[:1])
    assert [x['Tgt'] for x in before] == ['a', 'b']
    assert before[0]['cand_sim_prob'] == pytest.approx(.8)
    assert after[0]['cand_sim_prob'] == 1 and after[0]['cand_sim_src_mean'] == .8
    items = [[{'key': 'shared', 'strength': 1}, {'key': 'unique', 'strength': .5}],
             [{'key': 'shared', 'strength': 1}]]
    assert distinctive(items) == [1., 0.]
    assert distinctive(items[:1]) == [0.]


@pytest.fixture
def offline_native(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from exact.core.entities.configs.config import ConfigModel
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model_dir = tmp_path / 'encoder'
    model_dir.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'heart', 'lung', 'organ', 'tissue', 'disease', 'source', 'target', 'parent']
    (model_dir / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    BertTokenizerFast(vocab_file=str(model_dir / 'vocab.txt'), do_lower_case=True).save_pretrained(model_dir)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32, max_position_embeddings=512)).save_pretrained(model_dir)
    cfg = ConfigModel().model_dump(mode='json', by_alias=True)
    cfg['dataset']['num_workers'] = 1
    cfg['dataset']['verbalization_mode'] = 'deterministic'
    cfg['candidates'].update(encoder=str(model_dir), encoder_revision=None,
                             top_k=3, use_amp=False, retrieval_strategy='hybrid')
    cfg['pipeline'][0]['params'].update(lexical_model_name=str(model_dir), context_model_name=str(model_dir),
                                      lexical_model_revision=None, context_model_revision=None,
                                      fp16_inference=False, use_llm=False, cache_dir=str(tmp_path/'cache'))
    config = tmp_path / 'config.yaml'
    config.write_text(json.dumps(cfg))
    source, target = tmp_path / 'source.owl', tmp_path / 'target.owl'
    template = '''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#">
      <owl:Ontology rdf:about="urn:{side}:ontology"/>
      <owl:Class rdf:about="urn:{side}:heart"><rdfs:label>heart</rdfs:label><rdfs:subClassOf rdf:resource="urn:{side}:organ"/></owl:Class>
      <owl:Class rdf:about="urn:{side}:lung"><rdfs:label>lung {side}</rdfs:label><rdfs:subClassOf rdf:resource="urn:{side}:organ"/></owl:Class>
      <owl:Class rdf:about="urn:{side}:organ"><rdfs:label>organ</rdfs:label></owl:Class>
      <owl:Class rdf:about="urn:context:{side}"><rdfs:label>disease</rdfs:label></owl:Class>
    </rdf:RDF>'''
    source.write_text(template.format(side='s'));target.write_text(template.format(side='t'))
    candidates = tmp_path / 'candidates.tsv'
    candidates.write_text("SrcEntity\tTgtEntity\tTgtCandidates\n"
                          "urn:s:heart\turn:t:heart\t['urn:t:lung', 'urn:t:heart', 'urn:t:heart']\n"
                          "urn:s:heart\turn:PRIVATE:GOLD\t['urn:t:organ', 'urn:t:heart']\n"
                          "urn:s:lung\turn:t:lung\t['urn:t:heart', 'urn:t:lung']\n")
    return dict(source=source,target=target,config=config,candidates=candidates,model=model_dir,root=tmp_path)


@pytest.mark.parametrize('mode', ['local', 'global'])
def test_real_offline_native_cli_and_decision_plan(offline_native, mode):
    from exact.delivery.cli.score import main
    from exact.decisions.pipeline import prepare
    from exact.acceptance.features import RAW
    f=offline_native; output=f['root']/mode
    argv=['--source',str(f['source']),'--target',str(f['target']),'--config',str(f['config']),
          '--output',str(output),'--mode',mode,'--device','cpu','--sources-per-shard','1']
    if mode=='local':argv+=['--candidates',str(f['candidates'])]
    else:argv+=['--source-prefix','urn:s:','--target-prefix','urn:t:']
    assert main(argv)==0
    rows=[json.loads(x) for x in (output/'pairs.jsonl').read_text().splitlines()]
    assert rows and len({(r['Src'],r['Tgt']) for r in rows})==len(rows)
    assert all(r['Src'].startswith('urn:s:') and r['Tgt'].startswith('urn:t:') for r in rows)
    assert 'urn:PRIVATE:GOLD' not in (output/'pairs.jsonl').read_text()
    assert all('llm_structured_evidence_packet' in r and 'llm_evidence_packet' in r for r in rows)
    if mode=='local':
        expected={('urn:s:heart','urn:t:heart'),('urn:s:heart','urn:t:lung'),('urn:s:heart','urn:t:organ'),
                  ('urn:s:lung','urn:t:heart'),('urn:s:lung','urn:t:lung')}
        assert {(r['Src'],r['Tgt']) for r in rows}==expected
        assert next(r for r in rows if r['Src']=='urn:s:heart' and r['Tgt']=='urn:t:heart')['exact_anchor']
    else:
        assert set(RAW)<=set(rows[0]),set(RAW)-set(rows[0])
        assert all(not r['exact_anchor'] for r in rows)
    plan=prepare(output/'pairs.jsonl',output/'sources.txt',output/'decision-plan',policy='global' if mode=='global' else 'local-application')
    assert json.loads(plan.read_text())['request_count'] <= 5
    before=(output/'pairs.jsonl').read_bytes()
    receipt=json.loads((output/'complete.json').read_text())
    assert receipt['provider_calls']==0 and not receipt['reference_labels_loaded']
    assert main(argv)==0 and (output/'pairs.jsonl').read_bytes()==before
    # A source-boundary restart reuses completed shard content after a process interruption.
    (output/'complete.json').unlink()
    assert main(argv)==0 and (output/'pairs.jsonl').read_bytes()==before
    # The scored native artifact must be directly consumable downstream.
    if mode == 'global':
        from exact.decisions.pipeline import run as decide, apply as fuse
        from exact.delivery.cli.acceptance import main as acceptance
        class OfflineDecisions:
            def predict(self, payload):
                return {'probability': .8, 'local_model_call_attempted': True}
        decide(plan, output/'answers.sqlite', engine=OfflineDecisions())
        fused=output/'fused.jsonl'
        fuse(output/'pairs.jsonl', plan, output/'answers.sqlite', fused)
        train=output/'train.tsv'
        train.write_text('SrcEntity\tTgtEntity\nurn:s:heart\turn:t:heart\nurn:s:lung\turn:t:lung\n')
        assert acceptance(['fit','--candidates',str(fused),'--train',str(train),'--output',str(output/'model'),
                           '--folds','2','--family','logistic']) == 0
        for policy in ('full','partial'):
            assert acceptance(['apply','--candidates',str(fused),'--model',str(output/'model'),
                               '--policy',policy,'--output',str(output/policy)]) == 0
            assert (output/policy/'alignment.rdf').is_file()


def test_native_checkpoint_rejects_changed_local_encoder_weights(offline_native):
    from exact.delivery.cli.score import main
    from transformers import BertModel
    import torch
    f = offline_native
    argv = ['--source', str(f['source']), '--target', str(f['target']), '--config', str(f['config']),
            '--output', str(f['root']/'bound'), '--mode', 'local', '--candidates', str(f['candidates'])]
    assert main(argv) == 0
    model = BertModel.from_pretrained(f['model'], local_files_only=True)
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    model.save_pretrained(f['model'])
    with pytest.raises(ValueError, match='changed|identity|new output'):
        main(argv)


def test_owned_scope_does_not_change_original_pair_primitives(offline_native):
    from exact.delivery.cli.score import main
    f = offline_native
    config = json.loads(f['config'].read_text())
    config['candidates']['top_k'] = 4  # Include the context target before ownership filtering.
    f['config'].write_text(json.dumps(config))
    common = ['--source',str(f['source']),'--target',str(f['target']),'--config',str(f['config']),
              '--mode','global','--source-prefix','urn:s:','--device','cpu']
    outputs = []
    for name, prefix in [('narrow','urn:t:'),('broad','urn:')]:
        out = f['root']/name
        assert main(common+['--target-prefix',prefix,'--output',str(out)]) == 0
        outputs.append({(r['Src'],r['Tgt']):r for r in
                        map(json.loads,(out/'pairs.jsonl').read_text().splitlines())})
    narrow,broad = outputs
    assert set(narrow)<set(broad)
    for key, row in narrow.items():
        for field in ('S_base','U','s_label','S_struct','s_hier','s_sim','s_diff','s_attr'):
            assert row[field] == broad[key][field], (key,field,row[field],broad[key][field])


def test_rank_publication_rejects_changed_candidate_multiplicity(tmp_path):
    from exact.delivery.cli.rank import write_rankings
    queries = [dict(qid=0, source='s', candidates=['a','a','b'])]
    result = {'queries':[dict(qid=0, source='s', candidates=[dict(target=t) for t in ['a','b','b']])]}
    with pytest.raises(ValueError, match='membership|permutation|multiplicity'):
        write_rankings(tmp_path/'rankings.tsv', result, queries)


def _write_local_evaluation(path, rows, *, labelled=False):
    import csv
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['SrcEntity', 'TgtEntity', 'TgtCandidates'] if labelled
                        else ['SrcEntity', 'TgtCandidates'])
        writer.writerows(rows)


def test_public_rankings_evaluate_without_gold_or_query_collapse(tmp_path):
    from exact.delivery.cli.eval import main
    from exact.core.entities.evaluation import EvaluationRequest
    from exact.impl.evaluators.bioml import _UpstreamBioMLApi
    from exact.utils.provenance import sha256_file
    ranked, refs, output = tmp_path/'rankings.tsv', tmp_path/'test.cands.tsv', tmp_path/'eval'
    _write_local_evaluation(ranked, [('same', "['b', 'a']"), ('same', "['b', 'c']")])
    _write_local_evaluation(refs, [('same', 'b', "['a', 'b']"),
                                   ('same', 'c', "['c', 'b']")], labelled=True)
    original = ranked.read_bytes()
    assert main(['-a', str(ranked), '-c', str(refs), '-o', str(output), '-e']) == 0
    report = json.loads((output/'evaluation_results.json').read_text())
    # Independent expected ranks are 1 and 2; source de-duplication is invalid.
    assert report['builtin']['MRR'] == .75
    assert report['builtin']['Hits@1'] == .5
    assert report['meta']['refs']['alignment']['sha256'] == sha256_file(ranked)
    assert report['meta']['refs']['reference_candidates']['sha256'] == sha256_file(refs)
    assert ranked.read_bytes() == original and b'TgtEntity' not in original
    inputs = _UpstreamBioMLApi._ranking_inputs(
        EvaluationRequest(alignment=ranked, reference_candidates=refs))
    assert inputs == ({0: ['b', 'a'], 1: ['b', 'c']}, {0: 'b', 1: 'c'})


@pytest.mark.parametrize('rows, message', [
    ([('s', "['a', 'b', 'b']")], 'multiplicity'),
    ([('different', "['a', 'a', 'b']")], 'source'),
    ([], 'counts'),
    ([('s', "('a', 'a', 'b')")], 'list'),
])
def test_public_ranking_evaluation_rejects_changed_queries(tmp_path, rows, message):
    from exact.utils.eval import MetricUtils
    from exact.core.entities.evaluation import EvaluationRequest
    from exact.impl.evaluators.bioml import _UpstreamBioMLApi
    ranked, refs = tmp_path/'rankings.tsv', tmp_path/'test.cands.tsv'
    _write_local_evaluation(ranked, rows)
    _write_local_evaluation(refs, [('s', 'a', "['a', 'a', 'b']")], labelled=True)
    with pytest.raises(ValueError, match=message):
        MetricUtils.read_candidate_mappings(str(ranked), str(refs))
    with pytest.raises(ValueError, match=message):
        _UpstreamBioMLApi._ranking_inputs(
            EvaluationRequest(alignment=ranked, reference_candidates=refs))


def test_public_ranking_evaluation_keeps_missing_and_empty_reference_queries(tmp_path):
    from exact.impl.evaluators.builtin import BuiltinEvaluator
    ranked, refs = tmp_path/'rankings.tsv', tmp_path/'test.cands.tsv'
    _write_local_evaluation(ranked, [('s', "['a', 'a', 'b']"), ('empty', '[]')])
    _write_local_evaluation(refs, [('s', 'b', "['b', 'a', 'a']"),
                                   ('empty', 'missing', '[]')], labelled=True)
    result = BuiltinEvaluator.local_eval(ranked, refs, K=[1, 3])
    assert result == {'MRR': round(1/6, 3), 'Hits@1': 0., 'Hits@3': .5}
