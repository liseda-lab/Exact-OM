"""Runtime fitted heads are explicit, verified deployment inputs."""
import json
from types import SimpleNamespace
import pytest
from exact.utils.frozen_inference import freeze_runtime_fitted_artifacts, inject_runtime_fitted_artifacts
from exact.utils.provenance import dataset_signature_for_paths


def fixture(tmp_path):
    source, target = tmp_path / 'source.owl', tmp_path / 'target.owl'
    source.write_text('source'); target.write_text('target')
    config = tmp_path / 'config.yaml'; config.write_text('saved before fitting')
    head = tmp_path / 'head.json'; head.write_text('{"weights": [1]}')
    trainer = SimpleNamespace(output_dir=tmp_path,
        dataset=SimpleNamespace(dataset_signature=dataset_signature_for_paths(source, target)),
        model=SimpleNamespace(), models=[SimpleNamespace(), SimpleNamespace(rerank_config={'artifact': str(head)})])
    freeze_runtime_fitted_artifacts(trainer)
    return config, source, target, head, tmp_path / 'fitting/deployment-artifacts.json'


def test_runtime_receipt_injects_exact_head_and_rejects_mutation(tmp_path):
    config, source, target, head, receipt = fixture(tmp_path)
    mapping = {'selector': {'rerank': {'artifact': None}}}
    inject_runtime_fitted_artifacts(mapping, receipt, selected_config=config, source=source, target=target)
    assert mapping['selector']['rerank']['artifact'] == str(head)
    head.write_text('{"weights": [2]}')
    with pytest.raises(ValueError, match='changed'):
        inject_runtime_fitted_artifacts(mapping, receipt, selected_config=config, source=source, target=target)


def test_receipt_rejects_changed_recipe_and_ontology_pair(tmp_path):
    config, source, target, head, receipt = fixture(tmp_path)
    target.write_text('another ontology')
    with pytest.raises(ValueError, match='another recipe or ontology pair'):
        inject_runtime_fitted_artifacts({}, receipt, selected_config=config, source=source, target=target)


def test_receipt_cannot_inject_scientific_settings(tmp_path):
    config, source, target, head, receipt = fixture(tmp_path)
    data = json.loads(receipt.read_text())
    data['artifacts']['matching.threshold'] = data['artifacts'].pop('selector.rerank.artifact')
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Invalid'):
        inject_runtime_fitted_artifacts({}, receipt, selected_config=config, source=source, target=target)
