import copy

import pytest

from tools.prepared_batch import binding
from tools.recover_hosted_spending import _ALLOWED, verify_code
from tools.recover_training_retention import verify_identity


def _sources(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        for name in _ALLOWED | {"exact/impl/scorer.py", "exact/experiments/reporting.py"}:
            if root == old and name == "exact/utils/hosted_spending.py":
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unchanged = True\n")
    for name in _ALLOWED:
        (new / name).write_text("accounting = True\n")
    repair = dict(
        migration="hosted-spending-checkpoint-v1",
        scientific_choices_unchanged=True,
        reporting_labels_exposed=False,
        changes={
            name: dict(
                before=binding(old / name)["sha256"] if (old / name).exists() else None,
                after=binding(new / name)["sha256"],
            )
            for name in _ALLOWED
        },
    )
    return old, new, repair


def test_only_exact_bound_administrative_changes_can_migrate(tmp_path):
    old, new, repair = _sources(tmp_path)
    assert (
        verify_code(old, new, repair)["files"]["exact/impl/scorer.py"]
        == binding(old / "exact/impl/scorer.py")["sha256"]
    )
    (new / "exact/llm/ledger.py").write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="source hashes"):
        verify_code(old, new, repair)


@pytest.mark.parametrize("name", ["exact/impl/scorer.py", "exact/experiments/reporting.py"])
def test_unrelated_scientific_or_reporting_changes_reject_reuse(tmp_path, name):
    old, new, repair = _sources(tmp_path)
    (new / name).write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="outside"):
        verify_code(old, new, repair)


@pytest.mark.parametrize(
    "field", ["inputs", "parameters", "parents", "dependencies", "seed", "role"]
)
def test_checkpoint_cannot_cross_input_config_seed_or_role(field):
    original = dict(
        artifact_id="old",
        implementation={"files": {"old": "sha"}},
        inputs={"data": "sha"},
        parameters={"cap": 300},
        parents=["input"],
        dependencies={"torch": "version"},
        seed=17,
        role="development",
    )
    expected = copy.deepcopy(original)
    expected.update(artifact_id="new", implementation={"files": {"new": "sha"}})
    verify_identity(original, expected, original["implementation"], stage="extraction")
    expected[field] = "changed"
    with pytest.raises(ValueError, match="identity differs"):
        verify_identity(original, expected, original["implementation"], stage="extraction")
