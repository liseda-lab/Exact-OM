import pytest
from pathlib import Path
from exact.delivery.api.eval import EvalutionRunner

DATA_DIR = Path.cwd() / "data"
NCIT_DIR = DATA_DIR / "ncit-doid"
EXP_DIR = Path.cwd() / "exp"

@pytest.fixture
def eval_runner():
    source_ontology_file = NCIT_DIR / "ncit.owl"
    target_ontology_file = NCIT_DIR / "doid.owl"
    output_dir = EXP_DIR / "test"
    reference_file = NCIT_DIR / "train.tsv"
    full_reference_file = NCIT_DIR / "test.tsv"
    candidates_file = str(NCIT_DIR / "test.cands.tsv")
    alignment_file_path = EXP_DIR / "test" / "model" / 'alignment' / "src2tgt.maps_local.tsv"
    memory = "32G"
    k = [1, 5, 10]
    error_on_fail = False

    return EvalutionRunner(
        source_ontology_file=str(source_ontology_file),
        target_ontology_file=str(target_ontology_file),
        output_dir=str(output_dir),
        reference_file=str(reference_file),
        full_reference_file=str(full_reference_file),
        candidates_file=candidates_file,
        alignment_file_path=str(alignment_file_path),
        memory=memory,
        k=k,
        error_on_fail=error_on_fail
    )

def test_eval_runner_run(eval_runner):
    try:
        eval_runner.run()
    except Exception as e:
        pytest.fail(f"EvalutionRunner.run() raised an exception: {e}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
