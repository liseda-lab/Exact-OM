from pathlib import Path

import pytest

from exact import AlignmentRunner

NCIT_DIR = Path.cwd() / "data" / "ncit-doid"
TEST_DIR = Path.cwd() / "tests" / "test_data"

pytestmark = pytest.mark.requires_data


@pytest.fixture
def supervised_local_alignment_runner():
    source_ontology_file = str(NCIT_DIR / "ncit.owl")
    target_ontology_file = str(NCIT_DIR / "doid.owl")
    output_dir = str("tests" / "test_data" / "test_prompt")
    config_file = str("tests" / "test_data" / "test_prompt" / "config.yaml")
    reference_file = str("tests" / "test_data" / "train.tsv")
    candidates_file = str("tests" / "test_data" / "test.cands.tsv")
    full_reference_file = str("tests" / "test_data" / "test.tsv")
    memory = "32G"

    return AlignmentRunner(
        source_ontology_file=source_ontology_file,
        target_ontology_file=target_ontology_file,
        output_dir=output_dir,
        training_reference_file=reference_file,
        full_reference_file=full_reference_file,
        candidates_file=candidates_file,
        config_file=config_file,
        save_logs=True,
        jvm_heap_size=memory,
        run_eval=True,
        device=0,
    )


@pytest.fixture
def unsup_local_alignment_runner():
    source_ontology_file = str(NCIT_DIR / "ncit.owl")
    target_ontology_file = str(NCIT_DIR / "doid.owl")
    output_dir = str("tests" / "test_data" / "test_prompt")
    config_file = str("tests" / "test_data" / "test_prompt" / "config.yaml")
    candidates_file = str("tests" / "test_data" / "test.cands.tsv")
    full_reference_file = str("tests" / "test_data" / "test.tsv")
    memory = "32G"

    return AlignmentRunner(
        source_ontology_file=source_ontology_file,
        target_ontology_file=target_ontology_file,
        output_dir=output_dir,
        full_reference_file=full_reference_file,
        candidates_file=candidates_file,
        config_file=config_file,
        save_logs=True,
        jvm_heap_size=memory,
        run_eval=True,
        device=0,
    )


@pytest.fixture
def supervised_global_alignment_runner():
    source_ontology_file = str(NCIT_DIR / "ncit.owl")
    target_ontology_file = str(NCIT_DIR / "doid.owl")
    output_dir = str("tests" / "test_data" / "test_prompt")
    config_file = str("tests" / "test_data" / "test_prompt" / "config.yaml")
    reference_file = str("tests" / "test_data" / "train.tsv")
    full_reference_file = str("tests" / "test_data" / "test.tsv")
    memory = "32G"

    return AlignmentRunner(
        source_ontology_file=source_ontology_file,
        target_ontology_file=target_ontology_file,
        output_dir=output_dir,
        training_reference_file=reference_file,
        full_reference_file=full_reference_file,
        config_file=config_file,
        save_logs=True,
        jvm_heap_size=memory,
        run_eval=True,
        device=0,
    )


@pytest.fixture
def unsup_global_alignment_runner():
    source_ontology_file = str(NCIT_DIR / "ncit.owl")
    target_ontology_file = str(NCIT_DIR / "doid.owl")
    output_dir = str("tests" / "test_data" / "test_prompt")
    config_file = str("tests" / "test_data" / "test_prompt" / "config.yaml")
    full_reference_file = str("tests" / "test_data" / "test.tsv")
    memory = "32G"

    return AlignmentRunner(
        source_ontology_file=source_ontology_file,
        target_ontology_file=target_ontology_file,
        output_dir=output_dir,
        full_reference_file=full_reference_file,
        config_file=config_file,
        save_logs=True,
        jvm_heap_size=memory,
        run_eval=True,
        device=0,
    )


def test_local_supervised(supervised_local_alignment_runner):
    try:
        supervised_local_alignment_runner.run()
    except Exception as e:
        pytest.fail(f"supervised_local_alignment_runner.run() raised an exception: {e}")


def test_local_unsupervised(unsup_local_alignment_runner):
    try:
        unsup_local_alignment_runner.run()
    except Exception as e:
        pytest.fail(f"unsup_local_alignment_runner.run() raised an exception: {e}")


def test_global_supervised(supervised_global_alignment_runner):
    try:
        supervised_global_alignment_runner.run()
    except Exception as e:
        pytest.fail(f"supervised_global_alignment_runner.run() raised an exception: {e}")


def test_global_unsupervised(unsup_global_alignment_runner):
    try:
        unsup_global_alignment_runner.run()
    except Exception as e:
        pytest.fail(f"unsup_global_alignment_runner.run() raised an exception: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
