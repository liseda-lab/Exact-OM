
import subprocess
from pathlib import Path
import os



DATA_DIR = Path.cwd() / "data"
NCIT_DIR = DATA_DIR / "ncit-doid"
EXP_DIR = Path.cwd() / "exp"

def main():
    source_ontology_file = str(NCIT_DIR / "ncit.owl")
    target_ontology_file = str(NCIT_DIR / "doid.owl")
    output_dir = str(EXP_DIR / "test_tune")
    config_file = str(EXP_DIR / "test_tune" / "config_tune.yaml")
    reference_file = str(NCIT_DIR / "train.tsv")
    candidates_file = str(NCIT_DIR / "test.cands.tsv")
    full_reference_file = str(NCIT_DIR / "test.tsv")
    memory = "30G"
    devices = [0]
    max_combinations = 2

    from exact import TuningAlignmentRunner

    runner = TuningAlignmentRunner(
        source_ontology_file=source_ontology_file,
        target_ontology_file=target_ontology_file,
        output_dir=output_dir,
        reference_file=reference_file,
        full_reference_file=full_reference_file,
        candidates_file=candidates_file,
        config_file=config_file,
        save_logs=True,
        jvm_heap_size=memory,
        devices=devices,
        max_combinations=max_combinations
    )
    runner.run()


if __name__ == "__main__":
    main()