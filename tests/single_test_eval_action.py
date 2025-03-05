import subprocess
from pathlib import Path
import os



DATA_DIR = Path.cwd() / "data"
NCIT_DIR = DATA_DIR / "ncit-doid"
EXP_DIR = Path.cwd() / "exp"

def main():
    source_ontology_file = NCIT_DIR / "ncit.owl"
    target_ontology_file = NCIT_DIR / "doid.owl"
    output_dir = EXP_DIR / "test"
    reference_file = NCIT_DIR / "train.tsv"
    full_reference_file = NCIT_DIR / "test.tsv"
    candidates_file = str(NCIT_DIR / "test.cands.tsv")
    alignment_file_path = EXP_DIR / "test" / "model" / 'alignment' / "src2tgt.maps_local.tsv"
    memory = "32G"

    from matcha_dl import init_jvm

    init_jvm(memory=memory)

    from matcha_dl.core.actions.evaluation import EvaluationAction

    EvaluationAction.run(
                alignment=alignment_file_path,
                output_dir_path=output_dir,
                error_on_fail=False,
                K=[1, 5, 10],
                source_file_path=source_ontology_file,
                target_file_path=target_ontology_file,
                train_reference_file_path=reference_file,
                full_reference_file_path=None,
                reference_candidates=candidates_file,
            )


if __name__ == "__main__":
    main()