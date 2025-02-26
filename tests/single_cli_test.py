import subprocess
from pathlib import Path
import os



DATA_DIR = Path.cwd() / "data"
NCIT_DIR = DATA_DIR / "ncit-doid"
EXP_DIR = Path.cwd() / "exp"

def main():
    source_ontology_file = str(NCIT_DIR / "ncit.owl")
    target_ontology_file = str(NCIT_DIR / "doid.owl")
    output_dir = str(EXP_DIR / "test")
    config_file = str(EXP_DIR / "test" / "config.yaml")
    reference_file = str(NCIT_DIR / "train.tsv")
    candidates_file = str(NCIT_DIR / "test.cands.tsv")
    full_reference_file = str(NCIT_DIR / "test.tsv")
    memory = "32G"

    result = subprocess.run(
        [
            "poetry",
            "run",
            "matchadl",
            "-s",
            source_ontology_file,
            "-t",
            target_ontology_file,
            "-o",
            output_dir,
            "-r",
            reference_file,
            "-f",
            full_reference_file,
            "-c",
            candidates_file,
            "-y",
            config_file,
            "-l",
            "-e",
            "-m",
            memory,
        ],
        check=True,
    )

    print(result)


if __name__ == "__main__":
    main()
