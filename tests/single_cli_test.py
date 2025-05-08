import subprocess
from pathlib import Path
import pytest

DATA_DIR = Path.cwd() / "data"
NCIT_DIR = DATA_DIR / "ncit-doid"
EXP_DIR = Path.cwd() / "exp"

@pytest.fixture
def cli_command():
    source_ontology_file = str(NCIT_DIR / "ncit.owl")
    target_ontology_file = str(NCIT_DIR / "doid.owl")
    output_dir = str(EXP_DIR / "test")
    config_file = str(EXP_DIR / "test" / "config.yaml")
    reference_file = str(NCIT_DIR / "train.tsv")
    candidates_file = str(NCIT_DIR / "test.cands.tsv")
    full_reference_file = str(NCIT_DIR / "test.tsv")
    memory = "32G"
    device = "0"

    return [
        "poetry",
        "run",
        "matchadl",
        "-s", source_ontology_file,
        "-t", target_ontology_file,
        "-o", output_dir,
        "-r", reference_file,
        "-f", full_reference_file,
        "-c", candidates_file,
        "-y", config_file,
        "-l",
        "-e",
        "-m", memory,
        "--device", device
    ]

def test_cli_execution(cli_command):
    try:
        result = subprocess.run(cli_command, check=True, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed with return code {result.returncode}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command raised an exception: {e.stderr}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
