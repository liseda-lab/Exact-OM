import os
import subprocess
from pathlib import Path

import pytest

DATA_DIR = Path.cwd() / "data"
NCIT_DIR = DATA_DIR / "ncit-doid"
EXP_DIR = Path.cwd() / "exp"

pytestmark = [pytest.mark.requires_data, pytest.mark.slow]


@pytest.fixture
def cli_command():
    data_dir = Path(os.environ.get("EXACT_TEST_DATA_DIR", NCIT_DIR))
    source_ontology_file = str(data_dir / "ncit.owl")
    target_ontology_file = str(data_dir / "doid.owl")
    output_dir = os.environ.get("EXACT_TEST_OUTPUT_DIR", str(EXP_DIR / "test"))
    config_file = os.environ.get("EXACT_TEST_CONFIG", str(EXP_DIR / "test" / "config.yaml"))
    reference_file = str(data_dir / "train.tsv")
    candidates_file = str(data_dir / "test.cands.tsv")
    full_reference_file = str(data_dir / "test.tsv")

    required = [
        source_ontology_file,
        target_ontology_file,
        config_file,
        reference_file,
        candidates_file,
        full_reference_file,
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        pytest.skip(f"Integration data is not configured; missing: {', '.join(missing)}")

    command = [
        "poetry",
        "run",
        "exact",
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
    ]

    device = os.environ.get("EXACT_TEST_DEVICE")
    if device is not None:
        command.extend(["--device", device])
    return command


def test_cli_execution(cli_command):
    try:
        result = subprocess.run(cli_command, check=True, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed with return code {result.returncode}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command raised an exception: {e.stderr}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
