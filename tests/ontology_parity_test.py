import json
from pathlib import Path

import pytest
import zstandard

from tools.capture_backend_baseline import capture

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
BASELINES = Path(__file__).parent / "baselines"


@pytest.mark.parametrize("name", ["mini_src", "mini_tgt"])
def test_fixture_snapshot_parity(name):
    ontology = FIXTURES / f"{name}.owl"
    baseline_path = BASELINES / f"{name}.backend.json.zst"
    expected = json.loads(zstandard.ZstdDecompressor().decompress(baseline_path.read_bytes()))
    actual = json.loads(json.dumps(capture(ontology), sort_keys=True))
    assert actual == expected
