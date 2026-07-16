from pathlib import Path

from exact.core.entities.configs.config import ConfigModel

PROFILE = Path(__file__).parents[1] / "configs" / "profiles" / "biokg.yaml"


def test_biokg_profile_preserves_the_complete_ranked_candidate_pool() -> None:
    config = ConfigModel.load_config(PROFILE)

    assert config.io.input_format == "csv-kg"
    assert config.io.output_formats == ["typed-tsv"]
    assert config.matching.threshold is None
    assert config.matching.cardinality is None
    assert config.matching.target_cardinality is None
    assert config.matching.relation_prediction == "hierarchy_heuristic"
    assert config.evaluation.backends == ["builtin", "bioml"]
