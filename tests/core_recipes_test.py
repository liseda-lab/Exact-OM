from exact.core.entities.configs.config import ConfigModel
from exact.experiments.core_recipes import core_arms, core_requirements
from exact.experiments.harness import deep_merge
from exact.experiments.schema import ArmConfig


def test_bounded_core_arms_preserve_model_pins_and_are_strict():
    base = ConfigModel.load_config("exact/default_config.yaml").model_dump(mode="python")
    limits = {
        "E00": 2,
        "E01": 5,
        "E03": 5,
        "E05": 6,
        "E06": 4,
        "E08": 4,
        "E09": 4,
        "E15": 4,
        "E16": 3,
        "E17": 3,
        "E24": 5,
        "E25": 11,
        "E26": 9,
    }
    requirements = core_requirements(base)
    for family, arms in core_arms(base).items():
        assert len(arms) <= limits[family]
        assert len({arm["id"] for arm in arms}) == len(arms)
        for raw in arms:
            arm = ArmConfig.model_validate(raw)
            resolved = ConfigModel.from_mapping(deep_merge(base, arm.overlay))
            for original, component in zip(base["pipeline"], resolved.pipeline):
                for key, value in original["params"].items():
                    if "model_name" in key or "model_revision" in key:
                        assert component.params[key] == value
            assert f"{family}/{arm.id}" in requirements
