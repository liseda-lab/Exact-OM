from exact.core.entities.configs.config import ConfigModel
from exact.experiments.fitting_recipes import fitting_arms, fitting_requirements
from exact.experiments.harness import deep_merge
from exact.experiments.schema import ArmConfig


def test_bounded_fitting_recipes_validate_with_defaults_without_replacing_models():
    default = ConfigModel.load_config("exact/default_config.yaml").model_dump(mode="python")
    recipes = fitting_arms()
    limits = {"E04": 5, "E07": 7, "E10": 8, "E18": 5, "E19": 4, "E20": 4, "E21": 5, "E22": 5}
    requirements = fitting_requirements()
    for family, arms in recipes.items():
        assert len(arms) <= limits[family]
        assert len({arm["id"] for arm in arms}) == len(arms)
        for raw in arms:
            arm = ArmConfig.model_validate(raw)
            config = ConfigModel.from_mapping(deep_merge(default, arm.overlay))
            assert config.pipeline == ConfigModel.from_mapping(default).pipeline
            assert family + "/" + arm.id in requirements


def test_label_free_recipe_clears_inherited_supervision_components():
    default = ConfigModel.load_config("exact/default_config.yaml").model_dump(mode="python")
    default["supervision"]["components"] = {
        "rerank": "supervised",
        "accept": "supervised",
        "fusion": "supervised",
    }
    arm = next(arm for arm in fitting_arms()["E22"] if arm["id"] == "label_free")
    resolved = ConfigModel.from_mapping(deep_merge(default, arm["overlay"]))
    assert all(value == "label_free" for value in resolved.supervision.components.values())
