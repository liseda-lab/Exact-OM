from pathlib import Path

from exact.core.entities.configs.config import ConfigModel


def test_load_config_preserves_default_model_params_on_partial_override(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  params:",
                "    tau_LLM: 0.5",
                "    generate_llm_rationales: False",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = ConfigModel.load_config(cfg_path)

    assert cfg.model.params["tau_LLM"] == 0.5
    assert cfg.model.params["generate_llm_rationales"] is False
    assert cfg.model.params["return_explanations"] is True
    assert cfg.model.params["use_llm"] is True
