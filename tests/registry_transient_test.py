import importlib.util

from exact.core.entities.registry import ComponentRegistry, ComponentType


def test_transient_module_execution_does_not_register_unimportable_component(tmp_path):
    module_path = tmp_path / "transient_component.py"
    module_path.write_text(
        "\n".join(
            [
                "from exact.core.contracts.base import SelfRegisteringComponent",
                "from exact.core.entities.registry import ComponentType",
                "class TransientComponent(SelfRegisteringComponent):",
                "    component_type = ComponentType.MODEL",
                "",
            ]
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("transient_component", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "TransientComponent" not in ComponentRegistry.list(ComponentType.MODEL)
