"""Current distribution qualification must not reuse historical stack pins."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGES = ("pyowl-core", "pyowl2vec-star-projector", "pyelk-reasoner", "pyhermit")


@pytest.fixture
def distribution_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return (
        importlib.import_module("verify_distribution_artifacts"),
        importlib.import_module("smoke_distribution"),
    )


@pytest.fixture
def contract():
    return json.loads((ROOT / "release/core-compatibility.json").read_text())


def test_current_published_constraints_preserve_the_historical_baseline(
    distribution_tools, contract
):
    verify, _smoke = distribution_tools
    before = json.dumps(contract, sort_keys=True)
    versions = verify._published_stack(contract)
    assert {versions[name] for name in NATIVE_PACKAGES} == {"0.2.1"}
    assert {contract["tested_stack"][name]["version"] for name in NATIVE_PACKAGES} == {"0.2.0"}
    assert versions["oaei-bioml-eval"] == contract["tested_stack"]["oaei-bioml-eval"]["version"]
    assert json.dumps(contract, sort_keys=True) == before


@pytest.mark.parametrize("version", ["0.2.0", "0.3.0", "0.2.1rc1", "0.2.1.dev1", "0.2.1+local"])
def test_distribution_rejects_old_or_unpublished_native_versions(
    distribution_tools, contract, version
):
    verify, _smoke = distribution_tools
    contract["published_native_stack"]["pyowl-core"]["version"] = version
    with pytest.raises(SystemExit):
        verify._published_stack(contract)


def test_distribution_never_falls_back_to_old_tested_stack(distribution_tools, contract):
    verify, _smoke = distribution_tools
    contract.pop("published_native_stack")
    with pytest.raises(SystemExit, match="published_native_stack"):
        verify._published_stack(contract)


def test_smoke_checks_exact_current_installed_versions(distribution_tools, contract, monkeypatch):
    _verify, smoke = distribution_tools
    versions = {name: "0.2.1" for name in NATIVE_PACKAGES}
    versions["exact-om"] = contract["exact"]["version"]
    monkeypatch.setattr(smoke, "_distribution_version", versions.__getitem__)
    smoke._assert_installed_versions(contract, NATIVE_PACKAGES)
    versions["pyowl-core"] = "0.2.0"
    with pytest.raises(SystemExit, match="installed pyowl-core version"):
        smoke._assert_installed_versions(contract, NATIVE_PACKAGES)


def test_current_native_distribution_helpers_qualify_projection_and_reasoners(
    distribution_tools, contract
):
    _verify, smoke = distribution_tools
    smoke._native_smoke(contract)
    smoke._reasoning_smoke(contract)
