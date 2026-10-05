"""Bind prospective launch approval to guarded hosted code or reviewed no-call evidence.

Receipts certify reviewed source and environment files; they do not infer whether a
scientific task uses hosted calls from its name. Attach after the storage wrapper.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

REQUIRED_SOURCE_FILES = (
    "exact/llm/prompt_budget.py",
    "exact/llm/routing.py",
    "exact/impl/models/semantic_llm.py",
)


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _verify(item):
    if (
        not isinstance(item, dict)
        or set(item) != {"path", "sha256"}
        or not isinstance(item["path"], str)
        or not Path(item["path"]).is_absolute()
        or not isinstance(item["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        or binding(item["path"]) != item
    ):
        raise ValueError("Hosted prompt guard binding is missing or changed")
    return Path(item["path"])


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_policy(config):
    from exact.llm.prompt_budget import PromptBudgetError, load_prompt_budget_policy

    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Hosted prompt launch guard requires schema_version 1")
    _verify(config.get("prompt_policy"))
    try:
        load_prompt_budget_policy(config["prompt_policy"], required=True)
    except PromptBudgetError as error:
        raise ValueError("Hosted prompt guard policy is invalid: " + str(error)) from error
    allowed = config.get("allowed_source_sha256")
    if not isinstance(allowed, dict) or not set(REQUIRED_SOURCE_FILES) <= set(allowed):
        raise ValueError(
            "Hosted prompt guard requires approved transport and semantic source hashes"
        )
    for name, hashes in allowed.items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(hashes, list)
            or not hashes
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in hashes
            )
        ):
            raise ValueError("Hosted prompt guard source approval is invalid")


def validate_launch(launch, policy):
    """Reject an unreviewed future worker before the dispatcher starts tmux/srun."""
    config = policy.get("hosted_prompt_guard")
    if config is None:
        return
    validate_policy(config)
    attached = launch.get("hosted_prompt_guard")
    if attached not in launch.get("bindings", []):
        raise ValueError("Prepared launch lacks a bound hosted prompt guard receipt")
    receipt = json.loads(_verify(attached).read_text())
    _validate_receipt(launch, policy, receipt)


def _validate_receipt(launch, policy, receipt):
    config = policy["hosted_prompt_guard"]
    if not isinstance(receipt, dict):
        raise ValueError("Hosted prompt guard receipt must be an object")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("policy_sha256") != _identity(config)
        or receipt.get("dispatch_nonce") != launch.get("nonce")
        or receipt.get("run_id") != launch.get("run", {}).get("id")
        or receipt.get("worker") != binding(launch["argv"][-1])
    ):
        raise ValueError("Hosted prompt guard receipt does not match this reviewed launch")

    def bound(item):
        if item not in launch["bindings"]:
            raise ValueError("Hosted prompt guard dependency is not a launch binding")
        return _verify(item)

    bound(receipt["worker"])
    recipe = json.loads(bound(receipt.get("recipe")).read_text())
    if receipt.get("mode") == "no_new_hosted_calls":
        evidence = receipt.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("No-new-hosted-calls exemption requires explicit reviewed evidence")
        for item in evidence:
            bound(item)
        return
    if receipt.get("mode") != "guarded_hosted":
        raise ValueError("Unknown hosted prompt guard launch mode")
    if (
        receipt.get("code_root") != recipe.get("code_root")
        or receipt.get("commit") != recipe.get("commit")
        or not isinstance(recipe.get("code_root"), str)
        or not Path(recipe["code_root"]).is_absolute()
        or not isinstance(recipe.get("commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", recipe["commit"])
        or receipt.get("environment") != recipe.get("environment")
        or receipt.get("prompt_policy") != config["prompt_policy"]
    ):
        raise ValueError("Hosted prompt guard source/environment differs from the frozen recipe")
    bound(receipt["prompt_policy"])
    environment = json.loads(bound(receipt["environment"]).read_text())
    spending = policy.get("hosted_spending_policy")
    if spending is not None:
        from exact.utils.hosted_spending import load_spending_policy

        if receipt.get("spending_policy") != spending:
            raise ValueError("Hosted worker lacks the reviewed campaign spending policy")
        bound(spending)
        selected = load_spending_policy(spending)
        if selected is None or selected["policy"].get("mode") != "hard_pause":
            raise ValueError("Hosted launch requires the approved spending pause policy")
        expected = {
            "EXACT_HOSTED_SPENDING_POLICY_PATH": spending["path"],
            "EXACT_HOSTED_SPENDING_POLICY_SHA256": spending["sha256"],
            "EXACT_HOSTED_CAMPAIGN_ID": selected["policy"]["campaign_id"],
        }
        if any(environment.get(key) != value for key, value in expected.items()):
            raise ValueError("Hosted worker environment lacks the bound spending policy")
        scope = launch["run"].get("hosted_scope")
        if (
            not isinstance(scope, dict)
            or set(scope) != {"campaign_id", "experiment_id"}
            or receipt.get("hosted_scope") != scope
            or scope["campaign_id"] != selected["policy"]["campaign_id"]
        ):
            raise ValueError("Hosted launch lacks its bound scientific spending scope")
        from exact.core.entities.configs.yaml_io import load_yaml_mapping

        campaign = load_yaml_mapping(bound(recipe.get("base_campaign")))
        step = next(
            (row for row in campaign["steps"] if row["id"] == recipe.get("scientific_step")), None
        )
        if (
            campaign.get("campaign_id") != scope["campaign_id"]
            or step is None
            or step.get("family") != scope["experiment_id"]
        ):
            raise ValueError("Hosted launch spending scope differs from its declared family")
        protected = {
            "exact/llm/ledger.py",
            "exact/llm/spending_admission.py",
            "exact/utils/hosted_spending.py",
            "exact/experiments/runtime.py",
            "exact/experiments/harness.py",
            "tools/prepared_batch.py",
        }
        if not protected <= set(config["allowed_source_sha256"]):
            raise ValueError("Hosted launch lacks reviewed spending admission and scope code")
    required_environment = {
        "EXACT_EXPERIMENT_MODE": "1",
        "EXACT_LLM_PROMPT_POLICY_PATH": config["prompt_policy"]["path"],
        "EXACT_LLM_PROMPT_POLICY_SHA256": config["prompt_policy"]["sha256"],
    }
    if any(environment.get(name) != value for name, value in required_environment.items()):
        raise ValueError("Hosted worker environment lacks the required bound prompt policy")
    source = receipt.get("source_files")
    if not isinstance(source, dict) or set(source) != set(config["allowed_source_sha256"]):
        raise ValueError("Hosted prompt guard receipt omits reviewed source files")
    for name, approved in config["allowed_source_sha256"].items():
        path = bound(source[name])
        if path != Path(recipe["code_root"]) / name or source[name]["sha256"] not in approved:
            raise ValueError("Hosted worker uses unapproved prompt-guard source: " + name)


def guard_launch(launch, policy, *, recipe_path, receipt_path, mode="guarded_hosted", evidence=()):
    """Return a descriptor with an immutable receipt; never edit an existing receipt."""
    config = policy.get("hosted_prompt_guard")
    if config is None:
        return launch
    validate_policy(config)
    launch = copy.deepcopy(launch)
    if launch.get("hosted_prompt_guard") is not None:
        validate_launch(launch, policy)
        return launch
    recipe_binding = binding(recipe_path)
    recipe = json.loads(Path(recipe_path).read_text())
    receipt = {
        "schema_version": 1,
        "mode": mode,
        "policy_sha256": _identity(config),
        "dispatch_nonce": launch["nonce"],
        "run_id": launch["run"]["id"],
        "worker": binding(launch["argv"][-1]),
        "recipe": recipe_binding,
    }
    dependencies = [receipt["worker"], recipe_binding]
    if mode == "no_new_hosted_calls":
        receipt["evidence"] = list(evidence)
        dependencies.extend(evidence)
    elif mode == "guarded_hosted":
        sources = {
            name: binding(Path(recipe["code_root"]) / name)
            for name in config["allowed_source_sha256"]
        }
        receipt.update(
            code_root=recipe["code_root"],
            commit=recipe["commit"],
            source_files=sources,
            environment=recipe["environment"],
            prompt_policy=config["prompt_policy"],
        )
        dependencies.extend([recipe["environment"], config["prompt_policy"], *sources.values()])
        if policy.get("hosted_spending_policy") is not None:
            receipt["spending_policy"] = policy["hosted_spending_policy"]
            receipt["hosted_scope"] = launch["run"].get("hosted_scope")
            dependencies.extend([receipt["spending_policy"], recipe.get("base_campaign")])
    else:
        raise ValueError("Unknown hosted prompt guard launch mode")
    for item in dependencies:
        if item not in launch["bindings"]:
            launch["bindings"].append(item)
    _validate_receipt(launch, policy, receipt)
    path = Path(receipt_path).resolve()
    content = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Immutable hosted prompt guard receipt changed")
    else:
        with path.open("x") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    launch["hosted_prompt_guard"] = binding(path)
    launch["bindings"].append(launch["hosted_prompt_guard"])
    validate_launch(launch, policy)
    return launch
