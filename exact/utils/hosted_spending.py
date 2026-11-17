"""Content-bound operational amendments to aggregate hosted spending limits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

POLICY_PATH_ENV = "EXACT_HOSTED_SPENDING_POLICY_PATH"
POLICY_SHA256_ENV = "EXACT_HOSTED_SPENDING_POLICY_SHA256"


def load_spending_policy(binding: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Verify an explicitly selected amendment; absent amendments retain old limits."""
    path = binding.get("path") if binding is not None else os.environ.get(POLICY_PATH_ENV)
    expected = binding.get("sha256") if binding is not None else os.environ.get(POLICY_SHA256_ENV)
    if not path and not expected:
        return None
    if not path or not expected:
        raise ValueError("Hosted spending policy requires its path and SHA256 binding")
    location = Path(path).resolve()
    raw = location.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Hosted spending policy bytes differ from their SHA256 binding")
    policy = json.loads(raw)
    if (
        not isinstance(policy, dict)
        or type(policy.get("schema_version")) is not int
        or policy["schema_version"] not in (1, 2)
        or policy.get("kind") != "exact_om_hosted_spending_policy"
        or type(policy.get("notification_tokens")) is not int
        or policy["notification_tokens"] <= 0
        or not isinstance(policy.get("authorization"), str)
        or not policy["authorization"].strip()
    ):
        raise ValueError("Invalid recorded hosted spending policy")
    if policy["schema_version"] == 1:
        if policy.get("mode") != "notification_only":
            raise ValueError("Invalid notification-only hosted spending policy")
    else:
        if policy.get("mode") != "hard_pause":
            raise ValueError("Invalid hard-pause hosted spending policy")
        for field in ("campaign_tokens_cap", "experiment_tokens_cap", "experiment_warning_tokens"):
            value = policy.get(field)
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError("Hosted spending policy requires positive integer " + field)
        overrides = policy.get("experiment_tokens_caps", {})
        if not isinstance(overrides, dict) or any(
            not isinstance(name, str)
            or not re.fullmatch(r"E[0-9]{2}|G0", name)
            or type(cap) is not int
            or not 0 < cap < 2**63
            for name, cap in overrides.items()
        ):
            raise ValueError("Invalid explicit per-experiment allowance overrides")
        if policy["experiment_warning_tokens"] > policy["experiment_tokens_cap"]:
            raise ValueError("Experiment warning cannot exceed its hard allowance")
        if not isinstance(policy.get("campaign_id"), str) or not policy["campaign_id"].strip():
            raise ValueError("Hosted spending policy requires a stable campaign identity")
        store = policy.get("admission_store")
        if (
            not isinstance(store, str)
            or not Path(store).is_absolute()
            or str(Path(store).resolve()) != store
        ):
            raise ValueError("Hosted admission store must be an absolute canonical path")
        bootstrap = policy.get("bootstrap")
        if (
            not isinstance(bootstrap, dict)
            or set(bootstrap) != {"path", "sha256"}
            or not isinstance(bootstrap["path"], str)
            or not Path(bootstrap["path"]).is_absolute()
            or not isinstance(bootstrap["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", bootstrap["sha256"])
        ):
            raise ValueError("Hosted spending policy requires a bound historical bootstrap")
    return {"path": str(location), "sha256": expected, "policy": policy}


def spending_policy_environment(policy: Mapping[str, Any]) -> dict[str, str]:
    """Use the selected shared policy while preserving finite request/retry guards."""
    return {
        POLICY_PATH_ENV: str(policy["path"]),
        POLICY_SHA256_ENV: str(policy["sha256"]),
        "EXACT_OPENROUTER_REQUEST_CAP": "",
        "EXACT_OPENROUTER_TOKEN_CAP": "",
        "EXACT_OPENROUTER_RETRY_UNKNOWN": "0",
    }


def record_spending_policy(directory: Path, policy: Mapping[str, Any]) -> None:
    """Durably retain policy content before an amended worker or paid attempt starts."""
    directory = Path(directory) / "spending-policies"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (str(policy["sha256"]) + ".json")
    content = json.dumps(dict(policy), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text() != content:
            raise ValueError("Retained hosted spending policy receipt changed")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".policy-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError:
            if target.read_text() != content:
                raise ValueError("Retained hosted spending policy receipt changed")
    finally:
        Path(temporary).unlink(missing_ok=True)
