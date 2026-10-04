"""Content-bound operational amendments to aggregate hosted spending limits."""

from __future__ import annotations

import hashlib
import json
import os
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
        or policy.get("schema_version") != 1
        or policy.get("kind") != "exact_om_hosted_spending_policy"
        or policy.get("mode") != "notification_only"
        or type(policy.get("notification_tokens")) is not int
        or policy["notification_tokens"] <= 0
        or not isinstance(policy.get("authorization"), str)
        or not policy["authorization"].strip()
    ):
        raise ValueError("Invalid recorded hosted spending policy")
    return {"path": str(location), "sha256": expected, "policy": policy}


def spending_policy_environment(policy: Mapping[str, Any]) -> dict[str, str]:
    """Keep finite per-request bounds and retry guards without an aggregate cutoff."""
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
