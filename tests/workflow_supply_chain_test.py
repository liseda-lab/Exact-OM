"""Supply-chain invariants for repository automation."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTION_REFERENCE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s+(\S+)")


def test_external_workflow_actions_are_pinned_to_commits() -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    workflows = sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
    assert workflows
    for workflow in workflows:
        for line_number, line in enumerate(
            workflow.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            match = USES_LINE.match(line)
            if match is None or match.group(1).startswith("./"):
                continue
            reference = match.group(1)
            assert ACTION_REFERENCE.fullmatch(
                reference
            ), f"{workflow.name}:{line_number}: mutable action reference {reference!r}"
