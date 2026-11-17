"""Small durable local-upload state machine; ZIP extraction never triggers computation."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .contracts import DomainError


class ImportJobs:
    """Track bounded uploads and cooperative cancellation without an external job platform."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # No uploader/extractor survives its serving process; partial staging was never published.
        for path in self.root.glob("*.json"):
            state = json.loads(path.read_bytes())
            if state["status"] in {"uploading", "validating"}:
                atomic_json(
                    path,
                    {
                        **state,
                        "status": "interrupted",
                        "reason": "Retry the upload after service restart",
                    },
                )

    def create(self) -> dict[str, Any]:
        """Allocate an opaque handle before upload so another tab can track or cancel it."""
        job_id = uuid.uuid4().hex
        state = {
            "job_id": job_id,
            "status": "pending",
            "received_bytes": 0,
            "cancel_requested": False,
            "package_id": None,
            "reason": None,
        }
        atomic_json(self.root / (job_id + ".json"), state)
        return state

    def get(self, job_id: str) -> dict[str, Any]:
        """Read public progress without exposing temporary paths or upload content."""
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise DomainError("unknown_import", "Unknown import job", 404)
        path = self.root / (job_id + ".json")
        if not path.is_file():
            raise DomainError("unknown_import", "Unknown import job", 404)
        return dict(json.loads(path.read_bytes()))

    def claim(self, job_id: str) -> None:
        """Prevent two uploads from writing or publishing under the same handle."""
        with self._lock:
            state = self.get(job_id)
            if state["status"] != "pending" or state["cancel_requested"]:
                raise DomainError(
                    "import_conflict", "Import job is already claimed or cancelled", 409
                )
            atomic_json(self.root / (job_id + ".json"), {**state, "status": "uploading"})

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        """Atomically persist only server-owned state changes."""
        with self._lock:
            state = {**self.get(job_id), **changes}
            atomic_json(self.root / (job_id + ".json"), state)
            return state

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Request cancellation; a completed atomic publication remains a valid library item."""
        with self._lock:
            state = self.get(job_id)
            if state["status"] == "available":
                raise DomainError("import_complete", "Import has already completed", 409)
            state["cancel_requested"] = True
            if state["status"] == "pending":
                state["status"] = "cancelled"
            atomic_json(self.root / (job_id + ".json"), state)
            return state
