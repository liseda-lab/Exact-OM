from __future__ import annotations

from .selector import CandidateSetSelector


class SecondPassReranker(CandidateSetSelector):
    """Backward-compatible registered alias for ``CandidateSetSelector``."""


__all__ = ["CandidateSetSelector", "SecondPassReranker"]
