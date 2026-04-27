from __future__ import annotations

from .candidate_set_selector import CandidateSetSelector


class SecondPassReranker(CandidateSetSelector):
    """
    Backward-compatible alias for the old second-stage model name.

    Legacy configs that still reference ``SecondPassReranker`` now receive the
    unsupervised candidate-set selector implementation. Unknown legacy
    parameters are accepted and ignored by ``CandidateSetSelector``.
    """

    pass
