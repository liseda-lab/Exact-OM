from pathlib import Path

import pandas as pd
import pytest

from exact.core.entities.evaluation import EvaluationRequest
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.impl.evaluators.builtin import BuiltinEvaluator
from exact.io.writers import write


def test_builtin_global_metrics_preserve_historical_names_and_math() -> None:
    predictions = [
        EntityMapping("s1", "t1", "=", 0.9),
        EntityMapping("s2", "t2", "=", 0.8),
    ]
    reference = [ReferenceMapping("s1", "t1", "="), ReferenceMapping("s3", "t3", "=")]

    result = BuiltinEvaluator.run(
        EvaluationRequest(alignment=predictions, full_reference=reference)
    )

    assert result.metrics == {"P": 0.5, "R": 0.5, "F1": 0.5}


def test_builtin_local_metrics_use_requested_cutoffs() -> None:
    reference = ReferenceMapping("s", "gold", "=")
    candidates = [
        EntityMapping("s", "other", "=", 0.9),
        EntityMapping("s", "gold", "=", 0.8),
    ]

    result = BuiltinEvaluator.run(
        EvaluationRequest(alignment=[(reference, candidates)], k=(1, 2))
    )

    assert result.metrics == {"Hits@1": 0.0, "Hits@2": 1.0, "MRR": 0.5}


@pytest.mark.parametrize("output_format", ["typed-tsv", "json", "oaei-rdf"])
def test_builtin_local_eval_projects_scored_artifacts_onto_candidate_pool(
    tmp_path: Path,
    output_format: str,
) -> None:
    pool_path = tmp_path / "test.cands.tsv"
    pd.DataFrame(
        [
            {
                "SrcEntity": "s",
                "TgtEntity": "gold",
                "TgtCandidates": repr(["gold", "wrong"]),
            }
        ]
    ).to_csv(pool_path, sep="\t", index=False)
    artifact = write(
        output_format,
        pd.DataFrame(
            [
                {
                    "SrcEntity": "s",
                    "TgtEntity": "wrong",
                    "Relation": "=",
                    "Score": 0.9,
                }
            ]
        ),
        tmp_path,
    )

    metrics = BuiltinEvaluator.local_eval(
        artifact,
        reference_candidates=pool_path,
        K=[1, 2],
    )

    assert metrics == {"Hits@1": 0.0, "Hits@2": 1.0, "MRR": 0.5}
