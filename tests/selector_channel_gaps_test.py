"""The optimized competition features retain the legacy numeric contract."""
import random

import pandas as pd
import pytest

from exact.impl.models.selector.features import FeatureEngineeringMixin


def legacy(selector, group, indices):
    columns = [c for c in ("s_label", "S_struct", "s_hier", "s_sim", "s_attr", "cand_sim")
               if c in group]
    result = {}
    for index in indices:
        gaps = []
        for column in columns:
            values = [selector._safe_float(group.at[i, column], 0.0) for i in indices]
            pos = list(indices).index(index)
            gaps.append(0.0 if len(values) <= 1 else
                        max(0.0, values[pos] - max(values[:pos] + values[pos + 1:])))
        result[index] = sum(gaps) / max(1, len(gaps))
    return result


@pytest.mark.parametrize("values", [[], [0.4], [0.9, 0.9, 0.2], [-1, -2, -3],
    [None, float("nan"), "bad", "0.5"], [float("inf"), float("inf"), -float("inf")],
    ["nan", 0.2, 0.9], [0.2, "nan", 0.9], [0.2, 0.9, "nan"]])
def test_gaps_preserve_ties_missing_and_nonfinite_values(values):
    selector = FeatureEngineeringMixin()
    frame = pd.DataFrame({"s_label": values, "s_attr": list(reversed(values))},
                         index=[3 * i + 2 for i in range(len(values))])
    for indices in (list(frame.index), list(reversed(frame.index))):
        assert selector._channel_gaps(frame, indices) == legacy(selector, frame, indices)


def test_all_channels_and_subset_preserve_exact_arithmetic():
    rng = random.Random(17)
    frame = pd.DataFrame({c: [rng.random() for _ in range(101)]
                          for c in ("s_label", "S_struct", "s_hier", "s_sim", "s_attr", "cand_sim")})
    selector = FeatureEngineeringMixin()
    for indices in (list(frame.index), list(frame.index)[::3]):
        assert selector._channel_gaps(frame, indices) == legacy(selector, frame, indices)
    assert selector._channel_gaps(frame.drop(columns=list(frame)), list(frame.index)) == {
        i: 0.0 for i in frame.index}


def test_same_winner_preserves_float_summation_across_channels():
    frame = pd.DataFrame({"s_label": [0.1, 0.0], "S_struct": [0.2, 0.0],
                          "s_hier": [0.3, 0.0]})
    selector = FeatureEngineeringMixin()
    assert selector._channel_gaps(frame, [0, 1]) == legacy(selector, frame, [0, 1])


def test_channel_values_are_normalized_once_per_candidate(monkeypatch):
    selector = FeatureEngineeringMixin()
    frame = pd.DataFrame({"s_label": [0.5] * 100, "s_attr": [0.7] * 100})
    original = selector._safe_float
    calls = []
    monkeypatch.setattr(selector, "_safe_float", lambda value, default: (
        calls.append(value), original(value, default))[1])
    selector._channel_gaps(frame, list(frame.index))
    assert len(calls) == 200
