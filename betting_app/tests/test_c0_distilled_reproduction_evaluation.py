import numpy as np
import pandas as pd
import pytest

from src.analysis.c0_distilled_reproduction_evaluation import (
    _exact_join,
    _paired_month_interval,
    _strict_probabilities,
)


def test_exact_join_requires_exact_unique_ids_and_preserves_canonical_order():
    frame = pd.DataFrame({"golgg_match_id": [11, 22], "y": [0, 1]})
    predictions = pd.DataFrame({"golgg_match_id": [22, 11], "p": [0.8, 0.2]})
    joined = _exact_join(frame, predictions, "predictions", ["p"])
    assert joined.golgg_match_id.tolist() == [11, 22]
    assert joined.p.tolist() == [0.2, 0.8]
    with pytest.raises(ValueError, match="exactly cover"):
        _exact_join(frame, predictions.iloc[:1], "predictions", ["p"])
    with pytest.raises(ValueError, match="unique"):
        _exact_join(
            frame, pd.concat([predictions, predictions.iloc[:1]]), "predictions", ["p"]
        )

def test_exact_join_rejects_reversed_outcomes_for_identical_match_ids():
    frame = pd.DataFrame({"golgg_match_id": ["a", "b"], "y": [0, 1]})
    reversed_sides = pd.DataFrame({"golgg_match_id": ["a", "b"], "y": [1, 0], "p": [.8, .2]})
    with pytest.raises(ValueError):
        _exact_join(frame, reversed_sides, "candidate", ["p"])



def test_strict_probability_contract_rejects_nonfinite_and_boundary_values():
    assert np.array_equal(_strict_probabilities([0.2, 0.8], "p", 2), [0.2, 0.8])
    for values in ([0.0], [1.0], [np.nan], [np.inf], [-0.1], [0.2, 0.3]):
        with pytest.raises(ValueError, match="no clipping"):
            _strict_probabilities(values, "p", 1)


def test_month_bootstrap_uses_paired_differences_and_equal_month_estimand():
    dates = pd.Series(
        pd.to_datetime(
            [f"2026-{month:02d}-10" for month in range(1, 10) for _ in range(2)]
        )
    )
    y = np.ones(18, dtype=int)
    ref = np.full(18, 0.8)
    candidate = np.repeat(np.linspace(0.5, 0.9, 9), 2)
    result = _paired_month_interval(y, candidate, ref, dates, 20261021, 5000)
    expected = np.log(ref / candidate).reshape(9, 2).mean(axis=1)
    assert result["month_count"] == 9
    assert result["match_weighted_delta_log_loss"] == pytest.approx(expected.mean())
    assert result["equal_month_delta_log_loss"] == pytest.approx(expected.mean())
    assert result["equal_month_ci_97_5"][0] <= result["equal_month_ci_97_5"][1]


def test_month_bootstrap_rejects_missing_month_blocks():
    with pytest.raises(ValueError, match="exactly 9 monthly"):
        _paired_month_interval(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            np.array([0.3, 0.7]),
            pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"])),
            1,
            5000,
        )
