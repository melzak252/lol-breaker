"""Replacement-guard (subgroup non-inferiority) evaluation on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import replacement_guards as rg


def _synthetic(n_per_month: int = 150, n_months: int = 8, seed: int = 7):
    rng = np.random.default_rng(seed)
    months = np.repeat(np.arange(n_months), n_per_month)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(months * 31 + rng.integers(0, 28, len(months)), unit="D")
    p_true = rng.uniform(0.2, 0.8, len(months))
    y = (rng.uniform(size=len(months)) < p_true).astype(int)
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "best_of": rng.choice([1, 3, 5], len(months)),
            "tournament_name": rng.choice(["LPL Spring 2026", "LCK 2026", "Worlds 2026"], len(months)),
            "competition_tier": rng.choice(["major", "minor"], len(months)),
            "unknown_players": rng.choice([0, 1], len(months)),
        }
    )
    return y, p_true, frame


def _masks(frame):
    return {"BO1": (frame["best_of"] == 1).to_numpy(), "BO3": (frame["best_of"] == 3).to_numpy()}


def test_identical_models_pass_every_guard():
    y, p, frame = _synthetic()
    result = rg.evaluate_replacement_guards(y, p, p, frame, rg.standard_slice_masks(frame))
    assert set(result["guards"]) == set(rg.GUARD_SLICE_NAMES)
    assert result["overall_pass"] is True
    assert result["overall_status"] == rg.STATUS_PASS
    for guard in result["guards"].values():
        assert guard["status"] == rg.STATUS_PASS and guard["passes"] is True
        assert guard["delta_ll"] == pytest.approx(0.0, abs=1e-12)
        assert guard["delta_ll_equal_block"] == pytest.approx(0.0, abs=1e-12)
        assert guard["ci_upper"] == pytest.approx(0.0, abs=1e-12)
    assert result["one_sided_confidence"] == pytest.approx(1 - 0.05 / 6)
    assert result["two_sided_confidence"] == pytest.approx(1 - 2 * 0.05 / 6)
    assert result["ci_type"] == "equal-tailed_two-sided_interval; upper bound also used for one-sided guard"

def test_degraded_candidate_fails_and_overall_fails():
    y, p, frame = _synthetic()
    degraded = np.full_like(p, 0.5)  # uninformative candidate vs informative reference
    result = rg.evaluate_replacement_guards(y, degraded, p, frame, rg.standard_slice_masks(frame))
    assert result["overall_pass"] is False
    assert result["overall_status"] == rg.STATUS_FAIL
    assert all(g["status"] == rg.STATUS_FAIL for g in result["guards"].values())
    assert all(g["delta_ll"] > 0 for g in result["guards"].values())


def test_one_failing_guard_blocks_overall_pass():
    y, p, frame = _synthetic()
    candidate = p.copy()
    bo1 = (frame["best_of"] == 1).to_numpy()
    candidate[bo1] = 0.5
    result = rg.evaluate_replacement_guards(y, candidate, p, frame, _masks(frame))
    assert result["guards"]["BO1"]["status"] == rg.STATUS_FAIL
    assert result["guards"]["BO3"]["status"] == rg.STATUS_PASS
    assert result["overall_pass"] is False


def test_empty_slice_is_not_evaluable_and_never_passes():
    y, p, frame = _synthetic()
    masks = {**_masks(frame), "Empty": np.zeros(len(frame), dtype=bool)}
    result = rg.evaluate_replacement_guards(y, p, p, frame, masks)
    empty = result["guards"]["Empty"]
    assert empty["status"] == rg.STATUS_NOT_EVALUABLE
    assert empty["passes"] is False
    assert empty["n"] == 0 and empty["ci_upper"] is None
    assert result["overall_pass"] is False
    assert result["overall_status"] == rg.STATUS_NOT_EVALUABLE


def test_single_month_slice_is_not_evaluable():
    y, p, frame = _synthetic()
    first_month = (pd.to_datetime(frame["date"]).dt.to_period("M") == pd.Period("2026-01")).to_numpy()
    result = rg.evaluate_replacement_guards(y, p, p, frame, {"OneMonth": first_month})
    guard = result["guards"]["OneMonth"]
    assert guard["n_blocks"] == 1 and guard["n"] > 0
    assert guard["status"] == rg.STATUS_NOT_EVALUABLE and guard["passes"] is False


def test_uses_calendar_month_blocks(monkeypatch):
    y, p, frame = _synthetic()
    calls = []
    real = rg.monthly_block_bootstrap_delta

    def spy(data, delta_column, **kwargs):
        calls.append((data.copy(), kwargs))
        return real(data, delta_column, **kwargs)

    monkeypatch.setattr(rg, "monthly_block_bootstrap_delta", spy)
    result = rg.evaluate_replacement_guards(y, p, p, frame, _masks(frame))
    assert len(calls) == 2
    for guard_name, (data, kwargs) in zip(("BO1", "BO3"), calls):
        assert kwargs["n_bootstraps"] == 5000
        expected_blocks = pd.to_datetime(data["date"]).dt.to_period("M").nunique()
        assert result["guards"][guard_name]["n_blocks"] == expected_blocks
    assert result["blocking"] == "calendar_month"


def test_match_weighted_and_equal_block_deltas_differ_when_blocks_unbalanced():
    dates = ["2026-01-05"] * 90 + ["2026-02-05"] * 10
    frame = pd.DataFrame({"date": dates})
    y = np.ones(100, dtype=int)
    reference = np.full(100, 0.8)
    candidate = np.concatenate([np.full(90, 0.8), np.full(10, 0.6)])
    result = rg.evaluate_replacement_guards(y, candidate, reference, frame, {"All": np.ones(100, bool)})
    guard = result["guards"]["All"]
    per_row = np.log(0.8 / 0.6)
    assert guard["n_blocks"] == 2
    assert guard["delta_ll"] == pytest.approx(0.1 * per_row)
    assert guard["delta_ll_equal_block"] == pytest.approx(0.5 * per_row)


def test_bonferroni_widens_upper_bound_with_more_guards():
    y, p, frame = _synthetic()
    candidate = np.clip(p + 0.03, 0.01, 0.99)
    masks = {"All": np.ones(len(frame), dtype=bool)}
    one = rg.evaluate_replacement_guards(y, candidate, p, frame, masks, n_guards=1)
    six = rg.evaluate_replacement_guards(y, candidate, p, frame, masks, n_guards=6)
    assert six["guards"]["All"]["ci_upper"] >= one["guards"]["All"]["ci_upper"]
    assert six["guards"]["All"]["ci_lower"] <= one["guards"]["All"]["ci_lower"]
    assert six["guards"]["All"]["delta_ll"] == pytest.approx(one["guards"]["All"]["delta_ll"])


def test_input_validation():
    y, p, frame = _synthetic()
    masks = _masks(frame)
    with pytest.raises(ValueError, match="n_boot"):
        rg.evaluate_replacement_guards(y, p, p, frame, masks, n_boot=100)
    with pytest.raises(ValueError, match="length mismatch"):
        rg.evaluate_replacement_guards(y[:-1], p, p, frame, masks)
    with pytest.raises(ValueError, match="n_guards"):
        rg.evaluate_replacement_guards(y, p, p, frame, masks, n_guards=1)
    with pytest.raises(ValueError, match="finite"):
        bad = p.copy()
        bad[0] = np.nan
        rg.evaluate_replacement_guards(y, bad, p, frame, masks)


def test_standard_slice_masks_follow_assess_definitions():
    frame = pd.DataFrame(
        {
            "best_of": [1, 3, 5, 3, 3],
            "tournament_name": ["LPL Spring 2026", "KeSPA Cup 2026", "Worlds 2026", "LCK 2026", "LPL Split 2"],
            "competition_tier": ["minor", "major", "major", "minor", "minor"],
            "unknown_players": [0, 0, 2, 0, 1],
        }
    )
    masks = rg.standard_slice_masks(frame)
    assert list(masks) == list(rg.GUARD_SLICE_NAMES)
    assert masks["BO1"].tolist() == [True, False, False, False, False]
    assert masks["BO3"].tolist() == [False, True, False, True, True]
    assert masks["Major"].tolist() == [False, False, True, False, False]
    assert masks["LPL"].tolist() == [True, False, False, False, True]
    assert masks["Unseen_Players"].tolist() == [False, False, True, False, True]


def test_bonferroni_delta_interval_reports_level_and_blocks():
    y, p, frame = _synthetic()
    better = np.clip(p, 0.01, 0.99)
    worse = np.full_like(p, 0.5)
    out = rg.bonferroni_delta_interval(y, better, worse, frame, n_comparisons=20)
    assert out["confidence"] == pytest.approx(1 - 0.05 / 20)
    assert out["one_sided_alpha"] == pytest.approx(0.05 / 40)
    assert out["two_sided_confidence"] == out["confidence"]
    assert out["n_blocks"] == pd.to_datetime(frame["date"]).dt.to_period("M").nunique()
    assert out["n"] == len(y)
    assert out["delta_match_weighted"] < 0
    assert out["excludes_zero_favouring_candidate"] is True
    wide = rg.bonferroni_delta_interval(y, better, worse, frame, n_comparisons=200)
    assert wide["ci_upper"] >= out["ci_upper"]


def test_assert_chronological_order():
    rg.assert_chronological_order(["2026-01-01", "2026-01-01", "2026-02-01"])
    with pytest.raises(ValueError, match="not chronologically ordered"):
        rg.assert_chronological_order(["2026-02-01", "2026-01-01"])
    with pytest.raises(ValueError, match="missing"):
        rg.assert_chronological_order(["2026-01-01", None])
