"""Side/draft are unknown pre-match: the Fearless engine must preserve the input series probability."""
from __future__ import annotations

import random

import pytest

import src.models.fearless_series_engine as eng
from src.models.fearless_series_engine import (
    calibrate_map_logit,
    fearless_series_win_probability,
    simulate_fearless_series_maps,
)


@pytest.fixture
def no_depletion(monkeypatch):
    monkeypatch.setattr(eng, "compute_team_depletion", lambda *a, **k: 0.0)


@pytest.mark.parametrize("best_of", [1, 3, 5, 7])
@pytest.mark.parametrize("p", [0.5, 0.7, 0.3, 0.93])
def test_analytic_series_probability_matches_input(best_of, p):
    logit = calibrate_map_logit(p, best_of)
    assert fearless_series_win_probability(logit, best_of) == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("best_of", [1, 3, 5])
@pytest.mark.parametrize("p", [0.5, 0.7])
def test_analytic_swap_sums_to_one(best_of, p):
    p_ab = fearless_series_win_probability(calibrate_map_logit(p, best_of), best_of)
    p_ba = fearless_series_win_probability(calibrate_map_logit(1.0 - p, best_of), best_of)
    assert p_ab + p_ba == pytest.approx(1.0, abs=1e-9)


def _mc_win_rate(team_a, team_b, best_of, p, n, seed):
    rng = random.Random(seed)
    wins = 0
    for _ in range(n):
        res = simulate_fearless_series_maps(
            team_a=team_a, team_b=team_b, best_of=best_of, base_series_prob=p, rng=rng
        )
        wins += res.winner == team_a
    return wins / n


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_simulated_even_series_is_fifty_fifty(no_depletion, best_of):
    n = 20000  # SE ~ 0.0035; 5 sigma ~ 0.018
    assert _mc_win_rate("A", "B", best_of, 0.5, n, seed=11) == pytest.approx(0.5, abs=0.018)


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_simulated_series_probability_and_swap(no_depletion, best_of):
    n = 20000
    p_ab = _mc_win_rate("A", "B", best_of, 0.7, n, seed=21)
    p_ba = _mc_win_rate("B", "A", best_of, 0.3, n, seed=22)
    assert p_ab == pytest.approx(0.7, abs=0.018)
    assert p_ab + p_ba == pytest.approx(1.0, abs=0.025)


def test_unknown_game1_side_is_randomized(no_depletion):
    rng = random.Random(5)
    blues = [
        simulate_fearless_series_maps(
            team_a="A", team_b="B", best_of=1, base_series_prob=0.5, rng=rng
        ).map_history[0].blue_team
        for _ in range(400)
    ]
    assert 150 < blues.count("A") < 250


def test_unknown_higher_seed_rejected():
    with pytest.raises(ValueError):
        simulate_fearless_series_maps(
            team_a="A", team_b="B", best_of=3, higher_seed="C", base_series_prob=0.5, rng=random.Random(1)
        )


def test_seeded_runs_are_deterministic():
    kw = dict(team_a="A", team_b="B", best_of=5, base_series_prob=0.6)
    r1 = simulate_fearless_series_maps(rng=random.Random(99), **kw)
    r2 = simulate_fearless_series_maps(rng=random.Random(99), **kw)
    assert r1.scoreline == r2.scoreline
    assert [m.blue_team for m in r1.map_history] == [m.blue_team for m in r2.map_history]
