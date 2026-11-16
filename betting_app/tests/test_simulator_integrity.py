"""Focused regression coverage for simulator conservation and probability semantics."""

import math
import random

import pytest

from src.models.advanced_tournament_simulator import (
    AdvancedTournamentSimulator,
    _assert_mass_conserved,
)
from src.models.tournament_bayesian_momentum import InPathTournamentBayesianFilter


def test_mass_assertion_rejects_unnormalized_broken_mass() -> None:
    with pytest.raises(AssertionError, match="max deviation"):
        _assert_mass_conserved("deliberately broken", (0.9, 1.0), (1.8, 2.0))


def test_two_team_key_is_already_series_probability() -> None:
    simulator = AdvancedTournamentSimulator(shrinkage_lambda=0.0, temperature=1.0)
    probability = simulator.get_matchup_probability(
        "A",
        "B",
        best_of=3,
        base_probabilities={("A", "B"): 0.7},
        team_ratings={"A": 1500.0, "B": 1500.0},
    )
    assert math.isclose(probability, 0.7, abs_tol=1e-12)


def test_favourite_momentum_update_has_zero_expected_drift() -> None:
    probability = 0.65
    rng = random.Random(104)
    offsets = []
    for _ in range(2000):
        filt = InPathTournamentBayesianFilter(
            learning_rate=0.12,
            adaptation_volatility=0.0,
            map_margin_weight=0.03,
            rng=rng,
        )
        won = rng.random() < probability
        filt.update_after_series(
            "A", "B", "A" if won else "B", 2 if won else 0,
            0 if won else 2, probability, best_of=3,
        )
        offsets.append(filt.get_state("A").current_logit_offset)
    assert abs(sum(offsets) / len(offsets)) < 0.01
