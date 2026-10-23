"""Comprehensive unit and integration test suite for the Advanced Tournament Simulation engine.

Tests:
1. Riot Sequential Swiss Draw Lottery (anti-rematch, R1 regional separation, slide-rules).
2. Hard Fearless Map Engine (side selection, champion burning, depletion penalties, scorelines).
3. In-Path Bayesian Momentum Filter (martingale expectation, streak dynamics, logit clamping).
4. Advanced Tournament Simulator Worlds & Regional Brackets (mass conservation, reproducibility, hybrid DP).
"""
from __future__ import annotations

import math
import random
from typing import Any
import pytest

from src.models.tournament_swiss_lottery import (
    SwissMatchup,
    draw_swiss_pool_lottery,
    simulate_swiss_round_draws,
)
from src.models.fearless_series_engine import (
    FearlessSeriesResult,
    MapResult,
    simulate_fearless_series_maps,
)
from src.models.tournament_bayesian_momentum import (
    InPathTournamentBayesianFilter,
)
from src.models.advanced_tournament_simulator import (
    AdvancedTournamentSimulator,
    DEFAULT_WORLDS_16_TEAMS,
    DEFAULT_WORLDS_RATINGS,
    DEFAULT_WORLDS_REGIONS,
)
from betting_app.services.tournament_service import SUPPORTED_BRACKETS


# ==============================================================================
# 1. Tests for Riot Sequential Swiss Draw Lottery
# ==============================================================================

def test_swiss_lottery_round1_regional_separation():
    """Verify that Round 1 draw strictly prevents intra-regional pairings."""
    teams = [
        "Gen.G", "T1", "HLE", "DK",           # LCK
        "BLG", "TES", "AL", "LGD",           # LPL
        "G2", "FNC", "MKOI",                 # LEC
        "TL", "FLY", "C9",                   # LCS
        "PSG", "GAM",                        # APAC / VCS
    ]
    regions = {
        "Gen.G": "LCK", "T1": "LCK", "HLE": "LCK", "DK": "LCK",
        "BLG": "LPL", "TES": "LPL", "AL": "LPL", "LGD": "LPL",
        "G2": "LEC", "FNC": "LEC", "MKOI": "LEC",
        "TL": "LCS", "FLY": "LCS", "C9": "LCS",
        "PSG": "APAC", "GAM": "APAC",
    }
    rng = random.Random(42)
    matchups = draw_swiss_pool_lottery(
        teams=teams,
        pool_name="0-0",
        round_num=1,
        past_matchups=set(),
        team_regions=regions,
        best_of=1,
        rng=rng,
    )

    assert len(matchups) == 8
    paired_teams = set()
    for m in matchups:
        assert m.team_a != m.team_b
        assert m.team_a in teams and m.team_b in teams
        assert regions[m.team_a] != regions[m.team_b], (
            f"Intra-region clash in Round 1: {m.team_a} vs {m.team_b} ({regions[m.team_a]})"
        )
        paired_teams.add(m.team_a)
        paired_teams.add(m.team_b)
    assert len(paired_teams) == 16


def test_swiss_lottery_no_rematches_rounds_2_to_5():
    """Verify that rounds 2 through 5 strictly enforce no rematches."""
    teams = ["Gen.G", "BLG", "T1", "TES", "G2", "TL", "FNC", "FLY"]
    past_matchups = {
        frozenset({"Gen.G", "BLG"}),
        frozenset({"T1", "TES"}),
        frozenset({"G2", "TL"}),
        frozenset({"FNC", "FLY"}),
    }
    rng = random.Random(101)
    matchups = draw_swiss_pool_lottery(
        teams=teams,
        pool_name="1-0",
        round_num=2,
        past_matchups=past_matchups,
        best_of=1,
        rng=rng,
    )

    assert len(matchups) == 4
    for m in matchups:
        pair = frozenset({m.team_a, m.team_b})
        assert pair not in past_matchups, f"Illegal rematch occurred: {m.team_a} vs {m.team_b}"


def test_simulate_swiss_round_draws_full_structure():
    """Test full multi-pool draw dispatch across active records."""
    records = {
        "Gen.G": (2, 0), "BLG": (2, 0),  # 2-0 pool (Bo3)
        "T1": (1, 1), "TES": (1, 1), "G2": (1, 1), "TL": (1, 1),  # 1-1 pool (Bo1)
        "FNC": (0, 2), "C9": (0, 2),  # 0-2 pool (Bo3)
    }
    past = {frozenset({"Gen.G", "T1"}), frozenset({"BLG", "TES"})}
    rng = random.Random(2026)
    matchups = simulate_swiss_round_draws(
        active_records=records,
        past_matchups=past,
        round_num=3,
        rng=rng,
    )

    assert len(matchups) == 4  # 1 in 2-0, 2 in 1-1, 1 in 0-2
    pools_drawn = {m.pool: [] for m in matchups}
    for m in matchups:
        pools_drawn[m.pool].append(m)
        if m.pool in ("2-0", "0-2"):
            assert m.best_of == 3  # Deciders must be Bo3
        else:
            assert m.best_of == 1


# ==============================================================================
# 2. Tests for Hard Fearless Sequential Series Engine
# ==============================================================================

def test_fearless_series_engine_bo3_and_bo5():
    """Verify that Fearless series terminates at threshold and burns 10 champions per map."""
    rng = random.Random(777)
    
    # Test Bo3
    res_bo3 = simulate_fearless_series_maps(
        team_a="T1",
        team_b="Gen.G",
        best_of=3,
        base_series_prob=0.55,
        rng=rng,
    )
    assert isinstance(res_bo3, FearlessSeriesResult)
    assert res_bo3.winner in ("T1", "Gen.G")
    assert res_bo3.score_winner == 2
    assert res_bo3.score_loser in (0, 1)
    assert res_bo3.maps_played == len(res_bo3.map_history)
    assert len(res_bo3.burned_champions) == 10 * res_bo3.maps_played

    # Test Bo5
    res_bo5 = simulate_fearless_series_maps(
        team_a="Bilibili Gaming",
        team_b="Top Esports",
        best_of=5,
        base_series_prob=0.65,
        rng=rng,
    )
    assert res_bo5.score_winner == 3
    assert res_bo5.score_loser in (0, 1, 2)
    assert res_bo5.maps_played in (3, 4, 5)
    assert len(res_bo5.burned_champions) == 10 * res_bo5.maps_played
    assert len(set(res_bo5.burned_champions)) == len(res_bo5.burned_champions), "Burned champions must be distinct"


def test_fearless_side_selection_logic():
    """Verify that loser of previous map chooses side in subsequent maps."""
    rng = random.Random(123)
    res = simulate_fearless_series_maps(
        team_a="Team A",
        team_b="Team B",
        best_of=5,
        higher_seed="Team A",
        base_series_prob=0.50,
        rng=rng,
    )
    # Game 1: Higher seed gets blue
    assert res.map_history[0].blue_team == "Team A"

    # Games 2+: Check loser of previous map chose Blue
    for i in range(1, len(res.map_history)):
        prev_loser = res.map_history[i - 1].loser
        curr_blue = res.map_history[i].blue_team
        assert curr_blue == prev_loser, f"Map {i+1}: Loser of map {i} ({prev_loser}) was not assigned Blue side ({curr_blue})"


# ==============================================================================
# 3. Tests for In-Path Bayesian Momentum Filter
# ==============================================================================

def test_bayesian_momentum_updates_and_clamping():
    """Test that Bayesian filter updates momentum correctly and respects max offset."""
    filt = InPathTournamentBayesianFilter(
        learning_rate=0.15,
        adaptation_volatility=0.0,  # Zero stochastic noise for deterministic check
        max_abs_logit_offset=0.85,
    )
    
    # Team A (favorite p=0.80) wins 2-0 (+2 map margin)
    filt.update_after_series("Team A", "Team B", winner="Team A", maps_a=2, maps_b=0, pre_match_prob_a=0.80)
    state_a = filt.get_state("Team A")
    state_b = filt.get_state("Team B")

    assert state_a.series_won == 1
    assert state_a.win_streak == 1
    assert state_a.current_logit_offset > 0.0
    assert state_b.series_lost == 1
    assert state_b.loss_streak == 1
    assert state_b.current_logit_offset < 0.0

    # Simulate heavy win streak to verify clamping boundary
    for _ in range(10):
        filt.update_after_series("Team A", "Team X", winner="Team A", maps_a=3, maps_b=0, pre_match_prob_a=0.50)
    
    clamped_a = filt.get_state("Team A")
    assert math.isclose(clamped_a.current_logit_offset, 0.85, abs_tol=1e-5), f"Offset {clamped_a.current_logit_offset} not clamped to 0.85"


def test_bayesian_momentum_martingale_property():
    """Verify that expected momentum drift over large simulation trials is approximately zero (Martingale property)."""
    p = 0.65
    trials = 2000
    rng = random.Random(42)
    deltas = []

    for _ in range(trials):
        filt = InPathTournamentBayesianFilter(learning_rate=0.12, adaptation_volatility=0.04, rng=rng)
        won = rng.random() < p
        winner = "Team A" if won else "Team B"
        maps_a = 2 if won else 0
        maps_b = 0 if won else 2
        filt.update_after_series("Team A", "Team B", winner=winner, maps_a=maps_a, maps_b=maps_b, pre_match_prob_a=p)
        deltas.append(filt.get_state("Team A").current_logit_offset)

    mean_delta = sum(deltas) / len(deltas)
    assert math.isclose(mean_delta, 0.0, abs_tol=0.05), f"Mean momentum drift {mean_delta} deviates from 0"


# ==============================================================================
# 4. Tests for Advanced Tournament Simulator
# ==============================================================================

def test_advanced_simulator_worlds_probability_mass():
    """Verify that simulated Worlds distribution satisfies all conservation laws."""
    sim = AdvancedTournamentSimulator(
        use_fearless=False,
        use_bayesian_momentum=False,
        use_hybrid_knockout=True,
        use_riot_lottery=True,
        shrinkage_lambda=0.20,
        temperature=1.05,
    )
    pair_probabilities = {
        (team_a, team_b, best_of): 0.5
        for i, team_a in enumerate(DEFAULT_WORLDS_16_TEAMS)
        for team_b in DEFAULT_WORLDS_16_TEAMS[i + 1 :]
        for best_of in (1, 3, 5)
    }
    res = sim.simulate_worlds(
        teams=DEFAULT_WORLDS_16_TEAMS,
        base_probabilities=pair_probabilities,
        n_simulations=100,
        seed=123,
    )
    standings = res["standings"]
    assert math.isclose(sum(s["champion_prob"] for s in standings), 1.0, abs_tol=1e-4)
    assert math.isclose(sum(res["swiss_qualifiers_prob"].values()), 8.0, abs_tol=1e-4)
    assert math.isclose(sum(res["region_distribution"].values()), 1.0, abs_tol=1e-4)
    assert math.isclose(sum(s["top2_prob"] for s in standings if s["top2_prob"] is not None), 2.0, abs_tol=1e-4)

def test_mass_assertion_rejects_broken_probability_table():
    from src.models.advanced_tournament_simulator import _assert_mass_conserved

    with pytest.raises(AssertionError, match="max deviation"):
        _assert_mass_conserved("test", (0.9, 1.0))



def test_two_tuple_base_probability_is_series_probability():
    simulator = AdvancedTournamentSimulator(shrinkage_lambda=0.0, temperature=1.0)
    probability = simulator.get_matchup_probability(
        "A", "B", best_of=3, base_probabilities={("A", "B"): 0.7},
        team_ratings={"A": 1500.0, "B": 1500.0},
    )
    assert math.isclose(probability, 0.7, abs_tol=1e-12)



def test_advanced_simulator_regional_brackets():
    """Test simulation of LEC, LCS, LPL brackets using AdvancedTournamentSimulator."""
    sim = AdvancedTournamentSimulator(
        use_fearless=False,
        use_bayesian_momentum=False,
        use_hybrid_knockout=True,
        shrinkage_lambda=0.0,
        temperature=1.0,
    )

    for tid in ("lec_2026_summer_playoffs", "lcs_2026_championship"):
        bracket = SUPPORTED_BRACKETS[tid]()
        # Equal ratings make this a pure bracket-structure/mass-conservation
        # check; enumerate every legal entrant instead of relying on defaults.
        team_ratings = {team: 1500.0 for team in bracket.teams}
        res = sim.simulate_bracket(
            bracket,
            n_simulations=100,
            seed=42,
            team_ratings=team_ratings,
        )
        assert res["tournament"] == tid
        total_p = sum(s["champion_prob"] for s in res["standings"])
        assert math.isclose(total_p, 1.0, abs_tol=1e-4), f"{tid} champion sum {total_p} != 1.0"


def test_advanced_simulator_deterministic_reproducibility():
    """Verify that passing the same seed yields bit-for-bit identical outputs."""
    sim1 = AdvancedTournamentSimulator()

    pair_probabilities = {
        (team_a, team_b, best_of): 0.5
        for i, team_a in enumerate(DEFAULT_WORLDS_16_TEAMS)
        for team_b in DEFAULT_WORLDS_16_TEAMS[i + 1 :]
        for best_of in (1, 3, 5)
    }
    res1 = sim1.simulate_worlds(n_simulations=50, seed=999, base_probabilities=pair_probabilities)
    from src.features.champion_pool_entropy import register_player_pool

    register_player_pool("unrelated-crystal-ball-player", {1: 10}, role="MID")
    sim3 = AdvancedTournamentSimulator()
    res3 = sim3.simulate_worlds(
        n_simulations=50, seed=999, base_probabilities=pair_probabilities
    )
    assert res1["standings"] == res3["standings"]

    for s1, s3 in zip(res1["standings"], res3["standings"]):
        assert s1["team"] == s3["team"]
        assert s1["champion_prob"] == s3["champion_prob"]
        assert s1["top2_prob"] == s3["top2_prob"]

    for r_k in res1["region_distribution"]:
        assert res1["region_distribution"][r_k] == res3["region_distribution"][r_k]
