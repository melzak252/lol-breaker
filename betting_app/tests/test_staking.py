"""Unit tests for portfolio-aware fractional Kelly staking."""

from __future__ import annotations

import pytest

from betting_app.core.staking import (
    fixed_stake,
    fractional_kelly_stake,
    percent_stake,
)


def test_fractional_kelly_standard():
    # Bankroll = 1000 PLN, Win Prob = 0.60, Decimal Odds = 2.00, Tax = 12%
    # Net Decimal = 2.0 * 0.88 = 1.76, b = 0.76
    # Full Kelly = (0.76 * 0.60 - 0.40) / 0.76 = (0.456 - 0.40) / 0.76 = 0.056 / 0.76 ~= 0.07368 (7.37%)
    # 5% fractional Kelly = 0.07368 * 0.05 = 0.003684 (0.37% of 1000 = 3.68 PLN)
    stake = fractional_kelly_stake(1000.0, 0.60, 2.00, fraction=0.05, tax_rate=0.12)
    assert 3.5 <= stake <= 4.0


def test_quarter_kelly_default():
    # Default fraction is now 0.25 (1/4 Kelly)
    # 1000 PLN * 0.07368 * 0.25 = 18.42 PLN
    stake = fractional_kelly_stake(1000.0, 0.60, 2.00, tax_rate=0.12)
    assert 18.0 <= stake <= 19.0

def test_fractional_kelly_limits_stake_to_aggregate_headroom():
    # Nominal equity is 1,000; 240 is already reserved, so only 10 of the
    # 25%-of-equity aggregate limit remains even though 760 is free cash.
    stake = fractional_kelly_stake(
        1000.0, 0.99, 3.0, fraction=1.0, tax_rate=0.0,
        reserved_bankroll=240.0,
    )
    assert stake == pytest.approx(10.0)


def test_fractional_kelly_returns_zero_when_aggregate_cap_is_exhausted():
    stake = fractional_kelly_stake(
        1000.0, 0.99, 3.0, fraction=1.0, tax_rate=0.0,
        reserved_bankroll=250.0,
    )
    assert stake == 0.0


def test_fractional_kelly_does_not_raise_stake_to_minimum_above_headroom():
    stake = fractional_kelly_stake(
        1000.0, 0.99, 3.0, fraction=1.0, tax_rate=0.0,
        min_stake=2.0, reserved_bankroll=248.5,
    )
    assert stake == 0.0


def test_fractional_kelly_uses_free_cash_for_kelly_and_respects_maximum():
    # Kelly on free cash (600) exceeds max_stake; neither nominal bankroll
    # nor the aggregate headroom should replace the single-ticket maximum.
    stake = fractional_kelly_stake(
        1000.0, 0.99, 3.0, fraction=1.0, tax_rate=0.0,
        max_stake=80.0, reserved_bankroll=100.0,
    )
    assert stake == 80.0


def test_fractional_kelly_cannot_exceed_free_cash():
    stake = fractional_kelly_stake(
        1000.0, 0.99, 3.0, fraction=1.0, tax_rate=0.0,
        max_stake=1000.0, reserved_bankroll=900.0,
        max_portfolio_exposure=1.0,
    )
    assert stake == pytest.approx(98.5)
    assert stake <= 100.0


def test_fractional_kelly_enforces_cap_for_over_cap_existing_exposure():
    stake = fractional_kelly_stake(
        900.0, 0.99, 3.0, fraction=1.0, tax_rate=0.0,
        reserved_bankroll=250.0,
    )
    assert stake == 0.0


def test_step_compounded_integer_rounding_stays_within_aggregate_headroom():
    from betting_app.core.staking import step_compounded_flat_stake

    stake = step_compounded_flat_stake(
        1000.0, unit_fraction=1.0, min_stake=2.0, max_stake=150.0,
        reserved_bankroll=244.3, integer_round=True,
    )
    assert stake <= 5.7


def test_wasserstein_integer_rounding_stays_within_aggregate_headroom():
    from betting_app.core.staking import capped_wasserstein_kelly_stake

    stake = capped_wasserstein_kelly_stake(
        1000.0, 0.99, 3.0, wasserstein_epsilon=0.0,
        fraction=1.0, tax_rate=0.0, min_stake=2.0, max_stake=150.0,
        reserved_bankroll=244.3, integer_round=True,
    )
    assert stake <= 5.7

def test_fractional_kelly_applies_exposure_limit_when_no_cap_is_used():
    # With no existing reservations, aggregate headroom is 20% of equity.
    stake = fractional_kelly_stake(
        100.0,
        0.90,
        2.50,
        fraction=1.0,  # Full Kelly
        reserved_bankroll=0.0,
        max_portfolio_exposure=0.20,
    )
    assert stake <= 20.0


def test_fractional_kelly_negative_ev():
    # Odds = 1.50, Prob = 0.50 -> Net = 1.32, b = 0.32 -> EV is negative
    stake = fractional_kelly_stake(1000.0, 0.50, 1.50)
    assert stake == 0.0


def test_fixed_and_percent_stake():
    assert fixed_stake(50.0, 100.0) == 50.0
    assert fixed_stake(150.0, 100.0) == 100.0
    assert percent_stake(100.0, 0.05) == 5.0
    assert percent_stake(10.0, 0.01) == 2.0  # clamped to min_stake 2.0
    assert percent_stake(1.0, 0.01) == 0.0  # cannot meet min_stake within 1.0 bankroll
