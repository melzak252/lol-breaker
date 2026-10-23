"""Staking helpers for diagnostic bet sizing."""

from __future__ import annotations


def fixed_stake(amount: float, bankroll: float) -> float:
    """Return a fixed stake capped by bankroll."""

    return max(0.0, min(float(amount), float(bankroll)))


def percent_stake(bankroll: float, fraction: float, min_stake: float = 2.0, max_stake: float = 100.0) -> float:
    """Return a bankroll-percent stake with min/max limits."""

    if bankroll <= 0:
        return 0.0
    stake = bankroll * fraction
    stake = min(max(stake, min_stake), max_stake)
    return stake if stake <= bankroll else 0.0


def fractional_kelly_stake(
    bankroll: float,
    probability: float,
    decimal_odds: float,
    fraction: float = 0.25,
    tax_rate: float = 0.12,
    min_stake: float = 2.0,
    max_stake: float = 150.0,
    reserved_bankroll: float = 0.0,
    max_portfolio_exposure: float = 0.25,
) -> float:
    """Calculate fractional Kelly with free-cash sizing and total-bankroll headroom.

    Kelly sizing uses free cash (nominal bankroll less nonnegative reservations);
    the per-bet exposure ceiling is remaining headroom against nominal bankroll.
    """
    total_bankroll = max(0.0, float(bankroll))
    reserved = max(0.0, float(reserved_bankroll))
    effective_bankroll = max(0.0, total_bankroll - reserved)
    headroom = max(0.0, total_bankroll * max_portfolio_exposure - reserved)
    if effective_bankroll <= 0 or headroom <= 0 or decimal_odds <= 1.0 or not 0 < probability < 1:
        return 0.0
    net_decimal = decimal_odds * (1.0 - tax_rate)
    b_value = net_decimal - 1.0
    if b_value <= 0:
        return 0.0
    full_kelly = ((b_value * probability) - (1.0 - probability)) / b_value
    if full_kelly <= 0:
        return 0.0
    stake = effective_bankroll * full_kelly * fraction
    effective_max = min(float(max_stake), headroom)
    if effective_max < min_stake:
        return 0.0
    stake = min(max(stake, min_stake), effective_max)
    return stake if stake <= min(effective_bankroll, headroom) else 0.0


def step_compounded_flat_stake(
    bankroll: float,
    unit_fraction: float = 0.03,
    min_stake: float = 2.0,
    max_stake: float = 150.0,
    reserved_bankroll: float = 0.0,
    max_portfolio_exposure: float = 0.25,
    integer_round: bool = True,
) -> float:
    """Calculate flat unit from free cash, capped by remaining exposure headroom.

    Integer rounding is clamped downward to ensure the stake never exceeds
    available cash or the portfolio exposure allowance.
    """
    total_bankroll = max(0.0, float(bankroll))
    reserved = max(0.0, float(reserved_bankroll))
    effective_bankroll = max(0.0, total_bankroll - reserved)
    headroom = max(0.0, total_bankroll * max_portfolio_exposure - reserved)
    if effective_bankroll < min_stake or headroom <= 0 or unit_fraction <= 0:
        return 0.0
    raw_unit = effective_bankroll * float(unit_fraction)
    if integer_round:
        raw_unit = float(round(raw_unit))
    effective_max = min(float(max_stake), headroom)
    if integer_round:
        effective_max = float(int(effective_max))
    if effective_max < min_stake:
        return 0.0
    stake = min(max(raw_unit, float(min_stake)), effective_max)
    return stake if stake <= min(effective_bankroll, headroom) else 0.0


def capped_wasserstein_kelly_stake(
    bankroll: float,
    probability: float,
    decimal_odds: float,
    wasserstein_epsilon: float = 0.025,
    fraction: float = 0.25,
    tax_rate: float = 0.12,
    min_stake: float = 2.0,
    max_stake: float = 150.0,
    reserved_bankroll: float = 0.0,
    max_portfolio_exposure: float = 0.25,
    integer_round: bool = False,
) -> float:
    """Calculate robust Kelly stake using free cash and remaining exposure headroom.

    The probability adjustment remains ``max(0, probability - epsilon)``; integer
    rounding is clamped to avoid exceeding available cash or exposure headroom.
    """
    prob_dro = max(0.0, float(probability) - max(0.0, float(wasserstein_epsilon)))
    if prob_dro <= 0.0:
        return 0.0
    stake = fractional_kelly_stake(
        bankroll=bankroll,
        probability=prob_dro,
        decimal_odds=decimal_odds,
        fraction=fraction,
        tax_rate=tax_rate,
        min_stake=min_stake,
        max_stake=max_stake,
        reserved_bankroll=reserved_bankroll,
        max_portfolio_exposure=max_portfolio_exposure,
    )
    if integer_round and stake > 0.0:
        total_bankroll = max(0.0, float(bankroll))
        reserved = max(0.0, float(reserved_bankroll))
        effective_bankroll = max(0.0, total_bankroll - reserved)
        headroom = max(0.0, total_bankroll * max_portfolio_exposure - reserved)
        integer_cap = min(int(max_stake), int(headroom), int(effective_bankroll))
        if integer_cap < min_stake:
            return 0.0
        stake = float(min(integer_cap, max(int(min_stake), round(stake))))
        if stake > effective_bankroll or stake > headroom:
            return 0.0
    return stake
