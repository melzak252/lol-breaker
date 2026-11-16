from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from betting_app.core.staking import fractional_kelly_stake
from src.analysis.unified_evaluation_engine import simulate_operational_portfolio


DECISION = datetime(2026, 6, 1, 12, tzinfo=UTC)
TAX_RATE = 0.12


def _candidate(
    opportunity_id,
    canonical_match_id,
    *,
    decision_at=DECISION,
    settle_at=None,
    side="team_a",
    bookmaker="Book",
    odds=2.0,
    probability=0.60,
    y=None,
):
    return {
        "opportunity_id": opportunity_id,
        "canonical_match_id": canonical_match_id,
        "decision_at": decision_at,
        "settle_at": settle_at or decision_at + timedelta(hours=4),
        "side": side,
        "bookmaker": bookmaker,
        "odds": odds,
        "prob_for_ev": probability,
        "ev_net": probability * odds * (1.0 - TAX_RATE) - 1.0,
        "y": y,
    }


def _frames(candidates):
    opportunities_by_id = {}
    for candidate in candidates:
        opportunities_by_id.setdefault(
            candidate["opportunity_id"],
            {
                "opportunity_id": candidate["opportunity_id"],
                "canonical_match_id": candidate["canonical_match_id"],
                "decision_at": candidate["decision_at"],
                "start_at": candidate["decision_at"] + timedelta(hours=1),
                "settle_at": candidate["settle_at"],
                "league": "LCK",
                "best_of": 3,
                "y": candidate["y"],
                "absence_flags": [],
            },
        )
    return pd.DataFrame(candidates), pd.DataFrame(opportunities_by_id.values())


def _simulate(candidates, *, initial_bankroll=10_000, **kwargs):
    candidate_frame, opportunities = _frames(candidates)
    return simulate_operational_portfolio(
        candidate_frame,
        opportunities,
        initial_bankroll=initial_bankroll,
        **kwargs,
    )


def _assert_decimal_accounting(summary):
    for field in ("final_cash", "final_reserved", "final_equity", "profit", "turnover"):
        assert isinstance(summary[field], Decimal)


def test_overlapping_open_bets_size_against_cash_free_of_reserved_stakes():
    candidates = [
        _candidate("first", 101, y=None),
        _candidate(
            "second",
            102,
            decision_at=DECISION + timedelta(minutes=1),
            y=None,
        ),
    ]

    result = _simulate(candidates, initial_bankroll=1_000)
    ledger = result["ledger"]
    first_stake, second_stake = ledger["stake"].tolist()
    expected_first = fractional_kelly_stake(1_000, 0.60, 2.0)
    expected_second = fractional_kelly_stake(
        1_000, 0.60, 2.0, reserved_bankroll=expected_first
    )

    assert ledger["opportunity_id"].tolist() == ["first", "second"]
    assert isinstance(first_stake, Decimal)
    assert isinstance(second_stake, Decimal)
    assert float(first_stake) == pytest.approx(expected_first)
    assert float(second_stake) == pytest.approx(expected_second)
    assert second_stake < first_stake
    assert (
        result["summary"]["final_cash"] == Decimal("1000") - first_stake - second_stake
    )
    assert result["summary"]["final_reserved"] == first_stake + second_stake
    assert result["summary"]["final_equity"] == Decimal("1000")
    assert result["summary"]["profit"] == Decimal("0")
    assert result["summary"]["max_drawdown"] == 0


def test_settlement_precedes_a_decision_at_the_same_utc_instant():
    settlement = DECISION + timedelta(hours=1)
    candidates = [
        _candidate("winner", 201, probability=0.99, y=1, settle_at=settlement),
        _candidate(
            "next",
            202,
            decision_at=settlement,
            y=None,
        ),
    ]

    result = _simulate(candidates, initial_bankroll=1_000)
    ledger = result["ledger"]
    first_stake, second_stake = ledger["stake"].tolist()
    assert first_stake == Decimal("150")
    expected_second = fractional_kelly_stake(1_114, 0.60, 2.0)

    assert ledger["opportunity_id"].tolist() == ["winner", "next"]
    assert isinstance(second_stake, Decimal)
    assert float(second_stake) == pytest.approx(expected_second)
    assert result["summary"]["final_cash"] == Decimal("1114") - second_stake
    assert result["summary"]["final_reserved"] == second_stake
    assert result["summary"]["final_equity"] == Decimal("1114")
    assert result["summary"]["profit"] == Decimal("114")


def test_winning_payout_applies_exact_twelve_percent_turnover_tax():
    result = _simulate(
        [_candidate("winner", 301, probability=0.99, y=1)],
        initial_bankroll=10_000,
    )
    ledger = result["ledger"]
    summary = result["summary"]

    assert ledger.loc[0, "stake"] == Decimal("150")
    assert summary["final_cash"] == Decimal("10114")
    assert summary["final_reserved"] == Decimal("0")
    assert summary["final_equity"] == Decimal("10114")
    assert summary["profit"] == Decimal("114")
    assert summary["turnover"] == Decimal("150")
    _assert_decimal_accounting(summary)
    assert Decimal(str(summary["roi"])) == Decimal("114") / Decimal("150")


def test_unresolved_outcome_keeps_stake_reserved_without_recording_a_loss():
    result = _simulate(
        [_candidate("pending", 401, y=None)],
        initial_bankroll=1_000,
    )
    summary = result["summary"]
    stake = result["ledger"].loc[0, "stake"]

    assert isinstance(stake, Decimal)
    assert summary["final_cash"] == Decimal("1000") - stake
    assert summary["final_reserved"] == stake
    assert summary["final_equity"] == Decimal("1000")
    assert summary["profit"] == Decimal("0")
    assert summary["turnover"] == stake
    assert summary["max_drawdown"] == 0


def test_duplicate_fixture_side_and_book_is_not_staked_twice():
    duplicate = _candidate("same-opportunity", 501, y=None)
    result = _simulate([duplicate, duplicate], initial_bankroll=1_000)
    ledger = result["ledger"]

    assert len(ledger) == 1
    assert not ledger.duplicated(["canonical_match_id", "side", "bookmaker"]).any()
    assert result["summary"]["final_reserved"] == ledger.loc[0, "stake"]


def test_stakes_respect_kelly_minimum_maximum_and_available_capital():
    candidates = [
        _candidate("large-value", 601, probability=0.99, y=None),
        _candidate("minimum-value", 602, probability=0.5683, y=None),
    ]
    result = _simulate(candidates)
    ledger = result["ledger"]
    summary = result["summary"]

    stakes = ledger["stake"].tolist()
    assert stakes == [Decimal("150"), Decimal("2")]
    assert all(Decimal("2") <= stake <= Decimal("150") for stake in stakes)
    assert summary["final_cash"] >= 0
    assert summary["final_reserved"] == sum(stakes, Decimal("0"))
    assert summary["final_cash"] + summary["final_reserved"] == summary["final_equity"]
    assert summary["final_equity"] == Decimal("10000")

    capital_path = result["capital_path"]
    for row in capital_path.itertuples(index=False):
        assert isinstance(row.cash, Decimal)
        assert isinstance(row.reserved, Decimal)
        assert isinstance(row.equity, Decimal)
        assert row.cash >= 0
        assert row.reserved >= 0
        assert row.cash + row.reserved == row.equity


def test_same_time_ev_priority_and_ties_are_input_permutation_stable():
    candidates = [
        _candidate("a-low", 701, probability=0.60, y=None),
        _candidate("c-tie", 702, probability=0.65, y=None),
        _candidate("z-high", 703, probability=0.70, y=None),
        _candidate("b-tie", 704, probability=0.65, y=None),
    ]
    forward = _simulate(candidates, initial_bankroll=1_000)
    reverse = _simulate(list(reversed(candidates)), initial_bankroll=1_000)

    assert forward["ledger"]["opportunity_id"].tolist() == [
        "z-high",
        "b-tie",
        "c-tie",
        "a-low",
    ]
    pd.testing.assert_frame_equal(forward["ledger"], reverse["ledger"])
    pd.testing.assert_frame_equal(forward["capital_path"], reverse["capital_path"])
    assert forward["summary"] == reverse["summary"]
    assert {
        "final_cash",
        "final_reserved",
        "final_equity",
        "profit",
        "turnover",
        "roi",
        "max_drawdown",
    } <= forward["summary"].keys()


def test_best_price_selected_before_ev_priority_and_distinct_opportunities_counted():
    worse = _candidate("quote-low", 801, odds=2.0, probability=0.70)
    better = _candidate("quote-best", 801, odds=3.0, probability=0.60)
    other = _candidate("other", 802, probability=0.70)
    result = _simulate([worse, other, better], initial_bankroll=1_000)

    fixture_bet = (
        result["ledger"].loc[result["ledger"]["canonical_match_id"] == 801].iloc[0]
    )
    assert fixture_bet["opportunity_id"] == "quote-best"
    assert result["summary"]["unselected_distinct_opportunity_count"] == 1


def test_initial_capital_snapshot_exists_when_no_candidates():
    result = simulate_operational_portfolio(
        pd.DataFrame(), pd.DataFrame(), initial_bankroll=1_000
    )
    assert len(result["capital_path"]) == 1
    snapshot = result["capital_path"].iloc[0]
    assert snapshot["event"] == "initial"
    assert snapshot["cash"] == Decimal("1000")
    assert snapshot["reserved"] == Decimal("0")


def test_opposite_sides_settle_using_oriented_outcomes():
    a = _candidate("a", 901, side="team_a", y=1)
    b = _candidate("b", 901, side="team_b", y=1)
    result = _simulate([a, b], initial_bankroll=1_000)
    assert result["ledger"]["y"].tolist() == [1, 0]
    assert result["summary"]["settled_bets_count"] == 2
