"""Consumer contracts for match recommendations; no database required."""

import pytest
from datetime import UTC, datetime

from betting_app.api.routers.matches import _build_match_recommendation
from betting_app.api.schemas import BookmakerOddsRow
from betting_app.services.market_service import kelly_fraction
from betting_app.services.prematch_quote_service import (
    PrematchMarketContext,
    PrematchQuote,
    quotes_consensus,
)


def market_context(odds_by_book=None):
    now = datetime.now(UTC)
    odds_by_book = odds_by_book or {
        1: (1.6, 2.4),
        2: (2.0, 2.0),
        3: (2.0, 2.0),
    }
    quotes = tuple(
        PrematchQuote(
            bookmaker_id, 1, bookmaker_id, "sts" if bookmaker_id == 1 else f"book{bookmaker_id}",
            now, "Alpha", "Beta", odds_a, odds_b, None, None,
        )
        for bookmaker_id, (odds_a, odds_b) in odds_by_book.items()
    )
    return PrematchMarketContext(
        1, "alpha", "beta", "LCK", "upcoming", None, now, quotes,
        quotes_consensus(quotes).prob_a, (),
    )


def test_recommendation_without_quote_context_withholds_even_when_display_odds_exist():
    params = dict(
        team_a_name="Alpha", team_b_name="Beta", has_unmapped_teams=False,
        odds_rows=[
            BookmakerOddsRow(bookmaker="sts", canonical_odds_a=1.6, canonical_odds_b=2.4,
                             odds_snapshot_id=1, quote_status="actionable")
        ],
        hybrid_prob_a=0.45, hybrid_prob_b=0.55,
        hybrid_diagnostics={"base_prob_a": 0.4},
    )
    rec = _build_match_recommendation(**params)
    assert not rec.has_value




def recommendation(**changes):
    params = dict(
        team_a_name="Alpha",
        team_b_name="Beta",
        has_unmapped_teams=False,
        odds_rows=[
            BookmakerOddsRow(
                bookmaker="sts", canonical_odds_a=1.6, canonical_odds_b=2.4,
                odds_snapshot_id=1, quote_scraped_at=datetime.now(UTC).isoformat(), quote_status="actionable",
            )
        ],
        hybrid_prob_a=0.45,
        hybrid_prob_b=0.55,
        hybrid_diagnostics={
            "base_prob_a": 0.4, "alpha": 0.5, "temperature": 1.0,
            "blending_mode": "logit_shrinkage", "uncertainty_required": False,
            "rating_disagreement": 0.0,
        },
        context=market_context(),
    )
    params.update(changes)
    return _build_match_recommendation(**params)


def test_unmapped_or_missing_odds_cannot_recommend():
    assert not recommendation(has_unmapped_teams=True).has_value
    assert not recommendation(odds_rows=[]).has_value


def test_qualified_side_uses_focal_probability_for_ev_and_stake():
    rec = recommendation()
    assert rec.has_value and rec.side == "b"
    expected_probability = 0.6 ** 0.5 / (0.4 ** 0.5 + 0.6 ** 0.5)
    assert rec.hybrid_prob == pytest.approx(round(expected_probability, 4))
    assert rec.ev == pytest.approx(expected_probability * 2.4 * 0.88 - 1, abs=1e-4)
    assert rec.quarter_kelly == pytest.approx(
        kelly_fraction(expected_probability, 2.4, 0.12) / 4, abs=1e-4
    )


def test_missing_hybrid_prediction_cannot_be_replaced_by_pure_prediction():
    rec = recommendation(
        hybrid_diagnostics=None,
    )
    assert not rec.has_value



def test_missing_hybrid_base_probability_fails_closed():
    rec = recommendation(hybrid_diagnostics={})
    assert not rec.has_value
    assert "prediction" in rec.verdict


def test_changed_sources_cannot_restore_pure_prediction_when_context_is_withheld():
    rec = recommendation(context=None, hybrid_prob_a=None, hybrid_prob_b=None)
    assert not rec.has_value


def test_focal_consensus_requires_two_other_bookmakers():
    context = market_context({1: (1.6, 2.4), 2: (2.0, 2.0)})
    rec = recommendation(context=context)
    assert not rec.has_value

def test_tier1_mean_market_gap_still_blocks_recommendation():
    rec = recommendation(
        league="LCK 2026",
        hybrid_prob_a=0.2,
        hybrid_prob_b=0.8,
        hybrid_diagnostics={
            "base_prob_a": 0.1, "alpha": 0.5, "temperature": 1.0,
            "blending_mode": "logit_shrinkage", "uncertainty_required": False,
        },
    )
    assert not rec.has_value


@pytest.mark.parametrize(
    ("odds_a", "expected_side"),
    [(2.58, "a"), (2.6, "b")],
)
def test_side_with_larger_mean_ev_respects_ev_cap(odds_a, expected_side):
    context = market_context({1: (odds_a, 3.0), 2: (2.0, 2.0), 3: (2.0, 2.0)})
    rec = recommendation(
        hybrid_prob_a=0.55,
        hybrid_prob_b=0.45,
        odds_rows=[
            BookmakerOddsRow(
                bookmaker="sts", canonical_odds_a=odds_a, canonical_odds_b=3.0,
                odds_snapshot_id=1, quote_scraped_at=datetime.now(UTC).isoformat(), quote_status="actionable",
            )
        ],
        context=context,
        hybrid_diagnostics={
            "base_prob_a": 0.599009900990099, "alpha": 0.5, "temperature": 1.0,
            "blending_mode": "logit_shrinkage", "uncertainty_required": False,
            "rating_disagreement": 0.0,
        },
    )
    assert rec.has_value and rec.side == expected_side
    expected_odds = odds_a if expected_side == "a" else 3.0
    expected_prob = 0.55 if expected_side == "a" else 0.45
    assert rec.ev == pytest.approx(expected_prob * expected_odds * 0.88 - 1, abs=1e-4)
    assert rec.hybrid_prob == pytest.approx(expected_prob)
    assert rec.model_prob == pytest.approx(0.5990 if expected_side == "a" else 0.4010)


def test_quote_ev_above_signal_cap_is_not_recommended():
    context = market_context({1: (8.0, 8.0), 2: (2.0, 2.0), 3: (2.0, 2.0)})
    rec = recommendation(
        odds_rows=[
            BookmakerOddsRow(
                bookmaker="sts", canonical_odds_a=8.0, canonical_odds_b=8.0,
                odds_snapshot_id=1, quote_status="actionable",
            )
        ],
        context=context,
    )
    assert not rec.has_value


def test_bo1_market_gap_gate_still_blocks_match_recommendation():
    context = market_context({1: (1.7, 2.4), 2: (2.0, 2.0), 3: (2.0, 2.0)})
    rec = recommendation(
        odds_rows=[
            BookmakerOddsRow(
                bookmaker="sts", canonical_odds_a=1.7, canonical_odds_b=2.4,
                odds_snapshot_id=1, quote_status="actionable",
            )
        ],
        hybrid_prob_a=0.7,
        hybrid_prob_b=0.3,
        hybrid_diagnostics={
            "base_prob_a": 0.9, "alpha": 0.5, "temperature": 1.0,
            "blending_mode": "logit_shrinkage", "uncertainty_required": False,
        },
        context=context,
        best_of=1,
    )
    assert not rec.has_value


def test_recommendation_evaluates_other_independent_book_quotes():
    context = market_context({
        1: (8.0, 8.0),
        2: (2.4, 2.4),
        3: (2.4, 2.4),
    })
    rec = recommendation(
        odds_rows=[
            BookmakerOddsRow(
                bookmaker=f"book{book_id}", canonical_odds_a=odds_a, canonical_odds_b=odds_b,
                odds_snapshot_id=book_id, quote_status="actionable",
            )
            for book_id, odds_a, odds_b in (
                (1, 8.0, 8.0), (2, 2.4, 2.4), (3, 2.4, 2.4),
            )
        ],
        hybrid_prob_a=0.65,
        hybrid_prob_b=0.35,
        hybrid_diagnostics={
            "base_prob_a": 0.6, "alpha": 0.5, "temperature": 1.0,
            "blending_mode": "logit_shrinkage", "uncertainty_required": False,
        },
        context=context,
    )
    assert rec.has_value
    assert rec.bookmaker in {"book2", "book3"}
