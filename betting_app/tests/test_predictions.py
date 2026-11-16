"""Tests for /api/predictions endpoint."""

from datetime import UTC, datetime, timedelta

import pytest

from fastapi.testclient import TestClient

class TestPredictions:
    def test_empty_returns_zero(self, client: TestClient):
        resp = client.get("/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["signals"] == []


@pytest.mark.parametrize(
    "basis, required, bound, expected_ev",
    [
        ("mean", False, None, 0.2144),
        ("measured_bounds", True, 0.44, 0.1616),
        ("assumed_research_haircut", True, 0.44, 0.1616),
        ("missing_bounds", True, None, None),
    ],
)
def test_prediction_signals_disclose_actual_ev_score_without_inventing_bound(
    monkeypatch, basis, required, bound, expected_ev,
):
    from betting_app.api.routers import predictions
    from betting_app.core.models import C0_NATIVE
    from betting_app.services.prematch_quote_service import (
        PrematchMarketContext, PrematchQuote, quotes_consensus,
        serialize_focal_consensus, serialize_market_context,
    )

    now = datetime.now(UTC)
    quotes = tuple(
        PrematchQuote(i, 4, i, f"book{i}", now - timedelta(seconds=10),
                      "Alpha", "Beta", 3.0, 2.0, None, None)
        for i in (1, 2, 3)
    )
    context = PrematchMarketContext(
        4, "alpha", "beta", "LCK", "upcoming", now + timedelta(hours=1),
        now, quotes, quotes_consensus(quotes).prob_a, (),
    )
    diagnostics = {"uncertainty_required": required, "uncertainty_basis": basis}
    if bound is not None:
        diagnostics.update(
            p_low_a=bound, p_low_b=0.52, epistemic_sigma_z=0.15,
            rating_disagreement=0.0,
        )
    focal = quotes_consensus(quotes, exclude_bookmaker_id=1)
    row = dict(
        id=8, prediction_id=9, prediction_match_id=4, canonical_prediction_id=9,
        canonical_match_id=4, bookmaker_id=1, odds_snapshot_id=1, side="a", odds=3.0,
        model_name=C0_NATIVE.name, model_version=C0_NATIVE.version,
        prob_a=0.46, prob_b=0.54, model_prob=0.46, market_prob=focal.prob_a,
        ev=9.0, evaluated_at=now, status="new", prediction_status="active",
        predicted_at=now - timedelta(seconds=20),
        data_cutoff_at=now - timedelta(minutes=1),
        team_a_name="Alpha", team_b_name="Beta", league="LCK", best_of=3,
        start_time_normalized=context.start_at.isoformat(), bookmaker_name="book1",
        diagnostics_json={**diagnostics, "base_prob_a": 0.46},
        market_context_json={
            **serialize_market_context(
                context, model_name=C0_NATIVE.name, model_version=C0_NATIVE.version,
            ),
            "focal_consensus": serialize_focal_consensus(focal),
        },
    )
    monkeypatch.setattr(predictions, "query_df", lambda *args: [row])
    monkeypatch.setattr(predictions, "load_prematch_quotes", lambda *args, **kwargs: {4: context})
    result = predictions.list_predictions(
        min_ev=0.0, max_ev=None, min_books=3, model_name=C0_NATIVE.name,
        model_version=C0_NATIVE.version, limit=50, db=object(),
    )

    if expected_ev is None:
        assert result.total == 0
        assert result.signals == []
    else:
        assert result.total == 1
        signal = result.signals[0]
        payload = result.model_dump()["signals"][0]
        score = 0.46 if bound is None else bound
        assert payload["prob_conservative"] == bound
        assert payload["prob_for_ev"] == score
        assert payload["uncertainty_basis"] == basis
        assert payload["uncertainty_reason"] == {
            "mean": "uncertainty_not_requested",
            "measured_bounds": "provided_lower_bound",
            "assumed_research_haircut": "explicit_research_assumption",
        }[basis]
        assert signal.ev == pytest.approx(expected_ev, abs=1e-12)
        assert signal.kelly == pytest.approx(expected_ev / (3.0 * 0.88 - 1.0), abs=1e-12)


@pytest.mark.parametrize("prob_a, expected_bets", [(0.5, 0), (0.7, 1)])
def test_timing_research_haircut_preserves_empty_conditional_results(
    monkeypatch, prob_a, expected_bets,
):
    from betting_app.api.routers import timing

    start = datetime.now(UTC) - timedelta(hours=1)
    matches = [{
        "canonical_match_id": 4, "team_a_name": "Alpha", "team_b_name": "Beta",
        "winner_side": "team_a", "start_time_normalized": start.isoformat(),
        "league": "LCK", "best_of": 3,
    }]
    predictions = [{
        "canonical_match_id": 4, "prob_a": prob_a, "prob_b": 1.0 - prob_a,
    }]
    quotes = [{
        "canonical_match_id": 4, "bookmaker": "book1", "odds_a": 2.0, "odds_b": 2.0,
        "scraped_at": (start - timedelta(hours=2)).isoformat(),
    }]

    def query(db, sql, params=None):
        if "FROM canonical_matches" in sql:
            return matches
        if "FROM canonical_predictions" in sql:
            return predictions
        if "FROM odds_snapshots" in sql:
            return quotes
        raise AssertionError("Unexpected validation-report query")

    monkeypatch.setattr(timing, "query_df", query)
    result = timing.validation_report(db=object())
    totals = result["betting_totals"]
    assert totals["uncertainty_basis"] == "assumed_research_haircut"
    assert totals["uncertainty_reason"] == "explicit_research_assumption"
    assert totals["assumed_sigma_z"] == 0.15
    assert totals["total_bets"] == expected_bets
    if expected_bets == 0:
        assert totals["result_status"] == "no_eligible_bets"
        assert totals["roi_pct"] is None
        assert totals["win_rate_pct"] is None
        assert totals["avg_clv_pct"] is None
        assert totals["pos_clv_pct"] is None
        assert totals["total_pnl_pln"] == 0.0
        assert totals["total_staked_pln"] == 0.0
        assert result["odds_brackets"] == []
    else:
        assert totals["result_status"] == "computed"
        assert totals["win_rate_pct"] == 100.0
        assert totals["roi_pct"] == 76.0
        assert totals["total_staked_pln"] == 100.0
