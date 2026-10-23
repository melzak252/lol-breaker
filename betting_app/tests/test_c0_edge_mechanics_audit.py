import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from betting_app.api.routers.matches import (
    _latest_display_states,
    _serving_predictions,
)
from betting_app.core.models.engine import (
    _series_probability_binomial,
    blend_market_probability,
)
from betting_app.core.models.registry import get_active_hybrid
from betting_app.core.staking import fractional_kelly_stake
from betting_app.services.bet_qualification_service import is_bet_eligible
from betting_app.services.market_service import expected_value
from betting_app.services.prematch_quote_service import (
    PrematchMarketContext,
    PrematchQuote,
    prediction_time_is_actionable,
    quotes_consensus,
)
from src.analysis.unified_evaluation_engine import simulate_operational_portfolio


TAX = 0.12
NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


def test_series_probability_is_blended_once_and_a_b_remain_complements():
    # A map probability is converted to its BO3 series probability by the model
    # family once; the market blend consumes that series probability unchanged.
    p_map = 0.62
    expected_series = p_map**2 * (3 - 2 * p_map)
    actual_series = _series_probability_binomial(p_map, 3)
    assert actual_series == pytest.approx(expected_series)
    p_a = blend_market_probability(
        actual_series, 0.50, alpha=1.0, temperature=1.0,
        blending_mode="logit_shrinkage",
    )
    p_b = 1.0 - p_a
    assert p_a == pytest.approx(expected_series)
    assert p_a + p_b == pytest.approx(1.0)


def test_alpha_zero_one_and_point_35_mean_sports_weight_not_market_weight():
    p_sports, q_market = 0.72, 0.43

    def independent(alpha):
        z = alpha * math.log(p_sports / (1 - p_sports)) + (1 - alpha) * math.log(q_market / (1 - q_market))
        return 1 / (1 + math.exp(-z))

    common = {"temperature": 1.0, "blending_mode": "logit_shrinkage"}
    assert blend_market_probability(
        p_sports, q_market, alpha=0.0, **common
    ) == pytest.approx(q_market)
    assert blend_market_probability(
        p_sports, q_market, alpha=1.0, **common
    ) == pytest.approx(p_sports)
    p_at_35 = blend_market_probability(
        p_sports, q_market, alpha=0.35, **common
    )
    assert p_at_35 == pytest.approx(independent(0.35))
    assert p_at_35 != pytest.approx(independent(0.65))


def test_taxed_break_even_probability_matches_net_ev():
    odds = 2.0
    break_even = 1.0 / (odds * (1.0 - TAX))
    assert expected_value(break_even, odds, TAX) == pytest.approx(0.0, abs=1e-12)


def test_dro_is_an_eligibility_hurdle_but_kelly_uses_nominal_score():
    probability, odds, epsilon = 0.36, 3.5, 0.025
    eligible, reason, diag = is_bet_eligible(
        prob_model=probability,
        odds=odds,
        tax_rate=TAX,
        min_ev_net=0.05,
        dynamic_hurdle=False,
        wasserstein_epsilon=epsilon,
        max_ev_net=None,
        max_ev_longshot=None,
        max_logit_disagreement=None,
        max_negative_clv_drift=None,
    )
    assert eligible and reason == "eligible"
    assert diag["ev_net"] == pytest.approx(0.1088)
    assert diag["ev_wasserstein"] == pytest.approx(0.0318)
    nominal_stake = fractional_kelly_stake(10_000, probability, odds, tax_rate=TAX)
    dro_score_stake = fractional_kelly_stake(
        10_000, max(0.0, probability - epsilon), odds, tax_rate=TAX,
    )
    assert nominal_stake > dro_score_stake > 0.0
    assert fractional_kelly_stake(0.0, probability, odds, tax_rate=TAX) == 0.0


def test_quote_consensus_excludes_focal_book_and_uses_two_other_books():
    quotes = (
        PrematchQuote(1, 10, 1, "focal", NOW, "A", "B", 1.50, 3.00, None, None),
        PrematchQuote(2, 10, 2, "peer-1", NOW, "A", "B", 2.50, 1.60, None, None),
        PrematchQuote(3, 10, 3, "peer-2", NOW, "A", "B", 3.50, 1.30, None, None),
    )
    consensus = quotes_consensus(quotes, exclude_bookmaker_id=1)
    peer_1 = (1.0 / 2.50) / ((1.0 / 2.50) + (1.0 / 1.60))
    peer_2 = (1.0 / 3.50) / ((1.0 / 3.50) + (1.0 / 1.30))
    assert consensus.bookmaker_ids == (2, 3)
    assert consensus.excluded_bookmaker_id == 1
    assert consensus.prob_a == pytest.approx((peer_1 + peer_2) / 2.0)

def test_ui_latest_invalid_quote_state_does_not_revive_older_valid_quote():
    older_valid = {
        "bookmaker_id": 7,
        "odds_snapshot_id": 1,
        "scraped_at": (NOW - timedelta(minutes=1)).isoformat(),
        "raw_team_a": "Alpha",
        "raw_team_b": "Beta",
        "odds_a": 2.5,
        "odds_b": 1.6,
        "league": "LCK",
        "start_time_normalized": (NOW + timedelta(hours=1)).isoformat(),
    }
    latest_invalid = {
        **older_valid,
        "odds_snapshot_id": 2,
        "scraped_at": NOW.isoformat(),
        "odds_a": None,
    }
    selected = _latest_display_states([older_valid, latest_invalid], NOW)
    assert [row["odds_snapshot_id"] for row in selected] == [2]
    assert selected[0]["odds_a"] is None


def test_latest_invalid_prediction_is_not_replaced_by_older_valid_prediction():
    earlier = {
        "id": 1,
        "canonical_match_id": 10,
        "model_name": "model",
        "model_version": "v1",
        "prediction_status": "active",
        "data_cutoff_at": NOW.isoformat(),
        "predicted_at": NOW.isoformat(),
    }
    latest_invalid = {
        **earlier,
        "id": 2,
        "prediction_status": "inactive",
        "predicted_at": (NOW + timedelta(minutes=1)).isoformat(),
    }
    # latest_predictions is audited by the application helper via the same
    # public actionable contract: once latest is selected, it fails closed.
    from betting_app.services.prematch_quote_service import latest_predictions

    selected = latest_predictions([earlier, latest_invalid])
    assert [row["id"] for row in selected] == [2]
    assert prediction_time_is_actionable(
        selected[0], decision_at=NOW + timedelta(minutes=2),
        start_at=NOW + timedelta(hours=1), model_name="model", model_version="v1",
    ) == (False, "prediction_inactive")


def test_unknown_winner_remains_reserved_and_settled_roi_is_undefined():
    decision_at = NOW
    candidate = {
        "opportunity_id": "pending",
        "canonical_match_id": 101,
        "decision_at": decision_at,
        "settle_at": decision_at + timedelta(hours=1),
        "side": "a",
        "bookmaker": "Book",
        "odds": 2.4,
        "prob_for_ev": 0.70,
        "ev_net": 0.70 * 2.4 * (1.0 - TAX) - 1.0,
        "y": None,
    }
    opportunity = {
        "opportunity_id": "pending",
        "canonical_match_id": 101,
        "decision_at": decision_at,
        "settle_at": candidate["settle_at"],
        "y": None,
    }
    result = simulate_operational_portfolio(
        pd.DataFrame([candidate]), pd.DataFrame([opportunity]),
        initial_bankroll=1_000.0, tax_rate=TAX,
    )
    assert result["summary"]["bets_count"] == 1
    assert result["summary"]["pending_bets_count"] == 1
    assert result["summary"]["settled_turnover"] == 0
    assert result["summary"]["roi"] is None
    assert result["summary"]["final_reserved"] == result["ledger"].loc[0, "stake"]
    assert pd.isna(result["ledger"].loc[0, "realized_pnl"])


def test_empty_candidate_and_opportunity_frames_return_zero_bets():
    result = simulate_operational_portfolio(
        pd.DataFrame(), pd.DataFrame(), initial_bankroll=1_000.0, tax_rate=TAX
    )
    assert result["summary"]["candidate_count"] == 0
    assert result["summary"]["bets_count"] == 0
    assert result["summary"]["roi"] is None

def test_same_clock_settlement_is_rejected_as_not_predecision_evidence():
    # An outcome that becomes available at the decision instant is not known
    # before the bet; accepting it and leaving it pending corrupts settled ROI.
    candidate = {
        "opportunity_id": "same-clock",
        "canonical_match_id": 102,
        "decision_at": NOW,
        "settle_at": NOW,
        "side": "a",
        "bookmaker": "Book",
        "odds": 2.4,
        "prob_for_ev": 0.70,
        "ev_net": 0.70 * 2.4 * (1.0 - TAX) - 1.0,
        "y": 1,
    }
    opportunity = {
        "opportunity_id": "same-clock",
        "canonical_match_id": 102,
        "decision_at": NOW,
        "settle_at": NOW,
        "y": 1,
    }
    result = simulate_operational_portfolio(
        pd.DataFrame([candidate]), pd.DataFrame([opportunity]),
        initial_bankroll=1_000.0, tax_rate=TAX,
    )
    assert result["ledger"].empty
    assert result["summary"]["bets_count"] == 0


def test_prior_settlement_precedes_equal_clock_later_fixture_decision():
    prior = {
        "opportunity_id": "prior",
        "canonical_match_id": 103,
        "decision_at": NOW - timedelta(hours=1),
        "settle_at": NOW,
        "side": "a",
        "bookmaker": "Book",
        "odds": 2.4,
        "prob_for_ev": 0.70,
        "ev_net": 0.70 * 2.4 * (1.0 - TAX) - 1.0,
        "y": 1,
    }
    next_bet = {
        "opportunity_id": "next",
        "canonical_match_id": 104,
        "decision_at": NOW,
        "settle_at": NOW + timedelta(hours=1),
        "side": "a",
        "bookmaker": "Book",
        "odds": 2.4,
        "prob_for_ev": 0.70,
        "ev_net": 0.70 * 2.4 * (1.0 - TAX) - 1.0,
        "y": None,
    }
    opportunities = pd.DataFrame([
        {"opportunity_id": "prior", "y": 1},
        {"opportunity_id": "next", "y": None},
    ])
    result = simulate_operational_portfolio(
        pd.DataFrame([prior, next_bet]), opportunities,
        initial_bankroll=1_000.0, tax_rate=TAX,
    )
    at_same_clock = result["capital_path"].loc[
        result["capital_path"]["timestamp"].eq(NOW), "event"
    ].tolist()
    assert at_same_clock == ["settlement", "decision"]
    assert result["ledger"]["opportunity_id"].tolist() == ["prior", "next"]


def test_ui_does_not_resurface_older_prediction_when_newest_clock_is_unknown():
    active_hybrid = get_active_hybrid()
    older_known = {
        "id": 1,
        "canonical_match_id": 105,
        "model_name": active_hybrid.base_model.name,
        "model_version": active_hybrid.base_model.version,
        "prediction_status": "active",
        "data_cutoff_at": (NOW - timedelta(minutes=2)).isoformat(),
        "predicted_at": (NOW - timedelta(minutes=1)).isoformat(),
        "prob_a": 0.70,
        "prob_b": 0.30,
        "diagnostics_json": None,
    }
    latest_unknown_clock = {
        **older_known,
        "id": 2,
        "predicted_at": None,
    }
    context = PrematchMarketContext(
        canonical_match_id=105,
        team_a_name="A",
        team_b_name="B",
        league=None,
        status=None,
        start_at=NOW + timedelta(hours=1),
        decision_at=NOW,
        quotes=(),
        prob_a=0.55,
        rejections=(),
    )
    served = _serving_predictions(
        None, [older_known, latest_unknown_clock], context
    )
    assert served == {}


def test_audit_current_mechanics_reports_evidence_and_numeric_proofs():
    from src.analysis.c0_edge_mechanics_audit import audit_current_mechanics

    repo_root = Path(__file__).resolve().parents[2]
    protocol_path = (
        repo_root
        / "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = audit_current_mechanics(protocol)
    expected_surfaces = {
        "probability_series",
        "orientation",
        "alpha_blend",
        "ev_tax",
        "qualification_dro",
        "kelly_wallet",
        "reservations_settlement",
        "roi",
        "consensus_prediction_state",
        "clv",
        "historical_audit",
    }
    assert report["run_id"] == protocol["run_id"]
    assert report["protocol_binding"]["timing_root"] == protocol["inputs"]["timing_root"]
    assert {item["surface"] for item in report["findings"]} == expected_surfaces
    allowed = {"verified-current", "historical-fixed", "defect", "unavailable"}
    assert {item["classification"] for item in report["findings"]} <= allowed
    assert all(item["evidence_paths"] and item["proof_cases"] for item in report["findings"])
    prediction_state = next(
        item for item in report["findings"]
        if item["surface"] == "consensus_prediction_state"
    )
    assert prediction_state["safe_repair"]
    settlement_state = next(
        item for item in report["findings"]
        if item["surface"] == "reservations_settlement"
    )
    assert "settle_at <= decision_at" in settlement_state["safe_repair"]
    assert report["source_pins"]["pre_repair_unified_evaluation_engine"]["sha256"] == (
        "1d1c905939c6cb77ea806a9d66ed28016008932bad231f409690c03ea5fd88dc"
    )
    proofs = {
        proof["id"]: proof
        for finding in report["findings"]
        for proof in finding["proof_cases"]
    }
    assert proofs["series_probability_complement"]["actual"]["sum"] == pytest.approx(1.0)
    assert proofs["serving_orientation"]["actual"]["probabilities"].keys() == {"a", "b"}
    assert proofs["taxed_break_even"]["actual"]["net_ev"] == pytest.approx(0.0, abs=1e-12)
    assert proofs["unknown_outcome_roi"]["actual"]["roi"] is None
    serialization = proofs["unknown_outcome_roi"]["actual"]["serialization"]
    assert serialization["winner"] is None
    assert serialization["nonfinite_metric"] is None
    wallet_proof = proofs["wallet_sizing_scenarios"]["actual"]
    assert wallet_proof["zero_wallet_stake"] == 0.0
    assert wallet_proof["wallet_amount_is_historical_input"] is False
    assert proofs["same_clock_own_settlement"]["actual"]["accepted"] is False
    latest_quote = proofs["latest_invalid_quote_does_not_revive_older"]["actual"]
    assert latest_quote["selected_ids"] == [2]
    assert latest_quote["selected_odds_a"] is None
    alternatives = proofs["same_fixture_quotes_are_alternatives_in_portfolio_replay"]["actual"]
    assert alternatives["candidate_rows"] == 2
    assert alternatives["ledger_bets"] == 1
    assert alternatives["selected_odds"] == pytest.approx(2.4)
    assert proofs["closing_line"]["actual"]["true_clv"] is None
    assert proofs["series_probability_complement"]["passed"] is True
    assert proofs["serving_orientation"]["passed"] is True
    assert proofs["alpha_endpoints_and_35_semantics"]["passed"] is True
    assert proofs["dro_eligibility_vs_nominal_sizing"]["passed"] is True
    assert proofs["same_clock_own_settlement"]["passed"] is True
    assert proofs["prior_settlement_before_equal_clock_decision"]["passed"] is True
    assert proofs["same_fixture_quotes_are_alternatives_in_portfolio_replay"]["passed"] is True
    assert proofs["settled_roi_after_tax"]["passed"] is True
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    decoded = json.loads(encoded)
    encoded_after_null_roundtrip = json.dumps(
        decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert hashlib.sha256(encoded.encode("utf-8")).digest() == hashlib.sha256(
        encoded_after_null_roundtrip.encode("utf-8")
    ).digest()

