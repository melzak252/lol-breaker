#!/usr/bin/env python3
"""Executable, side-effect-free audit of current C0 bet mechanics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROTOCOL = ROOT / "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol.json"
TAX = 0.12
NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _json_value(value: Any) -> Any:
    """Convert proof values to strict JSON values; nonfinite numbers become null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return _finite(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _case(case_id: str, expected: Any, actual: Any, passed: bool) -> dict[str, Any]:
    return {
        "id": case_id,
        "expected": _json_value(expected),
        "actual": _json_value(actual),
        "passed": bool(passed),
    }


def _finding(
    surface: str,
    classification: str,
    claim: str,
    evidence_paths: list[str],
    proof_cases: list[dict[str, Any]],
    limitations: list[str] | None = None,
    safe_repair: str | None = None,
) -> dict[str, Any]:
    return {
        "surface": surface,
        "classification": classification,
        "claim": claim,
        "evidence_paths": evidence_paths,
        "proof_cases": proof_cases,
        "limitations": limitations or [],
        "safe_repair": safe_repair,
    }


def _independent_logit_blend(sports: float, market: float, alpha: float) -> float:
    z = alpha * math.log(sports / (1.0 - sports))
    z += (1.0 - alpha) * math.log(market / (1.0 - market))
    return 1.0 / (1.0 + math.exp(-z))


def _sample_candidate(
    opportunity_id: str,
    match_id: int,
    decision_at: datetime,
    settle_at: datetime,
    outcome: int | None,
    tax_rate: float,
) -> dict[str, Any]:
    return {
        "opportunity_id": opportunity_id,
        "canonical_match_id": match_id,
        "decision_at": decision_at,
        "settle_at": settle_at,
        "side": "a",
        "bookmaker": "audit-book",
        "odds": 2.4,
        "prob_for_ev": 0.70,
        "ev_net": 0.70 * 2.4 * (1.0 - tax_rate) - 1.0,
        "y": outcome,
    }


def audit_current_mechanics(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe findings and executable numeric proofs for current mechanics.

    The function uses only in-process arithmetic and pure application helpers. It
    never opens an application database, mutates app state, loads/trains model
    weights, or treats an unavailable closing quote as CLV.
    """
    import pandas as pd

    from betting_app.api.routers.matches import _latest_display_states, _serving_predictions
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
        latest_predictions,
        quotes_consensus,
    )
    from src.analysis.unified_evaluation_engine import simulate_operational_portfolio

    portfolio = protocol.get("portfolio", {})
    inputs = protocol.get("inputs", {})
    tax_rate = _finite(portfolio.get("tax_rate")) or TAX
    fraction = _finite(portfolio.get("kelly_fraction")) or 0.25
    initial_bankroll = _finite(portfolio.get("initial_bankroll")) or 10_000.0
    max_stake = _finite(portfolio.get("max_stake")) or 150.0
    min_stake = _finite(portfolio.get("min_stake")) or 2.0
    exposure = _finite(portfolio.get("max_portfolio_exposure")) or 0.25
    active_hybrid = get_active_hybrid()

    p_map = 0.62
    independent_series = p_map**2 * (3.0 - 2.0 * p_map)
    runtime_series = _series_probability_binomial(p_map, 3)
    second_bo_counterfactual = _series_probability_binomial(runtime_series, 3)
    p_series_a = blend_market_probability(
        runtime_series, 0.5, alpha=1.0, temperature=1.0,
        blending_mode="logit_shrinkage",
    )
    p_series_b = 1.0 - p_series_a
    series_case = _case(
        "series_probability_complement",
        {
            "series": independent_series,
            "a_plus_b": 1.0,
            "second_bo_changes_probability": True,
        },
        {
            "series": runtime_series,
            "double_bo_counterfactual": second_bo_counterfactual,
            "a": p_series_a,
            "b": p_series_b,
            "sum": p_series_a + p_series_b,
        },
        math.isclose(runtime_series, independent_series, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(p_series_a, runtime_series, rel_tol=0.0, abs_tol=1e-12)
        and not math.isclose(second_bo_counterfactual, runtime_series, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(p_series_a + p_series_b, 1.0, rel_tol=0.0, abs_tol=1e-12),
    )

    serving_row = {
        "id": 1,
        "canonical_match_id": 7101,
        "model_name": active_hybrid.base_model.name,
        "model_version": active_hybrid.base_model.version,
        "prediction_status": "active",
        "data_cutoff_at": (NOW - timedelta(minutes=2)).isoformat(),
        "predicted_at": (NOW - timedelta(minutes=1)).isoformat(),
        "prob_a": 0.72,
        "prob_b": 0.28,
        "diagnostics_json": None,
    }
    market_context = PrematchMarketContext(
        canonical_match_id=7101,
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
    serving_result = _serving_predictions(None, [serving_row], market_context)
    served_a = _finite(serving_result.get("hybrid_prob_a"))
    served_b = _finite(serving_result.get("hybrid_prob_b"))
    single_blend_expected = blend_market_probability(
        serving_row["prob_a"], market_context.prob_a,
        alpha=active_hybrid.alpha,
        temperature=active_hybrid.temperature,
        blending_mode=active_hybrid.blending_mode,
    )
    double_blend_counterfactual = blend_market_probability(
        single_blend_expected, market_context.prob_a,
        alpha=active_hybrid.alpha,
        temperature=active_hybrid.temperature,
        blending_mode=active_hybrid.blending_mode,
    )
    orientation_case = _case(
        "serving_orientation",
        {
            "probabilities": {"a": "A-side", "b": "B-side"},
            "sum": 1.0,
            "one_blend_probability": single_blend_expected,
        },
        {
            "probabilities": {"a": served_a, "b": served_b},
            "sum": (served_a + served_b) if served_a is not None and served_b is not None else None,
            "double_blend_counterfactual": double_blend_counterfactual,
        },
        served_a is not None and served_b is not None
        and math.isclose(served_a, single_blend_expected, rel_tol=0.0, abs_tol=1e-12)
        and not math.isclose(
            double_blend_counterfactual, single_blend_expected, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(served_a + served_b, 1.0, rel_tol=0.0, abs_tol=1e-12),
    )

    sports, market = 0.72, 0.43
    alpha_values = {}
    for alpha in (0.0, 0.35, 0.65, 1.0):
        alpha_values[str(alpha)] = blend_market_probability(
            sports, market, alpha=alpha, temperature=1.0,
            blending_mode="logit_shrinkage",
        )
    alpha_case = _case(
        "alpha_endpoints_and_35_semantics",
        {
            "alpha_0": market,
            "alpha_1": sports,
            "alpha_035": _independent_logit_blend(sports, market, 0.35),
            "alpha_065_differs": True,
        },
        {
            "sports_weight": 0.35,
            "market_weight": 0.65,
            "active_registry_alpha": _finite(active_hybrid.alpha),
            "probabilities": alpha_values,
            "protocol_semantics": protocol.get("alpha_semantics"),
        },
        math.isclose(alpha_values["0.0"], market, abs_tol=1e-12)
        and math.isclose(alpha_values["1.0"], sports, abs_tol=1e-12)
        and math.isclose(
            alpha_values["0.35"], _independent_logit_blend(sports, market, 0.35),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and not math.isclose(alpha_values["0.35"], alpha_values["0.65"], abs_tol=1e-12),
    )

    odds = 2.0
    break_even = 1.0 / (odds * (1.0 - tax_rate))
    net_ev_at_break_even = expected_value(break_even, odds, tax_rate)
    ev_case = _case(
        "taxed_break_even",
        {"probability": break_even, "net_ev": 0.0},
        {"probability": break_even, "net_ev": net_ev_at_break_even},
        math.isclose(net_ev_at_break_even, 0.0, rel_tol=0.0, abs_tol=1e-12),
    )

    dro_probability, dro_odds, epsilon = 0.36, 3.5, 0.025
    dro_eligible, dro_reason, dro_diag = is_bet_eligible(
        prob_model=dro_probability,
        odds=dro_odds,
        tax_rate=tax_rate,
        min_ev_net=0.05,
        dynamic_hurdle=False,
        wasserstein_epsilon=epsilon,
        max_ev_net=None,
        max_ev_longshot=None,
        max_logit_disagreement=None,
        max_negative_clv_drift=None,
    )
    nominal_stake = fractional_kelly_stake(
        initial_bankroll, dro_probability, dro_odds, fraction=fraction, tax_rate=tax_rate,
        min_stake=min_stake, max_stake=max_stake,
        max_portfolio_exposure=exposure,
    )
    stress_stake = fractional_kelly_stake(
        initial_bankroll, max(0.0, dro_probability - epsilon), dro_odds,
        fraction=fraction, tax_rate=tax_rate, min_stake=min_stake,
        max_stake=max_stake, max_portfolio_exposure=exposure,
    )
    qualification_case = _case(
        "dro_eligibility_vs_nominal_sizing",
        {"eligible": True, "stress_ev_positive": True, "nominal_stake_gt_stress_stake": True},
        {
            "eligible": bool(dro_eligible),
            "reason": dro_reason,
            "nominal_ev": _finite(dro_diag.get("ev_net")),
            "stress_ev": _finite(dro_diag.get("ev_wasserstein")),
            "nominal_stake": nominal_stake,
            "stress_probability_stake": stress_stake,
        },
        bool(dro_eligible)
        and _finite(dro_diag.get("ev_wasserstein")) is not None
        and dro_diag["ev_wasserstein"] > 0.0
        and nominal_stake > stress_stake,
    )

    wallet_scenario = 1_000.0
    wallet_stake = fractional_kelly_stake(
        wallet_scenario, 0.70, 2.4, fraction=fraction, tax_rate=tax_rate,
        min_stake=min_stake, max_stake=max_stake,
        max_portfolio_exposure=exposure,
    )
    comparison_stake_100 = fractional_kelly_stake(
        100.0, 0.70, 2.4, fraction=fraction, tax_rate=tax_rate,
        min_stake=min_stake, max_stake=max_stake,
        max_portfolio_exposure=exposure,
    )
    zero_wallet_stake = fractional_kelly_stake(
        0.0, 0.70, 2.4, fraction=fraction, tax_rate=tax_rate,
        min_stake=min_stake, max_stake=max_stake,
        max_portfolio_exposure=exposure,
    )
    wallet_case = _case(
        "wallet_sizing_scenarios",
        {"wallet_scenario": 1_000.0, "zero_wallet_stake": 0.0, "wallet_stake_gt_100_scenario": True},
        {
            "scenario_wallet_pln": wallet_scenario,
            "wallet_scenario_stake": wallet_stake,
            "explicit_100pln_scenario_stake": comparison_stake_100,
            "zero_wallet_stake": zero_wallet_stake,
            "wallet_amount_is_historical_input": False,
        },
        zero_wallet_stake == 0.0 and wallet_stake > comparison_stake_100,
    )

    pending_candidate = _sample_candidate(
        "pending", 7201, NOW, NOW + timedelta(hours=1), None, tax_rate
    )
    pending = simulate_operational_portfolio(
        pd.DataFrame([pending_candidate]),
        pd.DataFrame([{"opportunity_id": "pending", "y": None}]),
        initial_bankroll=initial_bankroll,
        tax_rate=tax_rate,
        fraction=fraction,
        max_stake=max_stake,
        min_stake=min_stake,
        max_portfolio_exposure=exposure,
    )
    pending_summary = pending["summary"]
    pending_roi = _finite(pending_summary.get("settled_roi"))
    pending_serialization = _json_value({
        "winner": None,
        "nonfinite_metric": float("nan"),
        "pending_funds": pending_summary["final_reserved"],
    })
    pending_json_roundtrip = json.loads(json.dumps(pending_serialization, allow_nan=False))
    pending_case = _case(
        "unknown_outcome_roi",
        {
            "pending_bets": 1,
            "roi": None,
            "winner": None,
            "nonfinite_metric": None,
            "reserved_positive": True,
        },
        {
            "pending_bets": int(pending_summary["pending_bets_count"]),
            "roi": pending_roi,
            "final_reserved": _finite(pending_summary["final_reserved"]),
            "settled_turnover": _finite(pending_summary["settled_turnover"]),
            "serialization": pending_json_roundtrip,
        },
        int(pending_summary["pending_bets_count"]) == 1
        and pending_roi is None
        and _finite(pending_summary["final_reserved"]) is not None
        and pending_summary["final_reserved"] > 0
        and pending_json_roundtrip["winner"] is None
        and pending_json_roundtrip["nonfinite_metric"] is None,
    )

    same_clock_candidate = _sample_candidate("same-clock", 7202, NOW, NOW, 1, tax_rate)
    same_clock = simulate_operational_portfolio(
        pd.DataFrame([same_clock_candidate]),
        pd.DataFrame([{"opportunity_id": "same-clock", "y": 1}]),
        initial_bankroll=initial_bankroll,
        tax_rate=tax_rate,
        fraction=fraction,
        max_stake=max_stake,
        min_stake=min_stake,
        max_portfolio_exposure=exposure,
    )
    same_clock_reasons = same_clock["execution_rejections"].get(
        "reason", pd.Series(dtype=str)
    ).astype(str).tolist()
    same_clock_accepted = not same_clock["ledger"].empty
    same_clock_case = _case(
        "same_clock_own_settlement",
        {"accepted": False, "rejection_reason": "settlement_before_decision"},
        {
            "accepted": same_clock_accepted,
            "rejection_reasons": same_clock_reasons,
            "pre_repair_observation": {
                "accepted": True,
                "pending_bets": 1,
                "source_archive": "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/pre_repair_source/unified_evaluation_engine.py",
                "source_sha256": "1d1c905939c6cb77ea806a9d66ed28016008932bad231f409690c03ea5fd88dc",
            },
        },
        not same_clock_accepted and "settlement_before_decision" in same_clock_reasons,
    )

    prior = _sample_candidate(
        "prior", 7203, NOW - timedelta(hours=1), NOW, 1, tax_rate
    )
    next_bet = _sample_candidate(
        "next", 7204, NOW, NOW + timedelta(hours=1), None, tax_rate
    )
    prior_and_next = simulate_operational_portfolio(
        pd.DataFrame([prior, next_bet]),
        pd.DataFrame([{"opportunity_id": "prior", "y": 1}, {"opportunity_id": "next", "y": None}]),
        initial_bankroll=initial_bankroll,
        tax_rate=tax_rate,
        fraction=fraction,
        max_stake=max_stake,
        min_stake=min_stake,
        max_portfolio_exposure=exposure,
    )
    same_clock_events = prior_and_next["capital_path"].loc[
        prior_and_next["capital_path"]["timestamp"].eq(NOW), "event"
    ].astype(str).tolist()
    order_case = _case(
        "prior_settlement_before_equal_clock_decision",
        ["settlement", "decision"],
        same_clock_events,
        same_clock_events == ["settlement", "decision"],
    )

    lower_offer = _sample_candidate(
        "lower-offer", 7206, NOW, NOW + timedelta(hours=1), None, tax_rate
    )
    lower_offer["odds"] = 2.3
    higher_offer = {**lower_offer, "opportunity_id": "higher-offer", "odds": 2.4}
    alternatives = simulate_operational_portfolio(
        pd.DataFrame([lower_offer, higher_offer]),
        pd.DataFrame([
            {"opportunity_id": "lower-offer", "y": None},
            {"opportunity_id": "higher-offer", "y": None},
        ]),
        initial_bankroll=initial_bankroll,
        tax_rate=tax_rate,
        fraction=fraction,
        max_stake=max_stake,
        min_stake=min_stake,
        max_portfolio_exposure=exposure,
    )
    alternative_case = _case(
        "same_fixture_quotes_are_alternatives_in_portfolio_replay",
        {"candidate_rows": 2, "ledger_bets": 1, "selected_odds": 2.4},
        {
            "candidate_rows": int(alternatives["summary"]["candidate_count"]),
            "ledger_bets": int(alternatives["summary"]["bets_count"]),
            "selected_odds": (
                _finite(alternatives["ledger"].iloc[0]["odds"])
                if not alternatives["ledger"].empty else None
            ),
        },
        int(alternatives["summary"]["candidate_count"]) == 2
        and int(alternatives["summary"]["bets_count"]) == 1
        and not alternatives["ledger"].empty
        and math.isclose(
            _finite(alternatives["ledger"].iloc[0]["odds"]) or 0.0,
            2.4, rel_tol=0.0, abs_tol=1e-12,
        ),
    )

    settled_candidate = _sample_candidate(
        "settled-win", 7205, NOW, NOW + timedelta(hours=1), 1, tax_rate
    )
    settled = simulate_operational_portfolio(
        pd.DataFrame([settled_candidate]),
        pd.DataFrame([{"opportunity_id": "settled-win", "y": 1}]),
        initial_bankroll=initial_bankroll,
        tax_rate=tax_rate,
        fraction=fraction,
        max_stake=max_stake,
        min_stake=min_stake,
        max_portfolio_exposure=exposure,
    )
    settled_summary = settled["summary"]
    expected_win_roi = 2.4 * (1.0 - tax_rate) - 1.0
    actual_win_roi = _finite(settled_summary.get("settled_roi"))
    roi_case = _case(
        "settled_roi_after_tax",
        {"roi": expected_win_roi, "settled_bets": 1},
        {
            "roi": actual_win_roi,
            "settled_bets": int(settled_summary["settled_bets_count"]),
            "profit": _finite(settled_summary["profit"]),
            "settled_turnover": _finite(settled_summary["settled_turnover"]),
        },
        actual_win_roi is not None
        and math.isclose(actual_win_roi, expected_win_roi, rel_tol=0.0, abs_tol=1e-12)
        and int(settled_summary["settled_bets_count"]) == 1,
    )

    quotes = (
        PrematchQuote(1, 7301, 1, "focal", NOW, "A", "B", 1.50, 3.00, None, None),
        PrematchQuote(2, 7301, 2, "peer-1", NOW, "A", "B", 2.50, 1.60, None, None),
        PrematchQuote(3, 7301, 3, "peer-2", NOW, "A", "B", 3.50, 1.30, None, None),
    )
    consensus = quotes_consensus(quotes, exclude_bookmaker_id=1)
    peer_probabilities = [
        (1.0 / 2.50) / ((1.0 / 2.50) + (1.0 / 1.60)),
        (1.0 / 3.50) / ((1.0 / 3.50) + (1.0 / 1.30)),
    ]
    expected_loo = sum(peer_probabilities) / len(peer_probabilities)
    consensus_case = _case(
        "focal_excluded_consensus",
        {"excluded_bookmaker_id": 1, "peer_bookmaker_ids": [2, 3], "prob_a": expected_loo},
        {
            "excluded_bookmaker_id": consensus.excluded_bookmaker_id,
            "peer_bookmaker_ids": list(consensus.bookmaker_ids),
            "prob_a": _finite(consensus.prob_a),
        },
        consensus.excluded_bookmaker_id == 1
        and consensus.bookmaker_ids == (2, 3)
        and math.isclose(consensus.prob_a, expected_loo, rel_tol=0.0, abs_tol=1e-12),
    )

    older_valid_quote = {
        "bookmaker_id": 7,
        "odds_snapshot_id": 1,
        "scraped_at": (NOW - timedelta(minutes=1)).isoformat(),
        "raw_team_a": "A",
        "raw_team_b": "B",
        "odds_a": 2.5,
        "odds_b": 1.6,
        "league": "LCK",
        "start_time_normalized": (NOW + timedelta(hours=1)).isoformat(),
    }
    latest_invalid_quote = {
        **older_valid_quote,
        "odds_snapshot_id": 2,
        "scraped_at": NOW.isoformat(),
        "odds_a": None,
    }
    latest_quote_rows = _latest_display_states(
        [older_valid_quote, latest_invalid_quote], NOW
    )
    quote_invalid_case = _case(
        "latest_invalid_quote_does_not_revive_older",
        {"selected_ids": [2], "selected_odds_a": None},
        {
            "selected_ids": [int(row["odds_snapshot_id"]) for row in latest_quote_rows],
            "selected_odds_a": (
                _finite(latest_quote_rows[0].get("odds_a"))
                if latest_quote_rows else None
            ),
        },
        len(latest_quote_rows) == 1
        and int(latest_quote_rows[0]["odds_snapshot_id"]) == 2
        and latest_quote_rows[0].get("odds_a") is None,
    )

    older_prediction = {
        "id": 1,
        "canonical_match_id": 7401,
        "model_name": active_hybrid.base_model.name,
        "model_version": active_hybrid.base_model.version,
        "prediction_status": "active",
        "data_cutoff_at": (NOW - timedelta(minutes=2)).isoformat(),
        "predicted_at": (NOW - timedelta(minutes=1)).isoformat(),
        "prob_a": 0.70,
        "prob_b": 0.30,
        "diagnostics_json": None,
    }
    newest_unknown = {**older_prediction, "id": 2, "predicted_at": None}
    unknown_context = PrematchMarketContext(
        canonical_match_id=7401,
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
    latest_rows = latest_predictions([older_prediction, newest_unknown])
    unknown_served = _serving_predictions(
        None, [older_prediction, newest_unknown], unknown_context
    )
    stale_fallback_served = _finite(unknown_served.get("model_prob_a")) is not None
    unknown_clock_case = _case(
        "unknown_prediction_clock_fails_closed",
        {"older_prediction_resurfaced": False},
        {
            "selected_ids": [int(row["id"]) for row in latest_rows],
            "served_prediction_ids": [
                int(row["id"]) for row in latest_rows
                if row.get("predicted_at") is not None
            ] if stale_fallback_served else [],
            "stale_fallback_served": stale_fallback_served,
        },
        not stale_fallback_served,
    )

    same_cutoff_open = _finite(consensus.prob_a)
    same_cutoff_close = same_cutoff_open
    same_cutoff_drift = (
        same_cutoff_close - same_cutoff_open
        if same_cutoff_open is not None and same_cutoff_close is not None else None
    )
    clv_case = _case(
        "closing_line",
        {"true_clv": None, "same_cutoff_drift": 0.0},
        {
            "true_clv": None,
            "same_cutoff_open": same_cutoff_open,
            "same_cutoff_close": same_cutoff_close,
            "same_cutoff_drift": same_cutoff_drift,
        },
        same_cutoff_drift == 0.0,
    )

    loo_history_case = _case(
        "historical_quote_admission_now_uses_focal_exclusion",
        {"focal_excluded": True, "independent_peer_count": 2},
        {
            "focal_excluded": consensus.excluded_bookmaker_id == 1,
            "independent_peer_count": len(consensus.bookmaker_ids),
            "peer_probability": _finite(consensus.prob_a),
        },
        consensus.excluded_bookmaker_id == 1 and len(consensus.bookmaker_ids) == 2,
    )

    findings = [
        _finding(
            "probability_series", "verified-current",
            "BO conversion is applied at the model-family boundary, then blending consumes the series probability once; returned A/B probabilities complement.",
            [
                "betting_app/core/models/engine.py:19-30,107-120,141-155,184-195",
                "betting_app/core/models/native_c0.py:854-895",
            ],
            [series_case],
            ["The checker does not load or retrain native C0 weights; the proof exercises the application transform and independent BO3 arithmetic."],
        ),
        _finding(
            "orientation", "verified-current",
            "The serving recommendation consumer emits side A and side B from one current hybrid transform with p(A)+p(B)=1.",
            [
                "betting_app/api/routers/matches.py:176-228,1923-1974",
                "betting_app/api/schemas.py:429-445",
            ],
            [orientation_case],
            ["The synthetic in-process fixture uses the actual serving helper and no database session."],
        ),
        _finding(
            "alpha_blend", "verified-current",
            "Logit alpha=0 selects market, alpha=1 selects sports, and alpha=.35 means 35% sports plus 65% market under the frozen protocol; registry configs retain their declared blend mode and temperature.",
            [
                "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol.json:23-24",
                "betting_app/core/models/engine.py:33-90",
                "betting_app/core/models/registry.py:451-467",
            ],
            [alpha_case],
            ["Protocol .35 logit/T=1 grid semantics are not a claim that the separate thesis .35/T=.80 linear model is the active hybrid."],
        ),
        _finding(
            "ev_tax", "verified-current",
            "Expected value and break-even use odds multiplied by one minus tax; the 12% stake tax reduces the stake used for payout multiplication, rather than taxing net profit.",
            [
                "betting_app/services/market_service.py:13-31",
                "betting_app/services/bet_qualification_service.py:353-355",
                "betting_app/core/staking.py:22-72",
            ],
            [ev_case],
        ),
        _finding(
            "qualification_dro", "verified-current",
            "The nominal net-EV hurdle is checked first; the Wasserstein-stressed probability is a positive-EV eligibility test, while Kelly receives the nominal probability.",
            [
                "betting_app/services/bet_qualification_service.py:353-404",
                "betting_app/services/upcoming_inference_service.py:3015-3031,3127-3137",
            ],
            [qualification_case],
        ),
        _finding(
            "kelly_wallet", "verified-current",
            "The canonical upcoming pipeline and standalone EV-signal CLI resolve an explicit bankroll or the actual PLN wallet and otherwise pass zero; no live wallet balance is read by this audit.",
            [
                "betting_app/scripts/run_upcoming_prediction_pipeline.py:245-262",
                "betting_app/services/bankroll_service.py:11-48",
                "betting_app/scripts/generate_model_ev_signals.py:21-33",
                "betting_app/services/alert_service.py:289-393,408-478",
                "betting_app/api/routers/alerts.py:130-135",
                "betting_app/scheduler/tasks/predict.py:36-40",
            ],
            [wallet_case],
            ["No live wallet balance is read or inferred; 1000 PLN and zero PLN are explicit arithmetic scenarios, not historical wallet evidence."],
        ),
        _finding(
            "reservations_settlement", "historical-fixed" if not same_clock_accepted else "defect",
            "Unknown outcomes remain reserved at cost. An own settlement at or before its decision is now rejected; a prior bet still settles before a different same-clock decision.",
            [
                "src/analysis/unified_evaluation_engine.py:271-286,320-403",
                "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/pre_repair_source/unified_evaluation_engine.py",
            ],
            [pending_case, same_clock_case, order_case, alternative_case],
            [
                "The archived pre-repair engine is pinned at SHA-256 1d1c905939c6cb77ea806a9d66ed28016008932bad231f409690c03ea5fd88dc. Parent-confirmed pre-repair RED was accepted with one pending bet; no source repair is claimed to explain observed finance losses.",
                "The portfolio replay selects a best quote among same-fixture/side alternatives; this is not a global exposure-cap claim and does not establish that separately dispatched UI alerts are additive executions.",
            ],
            safe_repair="Reject own settle_at <= decision_at as settlement_before_decision; retain the existing settle_through(decision_at) before each later decision.",
        ),
        _finding(
            "roi", "verified-current",
            "Settled ROI is realized after-tax profit divided by settled turnover; pending bets do not enter the denominator.",
            [
                "src/analysis/unified_evaluation_engine.py:320-342,400-423",
                "betting_app/services/wallet_service.py:167-342",
            ],
            [roi_case],
        ),
        _finding(
            "consensus_prediction_state", "defect" if stale_fallback_served else "historical-fixed",
            "Focal-book quotes are excluded from LOO consensus; latest-invalid quote selection does not revive the older valid price. The latest prediction selector retains an unorderable newest state, so the serving consumer fails closed rather than reviving an older forecast.",
            [
                "betting_app/services/prematch_quote_service.py:119-131,237-326",
                "betting_app/models/prediction.py:184-188",
                "betting_app/api/routers/matches.py:176-201,248-313,450-459,2150-2154",
                "betting_app/services/upcoming_inference_service.py:2792-2801",
            ],
            [consensus_case, quote_invalid_case, unknown_clock_case],
            ["The canonical_predictions.predicted_at column is nullable. Reachability/count in the frozen October financial export is unavailable because it contains no predictions table."],
            safe_repair="Fail closed when the newest model/fixture prediction state has no usable predicted_at; do not rank that state below a timed row and resurface the older forecast.",
        ),
        _finding(
            "clv", "unavailable",
            "Current market edge is not closing-line value; the live signal path reuses its same-cutoff consensus as both open and close, so drift is zero by construction. Authentic CLV requires an observed same-book entry and pre-start close.",
            [
                "betting_app/services/upcoming_inference_service.py:2949-2953,3015-3031",
                "betting_app/api/routers/matches.py:1923-1974",
                "betting_app/api/routers/timing.py:1523-1806",
                "src/analysis/c0_policy_runner.py:170-174",
                "betting_app/README.md:28-30",
                "client/src/pages/MatchDetail.tsx:1325-1332",
            ],
            [clv_case],
            ["Frozen protocol reports zero close coverage and true_clv=null; no missing close is imputed as 0.5."],
        ),
        _finding(
            "historical_audit", "historical-fixed",
            "Prior audit findings on validity-before-ranking, timestamp joins admitting other market/live rows, unrelated orientation, and devig fallback are historical for the current shared quote path. Focal-excluded latest-state consensus is reproducible; the repaired unknown-prediction-clock fallback and its retained pre-repair reproduction are reported separately.",
            [
                "data/08_reporting/c0_betting_research/c0_betting_research_20261006_01/code_audit/summary.json",
                "data/08_reporting/c0_betting_research/c0_betting_research_20261006_01/code_audit/action_design.json",
                "betting_app/services/prematch_quote_service.py:81-160,237-305",
                "betting_app/services/upcoming_inference_service.py:2287-2299",
                "betting_app/api/routers/matches.py:248-313",
            ],
            [loo_history_case],
            ["Historical source defects are not re-attributed to current outcomes without exact-ID impact evidence."],
        ),
    ]

    counts = Counter(finding["classification"] for finding in findings)
    result = {
        "schema": "c0_edge_mechanics_audit_v1",
        "run_id": str(protocol.get("run_id", "unknown")),
        "protocol_binding": {
            "schema": protocol.get("schema"),
            "timing_root": inputs.get("timing_root"),
            "output_root": protocol.get("output_root"),
            "alpha_semantics": protocol.get("alpha_semantics"),
            "portfolio": portfolio,
        },
        "scope": "HISTORICAL_DEVELOPMENT_ONLY",
        "source_pins": {
            "pre_repair_unified_evaluation_engine": {
                "path": "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/pre_repair_source/unified_evaluation_engine.py",
                "sha256": "1d1c905939c6cb77ea806a9d66ed28016008932bad231f409690c03ea5fd88dc",
            },
        },
        "classification_counts": dict(sorted(counts.items())),
        "findings": findings,
        "limitations": [
            "This audit exercises current pure mechanics and synthetic arithmetic only; it does not query, mutate, or infer from a live application database.",
            "No native C0 model weights are loaded or retrained, no current wallet is observed, and no source is activated or deployed.",
            "Frozen timing exports contain no prediction table, so the count of null-clock prediction states and their impact on the historical finance cohort are unavailable.",
        ],
    }
    return _json_value(result)


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = audit_current_mechanics(protocol)
    output = args.output or Path(protocol["output_root"]) / "mechanics" / "current_mechanics_audit.json"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_compact_json(report) + "\n", encoding="utf-8")
    digest = hashlib.sha256(_compact_json(report).encode("utf-8")).hexdigest()
    print(_compact_json({
        "phase": "mechanics_audit_complete",
        "findings": len(report["findings"]),
        "classification_counts": report["classification_counts"],
        "report_sha256": digest,
        "output": str(output),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
