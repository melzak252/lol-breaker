from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.c0_region_finance import run_financial_comparison


def test_finance_refuses_partial_receipt_without_overwriting_it(tmp_path):
    receipt = tmp_path / "financial_receipt.json"
    receipt.write_text('{"status":"PARTIAL"}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_financial_comparison({}, pd.DataFrame(), output_dir=tmp_path)

    assert receipt.read_text(encoding="utf-8") == '{"status":"PARTIAL"}'


from src.analysis.c0_edge_policy_audit import BASE_POLICY
from src.analysis.c0_region_finance import replay_candidate_events


PORTFOLIO = {
    "initial_bankroll": 10_000,
    "kelly_fraction": 0.25,
    "max_stake": 150,
    "min_stake": 2,
    "max_portfolio_exposure": 0.25,
}

REPLAY_POLICY = {
    **BASE_POLICY,
    "max_ev_net": None,
    "max_ev_longshot": None,
    "max_tier1_market_gap": None,
    "max_bo1_market_gap": None,
    "max_logit_disagreement": None,
}


def _panels(*, p=0.78, outcome=np.nan):
    event = pd.DataFrame(
        [
            {
                "opportunity_id": "opp-1",
                "canonical_match_id": 41,
                "quote_id": "q-1",
                "bookmaker_id": "book-1",
                "bookmaker": "book-1",
                "decision_at": "2026-06-01T12:00:00Z",
                "start_at": "2026-06-02T12:00:00Z",
                "settle_at": "2026-06-03T00:00:00Z",
                "result_recorded_at": pd.NaT,
                "label_available_at": pd.NaT,
                "y": outcome,
                "side": "a",
                "odds": 2.0,
                "market_prob_a": 0.64,
                "p_base": 0.55,
                "prob_model": p,
                "ev_net": p * 2.0 * (1.0 - 0.12) - 1.0,
                "prob_for_ev": p,
                "best_of": 3.0,
                "competition_tier": "Tier-1 International",
                "partition": "ASSESS",
                "cash_partition": "ASSESS",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "canonical_match_id": 41,
                "requested_utc_day": "2026-06-01",
                "decision_at": "2026-06-01T12:00:00Z",
                "forecast_at": "2026-06-01T11:00:00Z",
                "usable_for_same_day_projection": True,
                "strict_prior_feature_day_proof": True,
                "daily_state_cutoff_day": "2026-06-01",
                "feature_source_max_day": "2026-05-31",
                "model_query_matched": True,
                "p_candidate": 0.78,
            }
        ]
    )
    return event, predictions


def test_candidate_replay_orients_same_clock_canonical_a_probability_to_b_side():
    events, predictions = _panels()
    events.loc[0, "side"] = "b"

    replay = replay_candidate_events(
        events,
        predictions,
        probability_column="p_candidate",
        timing="quote_triggered_first_eligible",
        policy=REPLAY_POLICY,
        portfolio=PORTFOLIO,
    )

    assert replay["panel"].loc[0, "p_base"] == 0.78
    assert np.isclose(replay["panel"].loc[0, "prob_base"], 0.22)
    assert replay["selected"].empty


def test_candidate_replay_rejects_future_day_or_unavailable_model_state():
    events, predictions = _panels()
    predictions.loc[0, "requested_utc_day"] = "2026-06-02"
    predictions.loc[0, "forecast_at"] = "2026-06-02T00:00:00Z"

    future = replay_candidate_events(
        events,
        predictions,
        probability_column="p_candidate",
        timing="quote_triggered_first_eligible",
        policy=REPLAY_POLICY,
        portfolio=PORTFOLIO,
    )
    assert future["panel"]["p_base"].isna().all()
    assert future["selected"].empty

    predictions.loc[0, "requested_utc_day"] = "2026-06-01"
    predictions.loc[0, "forecast_at"] = "2026-06-01T11:00:00Z"
    predictions.loc[0, "usable_for_same_day_projection"] = False
    unavailable = replay_candidate_events(
        events,
        predictions,
        probability_column="p_candidate",
        timing="quote_triggered_first_eligible",
        policy=REPLAY_POLICY,
        portfolio=PORTFOLIO,
    )
    assert unavailable["panel"]["p_base"].isna().all()
    assert unavailable["selected"].empty


def test_candidate_replay_recomputes_ev_gate_from_candidate_probability():
    events, predictions = _panels(p=0.99)
    predictions.loc[0, "p_candidate"] = 0.52
    policy = {**BASE_POLICY, "min_ev_net": 0.05, "dynamic_hurdle": False}

    replay = replay_candidate_events(
        events,
        predictions,
        probability_column="p_candidate",
        timing="quote_triggered_first_eligible",
        policy=policy,
        portfolio=PORTFOLIO,
        sports_alpha=1.0,
    )

    assert replay["panel"]["prob_model"].iloc[0] == 0.52
    assert not replay["gate_masks"]["minimum_ev"][0]
    assert replay["selected"].empty


def test_candidate_replay_selects_earliest_qualifying_clock_not_future_best_ev():
    events, predictions = _panels(p=0.78)
    later = events.iloc[0].copy()
    later["opportunity_id"] = "opp-later"
    later["quote_id"] = "q-later"
    later["decision_at"] = "2026-06-01T14:00:00Z"
    later["odds"] = 3.0
    events = pd.concat([events, pd.DataFrame([later])], ignore_index=True)

    replay = replay_candidate_events(
        events,
        predictions,
        probability_column="p_candidate",
        timing="quote_triggered_first_eligible",
        policy=REPLAY_POLICY,
        portfolio=PORTFOLIO,
    )

    assert replay["selected"]["opportunity_id"].tolist() == ["opp-1"]


def test_candidate_replay_keeps_unknown_result_pending_and_cash_reserved():
    events, predictions = _panels()
    events.loc[0, "market_prob_a"] = 0.71
    events.loc[0, "odds"] = 1.85
    events["y"] = np.nan

    replay = replay_candidate_events(
        events,
        predictions,
        probability_column="p_candidate",
        timing="quote_triggered_first_eligible",
        policy=BASE_POLICY,
        portfolio=PORTFOLIO,
    )

    wallet = replay["wallet"]
    assert wallet["summary"]["bets_count"] == 1
    assert wallet["summary"]["pending_bets_count"] == 1
    assert wallet["summary"]["settled_bets_count"] == 0
    assert wallet["summary"]["settled_turnover"] == 0
    assert wallet["summary"]["final_reserved"] == wallet["ledger"]["stake"].iloc[0]
    assert (
        wallet["summary"]["final_cash"] + wallet["summary"]["final_reserved"] == 10_000
    )


def test_candidate_replay_preserves_frozen_baseline_policy_values():
    events, predictions = _panels()
    before = dict(BASE_POLICY)

    replay_candidate_events(
        events,
        predictions,
        probability_column="p_candidate",
        timing="quote_triggered_first_eligible",
        policy=BASE_POLICY,
        portfolio=PORTFOLIO,
    )

    assert BASE_POLICY == before
    assert BASE_POLICY["tax_rate"] == 0.12
    assert BASE_POLICY["wasserstein_epsilon"] == 0.025
