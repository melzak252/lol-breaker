from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.analysis.c0_policy_evaluation import evaluate_policy_candidates
from betting_app.services.bet_qualification_service import SIGNAL_GATE_KWARGS

UTC = timezone.utc
DECISION = datetime(2026, 6, 1, 12, tzinfo=UTC)


def frames(*, books=3, p_base=0.60, p_exp039=0.60, odds_a=None, include_forecast=True):
    opportunity = pd.DataFrame(
        [
            {
                "opportunity_id": "opp-1",
                "canonical_match_id": 101,
                "decision_at": DECISION,
                "start_at": DECISION + timedelta(hours=1),
                "settle_at": DECISION + timedelta(hours=3),
                "league": "LCK",
                "best_of": 3,
                "y": 1,
                "absence_flags": [],
            }
        ]
    )
    quote_rows = []
    for index in range(books):
        quote_rows.append(
            {
                "quote_id": f"quote-{index}",
                "canonical_match_id": 101,
                "bookmaker_id": index + 1,
                "bookmaker": f"book-{index + 1}",
                "quoted_at": DECISION - timedelta(minutes=2),
                "odds_a": (odds_a if index == 0 and odds_a is not None else 2.0),
                "odds_b": 2.0,
                "is_live": False,
                "identity_verified": True,
                "raw_a_is_canonical_a": True,
            }
        )
    quotes = pd.DataFrame(quote_rows)
    forecasts = pd.DataFrame(
        [
            {
                "canonical_match_id": 101,
                "p_base": p_base,
                "p_exp039": p_exp039,
                "rating_disagreement": 0.0,
                "forecast_available": include_forecast,
                "forecast_at": DECISION - timedelta(minutes=5),
                "exp039_forecast_at": DECISION - timedelta(minutes=2)
                if p_exp039 is not None
                else None,
                "identity_verified": True,
            }
        ]
    )
    if not include_forecast:
        forecasts = forecasts.iloc[0:0]
    return opportunity, quotes, forecasts


def policy():
    return {
        **dict(SIGNAL_GATE_KWARGS),
        "alpha": 0.5,
        "tax_rate": 0.12,
        "wasserstein_epsilon": 0.025,
        "max_logit_disagreement": 0.85,
    }


def evaluate(opportunities, quotes, forecasts, *, arm="complete"):
    return evaluate_policy_candidates(
        opportunities, quotes, forecasts, policy=policy(), arm=arm, ttl_seconds=300
    )


def test_two_total_books_cannot_satisfy_two_other_book_consensus():
    result = evaluate(*frames(books=2))

    assert result["candidates"].empty
    assert result["selected"].empty
    assert len(result["opportunities"]) == 1
    assert result["opportunities"].candidate_absence_reason.tolist() == [
        "insufficient_other_books"
    ]


def test_focal_price_does_not_change_its_consensus_or_blended_probability():
    opportunities, quotes, forecasts = frames()
    baseline = evaluate(opportunities, quotes, forecasts)["candidates"]
    quotes.loc[quotes.bookmaker_id.eq(1), "odds_a"] = 3.0
    changed = evaluate(opportunities, quotes, forecasts)["candidates"]

    before = baseline.loc[baseline.bookmaker_id.eq(1)].iloc[0]
    after = changed.loc[changed.bookmaker_id.eq(1)].iloc[0]
    assert before.market_prob_a == pytest.approx(0.5)
    assert after.market_prob_a == pytest.approx(before.market_prob_a)
    assert after.prob_a == pytest.approx(before.prob_a)


def test_full_ev_cap_rejects_candidate_above_frozen_signal_maximum():
    result = evaluate(*frames(p_base=0.95, p_exp039=0.95, odds_a=2.4))
    focal = result["candidates"].loc[lambda rows: rows.bookmaker_id.eq(1)].iloc[0]

    assert not bool(focal.eligible)
    assert focal.diagnostics["ev_net"] > focal.diagnostics["max_ev_net"]


def test_dro_arm_rejects_fragile_three_to_one_bet_but_no_dro_accepts():
    opportunities, quotes, forecasts = frames(
        p_base=0.3076923076923,
        p_exp039=0.3076923076923,
        odds_a=3.0,
    )
    opportunities["league"] = None
    dro = evaluate(opportunities, quotes, forecasts, arm="complete")["candidates"]
    no_dro = evaluate(opportunities, quotes, forecasts, arm="no_dro")["candidates"]
    dro_focal = dro.loc[dro.bookmaker_id.eq(1)].iloc[0]
    no_dro_focal = no_dro.loc[no_dro.bookmaker_id.eq(1)].iloc[0]

    assert not bool(dro_focal.eligible)
    assert dro_focal.diagnostics["ev_wasserstein"] <= 0
    assert (
        no_dro_focal.diagnostics["ev_net"]
        > no_dro_focal.diagnostics["effective_min_ev"]
    )
    assert bool(no_dro_focal.eligible)


def test_guard_b_compares_market_with_base_not_hybrid_blend():
    opportunities, quotes, forecasts = frames(p_base=0.75, p_exp039=0.75, odds_a=2.0)
    opportunities["league"] = None  # Avoid the tier-1 guard; exercise Guard B itself.
    result = evaluate(opportunities, quotes, forecasts)["candidates"]
    focal = result.loc[result.bookmaker_id.eq(1)].iloc[0]

    assert focal.market_prob_a == pytest.approx(0.5)
    assert focal.prob_a > 0.5
    assert not bool(focal.eligible)
    assert focal.diagnostics["logit_disagreement"] > policy()["max_logit_disagreement"]


def test_fresh_exp039_veto_is_one_factor_and_requires_its_own_timestamp():
    opportunities, quotes, forecasts = frames(p_base=0.60, p_exp039=0.75, odds_a=2.4)
    opportunities["league"] = None
    complete = evaluate(opportunities, quotes, forecasts)["candidates"]
    no_exp039 = evaluate(opportunities, quotes, forecasts, arm="no_exp039")[
        "candidates"
    ]
    complete_focal = complete.loc[complete.bookmaker_id.eq(1)].iloc[0]
    no_exp_focal = no_exp039.loc[no_exp039.bookmaker_id.eq(1)].iloc[0]

    assert complete_focal.exp039_status == "evaluated"
    assert not bool(complete_focal.eligible)
    assert bool(no_exp_focal.eligible)


def test_missing_exp039_is_not_evaluated_not_fabricated_agreement():
    opportunities, quotes, forecasts = frames(p_exp039=None)
    forecasts = forecasts.drop(columns="p_exp039")
    result = evaluate(opportunities, quotes, forecasts)["candidates"]

    assert set(result.exp039_status) == {"not_evaluated"}


def test_ineligible_highest_price_does_not_block_lower_price_eligible_book():
    opportunities, quotes, forecasts = frames(p_base=0.65, p_exp039=0.65)
    quotes.loc[quotes.bookmaker_id.eq(1), ["odds_a", "odds_b"]] = 4.5
    quotes.loc[quotes.bookmaker_id.ne(1), ["odds_a", "odds_b"]] = 2.2

    result = evaluate(opportunities, quotes, forecasts)

    assert (
        not result["candidates"]
        .loc[lambda rows: rows.bookmaker_id.eq(1), "eligible"]
        .any()
    )
    assert result["selected"].bookmaker_id.tolist() == [2]


def test_outcomes_and_future_close_quotes_cannot_change_selection():
    opportunities, quotes, forecasts = frames(p_base=0.70, p_exp039=0.70)
    opportunities["league"] = None
    quotes[["odds_a", "odds_b"]] = 2.2
    baseline = evaluate(opportunities, quotes, forecasts)["selected"]
    opportunities["y"] = 0
    opportunities["settle_at"] = DECISION + timedelta(days=30)
    future_close = quotes.iloc[[0]].copy()

    future_close["quoted_at"] = DECISION + timedelta(days=2)
    future_close["odds_a"] = 1.01
    changed = evaluate(
        opportunities, pd.concat([quotes, future_close], ignore_index=True), forecasts
    )["selected"]

    assert changed.opportunity_id.tolist() == baseline.opportunity_id.tolist()
    assert changed.bookmaker_id.tolist() == baseline.bookmaker_id.tolist()


def test_opportunity_without_forecast_is_retained_but_not_selected():
    opportunities, quotes, forecasts = frames(include_forecast=False)
    result = evaluate(opportunities, quotes, forecasts)

    assert result["opportunities"].opportunity_id.tolist() == ["opp-1"]
    assert result["selected"].empty


def test_financial_rows_keep_full_precision_ev_and_canonical_probability_orientation():
    opportunities, quotes, forecasts = frames(p_base=0.651234567, p_exp039=0.651234567)
    quotes[["odds_a", "odds_b"]] = 2.2
    candidates = evaluate(opportunities, quotes, forecasts)["candidates"]
    a = candidates.loc[(candidates.bookmaker_id == 1) & (candidates.side == "a")].iloc[
        0
    ]
    b = candidates.loc[(candidates.bookmaker_id == 1) & (candidates.side == "b")].iloc[
        0
    ]
    assert a.model_ev == pytest.approx(
        a.prob_for_ev * a.odds * 0.88 - 1, rel=0, abs=1e-14
    )
    assert b.prob_a == pytest.approx(a.prob_a, rel=0, abs=1e-14)
