"""Frozen C0 policy candidate construction and shared-gate evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from betting_app.core.models.engine import blend_market_probability
from betting_app.services.bet_qualification_service import (
    SIGNAL_GATE_KWARGS,
    is_bet_eligible,
    qualification_tier,
)


ARMS = (
    "complete",
    "no_dro",
    "no_guard_b",
    "no_exp039",
    "min_ev_only_safety_ablated",
    "pure_c0_same_gates",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool:
    return bool(value) if not pd.isna(value) else False


def evaluate_policy_candidates(
    opportunities: pd.DataFrame,
    quotes: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    policy: Mapping[str, Any],
    arm: str = "complete",
    ttl_seconds: int = 300,
) -> dict[str, pd.DataFrame]:
    """Evaluate scenario-qualified quotes under one frozen policy arm.

    The focal book is excluded from the multiplicative no-vig consensus. At least
    two distinct external books are required; opportunity-level absence reasons
    preserve rows without a consensus-qualified quote.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown policy arm: {arm}")
    if ttl_seconds < 0:
        raise ValueError("ttl_seconds must be nonnegative")

    # The producer resolves quote state per opportunity/book. Re-run admission here
    # so arbitrary canonical-looking frames cannot bypass freshness or de-duplication.
    from src.analysis.c0_policy_data import admit_policy_quotes

    opp_frame = opportunities.copy().reset_index(drop=True)
    if opp_frame.empty:
        return {
            "opportunities": opp_frame,
            "candidates": pd.DataFrame(),
            "selected": pd.DataFrame(),
        }
    if "opportunity_id" not in opp_frame:
        raise ValueError("opportunities require opportunity_id")
    quote_rows = admit_policy_quotes(opportunities, quotes, ttl_seconds=ttl_seconds)
    forecast_rows = forecasts.copy()
    forecast_by_match: dict[Any, pd.Series] = {}
    for _, forecast in forecast_rows.iterrows():
        key = forecast.get("canonical_match_id")
        if key not in forecast_by_match and _truth(
            forecast.get("identity_verified", False)
        ):
            forecast_by_match[key] = forecast

    opp_by_match: dict[Any, list[pd.Series]] = {}
    for _, opportunity in opp_frame.iterrows():
        opp_by_match.setdefault(opportunity.get("canonical_match_id"), []).append(
            opportunity
        )

    def quote_odds(row: pd.Series, side: str) -> float | None:
        return _number(row.get(f"canonical_odds_{side}", row.get(f"odds_{side}")))

    def decision_time(row: pd.Series):
        return pd.to_datetime(row.get("decision_at"), utc=True, errors="coerce")

    by_opp: dict[Any, list[pd.Series]] = {}
    for _, quote in quote_rows.iterrows():
        key = quote.get("canonical_match_id")
        for opportunity in opp_by_match.get(key, ()):
            # If the producer did not retain an opportunity link, require the quote
            # to be available by this fixture's decision clock and within TTL.
            decision = decision_time(opportunity)
            quoted = pd.to_datetime(quote.get("quoted_at"), utc=True, errors="coerce")
            if pd.isna(decision) or pd.isna(quoted) or quoted > decision:
                continue
            if (decision - quoted).total_seconds() > ttl_seconds:
                continue
            if quote.get("opportunity_id") not in (
                None,
                opportunity.get("opportunity_id"),
            ):
                continue
            by_opp.setdefault(opportunity.get("opportunity_id"), []).append(quote)
    absence_reasons: dict[Any, str | None] = {}
    distinct_books: dict[Any, int] = {}
    forecast_available: dict[Any, bool] = {}
    for _, opportunity in opp_frame.iterrows():
        opportunity_id = opportunity.get("opportunity_id")
        rows = by_opp.get(opportunity_id, [])
        count = len(
            {
                row.get("bookmaker_id")
                for row in rows
                if not pd.isna(row.get("bookmaker_id"))
            }
        )
        distinct_books[opportunity_id] = count
        forecast = forecast_by_match.get(opportunity.get("canonical_match_id"))
        forecast_clock = (
            pd.to_datetime(forecast.get("forecast_at"), utc=True, errors="coerce")
            if forecast is not None
            else pd.NaT
        )
        opportunity_clock = decision_time(opportunity)
        available = (
            forecast is not None
            and _truth(forecast.get("forecast_available", False))
            and not pd.isna(forecast_clock)
            and not pd.isna(opportunity_clock)
            and forecast_clock <= opportunity_clock
        )
        forecast_available[opportunity_id] = available
        absence_reasons[opportunity_id] = (
            "no_admissible_quotes"
            if count == 0
            else "insufficient_other_books"
            if count < 3
            else "forecast_unavailable"
            if not available
            else None
        )
    opp_frame["admitted_quote_count"] = opp_frame["opportunity_id"].map(
        lambda opportunity_id: len(by_opp.get(opportunity_id, []))
    )
    opp_frame["distinct_bookmaker_count"] = opp_frame["opportunity_id"].map(
        distinct_books
    )
    opp_frame["candidate_absence_reason"] = opp_frame["opportunity_id"].map(
        absence_reasons
    )

    candidate_rows: list[dict[str, Any]] = []
    for _, opportunity in opp_frame.iterrows():
        oid = opportunity.get("opportunity_id")
        match_id = opportunity.get("canonical_match_id")
        forecast = forecast_by_match.get(match_id)
        p_base = (
            _number(forecast.get("p_base", forecast.get("p_C0")))
            if forecast is not None
            else None
        )
        forecast_clock = (
            pd.to_datetime(forecast.get("forecast_at"), utc=True, errors="coerce")
            if forecast is not None
            else pd.NaT
        )
        opportunity_clock = decision_time(opportunity)
        forecast_ok = (
            forecast is not None
            and _truth(forecast.get("forecast_available", False))
            and not pd.isna(forecast_clock)
            and not pd.isna(opportunity_clock)
            and forecast_clock <= opportunity_clock
        )
        if not forecast_ok:
            p_base = None
        quotes_for_opportunity = by_opp.get(oid, [])

        # Build one de-vigged probability per underlying external bookmaker.
        for focal in quotes_for_opportunity:
            fa, fb = quote_odds(focal, "a"), quote_odds(focal, "b")
            focal_id = focal.get("bookmaker_id")
            others: list[float] = []
            for other in quotes_for_opportunity:
                if other.get("bookmaker_id") == focal_id:
                    continue
                other_oriented = (
                    True
                    if {"canonical_odds_a", "canonical_odds_b"}.issubset(other.index)
                    else _truth(other.get("raw_a_is_canonical_a", True))
                )
                if (
                    not _truth(other.get("identity_verified", True))
                    or not other_oriented
                ):
                    continue
                if _truth(other.get("is_live", False)):
                    continue
                oa, ob = quote_odds(other, "a"), quote_odds(other, "b")
                if oa is None or ob is None or oa <= 1 or ob <= 1:
                    continue
                others.append((1.0 / oa) / ((1.0 / oa) + (1.0 / ob)))
            consensus_a = sum(others) / len(others) if len(others) >= 2 else None
            if len(others) < 2:
                continue
            verified = _truth(focal.get("identity_verified", True))
            orientation = (
                True
                if {"canonical_odds_a", "canonical_odds_b"}.issubset(focal.index)
                else _truth(focal.get("raw_a_is_canonical_a", True))
            )
            live = _truth(focal.get("is_live", False))
            for side in ("a", "b"):
                odds = fa if side == "a" else fb
                market_p = (
                    consensus_a
                    if side == "a"
                    else (1.0 - consensus_a if consensus_a is not None else None)
                )
                reason = "eligible"
                eligible = False
                p_side = base_side = blended_side = None
                diag: dict[str, Any] = {}
                exp039_veto = False
                exp039_status = "not_evaluated"
                if odds is None or odds <= 1:
                    reason = "invalid_odds"
                elif not verified or not orientation:
                    reason = "identity_unverified"
                elif live:
                    reason = "live_quote"
                elif len(others) < 2:
                    reason = "insufficient_other_books"
                elif p_base is None or not (0 < p_base < 1):
                    reason = "forecast_unavailable"
                else:
                    base_side = p_base if side == "a" else 1.0 - p_base
                    q_side = market_p
                    alpha = float(policy.get("alpha", 0.5))
                    temperature = float(policy.get("temperature", 1.0))
                    blend_mode = str(policy.get("blending_mode", "logit_shrinkage"))
                    blended_side = (
                        base_side
                        if arm == "pure_c0_same_gates"
                        else blend_market_probability(
                            base_side,
                            q_side,
                            alpha=alpha,
                            temperature=temperature,
                            blending_mode=blend_mode,
                        )
                    )
                    p_side = base_side if arm == "pure_c0_same_gates" else blended_side
                    # EXP039 is a distinct fresh-check veto. Absence is never agreement.
                    if (
                        arm != "no_exp039"
                        and forecast is not None
                        and "p_exp039" in forecast.index
                    ):
                        exp = _number(forecast.get("p_exp039"))
                        exp_clock = pd.to_datetime(
                            forecast.get("exp039_forecast_at"),
                            utc=True,
                            errors="coerce",
                        )
                        opportunity_clock = decision_time(opportunity)
                        fresh_exp = (
                            not pd.isna(exp_clock)
                            and not pd.isna(opportunity_clock)
                            and exp_clock <= opportunity_clock
                            and (opportunity_clock - exp_clock).total_seconds()
                            <= ttl_seconds
                        )
                        if fresh_exp and exp is not None and 0 < exp < 1:
                            exp039_status = "evaluated"
                            exp039_veto = abs(p_base - exp) > 0.10
                    kwargs = dict(SIGNAL_GATE_KWARGS)
                    kwargs.update(
                        {
                            key: value
                            for key, value in policy.items()
                            if key
                            in {
                                "tax_rate",
                                "min_ev_net",
                                "dynamic_hurdle",
                                "hurdle_odds_cutoff",
                                "hurdle_penalty_scale",
                                "max_rating_disagreement",
                                "max_tier1_market_gap",
                                "max_ev_net",
                                "tier1_only",
                                "max_ev_longshot",
                                "longshot_odds_threshold",
                                "max_bo1_market_gap",
                                "bo1_max_odds",
                                "max_negative_clv_drift",
                                "max_coinflip_market_gap",
                                "coinflip_min_odds",
                                "coinflip_max_odds",
                                "max_logit_disagreement",
                                "universal_rating_guard",
                                "max_odds",
                                "wasserstein_epsilon",
                            }
                        }
                    )
                    if arm == "no_dro":
                        kwargs["wasserstein_epsilon"] = None
                    if arm == "no_guard_b":
                        kwargs["max_logit_disagreement"] = None
                    if arm == "min_ev_only_safety_ablated":
                        kwargs.update(
                            {
                                "max_rating_disagreement": None,
                                "max_tier1_market_gap": None,
                                "max_ev_net": None,
                                "max_ev_longshot": None,
                                "max_bo1_market_gap": None,
                                "max_negative_clv_drift": None,
                                "max_coinflip_market_gap": None,
                                "max_logit_disagreement": None,
                                "universal_rating_guard": False,
                                "max_odds": None,
                                "wasserstein_epsilon": None,
                            }
                        )
                    eligible, reason, diag = is_bet_eligible(
                        prob_model=p_side,
                        prob_conservative=None,
                        odds=odds,
                        prob_market_novig=market_p,
                        rating_disagreement=_number(
                            forecast.get("rating_disagreement")
                        ),
                        competition_tier=qualification_tier(
                            opportunity.get("league"), opportunity.get("decision_at")
                        ),
                        best_of=(
                            int(opportunity["best_of"])
                            if pd.notna(opportunity.get("best_of"))
                            else None
                        ),
                        prob_market_close_novig=market_p,
                        prob_base=base_side,
                        base_prob_required=True,
                        uncertainty_required=False,
                        uncertainty_basis="mean",
                        **kwargs,
                    )
                    if exp039_veto and reason == "eligible":
                        reason = "exp039_divergence"
                        eligible = False
                    elif reason == "eligible":
                        eligible = True
                probability_for_ev = diag.get("prob_for_ev")
                full_precision_ev = (
                    probability_for_ev
                    * odds
                    * (1 - float(policy.get("tax_rate", 0.12)))
                    - 1
                    if probability_for_ev is not None
                    else None
                )
                candidate_rows.append(
                    {
                        **opportunity.to_dict(),
                        "quote_id": focal.get("quote_id"),
                        "bookmaker_id": focal_id,
                        "bookmaker": focal.get("bookmaker"),
                        "quoted_at": focal.get("quoted_at"),
                        "side": side,
                        "odds": odds,
                        "market_prob_a": consensus_a,
                        "market_prob": market_p,
                        "p_base": p_base,
                        "base_prob": base_side,
                        "prob_a": p_side
                        if side == "a" or p_side is None
                        else 1 - p_side,
                        "prob_side": p_side,
                        "prob_for_ev": diag.get("prob_for_ev"),
                        "ev_net": full_precision_ev,
                        "model_ev": full_precision_ev,
                        "diagnostics": diag,
                        "eligible": bool(eligible),
                        "reason": reason,
                        "exp039_status": exp039_status,
                        "uncertainty_basis": "mean",
                        "native_bound_status": "unavailable",
                    }
                )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        selected = candidates.copy()
    else:
        # Full precision eligibility precedes price ranking; never let a rejected
        # high-price book suppress a lower-price qualifying quote.
        passing = candidates.loc[candidates["eligible"]].copy()
        selected = (
            passing.sort_values(
                ["opportunity_id", "side", "odds", "bookmaker_id"],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .drop_duplicates(["opportunity_id", "side"], keep="first")
            .reset_index(drop=True)
        )
    return {
        "opportunities": opp_frame.reset_index(drop=True),
        "candidates": candidates.reset_index(drop=True),
        "selected": selected.reset_index(drop=True),
    }
