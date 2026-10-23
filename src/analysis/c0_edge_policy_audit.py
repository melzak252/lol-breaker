"""Frozen C0 edge-recovery policy search over the prepared causal cohort."""

from __future__ import annotations

import argparse
import hashlib
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping
import inspect

import numpy as np
import pandas as pd

from betting_app.services.bet_qualification_service import (
    DEFAULT_BASE_MIN_EV,
    DEFAULT_BO1_MAX_ODDS,
    DEFAULT_COINFLIP_MAX_ODDS,
    DEFAULT_COINFLIP_MIN_ODDS,
    DEFAULT_HURDLE_ODDS_CUTOFF,
    DEFAULT_HURDLE_PENALTY_SCALE,
    DEFAULT_LONGSHOT_ODDS_THRESHOLD,
    DEFAULT_MAX_BO1_MARKET_GAP,
    DEFAULT_MAX_EV_LONGSHOT,
    DEFAULT_MAX_LOGIT_DISAGREEMENT,
    DEFAULT_MAX_NEGATIVE_CLV_DRIFT,
    DEFAULT_MAX_RATING_DISAGREEMENT,
    DEFAULT_MAX_TIER1_MARKET_GAP,
    SIGNAL_GATE_KWARGS,
    qualification_tier,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_KEYS = (
    "tax_rate", "min_ev_net", "dynamic_hurdle", "hurdle_odds_cutoff",
    "hurdle_penalty_scale", "max_rating_disagreement", "max_tier1_market_gap",
    "max_ev_net", "tier1_only", "max_ev_longshot", "longshot_odds_threshold",
    "max_bo1_market_gap", "bo1_max_odds", "max_negative_clv_drift",
    "max_coinflip_market_gap", "coinflip_min_odds", "coinflip_max_odds",
    "max_logit_disagreement", "universal_rating_guard", "max_odds",
    "wasserstein_epsilon",
)
BASE_POLICY = {
    "tax_rate": 0.12,
    "min_ev_net": DEFAULT_BASE_MIN_EV,
    "dynamic_hurdle": True,
    "hurdle_odds_cutoff": DEFAULT_HURDLE_ODDS_CUTOFF,
    "hurdle_penalty_scale": DEFAULT_HURDLE_PENALTY_SCALE,
    "max_rating_disagreement": SIGNAL_GATE_KWARGS["max_rating_disagreement"],
    "max_tier1_market_gap": DEFAULT_MAX_TIER1_MARKET_GAP,
    "max_ev_net": SIGNAL_GATE_KWARGS["max_ev_net"],
    "tier1_only": False,
    "max_ev_longshot": DEFAULT_MAX_EV_LONGSHOT,
    "longshot_odds_threshold": DEFAULT_LONGSHOT_ODDS_THRESHOLD,
    "max_bo1_market_gap": DEFAULT_MAX_BO1_MARKET_GAP,
    "bo1_max_odds": None,
    "max_negative_clv_drift": DEFAULT_MAX_NEGATIVE_CLV_DRIFT,
    "max_coinflip_market_gap": None,
    "coinflip_min_odds": DEFAULT_COINFLIP_MIN_ODDS,
    "coinflip_max_odds": DEFAULT_COINFLIP_MAX_ODDS,
    "max_logit_disagreement": 0.85,
    "universal_rating_guard": SIGNAL_GATE_KWARGS["universal_rating_guard"],
    "max_odds": SIGNAL_GATE_KWARGS["max_odds"],
    "wasserstein_epsilon": 0.025,
}


def _digest(value: Any) -> str:
    from src.analysis.c0_edge_alpha_audit import canonical_json_sha256

    return canonical_json_sha256(value)


def _semantic_policy_values(config: Mapping[str, Any]) -> dict[str, Any]:
    values = {}
    for key in ("alpha", *POLICY_KEYS):
        value = config[key]
        if (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        ):
            value = float(value)
        values[key] = value
    return values


def build_policy_configs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the complete, deduplicated search frozen in the machine protocol."""
    grid = protocol["policy_grid"]
    core = grid["core"]
    alpha_grid = protocol["alpha_grid"]
    configs: list[dict[str, Any]] = []

    def add(family: str, alpha: float, overrides: Mapping[str, Any] | None = None) -> None:
        policy = dict(BASE_POLICY)
        policy.update({key: value for key, value in (overrides or {}).items() if key in POLICY_KEYS})
        config = {"family": family, "alpha": float(alpha), **policy}
        config["config_id"] = "edge-" + _digest(
            {"family": family, **_semantic_policy_values(config)}
        )[:16]
        configs.append(config)

    for alpha in core["alpha"]:
        for min_ev in core["min_ev_net"]:
            for epsilon in core["wasserstein_epsilon"]:
                add("core", alpha, {"min_ev_net": min_ev, "wasserstein_epsilon": epsilon})

    baseline_min_ev = float(grid["baseline_core"]["min_ev_net"])
    baseline_epsilon = float(grid["baseline_core"]["wasserstein_epsilon"])
    for alpha in alpha_grid:
        for gate, values in grid["one_factor_at_each_alpha"].items():
            for value in values:
                add(
                    "one_factor:" + gate,
                    alpha,
                    {"min_ev_net": baseline_min_ev, "wasserstein_epsilon": baseline_epsilon, gate: value},
                )
        for gate, values in grid.get("availability_sensitivities", {}).items():
            for value in values:
                add(
                    "availability:" + gate,
                    alpha,
                    {"min_ev_net": baseline_min_ev, "wasserstein_epsilon": baseline_epsilon, gate: value},
                )

    unique: dict[str, dict[str, Any]] = {}
    for config in configs:
        signature = _digest(_semantic_policy_values(config))
        unique.setdefault(signature, config)
    return list(unique.values())


def build_diagnostic_policy_configs(
    protocol: Mapping[str, Any], configs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build fixed, explicitly non-selection diagnostic controls from protocol v2."""
    baseline = protocol["policy_grid"]["baseline_core"]
    diagnostics: list[dict[str, Any]] = []
    for control in protocol.get("diagnostic_controls", []):
        if control.get("selection_eligible") is not False:
            raise ValueError("Diagnostic controls must be explicitly excluded from policy selection")
        overrides = control.get("overrides", {})
        unknown = set(overrides).difference(POLICY_KEYS)
        if unknown:
            raise ValueError(f"Diagnostic control contains unknown policy keys: {sorted(unknown)}")
        for alpha in control.get("alphas", []):
            base = next(
                config for config in configs
                if config["family"] == "core"
                and config["alpha"] == float(alpha)
                and config["min_ev_net"] == baseline["min_ev_net"]
                and config["wasserstein_epsilon"] == baseline["wasserstein_epsilon"]
            )
            diagnostic = dict(base)
            diagnostic.update(overrides)
            diagnostic["family"] = f"diagnostic:{control['name']}"
            diagnostic["control_name"] = control["name"]
            diagnostic["selection_eligible"] = False
            diagnostic["diagnostic_interpretation"] = control["interpretation"]
            diagnostic["config_id"] = "edge-" + _digest({
                "control_name": control["name"],
                **_semantic_policy_values(diagnostic),
            })[:16]
            diagnostics.append(diagnostic)
    return diagnostics


def _float_column(frame: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _optional(
    policy: Mapping[str, Any], key: str, default: float | None = None
) -> float | None:
    value = policy.get(key, default)
    return None if value is None or pd.isna(value) else float(value)


def vectorized_gate_passes(frame: pd.DataFrame, policy: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return each scalar qualification gate as a vectorized pass mask."""
    count = len(frame)
    p = _float_column(frame, "prob_model")
    base = _float_column(frame, "prob_base")
    odds = _float_column(frame, "odds")
    market = _float_column(frame, "prob_market_novig")
    rating = _float_column(frame, "rating_disagreement")
    close = _float_column(frame, "prob_market_close_novig")
    best_of = _float_column(frame, "best_of")
    finite = np.isfinite(p) & (p > 0.0) & (p < 1.0) & np.isfinite(odds) & (odds > 1.0)
    valid_market = np.isfinite(market) & (market > 0.0) & (market < 1.0)
    valid_base = np.isfinite(base) & (base > 0.0) & (base < 1.0)
    tax = float(policy.get("tax_rate", 0.12))
    ret = odds * (1.0 - tax)
    ev = p * ret - 1.0
    minimum = float(policy.get("min_ev_net", DEFAULT_BASE_MIN_EV))
    if bool(policy.get("dynamic_hurdle", True)):
        cutoff = float(policy.get("hurdle_odds_cutoff", DEFAULT_HURDLE_ODDS_CUTOFF))
        penalty = float(policy.get("hurdle_penalty_scale", DEFAULT_HURDLE_PENALTY_SCALE))
        minimum = minimum + np.maximum(0.0, (cutoff - odds) / cutoff) * penalty
    else:
        minimum = np.full(count, minimum, dtype=float)
    passes: dict[str, np.ndarray] = {
        "valid_probability_odds": finite,
        "minimum_ev": finite & (ev >= minimum),
    }
    epsilon = _optional(policy, "wasserstein_epsilon")
    passes["wasserstein_dro"] = (
        np.ones(count, dtype=bool)
        if epsilon is None or epsilon <= 0.0
        else finite & ((np.maximum(0.0, p - epsilon) * ret - 1.0) > 0.0)
    )
    max_ev = _optional(policy, "max_ev_net")
    passes["max_ev_net"] = np.ones(count, dtype=bool) if max_ev is None else ev <= max_ev
    max_odds = _optional(policy, "max_odds")
    passes["max_odds"] = np.ones(count, dtype=bool) if max_odds is None else odds <= max_odds
    longshot = float(policy.get("longshot_odds_threshold", DEFAULT_LONGSHOT_ODDS_THRESHOLD))
    max_longshot_ev = _optional(policy, "max_ev_longshot", DEFAULT_MAX_EV_LONGSHOT)
    passes["longshot_ev"] = (
        np.ones(count, dtype=bool)
        if max_longshot_ev is None
        else ~((odds > longshot) & (ev > max_longshot_ev))
    )
    bo1 = best_of == 1.0
    bo1_max = _optional(policy, "bo1_max_odds")
    passes["bo1_max_odds"] = np.ones(count, dtype=bool) if bo1_max is None else ~(bo1 & (odds > bo1_max))
    bo1_gap = _optional(policy, "max_bo1_market_gap", DEFAULT_MAX_BO1_MARKET_GAP)
    passes["bo1_market_gap"] = (
        np.ones(count, dtype=bool)
        if bo1_gap is None
        else ~(bo1 & valid_market & (np.abs(p - market) >= bo1_gap))
    )
    drift_limit = _optional(policy, "max_negative_clv_drift", DEFAULT_MAX_NEGATIVE_CLV_DRIFT)
    drift = close - market
    passes["negative_clv_drift"] = (
        np.ones(count, dtype=bool)
        if drift_limit is None
        else ~(np.isfinite(close) & valid_market & (drift <= drift_limit))
    )
    rating_limit = _optional(policy, "max_rating_disagreement", DEFAULT_MAX_RATING_DISAGREEMENT)
    rating_guard = bool(policy.get("universal_rating_guard", False)) | (
        odds <= float(policy.get("hurdle_odds_cutoff", DEFAULT_HURDLE_ODDS_CUTOFF))
    )
    passes["rating_disagreement"] = (
        np.ones(count, dtype=bool)
        if rating_limit is None
        else ~(rating_guard & np.isfinite(rating) & (rating > rating_limit))
    )
    if "competition_tier" in frame:
        tier1 = frame["competition_tier"].isin(("Tier-1 International", "Tier-1 Domestic")).to_numpy()
    elif "league" in frame:
        tier1 = np.asarray(
            [qualification_tier(value, at) in ("Tier-1 International", "Tier-1 Domestic")
             for value, at in zip(frame["league"], frame.get("decision_at", pd.Series([None] * count)))],
            dtype=bool,
        )
    else:
        tier1 = np.zeros(count, dtype=bool)
    tier_gap = _optional(policy, "max_tier1_market_gap", DEFAULT_MAX_TIER1_MARKET_GAP)
    passes["tier1_market_gap"] = (
        np.ones(count, dtype=bool)
        if tier_gap is None
        else ~(tier1 & valid_market & ((p - market) > tier_gap))
    )
    coin_gap = _optional(policy, "max_coinflip_market_gap")
    coin_band = (odds >= float(policy.get("coinflip_min_odds", DEFAULT_COINFLIP_MIN_ODDS))) & (
        odds <= float(policy.get("coinflip_max_odds", DEFAULT_COINFLIP_MAX_ODDS))
    )
    passes["coinflip_market_gap"] = (
        np.ones(count, dtype=bool)
        if coin_gap is None
        else ~(coin_band & valid_market & (np.abs(p - market) >= coin_gap))
    )
    logit_gap = _optional(policy, "max_logit_disagreement", DEFAULT_MAX_LOGIT_DISAGREEMENT)
    if logit_gap is None:
        passes["logit_disagreement"] = np.ones(count, dtype=bool)
    else:
        required_base = ~valid_base
        gap = np.full(count, np.inf, dtype=float)
        active = valid_base & valid_market
        gap[active] = np.abs(np.log(base[active] / (1.0 - base[active])) - np.log(market[active] / (1.0 - market[active])))
        gap[valid_base & ~valid_market] = 0.0
        passes["logit_disagreement"] = ~required_base & (gap <= logit_gap)
    passes["tier1_only"] = (
        np.ones(count, dtype=bool)
        if not bool(policy.get("tier1_only", False))
        else tier1
    )
    return passes


def vectorized_is_bet_eligible(frame: pd.DataFrame, policy: Mapping[str, Any]) -> np.ndarray:
    """Vectorized equivalent of ``is_bet_eligible`` for prepared mean-probability rows."""
    passes = vectorized_gate_passes(frame, policy)
    result = np.ones(len(frame), dtype=bool)
    for mask in passes.values():
        result &= mask
    return result


def choose_first_eligible(events: pd.DataFrame, mask: np.ndarray, timing: str) -> pd.DataFrame:
    """Choose one earliest qualified event per fixture; never use future EV to select time."""
    if len(events) != len(mask):
        raise ValueError("eligibility mask must align with event rows")
    if timing not in {
        "first_ready", "quote_triggered_first_eligible", "utc_even_2h_first_eligible",
        "utc_odd_2h_first_eligible", "noncausal_all_snapshot_oracle",
    }:
        raise ValueError(f"unknown timing policy: {timing}")
    eligible = events.loc[np.asarray(mask, dtype=bool)].copy()
    if eligible.empty:
        return eligible.reset_index(drop=True)
    clocks = pd.to_datetime(eligible["decision_at"], utc=True, errors="coerce")
    eligible = eligible.loc[clocks.notna()].copy()
    eligible["_clock"] = pd.to_datetime(eligible["decision_at"], utc=True, errors="coerce")
    if timing == "noncausal_all_snapshot_oracle":
        order = ["canonical_match_id", "ev_net", "odds", "bookmaker_id", "side"]
        ascending = [True, False, False, True, True]
        return eligible.sort_values(order, ascending=ascending, kind="mergesort").drop_duplicates(
            "canonical_match_id", keep="first"
        ).drop(columns="_clock").reset_index(drop=True)
    if timing in {"utc_even_2h_first_eligible", "utc_odd_2h_first_eligible"}:
        parity = 0 if timing.startswith("utc_even") else 1
        grid_clock = (
            eligible["_clock"].dt.minute.eq(0)
            & eligible["_clock"].dt.second.eq(0)
            & eligible["_clock"].dt.microsecond.eq(0)
            & eligible["_clock"].dt.nanosecond.eq(0)
            & eligible["_clock"].dt.hour.mod(2).eq(parity)
        )
        eligible = eligible.loc[grid_clock].copy()
    if eligible.empty:
        return eligible.drop(columns="_clock", errors="ignore").reset_index(drop=True)
    eligible["_ev_order"] = pd.to_numeric(eligible.get("ev_net"), errors="coerce")
    eligible["_odds_order"] = pd.to_numeric(eligible.get("odds"), errors="coerce")
    eligible = eligible.sort_values(
        ["canonical_match_id", "_clock", "_ev_order", "_odds_order", "bookmaker_id", "side"],
        ascending=[True, True, False, False, True, True],
        kind="mergesort",
    )
    selected = eligible.drop_duplicates("canonical_match_id", keep="first")
    return selected.drop(columns=["_clock", "_ev_order", "_odds_order"], errors="ignore").reset_index(drop=True)


def _event_panel(events: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    panel = events.copy()
    if panel.empty:
        return panel
    side_b = panel["side"].astype(str).str.lower().isin(("b", "team_b", "away", "2")).to_numpy()
    base_a = _float_column(panel, "p_base")
    market_a = _float_column(panel, "market_prob_a")
    if "market_prob" in panel:
        market_side = _float_column(panel, "market_prob")
    else:
        market_side = np.where(side_b, 1.0 - market_a, market_a)
    panel["prob_base"] = np.where(side_b, 1.0 - base_a, base_a)
    panel["prob_market_novig"] = market_side
    panel["prob_market_close_novig"] = market_side
    if "rating_disagreement" not in panel:
        panel["rating_disagreement"] = np.nan
    if "best_of" not in panel and "canonical_best_of" in panel:
        panel["best_of"] = panel["canonical_best_of"]
    if "competition_tier" not in panel:
        panel["competition_tier"] = [
            qualification_tier(league, at)
            for league, at in zip(
                panel.get("league", pd.Series([None] * len(panel))),
                panel["decision_at"],
            )
        ]
    fixture_columns = [
        name for name in (
            "y", "start_at", "settle_at", "result_recorded_at",
            "label_available_at", "partition", "cash_partition",
            "cash_decision_at", "cash_clock_basis",
        ) if name in fixtures
    ]
    if fixture_columns:
        panel = panel.drop(columns=[name for name in fixture_columns if name in panel])
        panel = panel.merge(
            fixtures[["canonical_match_id", *fixture_columns]].drop_duplicates("canonical_match_id"),
            on="canonical_match_id",
            how="left",
            validate="many_to_one",
        )
    return panel


def _with_alpha(panel: pd.DataFrame, alpha: float) -> pd.DataFrame:
    from scipy.special import expit, logit

    result = panel.copy()
    if result.empty:
        for column in ("prob_model", "prob_for_ev", "ev_net", "model_ev"):
            result[column] = pd.Series(dtype=float)
        return result
    base = _float_column(result, "prob_base")
    market = _float_column(result, "prob_market_novig")
    eps = 1e-6
    if float(alpha) == 0.0:
        blended = np.clip(market, eps, 1.0 - eps)
    elif float(alpha) == 1.0:
        blended = np.clip(base, eps, 1.0 - eps)
    else:
        p = np.clip(base, eps, 1.0 - eps)
        q = np.clip(market, eps, 1.0 - eps)
        blended = expit(float(alpha) * logit(p) + (1.0 - float(alpha)) * logit(q))
    odds = _float_column(result, "odds")
    result["prob_model"] = blended
    result["prob_for_ev"] = blended
    result["ev_net"] = blended * odds * 0.88 - 1.0
    result["model_ev"] = result["ev_net"]
    return result


def _orient_model_probabilities(
    panel: pd.DataFrame,
    model_rows: pd.DataFrame,
    probability_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only proven same-day model states, then orient canonical-A values."""
    required = {
        probability_column, "requested_utc_day", "forecast_at",
        "usable_for_same_day_projection", "strict_prior_feature_day_proof",
        "daily_state_cutoff_day", "feature_source_max_day",
    }
    missing = required.difference(model_rows.columns)
    if missing:
        raise ValueError(f"Daily model rows lack same-day proof fields: {sorted(missing)}")
    if len(model_rows) != len(panel):
        raise ValueError("Model prediction rows and event rows must have identical positional order")
    canonical = pd.to_numeric(
        model_rows[probability_column], errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    requested_times = pd.to_datetime(
        model_rows["requested_utc_day"], utc=True, errors="coerce"
    )
    requested_days = requested_times.dt.strftime("%Y-%m-%d").to_numpy()
    cutoff_times = pd.to_datetime(
        model_rows["daily_state_cutoff_day"], utc=True, errors="coerce"
    )
    feature_times = pd.to_datetime(
        model_rows["feature_source_max_day"], utc=True, errors="coerce"
    )
    requested_ns = requested_times.array.as_unit("ns").asi8
    cutoff_ns = cutoff_times.array.as_unit("ns").asi8
    feature_ns = feature_times.array.as_unit("ns").asi8
    event_times = pd.to_datetime(panel["decision_at"], utc=True, errors="coerce")
    prediction_times = pd.to_datetime(model_rows["forecast_at"], utc=True, errors="coerce")
    event_days = event_times.dt.strftime("%Y-%m-%d").to_numpy()
    event_ns = event_times.array.as_unit("ns").asi8
    prediction_ns = prediction_times.array.as_unit("ns").asi8
    nat = np.iinfo(np.int64).min
    usable = model_rows["usable_for_same_day_projection"].eq(True).to_numpy()
    strict_prior = model_rows["strict_prior_feature_day_proof"].eq(True).to_numpy()
    causal_same_day = (
        usable
        & strict_prior
        & (requested_days == event_days)
        & (requested_ns == cutoff_ns)
        & (feature_ns != nat)
        & (feature_ns < requested_ns)
        & (event_ns != nat)
        & (prediction_ns != nat)
        & (prediction_ns <= event_ns)
        & (prediction_times.dt.strftime("%Y-%m-%d").to_numpy() == requested_days)
    )
    canonical[~causal_same_day] = np.nan
    side_b = panel["side"].astype(str).str.lower().isin(
        ("b", "team_b", "away", "2")
    ).to_numpy()
    candidate = np.where(side_b & np.isfinite(canonical), 1.0 - canonical, canonical)
    return canonical, candidate


def _apply_model_variant(
    panel: pd.DataFrame, model_rows: pd.DataFrame, probability_column: str
) -> pd.DataFrame:
    canonical, candidate = _orient_model_probabilities(
        panel, model_rows, probability_column
    )
    result = panel.copy()
    result["p_base"] = canonical
    result["prob_base"] = candidate
    return result


def _first_ready_events(panel: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or rows.empty:
        return panel.iloc[0:0].copy()
    required = ("opportunity_id", "quote_id", "bookmaker_id")
    if not set(required).issubset(panel.columns) or not set(required).issubset(rows.columns):
        raise ValueError("prepared first-ready cohort lacks exact opportunity/book/quote identity")
    keys = rows[list(required)].drop_duplicates()
    marked = panel.merge(keys.assign(_first_ready=True), on=list(required), how="inner", validate="many_to_one")
    return marked.drop(columns="_first_ready").copy()


def _within_decision_window(events: pd.DataFrame) -> pd.DataFrame:
    panel = events.copy()
    if panel.empty:
        return panel
    decision = pd.to_datetime(panel["decision_at"], utc=True, errors="coerce")
    start = pd.to_datetime(panel["start_at"], utc=True, errors="coerce")
    lead_hours = (start - decision).dt.total_seconds() / 3600.0
    keep = decision.notna() & start.notna() & (lead_hours > 0.0) & (lead_hours < 48.0)
    return panel.loc[keep].copy()


def _monthly_portfolio_summary(
    ledger: pd.DataFrame,
    universe: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    family_size: int,
    seed_offset: int,
    fixture_month_universe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    bootstrap = protocol["bootstrap"]
    lower, upper = map(float, protocol["reliability_gates"]["near_ev_band"])
    months = sorted(
        pd.to_datetime(
            universe.get("cash_decision_at", universe.get("decision_at", pd.Series(dtype="datetime64[ns, UTC]"))),
            utc=True, errors="coerce",
        ).dropna().dt.strftime("%Y-%m").unique().tolist()
    )
    monthly_pnl = np.zeros(len(months), dtype=float)
    monthly_stake = np.zeros(len(months), dtype=float)
    near_error_sum = np.zeros(len(months), dtype=float)
    near_bet_count = np.zeros(len(months), dtype=float)
    near_stake = np.zeros(len(months), dtype=float)
    near_weighted_error = np.zeros(len(months), dtype=float)
    if not ledger.empty and months:
        month_index = {month: i for i, month in enumerate(months)}
        calendar_universe = fixture_month_universe if fixture_month_universe is not None else universe
        fixture_month = {
            int(row["canonical_match_id"]): pd.Timestamp(
                row.get("cash_decision_at", row.get("decision_at"))
            ).strftime("%Y-%m")
            for row in calendar_universe.to_dict("records")
            if pd.notna(row.get("canonical_match_id"))
            and pd.notna(row.get("cash_decision_at", row.get("decision_at")))
        }
        settled = ledger.loc[ledger["settled"].fillna(False)].copy()
        near_ev = pd.to_numeric(settled["model_ev"], errors="coerce")
        near = settled.loc[near_ev.gt(lower) & near_ev.le(upper)]
        for row in settled.to_dict("records"):
            month = fixture_month.get(int(row["canonical_match_id"]))
            if month is None or month not in month_index:
                raise ValueError("Settled bet cannot be placed on its full-universe preselection calendar")
            index = month_index[month]
            monthly_pnl[index] += float(row["realized_pnl"])
            monthly_stake[index] += float(row["stake"])
        for row in near.to_dict("records"):
            month = fixture_month.get(int(row["canonical_match_id"]))
            if month is None or month not in month_index:
                raise ValueError("Near-EV bet cannot be placed on its full-universe preselection calendar")
            index = month_index[month]
            error = float(row["model_ev"]) - float(row["realized_return"])
            near_error_sum[index] += error
            near_bet_count[index] += 1.0
            near_stake[index] += float(row["stake"])
            near_weighted_error[index] += float(row["stake"]) * error
    draws = int(bootstrap["draws"])
    if months:
        rng = np.random.default_rng(int(bootstrap["seed"]) + seed_offset)
        samples = rng.integers(0, len(months), size=(draws, len(months)))
        pnl_draw = monthly_pnl[samples].sum(axis=1)
        stake_draw = monthly_stake[samples].sum(axis=1)
        roi = np.divide(pnl_draw, stake_draw, out=np.full(draws, np.nan), where=stake_draw > 0)
        error_draw = near_error_sum[samples].sum(axis=1)
        count_draw = near_bet_count[samples].sum(axis=1)
        near_error = np.divide(
            error_draw, count_draw, out=np.full(draws, np.nan), where=count_draw > 0
        )
        weighted_error_draw = near_weighted_error[samples].sum(axis=1)
        near_stake_draw = near_stake[samples].sum(axis=1)
        near_error_stake = np.divide(
            weighted_error_draw,
            near_stake_draw,
            out=np.full(draws, np.nan),
            where=near_stake_draw > 0,
        )
    else:
        roi = near_error = near_error_stake = np.full(draws, np.nan)
    finite = np.isfinite(roi)
    finite_near = np.isfinite(near_error)
    finite_near_stake = np.isfinite(near_error_stake)
    confidence = float(bootstrap.get("confidence", 0.95))
    adjusted_tail = (1.0 - confidence) / (2.0 * max(1, family_size))

    def interval(values: np.ndarray, tail: float) -> list[float] | None:
        return (
            np.quantile(values, [tail, 1.0 - tail]).tolist()
            if np.isfinite(values).all() and len(months) >= 2 else None
        )

    unadjusted_tail = (1.0 - confidence) / 2.0

    return {
        "months": months,
        "month_count": len(months),
        "draws": draws,
        "search_family_size": family_size,
        "undefined_roi_replicates": int((~finite).sum()),
        "search_adjusted_roi_ci": interval(roi, adjusted_tail),
        "near_ev_error_bet_weighted_ci": interval(near_error, adjusted_tail),
        "near_ev_error_bet_weighted_95_ci": interval(near_error, unadjusted_tail),
        "near_ev_error_bet_undefined_replicates": int((~finite_near).sum()),
        "near_ev_error_stake_weighted_ci": interval(near_error_stake, adjusted_tail),
        "near_ev_error_stake_undefined_replicates": int((~finite_near_stake).sum()),
        "monthly_pnl": monthly_pnl.tolist(),
        "monthly_stake": monthly_stake.tolist(),
    }


def _action_partition_rows(
    selected: pd.DataFrame, partition: str | None, protocol: Mapping[str, Any]
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    required = {"partition", "decision_at"}
    if partition in (None, "TRAIN"):
        required.add("label_available_at")
    if not required.issubset(selected.columns):
        raise ValueError(f"Cash candidates lack strict action-partition fields: {sorted(required.difference(selected.columns))}")
    decision = pd.to_datetime(selected["decision_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(protocol["cutoff_at"])
    end = pd.Timestamp(protocol["assessment_end_at"])
    train = selected["partition"].eq("TRAIN") & decision.lt(cutoff)
    if "label_available_at" in selected:
        labels = pd.to_datetime(selected["label_available_at"], utc=True, errors="coerce")
        train &= labels.notna() & labels.le(cutoff)
    assess = selected["partition"].eq("ASSESS") & decision.ge(cutoff) & decision.lt(end)
    keep = train | assess if partition is None else (train if partition == "TRAIN" else assess)
    return selected.loc[keep].copy()


def _partition_fixtures(fixtures: pd.DataFrame, partition: str) -> pd.DataFrame:
    if "cash_partition" in fixtures:
        cash_partition = "TRAIN_CLOCK" if partition == "TRAIN" else "ASSESS_CLOCK"
        return fixtures.loc[fixtures["cash_partition"].eq(cash_partition)].copy()
    if "partition" not in fixtures:
        return fixtures.iloc[0:0].copy()
    return fixtures.loc[fixtures["partition"].eq(partition)].copy()


def _simulate(
    selected: pd.DataFrame,
    partition: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    from src.analysis.unified_evaluation_engine import simulate_operational_portfolio

    if not selected.empty:
        selected = _action_partition_rows(selected, partition, protocol)
    portfolio = protocol["portfolio"]
    return simulate_operational_portfolio(
        selected,
        selected,
        initial_bankroll=portfolio["initial_bankroll"],
        tax_rate=portfolio["tax_rate"],
        fraction=portfolio["kelly_fraction"],
        max_stake=portfolio["max_stake"],
        min_stake=portfolio["min_stake"],
        max_portfolio_exposure=portfolio["max_portfolio_exposure"],
    )


def _summary_row(
    config: Mapping[str, Any],
    timing: str,
    partition: str,
    selected: pd.DataFrame,
    wallet: dict[str, Any],
    universe: pd.DataFrame,
    protocol: Mapping[str, Any],
    family_size: int,
    *,
    fixture_month_universe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    summary = wallet["summary"]
    partition_selected = _action_partition_rows(selected, partition, protocol)
    ledger = wallet["ledger"]
    settled = ledger.loc[ledger["settled"].fillna(False)] if not ledger.empty else ledger
    active_months = (
        pd.to_datetime(settled["decision_at"], utc=True, errors="coerce")
        .dt.strftime("%Y-%m").nunique()
        if not settled.empty else 0
    )
    near_ev_values = (
        pd.to_numeric(settled.get("model_ev", pd.Series(dtype=float)), errors="coerce")
        if not settled.empty else pd.Series(dtype=float)
    )
    near_lower, near_upper = map(float, protocol["reliability_gates"]["near_ev_band"])
    near = (
        settled.loc[near_ev_values.gt(near_lower) & near_ev_values.le(near_upper)]
        if not settled.empty else settled
    )
    near_error = (
        pd.to_numeric(near["model_ev"], errors="coerce")
        - pd.to_numeric(near["realized_return"], errors="coerce")
        if not near.empty else pd.Series(dtype=float)
    )
    near_stake = (
        pd.to_numeric(near["stake"], errors="coerce").to_numpy(dtype=float)
        if not near.empty else np.array([], dtype=float)
    )
    monthly = _monthly_portfolio_summary(
        ledger, universe, protocol, family_size=family_size,
        seed_offset=int(config["config_id"][-8:], 16) + (0 if partition == "TRAIN" else 1),
        fixture_month_universe=fixture_month_universe,
    )
    return {
        **{key: config[key] for key in POLICY_KEYS},
        "config_id": config["config_id"],
        "family": config["family"],
        "alpha": config["alpha"],
        "timing": timing,
        "partition": partition,
        "preselection_fixture_count": len(universe),
        "selected_candidate_count": len(partition_selected),
        "bets": summary["bets_count"],
        "settled_bets": summary["settled_bets_count"],
        "pending_bets": summary["pending_bets_count"],
        "active_settled_months": int(active_months),
        "net_profit": float(summary["profit"]),
        "turnover": float(summary["turnover"]),
        "settled_turnover": float(summary["settled_turnover"]),
        "roi": float(summary["roi"]) if summary["roi"] is not None else None,
        "profit_per_preselection_fixture": (
            float(summary["profit"]) / len(universe) if len(universe) else None
        ),
        "final_cash": float(summary["final_cash"]),
        "final_reserved": float(summary["final_reserved"]),
        "final_equity": float(summary["final_equity"]),
        "peak_reserved_fraction": float(summary["peak_reserved_fraction"]),
        "execution_rejections": len(wallet["execution_rejections"]),
        "near_ev_bets": len(near),
        "near_ev_realized_return_mean": (
            float(pd.to_numeric(near["realized_return"], errors="coerce").mean())
            if len(near) else None
        ),
        "near_ev_positive_value_status": (
            "INSUFFICIENT_SUPPORT" if len(near) == 0
            else "POSITIVE" if float(pd.to_numeric(near["realized_return"], errors="coerce").mean()) > 0.0
            else "NON_POSITIVE"
        ),
        "near_ev_error_mean": float(near_error.mean()) if len(near_error) else None,
        "near_ev_error_stake_weighted": (
            float(np.average(near_error.to_numpy(dtype=float), weights=near_stake))
            if len(near_error) and np.isfinite(near_stake).all() and near_stake.sum() > 0
            else None
        ),
        "near_ev_containment_status": (
            "INSUFFICIENT_SUPPORT"
            if len(near) == 0 or monthly["near_ev_error_bet_weighted_95_ci"] is None
            else "PASS"
            if monthly["near_ev_error_bet_weighted_95_ci"][0] >= -0.02
            and monthly["near_ev_error_bet_weighted_95_ci"][1] <= 0.02
            else "FAIL"
        ),
        "near_ev_ci_contains_within_2pp": (
            bool(
                monthly["near_ev_error_bet_weighted_95_ci"][0] >= -0.02
                and monthly["near_ev_error_bet_weighted_95_ci"][1] <= 0.02
            )
            if len(near) and monthly["near_ev_error_bet_weighted_95_ci"] is not None
            else False
        ),
        "monthly_uncertainty": monthly,
    }


def _supported_policy(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> bool:
    return (
        summary["partition"] == "TRAIN"
        and summary["settled_bets"] >= int(protocol["min_policy_train_bets"])
        and summary["active_settled_months"] >= int(protocol["min_policy_train_months"])
    )


def _selection_rank(summary: Mapping[str, Any], baseline_id: str) -> tuple[Any, ...]:
    config = summary["_config"]
    distance = abs(float(config["alpha"]) - 0.5) + sum(
        config[key] != BASE_POLICY[key] for key in POLICY_KEYS
    )
    relaxed = 0
    for key in (
        "max_rating_disagreement", "max_tier1_market_gap", "max_ev_net",
        "max_ev_longshot", "max_bo1_market_gap", "max_odds",
        "max_logit_disagreement", "max_negative_clv_drift",
        "max_coinflip_market_gap", "bo1_max_odds",
    ):
        value = config[key]
        baseline = BASE_POLICY[key]
        if value is None and baseline is not None:
            relaxed += 1
        elif value is not None and baseline is not None and value > baseline:
            relaxed += 1
    if config["min_ev_net"] < BASE_POLICY["min_ev_net"]:
        relaxed += 1
    if config["wasserstein_epsilon"] < BASE_POLICY["wasserstein_epsilon"]:
        relaxed += 1
    if BASE_POLICY["dynamic_hurdle"] and not config["dynamic_hurdle"]:
        relaxed += 1
    if BASE_POLICY["universal_rating_guard"] and not config["universal_rating_guard"]:
        relaxed += 1
    return (
        -float(summary["profit_per_preselection_fixture"]),
        distance,
        relaxed,
        summary["config_id"],
    )


def _alpha_predictions(rows: pd.DataFrame, alpha: float) -> np.ndarray:
    from scipy.special import expit, logit

    p = _float_column(rows, "p_base")
    q = _float_column(rows, "market_prob_a")
    eps = 1e-6
    return expit(float(alpha) * logit(np.clip(p, eps, 1 - eps)) +
                 (1.0 - float(alpha)) * logit(np.clip(q, eps, 1 - eps)))

def _proper_scores(rows: pd.DataFrame, alpha: float) -> dict[str, Any]:
    from src.analysis.c0_edge_alpha_audit import proper_score_metrics

    probability = _alpha_predictions(rows, alpha)
    outcomes = _float_column(rows, "y")
    weights = _float_column(rows, "fixture_weight", default=1.0)
    available = np.isfinite(outcomes) & np.isfinite(probability) & np.isfinite(weights) & (weights > 0)
    metrics = proper_score_metrics(
        outcomes[available], probability[available], weights[available]
    )
    return {
        **metrics,
        "fixtures": int(rows.loc[available, "canonical_match_id"].nunique()),
        "ece_10_equal_width": metrics.get("ece"),
        "mean_prediction": (
            float(np.average(probability[available], weights=weights[available]))
            if available.any() else None
        ),
        "observed_rate": (
            float(np.average(outcomes[available], weights=weights[available]))
            if available.any() else None
        ),
    }


def _event_partition_panel(panel: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    if "partition" in panel:
        return panel
    if rows.empty or "partition" not in rows:
        return panel
    memberships = rows[["canonical_match_id", "partition"]].drop_duplicates("canonical_match_id")
    return panel.merge(memberships, on="canonical_match_id", how="left", validate="many_to_one")


def _market_ready_partition_panel(
    panel: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    readiness_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Partition market-only actions by earliest valid quote readiness, not C0 readiness."""
    if panel.empty:
        return panel
    readiness = panel if readiness_panel is None else readiness_panel
    window = _within_decision_window(readiness)
    if window.empty:
        result = panel.copy()
        result["market_ready_decision_at"] = pd.NaT
        result["partition"] = "NO_MARKET_READY_STATE"
        result["partition_reason"] = "no_valid_market_ready_clock"
        return result
    first = (
        window.sort_values(
            ["canonical_match_id", "decision_at", "opportunity_id"],
            kind="mergesort",
        )
        .drop_duplicates("canonical_match_id", keep="first")
    )
    from src.analysis.c0_edge_alpha_audit import _partition

    cutoff = pd.Timestamp(protocol["cutoff_at"])
    end = pd.Timestamp(protocol["assessment_end_at"])
    first["partition_result"] = [
        _partition(decision, label, outcome, cutoff, end)
        for decision, label, outcome in zip(
            first["decision_at"], first["label_available_at"], first["y"]
        )
    ]
    ready_clock = first.set_index("canonical_match_id")["decision_at"]
    result = panel.copy()
    result["market_ready_decision_at"] = result["canonical_match_id"].map(ready_clock)
    partition_map = first.set_index("canonical_match_id")["partition_result"]
    result["partition_result"] = result["canonical_match_id"].map(partition_map)
    result["partition"] = result["partition_result"].map(
        lambda value: value[0] if isinstance(value, tuple) else "NO_MARKET_READY_STATE"
    )
    result["partition_reason"] = result["partition_result"].map(
        lambda value: value[1] if isinstance(value, tuple) else "no_valid_market_ready_clock"
    )
    return result.drop(columns="partition_result")


def _paired_monthly_contrast(
    selected: pd.DataFrame,
    control: pd.DataFrame,
    universe: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    family_size: int,
    seed_offset: int,
) -> dict[str, Any]:
    months = sorted(
        pd.to_datetime(
            universe.get("cash_decision_at", universe.get("decision_at", pd.Series(dtype="datetime64[ns, UTC]"))),
            utc=True, errors="coerce",
        ).dropna().dt.strftime("%Y-%m").unique().tolist()
    )
    month_index = {month: index for index, month in enumerate(months)}
    counts = np.zeros(len(months), dtype=float)
    pnl_selected = np.zeros(len(months), dtype=float)
    stake_selected = np.zeros(len(months), dtype=float)
    pnl_control = np.zeros(len(months), dtype=float)
    stake_control = np.zeros(len(months), dtype=float)
    fixture_month = {
        int(row["canonical_match_id"]): pd.Timestamp(
            row.get("cash_decision_at", row.get("decision_at"))
        ).strftime("%Y-%m")
        for row in universe.to_dict("records")
        if pd.notna(row.get("canonical_match_id"))
        and pd.notna(row.get("cash_decision_at", row.get("decision_at")))
    }
    for month in fixture_month.values():
        if month in month_index:
            counts[month_index[month]] += 1.0

    def accumulate(ledger: pd.DataFrame, pnl: np.ndarray, stake: np.ndarray) -> None:
        if ledger.empty:
            return
        settled = ledger.loc[ledger["settled"].fillna(False)]
        for row in settled.to_dict("records"):
            month = fixture_month.get(int(row["canonical_match_id"]))
            if month in month_index:
                index = month_index[month]
                pnl[index] += float(row["realized_pnl"])
                stake[index] += float(row["stake"])

    accumulate(selected, pnl_selected, stake_selected)
    accumulate(control, pnl_control, stake_control)
    pnl_delta = pnl_selected - pnl_control
    draws = int(protocol["bootstrap"]["draws"])
    if months:
        rng = np.random.default_rng(int(protocol["bootstrap"]["seed"]) + seed_offset)
        samples = rng.integers(0, len(months), size=(draws, len(months)))
        count_draw = counts[samples].sum(axis=1)
        delta_draw = pnl_delta[samples].sum(axis=1)
        match_delta = np.divide(
            delta_draw, count_draw, out=np.full(draws, np.nan), where=count_draw > 0
        )
        active_months = counts[samples] > 0
        per_month = np.divide(
            pnl_delta[samples], counts[samples],
            out=np.full(samples.shape, np.nan), where=counts[samples] > 0,
        )
        equal_month_delta = np.divide(
            np.nansum(per_month, axis=1), active_months.sum(axis=1),
            out=np.full(draws, np.nan), where=active_months.sum(axis=1) > 0,
        )
        selected_pnl_draw = pnl_selected[samples].sum(axis=1)
        selected_stake_draw = stake_selected[samples].sum(axis=1)
        control_pnl_draw = pnl_control[samples].sum(axis=1)
        control_stake_draw = stake_control[samples].sum(axis=1)
        selected_roi = np.divide(
            selected_pnl_draw, selected_stake_draw,
            out=np.full(draws, np.nan), where=selected_stake_draw > 0,
        )
        control_roi = np.divide(
            control_pnl_draw, control_stake_draw,
            out=np.full(draws, np.nan), where=control_stake_draw > 0,
        )
        roi_delta = selected_roi - control_roi
    else:
        match_delta = equal_month_delta = roi_delta = np.full(draws, np.nan)
    tail = (1.0 - float(protocol["bootstrap"]["confidence"])) / (
        2.0 * max(1, family_size)
    )

    def ci(values: np.ndarray) -> list[float] | None:
        return (
            np.quantile(values, [tail, 1.0 - tail]).tolist()
            if np.isfinite(values).all() and len(months) >= 2 else None
        )

    return {
        "months": months,
        "preselection_fixtures": int(len(fixture_month)),
        "monthly_selected_pnl": pnl_selected.tolist(),
        "monthly_control_pnl": pnl_control.tolist(),
        "monthly_preselection_fixture_count": counts.tolist(),
        "match_weighted_pnl_per_fixture_delta": (
            float(pnl_delta.sum() / counts.sum()) if counts.sum() else None
        ),
        "match_weighted_pnl_per_fixture_delta_ci": ci(match_delta),
        "match_weighted_pnl_per_fixture_undefined_replicates": int(
            (~np.isfinite(match_delta)).sum()
        ),
        "equal_month_pnl_per_fixture_delta": (
            float(np.mean(pnl_delta[counts > 0] / counts[counts > 0]))
            if (counts > 0).any() else None
        ),
        "equal_month_pnl_per_fixture_delta_ci": ci(equal_month_delta),
        "equal_month_undefined_replicates": int((~np.isfinite(equal_month_delta)).sum()),
        "selected_minus_control_roi_delta_ci": ci(roi_delta),
        "roi_undefined_replicates": int((~np.isfinite(roi_delta)).sum()),
        "search_family_size": family_size,
        "draws": draws,
    }


def _selection_candidates(
    panel: pd.DataFrame,
    rows: pd.DataFrame,
    eligible: np.ndarray,
    timing: str,
    protocol: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if timing == "first_ready":
        candidates = _first_ready_events(panel, rows)
        if candidates.empty:
            return candidates
        keys = ["opportunity_id", "quote_id", "bookmaker_id", "side"]
        eligible_keys = panel.loc[np.asarray(eligible, dtype=bool), keys].drop_duplicates()
        candidates = candidates.merge(
            eligible_keys,
            on=keys,
            how="inner",
            validate="many_to_one",
        )
    else:
        candidates = panel.loc[np.asarray(eligible, dtype=bool)].copy()
    candidates = _within_decision_window(candidates)
    selected = choose_first_eligible(candidates, np.ones(len(candidates), dtype=bool), timing)
    return selected if protocol is None else _action_partition_rows(selected, None, protocol)


def _wallet_export(role: str, timing: str, partition: str, wallet: Mapping[str, Any]) -> dict[str, Any]:
    ledger = wallet["ledger"]
    pending = ledger.loc[~ledger["settled"].fillna(False)] if not ledger.empty else ledger
    records = ledger.to_dict("records") if not ledger.empty else []
    for record in records:
        for key, value in record.items():
            if isinstance(value, Decimal):
                record[key] = str(value)
    summary = {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in wallet["summary"].items()
    }
    pending_funds = sum(pending["stake"], Decimal("0")) if not pending.empty else Decimal("0")
    return {
        "role": role,
        "timing": timing,
        "partition": partition,
        "summary": summary,
        "pending_funds": str(pending_funds),
        "ledger": records,
    }


def _write_json(path: Path, value: Any) -> None:
    from src.analysis.c0_edge_alpha_audit import canonical_json_dumps

    path.write_text(canonical_json_dumps(value) + "\n", encoding="utf-8")


def run_policy_audit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Run the frozen alpha × qualification × timing policy audit."""
    from src.analysis.c0_edge_alpha_audit import (
        _load_prepared,
        canonical_json_sha256,
        fit_alpha_from_rows,
    )

    root = Path(protocol["output_root"])
    if not root.is_absolute():
        root = ROOT / root
    output = root / "policy"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / protocol.get(
        "protocol_path",
        "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol.json",
    )
    print("[policy] loading verified prepared cohort", flush=True)
    phase_start = pd.Timestamp.now(tz="UTC")
    cohort = _load_prepared(protocol, include_events=True)
    rows = cohort["rows"].copy()
    fixtures = cohort["fixtures"].copy()
    events = _event_panel(cohort["events"], fixtures)
    events = _event_partition_panel(events, rows)
    daily_native_states = cohort["daily_native_states"]
    timing_root = ROOT / protocol["inputs"]["timing_root"]
    print("[policy] loading/building exact odd-hour quote projections", flush=True)
    odd_grid_events, odd_grid_provenance = _load_or_build_odd_grid_events(
        protocol, fixtures, daily_native_states, timing_root, cohort,
    )
    market_odd_grid_events, market_odd_grid_provenance = _load_or_build_odd_grid_events(
        protocol, fixtures, daily_native_states, timing_root, cohort, require_native=False,
    )
    odd_events = _event_partition_panel(_event_panel(odd_grid_events, fixtures), rows)
    configs = build_policy_configs(protocol)
    diagnostic_configs = build_diagnostic_policy_configs(protocol, configs)
    timings = list(protocol["timing_policies"])
    baseline_core = protocol["policy_grid"]["baseline_core"]

    def core_config(alpha: float) -> dict[str, Any]:
        return next(
            config for config in configs
            if config["family"] == "core"
            and config["alpha"] == alpha
            and config["min_ev_net"] == baseline_core["min_ev_net"]
            and config["wasserstein_epsilon"] == baseline_core["wasserstein_epsilon"]
        )

    baseline = core_config(float(baseline_core["alpha"]))
    matched_market = core_config(0.0)
    c0_control = core_config(1.0)
    evidence = {
        "schema": "c0_edge_policy_input_evidence_v1",
        "run_id": protocol["run_id"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if protocol_path.is_file() else None,
        "alpha_input_hashes": cohort.get("evidence", {}).get("input_hashes", {}),
        "daily_native_states_sha256": _digest(daily_native_states.to_dict("records")),
        "odd_grid_provenance": odd_grid_provenance,
        "fixture_count": int(len(fixtures)),
        "unknown_outcome_count": int(fixtures.get("y", pd.Series(dtype=float)).isna().sum()),
        "config_count": len(configs),
        "config_ids_sha256": canonical_json_sha256([config["config_id"] for config in configs]),
        "diagnostic_control_count": len(diagnostic_configs),
        "diagnostic_controls": [
            {
                "config_id": config["config_id"],
                "control_name": config["control_name"],
                "selection_eligible": False,
                "alpha": config["alpha"],
                "policy": {key: config[key] for key in POLICY_KEYS},
            }
            for config in diagnostic_configs
        ],
    }
    _write_json(output / "input_evidence.json", evidence)
    fixtures.to_parquet(output / "fixture_universe.parquet", index=False)
    phase_costs: dict[str, float] = {
        "cohort_and_inventory_seconds": (pd.Timestamp.now(tz="UTC") - phase_start).total_seconds()
    }
    print(
        f"[policy] cohort rows={len(rows)} events={len(events)} fixtures={len(fixtures)} "
        f"configs={len(configs)}",
        flush=True,
    )
    print("[policy] fitting TRAIN alpha and caching probabilities", flush=True)
    phase_start = pd.Timestamp.now(tz="UTC")
    alpha_fit = fit_alpha_from_rows(
        rows,
        cutoff_at=protocol["cutoff_at"],
        min_fixtures=int(protocol["min_alpha_train_fixtures"]),
        min_months=int(protocol["min_alpha_train_months"]),
    )
    alpha_grid = [float(alpha) for alpha in protocol["alpha_grid"]]
    diagnostic_alphas = [float(config["alpha"]) for config in diagnostic_configs]
    cached_alphas = sorted(set(alpha_grid + diagnostic_alphas))
    cached = {
        alpha: _event_partition_panel(_with_alpha(events, alpha), rows)
        for alpha in cached_alphas
    }
    cached_odd = {
        alpha: _event_partition_panel(_with_alpha(odd_events, alpha), rows)
        for alpha in cached_alphas
    }
    score_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        for split in ("ALL", "TRAIN", "ASSESS"):
            selected_rows = rows if split == "ALL" else rows.loc[rows["partition"].eq(split)]
            score_rows.append(
                {"alpha": alpha, "partition": split, **_proper_scores(selected_rows, alpha)}
            )
    if alpha_fit.get("status") == "SELECTED":
        continuous_alpha = float(alpha_fit["alpha"])
        for split in ("ALL", "TRAIN", "ASSESS"):
            selected_rows = rows if split == "ALL" else rows.loc[rows["partition"].eq(split)]
            score_rows.append({
                "alpha": continuous_alpha,
                "partition": split,
                "selection_basis": "strict_TRAIN_proper_score_fit",
                **_proper_scores(selected_rows, continuous_alpha),
            })
    phase_costs["alpha_and_probability_cache_seconds"] = (
        pd.Timestamp.now(tz="UTC") - phase_start
    ).total_seconds()
    print(f"[policy] alpha fit={alpha_fit.get('status')} alpha={alpha_fit.get('alpha')}", flush=True)

    print("[policy] scoring frozen gate cells and timing policies", flush=True)
    phase_start = pd.Timestamp.now(tz="UTC")
    summaries: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    selected_frames: dict[tuple[str, str], pd.DataFrame] = {}
    family_size = len(configs) * len(timings)
    case_index = 0
    for config in configs:
        event_panels = {
            "observed": cached[config["alpha"]],
            "odd_grid": cached_odd[config["alpha"]],
        }
        timing_masks: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
        for source_name, panel in event_panels.items():
            gate_masks = vectorized_gate_passes(panel, config)
            eligible = np.ones(len(panel), dtype=bool)
            for mask in gate_masks.values():
                eligible &= mask
            if source_name == "observed":
                timing_masks["first_ready"] = (panel, eligible)
                timing_masks["quote_triggered_first_eligible"] = (panel, eligible)
                timing_masks["utc_even_2h_first_eligible"] = (panel, eligible)
            else:
                timing_masks["utc_odd_2h_first_eligible"] = (panel, eligible)
            for gate, mask in gate_masks.items():
                other = np.ones(len(panel), dtype=bool)
                for other_gate, other_mask in gate_masks.items():
                    if gate != other_gate:
                        other &= other_mask
                gate_rows.append({
                    "config_id": config["config_id"],
                    "family": config["family"],
                    "alpha": config["alpha"],
                    "event_source": source_name,
                    "gate": gate,
                    "candidate_rows": len(panel),
                    "pass_without_other_gates": int(mask.sum()),
                    "eligible_without_gate": int(other.sum()),
                    "veto_count": int((other & ~mask).sum()),
                    "active_veto": bool((other & ~mask).any()),
                    "eligible_after_all_gates": int(eligible.sum()),
                })
        for timing in timings:
            panel, eligible = timing_masks[timing]
            chosen = _selection_candidates(panel, rows, eligible, timing, protocol)
            selected_frames[(config["config_id"], timing)] = chosen
            for partition in ("TRAIN", "ASSESS"):
                universe = _partition_fixtures(fixtures, partition)
                wallet = _simulate(chosen, partition, protocol)
                result = _summary_row(
                    config, timing, partition, chosen, wallet, universe,
                    protocol, family_size,
                )
                result["_config"] = config
                summaries.append(result)
            case_index += 1
            if case_index % 100 == 0:
                print(
                    f"[policy] scored config/timing cases={case_index}/{family_size}",
                    flush=True,
                )
        if config["config_id"] == baseline["config_id"]:
            oracle = _selection_candidates(
                cached[config["alpha"]],
                rows,
                np.logical_and.reduce(list(vectorized_gate_passes(cached[config["alpha"]], config).values())),
                "noncausal_all_snapshot_oracle",
                protocol,
            )
            selected_frames[(config["config_id"], "noncausal_all_snapshot_oracle")] = oracle
            for partition in ("TRAIN", "ASSESS"):
                universe = _partition_fixtures(fixtures, partition)
                wallet = _simulate(oracle, partition, protocol)
                result = _summary_row(
                    config,
                    "noncausal_all_snapshot_oracle",
                    partition,
                    oracle,
                    wallet,
                    universe,
                    protocol,
                    family_size,
                )
                result["_config"] = config
                summaries.append(result)
    if alpha_fit.get("status") == "SELECTED":
        continuous_config = dict(baseline)
        continuous_config.update({
            "family": "proper_score_alpha_baseline",
            "alpha": float(alpha_fit["alpha"]),
        })
        continuous_config["config_id"] = "edge-" + _digest(
            {"family": continuous_config["family"], "alpha": continuous_config["alpha"],
             **{key: continuous_config[key] for key in POLICY_KEYS}}
        )[:16]
        continuous_panels = {
            "observed": _event_partition_panel(_with_alpha(events, continuous_config["alpha"]), rows),
            "odd_grid": _event_partition_panel(_with_alpha(odd_events, continuous_config["alpha"]), rows),
        }
        for timing in timings:
            source = "odd_grid" if timing == "utc_odd_2h_first_eligible" else "observed"
            panel = continuous_panels[source]
            masks = vectorized_gate_passes(panel, continuous_config)
            eligible = np.logical_and.reduce(list(masks.values()))
            selected = _selection_candidates(panel, rows, eligible, timing, protocol)
            selected_frames[(continuous_config["config_id"], timing)] = selected
            for partition in ("TRAIN", "ASSESS"):
                universe = _partition_fixtures(fixtures, partition)
                wallet = _simulate(selected, partition, protocol)
                result = _summary_row(
                    continuous_config, timing, partition, selected, wallet, universe,
                    protocol, family_size,
                )
                result["_config"] = continuous_config
                summaries.append(result)
        alpha_fit["baseline_policy_comparator_config_id"] = continuous_config["config_id"]
        alpha_fit["baseline_policy_comparator_family"] = continuous_config["family"]
    else:
        alpha_fit["baseline_policy_comparator_status"] = "NOT_EVALUATED_ALPHA_NOT_SELECTED"
    print("[policy] scoring fixed source-safety diagnostics", flush=True)
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_wallets: list[dict[str, Any]] = []
    for config in diagnostic_configs:
        diagnostic_panels = {
            "observed": cached[float(config["alpha"])],
            "odd_grid": cached_odd[float(config["alpha"])],
        }
        for timing in timings:
            source_name = (
                "odd_grid" if timing == "utc_odd_2h_first_eligible" else "observed"
            )
            panel = diagnostic_panels[source_name]
            masks = vectorized_gate_passes(panel, config)
            eligible = np.logical_and.reduce(list(masks.values()))
            selected = _selection_candidates(panel, rows, eligible, timing, protocol)
            for partition in ("TRAIN", "ASSESS"):
                wallet = _simulate(selected, partition, protocol)
                summary = _summary_row(
                    config, timing, partition, selected, wallet,
                    _partition_fixtures(fixtures, partition), protocol, 1,
                )
                diagnostic_rows.append({
                    "control_name": config["control_name"],
                    "control_interpretation": config["diagnostic_interpretation"],
                    "selection_eligible": False,
                    **{key: value for key, value in summary.items() if key != "_config"},
                })
                diagnostic_wallets.append(
                    _wallet_export(
                        f"diagnostic:{config['control_name']}:alpha{config['alpha']}",
                        timing, partition, wallet,
                    )
                )
        print(
            f"[policy] diagnostic {config['control_name']} alpha={config['alpha']} "
            f"timings={len(timings)} selection_eligible=false",
            flush=True,
        )
    if diagnostic_configs:
        pd.DataFrame(diagnostic_rows).to_parquet(
            output / "source_safety_only_controls.parquet", index=False
        )
        from src.analysis.c0_edge_alpha_audit import canonical_json_dumps
        (output / "source_safety_only_control_wallets.json").write_text(
            canonical_json_dumps(diagnostic_wallets) + "\n", encoding="utf-8"
        )
    phase_costs["fixed_policy_scoring_seconds"] = (
        pd.Timestamp.now(tz="UTC") - phase_start
    ).total_seconds()

    print("[policy] selecting policies on strict TRAIN only", flush=True)
    phase_start = pd.Timestamp.now(tz="UTC")
    selected_ids: dict[str, Any] = {}
    for timing in timings:
        train = [
            summary for summary in summaries
            if summary["timing"] == timing
            and summary["partition"] == "TRAIN"
            and summary["family"] != "proper_score_alpha_baseline"
            and _supported_policy(summary, protocol)
        ]
        if train:
            winner = min(train, key=lambda summary: _selection_rank(summary, baseline["config_id"]))
            selected_ids[timing] = {
                "status": "SELECTED",
                "config": {
                    key: winner["_config"][key]
                    for key in ("config_id", "family", "alpha", *POLICY_KEYS)
                },
                "train_bets": winner["settled_bets"],
                "train_active_months": winner["active_settled_months"],
                "train_profit_per_preselection_fixture": winner["profit_per_preselection_fixture"],
                "config_id": winner["config_id"],
                "alpha": winner["alpha"],
            }
        else:
            selected_ids[timing] = {
                "status": "NOT_SELECTED_INSUFFICIENT_SUPPORT",
                "config_id": None,
                "train_bets": max(
                    (
                        summary["settled_bets"] for summary in summaries
                        if summary["timing"] == timing and summary["partition"] == "TRAIN"
                    ),
                    default=0,
                ),
                "required_bets": protocol["min_policy_train_bets"],
                "required_months": protocol["min_policy_train_months"],
            }
    print("[policy] applying locked C0 policies to genuine model predictions", flush=True)
    model_root = Path(protocol["output_root"])
    if not model_root.is_absolute():
        model_root = ROOT / model_root
    model_root = model_root / "models"
    model_receipt_path = model_root / "model_audit_receipt.json"
    model_comparison_rows: list[dict[str, Any]] = []
    model_wallet_exports: list[dict[str, Any]] = []
    if model_receipt_path.is_file():
        model_receipt = json.loads(model_receipt_path.read_text(encoding="utf-8"))
        if model_receipt.get("run_id") != protocol["run_id"]:
            raise ValueError("Model audit receipt is bound to a different recovery run")
        if int(model_receipt.get("matched_daily_model_predictions", -1)) != 1126:
            raise ValueError("Model receipt does not bind all 1,126 native daily predictions")
        artifact = model_receipt.get("prediction_artifacts", {}).get("daily_model_predictions", {})
        prediction_path = Path(artifact.get("path", ""))
        if not prediction_path.is_absolute():
            prediction_path = ROOT / prediction_path
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Model receipt daily predictions are unavailable: {prediction_path}")
        daily_predictions = pd.read_parquet(prediction_path)
        if len(daily_predictions) != 1126:
            raise ValueError("Model daily artifact does not contain all 1,126 frozen native keys")
        if (
            "model_query_matched" not in daily_predictions
            or not daily_predictions["model_query_matched"].fillna(False).all()
        ):
            raise ValueError("Model daily artifact contains unauthenticated or absent query predictions")
        required_keys = {
            "canonical_match_id", "requested_utc_day", "decision_at", "forecast_at",
            "usable_for_same_day_projection", "strict_prior_feature_day_proof",
            "daily_state_cutoff_day", "feature_source_max_day",
        }
        if not required_keys.issubset(daily_predictions.columns):
            raise ValueError("Genuine model daily artifact lacks same-day proof fields")
        if (
            not daily_predictions["usable_for_same_day_projection"].eq(True).all()
            or not daily_predictions["strict_prior_feature_day_proof"].eq(True).all()
        ):
            raise ValueError("Model daily artifact includes rows without strict prior-feature proof")
        daily_predictions["_model_day"] = pd.to_datetime(
            daily_predictions["requested_utc_day"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        model_columns = {
            "C0": "p_c0",
            "C0-D-Core": "p_c0d_core",
            "C0-D-Core-17T": "p_c0d_core_17t",
            "C0-Slim": "p_c0_slim",
            "fresh18": "p_c0d_18t",
            "fresh20": "p_c0d_20t",
        }
        missing_model_columns = set(model_columns.values()).difference(daily_predictions.columns)
        if missing_model_columns:
            raise ValueError(f"Genuine daily model artifact lacks probabilities: {sorted(missing_model_columns)}")
        for probability_column in model_columns.values():
            model_probability = pd.to_numeric(
                daily_predictions[probability_column], errors="coerce"
            )
            if (
                not np.isfinite(model_probability.to_numpy(dtype=float)).all()
                or ((model_probability <= 0.0) | (model_probability >= 1.0)).any()
            ):
                raise ValueError(
                    f"Genuine daily model artifact has invalid probabilities in {probability_column}"
                )
        if daily_predictions.duplicated(["canonical_match_id", "_model_day"]).any():
            raise ValueError("Genuine daily model artifact has duplicate fixture/day predictions")
        model_keyed = daily_predictions.set_index(
            [daily_predictions.canonical_match_id.astype(str), "_model_day"]
        )
        for model_name, probability_column in model_columns.items():
            for timing in timings:
                source_panel = (
                    cached_odd[float(baseline["alpha"])]
                    if timing == "utc_odd_2h_first_eligible"
                    else cached[float(baseline["alpha"])]
                )
                panel = source_panel.copy()
                panel_days = pd.to_datetime(
                    panel["decision_at"], utc=True, errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                key_index = pd.MultiIndex.from_arrays([
                    panel["canonical_match_id"].astype(str).to_numpy(),
                    panel_days.to_numpy(),
                ])
                model_rows = model_keyed.reindex(key_index)
                model_variant_panel = _apply_model_variant(
                    panel, model_rows, probability_column
                )
                baseline_config = dict(baseline)
                selected_id = selected_ids[timing]["config_id"]
                locked_configs = [("baseline", baseline_config)]
                if selected_id is not None:
                    locked_configs.append(("train_selected", next(
                        config for config in configs if config["config_id"] == selected_id
                    )))
                for role, alpha in (("matched_market_alpha0", 0.0), ("pure_sport_alpha1", 1.0)):
                    control_config = dict(baseline)
                    control_config.update({"alpha": alpha, "family": role})
                    control_config["config_id"] = "edge-" + _digest({
                        "family": role, "alpha": alpha,
                        **{key: control_config[key] for key in POLICY_KEYS},
                    })[:16]
                    locked_configs.append((role, control_config))
                for locked_role, locked_config in locked_configs:
                    model_panel = _with_alpha(
                        model_variant_panel, locked_config["alpha"]
                    )
                    gate_masks = vectorized_gate_passes(model_panel, locked_config)
                    eligible = np.logical_and.reduce(list(gate_masks.values()))
                    selected_model = _selection_candidates(model_panel, rows, eligible, timing, protocol)
                    for partition in ("TRAIN", "ASSESS"):
                        wallet = _simulate(selected_model, partition, protocol)
                        summary = _summary_row(
                            locked_config, timing, partition, selected_model, wallet,
                            _partition_fixtures(fixtures, partition), protocol, family_size,
                        )
                        model_comparison_rows.append({
                            "model": model_name,
                            "locked_policy_role": locked_role,
                            **{key: value for key, value in summary.items() if key != "_config"},
                        })
                        model_wallet_exports.append(
                            _wallet_export(f"{model_name}:{locked_role}", timing, partition, wallet)
                        )
    else:
        raise FileNotFoundError(
            f"Run the genuine C0 edge model audit before policy evaluation: {model_receipt_path}"
        )
    pd.DataFrame(model_comparison_rows).to_parquet(
        output / "locked_model_policy_comparisons.parquet", index=False
    )
    _write_json(output / "locked_model_policy_wallets.json", model_wallet_exports)
    public_summaries = [{key: value for key, value in row.items() if key != "_config"} for row in summaries]
    pd.DataFrame(public_summaries).to_parquet(output / "case_summaries.parquet", index=False)
    pd.DataFrame(gate_rows).to_parquet(output / "gate_support.parquet", index=False)
    pd.DataFrame(score_rows).to_parquet(output / "alpha_score_curves.parquet", index=False)
    _write_json(
        output / "selection_ids.json",
        {
            "policy_selection_rule": protocol["policy_selection"],
            "selected_by_timing": selected_ids,
            "alpha_fit": alpha_fit,
            "assessment_winner_selection": "PROHIBITED; all assessment cells descriptive only",
            "baseline_config_id": baseline["config_id"],
            "cash_partition_role": "full-universe preselection denominator and zero-action month accounting only",
            "action_partition_rule": "strict fixture partition; TRAIN requires label_available_at<=cutoff and decision<cutoff; ASSESS requires cutoff<=decision<assessment_end",
            "diagnostic_controls": {
                "selection_eligible": False,
                "summary_path": str(output / "source_safety_only_controls.parquet")
                if diagnostic_configs else None,
                "wallet_path": str(output / "source_safety_only_control_wallets.json")
                if diagnostic_configs else None,
                "configuration_ids": [config["config_id"] for config in diagnostic_configs],
            },
        },
    )
    phase_costs["train_policy_selection_seconds"] = (
        pd.Timestamp.now(tz="UTC") - phase_start
    ).total_seconds()

    print("[policy] exporting selected/control wallets and market coverage", flush=True)
    phase_start = pd.Timestamp.now(tz="UTC")
    wallet_exports: list[dict[str, Any]] = []
    wallet_ledgers: dict[tuple[str, str, str], pd.DataFrame] = {}
    for timing in timings:
        roles: list[tuple[str, str]] = [
            (baseline["config_id"], "baseline"),
            (matched_market["config_id"], "matched_market_alpha0"),
            (c0_control["config_id"], "pure_c0_alpha1"),
        ]
        comparator_id = alpha_fit.get("baseline_policy_comparator_config_id")
        if comparator_id is not None:
            roles.append((comparator_id, "proper_score_alpha_baseline"))
        selected_id = selected_ids[timing]["config_id"]
        if selected_id is not None:
            roles.append((selected_id, "train_selected"))
        for config_id, role in dict.fromkeys(roles):
            selected = selected_frames[(config_id, timing)]
            for partition in ("TRAIN", "ASSESS"):
                wallet = _simulate(selected, partition, protocol)
                wallet_ledgers[(timing, partition, role)] = wallet["ledger"]
                wallet_exports.append(_wallet_export(role, timing, partition, wallet))
    oracle = selected_frames[(baseline["config_id"], "noncausal_all_snapshot_oracle")]
    for partition in ("TRAIN", "ASSESS"):
        wallet = _simulate(oracle, partition, protocol)
        wallet_exports.append(
            _wallet_export("noncausal_all_snapshot_oracle", "noncausal_all_snapshot_oracle", partition, wallet)
        )
    contrast_rows = []
    contrast_family_size = len(timings) * 2 * 3
    for timing_index, timing in enumerate(timings):
        selected_id = selected_ids[timing]["config_id"]
        for partition_index, partition in enumerate(("TRAIN", "ASSESS")):
            universe = _partition_fixtures(fixtures, partition)
            for control_index, control_role in enumerate(
                ("baseline", "matched_market_alpha0", "pure_c0_alpha1")
            ):
                if selected_id is None:
                    contrast_rows.append({
                        "timing": timing,
                        "partition": partition,
                        "control": control_role,
                        "selection_status": selected_ids[timing]["status"],
                        "contrast": None,
                    })
                    continue
                contrast = _paired_monthly_contrast(
                    wallet_ledgers[(timing, partition, "train_selected")],
                    wallet_ledgers[(timing, partition, control_role)],
                    universe,
                    protocol,
                    family_size=contrast_family_size,
                    seed_offset=100_000 + timing_index * 100 + partition_index * 10 + control_index,
                )
                contrast_rows.append({
                    "timing": timing,
                    "partition": partition,
                    "selected_config_id": selected_id,
                    "control": control_role,
                    "selection_status": selected_ids[timing]["status"],
                    "contrast": contrast,
                })
    from src.analysis.c0_edge_alpha_audit import canonical_json_dumps

    (output / "selected_control_wallets.json").write_text(
        canonical_json_dumps(wallet_exports) + "\n", encoding="utf-8"
    )
    _write_json(output / "policy_contrasts.json", {
        "selection": "strict TRAIN only; ASSESS contrasts are descriptive and all fixed cells remain unselected",
        "monthly_unit": "shared full preselection fixture-months; no-bet fixture PnL is zero; unknown outcomes are not losses",
        "contrasts": contrast_rows,
    })
    native_present = _float_column(events, "p_base")
    market_only = (
        int((~np.isfinite(native_present) & np.isfinite(_float_column(events, "prob_market_novig"))).sum())
        if not events.empty else 0
    )
    market_coverage = {
        "matched_native_covered_alpha0_control_config_id": matched_market["config_id"],
        "all_quote_events_without_native_probability": market_only,
        "raw_quote_fixture_count": odd_grid_provenance["raw_quote_fixture_count"],
        "native_daily_state_fixture_count": odd_grid_provenance["native_daily_state_fixture_count"],
        "market_only_fixture_count": odd_grid_provenance["market_only_fixture_count"],
        "all_quote_cash_status": "EVALUATED_AS_A_SEPARATE_MARKET_ONLY_CONTROL",
        "market_only_cash_result": "MARKET_ONLY_PROBABILITY; SPORT_BASE_AND_UNAVAILABLE_GUARDS_NOT_IMPUTED",
        "no_c0_imputation": True,
    }
    market_only_config = dict(baseline)
    market_only_config.update({
        "alpha": 0.0,
        "family": "market_only_all_quotes",
        "max_rating_disagreement": None,
        "max_tier1_market_gap": None,
        "max_bo1_market_gap": None,
        "max_coinflip_market_gap": None,
        "max_logit_disagreement": None,
        "universal_rating_guard": False,
    })
    market_only_config["config_id"] = "edge-" + _digest({
        "family": market_only_config["family"],
        **{key: market_only_config[key] for key in ("alpha", *POLICY_KEYS)},
    })[:16]
    market_observed_unpartitioned = _event_panel(events, fixtures)
    market_observed_unpartitioned["p_base"] = np.nan
    market_observed_unpartitioned["prob_base"] = np.nan
    market_observed_unpartitioned = _with_alpha(market_observed_unpartitioned, 0.0)
    market_odd_unpartitioned = _event_panel(market_odd_grid_events, fixtures)
    market_odd_unpartitioned["p_base"] = np.nan
    market_odd_unpartitioned["prob_base"] = np.nan
    market_odd_unpartitioned = _with_alpha(market_odd_unpartitioned, 0.0)
    market_readiness = pd.concat(
        [market_observed_unpartitioned, market_odd_unpartitioned],
        ignore_index=True,
        sort=False,
    )
    market_panel = _market_ready_partition_panel(
        market_observed_unpartitioned, protocol, readiness_panel=market_readiness
    )
    market_odd_panel = _market_ready_partition_panel(
        market_odd_unpartitioned, protocol, readiness_panel=market_readiness
    )
    market_clock_events = _within_decision_window(market_readiness)
    if market_clock_events.empty:
        market_clock_map = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        market_clock_map = (
            market_clock_events.sort_values(
                ["canonical_match_id", "decision_at", "opportunity_id"],
                kind="mergesort",
            )
            .drop_duplicates("canonical_match_id", keep="first")
            .set_index("canonical_match_id")["decision_at"]
        )
    market_cash_universe = fixtures.copy()
    cash_clock = pd.to_datetime(
        market_cash_universe["cash_decision_at"], utc=True, errors="coerce"
    )
    market_clock = market_cash_universe["canonical_match_id"].map(market_clock_map)
    market_clock_fallback = cash_clock.isna() & market_clock.notna()
    market_cash_universe["cash_decision_at"] = cash_clock.where(
        ~market_clock_fallback, market_clock
    )
    market_cash_clock_fallback_count = int(market_clock_fallback.sum())
    market_only_rows: list[dict[str, Any]] = []
    market_only_wallets: list[dict[str, Any]] = []
    for timing in timings:
        timing_panel = (
            market_odd_panel
            if timing == "utc_odd_2h_first_eligible"
            else market_panel
        )
        gates = vectorized_gate_passes(timing_panel, market_only_config)
        eligible = np.logical_and.reduce(list(gates.values()))
        if timing == "first_ready":
            all_window = _within_decision_window(timing_panel)
            clocks = pd.to_datetime(all_window["decision_at"], utc=True, errors="coerce")
            first_clock = clocks.groupby(all_window["canonical_match_id"]).transform("min")
            first_state = all_window.loc[clocks.eq(first_clock)]
            eligible_index = timing_panel.index[np.asarray(eligible, dtype=bool)]
            market_candidates = first_state.loc[first_state.index.isin(eligible_index)].copy()
        else:
            market_candidates = _within_decision_window(timing_panel.loc[eligible].copy())
        chosen_market = choose_first_eligible(
            market_candidates, np.ones(len(market_candidates), dtype=bool), timing
        )
        chosen_market = _action_partition_rows(chosen_market, None, protocol)
        for partition in ("TRAIN", "ASSESS"):
            wallet = _simulate(chosen_market, partition, protocol)
            summary = _summary_row(
                market_only_config, timing, partition, chosen_market, wallet,
                _partition_fixtures(market_cash_universe, partition),
                protocol, family_size,
                fixture_month_universe=market_cash_universe,
            )
            market_only_rows.append({
                "control_identity": "MARKET_ONLY_ALL_QUOTES_NO_SPORTS_BASE",
                **{key: value for key, value in summary.items() if key != "_config"},
            })
            market_only_wallets.append(
                _wallet_export("market_only_all_quotes", timing, partition, wallet)
            )
    market_quote_mask = np.isfinite(_float_column(market_panel, "prob_market_novig"))
    market_coverage.update({
        "market_only_config_id": market_only_config["config_id"],
        "market_only_event_count": int(market_quote_mask.sum()),
        "market_only_quote_fixture_count": int(
            market_panel.loc[market_quote_mask, "canonical_match_id"].nunique()
        ) if market_quote_mask.any() else 0,
        "market_only_summary_path": str(output / "market_only_all_quote_summaries.parquet"),
        "market_only_wallet_path": str(output / "market_only_all_quote_wallets.json"),
        "market_only_odd_grid_event_count": len(market_odd_panel),
        "market_only_odd_grid_provenance": market_odd_grid_provenance,
        "market_only_cash_denominator_fixture_count": len(market_cash_universe),
        "market_only_cash_clock_fallback_count": market_cash_clock_fallback_count,
        "market_only_partition_basis": (
            "earliest valid market quote readiness; strict TRAIN label clock, "
            "independent of native C0 fixture partition"
        ),
        "market_only_gates_disabled_without_sports_base": [
            "max_logit_disagreement", "max_rating_disagreement",
            "max_tier1_market_gap", "max_bo1_market_gap",
            "max_coinflip_market_gap",
        ],
        "market_only_utc_grid_source": (
            "odd-hour exact as-of raw quote projection; no native-state requirement "
            "or C0 imputation"
        ),
    })
    pd.DataFrame(market_only_rows).to_parquet(
        output / "market_only_all_quote_summaries.parquet", index=False
    )
    _write_json(output / "market_only_all_quote_wallets.json", market_only_wallets)
    _write_json(output / "market_only_coverage.json", market_coverage)
    phase_costs["selected_wallet_export_seconds"] = (
        pd.Timestamp.now(tz="UTC") - phase_start
    ).total_seconds()

    report = {
        "schema": "c0_edge_policy_audit_v1",
        "run_id": protocol["run_id"],
        "status": "COMPLETED_DESCRIPTIVE_HISTORICAL_AUDIT",
        "protocol_sha256": evidence["protocol_sha256"],
        "input_evidence_sha256": canonical_json_sha256(evidence),
        "config_count": len(configs),
        "core_config_count": sum(config["family"] == "core" for config in configs),
        "diagnostic_controls": {
            "selection_eligible": False,
            "config_count": len(diagnostic_configs),
            "summary_path": str(output / "source_safety_only_controls.parquet")
            if diagnostic_configs else None,
            "wallet_path": str(output / "source_safety_only_control_wallets.json")
            if diagnostic_configs else None,
            "purpose": "fixed source-safety gate ablation; not part of profit selection",
        },
        "timings": timings,
        "selection_by_timing": selected_ids,
        "alpha_fit": alpha_fit,
        "market_only_coverage": market_coverage,
        "oracle": "noncausal comparator only; future snapshot not executable",
        "assessment_rule": "All fixed assessment cells descriptive; no assessment-driven policy recommendation.",
        "fixture_universe_path": str(output / "fixture_universe.parquet"),
        "case_summaries_path": str(output / "case_summaries.parquet"),
        "gate_support_path": str(output / "gate_support.parquet"),
        "wallets_path": str(output / "selected_control_wallets.json"),
        "locked_model_comparisons_path": str(output / "locked_model_policy_comparisons.parquet"),
        "locked_model_wallets_path": str(output / "locked_model_policy_wallets.json"),
        "cash_partition_role": "denominator and zero-action accounting only; never action authorization",
        "action_partition_rule": "strict fixture partition; TRAIN label_available_at<=cutoff; ASSESS decisions in [cutoff, assessment_end)",
        "costs_seconds": phase_costs,
        "limitations": [
            "Historical snapshots are not authenticated issuance or live execution.",
            "Same-cutoff close drift is non-identifiable and is not treated as CLV.",
            "Missing EXP/rating evidence remains unavailable, never fabricated agreement.",
            "No future close, target-lineup probability, or market-only C0 imputation.",
            "All exposed 2026 data remain development, not genuinely unseen.",
        ],
    }
    _write_json(output / "results.json", report)
    print(
        f"[policy] complete case_rows={len(summaries)} selected="
        f"{sum(value['status'] == 'SELECTED' for value in selected_ids.values())}",
        flush=True,
    )
    return {
        "report": report,
        "output_root": output,
        "case_summaries": pd.DataFrame(public_summaries),
        "gate_support": pd.DataFrame(gate_rows),
        "selection_ids": selected_ids,
        "alpha_scores": pd.DataFrame(score_rows),
        "locked_model_comparisons": pd.DataFrame(model_comparison_rows),
        "diagnostic_controls": pd.DataFrame(diagnostic_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen C0 edge-recovery policy audit")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "data/05_model_input/c0_edge_recovery/"
        "c0_edge_recovery_20261022_01/protocol.json",
    )
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text())
    protocol["_protocol_sha256"] = _file_sha256(protocol_path)
    protocol["protocol_path"] = str(protocol_path)
    run_policy_audit(protocol)
    return 0




def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_or_build_odd_grid_events(
    protocol: Mapping[str, Any],
    fixtures: pd.DataFrame,
    daily_native_states: pd.DataFrame,
    timing_root: Path,
    cohort: Mapping[str, Any],
    *,
    require_native: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse an exact, receipt-validated data-only odd-clock projection."""
    from src.analysis.c0_policy_data import _utc, admit_policy_quotes

    receipt = cohort.get("receipt", {})
    source_hashes = {
        "quotes_sha256": _file_sha256(timing_root / "quotes.parquet"),
        "fixtures_sha256": receipt.get("fixtures_sha256"),
        "daily_native_states_sha256": receipt.get("daily_native_states_sha256"),
        "quote_admission_module_sha256": _file_sha256(ROOT / "src/analysis/c0_policy_data.py"),
        "latest_state_module_sha256": _file_sha256(
            ROOT / "betting_app/services/prematch_quote_service.py"
        ),
        "projection_code_sha256": hashlib.sha256(
            (
                inspect.getsource(_build_odd_grid_events)
                + inspect.getsource(admit_policy_quotes)
                + inspect.getsource(_utc)
            ).encode("utf-8")
        ).hexdigest(),
        "projection_parameters_sha256": _digest({
            "freshness_seconds": protocol["cohort"]["freshness_seconds"],
            "minimum_other_books": protocol["cohort"]["minimum_other_books"],
            "require_native": require_native,
            "grid": "exact UTC odd-hour minute00 second00; strict [decision-ttl,decision]",
        }),
    }
    if any(not value for value in source_hashes.values()):
        raise ValueError("Odd-grid cache requires verified fixture and native-state artifact hashes")
    cache_dir = ROOT / "data/04_feature" / str(protocol["run_id"]) / "policy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode = "native" if require_native else "market"
    events_path = cache_dir / f"odd_grid_{mode}.parquet"
    receipt_path = cache_dir / f"odd_grid_{mode}_receipt.json"
    if receipt_path.is_file():
        cached_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            cached_receipt.get("schema") != "c0_edge_policy_odd_grid_cache_v1"
            or cached_receipt.get("run_id") != protocol["run_id"]
            or cached_receipt.get("require_native") is not require_native
            or cached_receipt.get("input_hashes") != source_hashes
        ):
            raise ValueError(f"Odd-grid {mode} cache receipt does not match frozen inputs/code")
        if not events_path.is_file() or _file_sha256(events_path) != cached_receipt.get("events_sha256"):
            raise ValueError(f"Odd-grid {mode} cache artifact differs from its receipt")
        events = pd.read_parquet(events_path)
        if "native_input_hashes" in events:
            events["native_input_hashes"] = events["native_input_hashes"].map(
                lambda value: json.loads(value) if isinstance(value, str) and value.startswith("{") else value
            )
        provenance = dict(cached_receipt["provenance"])
        provenance["cache"] = {
            "used": True,
            "stage04_artifact": str(events_path.relative_to(ROOT)),
            "receipt": str(receipt_path.relative_to(ROOT)),
        }
        return events, provenance
    if events_path.exists():
        raise FileExistsError(f"Refusing to use unreceipted odd-grid {mode} cache")

    events, provenance = _build_odd_grid_events(
        protocol, fixtures, daily_native_states, timing_root, require_native=require_native
    )
    provenance = dict(provenance)
    provenance["cache"] = {
        "used": False,
        "stage04_artifact": str(events_path.relative_to(ROOT)),
        "receipt": str(receipt_path.relative_to(ROOT)),
    }
    parquet_events = events.copy()
    if "native_input_hashes" in parquet_events:
        parquet_events["native_input_hashes"] = parquet_events["native_input_hashes"].map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            if isinstance(value, dict) else value
        )
    parquet_events.to_parquet(events_path, index=False)
    _write_json(receipt_path, {
        "schema": "c0_edge_policy_odd_grid_cache_v1",
        "run_id": protocol["run_id"],
        "require_native": require_native,
        "input_hashes": source_hashes,
        "events_sha256": _file_sha256(events_path),
        "event_rows": int(len(events)),
        "provenance": provenance,
    })
    return events, provenance


def _build_odd_grid_events(
    protocol: Mapping[str, Any],
    fixtures: pd.DataFrame,
    daily_native_states: pd.DataFrame,
    timing_root: Path,
    *,
    require_native: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project exact odd-hour raw quote states, optionally requiring a native model state."""
    freshness_seconds = int(protocol["cohort"]["freshness_seconds"])
    minimum_other_books = int(protocol["cohort"]["minimum_other_books"])
    quote_path = timing_root / "quotes.parquet"
    quotes = pd.read_parquet(quote_path)
    states = daily_native_states.copy()
    quotes["quoted_at"] = pd.to_datetime(quotes["quoted_at"], utc=True, errors="coerce")
    if "forecast_at" in states:
        states["forecast_at"] = pd.to_datetime(states["forecast_at"], utc=True, errors="coerce")
    raw_a = pd.to_numeric(quotes["odds_a"], errors="coerce")
    raw_b = pd.to_numeric(quotes["odds_b"], errors="coerce")
    aligned = quotes["raw_a_is_canonical_a"].fillna(False).to_numpy(dtype=bool)
    canonical_a = np.where(aligned, raw_a, raw_b)
    canonical_b = np.where(aligned, raw_b, raw_a)
    quotes["_odds_canonical_a"] = canonical_a
    quotes["_odds_canonical_b"] = canonical_b
    quotes["_valid_price"] = (
        np.isfinite(canonical_a) & np.isfinite(canonical_b)
        & (canonical_a > 1) & (canonical_b > 1)
    )
    quotes = quotes.sort_values(["canonical_match_id", "quoted_at", "quote_id"], kind="mergesort")
    quote_groups = {int(match_id): group for match_id, group in quotes.groupby("canonical_match_id", sort=False)}
    native_groups = {
        int(match_id): group
        for match_id, group in states.groupby("canonical_match_id", sort=False)
    }
    fixture_by_id = {
        int(row["canonical_match_id"]): row
        for row in fixtures.to_dict("records")
        if pd.notna(row.get("canonical_match_id"))
    }
    records: list[dict[str, Any]] = []
    targets = 0
    no_native = 0
    insufficient_quotes = 0
    strict_day_proofs: list[str] = []
    for match_id, fixture in fixture_by_id.items():
        start = pd.to_datetime(fixture.get("start_at"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        qgroup = quote_groups.get(match_id)
        ngroup = native_groups.get(match_id)
        if qgroup is None or (require_native and ngroup is None):
            continue
        known_quotes = qgroup.loc[qgroup["quoted_at"].notna()]
        known_quote_ns = known_quotes["quoted_at"].array.as_unit("ns").asi8
        known_bookmakers = known_quotes["bookmaker_id"].to_numpy()
        for lag in range(0, 48):
            decision = start.floor("h") - pd.Timedelta(hours=lag)
            lead_hours = (start - decision).total_seconds() / 3600.0
            if not 0.0 < lead_hours < 48.0 or decision.hour % 2 != 1:
                continue
            targets += 1
            day = decision.strftime("%Y-%m-%d")
            freshness_start_ns = (decision - pd.Timedelta(seconds=freshness_seconds)).value
            fresh_start = int(np.searchsorted(known_quote_ns, freshness_start_ns, side="left"))
            fresh_stop = int(np.searchsorted(known_quote_ns, decision.value, side="right"))
            fresh_bookmakers = known_bookmakers[fresh_start:fresh_stop]
            fresh_book_count = pd.unique(fresh_bookmakers[pd.notna(fresh_bookmakers)]).size
            if fresh_book_count < minimum_other_books + 1:
                insufficient_quotes += 1
                continue
            native_row = None
            if require_native:
                native = ngroup.loc[
                    ngroup["forecast_available"].fillna(False)
                    & ngroup["identity_verified"].fillna(False)
                    & ngroup["forecast_at"].notna()
                    & ngroup["forecast_at"].le(decision)
                    & ngroup["requested_utc_day"].notna()
                    & ngroup["requested_utc_day"].astype(str).eq(day)
                    & ngroup["daily_state_cutoff_day"].notna()
                    & ngroup["daily_state_cutoff_day"].astype(str).eq(day)
                ]
                if "feature_source_max_day" in native:
                    native = native.loc[
                        pd.to_datetime(native["feature_source_max_day"], errors="coerce")
                        .lt(pd.Timestamp(day))
                    ]
                native = native.loc[
                    np.isfinite(pd.to_numeric(native["p_base"], errors="coerce"))
                    & pd.to_numeric(native["p_base"], errors="coerce").between(
                        0.0, 1.0, inclusive="neither"
                    )
                ]
                if "daily_state_key_sha256" in native:
                    native = native.loc[native["daily_state_key_sha256"].notna()]
                    if native["daily_state_key_sha256"].nunique() > 1:
                        raise ValueError("conflicting same-day native state keys for odd UTC grid")
                if native.empty:
                    no_native += 1
                    continue
                native_row = native.sort_values("forecast_at", kind="mergesort").iloc[-1]
            from src.analysis.c0_policy_data import admit_policy_quotes
            opportunity = pd.DataFrame([{
                "opportunity_id": f"edge-odd-grid-{match_id}-{decision.value}",
                "canonical_match_id": match_id,
                "decision_at": decision,
            }])
            latest = admit_policy_quotes(
                opportunity, qgroup, ttl_seconds=freshness_seconds
            )
            if latest.empty:
                insufficient_quotes += 1
                continue
            latest["_odds_canonical_a"] = pd.to_numeric(
                latest["canonical_odds_a"], errors="coerce"
            )
            latest["_odds_canonical_b"] = pd.to_numeric(
                latest["canonical_odds_b"], errors="coerce"
            )
            if latest["bookmaker_id"].nunique() < minimum_other_books + 1:
                insufficient_quotes += 1
                continue
            inverse_sum = 1.0 / latest["_odds_canonical_a"] + 1.0 / latest["_odds_canonical_b"]
            latest["_q_a"] = (1.0 / latest["_odds_canonical_a"]) / inverse_sum
            market_a = latest.set_index("bookmaker_id")["_q_a"]
            target_id = f"edge-odd-grid-{match_id}-{decision.value}"
            native_day = (
                str(native_row["native_effective_release_day"])
                if native_row is not None else None
            )
            input_hashes = native_row.get("native_input_hashes") if native_row is not None else None
            state_key = native_row.get("daily_state_key_sha256") if native_row is not None else None
            if native_row is not None:
                strict_day_proofs.append(
                    f"{match_id}|{day}|{native_day}|"
                    f"{native_row.get('feature_source_max_day')}|{state_key}"
                )
            for focal in latest.to_dict("records"):
                other = market_a.drop(index=focal["bookmaker_id"], errors="ignore")
                if len(other) < minimum_other_books:
                    continue
                q_a = float(other.mean())
                for side, odds, market in (
                    ("a", focal["_odds_canonical_a"], q_a),
                    ("b", focal["_odds_canonical_b"], 1.0 - q_a),
                ):
                    records.append({
                        **fixture,
                        "opportunity_id": target_id,
                        "canonical_match_id": match_id,
                        "decision_at": decision,
                        "start_at": start,
                        "side": side,
                        "odds": float(odds),
                        "p_base": float(native_row["p_base"]) if native_row is not None else np.nan,
                        "prob_base": (
                            float(native_row["p_base"] if side == "a" else 1.0 - native_row["p_base"])
                            if native_row is not None else np.nan
                        ),
                        "market_prob_a": q_a,
                        "market_prob": market,
                        "prob_market_novig": market,
                        "prob_market_close_novig": market,
                        "bookmaker_id": int(focal["bookmaker_id"]),
                        "bookmaker": focal["bookmaker"],
                        "quote_id": int(focal["quote_id"]),
                        "quoted_at": focal["quoted_at"],
                        "quote_age_seconds": float((decision - focal["quoted_at"]).total_seconds()),
                        "best_of": fixture.get("best_of"),
                        "league": fixture.get("league"),
                        "is_grid": True,
                        "is_snapshot": False,
                        "synthetic_grid_projection": True,
                        "native_effective_release_day": native_day,
                        "feature_source_max_day": (
                            native_row.get("feature_source_max_day")
                            if native_row is not None else None
                        ),
                        "daily_state_cutoff_day": (
                            native_row.get("daily_state_cutoff_day")
                            if native_row is not None else None
                        ),
                        "native_input_hashes": input_hashes,
                        "strict_same_utc_day_proof": native_row is not None,
                        "distinct_other_books": int(len(other)),
                    })
    grid_events = pd.DataFrame(records)
    if grid_events.empty:
        grid_events = pd.DataFrame(
            columns=[
                "canonical_match_id", "opportunity_id", "decision_at", "start_at",
                "settle_at", "y", "partition", "side", "odds", "p_base",
                "market_prob_a", "market_prob", "prob_market_novig",
                "prob_market_close_novig", "bookmaker_id", "bookmaker", "quote_id",
                "quoted_at", "best_of", "league", "is_grid", "is_snapshot",
            ]
        )
    native_matches = set(
        states.loc[
            states["forecast_available"].fillna(False)
            & states["identity_verified"].fillna(False),
            "canonical_match_id",
        ].dropna().astype(int)
    )
    quote_matches = set(quotes["canonical_match_id"].dropna().astype(int))
    provenance = {
        "native_state_required": require_native,
        "grid": "exact UTC odd-hour minute00 second00; latest admitted raw quotes as-of clock",
        "target_clocks": targets,
        "synthetic_event_rows": len(grid_events),
        "targets_without_same_day_native_state": no_native if require_native else None,
        "targets_without_three_fresh_identity_aligned_books": insufficient_quotes,
        "raw_quote_rows": len(quotes),
        "raw_quote_fixture_count": len(quote_matches),
        "native_daily_state_fixture_count": len(native_matches),
        "market_only_fixture_count": len(quote_matches - native_matches),
        "quotes_sha256": _file_sha256(quote_path),
        "daily_native_states_sha256": _digest(states.to_dict("records")),
        "strict_day_proof_sha256": hashlib.sha256("\\n".join(sorted(strict_day_proofs)).encode()).hexdigest(),
        "cross_utc_day_native_fill": False,
        "native_inference_rerun": False,
    }
    return grid_events, provenance


if __name__ == "__main__":
    raise SystemExit(main())
