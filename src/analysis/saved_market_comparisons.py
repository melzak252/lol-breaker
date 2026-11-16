"""Pure scoring for saved OddsPortal and Pinnacle comparison ledgers.

Losses are evaluated without clipping or refitting. Scores use one row per
fixture; paired intervals resample UTC calendar-month blocks.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.metrics import binary_log_loss_vector
from src.analysis.production_market_benchmark import paired_cluster_interval

ODDSPORTAL_ENDPOINTS = ("OPEN_OR_FIRST_OBSERVED", "LAST_CHANGE_OR_LAST_PRESTART")
PINNACLE_ENDPOINTS = ODDSPORTAL_ENDPOINTS


def _required(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _flags(values: pd.Series, name: str) -> np.ndarray:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, (int, np.integer)) and value in (0, 1):
            result.append(bool(value))
        else:
            raise ValueError(f"{name} must contain explicit booleans")
    return np.asarray(result, dtype=bool)


def _aware(values: pd.Series, name: str) -> pd.Series:
    parsed = []
    for value in values:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} has invalid timestamp") from exc
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
        parsed.append(stamp.tz_convert("UTC"))
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]")


def _prob(values: pd.Series, name: str) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not (np.isfinite(p) & (p > 0) & (p < 1)).all():
        raise ValueError(f"{name} must contain finite interior probabilities")
    return p


def _score_one(frame: pd.DataFrame, col: str, label: str) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0, "log_loss": None, "brier": None}
    y, p = frame[label].to_numpy(float), frame[col].to_numpy(float)
    ll = binary_log_loss_vector(y, p, epsilon=0.0)
    scores = pd.DataFrame({"gid": frame._gid.to_numpy(), "ll": ll, "brier": (p-y)**2})
    fixture = scores.groupby("gid", sort=False)[["ll", "brier"]].mean()
    return {"n": int(len(fixture)), "log_loss": float(fixture.ll.mean()), "brier": float(fixture.brier.mean())}


def _scores(frame: pd.DataFrame, model: str, market: str, label: str, extras: dict[str, str] | None = None) -> dict[str, Any]:
    result = {"sports_model": _score_one(frame, model, label), "market": _score_one(frame, market, label)}
    for name, col in (extras or {}).items():
        result[name] = _score_one(frame, col, label)
    return result


def _contrasts(frame: pd.DataFrame, model: str, market: str, label: str, month: str,
               draws: int, seed: int, confidence: float,
               oddsportal_months: Sequence[str] | None = None) -> dict[str, Any]:
    if frame.empty:
        return {"log_loss": None, "brier": None, "orientation": "sports_model_minus_market", "confidence_level": confidence}
    y, p, q = frame[label].to_numpy(float), frame[model].to_numpy(float), frame[market].to_numpy(float)
    data = pd.DataFrame({"gid": frame._gid.to_numpy(), "month": frame[month].astype(str).to_numpy(),
        "ll": binary_log_loss_vector(y, p, epsilon=0.0)-binary_log_loss_vector(y, q, epsilon=0.0),
        "brier": (p-y)**2-(q-y)**2})
    if data.groupby("gid").month.nunique().gt(1).any():
        raise ValueError("fixture has conflicting month labels")
    fixture = data.groupby("gid", sort=False).agg(ll=("ll", "mean"), brier=("brier", "mean"), month=("month", "first"))
    result = {}
    for metric in ("ll", "brier"):
        values, blocks = fixture[metric].to_numpy(float), fixture.month.to_numpy()
        key = "log_loss" if metric == "ll" else "brier"
        if oddsportal_months is not None:
            monthly = pd.DataFrame({"month": blocks, "delta": values, "n": 1.0}).groupby("month").sum()
            monthly = monthly.reindex(list(oddsportal_months), fill_value=0.0)
            result[key] = block_ratio_interval(monthly.delta, monthly.n, monthly.index, draws=draws, seed=seed)
        else:
            result[key] = paired_cluster_interval(values, blocks, draws=draws, seed=seed, confidence_level=confidence)
    return {**result, "orientation": "sports_model_minus_market", "confidence_level": confidence}


def _endpoint(frame: pd.DataFrame, model: str, market: str, label: str, month: str,
              draws: int, seed: int, coverage: dict[str, Any], confidence: float = .95,
              months: Sequence[str] | None = None, extras: dict[str, str] | None = None) -> dict[str, Any]:
    return {"scores": _scores(frame, model, market, label, extras),
            "contrasts": _contrasts(frame, model, market, label, month, draws, seed, confidence, months),
            "coverage": {**coverage, "scored_fixtures": int(frame._gid.nunique()) if len(frame) else 0}}


def _date_metadata(frame: pd.DataFrame, date_col: str = "_date") -> dict[str, Any]:
    if frame.empty:
        return {"date_min": None, "date_max": None, "months": []}
    dates = frame[date_col].dropna()
    return {"date_min": dates.min().date().isoformat(), "date_max": dates.max().date().isoformat(),
            "months": sorted(dates.dt.strftime("%Y-%m").unique().tolist())}


def summarize_oddsportal_comparisons(endpoints: pd.DataFrame, months: Sequence[str], *, draws: int = 5000, seed: int = 20261111) -> dict[str, Any]:
    """Reproduce earliest admitted focal-book quotes per fixture and endpoint."""
    _required(endpoints, ["source", "source_event_id", "golgg_match_id", "bookmaker", "horizon", "timed_price_eligible", "p_c0", "q_focal", "y", "label_status", "canonical_date", "quote_at"], "OddsPortal ledger")
    month_universe = list(map(str, months))
    if not month_universe or len(set(month_universe)) != len(month_universe):
        raise ValueError("months must be a nonempty unique month universe")
    frame = endpoints.copy()
    frame["_timed"] = _flags(frame.timed_price_eligible, "timed_price_eligible")
    mapped = frame.golgg_match_id.notna() & frame.golgg_match_id.astype(str).str.strip().ne("")
    frame["_mapped"] = mapped
    frame["_date"] = pd.to_datetime(frame.canonical_date, errors="coerce", utc=True)
    frame["_month"] = frame._date.dt.strftime("%Y-%m")
    frame["_eligible"] = frame._timed & frame.p_c0.notna() & frame.q_focal.notna()
    mapped_frame = frame.loc[mapped].copy()
    if mapped_frame.duplicated(["source", "source_event_id", "golgg_match_id", "bookmaker", "horizon"]).any():
        raise ValueError("duplicate mapped OddsPortal curated identity")
    mapped_frame["_gid"] = mapped_frame.golgg_match_id.astype(str)
    result = {}
    for horizon in ODDSPORTAL_ENDPOINTS:
        source = mapped_frame.loc[mapped_frame.horizon.eq(horizon)].copy()
        # Fixture-level label and canonical-month consistency precedes selection.
        for col, label in (("y", "label"), ("_month", "canonical month")):
            relevant = source.loc[source[col].notna()]
            if relevant.groupby("_gid")[col].nunique().gt(1).any():
                raise ValueError(f"conflicting OddsPortal fixture {label}")
        pool = source.loc[source._eligible].copy()
        if len(pool):
            pool["_quote"] = _aware(pool.quote_at, "quote_at")
            pool = pool.sort_values(["_quote", "bookmaker", "source_event_id"], kind="mergesort").drop_duplicates("_gid", keep="first")
        known_label = pd.to_numeric(pool.y, errors="coerce")
        scored = pool.loc[pool.label_status.eq("known") & known_label.isin([0, 1])].copy()
        if len(scored):
            if scored._month.isna().any():
                raise ValueError("selected OddsPortal fixture has invalid canonical date")
            scored["_label"] = pd.to_numeric(scored.y).astype(float)
            scored["p"] = _prob(scored.p_c0, "p_c0")
            scored["q"] = _prob(scored.q_focal, "q_focal")
            if not set(scored._month).issubset(set(month_universe)):
                raise ValueError("OddsPortal scored month outside supplied universe")
        selected = int(len(pool))
        coverage = {"source_records": int(frame.horizon.eq(horizon).sum()),
                    "mapped_records": int(len(source)),
                    "eligible_records": int(source._eligible.sum()),
                    "selected_price_fixtures": selected,
                    "scored_fixtures": int(scored._gid.nunique()) if len(scored) else 0,
                    "absences": {"missing_canonical_mapping": int((frame.horizon.eq(horizon) & ~frame._mapped).sum()),
                                 "not_timed_or_missing_forecast_or_price": int((~source._eligible).sum()),
                                 "unknown_or_invalid_outcome": int(selected-len(scored))},
                    **_date_metadata(scored)}
        result[horizon] = _endpoint(scored, "p", "q", "_label", "_month", draws, seed, coverage, months=month_universe)
    return {"endpoints": result, "metadata": {"source": "saved OddsPortal annual endpoint ledger", "month_count": len(month_universe), "period_months": month_universe,
            "units": "one earliest eligible focal quote per fixture; fixture-weighted mean loss; supplied-month ratio bootstrap",
            "draws": draws, "seed": seed, "protocol": "source-oriented p_c0 versus q_focal; exact unclipped log loss; no refitting"}}


def summarize_pinnacle_comparisons(ledger: pd.DataFrame, *, draws: int = 5000, seed: int = 20261112) -> dict[str, Any]:
    """Score canonical-aligned C20R/Pinnacle and saved secondary pair cohorts."""
    _required(ledger, ["horizon", "source_event_id", "golgg_match_id", "canonical_gid", "canonical_y", "canonical_date", "month", "p_c20r", "p_pinnacle", "label_available", "c20r_pair_eligible", "prestart_eligible", "endpoint_decision_at", "earliest_retained_start_at", "scheduled_start_at", "p_polish_consensus", "polish_pair_eligible", "three_way_eligible"], "Pinnacle ledger")
    frame = ledger.copy()
    for flag in ("label_available", "c20r_pair_eligible", "prestart_eligible", "polish_pair_eligible", "three_way_eligible"):
        frame["_"+flag] = _flags(frame[flag], flag)
    frame["_gid"] = frame.golgg_match_id.astype("string")
    frame["_month"] = frame.month.astype("string")
    frame["_y"] = pd.to_numeric(frame.canonical_y, errors="coerce")
    frame["_date"] = pd.to_datetime(frame.canonical_date, errors="coerce", utc=True)
    out, secondary = {}, {}
    for horizon in PINNACLE_ENDPOINTS:
        subset = frame.loc[frame.horizon.eq(horizon)].copy()
        source_fixtures = int(subset.golgg_match_id.nunique(dropna=True))
        primary = subset.loc[subset._c20r_pair_eligible].copy()
        if len(primary):
            if primary.golgg_match_id.isna().any() or primary.canonical_gid.isna().any() or primary._date.isna().any() or primary._month.isna().any():
                raise ValueError("admitted Pinnacle pair missing canonical identity/date")
            if not primary._label_available.all() or not primary._prestart_eligible.all() or not primary._y.isin([0, 1]).all():
                raise ValueError("admitted Pinnacle pair lacks a known binary label or prestart flag")
            if primary.duplicated("_gid").any():
                raise ValueError("duplicate admitted Pinnacle fixture")
            if _aware(primary.scheduled_start_at, "scheduled_start_at").dt.strftime("%Y-%m").ne(primary._month.astype(str)).any():
                raise ValueError("Pinnacle month conflicts with scheduled UTC start")
            if not primary.source_event_id.notna().all():
                raise ValueError("admitted Pinnacle row missing source event identity")
            decision, start = _aware(primary.endpoint_decision_at, "endpoint_decision_at"), _aware(primary.earliest_retained_start_at, "earliest_retained_start_at")
            if not (decision < start).all():
                raise ValueError("admitted Pinnacle endpoint is not strictly prestart")
            primary["_label"] = primary._y.astype(float)
            primary["p"] = _prob(primary.p_c20r, "p_c20r")
            primary["q"] = _prob(primary.p_pinnacle, "p_pinnacle")
        coverage = {"source_records": int(len(subset)), "source_fixtures": int(subset.source_event_id.nunique()),
                    "mapped_unique_fixtures": source_fixtures,
                    "admitted_records": int(len(primary)), "absences": {"not_c20r_pair_eligible": int((~subset._c20r_pair_eligible).sum())}, **_date_metadata(primary)}
        out[horizon] = _endpoint(primary, "p", "q", "_label", "_month", draws, seed, coverage)
        out[horizon]["contrasts"]["familywise_98_75_log_loss"] = _contrasts(primary, "p", "q", "_label", "_month", draws, seed, .9875)["log_loss"] if len(primary) else None
        scopes = {}
        for name, flag, model_col, market_col, extras, benchmark, model_label in (
            ("polish_vs_pinnacle", "_polish_pair_eligible", "p_polish_consensus", "p_pinnacle", {}, "Pinnacle", "Polish consensus"),
            ("three_way", "_three_way_eligible", "p_c20r", "p_polish_consensus", {"pinnacle": "p_pinnacle"}, "Polish consensus", "C20R")):
            cohort = subset.loc[subset[flag]].copy()
            if len(cohort):
                if cohort.golgg_match_id.isna().any() or cohort.canonical_gid.isna().any() or cohort._date.isna().any() or cohort._month.isna().any():
                    raise ValueError(f"admitted {name} row lacks canonical identity/date")
                if not cohort._label_available.all() or not cohort._prestart_eligible.all() or not cohort._y.isin([0, 1]).all():
                    raise ValueError(f"admitted {name} row lacks known binary label or prestart flag")
                if cohort.duplicated("_gid").any():
                    raise ValueError(f"duplicate identity in {name}")
                if _aware(cohort.scheduled_start_at, "scheduled_start_at").dt.strftime("%Y-%m").ne(cohort._month.astype(str)).any():
                    raise ValueError(f"inconsistent scheduled-start month in {name}")
                decision, start = _aware(cohort.endpoint_decision_at, "endpoint_decision_at"), _aware(cohort.earliest_retained_start_at, "earliest_retained_start_at")
                if not (decision < start).all():
                    raise ValueError(f"admitted {name} endpoint is not strictly prestart")
                cohort["_label"] = cohort._y.astype(float)
                cohort["p"] = _prob(cohort[model_col], model_col)
                cohort["q"] = _prob(cohort[market_col], market_col)
                if extras:
                    for key, col in extras.items():
                        cohort[col] = _prob(cohort[col], col)
            scopes[name] = _endpoint(cohort, "p", "q", "_label", "_month", draws, seed,
                {"source_records": int(len(subset)), "admitted_records": int(len(cohort)),
                 "absences": {"not_in_source_flag_cohort": int((~subset[flag]).sum())}, **_date_metadata(cohort)}, extras=extras)
            scopes[name]["benchmark"] = benchmark
            scopes[name]["sports_model_label"] = model_label
        secondary[horizon] = scopes
    return {"endpoints": out, "secondary_scopes": secondary,
            "metadata": {"source": "saved canonical-oriented Pinnacle comparison ledger", "units": "fixture-weighted mean loss; paired scheduled-start UTC-month bootstrap",
                         "draws": draws, "seed": seed, "familywise_confidence": .9875,
                         "protocol": "identical source-flag admitted rows per pair; no clipping, reprojection or refitting"}}
