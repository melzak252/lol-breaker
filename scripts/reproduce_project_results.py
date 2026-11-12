#!/usr/bin/env python3
"""Replay project-report scores and flat-stake diagnostics from installed ledgers.

No fitting, model inference, database access, or network access is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from betting_app.core.ev import expected_value
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.long_term_market_evaluation import (
    _month_universe,
    select_fixed_policy,
    summarize_unit_returns,
)
from src.analysis.metrics import binary_log_loss_vector, calculate_ece
from src.analysis.production_market_benchmark import paired_cluster_interval

ROOT = Path(__file__).resolve().parents[1]
YEAR_RANGE = tuple(range(2021, 2027))
DRAWS = 5000
SEED = 20261111
MODEL_RUN = "c0_model_improvement_20261026_01"
FINANCE_RUN = "fresh20_oddsportal_20261111_02"
FIRST = "OPEN_OR_FIRST_OBSERVED"
CONSENSUS_SEED = 20261023
C20R_VERSION = "c0-native-fresh20-2026-10-21-v1-regional-v1"
LEAD_BINS = ("[0,2)", "[2,6)", "[6,12)", "[12,24)", "[24,48)", "[48,96)", "[96,inf)")

PATHS = {
    "model_predictions": "data/07_model_output/c0_model_improvement/c0_model_improvement_20261026_01/predictions.parquet",
    "model_protocol": "data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/protocol.json",
    "model_binding": "data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/execution_binding.json",
    "model_verification": "data/08_reporting/c0_model_improvement/c0_model_improvement_20261026_01/sealed_verification.json",
    "serving_config": "conf/base/c0_serving.json",
    "odds_protocol": "data/05_model_input/fresh20_oddsportal/fresh20_oddsportal_20261111_02/protocol.json",
    "odds_evaluation": "data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/oddsportal_annual_endpoint_ledger.parquet",
    "odds_binding": "data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/execution_binding.json",
    "events": "data/04_feature/long_term_market_edge/long_term_market_edge_20261022_01/oddsportal_identity_complete_v3/events.parquet",
    "loo_panel": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_matched_panel.parquet",
    "loo_case_accounting": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_case_accounting.parquet",
    "loo_scores_reference": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_scores.csv",
    "loo_contrasts_reference": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_paired_contrasts.csv",
    "loo_verification": "data/08_reporting/production_probability/production_probability_20261023_02/verification/sealed_verification.json",
    "glicko_predictions": "data/artifacts/corrected-ratings-logloss-20260908-full/holdout_predictions.csv",
    "glicko_summary": "data/artifacts/corrected-ratings-logloss-20260908-full/summary.json",
    "glicko_verification": "data/artifacts/corrected-ratings-logloss-20260908-full/verification.json",
    "production_market_source": "src/analysis/production_market_benchmark.py",
    "selection_source": "src/analysis/long_term_market_evaluation.py",
    "score_source": "src/analysis/metrics.py",
    "bootstrap_source": "src/analysis/bootstrap.py",
    "ev_source": "betting_app/core/ev.py",
    "engine_source": "betting_app/core/models/engine.py",
    "guards_source": "src/analysis/betting_guards_a_to_d.py",
    "probability_metrics_source": "src/analysis/probability_metrics.py",
    "devig_source": "src/analysis/devig_comparison.py",
    "audit_serialization_source": "src/analysis/c0_edge_alpha_audit.py",
    "this_script": "scripts/reproduce_project_results.py",
}


def select_first_policy(panel: pd.DataFrame, *, alpha: float, tax: float) -> pd.DataFrame:
    """Apply the existing financial selection rules at one declared tax rate."""
    prepared = panel.loc[panel.timed_price_eligible.fillna(False)].copy()
    prepared["q"] = prepared["q_focal"]
    selected = select_fixed_policy(
        prepared,
        sports_alpha=alpha,
        tax_rate=tax,
        epsilon=0.025,
        minimum_ev=0.05,
        maximum_odds=4.0,
        timed=True,
    )
    return selected.reset_index(drop=True)


def score_probability(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(p)
    y, p = y[valid].astype(int), p[valid].astype(float)
    if not len(y):
        return {"n": 0, "log_loss": None, "brier": None, "ece10": None, "accuracy": None, "auc": None}
    if not ((p > 0) & (p < 1)).all():
        raise ValueError("all scored probabilities must be strictly between 0 and 1")
    return {
        "n": int(len(y)),
        "log_loss": float(np.mean(binary_log_loss_vector(y, p, epsilon=0.0))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece10": float(calculate_ece(y, p, n_bins=10)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "auc": float(roc_auc_score(y, p)),
    }


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], scope: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{scope}: missing columns {missing}")


def _unique_identities(frame: pd.DataFrame, columns: Sequence[str], scope: str) -> None:
    _require_columns(frame, columns, scope)
    if frame[list(columns)].isna().any().any() or frame[list(columns)].astype(str).eq("").any().any():
        raise ValueError(f"{scope}: missing identity")
    if frame[list(columns)].astype(str).duplicated().any():
        raise ValueError(f"{scope}: duplicate curated identities")


def _validate_probabilities(frame: pd.DataFrame, label: str, columns: Sequence[str]) -> None:
    _require_columns(frame, [label, *columns], "probability ledger")
    if frame.empty or not frame[label].isin([0, 1]).all():
        raise ValueError("probability ledger requires nonempty binary labels")
    values = frame[list(columns)].to_numpy(dtype=float)
    if not (np.isfinite(values) & (values > 0) & (values < 1)).all():
        raise ValueError("saved probabilities must be finite and strictly interior")


def _aware_clock(value: Any, name: str, *, allow_missing: bool = False) -> pd.Timestamp:
    if pd.isna(value):
        if allow_missing:
            return pd.NaT
        raise ValueError(f"{name}: missing clock")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: malformed clock {value!r}") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name}: timezone-aware clock required: {value!r}")
    return stamp.tz_convert("UTC")


def _paired_loss_contrast(candidate: pd.DataFrame, base: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    joined = candidate.merge(base, on="unit_id", how="outer", validate="one_to_one",
                             suffixes=("_candidate", "_base"), indicator=True)
    if not joined["_merge"].eq("both").all() or not joined.month_candidate.eq(joined.month_base).all():
        raise ValueError("paired loss identities/months differ")
    result = {}
    for metric in ("log_loss", "brier"):
        delta = joined[f"{metric}_candidate"].to_numpy(float) - joined[f"{metric}_base"].to_numpy(float)
        if not np.isfinite(delta).all() or joined.month_candidate.isna().any():
            raise ValueError("nonfinite paired differences or missing month; cohort must not shrink")
        interval = paired_cluster_interval(delta, joined.month_candidate, draws=DRAWS, seed=seed,
                                           confidence_level=0.95)
        result[metric] = {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in interval.items()
        }
    return result


def summarize_consensus_panel(panel: pd.DataFrame, cases: pd.DataFrame) -> dict[str, Any]:
    """Score saved LOO logits: quote losses -> equal books -> equal fixtures."""
    identity = ["canonical_match_id", "bookmaker", "snapshot_id"]
    proof_fields = ("contributors", "contributor_clocks", "ages_seconds",
                    "contributor_weights", "contributor_fair_probabilities")
    _require_columns(panel, ["clock", "month", "lead_bin", "n_contributors", *proof_fields], "LOO panel")
    _unique_identities(panel, identity, "LOO panel")
    _validate_probabilities(panel, "y", ["p_model", "p_market"])
    if not panel.month.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").fillna(False).all():
        raise ValueError("LOO panel has invalid scheduled-start months")
    if not panel.lead_bin.isin(LEAD_BINS).all():
        raise ValueError("LOO panel has an unknown scheduled-start lead bin")
    if panel.groupby("canonical_match_id")[["y", "month"]].nunique(dropna=False).ne(1).any().any():
        raise ValueError("conflicting per-fixture labels/months")
    for row in panel.to_dict("records"):
        decoded = {}
        for field in proof_fields:
            try:
                value = json.loads(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field}: malformed contributor JSON") from error
            if not isinstance(value, list) or len(value) != row["n_contributors"]:
                raise ValueError(f"{field}: contributor list length differs from n_contributors")
            decoded[field] = value
        for field in ("ages_seconds", "contributor_weights", "contributor_fair_probabilities"):
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in decoded[field]):
                raise ValueError(f"{field}: expected a flat list of numbers")
        contributors = decoded["contributors"]
        n = len(contributors)
        if (n < 2 or not all(isinstance(book, str) and book for book in contributors) or
                len(set(contributors)) != n or row["bookmaker"] in contributors):
            raise ValueError("consensus requires two distinct nonfocal contributors")
        clock = _aware_clock(row["clock"], "focal clock")
        clocks = [_aware_clock(value, "contributor clock") for value in decoded["contributor_clocks"]]
        ages = np.asarray(decoded["ages_seconds"], dtype=float)
        actual_ages = np.asarray([(clock - stamp).total_seconds() for stamp in clocks])
        weights = np.asarray(decoded["contributor_weights"], dtype=float)
        fair = np.asarray(decoded["contributor_fair_probabilities"], dtype=float)
        if (not np.isfinite(ages).all() or not ((ages >= 0) & (ages <= 300)).all() or
                not np.allclose(ages, actual_ages, rtol=0, atol=1e-6)):
            raise ValueError("future, stale or inconsistent contributor ages")
        if not np.allclose(weights, 1 / n, rtol=0, atol=1e-12):
            raise ValueError("saved consensus must use equal contributor logit weights")
        if not (np.isfinite(fair) & (fair > 0) & (fair < 1)).all():
            raise ValueError("invalid contributor fair probabilities")
        if not np.isclose(expit(np.mean(logit(fair))), row["p_market"], rtol=0, atol=1e-12):
            raise ValueError("saved consensus differs from mean other-book fair logits")

    accounting = cases.rename(columns={"focal_bookmaker": "bookmaker"})
    _unique_identities(accounting, identity, "LOO accounting")
    _require_columns(accounting, ["status", "absence_reason"], "LOO accounting")
    if not accounting.status.isin(["matched", "excluded"]).all():
        raise ValueError("LOO accounting has unknown status")
    excluded = accounting.loc[accounting.status.eq("excluded")]
    if excluded.absence_reason.isna().any() or excluded.absence_reason.eq("").any():
        raise ValueError("excluded LOO cases require an explicit absence reason")
    matched = accounting.loc[accounting.status.eq("matched"), identity]
    paired = panel[identity].merge(matched, on=identity, how="outer", indicator=True, validate="one_to_one")
    if not paired["_merge"].eq("both").all():
        raise ValueError("missing/extra matched accounting identities")

    probabilities = {
        "C20R": panel.p_model.to_numpy(float),
        "consensus": panel.p_market.to_numpy(float),
        "Hybrid-50": expit(0.5 * logit(panel.p_model.to_numpy(float)) +
                           0.5 * logit(panel.p_market.to_numpy(float))),
    }
    profiles = {}
    for profile in ("ALL", *LEAD_BINS):
        mask = np.ones(len(panel), dtype=bool) if profile == "ALL" else panel.lead_bin.eq(profile).to_numpy()
        part = panel.loc[mask]
        if part.empty:
            continue
        fixture_losses, scores = {}, {}
        for recipe, p in probabilities.items():
            y = part.y.to_numpy(float)
            scored = part[["canonical_match_id", "bookmaker", "month"]].copy()
            scored["log_loss"] = binary_log_loss_vector(y, p[mask], epsilon=0.0)
            scored["brier"] = (p[mask] - y) ** 2
            by_book = scored.groupby(["canonical_match_id", "bookmaker"], as_index=False, sort=False).agg(
                log_loss=("log_loss", "mean"), brier=("brier", "mean"), month=("month", "first"))
            fixture = by_book.groupby("canonical_match_id", as_index=False, sort=False).agg(
                log_loss=("log_loss", "mean"), brier=("brier", "mean"), month=("month", "first"))
            fixture_losses[recipe] = fixture.rename(columns={"canonical_match_id": "unit_id"})
            scores[recipe] = {metric: float(fixture[metric].mean()) for metric in ("log_loss", "brier")}
        contrasts = {
            f"{candidate}_minus_{base}": _paired_loss_contrast(fixture_losses[candidate], fixture_losses[base],
                                                             seed=CONSENSUS_SEED)
            for candidate, base in (("C20R", "consensus"), ("Hybrid-50", "consensus"), ("Hybrid-50", "C20R"))
        }
        profiles[profile] = {
            "scores": scores, "contrasts": contrasts,
            "coverage": {"matched_cases": len(part), "fixture_book_cells": len(by_book), "fixtures": len(fixture),
                         "scheduled_start_months": sorted(part.month.unique())},
        }
    overall = profiles.pop("ALL")
    overall["coverage"].update(
        focal_cases=len(cases), focal_fixtures=int(cases.canonical_match_id.nunique()),
        excluded_cases=len(excluded), excluded_reasons=excluded.absence_reason.value_counts().to_dict(),
    )
    return {
        **overall, "by_lead_bin": profiles,
        "weighting": "mean quote loss within fixture/book; equal books within fixture; equal fixtures",
        "market_reference": "equal mean other-book fair logits; focal book excluded; at least two fresh books",
        "interval_scope": "nominal historical 95%; fixture-weighted and equal-month; not adaptive-search corrected",
    }


def summarize_glicko_pair(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compare the saved, already BO-projected player/organization probabilities."""
    columns = [f"{entity}_gl_{kind}" for entity in ("player", "team") for kind in ("raw", "calibrated")]
    _require_columns(predictions, ["date", "BoN", "team1_id", "team2_id"], "Glicko-2 ledger")
    _unique_identities(predictions, ["golgg_match_id"], "Glicko-2 ledger")
    _validate_probabilities(predictions, "y_true", columns)
    if not predictions.BoN.isin([1, 3, 5]).all():
        raise ValueError("Glicko-2 binary scoring supports only saved BO1/3/5 rows")
    teams = predictions[["team1_id", "team2_id"]]
    if (teams.isna().any().any() or teams.astype(str).eq("").any().any() or
            teams.team1_id.astype(str).eq(teams.team2_id.astype(str)).any()):
        raise ValueError("Glicko-2 series require two distinct team IDs")
    dates = pd.to_datetime(predictions.date, errors="raise", utc=True, format="mixed")
    if dates.isna().any():
        raise ValueError("Glicko-2 ledger has a missing match date")
    months = dates.dt.strftime("%Y-%m")
    y = predictions.y_true.to_numpy(float)
    scores, contrasts = {}, {}
    for kind in ("raw", "calibrated"):
        scores[kind], losses = {}, {}
        for entity, name in (("player", "player"), ("team", "organization")):
            p = predictions[f"{entity}_gl_{kind}"].to_numpy(float)
            scores[kind][name] = score_probability(y, p)
            losses[name] = pd.DataFrame({
                "unit_id": predictions.golgg_match_id, "month": months,
                "log_loss": binary_log_loss_vector(y, p, epsilon=0.0), "brier": (p - y) ** 2,
            })
        contrasts[kind] = _paired_loss_contrast(losses["player"], losses["organization"], seed=SEED)
    return {
        "scores": scores, "contrasts": contrasts,
        "cohort": {"series": len(predictions), "first_date": dates.min().strftime("%Y-%m-%d"),
                   "last_date": dates.max().strftime("%Y-%m-%d"), "months": sorted(months.unique()),
                   "missing_four_column_predictions": 0, "unit": "equal series on identical saved rows"},
        "source_protocol": {
            "engine": "Glicko-2; corrected ratings-only day-batch replay",
            "update": "freeze entire day before results; actual-date weekly decay; paired pre-map opponent states; actual map participants",
            "probabilities": "saved independent-map BO1/3/5 series projection, used once without new clipping",
            "calibration": "separate positive zero-intercept player/organization slopes from preceding calendar year",
            "availability": "observed first-map rosters and day-only clocks; retrospective scenarios, not authenticated issuance",
        },
        "interval_scope": "player minus organization; nominal historical 95%; series-weighted and equal-month",
    }


def _check_reference_value(actual: float, expected: float, scope: str) -> float:
    difference = abs(float(actual) - float(expected))
    if not np.isfinite(difference) or difference > 1e-12:
        raise ValueError(f"saved-reference discrepancy in {scope}: replay={actual}, saved={expected}")
    return difference


def _check_consensus_references(result: dict[str, Any], scores: pd.DataFrame,
                                contrasts: pd.DataFrame) -> dict[str, Any]:
    scores = scores.loc[scores.cohort.eq("loo_consensus_conditional")]
    contrasts = contrasts.loc[contrasts.cohort.eq("loo_consensus_conditional")]
    _unique_identities(scores, ["lead_bin", "recipe"], "saved consensus scores")
    _unique_identities(contrasts, ["lead_bin", "alpha", "comparator", "metric"], "saved consensus contrasts")
    errors = []
    for profile, replayed in {"ALL": result, **result["by_lead_bin"]}.items():
        for recipe, saved_name in (("C20R", "model_only"), ("consensus", "market_only"), ("Hybrid-50", "alpha_0.5")):
            reference = scores.loc[scores.lead_bin.eq(profile) & scores.recipe.eq(saved_name)]
            if len(reference) != 1:
                raise ValueError(f"missing consensus score reference: {profile}/{saved_name}")
            row = reference.iloc[0]
            for metric in ("log_loss", "brier"):
                errors.append(_check_reference_value(replayed["scores"][recipe][metric], row[f"mean_{metric}"],
                                                      f"consensus/{profile}/{recipe}/{metric}"))
            for key, column in (("matched_cases", "quotes_scored"), ("fixture_book_cells", "fixture_book_cells"),
                                ("fixtures", "fixtures")):
                if replayed["coverage"][key] != row[column]:
                    raise ValueError(f"consensus reference coverage differs: {profile}/{key}")
        for candidate, base, alpha, comparator in (("C20R", "consensus", 1.0, "market_only"),
                                                   ("Hybrid-50", "consensus", 0.5, "market_only"),
                                                   ("Hybrid-50", "C20R", 0.5, "model_only")):
            for metric in ("log_loss", "brier"):
                reference = contrasts.loc[contrasts.lead_bin.eq(profile) & contrasts.alpha.eq(alpha) &
                                          contrasts.comparator.eq(comparator) & contrasts.metric.eq(metric)]
                if len(reference) != 1:
                    raise ValueError(f"missing consensus contrast reference: {profile}/{candidate}/{base}/{metric}")
                interval = replayed["contrasts"][f"{candidate}_minus_{base}"][metric]
                for key in ("n", "blocks", "draws", "seed", "delta", "ci_lower", "ci_upper",
                            "equal_block_delta", "equal_block_ci_lower", "equal_block_ci_upper"):
                    errors.append(_check_reference_value(interval[key], reference.iloc[0][key],
                                                          f"consensus/{profile}/{candidate}-{base}/{metric}/{key}"))
    return {"status": "PASS", "absolute_tolerance": 1e-12, "numeric_comparisons": len(errors),
            "maximum_absolute_error": max(errors)}


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recompute_fixed_tax(selected: pd.DataFrame, tax: float) -> pd.DataFrame:
    result = selected.copy()
    result["tax_rate"] = tax
    result["ev_net"] = [
        expected_value(float(probability), float(odds), tax)
        for probability, odds in zip(result["p_selected"], result["odds"])
    ]
    return result


def _known_labels(frame: pd.DataFrame) -> pd.Series:
    return frame.y.isin([0, 1]) & frame.label_status.eq("known")


def _financial_summary(selected: pd.DataFrame, months: Sequence[str]) -> dict[str, Any]:
    full = summarize_unit_returns(selected, months, draws=DRAWS, seed=SEED, stake=1.0)
    result = {key: value for key, value in full.items() if key != "monthly_totals"}
    observed = selected.loc[_known_labels(selected)]
    near = observed.loc[observed.ev_net.gt(0.05) & observed.ev_net.le(0.10)]
    result["mean_expected_ev_known_labels"] = float(observed.ev_net.mean()) if len(observed) else None
    result["mean_decimal_odds_known_labels"] = float(observed.odds.mean()) if len(observed) else None
    result["near_mean_expected_ev_known_labels"] = float(near.ev_net.mean()) if len(near) else None
    if len(near):
        win = np.where(near.side.eq("a"), near.y, 1 - near.y)
        result["near_realized_yield"] = float(np.mean([
            expected_value(int(w), float(o), float(t)) for w, o, t in zip(win, near.odds, near.tax_rate)
        ]))
    else:
        result["near_realized_yield"] = None
    if len(observed):
        sports = np.where(observed.side.eq("a"), observed.p_c0, 1 - observed.p_c0)
        result["same_ticket_sports_only_mean_ev_known_labels"] = float(np.mean(
            sports * observed.odds.to_numpy(float) * (1 - observed.tax_rate.to_numpy(float)) - 1))
    else:
        result["same_ticket_sports_only_mean_ev_known_labels"] = None
    if int(full.get("unknown_labels") or 0) > 0:
        result["return_interval_scope"] = "known-label-only diagnostic; unresolved selected labels excluded, not whole-portfolio ROI"
    else:
        result["return_interval_scope"] = "all selected outcomes known; interval covers the full selected cohort"
    return result


def pair_later_quotes(selected: pd.DataFrame, endpoint_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach one actual book's later endpoint to each selected entry, including absences."""
    keys = ["source", "source_event_id", "golgg_match_id", "bookmaker"]
    endpoint_columns = ["quote_at", "quote_at_a", "quote_at_b", "source_start_at", "odds_a", "odds_b",
                        "canonical_team_a_id", "canonical_team_b_id", "timed_price_eligible", "timed_price_absence"]
    computed = ["later_quote_pair_status", "selected_side_later_odds", "later_selected_novig_probability",
                "raw_price_movement_proxy", "net_no_vig_later_price_proxy", "later_quote_lead_hours",
                "genuine_clv_status"]
    if selected.empty:
        return pd.DataFrame(columns=[*selected.columns, *[f"{column}_later" for column in endpoint_columns], *computed])
    _require_columns(selected, [*keys, *endpoint_columns, "side", "odds", "tax_rate"], "selected entries")
    _unique_identities(selected, keys, "selected entries")
    _require_columns(endpoint_panel, [*keys, *endpoint_columns, "horizon"], "endpoint ledger")
    later = endpoint_panel.loc[endpoint_panel.horizon.eq("LAST_CHANGE_OR_LAST_PRESTART") &
                               endpoint_panel.golgg_match_id.notna(), [*keys, *endpoint_columns]].copy()
    _unique_identities(later, keys, "later endpoints")
    entries = selected.copy()
    for key in keys:
        entries[key], later[key] = entries[key].astype(str), later[key].astype(str)
    pairs = entries.merge(later, on=keys, how="left", suffixes=("", "_later"), validate="one_to_one")
    if not pairs.side.isin(["a", "b"]).all():
        raise ValueError("selected entry has an invalid side")
    tax = pairs.tax_rate.to_numpy(float)
    if not (np.isfinite(tax) & (tax >= 0) & (tax < 1)).all():
        raise ValueError("selected entry tax must be a finite fraction in [0,1)")

    def valid_price(row, suffix, clock):
        flag = row[f"timed_price_eligible{suffix}"]
        eligible = flag is True or (isinstance(flag, np.bool_) and bool(flag))
        prices = np.asarray([row[f"odds_a{suffix}"], row[f"odds_b{suffix}"]], dtype=float)
        a = _aware_clock(row[f"quote_at_a{suffix}"], f"side-a{suffix}", allow_missing=True)
        b = _aware_clock(row[f"quote_at_b{suffix}"], f"side-b{suffix}", allow_missing=True)
        fresh = (pd.notna(a) and pd.notna(b) and pd.notna(clock) and max(a, b) == clock and
                 all(0 <= (clock - stamp).total_seconds() <= 300 for stamp in (a, b)))
        return eligible and (np.isfinite(prices) & (prices > 1)).all() and fresh

    statuses, leads = [], []
    for row in pairs.to_dict("records"):
        entry_at = _aware_clock(row["quote_at"], "entry quote")
        later_at = _aware_clock(row["quote_at_later"], "later quote", allow_missing=True)
        start_values = [_aware_clock(row[name], name, allow_missing=True)
                        for name in ("source_start_at", "source_start_at_later")]
        starts = [stamp for stamp in start_values if pd.notna(stamp)]
        start = min(starts) if starts else pd.NaT
        entry_valid = valid_price(row, "", entry_at)
        later_valid = valid_price(row, "_later", later_at)
        entry_odds = float(row["odds"])
        if not np.isfinite(entry_odds) or not np.isclose(entry_odds, row[f"odds_{row['side']}"], rtol=0, atol=1e-12):
            raise ValueError("selected-side entry odds differ from the retained two-side quote")
        identity = all(pd.notna(row[name]) and pd.notna(row[f"{name}_later"]) and
                       row[name] == row[f"{name}_later"]
                       for name in ("canonical_team_a_id", "canonical_team_b_id"))
        if pd.isna(later_at):
            status = "no_retained_later_quote"
        elif not identity:
            status = "canonical_team_mismatch"
        elif later_at == entry_at:
            status = "same_clock_not_later"
        elif later_at < entry_at:
            status = "earlier_than_entry"
        elif pd.isna(start):
            status = "no_recorded_start"
        elif entry_at >= start or later_at >= start:
            status = "at_or_after_earliest_known_start"
        elif not entry_valid:
            status = "entry_quote_fails_timed_price_rules"
        elif not later_valid:
            status = "later_quote_fails_timed_price_rules"
        else:
            status = "matched_later_prestart_scenario"
        statuses.append(status)
        leads.append((start - later_at).total_seconds() / 3600 if pd.notna(start) and pd.notna(later_at) else np.nan)
    pairs["later_quote_pair_status"] = statuses
    pairs["later_quote_lead_hours"] = leads
    valid = pairs.later_quote_pair_status.eq("matched_later_prestart_scenario")
    pairs["selected_side_later_odds"] = np.where(pairs.side.eq("a"), pairs.odds_a_later, pairs.odds_b_later)
    for column in ("later_selected_novig_probability", "raw_price_movement_proxy", "net_no_vig_later_price_proxy"):
        pairs[column] = np.nan
    matched = pairs.loc[valid]
    fair_a = (1 / matched.odds_a_later) / (1 / matched.odds_a_later + 1 / matched.odds_b_later)
    q = np.where(matched.side.eq("a"), fair_a, 1 - fair_a)
    pairs.loc[valid, "later_selected_novig_probability"] = q
    pairs.loc[valid, "raw_price_movement_proxy"] = matched.odds / matched.selected_side_later_odds - 1
    pairs.loc[valid, "net_no_vig_later_price_proxy"] = (1 - matched.tax_rate) * matched.odds * q - 1
    pairs["genuine_clv_status"] = "not_available_no_certified_close_or_fill"
    return pairs


def _finance_mean_interval(frame: pd.DataFrame, column: str, months: Sequence[str]) -> dict[str, Any]:
    labels = pd.to_datetime(frame.canonical_date, errors="raise").dt.strftime("%Y-%m")
    values = frame[column].to_numpy(float)
    if not np.isfinite(values).all() or labels.isna().any() or not set(labels).issubset(months):
        raise ValueError("price statistics require finite values and dates within the finance month universe")
    totals = pd.DataFrame({"month": labels.to_numpy(), "value": values, "n": np.ones(len(frame))})
    totals = totals.groupby("month").sum().reindex(months, fill_value=0.0)
    interval = block_ratio_interval(totals.value, totals.n, totals.index, draws=DRAWS, seed=SEED)
    active = totals.loc[totals.n.gt(0)]
    interval["active_month_sensitivity"] = block_ratio_interval(
        active.value, active.n, active.index, draws=DRAWS, seed=SEED)
    return interval


def summarize_later_price_pairs(pairs: pd.DataFrame, months: Sequence[str]) -> dict[str, Any]:
    matched = pairs.loc[pairs.later_quote_pair_status.eq("matched_later_prestart_scenario")]
    known = matched.loc[_known_labels(matched)].copy()

    def statistics(frame):
        lead = frame.later_quote_lead_hours.to_numpy(float)
        return {
            "raw_price_movement": _finance_mean_interval(frame, "raw_price_movement_proxy", months),
            "net_no_vig_proxy": _finance_mean_interval(frame, "net_no_vig_later_price_proxy", months),
            "median_later_quote_lead_hours": float(np.median(lead)) if len(lead) else None,
            "within_five_minutes": int(((lead >= 0) & (lead <= 5 / 60)).sum()),
        }

    matched_known = statistics(known)
    if len(known):
        win = np.where(known.side.eq("a"), known.y, 1 - known.y)
        returns = np.asarray([expected_value(int(w), float(o), float(t))
                              for w, o, t in zip(win, known.odds, known.tax_rate)])
        ev = known.ev_net.to_numpy(float)
        proxy = known.net_no_vig_later_price_proxy.to_numpy(float)
        matched_known.update(
            mean_expected_ev=float(ev.mean()), realized_yield=float(returns.mean()),
            mean_ev_minus_proxy=float(np.mean(ev - proxy)),
            mean_proxy_minus_realized=float(np.mean(proxy - returns)),
        )
    else:
        matched_known.update(mean_expected_ev=None, realized_yield=None,
                             mean_ev_minus_proxy=None, mean_proxy_minus_realized=None)
    return {
        "total_selections": len(pairs), "known_selections": int(_known_labels(pairs).sum()),
        "matched_prices": len(matched), "matched_known_count": len(known),
        "unmatched_prices": len(pairs) - len(matched),
        "absence_counts": pairs.loc[~pairs.index.isin(matched.index), "later_quote_pair_status"].value_counts().to_dict(),
        "all_price_pairs": statistics(matched), "matched_known": matched_known,
        "genuine_clv": None,
        "genuine_clv_status": "not_available_no_certified_close_or_fill",
        "scope": "same selected entry/book/side; strictly later recorded pre-start two-side quote; not certified close or fill",
        "interval_scope": "unit-stake ratio on all finance preselection months; active-month sensitivity separate",
    }


def _check_inputs() -> dict[str, Path]:
    found = {key: ROOT / rel for key, rel in PATHS.items()}
    missing = [f"{key}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
               for key, path in found.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing replay prerequisites (install the bound local artifacts; no fallback is used):\n  "
                                + "\n  ".join(missing))
    return found


def _bound_hash(binding: dict[str, Any], relative_path: str) -> str:
    matches = [
        digest for name, digest in binding.items()
        if str(name).replace("\\", "/").endswith("/" + relative_path)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one bound hash for {relative_path}; found {len(matches)}")
    return str(matches[0])


def _check_additional_bindings(paths: dict[str, Path]) -> dict[str, Any]:
    consensus = json.loads(paths["loo_verification"].read_text(encoding="utf-8"))
    if (consensus.get("model") != C20R_VERSION or
            consensus.get("status") != "COMPLETED_HISTORICAL_DEVELOPMENT_DIAGNOSTIC"):
        raise ValueError("consensus seal has an unexpected model identity or status")
    for key in ("loo_panel", "loo_case_accounting", "loo_scores_reference", "loo_contrasts_reference"):
        expected = consensus.get("files", {}).get(PATHS[key], {}).get("sha256")
        if expected != _sha256(paths[key]):
            raise ValueError(f"consensus seal SHA-256 mismatch: {PATHS[key]}")
    ratings = json.loads(paths["glicko_verification"].read_text(encoding="utf-8"))
    if ratings.get("status") != "completed_retrospective_ratings_diagnostic":
        raise ValueError("ratings receipt has an unexpected status")
    for key in ("glicko_predictions", "glicko_summary"):
        if ratings.get("sha256", {}).get(PATHS[key]) != _sha256(paths[key]):
            raise ValueError(f"ratings receipt SHA-256 mismatch: {PATHS[key]}")
    summary = json.loads(paths["glicko_summary"].read_text(encoding="utf-8"))
    if summary.get("version") != "corrected-header-ratings-only-daybatch-20260908":
        raise ValueError("ratings summary is not the corrected ratings-only protocol")
    return summary


def replay() -> tuple[dict[str, Any], dict[str, str]]:
    paths = _check_inputs()
    glicko_source = _check_additional_bindings(paths)
    consensus = summarize_consensus_panel(pd.read_parquet(paths["loo_panel"]),
                                         pd.read_parquet(paths["loo_case_accounting"]))
    consensus["saved_reference_verification"] = _check_consensus_references(
        consensus, pd.read_csv(paths["loo_scores_reference"]), pd.read_csv(paths["loo_contrasts_reference"]))
    glicko = summarize_glicko_pair(pd.read_csv(paths["glicko_predictions"]))
    rating_errors = []
    for kind in ("raw", "calibrated"):
        for entity, name in (("player", "player"), ("team", "organization")):
            reference = glicko_source["metrics"]["overall"][f"{entity}_gl_{kind}"]
            for metric in ("n", "log_loss", "brier"):
                rating_errors.append(_check_reference_value(glicko["scores"][kind][name][metric],
                                                             reference[metric], f"ratings/{entity}/{kind}/{metric}"))
    receipt_cohort = json.loads(paths["glicko_verification"].read_text(encoding="utf-8"))["cohort"]
    if (glicko["cohort"]["series"] != receipt_cohort["n"] or
            glicko["cohort"]["first_date"] != receipt_cohort["first_date"] or
            glicko["cohort"]["last_date"] != receipt_cohort["last_date"]):
        raise ValueError("ratings cohort differs from saved verification")
    glicko["source_protocol"]["saved_version"] = glicko_source["version"]
    glicko["source_protocol"]["saved_protocol"] = glicko_source["protocol"]
    glicko["source_protocol"]["saved_limitations"] = glicko_source["limitations"]
    glicko["source_protocol"]["prior_year_slopes"] = {
        year: {key: slopes[key] for key in ("player_gl", "team_gl")}
        for year, slopes in glicko_source["calibration_slopes"].items()
    }
    if not all(np.isfinite(value) and value > 0
               for slopes in glicko["source_protocol"]["prior_year_slopes"].values() for value in slopes.values()):
        raise ValueError("ratings protocol requires positive finite prior-year calibration slopes")
    glicko["saved_reference_verification"] = {
        "status": "PASS", "absolute_tolerance": 1e-12, "numeric_comparisons": len(rating_errors),
        "maximum_absolute_error": max(rating_errors),
    }
    model_protocol = json.loads(paths["model_protocol"].read_text(encoding="utf-8"))
    odds_protocol = json.loads(paths["odds_protocol"].read_text(encoding="utf-8"))
    serving = json.loads(paths["serving_config"].read_text(encoding="utf-8"))
    if model_protocol.get("run_id") != MODEL_RUN or odds_protocol.get("run_id") != FINANCE_RUN:
        raise ValueError("installed protocols do not match the report's frozen run IDs")
    if serving.get("serving_model_version") != C20R_VERSION:
        raise ValueError("local serving identity differs from the report's stated C20R model")
    model_binding = json.loads(paths["model_binding"].read_text(encoding="utf-8"))
    model_verification = json.loads(paths["model_verification"].read_text(encoding="utf-8"))
    if model_binding.get("input_hashes", {}).get("protocol.json") != _sha256(paths["model_protocol"]):
        raise ValueError("model protocol hash does not match its saved execution binding")
    if model_verification.get("status") != "VERIFIED_PASS":
        raise ValueError("model output lacks a successful saved independent verification receipt")
    if model_verification.get("predictions_sha256") != _sha256(paths["model_predictions"]):
        raise ValueError("model predictions differ from the saved independent-verification hash")
    odds_binding = json.loads(paths["odds_binding"].read_text(encoding="utf-8"))
    bound_inputs = odds_binding.get("input_and_code_sha256", {})
    for key in ("events", "odds_protocol"):
        if _bound_hash(bound_inputs, PATHS[key]) != _sha256(paths[key]):
            raise ValueError(f"{key} hash does not match the saved OddsPortal evaluation binding")
    if not odds_binding.get("binding_precedes_scoring"):
        raise ValueError("OddsPortal binding does not declare pre-score binding")

    pred = pd.read_parquet(paths["model_predictions"])
    if len(pred) != 3103 or pred.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("expected 3,103 unique canonical assessment predictions")
    if int(pred["regional_supported"].sum()) != 3083:
        raise ValueError("regional support mask differs from the frozen 3,083-row cohort")
    _validate_probabilities(pred, "y", ["C0", "fresh20", "annual_A0"])
    _validate_probabilities(pred.loc[pred.regional_supported], "y", ["fresh20+regional"])
    y = pred["y"].to_numpy(dtype=int)
    support = pred["regional_supported"].to_numpy(dtype=bool)
    model_columns = {
        "historical_C0": "C0",
        "fresh20_without_regional": "fresh20",
        "annual_A0": "annual_A0",
    }
    model_scores: dict[str, Any] = {
        "full_3103": {
            label: score_probability(y, pred[column].to_numpy(dtype=float))
            for label, column in model_columns.items()
        },
        "regional_supported_3083": {
            **{
                label: score_probability(y[support], pred[column].to_numpy(dtype=float)[support])
                for label, column in model_columns.items()
            },
            "C20R": score_probability(
                y[support], pred["fresh20+regional"].to_numpy(dtype=float)[support]
            ),
        },
        "coverage": {
            "assessment_rows": int(len(pred)),
            "regional_supported_rows": int(support.sum()),
            "regional_unsupported_rows": int((~support).sum()),
            "regional_coverage": float(support.mean()),
            "first_date": str(pred.date.min()),
            "last_date": str(pred.date.max()),
            "unit": "one canonical series; equal-series score",
            "unsupported_probability_status": "missing; excluded from regional model score, not imputed",
        },
    }

    events = pd.read_parquet(paths["events"])
    months = _month_universe(events, YEAR_RANGE)
    endpoints = pd.read_parquet(paths["odds_evaluation"])
    curated = endpoints.loc[endpoints.golgg_match_id.notna()]
    _unique_identities(curated, ["source", "source_event_id", "golgg_match_id", "bookmaker", "horizon"],
                       "financial endpoint ledger")
    if curated.loc[_known_labels(curated)].groupby("golgg_match_id").y.nunique().gt(1).any():
        raise ValueError("financial endpoint ledger has inconsistent known fixture labels")
    panel = endpoints.loc[endpoints.horizon.eq(FIRST)].copy()
    sports_zero = select_first_policy(panel, alpha=1.0, tax=0.0)
    samebook_zero = select_first_policy(panel, alpha=0.5, tax=0.0)
    samebook_fixed_tax = _recompute_fixed_tax(samebook_zero, 0.12)
    samebook_requalified_tax = select_first_policy(panel, alpha=0.5, tax=0.12)
    financial = {
        "policy": {
            "window": FIRST,
            "years": list(YEAR_RANGE),
            "tax_for_zero_tax_policies": 0.0,
            "sports_alpha": {"sports_only": 1.0, "samebook_50_50_logit": 0.5},
            "samebook_reference": "focal bookmaker multiplicative no-vig probability; not the application hybrid",
            "minimum_net_ev": 0.05,
            "dro_epsilon": 0.025,
            "maximum_odds": 4.0,
            "selection": "first eligible retained quote, at most one per canonical fixture",
            "stake": 1.0,
        },
        "cohort": {
            "mapped_target_first_date": panel.canonical_date.dropna().min(),
            "mapped_target_last_date": panel.canonical_date.dropna().max(),
            "unit": "one-unit selected ticket; binary-known-label returns",
            "month_universe": list(map(str, months)),
            "model_scope": "annual year-eligible fresh20+regional 2021–2026; not the 400-fixture LOO panel",
        },
        "sports_only_zero_tax": _financial_summary(sports_zero, months),
        "samebook_50_50_zero_tax": _financial_summary(samebook_zero, months),
        "samebook_50_50_fixed_tickets_tax_12_percent": _financial_summary(samebook_fixed_tax, months),
        "samebook_50_50_requalified_tax_12_percent": _financial_summary(samebook_requalified_tax, months),
    }

    later_price_proxies = {
        name: summarize_later_price_pairs(pair_later_quotes(selected, endpoints), months)
        for name, selected in (
            ("samebook_50_50_zero_tax", samebook_zero),
            ("samebook_50_50_requalified_tax_12_percent", samebook_requalified_tax),
        )
    }

    hashes = {key: _sha256(path) for key, path in paths.items()}
    bootstrap_by_domain = {
        "finance": {
            "resamples": DRAWS, "seed": SEED, "confidence_level": 0.95,
            "blocks": "all preselection UTC months; zero-action months retained",
            "month_universe": list(map(str, months)),
            "estimand": "equal-unit-stake ratio of totals; equal-active-month sensitivity",
        },
        "consensus": {
            "resamples": DRAWS, "seed": CONSENSUS_SEED, "confidence_level": 0.95,
            "blocks": "fixture scheduled-start UTC months",
            "month_universe": consensus["coverage"]["scheduled_start_months"],
            "estimand": "equal fixture after mean quote loss within book and equal books within fixture",
        },
        "ratings": {
            "resamples": DRAWS, "seed": SEED, "confidence_level": 0.95,
            "blocks": "match-date UTC months",
            "month_universe": glicko["cohort"]["months"],
            "estimand": "equal series; equal-month sensitivity",
        },
    }
    snapshot = {
        "schema": "project_results_replay_v2",
        "status": "REPLAYED_FROM_INSTALLED_SAVED_ARTIFACTS",
        "reproduction": {
            "bootstrap_by_domain": bootstrap_by_domain,
            "input_paths": PATHS,
            "runtime_versions": {
                "python": platform.python_version(),
                **{package: version(package)
                   for package in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn")},
            },
            "source_sha256": hashes,
        },
        "model": {
            "display_name": "C20R",
            "description": "20-token player-history ensemble with regional adjustment; 19 history input slots plus one learned token",
            "weights_model_version": serving["model_version"],
            "serving_model_version": serving["serving_model_version"],
            "feature_version": serving["serving_feature_version"],
            "historical_C0_is_distinct": True,
            "history_input_slots": 19,
            "learned_model_tokens": 1,
            "historical_evaluation_environment": model_binding.get("environment", {}),
        },
        "model_scores": model_scores,
        "consensus_prediction_quality": consensus,
        "glicko2_comparison": glicko,
        "financial_scenarios": financial,
        "later_price_proxies": later_price_proxies,
        "scope_limits": [
            "Historical development results; no prospective or held-out confirmation.",
            "ROI is known-label return per unit stake (turnover), not bankroll growth.",
            "Unknown labels leave whole-portfolio P&L and ROI undefined.",
            "First observed is not certified opening price; no authenticated activity, fills, two-other-book hybrid, or full application-gate replay.",
            "Endpoint policy portfolios overlap and are not additive.",
            "Later-price statistic is a recorded-price proxy, not true CLV or executable close.",
            "No fit, inference, acquisition, database access, or network access in this replay.",
        ],
    }
    return snapshot, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new exclusive directory for aggregate summary and replay receipt")
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"exclusive output directory already exists: {output_dir}")
    summary, hashes = replay()
    receipt = {
        "schema": "project_results_replay_receipt_v2",
        "status": "PASS",
        "bootstrap_by_domain": summary["reproduction"]["bootstrap_by_domain"],
        "source_sha256": hashes,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "project_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_value, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=_json_value, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "source_sha256": hashes},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
