#!/usr/bin/env python3
"""Replay project scores, saved wallets, and recorded-price diagnostics.

No fitting, model inference, database access, or network access is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from betting_app.core.ev import expected_value
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.c0_edge_policy_audit import vectorized_gate_passes
from src.analysis.long_term_market_evaluation import (
    _month_universe,
    select_fixed_policy,
    summarize_unit_returns,
)
from src.analysis.metrics import binary_log_loss_vector, calculate_ece
from src.analysis.production_market_benchmark import paired_cluster_interval
from src.analysis.production_saved_evidence import (
    pair_latest_recorded_closes,
    pair_production_closes,
    summarize_production_closes,
    summarize_latest_recorded_closes,
    summarize_production_wallet,
)
from src.analysis.saved_market_comparisons import (
    summarize_oddsportal_comparisons,
    summarize_pinnacle_comparisons,
)

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
PRODUCTION_RUN = "c0_gate_optimizer_20261023_01"
PRODUCTION_INPUT = f"data/05_model_input/c0_gate_optimizer/{PRODUCTION_RUN}"
PRODUCTION_OUTPUT = f"data/08_reporting/c0_gate_optimizer/{PRODUCTION_RUN}"
PRODUCTION_CLOSE = "data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01"
PRODUCTION_WINDOWS = ("h0_2", "h2_6", "h6_12", "h12_24", "h24_48", "primary48")
PINNACLE_RUN = "pinnacle_model_comparison_20261112_043418"
PINNACLE_OUTPUT = f"data/08_reporting/pinnacle_model_comparison/{PINNACLE_RUN}"
SAMPLING_INPUT = "data/08_reporting/audit/model_betting_reconciliation_20261113_01"
SAMPLING_SEED = 20261113

PATHS = {
    "model_predictions": "data/07_model_output/c0_model_improvement/c0_model_improvement_20261026_01/predictions.parquet",
    "model_protocol": "data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/protocol.json",
    "model_binding": "data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/execution_binding.json",
    "model_verification": "data/08_reporting/c0_model_improvement/c0_model_improvement_20261026_01/sealed_verification.json",
    "serving_config": "conf/base/c0_serving.json",
    "odds_protocol": "data/05_model_input/fresh20_oddsportal/fresh20_oddsportal_20261111_02/protocol.json",
    "odds_evaluation": "data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/oddsportal_annual_endpoint_ledger.parquet",
    "odds_binding": "data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/execution_binding.json",
    "odds_quality_reference": f"data/08_reporting/fresh20_oddsportal/{FINANCE_RUN}/evaluation/probability_quality.parquet",
    "odds_evaluation_receipt": f"data/08_reporting/fresh20_oddsportal/{FINANCE_RUN}/evaluation/evaluation_receipt.json",
    "pinnacle_ledger": f"data/07_model_output/pinnacle_model_comparison/{PINNACLE_RUN}/comparison_ledger.parquet",
    "pinnacle_scores_reference": f"{PINNACLE_OUTPUT}/scores.csv",
    "pinnacle_summary_reference": f"{PINNACLE_OUTPUT}/summary.json",
    "pinnacle_verification": f"{PINNACLE_OUTPUT}/verification.json",
    "events": "data/04_feature/long_term_market_edge/long_term_market_edge_20261022_01/oddsportal_identity_complete_v3/events.parquet",
    "loo_panel": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_matched_panel.parquet",
    "loo_case_accounting": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_case_accounting.parquet",
    "loo_scores_reference": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_scores.csv",
    "loo_contrasts_reference": "data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_paired_contrasts.csv",
    "loo_verification": "data/08_reporting/production_probability/production_probability_20261023_02/verification/sealed_verification.json",
    "glicko_predictions": "data/artifacts/corrected-ratings-logloss-20260908-full/holdout_predictions.csv",
    "glicko_summary": "data/artifacts/corrected-ratings-logloss-20260908-full/summary.json",
    "glicko_verification": "data/artifacts/corrected-ratings-logloss-20260908-full/verification.json",
    "production_protocol": f"{PRODUCTION_INPUT}/protocol.json",
    "production_configurations": f"{PRODUCTION_INPUT}/configurations.json",
    "production_fixtures": f"{PRODUCTION_INPUT}/fixtures.parquet",
    "production_binding": f"{PRODUCTION_INPUT}/binding.json",
    "production_events": f"{PRODUCTION_INPUT}/events.parquet",
    "production_verification": f"{PRODUCTION_OUTPUT}/final_verification.json",
    "production_wallet_summary": f"{PRODUCTION_OUTPUT}/sweep/summary.parquet",
    "production_selected": f"{PRODUCTION_OUTPUT}/sweep/selected.parquet",
    "production_funded": f"{PRODUCTION_OUTPUT}/sweep/funded_ledger.parquet",
    "production_entry_clv": f"{PRODUCTION_CLOSE}/clv/entry_clv_panel.parquet",
    "production_clv_verification": f"{PRODUCTION_CLOSE}/verification/sealed_verification.json",
    "production_closing_states": f"{PRODUCTION_CLOSE}/clv/closing_states.parquet",
    "sampled_checks": f"{SAMPLING_INPUT}/sampled_checks.parquet",
    "sampled_suggestions_reference": f"{SAMPLING_INPUT}/checked_suggestions.parquet",
    "sampling_reference": f"{SAMPLING_INPUT}/results.json",
    "market_comparisons_source": "src/analysis/saved_market_comparisons.py",
    "gate_vectorization_source": "src/analysis/c0_edge_policy_audit.py",
    "production_saved_evidence_source": "src/analysis/production_saved_evidence.py",
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
    pairs["authenticated_final_market_quote"] = False
    return pairs


def _finance_mean_interval(frame: pd.DataFrame, column: str, months: Sequence[str], *,
                           draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    labels = pd.to_datetime(frame.canonical_date, errors="raise").dt.strftime("%Y-%m")
    values = frame[column].to_numpy(float)
    if not np.isfinite(values).all() or labels.isna().any() or not set(labels).issubset(months):
        raise ValueError("price statistics require finite values and dates within the finance month universe")
    totals = pd.DataFrame({"month": labels.to_numpy(), "value": values, "n": np.ones(len(frame))})
    totals = totals.groupby("month").sum().reindex(months, fill_value=0.0)
    interval = block_ratio_interval(totals.value, totals.n, totals.index, draws=draws, seed=seed)
    active = totals.loc[totals.n.gt(0)]
    interval["active_month_sensitivity"] = block_ratio_interval(
        active.value, active.n, active.index, draws=draws, seed=seed)
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
        "authenticated_final_market_quote": False,
        "execution_evidence": "not_recorded",
        "scope": "historical strictly-later same-book price diagnostic; separate from the latest-recorded closing convention",
        "interval_scope": "unit-stake ratio on all finance preselection months; active-month sensitivity separate",
    }


def select_sampled_opportunities(checks: pd.DataFrame, events: pd.DataFrame,
                                 policy: dict[str, Any]) -> pd.DataFrame:
    """Replay the fixed observed-timestamp checks, without closing-price admission."""
    _unique_identities(checks, ["canonical_match_id", "entry_clock"], "sampled checks")
    _unique_identities(events, ["event_id"], "production events")
    _require_columns(checks, ["fixture_start_utc", "band", "hours"], "sampled checks")
    _require_columns(events, ["snapshot_id", "bookmaker", "decision_at", "start", "side",
                             "sports_p", "q", "odds", "best_of", "y"], "production events")
    sampled = checks.copy()
    sampled["entry_clock"] = sampled.entry_clock.map(lambda value: _aware_clock(value, "entry clock"))
    sampled["fixture_start_utc"] = sampled.fixture_start_utc.map(lambda value: _aware_clock(value, "sample start"))
    source = events.copy()
    source["decision_at"] = source.decision_at.map(lambda value: _aware_clock(value, "event clock"))
    source["start"] = source.start.map(lambda value: _aware_clock(value, "event start"))
    joined = source.merge(sampled, left_on=["canonical_match_id", "decision_at"],
                          right_on=["canonical_match_id", "entry_clock"], how="inner",
                          validate="many_to_one")
    if not joined.start.eq(joined.fixture_start_utc).all() or not (joined.decision_at < joined.start).all():
        raise ValueError("sample/event fixture clocks conflict or are not prestart")
    if not joined.side.isin(["team_a", "team_b"]).all():
        raise ValueError("production events have an invalid selected side")
    for column in ("sports_p", "q"):
        values = joined[column].to_numpy(float)
        if not (np.isfinite(values) & (values > 0) & (values < 1)).all():
            raise ValueError("snapshot probabilities must be finite and interior")
    blended = expit(.5 * logit(joined.sports_p) + .5 * logit(joined.q))
    if "p_logit" in joined:
        if not np.allclose(joined.p_logit, blended, rtol=0, atol=1e-12):
            raise ValueError("saved snapshot probability differs from fixed logit shrinkage")
    else:
        joined["p_logit"] = blended
    joined["prob_model"] = joined.p_logit
    joined["prob_base"] = joined.sports_p
    joined["prob_market_novig"] = joined.q
    joined["rating_disagreement"] = np.nan
    joined["prob_market_close_novig"] = np.nan
    joined["ev_net"] = joined.prob_model * joined.odds - 1
    passes = vectorized_gate_passes(joined, {**policy, "tax_rate": 0.0})
    eligible = np.logical_and.reduce(list(passes.values()))
    ordered = joined.loc[eligible].sort_values(
        ["canonical_match_id", "decision_at", "ev_net", "odds", "bookmaker", "side"],
        ascending=[True, True, False, False, True, True], kind="stable")
    return ordered.drop_duplicates(["canonical_match_id", "decision_at"]).reset_index(drop=True)


def _unit_offer_summary(selected: pd.DataFrame, months: Sequence[str], *,
                        draws: int = DRAWS, seed: int = SAMPLING_SEED) -> dict[str, Any]:
    data = selected.copy()
    data["canonical_date"] = pd.to_datetime(data.start, errors="raise").dt.strftime("%Y-%m-%d")
    known = data.loc[data.y.isin([0, 1])].copy()
    known["unit_return"] = known.y * known.odds - 1
    counts = data.groupby("canonical_match_id").size()
    return {
        "observations": len(data), "unique_fixtures": int(data.canonical_match_id.nunique()),
        "known_outcome_observations": len(known),
        "known_outcome_fixtures": int(known.canonical_match_id.nunique()),
        "unknown_outcome_observations": len(data) - len(known),
        "actual_wins": int(known.y.sum()),
        "expected_hybrid_wins": float(known.p_logit.sum()) if len(known) else None,
        "expected_sports_wins": float(known.sports_p.sum()) if len(known) else None,
        "mean_expected_ev": float(data.ev_net.mean()) if len(data) else None,
        "mean_expected_ev_known_outcomes": float(known.ev_net.mean()) if len(known) else None,
        "realized_unit_yield": _finance_mean_interval(known, "unit_return", months, draws=draws, seed=seed),
        "maximum_observations_per_fixture": int(counts.max()) if len(counts) else 0,
        "unit": "zero-tax unit return, observation-weighted; repeated fixtures share outcomes",
    }


def summarize_sampled_opportunities(checks: pd.DataFrame, selected: pd.DataFrame,
                                    months: Sequence[str], *, draws: int = DRAWS,
                                    seed: int = SAMPLING_SEED) -> dict[str, Any]:
    _unique_identities(checks, ["canonical_match_id", "entry_clock"], "sampled checks")
    _unique_identities(selected, ["canonical_match_id", "decision_at"], "sampled suggestions")
    return {
        "checks": len(checks), "checked_fixtures": int(checks.canonical_match_id.nunique()),
        "suggestions": len(selected), "abstentions": len(checks) - len(selected),
        "suggestion_rate": len(selected) / len(checks) if len(checks) else None,
        **_unit_offer_summary(selected, months, draws=draws, seed=seed),
    }


def _sampled_evidence(paths: dict[str, Path], months: Sequence[str]) -> dict[str, Any]:
    binding = json.loads(paths["production_binding"].read_text(encoding="utf-8"))
    if binding["files"][PATHS["production_events"]]["sha256"] != _sha256(paths["production_events"]):
        raise ValueError("production event SHA-256 differs from the frozen policy binding")
    reference = json.loads(paths["sampling_reference"].read_text(encoding="utf-8"))
    for key in ("production_events", "production_configurations", "production_closing_states"):
        if reference["source_hashes"].get(PATHS[key]) != _sha256(paths[key]):
            raise ValueError(f"sampling source differs from reconciliation: {PATHS[key]}")
    checks = pd.read_parquet(paths["sampled_checks"])
    events = pd.read_parquet(paths["production_events"])
    expected = pd.read_parquet(paths["sampled_suggestions_reference"])
    configurations = json.loads(paths["production_configurations"].read_text(encoding="utf-8"))
    baseline = next(item for item in configurations if item["config_id"] == "baseline")
    selected = select_sampled_opportunities(checks, events, baseline["policy"])
    if (len(checks) != reference["sampling"]["sampled_unique_fixture_timestamps"]
            or checks.groupby("band").size().ne(reference["sampling"]["draws_per_band"]).any()):
        raise ValueError("saved sample differs from its fixed time-stratified contract")
    _unique_identities(expected, ["event_id"], "saved suggestion reference")
    if set(selected.event_id) != set(expected.event_id):
        raise ValueError("replayed snapshot suggestions differ from the saved sample")
    left, right = selected.set_index("event_id"), expected.set_index("event_id")
    right = right.reindex(left.index)
    for column in ("odds", "p_logit", "sports_p", "q", "ev_net", "y"):
        if not np.allclose(left[column], right[column], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"snapshot suggestion source mismatch: {column}")
    states = pd.read_parquet(paths["production_closing_states"])
    pairs = pair_latest_recorded_closes(selected, states)
    result = summarize_sampled_opportunities(checks, selected, months)
    result["recorded_close_clv"] = summarize_latest_recorded_closes(pairs, months)
    result["by_lead_band"] = []
    for band in checks.band.unique():
        band_checks = checks.loc[checks.band.eq(band)]
        band_selected = selected.loc[selected.band.eq(band)]
        band_result = summarize_sampled_opportunities(band_checks, band_selected, months)
        band_result["band"] = band
        band_result["recorded_close_clv"] = summarize_latest_recorded_closes(pairs.loc[pairs.band.eq(band)], months)
        result["by_lead_band"].append(band_result)
    fixed_tickets = pd.read_parquet(
        paths["production_selected"], filters=[("case_id", "==", "tax0__logit__baseline__primary48")])
    fixed = events.set_index("event_id").loc[fixed_tickets.event_id].reset_index()
    if fixed.canonical_match_id.duplicated().any():
        raise ValueError("fixed chronological policy must select one ticket per fixture")
    fixed["ev_net"] = fixed.p_logit * fixed.odds - 1
    fixed_result = _unit_offer_summary(fixed, months)
    fixed_result["recorded_close_clv"] = summarize_latest_recorded_closes(
        pair_latest_recorded_closes(fixed, states), months)
    return {
        "status": "HISTORICAL_EXPLORATORY_COUNTERFACTUAL",
        "sampling": reference["sampling"],
        "snapshot_weighted": result,
        "fixed_first_eligible_48h": fixed_result,
        "closing_definition": {
            "reference": "latest recorded strictly prestart state per fixture/book",
            "maximum_age_seconds": None, "requires_later_than_entry": False,
            "same_entry_state_allowed": True, "invalid_latest_state_fallback": False,
            "loo": "at least two other books; equal-weight mean fair logits from raw two-sided prices",
            "tax_rate": 0.0, "authenticated_final_market_quote": False,
            "execution_evidence": "not_recorded; not required for price CLV",
        },
        "unassessed_guards": ["rating_disagreement", "contemporaneous_clv_drift"],
        "scope": "offers recorded exactly at sampled clocks, not a complete carried-forward bookmaker board",
        "sample_provenance": "saved checks and suggestions hashed in this replay; not independently sealed",
    }


def _market_evidence(paths: dict[str, Path], endpoints: pd.DataFrame, months: Sequence[str],
                     consensus: dict[str, Any], source_events: pd.DataFrame) -> dict[str, Any]:
    receipt = json.loads(paths["odds_evaluation_receipt"].read_text(encoding="utf-8"))
    if (receipt.get("run_id") != FINANCE_RUN or
            receipt.get("status") != "HISTORICAL_INDIVIDUAL_BOOK_ENDPOINT_SCENARIOS_COMPLETE"):
        raise ValueError("unexpected annual OddsPortal evaluation receipt")
    annual = summarize_oddsportal_comparisons(endpoints, list(months))
    references = pd.read_parquet(paths["odds_quality_reference"])
    references = references.loc[references.year.astype(str).eq("2021-2026")]
    annual_errors = []
    for horizon, result in annual["endpoints"].items():
        rows = references.loc[references.horizon.eq(horizon)]
        if len(rows) != 1:
            raise ValueError("expected one pooled annual quality reference per endpoint")
        reference = rows.iloc[0]
        for name, prefix in (("sports_model", ""), ("market", "market_")):
            for metric in ("n", "log_loss", "brier"):
                expected = reference["n" if metric == "n" else prefix + metric]
                annual_errors.append(_check_reference_value(result["scores"][name][metric], expected,
                                                             f"OddsPortal/{horizon}/{name}/{metric}"))
        for key in ("estimate", "ci_lower", "ci_upper", "equal_block_estimate", "blocks",
                    "active_blocks", "valid_replicates", "empty_replicates"):
            annual_errors.append(_check_reference_value(
                result["contrasts"]["log_loss"][key], reference.delta_vs_market[key],
                f"OddsPortal/{horizon}/delta/{key}"))
    inventory = source_events.loc[source_events.source_year.isin(YEAR_RANGE)]
    _unique_identities(inventory, ["source", "source_event_id"], "annual OddsPortal source inventory")
    source_ids = set(zip(inventory.source, inventory.source_event_id))
    quoted_ids = set(zip(endpoints.source, endpoints.source_event_id))
    if not quoted_ids.issubset(source_ids):
        raise ValueError("annual endpoint quote is outside the retained source inventory")
    annual["source_coverage"] = {
        "source_events": len(inventory),
        "quoted_source_events": len(quoted_ids),
        "events_without_retained_quotes": len(source_ids - quoted_ids),
        "source_events_missing_canonical_mapping": int(inventory.golgg_match_id.isna().sum()),
        "retained_endpoint_quote_rows": len(endpoints),
        "mapped_unique_fixtures": int(endpoints.golgg_match_id.nunique()),
        "mapped_fixture_first_date": endpoints.canonical_date.dropna().min(),
        "mapped_fixture_last_date": endpoints.canonical_date.dropna().max(),
    }
    annual["model_scope"] = "year-eligible annual fresh20+regional models 2021–2026, not one 2026 checkpoint backcast"
    annual["saved_reference_verification"] = {
        "status": "PASS", "absolute_tolerance": 1e-12, "numeric_comparisons": len(annual_errors),
        "maximum_absolute_error": max(annual_errors),
        "binding_limit": "endpoint ledger and reference scores hashed here; not independently sealed outputs",
    }

    seal = json.loads(paths["pinnacle_verification"].read_text(encoding="utf-8"))
    source = json.loads(paths["pinnacle_summary_reference"].read_text(encoding="utf-8"))
    if (seal.get("status") != "PASS" or seal.get("run_id") != PINNACLE_RUN or
            source.get("run_id") != PINNACLE_RUN):
        raise ValueError("Pinnacle evidence has an unexpected verification or run identity")
    for key in ("pinnacle_ledger", "pinnacle_scores_reference"):
        if seal.get("sha256", {}).get(PATHS[key]) != _sha256(paths[key]):
            raise ValueError(f"Pinnacle saved-output SHA-256 mismatch: {PATHS[key]}")
    ledger = pd.read_parquet(paths["pinnacle_ledger"])
    pinnacle = summarize_pinnacle_comparisons(ledger)
    score_reference = pd.read_csv(paths["pinnacle_scores_reference"])
    pin_errors = []

    def check_scores(result, horizon, cohort, models):
        for key, model in models.items():
            rows = score_reference.loc[(score_reference.horizon.eq(horizon))
                                       & score_reference.cohort.eq(cohort)
                                       & score_reference.model.eq(model)]
            if len(rows) != 1:
                raise ValueError("missing or ambiguous Pinnacle score reference")
            for metric, column in (("n", "n"), ("log_loss", "logloss"), ("brier", "brier")):
                pin_errors.append(_check_reference_value(result["scores"][key][metric], rows.iloc[0][column],
                                                         f"Pinnacle/{horizon}/{cohort}/{model}/{metric}"))

    for horizon, result in pinnacle["endpoints"].items():
        check_scores(result, horizon, "c20r_pinnacle_pairs", {"sports_model": "C20R", "market": "Pinnacle"})
        saved = source["c20r_vs_pinnacle"][horizon]["paired_contrast"]
        for actual_name, saved_name in (("log_loss", "logloss"), ("brier", "brier"),
                                        ("familywise_98_75_log_loss", "logloss_98_75_four_contrast_family")):
            for key in ("n", "blocks", "delta", "ci_lower", "ci_upper", "equal_block_delta",
                        "equal_block_ci_lower", "equal_block_ci_upper"):
                pin_errors.append(_check_reference_value(result["contrasts"][actual_name][key], saved[saved_name][key],
                                                         f"Pinnacle/{horizon}/{actual_name}/{key}"))
        secondary = pinnacle["secondary_scopes"][horizon]
        check_scores(secondary["polish_vs_pinnacle"], horizon, "polish_pinnacle_pairs",
                     {"sports_model": "Polish consensus", "market": "Pinnacle"})
        check_scores(secondary["three_way"], horizon, "three_way_common",
                     {"sports_model": "C20R", "market": "Polish consensus", "pinnacle": "Pinnacle"})
    if len(ledger) != source["scope"]["retained_endpoint_rows"] or ledger.source_event_id.nunique() != source["scope"]["source_fixtures"]:
        raise ValueError("Pinnacle source population differs from retained scope")
    pinnacle["source_coverage"] = source["scope"]
    pinnacle["source_decision"] = source["decision"]
    pinnacle["metadata"]["period_months"] = sorted(ledger.loc[ledger.c20r_pair_eligible, "month"].unique())
    pinnacle["secondary_scope_limits"] = "Fortuna and Superbet only; 2 months; not the entire Polish market or independent confirmation"
    pinnacle["saved_reference_verification"] = {
        "status": "PASS", "absolute_tolerance": 1e-12, "numeric_comparisons": len(pin_errors),
        "maximum_absolute_error": max(pin_errors),
    }
    return {"oddsportal": annual, "pinnacle": pinnacle, "polish": consensus}


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


def _production_evidence(paths: dict[str, Path]) -> dict[str, Any]:
    """Recompute twelve fixed saved portfolios, without rerunning policy selection."""
    seal = json.loads(paths["production_verification"].read_text(encoding="utf-8"))
    binding = json.loads(paths["production_binding"].read_text(encoding="utf-8"))
    closing_seal = json.loads(paths["production_clv_verification"].read_text(encoding="utf-8"))
    if seal.get("status") != "COMPLETED" or seal.get("input_binding_sha256") != _sha256(paths["production_binding"]):
        raise ValueError("production wallet seal or input binding is invalid")
    if closing_seal.get("status") != "COMPLETED_HISTORICAL_CONDITIONAL_DIAGNOSTIC":
        raise ValueError("unexpected production closing-reference seal")
    for source, keys in (
        (binding, ("production_protocol", "production_configurations", "production_fixtures")),
        (seal, ("production_wallet_summary", "production_selected", "production_funded")),
        (closing_seal, ("production_entry_clv",)),
    ):
        for key in keys:
            if source.get("files", {}).get(PATHS[key], {}).get("sha256") != _sha256(paths[key]):
                raise ValueError(f"production source SHA-256 mismatch: {PATHS[key]}")
    protocol = json.loads(paths["production_protocol"].read_text(encoding="utf-8"))
    if protocol.get("model") != C20R_VERSION or protocol.get("sports_share") != 0.5:
        raise ValueError("production evidence has an unexpected model or sports share")
    configurations = json.loads(paths["production_configurations"].read_text(encoding="utf-8"))
    baseline = [item for item in configurations if item["config_id"] == "baseline"]
    if len(baseline) != 1 or baseline[0]["changes"] or baseline[0]["disabled_groups"]:
        raise ValueError("production baseline must retain its frozen policy")

    case_ids = [f"tax{tax:g}__logit__baseline__{window}"
                for tax in (0.0, 0.12) for window in PRODUCTION_WINDOWS]
    query = [("case_id", "in", case_ids)]
    references = pd.read_parquet(paths["production_wallet_summary"], filters=query)
    selected = pd.read_parquet(paths["production_selected"], filters=query)
    funded = pd.read_parquet(paths["production_funded"], filters=query)
    if references.case_id.duplicated().any() or set(references.case_id) != set(case_ids):
        raise ValueError("production evidence must contain exactly the twelve fixed cases")
    if not selected.event_id.eq(selected.opportunity_id).all():
        raise ValueError("production selected event and opportunity identities differ")
    selected = selected.set_index(["case_id", "opportunity_id"])
    funded = funded.set_index(["case_id", "opportunity_id"])
    if not selected.index.is_unique or not funded.index.is_unique:
        raise ValueError("production selected/funded ticket identities must be unique")
    if set(selected.index) != set(funded.index):
        raise ValueError("selected and funded production ticket identities differ")
    selected = selected.reindex(funded.index)
    for column in ("canonical_match_id", "decision_at", "settle_at", "side", "bookmaker", "y"):
        left, right = selected[column], funded[column]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError(f"selected/funded production mismatch: {column}")
    if any(Decimal(str(left)) != Decimal(str(right))
           for left, right in zip(selected.odds, funded.odds, strict=True)):
        raise ValueError("selected and funded Decimal odds differ")
    if not np.allclose(selected.ev_net.to_numpy(float), funded.model_ev.to_numpy(float), rtol=0, atol=1e-12):
        raise ValueError("selected and funded expected returns differ")
    funded["start"] = selected.start
    funded["snapshot_id"] = selected.snapshot_id
    funded = funded.reset_index()
    references = references.set_index("case_id")
    funded["tax_rate"] = funded.case_id.map(references.tax)

    fixtures = pd.read_parquet(paths["production_fixtures"])
    if (fixtures.canonical_match_id.isna().any() or fixtures.canonical_match_id.duplicated().any()
            or not isinstance(fixtures.start.dtype, pd.DatetimeTZDtype)):
        raise ValueError("production fixture identities and aware start clocks are required")
    fixture_months = fixtures.start.dt.tz_convert("UTC").dt.strftime("%Y-%m")
    months = sorted(fixture_months.unique())
    entries = pd.read_parquet(paths["production_entry_clv"])
    windows = {item["window_id"]: item for item in protocol["windows"]}
    portfolios = []
    for case_id in case_ids:
        reference = references.loc[case_id]
        if reference.method != "logit" or reference.config_id != "baseline":
            raise ValueError("production policy identity differs from the fixed baseline")
        ledger = funded.loc[funded.case_id.eq(case_id)].copy()
        window = windows[reference.window_id]
        lead_hours = (ledger.start - ledger.decision_at).dt.total_seconds() / 3600
        if not ((lead_hours >= window["lo"]) & (lead_hours < window["hi"])).all():
            raise ValueError(f"production tickets fall outside {reference.window_id}")
        if int(reference.fixture_denominator) != len(fixtures):
            raise ValueError("production fixture denominator was reduced")
        wallet = summarize_production_wallet(ledger, months, draws=DRAWS, seed=CONSENSUS_SEED)
        for actual_key, saved_key in (
            ("bets", "funded_count"), ("settled_bets", "settled_count"), ("pending_bets", "pending_count"),
        ):
            if wallet[actual_key] != int(reference[saved_key]):
                raise ValueError(f"production wallet count mismatch: {case_id}/{saved_key}")
        if len(selected.loc[case_id]) != int(reference.selected_count):
            raise ValueError("production selected-ticket count mismatch")
        for actual_key, saved_key, tolerance in (
            ("settled_profit", "profit", 1e-8), ("settled_turnover", "turnover", 1e-8),
            ("settled_roi", "roi", 1e-12),
        ):
            if not np.isclose(wallet[actual_key], float(reference[saved_key]), rtol=0, atol=tolerance):
                raise ValueError(f"production wallet arithmetic mismatch: {case_id}/{saved_key}")
        closes = {
            age: summarize_production_closes(
                pair_production_closes(ledger, entries, reference=age),
                months, draws=DRAWS, seed=CONSENSUS_SEED)
            for age in ("300", "3600", "anyage")
        }
        portfolios.append({
            "case_id": case_id, "tax_rate": float(reference.tax), "window_id": reference.window_id,
            "hours_before_start": {"lower_inclusive": window["lo"], "upper_exclusive": window["hi"]},
            "source_price_fixtures": int(reference.source_window_fixture_count),
            "source_loo_fixtures": int(reference.source_loo_window_fixture_count),
            "selected_bets": int(reference.selected_count), "wallet": wallet,
            "closing_reference_proxies": closes,
        })
    first_reference = references.iloc[0]
    for field in ("source_matched_fixture_count", "source_matched_quotes",
                  "source_loo_global_quotes", "source_loo_global_fixture_count"):
        if references[field].nunique() != 1:
            raise ValueError(f"production cohort changes across fixed cases: {field}")
    return {
        "model": protocol["model"], "status": "HISTORICAL_COUNTERFACTUAL_NOT_RECORDED_BETS",
        "cohort": {
            "fixtures": len(fixtures), "quotes": int(first_reference.source_matched_quotes),
            "entry_loo_quotes": int(first_reference.source_loo_global_quotes),
            "entry_loo_fixtures": int(first_reference.source_loo_global_fixture_count),
            "fixture_start_min": fixtures.start.min().isoformat(),
            "fixture_start_max": fixtures.start.max().isoformat(),
            "fixture_start_months": months,
            "fixtures_by_start_month": fixture_months.value_counts().sort_index().to_dict(),
        },
        "policy": {
            "configuration": baseline[0], "method": "logit", "sports_share": protocol["sports_share"],
            "taxes_requalified_independently": [0.0, 0.12], "portfolio": protocol["portfolio"],
            "selection": "first eligible recorded observation per fixture within each half-open interval",
            "unassessed_guards": ["rating_disagreement", "contemporaneous_clv_drift"],
        },
        "portfolios": portfolios, "authenticated_final_market_quote": False,
        "execution_evidence": "not_recorded",
        "closing_reference_ages_seconds": {"300": 300, "3600": 3600, "anyage": None},
        "scope_limits": [
            "Production-collected prices and saved counterfactual C20R probabilities, not authenticated issued forecasts or actual bets.",
            "Reference ages are seconds before conservative recorded start, not elapsed seconds after entry.",
            "Two-hour collection cadence does not authenticate actual start, a complete close, source permission, or fills.",
            "Closing metrics are conditional on covered entries; missing references are not zero CLV.",
            "Gross LOO proxy excludes stake tax; net LOO proxy includes the scenario's stake tax.",
            "The original per-ticket free-cash exposure cap is not a 25% aggregate exposure cap.",
            "Independent interval wallets overlap; do not sum them or select a winning interval after scoring.",
            "Monthly intervals reuse retained funded stakes, not newly simulated wallet paths or refitted policies.",
        ],
    }


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
    production_betting = _production_evidence(paths)
    bookmaker_comparisons = _market_evidence(paths, endpoints, months, consensus, events)
    production_opportunities = _sampled_evidence(paths, production_betting["cohort"]["fixture_start_months"])

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
        "production_betting": {
            "resamples": DRAWS, "seed": CONSENSUS_SEED, "confidence_level": 0.95,
            "blocks": "complete preselection fixture-start UTC months; zero-action months retained",
            "month_universe": production_betting["cohort"]["fixture_start_months"],
            "estimand": "saved-stake ROI ratio; equal-ticket closing proxies, stake-weighted sensitivity",
            "undefined_draws": "retained; any zero-denominator draw withholds the interval",
        },
        "oddsportal_prediction": {
            "resamples": DRAWS, "seed": SEED, "confidence_level": 0.95,
            "month_universe": list(months),
            "estimand": "equal fixture; earliest eligible focal quote within each retained endpoint",
        },
        "pinnacle_prediction": {
            "resamples": DRAWS, "seed": 20261112, "confidence_level": 0.95,
            "four_contrast_family_confidence_level": 0.9875,
            "month_universe": bookmaker_comparisons["pinnacle"]["metadata"]["period_months"],
            "estimand": "paired equal fixture; equal-month sensitivity",
        },
        "production_opportunities": {
            "resamples": DRAWS, "seed": SAMPLING_SEED, "confidence_level": 0.95,
            "month_universe": production_betting["cohort"]["fixture_start_months"],
            "estimand": "selected-observation mean; fixed one-per-fixture policy separate",
            "undefined_draws": "retained; any zero-denominator draw withholds the interval",
        },
    }
    snapshot = {
        "schema": "project_results_replay_v4",
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
        "bookmaker_comparisons": bookmaker_comparisons,
        "glicko2_comparison": glicko,
        "financial_scenarios": financial,
        "later_price_proxies": later_price_proxies,
        "production_betting": production_betting,
        "production_opportunities": production_opportunities,
        "scope_limits": [
            "Historical development results; no prospective or held-out confirmation.",
            "ROI is known-label return per unit stake (turnover), not bankroll growth.",
            "Unknown labels leave whole-portfolio P&L and ROI undefined.",
            "OddsPortal scenarios use a focal-book blend, not a two-other-book hybrid or full application-gate replay.",
            "Production scenarios use a two-other-book logit blend but leave rating/drift guards unassessed; no authenticated activity or fills.",
            "Endpoint policy portfolios overlap and are not additive.",
            "Recorded-close CLV uses latest saved prestart prices, not certified final quotes; fills are a separate question.",
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
        "schema": "project_results_replay_receipt_v4",
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
