"""Predeclared, source-ID-aligned probability comparison for retained production observations.

Inputs are normalized ledgers: one row per retained snapshot observation in ``snapshots``;
zero or more model predictions per canonical fixture in ``predictions``. All observations
remain in the accounting output. Only explicit valid, pre-start, identity-bound two-sided
moneyline rows with known outcomes enter scoring.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.production_market_benchmark import paired_cluster_interval

PROTOCOL = {
    "protocol_id": "production_probability_20261023_01",
    "primary_market": "individual_book_same_clock_two_sided_moneyline_no_vig",
    "probability_orientation": "canonical_match_team_A; canonical-link-only evidence conditional",
    "eligibility": "latest recorded/scenario forecast ASOF each quote; declared data cutoff <= forecast clock <= quote clock < minimum retained declared scheduled-start bound; source and outcome clocks capped at separately observed export bound",
    "schedule_clock_policy": "aware schedules are UTC-normalized; legacy UTC-naive upcoming schedules are additional conservative UTC-assumed bounds, not authenticated actual start times",
    "stored_cutoff_cohorts": "timestamp-only separately from DECLARED_DATE_UTC_UPPER_BOUND_SCENARIO; in the latter a recorded history date means an assumed latest included UTC day and is usable no earlier than following midnight; never a timestamp/issuance authentication claim",
    "lead_bins_hours": [[0, 2], [2, 6], [6, 12], [12, 24], [24, 48], [48, 96], [96, None]],
    "primary_weighting": "quote-level scores averaged within fixture/book/model/lead-bin, then equal fixture weight",
    "diagnostic_weighting": "one retained observation equal weight; descriptive only",
    "endpoints": ["first_retained", "last_retained", "ALL"],
    "endpoint_policy": "first/last retained prestart source state before eligibility; invalid or unavailable endpoint withheld, never revive earlier quotes; not certified market OPEN/CLOSE",
    "metrics": ["log_loss", "brier", "calibration"],
    "bootstrap": {"draws": 5000, "unit": "calendar_month", "paired": True, "seed": 20261023},
    "selection": "no best horizon or alpha selection; no financial gates or filters",
    "archive_authentication": "conditional canonical identity cohort is labeled unverified; strict source-proof cohort separate",
    "current20R": "earliest-quote daily historical counterfactual retained ASOF only later in the same UTC day; actual generation separate, never issued forecast or backward clock fanout",
    "outcome_adjudication": "withhold known contradictions between canonical winners and complete map winners aligned by immutable team ID; never choose a label by model error or payout; original canonical-label pass retained",
}
LEAD_BINS = tuple((lo, np.inf if hi is None else hi) for lo, hi in PROTOCOL["lead_bins_hours"])

SNAPSHOT_REQUIRED = {
    "snapshot_id", "canonical_match_id", "observed_at", "earliest_start_utc", "export_clock_utc",
    "outcome_known_at", "best_of", "source_match_id", "mapping_source_match_id", "mapping_status",
    "source_binding_basis", "bookmaker", "odds_a", "odds_b", "raw_a_is_team_a", "market_type", "is_live",
    "snapshot_status", "availability_basis", "duplicate_representation", "outcome", "outcome_status",
}
PREDICTION_REQUIRED = {
    "canonical_match_id", "model_group", "model_id", "model_source_match_id",
    "model_binding_status", "model_binding_basis", "model_clock_utc", "model_data_cutoff_utc",
    "prediction_status", "p_team1", "p_team2",
}


def _utc(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        t = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    return t.tz_convert("UTC") if t.tzinfo is not None and t.utcoffset() is not None else pd.NaT


def evaluate_production_probabilities(snapshots: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    """Return all-row audit, fixture-equal primary scores, literal-row diagnostics and contrasts.

    ``predictions`` has one row per fixture/model/forecast-clock. Distinct stored-production
    and current20R replay cohorts are preserved by ``model_group``.
    Independently proved source IDs must match. Canonical-FK-only bindings remain
    conditional; all model probabilities use canonical team A, not source-header order.
    """
    missing_s = SNAPSHOT_REQUIRED - set(snapshots.columns)
    missing_p = PREDICTION_REQUIRED - set(predictions.columns)
    if missing_s or missing_p:
        raise ValueError(f"missing required columns: snapshots={sorted(missing_s)}, predictions={sorted(missing_p)}")
    if snapshots["snapshot_id"].isna().any() or snapshots["snapshot_id"].duplicated().any():
        raise ValueError("snapshot_id must be non-null and unique")
    if predictions.duplicated(["canonical_match_id", "model_group", "model_id", "model_clock_utc"]).any():
        raise ValueError("prediction rows must be unique per fixture/model/clock")

    s = snapshots.copy().reset_index(drop=True)
    s["canonical_match_id"] = s.canonical_match_id.astype("string").str.replace(r"\.0$", "", regex=True)
    for column in ("observed_at", "earliest_start_utc", "export_clock_utc", "outcome_known_at"):
        # _utc rejects naive values; explicit UTC dtype also covers an all-NaT column.
        s[column] = (s[column].dt.tz_convert("UTC") if isinstance(s[column].dtype, pd.DatetimeTZDtype)
                     else pd.to_datetime(s[column].map(_utc), utc=True))
    s["lead_hours"] = (s.earliest_start_utc-s.observed_at).dt.total_seconds()/3600
    s["lead_bin"] = pd.Series(pd.NA, index=s.index, dtype="string")
    for lo, hi in LEAD_BINS:
        m = s.lead_hours.ge(lo) & s.lead_hours.lt(hi)
        s.loc[m, "lead_bin"] = f"[{lo},{'inf' if np.isinf(hi) else hi})"
    s["exclusion_reason"] = ""
    independent_source = s.source_binding_basis.eq("independent_source_proof")
    canonical_link = s.source_binding_basis.eq("canonical_link_only") & s.canonical_match_id.notna()
    source_id_match = _same_id(s.mapping_source_match_id, s.source_match_id)
    strict_source = independent_source & source_id_match & s.mapping_status.eq("verified")
    conditional_source = canonical_link
    s["identity_cohort"] = np.where(strict_source, "verified",
                                    np.where(conditional_source, "conditional_archived_canonical_binding",
                                             "source_id_conflict"))
    def reject(mask: pd.Series, reason: str) -> None:
        s.loc[s.exclusion_reason.eq("") & mask.fillna(True), "exclusion_reason"] = reason
    reject(~(strict_source | conditional_source), "source_mapping_conflict_or_unproven_basis")
    reject(s.bookmaker.isna() | s.bookmaker.astype("string").str.strip().eq(""), "missing_bookmaker")
    reject(~s.market_type.astype("string").str.strip().str.lower().str.replace(r"[ -]+", "_", regex=True).isin(["match_winner", "series_winner"]), "not_moneyline")
    reject(s.is_live.map(lambda x: _bool(x) is not False), "live_or_unknown_live")
    reject(~s.snapshot_status.map(_active), "snapshot_not_active")
    reject(s.duplicate_representation.map(lambda x: _bool(x) is True), "duplicate_snapshot_representation")
    reject(s.canonical_match_id.isna() | s.canonical_match_id.astype("string").str.strip().eq(""), "missing_canonical_fixture")
    reject(s.observed_at.isna() | s.earliest_start_utc.isna() | s.export_clock_utc.isna(), "missing_or_naive_clock")
    reject(s.observed_at.ge(s.earliest_start_utc), "not_prestart")
    reject(s.observed_at.gt(s.export_clock_utc), "quote_after_export_clock")
    reject(s.earliest_start_utc.ge(s.export_clock_utc), "fixture_not_started_by_export_clock")
    oa, ob = pd.to_numeric(s.odds_a, errors="coerce"), pd.to_numeric(s.odds_b, errors="coerce")
    valid_odds = np.isfinite(oa) & np.isfinite(ob) & oa.gt(1) & ob.gt(1)
    reject(~valid_odds, "invalid_two_sided_odds")
    s["raw_a_is_team_a"] = s.raw_a_is_team_a.map(_bool)
    reject(s.raw_a_is_team_a.isna(), "unresolved_side_orientation")
    if "outcome_source_conflict" in s:
        reject(s.outcome_source_conflict.map(lambda value: _bool(value) is True), "conflicting_result_sources")
    known = s.outcome_status.map(_known_outcome) & s.outcome.isin([0, 1])
    outcome_asof = s.outcome_known_at.notna() & s.outcome_known_at.le(s.export_clock_utc)
    reject(~(known & outcome_asof), "unknown_outcome_at_export_clock")
    s["p_book_raw_a"] = np.where(valid_odds, 1.0/oa, np.nan)
    s["p_book_raw_b"] = np.where(valid_odds, 1.0/ob, np.nan)
    s["book_overround"] = s.p_book_raw_a+s.p_book_raw_b-1.0
    # A source row's value timestamp is not proof that market values were available then.
    s["p_book_team1"] = np.where(s.raw_a_is_team_a.eq(True), (1/oa)/((1/oa)+(1/ob)), (1/ob)/((1/oa)+(1/ob)))
    s.loc[~valid_odds, "p_book_team1"] = np.nan

    p = predictions.copy()
    p["canonical_match_id"] = p.canonical_match_id.astype("string").str.replace(r"\.0$", "", regex=True)
    for column in ("model_clock_utc", "model_data_cutoff_utc"):
        p[column] = (p[column].dt.tz_convert("UTC") if isinstance(p[column].dtype, pd.DatetimeTZDtype)
                     else pd.to_datetime(p[column].map(_utc), utc=True))
    model_keys = ["model_group", "model_id"]
    model_catalog = p[model_keys].drop_duplicates()
    p["_invalid_actual_clock"] = False
    if "actual_generated_at" in p:
        actual = pd.to_datetime(p.actual_generated_at.map(_utc), utc=True, errors="coerce")
        p["_invalid_actual_clock"] = p.actual_generated_at.notna() & actual.isna()
        p["_available_at"] = pd.concat([p.model_clock_utc, actual], axis=1).max(axis=1)
    else:
        p["_available_at"] = p.model_clock_utc
    base = s.merge(model_catalog, how="cross")
    valid_left = base.observed_at.notna() & base.canonical_match_id.notna()
    valid_right = p._available_at.notna() & p.canonical_match_id.notna()
    joined = pd.merge_asof(
        base.loc[valid_left].sort_values("observed_at", kind="stable"),
        p.loc[valid_right].sort_values("_available_at", kind="stable"),
        by=["canonical_match_id", *model_keys],
        left_on="observed_at", right_on="_available_at",
        direction="backward", allow_exact_matches=True, suffixes=("", "_prediction"),
    )
    if (~valid_left).any():
        joined = pd.concat([joined, base.loc[~valid_left].reindex(columns=joined.columns)], ignore_index=True)
    joined["pair_exclusion_reason"] = joined.exclusion_reason
    no_forecast = joined.model_clock_utc.isna()
    joined.loc[joined.pair_exclusion_reason.eq("") & no_forecast, "pair_exclusion_reason"] = "no_prior_forecast"
    same_model_source = _same_id(joined.model_source_match_id, joined.mapping_source_match_id)
    model_verified = (joined.model_binding_basis.eq("independent_source_proof") & same_model_source &
                      joined.model_binding_status.eq("verified"))
    model_conditional = joined.model_binding_basis.eq("canonical_link_only") & joined.canonical_match_id.notna()
    joined["identity_cohort"] = np.where(
        joined.identity_cohort.eq("verified") & model_verified, "verified",
        np.where(joined.identity_cohort.ne("source_id_conflict") & model_conditional,
                 "conditional_archived_canonical_binding", joined.identity_cohort),
    )
    bad_model_binding = ~(model_verified | model_conditional)
    joined.loc[joined.pair_exclusion_reason.eq("") & bad_model_binding.fillna(True), "pair_exclusion_reason"] = "model_source_binding_conflict_or_unproven_basis"
    joined.loc[joined.pair_exclusion_reason.eq("") & joined._invalid_actual_clock.fillna(False), "pair_exclusion_reason"] = "invalid_actual_generation_clock"
    invalid_status = ~joined.prediction_status.astype("string").str.lower().isin({"valid", "active", "issued", "success", "ok", "completed", "superseded", "stale"})
    joined.loc[joined.pair_exclusion_reason.eq("") & invalid_status.fillna(True), "pair_exclusion_reason"] = "latest_forecast_invalid_or_withdrawn"
    bad_cutoff = joined.model_data_cutoff_utc.isna() | joined.model_data_cutoff_utc.gt(joined.model_clock_utc)
    joined.loc[joined.pair_exclusion_reason.eq("") & bad_cutoff, "pair_exclusion_reason"] = "invalid_or_future_data_cutoff"
    joined.loc[joined.pair_exclusion_reason.eq("") & joined.model_clock_utc.ge(joined.earliest_start_utc), "pair_exclusion_reason"] = "model_not_prestart"
    if "model_clock_scope" in joined:
        daily = joined.model_clock_scope.eq("same_UTC_day_counterfactual")
        stale_day = joined.model_clock_utc.dt.normalize().ne(joined.observed_at.dt.normalize())
        joined.loc[joined.pair_exclusion_reason.eq("") & daily & stale_day, "pair_exclusion_reason"] = "no_same_day_replay_state"
    joined["p_model"] = pd.to_numeric(joined.p_team1, errors="coerce")
    p_model_other = pd.to_numeric(joined.p_team2, errors="coerce")
    invalidp = (~np.isfinite(joined.p_model) | ~joined.p_model.between(0, 1) |
                ~np.isfinite(p_model_other) | ~p_model_other.between(0, 1))
    incoherent = (joined.p_model+p_model_other-1).abs().gt(2e-6)
    joined.loc[joined.pair_exclusion_reason.eq("") & invalidp, "pair_exclusion_reason"] = "invalid_model_probability"
    joined.loc[joined.pair_exclusion_reason.eq("") & incoherent, "pair_exclusion_reason"] = "noncomplementary_model_sides"
    joined["scorable"] = joined.pair_exclusion_reason.eq("")
    joined["p_book_team1"] = pd.to_numeric(joined.p_book_team1, errors="coerce")
    matched = joined.loc[joined.scorable].copy()
    matched["month"] = matched.earliest_start_utc.dt.strftime("%Y-%m")
    # Score actual per-quote probabilities; averaging probabilities would fabricate a forecast.
    y = matched.outcome.to_numpy(float)
    pm = matched.p_model.to_numpy(float)
    pb = matched.p_book_team1.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        matched["ll_model"] = np.where(y == 1, -np.log(pm), -np.log1p(-pm))
        matched["ll_book"] = np.where(y == 1, -np.log(pb), -np.log1p(-pb))
    matched["brier_model"] = (pm-y)**2
    matched["brier_book"] = (pb-y)**2
    endpoint_keys = ["canonical_match_id", "bookmaker", "market_type"]
    prestart = s.observed_at.lt(s.earliest_start_utc) & s.observed_at.le(s.export_clock_utc)
    prestart_times = s.observed_at.where(prestart)
    first_time = prestart_times.groupby([s[k] for k in endpoint_keys]).transform("min")
    last_time = prestart_times.groupby([s[k] for k in endpoint_keys]).transform("max")
    endpoints = s[["snapshot_id"]].assign(
        _first=s.observed_at.eq(first_time).to_numpy(),
        _last=s.observed_at.eq(last_time).to_numpy())
    joined = joined.merge(endpoints, on="snapshot_id", validate="many_to_one")
    matched = matched.merge(endpoints, on="snapshot_id", validate="many_to_one")
    all_rows = matched.assign(endpoint="ALL")
    endpoint_rows = pd.concat([
        matched.loc[matched._first].assign(endpoint="first_retained"),
        matched.loc[matched._last].assign(endpoint="last_retained"),
    ], ignore_index=True)
    scored_rows = pd.concat([all_rows, endpoint_rows], ignore_index=True)
    fixture_keys = ["canonical_match_id", "best_of", "identity_cohort", "model_group", "model_id",
                    "bookmaker", "lead_bin", "endpoint"]
    fixture = scored_rows.groupby(fixture_keys, dropna=False, sort=True).agg(
        y=("outcome", "first"), ll_model=("ll_model", "mean"), ll_book=("ll_book", "mean"),
        brier_model=("brier_model", "mean"), brier_book=("brier_book", "mean"),
        p_model_calibration=("p_model", "mean"), p_book_calibration=("p_book_team1", "mean"),
        month=("month", "first"), n_observations=("snapshot_id", "nunique")).reset_index()
    score_rows, raw_rows, contrast_rows = [], [], []
    dims = ["best_of", "identity_cohort", "model_group", "model_id", "bookmaker", "lead_bin", "endpoint"]
    for keys, part in fixture.groupby(dims, dropna=False, sort=True):
        n = len(part)
        for comparator, llcol, briercol, pcol in (
            ("model", "ll_model", "brier_model", "p_model_calibration"),
            ("book", "ll_book", "brier_book", "p_book_calibration"),
        ):
            score_rows.append(dict(zip(dims, keys), comparator=comparator, weighting="fixture_equal", n=int(n),
                                   log_loss=float(part[llcol].mean()), brier=float(part[briercol].mean()),
                                   calibration_gap=float(part[pcol].mean()-part.y.mean())))
        for metric, model_col, book_col in (("log_loss", "ll_model", "ll_book"), ("brier", "brier_model", "brier_book")):
            delta = part[model_col].to_numpy(float)-part[book_col].to_numpy(float)
            nonfinite = int((~np.isfinite(delta)).sum())
            interval = paired_cluster_interval(delta, part.month, draws=5000, seed=20261023)
            if nonfinite:
                interval.update(n=len(part), blocks=int(part.month.nunique()), delta=None,
                                ci_lower=None, ci_upper=None, equal_block_delta=None,
                                equal_block_ci_lower=None, equal_block_ci_upper=None)
            contrast_rows.append(dict(zip(dims, keys), metric=metric, nonfinite_count=nonfinite,
                                      interval_estimable=nonfinite == 0 and interval["blocks"] >= 2, **interval))
    all_book_keys = ["canonical_match_id", "best_of", "identity_cohort", "model_group",
                     "model_id", "lead_bin", "endpoint"]
    all_books_fixture = fixture.groupby(all_book_keys, dropna=False, sort=True).agg(
        ll_model=("ll_model", "mean"), ll_book=("ll_book", "mean"),
        brier_model=("brier_model", "mean"), brier_book=("brier_book", "mean"),
        y=("y", "first"), month=("month", "first"), bookmakers=("bookmaker", "nunique"),
    ).reset_index()
    all_books_scores, all_books_contrasts = [], []
    all_book_dims = ["best_of", "identity_cohort", "model_group", "model_id", "lead_bin", "endpoint"]
    for keys, part in all_books_fixture.groupby(all_book_dims, dropna=False, sort=True):
        for comparator, llcol, briercol in (
            ("model", "ll_model", "brier_model"),
            ("average_individual_book_loss", "ll_book", "brier_book"),
        ):
            all_books_scores.append(dict(zip(all_book_dims, keys), comparator=comparator, n_fixtures=len(part),
                                        mean_bookmakers=float(part.bookmakers.mean()),
                                        log_loss=float(part[llcol].mean()), brier=float(part[briercol].mean())))
        for metric, model_col, book_col in (("log_loss", "ll_model", "ll_book"), ("brier", "brier_model", "brier_book")):
            delta = part[model_col].to_numpy(float)-part[book_col].to_numpy(float)
            nonfinite = int((~np.isfinite(delta)).sum())
            interval = paired_cluster_interval(delta, part.month, draws=5000, seed=20261023)
            if nonfinite:
                interval.update(n=len(part), blocks=int(part.month.nunique()), delta=None,
                                ci_lower=None, ci_upper=None, equal_block_delta=None,
                                equal_block_ci_lower=None, equal_block_ci_upper=None)
            all_books_contrasts.append(dict(zip(all_book_dims, keys), metric=metric,
                                             nonfinite_count=nonfinite, interval_estimable=nonfinite == 0 and interval["blocks"] >= 2,
                                             **interval))
    raw_dims = ["best_of", "identity_cohort", "model_group", "model_id", "bookmaker", "lead_bin"]
    for keys, part in matched.groupby(raw_dims, dropna=False, sort=True):
        for comparator, llcol, briercol in (("model", "ll_model", "brier_model"), ("book", "ll_book", "brier_book")):
            raw_rows.append(dict(zip(raw_dims, keys), comparator=comparator,
                                 weighting="observation_equal_diagnostic", n=len(part),
                                 log_loss=float(part[llcol].mean()), brier=float(part[briercol].mean())))
    snapshot_accounting = s.copy()
    snapshot_accounting["exclusion_reason"] = snapshot_accounting.exclusion_reason.replace("", "eligible_source_row")
    source_counts = (snapshot_accounting.groupby(["bookmaker", "market_type", "best_of", "identity_cohort", "exclusion_reason"],
                                                  dropna=False).agg(rows=("snapshot_id", "nunique")).reset_index())
    model_counts = (joined.groupby(["model_group", "model_id", "bookmaker", "pair_exclusion_reason"],
                                    dropna=False).agg(rows=("snapshot_id", "nunique")).reset_index())
    return {"protocol": PROTOCOL, "accounting": joined, "source_accounting_counts": source_counts,
            "model_accounting_counts": model_counts, "fixture_scores": pd.DataFrame(score_rows),
            "all_books_fixture_scores": pd.DataFrame(all_books_scores),
            "all_books_monthly_contrasts": pd.DataFrame(all_books_contrasts),
            "observation_scores": pd.DataFrame(raw_rows), "paired_monthly_contrasts": pd.DataFrame(contrast_rows),
            "fixture_panel": fixture}
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_prepared_probability_comparison(
    prepared_dir: str | Path,
    output_dir: str | Path,
    replay_forecasts: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the protocol, score normalized ledgers, and write auditable outputs once."""
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    snapshot_path = prepared_dir / "snapshots.parquet"
    stored_path = prepared_dir / "stored_predictions.parquet"
    inputs = [snapshot_path, stored_path]
    if replay_forecasts is not None:
        inputs.append(Path(replay_forecasts))
    absent = [str(path) for path in inputs if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"prepared probability inputs missing: {absent}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing probability report: {output_dir}")
    input_hashes = {str(path): _sha256_file(path) for path in inputs}
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_doc = {**PROTOCOL, "sha256": hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    (output_dir / "protocol_frozen.json").write_text(json.dumps(protocol_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshots = pd.read_parquet(snapshot_path)
    # Parse each source clock once, rather than once for every retained model version.
    for column in ("observed_at", "earliest_start_utc", "export_clock_utc", "outcome_known_at"):
        snapshots[column] = pd.to_datetime(snapshots[column].map(_utc), utc=True)
    predictions = pd.read_parquet(stored_path)
    if replay_forecasts is not None:
        replay = pd.read_parquet(replay_forecasts)
        predictions = pd.concat([predictions, replay], ignore_index=True, sort=False)
    # Keep the large all-row audit bounded: one model's ASOF join resident at a time.
    parts: dict[str, list[pd.DataFrame]] = {}
    accounting_rows = scorable_pairs = 0
    accounting_dir = output_dir / "accounting"
    accounting_dir.mkdir()
    for index, (_, history) in enumerate(predictions.groupby(["model_group", "model_id"], sort=True, dropna=False)):
        columns = list(PREDICTION_REQUIRED) + [c for c in ("actual_generated_at", "prediction_id", "model_clock_scope") if c in history]
        scored = evaluate_production_probabilities(snapshots, history[columns])
        audit = scored.pop("accounting")
        audit.to_parquet(accounting_dir / f"part-{index:03d}.parquet", index=False)
        accounting_rows += len(audit)
        scorable_pairs += int(audit.scorable.sum())
        for name, frame in scored.items():
            if isinstance(frame, pd.DataFrame):
                if name == "source_accounting_counts" and name in parts:
                    continue
                parts.setdefault(name, []).append(frame)
        print(f"Scored model {index + 1}: {int(audit.scorable.sum())} quote pairs", flush=True)
    combined = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    for name, frame in combined.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    receipt = {
        "protocol_id": PROTOCOL["protocol_id"], "protocol_sha256": protocol_doc["sha256"],
        "input_sha256": input_hashes, "snapshot_rows": int(len(snapshots)),
        "prediction_history_rows": int(len(predictions)),
        "accounting_rows": accounting_rows,
        "scorable_quote_model_pairs": scorable_pairs,
        "fixture_score_rows": int(len(combined.get("fixture_scores", []))),
        "all_books_score_rows": int(len(combined.get("all_books_fixture_scores", []))),
        "limitations": [
            "scraped_at is not proof of market-value availability",
            "conditional archived canonical bindings are not verified or executable",
            "historical current20R rows are replay scenarios, not issued forecasts",
        ],
    }
    (output_dir / "comparison_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def _same_id(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.astype("string").str.strip().eq(right.astype("string").str.strip()).fillna(False)


def _active(value: Any) -> bool:
    return str(value).strip().lower() in {"active", "valid", "open", "observed_price"}


def _known_outcome(value: Any) -> bool:
    return str(value).strip().lower() in {"known", "final", "completed"}




def prediction_request_manifest(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate current20R inference at fixture/day, selecting causal earliest clock.

    Required columns: canonical_match_id, source_match_id, release_day, request_clock_utc.
    A daily cache can be consumed only if its caller independently marks daily contract verification.
    """
    required = {"canonical_match_id", "source_match_id", "release_day", "request_clock_utc"}
    if not required.issubset(fixtures.columns):
        raise ValueError(f"missing request fields: {sorted(required-set(fixtures.columns))}")
    f = fixtures.copy()
    f["request_clock_utc"] = f.request_clock_utc.map(_utc)
    if f[["canonical_match_id", "source_match_id", "release_day"]].isna().any().any():
        raise ValueError("request fixture/source/day identifiers must be present")
    if f.request_clock_utc.isna().any():
        raise ValueError("request clocks must be timezone-aware")
    source_counts = f.groupby(["canonical_match_id", "release_day"]).source_match_id.nunique(dropna=False)
    if source_counts.gt(1).any():
        raise ValueError("fixture/day maps to multiple source IDs; refusing fanout")
    ordered = f.sort_values(["canonical_match_id", "release_day", "request_clock_utc"], kind="stable")
    manifest = ordered.drop_duplicates(["canonical_match_id", "release_day"], keep="first").copy()
    index = pd.MultiIndex.from_frame(manifest[["canonical_match_id", "release_day"]])
    manifest["source_join_fanout"] = f.groupby(["canonical_match_id", "release_day"]).size().reindex(index).to_numpy()
    declared = manifest.get("original_daily_contract_verified", pd.Series(False, index=manifest.index))
    manifest["cache_reuse_permitted"] = declared.map(lambda value: _bool(value) is True)
    return manifest.reset_index(drop=True)
