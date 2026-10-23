"""Normalize the fresh production export into auditable quote/model ledgers.

The export remains untouched. This module selects safe JSONL fields, retains every odds
snapshot, uses the existing identity builder for orientation evidence, and labels canonical-FK
identity as conditional rather than independent source proof.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.analysis.production_market_identity import build_identity_tables, _norm
from src.analysis.production_probability_evaluation import PROTOCOL, prediction_request_manifest

RUN = "production_probability_20261023_01"
RAW_ROOT = Path("data/01_raw/production_probability") / RUN
NORMALIZED_ROOT = Path("data/03_primary/production_probability") / RUN

_FIELDS: dict[str, tuple[str, ...]] = {
    "odds_snapshots": ("id", "bookmaker_id", "match_id", "canonical_match_id", "market_type", "raw_team_a", "raw_team_b", "odds_a", "odds_b", "is_live", "scraped_at"),
    "canonical_predictions": ("id", "canonical_match_id", "model_artifact_id", "model_name", "model_version", "predicted_at", "prob_a", "prob_b", "data_cutoff_at", "prediction_status", "features_version", "ratings_version"),
    "canonical_matches": ("id", "canonical_key", "team_a_name", "team_b_name", "normalized_team_a", "normalized_team_b", "start_time_normalized", "status", "winner_side", "result_source", "result_source_match_id", "result_recorded_at", "best_of"),
    "golgg_match_mappings": ("id", "canonical_match_id", "golgg_match_id", "confidence", "mapped_by", "mapped_at"),
    "golgg_teams": ("id", "team_id"),
    "golgg_matches": ("match_id", "date", "team1_id", "team2_id", "team1_name", "team2_name", "best_of"),
    "team_aliases": ("id", "normalized_name", "alias", "raw_name", "normalized_alias", "source_system", "league_pattern", "tournament_pattern", "valid_from", "valid_to", "is_active", "is_blocked", "review_status"),
    "bookmakers": ("id", "name"),
    "bookmaker_events": ("id", "bookmaker_id", "bookmaker_event_id", "canonical_match_id", "match_start_time"),
    "bookmaker_markets": ("bookmaker_event_id", "bookmaker_market_key", "market_type", "line_id", "is_extra_market"),
    "upcoming_matches": ("canonical_match_id", "match_start_time", "team_a_golgg_id", "team_b_golgg_id"),
    "prop_odds_snapshots": ("id", "bookmaker_id", "match_id", "canonical_match_id", "market_type", "scraped_at", "is_live"),
    "odds_outcome_snapshots": ("id", "scrape_run_id", "bookmaker_event_id", "bookmaker_market_key", "outcome_key",
                               "outcome_name", "outcome_side", "decimal_odds", "scraped_at"),
}


def _read_jsonl(root: Path, table: str) -> pd.DataFrame:
    path = root / f"{table}.jsonl"
    columns = _FIELDS[table]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                raw = json.loads(line)
                rows.append({column: raw.get(column) for column in columns})
                if table == "canonical_predictions":
                    diagnostics = raw.get("diagnostics_json")
                    if isinstance(diagnostics, str):
                        try:
                            diagnostics = json.loads(diagnostics)
                        except (ValueError, TypeError):
                            diagnostics = {}
                    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
                    historical = any(diagnostics.get(k) is True for k in ("backfill", "is_backfill", "backtest", "is_backtest"))
                    historical |= any(any(term in str(diagnostics.get(k, "")).lower() for term in ("backfill", "backtest", "historical_replay"))
                                      for k in ("source", "origin", "mode", "prediction_source"))
                    rows[-1]["diagnostic_historical_reconstruction"] = historical
                    rows[-1]["actual_generated_at"] = next((diagnostics[k] for k in ("actual_generated_at", "forecast_generated_at", "generated_at", "actual_created_at")
                                                          if diagnostics.get(k)), None)
    extras = ("diagnostic_historical_reconstruction", "actual_generated_at") if table == "canonical_predictions" else ()
    return pd.DataFrame(rows, columns=(*columns, *extras))


def _id(value: Any) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _utc_series(values: pd.Series) -> pd.Series:
    # Legacy naive schedule values are conditional UTC declarations, not source authentication.
    return pd.to_datetime(values, format="mixed", utc=True, errors="coerce")


def _earliest_start(
    canonical: pd.DataFrame,
    events: pd.DataFrame,
    upcoming: pd.DataFrame,
    identities: pd.DataFrame,
    golgg_teams: pd.DataFrame,
) -> pd.DataFrame:
    chunks = [canonical[["canonical_match_id", "start_time_normalized"]].rename(
        columns={"start_time_normalized": "start"})]
    if not events.empty:
        chunks.append(events[["canonical_match_id", "match_start_time"]].rename(columns={"match_start_time": "start"}))
    if not upcoming.empty:
        identity_by_match = identities.set_index("canonical_match_id")
        native_ids = {_id(row["id"]): _id(row.get("team_id")) for row in golgg_teams.to_dict("records")}
        accepted = []
        for row in upcoming.to_dict("records"):
            identity = identity_by_match.loc[row["canonical_match_id"]] if row["canonical_match_id"] in identity_by_match.index else None
            if identity is None or pd.isna(identity.get("team1_id")) or pd.isna(identity.get("team2_id")):
                continue
            expected = (_id(identity.get("team1_id")), _id(identity.get("team2_id")))
            actual = (native_ids.get(_id(row.get("team_a_golgg_id"))), native_ids.get(_id(row.get("team_b_golgg_id"))))
            if None not in actual and actual in (expected, expected[::-1]):
                accepted.append({"canonical_match_id": row["canonical_match_id"], "start": row.get("match_start_time")})
        chunks.append(pd.DataFrame(accepted, columns=["canonical_match_id", "start"]))
    all_starts = pd.concat(chunks, ignore_index=True)
    all_starts["start"] = _utc_series(all_starts.start)
    return (all_starts.dropna(subset=["canonical_match_id", "start"])
            .groupby("canonical_match_id", as_index=False).start.min()
            .rename(columns={"start": "earliest_start_utc"}))


def normalize_production_probability_export(
    raw_root: str | Path = RAW_ROOT,
    output_root: str | Path = NORMALIZED_ROOT,
    extra_odds: pd.DataFrame | None = None,
    outcome_checks: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Write normalized snapshot, production prediction, and current20R request ledgers.

    No raw payload/diagnostics field is copied to normalized outputs. Every raw odds snapshot
    remains one row in ``snapshots``; prop and outcome observation inventories are counted too.
    """
    raw_root, output_root = Path(raw_root), Path(output_root)
    receipt = json.loads((raw_root / "export_receipt.json").read_text(encoding="utf-8"))
    export_clock = pd.Timestamp(receipt["observed_server_transaction_clock_from_separate_read_only_snapshot"]["utc"])
    matches = _read_jsonl(raw_root, "canonical_matches").rename(columns={"id": "canonical_match_id"})
    mappings = _read_jsonl(raw_root, "golgg_match_mappings")
    odds = _read_jsonl(raw_root, "odds_snapshots")
    original_odds_count = len(odds)
    if extra_odds is not None and not extra_odds.empty:
        odds = pd.concat([odds, extra_odds], ignore_index=True, sort=False)
    aliases = _read_jsonl(raw_root, "team_aliases")
    golgg = _read_jsonl(raw_root, "golgg_matches")
    golgg_teams = _read_jsonl(raw_root, "golgg_teams")
    books = _read_jsonl(raw_root, "bookmakers")
    events = _read_jsonl(raw_root, "bookmaker_events")
    upcoming = _read_jsonl(raw_root, "upcoming_matches")
    markets = _read_jsonl(raw_root, "bookmaker_markets")
    props = _read_jsonl(raw_root, "prop_odds_snapshots")
    raw_outcomes = _read_jsonl(raw_root, "odds_outcome_snapshots")
    predictions = _read_jsonl(raw_root, "canonical_predictions")

    # Fit the existing identity helper's source schema directly to exact IDs from mappings.
    mappings["golgg_match_id"] = mappings.golgg_match_id.map(_id)
    source = golgg.copy()
    source["t1_id"], source["t2_id"] = source.team1_id, source.team2_id
    source["t1_name"], source["t2_name"] = source.team1_name, source.team2_name
    source["sname_t1"], source["sname_t2"] = source.team1_name, source.team2_name
    source_records = source.to_dict("records")
    empty_forecasts = pd.DataFrame(columns=["golgg_match_id"])
    identity_odds = odds.drop_duplicates(["canonical_match_id", "raw_team_a", "raw_team_b"], keep="first")
    identity_odds = identity_odds[["id", "canonical_match_id", "raw_team_a", "raw_team_b", "bookmaker_id"]].rename(
        columns={"bookmaker_id": "bookmaker"})
    identities, quote_alignment, identity_receipt = build_identity_tables(
        matches, mappings, identity_odds, aliases, {"golgg_matches": source_records}, empty_forecasts,
    )
    identity_by_match = identities.set_index("canonical_match_id", drop=False)
    quote_alignment = quote_alignment.rename(columns={"id": "snapshot_id"})
    quote_lookup = {
        (_id(row["canonical_match_id"]), _id(row["raw_team_a"]), _id(row["raw_team_b"])): row
        for row in quote_alignment.to_dict("records")
    }

    start_by_match = _earliest_start(matches, events, upcoming, identities, golgg_teams).set_index("canonical_match_id")["earliest_start_utc"]
    canonical = matches.set_index("canonical_match_id")
    book_lookup = books.assign(bookmaker_id=books.id.map(_id)).set_index("bookmaker_id").name.to_dict()
    # Canonical team A is the binary target orientation used by the native replay.
    canonical_outcome: dict[Any, tuple[Any, Any]] = {}
    for row in matches.to_dict("records"):
        winner = str(row.get("winner_side") or "").strip().lower()
        y = 1 if winner in {"team_a", "a", "home"} else 0 if winner in {"team_b", "b", "away"} else None
        status = "known" if y in (0, 1) and str(row.get("status", "")).lower() == "finished" else "unknown"
        canonical_outcome[row["canonical_match_id"]] = (y if status == "known" else None, status)

    snapshots: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    odds["observed_utc"] = _utc_series(odds.scraped_at)
    duplicate_representation = odds.duplicated(
        ["canonical_match_id", "bookmaker_id", "market_type", "scraped_at", "raw_team_a", "raw_team_b", "odds_a", "odds_b", "is_live"],
        keep="first",
    )
    for quote_index, quote in enumerate(odds.to_dict("records")):
        mid = quote.get("canonical_match_id")
        identity = identity_by_match.loc[mid] if mid in identity_by_match.index else None
        aligned = quote_lookup.get((_id(mid), _id(quote.get("raw_team_a")), _id(quote.get("raw_team_b"))))
        raw_a_is_team_a = aligned.get("raw_a_is_canonical_a") if aligned is not None else None
        if raw_a_is_team_a is None or pd.isna(raw_a_is_team_a):
            if mid in canonical.index:
                fixture = canonical.loc[mid]
                ca, cb = _norm(fixture.get("team_a_name")), _norm(fixture.get("team_b_name"))
                qa, qb = _norm(quote.get("raw_team_a")), _norm(quote.get("raw_team_b"))
                if ca and cb and ca != cb and qa != qb:
                    raw_a_is_team_a = True if (qa, qb) == (ca, cb) else False if (qa, qb) == (cb, ca) else None
        golgg_id = identity.get("golgg_match_id") if identity is not None else None
        start = start_by_match.get(mid, pd.NaT)
        y, outcome_status = canonical_outcome.get(mid, (None, "unknown"))
        observed = quote.get("observed_utc")
        snapshots.append({
            "snapshot_id": _id(quote.get("id")), "canonical_match_id": mid,
            "bookmaker": book_lookup.get(_id(quote.get("bookmaker_id"))),
            "observed_at": quote.get("scraped_at"), "earliest_start_utc": start,
            "export_clock_utc": export_clock, "outcome_known_at": ((canonical.loc[mid].get("result_recorded_at") or export_clock) if mid in canonical.index and outcome_status == "known" else None),
            "outcome_clock_basis": "recorded_result_clock_or_finished_snapshot_upper_bound",
            "best_of": identity.get("canonical_best_of") if identity is not None and pd.notna(identity.get("canonical_best_of")) else (canonical.loc[mid].get("best_of") if mid in canonical.index else None),
            "source_match_id": None, "mapping_source_match_id": golgg_id,
            "mapping_status": "verified" if identity is not None and identity.get("identity_status") == "verified" else str(identity.get("identity_status") if identity is not None else "no_canonical_mapping"),
            "source_binding_basis": "canonical_link_only" if mid in canonical.index else "unresolved",
            "odds_a": quote.get("odds_a"), "odds_b": quote.get("odds_b"),
            "raw_a_is_team_a": raw_a_is_team_a, "market_type": quote.get("market_type"),
            "is_live": quote.get("is_live"), "snapshot_status": "observed_price",
            "availability_basis": "recorded_scraped_price_conditional",
            "observation_origin": quote.get("observation_origin") or "odds_snapshots",
            "live_status_basis": quote.get("live_status_basis") or "recorded_is_live_flag",
            "duplicate_representation": bool(duplicate_representation.iloc[quote_index]),
            "outcome": y, "outcome_status": outcome_status,
        })
        if (not pd.isna(observed) and identity is not None and golgg_id is not None and
                pd.notna(identity.get("canonical_a_is_team1"))):
            release_day = observed.date().isoformat()
            request_rows.append({
                "canonical_match_id": mid, "source_match_id": golgg_id,
                "release_day": release_day, "request_clock_utc": observed,
                "opportunity_id": f"{mid}:{release_day}", "golgg_match_id": golgg_id,
                "decision_at": observed, "start_at": start,
                "canonical_a_is_team1": bool(identity.get("canonical_a_is_team1")),
                "team1_id": identity.get("team1_id"), "team2_id": identity.get("team2_id"),
                "best_of": identity.get("canonical_best_of"), "identity_status": identity.get("identity_status"),
            })
    snapshot_frame = pd.DataFrame(snapshots)
    snapshot_frame["snapshot_id"] = snapshot_frame.snapshot_id.astype("string")
    if outcome_checks is not None:
        snapshot_frame = apply_outcome_adjudication(snapshot_frame, outcome_checks)
    predictions["prob_a_numeric"] = pd.to_numeric(predictions.prob_a, errors="coerce")
    predictions["prob_b_numeric"] = pd.to_numeric(predictions.prob_b, errors="coerce")
    pred_rows: list[dict[str, Any]] = []
    for row in predictions.to_dict("records"):
        mid = row.get("canonical_match_id")
        identity = identity_by_match.loc[mid] if mid in identity_by_match.index else None
        pa, pb = row.get("prob_a_numeric"), row.get("prob_b_numeric")
        p_team1, p_team2 = pa, pb  # Evaluator/replay interface: team1 is canonical team A.
        gid = identity.get("golgg_match_id") if identity is not None else None
        model_id = ":".join(str(part) for part in (row.get("model_artifact_id"), row.get("model_name"), row.get("model_version"), row.get("features_version"), row.get("ratings_version")))
        pred_rows.append({
            "canonical_match_id": mid, "model_group": f"stored-production:{row.get('model_name')}:{row.get('model_version')}",
            "model_id": model_id, "model_source_match_id": None,
            "model_binding_status": "unverified", "model_binding_basis": "canonical_link_only" if mid in canonical.index else "unresolved",
            "model_clock_utc": row.get("predicted_at"), "model_data_cutoff_utc": row.get("data_cutoff_at"),
            "prediction_status": "historical_reconstruction" if row.get("diagnostic_historical_reconstruction") else row.get("prediction_status"), "p_team1": p_team1, "p_team2": p_team2,
            "actual_generated_at": row.get("actual_generated_at"),
            "raw_prob_a": float(pa) if pd.notna(pa) else np.nan, "raw_prob_b": float(pb) if pd.notna(pb) else np.nan,
            "prediction_id": _id(row.get("id")), "model_artifact_id": row.get("model_artifact_id"),
        })
    prediction_frame = pd.DataFrame(pred_rows)
    if not prediction_frame.empty:
        prediction_frame["prediction_id"] = prediction_frame.prediction_id.astype("string")
    prediction_frame, duplicate_states = resolve_prediction_states(prediction_frame)

    requests = pd.DataFrame(request_rows)
    manifest = prediction_request_manifest(requests) if not requests.empty else pd.DataFrame()
    if not manifest.empty:
        manifest["decision_at"] = manifest.request_clock_utc
        manifest["opportunity_id"] = manifest.canonical_match_id.astype(str) + ":" + manifest.release_day.astype(str)

    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_frame.to_parquet(output_root / "snapshots.parquet", index=False)
    prediction_frame.to_parquet(output_root / "stored_predictions.parquet", index=False)
    duplicate_states.to_parquet(output_root / "prediction_duplicate_states.parquet", index=False)
    pd.DataFrame(props).to_parquet(output_root / "prop_observation_inventory.parquet", index=False)
    pd.DataFrame(raw_outcomes).to_parquet(output_root / "outcome_observation_inventory.parquet", index=False)
    if not manifest.empty:
        manifest.to_parquet(output_root / "current20r_request_manifest.parquet", index=False)
    inventory = {
        "run": RUN, "protocol": PROTOCOL, "raw_export_clock": export_clock.isoformat(),
        "clock_provenance": "separate read-only transaction; bulk snapshot exact clock/LSN unavailable in source receipt",
        "raw_counts": {"odds_snapshots": original_odds_count, "paired_outcome_quotes": len(odds)-original_odds_count,
                       "prop_odds_snapshots": len(props), "odds_outcome_snapshots": len(raw_outcomes),
                       "canonical_predictions": len(predictions), "canonical_matches": len(matches), "golgg_matches": len(golgg)},
        "normalized_counts": {"snapshots": len(snapshot_frame), "stored_predictions": len(prediction_frame),
                              "current20r_request_manifest": len(manifest)},
        "source_status_counts": snapshot_frame.mapping_status.value_counts(dropna=False).astype(int).to_dict(),
        "quote_identity_counts": quote_alignment.quote_identity_status.value_counts(dropna=False).astype(int).to_dict(),
        "identity_receipt": identity_receipt,
        "limitations": ["canonical FK and side-name alignment are conditional evidence, never independent source proof",
                        "quote scrape timestamp is not proof that market values were available at that time",
                        "outcome pairs require exact event, market, scrape-run and observation clock; unresolved observations remain in the pairing inventory"],
    }
    (output_root / "normalization_receipt.json").write_text(json.dumps(inventory, indent=2, default=str) + "\n", encoding="utf-8")
    return inventory


def resolve_prediction_states(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse equivalent copies; withhold conflicting states at an identical claimed clock."""
    keys = ["canonical_match_id", "model_group", "model_id", "model_clock_utc"]
    duplicate = predictions.duplicated(keys, keep=False)
    if not duplicate.any():
        return predictions.copy(), pd.DataFrame(columns=[*keys, "prediction_id", "duplicate_state_status"])
    grouped = predictions.loc[duplicate].copy()
    valid = {"valid", "active", "issued", "success", "ok", "completed", "superseded", "stale"}
    grouped["_eligibility_class"] = grouped.prediction_status.astype("string").str.lower().isin(valid)
    signature = ["p_team1", "p_team2", "model_data_cutoff_utc", "_eligibility_class",
                 "model_source_match_id", "model_binding_status", "model_binding_basis"]
    if "actual_generated_at" in grouped:
        signature.append("actual_generated_at")
    retained, audit = [], []
    for _, group in grouped.groupby(keys, dropna=False, sort=False):
        conflict = bool(group[signature].nunique(dropna=False).gt(1).any())
        row = group.sort_values("prediction_id", kind="stable").iloc[0].drop(labels="_eligibility_class").copy()
        status = "conflicting_same_clock_state_withheld" if conflict else "equivalent_copy_collapsed"
        if conflict:
            row["prediction_status"] = status
            row["p_team1"] = row["p_team2"] = float("nan")
        retained.append(row)
        for original in group.to_dict("records"):
            audit.append({**{k: original.get(k) for k in (*keys, "prediction_id")},
                          "duplicate_state_status": status})
    unique = pd.concat([predictions.loc[~duplicate], pd.DataFrame(retained)], ignore_index=True)
    return unique, pd.DataFrame(audit)


def apply_outcome_adjudication(snapshots: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    """Flag contradictions without choosing a convenient label or altering raw outcomes."""
    required = {"canonical_match_id", "canonical_outcome", "map_id_outcome", "source_team_id_pair_matches"}
    if required - set(checks):
        raise ValueError("outcome checks must identify canonical and map-team-ID results")
    checked = checks.copy()
    checked["_canonical_id"] = checked.canonical_match_id.map(_id)
    if checked._canonical_id.isna().any() or checked._canonical_id.duplicated().any():
        raise ValueError("outcome checks must have unique non-null canonical fixture IDs")
    conflict = (~checked.source_team_id_pair_matches.eq(True) |
                checked.canonical_outcome.ne(checked.map_id_outcome))
    lookup = pd.Series(conflict.to_numpy(), index=checked._canonical_id)
    result = snapshots.copy()
    verdict = result.canonical_match_id.map(_id).map(lookup)
    result["outcome_source_conflict"] = verdict.eq(True)
    result["outcome_adjudication_basis"] = verdict.map(
        {True: "conflicting_result_sources", False: "map_team_ID_agreement"}
    ).fillna("not_independently_adjudicated")
    return result


def declared_date_cutoff_scenario(predictions: pd.DataFrame) -> pd.DataFrame:
    """Separate conditional cohort: a cutoff date is usable only after that UTC day ends.

    This does not authenticate input release, generation or issuance. Unknown/malformed
    cutoffs stay invalid; no earlier forecast is revived when the latest state is invalid.
    """
    result = predictions.copy()
    original = result.model_data_cutoff_utc.astype("string")
    dates = original.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
    parsed = pd.to_datetime(original.where(dates), format="%Y-%m-%d", utc=True, errors="coerce")
    upper = parsed + pd.Timedelta(days=1)
    result["declared_data_cutoff_original"] = result.model_data_cutoff_utc
    result.loc[dates, "model_data_cutoff_utc"] = upper.loc[dates].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    result["cutoff_interpretation"] = "original_timestamp_or_invalid"
    result.loc[dates, "cutoff_interpretation"] = "conditional_latest_included_UTC_day_end_bound"
    result["original_model_group"] = result.model_group
    result["model_group"] = result.model_group.astype(str) + ":DECLARED_DATE_UTC_UPPER_BOUND_SCENARIO"
    result["model_id"] = result.model_id.astype(str) + ":date-upper-bound-scenario"
    return result
