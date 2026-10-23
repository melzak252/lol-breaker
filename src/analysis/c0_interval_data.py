"""Prepare historical C0 interval checkpoints and genuine native forecasts.

All clocks are retrospective scenarios, not authenticated forecast issuance.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_BASE_PROTOCOL = Path(
    "data/05_model_input/c0_policy_evaluation/"
    "c0_full_policy_20261021_01_integrity_v2/protocol.json"
)
_BASE_INPUTS = Path(
    "data/08_reporting/c0_policy_evaluation/"
    "c0_full_policy_20261021_01_integrity_v2"
)
_GRID = pd.Timedelta(hours=2)
_WINDOW = pd.Timedelta(hours=48)


def _utc(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        value = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if value.tzinfo is None:
        return pd.NaT
    return value.tz_convert("UTC")


def _as_utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.map(_utc), utc=True)


def _checkpoint_id(match_id: Any, clock: pd.Timestamp) -> str:
    return f"c0-{match_id}-interval-{clock.value}"


def build_interval_checkpoints(
    fixtures: pd.DataFrame, raw_quotes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct UTC even-hour grid and raw-snapshot checkpoints.

    Returns every scheduled checkpoint and a one-row-per-fixture inventory. Raw
    quote timestamps are retained as observation clocks even if their quote
    records later fail identity, freshness, or other market gates.
    """
    if "canonical_match_id" not in fixtures or "start_at" not in fixtures:
        raise ValueError("fixtures require canonical_match_id and start_at")
    if fixtures.canonical_match_id.duplicated().any():
        raise ValueError("fixture input must contain one row per canonical match")
    quote_clock_column = "quoted_at" if "quoted_at" in raw_quotes else "scraped_at"
    if not raw_quotes.empty and not {"canonical_match_id", quote_clock_column}.issubset(raw_quotes):
        raise ValueError("raw quotes require canonical_match_id and quoted_at/scraped_at")
    if not raw_quotes.empty and "quote_id" not in raw_quotes and "id" not in raw_quotes:
        raise ValueError("raw quotes require quote_id or id")

    fixture_rows = fixtures.copy()
    fixture_rows["start_at"] = _as_utc_series(fixture_rows.start_at)
    quote_rows = raw_quotes.copy()
    if quote_rows.empty:
        quote_rows["_checkpoint_clock"] = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        quote_rows["_checkpoint_clock"] = _as_utc_series(quote_rows[quote_clock_column])
        quote_rows["_checkpoint_quote_id"] = quote_rows.get(
            "quote_id", quote_rows.get("id")
        )
    quote_groups = {
        match_id: group for match_id, group in quote_rows.groupby(
            "canonical_match_id", sort=False
        )
    } if not quote_rows.empty else {}

    checkpoints: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    for fixture in fixture_rows.to_dict("records"):
        match_id = fixture["canonical_match_id"]
        start = fixture["start_at"]
        grid_clocks: list[pd.Timestamp] = []
        snapshot_ids: dict[pd.Timestamp, list[Any]] = {}
        if not pd.isna(start):
            lower = start - _WINDOW
            first_ns = ((lower.value + _GRID.value - 1) // _GRID.value) * _GRID.value
            grid_clocks = list(pd.date_range(
                pd.Timestamp(first_ns, tz="UTC"), start, freq=_GRID, inclusive="left"
            ))
            match_quotes = quote_groups.get(match_id)
            if match_quotes is not None:
                observed = match_quotes.loc[
                    match_quotes._checkpoint_clock.notna()
                    & match_quotes._checkpoint_clock.ge(lower)
                    & match_quotes._checkpoint_clock.lt(start)
                ]
                for clock, group in observed.groupby("_checkpoint_clock", sort=True):
                    snapshot_ids[pd.Timestamp(clock)] = group["_checkpoint_quote_id"].tolist()
        all_clocks = sorted(set(grid_clocks).union(snapshot_ids))
        grid_set = set(grid_clocks)
        fixture_absence = (
            "schedule_unknown" if pd.isna(start)
            else "identity_unverified"
            if fixture.get("identity_status") != "verified"
            else None
        )
        for clock in all_clocks:
            is_grid = clock in grid_set
            ids = snapshot_ids.get(clock, [])
            row = dict(fixture)
            row.update(
                opportunity_id=_checkpoint_id(match_id, clock),
                decision_at=clock,
                checkpoint_at=clock,
                is_grid=is_grid,
                is_snapshot=bool(ids),
                snapshot_quote_ids=ids,
                hours_before_start=(start - clock).total_seconds() / 3600.0,
                checkpoint_absence=fixture_absence,
            )
            checkpoints.append(row)
        accounting.append({
            **{
                name: value for name, value in fixture.items()
                if name not in {"opportunity_id", "decision_at", "settle_at"}
            },
            "source_opportunity_id": fixture.get("opportunity_id"),
            "canonical_match_id": match_id,
            "start_at": start,
            "identity_status": fixture.get("identity_status"),
            "fixture_valid": fixture_absence is None,
            "grid_count": len(grid_clocks),
            "snapshot_count": len(snapshot_ids),
            "checkpoint_count": len(all_clocks),
            "checkpoint_absence": fixture_absence,
        })
    checkpoint_frame = pd.DataFrame(checkpoints)
    if checkpoint_frame.empty:
        checkpoint_frame = fixture_rows.iloc[0:0].copy()
        for name, dtype in (("decision_at", "datetime64[ns, UTC]"),
                            ("checkpoint_at", "datetime64[ns, UTC]"),
                            ("is_grid", bool), ("is_snapshot", bool),
                            ("snapshot_quote_ids", object),
                            ("hours_before_start", float),
                            ("checkpoint_absence", object)):
            checkpoint_frame[name] = pd.Series(dtype=dtype)
    else:
        checkpoint_frame = checkpoint_frame.sort_values(
            ["canonical_match_id", "decision_at"], kind="mergesort"
        ).reset_index(drop=True)
    accounting_frame = pd.DataFrame(accounting)
    return checkpoint_frame, accounting_frame


def _representative_opportunities(checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one actual checkpoint per fixture/UTC day with explicit proof key."""
    if checkpoints.empty:
        return checkpoints.copy(), pd.DataFrame(columns=[
            "representative_opportunity_id", "representative_decision_at",
            "utc_cutoff_day", "fanout_opportunity_ids", "native_daily_key_sha256",
        ])
    valid = checkpoints.loc[
        checkpoints.decision_at.notna()
        & checkpoints.start_at.notna()
        & checkpoints.identity_status.eq("verified")
        & checkpoints.golgg_match_id.notna()
        & checkpoints.team1_id.notna()
        & checkpoints.team2_id.notna()
        & checkpoints.canonical_a_is_team1.isin((True, False))
        & checkpoints.canonical_best_of.isin((1, 3, 5))
    ].copy()
    if valid.empty:
        return valid, pd.DataFrame(columns=[
            "representative_opportunity_id", "representative_decision_at",
            "utc_cutoff_day", "fanout_opportunity_ids", "native_daily_key_sha256",
        ])
    valid["_utc_cutoff_day"] = valid.decision_at.dt.strftime("%Y-%m-%d")
    contract_fields = [
        "canonical_match_id", "golgg_match_id", "team1_id", "team2_id",
        "canonical_a_is_team1", "canonical_best_of", "start_at",
        "identity_status", "_utc_cutoff_day",
    ]
    representatives = []
    manifest = []
    for _, group in valid.sort_values("decision_at", kind="mergesort").groupby(
        ["canonical_match_id", "_utc_cutoff_day"], sort=True
    ):
        if len(group[contract_fields].drop_duplicates()) != 1:
            raise ValueError("Fixture/day native identity contract is not stable")
        rep = group.iloc[0].copy()
        key = {name: str(rep[name]) for name in contract_fields}
        key.update(
            roster_policy="strictly_prior_day_exact_context_observed_scenario_not_official_announcement",
            mode="full",
        )
        digest = hashlib.sha256(
            repr(sorted(key.items())).encode("utf-8")
        ).hexdigest()
        representatives.append(rep.drop(labels=["_utc_cutoff_day"]))
        manifest.append({
            "representative_opportunity_id": rep.opportunity_id,
            "representative_decision_at": rep.decision_at,
            "utc_cutoff_day": rep["_utc_cutoff_day"],
            "fanout_opportunity_ids": group.opportunity_id.tolist(),
            "native_daily_key_sha256": digest,
            "native_daily_key": key,
        })
    return pd.DataFrame(representatives).reset_index(drop=True), pd.DataFrame(manifest)


def _bind_native_manifest(
    manifest: pd.DataFrame, rep_forecasts: pd.DataFrame
) -> pd.DataFrame:
    """Receipt the exact prior-roster/context contract and feature hashes."""
    if manifest.empty:
        return manifest
    forecasts = rep_forecasts.set_index("opportunity_id", drop=False)
    rows = []
    for item in manifest.to_dict("records"):
        row = forecasts.loc[item["representative_opportunity_id"]].to_dict()
        context = {}
        for side in ("a", "b"):
            source = row.get(f"prior_source_{side}")
            context[side] = source.get("competition_context") if isinstance(source, dict) else None
        contract = {
            "daily_key": item["native_daily_key"],
            "competition_context_by_side": context,
            "prior_roster_a": row.get("prior_roster_a"),
            "prior_roster_b": row.get("prior_roster_b"),
            "native_input_hashes": row.get("native_input_hashes"),
            "feature_source_max_day": row.get("feature_source_max_day"),
        }
        for name, value in contract.items():
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                contract[name] = None
        item["native_feature_contract"] = contract
        item["native_feature_contract_sha256"] = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        item["native_input_hashes"] = contract["native_input_hashes"]
        rows.append(item)
    return pd.DataFrame(rows)


def _run_native_worker(opportunities: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if opportunities.empty:
        return pd.DataFrame(columns=[
            "opportunity_id", "canonical_match_id", "p_base", "forecast_available",
            "identity_verified", "forecast_at", "native_absence",
        ]), {"prior_context_scenarios": 0, "genuine_native_forecasts": 0}
    with tempfile.TemporaryDirectory(prefix="c0-interval-native-") as temporary:
        source, destination = Path(temporary) / "input.pkl", Path(temporary) / "output.pkl"
        source.write_bytes(pickle.dumps(opportunities, protocol=pickle.HIGHEST_PROTOCOL))
        completed = subprocess.run(
            [sys.executable, "-m", "src.analysis.c0_policy_data", "--native-worker",
             str(source), str(destination)],
            text=True, capture_output=True, check=False,
            env={**os.environ, "BETTING_APP_C0_WORKER": "1"},
        )
        if completed.returncode:
            raise RuntimeError("Native interval input preparation failed: " + completed.stderr)
        return pickle.loads(destination.read_bytes())


def _fanout_daily_forecasts(
    checkpoints: pd.DataFrame, manifest: pd.DataFrame, rep_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    forecast_by_id = rep_forecasts.set_index("opportunity_id", drop=False)
    checkpoints_by_id = checkpoints.set_index("opportunity_id", drop=False)
    rows = []
    for item in manifest.to_dict("records"):
        representative_id = item["representative_opportunity_id"]
        source = forecast_by_id.loc[representative_id].to_dict()
        representative_clock = item["representative_decision_at"]
        for opportunity_id in item["fanout_opportunity_ids"]:
            target = checkpoints_by_id.loc[opportunity_id]
            row = dict(source)
            row.update(
                opportunity_id=opportunity_id,
                canonical_match_id=target.canonical_match_id,
                forecast_at=target.decision_at,
                forecast_clock_basis="daily_state_projection_not_issuance",
                daily_state_representative_opportunity_id=representative_id,
                daily_state_representative_at=representative_clock,
                daily_state_cutoff_day=item["utc_cutoff_day"],
                daily_state_key_sha256=item["native_daily_key_sha256"],
                native_feature_contract_sha256=item.get("native_feature_contract_sha256"),
                daily_state_projection=True,
                issuance_authenticated=False,
            )
            rows.append(row)
    forecast = pd.DataFrame(rows)
    if not checkpoints.empty:
        have = set(forecast.opportunity_id) if not forecast.empty else set()
        missing = checkpoints.loc[~checkpoints.opportunity_id.isin(have),
                                  ["opportunity_id", "canonical_match_id", "decision_at"]].copy()
        if not missing.empty:
            missing["p_base"] = np.nan
            missing["forecast_available"] = False
            missing["identity_verified"] = missing.canonical_match_id.map(
                checkpoints.drop_duplicates("canonical_match_id").set_index("canonical_match_id").identity_status
            ).eq("verified")
            missing["forecast_at"] = missing.decision_at
            missing["native_absence"] = "identity_schedule_context_or_format_unavailable"
            missing["forecast_clock_basis"] = "simulated_historical_origin_not_issuance"
            missing["issuance_authenticated"] = False
            forecast = pd.concat([forecast, missing.drop(columns="decision_at")], ignore_index=True)
    return forecast


def prepare_interval_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    """Prepare all interval checkpoints and genuine, isolated native C0 inputs."""
    if protocol.get("schema") != "c0_interval_policy_v1" or protocol.get("run_id") != "c0_interval_policy_20261022_01":
        raise ValueError("Wrong frozen interval policy protocol")
    if protocol.get("scope") != "HISTORICAL_DEVELOPMENT_ONLY" or protocol.get("formal_fits") != 0:
        raise ValueError("Interval preparation is restricted to zero-fit historical development")
    base_protocol_path = Path(protocol["base_protocol"])
    if base_protocol_path.resolve() != _BASE_PROTOCOL.resolve():
        raise ValueError("Interval base protocol differs from frozen C0 policy protocol")
    base_protocol = json.loads(base_protocol_path.read_text())
    from src.analysis import c0_policy_data

    c0_policy_data._validate_protocol(base_protocol)
    source_evidence = c0_policy_data._verify_artifacts()
    prepared_root = Path(protocol["prepared_inputs"])
    if prepared_root.resolve() != _BASE_INPUTS.resolve():
        raise ValueError("Interval input source differs from frozen prepared policy inputs")
    fixtures = pd.read_parquet(prepared_root / "opportunities.parquet")
    quotes = pd.read_parquet(prepared_root / "quotes.parquet")
    if len(fixtures) != 1771 or len(quotes) != 248735:
        raise ValueError("Frozen fixture or raw quote-state universe changed")
    checkpoints, accounting = build_interval_checkpoints(fixtures, quotes)
    representatives, manifest = _representative_opportunities(checkpoints)
    rep_forecasts, native_evidence = _run_native_worker(representatives)
    manifest = _bind_native_manifest(manifest, rep_forecasts)
    forecasts = _fanout_daily_forecasts(checkpoints, manifest, rep_forecasts)

    release_by_match = {}
    if not rep_forecasts.empty and "canonical_match_id" in rep_forecasts:
        for row in rep_forecasts.to_dict("records"):
            release_by_match[row["canonical_match_id"]] = row.get("native_effective_release_day")
    settlement_by_match = {}
    for fixture in fixtures.to_dict("records"):
        start, recorded = _utc(fixture.get("start_at")), _utc(fixture.get("result_recorded_at"))
        day = release_by_match.get(fixture["canonical_match_id"])
        midnight = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1) if day else (
            start.normalize() + pd.Timedelta(days=1) if not pd.isna(start) else pd.NaT
        )
        settlement_by_match[fixture["canonical_match_id"]] = max(
            (clock for clock in (start + pd.Timedelta(hours=6) if not pd.isna(start) else pd.NaT,
                                 recorded, midnight) if not pd.isna(clock)),
            default=pd.NaT,
        )
    if not checkpoints.empty:
        checkpoints["settle_at"] = checkpoints.canonical_match_id.map(settlement_by_match)
        checkpoints = checkpoints.merge(
            forecasts.drop(columns=["canonical_match_id"], errors="ignore"),
            on="opportunity_id", how="left", validate="one_to_one",
        )
    evidence = {
        "scope": "HISTORICAL_DEVELOPMENT_ONLY",
        "universe_count": len(fixtures),
        "checkpoint_count": len(checkpoints),
        "raw_quote_state_count": len(quotes),
        "authenticated_history": False,
        "authenticated_issuance": False,
        "forecast_available_count": int(forecasts.forecast_available.fillna(False).sum()) if not forecasts.empty else 0,
        "issuance_provenance": "Native daily state projected to each actual simulated checkpoint clock; not official issuance",
        "daily_representative_count": len(representatives),
        "daily_projection_checkpoint_count": int(forecasts.daily_state_projection.fillna(False).sum()) if not forecasts.empty and "daily_state_projection" in forecasts else 0,
        "representative_manifest": manifest,
        "fixture_accounting": accounting,
        "source_artifact_evidence": source_evidence,
        "native": native_evidence,
    }
    return {"opportunities": checkpoints, "quotes": quotes, "forecasts": forecasts,
            "evidence": evidence}
