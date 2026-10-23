"""Source-clock native C0 prior-roster forecast preparation for long-term market edges."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "long_term_market_edge_20261022_01"
YEARS = tuple(range(2021, 2027))
SOURCES = ("oddsportal", "oddspapi")
TENSOR_KEYS = ("core", "team", "players", "w20", "gates", "history", "mask", "champions")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _path(value: str | Path) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        return None
    return result.tz_convert("UTC")


def _finite_pair(row: pd.Series, first: str, second: str) -> bool:
    try:
        return all(math.isfinite(float(row[key])) for key in (first, second))
    except (TypeError, ValueError, KeyError):
        return False
def _best_of(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or int(number) not in (1, 3, 5):
        return None
    return int(number)


def _roster(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    result = [str(item).strip() for item in value]
    if len(result) != 5 or any(not item or item.lower() == "nan" for item in result) or len(set(result)) != 5:
        return None
    return sorted(result)


def _history_root() -> Path:
    from src.analysis.ev_asof_replay import _resolve_research_root
    return _resolve_research_root(None)


def _source_inputs(source: str, layout: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    relative = layout["sources"][source]["root"]
    source_root = _path(relative)
    quotes_path, events_path = source_root / "quotes.parquet", source_root / "events.parquet"
    quotes = pd.read_parquet(quotes_path)
    events = pd.read_parquet(events_path)
    if "source_year" not in quotes:
        if source == "oddsportal":
            quotes["source_year"] = pd.to_datetime(quotes.quote_at, utc=True, errors="coerce").dt.year
        else:
            quotes["source_year"] = pd.to_datetime(quotes.quote_at, utc=True, errors="coerce").dt.year
    quotes["_quote_index"] = np.arange(len(quotes), dtype=np.int64)
    return quotes, events, source_root


def _latest_start(row: pd.Series, source: str) -> pd.Timestamp | None:
    fields = ("scheduled_start_at", "true_start_at", "source_start_at") if source == "oddspapi" else ("source_start_at",)
    values = [_timestamp(row.get(field)) for field in fields]
    known = [value for value in values if value is not None]
    return min(known) if known else None


def _quote_clock(row: pd.Series, source: str) -> pd.Timestamp | None:
    if source == "oddsportal":
        fields = ("source_clock1", "source_clock2")
    else:
        fields = ("outcome_1_quote_at", "outcome_2_quote_at")
    values = [_timestamp(row.get(field)) for field in fields]
    if any(value is None for value in values):
        return None
    return max(values)


def _effective_release(metadata: dict[str, Any]) -> date | None:
    for name in ("effective_label_release_day", "result_release_day", "result_day"):
        value = metadata.get(name)
        if value is None or pd.isna(value):
            continue
        try:
            return pd.Timestamp(value).date()
        except (TypeError, ValueError):
            continue
    return None


def _paired_clock_regression_proof() -> dict[str, Any]:
    earlier = "2021-01-01T10:00:00Z"
    later = "2021-01-01T10:01:00Z"
    proof = {}
    for source, fields in (
        ("oddsportal", ("source_clock1", "source_clock2")),
        ("oddspapi", ("outcome_1_quote_at", "outcome_2_quote_at")),
    ):
        observed = _quote_clock(pd.Series({fields[0]: earlier, fields[1]: later}), source)
        expected = pd.Timestamp(later)
        if observed != expected:
            raise AssertionError(f"{source} paired-clock MAX preflight failed")
        proof[source] = {"side_clock_1": earlier, "side_clock_2": later, "selected_paired_clock": observed.isoformat()}
    return proof


def prepare_timed_forecasts(protocol_path, source_layout_path, controls_manifest_path, output_dir):
    """Bind all inputs, then issue source-clock native C0 forecasts and explicit absences."""
    started = time.monotonic()
    protocol_path, layout_path, controls_path, out = map(_path, (protocol_path, source_layout_path, controls_manifest_path, output_dir))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Timed native output directory must be fresh and empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    implementation_path = Path(__file__).resolve()
    implementation_bytes = implementation_path.read_bytes()
    implementation_sha = hashlib.sha256(implementation_bytes).hexdigest()
    implementation_snapshot = out / "long_term_native_inputs.py.snapshot"
    implementation_snapshot.write_bytes(implementation_bytes)
    protocol, layout, controls = _json(protocol_path), _json(layout_path), _json(controls_path)
    if protocol.get("run_id") != RUN_ID or protocol.get("scope", {}).get("years") != list(YEARS):
        raise ValueError("Frozen long-term protocol identity/year scope mismatch")
    root = _history_root()
    from src.analysis import ev_asof_replay as replay
    from src.models.tournament_native import _Worker
    from src.models.long_term_native_controls import load_verified_2023_control
    metadata_path = root / replay.BANK_RELATIVE / "metadata.parquet"
    metadata_frame = pd.read_parquet(metadata_path)
    metadata = replay._metadata_index(metadata_frame)
    os.environ["BETTING_APP_C0_WORKER"] = "1"
    worker = _Worker("c0")
    model_origins = {}
    for year in YEARS:
        if year == 2023:
            verified = load_verified_2023_control(root)
            if not verified.get("timed_inference_available") or verified.get("provider_adapter_entry") is None:
                raise ValueError("Verified 2023 control does not permit timed inference")
            worker.adapters[year] = verified["provider_adapter_entry"]
            model_origins[str(year)] = {
                "method": "load_verified_2023_control", "timed_inference_available": True,
                "artifact_contract": verified["provider_adapter_entry"]["contract"],
                "identity": verified["provider_adapter_entry"]["identity"],
            }
        else:
            model_root = controls.get("control_roots", {}).get(str(year))
            if not model_root:
                raise ValueError(f"controls manifest omits model root for {year}")
            contract = worker.identity(year, root, controls.get("control_roots", {}))
            model_origins[str(year)] = {"model_root": str(_path(model_root)), "artifact_contract": contract}
    history = worker._history_adapter(root)
    history._load_history()
    prepared = history._prepared
    native_releases = replay._target_native_release_inventory(prepared["roster_events"])
    # The complete quote and identity inputs are loaded exactly once, with hashes sealed before inference.
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame, Path]] = {}
    for source in SOURCES:
        sources[source] = _source_inputs(source, layout)
    binding = {
        "schema": "long_term_timed_native_input_binding_v1", "run_id": RUN_ID,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "source_layout": {"path": str(layout_path), "sha256": _sha256(layout_path)},
        "controls_manifest": {"path": str(controls_path), "sha256": _sha256(controls_path)},
        "history_root": str(root), "metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
        "native_history": {
            "raw_history_path": prepared["raw_path"], "raw_history_sha256": prepared["raw_sha256"],
            "release_map_path": prepared["release_path"], "release_map_sha256": prepared["release_sha256"],
        },
        "source_inputs": {}, "model_origins": model_origins,
        "clock_rule": "maximum of both finite paired source quote clocks; strictly before earliest known scheduled/true/source event start",
        "paired_clock_preflight_proof": _paired_clock_regression_proof(),
        "active_state": "unknown OddsPortal status remains unauthenticated source-endpoint scenario; no production eligibility asserted",
    }
    all_queries: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for source, (quotes, events, source_root) in sources.items():
        alias_value = layout["sources"][source].get("aliases")
        alias_path = _path(alias_value) if alias_value else None
        binding["source_inputs"][source] = {
            "root": str(source_root),
            "quotes_sha256": _sha256(source_root / "quotes.parquet"),
            "events_sha256": _sha256(source_root / "events.parquet"),
            "receipt_sha256": _sha256(source_root / ("source_receipt.json" if source == "oddsportal" else "receipt.json")),
            "binding_sha256": _sha256(source_root / "source_binding.json"),
            "aliases_path": None if alias_path is None else str(alias_path),
            "aliases_sha256": None if alias_path is None else _sha256(alias_path),
        }
        merged = quotes.loc[quotes.source_year.isin(YEARS)].merge(
            events, on=["source", "source_event_id"], how="left", suffixes=("", "_event"), validate="many_to_one")
        for year in YEARS:
            year_rows = merged.loc[merged.source_year.eq(year)]
            rows = []
            for _, row in year_rows.iterrows():
                reason = None
                eid = str(row.get("source_event_id"))
                mid = row.get("golgg_match_id")
                if pd.isna(mid) or row.get("identity_status") != "candidate_unique":
                    reason = "unknown_or_ambiguous_canonical_identity"
                clock = _quote_clock(row, source)
                start = _latest_start(row, source)
                if reason is None and clock is None:
                    reason = "missing_paired_finite_source_quote_clocks"
                if reason is None and start is None:
                    reason = "unknown_source_event_start"
                if reason is None and not clock < start:
                    reason = "source_quote_not_strictly_before_start"
                meta = metadata.get(str(mid), {}) if not pd.isna(mid) else {}
                decision_year = clock.year if clock is not None else year
                if reason is None and decision_year not in YEARS:
                    reason = "unsupported_model_origin_year"
                if reason is None and not _finite_pair(row, "odds_a", "odds_b"):
                    reason = "missing_nonfinite_paired_source_odds"
                bo = _best_of(row.get("best_of"))
                if reason is None and bo is None:
                    reason = "unknown_or_unsupported_best_of"
                team_a, team_b = row.get("canonical_team_a_id"), row.get("canonical_team_b_id")
                context = row.get("competition_context")
                if reason is None and (pd.isna(team_a) or pd.isna(team_b) or str(team_a) == str(team_b)):
                    reason = "unknown_or_invalid_canonical_teams"
                if reason is None and (not isinstance(context, str) or not context.strip()):
                    reason = "unknown_exact_competition_context"
                effective_release = native_releases.get(str(mid)) if not pd.isna(mid) else None
                if effective_release is None:
                    effective_release = _effective_release(meta)
                if reason is None and (effective_release is None or effective_release < clock.date()):
                    reason = "target_native_effective_release_unknown_or_precedes_decision"
                quote_original = quotes.iloc[int(row["_quote_index"])].drop(labels=["_quote_index"], errors="ignore").to_dict()
                quote_record_sha = hashlib.sha256(json.dumps(
                    quote_original, sort_keys=True, separators=(",", ":"), default=str,
                ).encode()).hexdigest()
                rows.append({
                    **{f"quote_{key}": value for key, value in quote_original.items()},
                    "quote_record_sha256": quote_record_sha,
                    "source": source, "quote_index": int(row["_quote_index"]), "source_event_id": eid,
                    "source_row": row.get("source_row"), "source_year": year,
                    "golgg_match_id": None if pd.isna(mid) else str(mid),
                    "decision_at": None if clock is None else clock.isoformat(),
                    "source_start_at": None if start is None else start.isoformat(),
                    "decision_year": decision_year,
                    "canonical_team_a_id": None if pd.isna(team_a) else str(team_a),
                    "canonical_team_b_id": None if pd.isna(team_b) else str(team_b),
                    "source_team_a_id": None if pd.isna(team_a) else str(team_a),
                    "source_team_b_id": None if pd.isna(team_b) else str(team_b),
                    "identity_orientation_swapped": row.get("orientation_swapped"),
                    "competition_context": context,
                    "best_of": bo,
                    "odds_a": row.get("odds_a"), "odds_b": row.get("odds_b"),
                    "quote_status": row.get("quote_status"),
                    "active_state_proven": isinstance(row.get("active_state_proven"), (bool, np.bool_)) and bool(row.get("active_state_proven")),
                    "authenticated": False, "production_eligible": False,
                    "eligibility_scope": "unauthenticated_source_endpoint_scenario",
                    "forecast_absence": reason, "forecast_key": None,
                })
                if reason is None:
                    rows[-1]["_request"] = {
                        "team1_id": str(team_a), "team2_id": str(team_b), "best_of": bo,
                        "decision_at": clock.isoformat(), "start_at": start.isoformat(),
                        "competition_context": context.strip(), "mode": "full",
                    }
            all_queries[(source, year)] = rows
    indexed_events = replay._index_roster_events(prepared["roster_events"])
    binding["implementation"] = {
        "path": str(implementation_path), "sha256": implementation_sha,
        "snapshot_path": str(implementation_snapshot), "snapshot_sha256": implementation_sha,
    }
    binding["native_history"] = {
        "raw_history_path": prepared["raw_path"], "raw_history_sha256": prepared["raw_sha256"],
        "release_map_path": prepared["release_path"], "release_map_sha256": prepared["release_sha256"],
    }
    binding_path = out / "input_binding.json"
    binding_path.write_text(json.dumps(binding, sort_keys=True, indent=2, default=str, allow_nan=False) + "\n")
    binding_sha = _sha256(binding_path)
    ordered_requests = sorted(
        ((row["decision_at"], source, year, row) for (source, year), rows in all_queries.items()
         for row in rows if row["forecast_absence"] is None),
        key=lambda item: (item[0], item[1], item[3]["source_event_id"], item[3]["quote_index"]),
    )
    roster_cache: dict[tuple[str, str, date], list[str] | None] = {}
    plans = []
    unique_plans = {}
    for _, source, year, row in ordered_requests:
        request = row["_request"]
        cutoff = pd.Timestamp(request["decision_at"]).date()
        for team in (request["team1_id"], request["team2_id"]):
            prior_key = (team, request["competition_context"], cutoff)
            if prior_key not in roster_cache:
                selected = replay.select_prior_roster(indexed_events, team, request["competition_context"], cutoff)
                roster_cache[prior_key] = _roster(selected)
        roster_a = roster_cache[(request["team1_id"], request["competition_context"], cutoff)]
        roster_b = roster_cache[(request["team2_id"], request["competition_context"], cutoff)]
        if roster_a is None or roster_b is None or set(roster_a) & set(roster_b):
            row["forecast_absence"] = "five_disjoint_strictly_prior_exact_context_rosters_unavailable"
            continue
        native_request = {**request, "roster_a": roster_a, "roster_b": roster_b}
        request_identity = {
            "source": source, "source_year": year, "source_event_id": row["source_event_id"],
            "golgg_match_id": row["golgg_match_id"], "native_request": native_request,
        }
        request_sha = hashlib.sha256(json.dumps(request_identity, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        model_year = pd.Timestamp(request["decision_at"]).year
        model_identity = binding["model_origins"][str(model_year)]
        contract_sha = hashlib.sha256(json.dumps(model_identity, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        native_state = {
            "source": source, "decision_day_utc": cutoff.isoformat(), "model_year": model_year,
            "team1_id": request["team1_id"], "team2_id": request["team2_id"],
            "best_of": request["best_of"], "competition_context": request["competition_context"],
            "roster_a": roster_a, "roster_b": roster_b, "mode": request["mode"],
            "raw_history_sha256": prepared["raw_sha256"], "release_map_sha256": prepared["release_sha256"],
            "model_identity_sha256": contract_sha, "implementation_sha256": implementation_sha,
        }
        native_state_sha = hashlib.sha256(json.dumps(native_state, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        source_closure_sha = hashlib.sha256(json.dumps({
            "source": source, "source_year": year, "source_event_id": row["source_event_id"],
            "golgg_match_id": row["golgg_match_id"], "quote_record_sha256": row["quote_record_sha256"],
            "source_row": row["source_row"],
        }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        row["full_request_sha256"] = request_sha
        row["source_closure_sha256"] = source_closure_sha
        row["native_state_sha256"] = native_state_sha
        row["forecast_key"] = native_state_sha
        plan = (native_state_sha, source, year, model_year, row, native_request, cutoff, request_sha, source_closure_sha)
        plans.append(plan)
        unique_plans.setdefault(native_state_sha, plan)
    parity_candidate = next(
        (plan for plan in plans
         if plan[4]["decision_at"] != unique_plans[plan[0]][4]["decision_at"]),
        None,
    )
    parity = {"checked": False, "reason": "no repeated same-day native feature state with a later clock"}
    print(
        f"bound timed requests: quote_requests={len(ordered_requests)} roster_valid={len(plans)} "
        f"unique_native_feature_states={len(unique_plans)}; starting source-clock inference",
        flush=True,
    )
    cache = {}
    for state_sha, source, year, model_year, row, native_request, cutoff, request_sha, source_closure_sha in unique_plans.values():
        try:
            result = worker.predict(native_request, root, controls.get("control_roots", {}))
        except Exception as error:
            from src.models.tournament_native import NativeArtifactUnavailable
            if isinstance(error, NativeArtifactUnavailable):
                row["forecast_absence"] = str(error)
                continue
            raise
        proof = result["provenance"]
        tensors = proof["input_hashes"]
        missing = set(TENSOR_KEYS) - set(tensors)
        if missing:
            raise ValueError(f"native query proof omits tensor hashes: {sorted(missing)}")
        prior_a = _prior_events(indexed_events, native_request["team1_id"], native_request["competition_context"], cutoff)
        prior_b = _prior_events(indexed_events, native_request["team2_id"], native_request["competition_context"], cutoff)
        cache[state_sha] = {
            "native_state_sha256": state_sha, "representative_native_state_sha256": state_sha,
            "forecast_key": state_sha,
            "representative_full_request_sha256": request_sha,
            "source_closure_sha256": source_closure_sha,
            "raw_history_sha256": prepared["raw_sha256"], "release_map_sha256": prepared["release_sha256"],
            "source": source, "source_year": year, "source_event_id": row["source_event_id"],
            "golgg_match_id": row["golgg_match_id"], "decision_at": row["decision_at"],
            "source_start_at": row["source_start_at"], "projection_day_utc": cutoff.isoformat(),
            "daily_feature_state_projection": True, "model_year": model_year,
            "best_of": native_request["best_of"], "competition_context": native_request["competition_context"],
            "p_c0": proof["mode_probabilities"]["full"], "mode_probabilities": proof["mode_probabilities"],
            "source_team_a_id": native_request["team1_id"], "source_team_b_id": native_request["team2_id"],
            "prior_roster_a": native_request["roster_a"], "prior_roster_b": native_request["roster_b"],
            "prior_source_ids_a": [event.get("source_match_id") for event in prior_a],
            "prior_source_ids_b": [event.get("source_match_id") for event in prior_b],
            "prior_release_days_a": [event.get("effective_release_day") for event in prior_a],
            "prior_release_days_b": [event.get("effective_release_day") for event in prior_b],
            "tensor_hashes": {key: tensors[key] for key in TENSOR_KEYS},
            "last_feature_day": proof["native_history_provenance"].get("feature_source_max_day"),
            "artifact_contract_sha256": hashlib.sha256(json.dumps(proof["artifact_contract"], sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest(),
            "model_identity_sha256": hashlib.sha256(json.dumps(proof["model_identity"], sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest(),
            "authenticated": False,
        }
        if parity_candidate is not None and parity_candidate[0] == state_sha:
            later = parity_candidate
            from src.models.tournament_native import ResearchNativePairProvider
            normalized = ResearchNativePairProvider.validate_request(later[5])
            _, raw_later, hist_later, _ = history._raw_snapshot(normalized)
            later_hashes = history._array_digest({**raw_later, **hist_later})
            if later_hashes != cache[state_sha]["tensor_hashes"]:
                raise ValueError("Same-day later native request does not match earliest representative tensor hashes")
            parity = {
                "checked": True, "native_state_sha256": state_sha,
                "representative_decision_at": row["decision_at"],
                "later_actual_decision_at": later[4]["decision_at"], "tensor_hashes_equal": True,
            }
            parity_candidate = None
    for state_sha, _, _, _, representative, _, _, _, _ in unique_plans.values():
        if state_sha not in cache:
            absence = representative.get("forecast_absence") or "native_forecast_unavailable"
            for plan in plans:
                if plan[0] == state_sha:
                    plan[4]["forecast_absence"] = absence
                    plan[4]["forecast_key"] = None
    if hashlib.sha256(implementation_path.read_bytes()).hexdigest() != implementation_sha:
        raise RuntimeError("Timed native implementation changed after its durable pre-inference binding")
    forecasts = list(cache.values())
    for (source, year), rows in all_queries.items():
        forecast_rows = [item for item in forecasts if item["source"] == source and item["source_year"] == year]
        query_records = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
        query_frame = pd.DataFrame(query_records)
        if not query_records:
            query_frame = pd.DataFrame(columns=[
                *(f"quote_{key}" for key in sources[source][0].columns if key != "_quote_index"),
                "source", "quote_index", "source_event_id", "source_row", "source_year",
                "golgg_match_id", "decision_at", "source_start_at", "decision_year",
                "canonical_team_a_id", "canonical_team_b_id", "source_team_a_id", "source_team_b_id",
                "identity_orientation_swapped", "competition_context", "best_of", "odds_a", "odds_b",
                "quote_record_sha256", "full_request_sha256", "source_closure_sha256", "native_state_sha256",
                "quote_status", "active_state_proven", "authenticated", "production_eligible",
                "eligibility_scope", "forecast_absence", "forecast_key",
            ])
        query_path = out / f"{source}_{year}_quote_query_map.parquet"
        query_frame.to_parquet(query_path, index=False)
        print(f"{source} {year}: quotes={len(rows)} forecasts={len(forecast_rows)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    # Write model identity once/year and bulky prediction evidence once/cache-state.
    model_manifest = {"schema": "long_term_timed_native_model_manifest_v1", "run_id": RUN_ID, "origins": binding["model_origins"]}
    (out / "model_manifest.json").write_text(json.dumps(model_manifest, sort_keys=True, indent=2, default=str) + "\n")
    forecasts_path = out / "timed_prior_roster_forecasts.parquet"
    forecast_frame = pd.DataFrame(forecasts)
    if forecast_frame.empty:
        forecast_frame = pd.DataFrame(columns=[
            "native_state_sha256", "representative_native_state_sha256", "representative_full_request_sha256",
            "forecast_key", "source", "source_year", "source_event_id", "golgg_match_id",
            "decision_at", "source_start_at", "model_year", "best_of", "competition_context",
            "p_c0", "mode_probabilities",
            "source_team_a_id", "source_team_b_id", "prior_roster_a", "prior_roster_b",
            "prior_source_ids_a", "prior_source_ids_b", "prior_release_days_a", "prior_release_days_b",
            "tensor_hashes", "last_feature_day", "artifact_contract_sha256",
            "model_identity_sha256", "raw_history_sha256", "release_inventory_sha256",
            "source_closure_sha256", "full_request_sha256", "daily_feature_state_projection",
            "authenticated",
        ])
    forecast_frame.to_parquet(forecasts_path, index=False)
    # Final receipt is written only after every source-year query artifact is durable.
    receipt = {"schema": "long_term_timed_native_inputs_receipt_v1", "run_id": RUN_ID,
        "input_binding_sha256": binding_sha, "input_binding_path": str(binding_path),
        "model_manifest_sha256": _sha256(out / "model_manifest.json"), "forecast_path": str(forecasts_path),
        "forecast_sha256": _sha256(forecasts_path), "source_year_counts": {}, "elapsed_seconds": time.monotonic()-started,
        "history_series_count": len(prepared["series"]), "roster_event_count": len(prepared["roster_events"]),
        "history_loaded_once": True, "active_state_authentication": "not asserted; source scenarios only",
        "requested_quote_count": len(ordered_requests), "roster_valid_request_count": len(plans),
        "unique_native_feature_state_count": len(unique_plans), "daily_projection_parity": parity}
    for (source, year), rows in all_queries.items():
        query_path = out / f"{source}_{year}_quote_query_map.parquet"
        receipt["source_year_counts"][f"{source}:{year}"] = {
            "quote_rows": len(rows), "forecasts": sum(row.get("forecast_absence") is None and row.get("forecast_key") is not None for row in rows),
            "absences": sum(row.get("forecast_absence") is not None for row in rows),
            "query_path": str(query_path), "query_sha256": _sha256(query_path)}
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, default=str) + "\n")
    return receipt


def _prior_events(indexed: dict[str, list[dict[str, Any]]], team: str, context: str, cutoff: date) -> list[dict[str, Any]]:
    from src.analysis.ev_asof_replay import _canonical_competition_context, _as_day
    return [event for event in indexed.get(team, ())
            if _canonical_competition_context(event.get("competition_context")) == context
            and _as_day(event["effective_release_day"], "release") < cutoff][-1:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=f"data/05_model_input/long_term_market_edge/{RUN_ID}/protocol_policy_v2.json")
    parser.add_argument("--source-layout", default=f"data/05_model_input/long_term_market_edge/{RUN_ID}/source_layout_integrity_v3.json")
    parser.add_argument("--controls-manifest", default=f"data/08_reporting/long_term_market_edge/{RUN_ID}/annual_c0/manifest.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    receipt = prepare_timed_forecasts(args.protocol, args.source_layout, args.controls_manifest, args.output_dir)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
