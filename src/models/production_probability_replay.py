"""Read-only, research-labelled replay of the pinned fresh20 C0 ensemble."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd




def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("opportunity clocks must be timezone-aware UTC values")
    return stamp.tz_convert("UTC")

def _deduplicate_daily_cutoffs(panel: pd.DataFrame) -> pd.DataFrame:
    """Select earliest decision clock per fixture/UTC day, retaining quote fanout."""
    panel = panel.sort_values(["canonical_match_id", "decision_at", "opportunity_id"], kind="stable")
    groups = panel.groupby(
        [panel.canonical_match_id.astype(str), panel.decision_at.dt.date],
        sort=False,
        dropna=False,
    )
    keep = []
    fanout = {}
    for _, group in groups:
        for column in ("golgg_match_id", "team1_id", "team2_id", "best_of", "canonical_a_is_team1"):
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(f"conflicting {column} within fixture/day cutoff")
        first = group.iloc[0].copy()
        keep.append(first)
        fanout[str(first["opportunity_id"])] = len(group)
    result = pd.DataFrame(keep).reset_index(drop=True)
    result["source_join_fanout"] = result.opportunity_id.astype(str).map(fanout).astype(int)
    result["cache_reuse_permitted"] = False
    return result


def _canonical_probability(p_team_a: float, canonical_a_is_team1: bool) -> float:
    return float(p_team_a if canonical_a_is_team1 else 1.0 - p_team_a)


def _validate_opportunities(opportunities: pd.DataFrame) -> pd.DataFrame:
    opportunities = opportunities.copy()
    for target, source in (
        ("golgg_match_id", "source_match_id"),
        ("decision_at", "request_clock_utc"),
        ("best_of", "canonical_best_of"),
    ):
        if target not in opportunities and source in opportunities:
            opportunities[target] = opportunities[source]
    if "opportunity_id" not in opportunities and "row_id" in opportunities:
        opportunities["opportunity_id"] = opportunities["row_id"]
    required = {
        "opportunity_id", "canonical_match_id", "golgg_match_id", "decision_at",
        "start_at", "canonical_a_is_team1", "team1_id", "team2_id", "best_of",
        "identity_status",
    }
    missing = required.difference(opportunities.columns)
    if missing:
        raise ValueError(f"opportunities missing required columns: {sorted(missing)}")
    if opportunities.opportunity_id.isna().any() or opportunities.opportunity_id.astype(str).duplicated().any():
        raise ValueError("opportunity_id must be nonnull and unique")
    panel = opportunities.copy()
    panel["_replay_absence"] = None
    clocks, starts, best_of_values, orientations, absence_values = [], [], [], [], []
    for row in panel.to_dict("records"):
        reasons = []
        try:
            clock = _utc(row["decision_at"])
        except (TypeError, ValueError):
            clock = pd.NaT
            reasons.append("decision_clock_invalid_or_missing")
        try:
            start = _utc(row["start_at"]) if pd.notna(row["start_at"]) else pd.NaT
        except (TypeError, ValueError):
            start = pd.NaT
            reasons.append("start_clock_invalid")
        if pd.notna(clock) and pd.notna(start) and clock >= start:
            reasons.append("not_prestart")
        try:
            bo = int(row["best_of"])
            if bo not in (1, 3, 5):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            bo = 3
            reasons.append("best_of_unsupported")
        side = row["canonical_a_is_team1"]
        if isinstance(side, (bool, np.bool_)):
            orientation = bool(side)
        elif isinstance(side, str) and side.strip().lower() in ("true", "false"):
            orientation = side.strip().lower() == "true"
        else:
            orientation = False
            reasons.append("canonical_side_orientation_invalid")
        if pd.isna(row["canonical_match_id"]):
            reasons.append("canonical_match_id_missing")
        if row["identity_status"] != "verified":
            reasons.append("source_identity_" + str(row["identity_status"]))
        clocks.append(clock)
        starts.append(start)
        best_of_values.append(bo)
        orientations.append(orientation)
        absence_values.append(";".join(reasons) if reasons else None)
    panel["_replay_absence"] = absence_values
    # Positional assignment avoids depending on caller index labels.
    panel["decision_at"] = clocks
    panel["start_at"] = starts
    panel["best_of"] = best_of_values
    panel["canonical_a_is_team1"] = orientations
    panel["row_id"] = panel.opportunity_id.astype(str)
    panel["scenario_decision_at"] = panel.decision_at
    valid = panel.loc[panel._replay_absence.isna()].copy()
    invalid = panel.loc[panel._replay_absence.notna()].copy()
    if not valid.empty:
        valid = _deduplicate_daily_cutoffs(valid)
    if not invalid.empty:
        invalid["source_join_fanout"] = 1
        invalid["cache_reuse_permitted"] = False
    return pd.concat([valid, invalid], ignore_index=True)


def _base_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "opportunity_id": row["opportunity_id"],
        "canonical_match_id": row["canonical_match_id"],
        "golgg_match_id": row["golgg_match_id"],
        "identity_status": row["identity_status"],
        "decision_at": row["decision_at"],
        "model_clock_utc": row["decision_at"],
        "model_data_cutoff_utc": row["decision_at"].normalize() if pd.notna(row["decision_at"]) else pd.NaT,
        "scenario_clock_utc": row["decision_at"],
        "model_clock_scope": "same_UTC_day_counterfactual",
        "canonical_a_is_team1": row["canonical_a_is_team1"],
        "best_of": row["best_of"],
        "model_source_match_id": row["golgg_match_id"],
        "model_binding_status": "conditional_historical_replay",
        "model_binding_basis": "canonical_link_only",
        "model_id": "fresh20-regional-full",
        "source_join_fanout": int(row.get("source_join_fanout", 1)),
        "cache_reuse_permitted": False,
        "model_group": "production-fresh20-regional-research-replay",
        "forecast_available": False,
        "prediction_status": "unavailable",
        "p_team1": np.nan,
        "p_team2": np.nan,
        "native_absence": None,
        "forecast_clock_basis": "simulated_historical_origin_not_issuance",
        "issuance_authenticated": False,
    }


def run_production_probability_replay(
    opportunities: pd.DataFrame,
    *,
    raw_history_path: str | Path,
    release_map_path: str | Path,
    research_root: str | Path,
    chunk_size: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay every normalized opportunity using actual fresh20 + causal regional inputs.

    This is a local research replay, not evidence that the model issued forecasts at
    the historical decision clocks. Unavailable inputs remain explicit absences.
    """
    if not 1 <= int(chunk_size) <= 256:
        raise ValueError("chunk_size must be between 1 and 256")
    raw_path = Path(raw_history_path).resolve(strict=True)
    release_path = Path(release_map_path).resolve(strict=True)
    root = Path(research_root).resolve(strict=True)
    input_quote_count = len(opportunities)
    panel = _validate_opportunities(opportunities)
    started = time.perf_counter()
    was_worker = os.environ.get("BETTING_APP_C0_WORKER")
    old_raw = os.environ.get("C0_RAW_HISTORY_PATH")
    old_release = os.environ.get("C0_RELEASE_MAP_PATH")
    os.environ["BETTING_APP_C0_WORKER"] = "1"
    os.environ["C0_RAW_HISTORY_PATH"] = str(raw_path)
    os.environ["C0_RELEASE_MAP_PATH"] = str(release_path)
    try:
        from betting_app.core.models.native_c0 import NativeC0Adapter
        from src.analysis import ev_asof_replay as replay
        adapter = NativeC0Adapter()
        adapter.research_root = root
        identity = adapter.load()
        loaded_raw, raw_hash, release_hash, prepared = adapter._load_history()
        # Prior-roster selection and outcome-release exclusion are shared with the
        # established strict-prior research path; target rosters are never used.
        from src.analysis.c0_input_aligned_data import build_prior_targets
        release_days = {str(item.identifier): item.games[-1].day for item in prepared["series"]}
        event_context = {
            str(event["source_match_id"]): event["competition_context"]
            for event in prepared["roster_events"]
            if event.get("source_match_id") is not None and event.get("competition_context") is not None
        }
        eligible_panel = panel.loc[panel._replay_absence.isna()].copy()
        eligible_panel["competition_context"] = eligible_panel.golgg_match_id.astype(str).map(event_context)
        prior = (
            build_prior_targets(eligible_panel, prepared["roster_events"], target_native_release_days=release_days)
            if not eligible_panel.empty else pd.DataFrame()
        )
        requests = []
        regional_requests = []
        results = [
            {**_base_result(row), "native_absence": row["_replay_absence"],
             "forecast_generated_at": None}
            for row in panel.loc[panel._replay_absence.notna()].to_dict("records")
        ]
        raw_batches: list[dict[str, np.ndarray]] = []
        history_batches: list[dict[str, np.ndarray]] = []
        selected: list[dict[str, Any]] = []
        digests: dict[str, dict[str, str]] = {}
        regional_receipt = None
        prior_records = (
            prior.sort_values(["scenario_decision_at", "opportunity_id"]).to_dict("records")
            if not prior.empty else []
        )
        for row in prior_records:
            out = _base_result(row)
            out["forecast_generated_at"] = None
            if row["identity_status"] != "verified":
                out["native_absence"] = "source_identity_" + str(row["identity_status"])
            elif not row.get("forecast_eligible", False):
                out["native_absence"] = row.get("forecast_absence")
            elif not row.get("prior_roster_available", False):
                reasons = row.get("native_absence_reasons") or ["prior_roster_unavailable"]
                out["native_absence"] = ";".join(map(str, reasons))
            elif row.get("competition_context") is None:
                out["native_absence"] = "competition_context_missing"
            else:
                request = adapter._validate_request({
                    "team1_id": row["team1_id"], "team2_id": row["team2_id"],
                    "roster_a": row["roster_a"], "roster_b": row["roster_b"],
                    "best_of": int(row["best_of"]),
                    "decision_at": row["scenario_decision_at"].isoformat(),
                    "start_at": row["start_at"].isoformat() if pd.notna(row["start_at"]) else None,
                    "competition_context": row["competition_context"], "mode": "full",
                })
                requests.append(request)
                regional_requests.append({
                    "team1_id": request["team1_id"], "team2_id": request["team2_id"],
                    "roster_a": request["roster_a"], "roster_b": request["roster_b"],
                    "cutoff_day": request["decision_at"].date().isoformat(),
                })
                selected.append(out)
                out["prior_roster_a"] = row["roster_a"]
                out["prior_roster_b"] = row["roster_b"]
                out["prior_source_a"] = row["prior_source_a"]
                out["prior_source_b"] = row["prior_source_b"]
            results.append(out)

        regional_start = time.perf_counter()
        if regional_requests:
            from src.models.c0_regional_replay import build_regional_query_states
            regional_result = build_regional_query_states(
                regional_requests, raw_history_path=raw_path, release_map_path=release_path,
            )
            regional_receipt = regional_result["receipt"]
        else:
            regional_result = {"rows": []}
        regional_seconds = time.perf_counter() - regional_start
        if len(regional_result["rows"]) != len(requests):
            raise ValueError("regional replay did not return one causal state per eligible opportunity")
        feature_start = time.perf_counter()
        batch_indexes: list[int] = []
        for index, (request, result_row, regional) in enumerate(zip(
            requests, selected, regional_result["rows"], strict=True
        )):
            if regional.get("status") != "ok" or regional.get("offset_a") is None or regional.get("offset_b") is None:
                result_row["native_absence"] = "regional_state_unavailable"
                continue
            state = adapter._advance_history(request["decision_at"].date(), prepared)
            before_state = replay._logical_state_marker(state)
            target, raw, hist, provenance = adapter._raw_snapshot(request)
            if before_state != replay._logical_state_marker(state):
                raise ValueError("native feature query mutated historical state")
            request_id = str(result_row["opportunity_id"])
            digests[request_id] = adapter._array_digest({**raw, **hist})
            result_row.update(
                native_input_hashes=digests[request_id],
                feature_source_max_day=target["feature_source_max_day"],
                history_slot_count=int(hist["history"].shape[2]),
                history_mask_state=np.asarray(hist["mask"]).astype(bool).sum(axis=(0, 1)).tolist(),
                regional_status=regional["status"],
                regional_family_a=regional.get("family_a"),
                regional_family_b=regional.get("family_b"),
                regional_offset_a=regional["offset_a"], regional_offset_b=regional["offset_b"],
                source_provenance=provenance,
            )
            raw_batches.append(raw)
            history_batches.append(hist)
            batch_indexes.append(index)
            if len(raw_batches) >= chunk_size:
                _predict_batch(adapter, requests, selected, raw_batches, history_batches,
                               regional_result["rows"], batch_indexes)
                batch_indexes.clear()
        if raw_batches:
            _predict_batch(adapter, requests, selected, raw_batches, history_batches,
                           regional_result["rows"], batch_indexes)
        feature_and_inference_seconds = time.perf_counter() - feature_start
        return pd.DataFrame(results), {
            "schema": "production_probability_replay_v1",
            "model_identity": identity,
            "raw_history_path": str(loaded_raw), "raw_history_sha256": raw_hash,
            "release_map_path": str(prepared["release_path"]), "release_map_sha256": release_hash,
            "regional_replay_receipt": regional_receipt,
            "input_quote_count": input_quote_count,
            "opportunity_count": len(panel),
            "forecast_count": sum(bool(row["forecast_available"]) for row in results),
            "eligible_request_count": len(requests),
            "source_hashes": {"raw_history_sha256": raw_hash, "release_map_sha256": release_hash,
                              **identity.get("artifact_hashes", {})},
            "timings_seconds": {"total": time.perf_counter() - started,
                                "regional": regional_seconds,
                                "feature_and_inference": feature_and_inference_seconds},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "clock_policy": "scenario decision clocks; simulated historical origin, not authenticated issuance",
        }
    finally:
        for name, value in (("BETTING_APP_C0_WORKER", was_worker), ("C0_RAW_HISTORY_PATH", old_raw),
                            ("C0_RELEASE_MAP_PATH", old_release)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _predict_batch(
    adapter: Any,
    requests: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    raw_rows: list[dict[str, np.ndarray]],
    history_rows: list[dict[str, np.ndarray]],
    regional_rows: list[dict[str, Any]],
    indexes: list[int],
) -> None:
    """Predict an aligned bounded chunk with the adapter's pinned fresh20 module."""
    if not raw_rows:
        return
    if len(indexes) != len(raw_rows):
        raise ValueError("fresh20 batch indexes do not align with native inputs")
    all_raw = {key: np.stack([row[key] for row in raw_rows]) for key in raw_rows[0]}
    all_history = {key: np.stack([row[key] for row in history_rows]) for key in history_rows[0]}
    offsets_a = np.asarray([regional_rows[i]["offset_a"] for i in indexes], dtype=np.float32)
    offsets_b = np.asarray([regional_rows[i]["offset_b"] for i in indexes], dtype=np.float32)
    from src.analysis.c0_region_factorial import transform_batch_once
    transformed_raw, transformed_history = transform_batch_once(all_raw, all_history, offsets_a, offsets_b)
    for batch_index, request_index in enumerate(indexes):
        row = selected[request_index]
        row["model_input_hashes"] = adapter._array_digest({
            **{key: value[batch_index] for key, value in transformed_raw.items()},
            **{key: value[batch_index] for key, value in transformed_history.items()},
        })
    best_of = np.asarray([requests[i]["best_of"] for i in indexes], dtype=np.int64)
    probabilities = np.asarray(adapter._fresh_module._predict(
        adapter._models, {"raw": transformed_raw, "history": transformed_history}, best_of,
        "full", adapter._backend, adapter._shared, adapter._fresh_reference, adapter._calibration,
    ), dtype=np.float64).reshape(-1)
    if probabilities.shape != (len(indexes),) or not np.isfinite(probabilities).all() or ((probabilities <= 0) | (probabilities >= 1)).any():
        raise ValueError("fresh20 returned invalid batch probabilities")
    generated_at = datetime.now(timezone.utc).isoformat()
    for index, probability in zip(indexes, probabilities, strict=True):
        row = selected[index]
        row["p_team1"] = _canonical_probability(float(probability), row["canonical_a_is_team1"])
        row["p_team2"] = 1.0 - row["p_team1"]
        row["prediction_status"] = "valid"
        row["forecast_available"] = True
        row["native_absence"] = None
        row["forecast_generated_at"] = generated_at
    raw_rows.clear()
    history_rows.clear()
