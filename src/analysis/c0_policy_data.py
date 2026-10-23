from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_RAW = Path("data/01_raw/c0_production_counterfactual/c0_production_20261006_01")
_REPORT = Path(
    "data/08_reporting/c0_production_counterfactual/c0_production_20261006_01"
)


def _utc(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if parsed.tzinfo is None:
        return pd.NaT
    return parsed.tz_convert("UTC")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def admit_policy_quotes(
    opportunities: pd.DataFrame, quotes: pd.DataFrame, ttl_seconds: int = 300
) -> pd.DataFrame:
    """Admit each book's latest observed, identity-aligned state as of the origin."""
    columns = [
        "quote_id",
        "opportunity_id",
        "canonical_match_id",
        "bookmaker_id",
        "bookmaker",
        "quoted_at",
        "canonical_odds_a",
        "canonical_odds_b",
        "identity_verified",
        "raw_a_is_canonical_a",
        "is_live",
        "implied_a",
        "implied_b",
        "freshness_seconds",
    ]
    if ttl_seconds < 0:
        raise ValueError("ttl_seconds must be nonnegative")
    if opportunities.empty or quotes.empty:
        return pd.DataFrame(columns=columns)
    required = {"opportunity_id", "canonical_match_id", "decision_at"}
    if not required.issubset(opportunities.columns):
        raise ValueError(
            f"opportunities missing columns: {sorted(required - set(opportunities.columns))}"
        )
    opp = opportunities[["opportunity_id", "canonical_match_id", "decision_at"]].copy()
    opp["decision_at"] = opp["decision_at"].map(_utc)
    q = quotes.copy()
    rename = {}
    for target, candidates in {
        "quote_id": ("id",),
        "quoted_at": ("scraped_at",),
        "bookmaker_id": ("bookmaker",),
        "odds_a": ("raw_odds_a",),
        "odds_b": ("raw_odds_b",),
    }.items():
        if target not in q:
            candidate = next((name for name in candidates if name in q), None)
            if candidate:
                rename[candidate] = target
    q = q.rename(columns=rename)
    required_q = {
        "quote_id",
        "canonical_match_id",
        "bookmaker_id",
        "quoted_at",
        "odds_a",
        "odds_b",
        "identity_verified",
        "raw_a_is_canonical_a",
        "is_live",
    }
    if not required_q.issubset(q.columns):
        raise ValueError(
            f"quotes missing columns: {sorted(required_q - set(q.columns))}"
        )
    if "bookmaker" not in q:
        q["bookmaker"] = q["bookmaker_id"]
    if "market_type" in q:
        q = q[q["market_type"].eq("match_winner")]
    q["quoted_at"] = q["quoted_at"].map(_utc)
    joined = opp.merge(q, on="canonical_match_id", how="inner", suffixes=("", "_quote"))
    joined["freshness_seconds"] = (
        joined["decision_at"] - joined["quoted_at"]
    ).dt.total_seconds()
    # Exclude future records before latest-state selection: they are not yet observed.
    joined = joined[
        joined["quoted_at"].notna()
        & joined["decision_at"].notna()
        & joined["quoted_at"].le(joined["decision_at"])
    ]
    if joined.empty:
        return pd.DataFrame(columns=columns)
    # Deterministic tie handling: an ambiguous same-clock state blocks the book.
    joined = joined.sort_values(
        ["opportunity_id", "bookmaker_id", "quoted_at", "quote_id"],
        kind="mergesort",
        na_position="first",
    )
    latest = joined.drop_duplicates(
        ["opportunity_id", "bookmaker_id"], keep="last"
    ).copy()
    latest_keys = latest[["opportunity_id", "bookmaker_id", "quoted_at"]]
    ambiguous = joined.merge(
        latest_keys, on=["opportunity_id", "bookmaker_id", "quoted_at"], how="inner"
    )
    ambiguous_keys = (
        ambiguous.groupby(["opportunity_id", "bookmaker_id"])["quote_id"]
        .nunique()
        .gt(1)
    )
    # Whether the latest selected timestamp has conflicting quote IDs, independent of input order.
    conflict = set(ambiguous_keys[ambiguous_keys].index.tolist())

    def known_bool(value: Any, *, false_only: bool = False) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return not bool(value) if false_only else bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return not bool(value) if false_only else bool(value)
        return False

    verified = latest["identity_verified"].map(known_bool)
    oriented = latest["raw_a_is_canonical_a"].map(
        lambda x: (
            isinstance(x, (bool, np.bool_))
            or (isinstance(x, (int, np.integer)) and x in (0, 1))
        )
    )
    live = latest["is_live"].map(lambda x: known_bool(x, false_only=True))
    odds_a = pd.to_numeric(latest["odds_a"], errors="coerce")
    odds_b = pd.to_numeric(latest["odds_b"], errors="coerce")
    valid = (
        latest["freshness_seconds"].between(0, ttl_seconds, inclusive="both")
        & verified
        & oriented
        & live
        & np.isfinite(odds_a)
        & np.isfinite(odds_b)
        & odds_a.gt(1)
        & odds_b.gt(1)
    )
    if conflict:
        valid &= ~pd.MultiIndex.from_frame(
            latest[["opportunity_id", "bookmaker_id"]]
        ).isin(conflict)
    admitted = latest.loc[valid].copy()
    swap = ~admitted["raw_a_is_canonical_a"].astype(bool)
    admitted["canonical_odds_a"] = odds_a.loc[valid].where(~swap, odds_b.loc[valid])
    admitted["canonical_odds_b"] = odds_b.loc[valid].where(~swap, odds_a.loc[valid])
    admitted["implied_a"] = 1.0 / admitted["canonical_odds_a"]
    admitted["implied_b"] = 1.0 / admitted["canonical_odds_b"]
    return admitted.reindex(columns=columns).reset_index(drop=True)


def _verify_artifacts() -> dict[str, Any]:
    """Verify saved native and identity artifacts against their immutable receipts."""
    seal_path = _REPORT / "seal.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("status") != "SEALED" or not isinstance(seal.get("files"), list):
        raise ValueError("native artifact seal is invalid")
    for item in seal["files"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file() or _sha(path) != item.get("sha256"):
            raise ValueError(f"sealed native artifact mismatch: {path}")
    identity_path = _REPORT / "native_identity.json"
    identity = json.loads(identity_path.read_text())
    expected = (
        "Causal-C0",
        "c0-native-2026-w32-e12-v1",
        "native_c0",
        "native-c0-history16-v1",
        "a0_data_focus_20260929_192443",
        "control",
    )
    observed = (
        identity.get("name"),
        identity.get("version"),
        identity.get("family"),
        identity.get("feature_version"),
        identity.get("run_id"),
        identity.get("variant"),
    )
    if observed != expected:
        raise ValueError(
            "saved native model identity does not match the frozen C0 identity"
        )
    identity_receipt = json.loads((_REPORT / "identity_receipt.json").read_text())
    if (
        identity_receipt.get("identity_only") is not True
        or identity_receipt.get("outcomes_used_for_alignment") is not False
    ):
        raise ValueError(
            "saved identity receipt is not outcome-blind identity evidence"
        )
    alignment_path = _REPORT / "quote_alignment.parquet"
    outputs = identity_receipt.get("output_sha256", {})
    # Older v1 receipts did not include output hashes; the parser compatibility is
    # explicit and does not upgrade their identity/authentication claims.
    if outputs:
        if outputs.get("quote_alignment.parquet") != _sha(alignment_path):
            raise ValueError("quote alignment receipt hash mismatch")
        receipt_status = "identity_v2_outputs_verified_by_receipt"
    else:
        if identity_receipt.get("version") not in {
            "v1",
            "v1_legacy",
            "v2_coherent_id_name_pairs",
        }:
            raise ValueError(
                "identity receipt omits required output hashes and has an unknown schema"
            )
        receipt_status = (
            "legacy_v1_parser_compatibility_only"
            if identity_receipt["version"] in {"v1", "v1_legacy"}
            else "identity_v2_artifacts_verified_by_seal"
        )
    from betting_app.core.models.native_c0 import load_serving_config

    config, config_path, config_sha = load_serving_config()
    if config.get("source_contract_version") != "native-c0-serving-source-v2":
        raise ValueError("native serving source contract is not v2")
    replay_identity = identity.get("raw_feature_replay_source", {})
    replay_hash = (
        replay_identity.get("implementation_sha256")
        if isinstance(replay_identity, dict)
        else None
    )
    return {
        "native_identity": identity,
        "identity_receipt_status": receipt_status,
        "current_source_contract_version": config["source_contract_version"],
        "cached_replay_sha256": replay_hash,
        "cached_replay_matches_current": replay_hash
        == config.get("feature_replay_sha256"),
        "seal_sha256": _sha(seal_path),
        "identity_receipt_sha256": _sha(_REPORT / "identity_receipt.json"),
        "native_identity_sha256": _sha(identity_path),
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("run_id")
        not in {"c0_full_policy_20261021_01", "c0_full_policy_20261021_01_integrity_v2"}
        or protocol.get("scope") != "HISTORICAL_DEVELOPMENT_ONLY"
        or protocol.get("formal_fits") != 0
    ):
        raise ValueError(
            "policy protocol does not match the frozen C0 financial evaluation"
        )
    inputs = protocol.get("inputs", {})
    if (
        inputs.get("raw_snapshot") != _RAW.as_posix()
        or inputs.get("identity_snapshot") != _REPORT.as_posix()
    ):
        raise ValueError(
            "policy protocol input snapshots differ from the frozen source artifacts"
        )


def _canonical_request(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    keys = {
        "team1_id",
        "team2_id",
        "roster_a",
        "roster_b",
        "best_of",
        "decision_at",
        "competition_context",
        "start_at",
        "mode",
    }
    if set(value) != keys:
        return None
    return {
        "team1_id": str(value["team1_id"]),
        "team2_id": str(value["team2_id"]),
        "roster_a": sorted(map(str, value["roster_a"])),
        "roster_b": sorted(map(str, value["roster_b"])),
        "best_of": value["best_of"],
        "decision_at": _utc(value["decision_at"]),
        "competition_context": value["competition_context"],
        "start_at": _utc(value["start_at"]),
        "mode": value["mode"],
    }


def prepare_policy_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    """Retain the complete snapshot and generate causal, zero-fit C0 scenarios."""
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    _validate_protocol(protocol)
    receipt = _verify_artifacts()
    matches = pd.read_parquet(_RAW / "matches.parquet")
    quotes = pd.read_parquet(_RAW / "quotes.parquet")
    books = pd.read_parquet(_RAW / "bookmakers.parquet")
    events = pd.read_parquet(_RAW / "events.parquet")
    identities = pd.read_parquet(_REPORT / "identities.parquet")
    alignment = pd.read_parquet(_REPORT / "quote_alignment.parquet")
    if len(matches) != 1771 or matches.id.duplicated().any() or len(quotes) != 248735:
        raise ValueError("Frozen opportunity/quote universe changed")
    aligned = alignment[
        ["id", "canonical_match_id", "quote_identity_status", "raw_a_is_canonical_a"]
    ].rename(columns={"canonical_match_id": "alignment_match_id"})
    quotes = quotes.merge(aligned, on="id", validate="one_to_one")
    quotes["identity_verified"] = quotes.canonical_match_id.eq(
        quotes.alignment_match_id
    ) & quotes.quote_identity_status.eq("verified")
    quotes["quote_id"], quotes["quoted_at"] = quotes["id"], quotes["scraped_at"]
    quotes = quotes.merge(
        books[["id", "name"]].rename(
            columns={"id": "bookmaker_id", "name": "bookmaker"}
        ),
        on="bookmaker_id",
        validate="many_to_one",
    )
    starts = (
        events.assign(_start=events.match_start_time.map(_utc))
        .groupby("canonical_match_id")
        ._start.min()
    )
    rows = []
    for row in matches.to_dict("records"):
        clocks = [starts.get(row["id"], pd.NaT), _utc(row.get("start_time_normalized"))]
        clocks = [value for value in clocks if not pd.isna(value)]
        start = min(clocks) if clocks else pd.NaT
        rows.append(
            {
                "opportunity_id": f"c0-{row['id']}-t1h",
                "canonical_match_id": row["id"],
                "decision_at": start - pd.Timedelta(hours=1),
                "start_at": start,
                "league": row.get("league"),
                "best_of": row.get("best_of"),
                "y": {"team_a": 1.0, "team_b": 0.0, "A": 1.0, "B": 0.0}.get(
                    row.get("winner_side"), np.nan
                ),
                "result_recorded_at": _utc(row.get("result_recorded_at")),
                "absence_flags": ["schedule_unknown"] if pd.isna(start) else [],
            }
        )
    opportunities = pd.DataFrame(rows).merge(
        identities[
            [
                "canonical_match_id",
                "golgg_match_id",
                "identity_status",
                "canonical_a_is_team1",
                "team1_id",
                "team2_id",
                "canonical_best_of",
            ]
        ],
        on="canonical_match_id",
        how="left",
        validate="one_to_one",
    )
    with tempfile.TemporaryDirectory(prefix="c0-policy-inputs-") as temporary:
        source, destination = (
            Path(temporary) / "input.pkl",
            Path(temporary) / "output.pkl",
        )
        source.write_bytes(pickle.dumps(opportunities))
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.analysis.c0_policy_data",
                "--native-worker",
                str(source),
                str(destination),
            ],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "BETTING_APP_C0_WORKER": "1"},
        )
        if completed.returncode:
            raise RuntimeError(
                "Native policy input preparation failed: " + completed.stderr
            )
        forecasts, native_evidence = pickle.loads(destination.read_bytes())
    release = forecasts.set_index("opportunity_id")[
        "native_effective_release_day"
    ].to_dict()
    settlements = []
    for row in opportunities.to_dict("records"):
        start, recorded = row["start_at"], row["result_recorded_at"]
        day = release.get(row["opportunity_id"])
        midnight = (
            pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
            if day
            else (
                start.normalize() + pd.Timedelta(days=1)
                if not pd.isna(start)
                else pd.NaT
            )
        )
        clocks = [start + pd.Timedelta(hours=6), recorded, midnight]
        settlements.append(
            max((value for value in clocks if not pd.isna(value)), default=pd.NaT)
        )
    opportunities["settle_at"] = settlements
    evidence = {
        "scope": "HISTORICAL_DEVELOPMENT_ONLY",
        "universe_count": len(opportunities),
        "quote_count": len(quotes),
        "authenticated_history": False,
        "authenticated_issuance": False,
        "forecast_available_count": int(forecasts.forecast_available.sum()),
        "issuance_provenance": "Actual generation clock retained separately from simulated historical decision clock",
        "exp039_status": "not_evaluated",
        "rating_disagreement_status": "not_evaluated",
        "saved_native_identity_sha256": receipt["native_identity_sha256"],
        "native": native_evidence,
    }
    return {
        "opportunities": opportunities,
        "quotes": quotes,
        "forecasts": forecasts,
        "evidence": evidence,
    }


def _native_worker(source: Path, destination: Path) -> None:
    """Isolated native namespace; no archive objects cross the process boundary."""
    import pickle
    from datetime import datetime, timezone

    from src.analysis.c0_input_aligned_data import build_prior_targets
    from betting_app.core.models.native_c0 import NativeC0Adapter

    opportunities = pickle.loads(source.read_bytes())
    adapter = NativeC0Adapter()
    identity = adapter.load()
    raw_path, raw_sha, release_sha, prepared = adapter._load_history()
    release_days = {
        str(item.identifier): item.games[-1].day for item in prepared["series"]
    }
    contexts = {}
    for event in prepared["roster_events"]:
        contexts[str(event["source_match_id"])] = event["competition_context"]
    records = []
    valid = (
        opportunities.identity_status.eq("verified") & opportunities.decision_at.notna()
    )
    panel = opportunities.loc[valid].copy()
    panel["row_id"] = panel.opportunity_id
    panel["scenario_decision_at"] = panel.decision_at
    panel["competition_context"] = panel.golgg_match_id.astype(str).map(contexts)
    panel["best_of"] = panel.canonical_best_of
    panel = panel.loc[
        panel.competition_context.notna() & panel.best_of.isin((1, 3, 5))
    ].copy()
    prior = build_prior_targets(
        panel, prepared["roster_events"], target_native_release_days=release_days
    )
    arrays_raw, arrays_hist, queries = [], [], []
    from src.analysis import ev_asof_replay as replay

    for row in prior.sort_values(["scenario_decision_at", "row_id"]).to_dict("records"):
        record = {
            "opportunity_id": row["opportunity_id"],
            "canonical_match_id": row["canonical_match_id"],
            "p_base": np.nan,
            "forecast_available": False,
            "identity_verified": True,
            "forecast_at": row["scenario_decision_at"],
            "issuance_authenticated": False,
            "forecast_clock_basis": "simulated_historical_origin_not_issuance",
            "native_effective_release_day": row["target_native_effective_release_day"],
            "native_absence": row["native_absence"],
            "exp039_status": "not_evaluated",
        }
        records.append(record)
        if not row["forecast_eligible"] or not row["prior_roster_available"]:
            continue
        request = adapter._validate_request(
            {
                "team1_id": row["team1_id"],
                "team2_id": row["team2_id"],
                "roster_a": row["roster_a"],
                "roster_b": row["roster_b"],
                "best_of": int(row["best_of"]),
                "decision_at": row["scenario_decision_at"].isoformat(),
                "start_at": row["start_at"].isoformat(),
                "competition_context": row["competition_context"],
                "mode": "full",
            }
        )
        state = adapter._advance_history(request["decision_at"].date(), prepared)
        before = replay._logical_state_marker(state)
        target, raw, hist, provenance = adapter._raw_snapshot(request)
        if before != replay._logical_state_marker(state):
            raise ValueError("Native feature query mutated historical state")
        record.update(
            prior_roster_a=row["roster_a"],
            prior_roster_b=row["roster_b"],
            prior_source_a=row["prior_source_a"],
            prior_source_b=row["prior_source_b"],
            native_input_hashes=adapter._array_digest({**raw, **hist}),
            feature_source_max_day=target["feature_source_max_day"],
        )
        arrays_raw.append(raw)
        arrays_hist.append(hist)
        queries.append((record, request, bool(row["canonical_a_is_team1"])))
    if queries:
        from src.models import a0_data_focus

        raw_batch = {
            key: np.stack([value[key] for value in arrays_raw]) for key in arrays_raw[0]
        }
        hist_batch = {
            key: np.stack([value[key] for value in arrays_hist])
            for key in arrays_hist[0]
        }
        result = a0_data_focus.predict_models(
            adapter._backend,
            adapter._shared,
            adapter._models,
            adapter._calibration,
            raw_batch,
            hist_batch,
            np.array([request["best_of"] for _, request, _ in queries]),
        )
        values = np.asarray(result["full"])[:, 0]
        if (
            values.shape != (len(queries),)
            or not np.isfinite(values).all()
            or ((values <= 0) | (values >= 1)).any()
        ):
            raise ValueError("Native C0 returned invalid batch probabilities")
        generated = datetime.now(timezone.utc).isoformat()
        for value, (record, _, canonical) in zip(values, queries, strict=True):
            record.update(
                p_base=float(value if canonical else 1 - value),
                forecast_available=True,
                forecast_generated_at=generated,
                native_absence=None,
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "opportunity_id",
                "native_effective_release_day",
                "p_base",
                "forecast_available",
                "identity_verified",
                "forecast_at",
                "native_absence",
            ]
        )
    missing = opportunities.loc[
        ~opportunities.opportunity_id.isin(frame.opportunity_id)
    ].copy()
    missing = missing[["opportunity_id", "canonical_match_id"]]
    missing["native_absence"] = "identity_schedule_context_or_format_unavailable"
    frame = pd.concat([frame, missing], ignore_index=True)
    frame["forecast_available"] = frame.forecast_available.fillna(False).astype(bool)
    frame["identity_verified"] = frame.identity_verified.fillna(False).astype(bool)
    frame["native_effective_release_day"] = frame.native_effective_release_day.where(
        frame.native_effective_release_day.notna(), None
    )
    evidence = {
        "identity": identity,
        "raw_history_sha256": raw_sha,
        "release_map_sha256": release_sha,
        "raw_history_path": str(raw_path),
        "release_map_path": str(prepared["release_path"]),
        "prior_context_scenarios": len(prior),
        "genuine_native_forecasts": len(queries),
    }
    destination.write_bytes(pickle.dumps((frame, evidence)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 or sys.argv[1] != "--native-worker":
        raise SystemExit("Use the unified evaluation current-policy command")
    _native_worker(Path(sys.argv[2]), Path(sys.argv[3]))
