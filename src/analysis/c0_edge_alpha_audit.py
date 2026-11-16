"""Causal historical alpha audit over the sealed C0 interval cache.

Preparation is outcome/EV/eligibility-blind.  The only fitted quantity is one
bounded SPORTS-weight scalar selected on labels demonstrably available before
the frozen cutoff; all other probability curves are fixed diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EPSILON = 1e-6
DEFAULT_ALPHA_GRID = (0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 1.0)
_REQUIRED_INPUTS = (
    "opportunities.parquet",
    "quotes.parquet",
    "forecasts.parquet",
    "candidates.parquet",
)
_REQUIRED_ROW_COLUMNS = (
    "opportunity_id",
    "canonical_match_id",
    "decision_at",
    "start_at",
    "settle_at",
    "y",
    "bookmaker_id",
    "bookmaker",
    "quote_id",
    "quoted_at",
    "odds",
    "odds_b",
    "p_base",
    "market_prob_a",
    "league",
    "best_of",
    "native_effective_release_day",
    "result_recorded_at",
    "fixture_weight",
    "label_available_at",
    "partition",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _utc(value: Any) -> pd.Timestamp:
    if value is None or value is pd.NaT or value is pd.NA:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if stamp.tzinfo is None:
        return pd.NaT
    return stamp.tz_convert("UTC")


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.map(_utc), utc=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe_value(value: Any) -> Any:
    """Convert nested NumPy/pandas values to strict JSON values; nonfinite is null."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Decimal):
        return format(value, "f") if value.is_finite() else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        finite = float(value)
        return finite if math.isfinite(finite) else None
    if isinstance(value, np.datetime64):
        return json_safe_value(pd.Timestamp(value))
    if isinstance(value, (pd.Timestamp, datetime, date)):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC")
        return stamp.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return json_safe_value(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_safe_value(value.tolist())
    if isinstance(value, np.ndarray):
        return json_safe_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe_value(item) for item in sorted(value, key=repr)]
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return json_safe_value(value.item())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def canonical_json_dumps(value: Any) -> str:
    """Canonical compact JSON used for evidence bytes and JSON output."""
    return json.dumps(
        json_safe_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_dumps(value) + "\n", encoding="utf-8")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != "c0_edge_recovery_v1":
        raise ValueError("Unsupported C0 edge recovery protocol")
    if protocol.get("run_id") != "c0_edge_recovery_20261022_01":
        raise ValueError("Wrong frozen C0 edge recovery run")
    if protocol.get("scope") != "HISTORICAL_DEVELOPMENT_ONLY":
        raise ValueError("Alpha audit is restricted to historical development")
    if protocol.get("formal_base_fits") != 0:
        raise ValueError("The alpha audit cannot retrain a base model")
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs.get("timing_root"):
        raise ValueError("Protocol must name the sealed timing_root input")
    if tuple(float(x) for x in protocol.get("alpha_grid", ())) != DEFAULT_ALPHA_GRID:
        raise ValueError("Alpha grid differs from the frozen grid")
    if not protocol.get("cutoff_at") or not protocol.get("assessment_end_at"):
        raise ValueError("Protocol must freeze train and assessment boundaries")


def _cached_paths(protocol: Mapping[str, Any]) -> dict[str, Path]:
    timing_root = _repo_path(protocol["inputs"]["timing_root"])
    return {
        "opportunities.parquet": timing_root / "opportunities.parquet",
        "quotes.parquet": timing_root / "quotes.parquet",
        "forecasts.parquet": timing_root / "forecasts.parquet",
        "candidates.parquet": timing_root / "candidates.parquet",
        "fixture_accounting.json": timing_root / "fixture_accounting.json",
        "independent_input_receipt.json": timing_root / "independent_input_receipt.json",
    }


def _source_input_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _cached_paths(protocol)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cached C0 timing input(s) missing: " + ", ".join(missing))
    hashes = {name: _sha256_file(path) for name, path in paths.items()}

    seal_path_value = protocol.get("inputs", {}).get("timing_seal")
    if seal_path_value:
        seal_path = _repo_path(seal_path_value)
        if not seal_path.is_file():
            raise FileNotFoundError(f"Timing seal is missing: {seal_path}")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        artifact_hashes = seal.get("artifact_sha256", {})
        required_artifacts = (
            *_REQUIRED_INPUTS, "fixture_accounting.json", "independent_input_receipt.json",
        )
        for name in required_artifacts:
            path = paths[name]
            key = path.relative_to(ROOT).as_posix()
            expected = artifact_hashes.get(key)
            if not isinstance(expected, str) or hashes[name] != expected:
                raise ValueError(f"Cached timing artifact does not match its seal: {name}")
        hashes["timing_seal.json"] = _sha256_file(seal_path)

    receipt_path = paths["independent_input_receipt.json"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_hashes = receipt.get("input_sha256")
    if not isinstance(receipt_hashes, Mapping):
        raise ValueError("Independent timing receipt lacks input hashes")
    for name in _REQUIRED_INPUTS:
        expected = receipt_hashes.get(name)
        if not isinstance(expected, str) or hashes[name] != expected:
            raise ValueError(f"Cached timing input differs from independent receipt: {name}")
    hashes["independent_input_receipt.json"] = _sha256_file(receipt_path)
    if protocol.get("_protocol_sha256"):
        hashes["protocol.json"] = str(protocol["_protocol_sha256"])
    return hashes


def _load_cached_inputs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed interval cache only; never regenerate native forecasts."""
    paths = _cached_paths(protocol)
    hashes = _source_input_hashes(protocol)
    print("[alpha] reading sealed candidates/forecasts/opportunities", flush=True)
    inputs = {
        "opportunities": pd.read_parquet(paths["opportunities.parquet"]),
        "quotes": pd.read_parquet(paths["quotes.parquet"]),
        "forecasts": pd.read_parquet(paths["forecasts.parquet"]),
        "candidates": pd.read_parquet(paths["candidates.parquet"]),
        "fixture_accounting": pd.DataFrame(
            json.loads(paths["fixture_accounting.json"].read_text(encoding="utf-8"))
        ),
        "input_hashes": hashes,
    }
    expected_universe = int(protocol.get("cohort", {}).get("universe_fixtures", 1771))
    if len(inputs["fixture_accounting"]) != expected_universe:
        raise ValueError("Sealed fixture accounting ledger size differs from protocol")
    print(
        "[alpha] cached rows "
        f"fixtures={len(inputs['fixture_accounting'])} "
        f"checkpoints={len(inputs['opportunities'])} "
        f"events={len(inputs['candidates'])}",
        flush=True,
    )
    return inputs


def _numeric_equal(left: pd.Series, right: pd.Series, *, atol: float = 1e-12) -> np.ndarray:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    equal = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=0.0, atol=atol, equal_nan=True)
    return equal


def _attach_source_clocks(
    events: pd.DataFrame,
    opportunities: pd.DataFrame,
    forecasts: pd.DataFrame,
    quotes: pd.DataFrame,
) -> pd.DataFrame:
    """Validate exact-ID/source clocks and attach checkpoint flags without probability joins."""
    result = events.copy()
    if result.empty:
        for name, dtype in (
            ("is_grid", bool),
            ("is_snapshot", bool),
            ("quote_age_seconds", float),
            ("hours_before_start", float),
        ):
            if name not in result:
                result[name] = pd.Series(dtype=dtype)
        return result

    if not opportunities.opportunity_id.is_unique:
        raise ValueError("Saved opportunity checkpoint IDs must be unique")
    if not forecasts.opportunity_id.is_unique:
        raise ValueError("Saved forecast checkpoint IDs must be unique")
    opp = opportunities.set_index("opportunity_id", drop=False)
    forecast = forecasts.set_index("opportunity_id", drop=False)
    ids = result.opportunity_id
    if not ids.isin(opp.index).all() or not ids.isin(forecast.index).all():
        raise ValueError("Candidate rows must match exact saved checkpoint IDs")

    for name in (
        "canonical_match_id", "decision_at", "start_at", "settle_at", "result_recorded_at",
        "team1_id", "team2_id", "canonical_a_is_team1", "golgg_match_id",
        "competition_context",
    ):
        if name not in opp:
            continue
        source = ids.map(opp[name])
        if name in result:
            if name.endswith("_at"):
                observed = _utc_series(result[name])
                expected = _utc_series(source)
                same = observed.eq(expected) | (observed.isna() & expected.isna())
                if not same.all():
                    raise ValueError(f"Candidate and opportunity clocks differ in {name}")
            elif name == "canonical_match_id":
                if not result[name].eq(source).fillna(False).all():
                    raise ValueError("Candidate and opportunity fixture identities differ")
            elif name in {"team1_id", "team2_id", "canonical_a_is_team1", "golgg_match_id", "competition_context"}:
                same = result[name].eq(source) | (result[name].isna() & source.isna())
                if not same.all():
                    raise ValueError(f"Candidate and opportunity identity context differs in {name}")
        result[name] = source.to_numpy()

    for name in ("is_grid", "is_snapshot", "hours_before_start"):
        if name not in opp:
            continue
        source = ids.map(opp[name])
        if name in result:
            if name in {"is_grid", "is_snapshot"}:
                expected = source.fillna(False).astype(bool)
                observed = result[name].fillna(False).astype(bool)
                if not observed.eq(expected).all():
                    raise ValueError(f"Candidate clock flag differs from saved opportunity: {name}")
            elif not _numeric_equal(result[name], source).all():
                raise ValueError(f"Candidate lead differs from saved opportunity: {name}")
        result[name] = source.to_numpy()

    for name in (
        "p_base",
        "forecast_available",
        "identity_verified",
        "forecast_at",
        "forecast_clock_basis",
        "native_effective_release_day",
        "native_absence",
        "daily_state_cutoff_day",
        "daily_state_key_sha256",
        "native_feature_contract_sha256",
        "native_input_hashes",
        "feature_source_max_day",
        "prior_roster_a",
        "prior_roster_b",
        "prior_source_a",
        "prior_source_b",
        "daily_state_projection",
        "issuance_authenticated",
        "daily_state_representative_opportunity_id",
        "daily_state_representative_at",
    ):
        if name not in forecast:
            continue
        source = ids.map(forecast[name])
        if name in result:
            if name == "p_base":
                same = _numeric_equal(result[name], source).all()
            elif name.endswith("_at"):
                observed, expected = _utc_series(result[name]), _utc_series(source)
                same = (observed.eq(expected) | (observed.isna() & expected.isna())).all()
            elif name.endswith("_day"):
                same = all(
                    _day_text(observed) == _day_text(expected)
                    for observed, expected in zip(result[name], source)
                )
            else:
                same = all(
                    canonical_json_dumps(observed) == canonical_json_dumps(expected)
                    for observed, expected in zip(result[name], source)
                )
            if not same:
                raise ValueError(f"Candidate forecast provenance differs from exact checkpoint: {name}")
        result[name] = source.to_numpy()
    if {"prior_roster_a", "prior_roster_b", "team1_id", "team2_id"}.issubset(result.columns):
        result["prior_roster_orientation"] = "native_team1_team2"

    quote_id_column = "quote_id" if "quote_id" in quotes else "id"
    source_quotes = quotes.copy()
    if quote_id_column not in source_quotes:
        raise ValueError("Saved raw quote table lacks quote IDs")
    source_quotes = source_quotes.rename(columns={quote_id_column: "_source_quote_id"})
    if source_quotes._source_quote_id.duplicated().any():
        raise ValueError("Saved raw quote IDs must be unique")
    q = source_quotes.set_index("_source_quote_id", drop=False)
    if not result.quote_id.isin(q.index).all():
        raise ValueError("Candidate quote IDs must match saved raw quote states")
    for event_column, quote_column in (
        ("canonical_match_id", "canonical_match_id"),
        ("bookmaker_id", "bookmaker_id"),
    ):
        if quote_column in q:
            source = result.quote_id.map(q[quote_column])
            if not result[event_column].eq(source).fillna(False).all():
                raise ValueError(f"Candidate {event_column} differs from its exact raw quote")
    if "quoted_at" in q:
        source_clock = result.quote_id.map(q.quoted_at)
        observed_clock, expected_clock = _utc_series(result.quoted_at), _utc_series(source_clock)
        if not (observed_clock.eq(expected_clock) | (observed_clock.isna() & expected_clock.isna())).all():
            raise ValueError("Candidate quote clocks differ from exact raw quote states")

    result["decision_at"] = _utc_series(result.decision_at)
    result["start_at"] = _utc_series(result.start_at)
    result["settle_at"] = _utc_series(result.settle_at)
    result["forecast_at"] = _utc_series(result.forecast_at)
    result["quoted_at"] = _utc_series(result.quoted_at)
    result["quote_age_seconds"] = (
        result.decision_at - result.quoted_at
    ).dt.total_seconds()
    result["hours_before_start"] = (
        result.start_at - result.decision_at
    ).dt.total_seconds() / 3600.0
    return result


def _validate_raw_quote_orientation(events: pd.DataFrame, quotes: pd.DataFrame) -> None:
    if events.empty:
        return
    raw = quotes.copy()
    if "quote_id" not in raw and "id" in raw:
        raw = raw.rename(columns={"id": "quote_id"})
    raw = raw.set_index("quote_id", drop=False)
    ids = events.quote_id
    orientation = ids.map(raw.raw_a_is_canonical_a)
    odds_a = pd.to_numeric(ids.map(raw.odds_a), errors="coerce").to_numpy(dtype=float)
    odds_b = pd.to_numeric(ids.map(raw.odds_b), errors="coerce").to_numpy(dtype=float)
    orientation_values = orientation.map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            or (isinstance(value, (int, np.integer)) and value in (0, 1))
            else None
        )
    ).to_numpy(dtype=object)
    if any(value is None for value in orientation_values):
        raise ValueError("Causal candidate quotes require a known canonical A/B orientation")
    canonical_a = np.where(orientation_values.astype(bool), odds_a, odds_b)
    canonical_b = np.where(orientation_values.astype(bool), odds_b, odds_a)
    expected = np.where(events.side.astype(str).to_numpy() == "a", canonical_a, canonical_b)
    observed = pd.to_numeric(events.odds, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise ValueError("Saved alpha events require finite paired quote prices")
    if not np.isclose(expected, observed, rtol=0.0, atol=1e-12).all():
        raise ValueError("Candidate side prices differ from canonicalized raw quote prices")
    if "identity_verified" in raw:
        verified = ids.map(raw.identity_verified).fillna(False).astype(bool)
        if not verified.all():
            raise ValueError("Candidate event references an unverified raw quote")


def _valid_paired_events(events: pd.DataFrame, freshness_seconds: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    required = {
        "opportunity_id",
        "canonical_match_id",
        "decision_at",
        "start_at",
        "bookmaker_id",
        "quote_id",
        "quoted_at",
        "side",
        "odds",
        "market_prob_a",
        "market_prob",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Saved candidates lack alpha event fields: {sorted(missing)}")
    data = events.copy()
    data["side"] = data.side.astype(str)
    data["odds"] = pd.to_numeric(data.odds, errors="coerce")
    data["market_prob_a"] = pd.to_numeric(data.market_prob_a, errors="coerce")
    data["market_prob"] = pd.to_numeric(data.market_prob, errors="coerce")
    key = ["opportunity_id", "bookmaker_id", "quote_id"]
    valid = (
        data.side.isin(("a", "b"))
        & data.quote_id.notna()
        & data.bookmaker_id.notna()
        & np.isfinite(data.odds)
        & data.odds.gt(1.0)
        & np.isfinite(data.market_prob_a)
        & data.market_prob_a.between(0.0, 1.0, inclusive="both")
        & np.isfinite(data.market_prob)
        & data.market_prob.between(0.0, 1.0, inclusive="both")
        & data.quote_age_seconds.ge(0.0)
        & data.quote_age_seconds.le(float(freshness_seconds))
        & data.quoted_at.notna()
        & data.decision_at.notna()
        & data.start_at.notna()
    )
    data = data.loc[valid].copy()
    if data.empty:
        return data
    paired = data.groupby(key, sort=False).side.agg(["size", "nunique"])
    paired_keys = paired.index[(paired["size"].eq(2)) & (paired["nunique"].eq(2))]
    data = data.set_index(key, drop=False)
    data = data.loc[data.index.isin(paired_keys)].reset_index(drop=True)
    if data.empty:
        return data
    expected_market = np.where(
        data.side.to_numpy() == "a",
        data.market_prob_a.to_numpy(dtype=float),
        1.0 - data.market_prob_a.to_numpy(dtype=float),
    )
    if not np.isclose(
        data.market_prob.to_numpy(dtype=float), expected_market, rtol=0.0, atol=1e-12
    ).all():
        raise ValueError("Saved A/B LOO probabilities are not exact complements")
    pair_stats = data.groupby(key, sort=False).market_prob_a.agg(["min", "max"])
    if not np.isclose(pair_stats["min"], pair_stats["max"], rtol=0.0, atol=1e-12).all():
        raise ValueError("A/B sides of a quote state have different LOO consensus")
    if data.duplicated(key + ["side"]).any():
        raise ValueError("Saved quote state has duplicate A/B side records")
    side_a = data.loc[data.side.eq("a"), key + ["odds"]].rename(columns={"odds": "odds_a"})
    side_b = data.loc[data.side.eq("b"), key + ["odds"]].rename(columns={"odds": "odds_b"})
    prices = side_a.merge(side_b, on=key, how="inner", validate="one_to_one")
    if prices.duplicated(["opportunity_id", "bookmaker_id"]).any():
        raise ValueError("Saved opportunity contains multiple focal quotes for one bookmaker")
    inverse_a = 1.0 / prices.odds_a.to_numpy(dtype=float)
    inverse_b = 1.0 / prices.odds_b.to_numpy(dtype=float)
    prices["no_vig_prob_a"] = inverse_a / (inverse_a + inverse_b)
    book_count = prices.groupby("opportunity_id", sort=False).bookmaker_id.transform("nunique")
    supported = prices.loc[book_count.ge(3)].copy()
    if supported.empty:
        return data.iloc[0:0].copy()
    supported_count = supported.groupby("opportunity_id", sort=False).bookmaker_id.transform("nunique")
    supported_sum = supported.groupby("opportunity_id", sort=False).no_vig_prob_a.transform("sum")
    supported["loo_market_prob_a"] = (
        supported_sum - supported.no_vig_prob_a
    ) / (supported_count - 1)
    lookup = pd.Series(
        supported.loo_market_prob_a.to_numpy(dtype=float),
        index=pd.MultiIndex.from_frame(supported[key]),
    )
    data = data.loc[data.opportunity_id.isin(supported.opportunity_id.unique())].copy()
    expected_loo = lookup.reindex(pd.MultiIndex.from_frame(data[key])).to_numpy(dtype=float)
    if not np.isfinite(expected_loo).all() or not np.isclose(
        data.market_prob_a.to_numpy(dtype=float),
        expected_loo,
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise ValueError("Saved market probabilities differ from exact raw-quote leave-one-out consensus")
    return data


def _label_clock(release_day: Any, recorded_at: Any) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Return strict max clock, daily-release clock, and explicit evidence status."""
    day = None
    if release_day is not None and not pd.isna(release_day):
        try:
            parsed = pd.Timestamp(str(release_day))
            if parsed.tzinfo is not None:
                parsed = parsed.tz_convert("UTC").tz_localize(None)
            day = parsed.date()
        except (TypeError, ValueError, OverflowError):
            day = None
    daily_clock = (
        pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
        if day is not None
        else pd.NaT
    )
    result_clock = _utc(recorded_at)
    if pd.isna(daily_clock):
        return pd.NaT, pd.NaT, "native_release_clock_unknown"
    if pd.isna(result_clock):
        return daily_clock, daily_clock, "result_record_clock_unknown_release_clock_known"
    return max(daily_clock, result_clock), daily_clock, "release_and_result_clocks_known"


def _partition(decision_at: Any, label_available_at: Any, y: Any, cutoff: pd.Timestamp,
               assessment_end: pd.Timestamp) -> tuple[str, str]:
    decision, label = _utc(decision_at), _utc(label_available_at)
    known_y = y is not None and not pd.isna(y) and float(y) in (0.0, 1.0)
    if pd.isna(decision):
        return "NO_VALID_READY_CLOCK", "ready_decision_clock_unknown"
    if decision < cutoff:
        if known_y and not pd.isna(label) and label <= cutoff:
            return "TRAIN", "label_available_by_cutoff"
        if not known_y:
            return "EARLY_UNKNOWN_OUTCOME", "outcome_unknown_at_audit"
        if pd.isna(label):
            return "EARLY_LABEL_CLOCK_UNKNOWN", "not_reassigned_to_assessment"
        return "BOUNDARY_EXCLUDED", "label_available_after_cutoff_not_reassigned"
    if decision < assessment_end:
        return "ASSESS", "post_cutoff_fixed_assessment_window"
    return "OUTSIDE_FIXED_WINDOWS", "after_assessment_end"


def _build_fixture_inventory(
    fixture_accounting: pd.DataFrame,
    opportunities: pd.DataFrame,
    rows: pd.DataFrame,
    cutoff: pd.Timestamp,
    assessment_end: pd.Timestamp,
) -> pd.DataFrame:
    if fixture_accounting.empty:
        if opportunities.empty:
            base = pd.DataFrame(columns=["canonical_match_id", "y", "start_at", "result_recorded_at"])
        else:
            cols = [name for name in (
                "canonical_match_id", "y", "start_at", "result_recorded_at",
                "league", "best_of", "identity_status", "checkpoint_absence",
            ) if name in opportunities]
            base = opportunities[cols].drop_duplicates("canonical_match_id", keep="first").copy()
    else:
        base = fixture_accounting.copy()
        if "result_recorded_at" in base:
            # This JSON field is a millisecond-formatted display, not the precise clock authority.
            base = base.rename(
                columns={"result_recorded_at": "fixture_accounting_result_recorded_at_display"}
            )
            base["fixture_accounting_result_recorded_at_precision"] = "milliseconds"
    if "canonical_match_id" not in base:
        raise ValueError("Fixture absence ledger requires canonical_match_id")
    if base.canonical_match_id.duplicated().any():
        raise ValueError("Fixture absence ledger must contain one row per fixture")
    for column in ("start_at", "settle_at"):
        if column in base:
            base[column] = _utc_series(base[column])
    if "y" not in base:
        base["y"] = np.nan
    if "result_recorded_at" not in base:
        base["result_recorded_at"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")

    if rows.empty:
        ready = pd.DataFrame(columns=[
            "canonical_match_id", "opportunity_id", "decision_at", "settle_at",
            "native_effective_release_day", "label_available_at", "daily_release_label_available_at",
            "label_clock_status", "partition", "partition_reason", "focal_book_count",
        ])
    else:
        ready = rows.sort_values(
            ["canonical_match_id", "decision_at", "bookmaker_id"], kind="mergesort"
        ).drop_duplicates("canonical_match_id").copy()
        ready = ready[[
            name for name in (
                "canonical_match_id", "opportunity_id", "decision_at", "settle_at",
                "native_effective_release_day", "label_available_at",
                "daily_release_label_available_at", "label_clock_status", "partition",
                "partition_reason", "bookmaker_id",
            ) if name in ready
        ]].drop_duplicates("canonical_match_id")
        ready = ready.rename(columns={"bookmaker_id": "_one_bookmaker"})
        counts = rows.groupby("canonical_match_id", sort=False).bookmaker_id.nunique()
        ready["focal_book_count"] = ready.canonical_match_id.map(counts).astype("Int64")
        ready = ready.drop(columns="_one_bookmaker", errors="ignore")
    planned = pd.DataFrame(columns=["canonical_match_id", "cash_planned_decision_at"])
    if not opportunities.empty and {"canonical_match_id", "decision_at"}.issubset(opportunities.columns):
        planned = opportunities[["canonical_match_id", "decision_at"]].copy()
        planned["decision_at"] = _utc_series(planned.decision_at)
        planned = (
            planned.loc[planned.decision_at.notna()]
            .sort_values(["canonical_match_id", "decision_at"], kind="mergesort")
            .drop_duplicates("canonical_match_id")
            .rename(columns={"decision_at": "cash_planned_decision_at"})
        )
    inventory = base.merge(ready, on="canonical_match_id", how="left", validate="one_to_one", suffixes=("", "_ready"))
    inventory = inventory.merge(planned, on="canonical_match_id", how="left", validate="one_to_one")
    if not opportunities.empty and {
        "canonical_match_id", "result_recorded_at"
    }.issubset(opportunities.columns):
        source_clocks = opportunities[
            ["canonical_match_id", "result_recorded_at"]
        ].copy()
        source_clocks["result_recorded_at"] = _utc_series(source_clocks.result_recorded_at)
        if source_clocks.drop_duplicates().canonical_match_id.duplicated().any():
            raise ValueError("Saved opportunity result clocks differ within a fixture")
        source_clocks = source_clocks.drop_duplicates("canonical_match_id").rename(
            columns={"result_recorded_at": "_authoritative_result_recorded_at"}
        )
        inventory = inventory.merge(
            source_clocks,
            on="canonical_match_id",
            how="left",
            validate="one_to_one",
        )
        inventory["result_recorded_at"] = inventory.pop(
            "_authoritative_result_recorded_at"
        )
        inventory["result_recorded_at_source"] = "opportunities.parquet"
        inventory["result_recorded_at_precision"] = str(
            opportunities.result_recorded_at.dtype
        )
    inventory["has_first_ready_state"] = inventory.opportunity_id.notna()
    inventory["fixture_weight"] = np.where(inventory.has_first_ready_state, 1.0, 0.0)
    if "partition" not in inventory:
        inventory["partition"] = "NO_FIRST_READY_STATE"
    else:
        inventory["partition"] = inventory.partition.fillna("NO_FIRST_READY_STATE")
    if "label_clock_status" not in inventory:
        inventory["label_clock_status"] = "native_release_clock_unknown"
    else:
        inventory["label_clock_status"] = inventory.label_clock_status.fillna("native_release_clock_unknown")
    ready_clock = _utc_series(inventory.decision_at) if "decision_at" in inventory else pd.Series(pd.NaT, index=inventory.index, dtype="datetime64[ns, UTC]")
    planned_clock = _utc_series(inventory.cash_planned_decision_at)
    inventory["cash_decision_at"] = ready_clock.where(
        inventory.has_first_ready_state, planned_clock
    )
    inventory["cash_clock_basis"] = np.where(
        inventory.has_first_ready_state, "first_ready_decision",
        np.where(inventory.cash_decision_at.notna(), "earliest_saved_opportunity_decision_fallback", "unknown"),
    )
    cash_clock = _utc_series(inventory.cash_decision_at)
    start_clock = (
        _utc_series(inventory.start_at)
        if "start_at" in inventory
        else pd.Series(pd.NaT, index=inventory.index, dtype="datetime64[ns, UTC]")
    )
    unknown_start = start_clock.isna()
    inventory["cash_start_status"] = np.where(unknown_start, "UNKNOWN_START", "KNOWN_START")
    inventory["cash_partition"] = np.select(
        [
            unknown_start,
            cash_clock.isna(),
            cash_clock.lt(cutoff),
            cash_clock.ge(cutoff) & cash_clock.lt(assessment_end),
        ],
        ["UNASSIGNED", "UNKNOWN_PLANNED_CLOCK", "TRAIN_CLOCK", "ASSESS_CLOCK"],
        default="OUTSIDE_FIXED_WINDOWS",
    )
    inventory["cash_partition_uses_labels"] = False
    return inventory

def _day_text(value: Any) -> str | None:
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%d")


def _build_daily_native_states(
    opportunities: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Expose one saved native state per exact fixture/UTC-day key, including absences."""
    state_columns = (
        "p_base", "forecast_available", "identity_verified", "forecast_at",
        "forecast_clock_basis", "native_absence", "native_effective_release_day",
        "daily_state_cutoff_day", "daily_state_key_sha256",
        "native_feature_contract_sha256", "native_input_hashes",
        "feature_source_max_day", "prior_roster_a", "prior_roster_b",
        "prior_roster_orientation",
        "prior_source_a", "prior_source_b", "daily_state_projection",
        "issuance_authenticated", "daily_state_representative_opportunity_id",
        "daily_state_representative_at",
    )
    opportunity_columns = (
        "opportunity_id", "canonical_match_id", "decision_at", "start_at",
        "team1_id", "team2_id", "canonical_a_is_team1", "canonical_best_of",
        "best_of", "league", "golgg_match_id", "identity_status", "competition_context",
    )
    context_columns = tuple(
        name for name in opportunity_columns
        if name not in {"opportunity_id", "canonical_match_id", "decision_at"}
    )
    output_columns = (
        "canonical_match_id", "requested_utc_day", "representative_opportunity_id",
        "representative_decision_at", *context_columns, *state_columns,
        "same_day_cutoff_match", "strict_prior_feature_day_proof",
        "usable_for_same_day_projection",
    )
    if opportunities.empty:
        return pd.DataFrame(columns=output_columns)
    if not opportunities.opportunity_id.is_unique:
        raise ValueError("Saved opportunity IDs must be unique for daily-state projection")
    if not forecasts.empty and not forecasts.opportunity_id.is_unique:
        raise ValueError("Saved forecast IDs must be unique for daily-state projection")
    if "canonical_match_id" not in opportunities or "decision_at" not in opportunities:
        raise ValueError("Daily-state projections require exact fixture and decision clocks")
    if "canonical_match_id" in forecasts and not forecasts.empty:
        source_fixtures = forecasts.set_index("opportunity_id").canonical_match_id
        expected_fixtures = opportunities.set_index("opportunity_id").canonical_match_id
        shared = source_fixtures.index.intersection(expected_fixtures.index)
        if not source_fixtures.loc[shared].eq(expected_fixtures.loc[shared]).all():
            raise ValueError("Daily native forecast fixture identity differs from its opportunity")

    opp_columns = [name for name in opportunity_columns if name in opportunities]
    states = opportunities[opp_columns].copy()
    states["decision_at"] = _utc_series(states.decision_at)
    states["requested_utc_day"] = states.decision_at.dt.strftime("%Y-%m-%d")
    forecast_columns = ["opportunity_id", *[name for name in state_columns if name in forecasts]]
    forecasts_for_join = forecasts[forecast_columns].copy() if not forecasts.empty else pd.DataFrame(
        columns=forecast_columns
    )
    states = states.merge(
        forecasts_for_join,
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    if {"prior_roster_a", "prior_roster_b", "team1_id", "team2_id"}.issubset(states.columns):
        states["prior_roster_orientation"] = "native_team1_team2"
    states = states.loc[states.requested_utc_day.notna()].copy()
    invariant_columns = (
        "team1_id", "team2_id", "canonical_a_is_team1", "golgg_match_id",
        "competition_context", "p_base", "forecast_available", "identity_verified", "native_absence",
        "native_effective_release_day", "daily_state_cutoff_day",
        "daily_state_key_sha256", "native_feature_contract_sha256",
        "native_input_hashes", "feature_source_max_day", "prior_roster_a",
        "prior_roster_b", "prior_roster_orientation", "prior_source_a", "prior_source_b",
    )
    rows: list[dict[str, Any]] = []
    for (match_id, requested_day), group in states.groupby(
        ["canonical_match_id", "requested_utc_day"], sort=True, dropna=False
    ):
        for column in invariant_columns:
            if column in group:
                values = group[column].dropna().map(repr).unique()
                if len(values) > 1:
                    raise ValueError(
                        f"Saved fixture/day native state differs within {column}"
                    )
        representative = group.sort_values(
            ["decision_at", "opportunity_id"], kind="mergesort"
        ).iloc[0]
        row = {
            "canonical_match_id": match_id,
            "requested_utc_day": requested_day,
            "representative_opportunity_id": representative.opportunity_id,
            "representative_decision_at": representative.decision_at,
        }
        row.update({
            name: representative[name] if name in representative else None
            for name in context_columns
        })
        for column in state_columns:
            row[column] = representative[column] if column in representative else None
        state_day = _day_text(row.get("daily_state_cutoff_day"))
        feature_day = _day_text(row.get("feature_source_max_day"))
        same_day = state_day is not None and state_day == requested_day
        prior_day_proof = feature_day is not None and feature_day < requested_day
        row["same_day_cutoff_match"] = bool(same_day)
        row["strict_prior_feature_day_proof"] = bool(prior_day_proof)
        try:
            p_base = float(row.get("p_base"))
        except (TypeError, ValueError):
            p_base = np.nan
        forecast_available = row.get("forecast_available")
        identity_verified = row.get("identity_verified")
        row["usable_for_same_day_projection"] = bool(
            same_day
            and prior_day_proof
            and isinstance(forecast_available, (bool, np.bool_))
            and bool(forecast_available)
            and isinstance(identity_verified, (bool, np.bool_))
            and bool(identity_verified)
            and math.isfinite(p_base)
            and 0.0 < p_base < 1.0
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def _validate_fixture_outcome_clocks(
    sources: Iterable[pd.DataFrame],
    fixture_accounting: pd.DataFrame,
) -> None:
    """Validate outcome identity; exact clocks come from the saved opportunity parquet."""
    if fixture_accounting.empty:
        return
    if "canonical_match_id" not in fixture_accounting:
        raise ValueError("Outcome validation requires canonical fixture identities")
    if fixture_accounting.canonical_match_id.duplicated().any():
        raise ValueError("Fixture accounting ledger must be unique by canonical match")
    ledger = fixture_accounting.set_index("canonical_match_id", drop=False)
    sources = tuple(sources)
    for source in sources:
        if source.empty or "canonical_match_id" not in source:
            continue
        match_ids = source.canonical_match_id
        if not match_ids.isin(ledger.index).all():
            raise ValueError("Cached row refers to a fixture outside the accounting ledger")
        if "y" in source and "y" in ledger:
            observed = pd.to_numeric(source.y, errors="coerce")
            expected = pd.to_numeric(match_ids.map(ledger.y), errors="coerce")
            if not _numeric_equal(observed, expected).all():
                raise ValueError("Cached outcome labels differ from the full fixture accounting ledger")

    authority = next(
        (
            source
            for source in sources
            if {"canonical_match_id", "opportunity_id", "result_recorded_at"}.issubset(source.columns)
            and not source.empty
        ),
        None,
    )
    if authority is not None and "result_recorded_at" in ledger:
        clocks = authority[["canonical_match_id", "result_recorded_at"]].copy()
        clocks["result_recorded_at"] = _utc_series(clocks.result_recorded_at)
        if clocks.drop_duplicates().canonical_match_id.duplicated().any():
            raise ValueError("Exact opportunity result clocks differ within a fixture")
        exact_by_fixture = clocks.drop_duplicates("canonical_match_id").set_index(
            "canonical_match_id"
        ).result_recorded_at
        shared = ledger.index.intersection(exact_by_fixture.index)
        display = _utc_series(ledger.loc[shared, "result_recorded_at"])
        exact_display = _utc_series(exact_by_fixture.loc[shared]).dt.floor("ms")
        comparable = display.notna() & exact_display.notna()
        if not (display.loc[comparable].eq(exact_display.loc[comparable])).all():
            raise ValueError("Fixture accounting result-clock display differs from exact millisecond projection")




def prepare_edge_cohort(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare all earliest-ready A-focal rows and all retained quote-state events."""
    _validate_protocol(protocol)
    cached = _load_cached_inputs(protocol)
    opportunities = cached["opportunities"].copy()
    forecasts = cached["forecasts"].copy()
    quotes = cached["quotes"].copy()
    candidates = cached["candidates"].copy()
    fixture_accounting = cached.get("fixture_accounting", pd.DataFrame()).copy()
    _validate_fixture_outcome_clocks(
        (opportunities, candidates),
        fixture_accounting,
    )
    if not candidates.empty and "side" not in candidates:
        raise ValueError("Saved candidates must include canonical side fields 'a'/'b'")
    required_opportunities = {
        "opportunity_id", "canonical_match_id", "decision_at", "start_at",
        "settle_at", "is_grid", "is_snapshot", "team1_id", "team2_id",
        "canonical_a_is_team1", "golgg_match_id", "result_recorded_at",
    }
    missing_opportunities = required_opportunities.difference(opportunities.columns)
    if not opportunities.empty and missing_opportunities:
        raise ValueError(
            f"Saved opportunities lack exact checkpoint identity/clocks/flags: "
            f"{sorted(missing_opportunities)}"
        )
    if not forecasts.empty and not {"opportunity_id", "p_base", "forecast_available", "forecast_at"}.issubset(forecasts.columns):
        raise ValueError("Saved forecasts lack exact checkpoint probability/clock fields")

    events = _attach_source_clocks(candidates, opportunities, forecasts, quotes)
    if not events.empty:
        _validate_raw_quote_orientation(events, quotes)
    freshness = int(protocol.get("cohort", {}).get("freshness_seconds", 300))
    events = _valid_paired_events(events, freshness)
    boundary_correction = {
        "inclusive_0_to_48_fixtures": 0,
        "inclusive_0_to_48_focal_rows": 0,
        "inclusive_selected_at_exact_48h_fixtures": 0,
        "inclusive_selected_at_exact_48h_focal_rows": 0,
    }
    if events.empty:
        events = candidates.iloc[0:0].copy()
        for name, dtype in (("is_grid", bool), ("is_snapshot", bool),
                            ("quote_age_seconds", float), ("hours_before_start", float)):
            if name not in events:
                events[name] = pd.Series(dtype=dtype)
        if "side" not in events:
            events["side"] = pd.Series(dtype=object)
        focal = events.copy()
    else:
        events["decision_at"] = _utc_series(events.decision_at)
        events["start_at"] = _utc_series(events.start_at)
        events["settle_at"] = _utc_series(events.settle_at)
        events["forecast_at"] = _utc_series(events.forecast_at)
        events["quoted_at"] = _utc_series(events.quoted_at)
        events["result_recorded_at"] = _utc_series(events.result_recorded_at)
        events["y"] = pd.to_numeric(events.y, errors="coerce")
        events["p_base"] = pd.to_numeric(events.p_base, errors="coerce")
        events["market_prob_a"] = pd.to_numeric(events.market_prob_a, errors="coerce")
        cutoff = _utc(protocol["cutoff_at"])
        assessment_end = _utc(protocol["assessment_end_at"])
        if pd.isna(cutoff) or pd.isna(assessment_end) or cutoff >= assessment_end:
            raise ValueError("Invalid frozen alpha cutoff boundaries")
        lead = (events.start_at - events.decision_at).dt.total_seconds() / 3600.0
        native_ready = (
            events.forecast_available.fillna(False).astype(bool)
            & events.forecast_at.notna()
            & events.forecast_at.le(events.decision_at)
            & np.isfinite(events.p_base)
            & events.p_base.gt(0.0)
            & events.p_base.lt(1.0)
        )
        inclusive_first_ready = events.loc[
            events.side.eq("a")
            & native_ready
            & lead.gt(0.0)
            & lead.le(48.0)
        ].copy()
        if not inclusive_first_ready.empty:
            inclusive_counts = inclusive_first_ready.groupby(
                "opportunity_id", sort=False
            ).bookmaker_id.transform("nunique")
            inclusive_first_ready = inclusive_first_ready.loc[
                inclusive_counts.ge(3)
            ].sort_values(
                ["canonical_match_id", "decision_at", "opportunity_id", "bookmaker_id"],
                kind="mergesort",
            )
            inclusive_state = inclusive_first_ready.drop_duplicates(
                "canonical_match_id"
            )[["canonical_match_id", "opportunity_id"]]
            inclusive_selected = inclusive_first_ready.merge(
                inclusive_state,
                on=["canonical_match_id", "opportunity_id"],
                how="inner",
                validate="many_to_one",
                sort=False,
            )
            at_boundary = inclusive_selected.loc[
                (
                    (inclusive_selected.start_at - inclusive_selected.decision_at)
                    .dt.total_seconds()
                    .div(3600.0)
                ).eq(48.0)
            ]
            boundary_correction.update({
                "inclusive_0_to_48_fixtures": int(
                    inclusive_selected.canonical_match_id.nunique()
                ),
                "inclusive_0_to_48_focal_rows": int(len(inclusive_selected)),
                "inclusive_selected_at_exact_48h_fixtures": int(
                    at_boundary.canonical_match_id.nunique()
                ),
                "inclusive_selected_at_exact_48h_focal_rows": int(len(at_boundary)),
            })
        first_ready = events.loc[
            events.side.eq("a")
            & native_ready
            & lead.gt(0.0)
            & lead.lt(48.0)
        ].copy()
        if not first_ready.empty:
            # The production admission cache already represents latest fresh identity-aligned
            # books; this recount is only the frozen >=3-book readiness test.
            book_counts = first_ready.groupby("opportunity_id", sort=False).bookmaker_id.transform("nunique")
            first_ready = first_ready.loc[book_counts.ge(3)].copy()
            if not first_ready.empty:
                first_ready = first_ready.sort_values(
                    ["canonical_match_id", "decision_at", "opportunity_id", "bookmaker_id"],
                    kind="mergesort",
                )
                # Resolve duplicate exact-clock checkpoints deterministically to one
                # opportunity so a fixture cannot acquire weight twice.
                selected_state = first_ready.drop_duplicates("canonical_match_id")[
                    ["canonical_match_id", "opportunity_id"]
                ]
                focal = first_ready.merge(
                    selected_state,
                    on=["canonical_match_id", "opportunity_id"],
                    how="inner",
                    validate="many_to_one",
                    sort=False,
                )
                if focal.duplicated(["opportunity_id", "bookmaker_id"]).any():
                    raise ValueError("Earliest-ready state contains duplicate focal bookmaker rows")
                pair_key = ["opportunity_id", "bookmaker_id", "quote_id"]
                side_b = events.loc[events.side.eq("b"), pair_key + ["odds"]].copy()
                if side_b.duplicated(pair_key).any():
                    raise ValueError("A focal quote has duplicate canonical B side rows")
                side_b = side_b.rename(columns={"odds": "odds_b"})
                focal = focal.merge(side_b, on=pair_key, how="left", validate="one_to_one")
                if focal.odds_b.isna().any():
                    raise ValueError("Every canonical-A focal quote requires its exact paired B price")
                focal["market_prob"] = focal["market_prob_a"]
                focal["lead_hours"] = (focal.start_at - focal.decision_at).dt.total_seconds() / 3600.0
                book_counts = focal.groupby("canonical_match_id", sort=False).bookmaker_id.transform("nunique")
                focal["fixture_weight"] = 1.0 / book_counts.astype(float)
                clocks = [
                    _label_clock(release, recorded)
                    for release, recorded in zip(
                        focal.native_effective_release_day,
                        focal.result_recorded_at,
                    )
                ]
                focal["label_available_at"] = pd.to_datetime([item[0] for item in clocks], utc=True)
                focal["daily_release_label_available_at"] = pd.to_datetime([item[1] for item in clocks], utc=True)
                focal["label_clock_status"] = [item[2] for item in clocks]
                partitions = [
                    _partition(decision, label, y, cutoff, assessment_end)
                    for decision, label, y in zip(focal.decision_at, focal.label_available_at, focal.y)
                ]
                focal["partition"] = [item[0] for item in partitions]
                focal["partition_reason"] = [item[1] for item in partitions]
            else:
                focal = events.iloc[0:0].copy()
        else:
            focal = events.iloc[0:0].copy()

    boundary_correction.update({
        "strict_0_to_48_fixtures": int(focal.canonical_match_id.nunique()) if not focal.empty else 0,
        "strict_0_to_48_focal_rows": int(len(focal)),
        "inclusive_minus_strict_focal_rows": (
            int(boundary_correction["inclusive_0_to_48_focal_rows"] - len(focal))
        ),
    })
    if focal.empty:
        for name in _REQUIRED_ROW_COLUMNS:
            if name not in focal:
                focal[name] = pd.Series(dtype="object")
        for name, dtype in (("lead_hours", float), ("daily_release_label_available_at", "datetime64[ns, UTC]"),
                            ("label_clock_status", object), ("partition_reason", object)):
            if name not in focal:
                focal[name] = pd.Series(dtype=dtype)
    else:
        missing = set(_REQUIRED_ROW_COLUMNS).difference(focal.columns)
        if missing:
            raise ValueError(f"Prepared focal rows lack contract fields: {sorted(missing)}")
        weights = focal.groupby("canonical_match_id", sort=False).fixture_weight.sum()
        if not np.allclose(weights.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Focal row weights must sum to exactly one per fixture")

    inventory = _build_fixture_inventory(
        fixture_accounting,
        opportunities,
        focal,
        _utc(protocol["cutoff_at"]),
        _utc(protocol["assessment_end_at"]),
    )
    if not fixture_accounting.empty:
        ledger_ids = set(fixture_accounting.canonical_match_id)
        if len(inventory) != len(fixture_accounting) or set(inventory.canonical_match_id) != ledger_ids:
            raise ValueError("Prepared fixture inventory did not preserve the complete absence ledger")
        for source in (opportunities, forecasts, candidates):
            if "canonical_match_id" in source and not set(source.canonical_match_id.dropna()).issubset(ledger_ids):
                raise ValueError("Cached checkpoint rows refer to a fixture outside the full accounting ledger")
    if "input_hashes" in cached:
        input_hashes = dict(cached["input_hashes"])
    else:
        input_hashes = _source_input_hashes(protocol)
    month_counts: dict[str, dict[str, int]] = {}
    if not focal.empty:
        for month, group in focal.groupby(focal.decision_at.dt.strftime("%Y-%m"), sort=True):
            month_counts[str(month)] = {
                "fixtures": int(group.canonical_match_id.nunique()),
                "focal_rows": int(len(group)),
            }
    cash_month_counts = (
        inventory.assign(_month=_utc_series(inventory.cash_decision_at).dt.strftime("%Y-%m"))
        .dropna(subset=["_month"])
        .groupby(["cash_partition", "_month"], sort=True)
        .canonical_match_id.nunique()
        .astype(int)
        .to_dict()
    ) if not inventory.empty else {}
    support = {
        "universe_fixtures": int(len(inventory)),
        "candidate_event_rows": int(len(events)),
        "first_ready_fixtures": int(focal.canonical_match_id.nunique()) if not focal.empty else 0,
        "first_ready_focal_rows": int(len(focal)),
        "partition_fixture_counts": (
            inventory.groupby("partition", dropna=False).canonical_match_id.nunique().astype(int).to_dict()
            if not inventory.empty and "partition" in inventory else {}
        ),
        "cash_partition_fixture_counts": (
            inventory.groupby("cash_partition", dropna=False).canonical_match_id.nunique().astype(int).to_dict()
            if not inventory.empty else {}
        ),
        "cash_month_fixture_counts": {
            f"{partition}:{month}": count
            for (partition, month), count in cash_month_counts.items()
        },
        "cash_partition_is_preselection_clock_only": True,
        "first_ready_month_counts": month_counts,
        "all_absence_rows_retained": True,
        "eligibility_or_ev_used_for_readiness": False,
        "outcome_used_for_readiness": False,
        "first_ready_boundary_correction": boundary_correction,
    }
    evidence = {
        "input_hashes": input_hashes,
        "source_paths": {name: str(path) for name, path in _cached_paths(protocol).items()},
        "source_receipt": "sealed interval candidates/forecasts/opportunities plus full fixture_accounting.json",
        "cohort_contract": "Earliest native-ready 3-fresh-book state, strictly 0<lead<48; canonical-A one-row-per-focal-book; equal fixture total weight; no eligibility/EV/outcome selection.",
        "alpha_semantics": protocol.get("alpha_semantics"),
        "support": support,
    }
    daily_native_states = _build_daily_native_states(opportunities, forecasts)
    support["daily_native_state_rows"] = int(len(daily_native_states))
    support["daily_native_states_usable_same_day"] = int(
        daily_native_states.usable_for_same_day_projection.sum()
    ) if not daily_native_states.empty else 0
    if len(inventory) == 1771:
        expected_month_fixtures = {
            "2026-05": 11,
            "2026-06": 26,
            "2026-07": 71,
            "2026-08": 171,
            "2026-09": 23,
        }
        observed_month_fixtures = {
            month: counts["fixtures"] for month, counts in month_counts.items()
        }
        cohort_contract = protocol.get("cohort", {})
        expected_ready_fixtures = cohort_contract.get(
            "expected_strict_first_ready_fixtures"
        )
        expected_ready_rows = cohort_contract.get(
            "expected_strict_first_ready_focal_rows"
        )
        if (
            observed_month_fixtures != expected_month_fixtures
            or (
                expected_ready_fixtures is not None
                and support["first_ready_fixtures"] != int(expected_ready_fixtures)
            )
            or (
                expected_ready_rows is not None
                and support["first_ready_focal_rows"] != int(expected_ready_rows)
            )
        ):
            raise ValueError("Prepared strict first-ready cohort differs from its declared protocol counts")
        support["expected_strict_first_ready_fixtures"] = expected_ready_fixtures
        support["expected_strict_first_ready_focal_rows"] = expected_ready_rows
        support["frozen_scope_validated"] = True
    print(
        "[alpha] prepared focal cohort "
        f"fixtures={support['first_ready_fixtures']} focal_rows={support['first_ready_focal_rows']} "
        f"events={support['candidate_event_rows']} daily_states={len(daily_native_states)}",
        flush=True,
    )
    return {
        "rows": focal.reset_index(drop=True),
        "fixtures": inventory.reset_index(drop=True),
        "events": events.reset_index(drop=True),
        "daily_native_states": daily_native_states.reset_index(drop=True),
        "support": support,
        "evidence": evidence,
    }


def blend_probability(p_base: float | np.ndarray, market_prob_a: float | np.ndarray,
                      alpha: float, epsilon: float = EPSILON) -> float | np.ndarray:
    """Blend series C0 and focal-book LOO with alpha as the SPORTS weight."""
    if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Sports alpha must be finite and in [0, 1]")
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("Probability epsilon must be finite and in (0, 0.5)")
    model = np.asarray(p_base, dtype=float)
    market = np.asarray(market_prob_a, dtype=float)
    if not np.isfinite(model).all() or not np.isfinite(market).all():
        raise ValueError("Blend inputs must be finite")
    if ((model < 0.0) | (model > 1.0)).any() or ((market < 0.0) | (market > 1.0)).any():
        raise ValueError("Blend inputs must lie in [0, 1]")
    p = np.clip(model, epsilon, 1.0 - epsilon)
    q = np.clip(market, epsilon, 1.0 - epsilon)
    logit = float(alpha) * np.log(p / (1.0 - p)) + (1.0 - float(alpha)) * np.log(q / (1.0 - q))
    probability = np.clip(1.0 / (1.0 + np.exp(-logit)), epsilon, 1.0 - epsilon)
    return float(probability) if probability.ndim == 0 else probability


def _log_loss_vector(y: np.ndarray, p: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), epsilon, 1.0 - epsilon)
    labels = np.asarray(y, dtype=float)
    return -(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))


def _ece_weighted(y: np.ndarray, p: np.ndarray, weights: np.ndarray, bins: int = 10) -> float | None:
    if len(y) == 0 or weights.sum() <= 0.0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    result = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p > lower) & ((p <= upper) if index < bins - 1 else (p <= upper))
        mass = float(weights[mask].sum())
        if mass:
            result += (mass / total) * abs(
                float(np.average(y[mask], weights=weights[mask]))
                - float(np.average(p[mask], weights=weights[mask]))
            )
    return float(result)


def _calibration_fit(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> tuple[float | None, float | None]:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None, None
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    x = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(80):
        eta = np.clip(design @ beta, -35.0, 35.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        variance = np.maximum(mu * (1.0 - mu), 1e-12)
        gradient = design.T @ (weights * (y - mu))
        information = design.T @ ((weights * variance)[:, None] * design)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return None, None
        beta += step
        if not np.isfinite(beta).all():
            return None, None
        if float(np.max(np.abs(step))) < 1e-9:
            return float(beta[1]), float(beta[0])
    return float(beta[1]), float(beta[0])


def proper_score_metrics(y: Iterable[float], p: Iterable[float],
                         weights: Iterable[float] | None = None) -> dict[str, Any]:
    """Weighted LogLoss/Brier/ECE and descriptive calibration diagnostics."""
    labels = np.asarray(y, dtype=float)
    probabilities = np.asarray(p, dtype=float)
    weight = np.ones(len(labels), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or weight.ndim != 1 or not (
        len(labels) == len(probabilities) == len(weight)
    ):
        raise ValueError("Probability metric inputs must be aligned one-dimensional vectors")
    known = np.isfinite(labels) & np.isin(labels, (0.0, 1.0))
    if not np.isfinite(probabilities[known]).all() or not np.isfinite(weight[known]).all():
        raise ValueError("Scored probabilities and weights must be finite")
    if (weight[known] < 0.0).any():
        raise ValueError("Metric weights must be nonnegative")
    labels, probabilities, weight = labels[known], probabilities[known], weight[known]
    if len(labels) == 0 or weight.sum() <= 0.0:
        return {"n": int(len(labels)), "weight_sum": 0.0, "log_loss": None,
                "brier": None, "ece": None, "calibration_slope": None,
                "calibration_intercept": None}
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Scored probabilities must lie in [0, 1]")
    probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    logloss = _log_loss_vector(labels, probabilities)
    brier = np.square(probabilities - labels)
    slope, intercept = _calibration_fit(labels, probabilities, weight)
    return {
        "n": int(len(labels)),
        "weight_sum": float(weight.sum()),
        "log_loss": float(np.average(logloss, weights=weight)),
        "brier": float(np.average(brier, weights=weight)),
        "ece": _ece_weighted(labels, probabilities, weight),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "metric_epsilon": EPSILON,
    }


def _bounded_alpha_fit(p_base: np.ndarray, market: np.ndarray, y: np.ndarray,
                       weights: np.ndarray) -> tuple[float, float]:
    """Golden-section bounded scalar minimization, with exact endpoint checks."""
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(alpha: float) -> float:
        prediction = np.asarray(blend_probability(p_base, market, alpha), dtype=float)
        return float(np.average(_log_loss_vector(y, prediction), weights=weights))

    left, right = 0.0, 1.0
    c = right - inverse_phi * (right - left)
    d = left + inverse_phi * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(120):
        if right - left <= 1e-12:
            break
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - inverse_phi * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + inverse_phi * (right - left)
            fd = objective(d)
    candidates = ((0.0, objective(0.0)), (1.0, objective(1.0)),
                  ((left + right) / 2.0, objective((left + right) / 2.0)))
    alpha, loss = min(candidates, key=lambda item: (item[1], abs(item[0] - 0.5), item[0]))
    return float(alpha), float(loss)


def fit_alpha_from_rows(rows: pd.DataFrame, *, cutoff_at: Any = "2026-08-01T00:00:00Z",
                        min_fixtures: int = 50, min_months: int = 2,
                        label_column: str = "label_available_at",
                        require_train_partition: bool = True) -> dict[str, Any]:
    """Fit only a bounded sports-weight scalar on rows with earlier-known labels."""
    if rows.empty:
        return {"status": "NOT_SELECTED_INSUFFICIENT_SUPPORT", "alpha": None,
                "weighted_log_loss": None, "train_fixtures": 0, "train_months": 0,
                "minimum_fixtures": int(min_fixtures), "minimum_months": int(min_months)}
    data = rows.copy()
    if label_column == "daily_release_label_available_at" and label_column not in data:
        if "native_effective_release_day" not in data:
            return {
                "status": "NOT_SELECTED_LABEL_CLOCK_UNAVAILABLE",
                "alpha": None,
                "weighted_log_loss": None,
                "train_fixtures": 0,
                "train_months": 0,
                "minimum_fixtures": int(min_fixtures),
                "minimum_months": int(min_months),
            }
        data[label_column] = pd.Series(
            [_label_clock(day, pd.NaT)[1] for day in data.native_effective_release_day],
            index=data.index,
            dtype="datetime64[ns, UTC]",
        )
    required = {"canonical_match_id", "decision_at", "y", "p_base", "market_prob_a", label_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Alpha fit rows lack fields: {sorted(missing)}")
    cutoff = _utc(cutoff_at)
    data["decision_at"] = _utc_series(data.decision_at)
    data[label_column] = _utc_series(data[label_column])
    data["y"] = pd.to_numeric(data.y, errors="coerce")
    data["p_base"] = pd.to_numeric(data.p_base, errors="coerce")
    data["market_prob_a"] = pd.to_numeric(data.market_prob_a, errors="coerce")
    if "fixture_weight" not in data:
        data["fixture_weight"] = 1.0
    data["fixture_weight"] = pd.to_numeric(data.fixture_weight, errors="coerce")
    known = data.y.isin((0.0, 1.0))
    earlier = data.decision_at.notna() & data.decision_at.lt(cutoff)
    labeled = data[label_column].notna() & data[label_column].le(cutoff)
    if require_train_partition and "partition" in data:
        in_train = data.partition.eq("TRAIN")
    else:
        in_train = pd.Series(True, index=data.index)
    valid = (
        in_train & earlier & labeled & known
        & np.isfinite(data.p_base) & data.p_base.between(0.0, 1.0, inclusive="both")
        & np.isfinite(data.market_prob_a) & data.market_prob_a.between(0.0, 1.0, inclusive="both")
        & np.isfinite(data.fixture_weight) & data.fixture_weight.gt(0.0)
    )
    data = data.loc[valid].copy()
    months = (
        data.drop_duplicates("canonical_match_id").decision_at.dt.strftime("%Y-%m").nunique()
        if not data.empty else 0
    )
    fixtures = int(data.canonical_match_id.nunique()) if not data.empty else 0
    result = {
        "status": "NOT_SELECTED_INSUFFICIENT_SUPPORT",
        "alpha": None,
        "weighted_log_loss": None,
        "train_fixtures": fixtures,
        "train_months": int(months),
        "minimum_fixtures": int(min_fixtures),
        "minimum_months": int(min_months),
        "selection_objective": "equal-fixture-weighted series logloss",
        "selection_scope": "strict earlier-label TRAIN only" if require_train_partition else "separate daily-release-only scenario",
    }
    if fixtures < min_fixtures or months < min_months:
        return result
    p_base = data.p_base.to_numpy(dtype=float)
    market = data.market_prob_a.to_numpy(dtype=float)
    y = data.y.to_numpy(dtype=float)
    weight = data.fixture_weight.to_numpy(dtype=float)
    alpha, loss = _bounded_alpha_fit(p_base, market, y, weight)
    result.update({"status": "SELECTED", "alpha": alpha, "weighted_log_loss": loss})
    return result


def monthly_paired_intervals(values: Iterable[float], weights: Iterable[float],
                             months: Iterable[str], month_universe: Iterable[str],
                             draws: int = 5000, seed: int = 20261022) -> dict[str, Any]:
    """Paired full-month bootstrap; any undefined resample makes its CI unresolved."""
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    block = np.asarray(months, dtype=object)
    universe = tuple(str(month) for month in month_universe)
    if value.ndim != 1 or weight.ndim != 1 or block.ndim != 1 or not (
        len(value) == len(weight) == len(block)
    ):
        raise ValueError("Monthly interval values, weights and blocks must be aligned")
    if not np.isfinite(value).all() or not np.isfinite(weight).all() or (weight < 0.0).any():
        raise ValueError("Monthly interval values and weights must be finite/nonnegative")
    if pd.isna(block).any() or len(set(universe)) != len(universe):
        raise ValueError("Monthly interval block labels and universe must be unique/present")
    unknown_months = set(map(str, block)).difference(universe)
    if unknown_months:
        raise ValueError(f"Observed months absent from bootstrap universe: {sorted(unknown_months)}")
    if isinstance(draws, bool) or int(draws) <= 0:
        raise ValueError("Bootstrap draws must be a positive integer")
    if isinstance(seed, bool):
        raise ValueError("Bootstrap seed must be an integer")
    month_index = {month: i for i, month in enumerate(universe)}
    numerator = np.zeros(len(universe), dtype=float)
    denominator = np.zeros(len(universe), dtype=float)
    for row_value, row_weight, row_month in zip(value, weight, block):
        index = month_index[str(row_month)]
        numerator[index] += row_value * row_weight
        denominator[index] += row_weight
    active = denominator > 0.0
    match_point = float(numerator.sum() / denominator.sum()) if denominator.sum() > 0.0 else None
    equal_point = (
        float(np.mean(numerator / denominator))
        if len(universe) and active.all()
        else None
    )
    rng = np.random.default_rng(int(seed))
    if universe:
        sampled = rng.integers(0, len(universe), size=(int(draws), len(universe)))
        sampled_num = numerator[sampled].sum(axis=1)
        sampled_den = denominator[sampled].sum(axis=1)
        match_defined = sampled_den > 0.0
        match_values = np.full(int(draws), np.nan, dtype=float)
        match_values[match_defined] = sampled_num[match_defined] / sampled_den[match_defined]
        equal_supported = denominator[sampled] > 0.0
        equal_defined = equal_supported.all(axis=1)
        equal_values = np.full(int(draws), np.nan, dtype=float)
        if equal_defined.any():
            equal_values[equal_defined] = np.mean(
                numerator[sampled[equal_defined]] / denominator[sampled[equal_defined]], axis=1
            )
    else:
        match_defined = equal_defined = np.zeros(int(draws), dtype=bool)
        match_values = equal_values = np.full(int(draws), np.nan, dtype=float)
    def interval(point: float | None, samples: np.ndarray, defined: np.ndarray,
                 denominator_by_month: np.ndarray) -> dict[str, Any]:
        undefined = int((~defined).sum())
        ci = (
            [float(x) for x in np.quantile(samples, [0.025, 0.975])]
            if undefined == 0 and len(universe) >= 2
            else None
        )
        return {
            "estimate": point,
            "ci": ci,
            "confidence_level": 0.95,
            "draws": int(draws),
            "undefined_replicates": undefined,
            "unsupported_months": [universe[i] for i, count in enumerate(denominator_by_month) if count <= 0.0],
            "month_universe": list(universe),
            "undefined_policy": "retain every replicate; CI null if any replicate is undefined",
        }
    return {
        "match_weighted": interval(match_point, match_values, match_defined, denominator),
        "equal_month": interval(equal_point, equal_values, equal_defined, denominator),
    }


def _metric_descriptive_intervals(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                                  weights: np.ndarray) -> dict[str, Any]:
    if frame.empty:
        return {"log_loss_ci95_descriptive": None, "brier_ci95_descriptive": None}
    ids = frame.canonical_match_id.to_numpy()
    ll = _log_loss_vector(y, p)
    brier = np.square(p - y)
    per_fixture = pd.DataFrame({"fixture": ids, "ll": ll * weights, "brier": brier * weights})
    fixture_scores = per_fixture.groupby("fixture", sort=False)[["ll", "brier"]].sum()
    n = len(fixture_scores)
    if n < 2:
        return {"log_loss_ci95_descriptive": None, "brier_ci95_descriptive": None}
    intervals = {}
    for metric in ("ll", "brier"):
        values = fixture_scores[metric].to_numpy(dtype=float)
        mean = float(values.mean())
        half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(n)
        key = "log_loss" if metric == "ll" else "brier"
        intervals[f"{key}_ci95_descriptive"] = [mean - half_width, mean + half_width]
    return intervals


def _month_universe(rows: pd.DataFrame, fixtures: pd.DataFrame, partition: str) -> list[str]:
    months: set[str] = set()
    if not fixtures.empty and {"partition", "decision_at"}.issubset(fixtures.columns):
        selected = fixtures
        if partition != "ALL":
            selected = selected.loc[selected.partition.eq(partition)]
        values = _utc_series(selected.decision_at).dropna().dt.strftime("%Y-%m")
        months.update(values.astype(str).tolist())
    subset = rows if partition == "ALL" else rows.loc[rows.partition.eq(partition)]
    if not subset.empty:
        months.update(_utc_series(subset.decision_at).dropna().dt.strftime("%Y-%m").astype(str).tolist())
    return sorted(months)


def score_alpha_grid(cohort: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Score the fixed grid/controls and make train-only scalar selection."""
    rows = cohort["rows"].copy()
    cutoff = _utc(protocol["cutoff_at"])
    fit = fit_alpha_from_rows(
        rows,
        cutoff_at=cutoff,
        min_fixtures=int(protocol.get("min_alpha_train_fixtures", 50)),
        min_months=int(protocol.get("min_alpha_train_months", 2)),
    )
    daily_fit = fit_alpha_from_rows(
        rows,
        cutoff_at=cutoff,
        min_fixtures=int(protocol.get("min_alpha_train_fixtures", 50)),
        min_months=int(protocol.get("min_alpha_train_months", 2)),
        label_column="daily_release_label_available_at",
        require_train_partition=False,
    )
    alpha_grid = [float(value) for value in protocol["alpha_grid"]]
    models = [(f"alpha_{alpha:g}", alpha, "FIXED_GRID") for alpha in alpha_grid]
    if fit["status"] == "SELECTED":
        models.append(("train_selected_alpha", float(fit["alpha"]), "STRICT_TRAIN_FIT"))
    if daily_fit["status"] == "SELECTED":
        models.append(("daily_release_only_alpha", float(daily_fit["alpha"]), "NONPRIMARY_DAILY_RELEASE_SCENARIO"))

    score_rows: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    partitions = ("ALL", "TRAIN", "ASSESS")
    draws = int(protocol.get("bootstrap", {}).get("draws", 5000))
    seed = int(protocol.get("bootstrap", {}).get("seed", 20261022))
    for partition in partitions:
        if partition == "ALL":
            part = rows.copy()
            part_mask = pd.Series(True, index=part.index)
        elif partition == "TRAIN":
            part = rows.loc[rows.partition.eq("TRAIN")].copy()
            part_mask = pd.Series(True, index=part.index)
        else:
            part = rows.loc[rows.partition.eq("ASSESS")].copy()
            part_mask = pd.Series(True, index=part.index)
        if not part.empty:
            part["y"] = pd.to_numeric(part.y, errors="coerce")
            part["p_base"] = pd.to_numeric(part.p_base, errors="coerce")
            part["market_prob_a"] = pd.to_numeric(part.market_prob_a, errors="coerce")
            if "fixture_weight" not in part:
                part["fixture_weight"] = 1.0
            part["fixture_weight"] = pd.to_numeric(part.fixture_weight, errors="coerce")
            part_mask = (
                part.y.isin((0.0, 1.0))
                & np.isfinite(part.p_base)
                & part.p_base.between(0.0, 1.0, inclusive="both")
                & np.isfinite(part.market_prob_a)
                & part.market_prob_a.between(0.0, 1.0, inclusive="both")
                & np.isfinite(part.fixture_weight)
                & part.fixture_weight.gt(0.0)
            )
        scored = part.loc[part_mask].copy()
        if scored.empty:
            y = p_base = market = weights = np.asarray([], dtype=float)
        else:
            y = scored.y.to_numpy(dtype=float)
            p_base = scored.p_base.to_numpy(dtype=float)
            market = scored.market_prob_a.to_numpy(dtype=float)
            weights = scored.fixture_weight.to_numpy(dtype=float)
        model_predictions: dict[str, np.ndarray] = {}
        for model_name, alpha, model_role in models:
            prediction = np.asarray(blend_probability(p_base, market, alpha), dtype=float)
            model_predictions[model_name] = prediction
            metrics = proper_score_metrics(y, prediction, weights)
            score = {
                "partition": partition,
                "model": model_name,
                "model_role": model_role,
                "alpha_sports_weight": alpha,
                "market_weight": 1.0 - alpha,
                "rows": int(len(scored)),
                "fixtures": int(scored.canonical_match_id.nunique()) if not scored.empty else 0,
                **metrics,
                "selection_on_assessment": False,
            }
            score.update(_metric_descriptive_intervals(scored, y, prediction, weights))
            score_rows.append(score)

        if scored.empty:
            continue
        reference_name = "alpha_0.5"
        if reference_name not in model_predictions:
            continue
        reference = model_predictions[reference_name]
        month_values = _utc_series(scored.decision_at).dt.strftime("%Y-%m").astype(str).to_numpy()
        month_universe = _month_universe(rows, cohort["fixtures"], partition)
        if not month_universe:
            month_universe = sorted(set(month_values))
        comparison_models = [
            (name, alpha, role)
            for name, alpha, role in models
            if name != reference_name
        ]
        for model_name, alpha, model_role in comparison_models:
            candidate = model_predictions[model_name]
            deltas = {
                "log_loss": _log_loss_vector(y, candidate) - _log_loss_vector(y, reference),
                "brier": np.square(candidate - y) - np.square(reference - y),
            }
            for metric, delta in deltas.items():
                paired = monthly_paired_intervals(
                    delta,
                    weights,
                    month_values,
                    month_universe,
                    draws=draws,
                    seed=seed,
                )
                intervals.append({
                    "partition": partition,
                    "model": model_name,
                    "model_role": model_role,
                    "alpha_sports_weight": alpha,
                    "reference": "alpha_0.5",
                    "measure": metric,
                    **paired,
                })

    score_table = pd.DataFrame(score_rows)
    return {
        "score_table": score_table,
        "monthly_intervals": intervals,
        "fit": fit,
        "daily_release_only_scenario": {
            **daily_fit,
            "primary": False,
            "interpretation": "Separate release-clock-only sensitivity; never substitutes for the strict result-recorded-clock primary fit.",
        },
        "selection": {
            "basis": "strict TRAIN equal-fixture-weighted series logloss only",
            "assessment_selection_performed": False,
            "retrospective_best_alpha_promotion": False,
            "fixed_grid": alpha_grid,
            "comparison_reference": 0.5,
        },
        "metric_contract": {
            "probability_epsilon": EPSILON,
            "sports_weight_semantics": "sigmoid(alpha*logit(p_C0)+(1-alpha)*logit(q_focal_LOO))",
            "series_probability_used_once": True,
            "row_weighting": "each focal bookmaker row has weight 1/focal_count, total one per fixture",
            "unknown_outcomes": "retained in fixture inventory; excluded only from proper-score calculations",
            "monthly_bootstrap": "5000 paired full-month resamples; undefined draws retained and any undefined draw nulls the interval",
        },
    }


def _output_dir(protocol: Mapping[str, Any]) -> Path:
    return _repo_path(protocol["output_root"]) / "alpha"


def _persist_prepared(cohort: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Path]:
    output = _output_dir(protocol)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "prepared_cohort": output / "prepared_cohort.parquet",
        "events": output / "events.parquet",
        "fixtures": output / "fixtures.parquet",
        "daily_native_states": output / "daily_native_states.parquet",
        "receipt": output / "preparation_receipt.json",
    }
    if paths["receipt"].exists():
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        expected_inputs = cohort.get("evidence", {}).get("input_hashes", {})
        if receipt.get("input_hashes") != expected_inputs:
            raise FileExistsError("Alpha preparation output already exists for different input hashes")
        artifacts = (
            ("prepared_cohort_sha256", paths["prepared_cohort"]),
            ("events_sha256", paths["events"]),
            ("fixtures_sha256", paths["fixtures"]),
            ("daily_native_states_sha256", paths["daily_native_states"]),
        )
        for name, path in artifacts:
            if not path.is_file() or _sha256_file(path) != receipt.get(name):
                raise ValueError(f"Prepared alpha artifact differs from its receipt: {path.name}")
        return paths
    if any(path.exists() for name, path in paths.items() if name != "receipt"):
        raise FileExistsError("Refusing to overwrite an unreceipted alpha preparation artifact")
    daily_states = cohort.get(
        "daily_native_states",
        pd.DataFrame(columns=[
            "canonical_match_id", "requested_utc_day", "usable_for_same_day_projection",
        ]),
    )
    cohort["rows"].to_parquet(paths["prepared_cohort"], index=False)
    cohort["events"].to_parquet(paths["events"], index=False)
    cohort["fixtures"].to_parquet(paths["fixtures"], index=False)
    daily_states.to_parquet(paths["daily_native_states"], index=False)
    receipt = {
        "schema": "c0_edge_alpha_preparation_v1",
        "run_id": protocol["run_id"],
        "input_hashes": cohort.get("evidence", {}).get("input_hashes", {}),
        "support": cohort.get("support", {}),
        "evidence": cohort.get("evidence", {}),
        "prepared_cohort_sha256": _sha256_file(paths["prepared_cohort"]),
        "events_sha256": _sha256_file(paths["events"]),
        "fixtures_sha256": _sha256_file(paths["fixtures"]),
        "daily_native_states_sha256": _sha256_file(paths["daily_native_states"]),
        "prepared_cohort_rows": int(len(cohort["rows"])),
        "event_rows": int(len(cohort["events"])),
        "fixture_rows": int(len(cohort["fixtures"])),
        "daily_native_state_rows": int(len(daily_states)),
        "prepared_cohort_contract": "One canonical-A focal row per bookmaker at earliest native+3-fresh-book checkpoint; equal fixture weights; no selection gates.",
    }
    _write_json(paths["receipt"], receipt)
    return paths


def _load_prepared(
    protocol: Mapping[str, Any], *, include_events: bool = False
) -> dict[str, Any] | None:
    """Load prepared parquet artifacts only after validating their receipt and sources."""
    output = _output_dir(protocol)
    receipt_path = output / "preparation_receipt.json"
    paths = {
        "rows": output / "prepared_cohort.parquet",
        "fixtures": output / "fixtures.parquet",
        "events": output / "events.parquet",
        "daily_native_states": output / "daily_native_states.parquet",
    }
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "c0_edge_alpha_preparation_v1" or receipt.get("run_id") != protocol["run_id"]:
        raise ValueError("Prepared alpha receipt has wrong schema/run identity")
    current_hashes = _source_input_hashes(protocol)
    if receipt.get("input_hashes") != current_hashes:
        raise ValueError("Prepared alpha cohort input hashes no longer match sealed timing inputs")
    hash_names = {
        "rows": "prepared_cohort_sha256",
        "fixtures": "fixtures_sha256",
        "events": "events_sha256",
        "daily_native_states": "daily_native_states_sha256",
    }
    for key, path in paths.items():
        if not path.is_file() or _sha256_file(path) != receipt.get(hash_names[key]):
            raise ValueError(f"Prepared alpha artifact differs from receipt: {path.name}")
    rows = pd.read_parquet(paths["rows"])
    fixtures = pd.read_parquet(paths["fixtures"])
    daily_native_states = pd.read_parquet(paths["daily_native_states"])
    events = pd.read_parquet(paths["events"]) if include_events else pd.DataFrame(columns=["side"])
    evidence = dict(receipt.get("evidence", {"input_hashes": current_hashes}))
    evidence["input_hashes"] = current_hashes
    return {
        "rows": rows,
        "fixtures": fixtures,
        "events": events,
        "daily_native_states": daily_native_states,
        "support": receipt.get("support", {}),
        "evidence": evidence,
        "receipt": receipt,
    }



def _input_evidence(cohort: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "c0_edge_alpha_input_evidence_v1",
        "run_id": protocol["run_id"],
        "protocol_sha256": protocol.get("_protocol_sha256"),
        "scope": protocol.get("scope"),
        "input_hashes": cohort.get("evidence", {}).get("input_hashes", {}),
        "source_paths": cohort.get("evidence", {}).get("source_paths", {}),
        "support": cohort.get("support", {}),
        "fixtures": cohort["fixtures"],
        "alpha_grid": protocol.get("alpha_grid"),
        "cutoff_at": protocol.get("cutoff_at"),
        "assessment_end_at": protocol.get("assessment_end_at"),
        "label_availability_rule": protocol.get("label_availability"),
    }
    safe = json_safe_value(payload)
    safe["evidence_sha256"] = canonical_json_sha256(safe)
    return safe


def run_alpha_audit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Run fixed proper-score curves and strict earlier-label scalar alpha fit."""
    _validate_protocol(protocol)
    output = _output_dir(protocol)
    output.mkdir(parents=True, exist_ok=True)
    print("[alpha] preparing/reusing outcome-blind focal cohort", flush=True)
    cohort = _load_prepared(protocol)
    if cohort is None:
        cohort = prepare_edge_cohort(protocol)
        prepared_paths = _persist_prepared(cohort, protocol)
    else:
        prepared_paths = {
            "prepared_cohort": output / "prepared_cohort.parquet",
            "events": output / "events.parquet",
            "fixtures": output / "fixtures.parquet",
            "receipt": output / "preparation_receipt.json",
            "daily_native_states": output / "daily_native_states.parquet",
        }
    evidence = _input_evidence(cohort, protocol)
    evidence_path = output / "input_evidence.json"
    if evidence_path.exists():
        existing = json.loads(evidence_path.read_text(encoding="utf-8"))
        existing_hash = existing.pop("evidence_sha256", None)
        if existing_hash != canonical_json_sha256(existing):
            raise ValueError("Existing alpha input evidence failed its canonical hash")
        if existing.get("input_hashes") != evidence.get("input_hashes"):
            raise FileExistsError("Alpha input evidence already exists for different input hashes")
        existing["evidence_sha256"] = existing_hash
        evidence = existing
    else:
        _write_json(evidence_path, evidence)
    print(
        f"[alpha] input evidence persisted fixtures={len(cohort['fixtures'])} "
        f"focal_rows={len(cohort['rows'])} events={cohort.get('support', {}).get('candidate_event_rows', 'cached')}",
        flush=True,
    )
    print("[alpha] scoring fixed alpha grid and earlier-only fit", flush=True)
    analysis = score_alpha_grid(cohort, protocol)
    score_table = analysis["score_table"]
    score_path = output / "score_curves.csv"
    if score_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior alpha scores: {score_path}")
    score_table.to_csv(score_path, index=False)
    intervals_path = output / "paired_monthly_intervals.json"
    if intervals_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior alpha intervals: {intervals_path}")
    _write_json(intervals_path, analysis["monthly_intervals"])
    result_path = output / "alpha_audit.json"
    result = {
        "schema": "c0_edge_alpha_audit_v1",
        "run_id": protocol["run_id"],
        "scope": "HISTORICAL_DEVELOPMENT_ONLY",
        "input_evidence_path": str(evidence_path),
        "input_evidence_sha256": evidence["evidence_sha256"],
        "prepared_cohort_path": str(prepared_paths["prepared_cohort"]),
        "fixtures_path": str(prepared_paths["fixtures"]),
        "daily_native_states_path": str(
            prepared_paths.get("daily_native_states", output / "daily_native_states.parquet")
        ),
        "events_path": str(prepared_paths["events"]),
        "score_curves_path": str(score_path),
        "paired_monthly_intervals_path": str(intervals_path),
        "support": cohort.get("support", {}),
        "strict_train_fit": analysis["fit"],
        "daily_release_only_scenario": analysis["daily_release_only_scenario"],
        "selection": analysis["selection"],
        "metric_contract": analysis["metric_contract"],
        "score_curves": score_table,
        "paired_monthly_intervals": analysis["monthly_intervals"],
        "assessment_best_alpha_promotion": False,
        "pnl_selection_used": False,
        "base_neural_refit": False,
    }
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior alpha audit results: {result_path}")
    _write_json(result_path, result)
    print(f"[alpha] completed fixed score grid; result={result_path}", flush=True)
    result["input_evidence_path"] = evidence_path
    result["result_path"] = result_path
    result["score_curves_path"] = score_path
    result["paired_monthly_intervals_path"] = intervals_path
    return result


def _prepare_only(protocol: Mapping[str, Any]) -> dict[str, Any]:
    cohort = prepare_edge_cohort(protocol)
    paths = _persist_prepared(cohort, protocol)
    evidence_path = _output_dir(protocol) / "input_evidence.json"
    evidence = _input_evidence(cohort, protocol)
    if evidence_path.exists():
        old = json.loads(evidence_path.read_text(encoding="utf-8"))
        if old.get("evidence_sha256") != evidence.get("evidence_sha256"):
            raise FileExistsError("Alpha input evidence already exists with different content")
    else:
        _write_json(evidence_path, evidence)
    return {"support": cohort["support"], "paths": paths, "input_evidence_path": evidence_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal C0 alpha proper-score audit")
    parser.add_argument("--protocol", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["_protocol_sha256"] = _sha256_file(protocol_path)
    if args.prepare:
        print("[alpha] prepare phase starting", flush=True)
        prepared = _prepare_only(protocol)
        print(canonical_json_dumps({"status": "PREPARED", **prepared}), flush=True)
    else:
        run_alpha_audit(protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
