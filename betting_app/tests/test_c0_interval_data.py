"""Checkpoint construction preserves fixture denominators and causal clocks."""

import pandas as pd

from src.analysis.c0_interval_data import build_interval_checkpoints


def fixtures():
    return pd.DataFrame([
        {"opportunity_id": "f1", "canonical_match_id": 1,
         "decision_at": pd.Timestamp("2026-10-20T12:00:00Z"),
         "start_at": pd.Timestamp("2026-10-22T12:00:00Z"),
         "y": 1.0, "identity_status": "verified"},
        {"opportunity_id": "f2", "canonical_match_id": 2,
         "decision_at": pd.NaT, "start_at": pd.NaT,
         "y": float("nan"), "identity_status": "verified"},
    ])


def test_utc_grid_boundaries_are_strictly_before_start_and_include_zero_hour():
    checkpoints, accounting = build_interval_checkpoints(fixtures(), pd.DataFrame())
    first = checkpoints.loc[checkpoints.canonical_match_id.eq(1)]
    clocks = set(first.loc[first.is_grid, "decision_at"])
    assert pd.Timestamp("2026-10-20T12:00:00Z") in clocks
    assert pd.Timestamp("2026-10-22T10:00:00Z") in clocks
    assert first.opportunity_id.is_unique
    assert pd.Timestamp("2026-10-22T12:00:00Z") not in clocks
    assert all(clock < pd.Timestamp("2026-10-22T12:00:00Z") for clock in clocks)
    assert accounting.loc[accounting.canonical_match_id.eq(1), "grid_count"].iloc[0] == 24


def test_grid_uses_utc_clock_when_start_has_non_utc_offset():
    fixture = fixtures().iloc[[0]].copy()
    fixture.loc[:, "start_at"] = pd.Timestamp("2026-10-22T14:00:00+02:00")
    checkpoints, _ = build_interval_checkpoints(fixture, pd.DataFrame())
    clocks = set(checkpoints.loc[checkpoints.is_grid, "decision_at"])
    assert pd.Timestamp("2026-10-22T10:00:00Z") in clocks
    assert pd.Timestamp("2026-10-22T12:00:00Z") not in clocks


def test_snapshots_require_strict_prestart_time_and_exclude_future_records():
    quotes = pd.DataFrame([
        {"canonical_match_id": 1, "quoted_at": "2026-10-22T11:59:00Z", "quote_id": "before"},
        {"canonical_match_id": 1, "quoted_at": "2026-10-22T12:00:00Z", "quote_id": "at-start"},
        {"canonical_match_id": 1, "quoted_at": "2026-10-22T12:01:00Z", "quote_id": "after"},
        {"canonical_match_id": 1, "quoted_at": "2026-10-20T12:00:00Z", "quote_id": "lower-bound"},
        {"canonical_match_id": 1, "quoted_at": "2026-10-20T11:59:59Z", "quote_id": "outside"},
    ])
    checkpoints, _ = build_interval_checkpoints(fixtures().iloc[[0]], quotes)
    snapshots = checkpoints.loc[checkpoints.is_snapshot]
    assert snapshots.decision_at.tolist() == [
        pd.Timestamp("2026-10-20T12:00:00Z"),
        pd.Timestamp("2026-10-22T11:59:00Z"),
    ]
    assert snapshots.snapshot_quote_ids.tolist() == [["lower-bound"], ["before"]]
    assert snapshots.decision_at.max() < pd.Timestamp("2026-10-22T12:00:00Z")


def test_no_grid_or_invalid_fixture_is_retained_in_accounting():
    fixture_rows = pd.DataFrame([
        {"opportunity_id": "invalid", "canonical_match_id": 3,
         "decision_at": pd.NaT, "start_at": pd.NaT, "y": 1.0,
         "identity_status": "unverified"},
    ])
    checkpoints, accounting = build_interval_checkpoints(fixture_rows, pd.DataFrame())
    assert checkpoints.empty
    assert len(accounting) == 1
    assert accounting.iloc[0].canonical_match_id == 3
    assert accounting.iloc[0].grid_count == 0
    assert accounting.iloc[0].snapshot_count == 0
    assert accounting.iloc[0].checkpoint_absence == "schedule_unknown"


def test_scheduled_unverified_fixture_keeps_checkpoint_and_identity_absence():
    fixture = pd.DataFrame([{
        "opportunity_id": "unverified", "canonical_match_id": 4,
        "decision_at": pd.NaT, "start_at": pd.Timestamp("2026-10-22T12:00:00Z"),
        "y": 0.0, "identity_status": "unverified",
    }])
    checkpoints, accounting = build_interval_checkpoints(fixture, pd.DataFrame())
    assert len(checkpoints) == 24
    assert checkpoints.checkpoint_absence.eq("identity_unverified").all()
    assert not bool(accounting.iloc[0].fixture_valid)
    assert accounting.iloc[0].checkpoint_absence == "identity_unverified"


def test_distinct_subsecond_snapshots_keep_distinct_forecast_join_keys():
    quotes = pd.DataFrame([
        {"canonical_match_id": 1, "quoted_at": "2026-10-22T11:59:00.100Z", "quote_id": "first"},
        {"canonical_match_id": 1, "quoted_at": "2026-10-22T11:59:00.200Z", "quote_id": "second"},
    ])
    checkpoints, _ = build_interval_checkpoints(fixtures().iloc[[0]], quotes)
    snapshots = checkpoints.loc[checkpoints.is_snapshot]
    assert snapshots.opportunity_id.is_unique
    assert snapshots.snapshot_quote_ids.tolist() == [["first"], ["second"]]


def test_native_contract_hash_survives_json_roundtrip_with_missing_support():
    import hashlib
    import json
    from src.analysis.c0_interval_data import _bind_native_manifest

    manifest = pd.DataFrame([
        {"representative_opportunity_id": name, "native_daily_key": {"mode": "full"}}
        for name in ("supported", "missing")
    ])
    forecasts = pd.DataFrame([
        {"opportunity_id": "supported", "prior_roster_a": ["a"],
         "prior_roster_b": ["b"], "native_input_hashes": {"raw.core": "digest"},
         "feature_source_max_day": "2026-01-01"},
        {"opportunity_id": "missing", "prior_roster_a": float("nan"),
         "prior_roster_b": float("nan"), "native_input_hashes": float("nan"),
         "feature_source_max_day": float("nan")},
    ])
    bound = _bind_native_manifest(manifest, forecasts)
    saved = json.loads(bound.to_json(orient="records", default_handler=str))
    for row in saved:
        canonical = json.dumps(
            row["native_feature_contract"], sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == row["native_feature_contract_sha256"]
    assert saved[1]["native_feature_contract"]["native_input_hashes"] is None
