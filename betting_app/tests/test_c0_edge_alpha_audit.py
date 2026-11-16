import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.analysis import c0_edge_alpha_audit as audit


def _protocol():
    return {
        "schema": "c0_edge_recovery_v1",
        "run_id": "c0_edge_recovery_20261022_01",
        "scope": "HISTORICAL_DEVELOPMENT_ONLY",
        "formal_base_fits": 0,
        "inputs": {"timing_root": "data/08_reporting/c0_interval_policy/frozen"},
        "output_root": "data/08_reporting/c0_edge_recovery/c0_edge_recovery_20261022_01",
        "alpha_grid": [0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 1.0],
        "cutoff_at": "2026-08-01T00:00:00Z",
        "assessment_end_at": "2026-10-01T00:00:00Z",
        "min_alpha_train_fixtures": 50,
        "min_alpha_train_months": 2,
        "bootstrap": {"draws": 5000, "seed": 20261022},
        "cohort": {"universe_fixtures": 1771, "freshness_seconds": 300},
        "alpha_semantics": {
            "sports_weight": "alpha",
            "market_weight": "1-alpha",
            "blend": "sigmoid(alpha*logit(p_C0)+(1-alpha)*logit(q_focal_LOO))",
        },
        "label_availability": {
            "strict_clock": "max(midnight after native effective release day, result_recorded_at when known)",
        },
    }


def _saved_fixture_tables(match_id=11):
    """Build deterministic saved-artifact-shaped rows; no ignored cache dependency."""
    start = pd.Timestamp("2026-05-30T17:34:59Z")
    first_id = "c0-11-interval-1780066499000000000"
    second_id = "c0-11-interval-1780067099000000000"
    checkpoints = (
        (first_id, pd.Timestamp("2026-05-29T17:34:59Z"), False, True),
        (second_id, pd.Timestamp("2026-05-29T17:44:59Z"), True, False),
    )
    opportunities, forecasts, quotes, candidates = [], [], [], []
    for checkpoint_index, (opportunity_id, decision_at, is_grid, is_snapshot) in enumerate(checkpoints):
        opportunity = {
            "opportunity_id": opportunity_id,
            "canonical_match_id": match_id,
            "decision_at": decision_at,
            "start_at": start,
            "settle_at": start + pd.Timedelta(hours=8),
            "result_recorded_at": pd.NaT,
            "y": np.nan,
            "league": "LCK",
            "best_of": 3,
            "canonical_best_of": 3,
            "team1_id": "team-1",
            "team2_id": "team-2",
            "canonical_a_is_team1": True,
            "golgg_match_id": f"game-{match_id}",
            "competition_context": "LCK",
            "is_grid": is_grid,
            "is_snapshot": is_snapshot,
            "hours_before_start": (start - decision_at).total_seconds() / 3600.0,
            "checkpoint_absence": None,
        }
        opportunities.append(opportunity)
        forecast = {
            "opportunity_id": opportunity_id,
            "canonical_match_id": match_id,
            "p_base": 0.72,
            "forecast_available": True,
            "identity_verified": True,
            "forecast_at": decision_at,
            "native_absence": None,
            "native_effective_release_day": "2026-05-30",
            "daily_state_cutoff_day": "2026-05-29",
            "daily_state_key_sha256": "d" * 64,
            "native_feature_contract_sha256": "e" * 64,
            "native_input_hashes": {"raw": "a" * 64, "release": "b" * 64},
            "feature_source_max_day": "2026-05-28",
            "prior_roster_a": ["team-1-player-a"],
            "prior_roster_b": ["team-2-player-b"],
            "prior_source_a": {"source": "prior-anchor"},
            "prior_source_b": {"source": "prior-anchor"},
            "forecast_clock_basis": "daily_state_projection_not_issuance",
            "daily_state_projection": True,
            "issuance_authenticated": False,
            "daily_state_representative_opportunity_id": first_id,
            "daily_state_representative_at": checkpoints[0][1],
        }
        forecasts.append(forecast)
        book_prices = [
            (1.90 + book * 0.01, 2.00 + book * 0.01)
            for book in range(3)
        ]
        no_vig_probabilities = [
            (1.0 / odds_a) / ((1.0 / odds_a) + (1.0 / odds_b))
            for odds_a, odds_b in book_prices
        ]
        for book, (odds_a, odds_b) in enumerate(book_prices):
            bookmaker_id = 100 + book
            quote_id = 1100 + checkpoint_index * 10 + book
            quote = {
                "quote_id": quote_id,
                "canonical_match_id": match_id,
                "bookmaker_id": bookmaker_id,
                "bookmaker": f"book-{book}",
                "quoted_at": decision_at,
                "odds_a": odds_a,
                "odds_b": odds_b,
                "canonical_odds_a": odds_a,
                "canonical_odds_b": odds_b,
                "raw_a_is_canonical_a": True,
                "identity_verified": True,
                "is_live": False,
            }
            quotes.append(quote)
            focal_market_prob = sum(
                probability
                for other_book, probability in enumerate(no_vig_probabilities)
                if other_book != book
            ) / 2.0
            for side, odds, market_prob in (
                ("a", odds_a, focal_market_prob),
                ("b", odds_b, 1.0 - focal_market_prob),
            ):
                candidates.append(
                    {
                        **opportunity,
                        **{
                            key: value
                            for key, value in forecast.items()
                            if key != "opportunity_id"
                        },
                        "quote_id": quote_id,
                        "bookmaker_id": bookmaker_id,
                        "bookmaker": quote["bookmaker"],
                        "quoted_at": decision_at,
                        "side": side,
                        "odds": odds,
                        "market_prob_a": focal_market_prob,
                        "market_prob": market_prob,
                        "base_prob": 0.72 if side == "a" else 0.28,
                        "prob_a": 0.72,
                        "prob_side": 0.72 if side == "a" else 0.28,
                        "prob_for_ev": 0.72 if side == "a" else 0.28,
                        "ev_net": np.nan if checkpoint_index == 0 else 0.25,
                        "model_ev": np.nan if checkpoint_index == 0 else 0.25,
                        "diagnostics": {},
                        "eligible": bool(checkpoint_index == 1),
                        "reason": "eligible" if checkpoint_index == 1 else "safety_gate",
                        "exp039_status": "not_evaluated",
                        "uncertainty_basis": "mean",
                        "native_bound_status": "unavailable",
                    }
                )
    return {
        "opportunities": pd.DataFrame(opportunities),
        "forecasts": pd.DataFrame(forecasts),
        "quotes": pd.DataFrame(quotes),
        "candidates": pd.DataFrame(candidates),
        "fixture_accounting": pd.DataFrame(
            [
                {
                    "canonical_match_id": match_id,
                    "y": np.nan,
                    "start_at": start,
                    "result_recorded_at": pd.NaT,
                    "cash_status": "pending",
                    "pending_funds": np.nan,
                }
            ]
        ),
        "input_hashes": {
            "opportunities.parquet": "c" * 64,
            "forecasts.parquet": "b" * 64,
            "quotes.parquet": "a" * 64,
            "candidates.parquet": "d" * 64,
        },
    }


def _audit_rows():
    rows = []
    for i in range(60):
        train = i < 50
        y = float(i % 2)
        decision = (
            pd.Timestamp("2026-05-10T12:00:00Z") + pd.Timedelta(days=i % 20)
            if i < 25
            else pd.Timestamp("2026-06-10T12:00:00Z") + pd.Timedelta(days=i % 20)
            if train
            else pd.Timestamp("2026-09-10T12:00:00Z") + pd.Timedelta(days=i % 10)
        )
        rows.append(
            {
                "opportunity_id": f"opp-{i}",
                "canonical_match_id": i,
                "decision_at": decision,
                "start_at": decision + pd.Timedelta(hours=12),
                "settle_at": decision + pd.Timedelta(hours=20),
                "label_available_at": pd.Timestamp("2026-07-01T00:00:00Z")
                if train
                else pd.Timestamp("2026-09-20T00:00:00Z"),
                "y": y,
                "p_base": 0.9 if y else 0.1,
                "market_prob_a": 0.1 if y else 0.9,
                "fixture_weight": 1.0,
                "partition": "TRAIN" if train else "ASSESS",
                "bookmaker_id": 1,
                "bookmaker": "book-a",
                "quote_id": i,
                "quoted_at": decision,
                "odds": 2.0,
                "odds_b": 2.0,
                "league": "LCK",
                "best_of": 3,
                "native_effective_release_day": "2026-06-01",
                "result_recorded_at": pd.Timestamp("2026-06-30T12:00:00Z")
                if train
                else pd.Timestamp("2026-09-19T12:00:00Z"),
            }
        )
    return pd.DataFrame(rows)


def test_saved_non_grid_snapshot_is_focalized_even_when_every_gate_rejects_it(monkeypatch):
    cached = _saved_fixture_tables()
    source_candidate = cached["candidates"].loc[
        cached["candidates"].opportunity_id.eq(
            "c0-11-interval-1780066499000000000"
        )
    ]
    assert not source_candidate.empty
    assert source_candidate.is_grid.eq(False).all()
    assert source_candidate.is_snapshot.eq(True).all()
    assert source_candidate.eligible.eq(False).all()
    assert source_candidate.ev_net.isna().all()
    assert cached["candidates"].loc[
        cached["candidates"].opportunity_id.eq("c0-11-interval-1780067099000000000"),
        "ev_net",
    ].gt(0.0).all()

    monkeypatch.setattr(audit, "_load_cached_inputs", lambda protocol: cached)
    cohort = audit.prepare_edge_cohort(_protocol())

    focal = cohort["rows"].loc[
        cohort["rows"].canonical_match_id.eq(11)
    ]
    assert focal.opportunity_id.nunique() == 1
    assert focal.opportunity_id.iloc[0] == "c0-11-interval-1780066499000000000"
    assert focal.bookmaker_id.nunique() == 3
    assert focal.is_grid.eq(False).all()
    assert focal.is_snapshot.eq(True).all()
    assert focal.eligible.eq(False).all()
    assert focal.ev_net.isna().all()
    assert set(cohort["events"].side) == {"a", "b"}
    assert focal.team1_id.eq("team-1").all()
    assert focal.team2_id.eq("team-2").all()
    assert focal.canonical_a_is_team1.eq(True).all()
    assert focal.golgg_match_id.eq("game-11").all()
    assert focal.competition_context.eq("LCK").all()
    assert cohort["daily_native_states"].team1_id.eq("team-1").all()
    assert cohort["daily_native_states"].canonical_a_is_team1.eq(True).all()
    assert cohort["daily_native_states"].golgg_match_id.eq("game-11").all()
    assert cohort["daily_native_states"].competition_context.eq("LCK").all()
    assert focal.prior_roster_orientation.eq("native_team1_team2").all()
    assert cohort["daily_native_states"].prior_roster_orientation.eq(
        "native_team1_team2"
    ).all()
    assert focal.prior_roster_a.map(
        lambda roster: tuple(roster) == ("team-1-player-a",)
    ).all()
    assert focal.prior_roster_b.map(
        lambda roster: tuple(roster) == ("team-2-player-b",)
    ).all()
    assert cached["candidates"].loc[
        cached["candidates"].opportunity_id.eq("c0-11-interval-1780067099000000000"),
        "eligible",
    ].all()
    assert set(focal.columns).issuperset(
        {
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
        }
    )
    for row in focal.itertuples(index=False):
        paired_b = cohort["events"].loc[
            cohort["events"].opportunity_id.eq(row.opportunity_id)
            & cohort["events"].bookmaker_id.eq(row.bookmaker_id)
            & cohort["events"].side.eq("b")
        ]
        assert len(paired_b) == 1
        assert row.odds_b == paired_b.odds.iloc[0]


def test_first_ready_uses_strictly_less_than_48_hours(monkeypatch):
    cached = _saved_fixture_tables()
    first_id = "c0-11-interval-1780066499000000000"
    second_id = "c0-11-interval-1780067099000000000"
    start = cached["opportunities"].start_at.iloc[0]
    decisions = {
        first_id: start - pd.Timedelta(hours=48),
        second_id: start - pd.Timedelta(hours=47),
    }
    for table_name in ("opportunities", "forecasts"):
        table = cached[table_name]
        table["decision_at"] = table.opportunity_id.map(decisions)
        if "forecast_at" in table:
            table["forecast_at"] = table.opportunity_id.map(decisions)
    cached["opportunities"]["hours_before_start"] = (
        cached["opportunities"].start_at - cached["opportunities"].decision_at
    ).dt.total_seconds() / 3600.0
    candidates = cached["candidates"]
    candidates["decision_at"] = candidates.opportunity_id.map(decisions)
    candidates["forecast_at"] = candidates.opportunity_id.map(decisions)
    candidates["quoted_at"] = candidates.decision_at
    candidates["hours_before_start"] = (
        candidates.start_at - candidates.decision_at
    ).dt.total_seconds() / 3600.0
    quote_times = candidates.drop_duplicates("quote_id").set_index("quote_id").quoted_at
    cached["quotes"]["quoted_at"] = cached["quotes"].quote_id.map(quote_times)
    monkeypatch.setattr(audit, "_load_cached_inputs", lambda protocol: cached)

    cohort = audit.prepare_edge_cohort(_protocol())

    assert cohort["rows"].opportunity_id.unique().tolist() == [second_id]
    assert cohort["rows"].lead_hours.eq(47.0).all()
    assert cohort["support"]["first_ready_boundary_correction"][
        "inclusive_selected_at_exact_48h_focal_rows"
    ] == 3



def test_sports_alpha_weight_and_proper_score_controls_are_exact():
    y = np.array([1.0, 0.0])
    model = np.array([0.9, 0.1])
    market = np.array([0.8, 0.2])

    assert audit.blend_probability(model[0], market[0], alpha=0.0) == pytest.approx(0.8)
    assert audit.blend_probability(model[0], market[0], alpha=1.0) == pytest.approx(0.9)
    expected_sports_035 = 1 / (
        1 + np.exp(-((0.35 * np.log(0.9 / 0.1)) + (0.65 * np.log(0.8 / 0.2))))
    )
    assert audit.blend_probability(model[0], market[0], alpha=0.35) == pytest.approx(
        expected_sports_035
    )

    market_metrics = audit.proper_score_metrics(y, market)
    c0_metrics = audit.proper_score_metrics(y, model)
    assert market_metrics["n"] == c0_metrics["n"] == 2
    assert market_metrics["log_loss"] == pytest.approx(-np.log(0.8))
    assert market_metrics["brier"] == pytest.approx(0.04)
    assert c0_metrics["log_loss"] == pytest.approx(-np.log(0.9))
    assert c0_metrics["brier"] == pytest.approx(0.01)
    rows = _audit_rows()
    protocol = _protocol()
    protocol["bootstrap"]["draws"] = 2
    cohort = {
        "rows": rows,
        "fixtures": rows[["canonical_match_id", "decision_at", "partition"]].copy(),
    }
    curves = audit.score_alpha_grid(cohort, protocol)["score_table"]
    train = curves.loc[curves.partition.eq("TRAIN")].set_index("model")
    assert train.loc["alpha_0", "log_loss"] == pytest.approx(-np.log(0.1))
    assert train.loc["alpha_0", "brier"] == pytest.approx(0.81)
    assert train.loc["alpha_1", "log_loss"] == pytest.approx(-np.log(0.9))
    assert train.loc["alpha_1", "brier"] == pytest.approx(0.01)


def test_train_fit_is_bounded_support_checked_and_assessment_label_invariant():
    rows = _audit_rows()
    fit = audit.fit_alpha_from_rows(rows)
    assert fit["status"] == "SELECTED"
    assert fit["alpha"] == pytest.approx(1.0)
    assert fit["train_fixtures"] == 50
    assert fit["train_months"] >= 2

    altered = rows.copy()
    assess = altered.partition.eq("ASSESS")
    altered.loc[assess, "y"] = 1.0 - altered.loc[assess, "y"]
    altered.loc[assess, "label_available_at"] = pd.Timestamp("2026-10-01T00:00:00Z")
    assert audit.fit_alpha_from_rows(altered)["alpha"] == fit["alpha"]
    altered.loc[assess, "p_base"] = 1.0 - altered.loc[assess, "p_base"]
    altered.loc[assess, "market_prob_a"] = 1.0 - altered.loc[assess, "market_prob_a"]
    assert audit.fit_alpha_from_rows(altered.sample(frac=1, random_state=7))["alpha"] == pytest.approx(
        fit["alpha"]
    )
    assert audit.fit_alpha_from_rows(rows.sample(frac=1, random_state=19))["alpha"] == pytest.approx(
        fit["alpha"]
    )

    insufficient = rows.loc[rows.partition.eq("TRAIN")].iloc[:49].copy()
    rejected = audit.fit_alpha_from_rows(insufficient)
    assert rejected["status"] == "NOT_SELECTED_INSUFFICIENT_SUPPORT"
    assert rejected["alpha"] is None


def test_run_persists_null_safe_hash_receipt_before_scoring_and_keeps_pending_fixture(
    tmp_path, monkeypatch
):
    rows = _audit_rows()
    fixtures = pd.DataFrame(
        [
            {
                "canonical_match_id": 9001,
                "y": np.nan,
                "result_recorded_at": pd.NaT,
                "label_available_at": pd.NaT,
                "cash_status": "pending",
                "pending_funds": np.nan,
            }
        ]
    )
    cohort = {
        "rows": rows,
        "fixtures": fixtures,
        "events": pd.DataFrame(columns=["side", "quote_id"]),
        "support": {"all_fixture_count": 1, "protocol_universe_fixtures": 1771},
        "evidence": {
            "input_hashes": {
                "candidates.parquet": "a" * 64,
                "forecasts.parquet": "b" * 64,
                "opportunities.parquet": "c" * 64,
            }
        },
    }
    monkeypatch.setattr(audit, "prepare_edge_cohort", lambda protocol: cohort)
    alpha_root = tmp_path / "alpha"
    original_score = audit.score_alpha_grid

    def require_receipt_before_score(*args, **kwargs):
        receipt_path = alpha_root / "input_evidence.json"
        assert receipt_path.is_file()
        receipt = receipt_path.read_text()
        assert "NaN" not in receipt and "Infinity" not in receipt
        parsed = json.loads(receipt)
        assert parsed["input_hashes"]["candidates.parquet"] == "a" * 64
        assert parsed["fixtures"][0]["y"] is None
        assert parsed["fixtures"][0]["pending_funds"] is None
        assert parsed["fixtures"][0]["cash_status"] == "pending"
        return original_score(*args, **kwargs)

    monkeypatch.setattr(audit, "score_alpha_grid", require_receipt_before_score)
    protocol = _protocol()
    protocol["output_root"] = str(tmp_path)
    result = audit.run_alpha_audit(protocol)
    assert result["input_evidence_path"] == alpha_root / "input_evidence.json"
    evidence = json.loads((alpha_root / "input_evidence.json").read_text())
    assert evidence["input_hashes"] == cohort["evidence"]["input_hashes"]
    assert evidence["fixtures"][0]["y"] is None
    assert evidence["fixtures"][0]["cash_status"] == "pending"
    canonical_payload = dict(evidence)
    expected_hash = canonical_payload.pop("evidence_sha256")
    round_tripped = json.loads(
        json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    )
    assert audit.canonical_json_sha256(round_tripped) == expected_hash


def test_fixture_display_clock_does_not_replace_precise_opportunity_clock():
    tables = _saved_fixture_tables()
    exact = pd.Timestamp("2026-05-30T17:34:59.123456Z")
    tables["opportunities"]["result_recorded_at"] = exact
    tables["fixture_accounting"]["result_recorded_at"] = "2026-05-30T17:34:59.123Z"
    audit._validate_fixture_outcome_clocks(
        (tables["opportunities"], tables["candidates"]),
        tables["fixture_accounting"],
    )
    bad_display = tables["fixture_accounting"].copy()
    bad_display["result_recorded_at"] = "2026-05-30T17:34:59.124Z"
    with pytest.raises(ValueError, match="millisecond projection"):
        audit._validate_fixture_outcome_clocks(
            (tables["opportunities"], tables["candidates"]), bad_display
        )
    inventory = audit._build_fixture_inventory(
        tables["fixture_accounting"],
        tables["opportunities"],
        pd.DataFrame(),
        pd.Timestamp("2026-08-01T00:00:00Z"),
        pd.Timestamp("2026-10-01T00:00:00Z"),
    )

    assert inventory.result_recorded_at.iloc[0] == exact
    assert inventory.fixture_accounting_result_recorded_at_display.iloc[0] == (
        "2026-05-30T17:34:59.123Z"
    )
    assert inventory.fixture_accounting_result_recorded_at_precision.iloc[0] == "milliseconds"
    assert inventory.result_recorded_at_source.iloc[0] == "opportunities.parquet"


def test_candidate_result_clock_must_match_exact_opportunity_clock(monkeypatch):
    cached = _saved_fixture_tables()
    exact = pd.Timestamp("2026-05-30T17:34:59.123456Z")
    cached["opportunities"]["result_recorded_at"] = exact
    cached["candidates"]["result_recorded_at"] = exact + pd.Timedelta(microseconds=1)
    monkeypatch.setattr(audit, "_load_cached_inputs", lambda protocol: cached)

    with pytest.raises(ValueError, match="clocks differ in result_recorded_at"):
        audit.prepare_edge_cohort(_protocol())




def test_canonical_json_handles_python_and_numpy_strings_without_changing_decimal_contract():
    assert audit.canonical_json_dumps(
        {
            "python": "team",
            "numpy": np.str_("team"),
            "stake": Decimal("25.00"),
            "undefined": Decimal("NaN"),
        }
    ) == '{"numpy":"team","python":"team","stake":"25.00","undefined":null}'


def test_verified_prepared_cache_loads_events_and_rejects_hash_mismatch(tmp_path, monkeypatch):
    protocol = _protocol()
    output = tmp_path / "alpha"
    output.mkdir()
    frames = {
        "prepared_cohort.parquet": pd.DataFrame({"opportunity_id": ["opp-1"]}),
        "fixtures.parquet": pd.DataFrame({"canonical_match_id": [1]}),
        "events.parquet": pd.DataFrame({"quote_id": [10], "side": ["a"]}),
        "daily_native_states.parquet": pd.DataFrame({"requested_utc_day": ["2026-05-01"]}),
    }
    for name, frame in frames.items():
        frame.to_parquet(output / name, index=False)
    input_hashes = {"opportunities.parquet": "a" * 64}
    monkeypatch.setattr(audit, "_output_dir", lambda value: output)
    monkeypatch.setattr(audit, "_source_input_hashes", lambda value: input_hashes)
    receipt = {
        "schema": "c0_edge_alpha_preparation_v1",
        "run_id": protocol["run_id"],
        "input_hashes": input_hashes,
        "prepared_cohort_sha256": audit._sha256_file(output / "prepared_cohort.parquet"),
        "fixtures_sha256": audit._sha256_file(output / "fixtures.parquet"),
        "events_sha256": audit._sha256_file(output / "events.parquet"),
        "daily_native_states_sha256": audit._sha256_file(output / "daily_native_states.parquet"),
    }
    audit._write_json(output / "preparation_receipt.json", receipt)

    loaded = audit._load_prepared(protocol, include_events=True)

    assert loaded["rows"].opportunity_id.tolist() == ["opp-1"]
    assert loaded["events"].quote_id.tolist() == [10]
    assert loaded["daily_native_states"].requested_utc_day.tolist() == ["2026-05-01"]
    frames["events.parquet"].assign(quote_id=11).to_parquet(output / "events.parquet", index=False)
    with pytest.raises(ValueError, match="events.parquet"):
        audit._load_prepared(protocol, include_events=True)


def test_empty_saved_candidate_frame_retains_fixture_and_a_b_event_schema(monkeypatch):
    cached = _saved_fixture_tables()
    cached["candidates"] = cached["candidates"].iloc[0:0].copy()
    monkeypatch.setattr(audit, "_load_cached_inputs", lambda protocol: cached)

    cohort = audit.prepare_edge_cohort(_protocol())

    assert cohort["rows"].empty
    assert cohort["events"].empty
    assert "side" in cohort["events"]
    assert set(cohort["fixtures"].canonical_match_id) == {11}
    fixture = cohort["fixtures"].iloc[0]
    assert fixture.partition == "NO_FIRST_READY_STATE"
    assert fixture.cash_decision_at == pd.Timestamp("2026-05-29T17:34:59Z")
    assert fixture.cash_clock_basis == "earliest_saved_opportunity_decision_fallback"
    assert fixture.cash_partition == "TRAIN_CLOCK"
    assert not fixture.cash_partition_uses_labels

    unknown_start = _saved_fixture_tables()
    unknown_start["opportunities"]["start_at"] = pd.NaT
    unknown_start["fixture_accounting"]["start_at"] = pd.NaT
    unknown_start["candidates"] = unknown_start["candidates"].iloc[0:0].copy()
    monkeypatch.setattr(audit, "_load_cached_inputs", lambda protocol: unknown_start)
    unassigned = audit.prepare_edge_cohort(_protocol())["fixtures"].iloc[0]
    assert unassigned.cash_partition == "UNASSIGNED"
    assert unassigned.cash_start_status == "UNKNOWN_START"

    future_start = _saved_fixture_tables()
    future_start["opportunities"]["start_at"] = pd.Timestamp("2026-11-01T00:00:00Z")
    future_start["fixture_accounting"]["start_at"] = pd.Timestamp("2026-11-01T00:00:00Z")
    future_start["opportunities"]["decision_at"] = pd.Timestamp("2026-10-02T00:00:00Z")
    future_start["candidates"] = future_start["candidates"].iloc[0:0].copy()
    monkeypatch.setattr(audit, "_load_cached_inputs", lambda protocol: future_start)
    outside = audit.prepare_edge_cohort(_protocol())["fixtures"].iloc[0]
    assert outside.cash_decision_at == pd.Timestamp("2026-10-02T00:00:00Z")
    assert outside.cash_partition == "OUTSIDE_FIXED_WINDOWS"
    assert outside.cash_start_status == "KNOWN_START"


def test_monthly_intervals_keep_undefined_resamples_instead_of_dropping_them():
    result = audit.monthly_paired_intervals(
        values=np.array([0.25]),
        weights=np.array([1.0]),
        months=np.array(["2026-05"]),
        month_universe=["2026-05", "2026-06"],
        draws=5000,
        seed=20261022,
    )

    for estimate in (result["match_weighted"], result["equal_month"]):
        assert estimate["undefined_replicates"] > 0
        assert estimate["ci"] is None
