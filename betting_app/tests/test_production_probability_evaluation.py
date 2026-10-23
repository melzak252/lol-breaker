from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.production_probability_evaluation import (
    evaluate_production_probabilities,
    prediction_request_manifest,
)


def _inputs():
    snapshots = pd.DataFrame([
        {"snapshot_id": "q1", "canonical_match_id": "f1", "observed_at": "2026-01-01T00:00:00Z",
         "best_of": 3,
         "earliest_start_utc": "2026-01-01T04:00:00Z", "export_clock_utc": "2026-01-02T00:00:00Z",
         "outcome_known_at": "2026-01-01T06:00:00Z", "source_match_id": "s1", "mapping_source_match_id": "s1",
         "mapping_status": "verified", "source_binding_basis": "independent_source_proof", "bookmaker": "bookA",
         "odds_a": 2.0, "odds_b": 2.0, "raw_a_is_team_a": True, "market_type": "match_winner",
         "is_live": False, "snapshot_status": "observed_price", "availability_basis": "recorded_scraped_price_conditional",
         "outcome": 1, "outcome_status": "known"},
        {"snapshot_id": "q2", "canonical_match_id": "f1", "observed_at": "2026-01-01T01:00:00Z",
         "best_of": 3,
         "earliest_start_utc": "2026-01-01T04:00:00Z", "export_clock_utc": "2026-01-02T00:00:00Z",
         "outcome_known_at": "2026-01-01T06:00:00Z", "source_match_id": "s1", "mapping_source_match_id": "s1",
         "mapping_status": "verified", "source_binding_basis": "independent_source_proof", "bookmaker": "bookA",
         "odds_a": 4.0, "odds_b": 2.0, "raw_a_is_team_a": False, "market_type": "match_winner",
         "is_live": False, "snapshot_status": "observed_price", "availability_basis": "recorded_scraped_price_conditional",
         "outcome": 1, "outcome_status": "known"},
        {"snapshot_id": "q3", "canonical_match_id": "f2", "observed_at": "2026-01-01T00:00:00Z",
         "best_of": 3,
         "earliest_start_utc": "2026-01-01T04:00:00Z", "export_clock_utc": "2026-01-02T00:00:00Z",
         "outcome_known_at": None, "source_match_id": "s2", "mapping_source_match_id": "wrong",
         "mapping_status": "verified", "source_binding_basis": "independent_source_proof", "bookmaker": "bookB",
         "odds_a": 2.0, "odds_b": 2.0, "raw_a_is_team_a": True, "market_type": "match_winner",
         "is_live": False, "snapshot_status": "observed_price", "availability_basis": "recorded_scraped_price_conditional",
         "outcome": None, "outcome_status": "unknown"},
    ])
    snapshots["duplicate_representation"] = False
    predictions = pd.DataFrame([
        {"canonical_match_id": "f1", "model_group": "stored-production:v1", "model_id": "m1",
         "model_source_match_id": "s1", "model_binding_status": "verified",
         "model_binding_basis": "independent_source_proof", "model_clock_utc": "2026-01-01T00:00:00Z",
         "model_data_cutoff_utc": "2025-12-31T23:00:00Z", "prediction_status": "valid", "p_team1": .8, "p_team2": .2},
    ])
    return snapshots, predictions


def test_all_snapshots_accounted_and_primary_fixture_weighted_not_quote_weighted():
    snapshots, predictions = _inputs()
    result = evaluate_production_probabilities(snapshots, predictions)
    audit = result["accounting"]
    assert set(audit.snapshot_id) == {"q1", "q2", "q3"}
    assert audit.loc[audit.snapshot_id.eq("q3"), "pair_exclusion_reason"].item() == "source_mapping_conflict_or_unproven_basis"
    scored = result["fixture_panel"]
    all_bin = scored.loc[scored.endpoint.eq("ALL")]
    assert all_bin.canonical_match_id.tolist() == ["f1"]
    assert all_bin.p_model_calibration.item() == pytest.approx(.8)
    # Canonical A is raw B: (1/2) / ((1/4)+(1/2)) = 2/3.
    assert all_bin.p_book_calibration.item() == pytest.approx((.5 + 2/3)/2)
    assert all_bin.n_observations.item() == 2
    assert result["fixture_scores"].query("comparator == 'model' and endpoint == 'ALL'").n.item() == 1
    assert result["observation_scores"].query("comparator == 'model'").n.item() == 2


def test_quote_uses_latest_causal_forecast_and_does_not_revive_withdrawn_state():
    snapshots, predictions = _inputs()
    later = predictions.iloc[0].copy()
    later["model_clock_utc"] = "2026-01-01T00:30:00Z"
    later["prediction_status"] = "withdrawn"
    predictions = pd.concat([predictions, later.to_frame().T], ignore_index=True)
    result = evaluate_production_probabilities(snapshots.iloc[[1]], predictions)
    assert not result["accounting"].scorable.any()
    assert result["accounting"].pair_exclusion_reason.item() == "latest_forecast_invalid_or_withdrawn"


def test_prediction_after_quote_cannot_be_used_by_asof_join():
    snapshots, predictions = _inputs()
    predictions.loc[0, "model_clock_utc"] = "2026-01-01T00:01:00Z"
    result = evaluate_production_probabilities(snapshots.iloc[[0]], predictions)
    assert not result["accounting"].scorable.any()
    assert result["accounting"].pair_exclusion_reason.item() == "no_prior_forecast"


def test_request_manifest_deduplicates_fixture_day_at_earliest_causal_clock():
    fixtures = pd.DataFrame([
        {"canonical_match_id": "f1", "source_match_id": "s1", "release_day": "2026-01-01", "request_clock_utc": "2026-01-01T02:00:00Z"},
        {"canonical_match_id": "f1", "source_match_id": "s1", "release_day": "2026-01-01", "request_clock_utc": "2026-01-01T01:00:00Z"},
        {"canonical_match_id": "f1", "source_match_id": "s1", "release_day": "2026-01-02", "request_clock_utc": "2026-01-02T01:00:00Z"},
    ])
    result = prediction_request_manifest(fixtures)
    assert len(result) == 2
    assert result.iloc[0].request_clock_utc == pd.Timestamp("2026-01-01T01:00:00Z")
    assert result.source_join_fanout.tolist() == [2, 1]
    assert not result.cache_reuse_permitted.any()


def test_fixture_equal_scores_average_losses_not_probabilities():
    import math
    snapshots, predictions = _inputs()
    snapshots = snapshots.iloc[:2].copy()
    snapshots["raw_a_is_team_a"] = True
    snapshots["odds_a"] = [10 / 9, 10]
    snapshots["odds_b"] = [10, 10 / 9]
    snapshots["export_clock_utc"] = "2026-10-22T00:00:00Z"
    other = snapshots.iloc[0].copy()
    other.update(pd.Series({"snapshot_id": "q4", "canonical_match_id": "f2", "source_match_id": "s2",
                            "mapping_source_match_id": "s2", "outcome": 0, "odds_a": 1.25, "odds_b": 5}))
    for field in ("observed_at", "earliest_start_utc", "outcome_known_at"):
        other[field] = str(other[field]).replace("2026-01-01", "2026-02-01")
    snapshots = pd.concat([snapshots, other.to_frame().T], ignore_index=True)
    predictions.loc[0, ["p_team1", "p_team2"]] = [.6, .4]
    second = predictions.iloc[0].copy()
    second.update(pd.Series({"canonical_match_id": "f2", "model_source_match_id": "s2",
                             "model_clock_utc": "2026-02-01T00:00:00Z", "p_team1": .4, "p_team2": .6}))
    predictions = pd.concat([predictions, second.to_frame().T], ignore_index=True)
    panel = evaluate_production_probabilities(snapshots, predictions)["fixture_panel"]
    panel = panel.loc[panel.endpoint.eq("ALL")]
    assert panel.ll_model.mean() == pytest.approx(-math.log(.6))
    assert panel.ll_book.mean() == pytest.approx(((-math.log(.9)-math.log(.1))/2-math.log(.2))/2)
    assert panel.brier_model.mean() == pytest.approx(.16)
    assert panel.brier_book.mean() == pytest.approx(.525)


def test_poststart_quote_does_not_hide_last_retained_prestart_quote():
    snapshots, predictions = _inputs()
    poststart = snapshots.iloc[1].copy()
    poststart["snapshot_id"], poststart["observed_at"] = "late", "2026-01-01T05:00:00Z"
    snapshots = pd.concat([snapshots, poststart.to_frame().T], ignore_index=True)
    result = evaluate_production_probabilities(snapshots, predictions)
    last = result["fixture_panel"].query("endpoint == 'last_retained'")
    assert last.n_observations.item() == 1
    assert last.p_book_calibration.item() == pytest.approx(2 / 3)


@pytest.mark.parametrize("probability,expected", [(1.0, 0.0), (0.0, float("inf"))])
def test_unclipped_probability_endpoints_have_exact_loss(probability, expected):
    snapshots, predictions = _inputs()
    predictions.loc[0, ["p_team1", "p_team2"]] = [probability, 1-probability]
    panel = evaluate_production_probabilities(snapshots.iloc[:1], predictions)["fixture_panel"]
    assert panel.query("endpoint == 'ALL'").ll_model.item() == expected


def test_missing_quote_clock_remains_accounted_without_asof_failure():
    snapshots, predictions = _inputs()
    snapshots.loc[0, "observed_at"] = None
    audit = evaluate_production_probabilities(snapshots, predictions)["accounting"]
    assert audit.loc[audit.snapshot_id.eq("q1"), "pair_exclusion_reason"].item() == "missing_or_naive_clock"
    assert set(audit.snapshot_id) == {"q1", "q2", "q3"}


def test_actual_generation_clock_blocks_backdated_forecast_before_it_existed():
    snapshots, predictions = _inputs()
    predictions["actual_generated_at"] = "2026-01-01T00:30:00Z"
    audit = evaluate_production_probabilities(snapshots.iloc[:2], predictions)["accounting"].set_index("snapshot_id")
    assert audit.loc["q1", "pair_exclusion_reason"] == "no_prior_forecast"
    assert bool(audit.loc["q2", "scorable"])


def test_superseded_forecast_is_still_valid_before_replacement():
    snapshots, predictions = _inputs()
    predictions["prediction_status"] = "superseded"
    assert evaluate_production_probabilities(snapshots.iloc[:1], predictions)["accounting"].scorable.item()


def test_invalid_last_source_state_is_withheld_instead_of_reviving_earlier_quote():
    snapshots, predictions = _inputs()
    snapshots.loc[1, "snapshot_status"] = "withdrawn"
    result = evaluate_production_probabilities(snapshots, predictions)
    assert result["fixture_panel"].query("endpoint == 'last_retained'").empty
    last = result["accounting"].loc[lambda row: row.snapshot_id.eq("q2")]
    assert bool(last._last.item())
    assert last.pair_exclusion_reason.item() == "snapshot_not_active"


def test_daily_counterfactual_does_not_carry_to_next_utc_day():
    snapshots, predictions = _inputs()
    snapshots = snapshots.iloc[:1].copy()
    snapshots.loc[0, "observed_at"] = "2026-01-02T00:00:00Z"
    snapshots.loc[0, "earliest_start_utc"] = "2026-01-02T04:00:00Z"
    snapshots.loc[0, "outcome_known_at"] = "2026-01-02T06:00:00Z"
    snapshots.loc[0, "export_clock_utc"] = "2026-01-03T00:00:00Z"
    predictions["model_clock_scope"] = "same_UTC_day_counterfactual"
    audit = evaluate_production_probabilities(snapshots, predictions)["accounting"]
    assert audit.pair_exclusion_reason.item() == "no_same_day_replay_state"


def test_explicit_naive_actual_generation_clock_is_not_assumed_utc():
    snapshots, predictions = _inputs()
    predictions["actual_generated_at"] = "2026-01-01T00:00:00"
    audit = evaluate_production_probabilities(snapshots.iloc[:1], predictions)["accounting"]
    assert audit.pair_exclusion_reason.item() == "invalid_actual_generation_clock"


@pytest.mark.parametrize("conflict", [False, True])
def test_same_clock_copies_are_equivalent_or_explicitly_withheld(conflict):
    from src.analysis.production_probability_inputs import resolve_prediction_states
    _, predictions = _inputs()
    predictions["prediction_id"] = "1"
    second = predictions.iloc[0].copy()
    second["prediction_id"], second["prediction_status"] = "2", "superseded"
    if conflict:
        second["p_team1"], second["p_team2"] = .9, .1
    source = pd.concat([predictions, second.to_frame().T], ignore_index=True)
    states, accounting = resolve_prediction_states(source)
    assert len(states) == 1
    assert set(accounting.prediction_id) == {"1", "2"}
    if conflict:
        assert states.prediction_status.item() == "conflicting_same_clock_state_withheld"
        assert states.p_team1.isna().item()
    else:
        assert states.p_team1.item() == .8
        assert states.prediction_status.item() in {"valid", "superseded"}


def test_entirely_missing_cutoff_column_is_withheld_not_timezone_comparison_error():
    snapshots, predictions = _inputs()
    predictions["model_data_cutoff_utc"] = None
    audit = evaluate_production_probabilities(snapshots.iloc[:1], predictions)["accounting"]
    assert audit.pair_exclusion_reason.item() == "invalid_or_future_data_cutoff"


def test_conflicting_result_sources_are_withheld_without_rewriting_the_winner():
    from src.analysis.production_probability_inputs import apply_outcome_adjudication
    snapshots, predictions = _inputs()
    checks = pd.DataFrame([{"canonical_match_id": "f1", "canonical_outcome": 1,
                            "map_id_outcome": 0, "source_team_id_pair_matches": True}])
    adjudicated = apply_outcome_adjudication(snapshots, checks)
    assert adjudicated.loc[adjudicated.canonical_match_id.eq("f1"), "outcome"].tolist() == [1, 1]
    audit = evaluate_production_probabilities(adjudicated.iloc[:2], predictions)["accounting"]
    assert audit.pair_exclusion_reason.tolist() == ["conflicting_result_sources"] * 2
    assert not audit.scorable.any()


@pytest.mark.parametrize("date,scorable", [("2025-12-31", True), ("2026-01-01", False), ("NaN", False)])
def test_declared_date_scenario_requires_the_whole_cutoff_day_to_have_ended(date, scorable):
    from src.analysis.production_probability_inputs import declared_date_cutoff_scenario
    snapshots, predictions = _inputs()
    predictions["model_data_cutoff_utc"] = date
    scenario = declared_date_cutoff_scenario(predictions)
    assert predictions.model_data_cutoff_utc.item() == date
    audit = evaluate_production_probabilities(snapshots.iloc[:1], scenario)["accounting"]
    assert bool(audit.scorable.item()) is scorable
    if not scorable:
        assert audit.pair_exclusion_reason.item() == "invalid_or_future_data_cutoff"


def test_fractional_source_start_cannot_disappear_behind_later_whole_second_start():
    from src.analysis.production_probability_inputs import _earliest_start
    canonical = pd.DataFrame([{"canonical_match_id": 1, "start_time_normalized": "2026-01-01T04:00:00+00:00"}])
    events = pd.DataFrame([{"canonical_match_id": 1, "match_start_time": "2026-01-01T03:00:00.123456+00:00"}])
    result = _earliest_start(canonical, events, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert result.earliest_start_utc.item() == pd.Timestamp("2026-01-01T03:00:00.123456+00:00")


def test_frozen_probability_mode_rejects_ignored_bootstrap_override(monkeypatch):
    import sys
    from scripts.run_model_benchmark import parse_args
    monkeypatch.setattr(sys, "argv", ["run_model_benchmark", "--production-probability", "prepared",
                                     "--output-dir", "new", "--bootstraps", "10000"])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
