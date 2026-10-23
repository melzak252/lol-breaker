import pandas as pd
import pytest

from src.models.production_probability_replay import (
    _base_result,
    _canonical_probability,
    _validate_opportunities,
)


def _opportunities():
    return pd.DataFrame([
        {"opportunity_id": "late", "canonical_match_id": "f1", "golgg_match_id": "source1",
         "decision_at": "2026-10-23T10:00:00Z", "start_at": "2026-10-23T14:00:00Z",
         "canonical_a_is_team1": True, "team1_id": "1", "team2_id": "2", "best_of": 3,
         "identity_status": "verified"},
        {"opportunity_id": "early", "canonical_match_id": "f1", "golgg_match_id": "source1",
         "decision_at": "2026-10-23T09:00:00Z", "start_at": "2026-10-23T14:00:00Z",
         "canonical_a_is_team1": True, "team1_id": "1", "team2_id": "2", "best_of": 3,
         "identity_status": "verified"},
        {"opportunity_id": "next-day", "canonical_match_id": "f1", "golgg_match_id": "source1",
         "decision_at": "2026-10-24T09:00:00Z", "start_at": "2026-10-24T14:00:00Z",
         "canonical_a_is_team1": True, "team1_id": "1", "team2_id": "2", "best_of": 3,
         "identity_status": "verified"},
    ])


def test_fixture_day_requests_deduplicate_to_earliest_clock_and_keep_fanout():
    result = _validate_opportunities(_opportunities())
    assert result.opportunity_id.tolist() == ["early", "next-day"]
    assert result.source_join_fanout.tolist() == [2, 1]
    assert result.cache_reuse_permitted.tolist() == [False, False]
    assert result.decision_at.iloc[0] == pd.Timestamp("2026-10-23T09:00:00Z")


def test_fixture_day_dedup_rejects_conflicting_series_identity():
    rows = _opportunities()
    rows.loc[0, "team2_id"] = "different"
    with pytest.raises(ValueError, match="conflicting team2_id"):
        _validate_opportunities(rows)


def test_canonical_orientation_complements_team_a_probability():
    assert _canonical_probability(0.73, True) == pytest.approx(0.73)
    assert _canonical_probability(0.73, False) == pytest.approx(0.27)


def test_unavailable_model_output_is_not_filled_with_a_probability():
    row = _validate_opportunities(_opportunities()).iloc[0].to_dict()
    output = _base_result(row)
    assert output["forecast_available"] is False
    assert pd.isna(output["p_team1"])
    assert output["issuance_authenticated"] is False
    assert output["forecast_clock_basis"] == "simulated_historical_origin_not_issuance"
    assert output["model_binding_status"] == "conditional_historical_replay"
    assert output["model_binding_basis"] == "canonical_link_only"


def test_invalid_best_of_is_retained_as_an_absence_instead_of_aborting_the_batch():
    rows = _opportunities()
    rows.loc[0, "best_of"] = 7
    result = _validate_opportunities(rows)
    invalid = result.set_index("opportunity_id").loc["late"]
    assert invalid["_replay_absence"] == "best_of_unsupported"


def test_unparseable_orientation_is_retained_as_an_absence():
    rows = _opportunities()
    rows["canonical_a_is_team1"] = rows["canonical_a_is_team1"].astype(object)
    rows.loc[0, "canonical_a_is_team1"] = "not-a-boolean"
    result = _validate_opportunities(rows)
    invalid = result.set_index("opportunity_id").loc["late"]
    assert invalid["_replay_absence"] == "canonical_side_orientation_invalid"


def test_unverified_source_identity_is_retained_as_an_absence():
    rows = _opportunities()
    rows.loc[0, "identity_status"] = "source_mapping_conflict"
    result = _validate_opportunities(rows)
    invalid = result.set_index("opportunity_id").loc["late"]
    assert invalid["_replay_absence"] == "source_identity_source_mapping_conflict"


def test_poststart_request_stays_an_explicit_absence_instead_of_aborting_other_requests():
    rows = _opportunities()
    rows.loc[0, "decision_at"] = "2026-10-23T15:00:00Z"
    result = _validate_opportunities(rows).set_index("opportunity_id")
    assert result.loc["late", "_replay_absence"] == "not_prestart"
    assert pd.isna(result.loc["early", "_replay_absence"])
