"""Outcome-blind source mapping must not erase independently known labels."""
import json

import pandas as pd

from src.analysis.long_term_market_sources import map_source_events


def _raw(tmp_path):
    path = tmp_path / "matches.json"
    path.write_text(json.dumps([{"match_id": 101, "date": "2021-01-12",
        "name_1": "Alpha", "name_2": "Beta", "tid_1": "1", "tid_2": "2",
        "score_1": 2, "score_2": 0, "BoN": 3, "tournament": "Canonical Spring 2021"}]))
    return path


def _event(event_id="e1", **extra):
    return {"source": "oddspapi", "source_event_id": event_id,
            "source_start_at": "2021-01-12T16:00:00Z", "home_name": "Alpha", "away_name": "Beta",
            "competition_context": "provider-short-label", **extra}


def test_missing_provider_score_preserves_known_exact_canonical_outcome(tmp_path):
    result = map_source_events(pd.DataFrame([_event()]), _raw(tmp_path))
    row = result.iloc[0]
    assert row.golgg_match_id == "101"
    assert row.label_status == "known"
    assert row.source_result_status == "missing"
    assert row.y == 1
    assert row.best_of == 3
    assert row.competition_context == "Canonical Spring 2021"


def test_competing_source_claims_annotate_each_row_not_only_the_last_row(tmp_path):
    events = [_event("e1"), _event("e2"), _event("unmatched", home_name="Gamma", away_name="Delta")]
    result = map_source_events(pd.DataFrame(events), _raw(tmp_path))
    assert result.identity_status.tolist() == ["ambiguous_competing_source_events", "ambiguous_competing_source_events", "unmatched"]
    assert result.golgg_match_id.isna().all()


def test_conflicting_provider_result_never_selects_a_different_identity(tmp_path):
    result = map_source_events(pd.DataFrame([_event(homeResult=0, awayResult=2)]), _raw(tmp_path))
    assert result.golgg_match_id.tolist() == ["101"]
    assert result.label_status.tolist() == ["conflict"]
    assert result.source_result_status.tolist() == ["conflict"]
    assert result.y.isna().all()
