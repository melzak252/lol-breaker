import numpy as np
import pandas as pd
import pytest


def latest_close_inputs():
    entries = pd.DataFrame([{"event_id": "entry:team_a", "snapshot_id": "entry", "canonical_match_id": "1",
                             "bookmaker": "a", "side": "team_a", "decision_at": "2026-01-02T10:00:00Z",
                             "start": "2026-01-02T12:00:00Z", "odds": 2.5, "y": np.nan}])
    states = pd.DataFrame([{"canonical_match_id": "1", "bookmaker": book, "snapshot_id": f"close-{book}",
                            "clock": "2026-01-02T11:00:00Z", "fixture_start_utc": "2026-01-02T12:00:00Z",
                            "odds_team_a": 2., "odds_team_b": 2., "state_reason": "eligible"}
                           for book in ["a", "b", "c"]])
    return entries, states


def test_latest_close_measures_prices_without_requiring_outcome_fill_or_stake():
    from src.analysis.production_saved_evidence import pair_latest_recorded_closes, summarize_latest_recorded_closes

    entries, states = latest_close_inputs()
    pairs = pair_latest_recorded_closes(entries, states)
    assert pairs.loc[0, "samebook_price_clv"] == pytest.approx(.25)
    assert pairs.loc[0, "loo_fair_clv"] == pytest.approx(.25)
    summary = summarize_latest_recorded_closes(pairs, ["2026-01"], draws=100)
    assert summary["samebook"]["covered"] == summary["loo"]["covered"] == 1
    assert summary["loo"]["mean"] == pytest.approx(.25)


def test_latest_close_allows_same_entry_state_and_old_peer_references():
    from src.analysis.production_saved_evidence import pair_latest_recorded_closes

    entries, states = latest_close_inputs()
    states.loc[states.bookmaker.eq("a"), ["clock", "odds_team_a"]] = [entries.loc[0, "decision_at"], 2.5]
    states.loc[~states.bookmaker.eq("a"), "clock"] = "2025-12-30T10:00:00Z"
    pairs = pair_latest_recorded_closes(entries, states)
    assert pairs.loc[0, "samebook_price_clv"] == pytest.approx(0)
    assert pairs.loc[0, "loo_fair_clv"] == pytest.approx(.25)


def test_latest_close_uses_selected_side_b_and_excludes_focal_book():
    from src.analysis.production_saved_evidence import pair_latest_recorded_closes

    entries, states = latest_close_inputs()
    entries.loc[0, ["event_id", "side", "odds"]] = ["entry:team_b", "team_b", 1.8]
    states.loc[:, "odds_team_a"] = 3.
    states.loc[:, "odds_team_b"] = 1.5
    states.loc[states.bookmaker.eq("a"), ["odds_team_a", "odds_team_b"]] = [2., 2.]
    pairs = pair_latest_recorded_closes(entries, states)
    assert pairs.loc[0, "samebook_price_clv"] == pytest.approx(-.1)
    assert pairs.loc[0, "loo_fair_clv"] == pytest.approx(.2)


def test_latest_close_retains_invalid_latest_state_as_missing_not_zero():
    from src.analysis.production_saved_evidence import pair_latest_recorded_closes, summarize_latest_recorded_closes

    entries, states = latest_close_inputs()
    states.loc[states.bookmaker.eq("a"), "state_reason"] = "conflicting_latest_state"
    states.loc[states.bookmaker.eq("b"), "clock"] = "2026-01-02T12:30:00Z"
    pairs = pair_latest_recorded_closes(entries, states)
    assert pd.isna(pairs.loc[0, "samebook_price_clv"])
    assert pd.isna(pairs.loc[0, "loo_fair_clv"])
    result = summarize_latest_recorded_closes(pairs, ["2026-01"], draws=100)
    assert result["samebook"]["covered"] == result["loo"]["covered"] == 0
    assert result["loo"]["mean"] is None


def test_latest_close_rejects_ambiguous_book_endpoint_instead_of_selecting_a_price():
    from src.analysis.production_saved_evidence import pair_latest_recorded_closes

    entries, states = latest_close_inputs()
    with pytest.raises(ValueError):
        pair_latest_recorded_closes(entries, pd.concat([states, states.iloc[:1]], ignore_index=True))


def test_latest_close_reports_shortening_direction_and_empty_coverage():
    from src.analysis.production_saved_evidence import pair_latest_recorded_closes, summarize_latest_recorded_closes

    entries, states = latest_close_inputs()
    result = summarize_latest_recorded_closes(pair_latest_recorded_closes(entries, states),
                                              ["2026-01"], draws=100)
    assert result["samebook"]["shortened"] == 1
    assert result["samebook"]["lengthened"] == 0
    empty = summarize_latest_recorded_closes(pair_latest_recorded_closes(entries.iloc[:0], states),
                                             ["2026-01"], draws=100)
    assert empty["entries"] == 0
    assert empty["samebook"]["covered"] == empty["loo"]["covered"] == 0
    assert empty["loo"]["mean"] is None
