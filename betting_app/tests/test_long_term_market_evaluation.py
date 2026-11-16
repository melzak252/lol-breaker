"""Consumer-visible boundaries for source-separated historical price diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.long_term_market_evaluation import (
    build_timed_panel,
    canonical_start_bounds,
    endpoint_loo_consensus,
    select_fixed_policy,
    summarize_unit_returns,
    timed_loo_consensus,
)


def test_endpoint_consensus_excludes_focal_book_and_requires_two_distinct_peers():
    quotes = pd.DataFrame([
        {"source": "oddsportal", "source_event_id": "e1", "horizon": "OPEN_OR_FIRST_OBSERVED",
         "bookmaker": book, "odds_a": a, "odds_b": b}
        for book, a, b in (("focal", 2.0, 2.0), ("peer1", 2.5, 5 / 3), ("peer2", 4.0, 4 / 3))
    ])
    result = endpoint_loo_consensus(quotes)
    assert result.loc[result.bookmaker.eq("focal"), "q"].item() == pytest.approx(0.325)
    quotes.loc[quotes.bookmaker.eq("focal"), "odds_a"] = 10.0
    changed = endpoint_loo_consensus(quotes)
    assert changed.loc[changed.bookmaker.eq("focal"), "q"].item() == pytest.approx(0.325)
    assert endpoint_loo_consensus(quotes.iloc[:2]).q.isna().all()
    duplicate = pd.concat([quotes.iloc[:2], quotes.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate.*book"):
        endpoint_loo_consensus(duplicate)


def _policy_panel():
    return pd.DataFrame([{
        "source": "oddsportal", "source_event_id": "e1", "golgg_match_id": "101",
        "horizon": "OPEN_OR_FIRST_OBSERVED", "bookmaker": "focal", "p_c0": 0.9,
        "q": 0.1, "odds_a": 2.2, "odds_b": 1.9, "y": 1.0,
        "canonical_date": "2021-01-12", "label_status": "known",
    }])


def test_logit_shrinkage_dro_uses_max_hurdle_not_added_ev_floor():
    # logit(.9) and logit(.1) cancel. Mean EV=.10, DRO EV=.045>0:
    # the existing rule accepts, while incorrectly requiring DRO EV>=.05 rejects.
    selected = select_fixed_policy(_policy_panel(), sports_alpha=0.5, tax_rate=0.0, epsilon=0.025)
    assert selected.side.tolist() == ["a"]
    assert selected.p_selected.item() == pytest.approx(0.5)
    assert selected.ev_net.item() == pytest.approx(0.1)
    assert selected.ev_dro.item() == pytest.approx(0.045)
    assert selected.odds.item() == 2.2
    summary = summarize_unit_returns(selected, ["2021-01", "2021-02", "2021-03"], draws=5000, seed=20261022)
    assert summary["profit"] == pytest.approx(120.0)
    assert summary["roi"] == pytest.approx(1.2)
    assert summary["return_interval"]["blocks"] == 3
    assert summary["return_interval"]["empty_replicates"] > 0
    assert summary["return_interval"]["ci_lower"] is None


def test_selection_does_not_consult_winners_or_hide_unknown_selected_labels():
    panel = _policy_panel()
    second = panel.iloc[0].copy()
    second["bookmaker"] = "higher_price"
    second["odds_a"] = 2.4
    panel = pd.concat([panel, second.to_frame().T], ignore_index=True)
    selected = select_fixed_policy(panel, sports_alpha=1.0, tax_rate=0.12, epsilon=0.025)
    assert selected.bookmaker.tolist() == ["higher_price"]
    panel["y"] = np.nan
    panel["label_status"] = "unknown"
    unknown = select_fixed_policy(panel, sports_alpha=1.0, tax_rate=0.12, epsilon=0.025)
    assert unknown.bookmaker.tolist() == selected.bookmaker.tolist()
    summary = summarize_unit_returns(unknown, ["2021-01", "2021-02"], draws=5000, seed=20261022)
    assert summary["bets"] == 1
    assert summary["unknown_labels"] == 1
    assert summary["roi"] is None
    assert summary["profit"] is None
    assert summary["known_label_profit"] == 0.0


def test_timed_consensus_never_borrows_a_future_or_expired_price():
    rows = []
    for book, clock, expiry, a, b in (
        ("focal", "2021-01-12T10:00:00Z", "2021-01-12T10:05:00Z", 2.0, 2.0),
        ("peer1", "2021-01-12T09:59:30Z", "2021-01-12T10:05:00Z", 2.0, 2.0),
        ("future", "2021-01-12T10:00:01Z", "2021-01-12T10:05:00Z", 4.0, 4 / 3),
        ("expired", "2021-01-12T09:59:20Z", "2021-01-12T10:00:00Z", 4.0, 4 / 3),
    ):
        rows.append({"source": "oddspapi", "source_event_id": "e1", "bookmaker": book,
                     "horizon": "OPEN_OR_FIRST_OBSERVED", "odds_a": a, "odds_b": b,
                     "quote_at": clock, "quote_at_a": clock, "quote_at_b": clock,
                     "source_start_at": "2021-01-12T16:00:00Z",
                     "valid_until_exclusive_at": expiry, "active_state_proven": True})
    result = timed_loo_consensus(pd.DataFrame(rows), freshness_seconds=300)
    assert pd.isna(result.loc[result.bookmaker.eq("focal"), "q_timed"].item())
    rows[2]["quote_at"] = rows[2]["quote_at_a"] = rows[2]["quote_at_b"] = "2021-01-12T09:59:40Z"
    result = timed_loo_consensus(pd.DataFrame(rows), freshness_seconds=300)
    assert result.loc[result.bookmaker.eq("focal"), "q_timed"].item() == pytest.approx(0.375)


def test_later_inactive_peer_state_never_revives_its_older_active_quote():
    rows = []
    for book, clock, active, a, b in (
        ("focal", "2021-01-12T10:00:00Z", True, 2.0, 2.0),
        ("peer1", "2021-01-12T09:59:30Z", True, 2.0, 2.0),
        ("peer2", "2021-01-12T09:58:00Z", True, 4.0, 4 / 3),
        ("peer2", "2021-01-12T09:59:40Z", False, 4.0, 4 / 3),
    ):
        rows.append({"source": "oddspapi", "source_event_id": "e1", "bookmaker": book,
                     "horizon": "OPEN_OR_FIRST_OBSERVED", "odds_a": a, "odds_b": b,
                     "quote_at": clock, "quote_at_a": clock, "quote_at_b": clock,
                     "source_start_at": "2021-01-12T16:00:00Z",
                     "valid_until_exclusive_at": None, "active_state_proven": active})
    result = timed_loo_consensus(pd.DataFrame(rows), freshness_seconds=300)
    assert pd.isna(result.loc[result.bookmaker.eq("focal"), "q_timed"].item())


def test_raw_probability_scores_do_not_clip_rare_large_losses():
    from src.analysis.long_term_market_evaluation import _scores

    result = _scores([1], [1e-20])
    assert result["log_loss"] == pytest.approx(-np.log(1e-20))
    with pytest.raises(ValueError, match="interior"):
        _scores([1], [0.0])


def test_zero_weight_inputs_are_not_required_or_imputed():
    pure_sports = _policy_panel()
    pure_sports["q"] = np.nan
    selected = select_fixed_policy(pure_sports, sports_alpha=1.0, tax_rate=0.12, epsilon=0.025)
    assert selected.side.tolist() == ["a"]
    assert selected.q.isna().all()
    pure_market = _policy_panel()
    pure_market["p_c0"] = np.nan
    pure_market["q"] = 0.8
    selected = select_fixed_policy(pure_market, sports_alpha=0.0, tax_rate=0.12, epsilon=0.025)
    assert selected.side.tolist() == ["a"]
    assert selected.p_c0.isna().all()


def test_native_probabilities_are_oriented_by_exact_team_ids_not_match_id_alone():
    from src.analysis.long_term_market_evaluation import build_archival_panel

    events = pd.DataFrame([{
        "source": "oddsportal", "source_event_id": "e1", "golgg_match_id": "101",
        "source_start_at": "2021-01-12T16:00:00Z", "canonical_date": "2021-01-12",
        "canonical_team_a_id": "1", "canonical_team_b_id": "2", "canonical_y": 1,
        "y": 1, "label_status": "known",
    }])
    quotes = pd.DataFrame([{
        "source": "oddsportal", "source_event_id": "e1", "horizon": "OPEN_OR_FIRST_OBSERVED",
        "bookmaker": "focal", "odds_a": 2.0, "odds_b": 2.0,
        "quote_at_a": "2021-01-12T10:00:00Z", "quote_at_b": "2021-01-12T10:00:00Z",
    }])
    native = pd.DataFrame([{"golgg_match_id": "101", "team1_id": "2", "team2_id": "1",
                            "p_full_full": 0.2, "y": 0}])
    result = build_archival_panel(events, quotes, native)
    assert result.p_c0.item() == pytest.approx(0.8)
    assert result.native_a_is_canonical_a.item() == False
    native["team2_id"] = "3"
    with pytest.raises(ValueError, match="team pair"):
        build_archival_panel(events, quotes, native)
    native["team2_id"] = "1"
    native["y"] = 1
    with pytest.raises(ValueError, match="label"):
        build_archival_panel(events, quotes, native)


def test_timed_policy_selects_first_eligible_not_future_maximum_edge():
    first = _policy_panel().iloc[0].to_dict()
    first.update(bookmaker="first", quote_at="2021-01-12T10:00:00Z")
    later = {**first, "bookmaker": "later", "odds_a": 3.0, "quote_at": "2021-01-12T10:00:00.001Z"}
    chosen = select_fixed_policy(pd.DataFrame([later, first]), sports_alpha=1.0,
                                 tax_rate=0.12, epsilon=0.025, timed=True)
    assert chosen.bookmaker.tolist() == ["first"]


@pytest.mark.parametrize("a_clock,b_clock,eligible", [
    ("09:59:00", "09:59:59", True),
    ("09:59:00", "10:00:00", False),
    ("10:00:00", "09:59:59", False),
    ("09:54:58", "09:59:59", False),
    ("09:54:59", "09:59:59", True),
])
def test_timed_panel_requires_both_fresh_sides_before_earliest_known_start(a_clock, b_clock, eligible):
    def at(clock):
        return f"2021-01-12T{clock}Z"
    actual = max(at(a_clock), at(b_clock))
    events = pd.DataFrame([{
        "source": "oddspapi", "source_event_id": "e1", "source_year": 2021,
        "golgg_match_id": "101", "canonical_team_a_id": "a", "canonical_team_b_id": "b",
        "canonical_date": "2021-01-12", "competition_context": "league",
        "source_start_at": at("10:03:00"), "true_start_at": at("10:00:00"),
        "scheduled_start_at": at("10:02:00"), "y": 1.0, "label_status": "known",
    }])
    quotes = pd.DataFrame([{
        "source": "oddspapi", "source_event_id": "e1", "horizon": "FIRST",
        "bookmaker": "book", "odds_a": 2.0, "odds_b": 2.0,
        "quote_at_a": at(a_clock), "quote_at_b": at(b_clock),
        "quote_status": "available_source_scenario", "active_state_proven": True,
    }])
    queries = pd.DataFrame([{
        "quote_index": 0, "forecast_key": "state", "forecast_absence": None,
        "decision_at": actual, "source_start_at": at("10:00:00"),
    }])
    forecasts = pd.DataFrame([{
        "forecast_key": "state", "p_c0": .8, "source_team_a_id": "a",
        "source_team_b_id": "b", "decision_at": at("09:50:00"), "model_year": 2021,
    }])
    releases = pd.DataFrame([{
        "golgg_match_id": "101", "target_native_effective_release_day": "2021-01-13",
    }])
    starts = canonical_start_bounds({"oddspapi": (events, quotes)})
    panel = build_timed_panel(events, quotes, queries, forecasts, releases, starts)
    assert panel.quote_at.item() == pd.Timestamp(actual)
    assert bool(panel.timed_price_eligible.item()) is eligible
    assert panel.source_start_at.item() == pd.Timestamp(at("10:00:00"))
    queries["decision_at"] = at("09:00:00")
    with pytest.raises(ValueError, match="actual quote clock"):
        build_timed_panel(events, quotes, queries, forecasts, releases, starts)


def _timed_market_only_inputs():
    clock = "2021-01-12T10:00:00Z"
    events = pd.DataFrame([{
        "source": "oddspapi", "source_event_id": "e1", "source_year": 2021,
        "golgg_match_id": "101", "canonical_team_a_id": "a", "canonical_team_b_id": "b",
        "canonical_date": "2021-01-12", "competition_context": "league",
        "source_start_at": "2021-01-12T11:00:00Z", "y": 1.0, "label_status": "known",
    }])
    quotes = pd.DataFrame([{
        "source": "oddspapi", "source_event_id": "e1", "horizon": "FIRST",
        "bookmaker": book, "odds_a": a, "odds_b": b, "quote_at_a": clock, "quote_at_b": clock,
        "quote_status": "available_source_scenario", "active_state_proven": True,
    } for book, a, b in [("focal", 2.2, 2.2), ("peer1", 1.1, 9.9), ("peer2", 1.1, 9.9)]])
    queries = pd.DataFrame(columns=[
        "quote_index", "forecast_key", "forecast_absence", "decision_at", "source_start_at",
    ])
    forecasts = pd.DataFrame(columns=[
        "forecast_key", "p_c0", "source_team_a_id", "source_team_b_id", "decision_at", "model_year",
    ])
    releases = pd.DataFrame([{
        "golgg_match_id": "101", "target_native_effective_release_day": "2021-01-13",
    }])
    return events, quotes, queries, forecasts, releases


def test_unemitted_native_queries_do_not_drop_prices_or_invent_sports_probabilities():
    events, quotes, queries, forecasts, releases = _timed_market_only_inputs()
    starts = canonical_start_bounds({"oddspapi": (events, quotes)})
    panel = build_timed_panel(events, quotes, queries, forecasts, releases, starts, years=[2021])
    assert len(panel) == len(quotes)
    assert not panel.native_request_emitted.any()
    assert panel.p_c0.isna().all()
    assert panel.timed_price_eligible.all()
    selected = select_fixed_policy(panel, sports_alpha=0, tax_rate=0, epsilon=.025, timed=True)
    assert selected.bookmaker.tolist() == ["focal"]
    assert selected.p_selected.item() == pytest.approx(.9)
    assert select_fixed_policy(panel, sports_alpha=1, tax_rate=0, epsilon=.025, timed=True).empty


def test_global_earliest_start_blocks_same_fixture_without_borrowing_another_target_start():
    events, quotes, queries, forecasts, releases = _timed_market_only_inputs()
    earlier = events.assign(source="oddsportal", source_event_id="other", golgg_match_id="999",
                            source_start_at="2021-01-12T09:59:00Z")
    tables = {"oddspapi": (events, quotes),
              "oddsportal": (earlier, pd.DataFrame(columns=["source_event_id"]))}
    starts = canonical_start_bounds(tables)
    panel = build_timed_panel(events, quotes, queries, forecasts, releases, starts, years=[2021])
    assert panel.timed_price_eligible.all()  # Earlier game, same teams: not this target.
    earlier["golgg_match_id"] = "101"
    starts = canonical_start_bounds(tables)
    panel = build_timed_panel(events, quotes, queries, forecasts, releases, starts, years=[2021])
    assert not panel.timed_price_eligible.any()
    assert panel.source_start_at.eq(pd.Timestamp("2021-01-12T09:59:00Z")).all()
    assert select_fixed_policy(panel, sports_alpha=0, tax_rate=0, epsilon=.025, timed=True).empty
