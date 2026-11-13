import json

import numpy as np
import pandas as pd
import pytest

from scripts import reproduce_project_results as replay
from scripts.reproduce_project_results import select_first_policy


def test_tax_requalification_uses_ssot_guards_and_changes_selections():
    panel = pd.DataFrame([
        {
            "source": "oddsportal",
            "horizon": "OPEN_OR_FIRST_OBSERVED",
            "golgg_match_id": "fixture-1",
            "source_event_id": "event-1",
            "bookmaker": "book-a",
            "quote_at": "2026-01-01T10:00:00Z",
            "p_c0": 0.5,
            "q_focal": 0.55,
            "odds_a": 2.2,
            "odds_b": 1.8,
            "archive_eligible": True,
            "timed_price_eligible": True,
            "canonical_date": "2026-01-02",
            "y": 1.0,
            "label_status": "known",
        },
        {
            "source": "oddsportal",
            "horizon": "OPEN_OR_FIRST_OBSERVED",
            "golgg_match_id": "fixture-1",
            "source_event_id": "event-1",
            "bookmaker": "book-b",
            "quote_at": "2026-01-01T10:01:00Z",
            "p_c0": 0.75,
            "q_focal": 0.55,
            "odds_a": 3.0,
            "odds_b": 1.5,
            "archive_eligible": True,
            "timed_price_eligible": True,
            "canonical_date": "2026-01-02",
            "y": 1.0,
            "label_status": "known",
        },
    ])
    untimed = panel.iloc[0].copy()
    untimed["golgg_match_id"] = "fixture-2"
    untimed["source_event_id"] = "event-2"
    untimed["timed_price_eligible"] = False
    panel = pd.concat([panel, untimed.to_frame().T], ignore_index=True)

    selected_zero = select_first_policy(panel, alpha=0.5, tax=0.0)
    selected_taxed = select_first_policy(panel, alpha=0.5, tax=0.12)

    assert selected_zero.bookmaker.tolist() == ["book-a"]
    assert selected_taxed.bookmaker.tolist() == ["book-b"]
    assert selected_zero.golgg_match_id.tolist() == ["fixture-1"]
    assert selected_taxed.golgg_match_id.tolist() == ["fixture-1"]
    assert selected_taxed.tax_rate.tolist() == [0.12]


def consensus_inputs():
    rows = []
    for fixture, book, probabilities in [
        ("one", "focal-a", [0.9, 0.7, 0.8]),
        ("one", "focal-b", [0.4]),
        ("two", "focal-a", [0.6]),
    ]:
        for p in probabilities:
            rows.append({
                "canonical_match_id": fixture, "bookmaker": book,
                "snapshot_id": str(len(rows)), "p_model": p, "p_market": 0.5,
                "y": float(fixture == "one"),
                "clock": "2026-05-01T10:00:00Z", "lead_bin": "[0,2)",
                "month": "2026-05" if fixture == "one" else "2026-06",
                "n_contributors": 2,
                "contributors": json.dumps(["peer-a", "peer-b"]),
                "contributor_clocks": json.dumps(["2026-05-01T09:59:00Z"] * 2),
                "ages_seconds": json.dumps([60, 60]),
                "contributor_weights": json.dumps([0.5, 0.5]),
                "contributor_fair_probabilities": json.dumps([0.5, 0.5]),
            })
    panel = pd.DataFrame(rows)
    cases = panel[["canonical_match_id", "bookmaker", "snapshot_id"]].rename(
        columns={"bookmaker": "focal_bookmaker"})
    cases["status"], cases["absence_reason"] = "matched", ""
    return panel, cases


def test_consensus_scores_losses_with_equal_book_then_fixture_weight():
    panel, cases = consensus_inputs()
    result = replay.summarize_consensus_panel(panel, cases)
    book_one = -np.log([0.9, 0.7, 0.8]).mean()
    expected = ((book_one - np.log(0.4)) / 2 - np.log(0.4)) / 2
    assert result["scores"]["C20R"]["log_loss"] == pytest.approx(expected)
    assert result["scores"]["C20R"]["log_loss"] != pytest.approx(
        -np.log([0.9, 0.7, 0.8, 0.4, 0.4]).mean())
    expected_brier = ((np.mean(np.square([0.1, 0.3, 0.2])) + 0.6**2) / 2 + 0.6**2) / 2
    assert result["scores"]["C20R"]["brier"] == pytest.approx(expected_brier)
    repeated = panel.loc[panel.canonical_match_id.eq("one") & panel.bookmaker.eq("focal-a")].copy()
    repeated["snapshot_id"] += "-repeat"
    enlarged = pd.concat([panel, repeated], ignore_index=True)
    repeated_cases = cases.loc[cases.snapshot_id.isin(panel.loc[repeated.index, "snapshot_id"])].copy()
    repeated_cases["snapshot_id"] += "-repeat"
    more = replay.summarize_consensus_panel(enlarged, pd.concat([cases, repeated_cases], ignore_index=True))
    assert more["scores"] == result["scores"]
    assert more["coverage"]["matched_cases"] == len(enlarged)
    complemented = panel.copy()
    complemented["y"] = 1 - complemented.y
    complemented["p_model"] = 1 - complemented.p_model
    same = replay.summarize_consensus_panel(complemented, cases)
    for recipe in result["scores"]:
        for metric in ("log_loss", "brier"):
            assert same["scores"][recipe][metric] == pytest.approx(result["scores"][recipe][metric])


@pytest.mark.parametrize("corruption", [
    "duplicate", "missing_case", "extra_case", "label", "month", "probability", "empty",
    "malformed_json", "missing_proof", "length", "focal", "future", "stale", "naive",
    "weight", "market", "nested_numeric", "object_numeric",
])
def test_consensus_rejects_corrupted_or_unpaired_provenance(corruption):
    panel, cases = consensus_inputs()
    if corruption == "duplicate":
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    elif corruption == "missing_case":
        cases = cases.iloc[1:].copy()
    elif corruption == "extra_case":
        extra = cases.iloc[[0]].assign(snapshot_id="not-in-panel")
        cases = pd.concat([cases, extra], ignore_index=True)
    elif corruption == "label":
        panel.loc[0, "y"] = 0
    elif corruption == "month":
        panel.loc[0, "month"] = "2026-06"
    elif corruption == "probability":
        panel.loc[0, "p_model"] = np.nan
    elif corruption == "empty":
        panel = panel.iloc[:0]
    elif corruption == "malformed_json":
        panel.loc[0, "contributors"] = "[broken"
    elif corruption == "missing_proof":
        panel = panel.drop(columns="ages_seconds")
    elif corruption == "length":
        panel.loc[0, "ages_seconds"] = "[60]"
    elif corruption == "focal":
        panel.loc[0, "contributors"] = '["focal-a", "peer-b"]'
    elif corruption == "future":
        panel.loc[0, "contributor_clocks"] = '["2026-05-01T10:01:00Z", "2026-05-01T09:59:00Z"]'
    elif corruption == "stale":
        panel.loc[0, "ages_seconds"] = "[301, 60]"
    elif corruption == "naive":
        panel.loc[0, "clock"] = "2026-05-01T10:00:00"
    elif corruption == "weight":
        panel.loc[0, "contributor_weights"] = "[0.1, 0.9]"
    elif corruption == "market":
        panel.loc[0, "p_market"] = 0.6
    elif corruption == "nested_numeric":
        panel.loc[0, "ages_seconds"] = "[[60], [60]]"
    elif corruption == "object_numeric":
        panel.loc[0, "contributor_weights"] = '[{}, {}]'
    with pytest.raises(ValueError):
        replay.summarize_consensus_panel(panel, cases)


def glicko_inputs():
    return pd.DataFrame({
        "golgg_match_id": ["a", "b", "c"], "y_true": [1, 0, 1], "BoN": [1, 3, 5],
        "date": ["2024-01-02", "2024-01-03", "2024-02-01"],
        "team1_id": ["one", "one", "three"], "team2_id": ["two", "three", "two"],
        "player_gl_raw": [0.8, 0.3, 0.7], "team_gl_raw": [0.5, 0.4, 0.6],
        "player_gl_calibrated": [0.75, 0.35, 0.65], "team_gl_calibrated": [0.5, 0.45, 0.55],
    })


def test_glicko_scores_identical_saved_rows_without_projection_or_refitting():
    frame = glicko_inputs()
    result = replay.summarize_glicko_pair(frame)
    for kind in ("raw", "calibrated"):
        for entity, name in (("player", "player"), ("team", "organization")):
            p = frame[f"{entity}_gl_{kind}"].to_numpy()
            expected_ll = -np.mean(frame.y_true * np.log(p) + (1 - frame.y_true) * np.log(1 - p))
            score = result["scores"][kind][name]
            assert score["n"] == len(frame)
            assert score["log_loss"] == pytest.approx(expected_ll)
            assert score["brier"] == pytest.approx(np.mean((p - frame.y_true) ** 2))
        expected_delta = (result["scores"][kind]["player"]["log_loss"] -
                          result["scores"][kind]["organization"]["log_loss"])
        assert result["contrasts"][kind]["log_loss"]["delta"] == pytest.approx(expected_delta)
    complemented = frame.copy()
    complemented["y_true"] = 1 - complemented.y_true
    for column in frame.filter(regex="_gl_"):
        complemented[column] = 1 - frame[column]
    same = replay.summarize_glicko_pair(complemented)
    for kind in result["scores"]:
        for entity in result["scores"][kind]:
            for metric in ("log_loss", "brier"):
                assert same["scores"][kind][entity][metric] == pytest.approx(result["scores"][kind][entity][metric])


@pytest.mark.parametrize("corruption", ["duplicate", "missing_prediction", "missing_column", "bo",
                                       "label", "conflicting_label", "date", "same_team"])
def test_glicko_rejects_invalid_rows_without_selecting_separate_cohorts(corruption):
    frame = glicko_inputs()
    if corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif corruption == "conflicting_label":
        frame = pd.concat([frame, frame.iloc[[0]].assign(y_true=0)], ignore_index=True)
    elif corruption == "missing_prediction":
        frame.loc[0, "team_gl_calibrated"] = np.nan
    elif corruption == "missing_column":
        frame = frame.drop(columns="team_gl_raw")
    elif corruption == "bo":
        frame.loc[0, "BoN"] = 2
    elif corruption == "label":
        frame.loc[0, "y_true"] = 2
    elif corruption == "date":
        frame.loc[0, "date"] = None
    elif corruption == "same_team":
        frame.loc[0, "team2_id"] = frame.loc[0, "team1_id"]
    with pytest.raises(ValueError):
        replay.summarize_glicko_pair(frame)


def financial_inputs():
    frame = pd.DataFrame({
        "ev_net": [0.05, 0.10, 0.10001, 0.08, 0.09],
        "side": ["a", "b", "a", "a", "b"],
        "y": [1.0, 1.0, 0.0, 1.0, np.nan],
        "label_status": ["known", "known", "known", "conflicting", "known"],
        "odds": [2.5] * 5, "tax_rate": [0.12] * 5,
        "canonical_date": ["2026-01-01"] * 5, "p_c0": [0.6] * 5,
    })
    frame["p_selected"] = (1 + frame.ev_net) / (frame.odds * (1 - frame.tax_rate))
    return frame


def test_financial_summary_uses_known_label_masks_and_exact_near_band():
    selected = financial_inputs()
    result = replay._financial_summary(selected, ["2026-01", "2026-02"])
    assert (result["bets"], result["known_labels"], result["unknown_labels"]) == (5, 3, 2)
    assert result["profit"] is None and result["roi"] is None
    assert result["known_label_profit"] == pytest.approx(1.2 - 1 - 1)
    assert result["known_label_roi"] == pytest.approx((1.2 - 1 - 1) / 3)
    assert result["mean_expected_ev_known_labels"] == pytest.approx(np.mean([0.05, 0.1, 0.10001]))
    assert result["expected_wins_known_labels"] == pytest.approx(selected.iloc[:3].p_selected.sum())
    assert result["mean_decimal_odds_known_labels"] == 2.5
    assert result["near_bets_known_labels"] == 1
    assert result["near_mean_expected_ev_known_labels"] == 0.1
    assert result["near_realized_yield"] == -1
    assert result["near_return_error_interval"]["estimate"] == pytest.approx(1.1)
    assert result["return_error_interval"]["estimate"] == pytest.approx(
        result["mean_expected_ev_known_labels"] - result["known_label_roi"])
    assert result["near_reliability_pass"] is False
    assert result["return_interval"]["active_blocks"] == 1
    assert result["return_interval"]["blocks"] == 2
    assert result["return_interval"]["empty_replicates"] > 0
    assert result["return_interval"]["ci_lower"] is None


def test_financial_empty_selection_has_null_conditional_metrics():
    result = replay._financial_summary(financial_inputs().iloc[:0], ["2026-01", "2026-02"])
    assert result["bets"] == result["known_labels"] == result["near_bets_known_labels"] == 0
    for key in ("mean_expected_ev_known_labels", "mean_decimal_odds_known_labels",
                "near_mean_expected_ev_known_labels", "near_realized_yield", "known_label_roi"):
        assert result[key] is None
    for key in ("return_interval", "return_error_interval", "near_return_error_interval"):
        assert result[key]["estimate"] is None
        assert result[key]["ci_lower"] is None


def price_inputs(tax=0.0):
    entry = {
        "source": "archive", "source_event_id": "event", "golgg_match_id": "fixture",
        "bookmaker": "book-a", "horizon": replay.FIRST, "side": "a",
        "quote_at": "2026-01-01T10:00:00Z", "quote_at_a": "2026-01-01T10:00:00Z",
        "quote_at_b": "2026-01-01T10:00:00Z", "source_start_at": "2026-01-01T12:00:00Z",
        "canonical_team_a_id": "one", "canonical_team_b_id": "two",
        "odds_a": 2.5, "odds_b": 1.5, "odds": 2.5, "p_selected": 0.6, "p_c0": 0.7,
        "tax_rate": tax, "ev_net": 0.6 * 2.5 * (1 - tax) - 1,
        "y": 1.0, "label_status": "known", "canonical_date": "2026-01-01",
        "timed_price_eligible": True, "timed_price_absence": "",
    }
    later = {**entry, "horizon": "LAST_CHANGE_OR_LAST_PRESTART",
             "quote_at": "2026-01-01T11:00:00Z", "quote_at_a": "2026-01-01T11:00:00Z",
             "quote_at_b": "2026-01-01T11:00:00Z", "odds_a": 2.0, "odds_b": 2.0}
    return pd.DataFrame([entry]), pd.DataFrame([later])


@pytest.mark.parametrize("tax,proxy", [(0.0, 0.25), (0.12, 0.10)])
def test_later_price_pairs_keep_tax_separate_from_raw_movement(tax, proxy):
    selected, endpoints = price_inputs(tax)
    pairs = replay.pair_later_quotes(selected, endpoints)
    assert pairs.loc[0, "raw_price_movement_proxy"] == pytest.approx(0.25)
    assert pairs.loc[0, "net_no_vig_later_price_proxy"] == pytest.approx(proxy)
    result = replay.summarize_later_price_pairs(pairs, ["2026-01", "2026-02"])
    assert result["total_selections"] == result["known_selections"] == result["matched_prices"] == 1
    assert result["matched_known_count"] == 1
    assert result["matched_known"]["net_no_vig_proxy"]["estimate"] == pytest.approx(proxy)
    assert result["matched_known"]["median_later_quote_lead_hours"] == 1


@pytest.mark.parametrize("change,value,reason", [
    ("quote_at", "2026-01-01T10:00:00Z", "same_clock_not_later"),
    ("quote_at", "2026-01-01T09:00:00Z", "earlier_than_entry"),
    ("quote_at", "2026-01-01T12:00:00Z", "at_or_after_earliest_known_start"),
    ("source_start_at", "2026-01-01T10:30:00Z", "at_or_after_earliest_known_start"),
    ("timed_price_eligible", False, "later_quote_fails_timed_price_rules"),
    ("odds_b", 1.0, "later_quote_fails_timed_price_rules"),
    ("quote_at_b", "2026-01-01T10:54:59Z", "later_quote_fails_timed_price_rules"),
    ("canonical_team_a_id", "two", "canonical_team_mismatch"),
    ("bookmaker", "book-b", "no_retained_later_quote"),
    ("golgg_match_id", "other-fixture", "no_retained_later_quote"),
])
def test_later_price_matching_preserves_explicit_absences(change, value, reason):
    selected, endpoints = price_inputs()
    endpoints.loc[0, change] = value
    if change == "quote_at":
        endpoints.loc[0, ["quote_at_a", "quote_at_b"]] = value
    pairs = replay.pair_later_quotes(selected, endpoints)
    assert len(pairs) == 1
    assert pairs.loc[0, "later_quote_pair_status"] == reason
    assert pd.isna(pairs.loc[0, "net_no_vig_later_price_proxy"])
    result = replay.summarize_later_price_pairs(pairs, ["2026-01"])
    assert result["absence_counts"] == {reason: 1}
    assert result["matched_known"]["net_no_vig_proxy"]["estimate"] is None


def test_later_price_matching_uses_requalified_book_side_and_entry():
    zero, endpoints = price_inputs()
    taxed = zero.copy()
    taxed.loc[0, ["bookmaker", "side", "odds", "odds_a", "odds_b", "tax_rate"]] = [
        "book-b", "b", 3.0, 1.5, 3.0, 0.12]
    taxed.loc[0, ["quote_at", "quote_at_a", "quote_at_b"]] = "2026-01-01T10:30:00Z"
    other = endpoints.copy()
    other.loc[0, ["bookmaker", "odds_a", "odds_b"]] = ["book-b", 3.0, 1.5]
    both = pd.concat([endpoints, other], ignore_index=True)
    pair = replay.pair_later_quotes(taxed, both)
    assert pair.loc[0, "selected_side_later_odds"] == 1.5
    assert pair.loc[0, "raw_price_movement_proxy"] == pytest.approx(1.0)
    assert pair.loc[0, "net_no_vig_later_price_proxy"] == pytest.approx(3 * 0.88 * (2 / 3) - 1)
    assert pair.loc[0, "quote_at"] == taxed.loc[0, "quote_at"]


@pytest.mark.parametrize("corruption", ["duplicate", "naive", "malformed"])
def test_later_price_matching_rejects_ambiguous_or_unzoned_endpoints(corruption):
    selected, endpoints = price_inputs()
    if corruption == "duplicate":
        endpoints = pd.concat([endpoints, endpoints], ignore_index=True)
    else:
        endpoints.loc[0, "quote_at"] = "2026-01-01T11:00:00" if corruption == "naive" else "bad"
    with pytest.raises(ValueError):
        replay.pair_later_quotes(selected, endpoints)


def test_later_price_summary_retains_unknowns_and_empty_policy():
    selected, endpoints = price_inputs()
    unknown = selected.assign(source_event_id="unknown", golgg_match_id="unknown", y=np.nan)
    selected = pd.concat([selected, unknown], ignore_index=True)
    endpoints = pd.concat([endpoints, endpoints.assign(source_event_id="unknown", golgg_match_id="unknown")],
                          ignore_index=True)
    result = replay.summarize_later_price_pairs(replay.pair_later_quotes(selected, endpoints), ["2026-01"])
    assert (result["total_selections"], result["known_selections"], result["matched_prices"],
            result["matched_known_count"]) == (2, 1, 2, 1)
    empty = replay.summarize_later_price_pairs(replay.pair_later_quotes(selected.iloc[:0], endpoints), ["2026-01"])
    assert empty["total_selections"] == empty["matched_prices"] == 0
    assert empty["matched_known"]["raw_price_movement"]["estimate"] is None


def test_later_price_matching_preserves_saved_300_second_side_freshness():
    selected, endpoints = price_inputs()
    endpoints.loc[0, "quote_at_b"] = "2026-01-01T10:55:00Z"
    pair = replay.pair_later_quotes(selected, endpoints)
    assert pair.loc[0, "later_quote_pair_status"] == "matched_later_prestart_scenario"
    assert pair.loc[0, "raw_price_movement_proxy"] == pytest.approx(0.25)


def test_snapshot_opportunities_keep_checks_and_repeated_fixture_outcomes_separate():
    checks = pd.DataFrame({
        "canonical_match_id": ["1", "1", "2"],
        "entry_clock": pd.to_datetime(["2026-01-02T09:00:00Z", "2026-01-02T10:00:00Z", "2026-01-02T09:00:00Z"]),
        "fixture_start_utc": pd.to_datetime(["2026-01-02T12:00:00Z"] * 3),
        "band": ["2-6", "2-6", "2-6"],
        "hours": [3., 2., 3.],
    })
    events = pd.DataFrame({
        "canonical_match_id": ["1", "1", "2"], "snapshot_id": ["a", "b", "c"],
        "event_id": ["a:team_a", "b:team_a", "c:team_a"], "bookmaker": ["book"] * 3,
        "decision_at": checks.entry_clock, "start": checks.fixture_start_utc,
        "side": ["team_a"] * 3, "sports_p": [.65, .65, .45], "q": [.5] * 3,
        "odds": [2., 2., 1.6], "best_of": [3] * 3, "y": [0, 0, 1],
    })
    policy = {"tax_rate": 0., "min_ev_net": .05, "wasserstein_epsilon": .025, "max_odds": 4.}
    selected = replay.select_sampled_opportunities(checks, events, policy)
    summary = replay.summarize_sampled_opportunities(checks, selected, ["2026-01"], draws=100)
    assert selected.event_id.tolist() == ["a:team_a", "b:team_a"]
    assert summary["checks"] == 3
    assert summary["suggestions"] == 2
    assert summary["unique_fixtures"] == 1
    assert summary["suggestion_rate"] == pytest.approx(2 / 3)
    assert summary["realized_unit_yield"]["estimate"] == pytest.approx(-1.)
    assert summary["known_outcome_observations"] == 2


def test_snapshot_selection_is_not_conditioned_on_closing_price_availability():
    checks = pd.DataFrame({
        "canonical_match_id": ["1"], "entry_clock": pd.to_datetime(["2026-01-01T10:00:00Z"]),
        "fixture_start_utc": pd.to_datetime(["2026-01-03T12:00:00Z"]), "band": ["48+"], "hours": [50.],
    })
    events = pd.DataFrame({
        "canonical_match_id": ["1"], "snapshot_id": ["a"], "event_id": ["a:team_a"],
        "bookmaker": ["book"], "decision_at": checks.entry_clock, "start": checks.fixture_start_utc,
        "side": ["team_a"], "sports_p": [.65], "q": [.5], "odds": [2.], "best_of": [3], "y": [np.nan],
    })
    selected = replay.select_sampled_opportunities(checks, events, {"tax_rate": 0., "min_ev_net": .05})
    summary = replay.summarize_sampled_opportunities(checks, selected, ["2026-01"], draws=100)
    assert summary["suggestions"] == 1
    assert summary["known_outcome_observations"] == 0
    assert summary["realized_unit_yield"]["estimate"] is None
