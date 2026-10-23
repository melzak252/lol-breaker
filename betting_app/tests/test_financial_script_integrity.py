from scripts.generate_marginal_blocked_bets_audit import (
    build_ledger_prediction_frame_or_withheld,
    match_ledger_prediction,
)
import pandas as pd
from scripts.run_c0_distilled_financial_simulations import (
    leave_one_book_out_consensus,
    probability_within_tolerance,
)
from scripts.run_walk_forward_prod_db_match_level import (
    missing_close_count,
    net_clv_from_close,
    observed_clv_rows,
    quotes_observed_by_decision_time,
)

def test_raw_prediction_frame_is_enriched_by_exact_match_id_and_withheld_on_missing_metadata():
    raw_predictions = pd.DataFrame(
        [{"golgg_match_id": "match-1", "p_c0__full": 0.7, "y": 1}]
    )
    gold = pd.DataFrame(
        [{
            "match_id": "match-1",
            "date": "2026-01-02",
            "team1_name": "Alpha",
            "team2_name": "Beta",
        }]
    )
    monthly = pd.DataFrame([{"golgg_match_id": "match-1", "p_full": 0.6}])
    enriched = build_ledger_prediction_frame_or_withheld(raw_predictions, gold, monthly)
    assert enriched is not None
    assert len(enriched) == 1
    assert enriched.loc[0, "golgg_match_id"] == "match-1"
    assert (enriched.loc[0, "team1_name"], enriched.loc[0, "team2_name"]) == ("Alpha", "Beta")
    assert match_ledger_prediction(
        {"date": "2026-01-02", "team_selected": "Beta", "opponent": "Alpha", "p_c0": 0.3},
        enriched,
    ) == ("match-1", True, False)
    assert build_ledger_prediction_frame_or_withheld(
        raw_predictions, gold.iloc[0:0], monthly
    ) is None


def test_ledger_join_requires_identity_date_and_probability_boundary():
    predictions = pd.DataFrame(
        [
            {"golgg_match_id": "match", "date": "2026-01-02", "team1_name": "A", "team2_name": "B", "p_c0__full": 0.501},
        ]
    )
    assert match_ledger_prediction(
        {"date": "2026-01-01", "team_selected": "A", "opponent": "B", "p_c0": 0.5},
        predictions,
    ) == ("match", False, False)
    assert match_ledger_prediction(
        {"date": "2026-01-01", "team_selected": "Wrong", "opponent": "B", "p_c0": 0.5},
        predictions,
    ) is None
    assert match_ledger_prediction(
        {"date": "2025-12-31", "team_selected": "A", "opponent": "B", "p_c0": 0.5},
        predictions,
    ) is None
    assert match_ledger_prediction(
        {"date": "2026-01-01", "team_selected": "A", "opponent": "B", "p_c0": 0.499},
        predictions,
    ) is None

    ambiguous_predictions = pd.DataFrame(
        [
            {"golgg_match_id": "match_1", "date": "2026-01-02", "team1_name": "A", "team2_name": "B", "p_c0__full": 0.501},
            {"golgg_match_id": "match_2", "date": "2026-01-02", "team1_name": "A", "team2_name": "B", "p_c0__full": 0.501},
        ]
    )
    assert match_ledger_prediction(
        {"date": "2026-01-01", "team_selected": "A", "opponent": "B", "p_c0": 0.5},
        ambiguous_predictions,
    ) == ("", False, True)


def test_leave_one_book_out_requires_two_other_books():
    quotes = pd.DataFrame(
        [
            {"bookmaker_id": "focal", "odds_a": 2.0, "odds_b": 2.0},
            {"bookmaker_id": "other1", "odds_a": 2.0, "odds_b": 2.0},
            {"bookmaker_id": "other2", "odds_a": 2.0, "odds_b": 2.0},
        ]
    )
    assert leave_one_book_out_consensus(quotes, "focal") == 0.5
    assert leave_one_book_out_consensus(quotes.iloc[:2], "focal") is None

def test_missing_close_is_excluded_and_counted():
    bets = pd.DataFrame(
        [
            {"bet_id": "missing", "clv": None},
            {"bet_id": "observed", "clv": 0.025},
        ]
    )

    observed = observed_clv_rows(bets)

    assert observed["bet_id"].tolist() == ["observed"]
    assert missing_close_count(bets) == 1
    assert net_clv_from_close(2.0, None, 0.12) is None


def test_post_decision_quotes_are_excluded():
    quotes = pd.DataFrame(
        {
            "quote_id": ["before", "at-decision", "after"],
            "scraped_dt": [
                "2026-05-01T11:59:59Z",
                "2026-05-01T12:00:00Z",
                "2026-05-01T12:00:01Z",
            ],
        }
    )

    observed = quotes_observed_by_decision_time(quotes, "2026-05-01T12:00:00Z")

    assert observed["quote_id"].tolist() == ["before", "at-decision"]


def test_identity_join_probability_tolerance_is_explicit():
    assert probability_within_tolerance(0.501, 0.5)
    assert not probability_within_tolerance(0.5011, 0.5)
