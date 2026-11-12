import pandas as pd

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
