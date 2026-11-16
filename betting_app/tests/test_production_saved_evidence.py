import json

import numpy as np
import pandas as pd
import pytest


def production_ledger():
    return pd.DataFrame([
        {"opportunity_id": "snap:1:team_a", "canonical_match_id": "one", "bookmaker": "focal",
         "decision_at": "2026-01-01T10:00:00Z", "start": "2026-01-01T12:00:00Z",
         "settle_at": "2026-01-01T15:00:00Z", "side": "team_a", "odds": 2.5,
         "tax_rate": 0.12, "stake": "100.00", "y": 1, "realized_pnl": "120.00", "settled": True},
        {"opportunity_id": "snap:2:team_b", "canonical_match_id": "two", "bookmaker": "focal",
         "decision_at": "2026-02-01T10:00:00Z", "start": "2026-02-01T12:00:00Z",
         "settle_at": "2026-02-01T15:00:00Z", "side": "team_b", "odds": 3.0,
         "tax_rate": 0.12, "stake": "50.00", "y": 0, "realized_pnl": "-50.00", "settled": True},
    ])


def production_references(ledger):
    rows = []
    for ticket in ledger.itertuples(index=False):
        start = pd.Timestamp(ticket.start)
        rows.append({
            "snapshot_id": ticket.opportunity_id.rsplit(":", 1)[0],
            "canonical_match_id": ticket.canonical_match_id, "bookmaker": ticket.bookmaker,
            "entry_clock": ticket.decision_at, "fixture_start_utc": ticket.start,
            "entry_odds_team_a": ticket.odds if ticket.side == "team_a" else 1.5,
            "entry_odds_team_b": ticket.odds if ticket.side == "team_b" else 1.5,
            "samebook_close_clock": (start - pd.Timedelta(minutes=3)).isoformat(),
            "samebook_close_odds_team_a": 2.0, "samebook_close_odds_team_b": 2.5,
            "samebook_age_seconds_to_start": 180.0,
            "samebook_300_eligible": True, "samebook_300_missing_reason": "",
            "loo_300_eligible": True, "loo_300_missing_reason": "",
            "loo_300_fair_q_team_a": 0.6, "loo_300_n_books": 2,
            "loo_300_contributor_books": json.dumps(["peer-one", "peer-two"]),
            "loo_300_contributor_snapshot_ids": json.dumps(["peer-quote-one", "peer-quote-two"]),
            "loo_300_contributor_clocks": json.dumps([
                (start - pd.Timedelta(minutes=5)).isoformat(),
                (start - pd.Timedelta(minutes=2)).isoformat(),
            ]),
            "loo_300_contributor_ages_seconds": json.dumps([300.0, 120.0]),
        })
    return pd.DataFrame(rows)


def test_production_wallet_weights_returns_by_stake_without_flipping_side_b_again():
    from src.analysis.production_saved_evidence import summarize_production_wallet

    result = summarize_production_wallet(production_ledger(), ["2026-01", "2026-02", "2026-03"], draws=200)
    assert result["bets"] == result["settled_bets"] == 2
    assert result["wins"] == 1
    assert result["profit"] == pytest.approx(70)
    assert result["settled_turnover"] == pytest.approx(150)
    assert result["roi"] == pytest.approx(70 / 150)
    assert result["unit_yield"] == pytest.approx(0.1)
    assert result["roi_interval"]["blocks"] == 3
    assert result["roi_interval"]["empty_replicates"] > 0
    assert result["roi_interval"]["ci_lower"] is None


def test_production_wallet_retains_pending_risk_and_empty_conditional_returns():
    from src.analysis.production_saved_evidence import summarize_production_wallet

    ledger = production_ledger()
    pending = ledger.iloc[[0]].assign(opportunity_id="pending:team_a", canonical_match_id="pending",
                                     settled=False, y=np.nan, realized_pnl=None, settle_at=None, stake="30")
    result = summarize_production_wallet(pd.concat([ledger, pending], ignore_index=True), ["2026-01", "2026-02"])
    assert result["pending_bets"] == 1
    assert result["committed_stake"] == pytest.approx(180)
    assert result["settled_profit"] == pytest.approx(70)
    assert result["settled_roi"] == pytest.approx(70 / 150)
    assert result["profit"] is None and result["roi"] is None
    empty = summarize_production_wallet(ledger.iloc[:0], ["2026-01", "2026-02"])
    assert empty["bets"] == 0
    assert empty["roi"] is None and empty["unit_yield"] is None
    assert empty["roi_interval"]["estimate"] is None


@pytest.mark.parametrize("corruption", ["payout", "duplicate_ticket", "duplicate_fixture", "naive_clock"])
def test_production_wallet_rejects_inconsistent_saved_money_or_identity(corruption):
    from src.analysis.production_saved_evidence import summarize_production_wallet

    ledger = production_ledger()
    if corruption == "payout":
        ledger.loc[0, "realized_pnl"] = "121"
    elif corruption == "duplicate_ticket":
        ledger = pd.concat([ledger, ledger.iloc[[0]]], ignore_index=True)
    elif corruption == "duplicate_fixture":
        ledger.loc[1, "canonical_match_id"] = "one"
    else:
        ledger.loc[0, "decision_at"] = "2026-01-01T10:00:00"
    with pytest.raises(ValueError):
        summarize_production_wallet(ledger, ["2026-01", "2026-02"])


def test_production_closes_use_actual_selected_side_and_keep_tax_separate():
    from src.analysis.production_saved_evidence import pair_production_closes, summarize_production_closes

    ledger = production_ledger()
    pairs = pair_production_closes(ledger, production_references(ledger))
    assert pairs["snapshot_id"].tolist() == ["snap:1", "snap:2"]
    assert pairs["samebook_price_move"].tolist() == pytest.approx([0.25, 0.2])
    assert pairs["loo_gross_proxy"].tolist() == pytest.approx([0.5, 0.2])
    assert pairs["loo_net_proxy"].tolist() == pytest.approx([0.32, 0.056])
    result = summarize_production_closes(pairs, ["2026-01", "2026-02"])
    assert result["loo_covered"] == result["samebook_covered"] == 2
    assert result["loo_gross_proxy"]["estimate"] == pytest.approx(0.35)
    assert result["stake_weighted_loo_gross_proxy"]["estimate"] == pytest.approx(0.4)


def test_production_closes_preserve_missing_references_and_unknown_outcomes():
    from src.analysis.production_saved_evidence import pair_production_closes, summarize_production_closes

    ledger = production_ledger()
    references = production_references(ledger)
    references.loc[0, ["loo_300_eligible", "samebook_300_eligible"]] = False
    references.loc[0, ["loo_300_missing_reason", "samebook_300_missing_reason"]] = "stale_reference"
    pairs = pair_production_closes(ledger, references.iloc[[0]])
    result = summarize_production_closes(pairs, ["2026-01", "2026-02"])
    assert result["bets"] == 2 and result["loo_covered"] == result["samebook_covered"] == 0
    assert result["loo_gross_proxy"]["estimate"] is None
    assert sum(result["loo_absences"].values()) == 2
    assert pairs["loo_net_proxy"].isna().all()
    unknown = ledger.assign(y=np.nan, settled=False, realized_pnl=None, settle_at=None)
    result = summarize_production_closes(pair_production_closes(unknown, production_references(ledger)), ["2026-01", "2026-02"])
    assert result["loo_covered"] == 2
    empty = summarize_production_closes(pair_production_closes(ledger.iloc[:0], references), ["2026-01", "2026-02"])
    assert empty["bets"] == 0 and empty["loo_gross_proxy"]["estimate"] is None


@pytest.mark.parametrize("corruption", ["duplicate", "book", "fixture", "entry_odds", "focal_contributor",
                                       "future_contributor", "naive_contributor", "stale_eligible", "same_clock_close"])
def test_production_closes_reject_ambiguous_or_contradictory_reference_proofs(corruption):
    from src.analysis.production_saved_evidence import pair_production_closes

    ledger = production_ledger().iloc[[0]].copy()
    references = production_references(ledger)
    if corruption == "duplicate":
        references = pd.concat([references, references], ignore_index=True)
    elif corruption == "book":
        references.loc[0, "bookmaker"] = "different-book"
    elif corruption == "fixture":
        references.loc[0, "canonical_match_id"] = "different-fixture"
    elif corruption == "entry_odds":
        references.loc[0, "entry_odds_team_a"] = 2.6
    elif corruption == "focal_contributor":
        references.loc[0, "loo_300_contributor_books"] = json.dumps(["focal", "peer-two"])
    elif corruption == "future_contributor":
        references.loc[0, "loo_300_contributor_clocks"] = json.dumps(["2026-01-01T12:01:00Z", "2026-01-01T11:58:00Z"])
    elif corruption == "naive_contributor":
        references.loc[0, "loo_300_contributor_clocks"] = json.dumps(["2026-01-01T11:55:00", "2026-01-01T11:58:00Z"])
    elif corruption == "stale_eligible":
        references.loc[0, "loo_300_contributor_clocks"] = json.dumps(["2026-01-01T11:54:59Z", "2026-01-01T11:58:00Z"])
        references.loc[0, "loo_300_contributor_ages_seconds"] = json.dumps([301.0, 120.0])
    else:
        references.loc[0, "samebook_close_clock"] = ledger.loc[0, "decision_at"]
    with pytest.raises(ValueError):
        pair_production_closes(ledger, references)
