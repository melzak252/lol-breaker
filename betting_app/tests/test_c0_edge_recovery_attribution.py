import hashlib
import json
from pathlib import Path

import pandas as pd
from src.analysis.c0_edge_recovery_attribution import (
    LEGACY_LEDGER_PATH,
    LEGACY_PREOVERWRITE_PATH,
    _legacy_50_history_evidence,
    attribute_legacy_bets,
    cohort_b_provenance,
    reconcile_t1h_bets,
)

def test_attribution_separates_exact_id_side_price_changes_from_coverage_and_policy():
    legacy = pd.DataFrame(
        [
            {"legacy_bet_id": "L1", "canonical_match_id": "m1", "side": "a", "odds": 2.0, "bookmaker_id": "book1", "payout_profit": 1.0},
            {"legacy_bet_id": "L2", "canonical_match_id": "m2", "side": "a", "odds": 2.0, "bookmaker_id": "book1", "payout_profit": -1.0},
            {"legacy_bet_id": "L3", "canonical_match_id": "m3", "side": "a", "odds": 2.0, "bookmaker_id": "book1", "payout_profit": 1.0},
        ]
    )
    replay = pd.DataFrame(
        [
            # Same exact fixture and side, but executable price differs.
            {"canonical_match_id": "m1", "side": "a", "odds": 1.8, "bookmaker_id": "book1", "eligible": True},
            # Same fixture but picked side differs; never treat it as the old bet.
            {"canonical_match_id": "m2", "side": "b", "odds": 2.5, "bookmaker_id": "book1", "eligible": True},
            # Exact fixture and price exist, but current policy does not bet.
            {"canonical_match_id": "m3", "side": "a", "odds": 2.0, "bookmaker_id": "book1", "eligible": False},
            # A new replay fixture is coverage, not a recoverable legacy bet.
            {"canonical_match_id": "m4", "side": "a", "odds": 2.1, "bookmaker_id": "book1", "eligible": True},
        ]
    )

    result = attribute_legacy_bets(legacy, replay)
    by_id = result.set_index("legacy_bet_id")

    assert by_id.loc["L1", "status"] == "exact_identity_price_changed"
    assert by_id.loc["L2", "status"] == "side_changed"
    assert by_id.loc["L3", "status"] == "policy_ineligible"
    assert result.attrs["unmatched_replay_rows"] == 1
    assert result["recoverable_profit"].isna().all()


def test_attribution_does_not_probability_match_or_invent_unknown_identity():
    legacy = pd.DataFrame(
        [{"legacy_bet_id": "L1", "canonical_match_id": None, "side": "a", "odds": 2.0, "bookmaker_id": "book1", "payout_profit": 100.0, "p_c0": 0.61}]
    )
    replay = pd.DataFrame(
        [{"canonical_match_id": "nearby-probability-match", "side": "a", "odds": 2.0, "bookmaker_id": "book1", "eligible": True, "p_c0": 0.6101}]
    )

    result = attribute_legacy_bets(legacy, replay)

    assert result.loc[0, "status"] == "identity_unknown"
    assert pd.isna(result.loc[0, "recoverable_profit"])
    assert result.attrs["unmatched_replay_rows"] == 1


def test_saved_optimizer_cohort_b_without_run_binding_remains_unverified():
    saved = {
        "timestamp_utc": "2026-10-20T01:48:45.384185+00:00",
        "verification_results": {
            "C0-D-Core": {
                "top_unconstrained_config": {
                    "config": {"alpha": 0.4, "eps": 0.025, "guard_b": 1.0, "max_odds": 4.0},
                    "cohort_b_5check": {"n_bets": 21, "n_wins": 14, "profit_pln": 1440.48},
                }
            }
        },
    }

    result = cohort_b_provenance(saved)

    assert result["saved_result_provenance_status"] == "SAVED_B_LABEL_PROVENANCE_UNVERIFIED"
    assert result["missing_binding_fields"] == [
        "canonical_fixture_ids",
        "cohort_time_bounds",
        "model_probability_basis",
        "source_hashes",
    ]
    assert result["current_source_rule"] == "later_match_disjoint_cohort_b"


def test_prior_t1h_comparison_uses_exact_fixture_side_and_separates_price_change():
    prior = pd.DataFrame(
        [
            {"opportunity_id": "c0-1-t1h", "canonical_match_id": 1, "side": "a", "bookmaker": "sts", "odds": 2.0},
            {"opportunity_id": "c0-2-t1h", "canonical_match_id": 2, "side": "a", "bookmaker": "sts", "odds": 2.0},
            {"opportunity_id": "c0-3-t1h", "canonical_match_id": 3, "side": "a", "bookmaker": "sts", "odds": 2.0},
        ]
    )
    first_eligible = pd.DataFrame(
        [
            {"canonical_match_id": 1, "side": "a", "bookmaker": "sts", "odds": 1.8},
            {"canonical_match_id": 2, "side": "b", "bookmaker": "sts", "odds": 2.2},
        ]
    )
    oracle = pd.DataFrame(
        [
            {"canonical_match_id": 1, "side": "a", "bookmaker": "sts", "odds": 2.0},
            {"canonical_match_id": 3, "side": "a", "bookmaker": "lebull", "odds": 2.0},
        ]
    )

    result = reconcile_t1h_bets(
        prior,
        {"first_eligible_2h": first_eligible, "all_snapshot_oracle": oracle},
    ).set_index(["legacy_opportunity_id", "timing"])

    assert result.loc[("c0-1-t1h", "first_eligible_2h"), "status"] == "same_fixture_side_price_or_book_changed"
    assert result.loc[("c0-2-t1h", "first_eligible_2h"), "status"] == "fixture_selected_side_changed"
    assert result.loc[("c0-3-t1h", "first_eligible_2h"), "status"] == "fixture_not_selected"
    assert result.loc[("c0-1-t1h", "all_snapshot_oracle"), "status"] == "same_fixture_side_price_book_match"
    assert result.loc[("c0-3-t1h", "all_snapshot_oracle"), "status"] == "same_fixture_side_price_or_book_changed"
    assert result.attrs["exact_fixture_id_matching"] is True


def test_t1h_reconciliation_keeps_quote_identity_separate_and_measures_clock_age():
    prior = pd.DataFrame(
        [{
            "opportunity_id": "legacy-t1h",
            "canonical_match_id": 42,
            "side": "a",
            "bookmaker": "sts",
            "odds": 2.9,
            "decision_at": "2026-08-10T14:00:00Z",
            "start_at": "2026-08-10T15:00:00Z",
            "quoted_at": "2026-08-10T12:55:42Z",
            "reference_quote_id": 100,
        }]
    )
    replay = pd.DataFrame(
        [{
            "opportunity_id": "fixed-2h",
            "canonical_match_id": 42,
            "side": "a",
            "bookmaker": "sts",
            "odds": 3.05,
            "decision_at": "2026-08-09T00:55:52Z",
            "start_at": "2026-08-10T15:00:00Z",
            "quoted_at": "2026-08-09T00:55:52Z",
            "quote_id": 200,
        }]
    )

    result = reconcile_t1h_bets(prior, {"fixed_2h": replay}).iloc[0]

    assert result["status"] == "same_fixture_side_price_or_book_changed"
    assert result["legacy_decision_lead_seconds"] == 3600
    assert result["legacy_quote_age_seconds"] == 3858
    assert not result["legacy_quote_within_300s_at_decision"]
    assert result["legacy_quote_id"] is None
    assert result["legacy_reference_quote_id"] == 100
    assert result["replay_quote_id"] == 200
    assert result["quote_identity_status"] == "partial_unknown"
    assert result["replay_odds"] == 3.05


def test_prior_t1h_freshness_counts_use_latest_admitted_peer_books():
    from src.analysis.c0_edge_recovery_attribution import prior_t1h_freshness_counts

    prior = pd.DataFrame([{
        "opportunity_id": "c0-42-t1h",
        "canonical_match_id": 42,
        "decision_at": "2026-08-10T14:00:00Z",
        "reference_bookmaker_id": 1,
    }])
    quotes = pd.DataFrame([
        {
            "id": 101, "canonical_match_id": 42, "bookmaker_id": 1,
            "bookmaker": "sts", "quoted_at": "2026-08-10T13:59:50Z",
            "odds_a": 2.0, "odds_b": 2.1, "identity_verified": True,
            "raw_a_is_canonical_a": True, "is_live": False, "market_type": "match_winner",
        },
        {
            "id": 102, "canonical_match_id": 42, "bookmaker_id": 2,
            "bookmaker": "betclic", "quoted_at": "2026-08-10T13:59:40Z",
            "odds_a": 2.0, "odds_b": 2.1, "identity_verified": True,
            "raw_a_is_canonical_a": True, "is_live": False, "market_type": "match_winner",
        },
        {
            "id": 103, "canonical_match_id": 42, "bookmaker_id": 3,
            "bookmaker": "stale", "quoted_at": "2026-08-10T13:54:59Z",
            "odds_a": 2.0, "odds_b": 2.1, "identity_verified": True,
            "raw_a_is_canonical_a": True, "is_live": False, "market_type": "match_winner",
        },
        {
            "id": 104, "canonical_match_id": 42, "bookmaker_id": 4,
            "bookmaker": "boundary", "quoted_at": "2026-08-10T13:55:00Z",
            "odds_a": 2.0, "odds_b": 2.1, "identity_verified": True,
            "raw_a_is_canonical_a": True, "is_live": False, "market_type": "match_winner",
        },
    ])

    result = prior_t1h_freshness_counts(prior, quotes).iloc[0]

    assert result["fresh_bookmaker_count"] == 3
    assert result["fresh_peer_bookmaker_count"] == 2
    assert result["fresh_bookmaker_ids"] == [1, 2, 4]


def test_recovered_legacy50_snapshot_requires_both_source_hash_bindings(tmp_path: Path):
    earlier_path = tmp_path / LEGACY_PREOVERWRITE_PATH
    retained_path = tmp_path / LEGACY_LEDGER_PATH
    earlier_path.parent.mkdir(parents=True)
    retained_path.parent.mkdir(parents=True)
    earlier_path.write_text("pre-overwrite\n", encoding="utf-8")
    retained_path.write_text("retained\n", encoding="utf-8")

    receipt_path = tmp_path / "recovery" / "legacy_50_version_comparison.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({
        "earlier_version": LEGACY_PREOVERWRITE_PATH,
        "retained_version": LEGACY_LEDGER_PATH,
        "earlier_version_sha256": hashlib.sha256(earlier_path.read_bytes()).hexdigest(),
        "retained_version_sha256": hashlib.sha256(retained_path.read_bytes()).hexdigest(),
    }), encoding="utf-8")

    evidence = _legacy_50_history_evidence(tmp_path, retained_path, receipt_path)

    assert evidence["recovered_version_binding_status"] == "BOTH_VERSION_HASHES_MATCH_RECEIPT"
    assert evidence["source_history"]["original_selection_generator"] == "NOT_RECOVERED"
    assert evidence["source_history"]["identity_status"]["canonical_match_ids"] == "NOT_RECOVERED"
    assert evidence["source_hashes"][LEGACY_PREOVERWRITE_PATH] == hashlib.sha256(
        earlier_path.read_bytes()
    ).hexdigest()
    assert evidence["global_absence_claim"] is False
