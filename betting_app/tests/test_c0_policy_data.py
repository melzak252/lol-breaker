"""Policy quote admission boundaries, exercised with synthetic retained rows only."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.analysis.c0_policy_data import _canonical_request, admit_policy_quotes


ORIGIN = datetime(2026, 10, 1, 12, tzinfo=UTC)


def _opportunity(*, decision_at=ORIGIN, match_id=41, opportunity_id="opp-1"):
    return pd.DataFrame(
        [
            {
                "opportunity_id": opportunity_id,
                "canonical_match_id": match_id,
                "decision_at": decision_at,
                "start_at": decision_at + timedelta(hours=1),
                "settle_at": decision_at + timedelta(hours=4),
                "league": "L",
                "best_of": 3,
                "y": None,
                "absence_flags": [],
            }
        ]
    )


def _quote(
    quote_id,
    *,
    quoted_at=ORIGIN - timedelta(seconds=20),
    match_id=41,
    bookmaker_id=7,
    bookmaker="Book",
    odds_a=2.0,
    odds_b=2.0,
    is_live=False,
    identity_verified=True,
    raw_a_is_canonical_a=True,
    label_a="Misleading A",
    label_b="Misleading B",
):
    return {
        "quote_id": quote_id,
        "canonical_match_id": match_id,
        "bookmaker_id": bookmaker_id,
        "bookmaker": bookmaker,
        "quoted_at": quoted_at,
        "odds_a": odds_a,
        "odds_b": odds_b,
        "is_live": is_live,
        "identity_verified": identity_verified,
        "raw_a_is_canonical_a": raw_a_is_canonical_a,
        "label_a": label_a,
        "label_b": label_b,
    }


def _admit(quotes, *, ttl_seconds=300, opportunities=None):
    return admit_policy_quotes(
        opportunities if opportunities is not None else _opportunity(),
        pd.DataFrame(quotes),
        ttl_seconds=ttl_seconds,
    )


def test_latest_invalid_state_suppresses_older_valid_state():
    result = _admit(
        [
            _quote("old-valid", quoted_at=ORIGIN - timedelta(seconds=60)),
            _quote(
                "latest-live", quoted_at=ORIGIN - timedelta(seconds=10), is_live=True
            ),
        ]
    )
    assert result.empty


def test_future_quote_is_excluded():
    result = _admit([_quote("future", quoted_at=ORIGIN + timedelta(microseconds=1))])
    assert result.empty


def test_quote_exactly_at_ttl_cutoff_is_admitted_and_retains_quote_id():
    result = _admit([_quote("cutoff", quoted_at=ORIGIN - timedelta(seconds=300))])
    assert result["quote_id"].tolist() == ["cutoff"]


def test_identity_flag_withholds_unverified_quote():
    result = _admit([_quote("unverified", identity_verified=False)])
    assert result.empty


def test_raw_orientation_is_applied_once_and_display_labels_do_not_set_orientation():
    row = _quote(
        "swapped",
        odds_a=2.5,
        odds_b=1.6,
        raw_a_is_canonical_a=False,
        label_a="canonical A",
        label_b="canonical B",
    )
    result = _admit([row])
    assert len(result) == 1
    assert result.iloc[0]["canonical_odds_a"] == pytest.approx(1.6)
    assert result.iloc[0]["canonical_odds_b"] == pytest.approx(2.5)


def test_equal_timestamp_conflicts_have_input_order_independent_winner():
    rows = [
        _quote("tie-a", quoted_at=ORIGIN - timedelta(seconds=5), odds_a=1.8),
        _quote("tie-b", quoted_at=ORIGIN - timedelta(seconds=5), odds_a=2.4),
    ]
    forward = _admit(rows)
    reverse = _admit(list(reversed(rows)))
    assert forward["quote_id"].tolist() == reverse["quote_id"].tolist()


def test_future_state_does_not_suppress_latest_past_quote():
    result = _admit(
        [
            _quote("past", quoted_at=ORIGIN - timedelta(seconds=20)),
            _quote("future", quoted_at=ORIGIN + timedelta(seconds=10)),
        ]
    )
    assert result["quote_id"].tolist() == ["past"]


def test_nonfinite_or_unknown_live_quote_is_withheld():
    result = _admit(
        [
            _quote("infinite", odds_a=float("inf")),
            _quote("unknown-live", is_live=None),
        ]
    )
    assert result.empty


def test_adapter_accepts_native_quote_column_names():
    opportunities = _opportunity()
    quotes = pd.DataFrame(
        [
            {
                "id": "native",
                "canonical_match_id": 41,
                "bookmaker_id": 7,
                "scraped_at": ORIGIN - timedelta(seconds=10),
                "odds_a": 2.2,
                "odds_b": 1.8,
                "is_live": 0,
                "identity_verified": True,
                "raw_a_is_canonical_a": True,
                "market_type": "match_winner",
            }
        ]
    )
    result = admit_policy_quotes(opportunities, quotes)
    assert result["quote_id"].tolist() == ["native"]
    assert result.iloc[0]["canonical_odds_a"] == pytest.approx(2.2)


def test_latest_equal_clock_conflict_is_withheld_independent_of_input_order():
    rows = [
        _quote("tie-a", quoted_at=ORIGIN - timedelta(seconds=5), odds_a=1.8),
        _quote("tie-b", quoted_at=ORIGIN - timedelta(seconds=5), odds_a=2.4),
    ]
    assert _admit(rows).empty
    assert _admit(list(reversed(rows))).empty


def test_numeric_zero_live_flag_is_known_prematch_but_unknown_is_withheld():
    known = _admit([_quote("known-prematch", is_live=0)])
    unknown = _admit([_quote("unknown-live", is_live=None)])
    assert known["quote_id"].tolist() == ["known-prematch"]
    assert unknown.empty


def test_native_request_closure_requires_exact_schema_and_normalizes_roster_order():
    request = {
        "team1_id": "41",
        "team2_id": "42",
        "roster_a": ["5", "4", "3", "2", "1"],
        "roster_b": ["10", "9", "8", "7", "6"],
        "best_of": 3,
        "decision_at": "2026-10-01T12:00:00+00:00",
        "competition_context": "League",
        "start_at": "2026-10-01T13:00:00+00:00",
        "mode": "full",
    }
    closed = _canonical_request(request)
    assert closed["roster_a"] == ["1", "2", "3", "4", "5"]
    assert closed["decision_at"] == pd.Timestamp(ORIGIN)
    assert _canonical_request({**request, "fixture_id": "41"}) is None
