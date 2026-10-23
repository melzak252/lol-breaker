"""A paired price cannot exist before its later side observation."""
import pandas as pd
import pytest

from src.analysis.long_term_native_inputs import _latest_start, _quote_clock


@pytest.mark.parametrize("source,fields", [
    ("oddsportal", ("source_clock1", "source_clock2")),
    ("oddspapi", ("outcome_1_quote_at", "outcome_2_quote_at")),
])
def test_paired_price_clock_is_later_side_even_across_utc_midnight(source, fields):
    row = pd.Series({fields[0]: "2021-12-31T23:59:59Z", fields[1]: "2022-01-01T00:00:01Z"})
    assert _quote_clock(row, source) == pd.Timestamp("2022-01-01T00:00:01Z")
    row[fields[0]], row[fields[1]] = row[fields[1]], row[fields[0]]
    assert _quote_clock(row, source) == pd.Timestamp("2022-01-01T00:00:01Z")


@pytest.mark.parametrize("source,fields", [
    ("oddsportal", ("source_clock1", "source_clock2")),
    ("oddspapi", ("outcome_1_quote_at", "outcome_2_quote_at")),
])
def test_unknown_or_naive_side_clock_cannot_become_a_paired_decision(source, fields):
    row = pd.Series({fields[0]: "2021-01-12T10:00:00Z", fields[1]: None})
    assert _quote_clock(row, source) is None
    row[fields[1]] = "2021-01-12T10:00:01"
    assert _quote_clock(row, source) is None


def test_late_actual_start_does_not_relax_known_scheduled_start():
    row = pd.Series({"scheduled_start_at": "2021-01-12T16:00:00Z",
                     "true_start_at": "2021-01-12T16:12:00Z",
                     "source_start_at": "2021-01-12T16:12:00Z"})
    assert _latest_start(row, "oddspapi") == pd.Timestamp("2021-01-12T16:00:00Z")
