import numpy as np
import pandas as pd
import pytest


FIRST = "OPEN_OR_FIRST_OBSERVED"


def oddsportal_rows():
    return pd.DataFrame([
        {"source": "oddsportal", "source_event_id": "one", "golgg_match_id": "1", "bookmaker": "a", "horizon": FIRST, "timed_price_eligible": True, "p_c0": .8, "q_focal": .5, "y": 1, "label_status": "known", "canonical_date": "2026-01-01", "quote_at": "2025-12-31T12:00:00Z"},
        {"source": "oddsportal", "source_event_id": "one", "golgg_match_id": "1", "bookmaker": "b", "horizon": FIRST, "timed_price_eligible": True, "p_c0": .6, "q_focal": .4, "y": 1, "label_status": "known", "canonical_date": "2026-01-01", "quote_at": "2025-12-31T11:00:00Z"},
        {"source": "oddsportal", "source_event_id": "two", "golgg_match_id": "2", "bookmaker": "a", "horizon": FIRST, "timed_price_eligible": True, "p_c0": .2, "q_focal": .3, "y": 0, "label_status": "known", "canonical_date": "2026-02-01", "quote_at": "2026-01-31T12:00:00Z"},
    ])


def test_oddsportal_comparison_uses_earliest_eligible_focal_quote_per_fixture():
    from src.analysis.saved_market_comparisons import summarize_oddsportal_comparisons

    rows = oddsportal_rows()
    result = summarize_oddsportal_comparisons(rows, ["2026-01", "2026-02"], draws=100)
    scores = result["endpoints"][FIRST]["scores"]
    expected = (-np.log(.6) - np.log(.8)) / 2
    assert scores["sports_model"]["n"] == 2
    assert scores["sports_model"]["log_loss"] == pytest.approx(expected)
    assert scores["sports_model"]["log_loss"] != pytest.approx(-np.log([.8, .6, .8]).mean())
    opposite = rows.copy()
    for column in ["p_c0", "q_focal", "y"]:
        opposite[column] = 1 - opposite[column]
    swapped = summarize_oddsportal_comparisons(opposite, ["2026-01", "2026-02"], draws=100)
    assert swapped["endpoints"][FIRST]["scores"]["sports_model"]["log_loss"] == pytest.approx(expected)


@pytest.mark.parametrize("corruption", ["duplicate", "conflicting_label", "invalid_probability"])
def test_oddsportal_comparison_rejects_corrupted_paired_rows(corruption):
    from src.analysis.saved_market_comparisons import summarize_oddsportal_comparisons

    rows = oddsportal_rows()
    if corruption == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    elif corruption == "conflicting_label":
        rows.loc[1, "y"] = 0
    else:
        rows.loc[1, "p_c0"] = 1.2
    with pytest.raises(ValueError):
        summarize_oddsportal_comparisons(rows, ["2026-01", "2026-02"], draws=100)


def pinnacle_rows():
    rows = []
    for gid, p, q, y in [("1", .8, .7, 1), ("2", .3, .4, 0)]:
        rows.append({"source_event_id": gid, "golgg_match_id": gid, "canonical_gid": gid,
                     "horizon": FIRST, "p_c20r": p, "p_pinnacle": q, "canonical_y": y,
                     "label_available": True, "c20r_pair_eligible": True,
                     "polish_pair_eligible": False, "three_way_eligible": False,
                     "prestart_eligible": True, "month": "2026-01", "canonical_date": "2026-01-02",
                     "endpoint_decision_at": "2026-01-01T12:00:00Z",
                     "earliest_retained_start_at": "2026-01-02T12:00:00Z",
                     "scheduled_start_at": "2026-01-02T12:00:00Z",
                     "p_polish_consensus": np.nan})
    return pd.DataFrame(rows)


def test_pinnacle_comparison_uses_the_identical_canonical_pairs():
    from src.analysis.saved_market_comparisons import summarize_pinnacle_comparisons

    rows = pinnacle_rows()
    result = summarize_pinnacle_comparisons(rows, draws=100)
    scores = result["endpoints"][FIRST]["scores"]
    assert scores["sports_model"]["n"] == scores["market"]["n"] == 2
    assert scores["sports_model"]["log_loss"] == pytest.approx(-np.log([.8, .7]).mean())
    assert scores["market"]["brier"] == pytest.approx(((.7 - 1) ** 2 + .4 ** 2) / 2)


@pytest.mark.parametrize("corruption", ["duplicate", "missing_model", "poststart"])
def test_pinnacle_comparison_does_not_silently_drop_broken_admitted_pairs(corruption):
    from src.analysis.saved_market_comparisons import summarize_pinnacle_comparisons

    rows = pinnacle_rows()
    if corruption == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    elif corruption == "missing_model":
        rows.loc[0, "p_c20r"] = np.nan
    else:
        rows.loc[0, "endpoint_decision_at"] = "2026-01-02T13:00:00Z"
    with pytest.raises(ValueError):
        summarize_pinnacle_comparisons(rows, draws=100)


def test_market_comparisons_retain_unmapped_source_rows_as_coverage_absences():
    from src.analysis.saved_market_comparisons import summarize_oddsportal_comparisons, summarize_pinnacle_comparisons

    odds = oddsportal_rows()
    missing = odds.iloc[0].copy()
    missing["source_event_id"] = "unmapped"
    missing[["golgg_match_id", "canonical_date", "p_c0"]] = [None, None, np.nan]
    missing["timed_price_eligible"] = False
    result = summarize_oddsportal_comparisons(pd.concat([odds, missing.to_frame().T], ignore_index=True),
                                             ["2026-01", "2026-02"], draws=100)
    assert result["endpoints"][FIRST]["scores"]["sports_model"]["n"] == 2
    pin = pinnacle_rows()
    missing = pin.iloc[0].copy()
    missing["source_event_id"] = "unmapped"
    for column in ["golgg_match_id", "canonical_gid", "canonical_date", "month", "p_c20r", "canonical_y"]:
        missing[column] = None
    missing["c20r_pair_eligible"] = missing["label_available"] = False
    result = summarize_pinnacle_comparisons(pd.concat([pin, missing.to_frame().T], ignore_index=True), draws=100)
    assert result["endpoints"][FIRST]["scores"]["sports_model"]["n"] == 2


def test_pinnacle_family_interval_actually_uses_the_wider_confidence_level():
    from src.analysis.saved_market_comparisons import summarize_pinnacle_comparisons

    template = pinnacle_rows().iloc[0].to_dict()
    rows = []
    for i, month in enumerate(pd.date_range("2025-01-01", periods=12, freq="MS")):
        row = dict(template)
        row.update(source_event_id=str(i), golgg_match_id=str(i), canonical_gid=str(i),
                   month=month.strftime("%Y-%m"), canonical_date=month.strftime("%Y-%m-02"),
                   endpoint_decision_at=month.strftime("%Y-%m-01T12:00:00Z"),
                   earliest_retained_start_at=month.strftime("%Y-%m-02T12:00:00Z"),
                   scheduled_start_at=month.strftime("%Y-%m-02T12:00:00Z"),
                   p_c20r=.8, p_pinnacle=.50 + .035*i)
        rows.append(row)
    result = summarize_pinnacle_comparisons(pd.DataFrame(rows))
    intervals = result["endpoints"][FIRST]["contrasts"]
    nominal = intervals["log_loss"]
    adjusted = intervals["familywise_98_75_log_loss"]
    assert adjusted["ci_lower"] < nominal["ci_lower"]
    assert adjusted["ci_upper"] > nominal["ci_upper"]


def test_pinnacle_blocks_by_scheduled_start_not_canonical_or_earlier_actual_date():
    from src.analysis.saved_market_comparisons import summarize_pinnacle_comparisons

    rows = pinnacle_rows().iloc[:1].copy()
    rows.loc[0, "canonical_date"] = "2026-01-31"
    rows.loc[0, "month"] = "2026-02"
    rows.loc[0, "scheduled_start_at"] = "2026-02-01T00:00:00Z"
    rows.loc[0, "endpoint_decision_at"] = "2026-01-31T12:00:00Z"
    rows.loc[0, "earliest_retained_start_at"] = "2026-01-31T23:06:00Z"
    result = summarize_pinnacle_comparisons(rows, draws=100)
    assert result["endpoints"][FIRST]["scores"]["sports_model"]["n"] == 1
    rows.loc[0, "month"] = "2026-01"
    with pytest.raises(ValueError):
        summarize_pinnacle_comparisons(rows, draws=100)
