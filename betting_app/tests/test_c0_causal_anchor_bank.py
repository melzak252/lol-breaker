from __future__ import annotations

import numpy as np
import pytest

from src.models.c0_causal_anchor_bank import summarize_released_maps


CUTOFF = 10_000


def _vector(
    *,
    first_stat: float = 0.0,
    first_stat_known: bool = False,
    second_stat: float = 0.0,
    second_stat_known: bool = False,
) -> np.ndarray:
    vector = np.zeros(33, dtype=np.float32)
    vector[0] = first_stat
    vector[12] = float(first_stat_known)
    vector[1] = second_stat
    vector[13] = float(second_stat_known)
    return vector


def _record(
    map_id: str,
    *,
    age: int = 10,
    release_age: int = 1,
    vector: np.ndarray | None = None,
    series_id: str = "series-1",
    player_id: str = "player-1",
) -> dict:
    return {
        "map_id": map_id,
        "series_id": series_id,
        "player_id": player_id,
        "played_day": CUTOFF - age,
        "release_day": CUTOFF - release_age,
        "vector": _vector() if vector is None else vector,
    }


def _assert_same_summary(left: dict, right: dict) -> None:
    np.testing.assert_array_equal(left["history"], right["history"])
    assert left["mask"] == right["mask"]
    assert left["count"] == right["count"]
    assert left["members"] == right["members"]


@pytest.mark.parametrize("horizon_days", [90, 180, 365])
def test_released_map_window_uses_strict_played_day_boundaries(
    horizon_days: int,
) -> None:
    records = [
        _record(
            "same-day", age=0, vector=_vector(first_stat=100, first_stat_known=True)
        ),
        _record(
            "one-day-old", age=1, vector=_vector(first_stat=1, first_stat_known=True)
        ),
        _record(
            "last-in-window",
            age=horizon_days - 1,
            vector=_vector(first_stat=horizon_days - 1, first_stat_known=True),
        ),
        _record(
            "on-horizon",
            age=horizon_days,
            vector=_vector(first_stat=200, first_stat_known=True),
        ),
        _record(
            "past-horizon",
            age=horizon_days + 1,
            vector=_vector(first_stat=300, first_stat_known=True),
        ),
        _record(
            "release-on-cutoff",
            age=2,
            release_age=0,
            vector=_vector(first_stat=400, first_stat_known=True),
        ),
        _record(
            "release-after-cutoff",
            age=3,
            release_age=-1,
            vector=_vector(first_stat=500, first_stat_known=True),
        ),
    ]

    result = summarize_released_maps(
        records, cutoff_day=CUTOFF, horizon_days=horizon_days
    )

    assert result["count"] == 2
    assert result["mask"] is True
    assert set(result["members"]) == {"one-day-old", "last-in-window"}
    np.testing.assert_allclose(
        result["history"][0], (1 + horizon_days - 1) / 2, rtol=0, atol=1e-6
    )


def test_future_unreleased_and_target_event_perturbations_do_not_change_anchor() -> (
    None
):
    records = [
        _record("prior", age=5, vector=_vector(first_stat=7, first_stat_known=True)),
        _record(
            "future-map",
            age=-1,
            release_age=20,
            vector=_vector(first_stat=20, first_stat_known=True),
        ),
        _record(
            "unreleased-map",
            age=4,
            release_age=0,
            vector=_vector(first_stat=30, first_stat_known=True),
        ),
        _record(
            "target-map",
            age=0,
            release_age=2,
            vector=_vector(first_stat=40, first_stat_known=True),
        ),
    ]
    before = summarize_released_maps(records, cutoff_day=CUTOFF, horizon_days=90)

    records[1]["vector"][:] = -900
    records[1]["release_day"] = CUTOFF - 1
    records[2]["vector"][:] = 800
    records[2]["release_day"] = CUTOFF + 1
    records[3]["vector"][:] = 700
    records[3]["series_id"] = "perturbed-target-series"
    after = summarize_released_maps(records, cutoff_day=CUTOFF, horizon_days=90)

    _assert_same_summary(before, after)
    assert before["count"] == 1
    assert before["members"] == ["prior"]
    assert before["history"][0] == pytest.approx(7)


def test_identical_statistics_on_different_map_ids_count_as_distinct_maps() -> None:
    shared_vector = _vector(
        first_stat=12, first_stat_known=True, second_stat=3, second_stat_known=True
    )
    records = [
        _record("map-b", vector=shared_vector.copy()),
        _record("map-a", vector=shared_vector.copy()),
    ]

    result = summarize_released_maps(records, cutoff_day=CUTOFF, horizon_days=90)

    assert result["count"] == 2
    assert set(result["members"]) == {"map-a", "map-b"}
    assert len(result["members"]) == 2
    np.testing.assert_array_equal(result["history"], shared_vector)


def test_exact_duplicate_map_record_is_deduplicated_by_stable_map_id() -> None:
    record = _record("same-map", vector=_vector(first_stat=9, first_stat_known=True))
    records = [record, {**record, "vector": record["vector"].copy()}]

    result = summarize_released_maps(records, cutoff_day=CUTOFF, horizon_days=90)

    assert result["count"] == 1
    assert result["members"] == ["same-map"]
    assert result["history"][0] == pytest.approx(9)


def test_conflicting_records_for_one_stable_map_id_are_rejected() -> None:
    first = _record("same-map", vector=_vector(first_stat=9, first_stat_known=True))
    conflict = {**first, "vector": _vector(first_stat=10, first_stat_known=True)}

    with pytest.raises(ValueError):
        summarize_released_maps([first, conflict], cutoff_day=CUTOFF, horizon_days=90)


def test_unknown_statistics_remain_masked_in_native_summary() -> None:
    records = [
        _record("map-1", vector=_vector(second_stat=4, second_stat_known=True)),
        _record("map-2", age=12, vector=_vector(second_stat=8, second_stat_known=True)),
    ]

    result = summarize_released_maps(records, cutoff_day=CUTOFF, horizon_days=90)

    assert result["mask"] is True
    assert result["count"] == 2
    assert result["history"][0] == 0
    assert result["history"][12] == 0
    assert result["history"][1] == pytest.approx(6)
    assert result["history"][13] == pytest.approx(1)


def test_native_presence_channel_is_binary_when_any_selected_map_observes_stat() -> (
    None
):
    records = [
        _record("known", vector=_vector(first_stat=2, first_stat_known=True)),
        _record(
            "unknown", age=12, vector=_vector(first_stat=0, first_stat_known=False)
        ),
    ]

    result = summarize_released_maps(records, cutoff_day=CUTOFF, horizon_days=90)

    assert result["history"][0] == pytest.approx(2)
    assert result["history"][12] == pytest.approx(1)


def test_summary_does_not_mutate_input_records_or_vectors() -> None:
    records = [
        _record("map-1", vector=_vector(first_stat=5, first_stat_known=True)),
        _record(
            "map-2", age=20, vector=_vector(second_stat=11, second_stat_known=True)
        ),
    ]
    metadata_before = [
        {key: value for key, value in record.items() if key != "vector"}
        for record in records
    ]
    vectors_before = [record["vector"].copy() for record in records]

    summarize_released_maps(records, cutoff_day=CUTOFF, horizon_days=90)

    assert [
        {key: value for key, value in record.items() if key != "vector"}
        for record in records
    ] == metadata_before
    for record, expected in zip(records, vectors_before, strict=True):
        np.testing.assert_array_equal(record["vector"], expected)
