from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.models.c0_live_query import (
    assemble_player_map_records,
    build_player_history_anchors,
)


def _map(map_id: str, player_id: str, played_day: int, release_day: int, value: float):
    vector = np.zeros(33, dtype=np.float32)
    vector[0] = value
    vector[12] = 1.0
    return {
        "map_id": map_id,
        "series_id": f"series-{map_id}",
        "player_id": player_id,
        "played_day": played_day,
        "release_day": release_day,
        "vector": vector,
    }


def test_player_history_anchors_use_exact_full_90_180_365_day_windows():
    cutoff = date(2026, 10, 20).toordinal()
    records = {
        "p1": [
            _map("365-edge", "p1", cutoff - 365, cutoff - 1, 365.0),
            _map("364-in", "p1", cutoff - 364, cutoff - 1, 364.0),
            _map("180-edge", "p1", cutoff - 180, cutoff - 1, 180.0),
            _map("179-in", "p1", cutoff - 179, cutoff - 1, 179.0),
            _map("90-edge", "p1", cutoff - 90, cutoff - 1, 90.0),
            _map("89-in", "p1", cutoff - 89, cutoff - 1, 89.0),
        ]
    }

    anchors = build_player_history_anchors(
        records,
        roster_a=["p1", "p2", "p3", "p4", "p5"],
        roster_b=["q1", "q2", "q3", "q4", "q5"],
        cutoff_day=cutoff,
    )

    assert anchors["count_365"][0, 0] == 5
    assert anchors["history_365"][0, 0, 0, 0] == pytest.approx(
        np.mean([364, 180, 179, 90, 89])
    )
    assert anchors["count_180"][0, 0] == 3
    assert anchors["history_180"][0, 0, 0, 0] == pytest.approx(np.mean([179, 90, 89]))
    assert anchors["count_90"][0, 0] == 1
    assert anchors["history_90"][0, 0, 0, 0] == pytest.approx(89.0)


def test_player_history_anchors_exclude_played_or_released_on_cutoff():
    cutoff = date(2026, 10, 20).toordinal()
    records = {
        "p1": [
            _map("prior", "p1", cutoff - 1, cutoff - 1, 1.0),
            _map("released-on-cutoff", "p1", cutoff - 1, cutoff, 2.0),
            _map("played-on-cutoff", "p1", cutoff, cutoff - 1, 3.0),
        ]
    }

    anchors = build_player_history_anchors(
        records,
        roster_a=["p1", "p2", "p3", "p4", "p5"],
        roster_b=["q1", "q2", "q3", "q4", "q5"],
        cutoff_day=cutoff,
    )

    for horizon in (90, 180, 365):
        assert anchors[f"count_{horizon}"][0, 0] == 1
        assert anchors[f"history_{horizon}"][0, 0, 0, 0] == pytest.approx(1.0)


def test_player_history_anchors_preserve_sorted_side_and_slot_correspondence():
    cutoff = date(2026, 10, 20).toordinal()
    players_a = ["a3", "a1", "a5", "a2", "a4"]
    players_b = ["b2", "b4", "b1", "b5", "b3"]
    records = {
        player: [
            _map(f"map-{player}", player, cutoff - 1, cutoff - 1, float(index + 1))
        ]
        for index, player in enumerate(players_a + players_b)
    }

    anchors = build_player_history_anchors(
        records,
        roster_a=players_a,
        roster_b=players_b,
        cutoff_day=cutoff,
    )

    actual = anchors["history_365"][:, :, 0, 0]
    np.testing.assert_array_equal(
        actual, np.asarray([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        anchors["count_365"], np.ones((2, 5), dtype=np.uint32)
    )


def test_player_map_record_assembly_rejects_conflicting_source_map_ids():
    first = _map("shared-map", "p1", 700, 701, 1.0)
    conflict = _map("shared-map", "p2", 700, 701, 2.0)

    with pytest.raises(ValueError, match="(?i)(duplicate|conflict).*map"):
        assemble_player_map_records([first, conflict])


def test_player_map_record_assembly_rejects_inconsistent_series_release_identity():
    first = _map("map-a", "p1", 700, 701, 1.0)
    second = _map("map-b", "p2", 700, 702, 2.0)
    first["series_id"] = second["series_id"] = "same-series"

    with pytest.raises(
        ValueError, match="(?i)(series|release).*(conflict|inconsistent)"
    ):
        assemble_player_map_records([first, second])


def test_player_map_record_assembly_indexes_distinct_maps_and_binds_player_identity():
    first = _map("map-a", "p1", 700, 701, 1.0)
    second = _map("map-b", "p1", 702, 703, 2.0)

    by_player, receipt = assemble_player_map_records([first, second])

    assert [record["map_id"] for record in by_player["p1"]] == ["map-a", "map-b"]
    assert receipt["map_record_count"] == 2
    assert receipt["player_count"] == 1


def test_player_map_record_assembly_rejects_malformed_raw_vector():
    record = _map("map-a", "p1", 700, 701, 1.0)
    record["vector"] = np.zeros(32, dtype=np.float32)

    with pytest.raises(ValueError, match="33-vector"):
        assemble_player_map_records([record])


def test_player_history_anchors_reject_duplicate_roster_ids():
    with pytest.raises(ValueError, match="unique"):
        build_player_history_anchors(
            {},
            roster_a=["p1", "p1", "p3", "p4", "p5"],
            roster_b=["q1", "q2", "q3", "q4", "q5"],
            cutoff_day=date(2026, 10, 20),
        )


def test_player_history_anchors_match_masked_pooling_and_log_age_at_cutoff():
    cutoff = date(2026, 10, 20).toordinal()
    first = _map("map-a", "p1", cutoff - 10, cutoff - 1, 100.0)
    second = _map("map-b", "p1", cutoff - 20, cutoff - 1, 0.0)
    first["vector"][25] = second["vector"][25] = 0.0
    second["vector"][0] = 0.0
    second["vector"][12] = 0.0
    anchors = build_player_history_anchors(
        {"p1": [first, second]},
        roster_a=["p1", "p2", "p3", "p4", "p5"],
        roster_b=["q1", "q2", "q3", "q4", "q5"],
        cutoff_day=cutoff,
    )
    history = anchors["history_365"][0, 0, 0]
    assert history[0] == pytest.approx(100.0)
    assert history[12] == 1.0
    assert history[25] == pytest.approx((np.log1p(10) + np.log1p(20)) / 2)


def test_player_history_anchors_reject_overlapping_rosters():
    with pytest.raises(ValueError, match="disjoint"):
        build_player_history_anchors(
            {},
            roster_a=["p1", "p2", "p3", "p4", "p5"],
            roster_b=["p5", "q2", "q3", "q4", "q5"],
            cutoff_day=date(2026, 10, 20),
        )


def test_player_history_anchors_require_disjoint_rosters():
    with pytest.raises(ValueError, match="disjoint"):
        build_player_history_anchors(
            {},
            roster_a=["p1", "p2", "p3", "p4", "p5"],
            roster_b=["p5", "q2", "q3", "q4", "q5"],
            cutoff_day=date(2026, 10, 20),
        )
