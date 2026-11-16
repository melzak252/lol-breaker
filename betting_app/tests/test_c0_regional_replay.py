"""Causal raw-history regional replay contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.models.c0_regional_replay as regional_replay
from src.models.cross_region_input_normalizer import compute_dynamic_offsets_from_state
from src.models.c0_regional_replay import build_regional_query_states


def _series(
    match_id: str,
    day: str,
    tournament: str,
    team_a: str,
    team_b: str,
    players_a: tuple[str, ...],
    players_b: tuple[str, ...],
    *,
    wins_a: int = 2,
    wins_b: int = 0,
) -> dict:
    games = []
    for index in range(wins_a + wins_b):
        a_won = index < wins_a
        games.append(
            {
                "game_id": f"{match_id}-{index + 1}",
                "team1_id": team_a,
                "team2_id": team_b,
                "team1_win": int(a_won),
                "team2_win": int(not a_won),
                "draw": 0,
                "team1_players": {
                    f"p{i}": {"player_id": p} for i, p in enumerate(players_a)
                },
                "team2_players": {
                    f"p{i}": {"player_id": p} for i, p in enumerate(players_b)
                },
            }
        )
    return {
        "match_id": match_id,
        "date": day,
        "tournament_name": tournament,
        "team1_id": team_a,
        "team2_id": team_b,
        "team1_name": team_a,
        "team2_name": team_b,
        "games": games,
    }


def _write_inputs(tmp_path: Path, matches: list[dict], release_days: dict[str, str]):
    history_path = tmp_path / "history.json"
    release_path = tmp_path / "release-days.json"
    history_path.write_text(json.dumps(matches), encoding="utf-8")
    release_path.write_text(json.dumps(release_days), encoding="utf-8")
    return history_path, release_path


def _request(
    opportunity_id: str,
    cutoff: str,
    team_a: str,
    team_b: str,
    roster_a: tuple[str, ...],
    roster_b: tuple[str, ...],
) -> dict:
    return {
        "opportunity_id": opportunity_id,
        "cutoff_day": cutoff,
        "team1_id": team_a,
        "team2_id": team_b,
        "roster_a": list(roster_a),
        "roster_b": list(roster_b),
    }


def test_replay_requires_strict_whole_series_release_before_cutoff(
    tmp_path: Path,
) -> None:
    a = tuple(f"a{i}" for i in range(5))
    b = tuple(f"b{i}" for i in range(5))
    history, releases = _write_inputs(
        tmp_path,
        [
            _series("released", "2025-01-01", "LCK Spring", "a", "b", a, b),
            _series("equal-cutoff", "2025-01-02", "LCK Spring", "a", "b", a, b),
        ],
        {"released": "2025-01-02", "equal-cutoff": "2025-01-03"},
    )
    request = _request("q", "2025-01-03", "a", "b", a, b)

    result = build_regional_query_states(
        [request], raw_history_path=history, release_map_path=releases
    )

    row = result["rows"][0]
    assert row["opportunity_id"] == "q"
    assert row["max_release_day"] == "2025-01-02"
    assert row["status"] == "ok"
    # Complete series released on cutoff is not consumed at the cutoff boundary.
    equal_only_history, equal_only_releases = _write_inputs(
        tmp_path,
        [_series("equal-cutoff", "2025-01-02", "LCK Spring", "a", "b", a, b)],
        {"equal-cutoff": "2025-01-03"},
    )
    equal_only = build_regional_query_states(
        [request],
        raw_history_path=equal_only_history,
        release_map_path=equal_only_releases,
    )["rows"][0]
    assert equal_only["max_release_day"] is None
    assert equal_only["status"] != "ok"


def test_future_result_and_roster_perturbations_do_not_change_prior_state(
    tmp_path: Path,
) -> None:
    a = tuple(f"a{i}" for i in range(5))
    b = tuple(f"b{i}" for i in range(5))
    prior = _series("prior", "2025-01-01", "LCK Spring", "a", "b", a, b)
    future = _series("future", "2025-01-04", "LPL Spring", "c", "d", a, b)
    history, releases = _write_inputs(tmp_path, [prior], {"prior": "2025-01-02"})
    query = [_request("q", "2025-01-03", "a", "b", a, b)]
    baseline = build_regional_query_states(
        query, raw_history_path=history, release_map_path=releases
    )

    future["games"][0]["team1_win"] = 0
    future["games"][0]["team2_win"] = 1
    future["games"][0]["team1_players"] = {"p0": {"player_id": "future-substitute"}}
    history_changed, releases_changed = _write_inputs(
        tmp_path, [future, prior], {"prior": "2025-01-02", "future": "2025-01-05"}
    )
    changed = build_regional_query_states(
        query, raw_history_path=history_changed, release_map_path=releases_changed
    )

    assert changed["rows"] == baseline["rows"]


def test_replay_is_deterministic_under_source_and_request_reordering(
    tmp_path: Path,
) -> None:
    a = tuple(f"a{i}" for i in range(5))
    b = tuple(f"b{i}" for i in range(5))
    c = tuple(f"c{i}" for i in range(5))
    d = tuple(f"d{i}" for i in range(5))
    matches = [
        _series("first", "2025-01-01", "LCK Spring", "a", "b", a, b),
        _series("second", "2025-01-02", "LPL Spring", "c", "d", c, d),
    ]
    releases_map = {"first": "2025-01-02", "second": "2025-01-03"}
    history, releases = _write_inputs(tmp_path, matches, releases_map)
    requests = [
        _request("q1", "2025-01-04", "a", "c", a, c),
        _request("q2", "2025-01-04", "b", "d", b, d),
    ]
    canonical = build_regional_query_states(
        requests, raw_history_path=history, release_map_path=releases
    )
    reordered_history, reordered_releases = _write_inputs(
        tmp_path, list(reversed(matches)), releases_map
    )
    reordered = build_regional_query_states(
        list(reversed(requests)),
        raw_history_path=reordered_history,
        release_map_path=reordered_releases,
    )

    assert {row["opportunity_id"]: row for row in canonical["rows"]} == {
        row["opportunity_id"]: row for row in reordered["rows"]
    }
    assert [row["opportunity_id"] for row in reordered["rows"]] == ["q2", "q1"]


def test_family_requires_prior_domestic_evidence_and_offsets_are_side_correct(
    tmp_path: Path,
) -> None:
    a = tuple(f"a{i}" for i in range(5))
    b = tuple(f"b{i}" for i in range(5))
    prior_domestic = _series("domestic", "2025-01-01", "LCK Spring", "a", "b", a, b)
    future_domestic = _series(
        "future-domestic", "2025-01-04", "LPL Spring", "a", "b", a, b
    )
    international = _series(
        "international",
        "2025-01-02",
        "Worlds",
        "a",
        "c",
        a,
        tuple(f"c{i}" for i in range(5)),
    )
    matches = [prior_domestic, international, future_domestic]
    history, releases = _write_inputs(
        tmp_path,
        matches,
        {
            "domestic": "2025-01-02",
            "international": "2025-01-03",
            "future-domestic": "2025-01-05",
        },
    )
    query = _request("q", "2025-01-04", "a", "b", a, b)
    original_query = json.loads(json.dumps(query))
    forward = build_regional_query_states(
        [query], raw_history_path=history, release_map_path=releases
    )
    reverse_query = _request("q", "2025-01-04", "b", "a", b, a)
    reverse = build_regional_query_states(
        [reverse_query], raw_history_path=history, release_map_path=releases
    )["rows"][0]

    row = forward["rows"][0]
    assert row["family_a"] == "LCK"
    assert row["family_b"] == "LCK"
    assert row["offset_a"] == reverse["offset_b"]
    assert row["offset_b"] == reverse["offset_a"]
    assert query == original_query

    no_prior_history, no_prior_releases = _write_inputs(
        tmp_path, [future_domestic], {"future-domestic": "2025-01-05"}
    )
    missing = build_regional_query_states(
        [query], raw_history_path=no_prior_history, release_map_path=no_prior_releases
    )["rows"][0]
    assert missing["family_a"] in (None, "unknown")
    assert missing["family_b"] in (None, "unknown")
    assert missing["status"] != "ok"
    assert missing["offset_a"] is None
    assert missing["offset_b"] is None
    assert "regional_offsets_state.json" not in str(missing)


def test_replay_accepts_t1_t2_source_records(tmp_path: Path) -> None:
    a = tuple(f"a{i}" for i in range(5))
    b = tuple(f"b{i}" for i in range(5))
    record = _series("legacy", "2025-01-01", "LCK Spring", "a", "b", a, b)
    record["t1_id"], record["t2_id"] = record.pop("team1_id"), record.pop("team2_id")
    record["t1_name"], record["t2_name"] = (
        record.pop("team1_name"),
        record.pop("team2_name"),
    )
    for game in record["games"]:
        game["t1_id"], game["t2_id"] = game.pop("team1_id"), game.pop("team2_id")
        game["t1_win"], game["t2_win"] = game.pop("team1_win"), game.pop("team2_win")
        game["t1_players"], game["t2_players"] = (
            game.pop("team1_players"),
            game.pop("team2_players"),
        )
    history, releases = _write_inputs(tmp_path, [record], {"legacy": "2025-01-02"})

    result = build_regional_query_states(
        [_request("q", "2025-01-03", "a", "b", a, b)],
        raw_history_path=history,
        release_map_path=releases,
    )

    assert result["rows"][0]["status"] == "ok"
    assert result["rows"][0]["family_a"] == result["rows"][0]["family_b"] == "LCK"
    assert result["receipt"]["source_coverage"]["usable_series"] == 1


def test_replay_keeps_first_game_rosters_counts_substitutes_and_routes_release_periods(
    tmp_path: Path,
) -> None:
    a = tuple(f"a{i}" for i in range(5))
    b = tuple(f"b{i}" for i in range(5))
    prior = _series("prior", "2025-01-01", "LCK Spring", "a", "b", a, b)
    # A later-map replacement is coverage-only: the DB first-game roster remains the event roster.
    prior["games"][1]["team1_players"]["p0"]["player_id"] = "later-substitute"
    history, releases = _write_inputs(tmp_path, [prior], {"prior": "2025-01-02"})
    query = [
        _request("at-release", "2025-01-02", "a", "b", a, b),
        _request("after-release", "2025-01-03", "a", "b", a, b),
    ]

    result = build_regional_query_states(
        query, raw_history_path=history, release_map_path=releases
    )

    coverage = result["receipt"]["source_coverage"]
    assert coverage["usable_series"] == 1
    assert coverage["series_with_substitutes"] == 1
    assert coverage["distinct_substitute_players_excluded_from_rating_events"] == 1
    at_release, after_release = result["rows"]
    assert at_release["max_release_day"] is None
    assert at_release["status"] != "ok"
    assert after_release["max_release_day"] == "2025-01-02"
    assert after_release["status"] == "ok"

    # Changing a later-map substitute cannot change first-game rating events or states.
    prior["games"][1]["team1_players"]["p0"]["player_id"] = "a0"
    baseline_history, baseline_releases = _write_inputs(
        tmp_path, [prior], {"prior": "2025-01-02"}
    )
    baseline = build_regional_query_states(
        query, raw_history_path=baseline_history, release_map_path=baseline_releases
    )
    assert result["rows"] == baseline["rows"]
    assert baseline["receipt"]["source_coverage"]["series_with_substitutes"] == 0
    assert (
        baseline["receipt"]["source_coverage"][
            "distinct_substitute_players_excluded_from_rating_events"
        ]
        == 0
    )


def test_offsets_are_the_existing_dynamic_normalizer_output(
    tmp_path: Path, monkeypatch
) -> None:
    a, b = tuple(f"a{i}" for i in range(5)), tuple(f"b{i}" for i in range(5))
    c, d = tuple(f"c{i}" for i in range(5)), tuple(f"d{i}" for i in range(5))
    history, releases = _write_inputs(
        tmp_path,
        [
            _series("lck", "2025-01-01", "LCK Spring", "a", "b", a, b),
            _series("lpl", "2025-01-01", "LPL Spring", "c", "d", c, d),
        ],
        {"lck": "2025-01-02", "lpl": "2025-01-02"},
    )
    captured = {}
    actual_compute = compute_dynamic_offsets_from_state

    def capture(state):
        captured["offsets"] = actual_compute(state)
        return captured["offsets"]

    monkeypatch.setattr(regional_replay, "compute_dynamic_offsets_from_state", capture)
    result = build_regional_query_states(
        [_request("q", "2025-01-03", "a", "c", a, c)],
        raw_history_path=history,
        release_map_path=releases,
    )
    row = result["rows"][0]
    assert row["offset_a"] == captured["offsets"]["LCK"]
    assert row["offset_b"] == captured["offsets"]["LPL"]
    assert result["receipt"]["implementation_binding"]["dynamic_normalizer_sha256"]


def test_new_team_uses_prior_player_family_majority_only(tmp_path: Path) -> None:
    a, b = tuple(f"a{i}" for i in range(5)), tuple(f"b{i}" for i in range(5))
    newcomer = tuple(f"n{i}" for i in range(5))
    history, releases = _write_inputs(
        tmp_path,
        [_series("lck", "2025-01-01", "LCK Spring", "a", "b", a, b)],
        {"lck": "2025-01-02"},
    )
    result = build_regional_query_states(
        [_request("q", "2025-01-03", "new-team", "b", a, b)],
        raw_history_path=history,
        release_map_path=releases,
    )["rows"][0]
    assert result["family_a"] == "LCK"
    assert result["status"] == "ok"

    future_history, future_releases = _write_inputs(
        tmp_path,
        [_series("future", "2025-01-04", "LPL Spring", "new-team", "x", newcomer, b)],
        {"future": "2025-01-05"},
    )
    baseline = build_regional_query_states(
        [_request("q", "2025-01-03", "new-team", "b", newcomer, b)],
        raw_history_path=future_history,
        release_map_path=future_releases,
    )["rows"][0]
    assert baseline["family_a"] is None
    assert baseline["offset_a"] is None


def test_source_change_during_read_is_rejected(tmp_path: Path, monkeypatch) -> None:
    a, b = tuple(f"a{i}" for i in range(5)), tuple(f"b{i}" for i in range(5))
    history, releases = _write_inputs(
        tmp_path,
        [_series("prior", "2025-01-01", "LCK Spring", "a", "b", a, b)],
        {"prior": "2025-01-02"},
    )
    actual_records = regional_replay._records

    def mutate_after_read(path, release_days):
        records = actual_records(path, release_days)
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return records

    monkeypatch.setattr(regional_replay, "_records", mutate_after_read)
    with pytest.raises(ValueError, match="source changed during read"):
        build_regional_query_states(
            [_request("q", "2025-01-03", "a", "b", a, b)],
            raw_history_path=history,
            release_map_path=releases,
        )


@pytest.mark.parametrize("overlap", [False, True])
def test_partial_or_overlapping_earliest_natural_roster_is_rejected(
    tmp_path: Path,
    overlap: bool,
) -> None:
    a, b = tuple(f"a{i}" for i in range(5)), tuple(f"b{i}" for i in range(5))
    series = _series("prior", "2025-01-01", "LCK Spring", "a", "b", a, b)
    series["games"][0]["game_id"] = "map-2"
    series["games"][1]["game_id"] = "map-1"
    if overlap:
        series["games"][1]["team2_players"] = {
            f"p{i}": {"player_id": player} for i, player in enumerate(a)
        }
    else:
        series["games"][1]["team1_players"].pop("p0")
    history, releases = _write_inputs(tmp_path, [series], {"prior": "2025-01-02"})
    row = build_regional_query_states(
        [_request("q", "2025-01-03", "a", "b", a, b)],
        raw_history_path=history,
        release_map_path=releases,
    )["rows"][0]
    assert row["status"] != "ok"
    assert row["offset_a"] is None
    assert row["offset_b"] is None


def test_incomplete_series_header_uses_complete_first_game_orientation(
    tmp_path: Path,
) -> None:
    series = _series(
        "partial",
        "2025-01-01",
        "LCK Spring",
        "445",
        "442",
        tuple(f"a{i}" for i in range(5)),
        tuple(f"b{i}" for i in range(5)),
    )
    series["team2_id"] = None
    series["games"] = series["games"][:1]
    series["games"][0]["team1_id"] = "442"
    series["games"][0]["team2_id"] = "445"
    history, _ = _write_inputs(tmp_path, [series], {})
    records, _ = regional_replay._records(history, {})
    assert (records[0]["team_a"], records[0]["team_b"]) == ("442", "445")
    assert records[0]["scores"] == (1,)


@pytest.mark.parametrize("game_b", ["445", "third-team"])
def test_complete_header_allows_side_swap_but_rejects_new_participant(
    tmp_path: Path,
    game_b: str,
):
    series = _series(
        "conflict",
        "2025-01-01",
        "LCK Spring",
        "445",
        "442",
        tuple(f"a{i}" for i in range(5)),
        tuple(f"b{i}" for i in range(5)),
    )
    series["games"] = series["games"][:1]
    series["games"][0]["team1_id"] = "442"
    series["games"][0]["team2_id"] = game_b
    history, _ = _write_inputs(tmp_path, [series], {})
    if game_b == "third-team":
        with pytest.raises(ValueError, match="changes participating teams"):
            regional_replay._records(history, {})
    else:
        records, _ = regional_replay._records(history, {})
        assert records[0]["scores"] == (0,)
        assert records[0]["players_a"] == tuple(f"b{i}" for i in range(5))
