"""Tests for Live Native Feature Builder.

Validates extraction of live GOL.GG matches from DB models, conversion to
canonical raw match dictionaries, chronological merging with frozen C0 history,
strict causal cutoff filtering, and SHA-256 evidence hashing.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import pytest

from betting_app.services.live_native_feature_builder import (
    raw_match_from_db_records,
    export_db_golgg_matches,
    merge_c0_history,
    build_extended_c0_history_file,
    LiveNativeFeatureBuilder,
)


def _make_game_players(game_id: str, match_id: str) -> list[dict]:
    roles = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
    players = []
    # Side t1
    for i, role in enumerate(roles, start=1):
        players.append({
            "game_id": game_id,
            "match_id": match_id,
            "team_id": "100",
            "team_name": "Team Alpha",
            "side": "t1",
            "role": role,
            "player_id": f"p_a_{i}",
            "player_name": f"PlayerA{i}",
            "champion_id": f"10{i}",
            "champion_name": f"ChampA{i}",
            "stats_json": json.dumps({
                "kills": 3, "deaths": 1, "assists": 5, "csm": 9.2, "gpm": 450.0,
                "dpm": 600.0, "vspm": 1.5, "gd@15": 200.0, "csd@15": 10.0,
                "xpd@15": 150.0, "kp%": 65.0, "dmg%": 25.0,
            }),
            "raw_json": "{}",
        })
    # Side t2
    for i, role in enumerate(roles, start=1):
        players.append({
            "game_id": game_id,
            "match_id": match_id,
            "team_id": "200",
            "team_name": "Team Beta",
            "side": "t2",
            "role": role,
            "player_id": f"p_b_{i}",
            "player_name": f"PlayerB{i}",
            "champion_id": f"20{i}",
            "champion_name": f"ChampB{i}",
            "stats_json": json.dumps({
                "kills": 1, "deaths": 3, "assists": 2, "csm": 8.5, "gpm": 380.0,
                "dpm": 450.0, "vspm": 1.2, "gd@15": -200.0, "csd@15": -10.0,
                "xpd@15": -150.0, "kp%": 50.0, "dmg%": 20.0,
            }),
            "raw_json": "{}",
        })
    return players


def _make_sample_match_records(match_id: str = "99001", match_date: str = "2026-09-15") -> tuple[dict, list[dict], list[dict]]:
    match_row = {
        "match_id": match_id,
        "date": match_date,
        "tournament_name": "LCK Regional Finals 2026",
        "patch": "14.18",
        "team1_id": "100",
        "team2_id": "200",
        "team1_name": "Team Alpha",
        "team2_name": "Team Beta",
        "team1_score": 2,
        "team2_score": 0,
        "team1_win": 1,
        "team2_win": 0,
        "draw": 0,
        "games_played": 2,
        "best_of": 3,
        "winner_name": "Team Alpha",
        "loser_name": "Team Beta",
        "source_link": "https://gol.gg/game/stats/99001/",
        "raw_json": None,
    }
    games = [
        {
            "game_id": f"{match_id}_1",
            "match_id": match_id,
            "date": match_date,
            "tournament_name": "LCK Regional Finals 2026",
            "patch": "14.18",
            "team1_id": "100",
            "team2_id": "200",
            "team1_name": "Team Alpha",
            "team2_name": "Team Beta",
            "team1_win": 1,
            "team2_win": 0,
            "draw": 0,
            "team1_side": "blue",
            "team2_side": "red",
            "game_duration": 1850,
            "team1_stats_json": json.dumps({"kills": 15, "towers": 9, "dragons": 3, "barons": 1}),
            "team2_stats_json": json.dumps({"kills": 5, "towers": 2, "dragons": 1, "barons": 0}),
            "raw_json": "{}",
        },
        {
            "game_id": f"{match_id}_2",
            "match_id": match_id,
            "date": match_date,
            "tournament_name": "LCK Regional Finals 2026",
            "patch": "14.18",
            "team1_id": "100",
            "team2_id": "200",
            "team1_name": "Team Alpha",
            "team2_name": "Team Beta",
            "team1_win": 1,
            "team2_win": 0,
            "draw": 0,
            "team1_side": "red",
            "team2_side": "blue",
            "game_duration": 1920,
            "team1_stats_json": json.dumps({"kills": 18, "towers": 10, "dragons": 4, "barons": 1}),
            "team2_stats_json": json.dumps({"kills": 7, "towers": 3, "dragons": 0, "barons": 0}),
            "raw_json": "{}",
        },
    ]
    players = []
    for g in games:
        players.extend(_make_game_players(g["game_id"], match_id))
    return match_row, games, players


class DummySession:
    def __init__(self, match_rows: list[dict], game_rows: list[dict], player_rows: list[dict]):
        self._matches = match_rows
        self._games = game_rows
        self._players = player_rows

    def execute(self, stmt, params=None):
        params = params or {}
        sql = str(getattr(stmt, "text", stmt)).strip().lower()
        if "from golgg_matches" in sql:
            min_date = params.get("min_date")
            max_date = params.get("max_date")
            res = []
            for m in self._matches:
                d = m.get("date") or ""
                if min_date and d < min_date:
                    continue
                if max_date and d > max_date:
                    continue
                res.append(dict(m))
            return DummyResult(res)
        elif "from golgg_games" in sql:
            mids = set(params.get("match_ids", []))
            res = [dict(g) for g in self._games if not mids or g.get("match_id") in mids]
            return DummyResult(res)
        elif "from golgg_game_players" in sql:
            mids = set(params.get("match_ids", []))
            res = [dict(p) for p in self._players if not mids or p.get("match_id") in mids]
            return DummyResult(res)
        return DummyResult([])


class DummyResult:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def mappings(self):
        return self._rows

    def all(self):
        return self._rows


def test_raw_match_from_db_records_assembles_canonical_structure():
    m_row, g_rows, p_rows = _make_sample_match_records()
    match_dict = raw_match_from_db_records(m_row, g_rows, p_rows)

    assert match_dict["match_id"] == "99001"
    assert match_dict["date"] == "2026-09-15"
    assert match_dict["team1_name"] == "Team Alpha"
    assert match_dict["team2_name"] == "Team Beta"
    assert match_dict["best_of"] == 3
    assert len(match_dict["games"]) == 2

    game1 = match_dict["games"][0]
    assert game1["game_id"] == "99001_1"
    assert game1["t1_name"] == "Team Alpha"
    assert game1["t2_name"] == "Team Beta"
    assert game1["t1_side"] == "blue"
    assert game1["t1_win"] is True
    assert len(game1["t1_players"]) == 5
    assert len(game1["t2_players"]) == 5

    # Check top laner on t1
    top_p = game1["t1_players"]["TOP"]
    assert top_p["player_id"] == "p_a_1"
    assert top_p["player_name"] == "PlayerA1"
    assert top_p["champion_id"] == "101"
    assert top_p["stats"]["kills"] == 3
    assert top_p["stats"]["csm"] == 9.2


def test_raw_match_rejects_incomplete_rosters():
    m_row, g_rows, p_rows = _make_sample_match_records()
    # Remove one player from t1 in game 1
    p_rows_incomplete = [p for p in p_rows if not (p["game_id"] == "99001_1" and p["side"] == "t1" and p["role"] == "SUPPORT")]
    with pytest.raises(ValueError, match="must contain exactly 5 players"):
        raw_match_from_db_records(m_row, g_rows, p_rows_incomplete)


def test_export_db_golgg_matches_queries_and_filters():
    m1, g1, p1 = _make_sample_match_records("99001", "2026-09-15")
    m2, g2, p2 = _make_sample_match_records("99002", "2026-09-20")
    m3, g3, p3 = _make_sample_match_records("99003", "2026-09-05")

    session = DummySession([m1, m2, m3], g1 + g2 + g3, p1 + p2 + p3)

    # Filter >= 2026-09-10
    exported = export_db_golgg_matches(session, min_date="2026-09-10")
    assert len(exported) == 2
    assert [m["match_id"] for m in exported] == ["99001", "99002"]


def test_merge_c0_history_deduplicates_and_enforces_causal_cutoff():
    canonical = [
        {"match_id": "1001", "date": "2026-09-01", "games": []},
        {"match_id": "1002", "date": "2026-09-05", "games": []},
    ]
    db_matches = [
        {"match_id": "1002", "date": "2026-09-05", "games": ["updated"]},  # duplicate, should override or preserve
        {"match_id": "2001", "date": "2026-09-15", "games": []},
        {"match_id": "2002", "date": "2026-10-01", "games": []},
    ]

    # Cutoff at 2026-09-20: 2002 must be excluded (strictly date < cutoff)
    merged, audit = merge_c0_history(canonical, db_matches, cutoff_date=date(2026, 9, 20))

    assert len(merged) == 3
    ids = [m["match_id"] for m in merged]
    assert ids == ["1001", "1002", "2001"]
    assert audit["canonical_count"] == 2
    assert audit["db_new_count"] == 1
    assert audit["excluded_by_cutoff_count"] == 1
    assert audit["max_included_date"] == "2026-09-15"


def test_build_extended_c0_history_file_writes_atomic_json_with_sha256(tmp_path):
    canonical_file = tmp_path / "canonical_matches.json"
    canonical_data = [
        {"match_id": "1001", "date": "2026-09-01", "games": []},
    ]
    canonical_file.write_text(json.dumps(canonical_data), encoding="utf-8")

    db_matches = [
        {"match_id": "2001", "date": "2026-09-15", "games": []},
    ]

    out_file = tmp_path / "extended_history.json"
    path, digest, audit = build_extended_c0_history_file(
        canonical_source=canonical_file,
        db_matches=db_matches,
        output_path=out_file,
        cutoff_date=date(2026, 9, 20),
    )

    assert path == out_file
    assert out_file.is_file()
    assert len(digest) == 64
    assert digest == hashlib.sha256(out_file.read_bytes()).hexdigest()

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(loaded) == 2
    assert [m["match_id"] for m in loaded] == ["1001", "2001"]


def test_live_native_feature_builder_end_to_end_orchestration(tmp_path):
    m1, g1, p1 = _make_sample_match_records("99001", "2026-09-15")
    session = DummySession([m1], g1, p1)

    canonical_file = tmp_path / "canonical.json"
    canonical_file.write_text(json.dumps([{"match_id": "1001", "date": "2026-09-01", "games": []}]), encoding="utf-8")

    cache_dir = tmp_path / "cache"
    builder = LiveNativeFeatureBuilder(
        session=session,
        canonical_history_path=canonical_file,
        cache_dir=cache_dir,
    )

    path, digest, audit = builder.prepare_extended_history(cutoff_date=date(2026, 9, 20))

    assert path.is_file()
    assert audit["total_series_count"] == 2
    assert audit["db_new_count"] == 1
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    # Second call with same cutoff should return cached file idempotently
    path2, digest2, audit2 = builder.prepare_extended_history(cutoff_date=date(2026, 9, 20))
    assert path2 == path
    assert digest2 == digest


def test_live_native_feature_builder_force_refresh_invalidates_cache(tmp_path):
    m1, g1, p1 = _make_sample_match_records("99001", "2026-09-15")
    session = DummySession([m1], g1, p1)
    canonical_file = tmp_path / "canonical.json"
    canonical_file.write_text(json.dumps([{"match_id": "1001", "date": "2026-09-01", "games": []}]), encoding="utf-8")
    builder = LiveNativeFeatureBuilder(session=session, canonical_history_path=canonical_file, cache_dir=tmp_path / "cache")

    path1, digest1, audit1 = builder.prepare_extended_history(cutoff_date=date(2026, 9, 20))
    # Add second match to session
    m2, g2, p2 = _make_sample_match_records("99002", "2026-09-16")
    session._matches.append(m2)
    session._games.extend(g2)
    session._players.extend(p2)

    # Without force_refresh, returns cached
    path_cached, digest_cached, audit_cached = builder.prepare_extended_history(cutoff_date=date(2026, 9, 20))
    assert audit_cached.get("cached") is True
    assert digest_cached == digest1

    # With force_refresh, writes new extended file
    path2, digest2, audit2 = builder.prepare_extended_history(cutoff_date=date(2026, 9, 20), force_refresh=True)
    assert audit2.get("total_series_count") == 3
    assert audit2.get("db_new_count") == 2
    assert digest2 != digest1


def test_live_native_feature_builder_handles_empty_db(tmp_path):
    session = DummySession([], [], [])
    canonical_file = tmp_path / "canonical.json"
    canonical_file.write_text(json.dumps([{"match_id": "1001", "date": "2026-09-01", "games": []}]), encoding="utf-8")
    builder = LiveNativeFeatureBuilder(session=session, canonical_history_path=canonical_file, cache_dir=tmp_path / "cache")
    path, digest, audit = builder.prepare_extended_history(cutoff_date=date(2026, 9, 20))

    assert audit["total_series_count"] == 1
    assert audit["db_new_count"] == 0


def test_raw_match_corrupt_stats_falls_back_to_zeros():
    m_row, g_rows, p_rows = _make_sample_match_records()
    # Corrupt stats_json on one player
    for p in p_rows:
        if p["role"] == "MID" and p["side"] == "t1":
            p["stats_json"] = "not a valid json"
    match_dict = raw_match_from_db_records(m_row, g_rows, p_rows)
    mid_p = match_dict["games"][0]["t1_players"]["MID"]
    # Should fall back to 0.0 for all stats
    assert mid_p["stats"]["kills"] == 0.0
    assert mid_p["stats"]["dpm"] == 0.0


def test_merge_c0_history_exact_cutoff_boundary():
    canonical = [{"match_id": "1001", "date": "2026-09-01", "games": []}]
    db_matches = [
        {"match_id": "2001", "date": "2026-09-19", "games": []},
        {"match_id": "2002", "date": "2026-09-20", "games": []},  # Exactly on cutoff: MUST be excluded
        {"match_id": "2003", "date": "2026-09-21", "games": []},  # After cutoff: MUST be excluded
    ]
    merged, audit = merge_c0_history(canonical, db_matches, cutoff_date=date(2026, 9, 20))
    assert len(merged) == 2
    assert [m["match_id"] for m in merged] == ["1001", "2001"]
    assert audit["excluded_by_cutoff_count"] == 2


def test_live_extended_history_resolves_in_native_c0_adapter(tmp_path, monkeypatch):
    from betting_app.core.models import native_c0

    canonical_file = tmp_path / "canonical.json"
    canonical_file.write_text(json.dumps([{"match_id": "1001", "date": "2026-09-01", "games": []}]), encoding="utf-8")

    db_matches = [{"match_id": "2001", "date": "2026-09-15", "games": []}]
    out_file = tmp_path / "extended.json"
    extended_path, digest, _ = build_extended_c0_history_file(
        canonical_source=canonical_file,
        db_matches=db_matches,
        output_path=out_file,
    )

    config = {"raw_history_path": native_c0._CANONICAL_RAW_HISTORY_RELATIVE}
    raw_path, source_kind, is_canonical = native_c0._resolve_raw_history_source(config, override=str(extended_path))

    assert raw_path == extended_path.resolve()
    assert source_kind == "explicit_raw_history_override"
    assert is_canonical is False


def test_predict_matchup_integration_with_extended_history(tmp_path):
    from scripts.predict_matchup import load_raw_matches, resolve_team

    m1, g1, p1 = _make_sample_match_records("99001", "2026-09-15")
    m_doc = raw_match_from_db_records(m1, g1, p1)

    extended_file = tmp_path / "extended.json"
    extended_file.write_text(json.dumps([m_doc]), encoding="utf-8")

    raw_matches = load_raw_matches(extended_file)
    assert len(raw_matches) == 1

    t1_info = resolve_team("Team Alpha", raw_matches)
    assert t1_info["name"] == "Team Alpha"
    assert t1_info["id"] == "100"
    assert t1_info["last_played_date"] == "2026-09-15"
    assert len(t1_info["player_ids"]) == 5
