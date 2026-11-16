"""Regression tests for GOL.GG player-id resolution SQL on PostgreSQL."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import text

from betting_app.core.db import get_session
from betting_app.services.current_roster_service import resolve_golgg_player_id


def _seed_shared_handle(session) -> None:
    for match_id in ("9001", "9002"):
        session.execute(
            text("INSERT INTO golgg_matches(match_id) VALUES (:match_id)"),
            {"match_id": match_id},
        )
        session.execute(
            text("INSERT INTO golgg_games(game_id, match_id) VALUES (:game_id, :match_id)"),
            {"game_id": f"g{match_id}", "match_id": match_id},
        )
    # Same handle on two teams; the newest match belongs to the other team.
    for match_id, player_id, team_name in (
        ("9001", "111", "Expected Team"),
        ("9002", "222", "Other Team"),
    ):
        session.execute(
            text(
                """
                INSERT INTO golgg_game_players(game_id, match_id, team_name, side, role, player_id, player_name)
                VALUES (:game_id, :match_id, :team_name, 'blue', 'mid', :player_id, 'SharedHandle')
                """
            ),
            {
                "game_id": f"g{match_id}",
                "match_id": match_id,
                "team_name": team_name,
                "player_id": player_id,
            },
        )
    session.commit()


def test_resolve_player_id_prefers_expected_team_on_postgres(client: TestClient) -> None:
    session = get_session()
    try:
        _seed_shared_handle(session)

        # Search 1 (exact lowercase name), with and without an expected team.
        assert resolve_golgg_player_id(session, "SharedHandle", expected_team="Expected Team") == ("111", "SharedHandle")
        assert resolve_golgg_player_id(session, "SharedHandle") == ("222", "SharedHandle")

        # Search 2 (alphanumeric-squashed name) reached when the exact lowercase match misses.
        session.execute(text("UPDATE golgg_game_players SET player_name = 'Shared Handle'"))
        session.commit()
        assert resolve_golgg_player_id(session, "SharedHandle", expected_team="Expected Team") == ("111", "Shared Handle")
        assert resolve_golgg_player_id(session, "SharedHandle") == ("222", "Shared Handle")
    finally:
        session.close()
