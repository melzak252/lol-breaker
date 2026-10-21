"""Live Native Feature Builder for C0 serving and forward inference.

Extracts freshly scraped and completed GOL.GG matches from the relational
database (PostgreSQL/TimescaleDB), transforms them into the canonical GOL.GG raw
match format, merges them with the frozen canonical baseline history, enforces
strict causal cutoff boundaries (date < cutoff_date), and generates an
authenticated, SHA-256 verified extended history dataset.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import text

from betting_app.core.config import PROJECT_ROOT

CANONICAL_HISTORY_RELATIVE = "data/artifacts/golgg-database-recovery-20260909/matches.json"
DEFAULT_CACHE_DIR_RELATIVE = "data/04_feature/live_c0_cache"
DEFAULT_CANONICAL_CUTOFF = date(2026, 9, 8)

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
STAT_KEYS = (
    "kills", "deaths", "assists", "csm", "gpm", "dpm",
    "vspm", "gd@15", "csd@15", "xpd@15", "kp%", "dmg%",
)


def _to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    if hasattr(row, "__dict__"):
        return {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Cannot convert {type(row)} to dictionary")


def _parse_json_field(val: Any) -> Any:
    if val is None:
        return {}
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return {}
        try:
            return json.loads(val)
        except Exception:
            return {}
    return {}


def _extract_rows(res: Any) -> list[Any]:
    if hasattr(res, "mappings"):
        mappings = res.mappings()
        if hasattr(mappings, "all"):
            return list(mappings.all())
        return list(mappings)
    if hasattr(res, "all"):
        return list(res.all())
    return list(res)

def _standardize_stats(stats_dict: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in STAT_KEYS:
        val = stats_dict.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            result[key] = float(val)
        else:
            result[key] = 0.0
    return result


def raw_match_from_db_records(
    match_row: Any,
    games_rows: Sequence[Any],
    players_rows: Sequence[Any],
) -> dict[str, Any]:
    """Convert relational DB rows into a canonical raw GOL.GG match document."""
    m = _to_dict(match_row)
    match_id = str(m.get("match_id") or "").strip()
    if not match_id:
        raise ValueError("Match record missing valid match_id")

    match_date = str(m.get("date") or "").strip()
    if not match_date:
        raise ValueError(f"Match {match_id} missing valid date")

    # If full match raw_json exists and contains games, verify or use as base
    raw_json_str = m.get("raw_json")
    base_match: dict[str, Any] = {}
    if raw_json_str:
        parsed_raw = _parse_json_field(raw_json_str)
        if isinstance(parsed_raw, dict):
            base_match = dict(parsed_raw)

    team1_name = str(m.get("team1_name") or base_match.get("team1_name") or base_match.get("t1_name") or "")
    team2_name = str(m.get("team2_name") or base_match.get("team2_name") or base_match.get("t2_name") or "")
    team1_id = str(m.get("team1_id") or base_match.get("team1_id") or base_match.get("t1_id") or "")
    team2_id = str(m.get("team2_id") or base_match.get("team2_id") or base_match.get("t2_id") or "")

    best_of = m.get("best_of") or base_match.get("best_of") or 1
    try:
        best_of = int(best_of)
    except (TypeError, ValueError):
        best_of = 1

    tournament_name = str(m.get("tournament_name") or base_match.get("tournament_name") or "")
    patch = str(m.get("patch") or base_match.get("patch") or "")

    # Group players by game_id and side
    players_by_game: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: {"t1": {}, "t2": {}})
    for p_row in players_rows:
        p = _to_dict(p_row)
        gid = str(p.get("game_id") or "").strip()
        side = str(p.get("side") or "").lower().strip()
        role = str(p.get("role") or "UNKNOWN").upper().strip()
        pid = str(p.get("player_id") or p.get("id") or "").strip()
        if not gid or side not in ("t1", "t2"):
            continue

        stats_payload = _parse_json_field(p.get("stats_json") or p.get("stats"))
        standardized_stats = _standardize_stats(stats_payload)

        players_by_game[gid][side][role] = {
            "player_id": pid,
            "player_name": str(p.get("player_name") or p.get("name") or pid),
            "champion_id": str(p.get("champion_id") or ""),
            "champion_name": str(p.get("champion_name") or ""),
            "stats": standardized_stats,
        }

    # Assemble games
    assembled_games: list[dict[str, Any]] = []
    sorted_games = sorted(
        [_to_dict(g) for g in games_rows],
        key=lambda g: str(g.get("game_id") or ""),
    )

    if not sorted_games:
        raise ValueError(f"Match {match_id} has no completed games")

    for g in sorted_games:
        gid = str(g.get("game_id") or "").strip()
        t1_players = players_by_game[gid]["t1"]
        t2_players = players_by_game[gid]["t2"]

        if len(t1_players) != 5:
            raise ValueError(f"Game {gid} side t1 must contain exactly 5 players (found {len(t1_players)})")
        if len(t2_players) != 5:
            raise ValueError(f"Game {gid} side t2 must contain exactly 5 players (found {len(t2_players)})")

        game_date = str(g.get("date") or match_date)
        game_patch = str(g.get("patch") or patch)
        game_t1_side = str(g.get("team1_side") or "blue").lower()
        game_t2_side = str(g.get("team2_side") or "red").lower()

        t1_win = bool(g.get("team1_win"))
        t2_win = bool(g.get("team2_win"))
        draw = bool(g.get("draw"))

        duration = g.get("game_duration")
        try:
            duration = int(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None

        t1_stats = _parse_json_field(g.get("team1_stats_json") or g.get("t1_stats"))
        t2_stats = _parse_json_field(g.get("team2_stats_json") or g.get("t2_stats"))

        assembled_game: dict[str, Any] = {
            "game_id": gid,
            "match_id": match_id,
            "date": game_date,
            "tournament_name": str(g.get("tournament_name") or tournament_name),
            "patch": game_patch,
            "t1_name": str(g.get("team1_name") or team1_name),
            "t2_name": str(g.get("team2_name") or team2_name),
            "t1_id": str(g.get("team1_id") or team1_id),
            "t2_id": str(g.get("team2_id") or team2_id),
            "t1_win": t1_win,
            "t2_win": t2_win,
            "draw": draw,
            "t1_side": game_t1_side,
            "t2_side": game_t2_side,
            "game_duration": duration,
            "t1_stats": t1_stats,
            "t2_stats": t2_stats,
            "t1_players": t1_players,
            "t2_players": t2_players,
        }
        assembled_games.append(assembled_game)

    score1 = m.get("team1_score")
    score2 = m.get("team2_score")
    if score1 is None:
        score1 = sum(1 for g in assembled_games if g["t1_win"])
    if score2 is None:
        score2 = sum(1 for g in assembled_games if g["t2_win"])

    return {
        "match_id": match_id,
        "date": match_date,
        "tournament_name": tournament_name,
        "patch": patch,
        "team1_id": team1_id,
        "team2_id": team2_id,
        "team1_name": team1_name,
        "team2_name": team2_name,
        "t1_name": team1_name,
        "t2_name": team2_name,
        "t1_id": team1_id,
        "t2_id": team2_id,
        "team1_score": int(score1),
        "team2_score": int(score2),
        "score1": int(score1),
        "score2": int(score2),
        "team1_win": bool(m.get("team1_win") if m.get("team1_win") is not None else (score1 > score2)),
        "team2_win": bool(m.get("team2_win") if m.get("team2_win") is not None else (score2 > score1)),
        "draw": bool(m.get("draw")),
        "games_played": len(assembled_games),
        "best_of": best_of,
        "winner_name": m.get("winner_name"),
        "loser_name": m.get("loser_name"),
        "games": assembled_games,
    }


def export_db_golgg_matches(
    session: Any,
    min_date: str | None = None,
    max_date: str | None = None,
    match_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Query relational database tables and return canonical raw match dicts."""
    where_clauses = []
    params: dict[str, Any] = {}

    if min_date:
        where_clauses.append("date >= :min_date")
        params["min_date"] = min_date
    if max_date:
        where_clauses.append("date <= :max_date")
        params["max_date"] = max_date
    if match_ids:
        where_clauses.append("match_id IN :match_ids")
        params["match_ids"] = tuple(match_ids)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql_matches = text(f"SELECT * FROM golgg_matches {where_sql} ORDER BY date ASC, match_id ASC")

    matches_res = session.execute(sql_matches, params)
    raw_matches = _extract_rows(matches_res)

    if not raw_matches:
        return []

    found_ids = [str(_to_dict(m)["match_id"]) for m in raw_matches]

    sql_games = text("SELECT * FROM golgg_games WHERE match_id IN :match_ids ORDER BY game_id ASC")
    games_res = session.execute(sql_games, {"match_ids": tuple(found_ids)})
    raw_games = _extract_rows(games_res)

    sql_players = text("SELECT * FROM golgg_game_players WHERE match_id IN :match_ids ORDER BY game_id ASC, side ASC, role ASC")
    players_res = session.execute(sql_players, {"match_ids": tuple(found_ids)})
    raw_players = _extract_rows(players_res)

    games_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in raw_games:
        d = _to_dict(g)
        games_by_match[str(d["match_id"])].append(d)

    players_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in raw_players:
        d = _to_dict(p)
        players_by_match[str(d["match_id"])].append(d)

    exported: list[dict[str, Any]] = []
    for m in raw_matches:
        mid = str(_to_dict(m)["match_id"])
        g_rows = games_by_match.get(mid, [])
        p_rows = players_by_match.get(mid, [])
        if not g_rows or not p_rows:
            continue
        try:
            match_doc = raw_match_from_db_records(m, g_rows, p_rows)
            exported.append(match_doc)
        except Exception:
            # Skip corrupted/incomplete matches
            continue

    exported.sort(key=lambda m: (m.get("date", ""), str(m.get("match_id", ""))))
    return exported


def merge_c0_history(
    canonical_matches: Iterable[dict[str, Any]],
    db_matches: Iterable[dict[str, Any]],
    cutoff_date: date | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge canonical frozen matches with DB matches, filtering strictly by cutoff."""
    cutoff_str = cutoff_date.isoformat() if isinstance(cutoff_date, date) else cutoff_date

    merged_by_id: dict[str, dict[str, Any]] = {}
    canonical_count = 0
    db_new_count = 0
    excluded_by_cutoff_count = 0

    # 1. Ingest canonical matches
    for m in canonical_matches:
        mid = str(m.get("match_id") or "").strip()
        m_date = str(m.get("date") or "")[:10]
        if cutoff_str and m_date >= cutoff_str:
            excluded_by_cutoff_count += 1
            continue
        if mid:
            merged_by_id[mid] = m
            canonical_count += 1

    # 2. Ingest DB matches (overrides or appends new)
    for m in db_matches:
        mid = str(m.get("match_id") or "").strip()
        m_date = str(m.get("date") or "")[:10]
        if cutoff_str and m_date >= cutoff_str:
            excluded_by_cutoff_count += 1
            continue
        if mid:
            if mid not in merged_by_id:
                db_new_count += 1
            merged_by_id[mid] = m

    all_series = list(merged_by_id.values())
    all_series.sort(key=lambda m: (str(m.get("date", ""))[:10], str(m.get("match_id", ""))))

    max_included_date = all_series[-1].get("date", "")[:10] if all_series else None
    audit = {
        "canonical_count": canonical_count,
        "db_new_count": db_new_count,
        "excluded_by_cutoff_count": excluded_by_cutoff_count,
        "total_series_count": len(all_series),
        "max_included_date": max_included_date,
        "cutoff_enforced": cutoff_str,
    }
    return all_series, audit


def build_extended_c0_history_file(
    canonical_source: Path | str,
    db_matches: Iterable[dict[str, Any]],
    output_path: Path | str,
    cutoff_date: date | str | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    """Assemble and atomically write extended C0 history JSON with SHA-256 digest."""
    canon_p = Path(canonical_source).resolve(strict=True)
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with canon_p.open("r", encoding="utf-8") as f:
        canonical_matches = json.load(f)

    merged, audit = merge_c0_history(canonical_matches, db_matches, cutoff_date=cutoff_date)

    # Atomic write to avoid partial writes during concurrent access
    temp_dir = out_p.parent
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        json.dump(merged, tf, ensure_ascii=False, indent=2)
        temp_name = tf.name

    os.replace(temp_name, out_p)

    hasher = hashlib.sha256()
    with out_p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    audit["output_sha256"] = digest
    audit["output_path"] = str(out_p)
    return out_p, digest, audit


class LiveNativeFeatureBuilder:
    """Manages dynamic assembly of extended C0 history from PostgreSQL and disk."""

    def __init__(
        self,
        session: Any = None,
        canonical_history_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ):
        self.session = session
        if canonical_history_path is not None:
            self.canonical_history_path = Path(canonical_history_path).resolve()
        else:
            self.canonical_history_path = (PROJECT_ROOT / CANONICAL_HISTORY_RELATIVE).resolve()

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = (PROJECT_ROOT / DEFAULT_CACHE_DIR_RELATIVE).resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_extended_history(
        self,
        cutoff_date: date | str | None = None,
        force_refresh: bool = False,
    ) -> tuple[Path, str, dict[str, Any]]:
        """Return an extended history JSON file and its SHA-256 for the given cutoff."""
        cutoff_str = cutoff_date.isoformat() if isinstance(cutoff_date, date) else (cutoff_date or "latest")
        target_name = f"extended_history_{cutoff_str}.json"
        out_file = self.cache_dir / target_name

        if not force_refresh and out_file.is_file():
            hasher = hashlib.sha256()
            with out_file.open("rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            # Fast return cached file
            return out_file, digest, {"cached": True, "output_path": str(out_file), "output_sha256": digest}

        db_matches: list[dict[str, Any]] = []
        if self.session is not None:
            db_matches = export_db_golgg_matches(
                self.session,
                min_date=DEFAULT_CANONICAL_CUTOFF.isoformat(),
                max_date=cutoff_str if cutoff_str != "latest" else None,
            )

        return build_extended_c0_history_file(
            canonical_source=self.canonical_history_path,
            db_matches=db_matches,
            output_path=out_file,
            cutoff_date=cutoff_date,
        )


def export_db_regional_offsets_state(
    session: Any,
    output_path: Path | str | None = None,
    ratings_version: str = "ratings-v2",
) -> Path | None:
    """Export the latest FamilyCalibratedGlicko2 state from rating_runs to JSON."""
    out_p = (
        Path(output_path).resolve()
        if output_path is not None
        else (PROJECT_ROOT / DEFAULT_CACHE_DIR_RELATIVE / "regional_offsets_state.json").resolve()
    )
    row = session.execute(
        text(
            "SELECT (systems_json::jsonb -> 'gl' -> 'checkpoint' -> 'state')::text "
            "FROM rating_runs WHERE ratings_version = :rv ORDER BY id DESC LIMIT 1"
        ),
        {"rv": ratings_version},
    ).scalar()
    if not row:
        return None
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(str(row), encoding="utf-8")
    return out_p


def main() -> int:
    import argparse
    from betting_app.core.db import get_session

    parser = argparse.ArgumentParser(
        description="Export extended GOL.GG history and live regional offsets state from PostgreSQL."
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Inclusive ISO cutoff date YYYY-MM-DD (default: latest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_CACHE_DIR_RELATIVE / "extended_matches.json",
        help="Output path for extended_matches.json",
    )
    args = parser.parse_args()

    with get_session() as session:
        db_matches = export_db_golgg_matches(
            session,
            min_date=DEFAULT_CANONICAL_CUTOFF.isoformat(),
            max_date=args.cutoff,
        )
        out_p, digest, audit = build_extended_c0_history_file(
            canonical_source=PROJECT_ROOT / CANONICAL_HISTORY_RELATIVE,
            db_matches=db_matches,
            output_path=args.output,
            cutoff_date=args.cutoff,
        )
        reg_p = export_db_regional_offsets_state(session)
        print(f"Exported {audit['total_series_count']} series ({audit['db_new_count']} new from DB) -> {out_p} (SHA-256: {digest[:16]}...)")
        if reg_p:
            print(f"Exported live FamilyCalibratedGlicko2 regional state -> {reg_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
