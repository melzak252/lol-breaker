"""Reconstruct a causal research history extension from a read-only production export."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from betting_app.services.live_native_feature_builder import (
    DEFAULT_CANONICAL_CUTOFF,
    effective_release_day,
    merge_c0_history,
    raw_match_from_db_records,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_export_history(raw_root: Path, canonical_path: Path, output_path: Path) -> dict[str, Any]:
    """Keep frozen canonical records intact; append complete new source IDs with receipts.

    All exported rows remain in raw storage. Unusable new series get explicit absences,
    rather than partial-map substitution or silent loss. Queries enforce their own earlier
    whole-series release boundary; the assembled source is not an issuance-time claim.
    """
    if output_path.exists() or output_path.with_suffix(".receipt.json").exists():
        raise FileExistsError("History output already exists; use a new namespace")
    canonical = json.loads(canonical_path.read_text())
    if not isinstance(canonical, list):
        raise ValueError("Canonical history must contain a series list")
    ids = [str(row["match_id"]) for row in canonical]
    if len(set(ids)) != len(ids):
        raise ValueError("Frozen canonical history contains duplicate series IDs")
    canonical_ids = set(ids)
    new: dict[str, dict[str, Any]] = {}
    source_count = 0
    with (raw_root / "golgg_matches.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            source_count += 1
            mid = str(row["match_id"])
            if mid not in canonical_ids:
                if mid in new:
                    raise ValueError(f"Duplicate exported series ID: {mid}")
                new[mid] = row
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    players: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, target in (("golgg_games.jsonl", games), ("golgg_game_players.jsonl", players)):
        with (raw_root / name).open() as stream:
            for line in stream:
                row = json.loads(line)
                mid = str(row["match_id"])
                if mid in new:
                    target[mid].append(row)
    appended, absences = [], []
    for mid, row in sorted(new.items()):
        if str(row.get("date", ""))[:10] < DEFAULT_CANONICAL_CUTOFF.isoformat():
            absences.append({
                "match_id": mid,
                "reason": "outside_existing_serving_extension_frontier",
            })
            continue
        expected = row.get("games_played")
        actual = len(games[mid])
        reason = None
        if not actual or not players[mid]:
            reason = "missing_games_or_players"
        elif expected is not None and actual != int(expected):
            reason = "incomplete_or_conflicting_map_count"
        elif int(row.get("best_of") or 0) in (1, 3, 5) and not (row.get("team1_win") or row.get("team2_win")):
            reason = "unfinished_or_conflicting_series_result"
        if reason is not None:
            absences.append({"match_id": mid, "reason": reason})
            continue
        try:
            document = raw_match_from_db_records(row, games[mid], players[mid])
        except (ValueError, TypeError, KeyError) as error:
            absences.append({"match_id": mid, "reason": "invalid_source_series", "detail": str(error)})
            continue
        appended.append(document)
    history, audit = merge_c0_history(canonical, appended)
    if len(history) != len(canonical) + len(appended):
        raise ValueError("History union changed canonical membership")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(history, stream, separators=(",", ":"), ensure_ascii=False)
    receipt = {
        "schema": "production_probability_history_v1",
        "scope": "HISTORICAL_DEVELOPMENT_ONLY",
        "canonical_preserved": True,
        "canonical_count": len(canonical),
        "exported_series_count": source_count,
        "new_exported_ids": len(new),
        "appended_series_count": len(appended),
        "unavailable_new_series": absences,
        "max_effective_release_day": max(map(effective_release_day, history)),
        "source_hashes": {str(path): _sha256(path) for path in (
            canonical_path, raw_root / "golgg_matches.jsonl", raw_root / "golgg_games.jsonl",
            raw_root / "golgg_game_players.jsonl",
        )},
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "merge_audit": audit,
        "clock_policy": "Per-query whole-series release day strictly before UTC decision day; daily proxy, not authenticated publication",
    }
    with output_path.with_suffix(".receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt
