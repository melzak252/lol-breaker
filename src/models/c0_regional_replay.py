"""Causal historical replay of family-calibrated regional offsets."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.models.competition_tiers import (
    CompetitionScope,
    CompetitionTier,
    classify_competition,
)
from src.models.cross_region_input_normalizer import (
    compute_dynamic_offsets_from_state,
    resolve_side_domestic_family,
)
from src.ratings.family_calibrated_glicko2 import FamilyCalibratedGlicko2, RatingEvent


_DEFAULT_RELEASE_RELATIVE = Path(
    "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/release_days.json"
)
_PARAMETERS = {
    "initial_rating": 1500.0,
    "initial_rd": 350.0,
    "initial_volatility": 0.06,
    "tau": 0.5,
    "convergence_tolerance": 1e-6,
    "rating_period_days": 30.0,
    "regional_discount": 0.70,
    "initial_tier_deviation": 100.0,
    "initial_family_deviation": 150.0,
    "bridge_process_deviation_per_30_days": 1.0,
    "major_families": ["LCK", "LPL", "LEC", "LCS", "LTA"],
    "family_location_constraint": "mean_total_family_offset_zero",
    "bridge_evidence": "one_majority_outcome_per_cross_family_event",
    "competition_prestige_weight": False,
    "rating_period": "complete_calendar_date",
}


def _complete_roster(
    game: Mapping[str, Any], side: str
) -> tuple[tuple[str, ...], bool]:
    payload = game.get(f"team{side}_players", game.get(f"t{side}_players"))
    if not isinstance(payload, Mapping):
        return (), False
    players = []
    valid = True
    for item in payload.values():
        raw = (
            item.get("player_id", item.get("id")) if isinstance(item, Mapping) else None
        )
        if raw is None or not str(raw).strip():
            valid = False
            continue
        players.append(str(raw).strip())
    unique = set(players)
    return tuple(sorted(unique)), valid and len(players) == 5 and len(unique) == 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_json_array(path: Path):
    """Decode a top-level JSON array incrementally, retaining one source record."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        eof = False
        position = 0
        while True:
            while position >= len(buffer) and not eof:
                chunk = stream.read(1024 * 1024)
                eof = not chunk
                buffer = chunk
                position = 0
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if eof:
                    return
                continue
            if buffer[position] == "[":
                position += 1
                break
            raise ValueError("raw history must be a top-level JSON array")
        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) or eof:
                    break
                chunk = stream.read(1024 * 1024)
                eof = not chunk
                buffer = chunk
                position = 0
            if position >= len(buffer):
                raise ValueError("truncated raw-history JSON array")
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = stream.read(1024 * 1024)
                eof = not chunk
                buffer = buffer[position:] + chunk
                position = 0
                continue
            yield value
            position = end
            if position > 2 * 1024 * 1024:
                buffer = buffer[position:]
                position = 0


def _day(value: Any, label: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc


def _players(game: Mapping[str, Any], side: str) -> tuple[str, ...]:
    key = f"team{side}_players"
    payload = game.get(key, game.get(f"t{side}_players"))
    if not isinstance(payload, Mapping):
        return ()
    result = []
    for item in payload.values():
        if not isinstance(item, Mapping):
            continue
        raw = item.get("player_id", item.get("id"))
        if raw is not None and str(raw).strip():
            player = str(raw).strip()
            if player not in result:
                result.append(player)
    return tuple(sorted(result))


def _natural_id_key(value: Any) -> tuple[int, int | str, str]:
    text = str(value)
    if text.isascii() and text.isdecimal():
        return 0, int(text), text
    return 1, text, text


def _records(
    path: Path, release_days: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    seen: set[str] = set()
    skipped = defaultdict(int)
    for raw in _stream_json_array(path):
        if not isinstance(raw, Mapping):
            skipped["non_object"] += 1
            continue
        identifier = str(raw.get("match_id", "")).strip()
        if not identifier:
            skipped["missing_match_id"] += 1
            continue
        if identifier in seen:
            raise ValueError(f"duplicate raw series ID {identifier}")
        seen.add(identifier)
        games = raw.get("games")
        if not isinstance(games, list) or not games:
            skipped["no_games"] += 1
            continue
        header_a = str(raw.get("team1_id", raw.get("t1_id", "")) or "").strip()
        header_b = str(raw.get("team2_id", raw.get("t2_id", "")) or "").strip()
        header_complete = bool(header_a and header_b and header_a != header_b)
        if header_complete:
            team_a, team_b = header_a, header_b
        else:
            _, first_game = min(
                enumerate(games, 1),
                key=lambda item: (
                    _natural_id_key(
                        item[1].get("game_id", item[1].get("match_id", item[0]))
                    )
                    if isinstance(item[1], Mapping)
                    else (2, "", "")
                ),
            )
            if not isinstance(first_game, Mapping):
                raise ValueError(f"invalid game in series {identifier}")
            team_a = str(
                first_game.get("team1_id", first_game.get("t1_id", "")) or ""
            ).strip()
            team_b = str(
                first_game.get("team2_id", first_game.get("t2_id", "")) or ""
            ).strip()
            if not team_a or not team_b or team_a == team_b:
                skipped["unusable_series"] += 1
                continue
            if any(
                team and team not in (team_a, team_b) for team in (header_a, header_b)
            ):
                raise ValueError(f"series {identifier} changes participating teams")
        tournament = str(raw.get("tournament_name") or "")
        map_rows = []
        scores = []
        played_days = []
        score_complete = True
        missing_map_roster_sides = 0
        for index, game in enumerate(games, 1):
            if not isinstance(game, Mapping):
                raise ValueError(f"invalid game in series {identifier}")
            map_day = _day(
                game.get("date", raw.get("date")), f"series {identifier} map date"
            )
            played_days.append(map_day)
            ta = str(game.get("team1_id", game.get("t1_id", "")) or "").strip()
            tb = str(game.get("team2_id", game.get("t2_id", "")) or "").strip()
            if ta and ta not in (team_a, team_b) or tb and tb not in (team_a, team_b):
                raise ValueError(f"series {identifier} changes participating teams")
            if not ta:
                ta = team_a if not tb or tb == team_b else team_b
            if not tb:
                tb = team_b if ta == team_a else team_a
            if {ta, tb} != {team_a, team_b}:
                raise ValueError(f"series {identifier} changes participating teams")
            if not isinstance(
                game.get("team1_players", game.get("t1_players")), Mapping
            ):
                missing_map_roster_sides += 1
            if not isinstance(
                game.get("team2_players", game.get("t2_players")), Mapping
            ):
                missing_map_roster_sides += 1
            players_t1, valid_t1 = _complete_roster(game, "1")
            players_t2, valid_t2 = _complete_roster(game, "2")
            players_a, players_b = (
                (players_t1, players_t2) if ta == team_a else (players_t2, players_t1)
            )
            valid_a, valid_b = (
                (valid_t1, valid_t2) if ta == team_a else (valid_t2, valid_t1)
            )
            game_id = game.get("game_id", game.get("match_id", index))
            map_rows.append(
                (
                    _natural_id_key(game_id),
                    map_day,
                    players_a,
                    players_b,
                    valid_a,
                    valid_b,
                )
            )
            win_a = game.get("team1_win", game.get("t1_win"))
            win_b = game.get("team2_win", game.get("t2_win"))
            if (
                game.get("draw")
                or win_a not in (0, 1, False, True)
                or win_b not in (0, 1, False, True)
                or bool(win_a) == bool(win_b)
            ):
                score_complete = False
                continue
            scores.append(int(bool(win_a) if ta == team_a else bool(win_b)))
        if not played_days or team_a == team_b or not team_a or not team_b:
            skipped["unusable_series"] += 1
            continue
        played = max(played_days)
        release = (
            max(played, _day(release_days[identifier], f"release day for {identifier}"))
            if identifier in release_days
            else played
        )
        identity = classify_competition(tournament, played)
        if (
            identity.scope is CompetitionScope.UNKNOWN
            or identity.tier is CompetitionTier.UNKNOWN
        ):
            skipped["unknown_competition"] += 1
        first_map = min(map_rows, key=lambda row: row[0]) if map_rows else None
        first_players_a = first_map[2] if first_map is not None else ()
        first_players_b = first_map[3] if first_map is not None else ()
        first_roster_valid = (
            first_map is not None
            and first_map[4]
            and first_map[5]
            and len(first_players_a) == 5
            and len(first_players_b) == 5
            and not (set(first_players_a) & set(first_players_b))
        )
        all_players_a = {player for row in map_rows for player in row[2]}
        all_players_b = {player for row in map_rows for player in row[3]}
        substitute_player_count = len(all_players_a - set(first_players_a)) + len(
            all_players_b - set(first_players_b)
        )
        if not first_roster_valid:
            skipped["first_game_roster_incomplete"] += 1
        skip_reasons = []
        if not score_complete or len(scores) != len(games):
            skipped["incomplete_or_draw_series_scores"] += 1
            skip_reasons.append("incomplete_or_draw_series_scores")
        elif scores and sum(scores) * 2 == len(scores):
            skipped["draw_series"] += 1
            score_complete = False
            skip_reasons.append("draw_series")
        if not first_roster_valid:
            skip_reasons.append("first_game_roster_incomplete")
        skip_reason = ";".join(skip_reasons) or None
        records.append(
            {
                "id": identifier,
                "team_a": team_a,
                "team_b": team_b,
                "participant_pair_source": "series_header"
                if header_complete
                else "first_natural_map",
                "identity": identity,
                "played": played,
                "release": release,
                "players_a": first_players_a,
                "players_b": first_players_b,
                "missing_map_roster_sides": missing_map_roster_sides,
                "substitute_player_count": substitute_player_count,
                "rating_event_skip_reason": skip_reason,
                "scores": tuple(scores) if score_complete else (),
            }
        )
    records.sort(key=lambda r: (r["release"], r["played"], r["id"]))
    coverage = {
        "raw_series_seen": len(seen),
        "usable_series": len(records),
        "first_natural_map_participant_pair_series": sum(
            record["participant_pair_source"] == "first_natural_map"
            for record in records
        ),
        "release_map_bound_usable_series": sum(
            record["id"] in release_days for record in records
        ),
        "played_day_fallback_usable_series": sum(
            record["id"] not in release_days for record in records
        ),
        "first_game_roster_incomplete_series": sum(
            not record["players_a"] or not record["players_b"] for record in records
        ),
        "missing_map_roster_sides": sum(
            record["missing_map_roster_sides"] for record in records
        ),
        "series_with_substitutes": sum(
            record["substitute_player_count"] > 0 for record in records
        ),
        "distinct_substitute_players_excluded_from_rating_events": sum(
            record["substitute_player_count"] for record in records
        ),
        "rating_event_skips": {
            record["id"]: record["rating_event_skip_reason"]
            for record in records
            if record["rating_event_skip_reason"] is not None
        },
        "skipped": dict(sorted(skipped.items())),
    }
    return records, coverage


def _snapshot(
    engine: FamilyCalibratedGlicko2,
    affiliations: Mapping[str, tuple[str, str]],
    player_families: Mapping[str, str],
    cutoff: date,
    max_release: date | None,
) -> str:
    payload = {
        "parameter_law": _PARAMETERS,
        "engine": engine.to_state(),
        "team_affiliations": sorted(
            (key, value[0], value[1]) for key, value in affiliations.items()
        ),
        "player_families": sorted(player_families.items()),
        "cutoff_day": cutoff.isoformat(),
        "max_release_day": max_release.isoformat() if max_release else None,
        "ordering": "effective-release-day; all complete series of a day batch simultaneously; played-day only tie/provenance",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def build_regional_query_states(
    requests: Sequence[Mapping[str, Any]],
    *,
    raw_history_path: str | Path,
    release_map_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay complete released series once and return causal family offsets per request."""
    source = Path(raw_history_path)
    source_hash_before = _sha256(source)
    release_path = Path(release_map_path) if release_map_path is not None else None
    release_hash_before = None
    releases: Mapping[str, Any] = {}
    if release_path is not None:
        release_hash_before = _sha256(release_path)
        loaded = json.loads(release_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("release map must be an ID-to-day object")
        releases = loaded
    else:
        from src.analysis.ev_asof_replay import _resolve_research_root

        default = _resolve_research_root(None) / _DEFAULT_RELEASE_RELATIVE
        if default.is_file():
            release_path = default
            release_hash_before = _sha256(release_path)
            loaded = json.loads(release_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError("default release map must be an ID-to-day object")
            releases = loaded
    source_records, coverage = _records(source, releases)
    if _sha256(source) != source_hash_before:
        raise ValueError("raw history source changed during read")
    if release_path is not None and _sha256(release_path) != release_hash_before:
        raise ValueError("release map source changed during read")
    parsed_requests = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise ValueError("regional requests must be mappings")
        cutoff_value = request.get("cutoff_day", request.get("decision_at"))
        cutoff = _day(cutoff_value, "request cutoff day")
        parsed_requests.append((index, request, cutoff))
    engine = FamilyCalibratedGlicko2(
        tau=_PARAMETERS["tau"],
        convergence_tolerance=_PARAMETERS["convergence_tolerance"],
        rating_period_days=_PARAMETERS["rating_period_days"],
        initial_tier_deviation=_PARAMETERS["initial_tier_deviation"],
        initial_family_deviation=_PARAMETERS["initial_family_deviation"],
        bridge_process_deviation=_PARAMETERS["bridge_process_deviation_per_30_days"],
        regional_discount=_PARAMETERS["regional_discount"],
        major_families=_PARAMETERS["major_families"],
    )
    affiliations: dict[str, tuple[str, str]] = {}
    player_families: dict[str, str] = {}
    rows: list[dict[str, Any] | None] = [None] * len(parsed_requests)
    cursor = 0
    cache: dict[date, dict[str, Any]] = {}
    normalization_values: dict[str, dict[str, float]] = {}
    for cutoff in sorted({item[2] for item in parsed_requests}):
        while (
            cursor < len(source_records) and source_records[cursor]["release"] < cutoff
        ):
            release_day = source_records[cursor]["release"]
            period_start = cursor
            while (
                cursor < len(source_records)
                and source_records[cursor]["release"] == release_day
            ):
                cursor += 1
            period_records = source_records[period_start:cursor]
            domestic_candidates: dict[str, set[tuple[str, str]]] = defaultdict(set)
            for record in period_records:
                identity = record["identity"]
                if (
                    record["rating_event_skip_reason"] is None
                    and identity.scope is CompetitionScope.DOMESTIC
                    and identity.tier is not CompetitionTier.UNKNOWN
                ):
                    affiliation = (identity.family, identity.tier.value)
                    domestic_candidates[record["team_a"]].add(affiliation)
                    domestic_candidates[record["team_b"]].add(affiliation)
            period_affiliations = {
                team: sorted(candidates)[0]
                for team, candidates in domestic_candidates.items()
                if len(candidates) == 1
            }
            events = []
            for record in period_records:
                if record["rating_event_skip_reason"] is not None:
                    continue
                identity = record["identity"]
                if (
                    identity.scope is CompetitionScope.DOMESTIC
                    and identity.tier is not CompetitionTier.UNKNOWN
                ):
                    aff_a = period_affiliations.get(record["team_a"])
                    aff_b = period_affiliations.get(record["team_b"])
                else:
                    aff_a = affiliations.get(record["team_a"])
                    aff_b = affiliations.get(record["team_b"])
                team_a_aff = aff_a or ("unknown", "unknown")
                team_b_aff = aff_b or ("unknown", "unknown")
                events.append(
                    RatingEvent(
                        event_id=record["id"],
                        event_date=release_day,
                        team_a_id=record["team_a"],
                        team_b_id=record["team_b"],
                        players_a=record["players_a"],
                        players_b=record["players_b"],
                        family_a=team_a_aff[0],
                        family_b=team_b_aff[0],
                        tier_a=team_a_aff[1],
                        tier_b=team_b_aff[1],
                        scores=record["scores"],
                    )
                )
            if events:
                engine.process_period(events)
            affiliations.update(period_affiliations)
            player_candidates: dict[str, set[str]] = defaultdict(set)
            for record in period_records:
                identity = record["identity"]
                affiliation = (identity.family, identity.tier.value)
                if (
                    record["rating_event_skip_reason"] is None
                    and identity.scope is CompetitionScope.DOMESTIC
                    and identity.tier is not CompetitionTier.UNKNOWN
                ):
                    for team, players in (
                        (record["team_a"], record["players_a"]),
                        (record["team_b"], record["players_b"]),
                    ):
                        if period_affiliations.get(team) == affiliation:
                            for player in players:
                                player_candidates[player].add(identity.family)
            player_families.update(
                {
                    player: next(iter(families))
                    for player, families in player_candidates.items()
                    if len(families) == 1
                }
            )
        max_release = source_records[cursor - 1]["release"] if cursor else None
        state_hash = _snapshot(
            engine, affiliations, player_families, cutoff, max_release
        )
        normalized_offsets = compute_dynamic_offsets_from_state(engine.to_state())
        normalization_values[cutoff.isoformat()] = dict(
            sorted(normalized_offsets.items())
        )
        cache[cutoff] = {
            "max_release_day": max_release.isoformat() if max_release else None,
            "state_sha256": state_hash,
        }
        team_families = {
            team: affiliation[0] for team, affiliation in affiliations.items()
        }
        family_tiers: dict[str, str] = {}
        for family, tier in sorted(affiliations.values()):
            family_tiers.setdefault(family, tier)
        available_families = set(engine.family_ids)
        available_tiers = set(engine.tier_ids)
        for index, request, request_cutoff in parsed_requests:
            if request_cutoff != cutoff:
                continue
            family_a_name = resolve_side_domestic_family(
                str(request.get("team1_id", "")),
                request.get("roster_a", ()),
                team_families,
                player_families,
            )
            family_b_name = resolve_side_domestic_family(
                str(request.get("team2_id", "")),
                request.get("roster_b", ()),
                team_families,
                player_families,
            )
            family_a = (family_a_name, family_tiers.get(family_a_name, "unknown"))
            family_b = (family_b_name, family_tiers.get(family_b_name, "unknown"))
            support_a = (
                family_a[0] in available_families and family_a[1] in available_tiers
            )
            support_b = (
                family_b[0] in available_families and family_b[1] in available_tiers
            )
            offsets = [
                float(normalized_offsets[affiliation[0]])
                if supported and affiliation[0] in normalized_offsets
                else None
                for affiliation, supported in (
                    (family_a, support_a),
                    (family_b, support_b),
                )
            ]
            rows[index] = {
                "opportunity_id": request.get("opportunity_id"),
                "cutoff_day": cutoff.isoformat(),
                "family_a": family_a[0] if family_a[0] != "unknown" else None,
                "family_b": family_b[0] if family_b[0] != "unknown" else None,
                "offset_a": offsets[0],
                "offset_b": offsets[1],
                **cache[cutoff],
                "status": "ok"
                if all(offset is not None for offset in offsets)
                else "unavailable_family_or_support",
            }

    if _sha256(source) != source_hash_before:
        raise ValueError("raw history source changed during replay")
    if release_path is not None and _sha256(release_path) != release_hash_before:
        raise ValueError("release map source changed during replay")
    constructor_source = inspect.getsource(FamilyCalibratedGlicko2.__init__)
    normalizer_source = inspect.getsource(compute_dynamic_offsets_from_state)
    normalizer_module_sha256 = _sha256(
        Path(__file__).with_name("cross_region_input_normalizer.py")
    )
    receipt = {
        "source_kind": "raw_history_causal_family_calibrated_replay",
        "raw_history_path": str(source),
        "raw_history_sha256": source_hash_before,
        "release_map_path": str(release_path) if release_path else None,
        "release_map_sha256": release_hash_before,
        "source_coverage": coverage,
        "request_count": len(requests),
        "distinct_cutoff_days": len(cache),
        "parameter_law": dict(_PARAMETERS),
        "implementation_binding": {
            "engine_constructor_sha256": hashlib.sha256(
                constructor_source.encode()
            ).hexdigest(),
            "dynamic_normalizer_sha256": hashlib.sha256(
                normalizer_source.encode()
            ).hexdigest(),
            "dynamic_normalizer_module_sha256": normalizer_module_sha256,
            "constructor_parameters": {
                key: _PARAMETERS[key]
                for key in (
                    "tau",
                    "convergence_tolerance",
                    "rating_period_days",
                    "initial_tier_deviation",
                    "initial_family_deviation",
                    "bridge_process_deviation_per_30_days",
                    "regional_discount",
                    "major_families",
                )
            },
            "dynamic_normalizer": "compute_dynamic_offsets_from_state(engine.to_state())",
        },
        "normalization_values_by_cutoff": normalization_values,
        "ordering_assumptions": {
            "rating_period": "complete calendar date grouped by effective whole-series release day",
            "period_date": "effective release day (a conservative public-availability proxy), not played day",
            "same_day": "all series use frozen prior and are updated as one batch, IDs deterministically sorted by engine",
            "series_release": "max(latest map played day, corrected release-map day); strict release < cutoff",
            "substitutes": "all map rosters observed for coverage; rating events use only the earliest natural game-ID roster, with no substitute fallback",
            "domestic_affiliation": "only domestic series strictly released before cutoff; cross-family events use preceding team affiliation",
            "source_difference": "raw reconstructed source chronology; effective release-day periods and earliest natural game-ID roster; DB rebuild uses database match date and first-game roster",
        },
        "max_release_day": max(
            (r["release"].isoformat() for r in source_records), default=None
        ),
    }
    return {"rows": rows, "receipt": receipt}
