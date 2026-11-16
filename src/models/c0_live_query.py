"""Research-only full-history causal anchors and retained Worlds query batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_WIDTH = 33
HORIZONS = (90, 180, 365)
SNAPSHOT_DEFAULT = (
    PROJECT_ROOT / "data/04_feature/live_c0_cache/worlds_2026_swiss_r1_snapshots.pkl"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _day(value: Any) -> int:
    if isinstance(value, date):
        return value.toordinal()
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    text = str(value)
    try:
        return date.fromisoformat(text[:10]).toordinal()
    except ValueError as error:
        raise ValueError(f"Invalid player-map calendar day: {value!r}") from error


def assemble_player_map_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Validate source identities and index immutable raw vectors by player."""
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    map_ids: dict[str, tuple[str, str, int, int]] = {}
    series_release: dict[str, int] = {}
    for source in records:
        record = dict(source)
        map_id, series_id, player_id = (
            str(record.get(key) or "") for key in ("map_id", "series_id", "player_id")
        )
        if not map_id or not series_id or not player_id:
            raise ValueError(
                "Player-map record is missing map, series, or player identity"
            )
        played, release = (
            _day(record.get("played_day")),
            _day(record.get("release_day")),
        )
        vector = np.asarray(record.get("vector"), dtype=np.float32)
        if vector.shape != (HISTORY_WIDTH,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid raw 33-vector for source map {map_id}")
        identity = (series_id, player_id, played, release)
        prior = map_ids.get(map_id)
        if prior is not None:
            raise ValueError(
                f"Duplicate/conflicting source map ID {map_id}: {prior!r} vs {identity!r}"
            )
        map_ids[map_id] = identity
        previous_release = series_release.setdefault(series_id, release)
        if previous_release != release:
            raise ValueError(
                f"Series release conflict for {series_id}: inconsistent effective release day"
            )
        record.update(
            map_id=map_id,
            series_id=series_id,
            player_id=player_id,
            played_day=played,
            release_day=release,
            vector=vector,
        )
        by_player[player_id].append(record)
    for entries in by_player.values():
        entries.sort(key=lambda item: (item["played_day"], item["map_id"]))
    receipt = {
        "map_record_count": len(map_ids),
        "player_count": len(by_player),
        "source_series_count": len(series_release),
        "identity_policy": "unique source map ID; consistent series release",
    }
    return dict(by_player), receipt


def build_player_history_anchors(
    records_by_player: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    roster_a: Sequence[str],
    roster_b: Sequence[str],
    cutoff_day: int | str | date,
) -> dict[str, np.ndarray]:
    """Aggregate full raw released maps using the frozen training summarizer."""
    from src.models.c0_edge_model_audit import build_prior_roster_anchors

    return build_prior_roster_anchors(
        records_by_player,
        roster_a=list(roster_a),
        roster_b=list(roster_b),
        decision_day=cutoff_day,
    )


def _collect_worker(
    raw_path: Path, release_path: Path, research_root: str | Path | None
):
    from src.analysis import c0_emea_reconstruction as reconstruction
    from src.analysis import ev_asof_replay as replay

    root = replay._resolve_research_root(research_root)
    raw_hash, release_hash = _sha256(raw_path), _sha256(release_path)
    releases = json.loads(release_path.read_text(encoding="utf-8"))
    if not isinstance(releases, Mapping):
        raise ValueError("Corrected release map must be an ID-to-day mapping")
    backend, backend_hashes = replay._activate_feature_backend(root)
    sources: dict[str, Any] = {}

    def capture():
        for raw in backend["iter_matches"](raw_path):
            source = reconstruction._capture(raw)
            sid = source["source_series_id"]
            if sid in sources:
                raise ValueError(f"Duplicate source series ID {sid}")
            sources[sid] = source
            yield raw

    series, metadata, extra, audit = backend["collect_history"](capture())
    corrected = []
    for item in series:
        source = sources[item.identifier]
        effective = replay._corrected_series_day(item, releases)
        source.update(
            effective_release_day=effective.isoformat(),
            team1_id=item.team1,
            team2_id=item.team2,
            best_of=item.best_of,
            latest_played_day=max(game.day for game in item.games).isoformat(),
        )
        for game, record in zip(item.games, source["maps"], strict=True):
            record.update(
                effective_release_day=effective.isoformat(),
                team1_id=item.team1,
                team2_id=item.team2,
                roster_a=list(game.players1),
                roster_b=list(game.players2),
                w20_valid_a=game.history1 is not None,
                w20_valid_b=game.history2 is not None,
            )
        if effective != item.games[-1].day:
            from dataclasses import replace

            item = replace(
                item, games=[*item.games[:-1], replace(item.games[-1], day=effective)]
            )
        corrected.append(item)
    corrected.sort(key=lambda item: (item.games[-1].day, item.order))
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    team_histories: dict[str, list[Any]] = defaultdict(list)
    state = backend["TemporalState"]()
    state.vocabulary = dict(backend["champion_vocabulary"])
    for index, item in enumerate(corrected, 1):
        reconstruction._consume(
            state,
            item,
            metadata[item.identifier],
            extra[item.identifier],
            sources[item.identifier],
            histories,
            team_histories,
        )
        if index % 5000 == 0:
            print(
                f"c0_live_query reconstructed {index}/{len(corrected)} released series",
                file=sys.stderr,
                flush=True,
            )
    flat = []
    source_map_identity: dict[str, tuple[str, str, int]] = {}
    for player_id, entries in histories.items():
        for entry in entries:
            source_map_id = str(entry["source_map_id"])
            series_id = str(entry["source_series_id"])
            played_day = _day(entry["played_day"])
            release_day = _day(entry["effective_release_day"])
            map_identity = (series_id, str(entry["played_day"]), release_day)
            previous = source_map_identity.setdefault(source_map_id, map_identity)
            if previous != map_identity:
                raise ValueError(
                    f"Conflicting source map ID {source_map_id} across series or clocks"
                )
            vector = np.zeros(HISTORY_WIDTH, dtype=np.float32)
            vector[:12] = np.asarray(entry["raw_stats"], dtype=np.float32)
            vector[12:24] = np.asarray(entry["raw_stat_presence"], dtype=np.float32)
            vector[24] = float(entry["win"])
            vector[26] = float(entry["opponent_prior_rating"])
            role = str(entry["role"])
            vector[27:33] = [
                float(role == value)
                for value in ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT", "UNKNOWN")
            ]
            flat.append(
                {
                    "map_id": f"{source_map_id}:{player_id}",
                    "source_map_id": source_map_id,
                    "series_id": series_id,
                    "player_id": str(player_id),
                    "played_day": played_day,
                    "release_day": release_day,
                    "vector": vector,
                }
            )
    records, identity_receipt = assemble_player_map_records(flat)
    expected_source_maps = sum(len(source["maps"]) for source in sources.values())
    if len(source_map_identity) != expected_source_maps:
        raise ValueError(
            "Source map membership changed while assembling player histories"
        )
    identity_receipt["source_map_count"] = len(source_map_identity)
    identity_receipt["source_map_membership_sha256"] = hashlib.sha256(
        json.dumps(sorted(source_map_identity.items()), separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    identity_receipt["source_series_membership_sha256"] = hashlib.sha256(
        json.dumps(
            sorted((item.identifier, len(item.games)) for item in corrected),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if _sha256(raw_path) != raw_hash or _sha256(release_path) != release_hash:
        raise ValueError(
            "Raw history or corrected release map changed during reconstruction"
        )
    receipt = {
        **identity_receipt,
        "raw_history_sha256": raw_hash,
        "release_map_sha256": release_hash,
        "raw_history_path": str(raw_path),
        "release_map_path": str(release_path),
        "raw_series_reconstructed": len(corrected),
        "feature_backend_hashes": dict(sorted(backend_hashes.items())),
        "source_audit": audit,
        "source_kind": "full_raw_history_and_corrected_release_map",
        "effective_release_policy": "max(corrected override, latest source map day); whole-series strict exclusive cutoff",
    }
    return records, receipt


def _worker(source: Path, destination: Path) -> None:
    payload = pickle.loads(source.read_bytes())
    if payload["operation"] == "collect":
        result = _collect_worker(
            Path(payload["raw_path"]),
            Path(payload["release_path"]),
            payload["research_root"],
        )
    elif payload["operation"] == "worlds":
        result = _worlds_worker(payload)
    else:
        raise ValueError("Unknown isolated worker operation")
    destination.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))


def _invoke(payload: Mapping[str, Any], *, timeout: float | None = None):
    with tempfile.TemporaryDirectory(prefix="c0-live-query-") as directory:
        source, destination = (
            Path(directory) / "input.pkl",
            Path(directory) / "output.pkl",
        )
        source.write_bytes(
            pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
        )
        command = [
            sys.executable,
            "-m",
            "src.models.c0_live_query",
            "--worker",
            str(source),
            str(destination),
        ]
        environment = dict(os.environ)
        environment["BETTING_APP_C0_WORKER"] = "1"
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Isolated C0 live query worker failed: {completed.stdout.strip()}"
            )
        return pickle.loads(destination.read_bytes())


def collect_player_map_records(
    raw_history_path: str | Path,
    release_map_path: str | Path,
    *,
    research_root: str | Path | None = None,
):
    raw_path, release_path = _path(raw_history_path), _path(release_map_path)
    if not raw_path.is_file() or not release_path.is_file():
        raise FileNotFoundError("Raw history and corrected release map must both exist")
    return _invoke(
        {
            "operation": "collect",
            "raw_path": str(raw_path),
            "release_path": str(release_path),
            "research_root": str(research_root) if research_root is not None else None,
        },
        timeout=None,
    )


def _worlds_wire_request_with_native_slots(
    wire_request: Mapping[str, Any], validated_request: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact wire contract with roster slots aligned to native tensors."""
    ordered = dict(wire_request)
    ordered["roster_a"] = list(validated_request["roster_a"])
    ordered["roster_b"] = list(validated_request["roster_b"])
    return ordered


def _worlds_worker(payload: Mapping[str, Any]):
    import pandas as pd
    from betting_app.core.models.native_c0 import NativeC0Adapter

    snapshots = pickle.loads(Path(payload["snapshot_path"]).read_bytes())
    expected_ids = list(range(146609, 146617))
    if not isinstance(snapshots, Mapping) or set(snapshots) != set(expected_ids):
        raise ValueError(
            "Retained Worlds snapshot must contain exactly canonical membership 146609..146616"
        )
    if "records_by_player" in payload:
        records, receipt = payload["records_by_player"], payload["raw_map_receipt"]
    else:
        records, receipt = _collect_worker(
            Path(payload["raw_path"]),
            Path(payload["release_path"]),
            payload["research_root"],
        )
    adapter = NativeC0Adapter()
    adapter.load()
    requests, proofs, raws, histories, probabilities = [], [], [], [], []
    anchor_members_by_horizon = {str(horizon): [] for horizon in HORIZONS}
    for match_id in expected_ids:
        request, encoded, family_a, family_b, offset_a, offset_b = snapshots[match_id]
        request = dict(request)
        if request.get("competition_context") != "Worlds":
            raise ValueError(
                f"Cached query {match_id} has unexpected competition context"
            )
        wire_request = dict(request)
        validated_request = adapter._validate_request(wire_request)
        wire_request = _worlds_wire_request_with_native_slots(
            wire_request, validated_request
        )
        target, raw, hist = adapter._decode_snapshot(encoded, validated_request)
        request = validated_request
        roster_a = request["roster_a"]
        roster_b = request["roster_b"]
        cutoff = _day(target["cutoff_day"])
        roster_a = request["roster_a"]
        roster_b = request["roster_b"]
        for side, roster in (("a", roster_a), ("b", roster_b)):
            for key in (
                f"tensor_player_order_{side}",
                f"raw_tensor_player_order_{side}",
                f"temporal_tensor_player_order_{side}",
            ):
                if target.get(key) != roster:
                    raise ValueError(
                        f"Cached original C0 tensor slot identity mismatch: {key}"
                    )
        if (
            target.get("feature_source_max_day")
            and _day(target["feature_source_max_day"]) >= cutoff
        ):
            raise ValueError(
                "Cached original C0 tensor contains a non-prior feature day"
            )
        anchors = build_player_history_anchors(
            records, roster_a=roster_a, roster_b=roster_b, cutoff_day=cutoff
        )
        anchor_history = np.concatenate(
            [anchors[f"history_{h}"] for h in (365, 180, 90)], axis=2
        )
        anchor_mask = np.concatenate(
            [anchors[f"mask_{h}"] for h in (365, 180, 90)], axis=2
        )
        for horizon in HORIZONS:
            horizon_members = []
            for roster in (roster_a, roster_b):
                side_members = []
                for player_id in roster:
                    side_members.append(
                        sorted(
                            str(record["map_id"])
                            for record in records.get(player_id, ())
                            if 0 < cutoff - _day(record["played_day"]) < horizon
                            and _day(record["release_day"]) < cutoff
                        )
                    )
                horizon_members.append(side_members)
            anchor_members_by_horizon[str(horizon)].append(horizon_members)
        # Anchors precede recent16 as in the frozen fresh20 recipe.
        raws.append({key: np.asarray(value).copy() for key, value in raw.items()})
        histories.append(
            {
                "history": np.concatenate([anchor_history, hist["history"]], axis=2),
                "mask": np.concatenate([anchor_mask, hist["mask"]], axis=2),
                "champions": np.asarray(hist["champions"]).copy(),
            }
        )
        prediction = adapter.predict(
            wire_request, snapshot_value=encoded, canonical_match_id=int(match_id)
        )
        probabilities.append(float(prediction["prob_a"]))
        requests.append(wire_request)
        proofs.append(
            {
                "opportunity_id": str(match_id),
                "canonical_match_id": int(match_id),
                "cutoff_day": target["cutoff_day"],
                "feature_source_max_day": target.get("feature_source_max_day"),
                "roster_status": "conditional_roster_from_retained_snapshot_not_announcement_authenticated",
                "family_a": str(family_a),
                "family_b": str(family_b),
                "offset_a": float(offset_a),
                "offset_b": float(offset_b),
                "anchor_counts": {
                    str(h): anchors[f"count_{h}"].tolist() for h in HORIZONS
                },
                "anchor_members": {
                    str(h): anchor_members_by_horizon[str(h)][-1] for h in HORIZONS
                },
                "original_tensor_player_order_a": roster_a,
                "original_tensor_player_order_b": roster_b,
                "c0_diagnostics": prediction["diagnostics"],
            }
        )
    raw_batch = {key: np.stack([entry[key] for entry in raws]) for key in raws[0]}
    history_batch = {
        key: np.stack([entry[key] for entry in histories]) for key in histories[0]
    }
    if history_batch["history"].shape[3] != 19 or history_batch["mask"].shape[3] != 19:
        raise ValueError(
            "fresh20 input must contain exactly 19 observed slots: three full-source anchors plus 16 native slots"
        )
    return {
        "requests": requests,
        "batches": {
            "20t": {
                "candidate": "20t",
                "row_ids": np.asarray(list(map(str, expected_ids))),
                "metadata": pd.DataFrame(
                    {"best_of": [int(item["best_of"]) for item in requests]}
                ),
                "raw": raw_batch,
                "history": history_batch,
                "best_of": np.asarray(
                    [int(item["best_of"]) for item in requests], dtype=np.int64
                ),
            }
        },
        "p_c0": np.asarray(probabilities, dtype=np.float64),
        "query_proofs": proofs,
        "raw_map_receipt": receipt,
        "snapshot_sha256": _sha256(Path(payload["snapshot_path"])),
        "roster_status": "conditional_retained_snapshot_rosters_not_announcement_authenticated",
    }


def build_worlds_query(
    snapshot_path: str | Path = SNAPSHOT_DEFAULT,
    *,
    raw_history_path: str | Path,
    release_map_path: str | Path,
    research_root: str | Path | None = None,
    records_by_player: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    raw_map_receipt: Mapping[str, Any] | None = None,
):
    paths = {
        key: _path(value)
        for key, value in (
            ("snapshot_path", snapshot_path),
            ("raw_path", raw_history_path),
            ("release_path", release_map_path),
        )
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            "Worlds snapshot, raw history, and corrected release map must exist"
        )
    if records_by_player is None:
        records_by_player, collected_receipt = collect_player_map_records(
            paths["raw_path"], paths["release_path"], research_root=research_root
        )
        raw_map_receipt = collected_receipt
    elif raw_map_receipt is None:
        raise ValueError("Caller-supplied records require their reconstruction receipt")
    if raw_map_receipt.get("raw_history_sha256") != _sha256(
        paths["raw_path"]
    ) or raw_map_receipt.get("release_map_sha256") != _sha256(paths["release_path"]):
        raise ValueError(
            "Reused player-map cache receipt does not bind the supplied raw and release inputs"
        )
    return _invoke(
        {
            "operation": "worlds",
            **{key: str(value) for key, value in paths.items()},
            "research_root": str(research_root) if research_root is not None else None,
            "records_by_player": records_by_player,
            "raw_map_receipt": raw_map_receipt,
        },
        timeout=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    args = parser.parse_args(argv)
    if args.worker:
        _worker(Path(args.worker[0]), Path(args.worker[1]))
        return 0
    parser.error("--worker is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
