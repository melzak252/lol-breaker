"""Raw, release-aware C0 causal anchor history bank.

The bank is a new development experiment; source data and prior stage artifacts
are read-only. Archive feature code is activated only from the command process.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (90, 180, 365)
HISTORY_WIDTH = 33


def summarize_released_maps(
    records: Iterable[Mapping[str, Any]], *, cutoff_day: int, horizon_days: int
) -> dict[str, Any]:
    """Summarize unique maps played strictly within a released prior window.

    Per-coordinate means use the native presence-mask convention. Presence
    channels are one if any selected map observed that statistic; all remaining
    native context channels are averaged over selected map records.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    unique: dict[str, tuple[tuple[Any, ...], np.ndarray]] = {}
    for record in records:
        try:
            map_id = str(record["map_id"])
            series_id = str(record["series_id"])
            player_id = str(record["player_id"])
            played = int(record["played_day"])
            released = int(record["release_day"])
            vector = np.asarray(record["vector"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed released-map record") from error
        if (
            not map_id
            or vector.shape != (HISTORY_WIDTH,)
            or not np.issubdtype(vector.dtype, np.number)
        ):
            raise ValueError("Invalid map identity or native history vector")
        if not np.isfinite(vector).all():
            raise ValueError("Native history vector must be finite")
        signature = (
            series_id,
            player_id,
            played,
            released,
            vector.dtype.str,
            vector.tobytes(),
        )
        prior = unique.get(map_id)
        if prior is not None:
            if prior[0] != signature:
                raise ValueError(f"Conflicting records for stable map ID {map_id!r}")
            continue
        unique[map_id] = (signature, vector)

    selected = [
        (map_id, vector)
        for map_id, (signature, vector) in unique.items()
        if 0 < cutoff_day - signature[2] < horizon_days and signature[3] < cutoff_day
    ]
    selected.sort(key=lambda item: item[0])
    history = np.zeros(HISTORY_WIDTH, dtype=np.float32)
    if selected:
        values = np.stack(
            [np.asarray(vector, dtype=np.float64) for _, vector in selected]
        )
        # The first twelve native channels have matching presence bits at 12:24.
        for channel in range(12):
            known = values[:, channel + 12] > 0.5
            if known.any():
                history[channel] = np.float32(values[known, channel].mean())
        history[12:24] = (values[:, 12:24] > 0.5).any(axis=0).astype(np.float32)
        history[24:] = values[:, 24:].mean(axis=0).astype(np.float32)
    return {
        "history": history,
        "mask": bool(selected),
        "count": len(selected),
        "members": [map_id for map_id, _ in selected],
    }


def build_raw_anchor_bank(
    protocol: Mapping[str, Any] | str | Path, *, output_dir: str | Path
) -> dict[str, Any]:
    """Build a fresh bank from authoritative raw data; never alter source artifacts."""
    protocol_obj, protocol_path = _load_protocol(protocol)
    return _build_isolated(
        protocol_obj, protocol_path, Path(output_dir).expanduser().resolve(), "--build"
    )


def verify_raw_anchor_bank(
    protocol: Mapping[str, Any] | str | Path, manifest: Mapping[str, Any] | str | Path
) -> dict[str, Any]:
    """Independently verify a bank manifest and its causal source reconstruction."""
    protocol_obj, protocol_path = _load_protocol(protocol)
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
    else:
        manifest_path = (
            Path(str(manifest.get("manifest_path", ""))).expanduser().resolve()
        )
        if not manifest_path.is_file():
            raise ValueError(
                "manifest_path is required when passing an in-memory manifest"
            )
    return _build_isolated(protocol_obj, protocol_path, manifest_path, "--verify")


def _load_protocol(
    protocol: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path]:
    if isinstance(protocol, Mapping):
        obj = dict(protocol)
        path = Path(str(obj.get("protocol_path", "")))
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
    else:
        path = Path(protocol).expanduser().resolve()
        obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("scope") != "HISTORICAL_DEVELOPMENT_ONLY":
        raise ValueError("Raw anchor bank requires a historical development protocol")
    return obj, path


def _build_isolated(
    protocol: Mapping[str, Any], protocol_path: Path, destination: Path, action: str
) -> dict[str, Any]:
    import subprocess
    import tempfile

    temporary_protocol = None
    if not protocol_path.is_file():
        temporary_protocol = tempfile.TemporaryDirectory(prefix="c0-anchor-protocol-")
        protocol_path = Path(temporary_protocol.name) / "protocol.json"
        protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "src.models.c0_causal_anchor_bank",
        "--protocol",
        str(protocol_path),
        action,
    ]
    command.extend(
        ["--output-dir" if action == "--build" else "--manifest", str(destination)]
    )
    try:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, check=True, text=True, capture_output=True
        )
    finally:
        if temporary_protocol is not None:
            temporary_protocol.cleanup()
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Isolated raw-bank process returned invalid JSON") from error


def _worker(
    protocol_path: Path, *, output_dir: Path | None, manifest_path: Path | None
) -> dict[str, Any]:
    """Reconstruct the authoritative maps, then build or independently inspect a fresh bank."""
    import bisect
    import tempfile
    from collections import defaultdict
    from datetime import date

    import pandas as pd

    from src.analysis import ev_asof_replay as replay
    from src.analysis import c0_emea_reconstruction as reconstruction

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = replay._resolve_research_root(None)
    bank = root / replay.BANK_RELATIVE
    raw_path = PROJECT_ROOT / replay.RAW_HISTORY_RELATIVE
    release_path = bank / "release_days.json"
    source_paths = (
        protocol_path,
        Path(__file__).resolve(),
        raw_path,
        release_path,
        bank / "metadata.parquet",
        bank / "original_metadata.jsonl",
    )
    source_hashes = {str(path): _sha256(path) for path in source_paths}
    if (
        source_hashes[str(raw_path)] != replay.BANK_SOURCE_SHA256
        or source_hashes[str(release_path)] != replay.BANK_RELEASE_SHA256
    ):
        raise ValueError(
            "Canonical raw map or corrected release source hash differs from frozen authority"
        )
    releases = json.loads(release_path.read_text(encoding="utf-8"))
    # This isolated reconstruction uses the archived rating package, not the
    # application's newer lazy-export initializer. Keep the original manifest
    # check intact rather than accepting a changed source hash.
    import importlib.util
    ratings = root / "unified-ratings-20260911/code/src/ratings"
    manifest = json.loads((root / "unified-ratings-20260911/manifest.json").read_text())
    expected_init = manifest["files"]["code/src/ratings/__init__.py"]["sha256"]
    if _sha256(ratings / "__init__.py") != expected_init:
        raise ValueError("Archived rating package initializer differs from its manifest")
    for name in list(sys.modules):
        if name == "src.ratings" or name.startswith("src.ratings."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "src.ratings", ratings / "__init__.py", submodule_search_locations=[str(ratings)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["src.ratings"] = package
    spec.loader.exec_module(package)
    sys.modules["src"].ratings = package
    backend, backend_hashes = replay._activate_feature_backend(root)
    series, metadata, extra, sources, audit = reconstruction._collect(
        raw_path, releases, backend
    )
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    team_histories: dict[str, Any] = defaultdict(list)
    state = backend["TemporalState"]()
    state.vocabulary = dict(backend["champion_vocabulary"])
    for item in series:
        reconstruction._consume(
            state,
            item,
            metadata[item.identifier],
            extra[item.identifier],
            sources[item.identifier],
            histories,
            team_histories,
        )
    source_hashes.update(backend_hashes)
    source_hashes.update(reconstruction._verify_feature_imports(root))
    original = pd.read_json(
        bank / "original_metadata.jsonl",
        lines=True,
        convert_dates=False,
        dtype={"golgg_match_id": str},
    )
    rows = pd.read_parquet(bank / "metadata.parquet")
    if len(rows) != 40636 or len(original) != len(rows):
        raise ValueError(
            "Canonical bank row identity inventory must contain all 40636 rows"
        )
    ids = rows.golgg_match_id.astype(str).to_numpy()
    if original.golgg_match_id.astype(str).tolist() != ids.tolist() or len(
        set(ids)
    ) != len(ids):
        raise ValueError(
            "Canonical GOL.GG row IDs differ from original metadata or are not unique"
        )

    # Build immutable per-player map records once; query the chronological slice
    # for each decision row, retaining all raw history (no token truncation).
    by_player: dict[str, tuple[list[dict[str, Any]], list[int]]] = {}
    for player_id, entries in histories.items():
        records = []
        for entry in entries:
            played = date.fromisoformat(str(entry["played_day"])[:10]).toordinal()
            released = date.fromisoformat(
                str(entry["effective_release_day"])[:10]
            ).toordinal()
            stats = np.asarray(entry["raw_stats"], dtype=np.float32)
            present = np.asarray(entry["raw_stat_presence"], dtype=np.float32)
            vector = np.zeros(HISTORY_WIDTH, dtype=np.float32)
            vector[:12] = stats
            vector[12:24] = present
            vector[24] = float(entry["win"])
            vector[26] = float(entry["opponent_prior_rating"])
            role = str(entry["role"])
            vector[27:33] = [
                float(role == name)
                for name in ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT", "UNKNOWN")
            ]
            records.append(
                {
                    "played": played,
                    "released": released,
                    "map_id": str(entry["source_map_id"]),
                    "series_id": str(entry["source_series_id"]),
                    "vector": vector,
                }
            )
        records.sort(key=lambda record: (record["played"], record["map_id"]))
        by_player[player_id] = (records, [record["played"] for record in records])
    payload: dict[str, np.ndarray] = {
        "row_ids": np.asarray(ids, dtype=f"U{max(map(len, ids))}")
    }
    membership: dict[str, Any] = {}
    counts_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        # Singleton token axis is the replay owner’s concatenation contract.
        histories_array = np.zeros(
            (len(rows), 2, 5, 1, HISTORY_WIDTH), dtype=np.float32
        )
        masks = np.zeros((len(rows), 2, 5, 1), dtype=np.uint8)
        counts = np.zeros((len(rows), 2, 5), dtype=np.uint32)
        horizon_members: dict[str, Any] = {}
        for index, row in rows.iterrows():
            cutoff = date.fromisoformat(str(row["date"])[:10]).toordinal()
            for side, roster_field in enumerate(("roster_a", "roster_b")):
                roster = original.iloc[index][roster_field]
                if isinstance(roster, str):
                    roster = json.loads(roster)
                roster = [str(player_id) for player_id in roster]
                if len(roster) != 5 or len(set(roster)) != 5:
                    raise ValueError(
                        f"Invalid canonical {roster_field} for {ids[index]}"
                    )
                for slot, player_id in enumerate(roster):
                    records, played_days = by_player.get(player_id, ([], []))
                    start = bisect.bisect_right(played_days, cutoff - horizon)
                    end = bisect.bisect_left(played_days, cutoff)
                    selected = [
                        record
                        for record in records[start:end]
                        if 0 < cutoff - record["played"] < horizon
                        and record["released"] < cutoff
                    ]
                    selected.sort(key=lambda record: record["map_id"])
                    vectors = []
                    for record in selected:
                        vector = record["vector"].copy()
                        vector[25] = np.log1p(max(1, cutoff - record["played"]))
                        vectors.append(vector)
                    summary = summarize_released_maps(
                        (
                            {
                                "map_id": record["map_id"],
                                "series_id": record["series_id"],
                                "player_id": player_id,
                                "played_day": record["played"],
                                "release_day": record["released"],
                                "vector": vector,
                            }
                            for record, vector in zip(selected, vectors, strict=True)
                        ),
                        cutoff_day=cutoff,
                        horizon_days=horizon,
                    )
                    histories_array[index, side, slot, 0] = summary["history"]
                    masks[index, side, slot, 0] = summary["mask"]
                    counts[index, side, slot] = summary["count"]
                    horizon_members[f"{ids[index]}:{side}:{slot}"] = summary["members"]
        key = str(horizon)
        payload[f"history_{key}"] = histories_array
        payload[f"mask_{key}"] = masks
        payload[f"count_{key}"] = counts
        membership[key] = horizon_members
        counts_by_horizon[key] = {
            "valid_player_rows": int(masks.sum()),
            "maps": int(counts.sum()),
        }

    if any(_sha256(path) != source_hashes[str(path)] for path in source_paths):
        raise ValueError(
            "A protected raw, protocol, or metadata source changed during anchor construction"
        )
    if output_dir is None:
        if manifest_path is None:
            raise ValueError("Missing verification manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "c0_raw_causal_anchor_bank_v2":
            raise ValueError("Unexpected raw anchor bank schema")
        with tempfile.TemporaryDirectory(prefix="c0-anchor-check-") as temporary:
            checked = _write_bank(
                Path(temporary) / "replay",
                payload,
                membership,
                source_hashes,
                counts_by_horizon,
                audit,
            )
        expected_artifacts = manifest.get("artifacts")
        if not isinstance(expected_artifacts, dict):
            raise ValueError("Raw anchor bank manifest omits artifact hashes")
        actual_artifacts = {
            name: _sha256(manifest_path.parent / name) for name in expected_artifacts
        }
        if actual_artifacts != expected_artifacts:
            raise ValueError(
                "Raw anchor bank artifact hashes differ from immutable manifest"
            )
        if expected_artifacts != checked["artifacts"]:
            raise ValueError(
                "Raw anchor bank artifacts differ from causal source reconstruction"
            )
        if manifest.get("source_sha256") != dict(source_hashes):
            raise ValueError(
                "Raw anchor bank provenance differs from authoritative sources"
            )
        # Explicitly inspect membership, shape, and causal bounds independently
        # of the artifact hash comparison above.
        saved_membership = json.loads(
            (manifest_path.parent / "map_membership.json").read_text(encoding="utf-8")
        )
        record_by_player_map = {
            player_id: {record["map_id"]: record for record in records}
            for player_id, (records, _) in by_player.items()
        }
        perturbation_checks = 0
        for horizon in HORIZONS:
            key = str(horizon)
            if saved_membership.get(key) != membership[key]:
                raise ValueError(f"Raw membership differs for {horizon}-day history")
            history = np.load(
                manifest_path.parent / f"history_{key}.npy", allow_pickle=False
            )
            mask = np.load(manifest_path.parent / f"mask_{key}.npy", allow_pickle=False)
            counts = np.load(
                manifest_path.parent / f"count_{key}.npy", allow_pickle=False
            )
            if (
                history.shape != (40636, 2, 5, 1, HISTORY_WIDTH)
                or mask.shape != (40636, 2, 5, 1)
                or counts.shape != (40636, 2, 5)
            ):
                raise ValueError(
                    "Raw anchor arrays violate singleton-token replay contract"
                )
            if not np.array_equal(mask[..., 0].astype(bool), counts > 0):
                raise ValueError("Raw anchor mask/count membership disagreement")
            for index, row in rows.iterrows():
                cutoff = date.fromisoformat(str(row["date"])[:10]).toordinal()
                for side, roster_field in enumerate(("roster_a", "roster_b")):
                    roster = original.iloc[index][roster_field]
                    if isinstance(roster, str):
                        roster = json.loads(roster)
                    for slot, player_id in enumerate(map(str, roster)):
                        member_key = f"{ids[index]}:{side}:{slot}"
                        member_ids = saved_membership[key].get(member_key)
                        if not isinstance(member_ids, list) or member_ids != sorted(
                            set(member_ids)
                        ):
                            raise ValueError(
                                "Raw membership IDs are missing, duplicated, or not canonical"
                            )
                        by_map = record_by_player_map.get(player_id, {})
                        eligible = sorted(
                            map_id
                            for map_id, record in by_map.items()
                            if 0 < cutoff - record["played"] < horizon
                            and record["released"] < cutoff
                        )
                        if member_ids != eligible:
                            raise ValueError(
                                "Raw membership omits eligible maps or contains a noncausal map"
                            )
                        if counts[index, side, slot] != len(member_ids) or bool(
                            mask[index, side, slot, 0]
                        ) != bool(member_ids):
                            raise ValueError(
                                "Raw membership differs from saved count/mask tensors"
                            )
                        expected = np.zeros(HISTORY_WIDTH, dtype=np.float32)
                        if member_ids:
                            values = np.stack(
                                [
                                    by_map[map_id]["vector"].astype(
                                        np.float64, copy=True
                                    )
                                    for map_id in member_ids
                                ]
                            )
                            values[:, 25] = [
                                np.float32(np.log1p(max(1, cutoff - by_map[map_id]["played"])))
                                for map_id in member_ids
                            ]
                            for channel in range(12):
                                observed = values[:, channel + 12] > 0.5
                                if observed.any():
                                    expected[channel] = np.float32(
                                        values[observed, channel].mean()
                                    )
                            expected[12:24] = (
                                (values[:, 12:24] > 0.5).any(axis=0).astype(np.float32)
                            )
                            expected[24:] = (
                                values[:, 24:].mean(axis=0).astype(np.float32)
                            )
                        if not np.array_equal(history[index, side, slot, 0], expected):
                            raise ValueError(
                                "Raw summary differs from independent native-channel recomputation"
                            )
                        if index % 1024 == 0:
                            blocked = [
                                record
                                for record in by_map.values()
                                if record["played"] >= cutoff
                                or record["released"] >= cutoff
                            ]
                            if blocked:
                                probe = []
                                for record in [
                                    by_map[map_id] for map_id in member_ids
                                ] + blocked[:3]:
                                    vector = record["vector"].copy()
                                    vector[25] = np.log1p(
                                        max(1, cutoff - record["played"])
                                    )
                                    probe.append(
                                        {
                                            "map_id": record["map_id"],
                                            "series_id": record["series_id"],
                                            "player_id": player_id,
                                            "played_day": record["played"],
                                            "release_day": record["released"],
                                            "vector": vector,
                                        }
                                    )
                                for record in probe:
                                    if (
                                        record["played_day"] >= cutoff
                                        or record["release_day"] >= cutoff
                                    ):
                                        record["vector"][:] += 17
                                        record["vector"][24] = 1 - record["vector"][24]
                                        if record["release_day"] >= cutoff:
                                            record["release_day"] += 31
                                changed = summarize_released_maps(
                                    probe, cutoff_day=cutoff, horizon_days=horizon
                                )
                                if changed[
                                    "members"
                                ] != member_ids or not np.array_equal(
                                    changed["history"], expected
                                ):
                                    raise ValueError(
                                        "Future/target/unreleased-map perturbation changed a prior summary"
                                    )
                                perturbation_checks += 1
            if not np.isfinite(history).all():
                raise ValueError("Raw anchor contains non-finite summaries")
        return {
            "status": "VERIFIED",
            "row_count": len(ids),
            "manifest_path": str(manifest_path),
            "future_summary_perturbation_checks": perturbation_checks,
            "perturbation_scope": "Representative actual excluded map records; not a full raw-source mutation replay",
            "normalization": "Raw native vectors; original TRAIN-fitted shared HistoryNormalizer is applied by the model",
        }
    return _write_bank(
        output_dir, payload, membership, source_hashes, counts_by_horizon, audit
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bank(
    output_dir: Path,
    payload: Mapping[str, np.ndarray],
    membership: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    counts: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable raw anchor bank: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, str] = {}
    for key, array in payload.items():
        filename = f"{key}.npy"
        np.save(output_dir / filename, array, allow_pickle=False)
        artifacts[filename] = _sha256(output_dir / filename)
    membership_path = output_dir / "map_membership.json"
    membership_path.write_text(
        json.dumps(membership, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    artifacts[membership_path.name] = _sha256(membership_path)
    manifest = {
        "schema": "c0_raw_causal_anchor_bank_v2",
        "experiment": "new_daily_scenario_not_original_reproduction",
        "row_count": len(payload["row_ids"]),
        "horizons_days": list(HORIZONS),
        "tensor_contract": {
            "history": "(row,2,5,1,33)",
            "mask": "(row,2,5,1)",
            "count": "(row,2,5)",
            "row_ids": "original_metadata order",
        },
        "artifacts": artifacts,
        "artifact_sha256": hashlib.sha256(
            json.dumps(artifacts, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": dict(source_hashes),
        "counts": dict(counts),
        "provenance": {
            "history_audit": _json_safe(audit),
            "membership": membership_path.name,
            "native_width": HISTORY_WIDTH,
            "source": "authoritative raw maps and corrected release map",
            "summary": "native masked-mean 33-feature convention from train_c0_multiscale_20token",
            "release_rule": "effective_release_day < cutoff; 0 < cutoff-played_day < horizon",
            "normalization": "none; raw native history scale, no fit or transform",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.build:
        if args.manifest is not None:
            parser.error("--build forbids --manifest")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        run_id = str(protocol.get("run_id", "c0_raw_anchor_bank"))
        destination = args.output_dir or (
            PROJECT_ROOT / "data/04_feature" / run_id / "stage04run"
        )
        result = _worker(
            args.protocol.resolve(),
            output_dir=destination.resolve(),
            manifest_path=None,
        )
    else:
        if args.output_dir is not None:
            parser.error("--verify forbids --output-dir")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        run_id = str(protocol.get("run_id", "c0_raw_anchor_bank"))
        manifest_path = args.manifest or (
            PROJECT_ROOT / "data/04_feature" / run_id / "stage04run" / "manifest.json"
        )
        result = _worker(
            args.protocol.resolve(),
            output_dir=None,
            manifest_path=manifest_path.resolve(),
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
