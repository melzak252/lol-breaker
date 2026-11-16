"""Research-only audit of saved C0 distilled models and causal query inputs.

This module never fits or activates a serving model. Historical fresh studies are
reused as controls; saved checkpoints are loaded only for caller-supplied inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence
import os

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_MANIFEST = PROJECT_ROOT / "data/06_models/c0_distilled/manifest.json"
LEGACY_REPORT = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
FRESH_PROTOCOL = PROJECT_ROOT / "data/05_model_input/c0_distilled_verification/c0d_repro_20261021_01/protocol.json"
HORIZONS = (90, 180, 365)
HISTORY_WIDTH = 33
MODES = ("full", "no_w20", "no_organization")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _decode_json_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return value
        else:
            return value
    if isinstance(value, np.ndarray):
        return [_decode_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {key: _decode_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode_json_value(item) for item in value]
    return value
def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _day_ordinal(value: Any) -> int:
    if isinstance(value, date):
        return value.toordinal()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10]).toordinal()
        except ValueError as error:
            raise ValueError("decision_day must be an ordinal or ISO date") from error
    raise ValueError("decision_day must be an ordinal or ISO date")


def build_prior_roster_anchors(
    records_by_player: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    roster_a: Sequence[str],
    roster_b: Sequence[str],
    decision_day: int | str | date,
) -> dict[str, np.ndarray]:
    """Build native 33-coordinate map anchors for an explicit, prior roster pair.

    Map records are raw reconstructed maps. Every horizon applies the same strict
    play-window and release-day rules as the frozen C0 anchor-bank summarizer.
    The input roster order is retained slot-for-slot; no target-lineup fallback is
    accepted.
    """
    from src.models.c0_causal_anchor_bank import summarize_released_maps

    cutoff = _day_ordinal(decision_day)
    rosters: list[list[str]] = []
    for name, roster in (("roster_a", roster_a), ("roster_b", roster_b)):
        if not isinstance(roster, (list, tuple)) or len(roster) != 5:
            raise ValueError(f"{name} must contain exactly five explicit players")
        ids = [str(value).strip() for value in roster]
        if (
            any(
                value is None
                or not str(value).strip()
                or str(value).strip().lower() in {"none", "nan"}
                for value in roster
            )
            or len(set(ids)) != 5
        ):
            raise ValueError(f"{name} must contain five unique nonempty player IDs")
        rosters.append(ids)
    if set(rosters[0]) & set(rosters[1]):
        raise ValueError("Native team rosters must be disjoint")

    output: dict[str, np.ndarray] = {}
    for horizon in HORIZONS:
        histories = np.zeros((2, 5, 1, HISTORY_WIDTH), dtype=np.float32)
        masks = np.zeros((2, 5, 1), dtype=np.uint8)
        counts = np.zeros((2, 5), dtype=np.uint32)
        for side, roster in enumerate(rosters):
            for slot, player_id in enumerate(roster):
                query_records = []
                for record in records_by_player.get(player_id, ()):
                    try:
                        played = _day_ordinal(record["played_day"])
                        released = _day_ordinal(record["release_day"])
                        vector = np.asarray(record["vector"], dtype=np.float32)
                        map_id = str(record["map_id"])
                        series_id = str(record["series_id"])
                        record_player = str(record["player_id"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError("Malformed raw player-map history") from error
                    if record_player != player_id:
                        raise ValueError("Raw map player identity differs from roster slot")
                    if vector.shape != (HISTORY_WIDTH,) or not np.isfinite(vector).all():
                        raise ValueError("Raw map history must be a finite native 33-vector")
                    vector = vector.copy()
                    vector[25] = np.float32(math.log1p(max(1, cutoff - played)))
                    query_records.append({
                        "map_id": map_id,
                        "series_id": series_id,
                        "player_id": player_id,
                        "played_day": played,
                        "release_day": released,
                        "vector": vector,
                    })
                summary = summarize_released_maps(
                    query_records, cutoff_day=cutoff, horizon_days=horizon
                )
                histories[side, slot, 0] = summary["history"]
                masks[side, slot, 0] = int(summary["mask"])
                counts[side, slot] = summary["count"]
        key = str(horizon)
        output[f"history_{key}"] = histories
        output[f"mask_{key}"] = masks
        output[f"count_{key}"] = counts
    return output


def model_identity(name: str, *, protocol_path: str | Path = FRESH_PROTOCOL) -> dict[str, Any]:
    """Resolve a model label to its exact saved checkpoint/calibration identity."""
    label = str(name).strip()
    if label in {"C0-D", "C0-D-Core", "C0-D-Year", "C0-D-Year-Core"}:
        manifest = _read_json(LEGACY_MANIFEST)
        entry = manifest["distilled_models"]["C0-D"]
        checkpoint_paths = {}
        checkpoints = {}
        for item in entry["checkpoints"]:
            path = _path(item["path"])
            actual = _sha256(path)
            if actual != item["sha256"]:
                raise ValueError(f"Legacy C0-D checkpoint hash mismatch: {path}")
            seed = str(item["seed"])
            checkpoint_paths[seed] = str(path)
            checkpoints[seed] = actual
        calibration = entry["calibration"]
        calibration_hash = _canonical_hash(calibration)
        canonical_name = "C0-D-Core"
        aliases = list(entry["aliases"])
        calibration_path = None
    elif label in {"C0-Slim", "C0-D1", "C0-Slim-Year", "C0-D1-Year", "C0-Slim+"}:
        manifest = _read_json(LEGACY_MANIFEST)
        entry = manifest["distilled_models"]["C0-Slim"]
        item = entry["checkpoints"][0]
        path = _path(item["path"])
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"Legacy C0-Slim checkpoint hash mismatch: {path}")
        checkpoints = {str(item["seed"]): actual}
        checkpoint_paths = {str(item["seed"]): str(path)}
        calibration = entry["calibration"]
        calibration_hash = _canonical_hash(calibration)
        canonical_name = "C0-Slim"
        aliases = list(entry["aliases"])
        calibration_path = None
    elif label in {"C0-D-Core-17T", "C0-D-Core-17t"}:
        artifact = _read_json(LEGACY_REPORT / "track1_distillation.json")
        entry = artifact["students"]["student_1c_w32_kd_born_again"]
        checkpoints = {}
        checkpoint_paths = {}
        for seed in (20260910, 20260911, 20260912):
            path = LEGACY_REPORT / f"models/student_w32_kd/seed_{seed}/checkpoint_epoch16.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Saved historical 17-token checkpoint is missing: {path}")
            checkpoints[str(seed)] = _sha256(path)
            checkpoint_paths[str(seed)] = str(path)
        calibration = entry["calibration"]
        calibration_hash = _canonical_hash(calibration)
        canonical_name = "C0-D-Core-17T"
        aliases = ["C0-D-Core-17T"]
        calibration_path = str(LEGACY_REPORT / "track1_distillation.json")
    elif label in {"18t", "20t", "fresh18", "fresh20"}:
        candidate = "18t" if label in {"18t", "fresh18"} else "20t"
        protocol = _read_json(protocol_path)
        inputs = protocol.get("inputs")
        if not isinstance(inputs, Mapping):
            inputs = {}
        model_protocol_path = _path(inputs.get("model_protocol") or FRESH_PROTOCOL)
        model_protocol = _read_json(model_protocol_path)
        from src.models.c0_distilled_reproduction import _stage

        directory = _stage(model_protocol, "06_models") / candidate
        bundle = _read_json(directory / "bundle_manifest.json")
        checkpoints = {}
        checkpoint_paths = {}
        for seed, expected in bundle["checkpoint_sha256"].items():
            checkpoint = directory / f"seed_{seed}/checkpoint_epoch16.pt"
            actual = _sha256(checkpoint)
            if actual != expected:
                raise ValueError(f"Fresh {candidate} checkpoint hash mismatch: {checkpoint}")
            checkpoints[str(seed)] = actual
            checkpoint_paths[str(seed)] = str(checkpoint)
        calibration_path = directory / "calibration.json"
        calibration_hash = _sha256(calibration_path)
        if calibration_hash != bundle["calibration_sha256"]:
            raise ValueError(f"Fresh {candidate} calibration hash mismatch")
        calibration = _read_json(calibration_path)
        canonical_name = candidate
        aliases = [candidate]
    else:
        raise ValueError(f"Unknown saved C0 model identity: {label}")

    identity = {
        "canonical_name": canonical_name,
        "aliases": aliases,
        "checkpoint_sha256": checkpoints,
        "checkpoint_paths": checkpoint_paths,
        "calibration_path": None if calibration_path is None else str(calibration_path),
        "calibration_sha256": calibration_hash,
        "artifact_identity": _canonical_hash({
            "checkpoints": checkpoints,
            "calibration_sha256": calibration_hash,
        }),
        "calibration": calibration,
        "lineage_status": (
            "verified_fresh_raw_causal" if canonical_name in {"18t", "20t"}
            else "unverified_legacy_positional"
        ),
        "research_only": True,
    }
    return identity


def infer_saved_slim(
    batch: Mapping[str, Any], *, best_of: Sequence[int], mode: str = "full"
) -> np.ndarray:
    """Run the genuine saved one-seed Slim network and its saved Beta calibration."""
    if mode not in MODES:
        raise ValueError("Unknown saved inference mode")
    if batch.get("candidate") != "18t":
        raise ValueError("C0-Slim requires an 18-token input batch")
    n = len(batch["raw"]["players"])
    bo = np.asarray(best_of)
    if bo.shape != (n,) or not np.isin(bo, (1, 3, 5)).all():
        raise ValueError("best_of must contain one legal format per row")
    identity = model_identity("C0-Slim")
    request = (dict(batch), bo.astype(np.int64), mode, identity)
    with tempfile.TemporaryDirectory(prefix="c0-edge-slim-inference-") as temporary:
        source = Path(temporary) / "input.pkl"
        destination = Path(temporary) / "output.pkl"
        source.write_bytes(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        completed = subprocess.run(
            [sys.executable, "-m", "src.models.c0_edge_model_audit", "--slim-worker",
             str(source), str(destination)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Saved Slim inference failed: {completed.stderr.strip()}")
        result = np.asarray(pickle.loads(destination.read_bytes()), dtype=np.float64)
    if result.shape != (n,) or not np.isfinite(result).all() or np.any((result <= 0) | (result >= 1)):
        raise ValueError("Saved Slim inference returned invalid probabilities")
    return result

def infer_saved_17t(
    batch: Mapping[str, Any], *, best_of: Sequence[int], mode: str = "full"
) -> np.ndarray:
    if mode not in MODES:
        raise ValueError("Unknown saved inference mode")
    if batch.get("candidate") not in {"18t", "20t"}:
        raise ValueError("Historical 17-token core requires a native query batch")
    n = len(batch["raw"]["players"])
    bo = np.asarray(best_of)
    if bo.shape != (n,) or not np.isin(bo, (1, 3, 5)).all():
        raise ValueError("best_of must contain one legal format per row")
    identity = model_identity("C0-D-Core-17T")
    request = (dict(batch), bo.astype(np.int64), mode, identity)
    with tempfile.TemporaryDirectory(prefix="c0-edge-17t-inference-") as temporary:
        source = Path(temporary) / "input.pkl"
        destination = Path(temporary) / "output.pkl"
        source.write_bytes(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        completed = subprocess.run(
            [sys.executable, "-m", "src.models.c0_edge_model_audit", "--legacy17-worker",
             str(source), str(destination)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Saved historical 17-token inference failed: {completed.stderr.strip()}")
        result = np.asarray(pickle.loads(destination.read_bytes()), dtype=np.float64)
    if result.shape != (n,) or not np.isfinite(result).all() or np.any((result <= 0) | (result >= 1)):
        raise ValueError("Saved historical 17-token inference returned invalid probabilities")
    return result


def _legacy17_worker(source: str | Path, destination: str | Path) -> None:
    import joblib
    import pandas as pd
    import torch

    batch, best_of, mode, identity = pickle.loads(Path(source).read_bytes())
    live_identity = model_identity("C0-D-Core-17T")
    if identity["artifact_identity"] != live_identity["artifact_identity"]:
        raise ValueError("Historical 17-token weights/calibration changed during inference")
    from src.models import a0_capacity_experiment as cap_exp
    from src.models import a0_data_focus as focus
    from src.models import a0_native_refit as native
    from src.models import a0_target_experiment as target
    from src.models import a0_uncertainty
    root = _path((PROJECT_ROOT / "data/research_root.txt").read_text().strip())
    backend, _ = a0_uncertainty._activate_archive_backend(root)
    torch.set_num_threads(2)
    model_root = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
    shared = joblib.load(model_root / "shared/model.joblib")
    reference = joblib.load(model_root / "control/calibration/model.joblib")
    models = []
    for seed in (20260910, 20260911, 20260912):
        model = cap_exp.ParametricNonlinearHistoryModel(
            shared["baseline"], shared["normalizer"], width=32, seed=seed,
            epochs=16, coefficient=0.1,
        )
        state = torch.load(identity["checkpoint_paths"][str(seed)], map_location="cpu", weights_only=False)
        model.network.load_state_dict(state["network_state_dict"], strict=True)
        model.eta = torch.nn.Parameter(state["eta"])
        model.target_scaler = state["target_scaler"]
        model.network.eval()
        models.append(model)
    recent = {
        **batch["history"],
        "history": batch["history"]["history"][:, :, :, -16:],
        "mask": batch["history"]["mask"][:, :, :, -16:],
    }
    data = {"frame": pd.DataFrame({"best_of": best_of}), **batch, "history": recent}
    logits, _, _ = native._calculate_forecast(models, data, np.arange(len(best_of)), backend)
    other_logits = focus._other_logits(shared, batch["raw"], recent, best_of)
    beta = identity["calibration"]["beta_parameters"]
    calibration = {
        "beta_parameters": np.asarray([[item["scale"], item["rho"]] for item in beta]),
        "slopes": reference["slopes"],
        "mixtures": reference["mixtures"],
    }
    predictions, _, _ = target._predict_mixture(
        backend, logits, other_logits, calibration["slopes"], batch["raw"]["gates"],
        best_of, calibration,
    )
    values = np.asarray(predictions[:, MODES.index(mode)], dtype=np.float64)
    Path(destination).write_bytes(pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL))

def _slim_worker(source: str | Path, destination: str | Path) -> None:
    import joblib
    import pandas as pd
    import torch

    batch, best_of, mode, identity = pickle.loads(Path(source).read_bytes())
    live_identity = model_identity("C0-Slim")
    if identity["artifact_identity"] != live_identity["artifact_identity"]:
        raise ValueError("Saved Slim weights/calibration changed during inference")
    seed = int(next(iter(identity["checkpoint_sha256"])))
    checkpoint_path = Path(identity["checkpoint_paths"][str(seed)])
    if _sha256(checkpoint_path) != identity["checkpoint_sha256"][str(seed)]:
        raise ValueError("Saved Slim checkpoint changed before inference")
    from scripts.train_c0_year_anchor_distilled import YearAnchorNonlinearHistoryModel
    from src.models import a0_uncertainty
    from src.models import a0_data_focus as focus
    from src.models import a0_native_refit as native
    from src.models import a0_target_experiment as target

    root = _path((PROJECT_ROOT / "data/research_root.txt").read_text().strip())
    backend, _ = a0_uncertainty._activate_archive_backend(root)
    torch.set_num_threads(2)
    model_root = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
    shared = joblib.load(model_root / "shared/model.joblib")
    reference = joblib.load(model_root / "control/calibration/model.joblib")
    model = YearAnchorNonlinearHistoryModel(
        shared["baseline"], shared["normalizer"], width=32, seed=seed, epochs=16,
        coefficient=0.1,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.network.load_state_dict(state["network_state_dict"], strict=True)
    model.eta = torch.nn.Parameter(state["eta"])
    model.target_scaler = state["target_scaler"]
    model.network.eval()
    data = {"frame": pd.DataFrame({"best_of": best_of}), **batch}
    # The saved Slim artifact is one trained member, but native calibration
    # expects a fixed three-member axis. Run it once and replicate its logits.
    z, _, _ = native._calculate_forecast([model] * 3, data, np.arange(len(best_of)), backend)
    recent = {
        **batch["history"],
        "history": batch["history"]["history"][:, :, :, -16:],
        "mask": batch["history"]["mask"][:, :, :, -16:],
    }
    other_logits = focus._other_logits(shared, batch["raw"], recent, best_of)
    beta = identity["calibration"]["beta_parameters"]
    calibration = {
        "beta_parameters": np.asarray([[item["scale"], item["rho"]] for item in beta]),
        "slopes": reference["slopes"],
        "mixtures": reference["mixtures"],
    }
    predictions, _, _ = target._predict_mixture(
        backend, z, other_logits, calibration["slopes"], batch["raw"]["gates"],
        best_of, calibration,
    )
    values = np.asarray(predictions[:, MODES.index(mode)], dtype=np.float64)
    Path(destination).write_bytes(pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL))

def _native_query_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: item[key]
        for key in (
            "team1_id", "team2_id", "roster_a", "roster_b", "best_of",
            "decision_at", "start_at", "competition_context", "mode",
        )
    }
    for clock in ("decision_at", "start_at"):
        value = payload[clock]
        if hasattr(value, "isoformat"):
            payload[clock] = value.isoformat()
    return payload

def _query_worker(source: str | Path, destination: str | Path) -> None:
    """Reconstruct frozen raw history and native snapshots in a clean process."""
    requests, research_root, supplied_records = pickle.loads(Path(source).read_bytes())
    if not requests:
        raise ValueError("At least one exact prior-roster query is required")
    from betting_app.core.models.native_c0 import NativeC0Adapter

    validator = NativeC0Adapter()
    for item in requests:
        validator._validate_request(_native_query_payload(item))

    if supplied_records is None:
        records, receipt = _raw_map_records(research_root)
    else:
        records = supplied_records
        receipt = {"source_kind": "caller_supplied_readonly_raw_history"}
    query = build_causal_query_batches(
        requests, records, prior_source_receipt=receipt,
        native_adapter_class=NativeC0Adapter,
    )
    Path(destination).write_bytes(pickle.dumps(query, protocol=pickle.HIGHEST_PROTOCOL))

def _build_causal_query_batches_isolated(
    requests: Sequence[Mapping[str, Any]],
    *,
    research_root: str | Path | None = None,
    records_by_player: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not requests:
        raise ValueError("At least one exact prior-roster query is required")
    with tempfile.TemporaryDirectory(prefix="c0-edge-query-") as temporary:
        source = Path(temporary) / "input.pkl"
        destination = Path(temporary) / "output.pkl"
        source.write_bytes(pickle.dumps(
            (list(requests), research_root, records_by_player),
            protocol=pickle.HIGHEST_PROTOCOL,
        ))
        environment = dict(os.environ)
        environment["BETTING_APP_C0_WORKER"] = "1"
        completed = subprocess.run(
            [sys.executable, "-m", "src.models.c0_edge_model_audit",
             "--query-worker", str(source), str(destination)],
            cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Isolated native query worker failed: {completed.stderr.strip()}")
        return pickle.loads(destination.read_bytes())




def _raw_map_records(research_root: str | Path | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Reconstruct raw released player-map vectors once for arbitrary queries."""
    import importlib.util

    from src.analysis import c0_emea_reconstruction as reconstruction
    from src.analysis import ev_asof_replay as replay

    root = replay._resolve_research_root(research_root)
    bank = root / replay.BANK_RELATIVE
    raw_path = PROJECT_ROOT / replay.RAW_HISTORY_RELATIVE
    release_path = bank / "release_days.json"
    raw_hash = _sha256(raw_path)
    release_hash = _sha256(release_path)
    if raw_hash != replay.BANK_SOURCE_SHA256 or release_hash != replay.BANK_RELEASE_SHA256:
        raise ValueError("Raw history or corrected release map differs from frozen source")
    releases = _read_json(release_path)

    # Match the anchor-bank builder's isolated archived ratings import, rather
    # than accidentally using a newer application export.
    ratings_root = root / "unified-ratings-20260911/code/src/ratings"
    archived_manifest = _read_json(root / "unified-ratings-20260911/manifest.json")
    expected_init = archived_manifest["files"]["code/src/ratings/__init__.py"]["sha256"]
    init_path = ratings_root / "__init__.py"
    if os.environ.get("BETTING_APP_C0_WORKER") != "1":
        raise RuntimeError("_raw_map_records must run inside the isolated query worker")

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

    records_by_player: dict[str, list[dict[str, Any]]] = {}
    for player_id, entries in histories.items():
        records = []
        for entry in entries:
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
            records.append({
                "map_id": str(entry["source_map_id"]),
                "series_id": str(entry["source_series_id"]),
                "player_id": str(player_id),
                "played_day": _day_ordinal(entry["played_day"]),
                "release_day": _day_ordinal(entry["effective_release_day"]),
                "vector": vector,
            })
        records.sort(key=lambda value: (value["played_day"], value["map_id"]))
        records_by_player[str(player_id)] = records
    if _sha256(raw_path) != raw_hash or _sha256(release_path) != release_hash:
        raise ValueError("Protected raw history changed during query-map reconstruction")
    receipt = {
        "raw_history_sha256": raw_hash,
        "release_map_sha256": release_hash,
        "raw_series_reconstructed": len(series),
        "player_count": len(records_by_player),
        "map_record_count": sum(map(len, records_by_player.values())),
        "feature_backend_hashes": dict(sorted(backend_hashes.items())),
        "source_audit": audit,
        "source_kind": "frozen_raw_history_and_corrected_release_map",
    }
    return records_by_player, receipt


def _verify_prior_roster_source(request: Mapping[str, Any]) -> None:
    from src.analysis import ev_asof_replay as replay

    cutoff = _day_ordinal(str(request["decision_at"])[:10])
    target_id = str(request.get("canonical_match_id", request.get("row_id", "")))
    context = replay._canonical_competition_context(request.get("competition_context"))
    for side in ("a", "b"):
        roster = list(map(str, request[f"roster_{side}"]))
        source = request.get(f"prior_source_{side}")
        if not isinstance(source, Mapping):
            raise ValueError(f"Missing exact prior roster provenance for side {side}")
        source_id = str(source.get("source_match_id") or "")
        release = _day_ordinal(source.get("effective_release_day", ""))
        source_context = replay._canonical_competition_context(
            source.get("competition_context", source.get("source_tournament_name", source.get("tournament_name")))
        )
        source_roster = sorted(replay._event_roster(source))
        source_team = source.get("team_id")
        expected_team = request["team1_id" if side == "a" else "team2_id"]
        if source_team is not None and str(source_team) != str(expected_team):
            raise ValueError(f"Roster {side} provenance belongs to the wrong native team")
        if (
            not source_id
            or source_id == target_id
            or release >= cutoff
            or source_context != context
            or source_roster != sorted(roster)
        ):
            raise ValueError(f"Roster {side} is not an exact-context, strictly prior source roster")


def build_causal_query_batches(
    requests: Sequence[Mapping[str, Any]],
    records_by_player: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    prior_source_receipt: Mapping[str, Any] | None = None,
    native_adapter_class: Any | None = None,
) -> dict[str, Any]:
    """Build genuine native raw/history and 18t/20t batches at exact query clocks.

    Each request must carry explicit prior-roster source records. Native C0's
    offline raw-history adapter is asked for the exact decision day; saved
    fixture snapshots are not joined by fixture ID alone.
    """
    if not requests:
        raise ValueError("At least one exact prior-roster query is required")
    if native_adapter_class is None:
        from betting_app.core.models.native_c0 import NativeC0Adapter

        native_adapter_class = NativeC0Adapter
    from src.analysis import ev_asof_replay as replay
    from src.models import a0_data_focus

    ordered = sorted(
        requests,
        key=lambda value: (str(value["decision_at"]), str(value["canonical_match_id"])),
    )
    if len({str(value["opportunity_id"]) for value in ordered}) != len(ordered):
        raise ValueError("Causal model query IDs must be unique")
    if __import__("os").environ.get("BETTING_APP_C0_WORKER") != "1":
        raise RuntimeError("Causal native queries require the isolated --query-worker process")
    adapter = native_adapter_class()
    adapter.load()
    _, _, _, prepared = adapter._load_history()
    snapshots = []
    proofs = []
    for item in ordered:
        _verify_prior_roster_source(item)
        request = adapter._validate_request(_native_query_payload(item))
        adapter._advance_history(request["decision_at"].date(), prepared)
        target, raw, history, provenance = adapter._raw_snapshot(request)
        cutoff = _day_ordinal(target["cutoff_day"])
        if target.get("feature_source_max_day"):
            if _day_ordinal(target["feature_source_max_day"]) >= cutoff:
                raise ValueError("Native query contains a feature released on/after its decision day")
        if target.get("roster_a") != request["roster_a"] or target.get("roster_b") != request["roster_b"]:
            raise ValueError("Native query roster differs from the explicit prior roster request")
        for side in ("a", "b"):
            for field in (
                f"tensor_player_order_{side}",
                f"raw_tensor_player_order_{side}",
                f"temporal_tensor_player_order_{side}",
            ):
                if target.get(field) != request[f"roster_{side}"]:
                    raise ValueError("Native tensor player slots differ from prior roster anchor slots")
        tensors = {**raw, **history}
        tensor_hashes = {
            name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in sorted(tensors.items())
        }
        expected_hashes = item.get("native_input_hashes")
        if expected_hashes is not None:
            expected_hashes = _decode_json_value(expected_hashes)
            if not isinstance(expected_hashes, Mapping) or dict(expected_hashes) != tensor_hashes:
                raise ValueError("Reconstructed native input differs from sealed eight-tensor hashes")
        anchors = build_prior_roster_anchors(
            records_by_player,
            roster_a=request["roster_a"],
            roster_b=request["roster_b"],
            decision_day=cutoff,
        )
        snapshots.append((item, request, target, raw, history, anchors))
        proofs.append({
            "opportunity_id": str(item["opportunity_id"]),
            "canonical_match_id": str(item["canonical_match_id"]),
            "decision_day": target["cutoff_day"],
            "feature_source_max_day": target.get("feature_source_max_day"),
            "prior_source_a": replay._json_safe(item["prior_source_a"]),
            "prior_source_b": replay._json_safe(item["prior_source_b"]),
            "native_input_hashes": tensor_hashes,
            "native_history_provenance": replay._json_safe(provenance),
            "anchor_counts": {
                str(horizon): anchors[f"count_{horizon}"].tolist() for horizon in HORIZONS
            },
        })

    raw_batch = {
        name: np.stack([snapshot[3][name] for snapshot in snapshots])
        for name in snapshots[0][3]
    }
    history_batch = {
        name: np.stack([snapshot[4][name] for snapshot in snapshots])
        for name in snapshots[0][4]
    }
    best_of = np.asarray([snapshot[1]["best_of"] for snapshot in snapshots], dtype=np.int64)
    c0 = a0_data_focus.predict_models(
        adapter._backend, adapter._shared, adapter._models, adapter._calibration,
        raw_batch, history_batch, best_of,
    )
    native_modes = np.asarray(c0["full"], dtype=np.float64)
    if native_modes.shape != (len(snapshots), len(MODES)):
        raise ValueError(f"Native C0 returned an invalid context-mode matrix: {native_modes.shape}")
    p_c0 = native_modes[:, MODES.index("full")]
    if p_c0.shape != (len(snapshots),) or not np.isfinite(p_c0).all() or np.any((p_c0 <= 0) | (p_c0 >= 1)):
        raise ValueError("Native C0 query predictions are malformed")

    def candidate_batch(candidate: str, dtype: Any | None = None) -> dict[str, Any]:
        horizons = (365,) if candidate == "18t" else (365, 180, 90)
        history_parts = []
        mask_parts = []
        for horizon in horizons:
            history_parts.append(np.stack([snapshot[5][f"history_{horizon}"] for snapshot in snapshots]))
            mask_parts.append(np.stack([snapshot[5][f"mask_{horizon}"] for snapshot in snapshots]))
        history_parts.append(history_batch["history"])
        mask_parts.append(history_batch["mask"])
        combined_history = np.concatenate(history_parts, axis=3)
        if dtype is not None:
            combined_history = combined_history.astype(dtype, copy=False)
        return {
            "candidate": candidate,
            "row_ids": np.asarray([str(item["canonical_match_id"]) for item, *_ in snapshots]),
            "metadata": __import__("pandas").DataFrame({"best_of": best_of}),
            "raw": raw_batch,
            "history": {
                **history_batch,
                "history": combined_history,
                "mask": np.concatenate(mask_parts, axis=3),
            },
            "best_of": best_of,
        }

    return {
        "requests": ordered,
        "batches": {
            "18t": candidate_batch("18t"),
            "18t_legacy": candidate_batch("18t", np.float16),
            "20t": candidate_batch("20t"),
        },
        "p_c0": p_c0,
        "c0_modes": c0,
        "query_proofs": proofs,
        "raw_map_receipt": dict(prior_source_receipt or {}),
    }


def infer_saved_candidates(query: Mapping[str, Any], *, model_protocol_path: str | Path = FRESH_PROTOCOL) -> dict[str, np.ndarray]:
    """Exercise all genuine saved model configurations on one shared causal query batch."""
    from src.models.c0_distilled_replay import load_research_bundle, predict_research_batch
    from src.models.c0_distilled_reproduction import infer

    batch18 = query["batches"]["18t"]
    batch18_legacy = query["batches"]["18t_legacy"]
    batch20 = query["batches"]["20t"]
    best_of = batch18["best_of"]
    result = {"C0": np.asarray(query["p_c0"], dtype=np.float64)}
    legacy = load_research_bundle(model_protocol_path, candidate="18t")
    result["C0-D-Core"] = predict_research_batch(
        legacy, batch18_legacy, best_of=best_of, mode="full"
    )
    result["C0-D-Core-17T"] = infer_saved_17t(batch18, best_of=best_of, mode="full")
    result["C0-Slim"] = infer_saved_slim(batch18_legacy, best_of=best_of, mode="full")
    result["fresh18"] = np.asarray(
        infer(model_protocol_path, "18t", batch18, best_of, mode="full"), dtype=np.float64
    )
    result["fresh20"] = np.asarray(
        infer(model_protocol_path, "20t", batch20, best_of, mode="full"), dtype=np.float64
    )
    expected = len(result["C0"])
    for model, values in result.items():
        if values.shape != (expected,) or not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
            raise ValueError(f"Saved candidate {model} returned malformed probabilities")
    return result



def score_model_alphas(
    predictions: Mapping[str, Sequence[float]],
    market_probability: Sequence[float],
    outcomes: Sequence[int],
    *,
    fixture_weights: Sequence[float] | None = None,
    fixture_ids: Sequence[Any] | None = None,
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 1.0),
) -> list[dict[str, Any]]:
    """Proper weighted LL/Brier of saved forecasts and frozen market alphas."""
    from scipy.special import expit

    y = np.asarray(outcomes, dtype=np.int8)
    market = np.asarray(market_probability, dtype=np.float64)
    if y.ndim != 1 or market.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("Model scoring requires matched binary fixtures and market probabilities")
    if not np.isfinite(market).all() or np.any((market <= 0) | (market >= 1)):
        raise ValueError("Market probabilities must be strict finite probabilities")
    weights = np.ones(len(y), dtype=np.float64) if fixture_weights is None else np.asarray(fixture_weights, dtype=np.float64)
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Every matched scoring row requires a positive finite fixture weight")
    unique_fixture_count = len(y) if fixture_ids is None else len(set(map(str, fixture_ids)))
    if fixture_ids is not None:
        ids = np.asarray(list(map(str, fixture_ids)), dtype=object)
        if ids.shape != y.shape:
            raise ValueError("fixture_ids must match the scoring vectors")
        for fixture_id in set(ids.tolist()):
            if not np.isclose(weights[ids == fixture_id].sum(), 1.0, atol=1e-12, rtol=0):
                raise ValueError(f"Focal-row weights do not sum to one for fixture {fixture_id}")
            if len(set(y[ids == fixture_id].tolist())) != 1:
                raise ValueError(f"Conflicting outcomes for fixture {fixture_id}")
    weight_total = float(weights.sum())
    logit_market = np.log(np.clip(market, 1e-6, 1 - 1e-6) / (1 - np.clip(market, 1e-6, 1 - 1e-6)))
    results = []
    for name, raw in predictions.items():
        p = np.asarray(raw, dtype=np.float64)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
            raise ValueError(f"{name} probabilities are not exactly matched to outcomes")
        logit_model = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        for alpha in alpha_grid:
            if not 0 <= float(alpha) <= 1:
                raise ValueError("Alpha grid must stay in [0,1]")
            mixture = expit(float(alpha) * logit_model + (1.0 - float(alpha)) * logit_market)
            loss = -np.where(y == 1, np.log(mixture), np.log1p(-mixture))
            results.append({
                "model": name,
                "sports_alpha": float(alpha),
                "matched_row_count": int(len(y)),
                "matched_fixture_count": int(unique_fixture_count),
                "fixture_weight_sum": weight_total,
                "log_loss": float(np.dot(weights, loss) / weight_total),
                "brier": float(np.dot(weights, (mixture - y) ** 2) / weight_total),
            })
    return results


def _query_signature(
    canonical_match_id: Any,
    decision_at: Any,
    roster_a: Sequence[Any],
    roster_b: Sequence[Any],
    prior_source_a: Mapping[str, Any],
    prior_source_b: Mapping[str, Any],
) -> str:
    import pandas as pd

    roster_a = _decode_json_value(roster_a)
    roster_b = _decode_json_value(roster_b)
    prior_source_a = _decode_json_value(prior_source_a)
    prior_source_b = _decode_json_value(prior_source_b)
    if (
        not isinstance(roster_a, (list, tuple))
        or not isinstance(roster_b, (list, tuple))
        or not isinstance(prior_source_a, Mapping)
        or not isinstance(prior_source_b, Mapping)
    ):
        raise ValueError("Exact model query signature requires decoded native rosters and sources")
    decision = pd.to_datetime(decision_at, utc=True, errors="raise").isoformat()
    return _canonical_hash({
        "canonical_match_id": str(canonical_match_id),
        "decision_at": decision,
        "prior_roster_a": list(map(str, roster_a)),
        "prior_roster_b": list(map(str, roster_b)),
        "prior_source_a": str(prior_source_a.get("source_match_id")),
        "prior_source_b": str(prior_source_b.get("source_match_id")),
    })


def build_prior_query_requests(
    rows: Any, *, group_by_decision: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse to fixture and, when requested, one query per UTC-day state."""
    import pandas as pd

    if "canonical_match_id" not in rows.columns:
        raise ValueError("Model cohort lacks canonical_match_id")
    required = {
        "opportunity_id", "decision_at", "start_at", "team1_id", "team2_id",
        "canonical_a_is_team1", "prior_roster_orientation", "best_of",
        "prior_roster_a", "prior_roster_b", "prior_source_a", "prior_source_b",
    }
    missing_columns = sorted(required - set(rows.columns))
    if missing_columns:
        raise ValueError(
            "Model cohort is missing required native prior-query columns: "
            + ", ".join(missing_columns)
        )
    requests: list[dict[str, Any]] = []
    absence: list[dict[str, Any]] = []
    grouped_rows = rows
    if group_by_decision:
        grouped_rows = rows.copy()
        grouped_rows["_edge_query_utc_day"] = pd.to_datetime(
            grouped_rows["decision_at"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        group_columns = ["canonical_match_id", "_edge_query_utc_day"]
    else:
        group_columns = ["canonical_match_id"]
    for _, group in grouped_rows.groupby(group_columns, sort=True, dropna=False):
        match_id = group.iloc[0]["canonical_match_id"]
        first = group.sort_values(["decision_at", "opportunity_id"], kind="mergesort").iloc[0].copy()
        for name in ("prior_roster_a", "prior_roster_b", "prior_source_a", "prior_source_b",
                     "native_input_hashes"):
            if name in first:
                first[name] = _decode_json_value(first[name])
        if "native_input_hashes" in group:
            hash_values = [
                _decode_json_value(value)
                for value in group["native_input_hashes"].tolist()
                if value is not None and not (isinstance(value, float) and np.isnan(value))
            ]
            if hash_values and any(value != hash_values[0] for value in hash_values[1:]):
                absence.append({
                    "canonical_match_id": str(match_id),
                    "reason": "native_input_hashes_inconsistent",
                })
                continue
            if hash_values and not isinstance(first.get("native_input_hashes"), Mapping):
                first["native_input_hashes"] = hash_values[0]
        orientation = first["prior_roster_orientation"]
        if orientation != "native_team1_team2":
            absence.append({
                "canonical_match_id": str(match_id),
                "reason": "prior_roster_orientation_invalid",
            })
            continue
        request_id = str(first["opportunity_id"])
        roster_a, roster_b = first["prior_roster_a"], first["prior_roster_b"]
        source_a, source_b = first["prior_source_a"], first["prior_source_b"]
        if not isinstance(source_a, Mapping) or not isinstance(source_b, Mapping):
            absence.append({"canonical_match_id": str(match_id), "reason": "prior_roster_source_missing"})
            continue
        from src.analysis.ev_asof_replay import _valid_roster
        if (
            not _valid_roster(roster_a) or not _valid_roster(roster_b)
            or set(map(str, roster_a)) & set(map(str, roster_b))
        ):
            absence.append({"canonical_match_id": str(match_id), "reason": "prior_roster_invalid"})
            continue
        is_a_team1 = first["canonical_a_is_team1"]
        if not isinstance(is_a_team1, (bool, np.bool_)):
            absence.append({"canonical_match_id": str(match_id), "reason": "side_orientation_missing"})
            continue
        context = first.get("competition_context")
        if not isinstance(context, str) or not context.strip():
            context = source_a.get(
                "competition_context",
                source_a.get("source_tournament_name", source_a.get("tournament_name")),
            )
        source_context_a = source_a.get(
            "competition_context", source_a.get("source_tournament_name", source_a.get("tournament_name"))
        )
        source_context_b = source_b.get(
            "competition_context", source_b.get("source_tournament_name", source_b.get("tournament_name"))
        )
        from src.analysis.ev_asof_replay import _canonical_competition_context
        if (
            not isinstance(context, str)
            or not context.strip()
            or _canonical_competition_context(source_context_a) != _canonical_competition_context(context)
            or _canonical_competition_context(source_context_b) != _canonical_competition_context(context)
        ):
            absence.append({"canonical_match_id": str(match_id), "reason": "prior_roster_context_mismatch"})
            continue
        team1_id = str(first["team1_id"]).strip()
        team2_id = str(first["team2_id"]).strip()
        try:
            best_of = int(first["best_of"])
            decision = pd.to_datetime(first["decision_at"], utc=True, errors="raise")
            start = pd.to_datetime(first["start_at"], utc=True, errors="raise")
        except (TypeError, ValueError, OverflowError):
            absence.append({"canonical_match_id": str(match_id), "reason": "native_query_clock_or_format_invalid"})
            continue
        if (
            not team1_id or not team2_id or team1_id.lower() in {"none", "nan"}
            or team2_id.lower() in {"none", "nan"} or team1_id == team2_id
            or isinstance(first["best_of"], (bool, np.bool_)) or best_of not in (1, 3, 5)
            or pd.isna(decision) or pd.isna(start) or decision >= start
        ):
            absence.append({"canonical_match_id": str(match_id), "reason": "native_query_clock_or_format_invalid"})
            continue
        try:
            signature_decision = (
                pd.to_datetime(first["decision_at"], utc=True).date().isoformat()
                if group_by_decision else first["decision_at"]
            )
            signature = _query_signature(
                match_id, signature_decision, roster_a, roster_b, source_a, source_b
            )
            consistent = all(
                _query_signature(
                    match_id, signature_decision,
                    _decode_json_value(row["prior_roster_a"]),
                    _decode_json_value(row["prior_roster_b"]),
                    _decode_json_value(row["prior_source_a"]),
                    _decode_json_value(row["prior_source_b"]),
                ) == signature
                and (
                    not group_by_decision
                    or pd.to_datetime(row["decision_at"], utc=True).date()
                    == pd.to_datetime(first["decision_at"], utc=True).date()
                )
                and isinstance(row["canonical_a_is_team1"], (bool, np.bool_))
                and bool(row["canonical_a_is_team1"]) == bool(is_a_team1)
                and str(row["team1_id"]) == str(first["team1_id"])
                and str(row["team2_id"]) == str(first["team2_id"])
                and row["prior_roster_orientation"] == "native_team1_team2"
                and int(row["best_of"]) == int(first["best_of"])
                for _, row in group.iterrows()
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            consistent = False
        if not consistent:
            absence.append({
                "canonical_match_id": str(match_id),
                "reason": "fixture_rows_have_inconsistent_native_query_identity",
            })
            continue
        # Forecast roster/source a,b are native team1/team2 side orientation.
        # canonical_a_is_team1 changes probability orientation only.
        query_aliases = []
        seen_aliases = set()
        for _, row in group.iterrows():
            alias_roster_a = _decode_json_value(row["prior_roster_a"])
            alias_roster_b = _decode_json_value(row["prior_roster_b"])
            alias_source_a = _decode_json_value(row["prior_source_a"])
            alias_source_b = _decode_json_value(row["prior_source_b"])
            alias_key = _query_signature(
                match_id, row["decision_at"], alias_roster_a, alias_roster_b,
                alias_source_a, alias_source_b,
            )
            if alias_key in seen_aliases:
                continue
            seen_aliases.add(alias_key)
            query_aliases.append({
                "opportunity_id": str(row["opportunity_id"]),
                "decision_at": row["decision_at"],
                "prior_roster_a": list(map(str, alias_roster_a)),
                "prior_roster_b": list(map(str, alias_roster_b)),
                "prior_source_a": dict(alias_source_a),
                "prior_source_b": dict(alias_source_b),
            })
        request = {
            "opportunity_id": request_id,
            "canonical_match_id": str(match_id),
            "decision_at": first["decision_at"],
            "start_at": first["start_at"],
            "team1_id": team1_id,
            "team2_id": team2_id,
            "roster_a": list(map(str, roster_a)),
            "roster_b": list(map(str, roster_b)),
            "prior_source_a": dict(source_a),
            "prior_source_b": dict(source_b),
            "canonical_prior_roster_a": list(map(str, roster_a)),
            "canonical_prior_roster_b": list(map(str, roster_b)),
            "canonical_a_is_team1": bool(is_a_team1),
            "prior_roster_orientation": "native_team1_team2",
            "best_of": best_of,
            "native_input_hashes": first.get("native_input_hashes"),
            "competition_context": context,
            "mode": "full",
            "focal_opportunity_ids": group.opportunity_id.astype(str).tolist(),
            "query_aliases": query_aliases,
        }
        try:
            _verify_prior_roster_source(request)
        except (TypeError, ValueError, KeyError, AttributeError):
            absence.append({"canonical_match_id": str(match_id), "reason": "prior_roster_source_invalid"})
            continue
        requests.append(request)
    return requests, absence


def run_model_audit(
    protocol_path: str | Path,
    *,
    model_protocol_path: str | Path | None = None,
    model_output_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the exact matched-cohort audit and persist compact research artifacts."""
    import pandas as pd
    from src.analysis import c0_edge_alpha_audit
    safe = c0_edge_alpha_audit.json_safe_value

    protocol_path = _path(protocol_path)
    protocol = _read_json(protocol_path)
    protocol["_protocol_sha256"] = _sha256(protocol_path)
    if protocol.get("run_id") != "c0_edge_recovery_20261022_01":
        raise ValueError("Model audit only accepts the frozen C0 edge-recovery run protocol")
    model_inputs = protocol.get("inputs")
    if not isinstance(model_inputs, Mapping):
        model_inputs = {}
    model_protocol = _path(model_protocol_path or model_inputs.get("model_protocol") or FRESH_PROTOCOL)
    report_root = _path(output_root or protocol["output_root"]) / "models"
    report_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        name: report_root / name for name in (
            "matched_focal_predictions.parquet",
            "fixture_predictions.parquet",
            "daily_model_predictions.parquet",
            "model_alpha_scores.csv",
            "query_absence_ledger.json",
            "model_audit_receipt.json",
        )
    }
    existing = [str(path) for path in artifacts.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing model audit artifacts: {existing}")
    print("[model-audit] preparing frozen first-ready cohort", flush=True)
    cohort = c0_edge_alpha_audit._load_prepared(protocol)
    if cohort is None:
        cohort = c0_edge_alpha_audit.prepare_edge_cohort(protocol)
    existing = [str(path) for path in artifacts.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing model audit artifacts: {existing}")
    rows = cohort["rows"]
    daily_states = cohort["daily_native_states"].copy()
    print(f"[model-audit] focal_rows={len(rows)} fixtures={rows.canonical_match_id.nunique() if not rows.empty else 0}", flush=True)
    if "usable_for_same_day_projection" not in daily_states:
        raise ValueError("Daily native state schema lacks usability proof")
    daily_usable = daily_states.loc[daily_states.usable_for_same_day_projection.eq(True)].copy()
    daily_usable = daily_usable.rename(columns={
        "representative_opportunity_id": "opportunity_id",
        "representative_decision_at": "decision_at",
    })
    if daily_usable.empty:
        raise ValueError("No usable same-day native states are available for model projection")
    query_rows = pd.concat([rows, daily_usable], ignore_index=True, sort=False)
    print("[model-audit] rebuilding exact prior-roster native queries", flush=True)
    all_predictions, prediction_receipt = predict_model_cohort(
        query_rows,
        model_protocol_path=model_protocol,
        group_by_decision=True,
    )
    all_predictions = all_predictions.sort_values(
        ["canonical_match_id", "decision_at"], kind="mergesort"
    ).reset_index(drop=True)
    matched_rows = attach_model_probabilities(rows, all_predictions)
    fixture_predictions = (
        matched_rows.sort_values(["canonical_match_id", "decision_at"], kind="mergesort")
        .drop_duplicates("canonical_match_id", keep="first")
    )
    daily_predictions = attach_model_probabilities(daily_usable, all_predictions)
    print(
        f"[model-audit] saved_model_queries={len(all_predictions)} "
        f"daily_states={int(daily_predictions.model_query_matched.sum())}/{len(daily_predictions)} "
        f"query_absences={len(prediction_receipt.get('query_absence', []))}",
        flush=True,
    )
    if (
        rows.canonical_match_id.nunique() != 302
        or len(fixture_predictions) != 302
        or not fixture_predictions.model_query_matched.all()
    ):
        raise ValueError("All 302 known first-ready cohort fixtures must have genuine model probabilities")
    if len(daily_predictions) != 1126 or not daily_predictions.model_query_matched.all():
        raise ValueError("All 1,126 native daily query states must have genuine model probabilities")
    model_columns = (
        "p_c0", "p_c0d_core", "p_c0d_core_17t", "p_c0_slim", "p_c0d_18t", "p_c0d_20t",
        "a_c0", "b_c0", "a_c0d_core", "b_c0d_core", "a_c0d_core_17t", "b_c0d_core_17t",
        "a_c0_slim", "b_c0_slim", "a_c0d_18t", "b_c0d_18t", "a_c0d_20t", "b_c0d_20t",
    )
    for column in model_columns:
        if column not in matched_rows:
            raise ValueError(f"Model predictions are missing required probability column {column}")
    scoring = matched_rows.loc[
        matched_rows.model_query_matched
        & pd.to_numeric(matched_rows["y"], errors="coerce").isin((0, 1))
    ].copy()
    if not np.allclose(
        scoring["p_c0"].to_numpy(dtype=float),
        scoring["p_base"].to_numpy(dtype=float),
        atol=2e-6, rtol=0,
    ):
        delta = float(np.max(np.abs(
            scoring["p_c0"].to_numpy(dtype=float) - scoring["p_base"].to_numpy(dtype=float)
        )))
        raise ValueError(f"Reconstructed native C0 differs from sealed p_base (max_abs={delta})")
    if not np.allclose(
        daily_predictions["p_c0"].to_numpy(dtype=float),
        daily_predictions["p_base"].to_numpy(dtype=float),
        atol=2e-6, rtol=0,
    ):
        delta = float(np.max(np.abs(
            daily_predictions["p_c0"].to_numpy(dtype=float)
            - daily_predictions["p_base"].to_numpy(dtype=float)
        )))
        raise ValueError(f"Daily native C0 differs from sealed p_base (max_abs={delta})")
    scores = []
    partitions = (
        ("ALL", scoring),
        ("TRAIN", scoring.loc[scoring.partition.astype(str).str.upper().eq("TRAIN")]),
        ("ASSESS", scoring.loc[scoring.partition.astype(str).str.upper().eq("ASSESS")]),
    )
    for partition_name, partition_rows in partitions:
        if partition_rows.empty:
            continue
        partition_scores = score_model_alphas(
            {
                "C0": partition_rows["p_c0"].to_numpy(dtype=float),
                "C0-D-Core": partition_rows["p_c0d_core"].to_numpy(dtype=float),
                "C0-D-Core-17T": partition_rows["p_c0d_core_17t"].to_numpy(dtype=float),
                "C0-Slim": partition_rows["p_c0_slim"].to_numpy(dtype=float),
                "fresh18": partition_rows["p_c0d_18t"].to_numpy(dtype=float),
                "fresh20": partition_rows["p_c0d_20t"].to_numpy(dtype=float),
            },
            partition_rows["market_prob_a"].to_numpy(dtype=float),
            pd.to_numeric(partition_rows["y"], errors="raise").to_numpy(dtype=int),
            fixture_weights=partition_rows["fixture_weight"].to_numpy(dtype=float),
            fixture_ids=partition_rows["canonical_match_id"].astype(str).to_numpy(),
            alpha_grid=protocol["alpha_grid"],
        )
        for item in partition_scores:
            item["partition"] = partition_name
        scores.extend(partition_scores)
    query_absence = list(prediction_receipt.get("query_absence", []))
    matched_ids = set(fixture_predictions.get("canonical_match_id", pd.Series(dtype=str)).astype(str))
    for match_id in rows.canonical_match_id.astype(str).drop_duplicates():
        if match_id not in matched_ids:
            query_absence.append({"canonical_match_id": match_id, "reason": "native_query_not_matched"})
    scored_fixtures = (
        matched_rows.loc[matched_rows.model_query_matched, "canonical_match_id"].nunique()
        if not matched_rows.empty else 0
    )
    receipt = {
        "schema": "c0_edge_recovery_model_audit_v1",
        "run_id": protocol["run_id"],
        "research_only": True,
        "source_activation": False,
        "deployment": False,
        "model_protocol": str(model_protocol),
        "input_cohort_rows": int(len(rows)),
        "input_cohort_fixtures": int(rows.canonical_match_id.nunique()) if not rows.empty else 0,
        "matched_prior_roster_fixtures": int(len(fixture_predictions)),
        "matched_focal_rows": int(matched_rows.model_query_matched.sum()) if not matched_rows.empty else 0,
        "matched_scored_fixtures": int(scored_fixtures),
        "daily_native_state_count": int(len(daily_states)),
        "matched_daily_model_predictions": int(daily_predictions.model_query_matched.sum()),
        "max_abs_p_c0_vs_p_base_diff": float(np.max(np.abs(
            scoring["p_c0"].to_numpy(dtype=float) - scoring["p_base"].to_numpy(dtype=float)
        ))) if not scoring.empty else None,
        "max_abs_daily_p_c0_vs_p_base_diff": float(np.max(np.abs(
            daily_predictions["p_c0"].to_numpy(dtype=float)
            - daily_predictions["p_base"].to_numpy(dtype=float)
        ))),
        "model_query_absence_count": int(len(query_absence)),
        "scored_fixture_weight_sum": float(
            matched_rows.loc[matched_rows.model_query_matched, "fixture_weight"].sum()
        ) if not matched_rows.empty else 0.0,
        "saved_model_identities": prediction_receipt.get("model_identities", {}),
        "native_query_proofs": prediction_receipt.get("query_proofs", []),
        "raw_map_receipt": prediction_receipt.get("raw_map_receipt", {}),
        "scores": scores,
        "absence_ledger": "query_absence_ledger.json",
        "prediction_artifacts": {
            "daily_model_predictions": {
                "path": str(artifacts["daily_model_predictions.parquet"]),
                "grain": "canonical fixture and requested UTC day native state",
                "key_fields": ["canonical_match_id", "requested_utc_day", "decision_at"],
            },
            "matched_focal_predictions": {
                "path": str(artifacts["matched_focal_predictions.parquet"]),
                "grain": "first-ready focal quote row",
                "key_fields": [
                    "canonical_match_id", "decision_at", "prior_roster_a", "prior_roster_b",
                    "prior_source_a", "prior_source_b",
                ],
            },
            "probability_columns": list(model_columns),
        },
    }
    def _serializable_frame(frame: Any) -> Any:
        output = frame.copy()
        for column in output.columns:
            if output[column].dtype == object:
                output[column] = output[column].map(
                    lambda value: json.dumps(safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
                    if isinstance(value, (dict, list, tuple)) else value
                )
        return output

    _serializable_frame(matched_rows).to_parquet(artifacts["matched_focal_predictions.parquet"], index=False)
    _serializable_frame(daily_predictions).to_parquet(
        artifacts["daily_model_predictions.parquet"], index=False
    )
    _serializable_frame(fixture_predictions).to_parquet(artifacts["fixture_predictions.parquet"], index=False)
    pd.DataFrame(scores).to_csv(artifacts["model_alpha_scores.csv"], index=False)
    artifacts["query_absence_ledger.json"].write_text(
        json.dumps(safe(query_absence), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts["model_audit_receipt.json"].write_text(
        json.dumps(safe(receipt), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[model-audit] wrote matched_fixtures={len(fixture_predictions)} "
        f"scored_fixtures={scored_fixtures} to {report_root}",
        flush=True,
    )
    return {"receipt": receipt, "artifacts": {key: str(path) for key, path in artifacts.items()}}


def attach_model_probabilities(rows: Any, predictions: Any) -> Any:
    """Attach saved forecasts only on exact fixture/time/side/prior-roster keys."""
    if rows.empty:
        raise ValueError("Cannot attach model probabilities to an empty cohort")
    values = predictions.copy()
    values["_query_signature"] = [
        _query_signature(row.canonical_match_id, row.decision_at, row.prior_roster_a,
                         row.prior_roster_b, row.prior_source_a, row.prior_source_b)
        for row in values.itertuples(index=False)
    ]
    if values._query_signature.duplicated().any():
        raise ValueError("Model probability query keys must be unique")
    left = rows.copy()
    signatures = []
    for row in left.itertuples(index=False):
        try:
            signatures.append(_query_signature(
                row.canonical_match_id, row.decision_at, row.prior_roster_a,
                row.prior_roster_b, row.prior_source_a, row.prior_source_b,
            ))
        except (TypeError, ValueError, AttributeError):
            signatures.append(None)
    left["_query_signature"] = signatures
    probability_columns = [
        column for column in values.columns
        if column.startswith(("p_", "a_", "b_"))
    ]
    if len(probability_columns) != len(set(probability_columns)):
        raise ValueError("Model probability columns must be unique")
    if set(probability_columns) & set(left.columns):
        raise ValueError("Cohort already contains model probability columns")
    joined = left.merge(values[["_query_signature", *probability_columns]], on="_query_signature",
                        how="left", validate="many_to_one", indicator=True)
    joined["model_query_matched"] = joined["_merge"].eq("both")
    return joined.drop(columns=["_query_signature", "_merge"])


def predict_model_cohort(
    rows: Any,
    *,
    model_protocol_path: str | Path = FRESH_PROTOCOL,
    records_by_player: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    group_by_decision: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Run every available saved model on the same exact prior-roster queries."""
    import pandas as pd

    requests, query_absence = build_prior_query_requests(
        rows, group_by_decision=group_by_decision
    )
    if not requests:
        return pd.DataFrame(), {
            "status": "NO_MATCHED_PRIOR_ROSTER_QUERIES",
            "query_absence": query_absence,
            "matched_fixture_count": 0,
        }
    query = _build_causal_query_batches_isolated(
        requests, records_by_player=records_by_player
    )
    forecasts = infer_saved_candidates(query, model_protocol_path=model_protocol_path)
    result_rows = []
    for index, request in enumerate(query["requests"]):
        oriented = {
            name: float(value[index] if request["canonical_a_is_team1"] else 1.0 - value[index])
            for name, value in forecasts.items()
        }
        aliases = request.get("query_aliases") or [{
            "opportunity_id": request["opportunity_id"],
            "decision_at": request["decision_at"],
            "prior_roster_a": request["canonical_prior_roster_a"],
            "prior_roster_b": request["canonical_prior_roster_b"],
            "prior_source_a": request["prior_source_a"],
            "prior_source_b": request["prior_source_b"],
        }]
        for alias in aliases:
            result_rows.append({
                "opportunity_id": alias["opportunity_id"],
                "canonical_match_id": request["canonical_match_id"],
                "decision_at": alias["decision_at"],
                "prior_roster_a": alias["prior_roster_a"],
                "prior_roster_b": alias["prior_roster_b"],
                "prior_source_a": alias["prior_source_a"],
                "prior_source_b": alias["prior_source_b"],
                "p_c0": oriented["C0"],
                "p_c0d_core": oriented["C0-D-Core"],
                "p_c0d_core_17t": oriented["C0-D-Core-17T"],
                "p_c0_slim": oriented["C0-Slim"],
                "p_c0d_18t": oriented["fresh18"],
                "p_c0d_20t": oriented["fresh20"],
                "a_c0": oriented["C0"],
                "b_c0": 1.0 - oriented["C0"],
                "a_c0d_core": oriented["C0-D-Core"],
                "b_c0d_core": 1.0 - oriented["C0-D-Core"],
                "a_c0d_core_17t": oriented["C0-D-Core-17T"],
                "b_c0d_core_17t": 1.0 - oriented["C0-D-Core-17T"],
                "a_c0_slim": oriented["C0-Slim"],
                "b_c0_slim": 1.0 - oriented["C0-Slim"],
                "a_c0d_18t": oriented["fresh18"],
                "b_c0d_18t": 1.0 - oriented["fresh18"],
                "a_c0d_20t": oriented["fresh20"],
                "b_c0d_20t": 1.0 - oriented["fresh20"],
            })
    predictions = pd.DataFrame(result_rows)
    return predictions, {
        "status": "COMPLETE",
        "matched_fixture_count": int(predictions.canonical_match_id.nunique()),
        "native_query_count": int(len(query["requests"])),
        "matched_query_key_count": int(len(predictions)),
        "query_absence": query_absence,
        "query_proofs": query["query_proofs"],
        "raw_map_receipt": query["raw_map_receipt"],
        "model_identities": {
            name: model_identity(name, protocol_path=model_protocol_path)
            for name in ("C0-D-Core", "C0-D-Core-17T", "C0-Slim", "18t", "20t")
        },
    }
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim-worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--query-worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--legacy17-worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--protocol", help="Run frozen model audit from the C0 edge-recovery protocol")
    parser.add_argument("--model-protocol")
    parser.add_argument("--model-output-root")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    if args.slim_worker:
        _slim_worker(*args.slim_worker)
        return 0
    if args.legacy17_worker:
        _legacy17_worker(*args.legacy17_worker)
        return 0
    if args.query_worker:
        _query_worker(*args.query_worker)
        return 0
    if args.protocol:
        run_model_audit(
            args.protocol,
            model_protocol_path=args.model_protocol,
            model_output_root=args.model_output_root,
            output_root=args.output_root,
        )
        return 0
    parser.error("Supply --protocol or use the internal worker operation")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
