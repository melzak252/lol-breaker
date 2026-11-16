"""Immutable input binding and numerical proofs for the frozen C0-D replay."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "c0d_repro_20261021_01"
PROTOCOL_DEFAULT = (
    PROJECT_ROOT
    / "data/05_model_input/c0_distilled_verification"
    / RUN_ID
    / "protocol.json"
)
REPORT_ROOT = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
MODEL_ROOT = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
TEACHER_ARCHIVE = (
    PROJECT_ROOT / "data/06_models/a0_capacity/native_capacity_20261004_01"
)


def _protocol_path(protocol: str | Path) -> Path:
    path = Path(protocol)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _research_root() -> Path:
    configured = Path((PROJECT_ROOT / "data/research_root.txt").read_text().strip())
    return (
        configured if configured.is_absolute() else PROJECT_ROOT / configured
    ).resolve()


def _input_inventory(protocol_path: Path) -> dict[str, str]:
    root = _research_root()
    bank = root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    entries: set[Path] = {
        protocol_path,
        PROJECT_ROOT / "data/research_root.txt",
        REPORT_ROOT / "year_anchor_distilled.json",
        REPORT_ROOT / "year_anchor_distilled_predictions.parquet",
        REPORT_ROOT / "multiscale_20token_distilled.json",
        REPORT_ROOT / "multiscale_20token_predictions.parquet",
        REPORT_ROOT / "teacher_probs_B.npy",
        MODEL_ROOT / "shared/model.joblib",
        MODEL_ROOT / "control/calibration/model.joblib",
    }
    entries.update({
        root / "nonlinear-history-20260912/data/03_primary/aligned_scoreline_labels.parquet",
        root / "unified-ratings-20260911/data/04_feature/roster-optional-v1/metadata.jsonl",
        *(root / "auxiliary-player-20260912/data/03_primary" / name
          for name in ("schema.json", "targets.npy", "target_mask.npy")),
    })
    for model_name in ("year_anchor_distilled", "multiscale_20token_distilled"):
        for seed in (20260910, 20260911, 20260912):
            entries.add(
                REPORT_ROOT
                / "models"
                / model_name
                / f"seed_{seed}"
                / "checkpoint_epoch16.pt"
            )
    old_anchor_files = [
        path
        for path in (REPORT_ROOT / "year_anchor_bank").glob("*.npy")
        if path.is_file()
    ]
    if len(old_anchor_files) != 7:
        raise ValueError(
            f"expected six historical anchor arrays and one count array, found {len(old_anchor_files)}"
        )
    # Bind every original canonical-bank artifact, including its completion and release metadata.
    entries.update(path for path in bank.iterdir() if path.is_file())
    # Old annual/half-year/quarter anchor arrays are inputs to historical saved candidates.
    entries.update(
        path
        for path in (REPORT_ROOT / "year_anchor_bank").glob("*.npy")
        if path.is_file()
    )
    for width, epoch in ((32, 12), (32, 24), (64, 12)):
        for seed in (20260910, 20260911, 20260912):
            entries.add(
                TEACHER_ARCHIVE / f"w{width}_seed{seed}" / f"model_epoch{epoch}.joblib"
            )
    # Original recipe scripts and the raw-bank reproduction source are frozen alongside inference code.
    entries.update(
        {
            PROJECT_ROOT / "scripts/run_track1_distillation.py",
            PROJECT_ROOT / "scripts/train_c0_year_anchor_distilled.py",
            PROJECT_ROOT / "scripts/train_c0_multiscale_20token.py",
            root / "a0-phase-walkforward-20260914/code/rebuild_corrected_bank.py",
            root / "a0-phase-walkforward-20260914/code/verify_corrected_bank.py",
            root / "a0-phase-walkforward-20260914/code/causal_predictor.py",
            PROJECT_ROOT / "src/models/c0_distilled_replay.py",
            PROJECT_ROOT / "src/models/c0_distilled_proof.py",
        }
    )
    missing = sorted(str(path) for path in entries if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "execution binding inputs are missing: " + ", ".join(missing)
        )
    return {
        str(path.relative_to(PROJECT_ROOT))
        if path.is_relative_to(PROJECT_ROOT)
        else str(path): _hash_file(path)
        for path in sorted(entries, key=lambda item: str(item))
    }


def _legacy_anchor_counts() -> dict[str, int]:
    files = sorted((REPORT_ROOT / "year_anchor_bank").glob("*.npy"))
    if len(files) != 7:
        raise ValueError(
            f"expected six historical anchor arrays and one count array, found {len(files)}"
        )
    return {
        str(path.relative_to(PROJECT_ROOT)): int(
            np.load(path, mmap_mode="r", allow_pickle=False).shape[0]
        )
        for path in files
    }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def bind_execution(protocol: str | Path) -> dict[str, Any]:
    path = _protocol_path(protocol)
    document = json.loads(path.read_text())
    if document.get("run_id") != RUN_ID:
        raise ValueError(f"unexpected C0-D run_id: {document.get('run_id')!r}")
    inventory = _input_inventory(path)
    binding = {
        "schema": "c0_distilled_legacy_execution_binding_v1",
        "run_id": RUN_ID,
        "lineage_status": "unverified_legacy_positional",
        "protocol_sha256": _hash_file(path),
        "source_sha256": inventory,
        "legacy_anchor_counts": _legacy_anchor_counts(),
    }
    binding["binding_sha256"] = _canonical_hash(binding)
    destination = path.parent / "execution_binding.json"
    encoded = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode()
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError(
                f"immutable execution binding already exists with different bytes: {destination}"
            )
    else:
        _atomic_create(destination, encoded)
    return binding


def _atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(
                    f"immutable artifact already exists with different bytes: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "copy"):
        try:
            return value.copy(deep=True)
        except TypeError:
            return value.copy()
    return value


def _arrays_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _arrays_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right, equal_nan=np.issubdtype(left.dtype, np.inexact))
    if hasattr(left, "equals"):
        return bool(left.equals(right))
    return left == right


def _saved_candidate(protocol: dict[str, Any], candidate: str):
    import pandas as pd

    name = (
        "year_anchor_distilled_predictions.parquet"
        if candidate == "18t"
        else "multiscale_20token_predictions.parquet"
    )
    column = "p_c0_d_year_core__full" if candidate == "18t" else "p_c0_d_20t_core__full"
    frame = pd.read_parquet(REPORT_ROOT / name)
    if len(frame) != 3103 or not {"golgg_match_id", "best_of", column}.issubset(
        frame.columns
    ):
        raise ValueError(
            f"saved {candidate} inference does not have the required 3103 rows/columns"
        )
    ids = frame["golgg_match_id"].astype(str).to_numpy()
    if len(set(ids.tolist())) != len(ids) or frame["best_of"].isna().any():
        raise ValueError(f"saved {candidate} inference has invalid IDs or BO values")
    return frame, ids, column


def _structural_checks(
    protocol: str | Path, candidate: str, row_id: str, second_id: str
) -> dict[str, Any]:
    from src.models.c0_distilled_replay import (
        build_research_batch,
        load_research_bundle,
        predict_research_batch,
    )

    bundle = load_research_bundle(protocol, candidate=candidate)
    original = build_research_batch(bundle, [row_id])
    pristine = _deep_copy(original)
    predictions: dict[str, dict[str, float]] = {}
    swapped = _deep_copy(original)
    for feature in swapped["raw"].values():
        if (
            isinstance(feature, np.ndarray)
            and feature.ndim >= 2
            and feature.shape[1] == 2
        ):
            feature[...] = feature[:, ::-1, ...]
    for feature in swapped["history"].values():
        if (
            isinstance(feature, np.ndarray)
            and feature.ndim >= 2
            and feature.shape[1] == 2
        ):
            feature[...] = feature[:, ::-1, ...]
    for bo in (1, 3, 5):
        predictions[str(bo)] = {}
        for mode in ("full", "no_w20", "no_organization"):
            values = predict_research_batch(bundle, original, best_of=[bo], mode=mode)
            if values.shape != (1,):
                raise ValueError(
                    f"{candidate} structural singleton returned invalid shape"
                )
            predictions[str(bo)][mode] = float(values[0])
        swapped_value = float(
            predict_research_batch(bundle, swapped, best_of=[bo], mode="full")[0]
        )
        predictions[str(bo)]["full_swapped"] = swapped_value
        if abs(predictions[str(bo)]["full"] + swapped_value - 1.0) > 2e-6:
            raise ValueError(f"{candidate} side swap does not complement for BO{bo}")
    if not _arrays_equal(original, pristine):
        raise ValueError(f"{candidate} replay mutated its batch input")
    cold = _deep_copy(original)
    for feature in cold["raw"].values():
        if not np.issubdtype(np.asarray(feature).dtype, np.number):
            raise ValueError("cold-start raw inputs must be numeric")
        feature[...] = 0
    for feature in cold["history"].values():
        if isinstance(feature, np.ndarray) and np.issubdtype(feature.dtype, np.number):
            feature[...] = 0
    for bo in (1, 3, 5):
        probability = float(
            predict_research_batch(bundle, cold, best_of=[bo], mode="full")[0]
        )
        if abs(probability - 0.5) > 2e-6:
            raise ValueError(
                f"{candidate} cold-start probability is not neutral: {probability}"
            )
    batch = build_research_batch(bundle, [row_id, second_id])
    batched = float(
        predict_research_batch(bundle, batch, best_of=[3, 3], mode="full")[0]
    )
    singleton = float(
        predict_research_batch(bundle, original, best_of=[3], mode="full")[0]
    )
    if abs(singleton - batched) > 2e-6:
        raise ValueError(f"{candidate} batch/singleton inconsistency")
    return {
        "candidate": candidate,
        "row_id": row_id,
        "bo_modes": predictions,
        "cold_start_neutral": True,
        "side_swap_complement": True,
        "batch_singleton_consistent": True,
        "input_nonmutation": True,
    }


def run_verification(protocol: str | Path) -> dict[str, Any]:
    path = _protocol_path(protocol)
    output_root = (
        PROJECT_ROOT / "data/07_model_output/c0_distilled_verification" / RUN_ID
    )
    receipt_root = PROJECT_ROOT / "data/08_reporting/c0_distilled_verification" / RUN_ID
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    diagnostic: dict[str, Any] = {
        "run_id": RUN_ID,
        "lineage_status": "unverified_legacy_positional",
        "causal_reproduction_certified": False,
        "status": "failed",
    }
    try:
        binding = verify_binding(path)
        teacher = verify_teacher_b(path)
        if (
            not teacher.get("verified")
            or teacher.get("binding_sha256") != binding["binding_sha256"]
        ):
            raise ValueError(
                "verified Teacher B proof is absent or bound to different inputs"
            )
        document = json.loads(path.read_text())
        tolerance = float(document["legacy_replay"]["probability_abs_tolerance"])
        if tolerance != 2e-6:
            raise ValueError(
                "legacy replay tolerance differs from frozen 2e-6 contract"
            )
        from src.models.c0_distilled_replay import (
            build_research_batch,
            load_research_bundle,
            predict_research_batch,
        )

        replayed: dict[str, Any] = {}
        checks = []
        for candidate, label in (
            ("18t", "18-token distilled core"),
            ("20t", "20-token distilled core"),
        ):
            saved, ids, column = _saved_candidate(document, candidate)
            bundle = load_research_bundle(path, candidate=candidate)
            if list(map(str, bundle["assess_row_ids"])) != ids.tolist():
                raise ValueError(
                    f"{candidate} public assess IDs differ from the exact saved assessment ID sequence"
                )
            batch = build_research_batch(bundle, ids.tolist())
            if not np.array_equal(
                np.asarray(batch["metadata"]["best_of"], dtype=np.int64),
                saved["best_of"].to_numpy(dtype=np.int64),
            ):
                raise ValueError(
                    f"{candidate} BO values differ from saved outputs by exact ID"
                )
            predictions = predict_research_batch(
                bundle,
                batch,
                best_of=saved["best_of"].to_numpy(dtype=np.int64),
                mode="full",
            )
            expected = saved[column].to_numpy(dtype=np.float64)
            error = float(np.max(np.abs(predictions - expected)))
            if error > tolerance:
                raise ValueError(
                    f"{candidate} saved inference mismatch max_abs_error={error:.12g}"
                )
            if len(ids) != 3103:
                raise ValueError(f"{candidate} expected 3103 saved IDs, got {len(ids)}")
            replayed[candidate] = {
                "candidate": label,
                "row_ids": ids.tolist(),
                "best_of": saved["best_of"].astype(int).tolist(),
                "probabilities": predictions.tolist(),
                "saved_probability_column": column,
                "max_abs_error": error,
                "n": len(ids),
            }
            checks.append(_structural_checks(path, candidate, str(ids[0]), str(ids[1])))
        result = {
            "run_id": RUN_ID,
            "status": "verified",
            "verified": True,
            "binding_sha256": binding["binding_sha256"],
            "teacher_proof_sha256": _hash_file(
                receipt_root / "teacher_verification.json"
            ),
            "lineage_status": "unverified_legacy_positional",
            "causal_reproduction_certified": False,
            "teacher_verification": teacher,
            "candidate_replays": replayed,
            "structural_checks": checks,
            "probability_abs_tolerance": tolerance,
        }
        out_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        probability_path = output_root / "replayed_probabilities.json"
        _atomic_create(probability_path, out_bytes)
        receipt = {
            key: value for key, value in result.items() if key != "candidate_replays"
        }
        receipt["replayed_probabilities"] = {
            "path": str(probability_path),
            "sha256": hashlib.sha256(out_bytes).hexdigest(),
        }
        rec_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
        _atomic_create(receipt_root / "verification_receipt.json", rec_bytes)
        return result
    except Exception as error:
        diagnostic.update({"error_type": type(error).__name__, "error": str(error)})
        diagnostic["diagnostic_sha256"] = _canonical_hash(diagnostic)
        _atomic_create(
            receipt_root / ("verification_failure_" + diagnostic["diagnostic_sha256"][:12] + ".json"),
            (json.dumps(diagnostic, indent=2, sort_keys=True) + "\n").encode(),
        )
        raise


def verify_binding(protocol: str | Path) -> dict[str, Any]:
    path = _protocol_path(protocol)
    binding_path = path.parent / "execution_binding.json"
    binding = json.loads(binding_path.read_text())
    if (
        binding.get("schema") != "c0_distilled_legacy_execution_binding_v1"
        or binding.get("run_id") != RUN_ID
    ):
        raise ValueError("invalid C0-D execution binding schema or run_id")
    claimed = binding.get("binding_sha256")
    unhashed = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if claimed != _canonical_hash(unhashed):
        raise ValueError("execution binding self-hash mismatch")
    if binding.get("protocol_sha256") != _hash_file(path):
        raise ValueError("frozen protocol changed after execution binding")
    expected = binding.get("source_sha256")
    actual = _input_inventory(path)
    if expected != actual:
        changed = sorted(set(expected or {}) | set(actual))
        changed = [
            name for name in changed if (expected or {}).get(name) != actual.get(name)
        ]
        raise ValueError("execution binding inputs changed: " + ", ".join(changed))
    if binding.get("legacy_anchor_counts") != _legacy_anchor_counts():
        raise ValueError("historical anchor row counts changed after execution binding")
    return {
        "verified": True,
        "binding_sha256": claimed,
        "path": str(binding_path),
        "lineage_status": "unverified_legacy_positional",
    }


def _teacher_child(protocol: Path) -> dict[str, Any]:
    root = _research_root()
    from src.models import a0_uncertainty as uncertainty
    from src.models import a0_capacity_runner as runner

    backend, _ = uncertainty._activate_archive_backend(root)
    data = uncertainty._load_training_data(root)
    frame = data["frame"]
    ids = frame["golgg_match_id"].astype(str).to_numpy()
    partition = runner._partition(
        frame, smoke=False, history=data["history"], raw=data["raw"]
    )
    train_ix = np.asarray(partition["train"], dtype=np.int64)
    train_ids = ids[train_ix]
    import joblib
    import torch

    torch.set_num_threads(2)
    from src.models import a0_data_focus as focus

    bo = frame.iloc[train_ix]["best_of"].to_numpy(dtype=np.int64)
    bo_index = (bo - 1) // 2
    checkpoint_dir = (
        PROJECT_ROOT / "data/06_models/a0_capacity/native_capacity_20261004_01"
    )
    logits: list[np.ndarray] = []
    rhos: list[float] = []
    checkpoint_hashes: dict[str, str] = {}
    for width, epoch in ((32, 12), (32, 24), (64, 12)):
        for seed in (20260910, 20260911, 20260912):
            checkpoint = (
                checkpoint_dir / f"w{width}_seed{seed}" / f"model_epoch{epoch}.joblib"
            )
            model = joblib.load(checkpoint)
            logits.append(
                np.asarray(
                    model.map_logits(
                        data["raw"], data["history"], None, train_ix, bo_index
                    )
                )
            )
            rhos.append(float(0.5 * torch.sigmoid(model.eta.detach()).cpu()))
            checkpoint_hashes[str(checkpoint.relative_to(PROJECT_ROOT))] = _hash_file(
                checkpoint
            )
    shared = joblib.load(
        PROJECT_ROOT
        / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/shared/model.joblib"
    )
    raw, history = focus._subset_inputs(data, train_ix)
    experts = [
        uncertainty._unified_logits(
            shared["rating"], raw, history, bo_index, modes=("full",)
        )[:, 0]
    ]
    experts.extend(
        uncertainty._unified_logits(model, raw, history, bo_index, modes=("full",))[
            :, 0
        ]
        for model in shared["mlps"]
    )
    cached = np.load(REPORT_ROOT / "teacher_probs_B.npy", allow_pickle=False)
    if cached.shape != (36205, 6):
        raise ValueError(
            f"cached Teacher B must have shape (36205, 6); got {cached.shape}"
        )
    if len(train_ix) != len(cached):
        raise ValueError(
            f"TRAIN partition/cached Teacher B row count mismatch: {len(train_ix)} != {len(cached)}"
        )
    replay = np.zeros_like(cached, dtype=np.float64)
    for logits_i, rho in zip(logits, rhos):
        replay += np.exp(backend["beta_terminal"].terminal_logs(logits_i, bo, rho))
    mean_rho = float(np.mean(rhos))
    for expert_logits in experts:
        replay += np.exp(
            backend["beta_terminal"].terminal_logs(expert_logits, bo, mean_rho)
        )
    replay /= 13.0
    error = float(np.max(np.abs(replay - cached)))
    if error > 2e-6:
        raise ValueError(f"Teacher B mismatch max_abs_error={error:.12g}")
    return {
        "status": "VERIFIED",
        "verified": True,
        "n": len(train_ix),
        "constituents": 13,
        "max_abs_error": error,
        "row_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "checkpoint_sha256": checkpoint_hashes,
        "cached_shape": list(cached.shape),
        "binding_sha256": verify_binding(protocol)["binding_sha256"],
    }


def verify_teacher_b(protocol: str | Path) -> dict[str, Any]:
    path = _protocol_path(protocol)
    binding = verify_binding(path)
    cached_path = PROJECT_ROOT / "data/08_reporting/c0_distilled_verification" / RUN_ID / "teacher_verification.json"
    if cached_path.is_file():
        cached = json.loads(cached_path.read_text())
        if (cached.get("binding_sha256") != binding["binding_sha256"]
                or cached.get("status") != "VERIFIED" or cached.get("n") != 36205
                or cached.get("constituents") != 13 or not np.isfinite(cached.get("max_abs_error", np.nan))
                or cached["max_abs_error"] > 2e-6):
            raise ValueError("Existing Teacher B proof does not bind these verified inputs")
        return cached
    command = [
        sys.executable,
        "-c",
        "from src.models.c0_distilled_proof import _teacher_child; import json,sys; print(json.dumps(_teacher_child(__import__('pathlib').Path(sys.argv[1]))))",
        str(path),
    ]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            "Teacher B isolated replay failed: " + completed.stderr.strip()
        )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result.get("binding_sha256") != binding["binding_sha256"]:
        raise ValueError("Teacher B proof references a different execution binding")
    destination = (
        path.parent.parent.parent.parent
        / "08_reporting"
        / "c0_distilled_verification"
        / RUN_ID
        / "teacher_verification.json"
    )
    result["lineage_status"] = "unverified_legacy_positional"
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_create(destination, encoded)
    return result
