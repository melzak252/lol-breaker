"""Fixed six-fit, research-only C0-D reproduction on a causal frozen-source bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    "data/05_model_input/c0_distilled_verification/c0d_repro_20261021_01/protocol.json"
)
LEGACY = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
SHARED = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
SEEDS = (20260910, 20260911, 20260912)
MODES = ("full", "no_w20", "no_organization")


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    )
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Immutable artifact already differs: {path}")
        return
    with path.open("x") as stream:
        stream.write(encoded)


def _stage(protocol, stage):
    return (
        PROJECT_ROOT
        / "data"
        / stage
        / "c0_distilled_verification"
        / protocol["run_id"]
        / "fresh"
    )


def _raw_manifest(protocol):
    return (
        PROJECT_ROOT
        / "data/04_feature"
        / protocol["run_id"]
        / "stage04run/manifest.json"
    )


def _verify_raw(manifest_path, protocol):
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "c0_raw_causal_anchor_bank_v2":
        raise ValueError("Raw bank has unexpected schema")
    artifacts = manifest.get("artifacts", {})
    required = ["row_ids.npy", "map_membership.json"] + [
        f"{part}_{h}.npy"
        for h in (90, 180, 365)
        for part in ("history", "mask", "count")
    ]
    for name in required:
        expected = artifacts.get(name)
        if not isinstance(expected, str):
            raise ValueError(f"Raw bank lacks hash for {name}")
        if _sha256(manifest_path.parent / name) != expected:
            raise ValueError(f"Raw bank artifact hash mismatch: {name}")
    if hashlib.sha256(
        json.dumps(artifacts, sort_keys=True).encode()
    ).hexdigest() != manifest.get("artifact_sha256"):
        raise ValueError("Raw bank aggregate hash mismatch")
    ids = np.load(manifest_path.parent / "row_ids.npy", allow_pickle=False)
    if ids.shape != (40636,) or len(set(ids.tolist())) != 40636:
        raise ValueError(
            "Raw bank row IDs do not cover the full canonical bank uniquely"
        )
    for h in (90, 180, 365):
        for part, shape in [
            ("history", (40636, 2, 5, 1, 33)),
            ("mask", (40636, 2, 5, 1)),
            ("count", (40636, 2, 5)),
        ]:
            if (
                np.load(manifest_path.parent / f"{part}_{h}.npy", mmap_mode="r").shape
                != shape
            ):
                raise ValueError("Raw bank tensor shape mismatch")
    proof_path = manifest_path.parent / "verification_receipt.json"
    proof = _load_json(proof_path)
    if (
        proof.get("status") != "VERIFIED"
        or proof.get("row_count") != 40636
        or proof.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError(
            "Raw bank independent verification does not bind this manifest"
        )
    return {
        "path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "receipt_sha256": _sha256(proof_path),
        "row_count": 40636,
        "artifacts": artifacts,
    }


def _closure(protocol_path):
    from src.models.c0_distilled_proof import verify_binding, verify_teacher_b
    from src.analysis.c0_distilled_reproduction_evaluation import reference_paths

    protocol = _load_json(protocol_path)
    raw = _verify_raw(_raw_manifest(protocol), protocol)
    legacy_binding = verify_binding(protocol_path)
    teacher = verify_teacher_b(protocol_path)
    if (
        teacher.get("status") != "VERIFIED"
        or teacher.get("constituents") != 13
        or teacher.get("n") != 36205
    ):
        raise ValueError("Full Teacher B numerical proof is required before fitting")
    files = {
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/analysis/c0_distilled_reproduction_evaluation.py",
        PROJECT_ROOT / "src/analysis/replacement_guards.py",
        PROJECT_ROOT / "src/analysis/bootstrap.py",
        protocol_path,
        _raw_manifest(protocol),
        _raw_manifest(protocol).parent / "verification_receipt.json",
        protocol_path.parent / "execution_binding.json",
    }
    files.update(_path(value) for value in reference_paths(protocol).values())
    files.update(_raw_manifest(protocol).parent / name for name in raw["artifacts"])
    files.add(
        PROJECT_ROOT
        / "data/08_reporting/c0_distilled_verification"
        / protocol["run_id"]
        / "teacher_verification.json"
    )
    return {
        "schema": "c0_distilled_fresh_execution_binding_v1",
        "status": "BOUND",
        "raw_bank": raw,
        "legacy_binding": legacy_binding["binding_sha256"],
        "teacher": teacher,
        "inputs": {str(path): _sha256(path) for path in sorted(files)},
    }


def bind(protocol_path):
    protocol_path = _path(protocol_path)
    result = _closure(protocol_path)
    _write_json(protocol_path.parent / "fresh_execution_binding.json", result)
    return result


def _ready(protocol_path):
    protocol_path = _path(protocol_path)
    expected = _load_json(protocol_path.parent / "fresh_execution_binding.json")
    if expected != _closure(protocol_path):
        raise ValueError("Fresh execution inputs or code changed after binding")
    return _load_json(protocol_path), Path(expected["raw_bank"]["path"])


def _activate():
    from src.models import a0_uncertainty

    configured = Path((PROJECT_ROOT / "data/research_root.txt").read_text().strip())
    root = _path(configured)
    backend, _ = a0_uncertainty._activate_archive_backend(root)
    return a0_uncertainty, backend, root


def _dataset(data, bank, candidate):
    horizons = (365,) if candidate == "18t" else (365, 180, 90)
    hist = [np.load(bank / f"history_{h}.npy", mmap_mode="r") for h in horizons]
    mask = [np.load(bank / f"mask_{h}.npy", mmap_mode="r") for h in horizons]
    return {
        **data,
        "history": {
            **data["history"],
            "history": np.concatenate([*hist, data["history"]["history"]], axis=3),
            "mask": np.concatenate([*mask, data["history"]["mask"]], axis=3),
        },
    }


def _load_models(candidate, directory, shared):
    import torch

    if candidate == "18t":
        from scripts.train_c0_year_anchor_distilled import (
            YearAnchorNonlinearHistoryModel as Model,
        )
    elif candidate == "20t":
        from scripts.train_c0_multiscale_20token import (
            MultiScale20TokenNonlinearHistoryModel as Model,
        )
    else:
        raise ValueError("Unknown candidate")
    models = []
    for seed in SEEDS:
        state = torch.load(
            directory / f"seed_{seed}/checkpoint_epoch16.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = Model(
            shared["baseline"],
            shared["normalizer"],
            width=32,
            seed=seed,
            epochs=16,
            coefficient=0.1,
        )
        model.network.load_state_dict(state["network_state_dict"], strict=True)
        model.eta = torch.nn.Parameter(state["eta"])
        model.target_scaler = state["target_scaler"]
        model.network.eval()
        models.append(model)
    return models


def _predict(models, batch, best_of, mode, backend, shared, reference, calibration):
    import pandas as pd
    from src.models.a0_native_refit import _calculate_forecast
    from src.models.a0_data_focus import _other_logits
    from src.models.a0_target_experiment import _predict_mixture

    if mode not in MODES:
        raise ValueError("Unknown inference mode")
    n = len(batch["raw"]["players"])
    bo = np.asarray(best_of)
    if bo.ndim == 0:
        bo = np.full(n, bo.item())
    if bo.shape != (n,) or not np.isin(bo, (1, 3, 5)).all():
        raise ValueError("best_of must contain one legal format per row")
    bo = bo.astype(np.int64)
    data = {"frame": pd.DataFrame({"best_of": bo}), **batch}
    z, _, _ = _calculate_forecast(models, data, np.arange(n), backend)
    original = {
        **batch["history"],
        "history": batch["history"]["history"][:, :, :, -16:],
        "mask": batch["history"]["mask"][:, :, :, -16:],
    }
    others = _other_logits(shared, batch["raw"], original, bo)
    params = {
        "beta_parameters": np.array(
            [[p["scale"], p["rho"]] for p in calibration["beta_parameters"]]
        ),
        "slopes": reference["slopes"],
        "mixtures": reference["mixtures"],
    }
    p, _, _ = _predict_mixture(
        backend, z, others, params["slopes"], batch["raw"]["gates"], bo, params
    )
    result = np.asarray(p[:, MODES.index(mode)])
    if (
        result.shape != (n,)
        or not np.isfinite(result).all()
        or ((result <= 0) | (result >= 1)).any()
    ):
        raise ValueError("Fresh saved inference returned invalid probabilities")
    return result


def _batch(data, indices):
    return {
        "raw": {
            key: np.asarray(value)[indices].copy() for key, value in data["raw"].items()
        },
        "history": {
            key: np.asarray(value)[indices].copy()
            for key, value in data["history"].items()
        },
    }


def _verify_candidate(
    models, dataset, assess_ix, expected, calibration, backend, shared, reference
):
    batch = _batch(dataset, assess_ix)
    bo = dataset["frame"].best_of.to_numpy()[assess_ix]
    actual = _predict(
        models, batch, bo, "full", backend, shared, reference, calibration
    )
    error = float(np.max(np.abs(actual - expected)))
    if error > 2e-6:
        raise ValueError(f"Fresh saved prediction replay mismatch: {error}")
    sample = _batch(dataset, assess_ix[:3])
    before = pickle.dumps(sample)
    checks = 0
    for format_ in (1, 3, 5):
        for mode in MODES:
            baseline = _predict(
                models, sample, format_, mode, backend, shared, reference, calibration
            )
            swap = {
                group: {key: value[:, ::-1].copy() for key, value in arrays.items()}
                for group, arrays in sample.items()
            }
            cold = {
                group: {key: np.zeros_like(value) for key, value in arrays.items()}
                for group, arrays in sample.items()
            }
            ps = _predict(
                models, swap, format_, mode, backend, shared, reference, calibration
            )
            pc = _predict(
                models, cold, format_, mode, backend, shared, reference, calibration
            )
            single = {
                group: {key: value[:1].copy() for key, value in arrays.items()}
                for group, arrays in sample.items()
            }
            p1 = _predict(
                models, single, format_, mode, backend, shared, reference, calibration
            )
            if (
                max(
                    np.max(np.abs(ps + baseline - 1)),
                    np.max(np.abs(pc - 0.5)),
                    abs(p1[0] - baseline[0]),
                )
                > 2e-6
            ):
                raise ValueError("Fresh saved model violates cold/swap/batch invariant")
            checks += 1
    if before != pickle.dumps(sample):
        raise ValueError("Saved inference mutated input tensors")
    return {
        "status": "VERIFIED",
        "rows": len(actual),
        "max_abs_error": error,
        "format_mode_checks": checks,
    }


def _child(protocol_path, operation):
    import joblib
    import torch
    import pandas as pd

    protocol, manifest = _ready(protocol_path)
    torch.set_num_threads(2)
    uncertainty, backend, root = _activate()
    from src.models.a0_capacity_runner import _partition
    from src.models.competition_tiers import classify_competition
    from src.models.c0_distilled_replay import validate_teacher_probabilities
    from scripts import (
        train_c0_year_anchor_distilled as year,
        train_c0_multiscale_20token as multi,
    )

    data = uncertainty._load_training_data(root)
    frame = data["frame"]
    frame["competition_tier"] = [
        classify_competition(name, pd.Timestamp(day).date()).tier.value
        for name, day in zip(frame.tournament_name, frame.date)
    ]
    split = _partition(frame, smoke=False, history=data["history"], raw=data["raw"])
    train, cal, assess = split["train"], split["cal"], split["validation"]
    ids = frame.golgg_match_id.astype(str).to_numpy()
    if not np.array_equal(ids, np.load(manifest.parent / "row_ids.npy")):
        raise ValueError("Fresh raw bank is not aligned to every canonical row")
    shared = joblib.load(SHARED / "shared/model.joblib")
    reference = joblib.load(SHARED / "control/calibration/model.joblib")
    teacher = validate_teacher_probabilities(
        np.load(LEGACY / "teacher_probs_B.npy"), ids[train]
    )
    prediction_path = _stage(protocol, "07_model_output") / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "golgg_match_id": ids[assess],
            "date": frame.iloc[assess].date.to_numpy(),
            "y": frame.iloc[assess].y.to_numpy(),
            "best_of": frame.iloc[assess].best_of.to_numpy(),
        }
    )
    proofs = {}
    for candidate in ("18t", "20t"):
        dataset = _dataset(data, manifest.parent, candidate)
        directory = _stage(protocol, "06_models") / candidate
        if operation == "smoke":
            legacy_name = (
                "year_anchor_distilled"
                if candidate == "18t"
                else "multiscale_20token_distilled"
            )
            models = _load_models(candidate, LEGACY / "models" / legacy_name, shared)
            old = _load_json(
                LEGACY
                / (
                    "year_anchor_distilled.json"
                    if candidate == "18t"
                    else "multiscale_20token_distilled.json"
                )
            )
            calibration = (
                old["calibration"]["d_year_core"]
                if candidate == "18t"
                else old["calibration"]
            )
            _predict(
                models,
                _batch(dataset, train[:3]),
                frame.best_of.to_numpy()[train[:3]],
                "full",
                backend,
                shared,
                reference,
                calibration,
            )
            proofs[candidate] = {"input_rows": 3, "fits": 0, "calibration_fits": 0}
            continue
        if operation == "run":
            directory.mkdir(parents=True, exist_ok=True)
            trainer = (
                year.train_or_load_year_anchor_model
                if candidate == "18t"
                else multi.train_or_load_20token_model
            )
            models = []
            for seed in SEEDS:
                checkpoint = directory / f"seed_{seed}/checkpoint_epoch16.pt"
                started = directory / f"seed_{seed}/fit_started.json"
                if started.exists() and not checkpoint.exists():
                    raise RuntimeError(
                        "An interrupted fit exists; no automatic extra fit is authorized"
                    )
                if not checkpoint.exists():
                    _write_json(
                        started,
                        {
                            "seed": seed,
                            "epochs": 16,
                            "candidate": candidate,
                            "binding_sha256": _sha256(
                                protocol_path.parent / "fresh_execution_binding.json"
                            ),
                        },
                    )
                models.append(
                    trainer(
                        32,
                        seed,
                        16,
                        0.6,
                        teacher,
                        directory / f"seed_{seed}",
                        dataset,
                        train,
                        (frame.best_of.to_numpy()[train].astype(np.int64) - 1) // 2,
                        shared,
                    )
                )
            calibration_path = directory / "calibration.json"
            if calibration_path.exists():
                calibration = _load_json(calibration_path)
                values = _predict(
                    models,
                    _batch(dataset, assess),
                    frame.best_of.to_numpy()[assess],
                    "full",
                    backend,
                    shared,
                    reference,
                    calibration,
                )
            else:
                values, _, _, calibration, _, _ = (
                    year.fit_beta_calibration_and_assemble(
                        models, dataset, data, cal, assess, backend, shared, reference
                    )
                )
                _write_json(calibration_path, calibration)
            _write_json(
                directory / "bundle_manifest.json",
                {
                    "candidate": candidate,
                    "bank_manifest_sha256": _sha256(manifest),
                    "calibration_sha256": _sha256(calibration_path),
                    "checkpoint_sha256": {
                        str(seed): _sha256(
                            directory / f"seed_{seed}/checkpoint_epoch16.pt"
                        )
                        for seed in SEEDS
                    },
                    "shared_sha256": _sha256(SHARED / "shared/model.joblib"),
                    "reference_calibration_sha256": _sha256(
                        SHARED / "control/calibration/model.joblib"
                    ),
                },
            )
            predictions["p_" + candidate] = values
        else:
            bundle = _load_json(directory / "bundle_manifest.json")
            for seed in SEEDS:
                if (
                    _sha256(directory / f"seed_{seed}/checkpoint_epoch16.pt")
                    != bundle["checkpoint_sha256"][str(seed)]
                ):
                    raise ValueError("Saved student checkpoint changed")
            if _sha256(directory / "calibration.json") != bundle["calibration_sha256"]:
                raise ValueError("Saved calibration changed")
            models = _load_models(candidate, directory, shared)
            calibration = _load_json(directory / "calibration.json")
            saved = (
                pd.read_parquet(prediction_path)
                .set_index("golgg_match_id")
                .loc[ids[assess]]
            )
            proofs[candidate] = _verify_candidate(
                models,
                dataset,
                assess,
                saved["p_" + candidate].to_numpy(),
                calibration,
                backend,
                shared,
                reference,
            )
    if operation == "run":
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if prediction_path.exists():
            raise FileExistsError("Use --verify for a completed prediction namespace")
        predictions.to_parquet(prediction_path, index=False)
        from src.analysis.c0_distilled_reproduction_evaluation import (
            evaluate_fresh_predictions,
        )

        result = evaluate_fresh_predictions(
            frame.iloc[assess].copy(), predictions, protocol
        )
        result.update(
            formal_neural_fits=6,
            base_fits=0,
            prediction_sha256=_sha256(prediction_path),
        )
        _write_json(_stage(protocol, "08_reporting") / "results.json", result)
        return result
    if operation == "verify":
        _write_json(
            _stage(protocol, "08_reporting") / "saved_inference_verification.json",
            proofs,
        )
    return proofs


def run(protocol_path, *, smoke=False):
    return _invoke(protocol_path, "smoke" if smoke else "run")


def _invoke(protocol_path, operation):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.c0_distilled_reproduction",
            "--protocol",
            str(protocol_path),
            "--worker",
            operation,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout + "\n" + completed.stderr)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def build_fresh_batch(protocol_path, candidate, row_ids):
    import pandas as pd
    protocol, manifest = _ready(_path(protocol_path))
    root = _path((PROJECT_ROOT / "data/research_root.txt").read_text().strip())
    bank = root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    frame = pd.read_parquet(bank / "metadata.parquet")
    positions = {str(value): index for index, value in enumerate(frame.golgg_match_id)}
    requested = list(map(str, row_ids))
    if candidate not in ("18t", "20t") or not requested or len(set(requested)) != len(requested):
        raise ValueError("Expected a known candidate and nonempty unique row IDs")
    try:
        indices = np.array([positions[value] for value in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown frozen input row: {error.args[0]}") from error
    horizons = (365,) if candidate == "18t" else (365, 180, 90)
    raw = {name: np.load(bank / f"{name}.npy", mmap_mode="r")[indices].copy()
           for name in ("players", "core", "team", "w20", "gates")}
    history = {}
    for name in ("history", "mask"):
        values = [np.load(manifest.parent / f"{name}_{h}.npy", mmap_mode="r")[indices] for h in horizons]
        history[name] = np.concatenate([*values, np.load(bank / f"{name}.npy", mmap_mode="r")[indices]], axis=3)
    history["champions"] = np.load(bank / "champions.npy", mmap_mode="r")[indices].copy()
    return {"raw": raw, "history": history}


def infer(protocol_path, candidate, batch, best_of, mode="full"):
    """Genuine saved series inference over caller-owned tensors, in isolation."""
    if candidate not in ("18t", "20t") or mode not in MODES:
        raise ValueError("Unknown saved candidate or mode")
    with tempfile.TemporaryDirectory(prefix="c0-fresh-inference-") as temporary:
        source, destination = Path(temporary) / "input.pkl", Path(temporary) / "output.pkl"
        source.write_bytes(pickle.dumps((candidate, batch, best_of, mode)))
        completed = subprocess.run(
            [sys.executable, "-m", "src.models.c0_distilled_reproduction", "--protocol",
             str(_path(protocol_path)), "--infer-worker", str(source), str(destination)],
            cwd=PROJECT_ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return pickle.loads(destination.read_bytes())


def _infer_worker(protocol_path, source, destination):
    import joblib
    import torch
    protocol, _ = _ready(protocol_path)
    _, backend, _ = _activate()
    torch.set_num_threads(2)
    candidate, batch, bo, mode = pickle.loads(Path(source).read_bytes())
    directory = _stage(protocol, "06_models") / candidate
    manifest = _load_json(directory / "bundle_manifest.json")
    for seed in SEEDS:
        if _sha256(directory / f"seed_{seed}/checkpoint_epoch16.pt") != manifest["checkpoint_sha256"][str(seed)]:
            raise ValueError("Saved inference weights changed")
    if _sha256(directory / "calibration.json") != manifest["calibration_sha256"]:
        raise ValueError("Saved inference calibration changed")
    shared = joblib.load(SHARED / "shared/model.joblib")
    reference = joblib.load(SHARED / "control/calibration/model.joblib")
    values = _predict(_load_models(candidate, directory, shared), batch, bo, mode, backend,
                      shared, reference, _load_json(directory / "calibration.json"))
    Path(destination).write_bytes(pickle.dumps(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("bind", "smoke", "run", "verify"):
        group.add_argument("--" + name, action="store_true")
    group.add_argument("--worker", choices=("smoke", "run", "verify"))
    group.add_argument("--infer-worker", nargs=2)
    group.add_argument("--predict", choices=("18t", "20t"))
    parser.add_argument("--row-ids", nargs="+")
    parser.add_argument("--best-of", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--mode", choices=MODES, default="full")
    args = parser.parse_args(argv)
    path = _path(args.protocol)
    if args.infer_worker:
        _infer_worker(path, *args.infer_worker)
        return 0
    if args.predict:
        if not args.row_ids:
            parser.error("--predict requires --row-ids")
        batch = build_fresh_batch(path, args.predict, args.row_ids)
        print(json.dumps(infer(path, args.predict, batch, args.best_of, args.mode).tolist()))
        return 0
    if args.bind:
        result = bind(path)
    elif args.worker:
        result = _child(path, args.worker)
    else:
        result = _invoke(
            path, "smoke" if args.smoke else "run" if args.run else "verify"
        )
    print(json.dumps(result, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
