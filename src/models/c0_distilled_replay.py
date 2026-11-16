"""Research-only replay of frozen C0 distilled checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
PROTOCOL_DEFAULT = (
    "data/05_model_input/c0_distilled_verification/c0d_repro_20261021_01/protocol.json"
)
MODES = ("full", "no_w20", "no_organization")


def _protocol_path(protocol: str | Path) -> Path:
    path = Path(protocol)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _research_root() -> Path:
    root = Path((PROJECT_ROOT / "data/research_root.txt").read_text().strip())
    return (root if root.is_absolute() else PROJECT_ROOT / root).resolve()


def validate_teacher_probabilities(
    values: Any, row_ids: Sequence[Any], *, expected_bins: int = 6
) -> np.ndarray:
    probabilities = np.asarray(values)
    identifiers = np.asarray(row_ids)
    if probabilities.ndim != 2 or probabilities.shape[1] != expected_bins:
        raise ValueError(
            f"expected a (N, {expected_bins}) probability matrix; got {probabilities.shape}"
        )
    if identifiers.ndim != 1 or len(identifiers) != len(probabilities):
        raise ValueError("probability row count does not match one-dimensional row IDs")
    string_ids = [str(value) for value in identifiers.tolist()]
    if any(
        value is None or not value.strip() or value == "None" for value in string_ids
    ):
        raise ValueError("row IDs must be non-empty identifiers")
    if len(set(string_ids)) != len(string_ids):
        raise ValueError("row IDs must be unique")
    if (
        not np.issubdtype(probabilities.dtype, np.number)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError("invalid probability values")
    return probabilities


def _metadata(protocol: str | Path, candidate: str) -> dict[str, Any]:
    if candidate not in {"18t", "20t"}:
        raise ValueError("candidate must be '18t' or '20t'")
    import pandas as pd

    root = _research_root()
    bank = root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    frame = pd.read_parquet(bank / "metadata.parquet")
    if "golgg_match_id" not in frame or frame["golgg_match_id"].isna().any():
        raise ValueError("canonical metadata lacks valid target IDs")
    ids = frame["golgg_match_id"].astype(str).to_numpy()
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("canonical metadata has duplicate target IDs")
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    assess = frame.loc[dates.dt.year.eq(2026), "golgg_match_id"].astype(str).tolist()
    return {
        "candidate": candidate,
        "metadata": frame,
        "row_ids": ids,
        "assess_row_ids": assess,
        "protocol_path": str(_protocol_path(protocol)),
        "bank": str(bank),
        "research_root": str(root),
    }


def load_research_bundle(protocol: str | Path, *, candidate: str) -> dict[str, Any]:
    bundle = _metadata(protocol, candidate)
    # Keep only immutable input references and metadata in the public process.
    bank = Path(bundle["bank"])
    bundle["input_references"] = {
        "metadata": str(bank / "metadata.parquet"),
        "arrays": str(bank / "arrays.npz"),
        "players": str(bank / "players.npy"),
        "core": str(bank / "core.npy"),
        "team": str(bank / "team.npy"),
        "w20": str(bank / "w20.npy"),
        "gates": str(bank / "gates.npy"),
        "history": str(bank / "history.npy"),
        "mask": str(bank / "mask.npy"),
        "champions": str(bank / "champions.npy"),
        "year_anchor_bank": str(REPORT_ROOT / "year_anchor_bank"),
    }
    bundle["protocol"] = json.loads(Path(bundle["protocol_path"]).read_text())
    return bundle


def _parent_batch(
    protocol: str | Path, candidate: str, row_ids: Sequence[Any]
) -> dict[str, Any]:
    import pandas as pd

    root = _research_root()
    bank = root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    frame = pd.read_parquet(bank / "metadata.parquet")
    ids = frame["golgg_match_id"].astype(str).to_numpy()
    pos = {value: index for index, value in enumerate(ids)}
    requested = [str(value) for value in row_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("requested row IDs must be unique")
    try:
        index = np.asarray([pos[value] for value in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unknown research row ID: {error.args[0]}") from error
    raw = {
        name: np.load(bank / f"{name}.npy", mmap_mode="r", allow_pickle=False)[
            index
        ].copy()
        for name in ("players", "core", "team", "w20", "gates")
    }
    with np.load(bank / "arrays.npz", allow_pickle=False) as arrays:
        for key in arrays.files:
            if key not in raw:
                raw[key] = arrays[key][index].copy()
    history_original = np.load(bank / "history.npy", mmap_mode="r", allow_pickle=False)
    mask_original = np.load(bank / "mask.npy", mmap_mode="r", allow_pickle=False)
    champions = np.load(bank / "champions.npy", mmap_mode="r", allow_pickle=False)[
        index
    ].copy()
    hist_parts, mask_parts = [], []
    anchor_bank = REPORT_ROOT / "year_anchor_bank"
    for label in (
        ("year_anchor",)
        if candidate == "18t"
        else ("year_anchor", "half_year_180d", "quarter_90d")
    ):
        hist_parts.append(
            np.load(
                anchor_bank / f"{label}_history.npy", mmap_mode="r", allow_pickle=False
            )[index]
        )
        mask_parts.append(
            np.load(
                anchor_bank / f"{label}_mask.npy", mmap_mode="r", allow_pickle=False
            )[index]
        )
    history = np.concatenate([*hist_parts, history_original[index]], axis=3).copy()
    mask = np.concatenate([*mask_parts, mask_original[index]], axis=3).copy()
    metadata = frame.iloc[index].copy().reset_index(drop=True)
    return {
        "candidate": candidate,
        "metadata": metadata,
        "row_ids": np.asarray(requested, dtype=str),
        "raw": raw,
        "history": {"history": history, "mask": mask, "champions": champions},
        "protocol_path": str(_protocol_path(protocol)),
    }


def build_research_batch(
    bundle: dict[str, Any], row_ids: Sequence[Any]
) -> dict[str, Any]:
    return _parent_batch(bundle["protocol_path"], bundle["candidate"], row_ids)


def _trusted_worker(
    protocol: str | Path, input_path: str | Path, output_path: str | Path
) -> None:
    """Child-only loading of historical joblib/model objects and trusted local IPC."""
    request = pickle.loads(Path(input_path).read_bytes())
    root = _research_root()
    from src.models import a0_uncertainty as uncertainty

    backend, _ = uncertainty._activate_archive_backend(root)
    import joblib
    import torch

    torch.set_num_threads(2)
    from src.models import a0_data_focus as focus
    from src.models import a0_native_refit as native
    from src.models import a0_target_experiment as target

    candidate = request["candidate"]
    if candidate == "18t":
        from scripts.train_c0_year_anchor_distilled import (
            YearAnchorNonlinearHistoryModel as Model,
        )

        artifact = json.loads((REPORT_ROOT / "year_anchor_distilled.json").read_text())
        beta = artifact["calibration"]["d_year_core"]["beta_parameters"]
        model_name = "year_anchor_distilled"
    elif candidate == "20t":
        from scripts.train_c0_multiscale_20token import (
            MultiScale20TokenNonlinearHistoryModel as Model,
        )

        artifact = json.loads(
            (REPORT_ROOT / "multiscale_20token_distilled.json").read_text()
        )
        beta = artifact["calibration"]["beta_parameters"]
        model_name = "multiscale_20token_distilled"
    else:
        raise ValueError("invalid candidate")
    model_root = (
        PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
    )
    shared = joblib.load(model_root / "shared/model.joblib")
    reference = joblib.load(model_root / "control/calibration/model.joblib")
    calibration = {
        "beta_parameters": np.asarray([[item["scale"], item["rho"]] for item in beta]),
        "slopes": reference["slopes"],
        "mixtures": reference["mixtures"],
    }
    models = []
    for seed in (20260910, 20260911, 20260912):
        checkpoint = torch.load(
            REPORT_ROOT / "models" / model_name / f"seed_{seed}/checkpoint_epoch16.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = Model(
            shared["baseline"], shared["normalizer"], width=32, seed=seed, epochs=16
        )
        model.network.load_state_dict(checkpoint["network_state_dict"], strict=True)
        model.eta = torch.nn.Parameter(checkpoint["eta"])
        model.target_scaler = checkpoint["target_scaler"]
        model.network.eval()
        models.append(model)
    batch = request["batch"]
    bo = np.asarray(request["best_of"], dtype=np.int64)
    frame = batch["metadata"].copy(deep=True).reset_index(drop=True)
    frame["best_of"] = bo
    data = {"frame": frame, "raw": batch["raw"], "history": batch["history"]}
    z, _, _ = native._calculate_forecast(models, data, np.arange(len(bo)), backend)
    original_history = {
        **batch["history"],
        "history": batch["history"]["history"][:, :, :, -16:],
        "mask": batch["history"]["mask"][:, :, :, -16:],
    }
    other_logits = focus._other_logits(shared, batch["raw"], original_history, bo)
    predictions, _, _ = target._predict_mixture(
        backend,
        z,
        other_logits,
        calibration["slopes"],
        batch["raw"]["gates"],
        bo,
        calibration,
    )
    output = np.asarray(predictions[:, MODES.index(request["mode"])], dtype=np.float64)
    if (
        output.shape != (len(bo),)
        or not np.isfinite(output).all()
        or np.any(output <= 0)
        or np.any(output >= 1)
    ):
        raise ValueError("replay returned invalid probabilities")
    Path(output_path).write_bytes(
        pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
    )


def predict_research_batch(
    bundle: dict[str, Any], batch: dict[str, Any], *, best_of: Any, mode: str
) -> np.ndarray:
    if mode not in MODES:
        raise ValueError("invalid mode")
    count = len(batch["row_ids"])
    bo = np.asarray(best_of)
    if bo.ndim == 0:
        bo = np.full(count, bo.item())
    if (
        bo.shape != (count,)
        or not np.issubdtype(bo.dtype, np.number)
        or not np.isfinite(bo).all()
        or not np.isin(bo, (1, 3, 5)).all()
    ):
        raise ValueError("best_of must contain only 1, 3, or 5, one per row")
    if batch.get("candidate") != bundle["candidate"]:
        raise ValueError("batch candidate differs from bundle")
    request = {
        "candidate": bundle["candidate"],
        "batch": batch,
        "best_of": bo.astype(np.int64),
        "mode": mode,
    }
    with tempfile.TemporaryDirectory(prefix="c0-distilled-replay-") as directory:
        source, destination = (
            Path(directory) / "input.pkl",
            Path(directory) / "output.pkl",
        )
        source.write_bytes(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.c0_distilled_replay",
                "--worker",
                str(source),
                str(destination),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"isolated replay failed: {result.stderr.strip()}")
        values = np.asarray(pickle.loads(destination.read_bytes()), dtype=np.float64)
    if (
        values.shape != (count,)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or np.any(values >= 1)
    ):
        raise ValueError("malformed replay probabilities")
    return values


def verify_teacher_b(protocol: str | Path) -> dict[str, Any]:
    from src.models.c0_distilled_proof import verify_teacher_b as verify

    return verify(protocol)


def run_verification(protocol: str | Path) -> dict[str, Any]:
    from src.models.c0_distilled_proof import run_verification as verify

    return verify(protocol)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "--worker":
        if len(args_list) != 3:
            raise SystemExit("--worker requires trusted input and output paths")
        _trusted_worker(PROTOCOL_DEFAULT, args_list[1], args_list[2])
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=PROTOCOL_DEFAULT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bind", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--candidate18t", action="store_true")
    group.add_argument("--candidate20t", action="store_true")
    parser.add_argument("--rows", nargs="*")
    parser.add_argument("--bo", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--mode", choices=MODES, default="full")
    args = parser.parse_args(args_list)
    if args.verify:
        result = run_verification(args.protocol)
    elif args.bind:
        from src.models.c0_distilled_proof import bind_execution

        result = bind_execution(args.protocol)
    else:
        candidate = "18t" if args.candidate18t else "20t"
        bundle = load_research_bundle(args.protocol, candidate=candidate)
        rows = args.rows if args.rows is not None else bundle["assess_row_ids"]
        batch = build_research_batch(bundle, rows)
        result = predict_research_batch(
            bundle, batch, best_of=args.bo, mode=args.mode
        ).tolist()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
