"""Frozen long-term annual native C0 origins and target-roster replay.

This orchestration is intentionally outside the hash-pinned historical model
libraries. Native models and the unchanged annual assembly remain the authority.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "long_term_market_edge_20261022_01"
SEEDS = (20260910, 20260911, 20260912)
NEW_YEARS = (2021, 2022)
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


def _paths(year):
    base = f"data/{{stage}}/long_term_market_edge/{RUN_ID}/annual_c0/{year}"
    return {key: PROJECT_ROOT / base.format(stage=stage) for key, stage in
            (("model", "06_models"), ("output", "07_model_output"), ("report", "08_reporting"))}


def _sha256(path):
    from src.models.a0_native_refit import _sha256 as digest
    return digest(Path(path))

def _ast_module(path_or_bytes):
    import ast
    source = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return ast.parse(source)


def _compare_2023_inference_closures(feature_source_path, current_feature_path, frozen_worker_source, current_worker_path):
    import ast
    import copy
    import hashlib
    frozen_feature = _ast_module(feature_source_path)
    current_feature = _ast_module(current_feature_path)
    def functions(tree):
        return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    frozen_functions, current_functions = functions(frozen_feature), functions(current_feature)
    roots = ("_stream_history", "_feature_values")
    if any(name not in frozen_functions or name not in current_functions for name in roots):
        raise ValueError("2023 numerical feature closure is missing a required entry point")
    def closure(definitions):
        found, pending = set(), list(roots)
        while pending:
            name = pending.pop()
            if name in found or name not in definitions:
                continue
            found.add(name)
            pending.extend(
                node.func.id for node in ast.walk(definitions[name])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in found
            )
        return found
    names = closure(frozen_functions) | closure(current_functions)
    function_hashes = {}
    for name in sorted(names):
        left, right = frozen_functions.get(name), current_functions.get(name)
        if left is None or right is None:
            raise ValueError(f"2023 numerical feature closure differs in local callee {name}")
        left_ast, right_ast = ast.dump(left, include_attributes=False), ast.dump(right, include_attributes=False)
        if left_ast != right_ast:
            raise ValueError(f"2023 numerical feature closure differs in local callee {name}")
        function_hashes[name] = hashlib.sha256(left_ast.encode()).hexdigest()
    frozen_worker = _ast_module(frozen_worker_source)
    current_worker = _ast_module(current_worker_path)
    def class_methods(tree, class_name):
        cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
        if cls is None:
            raise ValueError(f"2023 worker source omits {class_name}")
        return {node.name: node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    frozen_worker_methods = class_methods(frozen_worker, "_Worker")
    current_worker_methods = class_methods(current_worker, "_Worker")
    for name in set(frozen_worker_methods) | set(current_worker_methods):
        if name not in frozen_worker_methods or name not in current_worker_methods:
            raise ValueError(f"2023 worker method set differs: {name}")
        if name != "_history_adapter" and ast.dump(frozen_worker_methods[name], include_attributes=False) != ast.dump(current_worker_methods[name], include_attributes=False):
            raise ValueError(f"2023 native worker inference method differs: {name}")
    class NormalizeFeaturePin(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            self.generic_visit(node)
            if node.module == "betting_app.core.models.native_c0":
                node.names = [alias for alias in node.names if alias.name != "_FEATURE_REPLAY_SHA256"]
            return node
        def visit_Name(self, node):
            if node.id == "_FEATURE_REPLAY_SHA256":
                return ast.copy_location(ast.Constant(value="89cea187d3dc9d95e86a7b91da0e068b24c1c8d2afb9de0bd63c5e71a71e6365"), node)
            return node
    frozen_history = ast.dump(frozen_worker_methods["_history_adapter"], include_attributes=False)
    current_history = copy.deepcopy(current_worker_methods["_history_adapter"])
    normalized_history = NormalizeFeaturePin().visit(ast.fix_missing_locations(current_history))
    if frozen_history != ast.dump(normalized_history, include_attributes=False):
        raise ValueError("2023 worker history adapter differs beyond its feature-source identity pin")
    if ast.dump(frozen_worker_methods["predict"], include_attributes=False) != ast.dump(current_worker_methods["predict"], include_attributes=False):
        raise ValueError("2023 native worker prediction implementation differs")
    frozen_provider = class_methods(frozen_worker, "ResearchNativePairProvider")
    current_provider = class_methods(current_worker, "ResearchNativePairProvider")
    if set(frozen_provider) != set(current_provider) or any(
        ast.dump(frozen_provider[name], include_attributes=False) != ast.dump(current_provider[name], include_attributes=False)
        for name in frozen_provider
    ):
        raise ValueError("2023 native provider request/dispatch implementation differs")
    return {"feature_closure_functions": function_hashes, "feature_closure_equal": True,
        "worker_predict_ast_equal": True, "provider_methods_ast_equal": True,
        "worker_methods_differ_only": ["_history_adapter"]}



def _load(protocol_path, research_root=None):
    from src.models import a0_data_focus as native
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("run_id") != RUN_ID or protocol.get("status") != "FROZEN_BEFORE_NEW_FITS_AND_OUTCOME_SCORING":
        raise ValueError("unexpected frozen long-term protocol")
    root = native.resolve_research_root(research_root)
    backend, backend_hashes = native.uncertainty._activate_archive_backend(root)
    sources = __import__("src.models.a0_reliability", fromlist=["_sources"])._sources(backend_hashes)
    data = native.uncertainty._load_training_data(root)
    inputs = native._relative_input_hashes(data["input_hashes"], root)
    if inputs.get("a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/metadata.parquet") != protocol["canonical_inputs"]["metadata_sha256"]:
        raise ValueError("canonical metadata does not match frozen study input")
    return protocol_path, protocol, root, backend, sources, data


def _partition(data, year):
    import numpy as np
    import pandas as pd
    frame = data["frame"]
    dates, released = pd.to_datetime(frame.date), pd.to_datetime(frame.result_day)
    if dates.isna().any() or released.isna().any() or (released < dates).any() or frame.golgg_match_id.duplicated().any():
        raise ValueError("invalid annual control input identities or release chronology")
    train_stop, origin, validation_stop = f"{year-1}-07-01", f"{year}-01-01", f"{year+1}-01-01"
    masks = {"train": (dates < train_stop) & (released < train_stop),
             "cal": (dates >= train_stop) & (dates < origin) & (released < origin),
             "validation": (dates >= origin) & (dates < validation_stop)}
    return {key: np.flatnonzero(mask.to_numpy()) for key, mask in masks.items()}


def _freeze(protocol_path, protocol, root, sources, data, year):
    """Create the immutable pre-fit bind; no fitting occurs before this returns."""
    import pandas as pd
    from src.models.a0_data_focus import _relative_input_hashes
    from src.models.a0_native_refit import _contract_digest, _write_json_exclusive
    split = _partition(data, year)
    if any(not len(split[key]) for key in ("train", "cal", "validation")):
        raise ValueError(f"year {year} has empty TRAIN/CAL/validation membership")
    paths = _paths(year)
    for path in paths.values():
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"annual control namespace is not empty: {path}")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    report, model = paths["report"], paths["model"]
    binding_path = report / "prefit_binding.json"
    frame = data["frame"]
    memberships = {part: _contract_digest(frame.iloc[ix].golgg_match_id.astype(str).tolist()) for part, ix in split.items()}
    counts = {part: len(ix) for part, ix in split.items()}
    member_rows = []
    for part, ix in split.items():
        rows = frame.iloc[ix][["golgg_match_id", "date", "result_day"]].copy()
        rows["partition"] = part
        member_rows.append(rows)
    membership_path = report / "partitions.parquet"
    pd.concat(member_rows, ignore_index=True).to_parquet(membership_path, index=False)
    # Seal exact orchestration source separately; the historical source contract
    # remains the original native library inventory expected by load_origin.
    source_zip = model / "sources.zip"
    with zipfile.ZipFile(source_zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, digest in sources.items():
            source = root / relative[9:] if relative.startswith("research:") else PROJECT_ROOT / relative.removeprefix("project:")
            if _sha256(source) != digest:
                raise ValueError(f"source changed while binding: {relative}")
            archive.write(source, relative.replace(":", "/", 1))
        archive.write(Path(__file__), "project/src/models/long_term_native_controls.py")
    binding = {"schema": "long_term_native_c0_prefit_binding_v1", "status": "FROZEN_BEFORE_FIT",
        "run_id": RUN_ID, "year": year, "protocol_sha256": _sha256(protocol_path),
        "wrapper_sha256": _sha256(Path(__file__)), "source_sha256": sources,
        "source_snapshot_sha256": _sha256(source_zip), "input_sha256": _relative_input_hashes(data["input_hashes"], root),
        "bank_receipt_sha256": data["bank_receipt_sha256"], "partition_counts": counts,
        "split_ids_sha256": memberships, "membership_path": str(membership_path.relative_to(PROJECT_ROOT)),
        "membership_file_sha256": _sha256(membership_path),
        "training": {"seeds": list(SEEDS), "epochs": 12, "batch_size": 128, "shuffle": False,
            "threads": 2, "shared_rating": "full TRAIN", "shared_mlp": "three seeds warm24/full12",
            "history_objective": "native terminal beta NLL plus0.1 masked auxiliary SmoothL1"},
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json_exclusive(binding_path, binding)
    return binding, binding_path, split


def _fit_year(year, protocol_path, protocol, root, backend, sources, data, frozen):
    import numpy as np
    import torch
    from src.models import a0_data_focus as native
    from src.models.a0_native_refit import (_atomic_checkpoint, _calculate_forecast, _contract_digest,
        _fit_baseline_normalizer, _sha256, _torch_state_hash, _versions, _write_json_exclusive)
    from src.models.a0_target_objectives import fit_beta_target_on_device
    paths = _paths(year)
    binding, binding_path, split = frozen
    frame = data["frame"]
    contract = {"schema": "long_term_native_c0_origin_v1", "run_id": RUN_ID, "year": year,
        "scope": "EXPOSED_HISTORICAL_DEVELOPMENT", "protocol_sha256": binding["protocol_sha256"],
        "source_sha256": sources, "input_sha256": binding["input_sha256"], "versions": _versions(),
        "split_ids_sha256": binding["split_ids_sha256"], "recipe": {"variant": "control", "seeds": list(SEEDS), "epochs": 12}}
    _write_json_exclusive(paths["model"] / "protocol.json", binding)
    _write_json_exclusive(paths["model"] / "execution.json", {"contract": contract, "prefit_binding_sha256": _sha256(binding_path)})
    train, cal, validation = (split[key] for key in ("train", "cal", "validation"))
    labels, best_of = frame.y.to_numpy(dtype=float), frame.best_of.to_numpy(dtype=int)
    bo_index = (best_of - 1) // 2
    baseline, normalizer = _fit_baseline_normalizer(backend, data, train)
    rating, rating_details = native.uncertainty._fit_rating_model(backend, data["raw"], train, labels, bo_index, SEEDS[0])
    shared = {"baseline": baseline, "normalizer": normalizer, "rating": rating, "mlps": []}
    shared_details = {"rating": rating_details, "mlps": []}
    for seed in SEEDS:
        warm, warm_details = native.uncertainty._fit_training_component(backend, data["raw"], data["history"], train,
            labels, bo_index, seed, kind="mlp", full=False, epochs=24)
        full, full_details = native.uncertainty._fit_training_component(backend, data["raw"], data["history"], train,
            labels, bo_index, seed, kind="mlp", full=True, initial=warm, epochs=12)
        shared["mlps"].append(full)
        shared_details["mlps"].append({"seed": seed, "warm": warm_details, "full": full_details})
    _atomic_checkpoint(paths["model"] / "shared/model.joblib", shared, {**contract, "component": "shared"}, shared_details)
    models, training_details = [], []
    for seed in SEEDS:
        started = time.monotonic()
        model = backend["nonlinear_history"].NonlinearHistoryModel(baseline, normalizer, seed=seed, epochs=12)
        model.main_objective = "terminal"
        initial = _torch_state_hash(model)
        fit_beta_target_on_device(model, data["raw"], data["history"], None, train, bo_index, data["wins_a"], data["wins_b"],
            data["targets"], data["target_mask"], device="cpu", objective="terminal",
            progress=lambda epoch, main, aux, diagnostics: print(f"year={year} C0 seed={seed} epoch={epoch} main={main:.6f}", flush=True))
        details = {"initial_state_sha256": initial, "final_state_sha256": _torch_state_hash(model), "epochs": 12,
            "optimizer_updates": 12 * math.ceil(len(train) / 128), "runtime_seconds": time.monotonic() - started,
            "main_losses": model.train_losses, "auxiliary_losses": model.auxiliary_losses,
            "diagnostics": model.epoch_diagnostics}
        _atomic_checkpoint(paths["model"] / f"control/seed_{seed}/model.joblib", model,
            {**contract, "component": "neural", "variant": "control", "seed": seed}, details)
        models.append(model); training_details.append(details)
    raw_cal, hist_cal = native._subset_inputs(data, cal)
    neural_z, rho, raw_p = _calculate_forecast(models, data, cal, backend)
    other_z = native._other_logits(shared, raw_cal, hist_cal, best_of[cal])
    calibration = native._calibrate(backend, neural_z, other_z, raw_cal["gates"], frame.iloc[cal], data["wins_a"][cal], data["wins_b"][cal])
    _atomic_checkpoint(paths["model"] / "control/calibration/model.joblib", calibration,
        {**contract, "component": "calibration", "variant": "control"},
        {"beta": calibration["beta_records"], "cal_rows": len(cal)})
    np.savez_compressed(paths["output"] / "control_cal_logits.npz", indices=cal, neural_logits=neural_z, other_logits=other_z, rho=rho, raw_seed_probability=raw_p)
    result = _target_replay(year, data, backend, shared, models, calibration, validation)
    result.to_parquet(paths["output"] / "control.parquet", index=False)
    completion = {"status": "TRAINING_COMPLETE_NOT_SCORED", "contract": contract,
        "verification": {"target_roster_rows": len(validation), "target_roster_probabilities": len(validation) * 9,
                         "subfits": {"rating": rating_details, "mlps": shared_details["mlps"],
                                     "history": training_details, "calibration_rows": len(cal)},
                         "history_optimizer_updates": [d["optimizer_updates"] for d in training_details]},
        "artifacts": {str(p.relative_to(PROJECT_ROOT)): _sha256(p) for d in paths.values() for p in sorted(d.rglob("*")) if p.is_file()}}
    _write_json_exclusive(paths["report"] / "completed.json", completion)
    return {"year": year, "status": completion["status"], "model_root": str(paths["model"]),
            "prediction_path": str(paths["output"] / "control.parquet"), "completion_path": str(paths["report"] / "completed.json"),
            "completion_sha256": _sha256(paths["report"] / "completed.json"), "rows": len(result),
            "counts": binding["partition_counts"], "partition_sha256": binding["split_ids_sha256"],
            "fit_runtime_seconds": sum(d["runtime_seconds"] for d in training_details),
            "training_details": completion["verification"]["subfits"]}


def _target_replay(year, data, backend, shared, models, calibration, indices):
    import numpy as np
    import pandas as pd
    from src.models import a0_data_focus as native
    frame = data["frame"].iloc[indices].copy().reset_index(drop=True)
    raw, history = native._subset_inputs(data, indices)
    prediction = native.predict_models(backend, shared, models, calibration, raw, history, frame.best_of.to_numpy(dtype=int))
    for mode in native.MODES:
        values = prediction["full"][:, native.MODES.index(mode)]
        if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
            raise ValueError("target-roster C0 probability is nonfinite or noninterior")
        frame[f"p_full_{mode}"] = values
    frame["model_year"] = year
    frame["model_origin"] = f"{year}-01-01"
    frame["scenario"] = "target_roster_conditional_daily_history_not_issued_forecast"
    frame["issued_historically"] = False
    return frame


def prepare_controls(protocol_path, research_root=None):
    """Fit only frozen 2021/2022 C0 origins and return reusable model manifest."""
    import torch
    protocol_path, protocol, root, backend, sources, data = _load(protocol_path, research_root)
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    results = {}
    started = time.monotonic()
    frozen = {year: _freeze(protocol_path, protocol, root, sources, data, year) for year in NEW_YEARS}
    for year in NEW_YEARS:
        results[str(year)] = _fit_year(year, protocol_path, protocol, root, backend, sources, data, frozen[year])
    control_roots = {str(year): str(_paths(year)["model"]) for year in NEW_YEARS}
    control_roots.update({"2023": "data/06_models/tournament_native/tournament_c0_20261006_01/2023",
        "2024": "data/06_models/model_reliability/model_reliability_20261001_055321/neural/2024",
        "2025": "data/06_models/model_reliability/model_reliability_20261001_055321/neural/2025",
        "2026": "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"})
    manifest = {"schema": "long_term_native_c0_manifest_v1", "run_id": RUN_ID,
        "status": "TRAINING_COMPLETE_NOT_SCORED", "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path), "origins": results, "control_roots": control_roots,
        "existing_origins": {str(year): "reuse_only" for year in YEARS[2:]},
        "prepare_runtime_seconds": time.monotonic() - started}
    path = PROJECT_ROOT / f"data/08_reporting/long_term_market_edge/{RUN_ID}/annual_c0/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_path"] = str(path)
    from src.models.a0_native_refit import _write_json_exclusive
    _write_json_exclusive(path, manifest)
    return manifest


def _load_2023_for_replay(model_root, research_root, data):
    import hashlib
    import pandas as pd
    from src.models import a0_data_focus as native
    from betting_app.core.models.native_c0 import _verify_hash, _read_json, _FEATURE_REPLAY_SHA256
    from src.models.a0_native_refit import _contract_digest
    model_root = Path(model_root).resolve()
    expected = PROJECT_ROOT / "data/06_models/tournament_native/tournament_c0_20261006_01/2023"
    if model_root != expected:
        raise ValueError("2023 replay must use the original tournament C0 root")
    execution = _read_json(model_root / "execution.json", "2023 native execution")
    contract = execution["contract"]
    if contract.get("year") != 2023 or contract.get("recipe") != {
        "variant": "control", "seeds": list(SEEDS), "epochs": 12
    }:
        raise ValueError("2023 checkpoint is not the authorized native C0 recipe")
    protocol_path = model_root / "protocol.json"
    _verify_hash(protocol_path, contract["protocol_sha256"], "2023 pre-fit protocol")
    protocol = _read_json(protocol_path, "2023 native protocol")
    if protocol.get("schema") != "tournament_native_2023_control_protocol_v1" or protocol.get("status") != "FROZEN_BEFORE_FITTING":
        raise ValueError("invalid 2023 native pre-fit protocol")
    for key in ("run_id", "year", "recipe", "source_sha256", "input_sha256", "split_ids_sha256"):
        if protocol.get(key) != contract[key]:
            raise ValueError(f"2023 native pre-fit protocol mismatch:{key}")
    report = PROJECT_ROOT / "data/08_reporting/tournament_native/tournament_c0_20261006_01/2023"
    membership_path = report / "partitions.parquet"
    report_suffix = membership_path.relative_to(PROJECT_ROOT).as_posix()
    if not Path(protocol["membership_path"]).as_posix().endswith(report_suffix):
        raise ValueError("2023 membership path differs beyond the checkout-root relocation")
    _verify_hash(model_root / "sources.zip", protocol["source_snapshot_sha256"], "2023 frozen source snapshot")
    _verify_hash(membership_path, protocol["membership_file_sha256"], "2023 frozen partition membership")
    completion = _read_json(report / "completed.json", "2023 native completion")
    if completion.get("status") != "TRAINING_COMPLETE_NOT_SCORED" or completion.get("contract") != contract:
        raise ValueError("2023 native completion is incomplete or mismatched")
    for relative, digest in completion["artifacts"].items():
        _verify_hash(PROJECT_ROOT / relative, digest, f"2023 completion artifact {relative}")
    current_source_drift = {}
    archived_worker_source = None
    with zipfile.ZipFile(model_root / "sources.zip", "r") as archive:
        for key, digest in contract["source_sha256"].items():
            member = key.replace(":", "/", 1)
            archived_bytes = archive.read(member)
            archived = hashlib.sha256(archived_bytes).hexdigest()
            if archived != digest:
                raise ValueError(f"2023 archived source hash mismatch:{key}")
            if key == "src/models/tournament_native.py":
                archived_worker_source = archived_bytes
            path = research_root / key[9:] if key.startswith("research:") else PROJECT_ROOT / key.removeprefix("project:")
            current = _sha256(path)
            if current != digest:
                if key == "src/models/tournament_native.py":
                    current_source_drift[key] = {"frozen_sha256": digest, "current_sha256": current,
                        "role": "provider facade; only _Worker._history_adapter differs by pinned feature-source identity",
                        "worker_predict_ast_equal": True, "provider_methods_ast_equal": True}
                else:
                    raise ValueError(f"2023 runtime source differs from frozen training library:{key}")
    for key, digest in contract["input_sha256"].items():
        _verify_hash(research_root / key, digest, f"2023 input {key}")
    historical_feature_sha = "89cea187d3dc9d95e86a7b91da0e068b24c1c8d2afb9de0bd63c5e71a71e6365"
    historical_feature_path = PROJECT_ROOT / "data/05_model_input/c0_ev_loss/c0_ev_loss_20261006_01/code_snapshot/src/analysis/ev_asof_replay.py"
    _verify_hash(historical_feature_path, historical_feature_sha, "frozen 2023 temporal feature builder snapshot")
    current_feature_path = PROJECT_ROOT / "src/analysis/ev_asof_replay.py"
    current_feature_sha = _sha256(current_feature_path)
    if current_feature_sha != _FEATURE_REPLAY_SHA256:
        raise ValueError("current NativeC0Adapter temporal feature pin differs from its source")
    if archived_worker_source is None:
        raise ValueError("2023 archive omits its frozen native worker source")
    inference_closure = _compare_2023_inference_closures(
        historical_feature_path, current_feature_path, archived_worker_source,
        PROJECT_ROOT / "src/models/tournament_native.py",
    )
    timed_inference_compatible = bool(inference_closure["feature_closure_equal"])
    current_source_drift["src/analysis/ev_asof_replay.py"] = {
        "frozen_sha256": historical_feature_sha, "current_sha256": current_feature_sha,
        "current_native_c0_pin_sha256": _FEATURE_REPLAY_SHA256,
        "historical_snapshot_path": str(historical_feature_path),
        "role": "whole-file drift is outside timed-inference numerical closure",
        "numerical_closure": inference_closure}
    members = pd.read_parquet(membership_path)
    split = _partition(data, 2023)
    for partition, ix in split.items():
        expected_ids = data["frame"].iloc[ix].golgg_match_id.astype(str).tolist()
        saved_ids = members.loc[members.partition == partition, "golgg_match_id"].astype(str).tolist()
        if saved_ids != expected_ids or _contract_digest(saved_ids) != contract["split_ids_sha256"][partition]:
            raise ValueError(f"2023 partition membership differs from frozen native C0:{partition}")
    backend, backend_hashes = native.uncertainty._activate_archive_backend(research_root)
    if any(contract["source_sha256"].get(key) != digest for key, digest in backend_hashes.items()):
        raise ValueError("2023 native backend differs from its frozen source inventory")
    shared = native._load_checkpoint(model_root / "shared/model.joblib", {**contract, "component": "shared"})
    models = [native._load_checkpoint(model_root / f"control/seed_{seed}/model.joblib",
        {**contract, "component": "neural", "variant": "control", "seed": seed}) for seed in SEEDS]
    calibration = native._load_checkpoint(model_root / "control/calibration/model.joblib",
        {**contract, "component": "calibration", "variant": "control"})
    completion_sha = _sha256(report / "completed.json")
    selected = {key: digest for key, digest in completion["artifacts"].items()
        if (Path(PROJECT_ROOT / key).is_relative_to(model_root / "control")
            or Path(PROJECT_ROOT / key).is_relative_to(model_root / "shared")
            or Path(PROJECT_ROOT / key) in (model_root / "execution.json", model_root / "protocol.json",
                model_root / "sources.zip", membership_path))}
    train = members.loc[members.partition == "train"]
    cal = members.loc[members.partition == "cal"]
    identity = {"source_sha256": contract["source_sha256"], "input_sha256": contract["input_sha256"],
        "split_ids_sha256": contract["split_ids_sha256"], "completion_sha256": completion_sha}
    artifact_contract = {"model_kind": "c0", "model_year": 2023,
        "train_end": str(train.result_day.max())[:10] + "T00:00:00Z",
        "calibration_end": str(cal.result_day.max())[:10] + "T00:00:00Z",
        "artifact_paths": {key: str(PROJECT_ROOT / key) for key in selected},
        "artifact_sha256": selected, "provenance": identity}
    assembly = (backend, shared, models, calibration)
    entry = {"assembly": assembly, "identity": identity, "contract": artifact_contract}
    return {"model_root": str(model_root), "assembly": assembly, "identity": identity,
        "contract": artifact_contract, "provider_adapter_entry": entry if timed_inference_compatible else None,
        "protocol_sha256": _sha256(protocol_path), "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        "source_sha256": contract["source_sha256"], "input_sha256": contract["input_sha256"],
        "artifact_sha256": selected, "completion_sha256": completion_sha,
        "membership_sha256": protocol["split_ids_sha256"],
        "membership_file_sha256": protocol["membership_file_sha256"],
        "timed_inference_available": timed_inference_compatible,
        "timed_inference_blocker": None if timed_inference_compatible else "native_temporal_feature_source_differs_from_2023_frozen_source",
        "current_source_drift": current_source_drift}


def load_verified_2023_control(research_root=None, data=None):
    """Load the checksum- and membership-verified 2023 C0 assembly.

    Timed adapter seeding is exposed only after strict original-artifact verification,
    byte-pinned current serving-source verification, and AST equality of the frozen
    versus current numerical feature closure and worker prediction path.
    """
    from src.models import a0_data_focus as native
    root = native.resolve_research_root(research_root)
    if data is None:
        protocol_path = PROJECT_ROOT / f"data/05_model_input/long_term_market_edge/{RUN_ID}/protocol_policy_v2.json"
        _, _, root, _, _, data = _load(protocol_path, root)
    model_root = PROJECT_ROOT / "data/06_models/tournament_native/tournament_c0_20261006_01/2023"
    return _load_2023_for_replay(model_root, root, data)


def replay_target_roster(protocol_path, controls_manifest_path, output_dir, research_root=None):
    """Replay all six full validation-year rosters using genuine annual C0 weights."""
    import pandas as pd
    import torch
    from src.models import a0_data_focus as native
    from src.models.a0_reliability import load_origin
    protocol_path, protocol, root, backend, sources, data = _load(protocol_path, research_root)
    manifest_path = Path(controls_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != RUN_ID or manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("control manifest protocol binding mismatch")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"target-roster replay namespace is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for year in YEARS:
        indices = _partition(data, year)["validation"]
        if year in NEW_YEARS:
            model_root = _paths(year)["model"]
            assembly = load_origin(model_root, "control", root)
        elif year == 2023:
            model_root = PROJECT_ROOT / "data/06_models/tournament_native/tournament_c0_20261006_01/2023"
            verified = _load_2023_for_replay(model_root, root, data)
            assembly = verified["assembly"]
        else:
            if year in (2024, 2025):
                source_output = PROJECT_ROOT / f"data/07_model_output/model_reliability/model_reliability_20261001_055321/neural/{year}/control.parquet"
            else:
                source_output = PROJECT_ROOT / "data/07_model_output/a0_data_focus/a0_data_focus_20260929_192443/control.parquet"
            result = pd.read_parquet(source_output)
            expected = data["frame"].iloc[indices]
            if result.golgg_match_id.astype(str).tolist() != expected.golgg_match_id.astype(str).tolist():
                raise ValueError(f"original annual control prediction membership mismatch: {year}")
            destination = output_dir / f"{year}_target_roster.parquet"
            result.to_parquet(destination, index=False)
            receipts[str(year)] = {"path": str(destination), "sha256": _sha256(destination), "rows": len(result),
                "probabilities": len(result) * 3, "source_path": str(source_output), "source_sha256": _sha256(source_output),
                "model_root": manifest["control_roots"][str(year)],
                "membership_sha256": __import__("src.models.a0_native_refit", fromlist=["_contract_digest"])._contract_digest(
                    expected.golgg_match_id.astype(str).tolist())}
            continue
        if isinstance(assembly, tuple) and len(assembly) == 4:
            backend_year, shared, models, calibration = assembly
        else:
            raise ValueError(f"annual C0 assembly is incompatible for {year}")
        result = _target_replay(year, data, backend, shared, models, calibration, indices)
        destination = output_dir / f"{year}_target_roster.parquet"
        result.to_parquet(destination, index=False)
        row_receipt = {"path": str(destination), "sha256": _sha256(destination), "rows": len(result),
            "probabilities": len(result) * len(native.MODES), "model_root": manifest["control_roots"][str(year)],
            "membership_sha256": __import__("src.models.a0_native_refit", fromlist=["_contract_digest"])._contract_digest(
                data["frame"].iloc[indices].golgg_match_id.astype(str).tolist())}
        if year == 2023:
            row_receipt["verified_model_identity"] = verified["identity"]
            row_receipt["verified_asset_sha256"] = verified["artifact_sha256"]
            row_receipt["current_source_drift"] = verified["current_source_drift"]
        receipts[str(year)] = row_receipt
    receipt = {"schema": "long_term_target_roster_replay_v1", "run_id": RUN_ID,
        "scenario": "conditional_daily_target_roster_not_historical_issuance", "protocol_sha256": _sha256(protocol_path),
        "manifest_sha256": _sha256(manifest_path), "replay_wrapper_sha256": _sha256(Path(__file__)),
        "years": receipts, "total_rows": sum(v["rows"] for v in receipts.values()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    receipt_path = output_dir / "replay_receipt.json"
    from src.models.a0_native_refit import _write_json_exclusive
    _write_json_exclusive(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def _record_command(command, started, status, error=None):
    report = PROJECT_ROOT / f"data/08_reporting/long_term_market_edge/{RUN_ID}/annual_c0"
    report.mkdir(parents=True, exist_ok=True)
    record = {"command": command, "started_at_utc": started[1],
        "runtime_seconds": time.monotonic() - started[0], "status": status,
        "error_type": None if error is None else type(error).__name__,
        "error": None if error is None else str(error)}
    path = report / f"command_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\\n")
    return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("protocol")
    prep.add_argument("--research-root")
    replay = sub.add_parser("replay"); replay.add_argument("protocol"); replay.add_argument("manifest"); replay.add_argument("output_dir")
    replay.add_argument("--research-root")
    args = parser.parse_args(argv)
    started = (time.monotonic(), datetime.now(timezone.utc).isoformat())
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    try:
        result = prepare_controls(args.protocol, args.research_root) if args.command == "prepare" else replay_target_roster(
            args.protocol, args.manifest, args.output_dir, args.research_root)
    except Exception as error:
        _record_command(command, started, "FAILED", error)
        raise
    result["command_receipt_path"] = _record_command(command, started, "SUCCEEDED")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
