"""Frozen four-cell C0/fresh20 regional factorial replay (research-only)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATIONS = ("C0", "C0+regional", "fresh20", "fresh20+regional")
PRIMARY_CONTRASTS = (
    ("C0+regional", "C0"),
    ("fresh20+regional", "fresh20"),
    ("fresh20", "C0"),
    ("fresh20+regional", "C0+regional"),
    ("fresh20+regional", "C0"),
)


def _path(value: str | Path) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _binding_path(protocol: Mapping[str, Any]) -> Path:
    return _path(protocol["input_root"]) / "execution_binding.json"


def _stage_cache(protocol: Mapping[str, Any], stage: str) -> Path:
    return _path(protocol["input_root"]) / "stage_cache" / f"{stage}.json"


def _ensure_namespace(protocol: Mapping[str, Any]) -> None:
    roots = [
        _path(protocol[key])
        for key in ("feature_root", "input_root", "model_output_root", "output_root")
    ]
    binding = _binding_path(protocol)
    # Existing outputs are reusable only under an exact frozen binding.
    occupied = []
    input_root = _path(protocol["input_root"])
    allowed_input_files = {
        _path(protocol["_protocol_path"]),
        input_root / "protected_before.json",
    }
    for root in roots:
        if not root.exists():
            continue
        contents = [item for item in root.iterdir() if item not in allowed_input_files]
        if contents:
            occupied.append(root)
    if occupied:
        if not binding.is_file():
            raise FileExistsError(
                "Factorial output namespace is occupied without its execution binding: "
                + ", ".join(map(str, occupied))
            )
        prior = _json(binding)
        if prior.get("protocol_sha256") != _sha256(_path(protocol["_protocol_path"])):
            raise FileExistsError(
                "Factorial output namespace belongs to a different frozen protocol"
            )


_SEALED_INPUT_KEYS = (
    "sealed_daily_predictions",
    "sealed_daily_predictions_sha256",
    "sealed_historical_stage",
    "sealed_historical_stage_sha256",
    "sealed_historical_protocol",
    "sealed_historical_protocol_sha256",
    "sealed_historical_binding",
    "sealed_historical_binding_sha256",
)
_SEALED_SCIENTIFIC_SECTIONS = (
    "configurations",
    "anchors",
    "regional",
    "historical",
    "financial",
    "bootstrap",
    "decision",
    "prohibited",
)


def _validated_sealed_daily_predictions(
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, str]] | None:
    inputs = protocol["inputs"]
    present = [key for key in _SEALED_INPUT_KEYS if key in inputs]
    if not present:
        return None
    if len(present) != len(_SEALED_INPUT_KEYS):
        raise ValueError("sealed historical input set is incomplete")
    paths = {
        key: _path(inputs[key])
        for key in (
            "sealed_daily_predictions",
            "sealed_historical_stage",
            "sealed_historical_protocol",
            "sealed_historical_binding",
        )
    }
    hashes: dict[str, str] = {}
    for key, path_key in (
        ("daily_predictions", "sealed_daily_predictions"),
        ("historical_stage", "sealed_historical_stage"),
        ("historical_protocol", "sealed_historical_protocol"),
        ("historical_binding", "sealed_historical_binding"),
    ):
        path = paths[path_key]
        if not path.is_file():
            raise FileNotFoundError(f"Missing sealed historical {key}: {path}")
        actual = _sha256(path)
        expected = str(inputs[f"{path_key}_sha256"])
        if actual != expected:
            raise ValueError(f"sealed historical {key} SHA-256 mismatch")
        hashes[key] = actual

    old_protocol = _json(paths["sealed_historical_protocol"])
    old_binding = _json(paths["sealed_historical_binding"])
    stage = _json(paths["sealed_historical_stage"])
    if old_protocol.get("schema") != "c0_region_factorial_v1":
        raise ValueError("sealed historical protocol schema is invalid")
    if old_binding.get("schema") != "c0_region_factorial_execution_binding_v1":
        raise ValueError("sealed historical binding schema is invalid")
    if old_binding.get("protocol_sha256") != hashes["historical_protocol"]:
        raise ValueError("sealed historical binding protocol SHA-256 mismatch")
    if old_binding.get("run_id") != old_protocol.get("run_id"):
        raise ValueError("sealed historical binding identity mismatch")
    binding_sha = hashlib.sha256(
        json.dumps(old_binding, sort_keys=True, default=_json_default).encode()
    ).hexdigest()
    if (
        stage.get("schema") != "c0_region_factorial_stage_v1"
        or stage.get("stage") != "historical"
        or stage.get("binding_sha256") != binding_sha
        or stage.get("receipt", {}).get("status") != "COMPLETE"
    ):
        raise ValueError("sealed historical stage is not a bound completed stage")
    receipt = stage["receipt"]
    if receipt.get("canonical_rows") != 3103 or receipt.get("daily_rows") != 1126:
        raise ValueError("sealed historical stage row counts are invalid")
    receipt_daily = _path(receipt.get("daily_predictions", ""))
    if receipt_daily != paths["sealed_daily_predictions"]:
        raise ValueError("sealed historical receipt names a different daily artifact")
    artifact = stage.get("artifacts", {}).get("daily_model_predictions")
    if artifact is not None and (
        _path(artifact.get("path", "")) != paths["sealed_daily_predictions"]
        or artifact.get("sha256") != hashes["daily_predictions"]
    ):
        raise ValueError("sealed historical stage daily artifact binding mismatch")
    if not stage.get("completed_at_utc"):
        raise ValueError("sealed historical stage completion timestamp is missing")
    if len(pd.read_parquet(paths["sealed_daily_predictions"])) != 1126:
        raise ValueError("sealed historical daily artifact must contain 1126 rows")

    for section in _SEALED_SCIENTIFIC_SECTIONS:
        if section not in old_protocol or section not in protocol:
            raise ValueError(f"sealed historical protocol lacks {section}")
        if old_protocol[section] != protocol[section]:
            raise ValueError(
                f"sealed historical {section} differs from the pinned protocol"
            )
    old_inputs = dict(old_protocol.get("inputs", {}))
    new_inputs = dict(inputs)
    for key in _SEALED_INPUT_KEYS:
        new_inputs.pop(key, None)
    if old_inputs != new_inputs:
        raise ValueError(
            "sealed historical source inputs differ from the pinned protocol"
        )
    for section in ("model", "models", "source", "sources"):
        if old_protocol.get(section) != protocol.get(section):
            raise ValueError(
                f"sealed historical {section} inputs differ from the pinned protocol"
            )
    return paths["sealed_daily_predictions"], hashes


def _source_paths(protocol: Mapping[str, Any]) -> dict[str, Path]:
    inputs = protocol["inputs"]
    pointer = _path(inputs["research_root_pointer"])
    if not pointer.is_file():
        raise FileNotFoundError(f"Missing research-root pointer: {pointer}")
    research_root = _path(pointer.read_text().strip())
    release = research_root / inputs["release_map_relative"]
    canonical_manifest = _path(inputs["canonical_manifest"])
    manifest_data = _json(canonical_manifest)
    canonical_input = research_root / manifest_data["input"]["path"]
    if _sha256(canonical_input) != manifest_data["input"]["sha256"]:
        raise ValueError(
            "Canonical benchmark input hash differs from its frozen manifest"
        )
    metadata = research_root / inputs["metadata_relative"]
    entries = {
        **(
            {"worlds_records_cache": _path(inputs["worlds_records_cache"])}
            if inputs.get("worlds_records_cache")
            else {}
        ),
        **(
            {"worlds_team_name_binding": _path(inputs["worlds_team_name_binding"])}
            if inputs.get("worlds_team_name_binding")
            else {}
        ),
        "canonical_a0_input": canonical_input,
        "protocol": _path(protocol["_protocol_path"]),
        "research_root_pointer": pointer,
        "canonical_manifest": _path(inputs["canonical_manifest"]),
        "native_config": _path(inputs["native_config"]),
        "raw_history": _path(inputs["raw_history"]),
        "extended_history": _path(inputs["extended_history"]),
        "fresh_protocol": _path(inputs["fresh_protocol"]),
        "fresh_model_root": _path(inputs["fresh20_model_root"]),
        "c0_model_root": _path(
            "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
        ),
        "fresh_anchor_bank": _path(inputs["fresh_anchor_bank"]),
        "edge_protocol": _path(inputs["edge_protocol"]),
        "edge_root": _path(inputs["edge_root"]),
        "c0_control_predictions": _path(
            "data/07_model_output/a0_data_focus/a0_data_focus_20260929_192443/control.parquet"
        ),
        "fresh20_canonical_predictions": _path(
            "data/07_model_output/c0_distilled_verification/c0d_repro_20261021_01/fresh/predictions.parquet"
        ),
        "release_map": release,
        "metadata": metadata,
    }
    sealed = _validated_sealed_daily_predictions(protocol)
    if sealed is not None:
        entries.update(
            {
                "sealed_daily_predictions": sealed[0],
                "sealed_historical_stage": _path(inputs["sealed_historical_stage"]),
                "sealed_historical_protocol": _path(
                    inputs["sealed_historical_protocol"]
                ),
                "sealed_historical_binding": _path(inputs["sealed_historical_binding"]),
            }
        )
    return entries


def _worlds_request_metadata(
    protocol: Mapping[str, Any],
    requests: list[Mapping[str, Any]],
    row_ids: Any,
    query_proofs: list[Mapping[str, Any]],
) -> tuple[list[bool], list[dict[str, Any]]]:
    expected_ids = [int(value) for value in protocol["worlds"]["match_ids"]]
    ids = [int(value) for value in row_ids]
    if ids != expected_ids or len(requests) != len(ids):
        raise ValueError("Worlds native row IDs differ from frozen snapshot IDs")
    proofs_by_id: dict[str, Mapping[str, Any]] = {}
    for proof in query_proofs:
        key = str(proof.get("opportunity_id"))
        if key in proofs_by_id:
            raise ValueError("Worlds query proof IDs must be unique")
        proofs_by_id[key] = proof
    if set(proofs_by_id) != {str(value) for value in ids}:
        raise ValueError("Worlds query proof IDs differ from frozen snapshot IDs")

    bindings = {
        int(item["match_id"]): item for item in protocol["worlds"]["team_id_bindings"]
    }
    if set(bindings) != set(ids):
        raise ValueError("Worlds team ID bindings differ from frozen snapshot IDs")
    name_binding = _json(_path(protocol["inputs"]["worlds_team_name_binding"]))
    source_path = protocol["inputs"].get("extended_history")
    source_hash = _sha256(_path(source_path)) if source_path else None
    allowed_names = name_binding.get("names", {})
    if (
        name_binding.get("schema") != "retained_worlds_team_name_binding_v1"
        or name_binding.get("missing")
        or (source_hash and name_binding.get("raw_history_sha256") != source_hash)
    ):
        raise ValueError("Worlds team name source binding is invalid")
    canonical_a_is_team1 = []
    regional_requests = []
    for row_id, request in zip(ids, requests):
        bound = bindings[row_id]
        team1_id, team2_id = str(bound["team1_id"]), str(bound["team2_id"])
        if str(request["team1_id"]) != team1_id or str(request["team2_id"]) != team2_id:
            raise ValueError(
                f"Worlds native team IDs differ from frozen binding: {row_id}"
            )
        for team_id, name_key in (
            (team1_id, "team_a_name"),
            (team2_id, "team_b_name"),
        ):
            # Same pinned ID; ignore display punctuation, never alias team identities.
            name_token = "".join(
                c for c in str(bound[name_key]).casefold() if c.isalnum()
            )
            source_tokens = {
                "".join(c for c in str(name).casefold() if c.isalnum())
                for name in allowed_names.get(team_id, ())
            }
            if not name_token or name_token not in source_tokens:
                raise ValueError(
                    f"Worlds team name differs from source binding: {team_id}"
                )
        proof = proofs_by_id[str(row_id)]
        if int(proof.get("canonical_match_id", -1)) != row_id:
            raise ValueError(f"Worlds query proof canonical ID differs: {row_id}")
        if request.get("decision_at") is None:
            raise ValueError(f"Worlds native request lacks decision time: {row_id}")
        canonical_a_is_team1.append(True)
        regional_requests.append(
            {
                "opportunity_id": str(row_id),
                "cutoff_day": str(proof["cutoff_day"])[:10],
                "team1_id": team1_id,
                "team2_id": team2_id,
                "roster_a": list(map(str, request["roster_a"])),
                "roster_b": list(map(str, request["roster_b"])),
            }
        )
    return canonical_a_is_team1, regional_requests


def _inventory(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input {name} is absent: {path}")
        if path.is_file():
            result[name] = {
                "path": str(path),
                "kind": "file",
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        else:
            files = sorted(p for p in path.rglob("*") if p.is_file())
            result[name] = {
                "path": str(path),
                "kind": "directory",
                "files": {str(p.relative_to(path)): _sha256(p) for p in files},
            }
    return result


def _code_closure() -> dict[str, str]:
    names = (
        "src/analysis/c0_region_factorial.py",
        "src/models/c0_live_query.py",
        "src/models/c0_regional_replay.py",
        "src/models/c0_edge_model_audit.py",
        "src/models/c0_distilled_reproduction.py",
        "src/models/c0_distilled_replay.py",
        "src/models/c0_distilled_proof.py",
        "src/models/a0_data_focus.py",
        "src/models/a0_native_refit.py",
        "src/models/a0_target_experiment.py",
        "src/models/cross_region_input_normalizer.py",
        "src/analysis/c0_region_finance.py",
        "src/analysis/c0_edge_policy_audit.py",
        "betting_app/core/models/native_c0.py",
        "scripts/run_unified_betting_evaluation.py",
    )
    closure = {}
    for name in names:
        path = _path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Execution code closure is incomplete: {name}")
        closure[name] = _sha256(path)
    return closure


def bind(protocol_path: str | Path) -> dict[str, Any]:
    path = _path(protocol_path)
    protocol = _json(path)
    if (
        protocol.get("schema") != "c0_region_factorial_v1"
        or protocol.get("status") != "DESIGN_FROZEN_EXECUTION_BINDING_REQUIRED"
    ):
        raise ValueError(
            "Expected the frozen c0_region_factorial_v1 execution protocol"
        )
    protocol["_protocol_path"] = str(path)
    _ensure_namespace(protocol)
    roots = [
        _path(protocol[key])
        for key in ("feature_root", "input_root", "model_output_root", "output_root")
    ]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    source_inventory = _inventory(_source_paths(protocol))
    binding = {
        "schema": "c0_region_factorial_execution_binding_v1",
        "run_id": protocol["run_id"],
        "protocol_sha256": _sha256(path),
        "source_inventory": source_inventory,
        "code_sha256": _code_closure(),
        "configurations": list(CONFIGURATIONS),
        "financial_policy": protocol["financial"],
        "bootstrap": protocol["bootstrap"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "bound_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "formal_neural_fits": 0,
        "calibration_fits": 0,
        "source_requests": 0,
        "database_queries": 0,
    }
    dest = _binding_path(protocol)
    if dest.exists():
        previous = _json(dest)
        if previous != binding:
            # Volatile binding time is not used to bless a changed source closure.
            stable_previous = {k: v for k, v in previous.items() if k != "bound_at_utc"}
            stable_current = {k: v for k, v in binding.items() if k != "bound_at_utc"}
            if stable_previous != stable_current:
                raise ValueError(
                    "Frozen source/code closure changed after execution binding"
                )
            binding = previous
    else:
        _write_json(dest, binding)
    return binding


def _require_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = _binding_path(protocol)
    if not path.is_file():
        raise RuntimeError("Run prepare before scoring stages")
    binding = _json(path)
    if binding["protocol_sha256"] != _sha256(_path(protocol["_protocol_path"])):
        raise RuntimeError("Protocol differs from the frozen execution binding")
    current = _inventory(_source_paths(protocol))
    if (
        current != binding["source_inventory"]
        or _code_closure() != binding["code_sha256"]
    ):
        raise RuntimeError(
            "A bound source or code input changed; refusing cache reuse/scoring"
        )
    return binding


def _cached(
    protocol: Mapping[str, Any], stage: str, binding: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _stage_cache(protocol, stage)
    if not path.is_file():
        return None
    value = _json(path)
    if (
        value.get("binding_sha256")
        != hashlib.sha256(
            json.dumps(binding, sort_keys=True, default=_json_default).encode()
        ).hexdigest()
    ):
        raise RuntimeError(f"{stage} cache does not match frozen source closure")
    for name, metadata in value.get("artifacts", {}).items():
        artifact = _path(metadata["path"])
        if not artifact.is_file() or _sha256(artifact) != metadata["sha256"]:
            raise RuntimeError(
                f"{stage} cached artifact is missing or modified: {name}"
            )
    return value


def _save_stage(
    protocol: Mapping[str, Any],
    stage: str,
    binding: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema": "c0_region_factorial_stage_v1",
        "stage": stage,
        "binding_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True, default=_json_default).encode()
        ).hexdigest(),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "receipt": dict(receipt),
        "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(_stage_cache(protocol, stage), record)
    return record


def prepare(protocol_path: str | Path) -> dict[str, Any]:
    protocol = _json(_path(protocol_path))
    protocol["_protocol_path"] = str(_path(protocol_path))
    binding = bind(protocol_path)
    return {"stage": "prepare", "binding": binding, "status": "ready"}


def transform_batch_once(
    raw: Mapping[str, np.ndarray],
    history: Mapping[str, np.ndarray],
    off_a: Any,
    off_b: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Normalize cross-region rows only, preserving domestic float32 inputs bit-for-bit."""
    from src.models.cross_region_input_normalizer import normalize_cross_region_batch

    if "history" not in history or history["history"].ndim != 5:
        raise ValueError(
            "history must have shape (batch, side, player, observed-slot, feature)"
        )
    n = history["history"].shape[0]
    if history["history"].shape[3] != 19:
        raise ValueError(
            f"fresh20 regional input requires 19 observed history slots, got {history['history'].shape[3]}"
        )
    if history["mask"].shape != history["history"].shape[:-1]:
        raise ValueError("history mask must align exactly with observed history slots")
    if history["champions"].shape != history["mask"].shape[:-1] + (16,):
        raise ValueError("native champion IDs must retain the original 16 recent slots")
    if history["champions"].dtype != np.dtype("uint16"):
        raise ValueError("native champion IDs must use uint16 dtype")
    if (
        raw["players"].shape[0] != n
        or raw["players"].shape[-1] != 21
        or raw["team"].shape[-1] != 11
    ):
        raise ValueError(
            "native raw tensors require aligned batch size, players width 21, and team width 11"
        )
    if raw["core"].shape != raw["players"].shape[:-1]:
        raise ValueError("native core tensor must align with player slots")
    oa = np.asarray(off_a, dtype=np.float32).reshape(-1)
    ob = np.asarray(off_b, dtype=np.float32).reshape(-1)
    if oa.size == 1:
        oa = np.full(n, oa.item(), dtype=np.float32)
    if ob.size == 1:
        ob = np.full(n, ob.item(), dtype=np.float32)
    if (
        oa.shape != (n,)
        or ob.shape != (n,)
        or not np.isfinite(oa).all()
        or not np.isfinite(ob).all()
    ):
        raise ValueError(
            "regional offsets must be finite scalars or aligned per-row vectors"
        )
    affected = np.flatnonzero(np.abs(oa - ob) > 1.0)
    raw_out = {key: np.array(value, copy=True) for key, value in raw.items()}
    hist_out = {key: np.array(value, copy=True) for key, value in history.items()}
    if affected.size:
        raw_changed, history_changed = normalize_cross_region_batch(
            {key: value[affected] for key, value in raw.items()},
            {key: value[affected] for key, value in history.items()},
            oa[affected],
            ob[affected],
            z_per_100gl=0.28,
        )
        for key in raw_out:
            raw_out[key][affected] = raw_changed[key]
        # Keep zero-difference histories bit-identical; adjusted rows retain the normalizer's float16 rounding.
        hist_out["history"][affected] = history_changed["history"].astype(
            hist_out["history"].dtype, copy=False
        )
    return raw_out, hist_out


def assemble_four_configurations(
    canonical: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    key = "match_id"
    if key not in canonical or "canonical_a_is_team1" not in canonical:
        raise ValueError(
            "canonical rows require match_id and canonical side orientation"
        )
    if canonical[key].astype(str).duplicated().any():
        raise ValueError("canonical match IDs must be unique")
    required = {key, "configuration", "canonical_a_is_team1"}
    if not required.issubset(predictions.columns):
        raise ValueError("predictions lack identity/configuration/orientation keys")
    if predictions.duplicated([key, "configuration"]).any():
        raise ValueError("duplicate match/configuration prediction")
    if not set(predictions.configuration.dropna()).issubset(CONFIGURATIONS):
        raise ValueError("unknown factorial configuration")
    if "probability" not in predictions:
        predictions = predictions.assign(probability=np.nan)
    if "status" not in predictions:
        predictions = predictions.assign(status="observed")
    if "observed_history_slots" in predictions:
        fresh = predictions.configuration.astype(str).str.startswith("fresh20")
        if predictions.loc[fresh, "observed_history_slots"].ne(19).any():
            raise ValueError("fresh20 prediction requires 19 observed history slots")
    base = canonical.copy()
    base[key] = base[key].astype(str)
    pred = predictions.copy()
    pred[key] = pred[key].astype(str)
    if key in pred.columns and not pred.empty:
        merged = base.merge(
            pred,
            on=key,
            how="left",
            suffixes=("", "_prediction"),
            validate="one_to_many",
        )
    else:
        merged = base.iloc[0:0].copy()
    pieces = []
    for config in CONFIGURATIONS:
        subset = (
            merged.loc[merged.configuration.eq(config)].copy()
            if "configuration" in merged
            else base.iloc[0:0].copy()
        )
        absent_ids = set(base[key]) - set(subset[key])
        if absent_ids:
            absent = base.loc[base[key].isin(absent_ids)].copy()
            absent["configuration"] = config
            absent["probability"] = np.nan
            absent["status"] = "absent_prediction"
            subset = pd.concat([subset, absent], ignore_index=True, sort=False)
        if not subset.empty:
            orient = subset.get(
                "canonical_a_is_team1_prediction", subset["canonical_a_is_team1"]
            )
            valid = orient.notna() & subset["canonical_a_is_team1"].notna()
            mismatch = valid & (
                orient.astype(bool) != subset["canonical_a_is_team1"].astype(bool)
            )
            if mismatch.any():
                raise ValueError(
                    "prediction orientation differs from canonical identity"
                )
        pieces.append(subset)
    result = pd.concat(pieces, ignore_index=True, sort=False)
    result["configuration"] = pd.Categorical(
        result.configuration, categories=CONFIGURATIONS, ordered=True
    )
    result = result.sort_values([key, "configuration"], kind="mergesort").reset_index(
        drop=True
    )
    return result


def score_probabilities(y: Any, probability: Any) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(
            "outcomes and probabilities must be equal one-dimensional arrays"
        )
    valid = np.isfinite(p) & np.isin(y, (0, 1))
    y, p = y[valid], p[valid]
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not len(y):
        return {
            "n": 0,
            "log_loss": None,
            "brier": None,
            "auc": None,
            "accuracy": None,
            "ece": None,
            "tails": {
                "below_0_05": 0,
                "above_0_95": 0,
                "log_loss_ge_2_5": 0,
                "log_loss_ge_5": 0,
            },
        }
    with np.errstate(divide="ignore", invalid="ignore"):
        losses = np.where(y == 1, -np.log(p), -np.log1p(-p))
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    except (ImportError, ValueError):
        auc = None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(
        np.sum(bins == i)
        / len(y)
        * abs(float(p[bins == i].mean()) - float(y[bins == i].mean()))
        for i in range(10)
        if np.any(bins == i)
    )
    return {
        "n": int(len(y)),
        "log_loss": float(np.mean(losses)),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": auc,
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "ece": float(ece),
        "tails": {
            "below_0_05": int(np.sum(p < 0.05)),
            "above_0_95": int(np.sum(p > 0.95)),
            "log_loss_ge_2_5": int(np.sum(losses >= 2.5)),
            "log_loss_ge_5": int(np.sum(losses >= 5.0)),
        },
    }


def paired_monthly_contrast(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    draws: int = 5000,
    seed: int = 20261022,
    metric: str = "log_loss",
    confidence: float | None = None,
    month: str = "month",
    outcome: str = "y",
) -> dict[str, Any]:
    if metric not in ("log_loss", "brier"):
        raise ValueError("metric must be 'log_loss' or 'brier'")
    if confidence is None:
        confidence = 0.99 if metric == "log_loss" else 0.95
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    needed = {month, outcome, candidate, reference}
    if not needed.issubset(frame):
        raise ValueError(
            f"paired contrast missing columns: {sorted(needed - set(frame))}"
        )
    paired = frame.loc[frame[list(needed)].notna().all(axis=1)].copy()
    paired = paired.loc[np.isfinite(paired[candidate]) & np.isfinite(paired[reference])]
    if (
        ((paired[[candidate, reference]] < 0) | (paired[[candidate, reference]] > 1))
        .any()
        .any()
    ):
        raise ValueError("probabilities must lie in [0, 1]")
    y = paired[outcome].to_numpy(dtype=np.float64)
    pc = paired[candidate].to_numpy(dtype=np.float64)
    pr = paired[reference].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):

        def losses(p):
            return (
                np.where(y == 1, -np.log(p), -np.log1p(-p))
                if metric == "log_loss"
                else (p - y) ** 2
            )

        delta = losses(pc) - losses(pr)
    paired["_delta"] = delta
    grouped = paired.groupby(month, sort=True, dropna=False)["_delta"].agg(
        ["mean", "count"]
    )
    mw = (
        float(np.average(grouped["mean"], weights=grouped["count"]))
        if len(grouped)
        else None
    )
    em = float(grouped["mean"].mean()) if len(grouped) else None
    rng = np.random.default_rng(seed)
    months = len(grouped)
    samples = np.full((draws, 2), np.nan)
    if months:
        means = grouped["mean"].to_numpy(dtype=float)
        counts = grouped["count"].to_numpy(dtype=float)
        ix = rng.integers(0, months, size=(draws, months))
        samples[:, 0] = np.sum(means[ix] * counts[ix], axis=1) / np.sum(
            counts[ix], axis=1
        )
        samples[:, 1] = means[ix].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    interval = lambda x: (
        [float(np.nanquantile(x, tail)), float(np.nanquantile(x, 1 - tail))]
        if np.isfinite(x).any()
        else [None, None]
    )
    label = f"{confidence * 100:.4g}pct"
    return {
        "metric": metric,
        "confidence": confidence,
        "candidate": candidate,
        "reference": reference,
        "n_rows": int(len(paired)),
        "n_months": int(months),
        "match_weighted_delta": mw,
        "equal_month_delta": em,
        "draws": int(draws),
        "undefined_draw_count": int(np.sum(~np.isfinite(samples[:, 0]))),
        f"match_weighted_{label}": interval(samples[:, 0]),
        f"equal_month_{label}": interval(samples[:, 1]),
    }


def _run_worlds(
    protocol: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    cached = _cached(protocol, "worlds", binding)
    if cached:
        return cached
    import pickle
    from src.models.c0_live_query import build_worlds_query
    from src.models.c0_distilled_reproduction import infer

    inputs = protocol["inputs"]
    research_root = _path(_path(inputs["research_root_pointer"]).read_text().strip())
    raw_path = _path(inputs["extended_history"])
    release_map = research_root / inputs["release_map_relative"]
    records, raw_receipt = _get_raw_map_cache(
        protocol, research_root, source_path=raw_path
    )
    query = build_worlds_query(
        _path(inputs["worlds_snapshots"]),
        raw_history_path=raw_path,
        release_map_path=release_map,
        research_root=research_root,
        records_by_player=records,
        raw_map_receipt=raw_receipt,
    )
    batch = query["batches"]["20t"]
    requests = query["requests"]
    if len(requests) != len(batch["row_ids"]) or len(requests) != len(query["p_c0"]):
        raise ValueError("Worlds native query lost exact eight-match coverage")
    orientation, regional_requests = _worlds_request_metadata(
        protocol, requests, batch["row_ids"], query["query_proofs"]
    )
    forecasts = np.asarray(
        infer(_path(inputs["fresh_protocol"]), "20t", batch, batch["best_of"]),
        dtype=np.float64,
    )
    if len(forecasts) != len(requests):
        raise ValueError("Worlds fresh20 inference lost exact eight-match coverage")
    from src.models.c0_regional_replay import build_regional_query_states

    regional = build_regional_query_states(
        regional_requests, raw_history_path=raw_path, release_map_path=release_map
    )
    replay_rows = regional["rows"]
    offsets_a = np.asarray(
        [row.get("offset_a", np.nan) for row in replay_rows], dtype=np.float32
    )
    offsets_b = np.asarray(
        [row.get("offset_b", np.nan) for row in replay_rows], dtype=np.float32
    )
    supported = (
        (np.asarray([row.get("status") for row in replay_rows]) == "ok")
        & np.isfinite(offsets_a)
        & np.isfinite(offsets_b)
    )
    c0 = np.asarray(query["p_c0"], dtype=np.float64)
    c0_regional = np.full(len(c0), np.nan, dtype=np.float64)
    fresh_regional = np.full(len(c0), np.nan, dtype=np.float64)
    if supported.any():
        ix = np.flatnonzero(supported)
        raw, history = transform_batch_once(
            {key: value[ix] for key, value in batch["raw"].items()},
            {key: value[ix] for key, value in batch["history"].items()},
            offsets_a[ix],
            offsets_b[ix],
        )
        fresh_regional[ix] = infer(
            _path(inputs["fresh_protocol"]),
            "20t",
            {"raw": raw, "history": history},
            batch["best_of"][ix],
        )
        recent = {
            **history,
            "history": history["history"][:, :, :, -16:],
            "mask": history["mask"][:, :, :, -16:],
        }
        c0_regional[ix] = _isolated_native_c0(raw, recent, batch["best_of"][ix])
    side_a = np.asarray(orientation, dtype=bool)
    forecasts_by_config = {
        "C0": np.where(side_a, c0, 1 - c0),
        "C0+regional": np.where(side_a, c0_regional, 1 - c0_regional),
        "fresh20": np.where(side_a, forecasts, 1 - forecasts),
        "fresh20+regional": np.where(side_a, fresh_regional, 1 - fresh_regional),
    }
    proof_by_id = {str(row["opportunity_id"]): row for row in query["query_proofs"]}
    predictions = []
    for index, request in enumerate(requests):
        row_id = str(batch["row_ids"][index])
        proof = proof_by_id[row_id]
        for config in CONFIGURATIONS:
            value = forecasts_by_config[config][index]
            predictions.append(
                {
                    "match_id": str(batch["row_ids"][index]),
                    "configuration": config,
                    "canonical_a_is_team1": bool(side_a[index]),
                    "probability": value,
                    "status": "observed"
                    if np.isfinite(value)
                    else "regional_support_unavailable",
                    "cutoff_day": regional_requests[index]["cutoff_day"],
                    "feature_source_max_day": proof.get("feature_source_max_day"),
                    "decision_at": request.get("decision_at"),
                    "source_match_id": str(batch["row_ids"][index]),
                    "roster_a": request["roster_a"],
                    "roster_b": request["roster_b"],
                    "state_sha256": replay_rows[index].get("state_sha256"),
                }
            )
    root = _path(protocol["model_output_root"])
    root.mkdir(parents=True, exist_ok=True)
    features = _path(protocol["feature_root"])
    features.mkdir(parents=True, exist_ok=True)
    query_path = root / "worlds_query.pkl"
    query_path.write_bytes(pickle.dumps(query, protocol=pickle.HIGHEST_PROTOCOL))
    predictions_path = root / "worlds_predictions.parquet"
    pd.DataFrame(predictions).to_parquet(predictions_path, index=False)
    proofs_path = root / "worlds_proofs.json"
    _write_json(
        proofs_path,
        {
            "query_proofs": query["query_proofs"],
            "regional_receipt": regional.get("receipt"),
            "raw_map_receipt": raw_receipt,
            "source_clocks_are_conditional_snapshots": True,
        },
    )
    forecast_path = root / "worlds_predictions.pkl"
    forecast_path.write_bytes(
        pickle.dumps(
            {
                "query": query,
                "fresh20": forecasts,
                "regional": regional,
                "configuration_predictions": forecasts_by_config,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    cache_artifact = (
        _path(inputs["worlds_records_cache"])
        if inputs.get("worlds_records_cache")
        else features / "worlds_records.pkl"
    )
    artifacts = {
        "worlds_query": query_path,
        "worlds_predictions": predictions_path,
        "worlds_proofs": proofs_path,
        "worlds_pickle": forecast_path,
        "worlds_records": cache_artifact,
    }
    return _save_stage(
        protocol,
        "worlds",
        binding,
        artifacts,
        {
            "query_receipt": query.get("receipt"),
            "raw_map_receipt": raw_receipt,
            "regional_receipt": regional.get("receipt"),
            "rows": len(predictions),
            "regional_support_rows": int(supported.sum()),
            "interpretation": "Conditional replay of retained Oct-20 snapshot inputs; not authenticated issuance.",
        },
    )


def _get_raw_map_cache(
    protocol: Mapping[str, Any],
    research_root: Path,
    *,
    source_path: Path | None = None,
    cache_name: str = "worlds_records.pkl",
) -> tuple[dict[str, Any], dict[str, Any]]:
    import pickle

    inputs = protocol["inputs"]
    raw_path = source_path or _path(inputs["raw_history"])
    release_path = research_root / inputs["release_map_relative"]
    contracts = protocol["raw_map_cache_contracts"]
    source_key = next(
        (
            name
            for name in ("raw_history", "extended_history")
            if raw_path == _path(inputs[name])
        ),
        None,
    )
    if source_key is None:
        raise ValueError(f"Raw map cache source is not frozen: {raw_path}")
    contract = contracts[source_key]
    is_extended = source_key == "extended_history"
    cache_path = (
        _path(inputs["worlds_records_cache"])
        if is_extended and inputs.get("worlds_records_cache")
        else _path(protocol["feature_root"]) / cache_name
    )
    raw_hash, release_hash = _sha256(raw_path), _sha256(release_path)
    external_hash = inputs.get("worlds_records_cache_sha256") if is_extended else None
    if cache_path.is_file():
        if external_hash and _sha256(cache_path) != external_hash:
            raise RuntimeError(
                f"Raw map cache bytes differ from frozen hash: {cache_path}"
            )
        envelope = pickle.loads(cache_path.read_bytes())
        receipt = envelope.get("receipt")
        records = envelope.get("records_by_player")
        if (
            envelope.get("raw_history_sha256") != raw_hash
            or envelope.get("release_map_sha256") != release_hash
            or envelope.get("schema") != "c0_full_records_v1"
            or not isinstance(records, Mapping)
            or not isinstance(receipt, Mapping)
            or receipt.get("raw_history_sha256") != raw_hash
            or receipt.get("release_map_sha256") != release_hash
            or receipt.get("map_record_count") != contract["map_record_count"]
            or receipt.get("source_map_count") != contract["source_map_count"]
            or receipt.get("source_series_count") != contract["source_series_count"]
            or receipt.get("player_count") != len(records)
            or not receipt.get("source_map_membership_sha256")
            or not receipt.get("source_series_membership_sha256")
        ):
            raise RuntimeError(
                f"Raw map cache source hashes or receipt differ from frozen inputs: {cache_path}"
            )
        return envelope["records_by_player"], dict(receipt)
    if is_extended and inputs.get("worlds_records_cache"):
        raise FileNotFoundError(
            f"Frozen external raw map cache is absent: {cache_path}"
        )
    from src.models.c0_live_query import collect_player_map_records

    records, receipt = collect_player_map_records(
        raw_path, release_path, research_root=research_root
    )
    if (
        not isinstance(records, Mapping)
        or receipt.get("raw_history_sha256") != raw_hash
        or receipt.get("release_map_sha256") != release_hash
        or receipt.get("map_record_count") != contract["map_record_count"]
        or receipt.get("source_map_count") != contract["source_map_count"]
        or receipt.get("source_series_count") != contract["source_series_count"]
        or receipt.get("player_count") != len(records)
        or not receipt.get("source_map_membership_sha256")
        or not receipt.get("source_series_membership_sha256")
    ):
        raise RuntimeError(
            "Collected raw map cache receipt differs from source contract"
        )
    envelope = {
        "schema": "c0_full_records_v1",
        "raw_history_sha256": raw_hash,
        "release_map_sha256": release_hash,
        "records_by_player": records,
        "receipt": receipt,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(temporary, cache_path)
    return records, receipt


def _require_finite_probabilities(values: Any, expected: int, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (expected,) or not np.isfinite(probabilities).all():
        finite = int(np.isfinite(probabilities).sum())
        raise ValueError(
            f"{name} native probability coverage expected {expected} finite rows, got "
            f"{finite} finite from shape {probabilities.shape}"
        )
    return probabilities


def _run_historical(
    protocol: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    cached = _cached(protocol, "historical", binding)
    if cached:
        return cached
    inputs = protocol["inputs"]
    research_root = _path(_path(inputs["research_root_pointer"]).read_text().strip())
    metadata_path = research_root / inputs["metadata_relative"]
    metadata = pd.read_parquet(metadata_path)
    metadata = (
        metadata.loc[
            pd.to_datetime(metadata["date"], errors="coerce").dt.year.eq(
                int(protocol["historical"]["canonical_year"])
            )
        ]
        .copy()
        .reset_index(drop=True)
    )
    expected = int(protocol["historical"]["canonical_expected_rows"])
    if (
        len(metadata) != expected
        or metadata.golgg_match_id.astype(str).duplicated().any()
    ):
        raise ValueError(
            f"Canonical feature bank expected {expected} unique 2026 rows, got {len(metadata)}"
        )
    identifiers = metadata.golgg_match_id.astype(str).tolist()
    requests = [
        {
            "opportunity_id": str(row.golgg_match_id),
            "cutoff_day": pd.Timestamp(row.date).date().isoformat(),
            "team1_id": str(row.team1_id),
            "team2_id": str(row.team2_id),
            "roster_a": list(map(str, row.roster_a)),
            "roster_b": list(map(str, row.roster_b)),
        }
        for row in metadata.itertuples(index=False)
    ]
    from src.models.c0_regional_replay import build_regional_query_states

    regional = build_regional_query_states(
        requests,
        raw_history_path=_path(inputs["raw_history"]),
        release_map_path=research_root / inputs["release_map_relative"],
    )
    rows = regional["rows"]
    if (
        len(rows) != len(metadata)
        or [str(r["opportunity_id"]) for r in rows] != identifiers
    ):
        raise ValueError(
            "Regional replay failed one-to-one canonical row identity/order coverage"
        )
    offsets_a = np.asarray([r.get("offset_a", np.nan) for r in rows], dtype=np.float32)
    offsets_b = np.asarray([r.get("offset_b", np.nan) for r in rows], dtype=np.float32)
    status = np.asarray([r.get("status", "unknown") for r in rows], dtype=object)
    good = np.isfinite(offsets_a) & np.isfinite(offsets_b) & (status == "ok")
    from src.models.c0_distilled_reproduction import build_fresh_batch, infer

    fresh_protocol = _path(inputs["fresh_protocol"])
    canonical_c0 = np.full(expected, np.nan, dtype=np.float64)
    canonical_c0_regional = np.full(expected, np.nan, dtype=np.float64)
    fresh20 = np.full(expected, np.nan, dtype=np.float64)
    fresh20_regional = np.full(expected, np.nan, dtype=np.float64)
    batch_size = 128
    batch = build_fresh_batch(fresh_protocol, "20t", identifiers)
    for begin in range(0, expected, batch_size):
        end = min(expected, begin + batch_size)
        ix = np.arange(begin, end)
        raw = {key: value[begin:end] for key, value in batch["raw"].items()}
        history = {key: value[begin:end] for key, value in batch["history"].items()}
        best_of = metadata.best_of.iloc[begin:end].to_numpy(dtype=np.int64)
        fresh20[ix] = _require_finite_probabilities(
            infer(fresh_protocol, "20t", {"raw": raw, "history": history}, best_of),
            len(ix),
            "fresh20",
        )
        recent = {
            **history,
            "history": history["history"][:, :, :, -16:],
            "mask": history["mask"][:, :, :, -16:],
        }
        base_c0 = _isolated_native_c0(raw, recent, best_of)
        canonical_c0[ix] = _require_finite_probabilities(base_c0, len(ix), "C0")
        local = np.flatnonzero(good[ix])
        if local.size:
            usable = ix[local]
            raw_reg, hist_reg = transform_batch_once(
                {key: value[local] for key, value in raw.items()},
                {key: value[local] for key, value in history.items()},
                offsets_a[usable],
                offsets_b[usable],
            )
            reg_recent = {
                **hist_reg,
                "history": hist_reg["history"][:, :, :, -16:],
                "mask": hist_reg["mask"][:, :, :, -16:],
            }
            best_of = metadata.best_of.iloc[usable].to_numpy(dtype=np.int64)
            canonical_c0_regional[usable] = _require_finite_probabilities(
                _isolated_native_c0(raw_reg, reg_recent, best_of),
                len(usable),
                "C0+regional",
            )
            fresh20_regional[usable] = _require_finite_probabilities(
                infer(
                    fresh_protocol,
                    "20t",
                    {"raw": raw_reg, "history": hist_reg},
                    best_of,
                ),
                len(usable),
                "fresh20+regional",
            )
    canonical = pd.DataFrame(
        {
            "match_id": identifiers,
            "canonical_a_is_team1": True,
            "y": metadata.y.to_numpy(dtype=np.int8),
            "date": metadata.date.astype(str).to_numpy(),
            "month": pd.to_datetime(metadata.date).dt.strftime("%Y-%m").to_numpy(),
            "best_of": metadata.best_of.to_numpy(dtype=int),
            "tournament_name": metadata.tournament_name.astype(str).to_numpy(),
            "team1_id": metadata.team1_id.astype(str).to_numpy(),
            "team2_id": metadata.team2_id.astype(str).to_numpy(),
            "feature_source_max_day": metadata.feature_source_max_day.astype(
                str
            ).to_numpy(),
        }
    )
    c0_reference = pd.read_parquet(
        _path(
            "data/07_model_output/a0_data_focus/a0_data_focus_20260929_192443/control.parquet"
        )
    )
    fresh_reference = pd.read_parquet(
        _path(
            "data/07_model_output/c0_distilled_verification/c0d_repro_20261021_01/fresh/predictions.parquet"
        )
    )
    c0_reference = (
        c0_reference.assign(golgg_match_id=c0_reference.golgg_match_id.astype(str))
        .set_index("golgg_match_id")
        .loc[identifiers]
    )
    fresh_reference = (
        fresh_reference.assign(
            golgg_match_id=fresh_reference.golgg_match_id.astype(str)
        )
        .set_index("golgg_match_id")
        .loc[identifiers]
    )
    c0_error = float(
        np.max(
            np.abs(canonical_c0 - c0_reference.p_full_full.to_numpy(dtype=np.float64))
        )
    )
    fresh_error = float(
        np.max(np.abs(fresh20 - fresh_reference.p_20t.to_numpy(dtype=np.float64)))
    )
    if c0_error > 2e-6 or fresh_error > 2e-6:
        raise ValueError(
            f"Saved-model canonical ID replay mismatch: C0={c0_error:.9g}, fresh20={fresh_error:.9g}"
        )
    annual_path = (
        research_root / _json(_path(inputs["canonical_manifest"]))["input"]["path"]
    )
    annual = pd.read_parquet(annual_path)
    if not {"golgg_match_id", "p_full"}.issubset(annual):
        raise ValueError("Frozen annual A0 artifact lacks golgg_match_id/p_full")
    annual = annual.assign(golgg_match_id=annual.golgg_match_id.astype(str)).set_index(
        "golgg_match_id"
    )
    if not set(identifiers).issubset(annual.index):
        raise ValueError(
            "Frozen annual A0 control does not cover all canonical 2026 IDs"
        )
    canonical["annual_A0"] = annual.loc[identifiers, "p_full"].to_numpy(
        dtype=np.float64
    )
    canonical["tier"] = c0_reference.tier.to_numpy()
    canonical["family"] = c0_reference.family.to_numpy()
    canonical["unknown_players"] = c0_reference.unknown_players.to_numpy(dtype=int)
    prediction_rows = []
    for name, values in zip(
        CONFIGURATIONS, (canonical_c0, canonical_c0_regional, fresh20, fresh20_regional)
    ):
        prediction_rows.extend(
            {
                "match_id": identifier,
                "configuration": name,
                "canonical_a_is_team1": True,
                "probability": float(value) if np.isfinite(value) else np.nan,
                "reason": None
                if np.isfinite(value)
                else "regional_source_support_unavailable",
                "status": "observed"
                if np.isfinite(value)
                else "regional_support_unavailable",
                "observed_history_slots": 19 if name.startswith("fresh20") else 0,
            }
            for identifier, value in zip(identifiers, values)
        )
    assembled = assemble_four_configurations(canonical, pd.DataFrame(prediction_rows))
    score_rows = []
    for name, group in assembled.groupby("configuration", observed=True):
        score_rows.append(
            {
                "configuration": str(name),
                **score_probabilities(group.y.to_numpy(), group.probability.to_numpy()),
            }
        )
    wide = (
        assembled.pivot(index="match_id", columns="configuration", values="probability")
        .join(
            canonical.set_index("match_id")[
                [
                    "y",
                    "month",
                    "best_of",
                    "tier",
                    "family",
                    "unknown_players",
                    "annual_A0",
                ]
            ]
        )
        .reset_index()
    )
    common_support = wide[list(CONFIGURATIONS)].notna().all(axis=1)
    common_support_scores = [
        {
            "configuration": config,
            **score_probabilities(
                wide.loc[common_support, "y"], wide.loc[common_support, config]
            ),
        }
        for config in CONFIGURATIONS
    ]
    slices = {
        "BO1": wide.best_of.eq(1),
        "BO3": wide.best_of.eq(3),
        "BO5": wide.best_of.eq(5),
        "LPL": wide.family.eq("LPL"),
        "Major": wide.tier.astype(str).str.lower().eq("major"),
        "Unseen_Players": wide.unknown_players.gt(0),
    }
    slice_scores = []
    for slice_name, mask in slices.items():
        for config in CONFIGURATIONS:
            score = score_probabilities(wide.loc[mask, "y"], wide.loc[mask, config])
            slice_scores.append({"slice": slice_name, "configuration": config, **score})
    support_masks = {
        "international": {
            "international": wide.tier.astype(str).str.lower().eq("international")
        },
        "player_experience": {
            "all_known_players": wide.unknown_players.eq(0),
            "contains_unseen_players": wide.unknown_players.gt(0),
        },
        "family": {
            str(family): wide.family.eq(family)
            for family in sorted(wide.family.dropna().unique())
        },
    }
    support_metrics = []
    for kind, groups in support_masks.items():
        for group_name, mask in groups.items():
            for config in CONFIGURATIONS:
                score = score_probabilities(wide.loc[mask, "y"], wide.loc[mask, config])
                support_metrics.append(
                    {
                        "support": kind,
                        "slice": group_name,
                        "configuration": config,
                        **score,
                    }
                )
    margin = 0.005
    guard_confidence = 1.0 - 0.05 / 24.0
    guard_rows = []
    for slice_name, mask in slices.items():
        subset = wide.loc[mask].copy()
        for candidate in ("fresh20", "fresh20+regional"):
            for reference in ("C0", "annual_A0"):
                result = paired_monthly_contrast(
                    subset,
                    candidate,
                    reference,
                    draws=max(5000, int(protocol["bootstrap"]["draws"])),
                    seed=int(protocol["bootstrap"]["seed"]),
                    metric="log_loss",
                    confidence=guard_confidence,
                )
                upper = result[f"equal_month_{guard_confidence * 100:.4g}pct"][1]
                support_ok = (
                    result["n_rows"] > 0
                    and result["n_months"] >= 2
                    and upper is not None
                )
                guard_rows.append(
                    {
                        "slice": slice_name,
                        "candidate": candidate,
                        "reference": reference,
                        **result,
                        "margin": margin,
                        "upper_bound": upper,
                        "status": "INSUFFICIENT_SUPPORT"
                        if not support_ok
                        else ("PASS" if upper <= margin else "FAIL"),
                    }
                )
    contrasts = []
    for metric in ("log_loss", "brier"):
        for candidate, reference in PRIMARY_CONTRASTS:
            result = paired_monthly_contrast(
                wide,
                candidate,
                reference,
                draws=max(5000, int(protocol["bootstrap"]["draws"])),
                seed=int(protocol["bootstrap"]["seed"]),
                metric=metric,
                confidence=(
                    float(protocol["bootstrap"]["primary_adjusted_confidence"])
                    if metric == "log_loss"
                    else float(protocol["bootstrap"]["confidence"])
                ),
            )
            contrasts.append(result)
    out = _path(protocol["model_output_root"])
    report = _path(protocol["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    paths = {
        "canonical_predictions": out / "canonical_predictions.parquet",
        "historical_scores": report / "historical_scores.json",
        "historical_contrasts": report / "historical_contrasts.json",
        "historical_slices": report / "historical_slices.json",
        "canonical_proofs": report / "canonical_proofs.json",
        "regional_replay": _path(protocol["feature_root"]) / "regional_replay.parquet",
    }
    assembled.to_parquet(paths["canonical_predictions"], index=False)
    pd.DataFrame(rows).to_parquet(paths["regional_replay"], index=False)
    _write_json(
        paths["canonical_proofs"],
        {
            "regional_replay_receipt": regional.get("receipt"),
            "rows": rows,
            "row_identity_count": len(rows),
            "source_clock_policy": protocol["regional"]["availability"],
            "execution_binding_sha256": hashlib.sha256(
                json.dumps(binding, sort_keys=True, default=_json_default).encode()
            ).hexdigest(),
        },
    )
    _write_json(
        paths["historical_scores"],
        {
            "status": "COMPLETE",
            "denominator": expected,
            "observed_by_configuration": {
                str(x["configuration"]): int(x["n"]) for x in score_rows
            },
            "common_support_scores": common_support_scores,
            "support_slices": support_metrics,
            "regional_unavailable": int((~good).sum()),
            "paired_common_support": int(
                wide[list(CONFIGURATIONS)].notna().all(axis=1).sum()
            ),
            "coverage_policy": protocol["historical"]["coverage"],
            "scope_limitations": [
                protocol["historical"]["canonical_scope"],
                protocol["historical"]["year_limit"],
                protocol["regional"]["interpretation"],
            ],
        },
    )
    _write_json(
        paths["historical_contrasts"],
        {
            "primary": contrasts,
            "draws": int(protocol["bootstrap"]["draws"]),
            "seed": int(protocol["bootstrap"]["seed"]),
            "primary_family_size": int(protocol["bootstrap"]["primary_family_size"]),
            "primary_adjusted_confidence": float(
                protocol["bootstrap"]["primary_adjusted_confidence"]
            ),
            "six_slice_noninferiority": guard_rows,
        },
    )
    _write_json(
        paths["historical_slices"],
        {
            "scores": slice_scores,
            "noninferiority": guard_rows,
            "guard_family_size": 24,
            "margin": margin,
            "confidence": guard_confidence,
            "slices": list(slices),
        },
    )
    daily_path = out / "daily_model_predictions.parquet"
    _build_daily_predictions(protocol, daily_path, research_root)
    first_ready_path = out / "first_ready_predictions.parquet"
    _build_first_ready_predictions(protocol, daily_path, first_ready_path)
    paths["historical_records"] = (
        _path(protocol["feature_root"]) / "historical_records.pkl"
    )
    paths["first_ready_proofs"] = out / "first_ready_proofs.json"
    receipt = {
        "status": "COMPLETE",
        "canonical_rows": expected,
        "regional_replay": regional.get("receipt"),
        "configuration_ids": list(CONFIGURATIONS),
        "score_rows": len(assembled),
        "daily_predictions": str(daily_path),
        "daily_rows": int(len(pd.read_parquet(daily_path))),
        "first_ready_predictions": str(first_ready_path),
        "first_ready_rows": int(len(pd.read_parquet(first_ready_path))),
        "canonical_native_parity_max_abs_c0": c0_error,
        "canonical_saved_fresh20_parity_max_abs": fresh_error,
        "six_slice_noninferiority": {
            f"{row['candidate']}|{row['reference']}|{row['slice']}": row["status"]
            for row in guard_rows
        },
        "zero_fits": True,
        "source_requests": 0,
        "database_queries": 0,
    }
    return _save_stage(protocol, "historical", binding, paths, receipt)


def _isolated_native_c0(
    raw: Mapping[str, np.ndarray],
    history: Mapping[str, np.ndarray],
    best_of: np.ndarray,
) -> np.ndarray:
    import pickle, subprocess, tempfile

    with tempfile.TemporaryDirectory(prefix="c0-region-factorial-native-") as directory:
        source = Path(directory) / "input.pkl"
        destination = Path(directory) / "output.pkl"
        source.write_bytes(
            pickle.dumps((raw, history, best_of), protocol=pickle.HIGHEST_PROTOCOL)
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.analysis.c0_region_factorial",
                "--native-c0-worker",
                str(source),
                str(destination),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "Isolated native C0 inference failed: " + result.stderr[-4000:]
            )
        return pickle.loads(destination.read_bytes())


def _native_c0_worker(source: Path, destination: Path) -> None:
    import os, pickle

    os.environ["BETTING_APP_C0_WORKER"] = "1"
    from betting_app.core.models.native_c0 import NativeC0Adapter
    from src.models.a0_data_focus import predict_models

    raw, history, best_of = pickle.loads(source.read_bytes())
    adapter = NativeC0Adapter()
    adapter.load()
    probabilities = predict_models(
        adapter._backend,
        adapter._shared,
        adapter._models,
        adapter._calibration,
        raw,
        history,
        best_of,
    )["full"]
    from src.models.c0_edge_model_audit import MODES

    result = np.asarray(probabilities[:, MODES.index("full")], dtype=np.float64)
    if (
        result.shape != (len(best_of),)
        or not np.isfinite(result).all()
        or np.any((result <= 0) | (result >= 1))
    ):
        raise ValueError(
            "Saved native C0 inference returned malformed series probabilities"
        )
    destination.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))


def _build_first_ready_predictions(
    protocol: Mapping[str, Any], daily_path: Path, destination: Path
) -> None:
    source = (
        _path(protocol["inputs"]["edge_root"]) / "alpha" / "prepared_cohort.parquet"
    )
    if not source.is_file():
        raise FileNotFoundError("Frozen first-ready prepared cohort is missing")
    cohort = pd.read_parquet(source)
    expected = int(protocol["historical"]["market_first_ready_expected_rows"])
    if len(cohort) != expected:
        raise ValueError(f"Expected {expected} first-ready rows, got {len(cohort)}")
    expected_fixtures = int(
        protocol["historical"]["market_first_ready_expected_fixtures"]
    )
    if cohort.canonical_match_id.astype(str).nunique() != expected_fixtures:
        raise ValueError(
            "First-ready fixture denominator differs from the frozen 302-fixture cohort"
        )
    daily = pd.read_parquet(daily_path)
    cohort["canonical_match_id"] = cohort.canonical_match_id.astype(str)
    cohort["requested_utc_day"] = pd.to_datetime(
        cohort.decision_at, utc=True
    ).dt.strftime("%Y-%m-%d")
    daily["canonical_match_id"] = daily.canonical_match_id.astype(str)
    daily["requested_utc_day"] = daily.requested_utc_day.astype(str)
    cohort["decision_at"] = pd.to_datetime(cohort.decision_at, utc=True)
    daily["decision_at"] = pd.to_datetime(daily.decision_at, utc=True)
    model_columns = [
        "canonical_match_id",
        "requested_utc_day",
        "decision_at",
        "forecast_at",
        "feature_source_max_day",
        "daily_state_cutoff_day",
        "strict_prior_feature_day_proof",
        "usable_for_same_day_projection",
        "p_C0",
        "p_C0_regional",
        "p_fresh20",
        "p_fresh20_regional",
        "regional_status",
        "regional_state_sha256",
    ]
    model = daily[model_columns].copy()
    result = cohort.merge(
        model,
        on=["canonical_match_id", "requested_utc_day"],
        how="left",
        suffixes=("", "_query"),
        validate="many_to_one",
    )

    def query_column(name):
        key = f"{name}_query" if f"{name}_query" in result else name
        return result[key]

    query_clock = pd.to_datetime(result.decision_at_query, utc=True, errors="coerce")
    forecast_clock = pd.to_datetime(
        query_column("forecast_at"), utc=True, errors="coerce"
    )
    source_day = pd.to_datetime(
        query_column("feature_source_max_day"), utc=True, errors="coerce"
    )
    cutoff_day = pd.to_datetime(
        query_column("daily_state_cutoff_day"), utc=True, errors="coerce"
    )
    target_day = result.decision_at.dt.normalize()
    admissible = (
        query_clock.le(result.decision_at)
        & query_clock.dt.normalize().eq(target_day)
        & forecast_clock.le(result.decision_at)
        & source_day.lt(target_day)
        & cutoff_day.dt.normalize().eq(target_day)
        & query_column("strict_prior_feature_day_proof").eq(True)
        & query_column("usable_for_same_day_projection").eq(True)
    )
    bad = ~admissible
    for column in ("p_C0", "p_C0_regional", "p_fresh20", "p_fresh20_regional"):
        result.loc[bad, column] = np.nan
    result["factorial_query_status"] = np.where(
        bad, "daily_state_or_clock_absent", "observed"
    )
    result["configuration_ids"] = [list(CONFIGURATIONS)] * len(result)
    if len(result) != expected:
        raise ValueError("First-ready output changed the frozen denominator")
    result.to_parquet(destination, index=False)
    proof_columns = [
        "opportunity_id",
        "canonical_match_id",
        "decision_at",
        "decision_at_query",
        "forecast_at_query",
        "strict_prior_feature_day_proof",
        "daily_state_cutoff_day",
        "feature_source_max_day",
        "factorial_query_status",
        "regional_state_sha256",
    ]
    _write_json(
        destination.with_name("first_ready_proofs.json"),
        {
            "rows": result[
                [column for column in proof_columns if column in result]
            ].to_dict("records"),
            "row_count": len(result),
            "admitted_projection_count": int(admissible.sum()),
            "exact_clock_match_count": int(result.decision_at.eq(query_clock).sum()),
        },
    )


def _build_daily_predictions(
    protocol: Mapping[str, Any], destination: Path, research_root: Path
) -> None:
    """Build one shared causal query cache and export all four daily model cells."""
    inputs = protocol["inputs"]
    existing = _path(inputs["edge_root"]) / "models" / "daily_model_predictions.parquet"
    if not existing.is_file():
        raise FileNotFoundError("Frozen edge daily model predictions are required")
    daily = pd.read_parquet(existing)
    required = {
        "canonical_match_id",
        "requested_utc_day",
        "decision_at",
        "usable_for_same_day_projection",
        "forecast_at",
        "strict_prior_feature_day_proof",
        "daily_state_cutoff_day",
        "feature_source_max_day",
    }
    if not required.issubset(daily):
        raise ValueError(
            "Frozen daily model file lacks temporal proof fields: "
            + ", ".join(sorted(required - set(daily)))
        )
    expected = int(protocol["historical"]["market_daily_expected_usable_states"])
    daily = daily.loc[daily.usable_for_same_day_projection.fillna(False)].copy()
    if len(daily) != expected:
        raise ValueError(
            f"Expected {expected} usable frozen daily states, got {len(daily)}"
        )
    daily["canonical_match_id"] = daily.canonical_match_id.astype(str)
    daily["requested_utc_day"] = daily.requested_utc_day.astype(str)
    daily["decision_at"] = pd.to_datetime(daily.decision_at, utc=True)
    from src.models.c0_edge_model_audit import (
        _build_causal_query_batches_isolated,
        build_prior_query_requests,
    )

    requests, absences = build_prior_query_requests(daily, group_by_decision=True)
    if not requests:
        raise ValueError("No valid frozen daily causal native requests")
    records, raw_receipt = _get_raw_map_cache(
        protocol, research_root, cache_name="historical_records.pkl"
    )
    query = _build_causal_query_batches_isolated(
        requests, research_root=research_root, records_by_player=records
    )
    batch = query["batches"]["20t"]
    query_requests = query["requests"]
    if len(query_requests) != len(batch["best_of"]):
        raise ValueError("Daily causal query rows and model batch differ in length")
    from src.models.c0_regional_replay import build_regional_query_states

    regional_requests = [
        {
            "opportunity_id": str(item["opportunity_id"]),
            "cutoff_day": pd.to_datetime(item["decision_at"], utc=True)
            .date()
            .isoformat(),
            "team1_id": str(item["team1_id"]),
            "team2_id": str(item["team2_id"]),
            "roster_a": list(map(str, item["roster_a"])),
            "roster_b": list(map(str, item["roster_b"])),
        }
        for item in query_requests
    ]
    regional = build_regional_query_states(
        regional_requests,
        raw_history_path=_path(inputs["raw_history"]),
        release_map_path=research_root / inputs["release_map_relative"],
    )
    if len(regional["rows"]) != len(query_requests):
        raise ValueError("Daily regional replay did not preserve every query row")
    offsets_a = np.asarray(
        [r.get("offset_a", np.nan) for r in regional["rows"]], dtype=np.float32
    )
    offsets_b = np.asarray(
        [r.get("offset_b", np.nan) for r in regional["rows"]], dtype=np.float32
    )
    statuses = np.asarray([r.get("status") for r in regional["rows"]], dtype=object)
    supported = np.isfinite(offsets_a) & np.isfinite(offsets_b) & (statuses == "ok")
    from src.models.c0_distilled_reproduction import infer

    fresh20 = np.asarray(
        infer(_path(inputs["fresh_protocol"]), "20t", batch, batch["best_of"]),
        dtype=np.float64,
    )
    raw_regional = {
        key: np.array(value, copy=True) for key, value in batch["raw"].items()
    }
    history_regional = {
        key: np.array(value, copy=True) for key, value in batch["history"].items()
    }
    fresh20_regional = np.full(len(query_requests), np.nan, dtype=np.float64)
    c0_regional = np.full(len(query_requests), np.nan, dtype=np.float64)
    if supported.any():
        indices = np.flatnonzero(supported)
        raw_subset = {key: value[indices] for key, value in batch["raw"].items()}
        history_subset = {
            key: value[indices] for key, value in batch["history"].items()
        }
        raw_norm, history_norm = transform_batch_once(
            raw_subset, history_subset, offsets_a[indices], offsets_b[indices]
        )
        raw_regional = {
            key: np.array(value, copy=True) for key, value in batch["raw"].items()
        }
        history_regional = {
            key: np.array(value, copy=True) for key, value in batch["history"].items()
        }
        for key, values in raw_norm.items():
            raw_regional[key][indices] = values
        for key, values in history_norm.items():
            history_regional[key][indices] = values
        fresh20_regional[indices] = infer(
            _path(inputs["fresh_protocol"]),
            "20t",
            {"raw": raw_norm, "history": history_norm},
            batch["best_of"][indices],
        )
        recent = {
            **history_norm,
            "history": history_norm["history"][:, :, :, -16:],
            "mask": history_norm["mask"][:, :, :, -16:],
        }
        c0_regional[indices] = _isolated_native_c0(
            raw_norm, recent, batch["best_of"][indices]
        )
    c0 = np.asarray(query["p_c0"], dtype=np.float64)
    oriented_c0 = np.where(
        [bool(r["canonical_a_is_team1"]) for r in query_requests], c0, 1.0 - c0
    )
    oriented_c0_regional = np.where(
        [bool(r["canonical_a_is_team1"]) for r in query_requests],
        c0_regional,
        1.0 - c0_regional,
    )
    oriented_fresh = np.where(
        [bool(r["canonical_a_is_team1"]) for r in query_requests],
        fresh20,
        1.0 - fresh20,
    )
    oriented_fresh_regional = np.where(
        [bool(r["canonical_a_is_team1"]) for r in query_requests],
        fresh20_regional,
        1.0 - fresh20_regional,
    )
    model_rows = []
    for idx, request in enumerate(query_requests):
        day = pd.to_datetime(request["decision_at"], utc=True).date().isoformat()
        model_rows.append(
            {
                "canonical_match_id": str(request["canonical_match_id"]),
                "requested_utc_day": day,
                "decision_at": request["decision_at"],
                "p_C0": oriented_c0[idx],
                "p_C0_regional": oriented_c0_regional[idx],
                "p_fresh20": oriented_fresh[idx],
                "p_fresh20_regional": oriented_fresh_regional[idx],
                "regional_status": str(regional["rows"][idx].get("status")),
                "regional_state_sha256": regional["rows"][idx].get("state_sha256"),
            }
        )
    scored = pd.DataFrame(model_rows)
    output = daily.merge(
        scored,
        on=["canonical_match_id", "requested_utc_day", "decision_at"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_new"),
    )
    if len(output) != expected:
        raise ValueError(
            "Daily prediction join changed the frozen opportunity denominator"
        )
    for col in ("p_C0", "p_C0_regional", "p_fresh20", "p_fresh20_regional"):
        output[col] = pd.to_numeric(output[col], errors="coerce")
    output["raw_map_receipt_sha256"] = hashlib.sha256(
        json.dumps(raw_receipt, sort_keys=True, default=_json_default).encode()
    ).hexdigest()
    output["regional_status"] = output.regional_status.fillna("query_absent")
    output["regional_state_sha256"] = output.regional_state_sha256.where(
        output.regional_state_sha256.notna(), None
    )
    for predicted, observed in (("p_C0", "p_c0"), ("p_fresh20", "p_c0d_20t")):
        paired = output[[predicted, observed]].dropna()
        if len(paired) and not np.allclose(
            paired[predicted], paired[observed], atol=2e-6, rtol=0
        ):
            raise ValueError(
                f"Rebuilt daily {predicted} does not reproduce frozen {observed}"
            )
    _write_json(
        destination.with_name("daily_query_proofs.json"),
        {
            "raw_map_receipt": raw_receipt,
            "regional_receipt": regional.get("receipt"),
            "query_proofs": query.get("query_proofs"),
            "query_absences": absences,
            "usable_states": expected,
            "model_query_rows": len(query_requests),
            "regional_supported_rows": int(supported.sum()),
            "clock_policy": "exact fixture and UTC-day state; retained decision_at/forecast_at/proofs",
        },
    )
    output.to_parquet(destination, index=False)


def _run_finance(
    protocol: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    cached = _cached(protocol, "finance", binding)
    if cached:
        return cached
    from src.analysis.c0_region_finance import run_financial_comparison

    sealed = _validated_sealed_daily_predictions(protocol)
    projection_artifacts: dict[str, Path] = {}
    if sealed is not None:
        daily_path, source_hashes = sealed
        projection_root = _path(protocol["model_output_root"])
        if projection_root == daily_path.parent:
            raise ValueError("sealed projections must use the new model-output root")
        first_ready_path = projection_root / "first_ready_predictions.parquet"
        projection_root.mkdir(parents=True, exist_ok=True)
        _build_first_ready_predictions(protocol, daily_path, first_ready_path)
        projection_artifacts = {
            "first_ready_predictions": first_ready_path,
            "first_ready_proofs": first_ready_path.with_name("first_ready_proofs.json"),
        }
    else:
        history_stage = _cached(protocol, "historical", binding)
        if history_stage is None:
            raise RuntimeError("finance requires a completed historical stage")
        daily_path = (
            _path(protocol["model_output_root"]) / "daily_model_predictions.parquet"
        )
        if not daily_path.is_file():
            raise FileNotFoundError(
                "historical daily_model_predictions.parquet is required for finance"
            )
        source_hashes = {}
    output_dir = _path(protocol["output_root"])
    receipt = run_financial_comparison(protocol, daily_path, output_dir=output_dir)
    if sealed is not None:
        receipt["external_nn_provenance"] = {
            "protocol": str(_path(protocol["inputs"]["sealed_historical_protocol"])),
            "binding": str(_path(protocol["inputs"]["sealed_historical_binding"])),
            "historical_stage": str(
                _path(protocol["inputs"]["sealed_historical_stage"])
            ),
            "daily_predictions": str(daily_path),
            "source_sha256": source_hashes,
        }
    artifacts = {
        p.stem: p
        for p in output_dir.glob("*")
        if p.is_file() and p.name != "finance_stage.json"
    }
    artifacts.update(projection_artifacts)
    record = _save_stage(protocol, "finance", binding, artifacts, receipt)
    _write_json(output_dir / "finance_stage.json", record)
    return record


def run_stage(protocol_path: str | Path, stage: str) -> dict[str, Any]:
    if stage not in ("prepare", "worlds", "historical", "finance", "all"):
        raise ValueError("stage must be prepare, worlds, historical, finance, or all")
    if stage == "prepare":
        return prepare(protocol_path)
    protocol = _json(_path(protocol_path))
    protocol["_protocol_path"] = str(_path(protocol_path))
    binding = _require_binding(protocol)
    if stage == "worlds":
        return _run_worlds(protocol, binding)
    if stage == "historical":
        return _run_historical(protocol, binding)
    if stage == "finance":
        return _run_finance(protocol, binding)
    _run_worlds(protocol, binding)
    _run_historical(protocol, binding)
    return _run_finance(protocol, binding)


def main(argv: list[str] | None = None) -> int:
    import argparse

    values = list(sys.argv[1:] if argv is None else argv)
    if values[:1] == ["--native-c0-worker"]:
        if len(values) != 3:
            raise ValueError("native C0 worker requires input and output pickle paths")
        _native_c0_worker(Path(values[1]), Path(values[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "worlds", "historical", "finance", "all"),
    )
    args = parser.parse_args(values)
    print(
        json.dumps(
            run_stage(args.protocol, args.stage), sort_keys=True, default=_json_default
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
