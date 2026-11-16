from __future__ import annotations

from copy import deepcopy
from datetime import timezone

import numpy as np
import pytest

from betting_app.core.models import native_c0
from betting_app.core.models.native_c0 import (
    NativeC0Adapter,
    encode_snapshot,
)


def test_serving_source_contract_rejects_tampered_version_and_hash(tmp_path) -> None:
    import json

    config, _, _ = native_c0.load_serving_config()
    config_path = tmp_path / "serving.json"
    config["source_contract_version"] = "native-c0-serving-source-v1"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="source-contract version"):
        native_c0.load_serving_config(config_path)

    config, _, _ = native_c0.load_serving_config()
    config["feature_replay_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="feature replay source"):
        native_c0.load_serving_config(config_path)


def test_serving_config_rejects_noncanonical_raw_history_path(tmp_path) -> None:
    import json

    config, _, _ = native_c0.load_serving_config()
    config["raw_history_path"] = "data/other/matches.json"
    config_path = tmp_path / "serving.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_history_path"):
        native_c0.load_serving_config(config_path)




def test_client_identity_schema_tracks_configured_source_contract(monkeypatch) -> None:
    from betting_app.services import c0_inference

    config, config_path, config_sha256 = native_c0.load_serving_config()
    identity = {
        "name": native_c0.MODEL_NAME,
        "version": native_c0.MODEL_VERSION,
        "family": native_c0.MODEL_FAMILY,
        "feature_version": native_c0.FEATURE_VERSION,
        "weights_model_version": native_c0.WEIGHTS_MODEL_VERSION,
        "weights_feature_version": native_c0.WEIGHTS_FEATURE_VERSION,
        "regional_default_enabled": True,
        "regional_adjustment_version": native_c0.REGIONAL_ADJUSTMENT_VERSION,
        "has_uncertainty": False,
        "run_id": "c0d_repro_20261021_01",
        "variant": "20t",
        "artifact_path": str(native_c0.PROJECT_ROOT / native_c0._FRESH_MODEL_ROOT_RELATIVE),
        "execution_sha256": "e" * 64,
        "completion_sha256": native_c0._COMPLETION_SHA256,
        "artifact_hashes": {
            "bundle_manifest.json": native_c0._FRESH_BUNDLE_MANIFEST_SHA256,
            "calibration.json": native_c0._FRESH_CALIBRATION_SHA256,
            **{
                f"seed_{seed}/checkpoint_epoch16.pt": digest
                for seed, digest in native_c0._FRESH_CHECKPOINT_SHA256.items()
            },
        },
        "training_source_hashes": {"source": "b" * 64},
        "training_source_provenance": "fresh_execution_binding_with_original_shared_expert",
        "fresh_candidate": "20t",
        "fresh_protocol_path": str(native_c0.PROJECT_ROOT / config["fresh_protocol_path"]),
        "fresh_bundle_manifest_sha256": native_c0._FRESH_BUNDLE_MANIFEST_SHA256,
        "fresh_checkpoint_hashes": native_c0._FRESH_CHECKPOINT_SHA256,
        "fresh_calibration_sha256": native_c0._FRESH_CALIBRATION_SHA256,
        "shared_expert_sha256": native_c0._FRESH_SHARED_SHA256,
        "reference_calibration_sha256": native_c0._FRESH_REFERENCE_CALIBRATION_SHA256,
        "serving_source_hashes": {
            "project:src/analysis/ev_asof_replay.py": config["feature_replay_sha256"],
        },
        "raw_feature_replay_source": {
            "implementation_path": "/replay",
            "implementation_sha256": config["feature_replay_sha256"],
            "feature_archive_path": "/archive",
        },
        "canonical_raw_history_source": {
            "path": "/history",
            "sha256": config["raw_history_sha256"],
            "source_kind": config["raw_history_source_kind"],
        },
        "serving_config_path": str(config_path),
        "serving_config_sha256": config_sha256,
        "scope": "DEVELOPMENT_ONLY",
        "source_contract_version": config["source_contract_version"],
        "forward_status": config["forward_status"],
    }
    monkeypatch.setattr(
        c0_inference, "load_serving_config",
        lambda: (config, config_path, config_sha256),
    )
    assert c0_inference._validate_identity(identity) == identity
    for field, value in (
        ("source_contract_version", "native-c0-serving-source-v1"),
        ("serving_config_sha256", "0" * 64),
        ("version", native_c0.WEIGHTS_MODEL_VERSION),
        ("feature_version", native_c0.WEIGHTS_FEATURE_VERSION),
        ("regional_default_enabled", False),
        ("regional_adjustment_version", "unverified-regional-law"),
    ):
        altered = deepcopy(identity)
        altered[field] = value
        with pytest.raises(c0_inference.C0InferenceError):
            c0_inference._validate_identity(altered)

    altered = deepcopy(identity)
    altered["serving_source_hashes"] = {}
    with pytest.raises(c0_inference.C0InferenceError):
        c0_inference._validate_identity(altered)



DECISION = "2026-10-02T12:00:00+00:00"
ROSTER_A = ["a1", "a2", "a3", "a4", "a5"]
ROSTER_B = ["b1", "b2", "b3", "b4", "b5"]


def _adapter() -> NativeC0Adapter:
    return NativeC0Adapter.__new__(NativeC0Adapter)


def _request(best_of: int = 3, *, decision_at: str = DECISION, mode: str = "full") -> dict:
    return {
        "team1_id": "team-a",
        "team2_id": "team-b",
        "roster_a": list(reversed(ROSTER_A)),
        "roster_b": list(reversed(ROSTER_B)),
        "best_of": best_of,
        "decision_at": decision_at,
        "competition_context": "Worlds",
        "start_at": "2026-10-02T14:00:00Z",
        "mode": mode,
    }


def _snapshot() -> dict:
    raw = {
        "players": np.arange(2 * 5 * 21, dtype=np.float32).reshape(2, 5, 21),
        "core": np.arange(2 * 5, dtype=np.float32).reshape(2, 5),
        "team": np.arange(2 * 11, dtype=np.float32).reshape(2, 11),
        "w20": np.arange(2 * 10, dtype=np.float32).reshape(2, 10),
        "gates": np.arange(2 * 2, dtype=np.float32).reshape(2, 2),
    }
    hist = {
        "history": np.arange(2 * 5 * 19 * 33, dtype=np.float32).reshape(2, 5, 19, 33),
        "mask": np.arange(2 * 5 * 19, dtype=np.uint8).reshape(2, 5, 19) % 2,
        "champions": np.arange(2 * 5 * 16, dtype=np.uint16).reshape(2, 5, 16),
    }
    target = {
        "team1_id": "team-a",
        "team2_id": "team-b",
        "best_of": 3,
        "decision_at": DECISION,
        "effective_decision_at": DECISION,
        "start_at": "2026-10-02T14:00:00Z",
        "competition_context": "Worlds",
        "tournament_name": "Worlds",
        "roster_a": list(reversed(ROSTER_A)),
        "roster_b": list(reversed(ROSTER_B)),
        "cutoff_day": "2026-10-02",
        "feature_source_max_day": "2026-10-01",
        "feature_available": True,
    }
    for side, roster in (("a", ROSTER_A), ("b", ROSTER_B)):
        for key in (
            f"tensor_player_order_{side}",
            f"raw_tensor_player_order_{side}",
            f"temporal_tensor_player_order_{side}",
        ):
            target[key] = list(roster)
    return {"target": target, "raw": raw, "hist": hist}


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_native_request_accepts_supported_series_formats_and_manual_ids(best_of: int) -> None:
    request = _adapter()._validate_request(_request(best_of, mode="no_w20"))

    assert request["best_of"] == best_of
    assert request["team1_id"] == "team-a"
    assert request["roster_a"] == sorted(ROSTER_A)
    assert request["decision_at"].tzinfo == timezone.utc
    assert request["mode"] == "no_w20"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"decision_at": "2026-10-02T12:00:00"}, "timezone"),
        ({"decision_at": "2025-12-31T23:59:00Z"}, "pre-2026-01-01"),
        ({"best_of": 7}, "best_of"),
        ({"roster_a": ["a1", "a1", "a3", "a4", "a5"]}, "unique"),
    ],
)
def test_native_request_rejects_invalid_or_leaking_inputs(overrides: dict, message: str) -> None:
    request = _request()
    request.update(overrides)

    with pytest.raises(ValueError, match=message):
        _adapter()._validate_request(request)


def test_exact_native_snapshot_wire_round_trip_preserves_tensor_dtype_and_values() -> None:
    adapter = _adapter()
    request = adapter._validate_request(_request())
    encoded = encode_snapshot(_snapshot())
    target, raw, hist = adapter._decode_snapshot(encoded, request)

    assert target["feature_source_max_day"] == "2026-10-01"
    for family, expected in (("raw", _snapshot()["raw"]), ("hist", _snapshot()["hist"])):
        actual = raw if family == "raw" else hist
        for key, array in expected.items():
            assert actual[key].dtype == array.dtype
            np.testing.assert_array_equal(actual[key], array)


def test_native_snapshot_rejects_history_on_decision_day() -> None:
    adapter = _adapter()
    request = adapter._validate_request(_request())
    snapshot = _snapshot()
    snapshot["target"]["feature_source_max_day"] = "2026-10-02"

    with pytest.raises(ValueError, match="decision-day boundary"):
        adapter._decode_snapshot(encode_snapshot(snapshot), request)


def test_native_snapshot_requires_every_exact_sorted_player_slot_order() -> None:
    adapter = _adapter()
    request = adapter._validate_request(_request())
    snapshot = _snapshot()
    del snapshot["target"]["temporal_tensor_player_order_a"]

    with pytest.raises(ValueError, match="omits exact native player-slot order"):
        adapter._decode_snapshot(encode_snapshot(snapshot), request)


def test_native_snapshot_rejects_tensor_dtype_substitution() -> None:
    snapshot = _snapshot()
    snapshot["raw"]["players"] = snapshot["raw"]["players"].astype(np.float64)

    with pytest.raises(ValueError, match="requires shape"):
        encode_snapshot(snapshot)


def test_raw_history_defaults_to_canonical_source_and_allows_explicit_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_c0, "PROJECT_ROOT", tmp_path)
    canonical = tmp_path / native_c0._CANONICAL_RAW_HISTORY_RELATIVE
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical raw source")
    config = {"raw_history_path": native_c0._CANONICAL_RAW_HISTORY_RELATIVE}

    default_path, default_kind, default_is_canonical = native_c0._resolve_raw_history_source(config, None)
    assert default_path == canonical
    assert default_kind == native_c0._CANONICAL_RAW_HISTORY_KIND
    assert default_is_canonical is True

    override = tmp_path / "approved-alternative.json"
    override.write_bytes(b"explicit alternative")
    override_path, override_kind, override_is_canonical = native_c0._resolve_raw_history_source(
        config, str(override),
    )
    assert override_path == override
    assert override_kind == "explicit_raw_history_override"
    assert override_is_canonical is False


def test_missing_explicit_raw_history_fails_without_automatic_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_c0, "PROJECT_ROOT", tmp_path)
    canonical = tmp_path / native_c0._CANONICAL_RAW_HISTORY_RELATIVE
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical raw source")
    config = {"raw_history_path": native_c0._CANONICAL_RAW_HISTORY_RELATIVE}

    with pytest.raises(FileNotFoundError):
        native_c0._resolve_raw_history_source(config, str(tmp_path / "missing-override.json"))


@pytest.fixture(scope="module")
def genuine_native_client(require_native_artifacts):
    from betting_app.services.c0_inference import C0InferenceClient

    client = C0InferenceClient()
    try:
        yield client
    finally:
        client.close()


def test_genuine_native_diagnostics_bind_exact_request_without_changing_probability(
    genuine_native_client,
    monkeypatch,
) -> None:
    monkeypatch.delenv("C0_RAW_HISTORY_PATH", raising=False)
    monkeypatch.delenv("C0_RELEASE_MAP_PATH", raising=False)
    request = _request()
    original = deepcopy(request)
    ordinary = genuine_native_client.predict_request(request)
    evidence = genuine_native_client.predict_evidence(request)
    diagnostics = ordinary.diagnostics

    assert request == original
    assert evidence.prediction.prob_a == pytest.approx(ordinary.prob_a, abs=1e-12)
    assert ordinary.prob_b == pytest.approx(1.0 - ordinary.prob_a, abs=1e-12)
    assert ordinary.map_prob_a is None
    assert ordinary.map_prob_b is None
    assert ordinary.epistemic_sigma_z is None
    assert ordinary.p_low_a is None
    assert ordinary.p_low_b is None
    assert diagnostics["probability_target"] == "series_win_already_projected"
    assert diagnostics["uncertainty_available"] is False
    assert diagnostics["information_status_basis"] == "declared_input_provenance_not_authenticated"
    provenance = diagnostics["feature_provenance"]
    import hashlib
    from pathlib import Path

    config, _, _ = native_c0.load_serving_config()
    expected_history = native_c0.PROJECT_ROOT / config["serving_raw_history_path"]
    assert Path(provenance["raw_history_path"]).resolve() == expected_history.resolve()
    with expected_history.open("rb") as source:
        assert provenance["raw_history_sha256"] == hashlib.file_digest(source, "sha256").hexdigest()
    freshness = provenance["dataset_freshness"]
    consumed_day = freshness["feature_history_release_cutoff_day"]
    assert consumed_day < freshness["query_decision_day_exclusive_cutoff"]
    assert consumed_day <= freshness["raw_history_max_effective_release_day"]
    assert diagnostics["input_hashes"] == evidence.prediction.diagnostics["input_hashes"]
    assert diagnostics["c0_request"] == {
        **original,
        "roster_a": sorted(ROSTER_A),
        "roster_b": sorted(ROSTER_B),
        "start_at": "2026-10-02T14:00:00+00:00",
    }
 
def test_native_snapshot_rejects_retained_sixteen_slot_input_without_fallback() -> None:
    adapter = _adapter()
    request = adapter._validate_request(_request())
    snapshot = _snapshot()
    snapshot["hist"]["history"] = snapshot["hist"]["history"][:, :, 3:, :]
    snapshot["hist"]["mask"] = snapshot["hist"]["mask"][:, :, 3:]
    with pytest.raises(ValueError, match="shape"):
        adapter._decode_snapshot(encode_snapshot(snapshot), request)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"roster_a": ["a1", "a2", "a3", "a4", "replacement"]}, "roster_a differs"),
        ({"decision_at": "2026-10-02T12:01:00+00:00"}, "decision clocks differ"),
        ({"start_at": "2026-10-02T15:00:00+00:00"}, "start_at differs"),
        ({"competition_context": "MSI"}, "competition context differs"),
    ],
    ids=["roster", "decision_clock", "start_clock", "competition_context"],
)
def test_native_snapshot_withholds_changed_request_instead_of_relabeling_tensors(
    overrides: dict,
    message: str,
) -> None:
    adapter = _adapter()
    snapshot = encode_snapshot(_snapshot())
    original = deepcopy(snapshot)
    request = {**_request(), **overrides}

    with pytest.raises(ValueError, match=message):
        adapter._decode_snapshot(snapshot, adapter._validate_request(request))

    assert snapshot == original
