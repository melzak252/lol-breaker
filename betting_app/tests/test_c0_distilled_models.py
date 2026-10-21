"""Unit tests for the C0 Distilled Model Family (C0-D, C0-Slim, C0-Mini)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from betting_app.core.models import (
    A0_REFERENCE,
    ACTIVE_MODEL_NAME,
    C0_DISTILLED,
    C0_MINI,
    C0_NATIVE,
    C0_SLIM,
    EXP039_THESIS,
    MODEL_REGISTRY,
    REGISTERED_MODELS,
    get_active_model,
    get_model,
    list_registered_models,
    resolve_model_spec,
)
from scripts.predict_matchup import apply_model_calibration

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_c0_distilled_models_registration_and_properties():
    """Verify C0-D, C0-Slim, and C0-Mini specs have correct attributes and metadata."""
    # Active model invariants
    assert ACTIVE_MODEL_NAME == C0_NATIVE.name == "Causal-C0"
    assert get_active_model() is C0_NATIVE
    assert MODEL_REGISTRY is REGISTERED_MODELS

    # C0-D
    assert C0_DISTILLED.effective_short_code == "C0-D"
    assert C0_DISTILLED.short_code == "C0-D"
    assert C0_DISTILLED.name == "Causal-C0-D"
    assert C0_DISTILLED.effective_display_name == "BornAgain-Distilled-Attention-Ensemble"
    assert C0_DISTILLED.version == "c0-d-born-again-w32-v1"
    assert C0_DISTILLED.family == "distilled_c0"
    assert C0_DISTILLED.prediction_target == "series"
    assert C0_DISTILLED.has_uncertainty is False
    assert C0_DISTILLED.metadata["operational"] is True
    assert C0_DISTILLED.metadata["neural_params"] == 19914
    assert C0_DISTILLED.metadata["neural_params_total"] == 19914
    assert C0_DISTILLED.metadata["neural_params_per_seed"] == 6638
    assert C0_DISTILLED.metadata["log_loss"] == pytest.approx(0.534134)
    assert C0_DISTILLED.metadata["log_loss_pure"] == pytest.approx(0.534134)
    assert C0_DISTILLED.metadata["log_loss_with_connector_ons"] == pytest.approx(0.533984)
    assert C0_DISTILLED.metadata["log_loss_17token_core"] == pytest.approx(0.534418)
    assert C0_DISTILLED.metadata["brier_score"] == pytest.approx(0.180468)
    assert C0_DISTILLED.metadata["brier"] == pytest.approx(0.180468)
    assert C0_DISTILLED.metadata["ece"] == pytest.approx(0.018708)
    assert "C0-D-Year" in C0_DISTILLED.metadata["aliases"]
    assert "C0-D-Year-Core" in C0_DISTILLED.metadata["aliases"]
    # C0-Slim
    assert C0_SLIM.effective_short_code == "C0-Slim"
    assert C0_SLIM.short_code == "C0-Slim"
    assert C0_SLIM.name == "Causal-C0-Slim"
    assert C0_SLIM.effective_display_name == "SingleSeed-Distilled-Attention"
    assert C0_SLIM.version == "c0-slim-1seed-w32-v1"
    assert C0_SLIM.family == "distilled_c0"
    assert C0_SLIM.metadata["neural_params"] == 6638
    assert C0_SLIM.metadata["neural_params_total"] == 6638
    assert C0_SLIM.metadata["neural_params_per_seed"] == 6638
    assert C0_SLIM.metadata["log_loss"] == pytest.approx(0.534656)
    assert C0_SLIM.metadata["log_loss_pure"] == pytest.approx(0.534656)
    assert C0_SLIM.metadata["log_loss_17token"] == pytest.approx(0.535062)
    assert C0_SLIM.metadata["log_loss_connector"] == pytest.approx(0.534356)
    assert C0_SLIM.metadata["brier_score"] == pytest.approx(0.180667)
    assert C0_SLIM.metadata["brier"] == pytest.approx(0.180667)
    assert C0_SLIM.metadata["ece"] == pytest.approx(0.019236)
    assert "C0-D1" in C0_SLIM.metadata["aliases"]
    assert "C0-Slim-Year" in C0_SLIM.metadata["aliases"]
    assert "C0-D1-Year" in C0_SLIM.metadata["aliases"]
    assert "C0-Slim+" in C0_SLIM.metadata["aliases"]
    # C0-Mini
    assert C0_MINI.effective_short_code == "C0-Mini"
    assert C0_MINI.short_code == "C0-Mini"
    assert C0_MINI.name == "Causal-C0-Mini"
    assert C0_MINI.effective_display_name == "HalfWidth-Distilled-Attention"
    assert C0_MINI.version == "c0-mini-w16-kd-v1"
    assert C0_MINI.family == "distilled_c0"
    assert C0_MINI.metadata["neural_params_total"] == 8394
    assert C0_MINI.metadata["neural_params_per_seed"] == 2798
    assert C0_MINI.metadata["log_loss"] == pytest.approx(0.535845)
    assert "C0-Small" in C0_MINI.metadata["aliases"]

    # Registered list serialization check
    registered = list_registered_models()
    short_codes = {m["short_code"] for m in registered}
    assert {"C0", "C0-D", "C0-Slim", "C0-Mini", "EXP-078", "EXP-039"}.issubset(short_codes)


@pytest.mark.parametrize(
    ("identifier", "expected_spec"),
    [
        ("c0", C0_NATIVE),
        ("C0", C0_NATIVE),
        ("Causal-C0", C0_NATIVE),
        ("native_c0", C0_NATIVE),
        ("PlayerHistory-Attention-Ensemble", C0_NATIVE),
        ("c0-d", C0_DISTILLED),
        ("C0-D", C0_DISTILLED),
        ("c0_d", C0_DISTILLED),
        ("C0-D-Core", C0_DISTILLED),
        ("c0-d-core", C0_DISTILLED),
        ("c0_d_core", C0_DISTILLED),
        ("C0-D-Year", C0_DISTILLED),
        ("c0-d-year", C0_DISTILLED),
        ("c0_d_year", C0_DISTILLED),
        ("C0-D-Year-Core", C0_DISTILLED),
        ("c0-d-year-core", C0_DISTILLED),
        ("Causal-C0-D", C0_DISTILLED),
        ("c0-d-born-again-w32-v1", C0_DISTILLED),
        ("BornAgain-Distilled-Attention-Ensemble", C0_DISTILLED),
        ("c0-slim", C0_SLIM),
        ("C0-Slim", C0_SLIM),
        ("c0_slim", C0_SLIM),
        ("c0-d1", C0_SLIM),
        ("C0-D1", C0_SLIM),
        ("c0_d1", C0_SLIM),
        ("C0-Slim-Year", C0_SLIM),
        ("c0-slim-year", C0_SLIM),
        ("c0_slim_year", C0_SLIM),
        ("C0-D1-Year", C0_SLIM),
        ("c0-d1-year", C0_SLIM),
        ("C0-Slim+", C0_SLIM),
        ("c0-slim+", C0_SLIM),
        ("Causal-C0-Slim", C0_SLIM),
        ("c0-slim-1seed-w32-v1", C0_SLIM),
        ("SingleSeed-Distilled-Attention", C0_SLIM),
        ("c0-mini", C0_MINI),
        ("C0-Mini", C0_MINI),
        ("c0_mini", C0_MINI),
        ("c0-small", C0_MINI),
        ("C0-Small", C0_MINI),
        ("c0_small", C0_MINI),
        ("Causal-C0-Mini", C0_MINI),
        ("c0-mini-w16-kd-v1", C0_MINI),
        ("HalfWidth-Distilled-Attention", C0_MINI),
        ("a0", A0_REFERENCE),
        ("A0", A0_REFERENCE),
        ("Annual-Causal-A0", A0_REFERENCE),
        ("WalkForward-Reference-Ensemble", A0_REFERENCE),
        ("exp-039", EXP039_THESIS),
        ("EXP-039", EXP039_THESIS),
        ("exp039", EXP039_THESIS),
        ("ElasticNet-TeamForm-Baseline", EXP039_THESIS),
    ],
)
def test_resolve_model_spec_short_codes_and_aliases(identifier: str, expected_spec):
    """Verify resolve_model_spec resolves models case-insensitively across aliases."""
    assert resolve_model_spec(identifier) is expected_spec
    assert get_model(identifier) is expected_spec


def test_resolve_model_spec_unknown_raises_key_error():
    """Verify resolving an invalid identifier raises KeyError."""
    with pytest.raises(KeyError):
        resolve_model_spec("unknown-phantom-model")
    assert get_model("unknown-phantom-model") is None


def test_manifest_structure_and_metrics():
    """Verify data/06_models/c0_distilled/manifest.json exists and satisfies schema."""
    manifest_path = PROJECT_ROOT / "data" / "06_models" / "c0_distilled" / "manifest.json"
    assert manifest_path.is_file()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema"] == "c0_distilled_manifest_v1"

    # Baseline model
    base = manifest["baseline_model"]
    assert base["short_code"] == "C0"
    assert base["neural_parameters"] == 19818
    assert base["evaluation_metrics"]["unclipped_log_loss"] == pytest.approx(0.535449, abs=1e-5)

    # Distilled models
    models = manifest["distilled_models"]
    assert set(models.keys()) == {"C0-D", "C0-Slim", "C0-Mini"}

    # C0-D
    c0_d = models["C0-D"]
    assert c0_d["name"] == "Causal-C0-D"
    assert c0_d["version"] == "c0-d-born-again-w32-v1"
    assert c0_d["neural_parameters"] == 19914
    assert c0_d["neural_parameters_total"] == 19914
    assert c0_d["neural_parameters_per_seed"] == 6638
    assert c0_d["metrics"]["log_loss"] == pytest.approx(0.534134, abs=1e-5)
    assert c0_d["metrics"]["log_loss_pure"] == pytest.approx(0.534134, abs=1e-5)
    assert c0_d["metrics"]["log_loss_with_connector_ons"] == pytest.approx(0.533984, abs=1e-5)
    assert c0_d["metrics"]["log_loss_connector_ons"] == pytest.approx(0.533984, abs=1e-5)
    assert c0_d["metrics"]["log_loss_17token_core"] == pytest.approx(0.534418, abs=1e-5)
    assert c0_d["metrics"]["brier_score"] == pytest.approx(0.180468, abs=1e-5)
    assert c0_d["metrics"]["ece_10_bins"] == pytest.approx(0.018708, abs=1e-5)
    assert len(c0_d["checkpoints"]) == 3

    # C0-Slim
    c0_slim = models["C0-Slim"]
    assert c0_slim["name"] == "Causal-C0-Slim"
    assert c0_slim["version"] == "c0-slim-1seed-w32-v1"
    assert c0_slim["neural_parameters"] == 6638
    assert c0_slim["neural_parameters_total"] == 6638
    assert c0_slim["neural_parameters_per_seed"] == 6638
    assert c0_slim["metrics"]["log_loss"] == pytest.approx(0.534656, abs=1e-5)
    assert c0_slim["metrics"]["log_loss_pure"] == pytest.approx(0.534656, abs=1e-5)
    assert c0_slim["metrics"]["log_loss_17token"] == pytest.approx(0.535062, abs=1e-5)
    assert c0_slim["metrics"]["log_loss_connector"] == pytest.approx(0.534356, abs=1e-5)
    assert c0_slim["metrics"]["brier_score"] == pytest.approx(0.180667, abs=1e-5)
    assert c0_slim["metrics"]["ece_10_bins"] == pytest.approx(0.019236, abs=1e-5)
    assert c0_slim["runtime"]["inference_latency_ms"] == pytest.approx(277.7, abs=1.0)
    assert len(c0_slim["checkpoints"]) == 1

    # C0-Mini
    c0_mini = models["C0-Mini"]
    assert c0_mini["name"] == "Causal-C0-Mini"
    assert c0_mini["version"] == "c0-mini-w16-kd-v1"
    assert c0_mini["neural_parameters_total"] == 8394
    assert c0_mini["neural_parameters_per_seed"] == 2798
    assert c0_mini["metrics"]["log_loss"] == pytest.approx(0.535845, abs=1e-5)
    assert len(c0_mini["checkpoints"]) == 3

    # Provenance files
    prov = manifest["provenance_files"]
    assert "year_anchor_distilled_parquet" in prov
    assert "year_anchor_undistilled_parquet" in prov
    for key, item in prov.items():
        rel_path = item["path"]
        expected_sha = item["sha256"]
        full_path = PROJECT_ROOT / rel_path
        assert full_path.is_file(), f"Missing provenance file: {full_path}"
        computed_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
        assert computed_sha == expected_sha, f"SHA mismatch for {rel_path}"


def test_predict_matchup_cli_argument_parsing():
    """Verify CLI parser accepts --model options for C0 distilled family."""
    parser = argparse.ArgumentParser()
    parser.add_argument("team1")
    parser.add_argument("team2")
    parser.add_argument("--model", type=str, default="C0")
    parser.add_argument("--bo", type=int, choices=[1, 3, 5], default=3)

    args_c0_d = parser.parse_args(["T1", "GEN", "--model", "C0-D"])
    assert resolve_model_spec(args_c0_d.model) is C0_DISTILLED

    args_c0_slim = parser.parse_args(["C9", "KC", "--model", "C0-Slim"])
    assert resolve_model_spec(args_c0_slim.model) is C0_SLIM

    args_c0_d1 = parser.parse_args(["C9", "KC", "--model", "C0-D1"])
    assert resolve_model_spec(args_c0_d1.model) is C0_SLIM

    args_c0_mini = parser.parse_args(["FNC", "G2", "--model", "C0-Mini"])
    assert resolve_model_spec(args_c0_mini.model) is C0_MINI

    args_c0_small = parser.parse_args(["FNC", "G2", "--model", "C0-Small"])
    assert resolve_model_spec(args_c0_small.model) is C0_MINI

    args_default = parser.parse_args(["T1", "GEN"])
    assert resolve_model_spec(args_default.model) is C0_NATIVE


def test_apply_model_calibration_symmetry_and_identity():
    """Verify apply_model_calibration preserves probability invariants and side symmetry."""
    # Identity for C0 native
    p_a, p_b = apply_model_calibration(0.72, C0_NATIVE, best_of=3)
    assert p_a == pytest.approx(0.72)
    assert p_b == pytest.approx(0.28)

    # Distilled models maintain side-swap symmetry: f(1 - p) == 1 - f(p)
    for model in [C0_DISTILLED, C0_SLIM, C0_MINI]:
        for bo in [1, 3, 5]:
            p_a, p_b = apply_model_calibration(0.65, model, best_of=bo)
            assert p_a + p_b == pytest.approx(1.0)

            # Swap sides
            p_rev_a, p_rev_b = apply_model_calibration(0.35, model, best_of=bo)
            assert p_rev_a + p_rev_b == pytest.approx(1.0)
            assert p_a == pytest.approx(p_rev_b, abs=1e-6)
            assert p_b == pytest.approx(p_rev_a, abs=1e-6)
