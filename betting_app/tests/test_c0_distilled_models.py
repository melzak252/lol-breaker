"""Unit tests for the C0 Distilled Model Family (C0-D, C0-Slim, C0-Mini)."""

from __future__ import annotations

import argparse

import pytest

from betting_app.core.models import (
    A0_REFERENCE,
    ACTIVE_MODEL_NAME,
    C0_DISTILLED,
    C0_DISTILLED_20T,
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
    set_active_model,
)
from scripts.predict_matchup import apply_model_calibration



def test_nonoperational_references_have_unavailable_reasons_and_research_artifacts():
    for model in (C0_DISTILLED, C0_DISTILLED_20T, C0_SLIM, C0_MINI, resolve_model_spec("C0-Year"), A0_REFERENCE):
        assert model.metadata["operational"] is False
        assert model.metadata["unavailable_reason"]
        assert model.artifact_path is None
    for model in (C0_DISTILLED, C0_SLIM, C0_MINI, resolve_model_spec("C0-Year")):
        assert model.metadata["calibration_slopes_source"]
        assert "RESEARCH ONLY, not servable" in model.description
    assert A0_REFERENCE.metadata["unavailable_reason"]


def test_research_slope_calibration_requires_explicit_surrogate_opt_in():
    with pytest.raises(ValueError, match="allow_surrogate=True"):
        apply_model_calibration(0.65, C0_DISTILLED, best_of=3)
    p_a, p_b = apply_model_calibration(0.65, C0_DISTILLED, best_of=3, allow_surrogate=True)
    assert p_a + p_b == pytest.approx(1.0)


def test_unknown_model_error_lists_registered_models_dynamically():
    with pytest.raises(KeyError) as exc:
        resolve_model_spec("unknown-phantom-model")
    message = str(exc.value)
    for short_code in ("C0-D-20T", "C0-Year", "A1"):
        assert short_code in message




def test_a0_reference_is_rejected_without_missing_reason_key(monkeypatch):
    monkeypatch.setenv("ACTIVE_PREDICTION_MODEL", "A0")
    with pytest.raises(ValueError, match="offline walk-forward benchmark reference"):
        get_active_model()
    with pytest.raises(ValueError, match="offline walk-forward benchmark reference"):
        set_active_model(A0_REFERENCE)


@pytest.mark.parametrize("identifier", ["C0-D", "C0-D-20T", "C0-Slim", "C0-Mini", "C0-Year"])
def test_unservable_c0_variants_stay_resolvable_but_cannot_be_active(monkeypatch, identifier):
    spec = resolve_model_spec(identifier)
    assert spec.metadata["operational"] is False
    monkeypatch.setenv("ACTIVE_PREDICTION_MODEL", identifier)
    with pytest.raises(ValueError, match="no serving path"):
        get_active_model()
    with pytest.raises(ValueError, match="no serving path"):
        set_active_model(spec)


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
            p_a, p_b = apply_model_calibration(0.65, model, best_of=bo, allow_surrogate=True)
            assert p_a + p_b == pytest.approx(1.0)

            # Swap sides
            p_rev_a, p_rev_b = apply_model_calibration(0.35, model, best_of=bo, allow_surrogate=True)
            assert p_rev_a + p_rev_b == pytest.approx(1.0)
            assert p_a == pytest.approx(p_rev_b, abs=1e-6)
            assert p_b == pytest.approx(p_rev_a, abs=1e-6)
