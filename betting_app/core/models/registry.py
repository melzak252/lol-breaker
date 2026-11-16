"""Central model registry providing the Single Source of Truth for models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from betting_app.core.models.contract import HybridSpec, ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -----------------------------------------------------------------------------
# Known Model Specifications
# -----------------------------------------------------------------------------


_UNSERVED_MANIFEST_REASON = "no serving path builds 18/20-token tensors; artifact manifest not tracked"
_UNSERVED_REPORTING_JSON_REASON = (
    "no serving path builds 18/20-token tensors; artifact manifest not tracked "
    "(artifact_path is a reporting JSON, not a model)"
)
_UNSERVED_STUDENT_REASON = "no serving path loads distilled C0 student artifacts; artifact manifest not tracked"
# What the research CLI (scripts/predict_matchup.py) actually emits for non-C0 short codes:
# legacy 17-token C0 probability, then logit(p) * calibration_slopes[best_of].
_BO_SLOPE_SURROGATE = "legacy_c0_probability_with_bo_slope_surrogate"


C0_NATIVE = ModelSpec(
    name="Causal-C0",
    display_name="PlayerHistory-Attention-Ensemble",
    short_code="C0",
    version="c0-native-fresh20-2026-10-21-v1-regional-v1",
    feature_version="native-c0-fresh20-history19-regional-v1",
    artifact_path=PROJECT_ROOT / "data" / "06_models" / "c0_distilled_verification" / "c0d_repro_20261021_01" / "fresh" / "20t",
    description="Genuine three-seed fresh 20-token C0 inference with causal regional input correction. Nineteen observed history slots plus the learned model token.",
    metadata={
        "operational": True,
        "symmetric": True,
        "weights_model_version": "c0-native-fresh20-2026-10-21-v1",
        "weights_feature_version": "native-c0-fresh20-history19-v1",
        "regional_adjustment_version": "regional-input-correction-v1",
        "source_contract_version": "native-c0-serving-source-v3",
        "artifact_release": "data/06_models/c0_distilled_verification/c0d_repro_20261021_01/fresh/20t",
        "serving_protocol": "data/05_model_input/c0_distilled_verification/c0d_repro_20261021_01/protocol.json",
        "token_count": 20,
        "observed_history_slots": 19,
        "learned_model_tokens": 1,
    },
)

C0_DISTILLED = ModelSpec(
    name="Causal-C0-D",
    display_name="BornAgain-Distilled-Attention-Ensemble",
    short_code="C0-D",
    version="c0-d-born-again-w32-v1",
    feature_version="native-c0-history16-year1-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="distilled_c0",
    artifact_path=None,
    description="RESEARCH ONLY, not servable: 3-seed 18-Token (16 Recent Maps + 365d Year Anchor + Empty) Born-Again Distilled W32 Ensemble. No serving path builds 18-token tensors; the research CLI applies a BO logit-slope surrogate to legacy 17-token C0 probabilities, not distilled inference.",
    metadata={
        "operational": False,
        "unavailable_reason": _UNSERVED_MANIFEST_REASON,
        "served_as": _BO_SLOPE_SURROGATE,
        "calibration_slopes_source": "copied from the 18-token manifest (untracked); the research CLI applies them to legacy 17-token C0 probabilities",
        "symmetric": True,
        "distilled": True,
        "short_name": "C0-D",
        "aliases": [
            "C0-D",
            "C0-D-Core",
            "C0-D-Year",
            "C0-D-Year-Core",
            "Causal-C0-D",
            "c0-d-born-again-w32-v1",
            "c0_d",
            "c0_d_core",
            "BornAgain-Distilled-Attention-Ensemble",
        ],
        "neural_params": 19914,
        "neural_params_total": 19914,
        "neural_params_per_seed": 6638,
        "log_loss": 0.534134,
        "log_loss_pure": 0.534134,
        "log_loss_with_connector_ons": 0.533984,
        "log_loss_connector_ons": 0.533984,
        "log_loss_17token_core": 0.534418,
        "brier": 0.180468,
        "brier_score": 0.180468,
        "ece": 0.018708,
        "ci_95": [-0.00242, -0.00045],
        "calibration_slopes": {1: 1.004, 3: 1.008, 5: 1.012},
        "artifact_manifest": "data/06_models/c0_distilled/manifest.json",
        "financials": {
            "bets": 43,
            "win_rate": 0.605,
            "profit_pln": 1765.78,
            "yield_pct": 27.82,
            "roi_pct": 17.66,
            "sharpe": 1.56,
            "polish_longshots": {
                "odds_range": "2.30-4.00",
                "bets": 26,
                "win_rate": 0.577,
                "profit_pln": 1793.15,
                "yield_pct": 47.41,
                "sharpe": 1.79,
            },
        },
    },
)

C0_DISTILLED_20T = ModelSpec(
    name="Causal-C0-D-20T",
    display_name="MultiScale-20Token-Distilled-Ensemble",
    short_code="C0-D-20T",
    version="c0-d-20token-w32-v1",
    feature_version="native-c0-history16-ms3-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="distilled_c0",
    artifact_path=None,
    description="RESEARCH ONLY, not servable: 3-seed 20-Token Multi-Scale Born-Again Distilled W32 Ensemble. The reporting JSON is not an inference artifact; the research CLI applies a BO logit-slope surrogate to legacy 17-token C0 probabilities, not distilled inference.",
    metadata={
        "operational": False,
        "unavailable_reason": _UNSERVED_REPORTING_JSON_REASON,
        "served_as": _BO_SLOPE_SURROGATE,
        "calibration_slopes_source": "copied from C0-D; not present in the 20-token reporting JSON",
        "symmetric": True,
        "distilled": True,
        "short_name": "C0-D-20T",
        "aliases": ["C0-D-20T", "c0_d_20t", "c0-20t", "C0-20T", "MultiScale-20Token-Distilled-Ensemble"],
        "neural_params": 20106,
        "neural_params_total": 20106,
        "neural_params_per_seed": 6702,
        "log_loss": 0.533954,
        "log_loss_norm": 0.533600,
        "brier": 0.180388,
        "ece": 0.020309,
        "calibration_slopes": {1: 1.004, 3: 1.008, 5: 1.012},
        "reporting_json": "data/08_reporting/c0_upgrade_and_distillation_20261017/multiscale_20token_distilled.json",
    },
)

C0_SLIM = ModelSpec(
    name="Causal-C0-Slim",
    display_name="SingleSeed-Distilled-Attention",
    short_code="C0-Slim",
    version="c0-slim-1seed-w32-v1",
    feature_version="native-c0-history16-year1-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    artifact_path=None,
    description="RESEARCH ONLY, not servable: 1-seed 18-Token Distilled W32 research result. The research CLI applies a BO logit-slope surrogate to legacy 17-token C0 probabilities, not distilled inference.",
    metadata={
        "operational": False,
        "unavailable_reason": _UNSERVED_STUDENT_REASON,
        "served_as": _BO_SLOPE_SURROGATE,
        "symmetric": True,
        "distilled": True,
        "short_name": "C0-Slim",
        "aliases": [
            "C0-Slim",
            "C0-D1",
            "C0-Slim-Year",
            "C0-D1-Year",
            "C0-Slim+",
            "Causal-C0-Slim",
            "c0-slim-1seed-w32-v1",
            "c0_slim",
            "c0_d1",
            "SingleSeed-Distilled-Attention",
        ],
        "neural_params": 6638,
        "neural_params_total": 6638,
        "neural_params_per_seed": 6638,
        "log_loss": 0.534656,
        "log_loss_pure": 0.534656,
        "log_loss_17token": 0.535062,
        "log_loss_connector": 0.534356,
        "brier": 0.180667,
        "brier_score": 0.180667,
        "ece": 0.019236,
        "latency_ms": 277.7,
        "speedup": "3.1x",
        "calibration_slopes": {1: 0.992, 3: 1.002, 5: 1.018},
        "calibration_slopes_source": "research-only slope values recorded in registry; source artifact is untracked and values are not verified for serving",
        "artifact_manifest": "data/06_models/c0_distilled/manifest.json",
    },
)

C0_MINI = ModelSpec(
    name="Causal-C0-Mini",
    display_name="HalfWidth-Distilled-Attention",
    short_code="C0-Mini",
    version="c0-mini-w16-kd-v1",
    feature_version="native-c0-history16-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="distilled_c0",
    artifact_path=None,
    description="RESEARCH ONLY, not servable: 3-seed Width-16/Epoch-12 Distilled Student A research result. The research CLI applies a BO logit-slope surrogate to legacy 17-token C0 probabilities, not distilled inference.",
    metadata={
        "operational": False,
        "unavailable_reason": _UNSERVED_STUDENT_REASON,
        "served_as": _BO_SLOPE_SURROGATE,
        "symmetric": True,
        "distilled": True,
        "short_name": "C0-Mini",
        "aliases": ["C0-Small", "C0-Mini", "Causal-C0-Mini", "c0-mini-w16-kd-v1", "c0_mini", "c0_small", "HalfWidth-Distilled-Attention"],
        "neural_params_total": 8394,
        "neural_params_per_seed": 2798,
        "log_loss": 0.535845,
        "parameter_reduction": "57.65%",
        "calibration_slopes": {1: 1.002302, 3: 1.004804, 5: 1.002428},
        "calibration_slopes_source": "research-only slope values recorded in registry; source artifact is untracked and values are not verified for serving",
        "artifact_manifest": "data/06_models/c0_distilled/manifest.json",
    },
)

C0_YEAR = ModelSpec(
    name="Causal-C0-Year",
    display_name="YearAnchor-PlayerHistory-Attention-Ensemble",
    short_code="C0-Year",
    version="c0-year-18token-w32-e12-v1",
    feature_version="native-c0-history16-year1-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="c0_year_anchor",
    artifact_path=None,
    description="RESEARCH ONLY, not servable: 3-seed 18-Token Width-32/Epoch-12 Attention Ensemble. The reporting JSON is not an inference artifact; the research CLI applies a BO logit-slope surrogate to legacy 17-token C0 probabilities, not distilled inference.",
    metadata={
        "operational": False,
        "unavailable_reason": _UNSERVED_REPORTING_JSON_REASON,
        "served_as": _BO_SLOPE_SURROGATE,
        "symmetric": True,
        "short_name": "C0-Year",
        "aliases": ["C0-Year", "Causal-C0-Year", "c0_year", "c0-18t"],
        "neural_params_total": 19914,
        "neural_params_per_seed": 6638,
        "log_loss": 0.534777,
        "calibration_slopes": {1: 1.001, 3: 1.002, 5: 1.003},
        "calibration_slopes_source": "no source: these registry slope values are unverified and are not present in year_anchor_undistilled.json",
        "reporting_json": "data/08_reporting/c0_upgrade_and_distillation_20261017/year_anchor_undistilled.json",
    },
)

# Aliases for 18-token multi-horizon consolidated models
C0_D_YEAR = C0_DISTILLED
C0_SLIM_YEAR = C0_SLIM

A0_REFERENCE = ModelSpec(
    name="Annual-Causal-A0",
    display_name="WalkForward-Reference-Ensemble",
    short_code="A0",
    version="annual-causal-a0-v1",
    feature_version="causal-a0-research-bank-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="causal_a0",
    description="Annual walk-forward causal ensemble reference baseline (7 base checkpoints across 3 experts; quarterly cross-fitted calibration).",
    metadata={
        "operational": False,
        "unavailable_reason": "Annual-Causal-A0 is an offline walk-forward benchmark reference and is not servable in the live app.",
        "symmetric": True,
        "benchmark_reference": True,
        "short_name": "A0",
        "aliases": ["A0", "Annual-Causal-A0", "p_full", "WalkForward-Reference-Ensemble"],
    },
)

EXP078_LINEAR = ModelSpec(
    name="Operational-PlayerTeamRatings-W20",
    display_name="Linear-Ratings-TeamForm-Baseline",
    short_code="EXP-078",
    version="exp078-symmetric-series-v1",
    feature_version="ratings-w20-symmetric-series-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="linear_symmetric",
    artifact_path=PROJECT_ROOT / "betting_app" / "models" / "exp078_symmetric_series_v1.json",
    description="Symmetric series-level linear model on player/team ratings and W20 form.",
    metadata={"symmetric": True},
)

EXP039_THESIS = ModelSpec(
    name="Sym-Cal LR-ElasticNet-W20-Binomial",
    display_name="ElasticNet-TeamForm-Baseline",
    short_code="EXP-039",
    version="exp-039",
    feature_version="thesis-exp039",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="map",
    has_uncertainty=False,
    family="elastic_net",
    artifact_path=PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib",
    description="Map-level symmetric calibrated ElasticNet with 20-match team form (W20) and binomial series projection.",
    metadata={"frozen": True, "academic_baseline": True},
)

A1_CONSOLIDATED = ModelSpec(
    name="Consolidated-A1",
    display_name="Residual-Tier-Calibration-Challenger",
    short_code="A1",
    version="a1-consolidated-v1",
    feature_version="causal-a0-research-bank-residual-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="a1_engine",
    description="Research-only residual over WalkForward-Reference-Ensemble (Causal A0); no validated live A0 feature adapter or macro artifact is configured.",
    metadata={
        "operational": False,
        "unavailable_reason": "Consolidated A1 requires a validated Causal A0 adapter and out-of-sample macro artifact; rating consensus is not Causal A0.",
        "tier_scale": 0.94,
        "symmetric": True,
    },
)

ACTIVE_MODEL_NAME = C0_NATIVE.name
ACTIVE_MODEL_VERSION = C0_NATIVE.version
ACTIVE_MODEL_FAMILY = C0_NATIVE.family

BAYESIAN_SHRUNK_HYBRID = ModelSpec(
    name="Hybrid-Operational-Market",
    display_name="MarketShrunk-Hybrid-50",
    short_code="Hybrid-50",
    version=f"{C0_NATIVE.version}-a0.50-t1.00",
    feature_version=C0_NATIVE.feature_version,
    ratings_version=C0_NATIVE.ratings_version,
    w20_version=C0_NATIVE.w20_version,
    prediction_target="series",
    has_uncertainty=False,
    family="bayesian_market_hybrid",
    description="50/50 logit-space shrinkage blending PlayerHistory-Attention-Ensemble with 2-way Shin de-vigged market consensus.",
    metadata={
        "base_model_name": C0_NATIVE.name,
        "alpha": 0.50,
        "temperature": 1.0,
        "blending_mode": "logit_shrinkage",
        "symmetric": True,
    },
)

REGISTERED_MODELS: dict[str, ModelSpec] = {
    C0_NATIVE.name: C0_NATIVE,
    C0_NATIVE.display_name: C0_NATIVE,
    C0_NATIVE.short_code: C0_NATIVE,
    C0_DISTILLED.name: C0_DISTILLED,
    C0_DISTILLED.display_name: C0_DISTILLED,
    C0_DISTILLED.short_code: C0_DISTILLED,
    C0_SLIM.name: C0_SLIM,
    C0_SLIM.display_name: C0_SLIM,
    C0_SLIM.short_code: C0_SLIM,
    C0_MINI.name: C0_MINI,
    C0_MINI.display_name: C0_MINI,
    C0_MINI.short_code: C0_MINI,
    C0_YEAR.name: C0_YEAR,
    C0_YEAR.display_name: C0_YEAR,
    C0_YEAR.short_code: C0_YEAR,
    "Causal-C0-D-Year": C0_DISTILLED,
    "BornAgain-YearAnchor-Distilled-Ensemble": C0_DISTILLED,
    "C0-D-Year": C0_DISTILLED,
    "C0-D-Year-Core": C0_DISTILLED,
    "C0-D-Core": C0_DISTILLED,
    "Causal-C0-Slim-Year": C0_SLIM,
    "SingleSeed-YearAnchor-Distilled-Attention": C0_SLIM,
    "C0-Slim-Year": C0_SLIM,
    "C0-D1-Year": C0_SLIM,
    "C0-D1": C0_SLIM,
    "C0-Slim+": C0_SLIM,
    BAYESIAN_SHRUNK_HYBRID.name: BAYESIAN_SHRUNK_HYBRID,
    BAYESIAN_SHRUNK_HYBRID.display_name: BAYESIAN_SHRUNK_HYBRID,
    BAYESIAN_SHRUNK_HYBRID.short_code: BAYESIAN_SHRUNK_HYBRID,
    A1_CONSOLIDATED.name: A1_CONSOLIDATED,
    A1_CONSOLIDATED.display_name: A1_CONSOLIDATED,
    A1_CONSOLIDATED.short_code: A1_CONSOLIDATED,
    A0_REFERENCE.name: A0_REFERENCE,
    A0_REFERENCE.display_name: A0_REFERENCE,
    A0_REFERENCE.short_code: A0_REFERENCE,
    EXP078_LINEAR.name: EXP078_LINEAR,
    EXP078_LINEAR.display_name: EXP078_LINEAR,
    EXP078_LINEAR.short_code: EXP078_LINEAR,
    EXP039_THESIS.name: EXP039_THESIS,
    EXP039_THESIS.display_name: EXP039_THESIS,
    EXP039_THESIS.short_code: EXP039_THESIS,
    "c0": C0_NATIVE,
    "native_c0": C0_NATIVE,
    "c0-d": C0_DISTILLED,
    "c0_d": C0_DISTILLED,
    "c0-slim": C0_SLIM,
    "c0_slim": C0_SLIM,
    "c0-d1": C0_SLIM,
    "c0_d1": C0_SLIM,
    "c0-mini": C0_MINI,
    "c0_mini": C0_MINI,
    "c0-small": C0_MINI,
    "c0_small": C0_MINI,
    "c0-year": C0_YEAR,
    "c0_year": C0_YEAR,
    "c0-d-core": C0_DISTILLED,
    "c0_d_core": C0_DISTILLED,
    "c0-d-year": C0_DISTILLED,
    "c0_d_year": C0_DISTILLED,
    "c0-d-year-core": C0_DISTILLED,
    "c0_d_year_core": C0_DISTILLED,
    "c0-dy": C0_DISTILLED,
    "c0-slim-year": C0_SLIM,
    "c0_slim_year": C0_SLIM,
    "c0-d1-year": C0_SLIM,
    "c0_d1_year": C0_SLIM,
    "c0-slim+": C0_SLIM,
    "c0_slim+": C0_SLIM,
    "c0-sy": C0_SLIM,
    "a0": A0_REFERENCE,
    "shrunk_hybrid": BAYESIAN_SHRUNK_HYBRID,
    "bayesian_hybrid": BAYESIAN_SHRUNK_HYBRID,
    "hybrid_50": BAYESIAN_SHRUNK_HYBRID,
    "a1": A1_CONSOLIDATED,
    "a1_consolidated": A1_CONSOLIDATED,
    "exp078": EXP078_LINEAR,
    "exp039": EXP039_THESIS,
    "c0_d_20t": C0_DISTILLED_20T,
    "C0-D-20T": C0_DISTILLED_20T,
    "c0-d-20t": C0_DISTILLED_20T,
}

MODEL_REGISTRY: dict[str, ModelSpec] = REGISTERED_MODELS

# -----------------------------------------------------------------------------
# Active Model and Hybrid State
# -----------------------------------------------------------------------------
_ACTIVE_MODEL: ModelSpec = C0_NATIVE

# The separately selectable market-shrinkage policy uses native C0 once.
_ACTIVE_HYBRID: HybridSpec = HybridSpec(
    base_model=C0_NATIVE,
    hybrid_model_name=BAYESIAN_SHRUNK_HYBRID.name,
    alpha=0.50,
    temperature=1.0,
    blending_mode="logit_shrinkage",
    custom_version=BAYESIAN_SHRUNK_HYBRID.version,
)
# Thesis hybrid reference
THESIS_HYBRID: HybridSpec = HybridSpec(
    base_model=EXP039_THESIS,
    hybrid_model_name="Hybrid-Thesis-Market",
    alpha=0.35,
    temperature=0.80,
    blending_mode="linear",
    custom_version="a0.35-t0.80",
)


def get_hybrid_spec(model: ModelSpec) -> HybridSpec:
    """Resolve the pure base and blending policy declared by one model version."""
    if model.family != "bayesian_market_hybrid":
        raise ValueError("Expected a hybrid ModelSpec.")
    base = get_model(model.metadata.get("base_model_name", ""))
    if base is None or base.family == "bayesian_market_hybrid":
        raise ValueError("A hybrid must declare a registered pure sports model.")
    if base.metadata.get("operational") is False:
        raise ValueError(base.metadata.get("unavailable_reason", f"Model {base.name} is not operational"))
    return HybridSpec(
        base_model=base,
        hybrid_model_name=model.name,
        alpha=float(model.metadata["alpha"]),
        temperature=float(model.metadata["temperature"]),
        blending_mode=model.metadata.get("blending_mode", "logit_shrinkage"),
        custom_version=model.version,
    )


def get_active_model() -> ModelSpec:
    """Return the configured model, rejecting unknown or unavailable overrides."""
    override = os.getenv("ACTIVE_PREDICTION_MODEL")
    spec = get_model(override) if override else _ACTIVE_MODEL
    if spec is None:
        raise ValueError(f"Unknown ACTIVE_PREDICTION_MODEL: {override}")
    if spec.metadata.get("operational") is False:
        raise ValueError(spec.metadata.get("unavailable_reason", f"Model {spec.name} is not operational"))
    return spec


def get_active_model_spec() -> ModelSpec:
    """Return the active operational model specification."""
    return get_active_model()


def set_active_model(model_or_name: ModelSpec | str) -> None:
    """Switch the active sports model or explicitly selected hybrid model."""
    global _ACTIVE_MODEL, _ACTIVE_HYBRID
    if isinstance(model_or_name, str):
        spec = get_model(model_or_name)
        if spec is None:
            raise KeyError(f"Unknown model: {model_or_name}. Registered: {list(REGISTERED_MODELS.keys())}")
    else:
        spec = model_or_name
    if spec.metadata.get("operational") is False:
        raise ValueError(spec.metadata.get("unavailable_reason", f"Model {spec.name} is not operational"))
    if spec.family == "bayesian_market_hybrid":
        hybrid = get_hybrid_spec(spec)
    else:
        hybrid = HybridSpec(
            base_model=spec,
            hybrid_model_name=_ACTIVE_HYBRID.hybrid_model_name,
            alpha=_ACTIVE_HYBRID.alpha,
            temperature=_ACTIVE_HYBRID.temperature,
            blending_mode=_ACTIVE_HYBRID.blending_mode,
            custom_version=_ACTIVE_HYBRID.custom_version if _ACTIVE_HYBRID.base_model == spec else None,
        )
    _ACTIVE_MODEL = spec
    _ACTIVE_HYBRID = hybrid


def get_active_hybrid() -> HybridSpec:
    """Return market shrinkage synchronized with the currently selected sports model."""
    current_active = get_active_model()
    if current_active.family == "bayesian_market_hybrid":
        return get_hybrid_spec(current_active)
    if _ACTIVE_HYBRID.base_model != current_active:
        return HybridSpec(
            base_model=current_active,
            hybrid_model_name=_ACTIVE_HYBRID.hybrid_model_name,
            alpha=_ACTIVE_HYBRID.alpha,
            temperature=_ACTIVE_HYBRID.temperature,
            blending_mode=_ACTIVE_HYBRID.blending_mode,
        )
    return _ACTIVE_HYBRID

def set_active_hybrid(hybrid_spec: HybridSpec) -> None:
    """Set the separate market-shrinkage policy for the active sports model."""
    global _ACTIVE_HYBRID
    if hybrid_spec.base_model.family == "bayesian_market_hybrid":
        raise ValueError("A hybrid must use a pure sports model, not another hybrid.")
    if hybrid_spec.base_model.metadata.get("operational") is False:
        raise ValueError(
            hybrid_spec.base_model.metadata.get(
                "unavailable_reason", f"Model {hybrid_spec.base_model.name} is not operational"
            )
        )
    active = get_active_model()
    if active.family == "bayesian_market_hybrid":
        if hybrid_spec != get_hybrid_spec(active):
            raise ValueError("Select a separately versioned ModelSpec before changing its hybrid policy.")
    elif hybrid_spec.base_model != active:
        raise ValueError("Hybrid base must match the active sports model.")
    if (
        hybrid_spec.base_model == C0_NATIVE
        and hybrid_spec.hybrid_model_name == BAYESIAN_SHRUNK_HYBRID.name
        and hybrid_spec.alpha == BAYESIAN_SHRUNK_HYBRID.metadata["alpha"]
        and hybrid_spec.temperature == BAYESIAN_SHRUNK_HYBRID.metadata["temperature"]
        and hybrid_spec.blending_mode == BAYESIAN_SHRUNK_HYBRID.metadata["blending_mode"]
        and hybrid_spec.custom_version == BAYESIAN_SHRUNK_HYBRID.version
    ):
        _ACTIVE_HYBRID = hybrid_spec
    else:
        _ACTIVE_HYBRID = HybridSpec(
            base_model=hybrid_spec.base_model,
            hybrid_model_name=hybrid_spec.hybrid_model_name,
            alpha=hybrid_spec.alpha,
            temperature=hybrid_spec.temperature,
            blending_mode=hybrid_spec.blending_mode,
        )

def get_thesis_model() -> ModelSpec:
    """Return the frozen thesis baseline model specification."""
    return EXP039_THESIS


def get_thesis_hybrid() -> HybridSpec:
    """Return the frozen thesis hybrid specification."""
    return THESIS_HYBRID


def get_model(name_or_alias: str) -> ModelSpec | None:
    """Retrieve a model by key, alias, display_name, short_code, name, or version (case/dash insensitive)."""
    if name_or_alias in REGISTERED_MODELS:
        return REGISTERED_MODELS[name_or_alias]
    normalized = name_or_alias.lower().replace("-", "").replace("_", "").strip()
    for key, spec in REGISTERED_MODELS.items():
        candidates = [
            key.lower().replace("-", "").replace("_", ""),
            spec.name.lower().replace("-", "").replace("_", ""),
            spec.effective_display_name.lower().replace("-", "").replace("_", ""),
            spec.effective_short_code.lower().replace("-", "").replace("_", ""),
            spec.version.lower().replace("-", "").replace("_", ""),
        ]
        for alias in spec.metadata.get("aliases", []):
            candidates.append(str(alias).lower().replace("-", "").replace("_", ""))
        if normalized in candidates:
            return spec
    return None


def resolve_model_spec(identifier: str) -> ModelSpec:
    """Resolve a model specification by name, short_code, display_name, or alias case-insensitively.

    Raises KeyError if the identifier cannot be resolved.
    """
    spec = get_model(identifier)
    if spec is not None:
        return spec
    raise KeyError(
        f"Unknown model identifier: '{identifier}'. "
        f"Available models: {sorted({m.effective_short_code for m in REGISTERED_MODELS.values()})}"
    )

def list_registered_models() -> list[dict[str, Any]]:
    """List all registered models with their metadata for API responses."""
    active = get_active_model()
    seen = set()
    result = []
    for model in (
        C0_NATIVE,
        C0_DISTILLED,
        C0_DISTILLED_20T,
        C0_SLIM,
        C0_MINI,
        C0_YEAR,
        C0_D_YEAR,
        C0_SLIM_YEAR,
        BAYESIAN_SHRUNK_HYBRID,
        EXP078_LINEAR,
        EXP039_THESIS,
    ):
        if model.name in seen:
            continue
        seen.add(model.name)
        result.append({
            "name": model.name,
            "display_name": model.effective_display_name,
            "short_code": model.effective_short_code,
            "version": model.version,
            "feature_version": model.feature_version,
            "ratings_version": model.ratings_version,
            "w20_version": model.w20_version,
            "prediction_target": model.prediction_target,
            "target": model.prediction_target,
            "has_uncertainty": model.has_uncertainty,
            "has_epistemic_uncertainty": model.has_uncertainty,
            "family": model.family,
            "is_active_operational": model.name == active.name and model.version == active.version,
            "description": model.description,
            "metadata": model.metadata,
        })
    return result

def get_active_prediction_db_params() -> dict[str, str]:
    """Return SQL bind params for filtering active operational & hybrid predictions."""
    hybrid = get_active_hybrid()
    base_sports = hybrid.base_model
    return {
        "sn": base_sports.name,
        "sv": base_sports.version,
        "hn": hybrid.hybrid_model_name,
        "hv": hybrid.hybrid_model_version,
        "operational_name": base_sports.name,
        "operational_version": base_sports.version,
        "hybrid_name": hybrid.hybrid_model_name,
        "hybrid_version": hybrid.hybrid_model_version,
    }


def apply_model_calibration(
    p_a: float, spec: Any, best_of: int = 3, *, allow_surrogate: bool = False
) -> tuple[float, float]:
    """Apply a model's BO slope; research-only models require explicit surrogate opt-in.

    This adjustment does not run distilled inference.
    """
    import math

    if spec.metadata.get("operational") is False and not allow_surrogate:
        raise ValueError("Research-only calibration requires explicit allow_surrogate=True")
    slopes = spec.metadata.get("calibration_slopes", {}) if hasattr(spec, "metadata") else {}
    slope = float(slopes.get(best_of, slopes.get(str(best_of), slopes.get(3, 1.0))))
    p_clamped = max(1e-6, min(1.0 - 1e-6, float(p_a)))
    if abs(slope - 1.0) < 1e-9:
        return p_clamped, 1.0 - p_clamped
    l_val = math.log(p_clamped / (1.0 - p_clamped))
    p_adj = 1.0 / (1.0 + math.exp(-slope * l_val))
    p_adj = max(1e-6, min(1.0 - 1e-6, p_adj))
    return p_adj, 1.0 - p_adj
