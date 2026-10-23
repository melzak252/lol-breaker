"""Research-only scoring for the frozen fresh C0 distilled reproduction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.metrics import binary_log_loss_vector, calculate_ece
from src.analysis.replacement_guards import (
    evaluate_replacement_guards,
    standard_slice_masks,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = Path("data/08_reporting/c0_upgrade_and_distillation_20261017")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_paths(protocol: dict) -> dict[str, str]:
    """Return repository-relative files required to reproduce the references."""
    root_path = ROOT / "data/research_root.txt"
    research_root = (ROOT / root_path.read_text().strip()).resolve()
    config = ROOT / "conf/base/research_benchmark.json"
    benchmark = json.loads(config.read_text())
    return {
        "legacy_predictions": (LEGACY_ROOT / "track1_predictions.parquet").as_posix(),
        "annual_a0_predictions": str(
            (research_root / benchmark["input"]["path"]).resolve()
        ),
        "annual_a0_config": "conf/base/research_benchmark.json",
    }


def _strict_probabilities(values: Any, label: str, n: int) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
        raise ValueError(
            f"{label} must be a finite length-{n} vector strictly inside (0, 1); no clipping"
        )
    return p


def _metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    losses = binary_log_loss_vector(y, p, epsilon=0.0)
    return {
        "n": int(len(y)),
        "log_loss": float(losses.mean()),
        "brier": float(np.square(p - y).mean()),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "ece10": float(calculate_ece(y, p, n_bins=10)),
        "tail_log_loss_ge_2_5": int((losses >= 2.5).sum()),
        "tail_log_loss_ge_5": int((losses >= 5.0).sum()),
        "coverage": 1.0,
        "mean_probability": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def _paired_month_interval(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    dates: pd.Series,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    delta = binary_log_loss_vector(y, candidate, epsilon=0.0) - binary_log_loss_vector(
        y, reference, epsilon=0.0
    )
    months = pd.to_datetime(dates).dt.to_period("M").astype(str).to_numpy()
    month_values = np.unique(months)
    if len(month_values) != 9:
        raise ValueError(
            f"ASSESS must contain exactly 9 monthly blocks, got {len(month_values)}"
        )
    means = np.asarray([delta[months == month].mean() for month in month_values])
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(means), size=(draws, len(means)))
    sums = np.asarray([delta[months == month].sum() for month in month_values])
    counts = np.asarray([(months == month).sum() for month in month_values])
    weighted = sums[sampled_indices].sum(axis=1) / counts[sampled_indices].sum(axis=1)
    sampled = means[sampled_indices].mean(axis=1)
    return {
        "match_weighted_delta_log_loss": float(delta.mean()),
        "equal_month_delta_log_loss": float(means.mean()),
        "match_weighted_ci_97_5": [
            float(np.quantile(weighted, 0.0125)),
            float(np.quantile(weighted, 0.9875)),
        ],
        "equal_month_ci_97_5": [
            float(np.quantile(sampled, 0.0125)),
            float(np.quantile(sampled, 0.9875)),
        ],
        "bootstrap_draws": int(draws),
        "seed": int(seed),
        "month_count": int(len(means)),
        "delta_definition": "candidate minus reference; negative favours candidate",
    }


def _exact_join(
    frame: pd.DataFrame, other: pd.DataFrame, label: str, columns: list[str]
) -> pd.DataFrame:
    if "golgg_match_id" not in frame or "golgg_match_id" not in other:
        raise ValueError(f"{label} must contain golgg_match_id")
    if frame.golgg_match_id.isna().any() or other.golgg_match_id.isna().any():
        raise ValueError(f"{label} IDs may not be missing")
    if (
        frame.golgg_match_id.duplicated().any()
        or other.golgg_match_id.duplicated().any()
    ):
        raise ValueError(f"{label} IDs must be unique")
    if set(frame.golgg_match_id) != set(other.golgg_match_id):
        raise ValueError(f"{label} IDs do not exactly cover the canonical frame")
    aligned = other.set_index("golgg_match_id").loc[frame.golgg_match_id]
    for key in ("date", "y", "best_of"):
        if key in frame and key in aligned:
            left, right = frame[key].to_numpy(), aligned[key].to_numpy()
            if key == "date":
                left, right = pd.to_datetime(left).normalize(), pd.to_datetime(right).normalize()
            if not np.array_equal(left, right):
                raise ValueError(f"{label} canonical {key} orientation/format differs")
    return frame.merge(
        other[["golgg_match_id", *columns]],
        on="golgg_match_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )


def evaluate_fresh_predictions(
    frame: pd.DataFrame, predictions: pd.DataFrame, protocol: dict
) -> dict[str, Any]:
    """Evaluate two fresh forecasts against exact-ID aligned historical controls."""
    required = [
        "golgg_match_id",
        "date",
        "best_of",
        "y",
        "tournament_name",
        "competition_tier",
        "unknown_players",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"canonical ASSESS frame missing columns: {missing}")
    frame = frame.copy()
    predictions = predictions.copy()
    if frame.golgg_match_id.isna().any() or predictions.golgg_match_id.isna().any():
        raise ValueError("Forecast IDs may not be missing")
    frame["golgg_match_id"] = frame["golgg_match_id"].astype(str)
    predictions["golgg_match_id"] = predictions["golgg_match_id"].astype(str)
    if (
        len(frame) != 3103
        or frame.golgg_match_id.duplicated().any()
        or frame.golgg_match_id.isna().any()
    ):
        raise ValueError("canonical ASSESS frame must have 3103 unique nonmissing IDs")
    if not frame.y.isin([0, 1]).all():
        raise ValueError("canonical outcomes must be binary")
    pred_cols = ["p_18t", "p_20t"]
    joined = _exact_join(frame, predictions, "fresh predictions", pred_cols)
    n = len(joined)
    fresh = {col: _strict_probabilities(joined[col], col, n) for col in pred_cols}
    y = joined.y.to_numpy(dtype=int)
    dates = pd.to_datetime(joined.date)
    legacy_path = ROOT / reference_paths(protocol)["legacy_predictions"]
    legacy = pd.read_parquet(legacy_path)
    legacy["golgg_match_id"] = legacy["golgg_match_id"].astype(str)
    legacy_cols = ["p_c0__full", "w32_e16_kd_born_again__full"]
    legacy_joined = _exact_join(joined, legacy, "legacy predictions", legacy_cols)
    c0 = _strict_probabilities(legacy_joined[legacy_cols[0]], "legacy C0", n)
    kd17 = _strict_probabilities(
        legacy_joined[legacy_cols[1]], "17-token diagnostic", n
    )
    references: dict[str, np.ndarray] = {"C0": c0}
    annual_cfg = json.loads((ROOT / "conf/base/research_benchmark.json").read_text())
    annual_path = Path(reference_paths(protocol)["annual_a0_predictions"])
    expected_hash = annual_cfg["input"]["sha256"]
    observed_hash = _sha256(annual_path)
    if observed_hash != expected_hash:
        raise ValueError("pinned annual A0 joined-table input hash mismatch")
    annual = pd.read_parquet(annual_path)
    if "golgg_match_id" not in annual.columns or "p_full" not in annual.columns:
        raise ValueError("pinned annual A0 joined table lacks exact IDs or p_full")
    annual["golgg_match_id"] = annual["golgg_match_id"].astype(str)
    if annual["golgg_match_id"].duplicated().any():
        raise ValueError("Annual A0 contains duplicate canonical IDs")
    annual = annual.loc[annual.golgg_match_id.isin(frame.golgg_match_id)].copy()
    for key in ("y", "best_of"):
        aligned = frame[["golgg_match_id", key]].merge(
            annual[["golgg_match_id", key]],
            on="golgg_match_id",
            validate="one_to_one",
            suffixes=("_target", "_reference"),
        )
        if not np.array_equal(aligned[key + "_target"], aligned[key + "_reference"]):
            raise ValueError(f"Annual A0 target orientation/format mismatch: {key}")
    annual_joined = _exact_join(joined, annual, "annual A0", ["p_full"])
    references["annual_A0"] = _strict_probabilities(
        annual_joined.p_full, "annual A0", n
    )
    summary = {
        name: _metric_summary(y, p)
        for name, p in {
            **references,
            **{"18t": fresh["p_18t"], "20t": fresh["p_20t"], "17t_diagnostic": kd17},
        }.items()
    }
    boot = protocol.get("bootstrap", {})
    draws, seed = int(boot.get("draws", 5000)), int(boot.get("seed", 20261021))
    if draws != 5000:
        raise ValueError("protocol bootstrap must specify exactly 5000 draws")
    comparisons: dict[str, Any] = {}
    guards: dict[str, Any] = {}
    masks = standard_slice_masks(joined)
    for candidate_name, column in (("18t", "p_18t"), ("20t", "p_20t")):
        p = fresh[column]
        comparisons[candidate_name] = {
            "vs_C0": _paired_month_interval(y, p, references["C0"], dates, seed, draws),
            "vs_C0_brier_delta": float(
                np.square(p - y).mean() - np.square(references["C0"] - y).mean()
            ),
        }
        candidate_guards = {}
        for refname, refp in references.items():
            candidate_guards[refname] = evaluate_replacement_guards(
                y,
                p,
                refp,
                joined.assign(date=dates),
                masks,
                n_boot=draws,
                seed=seed,
                margin=0.005,
                n_guards=24,
                alpha=0.05,
            )
        guards[candidate_name] = candidate_guards
        primary = comparisons[candidate_name]["vs_C0"]
        all_pass = all(
            result["overall_status"] == "PASS" for result in candidate_guards.values()
        )
        comparisons[candidate_name]["decision"] = (
            "RESEARCH_ONLY_STRICT_PRIMARY_PASS"
            if primary["match_weighted_ci_97_5"][1] < 0
            and comparisons[candidate_name]["vs_C0_brier_delta"] <= 0
            and all_pass
            else "RESEARCH_ONLY_NOT_PROMOTED"
        )
    diagnostic = {
        name: _paired_month_interval(y, fresh["p_18t"], kd17, dates, seed, draws)
        for name in ("18t_vs_17t_diagnostic",)
    }
    diagnostic["20t_vs_17t_diagnostic"] = _paired_month_interval(
        y, fresh["p_20t"], kd17, dates, seed, draws
    )
    return {
        "schema": "c0_distilled_reproduction_evaluation_v1",
        "status": "RESEARCH_ONLY_HISTORICAL_EXPOSED",
        "n_assess": int(n),
        "metrics": summary,
        "primary_comparisons": comparisons,
        "replacement_guards": guards,
        "17t_diagnostic_contrasts": diagnostic,
        "multiplicity": {
            "primary": "Bonferroni across two candidates; two-sided 97.5% intervals",
            "guards": "one-sided family correction over 2 candidates x 2 references x 6 guards (24); margin 0.005",
            "historical_search": "Previously exposed historical variants remain nominal; no global-search correction claimed",
        },
        "decision_scope": "research-only; never a production promotion",
        "input_closure": {
            "references": reference_paths(protocol),
            "annual_a0_sha256": observed_hash,
            "annual_a0_expected_sha256": expected_hash,
        },
    }
