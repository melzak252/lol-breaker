"""Replacement-guard (subgroup non-inferiority) evaluation for probability models.

A candidate model may replace a reference model only if, on every guarded subgroup,
the paired monthly-block bootstrap upper bound of the per-row log-loss difference
``LL(candidate) - LL(reference)`` is at most ``margin`` (0.005 by default). The
upper bound is one-sided and Bonferroni-corrected for the number of guards
(``alpha / n_guards``).

Resampling and loss primitives are reused from :mod:`src.analysis.bootstrap`
(``monthly_block_bootstrap_delta``: calendar-month blocks) and
:mod:`src.analysis.metrics` (``binary_log_loss_vector``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.metrics import binary_log_loss_vector

DEFAULT_MARGIN = 0.005
DEFAULT_ALPHA = 0.05
DEFAULT_N_BOOT = 5000
DEFAULT_SEED = 20261021
DEFAULT_N_VARIANTS_SCORED = 20

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_EVALUABLE = "NOT_EVALUABLE"

# Calendar-month blocks need at least two blocks for a non-degenerate resample.
MIN_BLOCKS = 2

GUARD_SLICE_NAMES = ("BO1", "BO3", "BO5", "Major", "LPL", "Unseen_Players")


def assert_chronological_order(dates: Any, *, label: str = "dates") -> None:
    """Raise if ``dates`` is not monotonically non-decreasing.

    Causal rolling-window builders that iterate rows in storage order are only
    causal when storage order is chronological.
    """

    values = pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy(dtype="datetime64[ns]")
    if values.size < 2:
        return
    if np.isnat(values).any():
        raise ValueError(f"{label} contain missing dates; chronology cannot be verified")
    backwards = np.flatnonzero(np.diff(values.astype("int64")) < 0)
    if backwards.size:
        first = int(backwards[0])
        raise ValueError(
            f"{label} are not chronologically ordered: {backwards.size} backward steps, "
            f"first at row {first} -> {first + 1}"
        )


def standard_slice_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the six replacement-guard slices of the 2026 ASSESS frame.

    Definitions match ``evaluate_probability_stream`` in
    ``scripts/train_c0_year_anchor_distilled.py``.
    """

    best_of = frame["best_of"].to_numpy(dtype=int)
    tournament = frame["tournament_name"].to_numpy(dtype=str)
    tier = frame["competition_tier"].to_numpy(dtype=str)
    unknown = frame["unknown_players"].to_numpy(dtype=int)
    return {
        "BO1": best_of == 1,
        "BO3": best_of == 3,
        "BO5": best_of == 5,
        "Major": (tier == "major") & (tournament != "KeSPA Cup 2026"),
        "LPL": pd.Series(tournament).str.startswith("LPL ").to_numpy(),
        "Unseen_Players": unknown > 0,
    }


def _month_blocks(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period("M").astype(str)


def _paired_delta_bootstrap(
    y: np.ndarray,
    p_candidate: np.ndarray,
    p_reference: np.ndarray,
    dates: pd.Series,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap ``LL(candidate) - LL(reference)`` over calendar-month blocks."""

    delta = binary_log_loss_vector(y, p_candidate) - binary_log_loss_vector(y, p_reference)
    boot = pd.DataFrame({"date": pd.to_datetime(dates).to_numpy(), "delta": delta})
    observed, _, _, samples = monthly_block_bootstrap_delta(
        boot, "delta", date_column="date", n_bootstraps=n_boot, random_seed=seed
    )
    boot["month"] = _month_blocks(boot["date"]).to_numpy()
    month_means = boot.groupby("month")["delta"].mean()
    return {
        "n": int(len(boot)),
        "n_blocks": int(len(month_means)),
        "delta_match_weighted": float(observed),
        "delta_equal_block": float(month_means.mean()),
        "samples": samples,
    }


def _validate_inputs(
    y: Any, p_candidate: Any, p_reference: Any, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=int)
    cand = np.asarray(p_candidate, dtype=float)
    ref = np.asarray(p_reference, dtype=float)
    n = len(frame)
    if not (len(y_arr) == len(cand) == len(ref) == n):
        raise ValueError(
            f"length mismatch: y={len(y_arr)} candidate={len(cand)} reference={len(ref)} frame={n}"
        )
    if "date" not in frame.columns:
        raise ValueError("frame must contain a 'date' column")
    if not (np.isfinite(cand).all() and np.isfinite(ref).all()):
        raise ValueError("probabilities must be finite")
    if not np.isin(y_arr, (0, 1)).all():
        raise ValueError("y must be binary 0/1")
    return y_arr, cand, ref


def evaluate_replacement_guards(
    y: Any,
    p_candidate: Any,
    p_reference: Any,
    frame: pd.DataFrame,
    slice_masks: Mapping[str, Any],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    margin: float = DEFAULT_MARGIN,
    n_guards: int | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Evaluate subgroup non-inferiority guards of a candidate against a reference.

    For each slice the per-row log-loss delta (candidate - reference) is bootstrapped
    with calendar-month blocks (paired). A guard passes iff the one-sided upper bound
    at level ``1 - alpha / n_guards`` is ``<= margin``. ``ci_lower`` is the matching
    one-sided lower bound at level ``alpha / n_guards``. Slices with no rows, or with
    fewer than two calendar-month blocks, are ``NOT_EVALUABLE`` and never pass. The
    overall verdict passes only if all ``n_guards`` guards are evaluated and pass.

    Returns:
        ``{"guards": {name: {...}}, "overall_pass": bool, "overall_status": str, ...}``.
        Each guard has ``status``, ``passes``, ``n``, ``n_blocks``, ``delta_ll``
        (match-weighted), ``delta_ll_equal_block``, ``ci_lower``, ``ci_upper``.
    """

    if n_boot < 5000:
        raise ValueError(f"n_boot must be >= 5000, got {n_boot}")
    if not slice_masks:
        raise ValueError("at least one guard slice is required")
    n_guards_eff = len(slice_masks) if n_guards is None else int(n_guards)
    if n_guards_eff < len(slice_masks):
        raise ValueError(f"n_guards={n_guards_eff} is smaller than the {len(slice_masks)} slices given")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    y_arr, cand, ref = _validate_inputs(y, p_candidate, p_reference, frame)
    dates = frame["date"].reset_index(drop=True)
    per_guard_alpha = alpha / n_guards_eff

    guards: dict[str, dict[str, Any]] = {}
    for name, raw_mask in slice_masks.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (len(y_arr),):
            raise ValueError(f"slice '{name}' mask shape {mask.shape} != ({len(y_arr)},)")
        n_rows = int(mask.sum())
        entry: dict[str, Any] = {
            "n": n_rows,
            "n_blocks": 0,
            "delta_ll": None,
            "delta_ll_equal_block": None,
            "ci_lower": None,
            "ci_upper": None,
            "passes": False,
            "status": STATUS_NOT_EVALUABLE,
        }
        if n_rows > 0:
            result = _paired_delta_bootstrap(
                y_arr[mask],
                cand[mask],
                ref[mask],
                dates[mask].reset_index(drop=True),
                n_boot=n_boot,
                seed=seed,
            )
            entry["n_blocks"] = result["n_blocks"]
            if result["n_blocks"] >= MIN_BLOCKS:
                samples = result["samples"]
                upper = float(np.nanquantile(samples, 1.0 - per_guard_alpha))
                lower = float(np.nanquantile(samples, per_guard_alpha))
                passes = bool(upper <= margin)
                entry.update(
                    delta_ll=result["delta_match_weighted"],
                    delta_ll_equal_block=result["delta_equal_block"],
                    ci_lower=lower,
                    ci_upper=upper,
                    passes=passes,
                    status=STATUS_PASS if passes else STATUS_FAIL,
                )
            else:
                entry["delta_ll"] = result["delta_match_weighted"]
                entry["delta_ll_equal_block"] = result["delta_equal_block"]
        guards[name] = entry

    all_evaluable = all(g["status"] != STATUS_NOT_EVALUABLE for g in guards.values())
    overall_pass = bool(all_evaluable and all(g["passes"] for g in guards.values()))
    if not all_evaluable:
        overall_status = STATUS_NOT_EVALUABLE
    else:
        overall_status = STATUS_PASS if overall_pass else STATUS_FAIL
    return {
        "guards": guards,
        "overall_pass": overall_pass,
        "overall_status": overall_status,
        "margin": float(margin),
        "alpha": float(alpha),
        "n_guards": n_guards_eff,
        "per_guard_alpha": per_guard_alpha,
        "one_sided_confidence": 1.0 - per_guard_alpha,
        "two_sided_confidence": 1.0 - 2.0 * per_guard_alpha,
        "ci_type": "equal-tailed_two-sided_interval; upper bound also used for one-sided guard",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "blocking": "calendar_month",
        "delta_definition": "LL(candidate) - LL(reference); negative favours candidate",
    }


def bonferroni_delta_interval(
    y: Any,
    p_candidate: Any,
    p_reference: Any,
    frame: pd.DataFrame,
    *,
    n_comparisons: int,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Two-sided Bonferroni monthly-block interval for ``LL(candidate) - LL(reference)``.

    Quantiles are ``alpha / (2 * n_comparisons)`` and ``1 - alpha / (2 * n_comparisons)``
    (98.33% for 3 comparisons at ``alpha=0.05``). ``excludes_zero_favouring_candidate``
    is True only if the upper bound is below zero.
    """

    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y_arr, cand, ref = _validate_inputs(y, p_candidate, p_reference, frame)
    result = _paired_delta_bootstrap(
        y_arr, cand, ref, frame["date"].reset_index(drop=True), n_boot=n_boot, seed=seed
    )
    tail = alpha / (2.0 * n_comparisons)
    samples = result.pop("samples")
    lower = float(np.nanquantile(samples, tail))
    upper = float(np.nanquantile(samples, 1.0 - tail))
    return {
        **result,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": 1.0 - 2.0 * tail,
        "two_sided_confidence": 1.0 - 2.0 * tail,
        "one_sided_alpha": tail,
        "n_comparisons": int(n_comparisons),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "excludes_zero_favouring_candidate": bool(upper < 0.0),
    }
