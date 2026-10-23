#!/usr/bin/env python3
"""Train and Evaluate the 18-Token Distilled Multi-Horizon History Attention Architecture.

Adds each player's strictly prior 365-day whole-year anchor token:
- [empty_token, year_anchor_365d_token, map_1, ..., map_16] = 18 tokens
- Normalization: Exact same TRAIN HistoryNormalizer as C0 (z-scored coordinate alignment)
- Distillation: 16 epochs on TRAIN (N=36,205) using Teacher B (Multi-Capacity Super-Teacher)
  with distillation_weight=0.6 (0.40 * Hard_Beta_NLL + 0.60 * Soft_Teacher_KL + 0.10 * Aux_SmoothL1)

Bank provenance: ``scripts/build_year_anchor_bank.py`` rebuilds this token from the
release-corrected bank resolved through ``data/research_root.txt``. Its 33-value
map layout follows the historical ``temporal_player_features.py`` implementation
in the canonical benchmark. The builder asserts chronological metadata order,
retains stable metadata row alignment and uses only released maps strictly before
the current day and within 365 days.
Rebuild the bank with ``python scripts/build_year_anchor_bank.py`` before training.

Outputs saved to:
- data/08_reporting/c0_upgrade_and_distillation_20261017/year_anchor_distilled.json
- data/08_reporting/c0_upgrade_and_distillation_20261017/year_anchor_distilled_predictions.parquet
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, logsumexp
from sklearn.metrics import brier_score_loss, log_loss
import torch
import torch.nn as nn

# Enforce 2 threads to prevent CPU thread contention
torch.set_num_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.historical_production_simulator import HistoricalProductionSimulator
from src.analysis.metrics import calculate_ece
from src.models import a0_capacity_experiment as cap_exp
from src.models import a0_capacity_runner as runner
from src.models import a0_data_focus as data_focus
from src.models import a0_native_refit as native_refit
from src.models import a0_uncertainty as uncertainty
from src.models.a0_capacity_experiment import _backend, soft_terminal_loss

SEEDS = (20260910, 20260911, 20260912)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261017
REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
MODELS_DIR = REPORT_DIR / "models"
BANK_DIR = REPORT_DIR / "year_anchor_bank"

OUTPUT_JSON_PATH = REPORT_DIR / "year_anchor_distilled.json"
OUTPUT_PARQUET_PATH = REPORT_DIR / "year_anchor_distilled_predictions.parquet"


def resolve_research_root() -> Path:
    configured = Path((PROJECT_ROOT / "data/research_root.txt").read_text().strip())
    candidates = [
        configured if configured.is_absolute() else PROJECT_ROOT / configured,
        Path("/home/melzak/.codex/.chatgpt-projects/g-p-6944b69bdee08191acfcfeb7a38f372a"),
    ]
    for candidate in candidates:
        if (candidate / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/metadata.parquet").is_file():
            return candidate
    raise FileNotFoundError("Could not find valid research root containing release_corrected_bank")


# ---------------------------------------------------------------------------
# 18-Token Architecture Definition
# ---------------------------------------------------------------------------

class YearAnchorCommonMapResidual(cap_exp.ParametricCommonMapResidual):
    """18-Token Multi-Horizon History Attention Architecture.

    Tokens:
    - Slot 0: self.empty (learnable empty token)
    - Slot 1: 365-day whole-year anchor token passed through history_encoder + self.year_token_bias
    - Slots 2..17: 16 recent maps passed through history_encoder
    Total: 18 tokens in masked attention layer.
    """

    def __init__(self, width: int = 32):
        super().__init__(width)
        # Learnable bias for the 365-day whole-year anchor token
        self.year_token_bias = nn.Parameter(torch.zeros(1, 1, width))
        expected_params = (2 * width**2 + 142 * width + 13) + width  # 6605 + 32 = 6637 (+ eta = 6638)
        actual_params = sum(p.numel() for p in self.parameters())
        if actual_params != expected_params:
            raise AssertionError(f"Expected {expected_params} parameters, got {actual_params}")

    def _representation(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.player(x["players"])
        n = h.shape[0]

        # x['mask'] has shape (B, 2, 5, 17) -> flatten to (B*10, 17)
        mask_17 = x["mask"].reshape(-1, 17).bool()
        val_17 = x["history"].reshape(-1, 17, 33)
        val_17 = torch.where(mask_17[..., None], val_17, torch.zeros_like(val_17))

        # Split year anchor (slot 0) and recent 16 maps (slots 1..16)
        year_val = val_17[:, 0:1, :]
        recent_val = val_17[:, 1:, :]
        year_mask = mask_17[:, 0:1]
        recent_mask = mask_17[:, 1:]

        year_encoded = self.history_encoder(year_val) + self.year_token_bias
        recent_encoded = self.history_encoder(recent_val)

        # Prepend empty token to form 18 tokens
        empty_token = self.empty.expand(len(val_17), -1, -1)
        encoded_18 = torch.cat([empty_token, year_encoded, recent_encoded], dim=1)

        empty_valid = torch.ones((len(mask_17), 1), dtype=torch.bool, device=mask_17.device)
        valid_18 = torch.cat([empty_valid, year_mask, recent_mask], dim=1)

        weight = self.attention(encoded_18).squeeze(-1).masked_fill(~valid_18, -torch.inf).softmax(1)
        out = (encoded_18 * weight[..., None]).sum(1)
        h = h + out.reshape(n, 2, 5, self.width)
        return torch.tanh(h)


class YearAnchorNonlinearHistoryModel(cap_exp.ParametricNonlinearHistoryModel):
    """Container model matching the ParametricNonlinearHistoryModel API."""

    def __init__(
        self,
        baseline: Any,
        normalizer: Any,
        width: int = 32,
        seed: int = 20260910,
        epochs: int = 16,
        coefficient: float = 0.1,
    ):
        super().__init__(baseline, normalizer, width=width, seed=seed, epochs=epochs, coefficient=coefficient)
        torch.manual_seed(seed)
        self.network = YearAnchorCommonMapResidual(width)
        self.width = width
        self.seed = seed
        self.epochs = epochs


# ---------------------------------------------------------------------------
# Training and Evaluation Helpers
# ---------------------------------------------------------------------------

def compute_unclipped_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p_safe = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))))


def compute_tail_count(y: np.ndarray, p: np.ndarray, threshold: float = 2.5, eps: float = 1e-15) -> int:
    p_safe = np.clip(p, eps, 1.0 - eps)
    row_ll = -(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))
    return int(np.sum(row_ll >= threshold))


def evaluate_probability_stream(
    y: np.ndarray,
    p: np.ndarray,
    dates: np.ndarray,
    frame: pd.DataFrame,
    ref_c0_p: np.ndarray,
    ref_d_core_p: np.ndarray,
    n_bootstraps: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    ll = compute_unclipped_log_loss(y, p)
    brier = float(np.mean((p - y) ** 2))
    ece = float(calculate_ece(y, p, n_bins=10))
    tails = compute_tail_count(y, p, threshold=2.5)

    bo = frame["best_of"].to_numpy(dtype=int)
    t_name = frame["tournament_name"].to_numpy(dtype=str)
    tier = frame["competition_tier"].to_numpy(dtype=str)
    unknown = frame["unknown_players"].to_numpy(dtype=int)

    slices = {
        "BO1": bo == 1,
        "BO3": bo == 3,
        "BO5": bo == 5,
        "Major": (tier == "major") & (t_name != "KeSPA Cup 2026"),
        "LPL": pd.Series(t_name).str.startswith("LPL ").to_numpy(),
        "Unseen_Players": unknown > 0,
    }

    slice_metrics = {}
    for s_name, mask in slices.items():
        if mask.sum() > 0:
            slice_metrics[s_name] = {
                "n": int(mask.sum()),
                "log_loss": compute_unclipped_log_loss(y[mask], p[mask]),
                "brier": float(np.mean((p[mask] - y[mask]) ** 2)),
                "tail_count": compute_tail_count(y[mask], p[mask], threshold=2.5),
            }

    # Paired Monthly-Block Bootstrap vs C0 and vs C0-D-Core
    p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
    c0_safe = np.clip(ref_c0_p, 1e-15, 1.0 - 1e-15)
    d_core_safe = np.clip(ref_d_core_p, 1e-15, 1.0 - 1e-15)

    row_ll = -(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))
    c0_ll = -(y * np.log(c0_safe) + (1.0 - y) * np.log(1.0 - c0_safe))
    d_core_ll = -(y * np.log(d_core_safe) + (1.0 - y) * np.log(1.0 - d_core_safe))

    delta_c0 = row_ll - c0_ll
    delta_d_core = row_ll - d_core_ll

    boot_df = pd.DataFrame({
        "date": dates,
        "delta_c0": delta_c0,
        "delta_d_core": delta_d_core,
    })

    obs_c0, low_c0, high_c0, samples_c0 = monthly_block_bootstrap_delta(
        boot_df, "delta_c0", date_column="date", n_bootstraps=n_bootstraps, random_seed=seed
    )
    p_nonneg_c0 = float(np.mean(samples_c0 >= 0.0))

    obs_d_core, low_d_core, high_d_core, samples_d_core = monthly_block_bootstrap_delta(
        boot_df, "delta_d_core", date_column="date", n_bootstraps=n_bootstraps, random_seed=seed
    )
    p_nonneg_d_core = float(np.mean(samples_d_core >= 0.0))

    return {
        "n": int(len(y)),
        "unclipped_log_loss": ll,
        "brier_score": brier,
        "ece_10_bins": ece,
        "tail_count_ll_ge_2_5": tails,
        "slices": slice_metrics,
        "bootstrap_vs_c0": {
            "observed_delta": obs_c0,
            "ci95_lower": low_c0,
            "ci95_upper": high_c0,
            "p_nonnegative": p_nonneg_c0,
            "draws": n_bootstraps,
            "seed": seed,
        },
        "bootstrap_vs_c0_d_core": {
            "observed_delta": obs_d_core,
            "ci95_lower": low_d_core,
            "ci95_upper": high_d_core,
            "p_nonnegative": p_nonneg_d_core,
            "draws": n_bootstraps,
            "seed": seed,
        },
    }


def fit_beta_calibration_and_assemble(
    models: list[Any],
    data_neural: dict[str, Any],
    data_orig: dict[str, Any],
    cal_ix: np.ndarray,
    assess_ix: np.ndarray,
    backend: dict[str, Any],
    shared: dict[str, Any],
    ref_calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Fits Beta calibration on CAL and assembles full C0 prediction on ASSESS and CAL."""
    from src.models.a0_native_refit import _calculate_forecast
    from src.models.a0_target_experiment import _predict_mixture

    frame = data_orig["frame"]
    bo = frame.best_of.to_numpy(dtype=np.int64)
    wins_a, wins_b = data_orig["wins_a"], data_orig["wins_b"]

    cal_z, cal_rho, cal_raw_probs = _calculate_forecast(models, data_neural, cal_ix, backend)
    assess_z, assess_rho, assess_raw_probs = _calculate_forecast(models, data_neural, assess_ix, backend)

    raw_cal, hist_cal = data_focus._subset_inputs(data_orig, cal_ix)
    raw_assess, hist_assess = data_focus._subset_inputs(data_orig, assess_ix)

    other_cal = data_focus._other_logits(shared, raw_cal, hist_cal, bo[cal_ix])
    other_assess = data_focus._other_logits(shared, raw_assess, hist_assess, bo[assess_ix])

    parameters = []
    records = []
    for mode in range(3):
        fitted = backend["beta_terminal"].fit_joint(cal_z[:, mode], wins_a[cal_ix], wins_b[cal_ix], "beta")
        parameters.append((fitted["scale"], fitted["rho"]))
        records.append(fitted)

    cal_params = {
        "beta_parameters": np.asarray(parameters, dtype=float),
        "slopes": ref_calibration["slopes"],
        "mixtures": ref_calibration["mixtures"],
    }

    assembled_cal, _, _ = _predict_mixture(
        backend, cal_z, other_cal, cal_params["slopes"],
        raw_cal["gates"], bo[cal_ix], cal_params
    )
    assembled_assess, _, _ = _predict_mixture(
        backend, assess_z, other_assess, cal_params["slopes"],
        raw_assess["gates"], bo[assess_ix], cal_params
    )

    full_assembled_p = assembled_assess[:, 0]
    standalone_neural_p = assess_raw_probs[:, 0, :].mean(axis=1)

    cal_full_p = assembled_cal[:, 0]
    cal_neural_p = cal_raw_probs[:, 0, :].mean(axis=1)

    cal_summary = {
        "beta_parameters": [{"scale": float(p[0]), "rho": float(p[1])} for p in parameters],
        "neural_rho_mean": float(assess_rho.mean()),
    }
    return full_assembled_p, standalone_neural_p, assess_z[:, 0, :], cal_summary, cal_full_p, cal_neural_p


def train_or_load_year_anchor_model(
    width: int,
    seed: int,
    epochs: int,
    distillation_weight: float,
    teacher_probs: np.ndarray,
    ckpt_dir: Path,
    data_17: dict[str, Any],
    train_ix: np.ndarray,
    bo_index_train: np.ndarray,
    shared: dict[str, Any],
) -> YearAnchorNonlinearHistoryModel:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_epoch{epochs}.pt"

    if ckpt_path.is_file():
        print(f"Loading existing checkpoint: {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inst = YearAnchorNonlinearHistoryModel(
            shared["baseline"], shared["normalizer"], width=width, seed=seed, epochs=epochs, coefficient=0.1
        )
        inst.network.load_state_dict(state["network_state_dict"])
        inst.eta = torch.nn.Parameter(state["eta"])
        inst.target_scaler = state["target_scaler"]
        inst.epochs = epochs
        inst.epoch_diagnostics = state.get("epoch_diagnostics", [])
        inst.network.eval()
        return inst

    print(f"Training YearAnchor model: width={width}, seed={seed}, epochs={epochs}, lambda={distillation_weight}...")
    t0 = time.monotonic()
    model_inst = YearAnchorNonlinearHistoryModel(
        shared["baseline"], shared["normalizer"], width=width, seed=seed, epochs=epochs, coefficient=0.1
    )
    snapshots = model_inst.fit_trajectory(
        data_17["raw"], data_17["history"], None, train_ix, bo_index_train,
        data_17["wins_a"], data_17["wins_b"], data_17["targets"], data_17["target_mask"],
        checkpoint_dir=ckpt_dir, checkpoint_epochs=(epochs,), device="cpu",
        teacher_probabilities=teacher_probs, distillation_weight=distillation_weight,
        progress=lambda ep, loss, aux, rec: (
            print(f"  [w{width} s{seed}] epoch {ep}/{epochs}: loss={loss:.5f} aux={aux:.5f} ({time.monotonic()-t0:.1f}s)")
            if ep in (1, epochs // 4, epochs // 2, 3 * epochs // 4, epochs) else None
        )
    )
    print(f"Finished training w{width} seed {seed} in {time.monotonic() - t0:.1f}s")
    return snapshots[epochs]


def run_ons_recalibration(
    dates: np.ndarray,
    result_days: np.ndarray,
    logits: np.ndarray,
    y_true: np.ndarray,
    gamma: float = 0.005,
    eps: float = 1.0,
    a_bounds: tuple[float, float] = (0.7, 1.5),
    b_bounds: tuple[float, float] = (-0.5, 0.5),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Strictly causal day-by-day Online Platt Scaling (OPS) via Online Newton Step (ONS)."""
    unique_dates = sorted(np.unique(dates))
    n_samples = len(logits)

    p_recal = np.zeros(n_samples, dtype=np.float64)
    theta = np.array([1.0, 0.0], dtype=np.float64)
    hessian_acc = eps * np.eye(2, dtype=np.float64)

    theta_history = []
    processed_matches: set[int] = set()

    for current_date in unique_dates:
        d_mask = (dates == current_date)
        d_indices = np.where(d_mask)[0]

        z_d = logits[d_indices]
        p_d = expit(theta[0] * z_d + theta[1])
        p_recal[d_indices] = p_d

        theta_history.append({"date": str(current_date), "a": float(theta[0]), "b": float(theta[1])})

        available_mask = (result_days <= current_date) & ~np.isin(np.arange(n_samples), list(processed_matches))
        new_indices = np.where(available_mask)[0]

        if len(new_indices) > 0:
            for idx in new_indices:
                z_i = logits[idx]
                y_i = y_true[idx]
                x_i = np.array([z_i, 1.0], dtype=np.float64)

                p_i = float(expit(np.dot(theta, x_i)))
                p_i = max(1e-12, min(1.0 - 1e-12, p_i))

                g_i = (p_i - y_i) * x_i
                h_i = p_i * (1.0 - p_i) * np.outer(x_i, x_i)
                hessian_acc += h_i

                a_inv = np.linalg.inv(hessian_acc)
                step = gamma * (a_inv @ g_i)
                theta = theta - step

                theta[0] = np.clip(theta[0], a_bounds[0], a_bounds[1])
                theta[1] = np.clip(theta[1], b_bounds[0], b_bounds[1])

                processed_matches.add(idx)

    return p_recal, theta_history


def compute_three_part_decomposition(
    bets: list[dict[str, Any]],
    initial_bankroll: float = 10000.0,
) -> dict[str, Any]:
    """Compute performance metrics and exact three-part econometric decomposition."""
    if not bets:
        return {
            "n_bets": 0,
            "n_wins": 0,
            "win_rate_pct": 0.0,
            "total_turnover_pln": 0.0,
            "total_net_profit_pln": 0.0,
            "realized_yield_pct": 0.0,
            "bankroll_roi_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "claimed_ev_pct": 0.0,
            "true_edge_net_clv_pct": 0.0,
            "optimizers_curse_bias_pp": 0.0,
            "match_luck_noise_pp": 0.0,
            "identity_discrepancy_max": 0.0,
        }

    df = pd.DataFrame(bets)
    n_bets = len(df)
    n_wins = int(df["won"].sum())
    win_rate = (n_wins / n_bets) * 100.0 if n_bets > 0 else 0.0
    total_turnover = float(df["stake"].sum())
    total_net_profit = float(df["profit"].sum())
    realized_yield = (total_net_profit / total_turnover) * 100.0 if total_turnover > 0 else 0.0
    bankroll_roi = (total_net_profit / initial_bankroll) * 100.0

    equities = df["bankroll"].to_numpy(dtype=float)
    if len(equities) > 0:
        eq_curve = np.concatenate([[initial_bankroll], equities])
        running_max = np.maximum.accumulate(eq_curve)
        drawdowns = (running_max - eq_curve) / running_max * 100.0
        max_drawdown = float(np.max(drawdowns))
    else:
        max_drawdown = 0.0

    gross_wins = float(df[df["profit"] > 0]["profit"].sum())
    gross_losses = float(abs(df[df["profit"] < 0]["profit"].sum()))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    returns = df["profit"] / df["stake"]
    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (float(returns.mean()) / ret_std) * np.sqrt(n_bets) if ret_std > 0 else 0.0

    weights = df["stake"].to_numpy(dtype=float) / total_turnover
    claimed_ev = float(np.sum(weights * df["ev_claimed"].to_numpy(dtype=float))) * 100.0
    net_clv = float(np.sum(weights * df["net_clv"].to_numpy(dtype=float))) * 100.0
    bias_b = claimed_ev - net_clv
    luck_eta = realized_yield - net_clv

    unit_yield = df["profit"].to_numpy(dtype=float) / df["stake"].to_numpy(dtype=float)
    unit_ev = df["ev_claimed"].to_numpy(dtype=float)
    unit_clv = df["net_clv"].to_numpy(dtype=float)
    unit_diff = np.abs(unit_yield - (unit_ev - (unit_ev - unit_clv) + (unit_yield - unit_clv)))
    max_discrepancy = float(np.max(unit_diff))

    return {
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate_pct": round(win_rate, 2),
        "total_turnover_pln": round(total_turnover, 2),
        "total_net_profit_pln": round(total_net_profit, 2),
        "realized_yield_pct": round(realized_yield, 2),
        "bankroll_roi_pct": round(bankroll_roi, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "claimed_ev_pct": round(claimed_ev, 2),
        "true_edge_net_clv_pct": round(net_clv, 2),
        "optimizers_curse_bias_pp": round(bias_b, 2),
        "match_luck_noise_pp": round(luck_eta, 2),
        "identity_discrepancy_max": max_discrepancy,
    }


# ---------------------------------------------------------------------------
# Main Training & Evaluation Orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    torch.set_num_threads(2)
    total_start = time.monotonic()
    print("=" * 80)
    print("18-TOKEN MULTI-HORIZON HISTORY ATTENTION ARCHITECTURE (DISTILLED)")
    print("=" * 80)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    year_distilled_dir = MODELS_DIR / "year_anchor_distilled"
    year_distilled_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve research root and load data
    research_root = resolve_research_root()
    print(f"Research Root: {research_root}")
    backend, _ = uncertainty._activate_archive_backend(research_root)
    print("Activated archive backend.")

    data = uncertainty._load_training_data(research_root)
    frame = data["frame"]

    from src.models import competition_tiers as tiers
    identities = [tiers.classify_competition(name, pd.Timestamp(day).date())
                  for name, day in zip(frame.tournament_name, frame.date)]
    frame["competition_tier"] = [identity.tier.value for identity in identities]

    split = runner._partition(frame, smoke=False, history=data["history"], raw=data["raw"])
    train_ix = split["train"]
    cal_ix = split["cal"]
    assess_ix = split["validation"]

    print(f"Partitions: TRAIN={len(train_ix)}, CAL={len(cal_ix)}, ASSESS={len(assess_ix)}")
    assert len(train_ix) == 36205, f"Expected 36,205 TRAIN rows, got {len(train_ix)}"
    assert len(cal_ix) == 1328, f"Expected 1,328 CAL rows, got {len(cal_ix)}"
    assert len(assess_ix) == 3103, f"Expected 3,103 ASSESS rows, got {len(assess_ix)}"

    shared = joblib.load(PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/shared/model.joblib")
    ref_cal = joblib.load(PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/calibration/model.joblib")
    print("Loaded shared experts and reference calibration.")

    y_hist = np.load(BANK_DIR / "year_anchor_history.npy", mmap_mode="r")
    y_mask = np.load(BANK_DIR / "year_anchor_mask.npy", mmap_mode="r")
    year_ids = np.load(BANK_DIR / "year_anchor_row_ids.npy", mmap_mode="r", allow_pickle=False)
    np.testing.assert_array_equal(year_ids, frame["golgg_match_id"].astype(str).to_numpy(dtype=str))
    print(f"Loaded year anchor history {y_hist.shape} and mask {y_mask.shape}; row IDs match metadata.")
    assert y_hist.shape == (40636, 2, 5, 1, 33)
    assert y_mask.shape == (40636, 2, 5, 1)

    # Concatenate year anchor at index 0 and recent 16 maps at indices 1..16 -> 17 maps total
    print("Concatenating year anchor (1 map) + recent history (16 maps) -> 17 maps total...")
    h_17 = np.concatenate([y_hist, data["history"]["history"]], axis=3)
    m_17 = np.concatenate([y_mask, data["history"]["mask"]], axis=3)
    print(f"Concatenated history shape: {h_17.shape}, mask shape: {m_17.shape}")

    data_17 = {
        "raw": data["raw"],
        "history": {
            "history": h_17,
            "mask": m_17,
            "champions": data["history"].get("champions"),
        },
        "wins_a": data["wins_a"],
        "wins_b": data["wins_b"],
        "targets": data["targets"],
        "target_mask": data["target_mask"],
        "frame": data["frame"],
    }

    # Verify cold-start invariance and antisymmetry on architecture
    print("\nVerifying Cold-Start Neutrality and Antisymmetry Invariance...")
    torch.manual_seed(20260910)
    test_net = YearAnchorCommonMapResidual(width=32)
    B_test = 2
    x_cold = {
        "players": torch.zeros(B_test, 2, 5, 21),
        "context": torch.zeros(B_test, 2, 70),
        "history": torch.zeros(B_test, 2, 5, 17, 33),
        "mask": torch.zeros(B_test, 2, 5, 17, dtype=torch.bool),
    }
    z_cold = test_net(x_cold, torch.zeros(B_test, dtype=torch.long))
    p_cold = float(expit(z_cold[0].item()))
    print(f"  Cold-start probability: {p_cold:.6f} (expected exactly 0.500000)")
    assert abs(p_cold - 0.5) < 1e-6, f"Cold start neutrality failed: {p_cold}"

    # Side-swap antisymmetry test
    p_a = torch.randn(B_test, 5, 21)
    p_b = torch.randn(B_test, 5, 21)
    ctx_a = torch.randn(B_test, 70)
    ctx_b = torch.randn(B_test, 70)
    x_ab = {
        "players": torch.stack([p_a, p_b], dim=1),
        "context": torch.stack([ctx_a, ctx_b], dim=1),
        "history": torch.zeros(B_test, 2, 5, 17, 33),
        "mask": torch.zeros(B_test, 2, 5, 17, dtype=torch.bool),
    }
    x_ba = {
        "players": torch.stack([p_b, p_a], dim=1),
        "context": torch.stack([ctx_b, ctx_a], dim=1),
        "history": torch.zeros(B_test, 2, 5, 17, 33),
        "mask": torch.zeros(B_test, 2, 5, 17, dtype=torch.bool),
    }
    z_ab = test_net(x_ab, torch.zeros(B_test, dtype=torch.long))
    z_ba = test_net(x_ba, torch.zeros(B_test, dtype=torch.long))
    antisym_err = torch.max(torch.abs(z_ab + z_ba)).item()
    print(f"  Max antisymmetry error: {antisym_err:.10f} (expected 0.0)")
    assert antisym_err < 1e-6, f"Antisymmetry failed: {antisym_err}"
    print("Invariance verification passed.")

    # 3. Load Teacher B
    teacher_b_file = REPORT_DIR / "teacher_probs_B.npy"
    if not teacher_b_file.is_file():
        raise FileNotFoundError(f"Expected cached Teacher B at {teacher_b_file}")
    teacher_probs_B = np.load(teacher_b_file)
    print(f"Loaded cached Teacher B from {teacher_b_file}, shape: {teacher_probs_B.shape}")
    assert np.isfinite(teacher_probs_B).all() and np.allclose(teacher_probs_B.sum(axis=1), 1.0, atol=1e-10)

    # 4. Load baseline references (C0, C0-D-Core, C0-Slim)
    p1 = pd.read_parquet(REPORT_DIR / "track1_predictions.parquet")
    p4 = pd.read_parquet(REPORT_DIR / "track4_predictions.parquet")
    p_c0_full = p1["p_c0__full"].to_numpy(dtype=np.float64)
    p_d_core_full = p1["w32_e16_kd_born_again__full"].to_numpy(dtype=np.float64)
    p_slim_full = p1["w32_e16_kd_single__full"].to_numpy(dtype=np.float64)
    p_conn_full = p4["p_c0_connector"].to_numpy(dtype=np.float64)

    assess_frame = frame.iloc[assess_ix].reset_index(drop=True)
    y_assess = assess_frame["y"].to_numpy(dtype=int)
    dates_assess = assess_frame["date"].to_numpy(dtype=str)
    result_days_assess = p4["result_day"].to_numpy()

    ll_c0 = compute_unclipped_log_loss(y_assess, p_c0_full)
    ll_d_core = compute_unclipped_log_loss(y_assess, p_d_core_full)
    print(f"Baseline C0:        LL={ll_c0:.6f} (expected ~0.535449)")
    print(f"Baseline C0-D-Core: LL={ll_d_core:.6f} (expected ~0.534418)")

    # 5. Train 18-Token Distilled Models (3 seeds, 16 epochs, lambda=0.6)
    print("\n--- Training / Loading 18-Token Distilled Models (3 seeds, 16 epochs, lambda=0.6) ---")
    bo_train = frame.best_of.iloc[train_ix].to_numpy(dtype=np.int64)
    bo_index_train = (bo_train - 1) // 2

    year_distilled_models = []
    for seed in SEEDS:
        ckpt_dir = year_distilled_dir / f"seed_{seed}"
        m = train_or_load_year_anchor_model(
            width=32, seed=seed, epochs=16, distillation_weight=0.6,
            teacher_probs=teacher_probs_B, ckpt_dir=ckpt_dir,
            data_17=data_17, train_ix=train_ix, bo_index_train=bo_index_train, shared=shared
        )
        year_distilled_models.append(m)

    # 6. Construct & Calibrate Models on CAL
    print("\n--- Calibrating Models on CAL (N=1,328) & Assembling on ASSESS (N=3,103) ---")

    # A. C0-Slim-Year (Single seed 20260910)
    single_seed_year = [year_distilled_models[0]] * 3
    p_slim_year_full, p_slim_year_neural, _, cal_info_slim_year, p_slim_year_cal_full, p_slim_year_cal_neural = fit_beta_calibration_and_assemble(
        single_seed_year, data_17, data, cal_ix, assess_ix, backend, shared, ref_cal
    )

    # B. C0-D-Year-Core (3 seeds 20260910..12)
    p_d_year_core_full, p_d_year_core_neural, _, cal_info_d_year_core, p_d_year_core_cal_full, p_d_year_core_cal_neural = fit_beta_calibration_and_assemble(
        year_distilled_models, data_17, data, cal_ix, assess_ix, backend, shared, ref_cal
    )

    # Load C0-D-Core checkpoints to get CAL predictions for hybrid core logit blend
    d_core_dir = MODELS_DIR / "student_w32_kd"
    d_core_models = [
        cap_exp.ParametricNonlinearHistoryModel(shared["baseline"], shared["normalizer"], width=32, seed=s, epochs=16, coefficient=0.1)
        for s in SEEDS
    ]
    for s, m_c in zip(SEEDS, d_core_models):
        st = torch.load(d_core_dir / f"seed_{s}/checkpoint_epoch16.pt", map_location="cpu", weights_only=False)
        m_c.network.load_state_dict(st["network_state_dict"])
        m_c.eta = nn.Parameter(st["eta"])
        m_c.target_scaler = st["target_scaler"]
        m_c.network.eval()

    p_d_core_assess_check, _, _, _, p_d_core_cal_full, _ = fit_beta_calibration_and_assemble(
        d_core_models, data, data, cal_ix, assess_ix, backend, shared, ref_cal
    )
    assert np.allclose(p_d_core_assess_check, p_d_core_full, atol=1e-5), "C0-D-Core assemble mismatch"

    # C. C0-D-Year-HybridCore: CAL-fitted logit blend of C0-D-Year-Core + C0-D-Core
    y_cal = frame.iloc[cal_ix]["y"].to_numpy(dtype=float)
    z_year_cal = logit(np.clip(p_d_year_core_cal_full, 1e-12, 1.0 - 1e-12))
    z_core_cal = logit(np.clip(p_d_core_cal_full, 1e-12, 1.0 - 1e-12))

    def cal_blend_objective(w: float) -> float:
        z_b = w * z_year_cal + (1.0 - w) * z_core_cal
        p_b = expit(z_b)
        return compute_unclipped_log_loss(y_cal, p_b)

    opt_blend = minimize_scalar(cal_blend_objective, bounds=(0.0, 1.0), method="bounded")
    best_hybrid_weight = float(opt_blend.x)
    print(f"Optimal CAL logit blend weight for C0-D-Year-Core: {best_hybrid_weight:.4f} (C0-D-Core weight: {1.0 - best_hybrid_weight:.4f})")

    z_year_assess = logit(np.clip(p_d_year_core_full, 1e-12, 1.0 - 1e-12))
    z_core_assess = logit(np.clip(p_d_core_full, 1e-12, 1.0 - 1e-12))
    p_d_year_hybrid_core = expit(best_hybrid_weight * z_year_assess + (1.0 - best_hybrid_weight) * z_core_assess)

    # D. C0-D-Year: 0.50 * C0-D-Year-Core + 0.50 * c0_raw_connector + causal ONS
    p_d_year_pre_ons = 0.50 * p_d_year_core_full + 0.50 * p_conn_full
    p_d_year_full, ons_theta_hist = run_ons_recalibration(
        dates_assess, result_days_assess, logit(p_d_year_pre_ons), y_assess
    )

    # 7. Evaluate on ASSESS
    print("\n--- Evaluating Models on ASSESS (N=3,103) ---")
    metrics_slim_year_full = evaluate_probability_stream(
        y_assess, p_slim_year_full, dates_assess, assess_frame, p_c0_full, p_d_core_full
    )
    metrics_slim_year_neural = evaluate_probability_stream(
        y_assess, p_slim_year_neural, dates_assess, assess_frame, p_c0_full, p_d_core_full
    )
    metrics_d_year_core_full = evaluate_probability_stream(
        y_assess, p_d_year_core_full, dates_assess, assess_frame, p_c0_full, p_d_core_full
    )
    metrics_d_year_core_neural = evaluate_probability_stream(
        y_assess, p_d_year_core_neural, dates_assess, assess_frame, p_c0_full, p_d_core_full
    )
    metrics_d_year_hybrid_core = evaluate_probability_stream(
        y_assess, p_d_year_hybrid_core, dates_assess, assess_frame, p_c0_full, p_d_core_full
    )
    metrics_d_year_full = evaluate_probability_stream(
        y_assess, p_d_year_full, dates_assess, assess_frame, p_c0_full, p_d_core_full
    )

    print(f"C0 Baseline:                      LL={ll_c0:.6f}")
    print(f"C0-D-Core (Track 1 KD):           LL={ll_d_core:.6f}")
    print(f"C0-Slim-Year (1-Seed Assembled):  LL={metrics_slim_year_full['unclipped_log_loss']:.6f}, Brier={metrics_slim_year_full['brier_score']:.6f}")
    print(f"C0-Slim-Year (1-Seed Neural):     LL={metrics_slim_year_neural['unclipped_log_loss']:.6f}, Brier={metrics_slim_year_neural['brier_score']:.6f}")
    print(f"C0-D-Year-Core (3-Seed Assembled):LL={metrics_d_year_core_full['unclipped_log_loss']:.6f}, Brier={metrics_d_year_core_full['brier_score']:.6f}")
    print(f"C0-D-Year-Core (3-Seed Neural):   LL={metrics_d_year_core_neural['unclipped_log_loss']:.6f}, Brier={metrics_d_year_core_neural['brier_score']:.6f}")
    print(f"C0-D-Year-HybridCore:             LL={metrics_d_year_hybrid_core['unclipped_log_loss']:.6f}, Brier={metrics_d_year_hybrid_core['brier_score']:.6f}")
    print(f"C0-D-Year (Full System + ONS):    LL={metrics_d_year_full['unclipped_log_loss']:.6f}, Brier={metrics_d_year_full['brier_score']:.6f}")

    print(f"  Delta C0-D-Year-Core vs C0:     {metrics_d_year_core_full['unclipped_log_loss'] - ll_c0:+.6f}")
    print(f"  Delta C0-D-Year-Core vs D-Core: {metrics_d_year_core_full['unclipped_log_loss'] - ll_d_core:+.6f}")
    print(f"  Delta C0-D-Year vs C0:          {metrics_d_year_full['unclipped_log_loss'] - ll_c0:+.6f}")

    # 8. Longshot & Slump-Recovery Analysis
    print("\n--- Longshot & Slump-Recovery Analysis ---")

    # A. Longshot Upsets Analysis
    p_winner_c0 = np.where(y_assess == 1, p_c0_full, 1.0 - p_c0_full)
    c0_longshot_mask = p_winner_c0 <= 0.35

    mkt_raw = pd.read_parquet(PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet")
    mkt_sub = mkt_raw[["golgg_match_id", "q_consensus"]].copy()
    mkt_sub["golgg_match_id"] = mkt_sub["golgg_match_id"].astype(str)

    assess_mkt_df = pd.DataFrame({
        "golgg_match_id": assess_frame["golgg_match_id"].astype(str),
        "y": y_assess,
    }).merge(mkt_sub, on="golgg_match_id", how="left")

    q_win = np.where(assess_mkt_df["y"] == 1, assess_mkt_df["q_consensus"], 1.0 - assess_mkt_df["q_consensus"])
    odds_win = 1.0 / np.clip(q_win, 1e-6, 1.0)
    mkt_odds_bracket = (odds_win >= 2.50) & (odds_win <= 4.00) & assess_mkt_df["q_consensus"].notna().to_numpy()

    eval_models = {
        "C0": p_c0_full,
        "C0-D-Core": p_d_core_full,
        "C0-Slim": p_slim_full,
        "C0-Slim-Year": p_slim_year_full,
        "C0-D-Year-Core": p_d_year_core_full,
        "C0-D-Year": p_d_year_full,
    }

    longshot_analysis: dict[str, Any] = {
        "c0_priced_upset_p_le_0_35": {"n": int(c0_longshot_mask.sum())},
        "market_odds_bracket_2_50_to_4_00": {"n": int(mkt_odds_bracket.sum())},
    }

    for m_name, p_arr in eval_models.items():
        # C0 priced upset
        ll_sub = compute_unclipped_log_loss(y_assess[c0_longshot_mask], p_arr[c0_longshot_mask])
        brier_sub = float(np.mean((p_arr[c0_longshot_mask] - y_assess[c0_longshot_mask]) ** 2))
        p_win_m = np.where(y_assess == 1, p_arr, 1.0 - p_arr)
        mean_p_win = float(np.mean(p_win_m[c0_longshot_mask]))
        longshot_analysis["c0_priced_upset_p_le_0_35"][m_name] = {
            "log_loss": ll_sub,
            "brier": brier_sub,
            "mean_winner_prob": mean_p_win,
        }

        # Market bracket
        if mkt_odds_bracket.sum() > 0:
            ll_mkt = compute_unclipped_log_loss(y_assess[mkt_odds_bracket], p_arr[mkt_odds_bracket])
            brier_mkt = float(np.mean((p_arr[mkt_odds_bracket] - y_assess[mkt_odds_bracket]) ** 2))
            mean_p_mkt = float(np.mean(p_win_m[mkt_odds_bracket]))
            longshot_analysis["market_odds_bracket_2_50_to_4_00"][m_name] = {
                "log_loss": ll_mkt,
                "brier": brier_mkt,
                "mean_winner_prob": mean_p_mkt,
            }

    print(f"Longshot Upsets (C0 p_win <= 0.35, N={c0_longshot_mask.sum()}):")
    for m_name in eval_models:
        res = longshot_analysis["c0_priced_upset_p_le_0_35"][m_name]
        print(f"  {m_name:<15s}: LL={res['log_loss']:.4f}, mean_p_winner={res['mean_winner_prob']:.4f}")

    # B. Slumping Elite Teams Analysis
    assess_h = data["history"]["history"][assess_ix]
    assess_m = data["history"]["mask"][assess_ix].astype(bool)
    assess_yh = y_hist[assess_ix]
    assess_ym = y_mask[assess_ix].astype(bool)

    recent_wr = np.zeros((len(assess_ix), 2), dtype=float)
    year_wr = np.zeros((len(assess_ix), 2), dtype=float)

    for side in (0, 1):
        for i in range(len(assess_ix)):
            m_r = assess_m[i, side]
            if m_r.any():
                recent_wr[i, side] = assess_h[i, side, ..., 24][m_r].mean()
            else:
                recent_wr[i, side] = np.nan
            m_y = assess_ym[i, side]
            if m_y.any():
                year_wr[i, side] = assess_yh[i, side, ..., 24][m_y].mean()
            else:
                year_wr[i, side] = np.nan

    slump_team0 = (year_wr[:, 0] - recent_wr[:, 0]) >= 0.10
    slump_team1 = (year_wr[:, 1] - recent_wr[:, 1]) >= 0.10
    slump_either = slump_team0 | slump_team1

    slump_analysis: dict[str, Any] = {
        "slumping_team_either_side": {"n": int(slump_either.sum())},
        "slumping_team_0": {"n": int(slump_team0.sum())},
        "slumping_team_1": {"n": int(slump_team1.sum())},
    }

    for subset_name, sub_mask in [
        ("slumping_team_either_side", slump_either),
        ("slumping_team_0", slump_team0),
        ("slumping_team_1", slump_team1),
    ]:
        for m_name, p_arr in eval_models.items():
            ll_s = compute_unclipped_log_loss(y_assess[sub_mask], p_arr[sub_mask])
            brier_s = float(np.mean((p_arr[sub_mask] - y_assess[sub_mask]) ** 2))
            acc_s = float(np.mean((p_arr[sub_mask] >= 0.5) == y_assess[sub_mask]))
            slump_analysis[subset_name][m_name] = {
                "log_loss": ll_s,
                "brier": brier_s,
                "accuracy": acc_s,
            }

    print(f"\nSlumping Elite Teams (Anchor WR >= Recent WR + 10%, N={slump_either.sum()}):")
    for m_name in eval_models:
        res = slump_analysis["slumping_team_either_side"][m_name]
        print(f"  {m_name:<15s}: LL={res['log_loss']:.5f}, Brier={res['brier']:.5f}, Acc={res['accuracy']:.4f}")

    # 9. Full Financial Simulations (10,000 PLN Bankroll, NO EXP-039, All Gates + DRO Active)
    print("\n" + "=" * 80)
    print("FINANCIAL SIMULATIONS (10,000 PLN BANKROLL, NO EXP-039, GATES + DRO ACTIVE)")
    print("=" * 80)

    # A. 50-Bet / 45-Bet Polish Production Cohort
    ledger_path = PROJECT_ROOT / "data/08_reporting/optimal_strategy_simulation_ledger.csv"
    ledger_df = pd.read_csv(ledger_path)

    um_df = pd.read_parquet(PROJECT_ROOT / "data/07_model_output/a0_native_refit/native_refit_20260929_122617/uniform/uniform_monthly/candidate.parquet")
    um_2026 = um_df.set_index("golgg_match_id").loc[p1["golgg_match_id"].astype(str)].reset_index()
    p_c0_mon = um_2026["p_full"].to_numpy(dtype=np.float64)
    p1_dates = pd.to_datetime(p1["date"])
    gid_to_idx = {str(gid): i for i, gid in enumerate(p1["golgg_match_id"].values)}

    ledger_to_golgg: dict[int, tuple[str, bool]] = {}
    for _, r in ledger_df.iterrows():
        bid = int(r["bet_id"])
        dt = pd.to_datetime(str(r["date"])[:10])
        target_p = float(r["p_c0"])
        mask_dt = (p1_dates - dt).abs().dt.days <= 3
        idxs = np.where(mask_dt)[0]
        best = None
        for idx in idxs:
            for is_rev in (False, True):
                val_m = (1.0 - p_c0_mon[idx]) if is_rev else p_c0_mon[idx]
                val_s = (1.0 - p_c0_full[idx]) if is_rev else p_c0_full[idx]
                diff = min(abs(val_m - target_p), abs(val_s - target_p))
                if best is None or diff < best[0]:
                    best = (diff, str(p1["golgg_match_id"].iloc[idx]), is_rev)
        if best is not None:
            ledger_to_golgg[bid] = (best[1], best[2])

    assert len(ledger_to_golgg) == len(ledger_df)

    l_c0_led = logit(ledger_df["p_c0"].to_numpy(dtype=float))
    l_hyb_led = logit(ledger_df["p_hybrid"].to_numpy(dtype=float))
    l_mkt_led = 2.0 * l_hyb_led - l_c0_led
    p_thesis_led = ledger_df["p_thesis"].to_numpy(dtype=float)
    odds_led = ledger_df["odds_taken"].to_numpy(dtype=float)
    won_led = ledger_df["won"].to_numpy(dtype=int)

    financial_ledger_results: dict[str, Any] = {}

    for m_name, p_arr in eval_models.items():
        financial_ledger_results[m_name] = {}
        if m_name == "C0":
            p_m_vec = ledger_df["p_c0"].to_numpy(dtype=float)
        else:
            p_m_list = []
            for idx_row, r in ledger_df.iterrows():
                bid = int(r["bet_id"])
                gid, is_rev = ledger_to_golgg[bid]
                g_idx = gid_to_idx[gid]
                val_m = (1.0 - p_arr[g_idx]) if is_rev else p_arr[g_idx]
                val_s = (1.0 - p_c0_full[g_idx]) if is_rev else p_c0_full[g_idx]
                p_sel = float(expit(logit(float(r["p_c0"])) + (logit(val_m) - logit(val_s))))
                p_m_list.append(p_sel)
            p_m_vec = np.array(p_m_list)

        l_m = logit(np.clip(p_m_vec, 1e-6, 1.0 - 1e-6))
        l_hyb = 0.50 * l_m + 0.50 * l_mkt_led
        p_hyb = expit(l_hyb)
        ev_net = p_hyb * 0.88 * odds_led - 1.0

        guard_b = np.abs(l_m - l_mkt_led) <= 0.85
        odds_ok = odds_led <= 4.00
        ev_ok = (ev_net >= 0.05) & (ev_net <= 0.25)
        dro_ok = ((p_hyb - 0.025) * 0.88 * odds_led - 1.0) > 0.0
        consensus_veto = np.abs(p_m_vec - p_thesis_led) <= 0.10

        mask_no_veto = guard_b & odds_ok & ev_ok & dro_ok
        mask_with_veto = guard_b & odds_ok & ev_ok & dro_ok & consensus_veto

        for regime_label, mask in [
            ("without_veto_50_cohort", mask_no_veto),
            ("with_veto_45_cohort", mask_with_veto),
        ]:
            active_indices = np.where(mask)[0]
            sim_bets = []
            bankroll = 10000.0

            for i in active_indices:
                o_taken = odds_led[i]
                w = won_led[i]
                p_c = p_hyb[i]
                p_cl = (1.0 + ledger_df["raw_clv_pct"].iloc[i] / 100.0) / o_taken
                b_net = o_taken * 0.88 - 1.0
                f_star = max(0.0, (b_net * p_c - (1.0 - p_c)) / b_net) if b_net > 0 else 0.0
                stake = min(150.0, max(20.0, bankroll * 0.25 * f_star))
                profit = stake * (o_taken * 0.88 - 1.0) if w else -stake
                bankroll += profit
                sim_bets.append({
                    "odds_taken": o_taken,
                    "won": w,
                    "stake": stake,
                    "profit": profit,
                    "bankroll": bankroll,
                    "ev_claimed": p_c * 0.88 * o_taken - 1.0,
                    "net_clv": 0.88 * o_taken * p_cl - 1.0,
                })

            decomp = compute_three_part_decomposition(sim_bets, initial_bankroll=10000.0)
            financial_ledger_results[m_name][regime_label] = decomp

    print("\nPolish Production Cohort Results (1/4 Kelly, 12% Tax):")
    print(f"{'Model':<15s} | {'Regime':<24s} | {'Bets':<5s} | {'WinRate':<7s} | {'Profit PLN':<10s} | {'Yield %':<8s} | {'CLV %':<8s} | {'Bias pp':<8s}")
    for m_name in eval_models:
        for r_name in ("without_veto_50_cohort", "with_veto_45_cohort"):
            res = financial_ledger_results[m_name][r_name]
            print(f"{m_name:<15s} | {r_name:<24s} | {res['n_bets']:<5d} | {res['win_rate_pct']:<7.1f} | {res['total_net_profit_pln']:<+10.2f} | {res['realized_yield_pct']:<+8.2f} | {res['true_edge_net_clv_pct']:<+8.2f} | {res['optimizers_curse_bias_pp']:<+8.2f}")

    # B. Longshot Bets Subset (Odds in [2.30, 4.00])
    longshot_subset_results: dict[str, Any] = {"polish_books_ledger": {}, "pinnacle_t1": {}}

    for m_name in eval_models:
        # Polish books ledger longshots
        p_m_list = []
        for idx_row, r in ledger_df.iterrows():
            bid = int(r["bet_id"])
            gid, is_rev = ledger_to_golgg[bid]
            g_idx = gid_to_idx[gid]
            val_m = (1.0 - eval_models[m_name][g_idx]) if is_rev else eval_models[m_name][g_idx]
            val_s = (1.0 - p_c0_full[g_idx]) if is_rev else p_c0_full[g_idx]
            p_sel = float(expit(logit(float(r["p_c0"])) + (logit(val_m) - logit(val_s)))) if m_name != "C0" else float(r["p_c0"])
            p_m_list.append(p_sel)
        p_m_vec = np.array(p_m_list)

        l_m = logit(np.clip(p_m_vec, 1e-6, 1.0 - 1e-6))
        l_hyb = 0.50 * l_m + 0.50 * l_mkt_led
        p_hyb = expit(l_hyb)
        ev_net = p_hyb * 0.88 * odds_led - 1.0

        mask_longshot = (odds_led >= 2.30) & (odds_led <= 4.00) & (np.abs(l_m - l_mkt_led) <= 0.85) & (ev_net >= 0.05) & (ev_net <= 0.25) & (((p_hyb - 0.025) * 0.88 * odds_led - 1.0) > 0.0)

        active_indices = np.where(mask_longshot)[0]
        sim_bets = []
        bankroll = 10000.0

        for i in active_indices:
            o_taken = odds_led[i]
            w = won_led[i]
            p_c = p_hyb[i]
            p_cl = (1.0 + ledger_df["raw_clv_pct"].iloc[i] / 100.0) / o_taken
            b_net = o_taken * 0.88 - 1.0
            f_star = max(0.0, (b_net * p_c - (1.0 - p_c)) / b_net) if b_net > 0 else 0.0
            stake = min(150.0, max(20.0, bankroll * 0.25 * f_star))
            profit = stake * (o_taken * 0.88 - 1.0) if w else -stake
            bankroll += profit
            sim_bets.append({
                "odds_taken": o_taken,
                "won": w,
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
                "ev_claimed": p_c * 0.88 * o_taken - 1.0,
                "net_clv": 0.88 * o_taken * p_cl - 1.0,
            })

        longshot_subset_results["polish_books_ledger"][m_name] = compute_three_part_decomposition(sim_bets, initial_bankroll=10000.0)

    # C. Sharp Pinnacle T-1h Market Simulation (N=1,668)
    pinnacle_t1_path = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"
    mkt_raw = pd.read_parquet(pinnacle_t1_path)
    mkt_df = mkt_raw[["golgg_match_id", "date", "y", "q_consensus", "p_039"]].copy()
    mkt_df["golgg_match_id"] = mkt_df["golgg_match_id"].astype(str)

    p1_sub = pd.DataFrame({"golgg_match_id": p1["golgg_match_id"].astype(str)})
    for m_name, arr in eval_models.items():
        p1_sub[m_name] = arr

    merged_mkt = mkt_df.merge(p1_sub, on="golgg_match_id").sort_values(by=["date", "golgg_match_id"]).reset_index(drop=True)
    print(f"Constructed unified Pinnacle T-1h market cohort: {len(merged_mkt)} fixtures.")

    pinnacle_results: dict[str, Any] = {}

    regimes = [
        ("12_pct_polish_tax", 0.12),
        ("0_pct_international_tax", 0.00),
    ]

    strategies = [
        ("unconstrained_flat100", "unconstrained", False, False),
        ("hybrid50_flat100", "hybrid_flat", False, False),
        ("hybrid50_kelly", "hybrid_kelly", False, False),
        ("optimal_synergy_dro_no_veto", "optimal_dro_no_veto", False, False),
        ("optimal_synergy_dro_with_veto", "optimal_dro_with_veto", True, False),
        ("longshot_bracket_odds_2_30_4_00", "optimal_dro_no_veto", False, True),
    ]

    for regime_name, tax_rate in regimes:
        c = 1.0 - tax_rate
        pinnacle_results[regime_name] = {}

        for m_name in eval_models:
            pinnacle_results[regime_name][m_name] = {}

            p1_model = merged_mkt[m_name].to_numpy(dtype=float)
            p2_model = 1.0 - p1_model
            q1 = merged_mkt["q_consensus"].to_numpy(dtype=float)
            q2 = 1.0 - q1
            o1 = 1.0 / q1
            o2 = 1.0 / q2
            y_mkt = merged_mkt["y"].to_numpy(dtype=int)
            p_039_mkt = merged_mkt["p_039"].to_numpy(dtype=float)

            eps = 1e-12
            l_p1 = logit(np.clip(p1_model, eps, 1.0 - eps))
            l_q1 = logit(np.clip(q1, eps, 1.0 - eps))
            l_p2 = logit(np.clip(p2_model, eps, 1.0 - eps))
            l_q2 = logit(np.clip(q2, eps, 1.0 - eps))

            p1_hyb = expit(0.50 * l_p1 + 0.50 * l_q1)
            p2_hyb = expit(0.50 * l_p2 + 0.50 * l_q2)

            gap1 = np.abs(l_p1 - l_q1)
            gap2 = np.abs(l_p2 - l_q2)

            ev1_pure = p1_model * c * o1 - 1.0
            ev2_pure = p2_model * c * o2 - 1.0
            ev1_hyb = p1_hyb * c * o1 - 1.0
            ev2_hyb = p2_hyb * c * o2 - 1.0

            for strat_key, strat_kind, use_veto, longshot_only in strategies:
                sim_bets = []
                bankroll = 10000.0

                for i in range(len(merged_mkt)):
                    side = 0

                    if strat_kind == "unconstrained":
                        c1 = (ev1_pure[i] >= 0.05) and (o1[i] <= 4.00)
                        c2 = (ev2_pure[i] >= 0.05) and (o2[i] <= 4.00)
                        if c1 and (not c2 or ev1_pure[i] >= ev2_pure[i]):
                            side, o_taken, p_claim, p_close, won, ev_claim = 1, o1[i], p1_model[i], q1[i], int(y_mkt[i] == 1), ev1_pure[i]
                        elif c2:
                            side, o_taken, p_claim, p_close, won, ev_claim = 2, o2[i], p2_model[i], q2[i], int(y_mkt[i] == 0), ev2_pure[i]
                        stake = 100.0

                    elif strat_kind in ("hybrid_flat", "hybrid_kelly"):
                        c1 = (ev1_hyb[i] >= 0.05) and (ev1_hyb[i] <= 0.25) and (gap1[i] <= 0.85) and (o1[i] <= 4.00)
                        c2 = (ev2_hyb[i] >= 0.05) and (ev2_hyb[i] <= 0.25) and (gap2[i] <= 0.85) and (o2[i] <= 4.00)
                        if c1 and (not c2 or ev1_hyb[i] >= ev2_hyb[i]):
                            side, o_taken, p_claim, p_close, won, ev_claim = 1, o1[i], p1_hyb[i], q1[i], int(y_mkt[i] == 1), ev1_hyb[i]
                        elif c2:
                            side, o_taken, p_claim, p_close, won, ev_claim = 2, o2[i], p2_hyb[i], q2[i], int(y_mkt[i] == 0), ev2_hyb[i]

                        if strat_kind == "hybrid_flat":
                            stake = 100.0
                        else:
                            b_net = o_taken * c - 1.0 if side > 0 else 0.0
                            f_star = max(0.0, (b_net * p_claim - (1.0 - p_claim)) / b_net) if b_net > 0 else 0.0
                            stake = min(150.0, max(20.0, bankroll * 0.25 * f_star))

                    elif strat_kind in ("optimal_dro_no_veto", "optimal_dro_with_veto"):
                        veto1 = (not use_veto) or np.isnan(p_039_mkt[i]) or (abs(p1_model[i] - p_039_mkt[i]) <= 0.10)
                        veto2 = (not use_veto) or np.isnan(p_039_mkt[i]) or (abs(p2_model[i] - (1.0 - p_039_mkt[i])) <= 0.10)

                        min_o = 2.30 if longshot_only else 1.00
                        c1 = (ev1_hyb[i] >= 0.05) and (ev1_hyb[i] <= 0.25) and (gap1[i] <= 0.85) and veto1 and (o1[i] >= min_o) and (o1[i] <= 4.00)
                        c2 = (ev2_hyb[i] >= 0.05) and (ev2_hyb[i] <= 0.25) and (gap2[i] <= 0.85) and veto2 and (o2[i] >= min_o) and (o2[i] <= 4.00)

                        c1 = c1 and (((p1_hyb[i] - 0.025) * c * o1[i] - 1.0) > 0.0)
                        c2 = c2 and (((p2_hyb[i] - 0.025) * c * o2[i] - 1.0) > 0.0)

                        if c1 and (not c2 or ev1_hyb[i] >= ev2_hyb[i]):
                            side, o_taken, p_claim, p_close, won, ev_claim = 1, o1[i], p1_hyb[i], q1[i], int(y_mkt[i] == 1), ev1_hyb[i]
                        elif c2:
                            side, o_taken, p_claim, p_close, won, ev_claim = 2, o2[i], p2_hyb[i], q2[i], int(y_mkt[i] == 0), ev2_hyb[i]

                        b_net = o_taken * c - 1.0 if side > 0 else 0.0
                        f_star = max(0.0, (b_net * p_claim - (1.0 - p_claim)) / b_net) if b_net > 0 else 0.0
                        stake = min(150.0, max(20.0, bankroll * 0.25 * f_star))

                    if side == 0 or stake <= 0.0 or stake > bankroll:
                        continue

                    profit = stake * (o_taken * c - 1.0) if won else -stake
                    bankroll += profit
                    sim_bets.append({
                        "odds_taken": o_taken,
                        "won": won,
                        "stake": stake,
                        "profit": profit,
                        "bankroll": bankroll,
                        "ev_claimed": ev_claim,
                        "net_clv": c * o_taken * p_close - 1.0,
                    })

                pinnacle_results[regime_name][m_name][strat_key] = compute_three_part_decomposition(
                    sim_bets, initial_bankroll=10000.0
                )

    print("\nPinnacle Sharp T-1h Simulation Summary (12% Tax):")
    print(f"{'Model':<15s} | {'Strategy':<32s} | {'Bets':<5s} | {'WinRate':<7s} | {'Profit PLN':<10s} | {'Yield %':<8s} | {'CLV %':<8s}")
    for m_name in eval_models:
        for s_key in ("hybrid50_kelly", "optimal_synergy_dro_no_veto", "longshot_bracket_odds_2_30_4_00"):
            res = pinnacle_results["12_pct_polish_tax"][m_name][s_key]
            print(f"{m_name:<15s} | {s_key:<32s} | {res['n_bets']:<5d} | {res['win_rate_pct']:<7.1f} | {res['total_net_profit_pln']:<+10.2f} | {res['realized_yield_pct']:<+8.2f} | {res['true_edge_net_clv_pct']:<+8.2f}")

    print("\nPinnacle Sharp T-1h Simulation Summary (0% International Tax):")
    print(f"{'Model':<15s} | {'Strategy':<32s} | {'Bets':<5s} | {'WinRate':<7s} | {'Profit PLN':<10s} | {'Yield %':<8s} | {'CLV %':<8s}")
    for m_name in eval_models:
        for s_key in ("hybrid50_kelly", "optimal_synergy_dro_no_veto", "longshot_bracket_odds_2_30_4_00"):
            res = pinnacle_results["0_pct_international_tax"][m_name][s_key]
            print(f"{m_name:<15s} | {s_key:<32s} | {res['n_bets']:<5d} | {res['win_rate_pct']:<7.1f} | {res['total_net_profit_pln']:<+10.2f} | {res['realized_yield_pct']:<+8.2f} | {res['true_edge_net_clv_pct']:<+8.2f}")

    # 10. Save Artifacts: Predictions Parquet & Reporting JSON
    print("\n--- Exporting Artifacts ---")
    pred_df = pd.DataFrame({
        "golgg_match_id": assess_frame["golgg_match_id"].astype(str),
        "date": assess_frame["date"].astype(str),
        "y": y_assess,
        "best_of": assess_frame["best_of"].astype(int),
        "tournament_name": assess_frame["tournament_name"].astype(str),
        "competition_tier": assess_frame["competition_tier"].astype(str),
        "unknown_players": assess_frame["unknown_players"].astype(int),
        "p_c0__full": p_c0_full,
        "p_c0_d_core__full": p_d_core_full,
        "p_c0_slim__full": p_slim_full,
        "p_c0_slim_year__full": p_slim_year_full,
        "p_c0_slim_year__neural": p_slim_year_neural,
        "p_c0_d_year_core__full": p_d_year_core_full,
        "p_c0_d_year_core__neural": p_d_year_core_neural,
        "p_c0_d_year_hybrid_core__full": p_d_year_hybrid_core,
        "p_c0_d_year__full": p_d_year_full,
    })
    pred_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"Saved predictions to: {OUTPUT_PARQUET_PATH} ({len(pred_df)} rows)")

    report_payload = {
        "title": "18-Token Multi-Horizon History Attention Architecture (Distilled)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_seconds": float(time.monotonic() - total_start),
        "dataset": {
            "n_train": len(train_ix),
            "n_cal": len(cal_ix),
            "n_assess": len(assess_ix),
            "assessment_year": 2026,
            "monthly_blocks": 9,
        },
        "model_architecture": {
            "name": "YearAnchorCommonMapResidual",
            "tokens": 18,
            "token_layout": "[empty_token, year_anchor_365d_token, map_1, ..., map_16]",
            "parameters_per_seed": 6638,
            "parameters_3seeds": 19914,
            "cold_start_invariance": {
                "cold_start_neutral_p": p_cold,
                "max_antisymmetry_error": antisym_err,
            },
        },
        "controls": {
            "c0_reference": {
                "unclipped_log_loss": ll_c0,
                "brier_score": float(np.mean((p_c0_full - y_assess) ** 2)),
                "tail_count": compute_tail_count(y_assess, p_c0_full),
            },
            "c0_d_core_reference": {
                "unclipped_log_loss": ll_d_core,
                "brier_score": float(np.mean((p_d_core_full - y_assess) ** 2)),
                "tail_count": compute_tail_count(y_assess, p_d_core_full),
            },
        },
        "training_diagnostics": [
            {
                "seed": s,
                "final_diagnostics": m.epoch_diagnostics[-1] if hasattr(m, "epoch_diagnostics") and m.epoch_diagnostics else {},
            }
            for s, m in zip(SEEDS, year_distilled_models)
        ],
        "calibration": {
            "slim_year": cal_info_slim_year,
            "d_year_core": cal_info_d_year_core,
            "hybrid_core_cal_optimal_weight": best_hybrid_weight,
            "ons_recalibration_theta_history_length": len(ons_theta_hist),
            "ons_final_theta": ons_theta_hist[-1] if ons_theta_hist else {},
        },
        "assess_evaluation": {
            "c0_slim_year_assembled": metrics_slim_year_full,
            "c0_slim_year_neural": metrics_slim_year_neural,
            "c0_d_year_core_assembled": metrics_d_year_core_full,
            "c0_d_year_core_neural": metrics_d_year_core_neural,
            "c0_d_year_hybrid_core": metrics_d_year_hybrid_core,
            "c0_d_year_assembled_ons": metrics_d_year_full,
        },
        "longshot_upsets_analysis": longshot_analysis,
        "slumping_elite_teams_analysis": slump_analysis,
        "financial_simulations": {
            "polish_production_ledger_cohort": financial_ledger_results,
            "longshot_subset_ledger": longshot_subset_results,
            "pinnacle_t1_market": pinnacle_results,
        },
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved report to: {OUTPUT_JSON_PATH}")

    print("\n" + "=" * 80)
    print("YEAR ANCHOR DISTILLED PIPELINE COMPLETE SUCCESSFULLY")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
