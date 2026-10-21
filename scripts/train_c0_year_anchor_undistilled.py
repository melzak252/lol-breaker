#!/usr/bin/env python3
"""Train and evaluate the 18-Token Multi-Horizon History Attention Architecture.

Adds each player's strictly causal 365-Day Whole-Year Anchor Token
([empty_token, year_anchor_365d_token, map_1, ..., map_16])
to the 16 recent maps in Causal-C0.
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
import torch
import torch.nn as nn

# Set torch threads to 2 so parallel jobs do not contend for CPU cores
torch.set_num_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.metrics import calculate_ece
from src.models import a0_capacity_experiment as cap_exp
from src.models import a0_capacity_runner as runner
from src.models import a0_data_focus as data_focus
from src.models import a0_uncertainty as uncertainty
from src.models import competition_tiers as tiers
from src.models.a0_native_refit import _calculate_forecast
from src.models.a0_target_experiment import _predict_mixture

SEEDS = (20260910, 20260911, 20260912)
EPOCHS = 12
WIDTH = 32
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261017
REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
MODELS_DIR = REPORT_DIR / "models/year_anchor_undistilled"
YEAR_BANK_DIR = REPORT_DIR / "year_anchor_bank"


def resolve_research_root() -> Path:
    candidates = [
        Path("/home/melzak/.codex/.chatgpt-projects/g-p-6944b69bdee08191acfcfeb7a38f372a"),
        PROJECT_ROOT / "data/05_model_input/canonical_benchmark",
    ]
    for c in candidates:
        if (c / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/metadata.parquet").is_file():
            return c
    raise FileNotFoundError("Could not find valid research root containing release_corrected_bank")


def compute_unclipped_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p_safe = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))))


def compute_tail_count(y: np.ndarray, p: np.ndarray, threshold: float = 2.5, eps: float = 1e-15) -> int:
    p_safe = np.clip(p, eps, 1.0 - eps)
    row_ll = -(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))
    return int(np.sum(row_ll >= threshold))


class YearAnchorCommonMapResidual(cap_exp.ParametricCommonMapResidual):
    """18-Token Multi-Horizon History Attention Architecture.

    [empty_token, year_anchor_365d_token, map_1, ..., map_16]
    """

    def __init__(self, width: int = 32):
        super().__init__(width)
        # Year anchor bias initialized to zeros so at step 0 it behaves smoothly
        self.year_token_bias = nn.Parameter(torch.zeros(1, 1, width))
        expected_params = (2 * width**2 + 142 * width + 13) + width  # 6605 + 32 = 6637
        if sum(p.numel() for p in self.parameters()) != expected_params:
            raise AssertionError(f"Expected {expected_params} params, got {sum(p.numel() for p in self.parameters())}")

    def _representation(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.player(x["players"])
        n = h.shape[0]
        # x['mask'] is (B, 2, 5, 17) -> reshape to (B*10, 17)
        mask = x["mask"].reshape(-1, 17).bool()
        year_mask = mask[:, 0:1]  # (B*10, 1)
        recent_mask = mask[:, 1:17]  # (B*10, 16)

        # x['history'] is (B, 2, 5, 17, 33) -> reshape to (B*10, 17, 33)
        values = x["history"].reshape(-1, 17, 33)
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        year_val = values[:, 0:1, :]  # (B*10, 1, 33)
        recent_val = values[:, 1:17, :]  # (B*10, 16, 33)

        # 1-year anchor token passed through history_encoder + year_token_bias
        year_encoded = self.history_encoder(year_val) + self.year_token_bias
        recent_encoded = self.history_encoder(recent_val)

        empty_tok = self.empty.expand(len(values), -1, -1)  # (B*10, 1, width)
        # Concatenate [empty, year_encoded, recent_encoded] along dim 1 -> 18 tokens
        encoded_18 = torch.cat([empty_tok, year_encoded, recent_encoded], dim=1)  # (B*10, 18, width)

        valid_empty = torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device)
        valid_18 = torch.cat([valid_empty, year_mask, recent_mask], dim=1)  # (B*10, 18)

        # Masked softmax attention
        weight = self.attention(encoded_18).squeeze(-1).masked_fill(~valid_18, -torch.inf).softmax(1)
        out = (encoded_18 * weight[..., None]).sum(1)
        h = h + out.reshape(n, 2, 5, self.width)
        return torch.tanh(h)


class YearAnchorModel(cap_exp.ParametricNonlinearHistoryModel):
    """Wrapper matching C0 model interface for YearAnchorCommonMapResidual."""

    def __init__(
        self, baseline: Any, normalizer: Any, width: int = 32, seed: int = 20260910,
        epochs: int = 12, coefficient: float = 0.1,
    ):
        if epochs < 1 or not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("positive epochs and nonnegative auxiliary coefficient required")
        torch.manual_seed(seed)
        self.network = YearAnchorCommonMapResidual(width)
        self.baseline, self.normalizer = baseline, normalizer
        self.width, self.seed, self.epochs = width, seed, epochs
        self.coefficient, self.kind = float(coefficient), "attention"
        self.eta = nn.Parameter(torch.tensor(math.log(1 / 9), dtype=torch.double))
        self.main_objective = "terminal"


def verify_architectural_invariants(shared: dict[str, Any]) -> dict[str, Any]:
    """Strictly verify bitwise initialization match, cold-start neutrality, and antisymmetry."""
    print("Verifying architectural invariants and initialization draws...")
    torch.manual_seed(20260910)
    c0_ref = cap_exp.ParametricCommonMapResidual(WIDTH)
    torch.manual_seed(20260910)
    year_inst = YearAnchorCommonMapResidual(WIDTH)

    # 1. Parameter counts
    c0_params = sum(p.numel() for p in c0_ref.parameters()) + 1  # 6605 + 1 (eta) = 6606
    year_params = sum(p.numel() for p in year_inst.parameters()) + 1  # 6637 + 1 (eta) = 6638
    assert c0_params == 6606, f"Expected 6606 C0 params, got {c0_params}"
    assert year_params == 6638, f"Expected 6638 Year params, got {year_params}"

    # 2. Bitwise equal initialization draws
    d_c0 = dict(c0_ref.named_parameters())
    d_year = dict(year_inst.named_parameters())
    for name, p_c0 in d_c0.items():
        assert name in d_year, f"Missing parameter {name} in YearAnchorCommonMapResidual"
        p_year = d_year[name]
        assert torch.equal(p_c0, p_year), f"Initialization draw mismatch in {name}!"
    assert torch.equal(d_year["year_token_bias"], torch.zeros(1, 1, WIDTH)), "year_token_bias not initialized to 0"
    print("  ✓ Bitwise initialization draws match ParametricCommonMapResidual(32) exactly.")

    # 3. Cold-Start Invariance: zero prior history yields p = 0.500000 exactly
    # Set year_token_bias to arbitrary nonzero value to prove invariance holds regardless of bias
    with torch.no_grad():
        year_inst.year_token_bias.add_(0.75)
    B = 8
    cold_inputs = {
        "players": torch.randn(B, 2, 5, 21),
        "context": torch.randn(B, 2, 70),
        "history": torch.randn(B, 2, 5, 17, 33),
        "mask": torch.zeros(B, 2, 5, 17, dtype=torch.bool),
    }
    # Make cold symmetric match (team A == team B)
    cold_sym = {
        "players": torch.cat([cold_inputs["players"][:, :1], cold_inputs["players"][:, :1]], dim=1),
        "context": torch.cat([cold_inputs["context"][:, :1], cold_inputs["context"][:, :1]], dim=1),
        "history": torch.zeros(B, 2, 5, 17, 33),
        "mask": torch.zeros(B, 2, 5, 17, dtype=torch.bool),
    }
    z_cold_sym = year_inst(cold_sym)
    p_cold_sym = expit(z_cold_sym.detach().double().numpy())
    max_cold_dev = float(np.max(np.abs(p_cold_sym - 0.5)))
    assert max_cold_dev < 1e-12, f"Cold-start neutrality violated: max dev {max_cold_dev}"
    print(f"  ✓ Cold-start neutrality confirmed: max |p - 0.5| = {max_cold_dev:.2e}")

    # 4. Side-swap antisymmetry on random inputs: z(A, B) + z(B, A) == 0.0
    rand_inputs = {
        "players": torch.randn(B, 2, 5, 21),
        "context": torch.randn(B, 2, 70),
        "history": torch.randn(B, 2, 5, 17, 33),
        "mask": torch.randint(0, 2, (B, 2, 5, 17)).bool(),
    }
    z_orig = year_inst(rand_inputs)
    rand_swapped = {
        "players": rand_inputs["players"][:, [1, 0]],
        "context": rand_inputs["context"][:, [1, 0]],
        "history": rand_inputs["history"][:, [1, 0]],
        "mask": rand_inputs["mask"][:, [1, 0]],
    }
    z_swap = year_inst(rand_swapped)
    max_z_antisym = float((z_orig + z_swap).abs().max().item())
    p_orig = expit(z_orig.detach().double().numpy())
    p_swap = expit(z_swap.detach().double().numpy())
    max_p_antisym = float(np.max(np.abs(p_orig + p_swap - 1.0)))
    assert max_z_antisym < 1e-7, f"Logit antisymmetry violated: {max_z_antisym}"
    assert max_p_antisym < 1e-7, f"Probability antisymmetry violated: {max_p_antisym}"
    print(f"  ✓ Side-swap antisymmetry confirmed: max |z_A + z_B| = {max_z_antisym:.2e}, max |p_A + p_B - 1| = {max_p_antisym:.2e}")

    return {
        "c0_parameters_per_seed": c0_params,
        "year_anchor_parameters_per_seed": year_params,
        "parameter_delta": year_params - c0_params,
        "cold_start_neutrality_max_dev": max_cold_dev,
        "side_swap_logit_antisymmetry_max_dev": max_z_antisym,
        "side_swap_prob_antisymmetry_max_dev": max_p_antisym,
    }


def load_or_train_year_model(
    seed: int, epochs: int, ckpt_dir: Path,
    data_raw: dict[str, Any], data_hist: dict[str, Any], train_ix: np.ndarray, bo_index_train: np.ndarray,
    data_all: dict[str, Any], shared: dict[str, Any],
) -> YearAnchorModel:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_epoch{epochs}.pt"
    if ckpt_path.is_file():
        print(f"Loading existing checkpoint: {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inst = YearAnchorModel(
            shared["baseline"], shared["normalizer"], width=WIDTH, seed=seed, epochs=epochs, coefficient=0.1
        )
        inst.network.load_state_dict(state["network_state_dict"])
        inst.eta = nn.Parameter(state["eta"])
        inst.target_scaler = state["target_scaler"]
        inst.epochs = epochs
        inst.network.eval()
        return inst

    print(f"Training YearAnchorModel: seed={seed}, epochs={epochs}...")
    t0 = time.monotonic()
    model_inst = YearAnchorModel(
        shared["baseline"], shared["normalizer"], width=WIDTH, seed=seed, epochs=epochs, coefficient=0.1
    )
    snapshots = model_inst.fit_trajectory(
        data_raw, data_hist, None, train_ix, bo_index_train,
        data_all["wins_a"], data_all["wins_b"], data_all["targets"], data_all["target_mask"],
        checkpoint_dir=ckpt_dir, checkpoint_epochs=(epochs,), device="cpu",
        teacher_probabilities=None, distillation_weight=0.0,
        progress=lambda ep, loss, aux, rec: (
            print(f"  [seed {seed}] epoch {ep}/{epochs}: loss={loss:.5f} aux={aux:.5f} ({time.monotonic()-t0:.1f}s)")
            if ep in (1, epochs // 2, epochs) else None
        ),
    )
    print(f"Finished training seed {seed} in {time.monotonic() - t0:.1f}s")
    return snapshots[epochs]


def evaluate_probability_stream(
    y: np.ndarray, p: np.ndarray, dates: np.ndarray, frame: pd.DataFrame,
    ref_c0_p: np.ndarray, n_bootstraps: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    ll = compute_unclipped_log_loss(y, p)
    brier = float(np.mean((p - y) ** 2))
    ece = float(calculate_ece(y, p, n_bins=10))
    tails = compute_tail_count(y, p, threshold=2.5)

    # Slice evaluation
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
                "brier_score": float(np.mean((p[mask] - y[mask]) ** 2)),
                "tail_count": compute_tail_count(y[mask], p[mask], threshold=2.5),
            }

    # Paired Monthly-Block Bootstrap vs C0
    p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
    c0_safe = np.clip(ref_c0_p, 1e-15, 1.0 - 1e-15)

    row_ll = -(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))
    c0_ll = -(y * np.log(c0_safe) + (1.0 - y) * np.log(1.0 - c0_safe))

    delta_c0 = row_ll - c0_ll

    boot_df = pd.DataFrame({
        "date": dates,
        "delta_c0": delta_c0,
    })

    obs_c0, low_c0, high_c0, samples_c0 = monthly_block_bootstrap_delta(
        boot_df, "delta_c0", date_column="date", n_bootstraps=n_bootstraps, random_seed=seed
    )
    p_nonneg_c0 = float(np.mean(samples_c0 >= 0.0))

    return {
        "n": int(len(y)),
        "unclipped_log_loss": ll,
        "brier_score": brier,
        "ece_10_bins": ece,
        "tail_count_ll_ge_2_5": tails,
        "slices": slice_metrics,
        "bootstrap_vs_c0": {
            "observed_delta": float(obs_c0),
            "ci95_lower": float(low_c0),
            "ci95_upper": float(high_c0),
            "p_nonnegative": p_nonneg_c0,
            "draws": n_bootstraps,
            "seed": seed,
        },
    }


def analyze_longshots_and_divergence(
    y: np.ndarray,
    p_c0: np.ndarray,
    p_year: np.ndarray,
    p_year_neural: np.ndarray,
    p_blend: np.ndarray,
    year_hist_assess: np.ndarray,
    year_mask_assess: np.ndarray,
    recent_hist_assess: np.ndarray,
    recent_mask_assess: np.ndarray,
) -> dict[str, Any]:
    """Longshot & Slumping-Favorite Cohorts + Form-Divergence breakdown."""
    streams = {
        "C0": p_c0,
        "C0-Year": p_year,
        "C0-Year-Neural": p_year_neural,
        "C0-Year-Blend": p_blend,
    }

    results: dict[str, Any] = {
        "longshot_cohorts": {},
        "form_divergence_cohorts": {},
    }

    # 1. Longshot cohorts (underdog threshold <= 0.30 and <= 0.20)
    for threshold in (0.30, 0.20):
        cohort_key = f"underdog_p_le_{int(threshold*100)}"
        min_p = np.minimum(p_c0, 1.0 - p_c0)
        cohort_mask = min_p <= threshold

        # Favorite side in C0: team A if p_c0 > 0.5, else team B
        # Underdog wins if C0's favorite lost: (p_c0 > 0.5 and y == 0) or (p_c0 <= 0.5 and y == 1)
        underdog_won = cohort_mask & ((p_c0 > 0.5) != (y == 1))
        n_cohort = int(cohort_mask.sum())
        n_upsets = int(underdog_won.sum())

        cohort_data: dict[str, Any] = {
            "threshold": threshold,
            "n_matches": n_cohort,
            "n_upsets": n_upsets,
            "upset_rate": float(n_upsets / n_cohort) if n_cohort > 0 else 0.0,
            "models": {},
        }

        for model_name, p_stream in streams.items():
            ll = compute_unclipped_log_loss(y[cohort_mask], p_stream[cohort_mask])
            brier = float(np.mean((p_stream[cohort_mask] - y[cohort_mask]) ** 2))

            # Upset detection accuracy: when underdog won, did the model pick the underdog (i.e. correct side)?
            upset_pred_correct = ((p_stream > 0.5) == (y == 1))[underdog_won]
            acc_upset = float(np.mean(upset_pred_correct)) if n_upsets > 0 else 0.0

            # Mean probability assigned to the winning underdog
            # If y == 1, underdog was team A, prob is p_stream. If y == 0, underdog was team B, prob is 1 - p_stream.
            p_underdog = np.where(y == 1, p_stream, 1.0 - p_stream)
            mean_p_upset = float(np.mean(p_underdog[underdog_won])) if n_upsets > 0 else 0.0

            # Upset LogLoss & Brier strictly on the upset matches
            upset_ll = compute_unclipped_log_loss(y[underdog_won], p_stream[underdog_won]) if n_upsets > 0 else 0.0
            upset_brier = float(np.mean((p_stream[underdog_won] - y[underdog_won]) ** 2)) if n_upsets > 0 else 0.0

            cohort_data["models"][model_name] = {
                "cohort_log_loss": ll,
                "cohort_brier": brier,
                "upset_detection_accuracy": acc_upset,
                "upset_mean_prob_assigned": mean_p_upset,
                "upset_log_loss": upset_ll,
                "upset_brier": upset_brier,
            }

        results["longshot_cohorts"][cohort_key] = cohort_data

    # 2. Form-Divergence Cohorts
    # Feature 24: win rate
    # Feature 7: gd@15, Feature 19: gd@15 presence indicator
    ym = year_mask_assess.astype(bool)  # (N, 2, 5, 1)
    rm = recent_mask_assess.astype(bool)  # (N, 2, 5, 16)

    # 365-day win rate (feature 24)
    y_wr = year_hist_assess[..., 24]  # (N, 2, 5, 1)
    ym_count = ym.sum(axis=(2, 3))  # (N, 2)
    y_team_wr = (y_wr * ym).sum(axis=(2, 3)) / np.maximum(ym_count, 1)  # (N, 2)
    has_y_wr = (ym_count[:, 0] >= 3) & (ym_count[:, 1] >= 3)

    # 16-map recent win rate (feature 24)
    r_wr = recent_hist_assess[..., 24]  # (N, 2, 5, 16)
    rm_count = rm.sum(axis=(2, 3))  # (N, 2)
    r_team_wr = (r_wr * rm).sum(axis=(2, 3)) / np.maximum(rm_count, 1)  # (N, 2)
    has_r_wr = (rm_count[:, 0] >= 5) & (rm_count[:, 1] >= 5)

    delta_y_wr = y_team_wr[:, 0] - y_team_wr[:, 1]
    delta_r_wr = r_team_wr[:, 0] - r_team_wr[:, 1]
    disagree_wr = has_y_wr & has_r_wr & ((delta_y_wr * delta_r_wr) < 0)

    # GD@15 (feature 7, presence 19)
    y_gd = year_hist_assess[..., 7].astype(np.float32)
    y_pres = year_hist_assess[..., 19] > 0
    y_gd_mask = ym & y_pres
    y_team_gd = (y_gd * y_gd_mask).sum(axis=(2, 3)) / np.maximum(y_gd_mask.sum(axis=(2, 3)), 1)

    r_gd = recent_hist_assess[..., 7].astype(np.float32)
    r_pres = recent_hist_assess[..., 19] > 0
    r_gd_mask = rm & r_pres
    r_team_gd = (r_gd * r_gd_mask).sum(axis=(2, 3)) / np.maximum(r_gd_mask.sum(axis=(2, 3)), 1)

    has_y_gd = (y_gd_mask.sum(axis=(2, 3)) >= 3).all(axis=1)
    has_r_gd = (r_gd_mask.sum(axis=(2, 3)) >= 5).all(axis=1)
    delta_y_gd = y_team_gd[:, 0] - y_team_gd[:, 1]
    delta_r_gd = r_team_gd[:, 0] - r_team_gd[:, 1]
    disagree_gd = has_y_gd & has_r_gd & ((delta_y_gd * delta_r_gd) < 0)

    disagree_either = disagree_wr | disagree_gd

    divergence_masks = {
        "win_rate_divergence": disagree_wr,
        "gd15_divergence": disagree_gd,
        "either_divergence": disagree_either,
    }

    for div_name, mask in divergence_masks.items():
        n_div = int(mask.sum())
        div_data: dict[str, Any] = {
            "n_matches": n_div,
            "models": {},
        }
        if n_div > 0:
            for model_name, p_stream in streams.items():
                ll = compute_unclipped_log_loss(y[mask], p_stream[mask])
                brier = float(np.mean((p_stream[mask] - y[mask]) ** 2))
                acc = float(np.mean((p_stream[mask] > 0.5) == (y[mask] == 1)))
                delta_ll = ll - compute_unclipped_log_loss(y[mask], p_c0[mask])
                delta_brier = brier - float(np.mean((p_c0[mask] - y[mask]) ** 2))
                # Win rate vs C0 on individual matches
                closer_to_y = np.abs(p_stream[mask] - y[mask]) < np.abs(p_c0[mask] - y[mask])
                win_rate_vs_c0 = float(np.mean(closer_to_y))

                div_data["models"][model_name] = {
                    "log_loss": ll,
                    "brier_score": brier,
                    "accuracy": acc,
                    "delta_log_loss_vs_c0": delta_ll,
                    "delta_brier_vs_c0": delta_brier,
                    "win_rate_vs_c0": win_rate_vs_c0,
                }
        results["form_divergence_cohorts"][div_name] = div_data

    return results


def main() -> int:
    start_time = time.monotonic()
    print("=" * 80)
    print("C0 18-TOKEN MULTI-HORIZON HISTORY ATTENTION ARCHITECTURE (UNDISTILLED)")
    print("=" * 80)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Resolve research root and activate archive backend
    research_root = resolve_research_root()
    print(f"Research Root: {research_root}")
    backend, _ = uncertainty._activate_archive_backend(research_root)
    print("Activated archive backend.")

    # 2. Load data and verify canonical partitions
    data = uncertainty._load_training_data(research_root)
    frame = data["frame"]
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

    # 3. Load 365-day whole-year anchor bank
    year_hist_path = YEAR_BANK_DIR / "year_anchor_history.npy"
    year_mask_path = YEAR_BANK_DIR / "year_anchor_mask.npy"
    print(f"Loading 365-day bank from {YEAR_BANK_DIR}...")
    year_anchor_hist = np.load(year_hist_path)
    year_anchor_mask = np.load(year_mask_path)
    print(f"Loaded year_anchor_hist: {year_anchor_hist.shape}, dtype={year_anchor_hist.dtype}")
    print(f"Loaded year_anchor_mask: {year_anchor_mask.shape}, dtype={year_anchor_mask.dtype}")

    # Concatenate along axis 3 into a 17-map array (N, 2, 5, 17, 33)
    combined_hist = np.concatenate([year_anchor_hist, data["history"]["history"]], axis=3)
    combined_mask = np.concatenate([year_anchor_mask, data["history"]["mask"]], axis=3)
    print(f"Combined history shape: {combined_hist.shape}")
    print(f"Combined mask shape: {combined_mask.shape}")

    data_year = {
        "frame": data["frame"],
        "raw": data["raw"],
        "history": {
            "history": combined_hist,
            "mask": combined_mask,
            "champions": data["history"]["champions"],
        },
        "wins_a": data["wins_a"],
        "wins_b": data["wins_b"],
        "targets": data["targets"],
        "target_mask": data["target_mask"],
    }

    # 4. Load shared experts and reference calibration
    shared = joblib.load(PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/shared/model.joblib")
    ref_cal = joblib.load(PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/calibration/model.joblib")
    c0_control_dir = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control"
    c0_models = [joblib.load(c0_control_dir / f"seed_{s}/model.joblib") for s in SEEDS]
    print("Loaded shared experts, reference calibration, and C0 control models.")

    # Verify architectural invariants & parameter counts
    invariance_report = verify_architectural_invariants(shared)

    # 5. Baseline C0 Control Verification on CAL and ASSESS
    assess_frame = frame.iloc[assess_ix].reset_index(drop=True)
    y_assess = assess_frame["y"].to_numpy(dtype=int)
    dates_assess = assess_frame["date"].to_numpy(dtype=str)
    bo_all = frame.best_of.to_numpy(dtype=np.int64)
    wins_a, wins_b = data["wins_a"], data["wins_b"]

    print("Computing C0 baseline forecasts on CAL and ASSESS...")
    c0_cal_z, _, _ = _calculate_forecast(c0_models, data, cal_ix, backend)
    c0_assess_z, _, _ = _calculate_forecast(c0_models, data, assess_ix, backend)

    raw_cal, hist_cal_orig = data_focus._subset_inputs(data, cal_ix)
    raw_assess, hist_assess_orig = data_focus._subset_inputs(data, assess_ix)

    other_cal = data_focus._other_logits(shared, raw_cal, hist_cal_orig, bo_all[cal_ix])
    other_assess = data_focus._other_logits(shared, raw_assess, hist_assess_orig, bo_all[assess_ix])

    assembled_c0_cal, _, _ = _predict_mixture(
        backend, c0_cal_z, other_cal, ref_cal["slopes"],
        raw_cal["gates"], bo_all[cal_ix], ref_cal
    )
    p_c0_cal = assembled_c0_cal[:, 0]

    assembled_c0_assess, _, _ = _predict_mixture(
        backend, c0_assess_z, other_assess, ref_cal["slopes"],
        raw_assess["gates"], bo_all[assess_ix], ref_cal
    )
    p_c0_assess = assembled_c0_assess[:, 0]

    ll_c0_control = compute_unclipped_log_loss(y_assess, p_c0_assess)
    brier_c0_control = float(np.mean((p_c0_assess - y_assess) ** 2))
    print(f"C0 Baseline ASSESS: LL={ll_c0_control:.9f} (expected 0.535448701), Brier={brier_c0_control:.9f}")
    assert abs(ll_c0_control - 0.535448701) < 1e-6, f"C0 control mismatch: {ll_c0_control}"

    # 6. Train 3 seeds of YearAnchorModel
    bo_index_train = (bo_all[train_ix] - 1) // 2
    year_models: list[YearAnchorModel] = []
    for s in SEEDS:
        ckpt_dir = MODELS_DIR / f"seed_{s}"
        m = load_or_train_year_model(
            seed=s, epochs=EPOCHS, ckpt_dir=ckpt_dir,
            data_raw=data["raw"], data_hist=data_year["history"],
            train_ix=train_ix, bo_index_train=bo_index_train,
            data_all=data, shared=shared,
        )
        year_models.append(m)

    # 7. Calibration on CAL and Assembly on ASSESS
    print("Forecasting YearAnchor models on CAL and ASSESS...")
    year_cal_z, year_cal_rho, year_cal_raw_probs = _calculate_forecast(year_models, data_year, cal_ix, backend)
    year_assess_z, year_assess_rho, year_assess_raw_probs = _calculate_forecast(year_models, data_year, assess_ix, backend)

    # Fit Beta calibration (scale, rho) on CAL
    parameters = []
    records = []
    for mode in range(3):
        fitted = backend["beta_terminal"].fit_joint(year_cal_z[:, mode], wins_a[cal_ix], wins_b[cal_ix], "beta")
        parameters.append((fitted["scale"], fitted["rho"]))
        records.append(fitted)

    cal_params = {
        "beta_parameters": np.asarray(parameters, dtype=float),
        "slopes": ref_cal["slopes"],
        "mixtures": ref_cal["mixtures"],
    }
    print(f"Fitted Beta parameters: {cal_params['beta_parameters']}")

    # Assemble C0-Year on CAL and ASSESS
    assembled_year_cal, _, _ = _predict_mixture(
        backend, year_cal_z, other_cal, cal_params["slopes"],
        raw_cal["gates"], bo_all[cal_ix], cal_params
    )
    p_year_cal = assembled_year_cal[:, 0]

    assembled_year_assess, _, _ = _predict_mixture(
        backend, year_assess_z, other_assess, cal_params["slopes"],
        raw_assess["gates"], bo_all[assess_ix], cal_params
    )
    p_year_assess = assembled_year_assess[:, 0]
    p_year_neural_assess = year_assess_raw_probs[:, 0, :].mean(axis=1)

    # 8. Fit CAL-optimal logit blend with frozen C0
    print("Fitting CAL-optimal logit blend between C0-Year and frozen C0...")
    y_cal = frame.iloc[cal_ix]["y"].to_numpy(dtype=int)
    z_year_cal = logit(np.clip(p_year_cal, 1e-12, 1.0 - 1e-12))
    z_c0_cal = logit(np.clip(p_c0_cal, 1e-12, 1.0 - 1e-12))

    def blend_loss(alpha: float) -> float:
        z = alpha * z_year_cal + (1.0 - alpha) * z_c0_cal
        p = expit(z)
        return compute_unclipped_log_loss(y_cal, p)

    res = minimize_scalar(blend_loss, bounds=(0.0, 1.0), method="bounded")
    optimal_alpha = float(res.x)
    blend_cal_ll = float(res.fun)
    print(f"Optimal CAL logit blend alpha={optimal_alpha:.4f}, CAL LogLoss={blend_cal_ll:.6f} (vs C0 {compute_unclipped_log_loss(y_cal, p_c0_cal):.6f})")

    # Apply blend to ASSESS
    z_year_assess = logit(np.clip(p_year_assess, 1e-12, 1.0 - 1e-12))
    z_c0_assess = logit(np.clip(p_c0_assess, 1e-12, 1.0 - 1e-12))
    p_blend_assess = expit(optimal_alpha * z_year_assess + (1.0 - optimal_alpha) * z_c0_assess)

    # 9. Invariance verification on ASSESS predictions
    # Verify side-swap antisymmetry on ASSESS
    # Swap team A and team B in data_assess
    print("Verifying full ASSESS prediction pipeline antisymmetry...")
    raw_assess_swap = {
        k: v[:, [1, 0]] if v.ndim >= 2 and v.shape[1] == 2 else v
        for k, v in raw_assess.items()
    }
    hist_assess_year_swap = {
        "history": combined_hist[assess_ix][:, [1, 0]],
        "mask": combined_mask[assess_ix][:, [1, 0]],
        "champions": data["history"]["champions"][assess_ix][:, [1, 0]],
    }
    hist_assess_orig_swap = {
        "history": data["history"]["history"][assess_ix][:, [1, 0]],
        "mask": data["history"]["mask"][assess_ix][:, [1, 0]],
        "champions": data["history"]["champions"][assess_ix][:, [1, 0]],
    }
    data_assess_swap = {
        "frame": assess_frame,
        "raw": raw_assess_swap,
        "history": hist_assess_year_swap,
    }
    local_assess_ix = np.arange(len(assess_ix), dtype=np.int64)
    assess_z_swap, _, _ = _calculate_forecast(year_models, data_assess_swap, local_assess_ix, backend)
    other_assess_swap = data_focus._other_logits(shared, raw_assess_swap, hist_assess_orig_swap, bo_all[assess_ix])
    assembled_year_assess_swap, _, _ = _predict_mixture(
        backend, assess_z_swap, other_assess_swap, cal_params["slopes"],
        raw_assess_swap["gates"], bo_all[assess_ix], cal_params
    )
    p_year_assess_swap = assembled_year_assess_swap[:, 0]
    assess_antisym_dev = float(np.max(np.abs(p_year_assess + p_year_assess_swap - 1.0)))
    print(f"  ✓ Full ASSESS pipeline antisymmetry confirmed: max |p(A,B) + p(B,A) - 1.0| = {assess_antisym_dev:.2e}")
    assert assess_antisym_dev < 1e-7, f"ASSESS antisymmetry failed: {assess_antisym_dev}"

    # 10. Comprehensive Metric Evaluation on ASSESS
    print("Evaluating probability streams on ASSESS...")
    eval_c0 = evaluate_probability_stream(y_assess, p_c0_assess, dates_assess, assess_frame, p_c0_assess)
    eval_year = evaluate_probability_stream(y_assess, p_year_assess, dates_assess, assess_frame, p_c0_assess)
    eval_year_neural = evaluate_probability_stream(y_assess, p_year_neural_assess, dates_assess, assess_frame, p_c0_assess)
    eval_blend = evaluate_probability_stream(y_assess, p_blend_assess, dates_assess, assess_frame, p_c0_assess)

    print("-" * 80)
    print(f"C0 Baseline:         LL={eval_c0['unclipped_log_loss']:.6f}  Brier={eval_c0['brier_score']:.6f}  ECE={eval_c0['ece_10_bins']:.5f}  Tails={eval_c0['tail_count_ll_ge_2_5']}")
    print(f"C0-Year (Assembled): LL={eval_year['unclipped_log_loss']:.6f}  Brier={eval_year['brier_score']:.6f}  ECE={eval_year['ece_10_bins']:.5f}  Tails={eval_year['tail_count_ll_ge_2_5']}  Delta={eval_year['bootstrap_vs_c0']['observed_delta']:+.6f} [95% CI: {eval_year['bootstrap_vs_c0']['ci95_lower']:+.6f}, {eval_year['bootstrap_vs_c0']['ci95_upper']:+.6f}] (p_nonneg={eval_year['bootstrap_vs_c0']['p_nonnegative']:.3f})")
    print(f"C0-Year (Neural):    LL={eval_year_neural['unclipped_log_loss']:.6f}  Brier={eval_year_neural['brier_score']:.6f}  ECE={eval_year_neural['ece_10_bins']:.5f}  Tails={eval_year_neural['tail_count_ll_ge_2_5']}  Delta={eval_year_neural['bootstrap_vs_c0']['observed_delta']:+.6f} [95% CI: {eval_year_neural['bootstrap_vs_c0']['ci95_lower']:+.6f}, {eval_year_neural['bootstrap_vs_c0']['ci95_upper']:+.6f}] (p_nonneg={eval_year_neural['bootstrap_vs_c0']['p_nonnegative']:.3f})")
    print(f"C0-Year-Blend:       LL={eval_blend['unclipped_log_loss']:.6f}  Brier={eval_blend['brier_score']:.6f}  ECE={eval_blend['ece_10_bins']:.5f}  Tails={eval_blend['tail_count_ll_ge_2_5']}  Delta={eval_blend['bootstrap_vs_c0']['observed_delta']:+.6f} [95% CI: {eval_blend['bootstrap_vs_c0']['ci95_lower']:+.6f}, {eval_blend['bootstrap_vs_c0']['ci95_upper']:+.6f}] (p_nonneg={eval_blend['bootstrap_vs_c0']['p_nonnegative']:.3f})")
    print("-" * 80)

    # 11. Longshot & Form-Divergence Deep Analysis
    print("Running Longshot & Form-Divergence deep analysis...")
    deep_analysis = analyze_longshots_and_divergence(
        y=y_assess,
        p_c0=p_c0_assess,
        p_year=p_year_assess,
        p_year_neural=p_year_neural_assess,
        p_blend=p_blend_assess,
        year_hist_assess=year_anchor_hist[assess_ix],
        year_mask_assess=year_anchor_mask[assess_ix],
        recent_hist_assess=data["history"]["history"][assess_ix],
        recent_mask_assess=data["history"]["mask"][assess_ix],
    )

    # 12. Save Predictions Parquet
    pred_df = pd.DataFrame({
        "golgg_match_id": assess_frame["golgg_match_id"],
        "date": assess_frame["date"],
        "y": y_assess,
        "best_of": assess_frame["best_of"],
        "tournament_name": assess_frame["tournament_name"],
        "unknown_players": assess_frame["unknown_players"],
        "competition_tier": assess_frame["competition_tier"],
        "p_c0__full": p_c0_assess,
        "p_c0_year__full": p_year_assess,
        "p_c0_year__neural": p_year_neural_assess,
        "p_c0_year_blend__full": p_blend_assess,
    })
    pred_path = REPORT_DIR / "year_anchor_undistilled_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"Saved predictions to {pred_path} ({len(pred_df)} rows)")

    # 13. Assemble Final JSON Report
    total_elapsed = time.monotonic() - start_time
    final_report = {
        "experiment": "c0_18_token_year_anchor_undistilled",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "total_elapsed_seconds": float(total_elapsed),
        "parameters": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "width": WIDTH,
            "tokens_total": 18,
            "anchor_horizon_days": 365,
            "c0_params_per_seed": invariance_report["c0_parameters_per_seed"],
            "year_anchor_params_per_seed": invariance_report["year_anchor_parameters_per_seed"],
            "parameter_delta": invariance_report["parameter_delta"],
        },
        "invariance_verification": {
            **invariance_report,
            "pipeline_assess_antisymmetry_max_dev": assess_antisym_dev,
        },
        "calibration": {
            "beta_parameters": [
                {"scale": float(p[0]), "rho": float(p[1])}
                for p in cal_params["beta_parameters"]
            ],
            "neural_rho_cal_mean": float(year_cal_rho.mean()),
            "neural_rho_assess_mean": float(year_assess_rho.mean()),
            "logit_blend": {
                "alpha_year": optimal_alpha,
                "alpha_c0": float(1.0 - optimal_alpha),
                "cal_log_loss": blend_cal_ll,
            },
        },
        "models": {
            "c0_reference": eval_c0,
            "c0_year_assembled": eval_year,
            "c0_year_neural": eval_year_neural,
            "c0_year_blend": eval_blend,
        },
        "deep_analysis": deep_analysis,
        "verdict": {
            "c0_assess_log_loss": eval_c0["unclipped_log_loss"],
            "c0_year_assess_log_loss": eval_year["unclipped_log_loss"],
            "c0_year_blend_assess_log_loss": eval_blend["unclipped_log_loss"],
            "c0_year_delta_vs_c0": eval_year["bootstrap_vs_c0"]["observed_delta"],
            "c0_year_blend_delta_vs_c0": eval_blend["bootstrap_vs_c0"]["observed_delta"],
        },
    }

    report_path = REPORT_DIR / "year_anchor_undistilled.json"
    report_path.write_text(json.dumps(final_report, indent=2))
    print(f"Saved final report to {report_path}")

    print("=" * 80)
    print(f"COMPLETED EXPERIMENT IN {total_elapsed:.1f}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
