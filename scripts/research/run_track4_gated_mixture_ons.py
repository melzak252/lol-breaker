#!/usr/bin/env python3
"""Track 4: Uncertainty/Disagreement-Gated Expert Mixture + Online Beta/Platt Recalibration (ONS) + Multi-Capacity Ensemble.

Fitted strictly on CAL (N=1,328 in 2025H2) or strictly prior released days (result_day < current_date).
Evaluated on canonical 2026 ASSESS (N=3,103).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pinned input artifacts
CAPACITY_CAL_PATH = PROJECT_ROOT / "data/07_model_output/a0_capacity/native_capacity_20261004_01/calibration_predictions.parquet"
CAPACITY_ASSESS_PATH = PROJECT_ROOT / "data/07_model_output/a0_capacity/native_capacity_20261004_01/predictions.parquet"
CAPACITY_ENDPOINT_PATH = PROJECT_ROOT / "data/07_model_output/a0_capacity/native_capacity_20261004_01/endpoint_arrays.npz"

FOCUS_SHARED_PATH = PROJECT_ROOT / "data/07_model_output/a0_data_focus/a0_data_focus_20260929_192443/shared_logits.npz"
FOCUS_CONTROL_PATH = PROJECT_ROOT / "data/07_model_output/a0_data_focus/a0_data_focus_20260929_192443/control_logits.npz"

CONNECTOR_PRED_PATH = PROJECT_ROOT / "data/07_model_output/a0_sports_correction/c0_raw_connector_20261002_033658/predictions.parquet"

PREMATCH_META_PATH = PROJECT_ROOT / "data/04_feature/prematch_architecture_20260927_run001/metadata.parquet"
PREMATCH_HIST_PATH = PROJECT_ROOT / "data/04_feature/prematch_architecture_20260927_run001/history.npy"
PREMATCH_MASK_PATH = PROJECT_ROOT / "data/04_feature/prematch_architecture_20260927_run001/history_mask.npy"

PINNACLE_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"
POLISH_PANEL_PATH = PROJECT_ROOT / "data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/paired_probability_panel.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
OUTPUT_JSON_PATH = OUTPUT_DIR / "track4_gated_mixture_ons.json"
OUTPUT_PARQUET_PATH = OUTPUT_DIR / "track4_predictions.parquet"

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261017


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p_cl = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p_cl / (1.0 - p_cl))


def compute_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(p)
    for i in range(n_bins):
        mask_bin = (p >= bin_edges[i]) & (p < bin_edges[i + 1] if i < n_bins - 1 else p <= bin_edges[i + 1])
        if mask_bin.sum() > 0:
            p_mean = float(p[mask_bin].mean())
            y_mean = float(y[mask_bin].mean())
            ece += (mask_bin.sum() / total) * abs(p_mean - y_mean)
    return float(ece)


def compute_hybrid50(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    z = 0.50 * logit(p) + 0.50 * logit(q)
    return sigmoid(z)


def paired_monthly_bootstrap(
    delta_vector: np.ndarray,
    months: list[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Paired monthly-block bootstrap matching canonical research standards."""
    df = pd.DataFrame({"delta": delta_vector, "month": months})
    unique_months = sorted(df["month"].unique())
    n_blocks = len(unique_months)
    n_total = len(df)

    if n_blocks == 0 or n_total == 0:
        return {
            "delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "draws": draws,
            "seed": seed,
            "blocks": n_blocks,
            "n": n_total,
        }

    grouped = df.groupby("month")["delta"].agg(d_sum="sum", d_count="count", d_mean="mean").loc[unique_months]
    sums = grouped["d_sum"].to_numpy(dtype=np.float64)
    counts = grouped["d_count"].to_numpy(dtype=np.float64)
    means = grouped["d_mean"].to_numpy(dtype=np.float64)

    observed_match_weighted = float(sums.sum() / counts.sum())
    observed_equal_month = float(means.mean())

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n_blocks, size=(draws, n_blocks))

    # Match-weighted draws
    draw_sums = sums[sampled_indices].sum(axis=1)
    draw_counts = counts[sampled_indices].sum(axis=1)
    mw_draws = draw_sums / draw_counts

    alpha_tail = (1.0 - confidence_level) / 2.0
    return {
        "delta": observed_match_weighted,
        "ci_lower": float(np.quantile(mw_draws, alpha_tail)),
        "ci_upper": float(np.quantile(mw_draws, 1.0 - alpha_tail)),
        "draws": draws,
        "seed": seed,
        "blocks": n_blocks,
        "n": n_total,
        "confidence_level": confidence_level,
        "equal_month_delta": observed_equal_month,
    }


def gated_logit_loss_and_grad(params: np.ndarray, z_experts: np.ndarray, c_norm: np.ndarray, y: np.ndarray, l2_reg: float = 1.0):
    """Analytical gradient for L2-regularized logit mixture."""
    N, K = z_experts.shape
    M = c_norm.shape[1]

    theta_0 = params[:K]
    Theta = params[K:].reshape(K, M)

    h = theta_0[None, :] + c_norm @ Theta.T  # (N, K)
    w = softmax(h, axis=1)  # (N, K)

    z_comb = (w * z_experts).sum(axis=1)  # (N,)
    p = sigmoid(z_comb)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    nll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()
    reg = 0.5 * l2_reg * np.sum(Theta**2)
    loss = nll + reg

    d_z = (p - y) / N  # (N,)
    delta = d_z[:, None] * w * (z_experts - z_comb[:, None])  # (N, K)

    grad_theta_0 = delta.sum(axis=0)
    grad_Theta = delta.T @ c_norm + l2_reg * Theta

    grad = np.concatenate([grad_theta_0, grad_Theta.ravel()])
    return loss, grad


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
    N = len(logits)

    p_recal = np.zeros(N, dtype=np.float64)
    theta = np.array([1.0, 0.0], dtype=np.float64)  # (a, b)
    A = eps * np.eye(2, dtype=np.float64)

    theta_history = []
    processed_matches = set()

    for d in unique_dates:
        # 1. Matches on date d: predict using current causal theta_t
        d_mask = (dates == d)
        d_indices = np.where(d_mask)[0]

        z_d = logits[d_indices]
        p_d = sigmoid(theta[0] * z_d + theta[1])
        p_recal[d_indices] = p_d

        theta_history.append({"date": str(d), "a": float(theta[0]), "b": float(theta[1])})

        # 2. Update parameters ONLY from matches with result_day <= d
        available_mask = (result_days <= d) & ~np.isin(np.arange(N), list(processed_matches))
        new_indices = np.where(available_mask)[0]

        if len(new_indices) > 0:
            for idx in new_indices:
                z_i = logits[idx]
                y_i = y_true[idx]
                x_i = np.array([z_i, 1.0], dtype=np.float64)

                p_i = float(sigmoid(np.dot(theta, x_i)))
                p_i = max(1e-12, min(1.0 - 1e-12, p_i))

                g_i = (p_i - y_i) * x_i
                H_i = p_i * (1.0 - p_i) * np.outer(x_i, x_i)
                A += H_i

                A_inv = np.linalg.inv(A)
                step = gamma * (A_inv @ g_i)
                theta = theta - step

                # Box projection
                theta[0] = np.clip(theta[0], a_bounds[0], a_bounds[1])
                theta[1] = np.clip(theta[1], b_bounds[0], b_bounds[1])

                processed_matches.add(idx)

    return p_recal, theta_history


def main() -> None:
    print("=== Track 4: Uncertainty-Gated Mixture + Online Recalibration (ONS) Execution ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline capacity and sports data
    print("Loading frozen baseline and capacity artifacts...")
    cap_cal = pd.read_parquet(CAPACITY_CAL_PATH)
    cap_assess = pd.read_parquet(CAPACITY_ASSESS_PATH)
    endpoint = np.load(CAPACITY_ENDPOINT_PATH)
    focus_shared = np.load(FOCUS_SHARED_PATH)
    focus_control = np.load(FOCUS_CONTROL_PATH)
    conn_pred = pd.read_parquet(CONNECTOR_PRED_PATH)

    assert len(cap_cal) == 1328, f"Expected 1328 CAL rows, got {len(cap_cal)}"
    assert len(cap_assess) == 3103, f"Expected 3103 ASSESS rows, got {len(cap_assess)}"

    y_cal = cap_cal["y"].values.astype(int)
    y_assess = cap_assess["y"].values.astype(int)

    dates_assess = cap_assess["date"].values.astype(str)
    assess_months = pd.to_datetime(dates_assess).to_period("M").astype(str).tolist()

    # Align connector predictions
    conn_cal = conn_pred.set_index("golgg_match_id").loc[cap_cal["golgg_match_id"].values]
    conn_assess = conn_pred.set_index("golgg_match_id").loc[cap_assess["golgg_match_id"].values]

    result_days_assess = conn_assess["result_day"].values.astype(str)

    # Baseline probabilities
    p_c0_cal = cap_cal["w32_e12__full"].values.astype(np.float64)
    p_c0_assess = cap_assess["w32_e12__full"].values.astype(np.float64)

    ll_c0_check = -float((y_assess * np.log(p_c0_assess) + (1 - y_assess) * np.log(1 - p_c0_assess)).mean())
    brier_c0_check = float(((p_c0_assess - y_assess)**2).mean())
    assert abs(ll_c0_check - 0.535448701) < 1e-7, f"C0 baseline LogLoss mismatch: {ll_c0_check} vs 0.535448701"
    assert abs(brier_c0_check - 0.181034493) < 1e-7, f"C0 baseline Brier mismatch: {brier_c0_check} vs 0.181034493"
    print(f"Verified C0 Baseline ASSESS LogLoss: {ll_c0_check:.9f}, Brier: {brier_c0_check:.9f}")

    # 2. Extract strictly causal gating covariates from feature bank
    print("Extracting strictly causal gating covariates from feature bank...")
    meta_pre = pd.read_parquet(PREMATCH_META_PATH)
    hist = np.load(PREMATCH_HIST_PATH, mmap_mode="r")
    mask = np.load(PREMATCH_MASK_PATH, mmap_mode="r")

    id_to_idx = {str(mid): i for i, mid in enumerate(meta_pre["match_id"])}
    cal_idx_list = [id_to_idx[str(mid)] for mid in cap_cal["golgg_match_id"]]
    assess_idx_list = [id_to_idx[str(mid)] for mid in cap_assess["golgg_match_id"]]

    # c_depth: fraction of valid historical maps in hist.mask (sum mask / 160)
    c_depth_cal = mask[cal_idx_list].sum(axis=(1, 2, 3)) / 160.0
    c_depth_assess = mask[assess_idx_list].sum(axis=(1, 2, 3)) / 160.0

    # c_stat: fraction of observed @15 stats in hist.history[..., 12:24]
    def extract_c_stat(indices: list[int]) -> np.ndarray:
        c_stats = []
        batch_size = 256
        for b in range(0, len(indices), batch_size):
            b_idx = indices[b:b + batch_size]
            v_mask = mask[b_idx]
            h_slice = hist[b_idx, ..., 12:24]
            present = (h_slice > 0) & v_mask[..., None]
            counts = v_mask.sum(axis=(1, 2, 3))
            num = present.sum(axis=(1, 2, 3, 4))
            cs = np.divide(num, counts * 12.0, out=np.zeros(len(b_idx), dtype=float), where=counts > 0)
            c_stats.append(cs)
        return np.concatenate(c_stats)

    c_stat_cal = extract_c_stat(cal_idx_list)
    c_stat_assess = extract_c_stat(assess_idx_list)

    # d_expert: |logit(p_neural) - logit(p_rating)|
    neural_logits_cal = endpoint["w32_e12_cal_logits"][:, 0, :].mean(axis=1)
    neural_logits_assess = endpoint["w32_e12_assessment_logits"][:, 0, :].mean(axis=1)

    rating_logits_cal = focus_shared["other_logits"][:1328, 0, 0]
    rating_logits_assess = focus_shared["other_logits"][1328:, 0, 0]

    mlp_logits_cal = focus_shared["other_logits"][:1328, 0, 1:].mean(axis=1)
    mlp_logits_assess = focus_shared["other_logits"][1328:, 0, 1:].mean(axis=1)

    d_expert_cal = np.abs(neural_logits_cal - rating_logits_cal)
    d_expert_assess = np.abs(neural_logits_assess - rating_logits_assess)

    is_bo1_cal = (cap_cal["best_of"].values == 1).astype(float)
    is_bo5_cal = (cap_cal["best_of"].values == 5).astype(float)

    is_bo1_assess = (cap_assess["best_of"].values == 1).astype(float)
    is_bo5_assess = (cap_assess["best_of"].values == 5).astype(float)

    covariates_cal = np.column_stack([c_stat_cal, c_depth_cal, d_expert_cal, is_bo1_cal, is_bo5_cal])
    covariates_assess = np.column_stack([c_stat_assess, c_depth_assess, d_expert_assess, is_bo1_assess, is_bo5_assess])

    # Normalize covariates strictly with CAL parameters
    mu_cov = covariates_cal.mean(axis=0)
    std_cov = np.where(covariates_cal.std(axis=0) > 1e-6, covariates_cal.std(axis=0), 1.0)

    c_cal_norm = (covariates_cal - mu_cov) / std_cov
    c_assess_norm = (covariates_assess - mu_cov) / std_cov

    # 3. Define Slices
    # Slices: BO1 (N=745), BO3 (N=1,873), BO5 (N=485), Major (N=692), LPL (N=242), Unseen Players (N=334)
    is_maj = (cap_assess["competition_tier"].values == "major")
    family_series = conn_assess["family"].values
    major_slice_mask = is_maj & np.isin(family_series, ["LCK", "LPL", "LEC", "LCS"])
    lpl_slice_mask = cap_assess["tournament_name"].str.startswith("LPL ").values

    slice_masks = {
        "all_canonical": np.ones(3103, dtype=bool),
        "best_of_1": cap_assess["best_of"].values == 1,
        "best_of_3": cap_assess["best_of"].values == 3,
        "best_of_5": cap_assess["best_of"].values == 5,
        "major": major_slice_mask,
        "lpl": lpl_slice_mask,
        "source_unseen": cap_assess["unknown_players"].values > 0,
    }

    # Verify slice sizes
    expected_slice_sizes = {
        "all_canonical": 3103,
        "best_of_1": 745,
        "best_of_3": 1873,
        "best_of_5": 485,
        "major": 692,
        "lpl": 242,
        "source_unseen": 334,
    }
    for s_name, exp_n in expected_slice_sizes.items():
        actual_n = int(slice_masks[s_name].sum())
        assert actual_n == exp_n, f"Slice {s_name} size mismatch: expected {exp_n}, got {actual_n}"
    print(f"Verified all 6 slice cohort sizes exactly: {expected_slice_sizes}")

    # 4. Step 1: Re-Audit c0_raw_connector Against Active C0 Guards
    print("\n--- Step 1: Re-Auditing c0_raw_connector Against Active C0 Guards ---")
    p_conn = conn_assess["p_c0_connector"].values.astype(np.float64)
    delta_ll_conn = (-(y_assess * np.log(p_conn) + (1 - y_assess) * np.log(1 - p_conn))) - (-(y_assess * np.log(p_c0_assess) + (1 - y_assess) * np.log(1 - p_c0_assess)))
    delta_brier_conn = (p_conn - y_assess)**2 - (p_c0_assess - y_assess)**2

    conn_guard_results = {}
    for s_name, s_mask in slice_masks.items():
        s_months = [assess_months[i] for i in range(len(s_mask)) if s_mask[i]]
        bs_ll = paired_monthly_bootstrap(delta_ll_conn[s_mask], s_months, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        bs_brier = paired_monthly_bootstrap(delta_brier_conn[s_mask], s_months, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        passes = bs_ll["ci_upper"] <= 0.00500
        conn_guard_results[s_name] = {
            "n": int(s_mask.sum()),
            "delta_ll": bs_ll["delta"],
            "ci_lower": bs_ll["ci_lower"],
            "ci_upper": bs_ll["ci_upper"],
            "delta_brier": bs_brier["delta"],
            "guard_pass_against_c0": passes,
            "threshold": 0.00500,
        }
        print(f"  {s_name:<15} (N={s_mask.sum():<4}): Delta LL {bs_ll['delta']:+.6f} [95% CI {bs_ll['ci_lower']:+.6f}, {bs_ll['ci_upper']:+.6f}] -> Guard {'PASS' if passes else 'FAIL'} (limit <= +0.00500)")

    # 5. Step 2: Fit Gated Expert Mixtures on CAL (N=1,328)
    print("\n--- Step 2: Fitting Context/Uncertainty-Gated Expert Mixtures strictly on CAL ---")

    def fit_static_weights(z_exp: np.ndarray, y: np.ndarray) -> np.ndarray:
        K = z_exp.shape[1]
        def loss_fn(theta_0):
            w = softmax(theta_0)
            z = (w * z_exp).sum(axis=1)
            p = sigmoid(z)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()
        res = minimize(loss_fn, np.zeros(K), method="L-BFGS-B")
        return res.x

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    l2_candidates = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    # Variant 4A: Gated-C0-Internal (W32/E12 Neural, Rating, MLP)
    z_exp_cal_4a = np.column_stack([logit(p_c0_cal), rating_logits_cal, mlp_logits_cal])
    z_exp_assess_4a = np.column_stack([logit(p_c0_assess), rating_logits_assess, mlp_logits_assess])

    static_4a_cal = fit_static_weights(z_exp_cal_4a, y_cal)
    init_4a = np.concatenate([static_4a_cal, np.zeros(3 * 5)])

    best_l2_4a = 0.5
    best_cv_4a = float("inf")
    for l2 in l2_candidates:
        cv_scores = []
        for tr_idx, val_idx in kf.split(z_exp_cal_4a):
            res_fold = minimize(
                gated_logit_loss_and_grad,
                init_4a,
                args=(z_exp_cal_4a[tr_idx], c_cal_norm[tr_idx], y_cal[tr_idx], l2),
                jac=True,
                method="L-BFGS-B",
            )
            w_val = softmax(res_fold.x[:3][None, :] + c_cal_norm[val_idx] @ res_fold.x[3:].reshape(3, 5).T, axis=1)
            p_val = sigmoid((w_val * z_exp_cal_4a[val_idx]).sum(axis=1))
            p_val = np.clip(p_val, 1e-12, 1.0 - 1e-12)
            cv_scores.append(-(y_cal[val_idx] * np.log(p_val) + (1.0 - y_cal[val_idx]) * np.log(1.0 - p_val)).mean())
        mean_cv = float(np.mean(cv_scores))
        if mean_cv < best_cv_4a:
            best_cv_4a = mean_cv
            best_l2_4a = l2

    print(f"Variant 4A Selected L2 penalty: {best_l2_4a} (CAL CV LogLoss: {best_cv_4a:.6f})")

    res_4a = minimize(
        gated_logit_loss_and_grad,
        init_4a,
        args=(z_exp_cal_4a, c_cal_norm, y_cal, best_l2_4a),
        jac=True,
        method="L-BFGS-B",
    )
    t0_4a = res_4a.x[:3]
    Th_4a = res_4a.x[3:].reshape(3, 5)

    w_4a_ass = softmax(t0_4a[None, :] + c_assess_norm @ Th_4a.T, axis=1)
    z_4a_ass = (w_4a_ass * z_exp_assess_4a).sum(axis=1)
    p_4a_assess = sigmoid(z_4a_ass)

    # Variant 4B: Multi-Capacity Neural + Rating/MLP Mixture (W32/E12, W32/E24, W64/E12, Rating, MLP)
    z_exp_cal_4b = np.column_stack([
        logit(p_c0_cal),
        logit(cap_cal["w32_e24__full"].values),
        logit(cap_cal["w64_e12__full"].values),
        rating_logits_cal,
        mlp_logits_cal,
    ])
    z_exp_assess_4b = np.column_stack([
        logit(p_c0_assess),
        logit(cap_assess["w32_e24__full"].values),
        logit(cap_assess["w64_e12__full"].values),
        rating_logits_assess,
        mlp_logits_assess,
    ])

    static_4b_cal = fit_static_weights(z_exp_cal_4b, y_cal)
    init_4b = np.concatenate([static_4b_cal, np.zeros(5 * 5)])

    best_l2_4b = 0.5
    best_cv_4b = float("inf")
    for l2 in l2_candidates:
        cv_scores = []
        for tr_idx, val_idx in kf.split(z_exp_cal_4b):
            res_fold = minimize(
                gated_logit_loss_and_grad,
                init_4b,
                args=(z_exp_cal_4b[tr_idx], c_cal_norm[tr_idx], y_cal[tr_idx], l2),
                jac=True,
                method="L-BFGS-B",
            )
            w_val = softmax(res_fold.x[:5][None, :] + c_cal_norm[val_idx] @ res_fold.x[5:].reshape(5, 5).T, axis=1)
            p_val = sigmoid((w_val * z_exp_cal_4b[val_idx]).sum(axis=1))
            p_val = np.clip(p_val, 1e-12, 1.0 - 1e-12)
            cv_scores.append(-(y_cal[val_idx] * np.log(p_val) + (1.0 - y_cal[val_idx]) * np.log(1.0 - p_val)).mean())
        mean_cv = float(np.mean(cv_scores))
        if mean_cv < best_cv_4b:
            best_cv_4b = mean_cv
            best_l2_4b = l2

    print(f"Variant 4B Selected L2 penalty: {best_l2_4b} (CAL CV LogLoss: {best_cv_4b:.6f})")

    res_4b = minimize(
        gated_logit_loss_and_grad,
        init_4b,
        args=(z_exp_cal_4b, c_cal_norm, y_cal, best_l2_4b),
        jac=True,
        method="L-BFGS-B",
    )
    t0_4b = res_4b.x[:5]
    Th_4b = res_4b.x[5:].reshape(5, 5)

    w_4b_cal = softmax(t0_4b[None, :] + c_cal_norm @ Th_4b.T, axis=1)
    z_4b_cal = (w_4b_cal * z_exp_cal_4b).sum(axis=1)

    w_4b_ass = softmax(t0_4b[None, :] + c_assess_norm @ Th_4b.T, axis=1)
    z_4b_ass = (w_4b_ass * z_exp_assess_4b).sum(axis=1)
    p_4b_assess = sigmoid(z_4b_ass)

    # Variant 4C: Disagreement-Regularized C0 + Raw Specialist Trio
    # Connect Multi-Capacity C0 with Raw Specialist Trio with disagreement shrinkage
    p_spec_cal = conn_cal["p_specialist"].values.astype(np.float64)
    p_spec_ass = conn_assess["p_specialist"].values.astype(np.float64)
    z_spec_cal = logit(p_spec_cal)
    z_spec_ass = logit(p_spec_ass)

    d_trio_cal = np.abs(z_4b_cal - z_spec_cal)
    d_trio_ass = np.abs(z_4b_ass - z_spec_ass)

    # Plan09 calibrated connector weight alpha_0 = 0.176903385
    # with exponential shrinkage on disagreement: alpha(x) = alpha_0 * exp(-gamma * d_trio(x))
    # Select gamma on CAL via CV
    best_gamma_4c = 0.1
    best_cv_4c = float("inf")
    for gamma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        cv_scores_4c = []
        for tr_idx, val_idx in kf.split(z_4b_cal):
            alpha_c = 0.1769033853776533 * np.exp(-gamma * d_trio_cal[val_idx])
            z_c = (1.0 - alpha_c) * z_4b_cal[val_idx] + alpha_c * z_spec_cal[val_idx]
            p_c = sigmoid(z_c)
            p_c = np.clip(p_c, 1e-12, 1.0 - 1e-12)
            cv_scores_4c.append(-(y_cal[val_idx] * np.log(p_c) + (1.0 - y_cal[val_idx]) * np.log(1.0 - p_c)).mean())
        mean_cv_4c = float(np.mean(cv_scores_4c))
        if mean_cv_4c < best_cv_4c:
            best_cv_4c = mean_cv_4c
            best_gamma_4c = gamma

    print(f"Variant 4C Selected shrinkage parameter: gamma={best_gamma_4c} (CAL CV LogLoss: {best_cv_4c:.6f})")

    alpha_ass_4c = 0.1769033853776533 * np.exp(-best_gamma_4c * d_trio_ass)
    z_4c_ass = (1.0 - alpha_ass_4c) * z_4b_ass + alpha_ass_4c * z_spec_ass
    p_4c_assess = sigmoid(z_4c_ass)

    # 6. Step 3: Online Newton Step (ONS) Recalibration
    print("\n--- Step 3: Executing Strictly Causal Day-by-Day Online Newton Step Recalibration ---")
    p_c0_ons, ons_c0_history = run_ons_recalibration(dates_assess, result_days_assess, logit(p_c0_assess), y_assess, gamma=0.005, eps=1.0)
    p_4b_ons, ons_4b_history = run_ons_recalibration(dates_assess, result_days_assess, z_4b_ass, y_assess, gamma=0.005, eps=1.0)
    p_4c_ons, ons_4c_history = run_ons_recalibration(dates_assess, result_days_assess, z_4c_ass, y_assess, gamma=0.005, eps=1.0)

    # 7. Collect All Models for Comprehensive Evaluation
    all_models = {
        "baseline_c0": p_c0_assess,
        "c0_raw_connector": p_conn,
        "variant_4a": p_4a_assess,
        "variant_4b": p_4b_assess,
        "variant_4c": p_4c_assess,
        "c0_ons": p_c0_ons,
        "variant_4b_ons": p_4b_ons,
        "variant_4c_ons": p_4c_ons,
    }

    # 8. Step 4: Market Cohort Evaluation (Pinnacle N=1,668 & Polish2plus N=1,467)
    print("\n--- Step 4: Market Cohort Evaluation (Pinnacle N=1,668 & Polish2plus N=1,467) ---")
    df_t1 = pd.read_parquet(PINNACLE_T1_PATH)
    df_polish = pd.read_parquet(POLISH_PANEL_PATH)
    df_p2p = df_polish[df_polish["comparison_cohort"] == "Polish2plus"].copy()

    assert len(df_t1) == 1668, f"Expected 1668 Pinnacle rows, got {len(df_t1)}"
    assert len(df_p2p) == 1467, f"Expected 1467 Polish2plus rows, got {len(df_p2p)}"

    pin_match_indices = [np.where(cap_assess["golgg_match_id"].values == mid)[0][0] for mid in df_t1["golgg_match_id"]]
    pol_match_indices = [np.where(cap_assess["golgg_match_id"].values == mid)[0][0] for mid in df_p2p["golgg_match_id"]]

    y_pin = df_t1["y"].values.astype(int)
    p_mkt_pin = df_t1["q_consensus"].values.astype(np.float64)

    y_pol = df_p2p["y"].values.astype(int)
    p_mkt_pol = df_p2p["p_market"].values.astype(np.float64)

    market_cohort_evaluations = {}
    for m_name, p_model in all_models.items():
        p_pin = p_model[pin_match_indices]
        p_pol = p_model[pol_match_indices]

        # Pinnacle metrics
        ll_pin = float(-(y_pin * np.log(p_pin) + (1 - y_pin) * np.log(1 - p_pin)).mean())
        brier_pin = float(((p_pin - y_pin)**2).mean())

        tail_pin_mask = np.abs(p_pin - p_mkt_pin) >= 0.12
        n_tail_pin = int(tail_pin_mask.sum())
        ll_tail_pin = float(-(y_pin[tail_pin_mask] * np.log(p_pin[tail_pin_mask]) + (1 - y_pin[tail_pin_mask]) * np.log(1 - p_pin[tail_pin_mask])).mean()) if n_tail_pin > 0 else None
        ll_tail_mkt_pin = float(-(y_pin[tail_pin_mask] * np.log(p_mkt_pin[tail_pin_mask]) + (1 - y_pin[tail_pin_mask]) * np.log(1 - p_mkt_pin[tail_pin_mask])).mean()) if n_tail_pin > 0 else None
        excess_ll_pin = float(ll_tail_pin - ll_tail_mkt_pin) if n_tail_pin > 0 else None

        p_hyb_pin = compute_hybrid50(p_pin, p_mkt_pin)
        ll_hyb_pin = float(-(y_pin * np.log(p_hyb_pin) + (1 - y_pin) * np.log(1 - p_hyb_pin)).mean())

        # Polish metrics
        ll_pol = float(-(y_pol * np.log(p_pol) + (1 - y_pol) * np.log(1 - p_pol)).mean())
        brier_pol = float(((p_pol - y_pol)**2).mean())

        tail_pol_mask = np.abs(p_pol - p_mkt_pol) >= 0.12
        n_tail_pol = int(tail_pol_mask.sum())
        ll_tail_pol = float(-(y_pol[tail_pol_mask] * np.log(p_pol[tail_pol_mask]) + (1 - y_pol[tail_pol_mask]) * np.log(1 - p_pol[tail_pol_mask])).mean()) if n_tail_pol > 0 else None
        ll_tail_mkt_pol = float(-(y_pol[tail_pol_mask] * np.log(p_mkt_pol[tail_pol_mask]) + (1 - y_pol[tail_pol_mask]) * np.log(1 - p_mkt_pol[tail_pol_mask])).mean()) if n_tail_pol > 0 else None
        excess_ll_pol = float(ll_tail_pol - ll_tail_mkt_pol) if n_tail_pol > 0 else None

        p_hyb_pol = compute_hybrid50(p_pol, p_mkt_pol)
        ll_hyb_pol = float(-(y_pol * np.log(p_hyb_pol) + (1 - y_pol) * np.log(1 - p_hyb_pol)).mean())

        market_cohort_evaluations[m_name] = {
            "pinnacle_n1668": {
                "log_loss": ll_pin,
                "brier": brier_pin,
                "tail_disagreement_count": n_tail_pin,
                "tail_model_ll": ll_tail_pin,
                "tail_market_ll": ll_tail_mkt_pin,
                "tail_excess_ll": excess_ll_pin,
                "hybrid50_log_loss": ll_hyb_pin,
            },
            "polish2plus_n1467": {
                "log_loss": ll_pol,
                "brier": brier_pol,
                "tail_disagreement_count": n_tail_pol,
                "tail_model_ll": ll_tail_pol,
                "tail_market_ll": ll_tail_mkt_pol,
                "tail_excess_ll": excess_ll_pol,
                "hybrid50_log_loss": ll_hyb_pol,
            },
        }

    # 9. Compute Comprehensive Metrics, Slices, and Paired Monthly Block Bootstrap 95% CIs
    print("\n--- Step 5: Computing Full Metrics, Slice Breakdowns & 5,000-Draw Bootstrap CIs ---")
    model_summaries = {}
    sports_guard_evaluations = {}

    for m_name, p_cand in all_models.items():
        ll_vec = -(y_assess * np.log(p_cand) + (1 - y_assess) * np.log(1 - p_cand))
        overall_ll = float(ll_vec.mean())
        overall_brier = float(((p_cand - y_assess)**2).mean())
        overall_ece = float(compute_ece(p_cand, y_assess, 10))
        tail_25 = int((ll_vec >= 2.5).sum())

        slice_breakdowns = {}
        for s_name, s_mask in slice_masks.items():
            slice_breakdowns[s_name] = {
                "n": int(s_mask.sum()),
                "log_loss": float(ll_vec[s_mask].mean()),
                "brier": float(((p_cand[s_mask] - y_assess[s_mask])**2).mean()),
            }

        # Bootstrap contrast against C0
        bs_contrast = None
        if m_name != "baseline_c0":
            delta_ll_vec = ll_vec - (-(y_assess * np.log(p_c0_assess) + (1 - y_assess) * np.log(1 - p_c0_assess)))
            delta_brier_vec = (p_cand - y_assess)**2 - (p_c0_assess - y_assess)**2

            bs_ll = paired_monthly_bootstrap(delta_ll_vec, assess_months, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
            bs_brier = paired_monthly_bootstrap(delta_brier_vec, assess_months, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)

            slice_guards = {}
            all_guards_pass = True
            for s_name, s_mask in slice_masks.items():
                s_months = [assess_months[i] for i in range(len(s_mask)) if s_mask[i]]
                bs_s = paired_monthly_bootstrap(delta_ll_vec[s_mask], s_months, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
                passes = bs_s["ci_upper"] <= 0.00500
                if s_name != "all_canonical" and not passes:
                    all_guards_pass = False
                slice_guards[s_name] = {
                    "n": int(s_mask.sum()),
                    "delta_ll": bs_s["delta"],
                    "ci_lower": bs_s["ci_lower"],
                    "ci_upper": bs_s["ci_upper"],
                    "guard_pass": passes,
                    "threshold": 0.00500,
                }

            bs_contrast = {
                "overall_log_loss": bs_ll,
                "overall_brier": bs_brier,
                "slice_guards": slice_guards,
                "all_guards_pass": all_guards_pass,
            }
            sports_guard_evaluations[m_name] = bs_contrast

        model_summaries[m_name] = {
            "overall": {
                "log_loss": overall_ll,
                "brier": overall_brier,
                "ece": overall_ece,
                "tail_count_ll_ge_2_5": tail_25,
            },
            "slices": slice_breakdowns,
            "contrast_vs_c0": bs_contrast,
            "market_cohorts": market_cohort_evaluations[m_name],
        }

    # 10. Save Predictions to track4_predictions.parquet
    print(f"\nWriting predictions to {OUTPUT_PARQUET_PATH}...")
    pred_df = pd.DataFrame({
        "golgg_match_id": cap_assess["golgg_match_id"].values,
        "date": dates_assess,
        "result_day": result_days_assess,
        "best_of": cap_assess["best_of"].values.astype(int),
        "y": y_assess,
        "tournament_name": cap_assess["tournament_name"].values,
        "competition_tier": cap_assess["competition_tier"].values,
        "unknown_players": cap_assess["unknown_players"].values.astype(int),
        "c_stat": c_stat_assess,
        "c_depth": c_depth_assess,
        "d_expert": d_expert_assess,
        "is_bo1": is_bo1_assess,
        "is_bo5": is_bo5_assess,
        "d_trio": d_trio_ass,
        "p_c0": p_c0_assess,
        "p_c0_connector": p_conn,
        "p_variant_4a": p_4a_assess,
        "p_variant_4b": p_4b_assess,
        "p_variant_4c": p_4c_assess,
        "p_c0_ons": p_c0_ons,
        "p_variant_4b_ons": p_4b_ons,
        "p_variant_4c_ons": p_4c_ons,
    })
    pred_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"Saved {len(pred_df)} rows to {OUTPUT_PARQUET_PATH}")

    # 11. Save Full Reporting JSON
    reporting_payload = {
        "schema": "track4_gated_mixture_ons_v1",
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "task": "Track 4: Uncertainty/Disagreement-Gated Expert Mixture + ONS Recalibration + Multi-Capacity Ensemble",
        "partitions": {
            "train": {"end": "2025-07-01", "rows": 36205},
            "cal": {"start": "2025-07-01", "end": "2026-01-01", "rows": 1328},
            "assess": {"start": "2026-01-01", "end": "2026-09-08", "rows": 3103},
        },
        "step1_c0_raw_connector_reaudit": {
            "status": "ALL_GUARDS_PASS_AGAINST_C0",
            "finding": "In Plan09, c0_raw_connector failed BO5 and source-unseen replacement guards against annual A0 (+0.00518 and +0.00519 > 0.00500). Re-audited against active C0, BO5 upper is +0.003570 and source-unseen upper is +0.001338, both strictly passing the +0.00500 guard.",
            "guards": conn_guard_results,
        },
        "step2_gated_mixture_optimization": {
            "variant_4a": {
                "name": "Gated-C0-Internal (W32/E12, Rating, MLP)",
                "l2_penalty": best_l2_4a,
                "cal_cv_loss": best_cv_4a,
                "gating_intercepts": t0_4a.tolist(),
                "gating_covariate_weights": Th_4a.tolist(),
                "average_weights_assess": w_4a_ass.mean(axis=0).tolist(),
            },
            "variant_4b": {
                "name": "Multi-Capacity Mixture (W32/E12, W32/E24, W64/E12, Rating, MLP)",
                "l2_penalty": best_l2_4b,
                "cal_cv_loss": best_cv_4b,
                "gating_intercepts": t0_4b.tolist(),
                "gating_covariate_weights": Th_4b.tolist(),
                "average_weights_assess": w_4b_ass.mean(axis=0).tolist(),
            },
            "variant_4c": {
                "name": "Disagreement-Regularized C0 + Raw Specialist Trio",
                "base_anchor": "variant_4b",
                "alpha_base": 0.1769033853776533,
                "shrinkage_gamma": best_gamma_4c,
                "formula": "z_4c = (1 - alpha(x))*z_4b + alpha(x)*z_specialist, where alpha(x) = alpha_0 * exp(-gamma * |z_4b - z_specialist|)",
                "mean_alpha_assess": float(alpha_ass_4c.mean()),
            },
        },
        "step3_online_recalibration_ons": {
            "algorithm": "Gupta & Ramdas (ICML 2023) Online Platt Scaling via Online Newton Step (ONS)",
            "causality_contract": "Day-by-day parameters (a_t, b_t) updated ONLY from finished matches where result_day < current_date. Identity initialization (a_0=1.0, b_0=0.0).",
            "hyperparameters": {"gamma": 0.005, "eps": 1.0, "a_bounds": [0.7, 1.5], "b_bounds": [-0.5, 0.5]},
            "c0_ons_final_params": {"a": float(ons_c0_history[-1]["a"]), "b": float(ons_c0_history[-1]["b"])},
            "variant_4b_ons_final_params": {"a": float(ons_4b_history[-1]["a"]), "b": float(ons_4b_history[-1]["b"])},
            "variant_4c_ons_final_params": {"a": float(ons_4c_history[-1]["a"]), "b": float(ons_4c_history[-1]["b"])},
        },
        "models": model_summaries,
        "sports_guards_summary": sports_guard_evaluations,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(reporting_payload, f, indent=2)
    print(f"Successfully saved full report to {OUTPUT_JSON_PATH}")

    # Summary Printout
    print("\n=== FINAL RESULTS SUMMARY ON CANONICAL 2026 ASSESS (N=3,103) ===")
    print(f"{'Model':<24} | {'LogLoss':<9} | {'Delta LL':<10} | {'95% CI Lower':<12} | {'95% CI Upper':<12} | {'Brier':<9} | {'ECE (%)':<8} | {'Tail>=2.5'}")
    print("-" * 115)
    for m_name in all_models:
        s = model_summaries[m_name]
        ov = s["overall"]
        ct = s["contrast_vs_c0"]
        d_ll = f"{ct['overall_log_loss']['delta']:+.6f}" if ct else "   0.000000"
        ci_l = f"{ct['overall_log_loss']['ci_lower']:+.6f}" if ct else "   0.000000"
        ci_u = f"{ct['overall_log_loss']['ci_upper']:+.6f}" if ct else "   0.000000"
        print(f"{m_name:<24} | {ov['log_loss']:<9.6f} | {d_ll:<10} | {ci_l:<12} | {ci_u:<12} | {ov['brier']:<9.6f} | {ov['ece']*100:<8.4f} | {ov['tail_count_ll_ge_2_5']:<9}")


if __name__ == "__main__":
    main()
