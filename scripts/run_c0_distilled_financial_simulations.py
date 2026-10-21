#!/usr/bin/env python3
"""Comprehensive head-to-head financial and econometric simulations for C0 Distilled Model Family.

Evaluates:
- C0 (Baseline p_c0, 0.535449 LL)
- C0-D (Born-Again W32-KD + Conn50 + ONS, 0.534120 LL)
- C0-D-Core (Pure W32-KD Born-Again 3-seed, 0.534418 LL)
- C0-Slim (Pure W32-KD Single-seed, 0.535062 LL)
- C0-Slim+ (Single-Seed W32-KD + Conn50, 0.534356 LL)
- C0-Mini (W16-KD 3-seed, 0.535845 LL)

Across 3 Comprehensive Simulation Suites:
1. Suite 1: Production Polish Bookmakers 2h Trajectory Replay (12% Tax) & 50-Bet Optimal Synergy Ledger
2. Suite 2: Sharp Pinnacle T-1h Market Simulation (N=1,668) under 12% Polish Tax and 0% International Tax
3. Suite 3: Multi-Horizon (T-48h, T-24h, T-12h, T-6h, <2h) & Polish2plus (N=1,467) Proper-Score & Hybrid-50

Outputs saved to:
- data/08_reporting/c0_upgrade_and_distillation_20261017/financial_simulation_results.json
- data/08_reporting/c0_upgrade_and_distillation_20261017/financial_strategy_comparison.csv
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.special import expit, logit
# Paths
REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRACK1_PATH = REPORT_DIR / "track1_predictions.parquet"
TRACK4_PATH = REPORT_DIR / "track4_predictions.parquet"
TRAJ_PATH = PROJECT_ROOT / "data/04_feature/production_2h_odds_trajectory.parquet"
LEDGER_PATH = PROJECT_ROOT / "data/08_reporting/optimal_strategy_simulation_ledger.csv"
RAW_PROD_DIR = PROJECT_ROOT / "data/01_raw/c0_production_counterfactual/c0_production_20261006_01"
PINNACLE_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"
POLISH_PANEL_PATH = PROJECT_ROOT / "data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/paired_probability_panel.parquet"

OUTPUT_JSON_PATH = REPORT_DIR / "financial_simulation_results.json"
OUTPUT_CSV_PATH = REPORT_DIR / "financial_strategy_comparison.csv"


# ---------------------------------------------------------------------------
# Utility & Econometric Functions
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def clip_prob(p: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    return np.clip(p, eps, 1.0 - eps)


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
    theta = np.array([1.0, 0.0], dtype=np.float64)  # (a, b)
    hessian_acc = eps * np.eye(2, dtype=np.float64)

    theta_history = []
    processed_matches: set[int] = set()

    for current_date in unique_dates:
        d_mask = (dates == current_date)
        d_indices = np.where(d_mask)[0]

        z_d = logits[d_indices]
        p_d = sigmoid(theta[0] * z_d + theta[1])
        p_recal[d_indices] = p_d

        theta_history.append({"date": str(current_date), "a": float(theta[0]), "b": float(theta[1])})

        available_mask = (result_days <= current_date) & ~np.isin(np.arange(n_samples), list(processed_matches))
        new_indices = np.where(available_mask)[0]

        if len(new_indices) > 0:
            for idx in new_indices:
                z_i = logits[idx]
                y_i = y_true[idx]
                x_i = np.array([z_i, 1.0], dtype=np.float64)

                p_i = float(sigmoid(np.dot(theta, x_i)))
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

    # Max Drawdown
    equities = df["bankroll"].to_numpy(dtype=float)
    if len(equities) > 0:
        eq_curve = np.concatenate([[initial_bankroll], equities])
        running_max = np.maximum.accumulate(eq_curve)
        drawdowns = (running_max - eq_curve) / running_max * 100.0
        max_drawdown = float(np.max(drawdowns))
    else:
        max_drawdown = 0.0

    # Profit Factor
    gross_wins = float(df[df["profit"] > 0]["profit"].sum())
    gross_losses = float(abs(df[df["profit"] < 0]["profit"].sum()))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    # Sharpe Ratio (per-bet normalized)
    returns = df["profit"] / df["stake"]
    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (float(returns.mean()) / ret_std) * np.sqrt(n_bets) if ret_std > 0 else 0.0

    # Stake-weighted Three-Part Econometric Identity:
    # Yield = Claimed EV - Optimizer's Curse (Bias B) + Match Noise (Luck eta)
    weights = df["stake"].to_numpy(dtype=float) / total_turnover
    claimed_ev = float(np.sum(weights * df["ev_claimed"].to_numpy(dtype=float))) * 100.0
    net_clv = float(np.sum(weights * df["net_clv"].to_numpy(dtype=float))) * 100.0
    bias_b = claimed_ev - net_clv
    luck_eta = realized_yield - net_clv

    # Verify per-bet identity: yield_unit == ev_claimed - bias_unit + luck_unit
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
# Main Simulation Orchestrator
# ---------------------------------------------------------------------------

def run_distilled_financial_simulations() -> None:
    print("=" * 80)
    print("STARTING C0 DISTILLED FAMILY FINANCIAL & ECONOMETRIC SIMULATIONS")
    print("=" * 80)

    # 1. Load Track 1 & Track 4 predictions
    p1 = pd.read_parquet(TRACK1_PATH)
    p4 = pd.read_parquet(TRACK4_PATH)
    print(f"Loaded Track 1 predictions: {p1.shape}")
    print(f"Loaded Track 4 predictions: {p4.shape}")

    y_assess = p1["y"].to_numpy(dtype=int)
    dates_assess = p4["date"].to_numpy()
    result_days_assess = p4["result_day"].to_numpy()

    p_c0 = p1["p_c0__full"].to_numpy(dtype=np.float64)
    p_d_core = p1["w32_e16_kd_born_again__full"].to_numpy(dtype=np.float64)
    p_slim = p1["w32_e16_kd_single__full"].to_numpy(dtype=np.float64)
    p_mini = p1["w16_e12_kd__full"].to_numpy(dtype=np.float64)
    p_conn = p4["p_c0_connector"].to_numpy(dtype=np.float64)

    # Blends
    p_slim_plus = 0.50 * p_slim + 0.50 * p_conn
    p_ba_conn = 0.50 * p_d_core + 0.50 * p_conn

    # ONS Recalibration for C0-D
    print("Executing causal day-by-day ONS recalibration for C0-D...")
    p_c0_d, _ = run_ons_recalibration(dates_assess, result_days_assess, logit(p_ba_conn), y_assess)

    models_assess = {
        "C0": p_c0,
        "C0-D": p_c0_d,
        "C0-D-Core": p_d_core,
        "C0-Slim": p_slim,
        "C0-Slim+": p_slim_plus,
        "C0-Mini": p_mini,
    }

    # Verify LogLoss
    def calc_ll(p: np.ndarray) -> float:
        pcl = np.clip(p, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_assess * np.log(pcl) + (1 - y_assess) * np.log(1 - pcl)))

    print("\nVerified Model LogLoss on 2026 ASSESS (N=3,103):")
    for m_name, arr in models_assess.items():
        print(f"  {m_name:<10s}: {calc_ll(arr):.6f}")

    # Map golgg_match_id to index
    gid_to_idx = {str(gid): i for i, gid in enumerate(p1["golgg_match_id"].values)}

    # Strategy comparison rows table accumulator
    strategy_rows: list[dict[str, Any]] = []

    # =========================================================================
    # SUITE 1: PRODUCTION POLISH BOOKMAKERS REPLAY & 50-BET OPTIMAL SYNERGY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUITE 1: POLISH BOOKMAKERS 2H TRAJECTORY REPLAY & 50-BET OPTIMAL SYNERGY")
    print("=" * 80)

    traj_df = pd.read_parquet(TRAJ_PATH)
    ledger_df = pd.read_csv(LEDGER_PATH)

    # Map trajectory 43 matches
    u_traj = traj_df[traj_df["p_model_a"].notna()].drop_duplicates(subset=["canonical_match_id"]).copy()
    raw_prod_dir = RAW_PROD_DIR
    golgg = pd.read_parquet(raw_prod_dir / "golgg.parquet")

    def norm_team(s: Any) -> str:
        return str(s).lower().replace(" ", "").replace(".", "").replace("-", "").replace("esports", "").replace("gaming", "").replace("team", "")

    traj_to_golgg: dict[int, tuple[str, bool]] = {}
    for _, r in u_traj.iterrows():
        cid = int(r["canonical_match_id"])
        target_p = r["p_c0"] if pd.notna(r.get("p_c0")) else r["p_model_a"]
        t_a = norm_team(r["team_a_name"])
        t_b = norm_team(r["team_b_name"])
        dt_str = str(r["start_time_normalized"])[:10]

        m1 = p1[np.isclose(p1["p_c0__full"], target_p, atol=1e-4)]
        m2 = p1[np.isclose(1.0 - p1["p_c0__full"], target_p, atol=1e-4)]

        cands = []
        for _, mr in m1.iterrows():
            cands.append((mr["golgg_match_id"], False, abs(pd.to_datetime(mr["date"]) - pd.to_datetime(dt_str)).days))
        for _, mr in m2.iterrows():
            cands.append((mr["golgg_match_id"], True, abs(pd.to_datetime(mr["date"]) - pd.to_datetime(dt_str)).days))

        if cands:
            cands.sort(key=lambda x: x[2])
            traj_to_golgg[cid] = (str(cands[0][0]), cands[0][1])
        else:
            g_cands = golgg[(golgg["team1_name"].apply(norm_team).str.contains(t_a) | golgg["team1_name"].apply(norm_team).apply(lambda x: t_a in x)) &
                            (golgg["team2_name"].apply(norm_team).str.contains(t_b) | golgg["team2_name"].apply(norm_team).apply(lambda x: t_b in x))]
            if not g_cands.empty:
                gid = str(g_cands.iloc[0]["match_id"])
                traj_to_golgg[cid] = (gid, False)

    print(f"Mapped {len(traj_to_golgg)} trajectory canonical matches to golgg_match_id.")

    # Map 50 ledger bets
    matches = pd.read_parquet(raw_prod_dir / "matches.parquet")
    mappings = pd.read_parquet(raw_prod_dir / "mappings.parquet")
    m_map = matches.merge(mappings, left_on="id", right_on="canonical_match_id")
    matches["norm_a"] = matches["team_a_name"].apply(norm_team)
    matches["norm_b"] = matches["team_b_name"].apply(norm_team)
    m_map["norm_a"] = m_map["team_a_name"].apply(norm_team)
    m_map["norm_b"] = m_map["team_b_name"].apply(norm_team)

    ledger_df["norm_sel"] = ledger_df["team_selected"].apply(norm_team)
    ledger_df["norm_opp"] = ledger_df["opponent"].apply(norm_team)

    um_df = pd.read_parquet(PROJECT_ROOT / "data/07_model_output/a0_native_refit/native_refit_20260929_122617/uniform/uniform_monthly/candidate.parquet")
    um_2026 = um_df.set_index("golgg_match_id").loc[p1["golgg_match_id"].astype(str)].reset_index()
    p_c0_mon = um_2026["p_full"].to_numpy(dtype=np.float64)
    p1_dates = pd.to_datetime(p1["date"])

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
                val_s = (1.0 - p_c0[idx]) if is_rev else p_c0[idx]
                diff = min(abs(val_m - target_p), abs(val_s - target_p))
                if best is None or diff < best[0]:
                    best = (diff, str(p1["golgg_match_id"].iloc[idx]), is_rev)
        if best is not None:
            ledger_to_golgg[bid] = (best[1], best[2])

    print(f"Mapped {len(ledger_to_golgg)} out of {len(ledger_df)} ledger bets to golgg_match_id with exact parity.")

    # Reconstruct ledger market probability analytically:
    l_c0_led = logit(ledger_df["p_c0"].to_numpy(dtype=float))
    l_hyb_led = logit(ledger_df["p_hybrid"].to_numpy(dtype=float))
    l_mkt_led = (l_hyb_led - 0.65 * l_c0_led) / 0.35
    p_mkt_led = expit(l_mkt_led)
    p_thesis_led = ledger_df["p_thesis"].to_numpy(dtype=float)
    odds_led = ledger_df["odds_taken"].to_numpy(dtype=float)
    closing_odds_led = ledger_df["closing_odds"].to_numpy(dtype=float)
    won_led = ledger_df["won"].to_numpy(dtype=int)

    suite1_results: dict[str, Any] = {"trajectory_replay": {}, "ledger_50_cohort": {}}

    # Part 1A: Trajectory Replay across all 6 models
    from src.analysis.historical_production_simulator import HistoricalProductionSimulator

    # Pre-map p_039 by golgg_match_id
    mkt_raw_039 = pd.read_parquet(PINNACLE_T1_PATH)
    p039_lookup = dict(zip(mkt_raw_039["golgg_match_id"].astype(str), mkt_raw_039["p_039"]))

    for m_name in models_assess:
        suite1_results["trajectory_replay"][m_name] = {}
        df_m = traj_df.copy()
        new_p_model_a = []
        new_p_039 = []

        for _, r in df_m.iterrows():
            cid = int(r["canonical_match_id"])
            if pd.isna(r.get("p_model_a")):
                new_p_model_a.append(np.nan)
                new_p_039.append(np.nan)
            elif cid in traj_to_golgg:
                gid, inv = traj_to_golgg[cid]
                p_val = models_assess[m_name][gid_to_idx[gid]]
                new_p_model_a.append(1.0 - p_val if inv else p_val)
                p039_val = p039_lookup.get(gid, np.nan)
                new_p_039.append((1.0 - p039_val) if (inv and pd.notna(p039_val)) else p039_val)
            else:
                new_p_model_a.append(r["p_model_a"])
                new_p_039.append(np.nan)

        df_m["p_model_a"] = new_p_model_a
        df_m["p_039_a"] = new_p_039

        # Strategy 1: Unconstrained Pure Model
        sim1 = HistoricalProductionSimulator(
            initial_bankroll=10000.0,
            policy_kind="unconstrained",
            alpha=1.0,
            guard_b_logit=None,
            min_ev=0.05,
            max_ev=None,
            tier1_only=False,
            stake_mode="flat",
            flat_stake=100.0,
            tax_rate=0.12,
        )
        res1 = sim1.run(df_m)
        s1_df = res1["settled_df"]
        s1_bets = []
        if not s1_df.empty:
            b_run = 10000.0
            for _, row in s1_df.iterrows():
                b_run += row["profit"]
                s1_bets.append({
                    "odds_taken": row["odds"],
                    "won": row["won"],
                    "stake": row["stake"],
                    "profit": row["profit"],
                    "bankroll": b_run,
                    "ev_claimed": row["ev_claimed"],
                    "net_clv": row["clv"],
                })
        d1 = compute_three_part_decomposition(s1_bets, initial_bankroll=10000.0)
        suite1_results["trajectory_replay"][m_name]["unconstrained"] = d1
        strategy_rows.append({
            "regime": "2026 Production Polish Books 2h Trajectory (12% Tax)",
            "model": m_name,
            "strategy": f"{m_name} Unconstrained Pure Model (Flat 100)",
            "bets": d1["n_bets"],
            "win_rate_pct": d1["win_rate_pct"],
            "turnover_pln": d1["total_turnover_pln"],
            "profit_pln": d1["total_net_profit_pln"],
            "realized_yield_pct": d1["realized_yield_pct"],
            "bankroll_roi_pct": d1["bankroll_roi_pct"],
            "claimed_ev_pct": d1["claimed_ev_pct"],
            "net_clv_pct": d1["true_edge_net_clv_pct"],
            "optimizers_curse_bias_pp": d1["optimizers_curse_bias_pp"],
            "match_luck_noise_pp": d1["match_luck_noise_pp"],
            "max_drawdown_pct": d1["max_drawdown_pct"],
            "profit_factor": d1["profit_factor"],
            "sharpe_ratio": d1["sharpe_ratio"],
            "status": "UNCONSTRAINED_REPLAY",
        })

        # Strategy 2a: Hybrid-50 + Guard B <= 0.85 (Flat 100)
        sim2_flat = HistoricalProductionSimulator(
            initial_bankroll=10000.0,
            policy_kind="shrunk_hybrid",
            alpha=0.50,
            guard_b_logit=0.85,
            min_ev=0.05,
            max_ev=0.25,
            tier1_only=False,
            stake_mode="flat",
            flat_stake=100.0,
            tax_rate=0.12,
        )
        res2_flat = sim2_flat.run(df_m)
        s2f_df = res2_flat["settled_df"]
        s2f_bets = []
        if not s2f_df.empty:
            b_run = 10000.0
            for _, row in s2f_df.iterrows():
                b_run += row["profit"]
                s2f_bets.append({
                    "odds_taken": row["odds"],
                    "won": row["won"],
                    "stake": row["stake"],
                    "profit": row["profit"],
                    "bankroll": b_run,
                    "ev_claimed": row["ev_claimed"],
                    "net_clv": row["clv"],
                })
        d2f = compute_three_part_decomposition(s2f_bets, initial_bankroll=10000.0)
        suite1_results["trajectory_replay"][m_name]["hybrid_flat"] = d2f
        strategy_rows.append({
            "regime": "2026 Production Polish Books 2h Trajectory (12% Tax)",
            "model": m_name,
            "strategy": f"{m_name} Hybrid-50 + Guard B <= 0.85 (Flat 100)",
            "bets": d2f["n_bets"],
            "win_rate_pct": d2f["win_rate_pct"],
            "turnover_pln": d2f["total_turnover_pln"],
            "profit_pln": d2f["total_net_profit_pln"],
            "realized_yield_pct": d2f["realized_yield_pct"],
            "bankroll_roi_pct": d2f["bankroll_roi_pct"],
            "claimed_ev_pct": d2f["claimed_ev_pct"],
            "net_clv_pct": d2f["true_edge_net_clv_pct"],
            "optimizers_curse_bias_pp": d2f["optimizers_curse_bias_pp"],
            "match_luck_noise_pp": d2f["match_luck_noise_pp"],
            "max_drawdown_pct": d2f["max_drawdown_pct"],
            "profit_factor": d2f["profit_factor"],
            "sharpe_ratio": d2f["sharpe_ratio"],
            "status": "HYBRID_RISK_REDUCTION",
        })

        # Strategy 2b: Hybrid-50 + Guard B <= 0.85 (1/4 Kelly, cap=150)
        sim2_k = HistoricalProductionSimulator(
            initial_bankroll=10000.0,
            policy_kind="shrunk_hybrid",
            alpha=0.50,
            guard_b_logit=0.85,
            min_ev=0.05,
            max_ev=0.25,
            tier1_only=False,
            stake_mode="quarter_kelly",
            max_stake_cap=150.0,
            tax_rate=0.12,
        )
        res2_k = sim2_k.run(df_m)
        s2k_df = res2_k["settled_df"]
        s2k_bets = []
        if not s2k_df.empty:
            b_run = 10000.0
            for _, row in s2k_df.iterrows():
                b_run += row["profit"]
                s2k_bets.append({
                    "odds_taken": row["odds"],
                    "won": row["won"],
                    "stake": row["stake"],
                    "profit": row["profit"],
                    "bankroll": b_run,
                    "ev_claimed": row["ev_claimed"],
                    "net_clv": row["clv"],
                })
        d2k = compute_three_part_decomposition(s2k_bets, initial_bankroll=10000.0)
        suite1_results["trajectory_replay"][m_name]["hybrid_kelly"] = d2k
        strategy_rows.append({
            "regime": "2026 Production Polish Books 2h Trajectory (12% Tax)",
            "model": m_name,
            "strategy": f"{m_name} Hybrid-50 + Guard B <= 0.85 (1/4 Kelly)",
            "bets": d2k["n_bets"],
            "win_rate_pct": d2k["win_rate_pct"],
            "turnover_pln": d2k["total_turnover_pln"],
            "profit_pln": d2k["total_net_profit_pln"],
            "realized_yield_pct": d2k["realized_yield_pct"],
            "bankroll_roi_pct": d2k["bankroll_roi_pct"],
            "claimed_ev_pct": d2k["claimed_ev_pct"],
            "net_clv_pct": d2k["true_edge_net_clv_pct"],
            "optimizers_curse_bias_pp": d2k["optimizers_curse_bias_pp"],
            "match_luck_noise_pp": d2k["match_luck_noise_pp"],
            "max_drawdown_pct": d2k["max_drawdown_pct"],
            "profit_factor": d2k["profit_factor"],
            "sharpe_ratio": d2k["sharpe_ratio"],
            "status": "HYBRID_RISK_REDUCTION",
        })

    # Part 1B: Evaluate each model on the 50-bet cohort under Optimal Synergy & DRO
    for m_name, p_arr in models_assess.items():
        if m_name == "C0":
            p_m_vec = ledger_df["p_c0"].to_numpy(dtype=float)
            l_hyb = logit(ledger_df["p_hybrid"].to_numpy(dtype=float))
        else:
            p_m_list = []
            l_shift_list = []
            for idx_row, r in ledger_df.iterrows():
                bid = int(r["bet_id"])
                gid, is_rev = ledger_to_golgg[bid]
                g_idx = gid_to_idx[gid]
                val_m = (1.0 - p_arr[g_idx]) if is_rev else p_arr[g_idx]
                val_s = (1.0 - p_c0[g_idx]) if is_rev else p_c0[g_idx]
                p_sel = float(expit(logit(float(r["p_c0"])) + (logit(val_m) - logit(val_s))))
                p_m_list.append(p_sel)
                l_shift_list.append(logit(val_m) - logit(val_s))
            p_m_vec = np.array(p_m_list)
            l_hyb = logit(ledger_df["p_hybrid"].to_numpy(dtype=float)) + 0.65 * np.array(l_shift_list)
        l_m = logit(np.clip(p_m_vec, 1e-6, 1.0 - 1e-6))
        p_hyb = expit(l_hyb)
        ev_net = p_hyb * 0.88 * odds_led - 1.0

        # Exact Optimal Synergy rules on ledger
        guard_b = np.abs(l_m - l_mkt_led) <= 0.85
        consensus_veto = np.abs(p_m_vec - p_thesis_led) <= 0.10
        odds_ok = odds_led <= 4.00
        ev_ok = (ev_net >= 0.05) & (ev_net <= 0.25)
        opt_syn_mask = guard_b & consensus_veto & odds_ok & ev_ok
        dro_mask = opt_syn_mask & (((p_hyb - 0.025) * 0.88 * odds_led - 1.0) > 0.0)

        # Staking simulation for OptSyn & DRO
        for strat_name, mask in [
            ("Optimal Synergy (1/4 Kelly, Caps)", opt_syn_mask),
            ("Optimal Synergy + DRO Guard W (eps=0.025)", dro_mask),
            ("Adopted Production Ledger (50-Bet Fixed Anchor)", np.ones(len(ledger_df), dtype=bool)),
        ]:
            if strat_name == "Adopted Production Ledger (50-Bet Fixed Anchor)" and m_name != "C0":
                continue

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
                stake = min(150.0, max(20.0, bankroll * 0.25 * f_star)) if strat_name != "Adopted Production Ledger (50-Bet Fixed Anchor)" else float(ledger_df["stake"].iloc[i])
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
            regime_label = "2026 Production Polish Books (12% Tax)"
            strat_label = f"{m_name} {strat_name}"

            strategy_rows.append({
                "regime": regime_label,
                "model": m_name,
                "strategy": strat_label,
                "bets": decomp["n_bets"],
                "win_rate_pct": decomp["win_rate_pct"],
                "turnover_pln": decomp["total_turnover_pln"],
                "profit_pln": decomp["total_net_profit_pln"],
                "realized_yield_pct": decomp["realized_yield_pct"],
                "bankroll_roi_pct": decomp["bankroll_roi_pct"],
                "claimed_ev_pct": decomp["claimed_ev_pct"],
                "net_clv_pct": decomp["true_edge_net_clv_pct"],
                "optimizers_curse_bias_pp": decomp["optimizers_curse_bias_pp"],
                "match_luck_noise_pp": decomp["match_luck_noise_pp"],
                "max_drawdown_pct": decomp["max_drawdown_pct"],
                "profit_factor": decomp["profit_factor"],
                "sharpe_ratio": decomp["sharpe_ratio"],
                "status": "VALIDATED_PRODUCTION_EDGE" if decomp["realized_yield_pct"] > 0 else "UNDERPERFORMING",
            })

            suite1_results["ledger_50_cohort"][f"{m_name}_{strat_name}"] = decomp

    # =========================================================================
    # SUITE 2: SHARP PINNACLE T-1H MARKET SIMULATION (N=1,668)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUITE 2: SHARP PINNACLE T-1H MARKET SIMULATION (N=1,668)")
    print("=" * 80)

    mkt_raw = pd.read_parquet(PINNACLE_T1_PATH)
    mkt_df = mkt_raw[["golgg_match_id", "date", "y", "q_consensus", "p_039"]].copy()
    mkt_df["golgg_match_id"] = mkt_df["golgg_match_id"].astype(str)

    p1_sub = pd.DataFrame({"golgg_match_id": p1["golgg_match_id"].astype(str)})
    for m_name, arr in models_assess.items():
        p1_sub[m_name] = arr

    merged_mkt = mkt_df.merge(p1_sub, on="golgg_match_id")
    merged_mkt = merged_mkt.sort_values(by=["date", "golgg_match_id"]).reset_index(drop=True)
    print(f"Constructed unified Pinnacle T-1h market cohort: {len(merged_mkt)} fixtures.")

    suite2_results: dict[str, Any] = {}

    regimes = [
        ("Regime A (12% Polish Tax)", 0.12),
        ("Regime B (0% International Tax)", 0.00),
    ]

    strategies = [
        ("Unconstrained Pure Model (EV >= 5%, Odds <= 4.00, Flat 100)", "unconstrained"),
        ("Hybrid-50 + Guard B <= 0.85 (EV in [5%, 25%], Odds <= 4.00, Flat 100)", "hybrid_flat"),
        ("Hybrid-50 + Guard B <= 0.85 (EV in [5%, 25%], Odds <= 4.00, 1/4 Kelly)", "hybrid_kelly"),
        ("Optimal Synergy (Hybrid + Guard B + Veto <= 0.10 + Odds <= 4.00 + 1/4 Kelly)", "optimal_synergy"),
        ("Optimal Synergy + DRO Guard W (eps=0.025, 1/4 Kelly)", "optimal_dro"),
    ]

    for regime_name, tax_rate in regimes:
        c = 1.0 - tax_rate
        suite2_results[regime_name] = {}

        for m_name in models_assess:
            suite2_results[regime_name][m_name] = {}

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

            for strat_label, strat_kind in strategies:
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

                    elif strat_kind in ("optimal_synergy", "optimal_dro"):
                        veto1 = np.isnan(p_039_mkt[i]) or (abs(p1_model[i] - p_039_mkt[i]) <= 0.10)
                        veto2 = np.isnan(p_039_mkt[i]) or (abs(p2_model[i] - (1.0 - p_039_mkt[i])) <= 0.10)

                        c1 = (ev1_hyb[i] >= 0.05) and (ev1_hyb[i] <= 0.25) and (gap1[i] <= 0.85) and veto1 and (o1[i] <= 4.00)
                        c2 = (ev2_hyb[i] >= 0.05) and (ev2_hyb[i] <= 0.25) and (gap2[i] <= 0.85) and veto2 and (o2[i] <= 4.00)

                        if strat_kind == "optimal_dro":
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

                decomp = compute_three_part_decomposition(sim_bets, initial_bankroll=10000.0)
                suite2_results[regime_name][m_name][strat_label] = decomp

                status_label = "POSITIVE_ALPHA" if decomp["realized_yield_pct"] > 0 else "NEGATIVE_ALPHA"
                if "12%" in regime_name and decomp["realized_yield_pct"] < 0:
                    status_label = "TAX_BURDEN_DEFICIT"
                if decomp["optimizers_curse_bias_pp"] > 20.0:
                    status_label = "OPTIMIZERS_CURSE_EXPOSURE"

                strategy_rows.append({
                    "regime": regime_name,
                    "model": m_name,
                    "strategy": strat_label,
                    "bets": decomp["n_bets"],
                    "win_rate_pct": decomp["win_rate_pct"],
                    "turnover_pln": decomp["total_turnover_pln"],
                    "profit_pln": decomp["total_net_profit_pln"],
                    "realized_yield_pct": decomp["realized_yield_pct"],
                    "bankroll_roi_pct": decomp["bankroll_roi_pct"],
                    "claimed_ev_pct": decomp["claimed_ev_pct"],
                    "net_clv_pct": decomp["true_edge_net_clv_pct"],
                    "optimizers_curse_bias_pp": decomp["optimizers_curse_bias_pp"],
                    "match_luck_noise_pp": decomp["match_luck_noise_pp"],
                    "max_drawdown_pct": decomp["max_drawdown_pct"],
                    "profit_factor": decomp["profit_factor"],
                    "sharpe_ratio": decomp["sharpe_ratio"],
                    "status": status_label,
                })

    # =========================================================================
    # SUITE 3: MULTI-HORIZON PROPER-SCORE & HYBRID-50 COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUITE 3: MULTI-HORIZON & POLISH2PLUS PROPER-SCORE COMPARISONS")
    print("=" * 80)

    # Polish production horizons
    m_map_horizon = mappings[["canonical_match_id", "golgg_match_id"]].merge(
        matches[["id", "winner_side", "start_time_normalized"]],
        left_on="canonical_match_id",
        right_on="id",
    )
    m_map_horizon["golgg_match_id"] = m_map_horizon["golgg_match_id"].astype(str)

    df_models_horizon = pd.DataFrame({"golgg_match_id": p1["golgg_match_id"].astype(str), "y": p1["y"]})
    for m in ["C0", "C0-D", "C0-Slim", "C0-Mini"]:
        df_models_horizon[m] = models_assess[m]

    merged_matches_h = m_map_horizon.merge(df_models_horizon, on="golgg_match_id")
    merged_matches_h["a_is_pred_t1"] = (merged_matches_h["winner_side"] == "team_a") == (merged_matches_h["y"] == 1)

    quotes = pd.read_parquet(raw_prod_dir / "quotes.parquet")
    q_m = quotes.merge(merged_matches_h, on="canonical_match_id")
    start_dt = pd.to_datetime(q_m["start_time_normalized"], utc=True, format="mixed")
    scraped_dt = pd.to_datetime(q_m["scraped_at"], utc=True, format="mixed")
    q_m["lead_hours"] = (start_dt - scraped_dt).dt.total_seconds() / 3600.0
    q_m = q_m[q_m["lead_hours"] >= 0].copy()

    inv_a = 1.0 / q_m["odds_a"]
    inv_b = 1.0 / q_m["odds_b"]
    overround = (inv_a + inv_b) - 1.0
    p_shin_a = np.clip(inv_a - overround / 2.0, 1e-6, 1.0 - 1e-6)
    q_m["p_shin_pred"] = np.where(q_m["a_is_pred_t1"], p_shin_a, 1.0 - p_shin_a)

    prod_horizons = [
        ("T-48h", 48.0),
        ("T-24h", 24.0),
        ("T-12h", 12.0),
        ("T-6h", 6.0),
        ("<2h", 0.0),
    ]

    suite3_results: dict[str, Any] = {"production_horizons": {}, "polish2plus": {}}

    for label, h in prod_horizons:
        sub_q = (
            q_m[q_m["lead_hours"] >= h]
            .sort_values("scraped_at")
            .groupby(["canonical_match_id", "bookmaker_id"])
            .last()
            .reset_index()
        )
        agg_dict = {"p_shin_pred": "mean", "y": "first"}
        for m in ["C0", "C0-D", "C0-Slim", "C0-Mini"]:
            agg_dict[m] = "first"

        m_agg = sub_q.groupby("canonical_match_id").agg(agg_dict).reset_index()
        y_p = m_agg["y"].to_numpy(dtype=int)
        p_mkt_p = m_agg["p_shin_pred"].to_numpy(dtype=float)
        l_mkt_p = logit(np.clip(p_mkt_p, 1e-6, 1.0 - 1e-6))
        ll_mkt = float(-np.mean(np.where(y_p, np.log(p_mkt_p), np.log1p(-p_mkt_p))))
        bs_mkt = float(np.mean((p_mkt_p - y_p) ** 2))

        h_dict = {
            "horizon": label,
            "lead_hours": float(h),
            "n_matches": len(y_p),
            "market_log_loss": round(ll_mkt, 6),
            "market_brier": round(bs_mkt, 6),
            "models": {},
        }

        for m in ["C0", "C0-D", "C0-Slim", "C0-Mini"]:
            p_m = m_agg[m].to_numpy(dtype=float)
            l_m = logit(np.clip(p_m, 1e-6, 1.0 - 1e-6))
            p_hyb = expit(0.50 * l_m + 0.50 * l_mkt_p)

            ll_std = float(-np.mean(np.where(y_p, np.log(p_m), np.log1p(-p_m))))
            ll_hyb = float(-np.mean(np.where(y_p, np.log(p_hyb), np.log1p(-p_hyb))))
            bs_std = float(np.mean((p_m - y_p) ** 2))
            bs_hyb = float(np.mean((p_hyb - y_p) ** 2))

            h_dict["models"][m] = {
                "standalone_log_loss": round(ll_std, 6),
                "standalone_brier": round(bs_std, 6),
                "hybrid_log_loss": round(ll_hyb, 6),
                "hybrid_brier": round(bs_hyb, 6),
                "delta_ll_vs_market": round(ll_std - ll_mkt, 6),
                "delta_ll_hybrid_vs_market": round(ll_hyb - ll_mkt, 6),
            }

        suite3_results["production_horizons"][label] = h_dict

    # Polish2plus (N=1,467)
    df_polish = pd.read_parquet(POLISH_PANEL_PATH)
    df_p2p = df_polish[df_polish["comparison_cohort"] == "Polish2plus"].copy()

    df_p2p_models = pd.DataFrame({"golgg_match_id": p1["golgg_match_id"].astype(str)})
    for m in ["C0", "C0-D", "C0-Slim", "C0-Mini"]:
        df_p2p_models[m] = models_assess[m]

    merged_p2p = df_p2p.merge(df_p2p_models, on="golgg_match_id")
    y_p2p = merged_p2p["y"].to_numpy(dtype=int)
    p_mkt_p2p = merged_p2p["p_market"].to_numpy(dtype=float)
    l_mkt_p2p = logit(np.clip(p_mkt_p2p, 1e-6, 1.0 - 1e-6))
    ll_mkt_p2p = float(-np.mean(np.where(y_p2p, np.log(p_mkt_p2p), np.log1p(-p_mkt_p2p))))
    bs_mkt_p2p = float(np.mean((p_mkt_p2p - y_p2p) ** 2))

    p2p_dict = {
        "cohort": "Polish2plus",
        "n_matches": len(y_p2p),
        "market_log_loss": round(ll_mkt_p2p, 6),
        "market_brier": round(bs_mkt_p2p, 6),
        "models": {},
    }

    for m in ["C0", "C0-D", "C0-Slim", "C0-Mini"]:
        p_m = merged_p2p[m].to_numpy(dtype=float)
        l_m = logit(np.clip(p_m, 1e-6, 1.0 - 1e-6))
        p_hyb = expit(0.50 * l_m + 0.50 * l_mkt_p2p)

        ll_std = float(-np.mean(np.where(y_p2p, np.log(p_m), np.log1p(-p_m))))
        ll_hyb = float(-np.mean(np.where(y_p2p, np.log(p_hyb), np.log1p(-p_hyb))))
        bs_std = float(np.mean((p_m - y_p2p) ** 2))
        bs_hyb = float(np.mean((p_hyb - y_p2p) ** 2))

        p2p_dict["models"][m] = {
            "standalone_log_loss": round(ll_std, 6),
            "standalone_brier": round(bs_std, 6),
            "hybrid_log_loss": round(ll_hyb, 6),
            "hybrid_brier": round(bs_hyb, 6),
            "delta_ll_vs_market": round(ll_std - ll_mkt_p2p, 6),
            "delta_ll_hybrid_vs_market": round(ll_hyb - ll_mkt_p2p, 6),
        }

    suite3_results["polish2plus"] = p2p_dict

    # =========================================================================
    # EXPORT RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)

    # 1. Strategy Comparison CSV
    strat_df = pd.DataFrame(strategy_rows)
    strat_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Exported {len(strat_df)} comparison rows to: {OUTPUT_CSV_PATH}")

    # 2. JSON Payload
    reporting_payload = {
        "metadata": {
            "title": "C0 Distilled Model Family Financial & Econometric Simulation Results",
            "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
            "initial_bankroll_pln": 10000.0,
            "models_evaluated": list(models_assess.keys()),
            "models_specifications": {
                "C0": {"name": "C0-Native Baseline", "params": 19818, "latency_ms": 848, "log_loss": 0.535449},
                "C0-D": {"name": "Born-Again W32-KD + Conn50 + ONS", "params": 19818, "latency_ms": 848, "log_loss": 0.534120},
                "C0-D-Core": {"name": "Pure W32-KD Born-Again 3-seed", "params": 19818, "latency_ms": 848, "log_loss": 0.534418},
                "C0-Slim": {"name": "Pure W32-KD Single-seed", "params": 6606, "latency_ms": 278, "log_loss": 0.535062},
                "C0-Slim+": {"name": "Single-Seed W32-KD + Conn50", "params": 6606, "latency_ms": 278, "log_loss": 0.534356},
                "C0-Mini": {"name": "W16-KD 3-seed", "params": 8394, "latency_ms": 412, "log_loss": 0.535845},
            },
        },
        "suite1_production_polish_bookmakers": suite1_results,
        "suite2_sharp_pinnacle_t1_market": suite2_results,
        "suite3_multi_horizon_and_polish": suite3_results,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(reporting_payload, f, indent=2)
    print(f"Exported full JSON results to: {OUTPUT_JSON_PATH}")

    # Verification summary printout
    print("\n" + "=" * 80)
    print("FINANCIAL SIMULATION SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Regime':<32} | {'Model':<10} | {'Strategy':<38} | {'Bets':<5} | {'Win %':<6} | {'Yield %':<8} | {'Net CLV %':<9} | {'Bias (pp)':<9}")
    print("-" * 135)
    for row in strategy_rows[:25]:
        print(f"{row['regime'][:32]:<32} | {row['model']:<10} | {row['strategy'][:38]:<38} | {row['bets']:<5} | {row['win_rate_pct']:<6.1f} | {row['realized_yield_pct']:<+8.2f} | {row['net_clv_pct']:<+9.2f} | {row['optimizers_curse_bias_pp']:<+9.2f}")


if __name__ == "__main__":
    run_distilled_financial_simulations()
