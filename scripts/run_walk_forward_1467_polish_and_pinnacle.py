#!/usr/bin/env python3
"""Strict Chronological Match-Level Walk-Forward Simulation (1 Decision Per Match).

Runs realistic chronological walk-forward simulations enforcing:
1. Strictly 1 decision per match at t_dec = t_start - 1h.
2. OperationalWallet capital locking: overlapping matches lock capital until t_settle = t_start + 3h.
3. Expanding Monthly Walk-Forward Temperature / Logit Calibration Audit (Jan -> Sep 2026).
4. Evaluation across:
   - All 1,467 Polish Bookmaker Matches (Polish2plus cohort, raw best book odds & implied odds)
   - All 1,668 Sharp Pinnacle T-1h Matches (under 0% International and 12% Polish tax)
   - Proper scoring across all 3,103 canonical 2026 matches.

Saves output JSON to:
data/08_reporting/c0_upgrade_and_distillation_20261017/walk_forward_1467_polish_and_pinnacle.json
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
OUTPUT_JSON = REPORT_DIR / "walk_forward_1467_polish_and_pinnacle.json"

POLISH_PANEL_PATH = PROJECT_ROOT / "data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/paired_probability_panel.parquet"
POLISH_PAIRS_PATH = PROJECT_ROOT / "data/07_model_output/oddsportal/oddsportal_polish_consensus_20261009_152919Z/polish_quote_model_pairs.parquet"
PINNACLE_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"
PINNACLE_TIMES_PATH = PROJECT_ROOT / "data/07_model_output/oddsportal/c0_bookmaker_horizons_20261009_103721Z/native_event_query_ledger.parquet"
DISTILLED_PREDS_PATH = REPORT_DIR / "year_anchor_distilled_predictions.parquet"
CAL_PREDS_PATH = PROJECT_ROOT / "data/07_model_output/a0_capacity/native_capacity_20261004_01/calibration_predictions.parquet"


GATE_CONFIGURATIONS = {
    "G0 (Old Full Stack + EXP-039 Veto)": {
        "desc": "Hybrid-50 + DRO (eps=0.025) + Guard B (<=0.85) + MaxOdds (<=4.00) + MaxEV (<=25%) + MinEV (>=5%) + EXP-039 Veto (<=0.10)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": True,
    },
    "G1 (Full Stack: Hybrid-50 + DRO eps=0.025 + GuardB<=0.85 + MaxOdds<=4.0 + MaxEV<=25% + MinEV>=5%)": {
        "desc": "Hybrid-50 + DRO (eps=0.025) + Guard B (<=0.85) + MaxOdds (<=4.00) + MaxEV (<=25%) + MinEV (>=5%)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": False,
    },
    "G3 (Turn OFF Max EV <= 25%)": {
        "desc": "Turn OFF Max EV <= 25% (MaxEV = Infinity)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": None, "min_ev": 0.05, "veto": False,
    },
    "G4 (Turn OFF Max Odds <= 4.0 & Max EV <= 25%)": {
        "desc": "Turn OFF BOTH Max Odds AND Max EV",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": None, "max_ev": None, "min_ev": 0.05, "veto": False,
    },
    "G5 (ONLY Wasserstein DRO eps=0.025, DRO>0% + Logit Disagreement GuardB <= 0.85 — NO MaxOdds, NO MaxEV, NO MinEV, NO Veto)": {
        "desc": "ONLY Wasserstein DRO eps=0.025, DRO>0% + Logit Disagreement GuardB <= 0.85",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": None, "max_ev": None, "min_ev": None, "veto": False,
    },
    "G7b (Minimalist: ONLY Hybrid-50 + Wasserstein DRO > 0% — All Other Gates OFF)": {
        "desc": "Minimalist: ONLY Hybrid-50 + Wasserstein DRO > 0% — All Other Gates OFF",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": None, "max_odds": None, "max_ev": None, "min_ev": None, "veto": False,
    },
    "G9 (Control: Turn OFF DRO eps=0.0, keep GuardB/MaxOdds/MaxEV/MinEV)": {
        "desc": "Turn OFF DRO eps=0.0 -> keep GuardB/MaxOdds/MaxEV/MinEV",
        "hybrid": True, "dro_eps": None, "use_dro": False, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": False,
    },
}


def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    """Compute standard probabilistic scoring metrics."""
    p_cl = np.clip(p_pred, 1e-12, 1.0 - 1e-12)
    ll = float(-np.mean(y_true * np.log(p_cl) + (1.0 - y_true) * np.log(1.0 - p_cl)))
    bs = float(np.mean((p_pred - y_true) ** 2))
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        mask = (p_pred >= bin_edges[i]) & (p_pred < bin_edges[i + 1] if i < 9 else p_pred <= bin_edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(p_pred)) * abs(float(p_pred[mask].mean()) - float(y_true[mask].mean()))
    return {"log_loss": round(ll, 6), "brier_score": round(bs, 6), "ece_10_bins": round(float(ece), 6)}


def compute_three_part_decomposition(
    bets: list[dict[str, Any]],
    initial_bankroll: float = 10000.0,
    clv_by_construction: bool = False,
    net_clv_reason: str | None = None,
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
            "claimed_ev_pct": None,
            "net_clv_pct": None,
            "n_clv_bets": 0,
            "n_missing_close": 0,
            "net_clv_reason": net_clv_reason or "no_bets_with_independent_close",
            "clv_by_construction": clv_by_construction,
        }

    df = pd.DataFrame(bets)
    if "net_clv" not in df:
        df["net_clv"] = np.nan
    else:
        df["net_clv"] = pd.to_numeric(df["net_clv"], errors="coerce")
    if clv_by_construction:
        df["net_clv"] = np.nan
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

    clv_df = df.loc[df["net_clv"].notna()]
    n_clv_bets = len(clv_df)
    n_missing_close = n_bets - n_clv_bets
    if n_clv_bets:
        clv_turnover = float(clv_df["stake"].sum())
        clv_weights = clv_df["stake"].to_numpy(dtype=float) / clv_turnover
        claimed_ev = float(np.sum(clv_weights * clv_df["ev_claimed"].to_numpy(dtype=float))) * 100.0
        net_clv = float(np.sum(clv_weights * clv_df["net_clv"].to_numpy(dtype=float))) * 100.0
        bias_b = claimed_ev - net_clv
        clv_yield = float(clv_df["profit"].sum() / clv_turnover) * 100.0
        luck_eta = clv_yield - net_clv
        resolved_clv_reason = None
    else:
        claimed_ev = net_clv = bias_b = luck_eta = None
        resolved_clv_reason = net_clv_reason or (
            "clv_by_construction" if clv_by_construction else "no_bets_with_independent_close"
        )

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
        "claimed_ev_pct": round(claimed_ev, 2) if claimed_ev is not None else None,
        "net_clv_pct": round(net_clv, 2) if net_clv is not None else None,
        "n_clv_bets": n_clv_bets,
        "n_missing_close": n_missing_close,
        "net_clv_reason": resolved_clv_reason,
        "clv_by_construction": clv_by_construction,
        "optimizers_curse_bias_pp": round(bias_b, 2) if bias_b is not None else None,
        "match_luck_noise_pp": round(luck_eta, 2) if luck_eta is not None else None,
    }


def run_chronological_simulation(
    df_matches: pd.DataFrame,
    cfg: dict[str, Any],
    initial_capital: float = 10000.0,
    tax_rate: float = 0.12,
    close_reference_available: bool = False,
    clv_by_construction: bool = False,
    net_clv_reason: str | None = None,
) -> dict[str, Any]:
    c = 1.0 - tax_rate
    free_capital = float(initial_capital)
    reserved_capital = 0.0

    p1_m = df_matches["p1"].to_numpy(dtype=float)
    p2_m = 1.0 - p1_m
    q1 = df_matches["q1"].to_numpy(dtype=float)
    q2 = 1.0 - q1
    o1 = df_matches["o1"].to_numpy(dtype=float)
    o2 = df_matches["o2"].to_numpy(dtype=float)
    y = df_matches["y"].to_numpy(dtype=int)
    p039 = df_matches["p039"].to_numpy(dtype=float) if "p039" in df_matches.columns else np.full(len(df_matches), np.nan)

    eps = 1e-12
    l_p1 = logit(np.clip(p1_m, eps, 1.0 - eps))
    l_q1 = logit(np.clip(q1, eps, 1.0 - eps))
    l_p2 = logit(np.clip(p2_m, eps, 1.0 - eps))
    l_q2 = logit(np.clip(q2, eps, 1.0 - eps))

    p1_h = expit(0.50 * l_p1 + 0.50 * l_q1)
    p2_h = expit(0.50 * l_p2 + 0.50 * l_q2)

    gap1 = np.abs(l_p1 - l_q1)
    gap2 = np.abs(l_p2 - l_q2)

    use_hybrid = cfg.get("hybrid", True)
    p1_eff = p1_h if use_hybrid else p1_m
    p2_eff = p2_h if use_hybrid else p2_m

    ev1 = p1_eff * c * o1 - 1.0
    ev2 = p2_eff * c * o2 - 1.0

    dro_eps = cfg.get("dro_eps", 0.025)
    ev1_dro = (p1_eff - dro_eps) * c * o1 - 1.0 if dro_eps is not None else ev1
    ev2_dro = (p2_eff - dro_eps) * c * o2 - 1.0 if dro_eps is not None else ev2

    max_odds = cfg.get("max_odds")
    min_ev = cfg.get("min_ev")
    max_ev = cfg.get("max_ev")
    guard_b = cfg.get("guard_b")
    use_dro = cfg.get("use_dro", True)
    min_dro_ev = cfg.get("min_dro_ev", 0.0)
    use_veto = cfg.get("veto", False)

    c1 = np.ones(len(df_matches), dtype=bool)
    c2 = np.ones(len(df_matches), dtype=bool)

    if min_ev is not None:
        c1 &= (ev1 >= min_ev)
        c2 &= (ev2 >= min_ev)
    if max_ev is not None:
        c1 &= (ev1 <= max_ev)
        c2 &= (ev2 <= max_ev)
    if max_odds is not None:
        c1 &= (o1 <= max_odds)
        c2 &= (o2 <= max_odds)
    if guard_b is not None:
        c1 &= (gap1 <= guard_b)
        c2 &= (gap2 <= guard_b)
    if use_dro:
        c1 &= (ev1_dro > min_dro_ev)
        c2 &= (ev2_dro > min_dro_ev)
    if use_veto:
        veto1 = np.isnan(p039) | (np.abs(p1_m - p039) <= 0.10)
        veto2 = np.isnan(p039) | (np.abs(p2_m - (1.0 - p039)) <= 0.10)
        c1 &= veto1
        c2 &= veto2

    event_heap: list[tuple[pd.Timestamp, int, int, dict[str, Any]]] = []
    uid = 0
    for i in range(len(df_matches)):
        side = 0
        if c1[i] and (not c2[i] or ev1[i] >= ev2[i]):
            close_ref = q1[i] if close_reference_available else None
            side, o_taken, p_claim, p_close, won, ev_claim = 1, o1[i], p1_eff[i], close_ref, int(y[i] == 1), ev1[i]
        elif c2[i]:
            close_ref = q1[i] if close_reference_available else None
            p_close = 1.0 - close_ref if close_ref is not None else None
            side, o_taken, p_claim, won, ev_claim = 2, o2[i], p2_eff[i], int(y[i] == 0), ev2[i]

        if side != 0:
            t_dec = df_matches["t_dec"].iloc[i]
            t_settle = df_matches["t_settle"].iloc[i]
            mid = df_matches["match_id"].iloc[i]
            data = {
                "match_id": mid,
                "t_dec": t_dec,
                "t_settle": t_settle,
                "side": side,
                "odds": o_taken,
                "p_claim": p_claim,
                "p_close": p_close,
                "won": won,
                "ev_claim": ev_claim,
            }
            # Event type 1 = decision
            heapq.heappush(event_heap, (t_dec, 1, uid, data))
            uid += 1

    settled_bets = []
    skipped_capital_count = 0
    max_concurrent_bets = 0
    current_open_bets = 0
    min_free_capital = free_capital

    while event_heap:
        t, ev_type, _, data = heapq.heappop(event_heap)

        if ev_type == 0:
            # Settlement event (type 0 executes before type 1 at same timestamp)
            stake = data["stake"]
            reserved_capital -= stake
            current_open_bets -= 1
            if data["won"]:
                payout = stake * data["odds"] * c
                profit = payout - stake
                free_capital += payout
            else:
                payout = 0.0
                profit = -stake

            settled_bets.append({
                "match_id": data["match_id"],
                "side": data["side"],
                "odds_taken": data["odds"],
                "p_claim": data["p_claim"],
                "p_close": data["p_close"],
                "won": data["won"],
                "ev_claimed": data["ev_claim"],
                "net_clv": (
                    c * data["odds"] * data["p_close"] - 1.0
                    if data["p_close"] is not None else None
                ),
                "stake": stake,
                "profit": profit,
                "bankroll": free_capital + reserved_capital,
                "free_capital": free_capital,
                "reserved_capital": reserved_capital,
                "placed_at": data["t_dec"].isoformat() if hasattr(data["t_dec"], "isoformat") else str(data["t_dec"]),
                "settled_at": t.isoformat() if hasattr(t, "isoformat") else str(t),
            })

        elif ev_type == 1:
            # Decision opportunity at t_dec
            b_net = data["odds"] * c - 1.0
            if b_net > 0:
                f_star = max(0.0, (b_net * data["p_claim"] - (1.0 - data["p_claim"])) / b_net)
                raw_stake = free_capital * 0.25 * f_star
                stake = min(150.0, max(20.0, raw_stake))
                if free_capital >= 20.0:
                    stake = min(stake, free_capital)
                    free_capital -= stake
                    reserved_capital += stake
                    current_open_bets += 1
                    max_concurrent_bets = max(max_concurrent_bets, current_open_bets)
                    min_free_capital = min(min_free_capital, free_capital)
                    data["stake"] = stake
                    # Schedule settlement event at t_settle (type 0)
                    heapq.heappush(event_heap, (data["t_settle"], 0, uid, data))
                    uid += 1
                else:
                    skipped_capital_count += 1

    decomp = compute_three_part_decomposition(
        settled_bets,
        initial_bankroll=initial_capital,
        clv_by_construction=clv_by_construction,
        net_clv_reason=net_clv_reason,
    )
    decomp["operational_stats"] = {
        "max_concurrent_open_bets": max_concurrent_bets,
        "min_free_capital_pln": round(min_free_capital, 2),
        "skipped_due_to_capital": skipped_capital_count,
        "final_free_capital_pln": round(free_capital, 2),
        "final_reserved_capital_pln": round(reserved_capital, 2),
        "final_equity_pln": round(free_capital + reserved_capital, 2),
    }
    return decomp


def main() -> None:
    print("=" * 80)
    print("STRICT CHRONOLOGICAL MATCH-LEVEL WALK-FORWARD SIMULATION (1467 POLISH & 1668 PINNACLE)")
    print("=" * 80)

    # 1. Load C0 distilled predictions (ASSESS 2026, N=3,103)
    preds = pd.read_parquet(DISTILLED_PREDS_PATH)
    preds["golgg_match_id"] = preds["golgg_match_id"].astype(str)
    preds["month"] = pd.to_datetime(preds["date"]).dt.to_period("M").astype(str)
    months = sorted(preds["month"].unique())
    y_assess = preds["y"].to_numpy(dtype=float)
    print(f"Loaded 2026 distilled predictions: {preds.shape} across {len(months)} months ({months[0]} to {months[-1]}).")

    # 2. Load 2025H2 CAL baseline (N=1,328)
    cap_cal = pd.read_parquet(CAL_PREDS_PATH)
    y_cal = cap_cal["y"].to_numpy(dtype=float)
    p_cal_c0 = cap_cal["w32_e12__full"].to_numpy(dtype=float)
    z_cal_c0 = logit(np.clip(p_cal_c0, 1e-12, 1.0 - 1e-12))
    print(f"Loaded 2025H2 CAL baseline: {cap_cal.shape} (N={len(y_cal)}).")

    # 3. Expanding Monthly Walk-Forward Calibration Audit
    print("\n" + "-" * 80)
    print("EXPANDING MONTHLY WALK-FORWARD CALIBRATION AUDIT (JAN -> SEP 2026)")
    print("-" * 80)

    models_list = ["p_c0_d_year_core__full", "p_c0_slim_year__full", "p_c0__full"]
    model_labels = {
        "p_c0_d_year_core__full": "C0-D-Core",
        "p_c0_slim_year__full": "C0-Slim",
        "p_c0__full": "C0",
    }

    calibration_audit: dict[str, Any] = {
        "description": "Comparison between Static Pre-2026 Calibration (frozen on 2025H2 CAL) vs Expanding Monthly Walk-Forward Calibration (refit at start of each month M on 2025H2 + months < M).",
        "months": months,
        "n_total_matches": len(y_assess),
        "models": {},
    }

    def fit_cal_slope(z: np.ndarray, y: np.ndarray) -> float:
        res = minimize(
            lambda p: -np.mean(y * np.log(expit(p[0] * z)) + (1.0 - y) * np.log(1.0 - expit(p[0] * z))),
            [1.0],
            method="L-BFGS-B",
            bounds=[(0.5, 2.0)],
        )
        return float(res.x[0])

    expanding_probs: dict[str, np.ndarray] = {}

    for m_col in models_list:
        m_name = model_labels[m_col]
        z_assess = logit(np.clip(preds[m_col].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12))

        # Static scoring
        static_metrics = compute_metrics(y_assess, preds[m_col].to_numpy(dtype=float))

        # Expanding calibration
        p_exp = np.zeros(len(preds), dtype=float)
        monthly_slopes: dict[str, dict[str, float]] = {}
        monthly_metrics: dict[str, dict[str, float]] = {}

        curr_z = z_cal_c0.copy()
        curr_y = y_cal.copy()

        for m in months:
            # Fit slope on all data available strictly before month m starts
            s_m = fit_cal_slope(curr_z, curr_y)
            monthly_slopes[m] = {"slope": round(s_m, 4), "temperature": round(1.0 / s_m, 4)}

            m_mask = (preds["month"] == m).to_numpy()
            p_exp[m_mask] = expit(s_m * z_assess[m_mask])

            # Measure monthly metrics
            m_metrics = compute_metrics(y_assess[m_mask], p_exp[m_mask])
            m_metrics_static = compute_metrics(y_assess[m_mask], preds[m_col].to_numpy()[m_mask])
            monthly_metrics[m] = {
                "n": int(m_mask.sum()),
                "static_log_loss": m_metrics_static["log_loss"],
                "static_brier": m_metrics_static["brier_score"],
                "static_ece": m_metrics_static["ece_10_bins"],
                "expanding_log_loss": m_metrics["log_loss"],
                "expanding_brier": m_metrics["brier_score"],
                "expanding_ece": m_metrics["ece_10_bins"],
                "slope": round(s_m, 4),
            }

            # Update expanding historical training set
            curr_z = np.concatenate([curr_z, z_assess[m_mask]])
            curr_y = np.concatenate([curr_y, y_assess[m_mask]])

        expanding_metrics = compute_metrics(y_assess, p_exp)
        expanding_probs[m_col] = p_exp
        preds[f"{m_col}__expanding"] = p_exp

        calibration_audit["models"][m_name] = {
            "static_metrics": static_metrics,
            "expanding_metrics": expanding_metrics,
            "monthly_slopes": monthly_slopes,
            "monthly_breakdown": monthly_metrics,
        }

        print(f"\nModel: {m_name}")
        print(f"  Static Pre-2026:   LogLoss={static_metrics['log_loss']:.6f} | Brier={static_metrics['brier_score']:.6f} | ECE={static_metrics['ece_10_bins']*100:.3f}%")
        print(f"  Expanding Monthly: LogLoss={expanding_metrics['log_loss']:.6f} | Brier={expanding_metrics['brier_score']:.6f} | ECE={expanding_metrics['ece_10_bins']*100:.3f}%")
        for m, s_info in monthly_slopes.items():
            print(f"    Month {m}: slope={s_info['slope']:.4f} (T={s_info['temperature']:.4f})")

    # 4. Prepare Polish Bookmaker Matches Cohort (N=1,467)
    print("\n" + "-" * 80)
    print("CONSTRUCTING POLISH BOOKMAKER COHORT (N=1,467)")
    print("-" * 80)
    pol_raw = pd.read_parquet(POLISH_PANEL_PATH)
    pol_df = pol_raw[pol_raw["comparison_cohort"] == "Polish2plus"].copy().reset_index(drop=True)
    assert len(pol_df) == 1467, f"Expected 1,467 Polish matches, got {len(pol_df)}"

    pol_df["golgg_match_id"] = pol_df["golgg_match_id"].astype(str)
    pol_df["match_id"] = pol_df["event_id"]
    pol_df["t_start"] = pd.to_datetime(pol_df["source_start_utc"], utc=True)
    pol_df["t_dec"] = pol_df["t_start"] - pd.Timedelta(hours=1)
    pol_df["t_settle"] = pol_df["t_start"] + pd.Timedelta(hours=3)
    pol_df["y"] = pol_df["y"].astype(int)
    pol_df["q1"] = pol_df["p_polish_consensus"].astype(float)
    pol_df["q2"] = 1.0 - pol_df["q1"]

    # Extract exact raw odds from Polish quote pairs
    pairs_raw = pd.read_parquet(POLISH_PAIRS_PATH)
    close_pairs = pairs_raw[(pairs_raw["point"] == "close") & (pairs_raw["event_id"].isin(pol_df["event_id"]))].copy()
    close_pairs["odds_team1"] = np.where(close_pairs["reviewed_to_native_flip"], close_pairs["odds2"], close_pairs["odds1"])
    close_pairs["odds_team2"] = np.where(close_pairs["reviewed_to_native_flip"], close_pairs["odds1"], close_pairs["odds2"])

    best_odds = close_pairs.groupby("event_id").agg(
        best_odds_team1=("odds_team1", "max"),
        best_odds_team2=("odds_team2", "max"),
        book_count=("book", "count"),
    ).reset_index()

    pol_df = pol_df.merge(best_odds, on="event_id")
    pol_df["o1_raw"] = pol_df["best_odds_team1"].astype(float)
    pol_df["o2_raw"] = pol_df["best_odds_team2"].astype(float)

    # Implied odds with 5.8% standard overround
    book_cols = ["p_betfan", "p_lv_bet", "p_sts", "p_superbet", "p_fuksiarz"]
    min_p1 = pol_df[book_cols].min(axis=1)
    max_p1 = pol_df[book_cols].max(axis=1)
    min_p2 = 1.0 - max_p1
    pol_df["o1_implied"] = 1.0 / (min_p1 * 1.058)
    pol_df["o2_implied"] = 1.0 / (min_p2 * 1.058)

    # Merge distilled predictions and expanding predictions
    pred_cols_to_merge = ["golgg_match_id"] + models_list + [f"{c}__expanding" for c in models_list]
    pol_df = pol_df.merge(preds[pred_cols_to_merge], on="golgg_match_id")
    print(f"Constructed Polish dataset: {pol_df.shape} (N={len(pol_df)}).")

    # 5. Prepare Pinnacle T-1h Matches Cohort (N=1,668)
    print("\n" + "-" * 80)
    print("CONSTRUCTING SHARP PINNACLE T-1H COHORT (N=1,668)")
    print("-" * 80)
    pin_raw = pd.read_parquet(PINNACLE_T1_PATH)
    assert len(pin_raw) == 1668, f"Expected 1,668 Pinnacle matches, got {len(pin_raw)}"

    pin_times = pd.read_parquet(PINNACLE_TIMES_PATH)
    pin_times["golgg_match_id"] = pin_times["golgg_match_id"].astype(str)
    pin_time_map = pin_times.groupby("golgg_match_id")["start_at"].first()

    pin_df = pin_raw.copy().reset_index(drop=True)
    pin_df["golgg_match_id"] = pin_df["golgg_match_id"].astype(str)
    pin_df["match_id"] = pin_df["golgg_match_id"]
    pin_df["start_at"] = pin_df["golgg_match_id"].map(pin_time_map)
    pin_df["t_start"] = pd.to_datetime(pin_df["start_at"], format="ISO8601", utc=True)
    pin_df["t_dec"] = pin_df["t_start"] - pd.Timedelta(hours=1)
    pin_df["t_settle"] = pin_df["t_start"] + pd.Timedelta(hours=3)
    pin_df["y"] = pin_df["y"].astype(int)
    pin_df["q1"] = pin_df["q_consensus"].astype(float)
    pin_df["q2"] = 1.0 - pin_df["q1"]
    pin_df["o1"] = 1.0 / pin_df["q1"]
    pin_df["o2"] = 1.0 / pin_df["q2"]
    pin_df["p039"] = pin_df["p_039"].astype(float)

    pin_df = pin_df.merge(preds[pred_cols_to_merge], on="golgg_match_id")
    print(f"Constructed Pinnacle dataset: {pin_df.shape} (N={len(pin_df)}).")

    # 6. Run Simulations Across All Regimes
    results_payload: dict[str, Any] = {
        "metadata": {
            "title": "Strict Chronological Match-Level Walk-Forward Simulation (1467 Polish & 1668 Pinnacle)",
            "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
            "initial_bankroll_pln": 10000.0,
            "staking_policy": "Quarter Kelly (f* = [b*p - (1-p)]/b * 0.25), capped at [20.00, 150.00] PLN",
            "settlement_capital_locking": "3 hours after kickoff (t_settle = t_start + 3h)",
            "decision_lead_time": "1 hour before kickoff (t_dec = t_start - 1h)",
            "proper_scoring_universe": "Canonical 2026 ASSESS (N=3,103 matches)",
            "models_evaluated": list(model_labels.values()),
        },
        "calibration_audit": calibration_audit,
        "cohorts": {},
    }

    # Simulation execution helper
    def execute_cohort_suite(
        df_base: pd.DataFrame,
        odds_prefix: str,
        tax_rate: float,
        cohort_name: str,
        clv_by_construction: bool,
        net_clv_reason: str,
    ) -> dict[str, Any]:
        suite_dict: dict[str, Any] = {}
        for m_col in models_list:
            m_name = model_labels[m_col]
            suite_dict[m_name] = {"static_pre_2026": {}, "expanding_monthly": {}}

            df_sim = df_base.copy()
            df_sim["p1"] = df_sim[m_col]
            if odds_prefix:
                df_sim["o1"] = df_sim[f"o1_{odds_prefix}"]
                df_sim["o2"] = df_sim[f"o2_{odds_prefix}"]

            for gname, cfg in GATE_CONFIGURATIONS.items():
                suite_dict[m_name]["static_pre_2026"][gname] = run_chronological_simulation(
                    df_sim,
                    cfg,
                    initial_capital=10000.0,
                    tax_rate=tax_rate,
                    close_reference_available=False,
                    clv_by_construction=clv_by_construction,
                    net_clv_reason=net_clv_reason,
                )

            df_sim["p1"] = df_sim[f"{m_col}__expanding"]
            for gname, cfg in GATE_CONFIGURATIONS.items():
                suite_dict[m_name]["expanding_monthly"][gname] = run_chronological_simulation(
                    df_sim,
                    cfg,
                    initial_capital=10000.0,
                    tax_rate=tax_rate,
                    close_reference_available=False,
                    clv_by_construction=clv_by_construction,
                    net_clv_reason=net_clv_reason,
                )

        return suite_dict

    # A. Polish 1,467 matches - Raw Best Odds (12% tax)
    print("\nRunning Polish Bookmaker Matches (N=1,467) on Exact Raw Best Odds...")
    results_payload["cohorts"]["polish_1467_raw_odds"] = {
        "name": "All 1,467 Polish Bookmaker Matches (Exact Raw Closing Best Odds, 12% Polish Tax)",
        "tax_rate": 0.12,
        "n_matches": 1467,
        "odds_mode": "raw_best_closing_odds_across_active_polish_books",
        "clv_by_construction": False,
        "net_clv_reason": "independent closing reference unavailable; placed odds use closing quotes",
        "models": execute_cohort_suite(
            pol_df, "raw", 0.12, "polish_1467_raw", False,
            "independent closing reference unavailable; placed odds use closing quotes",
        ),
    }

    # B. Polish 1,467 matches - Implied Odds with 5.8% overround (12% tax)
    print("Running Polish Bookmaker Matches (N=1,467) on Implied Odds (5.8% Overround)...")
    results_payload["cohorts"]["polish_1467_implied_odds"] = {
        "name": "All 1,467 Polish Bookmaker Matches (Implied Odds with 5.8% Overround, 12% Polish Tax)",
        "tax_rate": 0.12,
        "n_matches": 1467,
        "odds_mode": "implied_odds_overround_5_8_pct",
        "clv_by_construction": False,
        "net_clv_reason": "independent closing reference unavailable; implied odds derive from closing quotes",
        "models": execute_cohort_suite(
            pol_df, "implied", 0.12, "polish_1467_implied", False,
            "independent closing reference unavailable; implied odds derive from closing quotes",
        ),
    }

    # C. Pinnacle T-1h Matches (N=1,668) - 0% International Tax
    print("Running Sharp Pinnacle T-1h Matches (N=1,668) under 0% International Tax...")
    results_payload["cohorts"]["pinnacle_1668_international_0tax"] = {
        "name": "All 1,668 Sharp Pinnacle T-1h Matches (0% International Tax)",
        "tax_rate": 0.0,
        "n_matches": 1668,
        "odds_mode": "pinnacle_t1_consensus_odds",
        "clv_by_construction": True,
        "net_clv_reason": "same consensus probability defines both fair odds and CLV reference",
        "models": execute_cohort_suite(
            pin_df, "", 0.0, "pinnacle_1668_0tax", True,
            "same consensus probability defines both fair odds and CLV reference",
        ),
    }

    # D. Pinnacle T-1h Matches (N=1,668) - 12% Polish Tax
    print("Running Sharp Pinnacle T-1h Matches (N=1,668) under 12% Polish Tax...")
    results_payload["cohorts"]["pinnacle_1668_polish_12tax"] = {
        "name": "All 1,668 Sharp Pinnacle T-1h Matches (12% Polish Turnover Tax)",
        "tax_rate": 0.12,
        "n_matches": 1668,
        "odds_mode": "pinnacle_t1_consensus_odds",
        "clv_by_construction": True,
        "net_clv_reason": "same consensus probability defines both fair odds and CLV reference",
        "models": execute_cohort_suite(
            pin_df, "", 0.12, "pinnacle_1668_12tax", True,
            "same consensus probability defines both fair odds and CLV reference",
        ),
    }

    # Save to JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nSuccessfully generated and saved: {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
