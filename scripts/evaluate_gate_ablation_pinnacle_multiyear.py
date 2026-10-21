#!/usr/bin/env python3
"""Rigorously evaluate turning off betting gates (G1 through G9) across:
1. Cohort 1: Sharp Pinnacle T-1h (N=1,668) under 12% Polish Tax
2. Cohort 2: Sharp Pinnacle T-1h (N=1,668) under 0% International Tax
3. Cohort 3: Multi-Year Historical OddsPortal Benchmark under 0% and 12% Tax:
   - 3A: OddsPortal Joint Consensus Benchmark (N=2,023 series) for C0-D & C0-Slim
   - 3B: OddsPortal Deduplicated Match Benchmark (N=920 matches) for C0-D & C0-Slim
   - 3C: OddsPortal Multi-Year Historical Macroeconomic Benchmark (2021-2026, N=9,872 matches)

Saves output JSON to:
data/08_reporting/c0_upgrade_and_distillation_20261017/gate_ablation_pinnacle_multiyear.json
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
OUTPUT_JSON = REPORT_DIR / "gate_ablation_pinnacle_multiyear.json"

DISTILLED_PATH = REPORT_DIR / "year_anchor_distilled_predictions.parquet"
PINNACLE_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"
JOINT_SCORING_PATH = PROJECT_ROOT / "data/07_model_output/oddsportal/c0_thesis_books_joint_20261009_115922Z/joint_scoring_cohorts.parquet"
ODDS_PATH = PROJECT_ROOT / "data/odds.csv"
MODEL_PRED_PATH = PROJECT_ROOT / "data/model_predictions_2020_2026.parquet"


CONFIGURATIONS = {
    "G0 (Full Stack + Veto)": {
        "desc": "Hybrid-50 + DRO (eps=0.025) + Guard B (<=0.85) + MaxOdds (<=4.00) + MaxEV (<=25%) + MinEV (>=5%) + EXP-039 Veto (<=0.10)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": True,
    },
    "G1 (Full Stack)": {
        "desc": "Hybrid-50 + DRO (eps=0.025) + Guard B (<=0.85) + MaxOdds (<=4.00) + MaxEV (<=25%) + MinEV (>=5%)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": False,
    },
    "G2 (Turn OFF Max Odds)": {
        "desc": "Turn OFF Max Odds <= 4.00 (MaxOdds = Infinity)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": None, "max_ev": 0.25, "min_ev": 0.05, "veto": False,
    },
    "G3 (Turn OFF Max EV)": {
        "desc": "Turn OFF Max EV <= 25% (MaxEV = Infinity)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": None, "min_ev": 0.05, "veto": False,
    },
    "G4 (Turn OFF Max Odds & Max EV)": {
        "desc": "Turn OFF BOTH Max Odds AND Max EV",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": None, "max_ev": None, "min_ev": 0.05, "veto": False,
    },
    "G5a (Require DRO_EV > 0% only, Guard B ON)": {
        "desc": "Turn OFF Min EV >= 5% -> require only DRO_EV > 0% (Guard B ON)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": None, "veto": False,
    },
    "G5b (Require DRO_EV >= 2% only, Guard B ON)": {
        "desc": "Turn OFF Min EV >= 5% -> require DRO_EV >= 2% (Guard B ON)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.02,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": None, "veto": False,
    },
    "G5c (Require DRO_EV > 0% only, Guard B OFF)": {
        "desc": "Turn OFF Min EV >= 5% -> require only DRO_EV > 0% (Guard B OFF)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": None, "max_odds": 4.0, "max_ev": 0.25, "min_ev": None, "veto": False,
    },
    "G5d (Require DRO_EV >= 2% only, Guard B OFF)": {
        "desc": "Turn OFF Min EV >= 5% -> require DRO_EV >= 2% (Guard B OFF)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.02,
        "guard_b": None, "max_odds": 4.0, "max_ev": 0.25, "min_ev": None, "veto": False,
    },
    "G6 (Turn OFF Guard B)": {
        "desc": "Turn OFF Guard B <= 0.85 (No divergence gate)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": None, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": False,
    },
    "G7a (Minimalist: Hybrid+DRO, EV>=5% & DRO_EV>0%)": {
        "desc": "Minimalist Hybrid+DRO: Turn OFF Guard B, MaxOdds, MaxEV (EV>=5% & DRO_EV>0%)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": None, "max_odds": None, "max_ev": None, "min_ev": 0.05, "veto": False,
    },
    "G7b (Minimalist: Hybrid+DRO, DRO_EV > 0% only)": {
        "desc": "Minimalist Hybrid+DRO: Turn OFF Guard B, MaxOdds, MaxEV, MinEV (DRO_EV > 0% only)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": None, "max_odds": None, "max_ev": None, "min_ev": None, "veto": False,
    },
    "G7c (Minimalist: Hybrid+DRO, DRO_EV >= 2% only)": {
        "desc": "Minimalist Hybrid+DRO: Turn OFF Guard B, MaxOdds, MaxEV, MinEV (DRO_EV >= 2% only)",
        "hybrid": True, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.02,
        "guard_b": None, "max_odds": None, "max_ev": None, "min_ev": None, "veto": False,
    },
    "G8a (Pure Model + DRO Full Stack)": {
        "desc": "Turn OFF Hybrid-50 Shrinkage -> Pure Model + DRO Full Stack",
        "hybrid": False, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": False,
    },
    "G8b (Pure Model + DRO Minimalist)": {
        "desc": "Turn OFF Hybrid-50 Shrinkage -> Pure Model + DRO Minimalist (DRO_EV > 0% only)",
        "hybrid": False, "dro_eps": 0.025, "use_dro": True, "min_dro_ev": 0.0,
        "guard_b": None, "max_odds": None, "max_ev": None, "min_ev": None, "veto": False,
    },
    "G9 (Turn OFF DRO eps=0.025, keep other gates)": {
        "desc": "Turn OFF DRO (eps=0.025) -> keep all other heuristic gates ON",
        "hybrid": True, "dro_eps": None, "use_dro": False, "min_dro_ev": 0.0,
        "guard_b": 0.85, "max_odds": 4.0, "max_ev": 0.25, "min_ev": 0.05, "veto": False,
    },
}


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
    }


def execute_simulation(
    df: pd.DataFrame,
    m_col: str,
    q_col: str,
    tax_rate: float,
    cfg: dict[str, Any],
    initial_bankroll: float = 10000.0,
) -> dict[str, Any]:
    c = 1.0 - tax_rate
    p1_m = df[m_col].to_numpy(dtype=float)
    p2_m = 1.0 - p1_m
    q1 = df[q_col].to_numpy(dtype=float)
    q2 = 1.0 - q1
    o1 = 1.0 / q1
    o2 = 1.0 / q2
    y = df["y"].to_numpy(dtype=int)
    p039 = df["p_039"].to_numpy(dtype=float) if "p_039" in df.columns else np.full(len(df), np.nan)

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

    c1 = np.ones(len(df), dtype=bool)
    c2 = np.ones(len(df), dtype=bool)

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

    sim_bets = []
    bankroll = initial_bankroll

    for i in range(len(df)):
        side = 0
        if c1[i] and (not c2[i] or ev1[i] >= ev2[i]):
            side, o_taken, p_claim, p_close, won, ev_claim = 1, o1[i], p1_eff[i], q1[i], int(y[i] == 1), ev1[i]
        elif c2[i]:
            side, o_taken, p_claim, p_close, won, ev_claim = 2, o2[i], p2_eff[i], q2[i], int(y[i] == 0), ev2[i]

        if side == 0:
            continue

        b_net = o_taken * c - 1.0
        f_star = max(0.0, (b_net * p_claim - (1.0 - p_claim)) / b_net) if b_net > 0 else 0.0
        stake = min(150.0, max(20.0, bankroll * 0.25 * f_star))

        if stake <= 0.0 or stake > bankroll:
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

    return compute_three_part_decomposition(sim_bets, initial_bankroll=initial_bankroll)


def main():
    print("=" * 80)
    print("STARTING GATE ABLATION EXPERIMENTS (PINNACLE T-1H & MULTI-YEAR BENCHMARKS)")
    print("=" * 80)

    # 1. Load C0 distilled models
    df_dist = pd.read_parquet(DISTILLED_PATH)
    df_dist["golgg_match_id"] = df_dist["golgg_match_id"].astype(str)
    print(f"Loaded year anchor distilled predictions: {df_dist.shape}")

    # 2. Load Pinnacle T-1h
    df_t1 = pd.read_parquet(PINNACLE_T1_PATH)
    df_t1["golgg_match_id"] = df_t1["golgg_match_id"].astype(str)
    m_pinnacle = df_t1.merge(
        df_dist[["golgg_match_id", "p_c0_d_year_core__full", "p_c0_slim_year__full", "p_c0__full"]],
        on="golgg_match_id",
    )
    m_pinnacle["date"] = pd.to_datetime(m_pinnacle["date"])
    m_pinnacle = m_pinnacle.sort_values(by=["date", "golgg_match_id"]).reset_index(drop=True)
    print(f"Constructed Pinnacle T-1h cohort: {len(m_pinnacle)} series.")

    # 3. Load Joint Scoring Cohorts (OddsPortal multi-bookmaker benchmark)
    df_joint = pd.read_parquet(JOINT_SCORING_PATH)
    df_joint["golgg_match_id"] = df_joint["golgg_match_id"].astype(str)
    m_joint = df_joint.merge(
        df_dist[["golgg_match_id", "p_c0_d_year_core__full", "p_c0_slim_year__full", "p_c0__full"]],
        on="golgg_match_id",
    )
    m_joint["date"] = pd.to_datetime(m_joint["date"])
    m_joint = m_joint.sort_values(by=["date", "golgg_match_id"]).reset_index(drop=True)
    print(f"Constructed OddsPortal Joint Quotes cohort: {len(m_joint)} series.")

    # 3B. Deduplicated match cohort
    m_joint_dedup = m_joint.groupby("golgg_match_id").first().reset_index()
    m_joint_dedup["date"] = pd.to_datetime(m_joint_dedup["date"])
    m_joint_dedup = m_joint_dedup.sort_values(by=["date", "golgg_match_id"]).reset_index(drop=True)
    print(f"Constructed OddsPortal Deduplicated Match cohort: {len(m_joint_dedup)} series.")

    # 4. Multi-Year Historical Benchmark (2021–2026)
    df_odds = pd.read_csv(ODDS_PATH)
    df_mp = pd.read_parquet(MODEL_PRED_PATH)
    df_odds["golgg_match_id"] = df_odds["golgg_match_id"].astype(str)
    df_mp["golgg_match_id"] = df_mp["golgg_match_id"].astype(str)

    m_my = df_odds.merge(df_mp, on="golgg_match_id")
    m_my["date"] = pd.to_datetime(m_my["date"])
    m_my = m_my[m_my["date"].dt.year >= 2021].sort_values("date").reset_index(drop=True)

    def norm(s):
        return str(s).lower().strip().replace(" ", "").replace("-", "").replace("_", "")

    t1_m = m_my["team1_name"].apply(norm)
    t2_m = m_my["team2_name"].apply(norm)
    t1_o = m_my["golgg_team1"].apply(norm)
    t2_o = m_my["golgg_team2"].apply(norm)
    swap = (t1_m == t2_o) | (t2_m == t1_o)

    o1 = np.where(swap, m_my["avg_odds_away"], m_my["avg_odds_home"])
    o2 = np.where(swap, m_my["avg_odds_home"], m_my["avg_odds_away"])

    valid = (o1 > 1.0) & (o2 > 1.0) & ~np.isnan(o1) & ~np.isnan(o2)
    m_my = m_my[valid].copy().reset_index(drop=True)
    o1 = o1[valid]
    o2 = o2[valid]

    inv1 = 1.0 / o1
    inv2 = 1.0 / o2
    overround = inv1 + inv2 - 1.0
    q1 = np.clip(inv1 - overround / 2.0, 1e-6, 1.0 - 1e-6)
    m_my["q_consensus"] = q1
    m_my["y"] = m_my["y_true"]
    m_my["p_benchmark_model"] = np.where(np.isnan(m_my["prob_exp039"]), 0.5, m_my["prob_exp039"])
    print(f"Constructed Multi-Year Macroeconomic Cohort (2021-2026): {len(m_my)} fixtures.")

    output_data: dict[str, Any] = {
        "metadata": {
            "created_at": "2026-10-19",
            "models_tested": ["C0-D (p_c0_d_year_core__full)", "C0-Slim (p_c0_slim_year__full)"],
            "configurations": {k: v["desc"] for k, v in CONFIGURATIONS.items()},
        },
        "cohort1_pinnacle_12pct_tax": {},
        "cohort2_pinnacle_0pct_tax": {},
        "cohort3_oddsportal_benchmark": {
            "3A_joint_consensus_quotes": {"0pct_tax": {}, "12pct_tax": {}},
            "3B_joint_deduplicated_matches": {"0pct_tax": {}, "12pct_tax": {}},
            "3C_multiyear_macroeconomic_2021_2026": {"0pct_tax": {}, "12pct_tax": {}},
        },
    }

    # Run Cohort 1 & 2: Pinnacle T-1h
    models_to_test = [
        ("C0-D", "p_c0_d_year_core__full"),
        ("C0-Slim", "p_c0_slim_year__full"),
        ("C0-Baseline", "p_c0__full"),
    ]

    for m_label, col in models_to_test:
        output_data["cohort1_pinnacle_12pct_tax"][m_label] = {}
        output_data["cohort2_pinnacle_0pct_tax"][m_label] = {}

        for cfg_name, cfg in CONFIGURATIONS.items():
            r12 = execute_simulation(m_pinnacle, col, "q_consensus", 0.12, cfg)
            r0 = execute_simulation(m_pinnacle, col, "q_consensus", 0.00, cfg)
            output_data["cohort1_pinnacle_12pct_tax"][m_label][cfg_name] = r12
            output_data["cohort2_pinnacle_0pct_tax"][m_label][cfg_name] = r0

    # Run Cohort 3A & 3B: OddsPortal Joint Consensus
    for m_label, col in models_to_test:
        output_data["cohort3_oddsportal_benchmark"]["3A_joint_consensus_quotes"]["0pct_tax"][m_label] = {}
        output_data["cohort3_oddsportal_benchmark"]["3A_joint_consensus_quotes"]["12pct_tax"][m_label] = {}
        output_data["cohort3_oddsportal_benchmark"]["3B_joint_deduplicated_matches"]["0pct_tax"][m_label] = {}
        output_data["cohort3_oddsportal_benchmark"]["3B_joint_deduplicated_matches"]["12pct_tax"][m_label] = {}

        for cfg_name, cfg in CONFIGURATIONS.items():
            r3a_0 = execute_simulation(m_joint, col, "p_consensus", 0.00, cfg)
            r3a_12 = execute_simulation(m_joint, col, "p_consensus", 0.12, cfg)
            r3b_0 = execute_simulation(m_joint_dedup, col, "p_consensus", 0.00, cfg)
            r3b_12 = execute_simulation(m_joint_dedup, col, "p_consensus", 0.12, cfg)

            output_data["cohort3_oddsportal_benchmark"]["3A_joint_consensus_quotes"]["0pct_tax"][m_label][cfg_name] = r3a_0
            output_data["cohort3_oddsportal_benchmark"]["3A_joint_consensus_quotes"]["12pct_tax"][m_label][cfg_name] = r3a_12
            output_data["cohort3_oddsportal_benchmark"]["3B_joint_deduplicated_matches"]["0pct_tax"][m_label][cfg_name] = r3b_0
            output_data["cohort3_oddsportal_benchmark"]["3B_joint_deduplicated_matches"]["12pct_tax"][m_label][cfg_name] = r3b_12

    # Run Cohort 3C: Multi-Year Macroeconomic Benchmark
    output_data["cohort3_oddsportal_benchmark"]["3C_multiyear_macroeconomic_2021_2026"]["0pct_tax"]["MultiYear_Benchmark_Model"] = {}
    output_data["cohort3_oddsportal_benchmark"]["3C_multiyear_macroeconomic_2021_2026"]["12pct_tax"]["MultiYear_Benchmark_Model"] = {}

    for cfg_name, cfg in CONFIGURATIONS.items():
        r3c_0 = execute_simulation(m_my, "p_benchmark_model", "q_consensus", 0.00, cfg)
        r3c_12 = execute_simulation(m_my, "p_benchmark_model", "q_consensus", 0.12, cfg)
        output_data["cohort3_oddsportal_benchmark"]["3C_multiyear_macroeconomic_2021_2026"]["0pct_tax"]["MultiYear_Benchmark_Model"][cfg_name] = r3c_0
        output_data["cohort3_oddsportal_benchmark"]["3C_multiyear_macroeconomic_2021_2026"]["12pct_tax"]["MultiYear_Benchmark_Model"][cfg_name] = r3c_12

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"\nSaved complete structured ablation results to: {OUTPUT_JSON}")
    print(f"File size: {OUTPUT_JSON.stat().st_size:,} bytes.")


if __name__ == "__main__":
    main()
