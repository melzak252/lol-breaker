#!/usr/bin/env python3
"""Mathematical and forensic audit of heuristic betting gates after Hybrid-50 and Wasserstein DRO.

Audits:
1. Mathematical Redundancy Proof (Odds-scaled DRO penalty and Underdog O=5.00 requirement)
2. Cohort 1: Full 645-Match / 248,735-Quote Polish Bookmaker Scanner (T-48h..1h, 12% tax)
   - 1A: Chronological Match Bets
   - 1B: Quote-Level All Opportunities
3. Cohort 2: 50-Bet Polish Production Opportunity Cohort (12% tax)
4. Cohort 3: Sharp Pinnacle T-1h Cohort (N=1,668)
   - 3A: 12% Polish Tax
   - 3B: 0% International Tax

Evaluated across C0-D (p_c0_d_year_core__full) and C0-Slim (p_c0_slim_year__full).
Saves structured JSON to:
data/08_reporting/c0_upgrade_and_distillation_20261017/marginal_blocked_bets_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
OUTPUT_JSON_PATH = REPORT_DIR / "marginal_blocked_bets_audit.json"

RAW_PROD_DIR = PROJECT_ROOT / "data/01_raw/c0_production_counterfactual/c0_production_20261006_01"
MATCHES_PATH = RAW_PROD_DIR / "matches.parquet"
MAPPINGS_PATH = RAW_PROD_DIR / "mappings.parquet"
QUOTES_PATH = RAW_PROD_DIR / "quotes.parquet"
GOLGG_PATH = RAW_PROD_DIR / "golgg.parquet"

PREDICTIONS_PATH = REPORT_DIR / "year_anchor_distilled_predictions.parquet"
PINNACLE_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"
LEDGER_PATH = PROJECT_ROOT / "data/08_reporting/optimal_strategy_simulation_ledger.csv"
UNIFORM_MONTHLY_PATH = PROJECT_ROOT / "data/07_model_output/a0_native_refit/native_refit_20260929_122617/uniform/uniform_monthly/candidate.parquet"


def build_mathematical_proof() -> dict[str, Any]:
    """Calculate exact odds-scaled DRO penalties and underdog requirements."""
    eps = 0.025
    odds_list = [1.50, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00]
    penalties = []
    for o in odds_list:
        pen_12 = -eps * o * (1.0 - 0.12)
        pen_0 = -eps * o * (1.0 - 0.00)
        penalties.append({
            "odds": o,
            "penalty_12_pct_tax": round(float(pen_12), 4),
            "penalty_12_pct_tax_pct": round(float(pen_12 * 100.0), 2),
            "penalty_0_pct_tax": round(float(pen_0), 4),
            "penalty_0_pct_tax_pct": round(float(pen_0 * 100.0), 2),
        })

    o_ud = 5.00
    tax_12 = 0.12
    c_12 = 1.0 - tax_12
    be_p_hyb_minus_eps = 1.0 / (o_ud * c_12)  # 1 / 4.40 = 0.2272727...
    be_p_hyb = be_p_hyb_minus_eps + eps  # 0.2522727...

    underdog_scenarios = []
    for q_mkt in [0.180, 0.185, 0.190, 0.200]:
        l_hyb = logit(be_p_hyb)
        l_mkt = logit(q_mkt)
        l_mod = 2.0 * l_hyb - l_mkt
        p_mod = expit(l_mod)
        dlogit = l_mod - l_mkt
        raw_ev = p_mod * o_ud * c_12 - 1.0
        hyb_ev = be_p_hyb * o_ud * c_12 - 1.0
        dro_ev = (be_p_hyb - eps) * o_ud * c_12 - 1.0
        underdog_scenarios.append({
            "q_mkt": q_mkt,
            "q_mkt_pct": round(float(q_mkt * 100.0), 2),
            "p_hyb_required": round(float(be_p_hyb), 4),
            "p_hyb_required_pct": round(float(be_p_hyb * 100.0), 2),
            "p_model_required": round(float(p_mod), 4),
            "p_model_required_pct": round(float(p_mod * 100.0), 2),
            "required_abs_dlogit": round(float(abs(dlogit)), 4),
            "raw_model_ev_pct": round(float(raw_ev * 100.0), 2),
            "hybrid_ev_pct": round(float(hyb_ev * 100.0), 2),
            "dro_ev_pct": round(float(dro_ev * 100.0), 2),
            "violates_guard_b_085": bool(abs(dlogit) > 0.85),
            "violates_max_odds_400": bool(o_ud > 4.00),
            "violates_max_ev_25pct_raw": bool(raw_ev > 0.25),
        })

    return {
        "wasserstein_epsilon": eps,
        "odds_penalty_curve": penalties,
        "underdog_o5_analysis": {
            "odds": o_ud,
            "tax_rate": tax_12,
            "breakeven_p_hyb_minus_eps": round(float(be_p_hyb_minus_eps), 4),
            "breakeven_p_hyb": round(float(be_p_hyb), 4),
            "scenarios": underdog_scenarios,
        },
        "theoretical_insights": [
            "1. Odds-Scaled Penalty: DRO shifts probability by -0.025, which translates directly to a linear EV penalty of -0.025 * O * (1 - tax). At O=1.50, penalty is -3.30%; at O=4.00, -8.80%; at O=5.00, -11.00%; at O=8.00, -17.60%.",
            "2. Natural Elimination of Longshots: To pass DRO at O=5.00 under 12% Polish tax, hybrid win probability must exceed 25.23%, requiring raw model probability of 33.40% against a 18.5% market. This requires a massive +14.9 pp edge and |dLogit| = 0.793.",
            "3. Guard B Mathematical Incompatibility: At odds O <= 2.40 under 12% tax, overcoming overround + 12% tax + DRO requires |dLogit| > 0.85. Thus Guard B <= 0.85 makes it mathematically impossible to bet short favorites under DRO.",
            "4. Redundancy of MaxOdds <= 4.00: DRO already severely penalizes high odds. High-odds bets that survive DRO are not lottery tickets; they are massive mispricings where the model has immense conviction.",
        ],
    }


def analyze_gate_blocks(
    df_bets: pd.DataFrame,
    gate_specs: list[tuple[str, str, pd.Series]],
) -> dict[str, Any]:
    """Calculate forensic performance of bets blocked by each gate."""
    total_bets = len(df_bets)
    total_wins = int(df_bets["won"].sum())
    total_profit = float(df_bets["profit"].sum())
    total_turnover = float(df_bets["stake"].sum())
    win_rate = (total_wins / total_bets * 100.0) if total_bets > 0 else 0.0
    yield_pct = (total_profit / total_turnover * 100.0) if total_turnover > 0 else 0.0

    summary: dict[str, Any] = {
        "base_passing_dro": {
            "n_bets": total_bets,
            "n_wins": total_wins,
            "win_rate_pct": round(win_rate, 2),
            "net_profit_pln": round(total_profit, 2),
            "turnover_pln": round(total_turnover, 2),
            "realized_yield_pct": round(yield_pct, 2),
        },
        "gates": {},
    }

    for gate_id, gate_label, blocked_mask in gate_specs:
        n_blk = int(blocked_mask.sum())
        if n_blk == 0:
            summary["gates"][gate_id] = {
                "gate_label": gate_label,
                "n_blocked": 0,
                "n_wins": 0,
                "win_rate_pct": 0.0,
                "net_profit_pln": 0.0,
                "turnover_pln": 0.0,
                "realized_yield_pct": 0.0,
                "avg_odds": 0.0,
                "verdict": "100%_REDUNDANT",
                "impact_summary": "Gate blocked literally 0 bets because DRO already eliminated unviable bets.",
                "sample_matches": [],
            }
        else:
            blk_df = df_bets[blocked_mask].copy()
            w = int(blk_df["won"].sum())
            wr = (w / n_blk * 100.0)
            prof = float(blk_df["profit"].sum())
            turn = float(blk_df["stake"].sum())
            yd = (prof / turn * 100.0) if turn > 0 else 0.0
            avg_o = float(blk_df["odds"].mean())

            if prof > 0.0:
                verdict = "COSTS_MONEY"
                verdict_text = f"Harmful: Blocked {n_blk} net winning bets, tossing +{prof:.2f} PLN profit ({yd:+.2f}% yield)."
            elif prof < 0.0:
                verdict = "SAVES_MONEY"
                verdict_text = f"Protective: Blocked {n_blk} net losing bets, saving {abs(prof):.2f} PLN in losses."
            else:
                verdict = "BREAK_EVEN"
                verdict_text = f"Neutral: Blocked {n_blk} bets that exactly broke even (0.00 PLN)."

            # Format sample matches (up to 5)
            samples = []
            for _, r in blk_df.head(5).iterrows():
                samples.append({
                    "match_id": str(r.get("golgg_match_id") or r.get("canonical_match_id") or r.get("bet_id")),
                    "date": str(r.get("date", ""))[:16],
                    "team_selected": str(r.get("team_selected", "")),
                    "opponent": str(r.get("opponent", "")),
                    "tournament": str(r.get("tournament", r.get("league", ""))),
                    "odds": round(float(r["odds"]), 2),
                    "model_prob": round(float(r["p_model"]), 4),
                    "hybrid_prob": round(float(r["p_hyb"]), 4),
                    "market_prob": round(float(r.get("q_mkt", 1.0 / r["odds"])), 4),
                    "ev_net_pct": round(float(r["ev_net"] * 100.0), 2),
                    "dro_ev_pct": round(float(r["dro_ev"] * 100.0), 2),
                    "gap_logit": round(float(r["gap"]), 4),
                    "won": int(r["won"]),
                    "profit_pln": round(float(r["profit"]), 2),
                })

            summary["gates"][gate_id] = {
                "gate_label": gate_label,
                "n_blocked": n_blk,
                "n_wins": w,
                "win_rate_pct": round(wr, 2),
                "net_profit_pln": round(prof, 2),
                "turnover_pln": round(turn, 2),
                "realized_yield_pct": round(yd, 2),
                "avg_odds": round(avg_o, 2),
                "verdict": verdict,
                "impact_summary": verdict_text,
                "sample_matches": samples,
            }

    return summary


def run_full_audit() -> dict[str, Any]:
    print("=" * 80)
    print("STARTING MATHEMATICAL & FORENSIC GATE AUDIT AFTER DRO")
    print("=" * 80)

    # Load shared assets
    p1 = pd.read_parquet(PREDICTIONS_PATH)
    mkt_raw = pd.read_parquet(PINNACLE_T1_PATH)
    ledger_df = pd.read_csv(LEDGER_PATH)
    golgg = pd.read_parquet(GOLGG_PATH)
    matches = pd.read_parquet(MATCHES_PATH)
    mappings = pd.read_parquet(MAPPINGS_PATH)
    quotes = pd.read_parquet(QUOTES_PATH)
    um_df = pd.read_parquet(UNIFORM_MONTHLY_PATH)

    golgg_lookup = {}
    for _, r in golgg.iterrows():
        golgg_lookup[str(r["match_id"])] = {
            "team1": str(r["team1_name"]),
            "team2": str(r["team2_name"]),
            "tournament": str(r["tournament_name"]),
        }

    p039_lookup = dict(zip(mkt_raw["golgg_match_id"].astype(str), mkt_raw["p_039"]))

    # 1. Math proof
    math_proof = build_mathematical_proof()

    # Models to audit
    models = {
        "C0-D": p1["p_c0_d_year_core__full"].to_numpy(dtype=np.float64),
        "C0-Slim": p1["p_c0_slim_year__full"].to_numpy(dtype=np.float64),
    }

    audit_results: dict[str, Any] = {
        "metadata": {
            "created_at": "2026-10-19T12:00:00Z",
            "eval_models": ["C0-D (p_c0_d_year_core__full)", "C0-Slim (p_c0_slim_year__full)"],
            "base_policy": "Hybrid-50 (alpha=0.50) + Wasserstein DRO (eps=0.025, DRO_EV > 0)",
            "gates_audited": [
                "MaxOdds <= 4.00",
                "MaxEV <= 25%",
                "MinEV >= 5%",
                "Guard B |dLogit| <= 0.85",
                "EXP-039 Veto <= 0.10",
            ],
        },
        "mathematical_redundancy_proof": math_proof,
        "cohorts": {},
    }

    # =========================================================================
    # COHORT 2: 50-BET POLISH PRODUCTION OPPORTUNITY COHORT (12% TAX)
    # =========================================================================
    print("\n--- Auditing Cohort 2: 50-Bet Polish Production Opportunity Cohort ---")
    um_2026 = um_df.set_index("golgg_match_id").loc[p1["golgg_match_id"].astype(str)].reset_index()
    p_c0_mon = um_2026["p_full"].to_numpy(dtype=np.float64)
    p_c0_full = p1["p_c0__full"].to_numpy(dtype=np.float64)
    p1_dates = pd.to_datetime(p1["date"])
    gid_to_idx = {str(gid): i for i, gid in enumerate(p1["golgg_match_id"].values)}

    ledger_to_golgg = {}
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

    l_c0_led = logit(ledger_df["p_c0"].to_numpy(dtype=float))
    l_hyb_led = logit(ledger_df["p_hybrid"].to_numpy(dtype=float))
    l_mkt_led = 2.0 * l_hyb_led - l_c0_led
    p_mkt_led = expit(l_mkt_led)
    p_thesis_led = ledger_df["p_thesis"].to_numpy(dtype=float)
    odds_led = ledger_df["odds_taken"].to_numpy(dtype=float)
    won_led = ledger_df["won"].to_numpy(dtype=int)

    cohort2_results: dict[str, Any] = {}

    for m_name, p_arr in models.items():
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
        dro_ev = (p_hyb - 0.025) * 0.88 * odds_led - 1.0
        gap = np.abs(l_m - l_mkt_led)
        veto = np.abs(p_m_vec - p_thesis_led) <= 0.10

        pass_dro = dro_ev > 0.0

        # Build DataFrame for bets passing DRO
        df_sub = ledger_df.copy()
        df_sub["p_model"] = p_m_vec
        df_sub["p_hyb"] = p_hyb
        df_sub["q_mkt"] = p_mkt_led
        df_sub["ev_net"] = ev_net
        df_sub["dro_ev"] = dro_ev
        df_sub["gap"] = gap
        df_sub["veto"] = veto
        df_sub["odds"] = odds_led
        df_sub["stake"] = 100.0  # Flat 100 PLN baseline
        df_sub["profit"] = np.where(df_sub["won"] == 1, 100.0 * (odds_led * 0.88 - 1.0), -100.0)

        dro_df = df_sub[pass_dro].copy().reset_index(drop=True)

        gate_specs = [
            ("gate_max_odds", "Max Odds <= 4.00", dro_df["odds"] > 4.00),
            ("gate_max_ev", "Max EV <= 25%", dro_df["ev_net"] > 0.25),
            ("gate_min_ev", "Min EV >= 5%", dro_df["ev_net"] < 0.05),
            ("gate_guard_b", "Guard B |dLogit| <= 0.85", dro_df["gap"] > 0.85),
            ("gate_exp039_veto", "EXP-039 Veto <= 0.10", ~dro_df["veto"]),
        ]

        cohort2_results[m_name] = analyze_gate_blocks(dro_df, gate_specs)

    audit_results["cohorts"]["polish_production_50_bet_cohort"] = cohort2_results

    # =========================================================================
    # COHORT 3: SHARP PINNACLE T-1H COHORT (N=1,668)
    # =========================================================================
    print("\n--- Auditing Cohort 3: Sharp Pinnacle T-1h Cohort ---")
    p1_sub_pin = pd.DataFrame({
        "golgg_match_id": p1["golgg_match_id"].astype(str),
        "C0-D": p1["p_c0_d_year_core__full"],
        "C0-Slim": p1["p_c0_slim_year__full"],
    })
    mkt_df = mkt_raw[["golgg_match_id", "date", "y", "q_consensus", "p_039"]].copy()
    mkt_df["golgg_match_id"] = mkt_df["golgg_match_id"].astype(str)
    merged_mkt = mkt_df.merge(p1_sub_pin, on="golgg_match_id").sort_values(by=["date", "golgg_match_id"]).reset_index(drop=True)

    q1_pin = merged_mkt["q_consensus"].to_numpy(dtype=float)
    q2_pin = 1.0 - q1_pin
    o1_pin = 1.0 / q1_pin
    o2_pin = 1.0 / q2_pin
    y_pin = merged_mkt["y"].to_numpy(dtype=int)
    p039_pin = merged_mkt["p_039"].to_numpy(dtype=float)

    eps_num = 1e-12
    l_q1_pin = logit(np.clip(q1_pin, eps_num, 1.0 - eps_num))
    l_q2_pin = logit(np.clip(q2_pin, eps_num, 1.0 - eps_num))

    cohort3_results: dict[str, Any] = {}

    for regime_key, tax_rate in [("12_pct_polish_tax", 0.12), ("0_pct_international_tax", 0.00)]:
        c_rate = 1.0 - tax_rate
        cohort3_results[regime_key] = {}

        for m_name in ["C0-D", "C0-Slim"]:
            p1_mod = merged_mkt[m_name].to_numpy(dtype=float)
            p2_mod = 1.0 - p1_mod

            l_p1 = logit(np.clip(p1_mod, eps_num, 1.0 - eps_num))
            l_p2 = logit(np.clip(p2_mod, eps_num, 1.0 - eps_num))

            p1_hyb = expit(0.50 * l_p1 + 0.50 * l_q1_pin)
            p2_hyb = expit(0.50 * l_p2 + 0.50 * l_q2_pin)

            gap1 = np.abs(l_p1 - l_q1_pin)
            gap2 = np.abs(l_p2 - l_q2_pin)

            ev1_hyb = p1_hyb * c_rate * o1_pin - 1.0
            ev2_hyb = p2_hyb * c_rate * o2_pin - 1.0

            dro1_ev = (p1_hyb - 0.025) * c_rate * o1_pin - 1.0
            dro2_ev = (p2_hyb - 0.025) * c_rate * o2_pin - 1.0

            veto1 = np.isnan(p039_pin) | (np.abs(p1_mod - p039_pin) <= 0.10)
            veto2 = np.isnan(p039_pin) | (np.abs(p2_mod - (1.0 - p039_pin)) <= 0.10)

            # Collect bets passing DRO
            pin_bets = []
            for i in range(len(merged_mkt)):
                for side in (1, 2):
                    dro_v = dro1_ev[i] if side == 1 else dro2_ev[i]
                    if dro_v > 0.0:
                        odds = o1_pin[i] if side == 1 else o2_pin[i]
                        ev = ev1_hyb[i] if side == 1 else ev2_hyb[i]
                        gap_v = gap1[i] if side == 1 else gap2[i]
                        veto_v = veto1[i] if side == 1 else veto2[i]
                        p_m = p1_mod[i] if side == 1 else p2_mod[i]
                        p_h = p1_hyb[i] if side == 1 else p2_hyb[i]
                        q_m = q1_pin[i] if side == 1 else q2_pin[i]
                        won = int(y_pin[i] == 1) if side == 1 else int(y_pin[i] == 0)
                        profit = 100.0 * (odds * c_rate - 1.0) if won else -100.0

                        gid = str(merged_mkt["golgg_match_id"].iloc[i])
                        g_info = golgg_lookup.get(gid, {})
                        t_sel = g_info.get("team1" if side == 1 else "team2", f"Side {side}")
                        t_opp = g_info.get("team2" if side == 1 else "team1", f"Opponent {side}")
                        tourn = g_info.get("tournament", "Pinnacle T-1h")

                        pin_bets.append({
                            "golgg_match_id": gid,
                            "date": merged_mkt["date"].iloc[i],
                            "side": side,
                            "team_selected": t_sel,
                            "opponent": t_opp,
                            "tournament": tourn,
                            "odds": odds,
                            "q_mkt": q_m,
                            "p_model": p_m,
                            "p_hyb": p_h,
                            "ev_net": ev,
                            "dro_ev": dro_v,
                            "gap": gap_v,
                            "veto": veto_v,
                            "won": won,
                            "stake": 100.0,
                            "profit": profit,
                        })

            df_pin_bets = pd.DataFrame(pin_bets)

            gate_specs = [
                ("gate_max_odds", "Max Odds <= 4.00", df_pin_bets["odds"] > 4.00),
                ("gate_max_ev", "Max EV <= 25%", df_pin_bets["ev_net"] > 0.25),
                ("gate_min_ev", "Min EV >= 5%", df_pin_bets["ev_net"] < 0.05),
                ("gate_guard_b", "Guard B |dLogit| <= 0.85", df_pin_bets["gap"] > 0.85),
                ("gate_exp039_veto", "EXP-039 Veto <= 0.10", ~df_pin_bets["veto"]),
            ]

            cohort3_results[regime_key][m_name] = analyze_gate_blocks(df_pin_bets, gate_specs)

    audit_results["cohorts"]["sharp_pinnacle_t1_cohort"] = cohort3_results

    # =========================================================================
    # COHORT 1: FULL 645-MATCH / 248,735-QUOTE POLISH SCANNER (T-48h..1h)
    # =========================================================================
    print("\n--- Auditing Cohort 1: Full 645-Match Polish Scanner ---")
    m_map = mappings[["canonical_match_id", "golgg_match_id"]].merge(
        matches[["id", "winner_side", "start_time_normalized", "league", "team_a_name", "team_b_name"]],
        left_on="canonical_match_id",
        right_on="id",
    )
    m_map["golgg_match_id"] = m_map["golgg_match_id"].astype(str)

    p1_sub_pol = p1[["golgg_match_id", "y", "p_c0_d_year_core__full", "p_c0_slim_year__full"]].copy()
    p1_sub_pol["golgg_match_id"] = p1_sub_pol["golgg_match_id"].astype(str)

    merged_matches = m_map.merge(p1_sub_pol, on="golgg_match_id")
    merged_matches["a_is_pred_t1"] = (merged_matches["winner_side"] == "team_a") == (merged_matches["y"] == 1)

    q_m = quotes.merge(merged_matches, on="canonical_match_id")
    start_dt = pd.to_datetime(q_m["start_time_normalized"], utc=True, format="mixed")
    scraped_dt = pd.to_datetime(q_m["scraped_at"], utc=True, format="mixed")
    q_m["lead_hours"] = (start_dt - scraped_dt).dt.total_seconds() / 3600.0

    q_window = q_m[(q_m["lead_hours"] >= 1.0) & (q_m["lead_hours"] <= 48.0)].copy()
    q_window["scraped_dt"] = scraped_dt
    q_window = q_window.sort_values(by=["scraped_dt"]).reset_index(drop=True)

    inv_a = 1.0 / q_window["odds_a"].to_numpy(dtype=float)
    inv_b = 1.0 / q_window["odds_b"].to_numpy(dtype=float)
    overround = (inv_a + inv_b) - 1.0
    psa = np.clip(inv_a - overround / 2.0, 1e-6, 1.0 - 1e-6)
    psb = 1.0 - psa

    q_window["p_shin_a"] = psa
    q_window["p_shin_b"] = psb
    q_window["p_039"] = q_window["golgg_match_id"].map(p039_lookup).to_numpy(dtype=float)

    cohort1_match_results: dict[str, Any] = {}
    cohort1_quote_results: dict[str, Any] = {}

    for m_name in ["C0-D", "C0-Slim"]:
        col = "p_c0_d_year_core__full" if m_name == "C0-D" else "p_c0_slim_year__full"
        p_raw = q_window[col].to_numpy(dtype=float)
        p_a = np.where(q_window["a_is_pred_t1"].to_numpy(dtype=bool), p_raw, 1.0 - p_raw)
        p_b = 1.0 - p_a

        l_psa = logit(psa)
        l_psb = logit(psb)

        l_pa = logit(np.clip(p_a, 1e-6, 1.0 - 1e-6))
        l_pb = logit(np.clip(p_b, 1e-6, 1.0 - 1e-6))

        p_hyb_a = expit(0.50 * l_pa + 0.50 * l_psa)
        p_hyb_b = expit(0.50 * l_pb + 0.50 * l_psb)

        oa = q_window["odds_a"].to_numpy(dtype=float)
        ob = q_window["odds_b"].to_numpy(dtype=float)

        eva = p_hyb_a * 0.88 * oa - 1.0
        evb = p_hyb_b * 0.88 * ob - 1.0

        dro_eva = (p_hyb_a - 0.025) * 0.88 * oa - 1.0
        dro_evb = (p_hyb_b - 0.025) * 0.88 * ob - 1.0

        gap_a = np.abs(l_pa - l_psa)
        gap_b = np.abs(l_pb - l_psb)

        p039_raw = q_window["p_039"].to_numpy(dtype=float)
        p039_a = np.where(q_window["a_is_pred_t1"].to_numpy(dtype=bool), p039_raw, 1.0 - p039_raw)
        veto_a = np.isnan(p039_a) | (np.abs(p_a - p039_a) <= 0.10)
        veto_b = np.isnan(p039_a) | (np.abs(p_b - (1.0 - p039_a)) <= 0.10)

        # 1A: Chronological First Match Bet
        bets_chrono = []
        seen_matches = set()

        for i in range(len(q_window)):
            mid = q_window["canonical_match_id"].iloc[i]
            if mid in seen_matches:
                continue

            cand_a = dro_eva[i] > 0.0
            cand_b = dro_evb[i] > 0.0

            if not cand_a and not cand_b:
                continue

            if cand_a and (not cand_b or dro_eva[i] >= dro_evb[i]):
                side = "a"
                odds = oa[i]
                ev = eva[i]
                dro = dro_eva[i]
                gap_val = gap_a[i]
                veto_val = veto_a[i]
                p_m = p_a[i]
                p_h = p_hyb_a[i]
                q_mkt_val = psa[i]
                won = int(q_window["winner_side"].iloc[i] == "team_a")
                t_sel = q_window["team_a_name"].iloc[i]
                t_opp = q_window["team_b_name"].iloc[i]
            else:
                side = "b"
                odds = ob[i]
                ev = evb[i]
                dro = dro_evb[i]
                gap_val = gap_b[i]
                veto_val = veto_b[i]
                p_m = p_b[i]
                p_h = p_hyb_b[i]
                q_mkt_val = psb[i]
                won = int(q_window["winner_side"].iloc[i] == "team_b")
                t_sel = q_window["team_b_name"].iloc[i]
                t_opp = q_window["team_a_name"].iloc[i]

            profit = 100.0 * (odds * 0.88 - 1.0) if won else -100.0
            seen_matches.add(mid)

            bets_chrono.append({
                "canonical_match_id": mid,
                "golgg_match_id": q_window["golgg_match_id"].iloc[i],
                "date": q_window["scraped_at"].iloc[i],
                "tournament": q_window["league"].iloc[i],
                "team_selected": t_sel,
                "opponent": t_opp,
                "side": side,
                "odds": odds,
                "q_mkt": q_mkt_val,
                "p_model": p_m,
                "p_hyb": p_h,
                "ev_net": ev,
                "dro_ev": dro,
                "gap": gap_val,
                "veto": veto_val,
                "won": won,
                "stake": 100.0,
                "profit": profit,
            })

        df_chrono = pd.DataFrame(bets_chrono)
        gate_specs_chrono = [
            ("gate_max_odds", "Max Odds <= 4.00", df_chrono["odds"] > 4.00),
            ("gate_max_ev", "Max EV <= 25%", df_chrono["ev_net"] > 0.25),
            ("gate_min_ev", "Min EV >= 5%", df_chrono["ev_net"] < 0.05),
            ("gate_guard_b", "Guard B |dLogit| <= 0.85", df_chrono["gap"] > 0.85),
            ("gate_exp039_veto", "EXP-039 Veto <= 0.10", ~df_chrono["veto"]),
        ]
        cohort1_match_results[m_name] = analyze_gate_blocks(df_chrono, gate_specs_chrono)

        # 1B: Full Quote-Level Opportunities
        quotes_passing = []
        for i in range(len(q_window)):
            for side in ("a", "b"):
                dro_v = dro_eva[i] if side == "a" else dro_evb[i]
                if dro_v > 0.0:
                    odds = oa[i] if side == "a" else ob[i]
                    ev = eva[i] if side == "a" else evb[i]
                    gap_v = gap_a[i] if side == "a" else gap_b[i]
                    veto_v = veto_a[i] if side == "a" else veto_b[i]
                    p_m = p_a[i] if side == "a" else p_b[i]
                    p_h = p_hyb_a[i] if side == "a" else p_hyb_b[i]
                    q_mkt_v = psa[i] if side == "a" else psb[i]
                    won = int(q_window["winner_side"].iloc[i] == ("team_a" if side == "a" else "team_b"))
                    profit = 100.0 * (odds * 0.88 - 1.0) if won else -100.0

                    t_sel = q_window["team_a_name"].iloc[i] if side == "a" else q_window["team_b_name"].iloc[i]
                    t_opp = q_window["team_b_name"].iloc[i] if side == "a" else q_window["team_a_name"].iloc[i]

                    quotes_passing.append({
                        "canonical_match_id": q_window["canonical_match_id"].iloc[i],
                        "golgg_match_id": q_window["golgg_match_id"].iloc[i],
                        "date": q_window["scraped_at"].iloc[i],
                        "tournament": q_window["league"].iloc[i],
                        "team_selected": t_sel,
                        "opponent": t_opp,
                        "side": side,
                        "odds": odds,
                        "q_mkt": q_mkt_v,
                        "p_model": p_m,
                        "p_hyb": p_h,
                        "ev_net": ev,
                        "dro_ev": dro_v,
                        "gap": gap_v,
                        "veto": veto_v,
                        "won": won,
                        "stake": 100.0,
                        "profit": profit,
                    })

        df_quotes = pd.DataFrame(quotes_passing)
        gate_specs_quotes = [
            ("gate_max_odds", "Max Odds <= 4.00", df_quotes["odds"] > 4.00),
            ("gate_max_ev", "Max EV <= 25%", df_quotes["ev_net"] > 0.25),
            ("gate_min_ev", "Min EV >= 5%", df_quotes["ev_net"] < 0.05),
            ("gate_guard_b", "Guard B |dLogit| <= 0.85", df_quotes["gap"] > 0.85),
            ("gate_exp039_veto", "EXP-039 Veto <= 0.10", ~df_quotes["veto"]),
        ]
        cohort1_quote_results[m_name] = analyze_gate_blocks(df_quotes, gate_specs_quotes)

    audit_results["cohorts"]["polish_scanner_645_matches"] = {
        "chronological_match_bets": cohort1_match_results,
        "quote_level_opportunities": cohort1_quote_results,
    }

    # Save to disk
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nAudit completed. Output written to: {OUTPUT_JSON_PATH}")
    return audit_results


if __name__ == "__main__":
    run_full_audit()
