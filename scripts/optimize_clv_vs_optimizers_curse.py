#!/usr/bin/env python3
"""Empirical Walk-Forward Optimization: Net CLV vs Optimizer's Curse (B_hyb vs B_DRO).

Executes rigorous walk-forward simulations across:
1. Production PostgreSQL Database (Cohort A: 646 exact matches, Cohort B: 830 matches up to Sep 21)
   - Realistic 5-Check Pre-Match Walk-Forward (T-24h, T-12h, T-6h, T-2h, T-1h)
   - Strictly max 1 bet per match
   - OperationalWallet 3-hour locked capital
   - 12% Polish tax (0.88 net decimal)
   - 1/4 Kelly sizing capped at [20.00, 150.00] PLN
   - Evaluates C0-D-Core (18-Token) and C0-Slim (18-Token)
2. Parametric Sweep across (alpha, eps, Guard B, MaxOdds):
   - alpha in {0.35, 0.40, 0.50}
   - eps in {0.015, 0.020, 0.025}
   - Guard B in {0.85, 0.95, 1.00, 1.10, 1.25}
   - MaxOdds in {4.00, None}
   Under strict condition: B_hyb < Net CLV (CCR_hyb < 1.0, Edge Retention > 50%)
   And stricter conditions: B_DRO <= 0 (CCR_DRO <= 0) and B_hyb <= 0.5 * Net CLV (CCR_hyb <= 0.5)
3. Cross-Cohort Verification:
   - 1,467-match Polish Bookmaker cohort (Polish2plus, raw best bookmaker odds)
   - 1,668-match Pinnacle T-1h cohort (under 12% Polish tax and 0% International tax)

Saves output to:
data/08_reporting/c0_upgrade_and_distillation_20261017/clv_vs_optimizers_curse_optimization.json
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t as t_dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.historical_production_simulator import OperationalWallet

REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
OUTPUT_JSON = REPORT_DIR / "clv_vs_optimizers_curse_optimization.json"

RAW_PROD_DIR = PROJECT_ROOT / "data/01_raw/c0_production_counterfactual/c0_production_20261006_01"
REP_DIR = PROJECT_ROOT / "data/08_reporting/c0_production_counterfactual/c0_production_20261006_01"
MATCHES_PATH = RAW_PROD_DIR / "matches.parquet"
BOOKMAKERS_PATH = RAW_PROD_DIR / "bookmakers.parquet"
MAPPINGS_PATH = RAW_PROD_DIR / "mappings.parquet"
QUOTES_PATH = RAW_PROD_DIR / "quotes.parquet"
QUOTE_ALIGNMENT_PATH = REP_DIR / "quote_alignment.parquet"
IDENTITIES_PATH = REP_DIR / "identities.parquet"
ALL_2026_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/all_2026_joined.parquet"
PRED_PATH = REPORT_DIR / "year_anchor_distilled_predictions.parquet"
SCORED_PANEL_PATH = REP_DIR / "scored_panel.parquet"
MARKET_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"

POLISH_PANEL_PATH = PROJECT_ROOT / "data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/paired_probability_panel.parquet"
POLISH_PAIRS_PATH = PROJECT_ROOT / "data/07_model_output/oddsportal/oddsportal_polish_consensus_20261009_152919Z/polish_quote_model_pairs.parquet"
PINNACLE_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"
PINNACLE_TIMES_PATH = PROJECT_ROOT / "data/07_model_output/oddsportal/c0_bookmaker_horizons_20261009_103721Z/native_event_query_ledger.parquet"

# -----------------------------------------------------------------------------
# 1. Data Preparation
# -----------------------------------------------------------------------------
def load_production_data() -> dict[str, Any]:
    print("[1/5] Loading production database and predictions with verified orientation...")
    matches = pd.read_parquet(MATCHES_PATH)
    quotes = pd.read_parquet(QUOTES_PATH)
    qa = pd.read_parquet(QUOTE_ALIGNMENT_PATH)
    bks = pd.read_parquet(BOOKMAKERS_PATH)
    ident = pd.read_parquet(IDENTITIES_PATH)
    ident = ident[(ident["identity_status"] == "verified") & (ident["date_within_one_day"] != False)].copy()

    p1 = pd.read_parquet(PRED_PATH)
    p26 = pd.read_parquet(ALL_2026_PATH)
    sp_df = pd.read_parquet(SCORED_PANEL_PATH).drop_duplicates("golgg_match_id")
    sc = sp_df.set_index(sp_df["golgg_match_id"].astype(str))

    yr_p26 = p1.merge(p26[["golgg_match_id", "team1_id", "team2_id", "y", "p_039"]], on="golgg_match_id", suffixes=("", "_p26"))
    yr_map = yr_p26.set_index(yr_p26["golgg_match_id"].astype(str))

    # Align quotes with quote_alignment.parquet (verified status only, orienting via raw_a_is_team1)
    q_aligned = quotes.merge(
        bks[["id", "name"]], left_on="bookmaker_id", right_on="id", suffixes=("", "_bk")
    ).merge(
        qa[["id", "quote_identity_status", "raw_a_is_team1"]], on="id"
    )
    q_aligned = q_aligned[
        (q_aligned["quote_identity_status"] == "verified") &
        (~q_aligned["is_live"].astype(bool)) &
        (q_aligned["market_type"] == "match_winner")
    ].copy()

    is_t1 = q_aligned["raw_a_is_team1"].astype(bool).to_numpy()
    q_aligned["odds_1"] = np.where(is_t1, q_aligned["odds_a"], q_aligned["odds_b"])
    q_aligned["odds_2"] = np.where(is_t1, q_aligned["odds_b"], q_aligned["odds_a"])
    q_aligned["scraped_dt"] = pd.to_datetime(q_aligned["scraped_at"], utc=True)
    q_aligned = q_aligned.sort_values("scraped_dt")

    # Devig using Shin to obtain fair p_shin_1
    inv1 = 1.0 / q_aligned["odds_1"].to_numpy(dtype=float)
    inv2 = 1.0 / q_aligned["odds_2"].to_numpy(dtype=float)
    s_tot = inv1 + inv2
    z_shin = np.maximum(0.0, s_tot - 1.0)
    denom = np.maximum(1e-9, 2.0 * (1.0 - z_shin))
    pi1 = (np.sqrt(z_shin**2 + 4.0 * (1.0 - z_shin) * (inv1**2) / s_tot) - z_shin) / denom
    pi2 = (np.sqrt(z_shin**2 + 4.0 * (1.0 - z_shin) * (inv2**2) / s_tot) - z_shin) / denom
    q_aligned["p_shin_1"] = pi1 / (pi1 + pi2)

    # Pre-compute pre-kickoff sharp closing line consensus
    print("Pre-computing closing lines...")
    closing_dict = {}
    m_ident = ident.merge(matches[["id", "winner_side", "start_time_normalized", "league"]], left_on="canonical_match_id", right_on="id")
    for _, r in m_ident.iterrows():
        cid = int(r["canonical_match_id"])
        if pd.isna(r["start_time_normalized"]):
            continue
        start_dt = pd.to_datetime(r["start_time_normalized"], utc=True)
        sub_cl = q_aligned[(q_aligned["canonical_match_id"] == cid) & (q_aligned["scraped_dt"] < start_dt) & (q_aligned["scraped_dt"] >= start_dt - pd.Timedelta(hours=6))]
        if len(sub_cl["bookmaker_id"].unique()) < 2:
            sub_cl = q_aligned[(q_aligned["canonical_match_id"] == cid) & (q_aligned["scraped_dt"] < start_dt)]
        if len(sub_cl["bookmaker_id"].unique()) >= 2:
            last_cl = sub_cl.groupby("bookmaker_id").last()
            closing_dict[cid] = float(last_cl["p_shin_1"].mean())

    # Build match info list with verified orientation
    match_rows = []
    for _, r in m_ident.iterrows():
        cid = int(r["canonical_match_id"])
        if pd.isna(r["winner_side"]) or r["winner_side"] not in ("team_a", "team_b", "a", "b"):
            continue
        start_dt = pd.to_datetime(r["start_time_normalized"], utc=True)
        if pd.isna(start_dt):
            continue
        gid_str = str(int(float(r["golgg_match_id"])))
        win_a = 1 if r["winner_side"] in ("team_a", "a") else 0
        win_1 = win_a if bool(r["canonical_a_is_team1"]) else (1 - win_a)

        if gid_str in yr_map.index:
            yrow = yr_map.loc[gid_str]
            same = (str(int(float(r["team1_id"]))) == str(int(float(yrow["team1_id"]))))
            p_c0d = float(yrow["p_c0_d_year_core__full"]) if same else (1.0 - float(yrow["p_c0_d_year_core__full"]))
            p_slim = float(yrow["p_c0_slim_year__full"]) if same else (1.0 - float(yrow["p_c0_slim_year__full"]))
            p_c0 = float(yrow["p_c0__full"]) if same else (1.0 - float(yrow["p_c0__full"]))
            is_cohort_a = True
        elif gid_str in sc.index:
            p_val = float(sc.loc[gid_str, "p_C0"])
            p_c0d = p_val
            p_slim = p_val
            p_c0 = p_val
            is_cohort_a = False
        else:
            continue

        match_rows.append({
            "canonical_match_id": cid,
            "golgg_str": gid_str,
            "is_cohort_a": is_cohort_a,
            "start_dt": start_dt,
            "win_1": win_1,
            "month": start_dt.strftime("%Y-%m"),
            "p_c0_d_year_core__full": p_c0d,
            "p_c0_slim_year__full": p_slim,
            "p_c0__full": p_c0,
        })

    all_matches_df = pd.DataFrame(match_rows)
    cohort_a = all_matches_df[all_matches_df["is_cohort_a"]].copy()
    cohort_b = all_matches_df.copy()

    quotes_by_match = {mid: df for mid, df in q_aligned.groupby("canonical_match_id")}

    print(f"Cohort A prepared: {len(cohort_a)} matches")
    print(f"Cohort B prepared: {len(cohort_b)} matches")

    return {
        "cohort_a": cohort_a,
        "cohort_b": cohort_b,
        "quotes_by_match": quotes_by_match,
        "closing_dict": closing_dict,
    }


def preextract_snapshots(matches_df: pd.DataFrame, quotes_by_match: dict[int, pd.DataFrame], closing_dict: dict[int, float], horizons: list[float] = [24.0, 12.0, 6.0, 2.0, 1.0]) -> list[dict[str, Any]]:
    snapshots = []
    for _, m in matches_df.iterrows():
        mid = int(m["canonical_match_id"])
        t_start = m["start_dt"]
        q_mid = quotes_by_match.get(mid)
        if q_mid is None:
            continue

        sub = q_mid[(q_mid["scraped_dt"] < t_start) & (q_mid["scraped_dt"] >= t_start - pd.Timedelta(hours=72))].copy()
        if sub.empty:
            continue

        sub["bucket_2h"] = sub["scraped_dt"].dt.floor("2h")
        live_books = {}
        for b_dt, b_df in sub.groupby("bucket_2h"):
            for _, qrow in b_df.iterrows():
                live_books[int(qrow["bookmaker_id"])] = {
                    "o1": float(qrow["odds_1"]),
                    "o2": float(qrow["odds_2"]),
                    "p1": float(qrow["p_shin_1"]),
                    "dt": qrow["scraped_dt"],
                }
            active = {k: v for k, v in live_books.items() if (b_dt - v["dt"]).total_seconds() <= 12 * 3600}
            if len(active) < 2:
                continue

            best_1 = None
            best_2 = None
            for fk_id, fk_val in active.items():
                oth = [v["p1"] for oid, v in active.items() if oid != fk_id]
                ql1 = float(np.mean(oth))
                ql2 = 1.0 - ql1
                lag1 = ql1 * fk_val["o1"] - 1.0
                lag2 = ql2 * fk_val["o2"] - 1.0

                if 0.00 <= lag1 <= 0.38:
                    if best_1 is None or fk_val["o1"] > best_1["odds"]:
                        best_1 = {"odds": fk_val["o1"], "q_loo": ql1}
                if 0.00 <= lag2 <= 0.38:
                    if best_2 is None or fk_val["o2"] > best_2["odds"]:
                        best_2 = {"odds": fk_val["o2"], "q_loo": ql2}

            if best_1 is None and best_2 is None:
                continue

            h_lead = (t_start - b_dt).total_seconds() / 3600.0
            snapshots.append({
                "canonical_match_id": mid,
                "horizon_h": round(h_lead, 1),
                "t_dec": b_dt,
                "t_start": t_start,
                "t_settle": t_start + pd.Timedelta(hours=3),
                "n_books": len(active),
                "q_mkt_a": best_1["q_loo"] if best_1 else np.nan,
                "q_mkt_b": best_2["q_loo"] if best_2 else np.nan,
                "best_odds_a": best_1["odds"] if best_1 else 1.0,
                "best_odds_b": best_2["odds"] if best_2 else 1.0,
                "winner_side": "team_a" if m["win_1"] == 1 else "team_b",
                "a_is_pred_t1": True,
                "p_c0_d_year_core__full": float(m["p_c0_d_year_core__full"]),
                "p_c0_slim_year__full": float(m["p_c0_slim_year__full"]),
                "p_c0__full": float(m["p_c0__full"]),
                "golgg_str": str(m["golgg_str"]),
                "p_close_a": float(closing_dict.get(mid, 0.5)),
            })
    return snapshots

# -----------------------------------------------------------------------------
# 2. Evaluation & Simulation Engine
# -----------------------------------------------------------------------------
def evaluate_snapshot(snap: dict[str, Any], model_col: str, alpha: float, eps: float, guard_b: float, max_odds: float | None, tax_rate: float = 0.12) -> dict[str, Any] | None:
    c_tax = 1.0 - tax_rate
    p_a = snap[model_col]
    p_b = 1.0 - p_a

    q_a = snap["q_mkt_a"]
    q_b = snap["q_mkt_b"]
    oa = snap["best_odds_a"]
    ob = snap["best_odds_b"]

    is_mo_active = (max_odds is not None) and (not pd.isna(max_odds))

    qa = False
    dro_ev_a = -999.0
    ev_a = -999.0
    p_hyb_a = 0.5
    if not np.isnan(q_a) and oa > 1.0:
        la = logit(np.clip(p_a, 1e-6, 1.0 - 1e-6))
        lqa = logit(np.clip(q_a, 1e-6, 1.0 - 1e-6))
        gap_a = abs(la - lqa)
        p_hyb_a = expit(alpha * la + (1.0 - alpha) * lqa)
        ev_a = p_hyb_a * c_tax * oa - 1.0
        dro_ev_a = (p_hyb_a - eps) * c_tax * oa - 1.0
        qa = (dro_ev_a > 0.0) and (gap_a <= guard_b) and (not is_mo_active or oa <= max_odds)

    qb = False
    dro_ev_b = -999.0
    ev_b = -999.0
    p_hyb_b = 0.5
    if not np.isnan(q_b) and ob > 1.0:
        lb = logit(np.clip(p_b, 1e-6, 1.0 - 1e-6))
        lqb = logit(np.clip(q_b, 1e-6, 1.0 - 1e-6))
        gap_b = abs(lb - lqb)
        p_hyb_b = expit(alpha * lb + (1.0 - alpha) * lqb)
        ev_b = p_hyb_b * c_tax * ob - 1.0
        dro_ev_b = (p_hyb_b - eps) * c_tax * ob - 1.0
        qb = (dro_ev_b > 0.0) and (gap_b <= guard_b) and (not is_mo_active or ob <= max_odds)

    if not qa and not qb:
        return None

    if qa and (not qb or dro_ev_a >= dro_ev_b):
        side = "a"
        odds = oa
        p_claim = p_hyb_a
        ev_claim = ev_a
        dro_ev = dro_ev_a
        q_mkt = q_a
        won = (snap["winner_side"] == "team_a")
        p_close = snap["p_close_a"]
    else:
        side = "b"
        odds = ob
        p_claim = p_hyb_b
        ev_claim = ev_b
        dro_ev = dro_ev_b
        q_mkt = q_b
        won = (snap["winner_side"] == "team_b")
        p_close = 1.0 - snap["p_close_a"]

    net_clv = odds * c_tax * p_close - 1.0
    b_hyb = ev_claim - net_clv
    b_dro = dro_ev - net_clv

    return {
        "match_id": snap["canonical_match_id"],
        "side": side,
        "odds": odds,
        "p_claim": p_claim,
        "ev_claim": ev_claim,
        "dro_ev": dro_ev,
        "q_mkt": q_mkt,
        "won": won,
        "p_close": p_close,
        "net_clv": net_clv,
        "b_hyb": b_hyb,
        "b_dro": b_dro,
        "t_dec": snap["t_dec"],
        "t_settle": snap["t_settle"],
        "horizon_h": snap["horizon_h"],
    }


def run_simulation(
    snaps: list[dict[str, Any]],
    model_col: str,
    alpha: float,
    eps: float,
    guard_b: float,
    max_odds: float | None,
    schedule_mode: str = "5-Check",
    initial_capital: float = 10000.0,
    tax_rate: float = 0.12,
) -> dict[str, Any]:
    c_tax = 1.0 - tax_rate
    wallet = OperationalWallet(initial_capital=initial_capital)

    if schedule_mode != "5-Check":
        h_target = {"T-24h": 24.0, "T-12h": 12.0, "T-6h": 6.0, "T-2h": 2.0, "T-1h": 1.0}[schedule_mode]
        active_snaps = [s for s in snaps if abs(s["horizon_h"] - h_target) <= 1.0]
    else:
        active_snaps = snaps

    heap: list[tuple[pd.Timestamp, int, int, str, dict[str, Any]]] = []
    tiebreaker = 0

    for s in active_snaps:
        heapq.heappush(heap, (s["t_dec"], 1, tiebreaker, "decision", s))
        tiebreaker += 1

    already_bet_matches: set[int] = set()
    placed_bets: dict[int, str] = {}
    capital_exhausted_bets = 0

    while heap:
        ts, prio, _, ev_type, payload = heapq.heappop(heap)

        if ev_type == "settle":
            mid = payload["match_id"]
            bet_id = placed_bets.get(mid)
            if bet_id and bet_id in wallet.open_bets:
                wallet.settle(bet_id, won=payload["won"], settled_at=ts, tax_rate=tax_rate)

        elif ev_type == "decision":
            s = payload
            mid = s["canonical_match_id"]
            if mid in already_bet_matches:
                continue

            cand = evaluate_snapshot(s, model_col, alpha, eps, guard_b, max_odds, tax_rate=tax_rate)
            if cand is None:
                continue

            b_net = cand["odds"] * c_tax - 1.0
            f_star = max(0.0, (b_net * cand["p_claim"] - (1.0 - cand["p_claim"])) / b_net) if b_net > 0 else 0.0
            raw_stake = wallet.free_capital * 0.25 * f_star
            stake = min(150.0, max(20.0, raw_stake))

            if stake > wallet.free_capital:
                stake = wallet.free_capital

            if stake < 20.0 or stake > wallet.free_capital:
                capital_exhausted_bets += 1
                continue

            bet_id = f"bet_{mid}"
            wallet.reserve(
                bet_id=bet_id,
                amount=stake,
                match_id=mid,
                side=cand["side"],
                odds=cand["odds"],
                placed_at=ts,
                settle_at=cand["t_settle"],
                ev_claimed=cand["ev_claim"],
                p_model=cand["p_claim"],
                p_market=cand["q_mkt"],
                clv=cand["net_clv"],
            )
            placed_bets[mid] = bet_id
            already_bet_matches.add(mid)

            tiebreaker += 1
            heapq.heappush(heap, (cand["t_settle"], 0, tiebreaker, "settle", {"match_id": mid, "won": cand["won"], "cand": cand}))

    settled = pd.DataFrame(wallet.settled_bets)
    if settled.empty:
        return {
            "n_bets": 0, "n_wins": 0, "win_rate_pct": 0.0, "turnover_pln": 0.0, "profit_pln": 0.0,
            "realized_yield_pct": 0.0, "net_clv_pct": 0.0, "ev_hyb_pct": 0.0, "ev_dro_pct": 0.0,
            "b_hyb_pp": 0.0, "b_dro_pp": 0.0, "ccr_hyb": 0.0, "ccr_dro": 0.0, "edge_retention_pct": 0.0,
            "total_clv_profit_pln": 0.0, "net_clv_unweighted_pct": 0.0, "t_clv": 0.0, "p_clv": 1.0,
            "max_drawdown_pct": 0.0, "profit_factor": 0.0, "sharpe": 0.0, "capital_exhausted": capital_exhausted_bets,
            "final_equity_pln": initial_capital,
        }

    n = len(settled)
    n_wins = int(settled["won"].sum())
    turnover = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    yd = (profit / turnover) * 100.0 if turnover > 0 else 0.0
    wr = float(settled["won"].mean()) * 100.0

    weights = settled["stake"].to_numpy(dtype=float) / turnover
    net_clvs = settled["clv"].to_numpy(dtype=float)
    ev_claims = settled["ev_claimed"].to_numpy(dtype=float)

    net_clv_weighted = float(np.sum(weights * net_clvs)) * 100.0
    ev_hyb_weighted = float(np.sum(weights * ev_claims)) * 100.0
    b_hyb_weighted = ev_hyb_weighted - net_clv_weighted
    odds_arr = settled["odds"].to_numpy(dtype=float)
    ev_dro_arr = ev_claims - eps * c_tax * odds_arr
    ev_dro_weighted = float(np.sum(weights * ev_dro_arr)) * 100.0
    b_dro_weighted = ev_dro_weighted - net_clv_weighted

    ccr_hyb = (b_hyb_weighted / net_clv_weighted) if net_clv_weighted != 0 else float("inf")
    ccr_dro = (b_dro_weighted / net_clv_weighted) if net_clv_weighted != 0 else float("inf")

    net_clv_mean_unw = float(np.mean(net_clvs))
    net_clv_std_unw = float(np.std(net_clvs, ddof=1)) if n > 1 else 0.0
    t_clv = (net_clv_mean_unw / (net_clv_std_unw / np.sqrt(n))) if (n > 1 and net_clv_std_unw > 0) else 0.0
    p_clv = float(1.0 - t_dist.cdf(t_clv, df=n - 1)) if n > 1 else 1.0

    total_clv_profit_pln = float(np.sum(settled["stake"].to_numpy(dtype=float) * net_clvs))

    equities = np.array([initial_capital] + [r["total_equity"] for r in wallet.settled_bets])
    running_max = np.maximum.accumulate(equities)
    dd = (running_max - equities) / running_max * 100.0
    max_dd = float(np.max(dd))

    gross_wins = float(settled[settled["profit"] > 0]["profit"].sum())
    gross_losses = float(abs(settled[settled["profit"] < 0]["profit"].sum()))
    pf = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    returns = settled["profit"] / settled["stake"]
    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (float(returns.mean()) / ret_std) * np.sqrt(n) if ret_std > 0 else 0.0

    return {
        "n_bets": n,
        "n_wins": n_wins,
        "win_rate_pct": round(wr, 2),
        "turnover_pln": round(turnover, 2),
        "profit_pln": round(profit, 2),
        "realized_yield_pct": round(yd, 2),
        "net_clv_pct": round(net_clv_weighted, 2),
        "ev_hyb_pct": round(ev_hyb_weighted, 2),
        "ev_dro_pct": round(ev_dro_weighted, 2),
        "b_hyb_pp": round(b_hyb_weighted, 2),
        "b_dro_pp": round(b_dro_weighted, 2),
        "ccr_hyb": round(ccr_hyb, 3),
        "ccr_dro": round(ccr_dro, 3),
        "edge_retention_pct": round((net_clv_weighted / ev_hyb_weighted) * 100.0, 2) if ev_hyb_weighted != 0 else 0.0,
        "total_clv_profit_pln": round(total_clv_profit_pln, 2),
        "net_clv_unweighted_pct": round(net_clv_mean_unw * 100.0, 2),
        "t_clv": round(float(t_clv), 3),
        "p_clv": round(p_clv, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "sharpe": round(float(sharpe), 2),
        "capital_exhausted": capital_exhausted_bets,
        "final_equity_pln": round(wallet.total_equity, 2),
    }


# -----------------------------------------------------------------------------
# 3. Cross-Cohort Verification (Polish & Pinnacle)
# -----------------------------------------------------------------------------
def load_cross_cohort_data(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[3/5] Loading 1,467 Polish Bookmaker cohort and 1,668 Pinnacle cohort...")
    pol_raw = pd.read_parquet(POLISH_PANEL_PATH)
    pol_df = pol_raw[pol_raw["comparison_cohort"] == "Polish2plus"].copy().reset_index(drop=True)
    pol_df["golgg_match_id"] = pol_df["golgg_match_id"].astype(str)
    pol_df["match_id"] = pol_df["event_id"]
    pol_df["t_start"] = pd.to_datetime(pol_df["source_start_utc"], utc=True)
    pol_df["t_dec"] = pol_df["t_start"] - pd.Timedelta(hours=1)
    pol_df["t_settle"] = pol_df["t_start"] + pd.Timedelta(hours=3)
    pol_df["y"] = pol_df["y"].astype(int)
    pol_df["q1"] = pol_df["p_polish_consensus"].astype(float)
    pol_df["q2"] = 1.0 - pol_df["q1"]

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
    pol_df = pol_df.merge(preds[["golgg_match_id", "p_c0_d_year_core__full", "p_c0_slim_year__full", "p_c0__full"]], on="golgg_match_id")

    pin_raw = pd.read_parquet(PINNACLE_T1_PATH)
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
    pin_df = pin_df.merge(preds[["golgg_match_id", "p_c0_d_year_core__full", "p_c0_slim_year__full", "p_c0__full"]], on="golgg_match_id")

    return pol_df, pin_df


def run_cohort_sim(
    df: pd.DataFrame,
    model_col: str,
    alpha: float,
    eps: float,
    guard_b: float,
    max_odds: float | None,
    odds_col1: str,
    odds_col2: str,
    tax_rate: float = 0.12,
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    c_tax = 1.0 - tax_rate
    wallet = OperationalWallet(initial_capital=initial_capital)

    p1 = df[model_col].to_numpy(dtype=float)
    p2 = 1.0 - p1
    q1 = df["q1"].to_numpy(dtype=float)
    q2 = df["q2"].to_numpy(dtype=float)
    o1 = df[odds_col1].to_numpy(dtype=float)
    o2 = df[odds_col2].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)

    eps_prob = 1e-12
    l_p1 = logit(np.clip(p1, eps_prob, 1.0 - eps_prob))
    l_q1 = logit(np.clip(q1, eps_prob, 1.0 - eps_prob))
    l_p2 = logit(np.clip(p2, eps_prob, 1.0 - eps_prob))
    l_q2 = logit(np.clip(q2, eps_prob, 1.0 - eps_prob))

    p1_h = expit(alpha * l_p1 + (1.0 - alpha) * l_q1)
    p2_h = expit(alpha * l_p2 + (1.0 - alpha) * l_q2)

    gap1 = np.abs(l_p1 - l_q1)
    gap2 = np.abs(l_p2 - l_q2)

    ev1 = p1_h * c_tax * o1 - 1.0
    ev2 = p2_h * c_tax * o2 - 1.0

    dro1 = (p1_h - eps) * c_tax * o1 - 1.0
    dro2 = (p2_h - eps) * c_tax * o2 - 1.0

    c1 = (dro1 > 0.0) & (gap1 <= guard_b)
    c2 = (dro2 > 0.0) & (gap2 <= guard_b)
    is_mo_active = (max_odds is not None) and (not pd.isna(max_odds))
    if is_mo_active:
        c1 &= (o1 <= max_odds)
        c2 &= (o2 <= max_odds)

    heap: list[tuple[pd.Timestamp, int, int, str, dict[str, Any]]] = []
    tie = 0
    for i in range(len(df)):
        side = 0
        if c1[i] and (not c2[i] or dro1[i] >= dro2[i]):
            side, o_taken, p_claim, p_close, won, ev_claim = 1, o1[i], p1_h[i], q1[i], int(y[i] == 1), ev1[i]
        elif c2[i]:
            side, o_taken, p_claim, p_close, won, ev_claim = 2, o2[i], p2_h[i], q2[i], int(y[i] == 0), ev2[i]

        if side != 0:
            mid = df["match_id"].iloc[i]
            t_dec = df["t_dec"].iloc[i]
            t_settle = df["t_settle"].iloc[i]
            net_clv = o_taken * c_tax * p_close - 1.0
            heapq.heappush(heap, (t_dec, 1, tie, "decision", {
                "match_id": mid, "side": side, "odds": o_taken, "p_claim": p_claim,
                "ev_claim": ev_claim, "p_close": p_close, "won": won, "net_clv": net_clv,
                "t_settle": t_settle, "eps": eps, "c_tax": c_tax
            }))
            tie += 1

    placed_bets: dict[Any, str] = {}
    capital_exhausted = 0
    while heap:
        ts, prio, _, ev_type, payload = heapq.heappop(heap)
        if ev_type == "settle":
            mid = payload["match_id"]
            bet_id = placed_bets.get(mid)
            if bet_id and bet_id in wallet.open_bets:
                wallet.settle(bet_id, won=payload["won"], settled_at=ts, tax_rate=tax_rate)
        elif ev_type == "decision":
            mid = payload["match_id"]
            b = payload
            b_net = b["odds"] * c_tax - 1.0
            f_star = max(0.0, (b_net * b["p_claim"] - (1.0 - b["p_claim"])) / b_net) if b_net > 0 else 0.0
            raw_stake = wallet.free_capital * 0.25 * f_star
            stake = min(150.0, max(20.0, raw_stake))
            if stake > wallet.free_capital:
                stake = wallet.free_capital
            if stake < 20.0 or stake > wallet.free_capital:
                capital_exhausted += 1
                continue
            bet_id = f"bet_{mid}"
            wallet.reserve(
                bet_id=bet_id, amount=stake, match_id=mid, side=str(b["side"]),
                odds=b["odds"], placed_at=ts, settle_at=b["t_settle"],
                ev_claimed=b["ev_claim"], p_model=b["p_claim"], p_market=b["p_close"],
                clv=b["net_clv"]
            )
            placed_bets[mid] = bet_id
            tie += 1
            heapq.heappush(heap, (b["t_settle"], 0, tie, "settle", {"match_id": mid, "won": b["won"]}))

    settled = pd.DataFrame(wallet.settled_bets)
    if settled.empty:
        return {"n_bets": 0, "net_clv_pct": 0.0, "ev_hyb_pct": 0.0, "b_hyb_pp": 0.0, "b_dro_pp": 0.0, "ccr_hyb": 0.0, "total_clv_profit_pln": 0.0, "profit_pln": 0.0, "realized_yield_pct": 0.0}

    n = len(settled)
    turnover = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    yd = (profit / turnover) * 100.0 if turnover > 0 else 0.0
    weights = settled["stake"].to_numpy(dtype=float) / turnover
    net_clvs = settled["clv"].to_numpy(dtype=float)
    ev_claims = settled["ev_claimed"].to_numpy(dtype=float)

    net_clv_weighted = float(np.sum(weights * net_clvs)) * 100.0
    ev_hyb_weighted = float(np.sum(weights * ev_claims)) * 100.0
    b_hyb_weighted = ev_hyb_weighted - net_clv_weighted

    odds_arr = settled["odds"].to_numpy(dtype=float)
    ev_dro_arr = ev_claims - eps * c_tax * odds_arr
    ev_dro_weighted = float(np.sum(weights * ev_dro_arr)) * 100.0
    b_dro_weighted = ev_dro_weighted - net_clv_weighted

    ccr_hyb = (b_hyb_weighted / net_clv_weighted) if net_clv_weighted != 0 else float("inf")
    ccr_dro = (b_dro_weighted / net_clv_weighted) if net_clv_weighted != 0 else float("inf")
    total_clv_profit_pln = float(np.sum(settled["stake"].to_numpy(dtype=float) * net_clvs))

    net_clv_mean_unw = float(np.mean(net_clvs))
    net_clv_std_unw = float(np.std(net_clvs, ddof=1)) if n > 1 else 0.0
    t_clv = (net_clv_mean_unw / (net_clv_std_unw / np.sqrt(n))) if (n > 1 and net_clv_std_unw > 0) else 0.0
    p_clv = float(1.0 - t_dist.cdf(t_clv, df=n - 1)) if n > 1 else 1.0

    return {
        "n_bets": n,
        "win_rate_pct": round(float(settled["won"].mean()) * 100.0, 2),
        "turnover_pln": round(turnover, 2),
        "profit_pln": round(profit, 2),
        "realized_yield_pct": round(yd, 2),
        "net_clv_pct": round(net_clv_weighted, 2),
        "ev_hyb_pct": round(ev_hyb_weighted, 2),
        "ev_dro_pct": round(ev_dro_weighted, 2),
        "b_hyb_pp": round(b_hyb_weighted, 2),
        "b_dro_pp": round(b_dro_weighted, 2),
        "ccr_hyb": round(ccr_hyb, 3),
        "ccr_dro": round(ccr_dro, 3),
        "edge_retention_pct": round((net_clv_weighted / ev_hyb_weighted) * 100.0, 2) if ev_hyb_weighted != 0 else 0.0,
        "total_clv_profit_pln": round(total_clv_profit_pln, 2),
        "t_clv": round(float(t_clv), 3),
        "p_clv": round(p_clv, 4),
        "final_equity_pln": round(wallet.total_equity, 2),
    }


# -----------------------------------------------------------------------------
# 4. Main Optimization & JSON Export
# -----------------------------------------------------------------------------
def main() -> None:
    t_start_total = time.time()
    print("=" * 80)
    print("EMPIRICAL NET CLV VS OPTIMIZER'S CURSE WALK-FORWARD OPTIMIZATION")
    print("=" * 80)

    prod_data = load_production_data()
    cohort_a = prod_data["cohort_a"]
    cohort_b = prod_data["cohort_b"]
    quotes_by_match = prod_data["quotes_by_match"]
    closing_dict = prod_data["closing_dict"]

    print("[2/5] Pre-extracting multi-horizon snapshots for Cohort A and B...")
    snaps_a = preextract_snapshots(cohort_a, quotes_by_match, closing_dict)
    snaps_b = preextract_snapshots(cohort_b, quotes_by_match, closing_dict)
    print(f"Cohort A snapshots: {len(snaps_a)} | Cohort B snapshots: {len(snaps_b)}")

    # Parameters to sweep
    alphas = [0.35, 0.40, 0.50]
    epsilons = [0.015, 0.020, 0.025]
    guard_bs = [0.85, 0.95, 1.00, 1.10, 1.25]
    max_odds_list = [4.00, None]
    param_grid = list(itertools.product(alphas, epsilons, guard_bs, max_odds_list))

    models = [
        ("C0-D-Core", "p_c0_d_year_core__full"),
        ("C0-Slim", "p_c0_slim_year__full"),
    ]

    all_sweep_results = {}
    top_configs_summary = {}

    for model_name, model_col in models:
        print(f"\n--- Running 90-parameter sweep for {model_name} on Realistic 5-Check Walk-Forward ---")
        model_results = []
        for alpha, eps, guard_b, max_odds in param_grid:
            # 5-Check simulation
            res_5check = run_simulation(snaps_a, model_col, alpha, eps, guard_b, max_odds, schedule_mode="5-Check")
            res_5check.update({
                "model": model_name,
                "alpha": alpha,
                "eps": eps,
                "guard_b": guard_b,
                "max_odds": max_odds,
                "schedule": "5-Check",
            })
            model_results.append(res_5check)

        df_m = pd.DataFrame(model_results)
        all_sweep_results[model_name] = model_results

        # Filter strictly satisfying B_hyb < Net CLV
        feasible = df_m[(df_m["net_clv_pct"] > 0) & (df_m["b_hyb_pp"] < df_m["net_clv_pct"]) & (df_m["n_bets"] > 0)]
        print(f"{model_name}: {len(feasible)} / {len(df_m)} configs strictly satisfy B_hyb < Net CLV (CCR_hyb < 1.0, Edge Retention > 50%)")

        # Stricter: B_DRO <= 0
        stricter_dro = feasible[feasible["b_dro_pp"] <= 0]
        print(f"{model_name}: {len(stricter_dro)} / {len(df_m)} configs strictly satisfy B_DRO <= 0 (CCR_DRO <= 0)")

        # Stricter: B_hyb <= 0.5 * Net CLV
        stricter_half = feasible[feasible["b_hyb_pp"] <= 0.5 * feasible["net_clv_pct"]]
        print(f"{model_name}: {len(stricter_half)} / {len(df_m)} configs strictly satisfy B_hyb <= 0.5 * Net CLV (CCR_hyb <= 0.5)")

        def sanitize_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
            if d is None:
                return None
            res = dict(d)
            if "max_odds" in res and (res["max_odds"] is None or pd.isna(res["max_odds"])):
                res["max_odds"] = None
            return res

        # Best configs by CLV Profit
        if not feasible.empty:
            best_clv_profit = sanitize_dict(feasible.sort_values("total_clv_profit_pln", ascending=False).iloc[0].to_dict())
            best_n_bets = sanitize_dict(feasible.sort_values(["n_bets", "total_clv_profit_pln"], ascending=[False, False]).iloc[0].to_dict())
            best_t_clv = sanitize_dict(feasible.sort_values(["t_clv", "total_clv_profit_pln"], ascending=[False, False]).iloc[0].to_dict())
            feasible_mo4 = feasible[feasible["max_odds"] == 4.0]
            best_mo4_clv = sanitize_dict(feasible_mo4.sort_values("total_clv_profit_pln", ascending=False).iloc[0].to_dict()) if not feasible_mo4.empty else best_clv_profit
        else:
            best_clv_profit = sanitize_dict(df_m.sort_values("total_clv_profit_pln", ascending=False).iloc[0].to_dict())
            best_n_bets = sanitize_dict(df_m.sort_values(["n_bets", "total_clv_profit_pln"], ascending=[False, False]).iloc[0].to_dict())
            best_t_clv = sanitize_dict(df_m.sort_values(["t_clv", "total_clv_profit_pln"], ascending=[False, False]).iloc[0].to_dict())
            best_mo4_clv = best_clv_profit

        top_configs_summary[model_name] = {
            "best_clv_profit": best_clv_profit,
            "best_n_bets": best_n_bets,
            "best_t_clv": best_t_clv,
            "best_max_odds_4": best_mo4_clv,
        }

    # Cross-Cohort Verification
    preds_df = pd.read_parquet(PRED_PATH)
    preds_df["golgg_match_id"] = preds_df["golgg_match_id"].astype(str)
    pol_df, pin_df = load_cross_cohort_data(preds_df)

    verification_results = {}
    for model_name, model_col in models:
        top_cfg = top_configs_summary[model_name]["best_clv_profit"]
        top_mo4 = top_configs_summary[model_name]["best_max_odds_4"]

        # Run on Cohort B (Extended Production DB)
        res_cohort_b_top = run_simulation(snaps_b, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"], top_cfg["max_odds"], schedule_mode="5-Check")
        res_cohort_b_mo4 = run_simulation(snaps_b, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"], top_mo4["max_odds"], schedule_mode="5-Check") if top_mo4 else None

        # Run on Polish Bookmakers (N=1,467)
        res_pol_top = run_cohort_sim(pol_df, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"], top_cfg["max_odds"], "o1_raw", "o2_raw", tax_rate=0.12)
        res_pol_mo4 = run_cohort_sim(pol_df, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"], top_mo4["max_odds"], "o1_raw", "o2_raw", tax_rate=0.12) if top_mo4 else None

        # Run on Pinnacle (N=1,668) under 12% Polish tax and 0% International tax
        res_pin_tax_top = run_cohort_sim(pin_df, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"], top_cfg["max_odds"], "o1", "o2", tax_rate=0.12)
        res_pin_tax_mo4 = run_cohort_sim(pin_df, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"], top_mo4["max_odds"], "o1", "o2", tax_rate=0.12) if top_mo4 else None

        res_pin_0tax_top = run_cohort_sim(pin_df, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"], top_cfg["max_odds"], "o1", "o2", tax_rate=0.00)
        res_pin_0tax_mo4 = run_cohort_sim(pin_df, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"], top_mo4["max_odds"], "o1", "o2", tax_rate=0.00) if top_mo4 else None

        # Horizon Breakdown on Cohort A
        horizons_breakdown = {}
        for h_name in ["T-24h", "T-12h", "T-6h", "T-2h", "T-1h"]:
            h_res = run_simulation(snaps_a, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"], top_cfg["max_odds"], schedule_mode=h_name)
            horizons_breakdown[h_name] = h_res

        verification_results[model_name] = {
            "top_unconstrained_config": {
                "config": {"alpha": top_cfg["alpha"], "eps": top_cfg["eps"], "guard_b": top_cfg["guard_b"], "max_odds": top_cfg["max_odds"]},
                "cohort_a_5check": top_cfg,
                "cohort_b_5check": res_cohort_b_top,
                "cohort_a_individual_horizons": horizons_breakdown,
                "polish_1467_cohort": res_pol_top,
                "pinnacle_1668_tax12": res_pin_tax_top,
                "pinnacle_1668_tax0": res_pin_0tax_top,
            },
            "top_max_odds_4_config": {
                "config": {"alpha": top_mo4["alpha"], "eps": top_mo4["eps"], "guard_b": top_mo4["guard_b"], "max_odds": top_mo4["max_odds"]} if top_mo4 else None,
                "cohort_a_5check": top_mo4,
                "cohort_b_5check": res_cohort_b_mo4,
                "polish_1467_cohort": res_pol_mo4,
                "pinnacle_1668_tax12": res_pin_tax_mo4,
                "pinnacle_1668_tax0": res_pin_0tax_mo4,
            }
        }

    # Structure Output JSON
    output_data = {
        "title": "Empirical Net CLV vs Optimizer's Curse Optimization & Frontier Exploration",
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "methodology": {
            "definition_net_clv": "Net CLV_i = q_close_i * O_i * 0.88 - 1 (Shin devigged closing probability under 12% Polish turnover tax)",
            "definition_ev_hyb": "EV_hyb_i = p_hyb_i * O_i * 0.88 - 1 (Pre-DRO claimed edge)",
            "definition_b_hyb": "B_hyb_i = EV_hyb_i - Net CLV_i = (p_hyb_i - q_close_i) * O_i * 0.88 (Pre-DRO Optimizer's Curse / Selection Disappointment)",
            "definition_ev_dro": "EV_dro_i = (p_hyb_i - eps) * O_i * 0.88 - 1 (Post-DRO claimed edge)",
            "definition_b_dro": "B_dro_i = EV_dro_i - Net CLV_i = (p_hyb_i - eps - q_close_i) * O_i * 0.88 (Post-DRO Post-Decision Surprise)",
            "definition_ccr_hyb": "Curse-to-CLV Ratio CCR_hyb = B_hyb / Net CLV. Hard rule: CCR_hyb < 1.0 <=> B_hyb < Net CLV <=> Edge Retention ERR > 50%",
            "definition_ccr_dro": "Curse-to-CLV Ratio CCR_dro = B_dro / Net CLV. Stricter rule: B_dro <= 0 <=> CCR_dro <= 0 (Conservative Claim)",
            "definition_clv_profit": "Total Expected Luck-Free CLV Profit = sum_i (Stake_i * Net CLV_i) in PLN",
            "t_clv_statistic": "t_CLV = mean(Net CLV_unweighted) / (std(Net CLV_unweighted) / sqrt(N))",
            "simulation_setup": "Realistic 5-Check Walk-Forward (T-24h, T-12h, T-6h, T-2h, T-1h), strictly max 1 bet per match, OperationalWallet 3h locked capital, 10,000 PLN initial capital, 1/4 Kelly sizing [20, 150] PLN",
        },
        "top_configurations_summary": top_configs_summary,
        "verification_results": verification_results,
        "full_grid_sweep": all_sweep_results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n[5/5] Successfully generated and saved: {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size:,} bytes).")
    print(f"Total pipeline elapsed time: {time.time() - t_start_total:.2f} seconds.")


if __name__ == "__main__":
    main()
