#!/usr/bin/env python3
"""Empirical Walk-Forward Optimization: Net CLV vs Optimizer's Curse (B_hyb vs B_DRO).

Executes rigorous walk-forward simulations across:
1. Production PostgreSQL Database (Cohort A: frozen exact matches; Cohort B: strictly later disjoint matches)
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
3. Independent later-period evaluation:
   - Polish and Pinnacle cohorts, plus production matches after Cohort A's end date

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

from betting_app.core.ev import fair_market_probabilities
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

def quotes_observed_by_decision_time(
    quotes: pd.DataFrame,
    decision_time: Any,
    observed_at_col: str = "scraped_dt",
) -> pd.DataFrame:
    """Return quotes known at the decision instant; future observations are never eligible."""
    decision_dt = pd.to_datetime(decision_time, utc=True)
    observed_at = pd.to_datetime(quotes[observed_at_col], utc=True)
    return quotes.loc[observed_at <= decision_dt]


def net_clv_from_close(odds: float, close_probability: float | None, tax_rate: float) -> float | None:
    """Compute net CLV only when an observed closing probability is available."""
    if close_probability is None or pd.isna(close_probability):
        return None
    return float(odds * (1.0 - tax_rate) * close_probability - 1.0)
def observed_clv_rows(bets: pd.DataFrame, clv_col: str = "clv") -> pd.DataFrame:
    """Keep only bets whose CLV uses an observed, independent closing reference."""
    return bets.loc[bets[clv_col].notna()]

def label_cohort_clv(
    result: dict[str, Any],
    clv_by_construction: bool = False,
    net_clv_reason: str | None = None,
) -> dict[str, Any]:
    """Null CLV claims when the cohort lacks an independent closing reference."""
    result["clv_by_construction"] = clv_by_construction
    n_bets = int(result.get("n_bets", 0))
    if clv_by_construction or net_clv_reason:
        result["n_clv_bets"] = 0
        result["n_missing_close"] = n_bets
        result["n_clv_excluded_by_construction"] = n_bets if clv_by_construction else 0
        for key in (
            "net_clv_pct", "net_clv_unweighted_pct", "b_hyb_pp", "b_dro_pp", "ccr_hyb", "ccr_dro",
            "edge_retention_pct", "total_clv_profit_pln", "t_clv", "p_clv",
        ):
            result[key] = None
        result["net_clv_reason"] = net_clv_reason or (
            "Odds are de-vigged fair odds from the same q used as the CLV reference."
        )
    else:
        result.setdefault("n_clv_bets", 0)
        result.setdefault("n_missing_close", n_bets)
        result["n_clv_excluded_by_construction"] = 0
        result.setdefault("net_clv_reason", None)
    return result


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

    q_aligned["p_shin_1"] = [
        fair_market_probabilities(float(odds_1), float(odds_2))[0]
        for odds_1, odds_2 in zip(q_aligned["odds_1"], q_aligned["odds_2"])
    ]

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
    selected_ids = set(cohort_a["canonical_match_id"])
    if cohort_a.empty:
        cohort_b = all_matches_df.iloc[0:0].copy()
    else:
        selection_end = cohort_a["start_dt"].max()
        cohort_b = all_matches_df[
            (all_matches_df["start_dt"] > selection_end)
            & ~all_matches_df["canonical_match_id"].isin(selected_ids)
        ].copy()
    in_sample_selection = cohort_b.empty

    quotes_by_match = {mid: df for mid, df in q_aligned.groupby("canonical_match_id")}

    print(f"Cohort A prepared: {len(cohort_a)} matches")
    print(f"Later disjoint Cohort B prepared: {len(cohort_b)} matches")

    return {
        "cohort_a": cohort_a,
        "cohort_b": cohort_b,
        "in_sample_selection": in_sample_selection,
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

        sub = q_mid[
            (q_mid["scraped_dt"] < t_start)
            & (q_mid["scraped_dt"] >= t_start - pd.Timedelta(hours=72))
        ].copy()
        if sub.empty:
            continue

        first_decision = (t_start - pd.Timedelta(hours=72)).ceil("2h")
        decision_times = pd.date_range(first_decision, t_start, freq="2h", inclusive="left")
        for t_dec in decision_times:
            known = quotes_observed_by_decision_time(sub, t_dec)
            latest = known.sort_values("scraped_dt").groupby("bookmaker_id").tail(1)
            if len(latest) < 3:
                continue

            best_1 = None
            best_2 = None
            for row in latest.itertuples(index=False):
                other = latest[latest["bookmaker_id"] != row.bookmaker_id]
                if other["bookmaker_id"].nunique() < 2:
                    continue
                ql1 = float(other["p_shin_1"].mean())
                ql2 = 1.0 - ql1
                odds_1 = float(row.odds_1)
                odds_2 = float(row.odds_2)
                if odds_1 > 1.0 and (best_1 is None or odds_1 > best_1["odds"]):
                    best_1 = {"odds": odds_1, "q_loo": ql1}
                if odds_2 > 1.0 and (best_2 is None or odds_2 > best_2["odds"]):
                    best_2 = {"odds": odds_2, "q_loo": ql2}


            if best_1 is None and best_2 is None:
                continue

            h_lead = (t_start - t_dec).total_seconds() / 3600.0
            snapshots.append({
                "canonical_match_id": mid,
                "horizon_h": round(h_lead, 1),
                "t_dec": t_dec,
                "t_start": t_start,
                "t_settle": t_start + pd.Timedelta(hours=3),
                "n_books": len(latest),
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
                "p_close_a": float(closing_dict[mid]) if mid in closing_dict and pd.notna(closing_dict[mid]) else np.nan,
                "market_prefilter_applied": False,
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

    net_clv = net_clv_from_close(odds, p_close, tax_rate)
    b_hyb = ev_claim - net_clv if net_clv is not None else None
    b_dro = dro_ev - net_clv if net_clv is not None else None

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
            "realized_yield_pct": 0.0, "net_clv_pct": None, "ev_hyb_pct": None, "ev_dro_pct": None,
            "b_hyb_pp": None, "b_dro_pp": None, "ccr_hyb": None, "ccr_dro": None, "edge_retention_pct": None,
            "total_clv_profit_pln": None, "net_clv_unweighted_pct": None, "t_clv": None, "p_clv": None,
            "n_clv_bets": 0, "n_missing_close": 0, "net_clv_reason": "no_bets_with_independent_close",
            "max_drawdown_pct": 0.0, "profit_factor": 0.0, "sharpe": 0.0, "capital_exhausted": capital_exhausted_bets,
            "final_equity_pln": initial_capital,
        }

    n = len(settled)
    n_wins = int(settled["won"].sum())
    turnover = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    yd = (profit / turnover) * 100.0 if turnover > 0 else 0.0
    wr = float(settled["won"].mean()) * 100.0

    clv_bets = observed_clv_rows(settled)
    n_clv_bets = len(clv_bets)
    n_missing_close = n - n_clv_bets
    if n_clv_bets:
        clv_turnover = float(clv_bets["stake"].sum())
        weights = clv_bets["stake"].to_numpy(dtype=float) / clv_turnover
        net_clvs = clv_bets["clv"].to_numpy(dtype=float)
        ev_claims = clv_bets["ev_claimed"].to_numpy(dtype=float)
        odds_arr = clv_bets["odds"].to_numpy(dtype=float)
        net_clv_weighted = float(np.sum(weights * net_clvs)) * 100.0
        ev_hyb_weighted = float(np.sum(weights * ev_claims)) * 100.0
        b_hyb_weighted = ev_hyb_weighted - net_clv_weighted
        ev_dro_weighted = float(np.sum(weights * (ev_claims - eps * c_tax * odds_arr))) * 100.0
        b_dro_weighted = ev_dro_weighted - net_clv_weighted
        ccr_hyb = b_hyb_weighted / net_clv_weighted if net_clv_weighted != 0 else None
        ccr_dro = b_dro_weighted / net_clv_weighted if net_clv_weighted != 0 else None
        net_clv_mean_unw = float(np.mean(net_clvs))
        net_clv_std_unw = float(np.std(net_clvs, ddof=1)) if n_clv_bets > 1 else 0.0
        t_clv = (net_clv_mean_unw / (net_clv_std_unw / np.sqrt(n_clv_bets))) if n_clv_bets > 1 and net_clv_std_unw > 0 else None
        p_clv = float(1.0 - t_dist.cdf(t_clv, df=n_clv_bets - 1)) if t_clv is not None else None
        total_clv_profit_pln = float(np.sum(clv_bets["stake"].to_numpy(dtype=float) * net_clvs))
        clv_reason = None
    else:
        net_clv_weighted = ev_hyb_weighted = ev_dro_weighted = None
        b_hyb_weighted = b_dro_weighted = ccr_hyb = ccr_dro = None
        net_clv_mean_unw = t_clv = p_clv = total_clv_profit_pln = None
        clv_reason = "no_bets_with_independent_close"

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
        "n_clv_bets": n_clv_bets,
        "n_missing_close": n_missing_close,
        "net_clv_reason": clv_reason,
        "n_wins": n_wins,
        "win_rate_pct": round(wr, 2),
        "turnover_pln": round(turnover, 2),
        "profit_pln": round(profit, 2),
        "realized_yield_pct": round(yd, 2),
        "net_clv_pct": round(net_clv_weighted, 2) if net_clv_weighted is not None else None,
        "ev_hyb_pct": round(ev_hyb_weighted, 2) if ev_hyb_weighted is not None else None,
        "ev_dro_pct": round(ev_dro_weighted, 2) if ev_dro_weighted is not None else None,
        "b_hyb_pp": round(b_hyb_weighted, 2) if b_hyb_weighted is not None else None,
        "b_dro_pp": round(b_dro_weighted, 2) if b_dro_weighted is not None else None,
        "ccr_hyb": round(ccr_hyb, 3) if ccr_hyb is not None else None,
        "ccr_dro": round(ccr_dro, 3) if ccr_dro is not None else None,
        "edge_retention_pct": round((net_clv_weighted / ev_hyb_weighted) * 100.0, 2) if net_clv_weighted is not None and ev_hyb_weighted != 0 else None,
        "total_clv_profit_pln": round(total_clv_profit_pln, 2) if total_clv_profit_pln is not None else None,
        "net_clv_unweighted_pct": round(net_clv_mean_unw * 100.0, 2) if net_clv_mean_unw is not None else None,
        "t_clv": round(float(t_clv), 3) if t_clv is not None else None,
        "p_clv": round(p_clv, 4) if p_clv is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "sharpe": round(float(sharpe), 2),
        "capital_exhausted": capital_exhausted_bets,
        "final_equity_pln": round(wallet.total_equity, 2),
    }


# -----------------------------------------------------------------------------
# 3. Independent Later-Period Evaluation (Polish & Pinnacle)
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
    clv_by_construction: bool = False,
    close_reference_available: bool = True,
    net_clv_reason: str | None = None,
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
            close_ref = q1[i] if close_reference_available else None
            side, o_taken, p_claim, p_close, won, ev_claim = 1, o1[i], p1_h[i], close_ref, int(y[i] == 1), ev1[i]
        elif c2[i]:
            close_ref = q1[i] if close_reference_available else None
            p_close = 1.0 - close_ref if close_ref is not None else None
            side, o_taken, p_claim, won, ev_claim = 2, o2[i], p2_h[i], int(y[i] == 0), ev2[i]

        if side != 0:
            mid = df["match_id"].iloc[i]
            t_dec = df["t_dec"].iloc[i]
            t_settle = df["t_settle"].iloc[i]
            net_clv = net_clv_from_close(o_taken, p_close, tax_rate)
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
        return label_cohort_clv({
            "n_bets": 0,
            "n_clv_bets": 0,
            "n_missing_close": 0,
            "net_clv_pct": None,
            "ev_hyb_pct": None,
            "ev_dro_pct": None,
            "b_hyb_pp": None,
            "b_dro_pp": None,
            "ccr_hyb": None,
            "ccr_dro": None,
            "edge_retention_pct": None,
            "total_clv_profit_pln": None,
            "profit_pln": 0.0,
            "realized_yield_pct": 0.0,
            "final_equity_pln": initial_capital,
        }, clv_by_construction, net_clv_reason)

    n = len(settled)
    turnover = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    yd = (profit / turnover) * 100.0 if turnover > 0 else 0.0
    all_weights = settled["stake"].to_numpy(dtype=float) / turnover
    all_ev_claims = settled["ev_claimed"].to_numpy(dtype=float)
    ev_hyb_weighted = float(np.sum(all_weights * all_ev_claims)) * 100.0
    odds_all = settled["odds"].to_numpy(dtype=float)
    ev_dro_weighted = float(np.sum(all_weights * (all_ev_claims - eps * c_tax * odds_all))) * 100.0

    clv_bets = observed_clv_rows(settled)
    if clv_by_construction:
        clv_bets = clv_bets.iloc[0:0]
    n_clv_bets = len(clv_bets)
    n_missing_close = n - n_clv_bets
    if n_clv_bets:
        clv_turnover = float(clv_bets["stake"].sum())
        clv_weights = clv_bets["stake"].to_numpy(dtype=float) / clv_turnover
        net_clvs = clv_bets["clv"].to_numpy(dtype=float)
        ev_claims = clv_bets["ev_claimed"].to_numpy(dtype=float)
        net_clv_weighted = float(np.sum(clv_weights * net_clvs)) * 100.0
        ev_hyb_clv_weighted = float(np.sum(clv_weights * ev_claims)) * 100.0
        bias_hyb = ev_hyb_clv_weighted - net_clv_weighted
        clv_yield = float(clv_bets["profit"].sum() / clv_turnover) * 100.0
        luck_eta = clv_yield - net_clv_weighted
        odds_arr = clv_bets["odds"].to_numpy(dtype=float)
        ev_dro_clv = ev_claims - eps * c_tax * odds_arr
        bias_dro = float(np.sum(clv_weights * ev_dro_clv)) * 100.0 - net_clv_weighted
        ccr_hyb = bias_hyb / net_clv_weighted if net_clv_weighted != 0 else None
        ccr_dro = bias_dro / net_clv_weighted if net_clv_weighted != 0 else None
        total_clv_profit_pln = float(np.sum(clv_bets["stake"].to_numpy(dtype=float) * net_clvs))
        net_clv_mean_unw = float(np.mean(net_clvs))
        net_clv_std_unw = float(np.std(net_clvs, ddof=1)) if n_clv_bets > 1 else 0.0
        t_clv = net_clv_mean_unw / (net_clv_std_unw / np.sqrt(n_clv_bets)) if n_clv_bets > 1 and net_clv_std_unw > 0 else None
        p_clv = float(1.0 - t_dist.cdf(t_clv, df=n_clv_bets - 1)) if t_clv is not None else None
        resolved_clv_reason = None
    else:
        net_clv_weighted = bias_hyb = bias_dro = luck_eta = None
        ccr_hyb = ccr_dro = total_clv_profit_pln = net_clv_mean_unw = t_clv = p_clv = None
        resolved_clv_reason = net_clv_reason or (
            "Odds are de-vigged fair odds from the same q used as the CLV reference."
            if clv_by_construction else "no_bets_with_independent_close"
        )

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

    return label_cohort_clv({
        "n_bets": n,
        "n_clv_bets": n_clv_bets,
        "n_missing_close": n_missing_close,
        "win_rate_pct": round(float(settled["won"].mean()) * 100.0, 2),
        "turnover_pln": round(turnover, 2),
        "profit_pln": round(profit, 2),
        "realized_yield_pct": round(yd, 2),
        "ev_hyb_pct": round(ev_hyb_weighted, 2),
        "ev_dro_pct": round(ev_dro_weighted, 2),
        "net_clv_pct": round(net_clv_weighted, 2) if net_clv_weighted is not None else None,
        "net_clv_unweighted_pct": round(net_clv_mean_unw * 100.0, 2) if net_clv_mean_unw is not None else None,
        "b_hyb_pp": round(bias_hyb, 2) if bias_hyb is not None else None,
        "b_dro_pp": round(bias_dro, 2) if bias_dro is not None else None,
        "ccr_hyb": round(ccr_hyb, 3) if ccr_hyb is not None else None,
        "ccr_dro": round(ccr_dro, 3) if ccr_dro is not None else None,
        "edge_retention_pct": round((net_clv_weighted / ev_hyb_weighted) * 100.0, 2) if net_clv_weighted is not None and ev_hyb_weighted != 0 else None,
        "total_clv_profit_pln": round(total_clv_profit_pln, 2) if total_clv_profit_pln is not None else None,
        "t_clv": round(float(t_clv), 3) if t_clv is not None else None,
        "p_clv": round(p_clv, 4) if p_clv is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "final_equity_pln": round(wallet.total_equity, 2),
    }, clv_by_construction, resolved_clv_reason)


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

    in_sample_selection = prod_data["in_sample_selection"]
    print("[2/5] Pre-extracting causal 2-hour snapshots for Cohort A and later disjoint Cohort B...")
    snaps_a = preextract_snapshots(cohort_a, quotes_by_match, closing_dict)
    snaps_b = preextract_snapshots(cohort_b, quotes_by_match, closing_dict)
    print(f"Cohort A snapshots: {len(snaps_a)} | Later disjoint Cohort B snapshots: {len(snaps_b)}")

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
        print(f"\n--- Running {len(param_grid)}-cell selection grid for {model_name} on Cohort A ---")
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
                "grid_cells_tried": len(param_grid),
                "market_prefilter_applied": False,
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

        # A configuration is selectable only when observed independent-close CLV
        # satisfies the stated eligibility rule; never promote a fallback row.
        if not feasible.empty:
            best_clv_profit = sanitize_dict(feasible.sort_values("total_clv_profit_pln", ascending=False).iloc[0].to_dict())
            best_n_bets = sanitize_dict(feasible.sort_values(["n_bets", "total_clv_profit_pln"], ascending=[False, False]).iloc[0].to_dict())
            best_t_clv = sanitize_dict(feasible.sort_values(["t_clv", "total_clv_profit_pln"], ascending=[False, False]).iloc[0].to_dict())
            feasible_mo4 = feasible[feasible["max_odds"] == 4.0]
            best_mo4_clv = sanitize_dict(feasible_mo4.sort_values("total_clv_profit_pln", ascending=False).iloc[0].to_dict()) if not feasible_mo4.empty else best_clv_profit
        else:
            best_clv_profit = None
            best_n_bets = None
            best_t_clv = None
            best_mo4_clv = None

        top_configs_summary[model_name] = {
            "selection_status": "SELECTED" if best_clv_profit is not None else "NOT_EVALUATED",
            "selection_reason": None if best_clv_profit is not None else "no_configuration_with_eligible_independent_close_CLV",
            "best_clv_profit": best_clv_profit,
            "best_n_bets": best_n_bets,
            "best_t_clv": best_t_clv,
            "best_max_odds_4": best_mo4_clv,
            "grid_cells_tried": len(param_grid),
        }

    # Evaluate only a genuinely selected configuration against a later,
    # match-disjoint production cohort. An empty Cohort B is not evaluation.
    independent_evaluation_results = {
        "status": "NOT_EVALUATED",
        "reason": (
            "no_later_disjoint_cohort_b"
            if in_sample_selection
            else "no_configuration_with_eligible_independent_close_CLV"
        ),
    }
    if not in_sample_selection and any(
        summary["best_clv_profit"] is not None
        for summary in top_configs_summary.values()
    ):
        preds_df = pd.read_parquet(PRED_PATH)
        preds_df["golgg_match_id"] = preds_df["golgg_match_id"].astype(str)
        pol_df, pin_df = load_cross_cohort_data(preds_df)

        independent_evaluation_results = {}
        for model_name, model_col in models:
            top_cfg = top_configs_summary[model_name]["best_clv_profit"]
            top_mo4 = top_configs_summary[model_name]["best_max_odds_4"]
            if top_cfg is None:
                independent_evaluation_results[model_name] = {
                    "status": "NOT_EVALUATED",
                    "reason": "no_configuration_with_eligible_independent_close_CLV",
                }
                continue

            res_cohort_b_top = run_simulation(
                snaps_b, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"],
                top_cfg["max_odds"], schedule_mode="5-Check",
            )
            res_cohort_b_mo4 = run_simulation(
                snaps_b, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"],
                top_mo4["max_odds"], schedule_mode="5-Check",
            ) if top_mo4 else None
            polish_clv_reason = "independent closing reference unavailable; best odds and consensus both use closing market"
            pinnacle_clv_reason = "fair Pinnacle odds are inverse of the same q used as CLV reference"

            # Run on Polish Bookmakers (N=1,467)
            res_pol_top = run_cohort_sim(
                pol_df, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"],
                top_cfg["max_odds"], "o1_raw", "o2_raw", tax_rate=0.12,
                close_reference_available=False, net_clv_reason=polish_clv_reason,
            )
            res_pol_mo4 = run_cohort_sim(
                pol_df, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"],
                top_mo4["max_odds"], "o1_raw", "o2_raw", tax_rate=0.12,
                close_reference_available=False, net_clv_reason=polish_clv_reason,
            ) if top_mo4 else None

            # Pinnacle odds are exact fair-odds inverses of the same q used as reference.
            res_pin_tax_top = run_cohort_sim(
                pin_df, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"],
                top_cfg["max_odds"], "o1", "o2", tax_rate=0.12,
                clv_by_construction=True, close_reference_available=False,
                net_clv_reason=pinnacle_clv_reason,
            )
            res_pin_tax_mo4 = run_cohort_sim(
                pin_df, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"],
                top_mo4["max_odds"], "o1", "o2", tax_rate=0.12,
                clv_by_construction=True, close_reference_available=False,
                net_clv_reason=pinnacle_clv_reason,
            ) if top_mo4 else None
            res_pin_0tax_top = run_cohort_sim(
                pin_df, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"],
                top_cfg["max_odds"], "o1", "o2", tax_rate=0.00,
                clv_by_construction=True, close_reference_available=False,
                net_clv_reason=pinnacle_clv_reason,
            )
            res_pin_0tax_mo4 = run_cohort_sim(
                pin_df, model_col, top_mo4["alpha"], top_mo4["eps"], top_mo4["guard_b"],
                top_mo4["max_odds"], "o1", "o2", tax_rate=0.00,
                clv_by_construction=True, close_reference_available=False,
                net_clv_reason=pinnacle_clv_reason,
            ) if top_mo4 else None

            horizons_breakdown = {}
            for h_name in ["T-24h", "T-12h", "T-6h", "T-2h", "T-1h"]:
                h_res = run_simulation(snaps_a, model_col, top_cfg["alpha"], top_cfg["eps"], top_cfg["guard_b"], top_cfg["max_odds"], schedule_mode=h_name)
                horizons_breakdown[h_name] = h_res

            independent_evaluation_results[model_name] = {
                "status": "EVALUATED",
                "top_unconstrained_config": {
                    "config": {"alpha": top_cfg["alpha"], "eps": top_cfg["eps"], "guard_b": top_cfg["guard_b"], "max_odds": top_cfg["max_odds"]},
                    "cohort_a_5check": top_cfg,
                    "later_disjoint_cohort_b_5check": res_cohort_b_top,
                    "cohort_a_individual_horizons": horizons_breakdown,
                    "polish_1467_cohort": res_pol_top,
                    "pinnacle_1668_tax12": res_pin_tax_top,
                    "pinnacle_1668_tax0": res_pin_0tax_top,
                },
                "top_max_odds_4_config": {
                    "config": {"alpha": top_mo4["alpha"], "eps": top_mo4["eps"], "guard_b": top_mo4["guard_b"], "max_odds": top_mo4["max_odds"]} if top_mo4 else None,
                    "cohort_a_5check": top_mo4,
                    "later_disjoint_cohort_b_5check": res_cohort_b_mo4,
                    "polish_1467_cohort": res_pol_mo4,
                    "pinnacle_1668_tax12": res_pin_tax_mo4,
                    "pinnacle_1668_tax0": res_pin_0tax_mo4,
                }
            }


    # Structure Output JSON
    output_data = {
        "title": "Empirical Net CLV vs Optimizer's Curse Optimization & Frontier Exploration",
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "selection": {
            "selected_on": "cohort_a",
            "grid_cells_tried_per_model": len(param_grid),
            "later_disjoint_cohort_b_available": not in_sample_selection,
            "in_sample_selection": in_sample_selection,
            "verification_status": (
                "EVALUATED"
                if not in_sample_selection and any(
                    summary["best_clv_profit"] is not None
                    for summary in top_configs_summary.values()
                )
                else "NOT_EVALUATED"
            ),
            "verification_reason": (
                None
                if not in_sample_selection and any(
                    summary["best_clv_profit"] is not None
                    for summary in top_configs_summary.values()
                )
                else (
                    "no_later_disjoint_cohort_b"
                    if in_sample_selection
                    else "no_configuration_with_eligible_independent_close_CLV"
                )
            ),
        },
        "methodology": {
            "definition_net_clv": "Net CLV_i = q_close_i * O_i * (1 - tax) - 1, only for bets with an observed independent pre-kickoff closing probability.",
            "definition_ev_hyb": "EV_hyb_i = p_hyb_i * O_i * (1 - tax) - 1 (Pre-DRO claimed edge)",
            "definition_b_hyb": "B_hyb_i = EV_hyb_i - Net CLV_i, computed only on bets with an observed independent close.",
            "definition_ev_dro": "EV_dro_i = (p_hyb_i - eps) * O_i * (1 - tax) - 1 (Post-DRO claimed edge)",
            "definition_b_dro": "B_dro_i = EV_dro_i - Net CLV_i, computed only on bets with an observed independent close.",
            "definition_ccr_hyb": "Curse-to-CLV Ratio CCR_hyb = B_hyb / Net CLV; null when no valid close statistic is available.",
            "definition_ccr_dro": "Curse-to-CLV Ratio CCR_dro = B_dro / Net CLV; null when no valid close statistic is available.",
            "definition_clv_profit": "Total Expected CLV Profit = sum_i (Stake_i * Net CLV_i), only for observed independent closes.",
            "market_prefilter_applied": False,
            "pinnacle_clv_by_construction": "Pinnacle fair odds are inverse probabilities from the same q used as the reference; all CLV-derived output fields are null.",
            "simulation_setup": "Causal 2-hour decision snapshots use only quotes observed at or before decision time; 5-Check Walk-Forward, max 1 bet per match, OperationalWallet 3-hour locked capital, quarter Kelly capped at [20, 150] PLN.",
        },
        "top_configurations_summary": top_configs_summary,
        "independent_evaluation_results": independent_evaluation_results,
        "full_grid_sweep": all_sweep_results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n[5/5] Successfully generated and saved: {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size:,} bytes).")
    print(f"Total pipeline elapsed time: {time.time() - t_start_total:.2f} seconds.")


if __name__ == "__main__":
    main()
