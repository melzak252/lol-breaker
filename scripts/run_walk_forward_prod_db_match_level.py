#!/usr/bin/env python3
"""Realistic Match-Level Chronological Walk-Forward Simulation (1 decision per match).

Simulates production betting operations across ALL settled matches in the Production PostgreSQL Database
(data/01_raw/c0_production_counterfactual/c0_production_20261006_01/):
- Strictly 1 decision evaluation per match (NO multi-snapshot scanning).
- Strict chronological walk-forward with OperationalWallet capital locking:
  * 10,000 PLN initial capital
  * 12% Polish tax (net decimal = 0.88 * odds)
  * 1/4 Kelly sizing: f* = [b*p - (1-p)]/b * 0.25, capped at [20.00, 150.00] PLN
  * Stake reserved at t_dec from free_capital, settled at t_start + 3h.
- 6 Walk-Forward Schedules:
  1. T-24h Fixed Walk-Forward
  2. T-12h Fixed Walk-Forward
  3. T-6h Fixed Walk-Forward
  4. T-2h Fixed Walk-Forward
  5. T-1h Fixed Walk-Forward
  6. Twice-Daily Human Check (10:00 & 18:00 UTC)
- 7 Gate Configurations:
  * G0: Full Stack + EXP-039 Veto (DRO eps=0.025, Guard B <= 0.85, MaxOdds <= 4.0, MaxEV <= 25%, MinEV >= 5%, Veto <= 0.10)
  * G1: Full Stack, No Veto (DRO eps=0.025, Guard B <= 0.85, MaxOdds <= 4.0, MaxEV <= 25%, MinEV >= 5%)
  * G3: Turn OFF Max EV <= 25%
  * G4: Turn OFF Max Odds <= 4.0 & Max EV <= 25%
  * G5: ONLY Wasserstein DRO (eps=0.025, DRO_EV > 0%) + Guard B <= 0.85 (NO MaxOdds, NO MaxEV, NO MinEV, NO Veto)
  * G7b: Minimalist (ONLY Hybrid-50 + Wasserstein DRO > 0%, all other gates OFF)
  * G9: Control (DRO eps=0.0 OFF, keep Guard B, MaxOdds, MaxEV, MinEV)
- Evaluates:
  * C0-D-Core (18-Token) (p_c0_d_year_core__full)
  * C0-Slim (18-Token) (p_c0_slim_year__full)
  * C0 (Baseline) (p_c0__full)
across Cohort A (the production training set) and a strictly later, match-disjoint Cohort B.

Saves structured JSON to:
  data/08_reporting/c0_upgrade_and_distillation_20261017/walk_forward_prod_db_match_level.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.historical_production_simulator import OperationalWallet
from betting_app.core.ev import fair_market_probabilities


# -----------------------------------------------------------------------------
# 1. Configuration & Asset Paths
# -----------------------------------------------------------------------------
RAW_PROD_DIR = PROJECT_ROOT / "data/01_raw/c0_production_counterfactual/c0_production_20261006_01"
REP_DIR = PROJECT_ROOT / "data/08_reporting/c0_production_counterfactual/c0_production_20261006_01"
MATCHES_PATH = RAW_PROD_DIR / "matches.parquet"
BOOKMAKERS_PATH = RAW_PROD_DIR / "bookmakers.parquet"
MAPPINGS_PATH = RAW_PROD_DIR / "mappings.parquet"
QUOTES_PATH = RAW_PROD_DIR / "quotes.parquet"
QUOTE_ALIGNMENT_PATH = REP_DIR / "quote_alignment.parquet"
IDENTITIES_PATH = REP_DIR / "identities.parquet"

REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
PRED_PATH = REPORT_DIR / "year_anchor_distilled_predictions.parquet"
ALL_2026_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/all_2026_joined.parquet"
SCORED_PANEL_PATH = REP_DIR / "scored_panel.parquet"
MARKET_T1_PATH = PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet"

OUTPUT_JSON_PATH = REPORT_DIR / "walk_forward_prod_db_match_level.json"

# -----------------------------------------------------------------------------
# 2. Decision Schedule Definitions
# -----------------------------------------------------------------------------
SCHEDULE_SPECS = [
    ("T-24h", "T-24h Fixed Walk-Forward (1 decision per match at kickoff - 24 hours)"),
    ("T-12h", "T-12h Fixed Walk-Forward (1 decision per match at kickoff - 12 hours)"),
    ("T-6h",  "T-6h Fixed Walk-Forward (1 decision per match at kickoff - 6 hours)"),
    ("T-2h",  "T-2h Fixed Walk-Forward (1 decision per match at kickoff - 2 hours)"),
    ("T-1h",  "T-1h Fixed Walk-Forward (1 decision per match at kickoff - 1 hour)"),
    ("Twice-Daily", "Twice-Daily Human Check (10:00 & 18:00 UTC, evaluate upcoming matches within [T-24h, T-1h] once)"),
]

GATE_SPECS = [
    ("G0", "G0 (Full Stack + EXP-039 Veto)"),
    ("G1", "G1 (Full Stack: Hybrid-50 + DRO eps=0.025 + GuardB<=0.85 + MaxOdds<=4.0 + MaxEV<=25% + MinEV>=5%)"),
    ("G3", "G3 (Turn OFF Max EV <= 25%)"),
    ("G4", "G4 (Turn OFF Max Odds <= 4.0 & Max EV <= 25%)"),
    ("G5", "G5 (ONLY Wasserstein DRO eps=0.025, DRO>0% + Logit Disagreement GuardB <= 0.85 — NO MaxOdds, NO MaxEV, NO MinEV, NO Veto)"),
    ("G7b", "G7b (Minimalist: ONLY Hybrid-50 + Wasserstein DRO > 0% — All Other Gates OFF)"),
    ("G9", "G9 (Control: Turn OFF DRO eps=0.0, keep GuardB/MaxOdds/MaxEV/MinEV)"),
]

def quotes_observed_by_decision_time(
    quotes: pd.DataFrame,
    decision_time: Any,
    observed_at_col: str = "scraped_dt",
) -> pd.DataFrame:
    """Exclude any quote whose observation timestamp is after the decision."""
    decision_dt = pd.to_datetime(decision_time, utc=True)
    observed_at = pd.to_datetime(quotes[observed_at_col], utc=True)
    return quotes.loc[observed_at <= decision_dt]


def net_clv_from_close(odds: float, close_probability: float | None, tax_rate: float) -> float | None:
    """Return CLV only when an observed independent closing probability exists."""
    if close_probability is None or pd.isna(close_probability):
        return None
    return float(odds * (1.0 - tax_rate) * close_probability - 1.0)


def observed_clv_rows(bets: pd.DataFrame, clv_col: str = "clv") -> pd.DataFrame:
    """Exclude unsettled-reference bets from all CLV-derived statistics."""
    return bets.loc[bets[clv_col].notna()]


def missing_close_count(bets: pd.DataFrame, clv_col: str = "clv") -> int:
    """Count bets excluded from CLV analysis because their close is unavailable."""
    return int(bets[clv_col].isna().sum())


# -----------------------------------------------------------------------------
# 3. Data Loading & Alignment
# -----------------------------------------------------------------------------
def load_and_prepare_data():
    print("Loading raw production matches, quotes, alignments, identities, and predictions...")
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
    p039_lookup = {}
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
            p039_raw = yrow.get("p_039", np.nan)
            p039 = float(p039_raw) if not pd.isna(p039_raw) else np.nan
            if not np.isnan(p039) and not same:
                p039 = 1.0 - p039
            is_cohort_a = True
        elif gid_str in sc.index:
            p_val = float(sc.loc[gid_str, "p_C0"])
            p_c0d = p_val
            p_slim = p_val
            p_c0 = p_val
            p039 = np.nan
            is_cohort_a = False
        else:
            continue

        p039_lookup[gid_str] = p039
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
            "p_039": p039,
        })

    all_matches_df = pd.DataFrame(match_rows)
    cohort_a_matches = all_matches_df[all_matches_df["is_cohort_a"]].copy()
    selected_ids = set(cohort_a_matches["canonical_match_id"])
    if cohort_a_matches.empty:
        cohort_b_matches = all_matches_df.iloc[0:0].copy()
    else:
        selection_end = cohort_a_matches["start_dt"].max()
        cohort_b_matches = all_matches_df[
            (all_matches_df["start_dt"] > selection_end)
            & ~all_matches_df["canonical_match_id"].isin(selected_ids)
        ].copy()
    in_sample_selection = cohort_b_matches.empty

    print(f"Cohort A prepared: {len(cohort_a_matches)} matches")
    print(f"Later disjoint Cohort B prepared: {len(cohort_b_matches)} matches")

    return {
        "quotes": q_aligned,
        "cohort_a": cohort_a_matches,
        "cohort_b": cohort_b_matches,
        "in_sample_selection": in_sample_selection,
        "closing_dict": closing_dict,
        "p039_lookup": p039_lookup,
    }


# -----------------------------------------------------------------------------
# 4. Decision Extraction for All 6 Schedules
# -----------------------------------------------------------------------------
def extract_decisions_for_schedule(
    matches_df: pd.DataFrame,
    quotes_aligned: pd.DataFrame,
    schedule_key: str,
    closing_dict: dict[int, float],
) -> list[dict[str, Any]]:
    quotes_by_match = {mid: df for mid, df in quotes_aligned.groupby("canonical_match_id")}
    decisions = []

    if schedule_key in ("T-24h", "T-12h", "T-6h", "T-2h", "T-1h"):
        hours_map = {"T-24h": 24.0, "T-12h": 12.0, "T-6h": 6.0, "T-2h": 2.0, "T-1h": 1.0}
        h = hours_map[schedule_key]

        for _, m in matches_df.iterrows():
            mid = int(m["canonical_match_id"])
            t_start = m["start_dt"]
            t_dec = t_start - pd.Timedelta(hours=h)

            q_mid = quotes_by_match.get(mid)
            if q_mid is None:
                continue

            # <= 12h freshness window before t_dec
            valid_q = quotes_observed_by_decision_time(q_mid, t_dec)
            valid_q = valid_q[(valid_q["scraped_dt"] < t_start) & (valid_q["scraped_dt"] >= t_dec - pd.Timedelta(hours=12))]
            if valid_q.empty:
                continue

            latest_b = valid_q.groupby("bookmaker_id").last()
            if len(latest_b) < 3:
                continue

            best_1 = None
            best_2 = None
            bks_list = latest_b.index.tolist()
            for bk in bks_list:
                row = latest_b.loc[bk]
                oth = latest_b.drop(bk)
                if oth.index.nunique() < 2:
                    continue
                o1 = float(row["odds_1"])
                o2 = float(row["odds_2"])
                ql1 = float(oth["p_shin_1"].mean())
                ql2 = 1.0 - ql1
                if o1 > 1.0 and (best_1 is None or o1 > best_1["odds"]):
                    best_1 = {"odds": o1, "q_loo": ql1}
                if o2 > 1.0 and (best_2 is None or o2 > best_2["odds"]):
                    best_2 = {"odds": o2, "q_loo": ql2}


            if best_1 is None and best_2 is None:
                continue

            decisions.append({
                "canonical_match_id": mid,
                "t_dec": t_dec,
                "t_start": t_start,
                "t_settle": t_start + pd.Timedelta(hours=3),
                "n_books": len(latest_b),
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
                "p_close_a": float(closing_dict[mid]) if mid in closing_dict else np.nan,
                "market_prefilter_applied": False,
            })

    elif schedule_key == "Twice-Daily":
        dates = pd.date_range("2026-05-28", "2026-09-23", freq="D", tz="UTC")
        check_times = []
        for d in dates:
            check_times.append(d + pd.Timedelta(hours=10))
            check_times.append(d + pd.Timedelta(hours=18))
        check_times.sort()

        evaluated_mids = set()
        matches_sorted = matches_df.sort_values("start_dt")

        for t_check in check_times:
            w_start = t_check + pd.Timedelta(hours=1)
            w_end = t_check + pd.Timedelta(hours=24)

            for _, m in matches_sorted.iterrows():
                mid = int(m["canonical_match_id"])
                if mid in evaluated_mids:
                    continue
                t_start = m["start_dt"]
                if not (w_start <= t_start <= w_end):
                    continue

                q_mid = quotes_by_match.get(mid)
                if q_mid is None:
                    continue

                valid_q = quotes_observed_by_decision_time(q_mid, t_check)
                valid_q = valid_q[(valid_q["scraped_dt"] < t_start) & (valid_q["scraped_dt"] >= t_check - pd.Timedelta(hours=12))]
                if valid_q.empty:
                    continue

                latest_b = valid_q.groupby("bookmaker_id").last()
                if len(latest_b) < 3:
                    continue

                best_1 = None
                best_2 = None
                bks_list = latest_b.index.tolist()
                for bk in bks_list:
                    row = latest_b.loc[bk]
                    oth = latest_b.drop(bk)
                    if oth.index.nunique() < 2:
                        continue
                    o1 = float(row["odds_1"])
                    o2 = float(row["odds_2"])
                    ql1 = float(oth["p_shin_1"].mean())
                    ql2 = 1.0 - ql1
                    if o1 > 1.0 and (best_1 is None or o1 > best_1["odds"]):
                        best_1 = {"odds": o1, "q_loo": ql1}
                    if o2 > 1.0 and (best_2 is None or o2 > best_2["odds"]):
                        best_2 = {"odds": o2, "q_loo": ql2}

                if best_1 is not None or best_2 is not None:
                    evaluated_mids.add(mid)
                    decisions.append({
                        "canonical_match_id": mid,
                        "t_dec": t_check,
                        "t_start": t_start,
                        "t_settle": t_start + pd.Timedelta(hours=3),
                        "n_books": len(latest_b),
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
                        "p_close_a": float(closing_dict[mid]) if mid in closing_dict else np.nan,
                        "market_prefilter_applied": False,
                    })

    return decisions


# -----------------------------------------------------------------------------
# 5. Core Walk-Forward Simulation Engine with OperationalWallet
# -----------------------------------------------------------------------------
def simulate_walk_forward(
    decisions: list[dict[str, Any]],
    model_col: str,
    gate_name: str,
    p039_lookup: dict[str, float],
    initial_capital: float = 10000.0,
    tax_rate: float = 0.12,
) -> dict[str, Any]:
    wallet = OperationalWallet(initial_capital=initial_capital)
    c_tax = 1.0 - tax_rate

    # 1. Candidate Decision Evaluation
    qualified_bets: dict[int, dict[str, Any]] = {}

    for d in decisions:
        mid = d["canonical_match_id"]
        p_raw = d[model_col]
        a_is_pred_t1 = d["a_is_pred_t1"]
        p_a = p_raw if a_is_pred_t1 else 1.0 - p_raw
        p_b = 1.0 - p_a

        q_a = d["q_mkt_a"]
        q_b = d["q_mkt_b"]
        oa = d["best_odds_a"]
        ob = d["best_odds_b"]

        def qualifies(ev: float, dro_ev: float, gap: float, odds: float, veto: bool) -> bool:
            if gate_name == "G0":
                return (dro_ev > 0.0) and (gap <= 0.85) and (odds <= 4.0) and (ev <= 0.25) and (ev >= 0.05) and veto
            elif gate_name == "G1":
                return (dro_ev > 0.0) and (gap <= 0.85) and (odds <= 4.0) and (ev <= 0.25) and (ev >= 0.05)
            elif gate_name == "G3":
                return (dro_ev > 0.0) and (gap <= 0.85) and (odds <= 4.0) and (ev >= 0.05)
            elif gate_name == "G4":
                return (dro_ev > 0.0) and (gap <= 0.85) and (ev >= 0.05)
            elif gate_name == "G5":
                # ONLY Wasserstein DRO eps=0.025 + Guard B <= 0.85 (NO MaxOdds, NO MaxEV, NO MinEV, NO Veto)
                return (dro_ev > 0.0) and (gap <= 0.85)
            elif gate_name == "G7b":
                # Minimalist: ONLY Hybrid-50 + DRO > 0%
                return (dro_ev > 0.0)
            elif gate_name == "G9":
                # Control: Turn OFF DRO eps=0.0, keep other gates
                return (ev >= 0.05) and (gap <= 0.85) and (odds <= 4.0) and (ev <= 0.25)
            else:
                raise ValueError(f"Unknown gate: {gate_name}")

        qa = False
        qb = False
        dro_ev_a = -999.0
        dro_ev_b = -999.0
        ev_a = -999.0
        ev_b = -999.0
        p_hyb_a = 0.5
        p_hyb_b = 0.5

        if not np.isnan(q_a) and oa > 1.0:
            la = logit(np.clip(p_a, 1e-6, 1.0 - 1e-6))
            lqa = logit(np.clip(q_a, 1e-6, 1.0 - 1e-6))
            gap_a = abs(la - lqa)
            p_hyb_a = expit(0.50 * la + 0.50 * lqa)
            ev_a = p_hyb_a * c_tax * oa - 1.0
            dro_ev_a = (p_hyb_a - 0.025) * c_tax * oa - 1.0
            p039_raw = p039_lookup.get(d["golgg_str"], np.nan)
            p039_a = p039_raw if a_is_pred_t1 else (1.0 - p039_raw if not np.isnan(p039_raw) else np.nan)
            veto_a = np.isnan(p039_a) or (abs(p_a - p039_a) <= 0.10)
            qa = qualifies(ev_a, dro_ev_a, gap_a, oa, veto_a)

        if not np.isnan(q_b) and ob > 1.0:
            lb = logit(np.clip(p_b, 1e-6, 1.0 - 1e-6))
            lqb = logit(np.clip(q_b, 1e-6, 1.0 - 1e-6))
            gap_b = abs(lb - lqb)
            p_hyb_b = expit(0.50 * lb + 0.50 * lqb)
            ev_b = p_hyb_b * c_tax * ob - 1.0
            dro_ev_b = (p_hyb_b - 0.025) * c_tax * ob - 1.0
            p039_raw = p039_lookup.get(d["golgg_str"], np.nan)
            p039_b = 1.0 - p039_raw if not np.isnan(p039_raw) else np.nan
            veto_b = np.isnan(p039_b) or (abs(p_b - p039_b) <= 0.10)
            qb = qualifies(ev_b, dro_ev_b, gap_b, ob, veto_b)

        if not qa and not qb:
            continue

        crit_a = ev_a if gate_name == "G9" else dro_ev_a
        crit_b = ev_b if gate_name == "G9" else dro_ev_b

        if qa and (not qb or crit_a >= crit_b):
            side = "a"
            odds = oa
            p_claim = p_hyb_a
            ev_claim = ev_a
            q_mkt = q_a
            won = (d["winner_side"] == "team_a")
            p_close = d["p_close_a"]
        else:
            side = "b"
            odds = ob
            p_claim = p_hyb_b
            ev_claim = ev_b
            q_mkt = q_b
            won = (d["winner_side"] == "team_b")
            p_close = 1.0 - d["p_close_a"]
        clv = net_clv_from_close(odds, p_close, tax_rate)
        qualified_bets[mid] = {
            "match_id": mid,
            "side": side,
            "odds": odds,
            "p_claim": p_claim,
            "ev_claim": ev_claim,
            "q_mkt": q_mkt,
            "won": won,
            "p_close": p_close,
            "clv": clv,
            "t_dec": d["t_dec"],
            "t_settle": d["t_settle"],
        }

    # 2. Build Chronological Event Queue
    events: list[dict[str, Any]] = []
    for mid, b in qualified_bets.items():
        events.append({"type": "decision", "timestamp": b["t_dec"], "match_id": mid, "bet": b})
        events.append({"type": "settle", "timestamp": b["t_settle"], "match_id": mid, "bet": b})

    # Sort events chronologically: settlements fire BEFORE decisions at exact same timestamp
    events.sort(key=lambda x: (x["timestamp"], 0 if x["type"] == "settle" else 1, x["match_id"]))

    # 3. Step Through Events with Capital Locking
    placed_bets: dict[int, str] = {}
    capital_exhausted_bets = 0

    for ev in events:
        mid = ev["match_id"]
        if ev["type"] == "decision":
            b = ev["bet"]
            b_net = b["odds"] * c_tax - 1.0
            f_star = max(0.0, (b_net * b["p_claim"] - (1.0 - b["p_claim"])) / b_net) if b_net > 0 else 0.0
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
                side=b["side"],
                odds=b["odds"],
                placed_at=ev["timestamp"],
                settle_at=b["t_settle"],
                ev_claimed=b["ev_claim"],
                p_model=b["p_claim"],
                p_market=b["q_mkt"],
                clv=b["clv"],
            )
            placed_bets[mid] = bet_id

        elif ev["type"] == "settle":
            bet_id = placed_bets.get(mid)
            if bet_id and bet_id in wallet.open_bets:
                wallet.settle(bet_id, won=ev["bet"]["won"], settled_at=ev["timestamp"], tax_rate=tax_rate)

    # 4. Summary & Econometric Performance Metrics
    settled = pd.DataFrame(wallet.settled_bets)
    if settled.empty:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "turnover_pln": 0.0,
            "profit_pln": 0.0,
            "realized_yield_pct": 0.0,
            "bankroll_roi_pct": 0.0,
            "net_clv_pct": None,
            "claimed_ev_pct": None,
            "optimizers_curse_bias_pp": None,
            "match_luck_noise_pp": None,
            "n_clv_bets": 0,
            "n_missing_close": 0,
            "net_clv_reason": "no_bets_with_independent_close",
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "capital_exhausted_bets": capital_exhausted_bets,
            "final_equity_pln": initial_capital,
        }

    n_bets = len(settled)
    n_wins = int(settled["won"].sum())
    turnover = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    yd = (profit / turnover) * 100.0 if turnover > 0 else 0.0
    roi = (profit / initial_capital) * 100.0
    wr = (n_wins / n_bets) * 100.0 if n_bets > 0 else 0.0

    clv_bets = observed_clv_rows(settled)
    n_clv_bets = len(clv_bets)
    n_missing_close = missing_close_count(settled)
    if n_clv_bets:
        clv_turnover = float(clv_bets["stake"].sum())
        clv_weights = clv_bets["stake"].to_numpy(dtype=float) / clv_turnover
        net_clv = float(np.sum(clv_weights * clv_bets["clv"].to_numpy(dtype=float))) * 100.0
        claimed_ev = float(np.sum(clv_weights * clv_bets["ev_claimed"].to_numpy(dtype=float))) * 100.0
        clv_yield = float(clv_bets["profit"].sum() / clv_turnover) * 100.0
        bias_pp = claimed_ev - net_clv
        luck_pp = clv_yield - net_clv
        net_clv_reason = None
    else:
        net_clv = claimed_ev = bias_pp = luck_pp = None
        net_clv_reason = "no_bets_with_independent_close"

    equities = np.array([initial_capital] + [r["total_equity"] for r in wallet.settled_bets])
    running_max = np.maximum.accumulate(equities)
    dd = (running_max - equities) / running_max * 100.0
    max_dd = float(np.max(dd))

    returns = settled["profit"] / settled["stake"]
    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (float(returns.mean()) / ret_std) * np.sqrt(n_bets) if ret_std > 0 else 0.0

    gross_wins = float(settled[settled["profit"] > 0]["profit"].sum())
    gross_losses = float(abs(settled[settled["profit"] < 0]["profit"].sum()))
    pf = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    return {
        "bets": n_bets,
        "wins": n_wins,
        "n_clv_bets": n_clv_bets,
        "n_missing_close": n_missing_close,
        "net_clv_reason": net_clv_reason,
        "win_rate_pct": round(wr, 2),
        "turnover_pln": round(turnover, 2),
        "profit_pln": round(profit, 2),
        "realized_yield_pct": round(yd, 2),
        "bankroll_roi_pct": round(roi, 2),
        "net_clv_pct": round(net_clv, 2) if net_clv is not None else None,
        "claimed_ev_pct": round(claimed_ev, 2) if claimed_ev is not None else None,
        "optimizers_curse_bias_pp": round(bias_pp, 2) if bias_pp is not None else None,
        "match_luck_noise_pp": round(luck_pp, 2) if luck_pp is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "sharpe_ratio": round(sharpe, 2),
        "capital_exhausted_bets": capital_exhausted_bets,
        "final_equity_pln": round(wallet.total_equity, 2),
    }


# -----------------------------------------------------------------------------
# 6. Main Orchestrator
# -----------------------------------------------------------------------------
def run_all_simulations():
    start_time = time.monotonic()
    print("=" * 80)
    print("REALISTIC MATCH-LEVEL WALK-FORWARD SIMULATION (1 DECISION PER MATCH)")
    print("=" * 80)

    data = load_and_prepare_data()
    quotes = data["quotes"]
    cohort_a = data["cohort_a"]
    cohort_b = data["cohort_b"]
    closing_dict = data["closing_dict"]
    p039_lookup = data["p039_lookup"]

    models = [
        ("C0-D-Core", "p_c0_d_year_core__full"),
        ("C0-Slim", "p_c0_slim_year__full"),
        ("C0", "p_c0__full"),
    ]

    cohorts = [
        ("cohort_a_645_exact", "Cohort A: 645-Match Production DB (Exact Frozen Distilled Matches)", cohort_a),
        ("cohort_b_later_disjoint", "Cohort B: Strictly Later, Disjoint Production DB Matches", cohort_b),
    ]

    results_payload: dict[str, Any] = {
        "title": "Realistic Match-Level Chronological Walk-Forward Simulation (1 Decision Per Match)",
        "selection": {
            "cohort_b_strictly_later_disjoint": True,
            "in_sample_selection": data["in_sample_selection"],
            "cohort_a_end_utc": cohort_a["start_dt"].max().isoformat() if not cohort_a.empty else None,
            "verification_status": "NOT_EVALUATED" if data["in_sample_selection"] else "EVALUATED",
            "verification_reason": "no_later_disjoint_cohort_b" if data["in_sample_selection"] else None,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            "initial_capital_pln": 10000.0,
            "polish_tax_rate": 0.12,
            "staking_strategy": "1/4 Kelly capped [20.00, 150.00] PLN from OperationalWallet free capital",
            "capital_locking": "Reserved at t_dec, settled at t_start + 3 hours",
            "market_consensus": "Mean Shin devigged probability across active books (require >= 2 books)",
            "playable_odds": "Best available bookmaker odds at t_dec",
            "quote_cutoff": "scraped_at <= decision_time",
            "market_prefilter_applied": False,
            "clv_missing_close_policy": "exclude missing-close bets from CLV-derived metrics",
        },
        "cohorts": {},
    }

    for c_key, c_label, c_df in cohorts:
        print(f"\n=======================================================")
        print(f"RUNNING COHORT: {c_label} (N={len(c_df)})")
        print(f"=======================================================")
        c_dict: dict[str, Any] = {
            "label": c_label,
            "evaluation_status": "EVALUATED" if not c_df.empty else "NOT_EVALUATED",
            "evaluation_reason": None if not c_df.empty else "cohort_has_no_matches",
            "n_matches": len(c_df),
            "schedules": {},
        }

        for s_key, s_label in SCHEDULE_SPECS:
            print(f"\n--- Schedule: {s_key} ({s_label}) ---")
            decisions = extract_decisions_for_schedule(c_df, quotes, s_key, closing_dict)
            print(f"  Extracted {len(decisions)} eligible match decision boards (>= 2 books).")

            s_dict: dict[str, Any] = {
                "label": s_label,
                "n_eligible_matches": len(decisions),
                "models": {},
            }

            for m_name, m_col in models:
                m_dict: dict[str, Any] = {}
                for g_key, g_label in GATE_SPECS:
                    res = simulate_walk_forward(
                        decisions=decisions,
                        model_col=m_col,
                        gate_name=g_key,
                        p039_lookup=p039_lookup,
                        initial_capital=10000.0,
                        tax_rate=0.12,
                    )
                    res["gate_label"] = g_label
                    m_dict[g_key] = res

                s_dict["models"][m_name] = m_dict

                # Print summary for C0-D-Core
                if m_name == "C0-D-Core":
                    g5_res = m_dict["G5"]
                    g1_res = m_dict["G1"]
                    g7b_res = m_dict["G7b"]
                    g1_clv = f"{g1_res['net_clv_pct']:+5.2f}%" if g1_res["net_clv_pct"] is not None else "N/A"
                    g5_clv = f"{g5_res['net_clv_pct']:+5.2f}%" if g5_res["net_clv_pct"] is not None else "N/A"
                    g7b_clv = f"{g7b_res['net_clv_pct']:+5.2f}%" if g7b_res["net_clv_pct"] is not None else "N/A"
                    print(f"    [{m_name}] G1: {g1_res['bets']:2d} bets | {g1_res['profit_pln']:+7.2f} PLN ({g1_res['realized_yield_pct']:+5.2f}%) | CLV: {g1_clv}")
                    print(f"    [{m_name}] G5: {g5_res['bets']:2d} bets | {g5_res['profit_pln']:+7.2f} PLN ({g5_res['realized_yield_pct']:+5.2f}%) | CLV: {g5_clv}")
                    print(f"    [{m_name}] G7b: {g7b_res['bets']:2d} bets | {g7b_res['profit_pln']:+7.2f} PLN ({g7b_res['realized_yield_pct']:+5.2f}%) | CLV: {g7b_clv}")

            c_dict["schedules"][s_key] = s_dict

        results_payload["cohorts"][c_key] = c_dict

    # Save to disk
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results_payload, f, indent=2)

    elapsed = time.monotonic() - start_time
    print(f"\nSaved structured simulation results to: {OUTPUT_JSON_PATH}")
    print(f"Total elapsed time: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    run_all_simulations()
