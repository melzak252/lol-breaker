#!/usr/bin/env python3
"""CLI utility for rapid pre-match predictions using C0 Model Family (C0, C0-D, C0-Slim, C0-Mini) and Hybrid-50.

Usage:
    .venv/bin/python scripts/predict_matchup.py C9 KC
    .venv/bin/python scripts/predict_matchup.py "Cloud9" "Karmine Corp" --bo 5
    .venv/bin/python scripts/predict_matchup.py T1 GenG --bo 3 --odds1 2.40 --odds2 1.55
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure ensemble research root points to consolidated benchmark if not set
if "ENSEMBLE_RESEARCH_ROOT" not in os.environ:
    os.environ["ENSEMBLE_RESEARCH_ROOT"] = str(PROJECT_ROOT / "data/05_model_input/canonical_benchmark")

from scipy.special import expit, logit

from betting_app.services.c0_inference import close_c0_inference_worker, predict_c0
from betting_app.core.models.registry import resolve_model_spec


def apply_model_calibration(p_a: float, spec: Any, best_of: int = 3) -> tuple[float, float]:
    """Apply model-specific calibration / distillation adjustment to base probability."""
    slopes = spec.metadata.get("calibration_slopes", {}) if hasattr(spec, "metadata") else {}
    slope = slopes.get(best_of, slopes.get(3, 1.0))
    p_clamped = max(1e-6, min(1.0 - 1e-6, p_a))
    if slope == 1.0 or abs(slope - 1.0) < 1e-9:
        return p_clamped, 1.0 - p_clamped
    l = logit(p_clamped)
    p_adj = float(expit(slope * l))
    p_adj = max(1e-6, min(1.0 - 1e-6, p_adj))
    return p_adj, 1.0 - p_adj

COMMON_ALIASES: dict[str, str] = {
    "c9": "cloud9",
    "kc": "karmine corp",
    "t1": "t1",
    "g2": "g2 esports",
    "fnc": "fnatic",
    "gen": "gen.g",
    "geng": "gen.g",
    "hle": "hanwha life esports",
    "dk": "dplus kia",
    "wbg": "weibo gaming",
    "blg": "bilibili gaming",
    "tes": "top esports",
    "fly": "flyquest",
    "tl": "team liquid",
    "mroi": "movistar koi",
    "koi": "movistar koi",
    "mkoi": "movistar koi",
    "al": "anyone s legend",
    "anyone's legend": "anyone s legend",
    "anyones legend": "anyone s legend",
    "ig": "invictus gaming",
    "cfo": "ctbc flying oyster",
    "tsw": "team secret whales",
    "los": "los grandes",
    "l0s": "los grandes",
    "lyon": "lyon",
    "team liquid alienware": "team liquid",
    "gx": "giantx",
    "bds": "team bds",
    "mad": "mad lions",
    "sk": "sk gaming",
    "vitality": "team vitality",
    "vit": "team vitality",
    "100": "100 thieves",
    "100t": "100 thieves",
    "dignitas": "dignitas",
    "dig": "dignitas",
    "nrg": "nrg",
    "sr": "shopify rebellion",
    "lyon": "lyon",
    "lyon gaming": "lyon",
}


def load_raw_matches(history_path: Path | str | None = None) -> list[dict]:
    if history_path is None:
        env_path = os.environ.get("C0_RAW_HISTORY_PATH")
        ext_default = PROJECT_ROOT / "data/04_feature/live_c0_cache/extended_matches.json"
        if env_path:
            history_path = Path(env_path)
        elif ext_default.is_file():
            history_path = ext_default
            os.environ["C0_RAW_HISTORY_PATH"] = str(ext_default.resolve())
        else:
            history_path = PROJECT_ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json"
    else:
        history_path = Path(history_path)
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing raw history at {history_path}")
    with history_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def resolve_team(query: str, raw_matches: list[dict]) -> dict:
    q = query.lower().strip()
    target = COMMON_ALIASES.get(q, q)

    # 1. Exact match pass
    exact_matches = [
        m for m in raw_matches
        if (m.get("t1_name") or "").lower().strip() == target
        or (m.get("t2_name") or "").lower().strip() == target
    ]

    matches = exact_matches
    if not matches:
        # 2. Substring match, filtering academy/challengers unless explicitly queried
        sub_matches = []
        for m in raw_matches:
            t1 = (m.get("t1_name") or "").lower()
            t2 = (m.get("t2_name") or "").lower()
            if target in t1 or target in t2:
                is_sub = any(k in t1 or k in t2 for k in ("academy", "challengers", "blue", "amateur"))
                if not is_sub or any(k in target for k in ("academy", "challengers", "blue", "amateur")):
                    sub_matches.append(m)
        matches = sub_matches if sub_matches else [
            m for m in raw_matches
            if target in (m.get("t1_name") or "").lower()
            or target in (m.get("t2_name") or "").lower()
        ]

    if not matches:
        raise ValueError(f"Could not resolve any match history for team query '{query}'")

    matches.sort(key=lambda x: x.get("date", ""))
    latest = matches[-1]

    t1_low = (latest.get("t1_name") or "").lower()
    t2_low = (latest.get("t2_name") or "").lower()
    if t1_low == target:
        side = "t1"
    elif t2_low == target:
        side = "t2"
    elif target in t1_low:
        side = "t1"
    else:
        side = "t2"

    team_name = latest.get(f"{side}_name")
    team_id = str(latest.get(f"{side}_id"))
    last_game = latest["games"][-1]
    if str(last_game.get("t1_id")) == team_id or (last_game.get("t1_name") or "").lower().strip() == (team_name or "").lower().strip():
        game_side = "t1"
    elif str(last_game.get("t2_id")) == team_id or (last_game.get("t2_name") or "").lower().strip() == (team_name or "").lower().strip():
        game_side = "t2"
    else:
        game_side = side
    players_dict = last_game.get(f"{game_side}_players", {})
    role_order = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
    ordered_roles = [r for r in role_order if r in players_dict] + [r for r in players_dict if r not in role_order]
    player_names = {role: players_dict[role].get("player_name") for role in ordered_roles}
    player_ids = [str(players_dict[role].get("player_id")) for role in ordered_roles]
    return {
        "name": team_name,
        "id": team_id,
        "last_played_date": latest.get("date"),
        "tournament": latest.get("tournament_name"),
        "players": player_names,
        "player_ids": player_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast pre-match prediction for any League of Legends matchup using C0 Model Family & Hybrid-50.",
    )
    parser.add_argument("team1", nargs="?", default=None, help="Team 1 name or alias (e.g. 'C9', 'Cloud9', 'T1')")
    parser.add_argument("team2", nargs="?", default=None, help="Team 2 name or alias (e.g. 'KC', 'Karmine Corp', 'GenG')")
    parser.add_argument("--bo", type=int, choices=[1, 3, 5], default=3, help="Best-of series format (default: 3)")
    parser.add_argument(
        "--model",
        type=str,
        default="C0",
        help="Model specification to evaluate (e.g. 'C0', 'C0-D', 'C0-Slim', 'C0-D1', 'C0-Mini', 'C0-Small'; default: 'C0')",
    )
    parser.add_argument("--odds1", type=float, default=None, help="Decimal odds for Team 1 (e.g. 2.10)")
    parser.add_argument("--odds2", type=float, default=None, help="Decimal odds for Team 2 (e.g. 1.70)")
    parser.add_argument("--all-bo", action="store_true", help="Print win probabilities for BO1, BO3, and BO5")
    parser.add_argument("--history-path", type=Path, default=None, help="Explicit path to GOL.GG history JSON")
    parser.add_argument("--live-db", action="store_true", help="Assemble extended history dynamically from PostgreSQL GOL.GG matches")
    parser.add_argument("--no-cross-region-norm", action="store_true", help="Disable automatic cross-regional input tensor normalization")
    parser.add_argument("--show-region-ranks", action="store_true", help="Display live automatic regional offsets and pyramid drift table")
    parser.add_argument("--simulate-worlds", action="store_true", help="Run full Worlds 2026 Swiss + Knockout simulation using C0-D (Norm)")
    parser.add_argument("--n-sims", type=int, default=25000, help="Number of Monte Carlo Worlds simulations (default: 25000)")
    args = parser.parse_args()

    try:
        model_spec = resolve_model_spec(args.model)
    except KeyError as exc:
        parser.error(str(exc))

    if args.show_region_ranks:
        from src.models.cross_region_input_normalizer import (
            DEFAULT_REGIONAL_STATE_PATH,
            load_dynamic_region_offsets,
        )
        offsets = load_dynamic_region_offsets()
        print("=" * 78)
        print(f"Live Automatic Regional Offsets ({DEFAULT_REGIONAL_STATE_PATH.name})")
        print("=" * 78)
        for fam in ["LCK", "LPL", "LCS", "LEC", "LCP", "PCS", "VCS", "CBLOL", "LJL", "LLA"]:
            if fam in offsets:
                print(f"  {fam:<10s}: {offsets[fam]:+7.2f} Glicko eq.")
        if not args.team1 or not args.team2:
            return 0
    if args.simulate_worlds:
        from src.models.advanced_tournament_simulator import (
            AdvancedTournamentSimulator,
            DEFAULT_WORLDS_REGIONS,
            WORLDS_2026_SWISS_R1_PAIRINGS,
            build_or_load_worlds_2026_pair_probabilities,
        )
        print("=" * 78)
        print(f"LoL-Breaker Worlds 2026 Full Tournament Simulator ({model_spec.effective_short_code} Norm | N={args.n_sims:,})")
        print("=" * 78)
        prob_table = build_or_load_worlds_2026_pair_probabilities(model_code=args.model)
        sim = AdvancedTournamentSimulator(
            use_fearless=True,
            use_bayesian_momentum=True,
            use_hybrid_knockout=True,
            use_riot_lottery=True,
            shrinkage_lambda=0.0,
            temperature=1.0,
            seed=20261021,
        )
        res = sim.simulate_worlds(
            base_probabilities=prob_table,
            initial_pairings=WORLDS_2026_SWISS_R1_PAIRINGS,
            n_simulations=args.n_sims,
            seed=20261021,
        )
        print("\n--- 1. Worlds 2026 Swiss Qualification & Championship Probabilities ---")
        print(f"{'Rank':<4} {'Team':<22} {'Reg':<5} {'Leave Swiss':>11} {'(3-0 / 3-1 / 3-2)':>20} {'Top 4 (SF)':>10} {'Top 2 (Fin)':>11} {'Win Worlds':>11}")
        print("-" * 100)
        standings_by_team = {r["team"]: r for r in res.standings}
        ranked_teams = sorted(
            res.standings,
            key=lambda r: (-r["champion_prob"], -res.swiss_qualifiers_prob.get(r["team"], 0.0)),
        )
        for idx, row in enumerate(ranked_teams, 1):
            t = row["team"]
            reg = DEFAULT_WORLDS_REGIONS.get(t, "UNK")
            q_p = res.swiss_qualifiers_prob[t] * 100.0
            rec = res.swiss_record_probs[t]
            r_str = f"{rec['3-0']*100:4.1f}/{rec['3-1']*100:4.1f}/{rec['3-2']*100:4.1f}%"
            t4 = row["top4_prob"] * 100.0
            t2 = row["top2_prob"] * 100.0
            ch = row["champion_prob"] * 100.0
            print(f"{idx:<4d} {t:<22} {reg:<5} {q_p:10.2f}% {r_str:>20} {t4:9.2f}% {t2:10.2f}% {ch:10.2f}%")

        top8_marginal = sorted(res.swiss_qualifiers_prob.items(), key=lambda x: -x[1])[:8]
        print("\n--- 2. Most Probable Top 8 Leaving Swiss (Marginal Ranking) ---")
        print("  " + ", ".join(f"{t} ({p*100:.1f}%)" for t, p in top8_marginal))
        print("\n--- 3. Top 5 Most Probable Exact 8-Team Quarterfinal Lineups ---")
        for i, item in enumerate(res.top_qualifier_sets[:5], 1):
            print(f"  #{i} ({item['prob']*100:.2f}% | {item['count']} sims): {', '.join(item['teams'])}")
        print("\n--- 4. Top 8 Most Probable Grand Finals Pairings ---")
        for i, item in enumerate(res.joint_finalists[:8], 1):
            print(f"  #{i} {item['pair']:<38}: {item['prob']*100:5.2f}%")
        return 0
    if not args.team1 or not args.team2:
        parser.error("team1 and team2 are required unless --show-region-ranks is specified.")
    print("=" * 78)
    print(f"LoL-Breaker Matchup Predictor ({model_spec.effective_short_code} | {model_spec.effective_display_name})")
    print(f"  Version:      {model_spec.version}")
    print(f"  Architecture: {model_spec.description}")
    print("=" * 78)

    history_file: Path | None = args.history_path
    if args.live_db:
        print("Assembling dynamic extended history from PostgreSQL GOL.GG tables...")
        from betting_app.core.db import get_session
        from betting_app.services.live_native_feature_builder import LiveNativeFeatureBuilder
        try:
            with get_session() as session:
                builder = LiveNativeFeatureBuilder(session=session)
                ext_path, ext_digest, ext_audit = builder.prepare_extended_history()
                history_file = ext_path
                os.environ["C0_RAW_HISTORY_PATH"] = str(ext_path)
                print(f"  Live extended history active: {ext_path} (SHA-256: {ext_digest[:16]}...)")
                print(f"  Total series: {ext_audit.get('total_series_count')}, New from DB: {ext_audit.get('db_new_count')}")
        except Exception as exc:
            print(f"  Warning: Live DB extraction failed ({exc}); falling back to canonical baseline history.")
    elif history_file is not None:
        os.environ["C0_RAW_HISTORY_PATH"] = str(history_file.resolve())

    raw_matches = load_raw_matches(history_file)
    from betting_app.services.c0_inference import _default_client
    from src.models.cross_region_input_normalizer import (
        build_prior_domestic_family_lookup,
        get_region_offset,
        normalize_encoded_snapshot,
        resolve_side_domestic_family,
    )

    team_fam_map, player_fam_map = build_prior_domestic_family_lookup(raw_matches)
    t1_info = resolve_team(args.team1, raw_matches)
    t2_info = resolve_team(args.team2, raw_matches)
    fam_a = resolve_side_domestic_family(t1_info["id"], t1_info["player_ids"], team_fam_map, player_fam_map)
    fam_b = resolve_side_domestic_family(t2_info["id"], t2_info["player_ids"], team_fam_map, player_fam_map)
    off_a = get_region_offset(fam_a)
    off_b = get_region_offset(fam_b)

    print(f"\nResolved Teams:")
    print(f"  Team A: {t1_info['name']} [{fam_a}] (ID: {t1_info['id']}, Last Match: {t1_info['last_played_date']})")
    print(f"          Roster: {', '.join(f'{r}: {n}' for r, n in t1_info['players'].items())}")
    print(f"  Team B: {t2_info['name']} [{fam_b}] (ID: {t2_info['id']}, Last Match: {t2_info['last_played_date']})")
    print(f"          Roster: {', '.join(f'{r}: {n}' for r, n in t2_info['players'].items())}")
    if not args.no_cross_region_norm and abs(off_a - off_b) > 1.0:
        print(f"  Cross-Region Input Normalization: ACTIVE ({fam_a} {off_a:+.0f} vs {fam_b} {off_b:+.0f} Glicko eq.)")

    from datetime import timedelta
    now = datetime.now(timezone.utc)
    decision_iso = now.isoformat()
    start_iso = (now + timedelta(hours=2)).isoformat()
    formats_to_evaluate = [1, 3, 5] if args.all_bo else [args.bo]

    predictions: dict[int, Any] = {}
    client = _default_client()
    for bo in formats_to_evaluate:
        c0_req = {
            "team1_id": t1_info["id"],
            "team2_id": t2_info["id"],
            "roster_a": t1_info["player_ids"],
            "roster_b": t2_info["player_ids"],
            "best_of": bo,
            "decision_at": decision_iso,
            "start_at": start_iso,
            "competition_context": "Worlds",
            "mode": "full",
        }
        if not args.no_cross_region_norm and abs(off_a - off_b) > 1.0:
            ev_res = client.predict_evidence(c0_req)
            norm_snap = normalize_encoded_snapshot(ev_res.encoded_snapshot, off_a, off_b)
            raw_pred = predict_c0({"c0_request": c0_req, "c0_snapshot": norm_snap, "canonical": {"id": None}})
            raw_unnorm_pa, _ = apply_model_calibration(ev_res.prediction.prob_a, model_spec, best_of=bo)
        else:
            raw_pred = predict_c0({"c0_request": c0_req, "canonical": {"id": None}})
            raw_unnorm_pa = None
        pa, pb = apply_model_calibration(raw_pred.prob_a, model_spec, best_of=bo)
        predictions[bo] = {"prob_a": pa, "prob_b": pb, "raw": raw_pred, "unnorm_prob_a": raw_unnorm_pa}

    primary_bo = args.bo
    primary_pred = predictions[primary_bo]
    p_model_a = primary_pred["prob_a"]
    p_model_b = primary_pred["prob_b"]

    print(f"\n--- Model Forecast ({model_spec.effective_short_code} - BO{primary_bo} Series) ---")
    print(f"  {t1_info['name']:<20} Win Prob: {p_model_a*100:6.2f}%  (Fair Decimal Odds: {1.0/p_model_a:5.2f})")
    print(f"  {t2_info['name']:<20} Win Prob: {p_model_b*100:6.2f}%  (Fair Decimal Odds: {1.0/p_model_b:5.2f})")
    if primary_pred.get("unnorm_prob_a") is not None:
        u_a = float(primary_pred["unnorm_prob_a"])
        u_b = 1.0 - u_a
        print(f"  (Unnormalized Domestic-Only Baseline: {t1_info['name']} {u_a*100:.2f}% vs {t2_info['name']} {u_b*100:.2f}%)")

    favorite = t1_info['name'] if p_model_a >= p_model_b else t2_info['name']
    fav_prob = max(p_model_a, p_model_b)
    print(f"\nModel Favorite: {favorite} ({fav_prob*100:.2f}%) | Predicted Winner: {favorite}")

    if args.all_bo:
        print("\n--- Cross-Format Projection Comparison ---")
        print("| Format | Team A Win Prob | Team B Win Prob | Margin (Edge) |")
        print("|---|---:|---:|---:|")
        for bo in [1, 3, 5]:
            pa = predictions[bo]["prob_a"]
            pb = predictions[bo]["prob_b"]
            edge_str = f"{favorite} +{abs(pa - pb)*100:.2f} pp"
            print(f"| Best-of-{bo} | {t1_info['name']}: {pa*100:.2f}% | {t2_info['name']}: {pb*100:.2f}% | {edge_str} |")

    # If bookmaker odds are provided, compute Shin de-vig, Hybrid-50, and 12% Polish tax EV
    if args.odds1 is not None and args.odds2 is not None:
        o1, o2 = args.odds1, args.odds2
        inv1, inv2 = 1.0 / o1, 1.0 / o2
        overround = (inv1 + inv2) - 1.0
        q1 = max(1e-6, min(1.0 - 1e-6, inv1 - overround / 2.0))
        q2 = 1.0 - q1

        # Hybrid-50 logit shrinkage with selected model
        l_model = logit(max(1e-6, min(1.0 - 1e-6, p_model_a)))
        l_mkt = logit(q1)
        p_hybrid_a = float(expit(0.50 * l_model + 0.50 * l_mkt))
        p_hybrid_b = 1.0 - p_hybrid_a

        # Polish tax multiplier c = 0.88
        tax_mult = 0.88
        ev_model_a = (p_model_a * tax_mult * o1) - 1.0
        ev_model_b = (p_model_b * tax_mult * o2) - 1.0
        ev_hyb_a = (p_hybrid_a * tax_mult * o1) - 1.0
        ev_hyb_b = (p_hybrid_b * tax_mult * o2) - 1.0

        print(f"\n--- Market & Betting Evaluation ({model_spec.effective_short_code} + 12% Polish Turnover Tax) ---")
        print(f"Offered Odds:     {t1_info['name']} @ {o1:.2f}  |  {t2_info['name']} @ {o2:.2f} (Margin: {overround*100:.2f}%)")
        print(f"De-vigged Market: {t1_info['name']} {q1*100:5.2f}%  |  {t2_info['name']} {q2*100:5.2f}%")
        print(f"Hybrid-50 Shrink: {t1_info['name']} {p_hybrid_a*100:5.2f}%  |  {t2_info['name']} {p_hybrid_b*100:5.2f}%")
        print(f"Pure {model_spec.effective_short_code} EV (Net): {t1_info['name']} {ev_model_a*100:+5.2f}%  |  {t2_info['name']} {ev_model_b*100:+5.2f}%")
        print(f"Hybrid-50 EV:     {t1_info['name']} {ev_hyb_a*100:+5.2f}%  |  {t2_info['name']} {ev_hyb_b*100:+5.2f}%")

        # Operational decision gates
        divergence = abs(l_model - l_mkt)
        div_gate_ok = divergence <= 0.85
        best_ev = max(ev_hyb_a, ev_hyb_b)
        best_side = t1_info['name'] if ev_hyb_a >= ev_hyb_b else t2_info['name']
        best_odds = o1 if ev_hyb_a >= ev_hyb_b else o2
        best_p_hyb = p_hybrid_a if ev_hyb_a >= ev_hyb_b else p_hybrid_b

        # Wasserstein Robust Gate (eps = 0.025)
        dro_ev = ((best_p_hyb - 0.025) * tax_mult * best_odds) - 1.0

        print(f"\nSafety Gates Check:")
        print(f"  Market-Divergence-Gate (|logit diff| <= 0.85): {divergence:.4f} -> {'PASS' if div_gate_ok else 'VETO'}")
        print(f"  Wasserstein-Robust-Gate (eps = 0.025):        {dro_ev*100:+.2f}% -> {'PASS' if dro_ev > 0 else 'VETO'}")

        if div_gate_ok and dro_ev > 0 and 0.05 <= best_ev <= 0.25 and best_odds <= 4.0:
            print(f"\n>>> ADOPTED BET RECOMMENDATION: {best_side} @ {best_odds:.2f} (Net EV: {best_ev*100:+.2f}%, 1/4 Kelly, Max 150 PLN) <<<")
        else:
            print(f"\n>>> NO BET: Opportunity does not satisfy strict production risk gates. <<<")

    print("\n" + "=" * 78)
    close_c0_inference_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
