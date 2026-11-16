#!/usr/bin/env python3
"""CLI utility for native C0 predictions and explicitly opted-in legacy research surrogates.

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
if __name__ == "__main__" and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.special import expit, logit

from betting_app.services.c0_inference import close_c0_inference_worker, predict_c0
from betting_app.core.models.registry import (
    C0_NATIVE, apply_model_calibration as _apply_model_calibration, resolve_model_spec,
)


POLISH_TURNOVER_TAX = 0.12  # fraction of stake, AGENTS.md convention
CANONICAL_HISTORY_RELATIVE = "data/artifacts/golgg-database-recovery-20260909/matches.json"


def apply_model_calibration(
    p_a: float, spec: Any, best_of: int = 3, *, allow_surrogate: bool = False
) -> tuple[float, float]:
    """Delegate registry calibration; research-only slopes require explicit opt-in."""
    return _apply_model_calibration(p_a, spec, best_of, allow_surrogate=allow_surrogate)

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
    "lyon gaming": "lyon",
}


def load_raw_matches(history_path: Path | str | None = None) -> list[dict]:
    if history_path is None:
        env_path = os.environ.get("C0_RAW_HISTORY_PATH")
        if env_path:
            history_path = Path(env_path)
        else:
            history_path = PROJECT_ROOT / CANONICAL_HISTORY_RELATIVE
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


def describe_served_model(spec: Any) -> list[str]:
    """Header lines stating what artifact actually produces the probability for ``spec``."""
    metadata = getattr(spec, "metadata", {}) or {}
    served_as = metadata.get("served_as")
    if served_as is None:
        return [
            f"Spec:         {spec.name} {spec.version}",
            f"Architecture: {spec.description}",
        ]
    return [
        f"Requested spec: {spec.name} {spec.version} (research label; operational={metadata.get('operational')})",
        f"Served as:      {served_as}",
        "Note: the probability below is the legacy 17-token C0 with a BO logit-slope surrogate, "
        "NOT an 18/20-token model.",
        f"Reason:         {metadata.get('unavailable_reason', 'n/a')}",
    ]


def _prepare_live_history(cutoff: str | None) -> Path:
    """Refresh and return live extended history for the exclusive cutoff day."""
    from betting_app.core.db import get_session
    from betting_app.services.live_native_feature_builder import LiveNativeFeatureBuilder

    with get_session() as session:
        builder = LiveNativeFeatureBuilder(session=session)
        ext_path, ext_digest, ext_audit = builder.prepare_extended_history(cutoff_date=cutoff)
    os.environ["C0_RAW_HISTORY_PATH"] = str(ext_path)
    print(f"  Live extended history active: {ext_path} (SHA-256: {ext_digest[:16]}...)")
    print("  (DB-backed history refreshed from PostgreSQL for this invocation.)")
    print(f"  Cutoff (exclusive): {ext_audit.get('cutoff_enforced')}, total series: {ext_audit.get('total_series_count')}, new from DB: {ext_audit.get('db_new_count')}")
    return ext_path


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
        help="Native C0 by default. Research C0-family labels require --allow-research-surrogate; use c0_distilled_replay for genuine saved 18/20-token inference.",
    )
    parser.add_argument("--allow-research-surrogate", action="store_true", help="Explicitly allow legacy C0-family BO-slope surrogates; these are not genuine distilled-model inference")
    parser.add_argument("--odds1", type=float, default=None, help="Decimal odds for Team 1 (e.g. 2.10)")
    parser.add_argument("--odds2", type=float, default=None, help="Decimal odds for Team 2 (e.g. 1.70)")
    parser.add_argument("--all-bo", action="store_true", help="Print win probabilities for BO1, BO3, and BO5")
    parser.add_argument("--history-path", type=Path, default=None, help="Explicit path to GOL.GG history JSON")
    parser.add_argument("--live-db", action="store_true", help="Assemble extended history from PostgreSQL GOL.GG matches released strictly before --cutoff; refresh DB input on every invocation")
    parser.add_argument("--cutoff", type=str, default=None, help="With --live-db: EXCLUSIVE prediction day YYYY-MM-DD; only series with effective release day strictly before it are used (default: today, UTC)")
    parser.add_argument("--allow-baseline-fallback", action="store_true", help="With --live-db: if the live history cannot be built, run on the frozen canonical baseline instead of exiting with an error")
    parser.add_argument("--research-cross-region-norm", action="store_true", help="Opt into unverified cached regional input normalization for research; off by default and not causal confirmation")
    parser.add_argument("--show-region-ranks", action="store_true", help="Display live automatic regional offsets and pyramid drift table")
    parser.add_argument("--simulate-worlds", action="store_true", help="Run the legacy research Worlds scenario; not certified tournament probabilities")
    parser.add_argument("--n-sims", type=int, default=25000, help="Number of Monte Carlo Worlds simulations (default: 25000)")
    args = parser.parse_args()

    try:
        model_spec = resolve_model_spec(args.model)
    except KeyError as exc:
        parser.error(str(exc))
    if model_spec is not C0_NATIVE and not (
        args.allow_research_surrogate
        and model_spec.metadata.get("operational") is False
        and model_spec.metadata.get("calibration_slopes_source")
    ):
        parser.error(
            "This named-matchup path serves native C0 only. C0-family research slopes "
            "require --allow-research-surrogate; genuine saved C0-D inference is "
            "available through python -m src.models.c0_distilled_replay."
        )

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
    print(f"LoL-Breaker Matchup Predictor (requested model code: {args.model})")
    for line in describe_served_model(model_spec):
        print(f"  {line}")
    print("=" * 78)

    history_file: Path | None = args.history_path
    if args.live_db:
        print("Assembling dynamic extended history from PostgreSQL GOL.GG tables...")
        try:
            history_file = _prepare_live_history(args.cutoff)
        except Exception as exc:
            if not args.allow_baseline_fallback:
                print(
                    f"ERROR: live history build failed ({exc}). Not falling back to the canonical "
                    "baseline; pass --allow-baseline-fallback to run on frozen history explicitly.",
                    file=sys.stderr,
                )
                return 2
            history_file = PROJECT_ROOT / CANONICAL_HISTORY_RELATIVE
            os.environ.pop("C0_RAW_HISTORY_PATH", None)
            print(f"  WARNING: live history build failed ({exc}); running on the FROZEN canonical baseline "
                  f"({history_file}) because --allow-baseline-fallback was given. Results are NOT live.")
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
    off_a = get_region_offset(fam_a) if args.research_cross_region_norm else 0.0
    off_b = get_region_offset(fam_b) if args.research_cross_region_norm else 0.0

    print(f"\nResolved Teams:")
    print(f"  Team A: {t1_info['name']} [{fam_a}] (ID: {t1_info['id']}, Last Match: {t1_info['last_played_date']})")
    print(f"          Roster: {', '.join(f'{r}: {n}' for r, n in t1_info['players'].items())}")
    print(f"  Team B: {t2_info['name']} [{fam_b}] (ID: {t2_info['id']}, Last Match: {t2_info['last_played_date']})")
    print(f"          Roster: {', '.join(f'{r}: {n}' for r, n in t2_info['players'].items())}")
    if args.research_cross_region_norm and abs(off_a - off_b) > 1.0:
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
        if args.research_cross_region_norm and abs(off_a - off_b) > 1.0:
            ev_res = client.predict_evidence(c0_req)
            norm_snap = normalize_encoded_snapshot(ev_res.encoded_snapshot, off_a, off_b)
            raw_pred = predict_c0({"c0_request": c0_req, "c0_snapshot": norm_snap, "canonical": {"id": None}})
            raw_unnorm_pa, _ = apply_model_calibration(
                ev_res.prediction.prob_a, model_spec, best_of=bo,
                allow_surrogate=args.allow_research_surrogate,
            )
        else:
            raw_pred = predict_c0({"c0_request": c0_req, "canonical": {"id": None}})
            raw_unnorm_pa = None
        pa, pb = apply_model_calibration(
            raw_pred.prob_a, model_spec, best_of=bo,
            allow_surrogate=args.allow_research_surrogate,
        )
        predictions[bo] = {"prob_a": pa, "prob_b": pb, "raw": raw_pred, "unnorm_prob_a": raw_unnorm_pa}

    primary_bo = args.bo
    primary_pred = predictions[primary_bo]
    p_model_a = primary_pred["prob_a"]
    p_model_b = primary_pred["prob_b"]

    served_name = getattr(primary_pred["raw"], "model_name", None)
    served_version = getattr(primary_pred["raw"], "model_version", None)
    served_feature = getattr(primary_pred["raw"], "feature_version", None)
    print(f"\n--- Model Forecast (requested {args.model} - BO{primary_bo} Series) ---")
    if model_spec.metadata.get("operational") is False:
        print("  DISCLOSURE: research-only BO-slope surrogate applied to legacy 17-token C0; this is not distilled-model inference.")
    print(f"  Probability produced by artifact: {served_name} {served_version} (feature_version={served_feature})")
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

    # Informational EV arithmetic only. This is NOT the application betting policy: it applies
    # none of the app's gates (thesis, BO1, longshot, freshness, >=2-bookmaker consensus, ...).
    if args.odds1 is not None and args.odds2 is not None:
        o1, o2 = args.odds1, args.odds2
        inv1, inv2 = 1.0 / o1, 1.0 / o2
        overround = (inv1 + inv2) - 1.0
        # Multiplicative de-vig, the method used by betting_app/services/prematch_quote_service.py.
        q1 = max(1e-6, min(1.0 - 1e-6, inv1 / (inv1 + inv2)))
        q2 = 1.0 - q1

        # Hybrid-50 logit shrinkage with selected model
        l_model = logit(max(1e-6, min(1.0 - 1e-6, p_model_a)))
        l_mkt = logit(q1)
        p_hybrid_a = float(expit(0.50 * l_model + 0.50 * l_mkt))
        p_hybrid_b = 1.0 - p_hybrid_a

        tax_mult = 1.0 - POLISH_TURNOVER_TAX
        ev_model_a = (p_model_a * tax_mult * o1) - 1.0
        ev_model_b = (p_model_b * tax_mult * o2) - 1.0
        ev_hyb_a = (p_hybrid_a * tax_mult * o1) - 1.0
        ev_hyb_b = (p_hybrid_b * tax_mult * o2) - 1.0

        print(f"\n--- EV ESTIMATE (informational, NOT the application betting policy; {POLISH_TURNOVER_TAX*100:.0f}% Polish turnover tax) ---")
        print(f"Offered Odds:     {t1_info['name']} @ {o1:.2f}  |  {t2_info['name']} @ {o2:.2f} (Margin: {overround*100:.2f}%)")
        print(f"De-vigged Market (multiplicative, single quote pair): {t1_info['name']} {q1*100:5.2f}%  |  {t2_info['name']} {q2*100:5.2f}%")
        print(f"Hybrid-50 Shrink: {t1_info['name']} {p_hybrid_a*100:5.2f}%  |  {t2_info['name']} {p_hybrid_b*100:5.2f}%")
        print(f"Pure model EV (net of tax): {t1_info['name']} {ev_model_a*100:+5.2f}%  |  {t2_info['name']} {ev_model_b*100:+5.2f}%")
        print(f"Hybrid-50 EV (net of tax):  {t1_info['name']} {ev_hyb_a*100:+5.2f}%  |  {t2_info['name']} {ev_hyb_b*100:+5.2f}%")

    print("\n" + "=" * 78)
    close_c0_inference_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
