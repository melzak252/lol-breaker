"""Automated tests for 2026 Roster Resolution Integrity and Cross-Regional Input Normalization."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest

from betting_app.core.models.registry import resolve_model_spec
from scripts.predict_matchup import apply_model_calibration, load_raw_matches, resolve_team
from src.models.cross_region_input_normalizer import (
    build_prior_domestic_family_lookup,
    decode_encoded_snapshot,
    get_region_offset,
    normalize_cross_region_batch,
    resolve_side_domestic_family,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_WORLDS_2026_ROSTERS: dict[str, tuple[str, str, list[str]]] = {
    "Gen.G": ("2804", "LCK", ["Kiin", "Canyon", "Chovy", "Ruler", "Duro"]),
    "Invictus Gaming": ("2850", "LPL", ["TheShy", "Wei", "RooKie", "JiaQi", "meiko"]),
    "Team Liquid": ("2813", "LCS", ["Morgan", "Josedeodo", "Quid", "Yeon", "CoreJJ"]),
    "CTBC Flying Oyster": ("2880", "LCP", ["Rest", "Shad0w", "Pout", "Doggo", "Kino"]),
    "Anyone's Legend": ("2848", "LPL", ["Breathe", "Tarzan", "Shanks", "Hope", "Kael"]),
    "Karmine Corp": ("2899", "LEC", ["Canna", "Yike", "Kyeahoo", "Caliste", "Busio"]),
    "G2 Esports": ("2897", "LEC", ["BrokenBlade", "SkewMond", "Caps", "Hans Sama", "Labrov"]),
    "Dplus KIA": ("2802", "LCK", ["Siwoo", "Lucid", "ShowMaker", "Smash", "Career"]),
    "Hanwha Life Esports": ("2805", "LCK", ["Zeus", "Kanavi", "Zeka", "Gumayusi", "Delight"]),
    "TOP Esports": ("2852", "LPL", ["ZUIAN", "Tian", "Creme", "JackeyLove", "Zhuo"]),
    "Los Grandes": ("2895", "CBLOL", ["Zest", "Curse", "Feisty", "Duduhh", "Ackerman"]),
    "T1": ("2809", "LCK", ["Doran", "Oner", "Faker", "Peyz", "Keria"]),
    "Team Secret Whales": ("2887", "LCP", ["Pun", "Hizto", "Dire", "Eddie", "Bie"]),
    "LYON": ("2954", "LCS", ["Dhokla", "Inspired", "Saint", "Berserker", "Isles"]),
    "Bilibili Gaming": ("2849", "LPL", ["Bin", "XUN", "Knight", "Viper", "ON"]),
    "Movistar KOI": ("2900", "LEC", ["Myrwn", "Elyoya", "jojopyun", "Supa", "Alvaro"]),
}


@pytest.fixture(scope="module")
def raw_matches() -> list[dict]:
    return load_raw_matches()


def test_all_16_worlds_2026_rosters_and_families_exact(raw_matches: list[dict]) -> None:
    """Verify resolve_team never swaps Red-side last-map rosters or resolves 2017 aliases."""
    team_fam_map, player_fam_map = build_prior_domestic_family_lookup(raw_matches)
    expected_roles = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]

    for query, (exp_id, exp_fam, exp_players) in EXPECTED_WORLDS_2026_ROSTERS.items():
        info = resolve_team(query, raw_matches)
        assert str(info["id"]) == exp_id, f"{query} resolved wrong team_id: {info['id']} != {exp_id}"
        assert list(info["players"].keys()) == expected_roles, f"{query} roles not in canonical order"
        actual_players = [info["players"][r] for r in expected_roles]
        assert actual_players == exp_players, f"{query} resolved wrong roster: {actual_players} != {exp_players}"
        assert len(set(info["player_ids"])) == 5

        fam = resolve_side_domestic_family(info["id"], info["player_ids"], team_fam_map, player_fam_map)
        assert fam == exp_fam, f"{query} resolved wrong domestic family: {fam} != {exp_fam}"


def test_cross_region_input_normalizer_domestic_invariance_and_antisymmetry() -> None:
    """Verify bit-exact identity on domestic matches and exact anti-symmetry under side swap."""
    snap_path = PROJECT_ROOT / "data/04_feature/live_c0_cache/worlds_2026_swiss_r1_snapshots.pkl"
    if not snap_path.is_file():
        pytest.skip("Cached Worlds 2026 snapshots not present")

    with snap_path.open("rb") as f:
        snaps = pickle.load(f)

    _, enc_snap, fa, fb, _, _ = snaps[146615]  # TSW (LCP) vs LYON (LCS)
    raw, hist = decode_encoded_snapshot(enc_snap)
    raw_b = {k: v[None, ...].copy() for k, v in raw.items()}
    hist_b = {k: v[None, ...].copy() for k, v in hist.items()}

    # 1. Domestic invariance (off_a == off_b)
    raw_dom, hist_dom = normalize_cross_region_batch(raw_b, hist_b, 160.0, 160.0)
    for k in raw_b:
        assert np.array_equal(raw_dom[k], raw_b[k]), f"Domestic match mutated raw.{k}"
    for k in hist_b:
        assert np.array_equal(hist_dom[k], hist_b[k]), f"Domestic match mutated hist.{k}"

    # 2. Side-swap anti-symmetry
    oa, ob = get_region_offset(fa), get_region_offset(fb)
    raw_ab, hist_ab = normalize_cross_region_batch(raw_b, hist_b, oa, ob)

    raw_swapped = {k: v[:, ::-1, ...].copy() for k, v in raw_b.items()}
    hist_swapped = {k: v[:, ::-1, ...].copy() for k, v in hist_b.items()}
    raw_ba, hist_ba = normalize_cross_region_batch(raw_swapped, hist_swapped, ob, oa)

    for k in raw_b:
        assert np.allclose(raw_ab[k][:, 0], raw_ba[k][:, 1], atol=1e-5)
        assert np.allclose(raw_ab[k][:, 1], raw_ba[k][:, 0], atol=1e-5)
    assert np.allclose(hist_ab["history"][:, 0], hist_ba["history"][:, 1], atol=1e-3)
    assert np.allclose(hist_ab["history"][:, 1], hist_ba["history"][:, 0], atol=1e-3)


def test_worlds_2026_swiss_r1_input_normalized_favorites_and_calibration() -> None:
    """Verify 100% odds-free Input-Normalized C0-D agrees on all 8/8 Worlds R1 favorites."""
    snap_path = PROJECT_ROOT / "data/04_feature/live_c0_cache/worlds_2026_swiss_r1_snapshots.pkl"
    if not snap_path.is_file():
        pytest.skip("Cached Worlds 2026 snapshots not present")

    os.environ["ENSEMBLE_RESEARCH_ROOT"] = str(PROJECT_ROOT / "data/05_model_input/canonical_benchmark")
    from src.models import a0_data_focus

    research_root = PROJECT_ROOT / "data/05_model_input/canonical_benchmark"
    backend, _ = a0_data_focus.uncertainty._activate_archive_backend(research_root)
    model_root = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
    shared = joblib.load(model_root / "shared/model.joblib")
    models = [joblib.load(model_root / f"control/seed_{s}/model.joblib") for s in a0_data_focus.SEEDS]
    calibration = joblib.load(model_root / "control/calibration/model.joblib")
    model_c0d = resolve_model_spec("c0-d")

    with snap_path.open("rb") as f:
        snaps = pickle.load(f)

    market_consensus = {
        146609: 76.9,  # Gen.G vs IG
        146610: 69.4,  # TL vs CFO
        146611: 65.0,  # AL vs KC
        146612: 43.3,  # G2 vs DK
        146613: 76.8,  # HLE vs TES
        146614: 16.8,  # LOS vs T1
        146615: 39.5,  # TSW vs LYON
        146616: 82.6,  # BLG vs MKOI
    }

    old_gaps, new_gaps = [], []
    for mid, (_, enc_snap, fa, fb, _, _) in snaps.items():
        raw, hist = decode_encoded_snapshot(enc_snap)
        raw_b = {k: v[None, ...].copy() for k, v in raw.items()}
        hist_b = {k: v[None, ...].copy() for k, v in hist.items()}
        oa, ob = get_region_offset(fa), get_region_offset(fb)

        p_old_raw = a0_data_focus.predict_models(
            backend, shared, models, calibration, raw_b, hist_b, np.array([1], dtype=np.int64)
        )["full"][0, 0]
        p_old, _ = apply_model_calibration(p_old_raw, model_c0d, best_of=1)

        raw_n, hist_n = normalize_cross_region_batch(raw_b, hist_b, oa, ob)
        p_new_raw = a0_data_focus.predict_models(
            backend, shared, models, calibration, raw_n, hist_n, np.array([1], dtype=np.int64)
        )["full"][0, 0]
        p_new, _ = apply_model_calibration(p_new_raw, model_c0d, best_of=1)

        q_mkt = market_consensus[mid]
        old_gaps.append(abs(p_old * 100.0 - q_mkt))
        new_gaps.append(abs(p_new * 100.0 - q_mkt))

        # Every single match's favorite (>50% vs <50%) must agree with market consensus
        assert (p_new >= 0.5) == (q_mkt >= 50.0), f"Match {mid} ({fa} vs {fb}) favorite mismatch: {p_new*100:.1f}% vs {q_mkt:.1f}%"

    assert np.mean(new_gaps) < 6.5, f"Expected mean market gap < 6.5pp, got {np.mean(new_gaps):.2f}pp"
    assert np.mean(new_gaps) < np.mean(old_gaps) - 7.0


def test_automatic_dynamic_region_rank_updates() -> None:
    """Verify region offsets are derived automatically from FamilyCalibratedGlicko2 state."""
    import json
    from src.models.cross_region_input_normalizer import (
        DEFAULT_REGIONAL_STATE_PATH,
        compute_dynamic_offsets_from_state,
        load_dynamic_region_offsets,
    )

    assert DEFAULT_REGIONAL_STATE_PATH.is_file()
    offsets = load_dynamic_region_offsets(DEFAULT_REGIONAL_STATE_PATH)

    # Verify exact automatic hierarchy from FamilyCalibratedGlicko2 + pyramid drift:
    # LCK & LPL > LCS > LEC > LCP > CBLOL > LJL > LLA
    assert offsets["LCK"] > offsets["LPL"] > offsets["LCS"] > offsets["LEC"]
    assert offsets["LEC"] > offsets["LCP"] > offsets["CBLOL"] > offsets["LLA"]

    # Verify changing international bridge results in state automatically shifts offsets
    state = json.loads(DEFAULT_REGIONAL_STATE_PATH.read_text(encoding="utf-8"))
    state["families"]["LEC"]["mean"] += 50.0
    updated = compute_dynamic_offsets_from_state(state)
    assert updated["LEC"] > offsets["LEC"] + 50.0
