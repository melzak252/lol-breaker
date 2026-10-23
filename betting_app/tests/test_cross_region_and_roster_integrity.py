"""Automated tests for 2026 Roster Resolution Integrity and Cross-Regional Input Normalization."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.predict_matchup import load_raw_matches, resolve_team
from src.models.cross_region_input_normalizer import (
    build_prior_domestic_family_lookup,
    normalize_cross_region_batch,
    resolve_side_domestic_family,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGIONAL_STATE = PROJECT_ROOT / "data/04_feature/live_c0_cache/regional_offsets_state.json"

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


@pytest.mark.requires_artifacts(PROJECT_ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json")
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
    """Verify domestic identity and side-swap symmetry on deterministic synthetic tensors."""
    rng = np.random.default_rng(211)
    raw_b = {
        "players": rng.normal(size=(1, 2, 5, 21)).astype(np.float32),
        "core": rng.normal(size=(1, 2, 5)).astype(np.float32),
        "team": rng.normal(size=(1, 2, 11)).astype(np.float32),
        "w20": rng.normal(size=(1, 2, 10)).astype(np.float32),
        "gates": rng.normal(size=(1, 2, 2)).astype(np.float32),
    }
    hist_b = {
        "history": rng.normal(size=(1, 2, 5, 16, 33)).astype(np.float32),
        "mask": rng.integers(0, 2, size=(1, 2, 5, 16)).astype(np.float32),
        "champions": rng.integers(0, 170, size=(1, 2, 5, 16)).astype(np.float32),
    }
    raw_dom, hist_dom = normalize_cross_region_batch(raw_b, hist_b, 160.0, 160.0)
    for key in raw_b:
        assert np.array_equal(raw_dom[key], raw_b[key]), f"Domestic match mutated raw.{key}"
    for key in hist_b:
        assert np.array_equal(hist_dom[key], hist_b[key]), f"Domestic match mutated hist.{key}"

    raw_ab, hist_ab = normalize_cross_region_batch(raw_b, hist_b, 350.0, 110.0)
    raw_swapped = {key: value[:, ::-1, ...].copy() for key, value in raw_b.items()}
    hist_swapped = {key: value[:, ::-1, ...].copy() for key, value in hist_b.items()}
    raw_ba, hist_ba = normalize_cross_region_batch(raw_swapped, hist_swapped, 110.0, 350.0)
    for key in raw_b:
        assert np.allclose(raw_ab[key][:, 0], raw_ba[key][:, 1], atol=1e-5)
        assert np.allclose(raw_ab[key][:, 1], raw_ba[key][:, 0], atol=1e-5)
    assert np.allclose(hist_ab["history"][:, 0], hist_ba["history"][:, 1], atol=1e-3)
    assert np.allclose(hist_ab["history"][:, 1], hist_ba["history"][:, 0], atol=1e-3)





@pytest.mark.requires_artifacts(REGIONAL_STATE)
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
