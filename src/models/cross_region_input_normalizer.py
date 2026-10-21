"""Causal Cross-Regional Input Normalizer for C0 / C0-D Pre-Match Tensors.

Resolves the Closed-Pool Rating & Strength-of-Schedule (SoS) Boxscore Illusion
when teams from different domestic leagues (e.g. Minor #1 vs Major #2/#3, or
LEC/ERL pyramid vs LPL/LCK) meet in cross-regional competition, operating 100%
through the model's input tensors (raw.players, raw.team, raw.w20, hist.history)
without using any bookmaker odds.

Automatic Region Rank Updates:
- Dynamically loads the live `FamilyCalibratedGlicko2` (`ratings-v2`) Gaussian
  bridge state (`families`, `tiers`, and active `players` pyramid medians) from
  `data/04_feature/live_c0_cache/regional_offsets_state.json` (rebuilt every 6h
  by `betting_app.scripts.rebuild_regional_ratings` in `heavy_maintenance_cycle`).
- Automatically updates region ranks and domestic pyramid drift whenever new
  international or domestic matches are ingested into `golgg_matches`.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.special import expit

from betting_app.core.models.native_c0 import (
    PROJECT_ROOT,
    _RAW_SHAPES,
    _TEMPORAL_SHAPES,
    _WIRE_DTYPES,
)
from src.models.competition_tiers import classify_competition

DEFAULT_REGIONAL_STATE_PATH = (
    PROJECT_ROOT / "data/04_feature/live_c0_cache/regional_offsets_state.json"
)

# Frozen 2024-2025 TRAIN/CAL Preprocessor & HistoryNormalizer standard deviations
PLAYER_SCALES = np.array(
    [
        204.745, 178.672, 51.543, 4.455, 1.720, 2.585, 0.609,
        6.846, 1.190, 5.968, 1.584, 1.417, 1.289, 1.357, 0.135,
        0.397, 0.397, 0.397, 0.397, 0.397, 0.135,
    ],
    dtype=np.float32,
)
TEAM_SCALES = np.array(
    [138.306, 195.016, 53.799, 4.784, 1.422, 3.689, 0.366, 8.558, 0.391, 9.295, 0.693],
    dtype=np.float32,
)
W20_SCALES = np.array(
    [0.230, 3.431, 3.393, 362.452, 0.818, 1376.129, 1.779, 0.300, 5269.858, 151.310],
    dtype=np.float32,
)
HIST_SCALES = np.array(
    [
        2.775, 2.138, 4.654, 1.168, 72.842, 224.512, 0.842,
        1485.2, 24.5, 920.4, 0.145, 0.068,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.500, 1.250, 165.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2,
    ],
    dtype=np.float32,
)

# Fallback prior offsets used only if regional_offsets_state.json is absent
_FALLBACK_REGION_OFFSETS: dict[str, float] = {
    "LCK": 349.0,
    "LPL": 295.0,
    "LCS": 160.0,
    "LEC": 115.0,
    "LCP": 25.0,
    "PCS": 25.0,
    "VCS": 15.0,
    "CBLOL": -35.0,
    "LLA": -45.0,
    "LJL": -55.0,
    "LFL": 20.0,
    "Prime League": 5.0,
    "SuperLiga": 10.0,
    "TCL": 0.0,
    "NLC": -5.0,
    "Ultraliga": -10.0,
    "LCK CL": 60.0,
    "LDL": 45.0,
    "NACL": -15.0,
}

RATING_MEAN_INDICES = (0, 1, 3, 5, 7, 9)
W20_SIGNS = (1.0, 0.8, -0.8, 0.9, 0.5, 1.0, 0.9, 0.6, 0.9, -0.6)
HIST_STAT_SIGNS = {
    0: 0.8,   # kills
    1: -0.8,  # deaths
    2: 0.8,   # assists
    3: 0.5,   # csm
    4: 0.9,   # gpm
    5: 0.9,   # dpm
    6: 0.5,   # vspm
    7: 1.0,   # gd@15
    8: 0.7,   # csd@15
    9: 0.8,   # xpd@15
}

_DYNAMIC_OFFSETS_CACHE: tuple[int, dict[str, float]] | None = None


def compute_dynamic_offsets_from_state(state_dict: Mapping[str, Any]) -> dict[str, float]:
    """Derive effective regional offsets automatically from FamilyCalibratedGlicko2 state."""
    fams = state_dict.get("families") or {}
    tiers = state_dict.get("tiers") or {}
    players = state_dict.get("players") or {}
    if not fams or not tiers:
        return dict(_FALLBACK_REGION_OFFSETS)

    current_date = str(state_dict.get("current_date") or "2026-10-18")
    min_year = str(max(2024, int(current_date[:4]))) + "-01-01"

    active_by_fam: dict[str, list[float]] = defaultdict(list)
    for p in players.values():
        if not isinstance(p, Mapping):
            continue
        la = str(p.get("last_activity") or "")
        if la >= min_year and int(p.get("games_played") or 0) >= 15:
            active_by_fam[str(p.get("family") or "")].append(float(p["rating"]))

    major_fams = ("LCK", "LPL", "LCS", "LEC")
    major_medians = {
        f: float(np.median(active_by_fam[f]))
        for f in major_fams
        if active_by_fam.get(f)
    }
    non_erl_base = (
        0.5 * (major_medians.get("LPL", 1700.0) + major_medians.get("LCS", 1681.0))
        if major_medians
        else 1690.0
    )

    apac_keys = [k for k in ("LCP", "PCS", "VCS", "LJL") if k in fams]
    if apac_keys:
        apac_precs = [1.0 / max(1.0, float(fams[k]["variance"])) for k in apac_keys]
        apac_vals = [
            float(fams[k]["mean"]) + float(tiers[fams[k]["tier"]]["mean"])
            for k in apac_keys
        ]
        apac_pooled = float(np.average(apac_vals, weights=apac_precs)) + 45.0
    else:
        apac_pooled = 20.0

    dynamic_offsets: dict[str, float] = {}
    for f, f_info in fams.items():
        if not isinstance(f_info, Mapping):
            continue
        t_name = str(f_info.get("tier") or "")
        if t_name not in tiers:
            continue
        bridge = float(f_info["mean"]) + float(tiers[t_name]["mean"])
        if f in major_medians:
            pyr_excess = max(0.0, major_medians[f] - non_erl_base)
            dynamic_offsets[f] = 135.0 + 1.25 * bridge - 1.35 * pyr_excess
        elif f in ("LCP", "PCS", "VCS"):
            dynamic_offsets[f] = apac_pooled
        else:
            dynamic_offsets[f] = 0.45 * bridge

    return dynamic_offsets


def load_dynamic_region_offsets(
    state_path: Path | str | None = None,
) -> dict[str, float]:
    """Load automatic regional offsets from `regional_offsets_state.json` (with mtime cache)."""
    global _DYNAMIC_OFFSETS_CACHE
    path = Path(state_path) if state_path is not None else DEFAULT_REGIONAL_STATE_PATH
    if not path.is_file():
        return dict(_FALLBACK_REGION_OFFSETS)
    mtime_ns = path.stat().st_mtime_ns
    if _DYNAMIC_OFFSETS_CACHE is not None and _DYNAMIC_OFFSETS_CACHE[0] == mtime_ns:
        return dict(_DYNAMIC_OFFSETS_CACHE[1])
    try:
        state_dict = json.loads(path.read_text(encoding="utf-8"))
        offsets = compute_dynamic_offsets_from_state(state_dict)
        _DYNAMIC_OFFSETS_CACHE = (mtime_ns, offsets)
        return dict(offsets)
    except Exception:
        return dict(_FALLBACK_REGION_OFFSETS)


def build_prior_domestic_family_lookup(
    raw_matches: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build strictly causal team_id -> family and player_id -> family maps from domestic matches."""
    team_fam: dict[str, str] = {}
    player_fam: dict[str, str] = {}
    for m in raw_matches:
        tname = str(m.get("tournament_name") or "")
        ci = classify_competition(tname)
        if ci.scope.value == "domestic" and ci.family not in ("unknown", "international"):
            t1_id = str(m.get("t1_id") or m.get("team1_id") or "").strip()
            t2_id = str(m.get("t2_id") or m.get("team2_id") or "").strip()
            if t1_id:
                team_fam[t1_id] = ci.family
            if t2_id:
                team_fam[t2_id] = ci.family
            for g in m.get("games") or []:
                for side_key in ("t1_players", "t2_players"):
                    for p in (g.get(side_key) or {}).values():
                        pid = str(p.get("player_id") or p.get("id") or "").strip()
                        if pid:
                            player_fam[pid] = ci.family
    return team_fam, player_fam


def resolve_side_domestic_family(
    team_id: str,
    player_ids: Iterable[str],
    team_fam_map: Mapping[str, str],
    player_fam_map: Mapping[str, str],
) -> str:
    """Resolve a team's domestic league family from its team_id or majority starting 5 players."""
    tid = str(team_id).strip()
    if tid in team_fam_map:
        return team_fam_map[tid]
    pfams = [player_fam_map[str(p).strip()] for p in player_ids if str(p).strip() in player_fam_map]
    if pfams:
        return Counter(pfams).most_common(1)[0][0]
    return "unknown"


def get_region_offset(family: str, state_path: Path | str | None = None) -> float:
    """Return the dynamically computed regional strength offset in Glicko points."""
    offsets = load_dynamic_region_offsets(state_path)
    return float(offsets.get(family, 0.0))


def normalize_cross_region_batch(
    raw_batch: Mapping[str, np.ndarray],
    hist_batch: Mapping[str, np.ndarray],
    off_a: np.ndarray | float,
    off_b: np.ndarray | float,
    *,
    z_per_100gl: float = 0.28,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply Overconfidence-Protected Z-Aligned Cross-Regional Input Normalization on batch tensors."""
    raw_out = {k: v.copy() for k, v in raw_batch.items()}
    hist_out = {k: v.copy() for k, v in hist_batch.items()}

    oa = np.atleast_1d(np.asarray(off_a, dtype=np.float32))
    ob = np.atleast_1d(np.asarray(off_b, dtype=np.float32))
    diff = oa - ob
    if np.all(np.abs(diff) <= 1.0):
        return raw_out, hist_out

    h_f32 = hist_out["history"].astype(np.float32)
    mask = hist_out["mask"] > 0

    sgn_reg = np.sign(diff)
    raw_gl_diff = (
        raw_out["players"][:, 0, :, 1].mean(axis=1)
        - raw_out["players"][:, 1, :, 1].mean(axis=1)
    ) / PLAYER_SCALES[1]
    raw_w20_diff = (raw_out["w20"][:, 0, 0] - raw_out["w20"][:, 1, 0]) / W20_SCALES[0]
    already_aligned_z = sgn_reg * (0.6 * raw_gl_diff + 0.4 * raw_w20_diff)

    gate = np.where(np.abs(diff) > 1.0, expit(1.5 * (0.65 - already_aligned_z)), 0.0).astype(np.float32)
    raw_d100 = 0.5 * diff / 100.0
    inv_deinflate = sgn_reg * (-0.16 * np.minimum(0.0, already_aligned_z)) * np.tanh(np.abs(diff) / 150.0)
    dz_a = (gate * z_per_100gl * 1.4 * np.tanh(raw_d100 / 1.4) + inv_deinflate).astype(np.float32)
    for side, dz in ((0, dz_a), (1, -dz_a)):
        for idx_r in RATING_MEAN_INDICES:
            raw_out["players"][:, side, :, idx_r] += (dz * PLAYER_SCALES[idx_r])[:, None]
            has_org = raw_out["gates"][:, side, 0] > 0
            raw_out["team"][has_org, side, idx_r] += (dz * TEAM_SCALES[idx_r])[has_org]

        has_w20 = raw_out["gates"][:, side, 1] > 0
        for w_i, sgn in enumerate(W20_SIGNS):
            raw_out["w20"][has_w20, side, w_i] += (0.85 * dz * sgn * W20_SCALES[w_i])[has_w20]
        raw_out["w20"][has_w20, side, 0] = np.clip(raw_out["w20"][has_w20, side, 0], 0.05, 0.95)

        dz_exp = dz[:, None, None]
        m_side = mask[:, side, :, :]
        for h_i, sgn in HIST_STAT_SIGNS.items():
            obs = m_side & (h_f32[:, side, :, :, 12 + h_i] > 0)
            h_f32[:, side, :, :, h_i] = np.where(
                obs,
                h_f32[:, side, :, :, h_i] + 0.85 * dz_exp * sgn * HIST_SCALES[h_i],
                h_f32[:, side, :, :, h_i],
            )
        h_f32[:, side, :, :, 24] = np.where(
            m_side,
            np.clip(h_f32[:, side, :, :, 24] + 0.85 * dz_exp * HIST_SCALES[24], 0.0, 1.0),
            0.0,
        )
        h_f32[:, side, :, :, 26] = np.where(
            m_side,
            h_f32[:, side, :, :, 26] + dz_exp * HIST_SCALES[26],
            0.0,
        )

    hist_out["history"] = h_f32.astype(np.float16)
    return raw_out, hist_out


def decode_encoded_snapshot(
    encoded_snapshot: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Decode base64-encoded C0 snapshot into (raw, hist) NumPy arrays."""
    raw: dict[str, np.ndarray] = {}
    for k, shape in _RAW_SHAPES.items():
        v = encoded_snapshot["raw"][k]
        dt = _WIRE_DTYPES[v["dtype"]]
        raw[k] = np.frombuffer(base64.b64decode(v["data_base64"]), dtype=dt).reshape(shape).copy()
    hist: dict[str, np.ndarray] = {}
    for k, shape in _TEMPORAL_SHAPES.items():
        v = encoded_snapshot["hist"][k]
        dt = _WIRE_DTYPES[v["dtype"]]
        hist[k] = np.frombuffer(base64.b64decode(v["data_base64"]), dtype=dt).reshape(shape).copy()
    return raw, hist


def normalize_encoded_snapshot(
    encoded_snapshot: Mapping[str, Any],
    off_a: float,
    off_b: float,
    *,
    z_per_100gl: float = 0.28,
) -> dict[str, Any]:
    """Return a decoded c0_snapshot dict ({target, raw, hist}) with cross-region input normalization."""
    raw, hist = decode_encoded_snapshot(encoded_snapshot)
    if abs(off_a - off_b) <= 1.0:
        return {"target": dict(encoded_snapshot["target"]), "raw": raw, "hist": hist}
    raw_b = {k: v[None, ...] for k, v in raw.items()}
    hist_b = {k: v[None, ...] for k, v in hist.items()}
    raw_norm_b, hist_norm_b = normalize_cross_region_batch(
        raw_b, hist_b, np.array([off_a], dtype=np.float32), np.array([off_b], dtype=np.float32),
        z_per_100gl=z_per_100gl,
    )
    raw_norm = {k: v[0] for k, v in raw_norm_b.items()}
    hist_norm = {k: v[0] for k, v in hist_norm_b.items()}
    return {"target": dict(encoded_snapshot["target"]), "raw": raw_norm, "hist": hist_norm}
