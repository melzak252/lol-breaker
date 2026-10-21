#!/usr/bin/env python3
"""Track 3: Continuous Champion Archetype Embeddings + 15% <UNK> Token Dropout (hist.champions).

Empirically executes and evaluates Track 3 on the canonical Causal-C0 dataset:
- TRAIN: 36,205 series < 2025-07-01
- CAL: 1,328 series in 2025H2
- ASSESS: 3,103 canonical 2026 series
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, logsumexp
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import a0_data_focus as focus
from src.models import a0_uncertainty as uncertainty
from src.analysis.metrics import calculate_ece
MODES = ("full", "no_w20", "no_organization")
MODE_FLAGS = ((True, True), (False, True), (False, False))
SEEDS = (20260910, 20260911, 20260912)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261017


def paired_monthly_bootstrap(
    delta_vector: np.ndarray,
    months: list[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Paired monthly-block bootstrap matching canonical research standards."""
    df = pd.DataFrame({"delta": delta_vector, "month": months})
    unique_months = sorted(df["month"].unique())
    n_blocks = len(unique_months)
    n_total = len(df)

    if n_blocks == 0 or n_total == 0:
        return {
            "delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "draws": draws,
            "seed": seed,
            "blocks": n_blocks,
            "n": n_total,
            "equal_month_delta": 0.0,
            "equal_month_ci_lower": 0.0,
            "equal_month_ci_upper": 0.0,
        }

    grouped = df.groupby("month")["delta"].agg(d_sum="sum", d_count="count", d_mean="mean").loc[unique_months]
    sums = grouped["d_sum"].to_numpy(dtype=np.float64)
    counts = grouped["d_count"].to_numpy(dtype=np.float64)
    means = grouped["d_mean"].to_numpy(dtype=np.float64)

    observed_match_weighted = float(sums.sum() / counts.sum())
    observed_equal_month = float(means.mean())

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n_blocks, size=(draws, n_blocks))

    draw_sums = sums[sampled_indices].sum(axis=1)
    draw_counts = counts[sampled_indices].sum(axis=1)
    mw_draws = draw_sums / draw_counts
    em_draws = means[sampled_indices].mean(axis=1)

    alpha_tail = (1.0 - confidence_level) / 2.0
    return {
        "delta": observed_match_weighted,
        "ci_lower": float(np.quantile(mw_draws, alpha_tail)),
        "ci_upper": float(np.quantile(mw_draws, 1.0 - alpha_tail)),
        "draws": draws,
        "seed": seed,
        "blocks": n_blocks,
        "n": n_total,
        "confidence_level": confidence_level,
        "equal_month_delta": observed_equal_month,
        "equal_month_ci_lower": float(np.quantile(em_draws, alpha_tail)),
        "equal_month_ci_upper": float(np.quantile(em_draws, 1.0 - alpha_tail)),
    }


def compute_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    p_c0: np.ndarray,
    months: list[str],
) -> dict[str, Any]:
    """Compute unclipped LogLoss, Brier, ECE10, tail count, and bootstrap CIs vs C0."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    p0 = np.asarray(p_c0, dtype=np.float64)

    # Unclipped LogLoss
    loss_cand = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    loss_c0 = -(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0))

    ll = float(np.mean(loss_cand))
    brier = float(np.mean((p - y) ** 2))
    ece = float(calculate_ece(y.astype(int), p, n_bins=10))
    tails = int(np.sum(loss_cand >= 2.5))

    delta_loss = loss_cand - loss_c0
    boot = paired_monthly_bootstrap(delta_loss, months)

    return {
        "n": len(y),
        "log_loss": ll,
        "brier": brier,
        "ece10": ece,
        "tail_loss_ge_2_5": tails,
        "delta_vs_c0": boot,
    }


def build_champion_archetypes(
    train_champs: np.ndarray,
    train_mask: np.ndarray,
    train_hist: np.ndarray,
    prior_weight_m: float = 50.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute empirical Bayes 18-dimensional continuous functional archetype vectors.

    Strictly causal: fitted exclusively on TRAIN rows where mask == 1 and champions > 0.
    Channels:
    - 0..11: 12 normalized stat tendencies (kills, deaths, assists, csm, gpm, dpm, vspm,
             gd@15, csd@15, xpd@15, kp%, dmg%)
    - 12..17: 6 centered role play-rate probabilities (TOP, JUNGLE, MID, ADC, SUPPORT, UNKNOWN)
    """
    valid_slot = (train_mask == 1) & (train_champs > 0)
    champs_flat = train_champs[valid_slot]
    hist_flat = train_hist[valid_slot]

    max_c = int(champs_flat.max())
    assert max_c <= 170, f"Expected max TRAIN champion <= 170, got {max_c}"

    stat_vals = hist_flat[:, :12]
    stat_obs = hist_flat[:, 12:24] > 0
    role_vals = hist_flat[:, 27:33].astype(np.float64)

    stat_sums = np.zeros(12, dtype=np.float64)
    stat_sq_sums = np.zeros(12, dtype=np.float64)
    stat_counts = np.zeros(12, dtype=np.float64)

    for k in range(12):
        obs_k = stat_obs[:, k]
        vals_k = stat_vals[obs_k, k].astype(np.float64)
        stat_sums[k] = vals_k.sum()
        stat_sq_sums[k] = (vals_k ** 2).sum()
        stat_counts[k] = len(vals_k)

    global_stat_mean = stat_sums / stat_counts
    global_stat_std = np.sqrt(np.maximum(stat_sq_sums / stat_counts - global_stat_mean ** 2, 1e-8))
    global_role_mean = role_vals.sum(axis=0) / len(role_vals)

    # Dimension 172 to accommodate index 0 (padding), 1..170 (seen champions), 171 (<UNK>)
    archetypes = np.zeros((172, 18), dtype=np.float32)

    # Seen champions 1..max_c
    for c in range(1, max_c + 1):
        c_mask = champs_flat == c
        n_c = int(c_mask.sum())
        if n_c == 0:
            continue
        c_hist = hist_flat[c_mask]

        # 12 stat tendencies with Empirical Bayes shrinkage toward 0.0 (global prior mean)
        for k in range(12):
            obs_k = c_hist[:, 12 + k] > 0
            n_obs = int(obs_k.sum())
            if n_obs > 0:
                z_vals = (c_hist[obs_k, k] - global_stat_mean[k]) / global_stat_std[k]
                archetypes[c, k] = float(z_vals.sum() / (n_obs + prior_weight_m))
            else:
                archetypes[c, k] = 0.0

        # 6 role play-rate probabilities with Empirical Bayes shrinkage toward global role mean
        # Standardized / centered by subtracting global_role_mean
        c_roles = c_hist[:, 27:33].astype(np.float64)
        role_counts = c_roles.sum(axis=0)
        p_c_role = (role_counts + prior_weight_m * global_role_mean) / (n_c + prior_weight_m)
        archetypes[c, 12:18] = (p_c_role - global_role_mean).astype(np.float32)

    metadata = {
        "num_valid_train_maps": int(len(champs_flat)),
        "num_seen_train_champions": int(len(np.unique(champs_flat))),
        "prior_weight_m": prior_weight_m,
        "global_stat_mean": global_stat_mean.tolist(),
        "global_stat_std": global_stat_std.tolist(),
        "global_role_mean": global_role_mean.tolist(),
    }
    return archetypes, metadata


def compute_pool_features(
    champions: np.ndarray,
    mask: np.ndarray,
    history: np.ndarray,
    archetypes: np.ndarray,
) -> np.ndarray:
    """Compute causal player champion pool breadth, comfort, and archetype alignment features.

    Returns:
        Array of shape (N, 2, 5, 7), dtype float32:
        - 0: pool_size_norm (unique champs / 16)
        - 1: entropy_norm (Shannon entropy / 4.0)
        - 2: top1_concentration (share on most played champion)
        - 3: top2_concentration (share on top 2 played champions)
        - 4: comfort_wr_diff (win rate on top-2 picks minus off-comfort picks)
        - 5: carry_inclination (mean dmg% tendency of played champions)
        - 6: tank_inclination (mean assists tendency of played champions)
    """
    n_series = champions.shape[0]
    n_players = n_series * 2 * 5

    valid_mask = (mask == 1) & (champions > 0)
    flat_valid = valid_mask.reshape(n_players, 16)
    flat_c = np.where(valid_mask, champions, 0).reshape(n_players, 16)
    flat_wins = history[..., 24].reshape(n_players, 16)

    # Accumulate counts per champion ID (bins 0..171)
    counts = np.zeros((n_players, 172), dtype=np.int16)
    row_idx = np.repeat(np.arange(n_players, dtype=np.int64), 16)
    safe_flat_c = np.clip(flat_c.ravel(), 0, 171).astype(np.int64)
    np.add.at(counts, (row_idx, safe_flat_c), 1)
    counts[:, 0] = 0  # Ignore padding / invalid

    champ_counts = counts[:, 1:]  # (n_players, 171)
    valid_games = champ_counts.sum(axis=1)  # (n_players,)
    has_games = valid_games > 0
    k_valid = np.maximum(valid_games.astype(np.float32), 1.0)

    # 1. Pool size norm
    unique_champs = (champ_counts > 0).sum(axis=1).astype(np.float32)
    pool_size_norm = np.where(has_games, unique_champs / 16.0, 0.0)

    # 2. Entropy norm: H / 4.0
    entropy_norm = np.zeros(n_players, dtype=np.float32)
    nz_rows, nz_cols = np.nonzero(champ_counts)
    if len(nz_rows) > 0:
        nz_counts = champ_counts[nz_rows, nz_cols].astype(np.float32)
        nz_terms = nz_counts * np.log2(nz_counts)
        sum_terms = np.zeros(n_players, dtype=np.float32)
        np.add.at(sum_terms, nz_rows, nz_terms)
        h_entropy = np.log2(k_valid) - (sum_terms / k_valid)
        entropy_norm = np.where(has_games, np.clip(h_entropy / 4.0, 0.0, 1.0), 0.0)

    # 3 & 4. Top-1 and Top-2 concentration
    top2_vals = np.partition(champ_counts, -2, axis=1)[:, -2:]
    top1_count = top2_vals.max(axis=1).astype(np.float32)
    top2_count = top2_vals.sum(axis=1).astype(np.float32)
    top1_conc = np.where(has_games, top1_count / k_valid, 0.0)
    top2_conc = np.where(has_games, top2_count / k_valid, 0.0)

    # 5. Comfort win-rate difference
    threshold = top2_vals.min(axis=1, keepdims=True)
    count_of_map_champ = counts[np.arange(n_players)[:, None], safe_flat_c.reshape(n_players, 16)]
    is_comfort = flat_valid & (count_of_map_champ >= threshold) & (count_of_map_champ > 0)
    is_off_comfort = flat_valid & (~is_comfort)

    n_comfort = is_comfort.sum(axis=1)
    n_off = is_off_comfort.sum(axis=1)
    wins_comfort = (flat_wins * is_comfort).sum(axis=1)
    wins_off = (flat_wins * is_off_comfort).sum(axis=1)

    wr_comfort = np.where(n_comfort > 0, (wins_comfort + 1.0) / (n_comfort + 2.0), 0.5)
    wr_off = np.where(n_off > 0, (wins_off + 1.0) / (n_off + 2.0), 0.5)
    comfort_wr_diff = np.where(has_games, wr_comfort - wr_off, 0.0)

    # 6 & 7. Player carry & tank inclinations from archetypes
    carry_arch = archetypes[:, 11]  # dmg% tendency
    tank_arch = archetypes[:, 2]   # assists tendency
    map_carry = carry_arch[safe_flat_c.reshape(n_players, 16)]
    map_tank = tank_arch[safe_flat_c.reshape(n_players, 16)]

    player_carry = np.where(has_games, (map_carry * flat_valid).sum(axis=1) / k_valid, 0.0)
    player_tank = np.where(has_games, (map_tank * flat_valid).sum(axis=1) / k_valid, 0.0)

    features = np.stack(
        [
            pool_size_norm,
            entropy_norm,
            top1_conc,
            top2_conc,
            comfort_wr_diff,
            player_carry,
            player_tank,
        ],
        axis=1,
    ).astype(np.float32)

    return features.reshape(n_series, 2, 5, 7)


class ChampionAugmentedResidual(nn.Module):
    """Augmented C0 CommonMapResidual with continuous archetypes, discrete embeddings, and pool features."""

    def __init__(
        self,
        base_network: nn.Module,
        archetype_vectors: np.ndarray,
        num_champions: int = 172,
        width: int = 32,
    ):
        super().__init__()
        self.width = width
        self.kind = "attention"
        self.has_history = True

        # Copy base network modules
        self.player = copy.deepcopy(base_network.player)
        self.history_encoder = copy.deepcopy(base_network.history_encoder)
        self.empty = nn.Parameter(base_network.empty.detach().clone())
        self.attention = copy.deepcopy(base_network.attention)
        self.head = copy.deepcopy(base_network.head)
        self.auxiliary_head = copy.deepcopy(base_network.auxiliary_head)

        # Buffer for continuous archetype table: shape (172, 18)
        self.register_buffer("archetypes", torch.as_tensor(archetype_vectors, dtype=torch.float32))

        # Small residual embedding for discrete champion tokens (8 dims)
        self.champion_emb = nn.Embedding(num_champions, 8, padding_idx=0)
        nn.init.normal_(self.champion_emb.weight, std=0.01)
        with torch.no_grad():
            self.champion_emb.weight[0].zero_()

        # Projection of champion representation: (18 archetype + 8 discrete) -> width (32)
        # bias=False ensures exact zero-output for padding / cold starts
        self.champ_token_proj = nn.Linear(18 + 8, width, bias=False)
        nn.init.normal_(self.champ_token_proj.weight, std=0.01)

        # Player champion pool features projection: 7 -> width (32)
        self.pool = nn.Linear(7, width, bias=False)
        nn.init.normal_(self.pool.weight, std=0.01)

    def _representation(self, x: dict[str, torch.Tensor], champ_dropout: float = 0.0) -> torch.Tensor:
        h = self.player(x["players"])
        if "pool" in x:
            h = h + self.pool(x["pool"])

        n = h.shape[0]
        mask = x["mask"].reshape(-1, 16).bool()
        values = x["history"].reshape(-1, 16, 33)
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        encoded = self.history_encoder(values)

        if "champions" in x:
            raw_champs = x["champions"].reshape(-1, 16).long()
            is_valid = mask & (raw_champs > 0)
            unk_idx = 171

            # Seen champions are 1..170; OOV tokens map to 171 (<UNK>)
            safe_discrete = torch.where((raw_champs > 0) & (raw_champs <= 170), raw_champs, unk_idx)
            safe_discrete = torch.where(is_valid, safe_discrete, torch.zeros_like(safe_discrete))

            # 15% UNK token dropout during training
            if self.training and champ_dropout > 0.0:
                drop_mask = torch.rand(safe_discrete.shape, device=safe_discrete.device) < champ_dropout
                safe_discrete = torch.where(is_valid & drop_mask, unk_idx, safe_discrete)

            # Continuous archetype vector (defaults to 0.0 for OOV / padding)
            safe_arch_idx = torch.where((raw_champs >= 0) & (raw_champs <= 170), raw_champs, 0)
            arch_feats = self.archetypes[safe_arch_idx]
            arch_feats = torch.where(is_valid[..., None], arch_feats, torch.zeros_like(arch_feats))

            emb_feats = self.champion_emb(safe_discrete)
            champ_combined = torch.cat([arch_feats, emb_feats], dim=-1)
            champ_token = self.champ_token_proj(champ_combined)
            champ_token = torch.where(mask[..., None], champ_token, torch.zeros_like(champ_token))

            encoded = encoded + champ_token

        encoded = torch.cat([self.empty.expand(len(encoded), -1, -1), encoded], 1)
        valid = torch.cat([torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask], 1)
        weight = self.attention(encoded).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        out = (encoded * weight[..., None]).sum(1)
        h = h + out.reshape(n, 2, 5, self.width)
        return torch.tanh(h)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        best_of_index: Any = None,
        return_aux: bool = False,
        champ_dropout: float = 0.0,
    ) -> Any:
        h = self._representation(x, champ_dropout=champ_dropout)
        team = torch.cat([x["context"], h.mean(2), h.std(2, unbiased=False)], -1)
        utility = self.head(team)
        rows = torch.arange(len(h), device=h.device)
        index = torch.zeros(len(x["players"]), dtype=torch.long, device=x["players"].device)
        z = utility[rows, 0, index] - utility[rows, 1, index]
        return (z, self.auxiliary_head(h)) if return_aux else z


def train_candidate_seed_fast(
    seed: int,
    base_model: Any,
    archetypes: np.ndarray,
    train_pool: np.ndarray,
    train_raw: dict[str, np.ndarray],
    train_mask: np.ndarray,
    train_history_norm: np.ndarray,
    train_champs: np.ndarray,
    train_targets: np.ndarray,
    train_t_mask: np.ndarray,
    best_of_train: np.ndarray,
    a_wins_train: np.ndarray,
    b_wins_train: np.ndarray,
    epochs: int = 8,
    batch_size: int = 256,
    champ_dropout: float = 0.15,
) -> tuple[ChampionAugmentedResidual, float]:
    """Train one seed of ChampionAugmentedResidual with in-memory batches for maximal efficiency."""
    from src.models.roster_optional import linear_design
    from src.models.auxiliary_player import masked_auxiliary_loss
    from src.models.beta_training_torch import beta_terminal_nll

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    network = ChampionAugmentedResidual(base_model.network, archetypes)
    eta = nn.Parameter(torch.tensor(math.log(1.0 / 9.0), dtype=torch.double))

    optimizer = torch.optim.AdamW(
        [
            {"params": list(network.parameters()), "weight_decay": 0.01},
            {"params": [eta], "weight_decay": 0.0},
        ],
        lr=0.001,
    )

    network.train()
    N = len(best_of_train)
    print(f"--- Training Seed {seed} for {epochs} epochs (batch_size={batch_size}) ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0

        for begin in range(0, N, batch_size):
            end = min(begin + batch_size, N)
            batch_slice = slice(begin, end)
            n_batch = end - begin

            # Gates dropout on raw
            part = {k: v[batch_slice].copy() for k, v in train_raw.items()}
            gates = part["gates"]
            gates[..., 1] *= (np_rng.random(gates.shape[:2]) >= 0.5)
            gates *= (np_rng.random(gates.shape[:2]) >= 0.2)[..., None]
            part["gates"] = gates

            p, c = base_model.baseline.normalizer.transform(part, True, True)
            offset = base_model.baseline.linear.decision_function(
                linear_design(c, (best_of_train[batch_slice] - 1) // 2)
            )

            inputs = {
                "players": torch.from_numpy(p.astype(np.float32)),
                "context": torch.from_numpy(c.astype(np.float32)),
                "mask": torch.from_numpy(train_mask[batch_slice]),
                "history": torch.from_numpy(train_history_norm[batch_slice]),
                "champions": torch.from_numpy(train_champs[batch_slice]),
                "pool": torch.from_numpy(train_pool[batch_slice]),
            }

            residual, pred = network(inputs, return_aux=True, champ_dropout=champ_dropout)
            logits = (torch.from_numpy(offset.astype(np.float32)) + residual).double()

            bo_t = torch.as_tensor(best_of_train[batch_slice])
            a_t = torch.as_tensor(a_wins_train[batch_slice])
            b_t = torch.as_tensor(b_wins_train[batch_slice])
            main_loss = beta_terminal_nll(logits, bo_t, a_t, b_t, eta).mean()

            tgt = torch.from_numpy(train_targets[batch_slice])
            bm = torch.from_numpy(train_t_mask[batch_slice])
            aux_loss = masked_auxiliary_loss(pred, tgt, bm)

            loss = main_loss + 0.1 * aux_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*network.parameters(), eta], 5.0)
            optimizer.step()

            epoch_main_loss += main_loss.item() * n_batch
            epoch_aux_loss += aux_loss.item() * n_batch

        mean_main = epoch_main_loss / N
        mean_aux = epoch_aux_loss / N
        ep_sec = time.time() - epoch_t0
        print(f"  [Seed {seed}] Epoch {epoch:2d}/{epochs:2d}: main_loss={mean_main:.4f}, aux_loss={mean_aux:.4f} ({ep_sec:.1f}s)")

    network.eval()
    rho = float(0.5 * torch.sigmoid(eta.detach()).item())
    total_sec = time.time() - start_time
    print(f"Seed {seed} finished in {total_sec:.1f}s (rho={rho:.4f}).")
    return network, rho


def compute_neural_logits_for_network(
    network: nn.Module,
    base_model: Any,
    data: dict[str, Any],
    pool_features: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute map logits across all three modes for a candidate network.

    Returns:
        Array of shape (len(indices), 3), dtype float64.
    """
    best_of = data["frame"]["best_of"].to_numpy(dtype=int)
    n = len(indices)
    modes_z = []

    for mode in MODES:
        w20 = mode != "no_organization" and mode != "no_w20"
        org = mode != "no_organization"
        mode_logits = []

        with torch.no_grad():
            for begin in range(0, n, batch_size):
                selected = indices[begin:begin + batch_size]
                inputs, offset = base_model.batch(
                    data["raw"], data["history"], None, selected, best_of[selected], w20=w20, org=org
                )
                inputs["champions"] = torch.as_tensor(data["history"]["champions"][selected], dtype=torch.long)
                inputs["pool"] = torch.as_tensor(pool_features[selected], dtype=torch.float32)

                residual = network(inputs, return_aux=False, champ_dropout=0.0)
                logits = (offset + residual).numpy()
                mode_logits.append(logits)

        modes_z.append(np.concatenate(mode_logits, axis=0))

    # Shape: (N, 3)
    return np.stack(modes_z, axis=1)


def main() -> None:
    torch.set_num_threads(8)
    output_dir = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Track 3: Continuous Champion Archetype Embeddings + 15% UNK Token Dropout ===")
    research_root = Path("/home/melzak/.codex/.chatgpt-projects/g-p-6944b69bdee08191acfcfeb7a38f372a")

    # 1. Load data and frozen C0 artifacts
    print("\n[Step 1] Loading canonical dataset and frozen C0 artifacts...")
    backend, _ = uncertainty._activate_archive_backend(research_root)
    from src.models.competition_tiers import classify_competition
    data = uncertainty._load_training_data(research_root)
    frame = data["frame"]
    split = focus.partition_indices(frame, history=data["history"], raw=data["raw"])

    train_ix = split["train"]
    cal_ix = split["cal"]
    assess_ix = split["validation"]

    print(f"Dataset split: TRAIN={len(train_ix):,}, CAL={len(cal_ix):,}, ASSESS={len(assess_ix):,}")
    assert len(train_ix) == 36205, f"Expected 36,205 TRAIN rows, got {len(train_ix)}"
    assert len(cal_ix) == 1328, f"Expected 1,328 CAL rows, got {len(cal_ix)}"
    assert len(assess_ix) == 3103, f"Expected 3,103 ASSESS rows, got {len(assess_ix)}"

    model_root = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
    shared = joblib.load(model_root / "shared/model.joblib")
    c0_models = [joblib.load(model_root / "control" / f"seed_{seed}/model.joblib") for seed in SEEDS]
    c0_calibration = joblib.load(model_root / "control/calibration/model.joblib")

    # Replicate C0 baseline on ASSESS
    raw_assess, hist_assess = focus._subset_inputs(data, assess_ix)
    hist_assess = {k: v[:] for k, v in hist_assess.items()}
    best_of_assess = frame["best_of"].to_numpy(dtype=int)[assess_ix]
    y_assess = frame["y"].to_numpy(dtype=float)[assess_ix]
    dates_assess = pd.to_datetime(frame["date"].iloc[assess_ix])
    months_assess = dates_assess.dt.strftime("%Y-%m").tolist()

    c0_assess_preds = focus.predict_models(backend, shared, c0_models, c0_calibration, raw_assess, hist_assess, best_of_assess)
    p_c0_full = c0_assess_preds["full"][:, 0]
    ll_c0 = float(-np.mean(y_assess * np.log(p_c0_full) + (1.0 - y_assess) * np.log(1.0 - p_c0_full)))
    brier_c0 = float(np.mean((p_c0_full - y_assess) ** 2))
    print(f"Verified Frozen C0 Baseline on ASSESS (N=3,103): LogLoss = {ll_c0:.9f}, Brier = {brier_c0:.9f}")
    assert abs(ll_c0 - 0.535448701) < 1e-6, f"Expected C0 LogLoss 0.535448701, got {ll_c0}"
    assert abs(brier_c0 - 0.181034493) < 1e-6, f"Expected C0 Brier 0.181034493, got {brier_c0}"

    # Also compute C0 on CAL
    raw_cal, hist_cal = focus._subset_inputs(data, cal_ix)
    hist_cal = {k: v[:] for k, v in hist_cal.items()}
    best_of_cal = frame["best_of"].to_numpy(dtype=int)[cal_ix]
    y_cal = frame["y"].to_numpy(dtype=float)[cal_ix]
    c0_cal_preds = focus.predict_models(backend, shared, c0_models, c0_calibration, raw_cal, hist_cal, best_of_cal)
    p_c0_cal = c0_cal_preds["full"][:, 0]
    ll_c0_cal = float(-np.mean(y_cal * np.log(p_c0_cal) + (1.0 - y_cal) * np.log(1.0 - p_c0_cal)))
    print(f"Frozen C0 Baseline on CAL (N=1,328): LogLoss = {ll_c0_cal:.6f}")

    # 2. Build strictly causal Continuous Champion Archetypes from TRAIN
    print("\n[Step 2] Building strictly causal continuous champion archetype vectors from TRAIN...")
    archetypes, arch_meta = build_champion_archetypes(
        data["history"]["champions"][train_ix],
        data["history"]["mask"][train_ix],
        data["history"]["history"][train_ix],
        prior_weight_m=50.0,
    )
    print(f"Archetype table built: {archetypes.shape} (172 rows x 18 dimensions).")
    print(f"Trained on {arch_meta['num_valid_train_maps']:,} valid maps across {arch_meta['num_seen_train_champions']} champions.")
    assert np.all(archetypes[0] == 0.0), "OOV archetype at index 0 must be strictly 0.0"
    assert np.all(archetypes[171] == 0.0), "OOV archetype at index 171 (<UNK>) must be strictly 0.0"

    # 3. Compute Player Champion Pool Breadth / Comfort Features
    print("\n[Step 3] Computing player champion pool breadth & comfort features...")
    pool_features = compute_pool_features(
        data["history"]["champions"],
        data["history"]["mask"],
        data["history"]["history"],
        archetypes,
    )
    print(f"Pool features computed: {pool_features.shape}")
    assert pool_features.shape == (len(frame), 2, 5, 7)

    # 4. Prepare in-memory arrays for fast training
    print("\n[Step 4] Pre-transforming TRAIN arrays into memory for fast training...")
    t_prep = time.time()
    normalizer = c0_models[0].normalizer
    train_history_chunks = []
    for begin in range(0, len(train_ix), 2048):
        ix = train_ix[begin:begin + 2048]
        mask_c = np.asarray(data["history"]["mask"][ix], dtype=bool)
        h_c = normalizer.transform(data["history"]["history"][ix], mask_c)
        train_history_chunks.append(h_c)
    train_history_norm = np.concatenate(train_history_chunks, axis=0)

    from src.models.roster_optional import subset
    from src.models.auxiliary_player import TargetScaler
    train_raw = subset(data["raw"], train_ix)
    train_mask = np.asarray(data["history"]["mask"][train_ix], dtype=bool)
    train_champs = np.asarray(data["history"]["champions"][train_ix], dtype=np.int64)
    train_pool = pool_features[train_ix]

    target_scaler = TargetScaler().fit(data["targets"], data["target_mask"], train_ix)
    train_targets = target_scaler.transform(data["targets"][train_ix], data["target_mask"][train_ix])
    train_t_mask = np.asarray(data["target_mask"][train_ix], bool)

    best_of_train = frame["best_of"].to_numpy(dtype=int)[train_ix]
    a_wins_train = data["wins_a"][train_ix]
    b_wins_train = data["wins_b"][train_ix]
    print(f"In-memory preparation finished in {time.time() - t_prep:.1f}s.")

    # Train 3 seeds with 15% UNK token dropout (8 epochs each)
    candidate_networks = []
    candidate_rhos = []
    for seed in SEEDS:
        ckpt_path = output_dir / f"checkpoint_seed_{seed}.pt"
        if ckpt_path.exists():
            print(f"Loading cached checkpoint for Seed {seed} from {ckpt_path}...")
            state = torch.load(ckpt_path)
            net = ChampionAugmentedResidual(c0_models[0].network, archetypes)
            net.load_state_dict(state["network_state_dict"])
            net.eval()
            rho = state["rho"]
        else:
            net, rho = train_candidate_seed_fast(
                seed=seed,
                base_model=c0_models[0],
                archetypes=archetypes,
                train_pool=train_pool,
                train_raw=train_raw,
                train_mask=train_mask,
                train_history_norm=train_history_norm,
                train_champs=train_champs,
                train_targets=train_targets,
                train_t_mask=train_t_mask,
                best_of_train=best_of_train,
                a_wins_train=a_wins_train,
                b_wins_train=b_wins_train,
                epochs=8,
                batch_size=256,
                champ_dropout=0.15,
            )
            torch.save({"network_state_dict": net.state_dict(), "rho": rho}, ckpt_path)
        candidate_networks.append(net)
        candidate_rhos.append(rho)

    # 5. Calibration on CAL (N=1,328)
    print("\n[Step 5] Performing native joint calibration on CAL...")
    cal_neural_z_seeds = [
        compute_neural_logits_for_network(net, c0_models[0], data, pool_features, cal_ix)
        for net in candidate_networks
    ]
    # Stack across seeds -> (N, 3_modes, 3_seeds)
    cal_neural_z = np.stack(cal_neural_z_seeds, axis=2)

    other_z_cal = focus._other_logits(shared, raw_cal, hist_cal, best_of_cal)
    cal_calibration = focus._calibrate(
        backend,
        cal_neural_z,
        other_z_cal,
        raw_cal["gates"],
        frame.iloc[cal_ix],
        data["wins_a"][cal_ix],
        data["wins_b"][cal_ix],
    )

    from src.models.a0_target_experiment import _predict_mixture
    cal_mix_full, _, _ = _predict_mixture(
        backend,
        cal_neural_z,
        other_z_cal,
        cal_calibration["slopes"],
        raw_cal["gates"],
        best_of_cal,
        cal_calibration,
    )
    p_cand_cal = cal_mix_full[:, 0]
    ll_cand_cal = float(-np.mean(y_cal * np.log(p_cand_cal) + (1.0 - y_cal) * np.log(1.0 - p_cand_cal)))
    print(f"Candidate Standalone LogLoss on CAL: {ll_cand_cal:.6f} vs C0: {ll_c0_cal:.6f} (delta = {ll_cand_cal - ll_c0_cal:+.6f})")

    # Fit CAL blending weight w in [0, 1] strictly on CAL
    def blend_objective(w: float) -> float:
        z_blend = (1.0 - w) * logit(p_c0_cal) + w * logit(p_cand_cal)
        p_blend = expit(z_blend)
        return float(-np.mean(y_cal * np.log(p_blend) + (1.0 - y_cal) * np.log(1.0 - p_blend)))

    opt_res = minimize_scalar(blend_objective, bounds=(0.0, 1.0), method="bounded")
    optimal_w = float(opt_res.x)
    ll_blend_cal = float(opt_res.fun)
    print(f"Optimal CAL blend weight w = {optimal_w:.4f} (CAL blended LL = {ll_blend_cal:.6f})")

    # 6. Evaluate on ASSESS (N=3,103)
    print("\n[Step 6] Evaluating on canonical 2026 ASSESS partition (N=3,103)...")
    assess_neural_z_seeds = [
        compute_neural_logits_for_network(net, c0_models[0], data, pool_features, assess_ix)
        for net in candidate_networks
    ]
    # Stack across seeds -> (N, 3_modes, 3_seeds)
    assess_neural_z = np.stack(assess_neural_z_seeds, axis=2)

    other_z_assess = focus._other_logits(shared, raw_assess, hist_assess, best_of_assess)
    assess_mix_full, _, _ = _predict_mixture(
        backend,
        assess_neural_z,
        other_z_assess,
        cal_calibration["slopes"],
        raw_assess["gates"],
        best_of_assess,
        cal_calibration,
    )
    p_cand_standalone = assess_mix_full[:, 0]

    # Blended predictions on ASSESS using optimal_w fitted strictly on CAL
    z_assess_blend = (1.0 - optimal_w) * logit(p_c0_full) + optimal_w * logit(p_cand_standalone)
    p_cand_blended = expit(z_assess_blend)

    # 7. Slices identification
    assess_frame = frame.iloc[assess_ix].reset_index(drop=True)
    bo_vals = assess_frame["best_of"].to_numpy(dtype=int)
    tnames = assess_frame["tournament_name"]

    # Major and LPL
    c_tiers = [classify_competition(name, pd.Timestamp(day).date()).tier.value
               for name, day in zip(tnames, assess_frame["date"])]
    is_major = (np.array(c_tiers) == "major") & (tnames != "KeSPA Cup 2026")
    is_lpl = tnames.str.startswith("LPL ").to_numpy()

    # Unseen players
    capacity_preds = pd.read_parquet(PROJECT_ROOT / "data/07_model_output/a0_capacity/native_capacity_20261004_01/predictions.parquet")
    is_unseen = (capacity_preds["unknown_players"] > 0).to_numpy()

    # OOV series: contains at least one champion not seen in TRAIN
    assess_champs = data["history"]["champions"][assess_ix]
    assess_mask = data["history"]["mask"][assess_ix]
    is_valid_map = (assess_mask == 1) & (assess_champs > 0)
    seen_train_set = set(np.unique(data["history"]["champions"][train_ix][(data["history"]["mask"][train_ix] == 1) & (data["history"]["champions"][train_ix] > 0)]))
    oov_maps = is_valid_map & (~np.isin(assess_champs, list(seen_train_set)))
    is_oov_series = oov_maps.any(axis=(1, 2, 3))

    slice_definitions = {
        "all": np.ones(len(assess_frame), dtype=bool),
        "bo1": (bo_vals == 1),
        "bo3": (bo_vals == 3),
        "bo5": (bo_vals == 5),
        "major": is_major,
        "lpl": is_lpl,
        "unseen_players": is_unseen,
        "oov_series": is_oov_series,
    }

    print("\nSlice counts verification:")
    for name, mask_s in slice_definitions.items():
        print(f"  - {name:15s}: N = {mask_s.sum():,}")

    assert slice_definitions["bo1"].sum() == 745, f"Expected 745 BO1, got {slice_definitions['bo1'].sum()}"
    assert slice_definitions["bo3"].sum() == 1873, f"Expected 1,873 BO3, got {slice_definitions['bo3'].sum()}"
    assert slice_definitions["bo5"].sum() == 485, f"Expected 485 BO5, got {slice_definitions['bo5'].sum()}"
    assert slice_definitions["major"].sum() == 692, f"Expected 692 Major, got {slice_definitions['major'].sum()}"
    assert slice_definitions["lpl"].sum() == 242, f"Expected 242 LPL, got {slice_definitions['lpl'].sum()}"
    assert slice_definitions["unseen_players"].sum() == 334, f"Expected 334 Unseen, got {slice_definitions['unseen_players'].sum()}"
    assert slice_definitions["oov_series"].sum() == 2932, f"Expected 2,932 OOV, got {slice_definitions['oov_series'].sum()}"

    # 8. Compute complete slice metrics and contrasts
    print("\n[Step 7] Computing metrics, contrasts, and paired monthly bootstrap CIs...")
    slice_results = {}
    for name, s_mask in slice_definitions.items():
        y_s = y_assess[s_mask]
        p_c0_s = p_c0_full[s_mask]
        p_stand_s = p_cand_standalone[s_mask]
        p_blend_s = p_cand_blended[s_mask]
        months_s = [months_assess[i] for i, val in enumerate(s_mask) if val]

        res_c0 = compute_metrics(y_s, p_c0_s, p_c0_s, months_s)
        res_stand = compute_metrics(y_s, p_stand_s, p_c0_s, months_s)
        res_blend = compute_metrics(y_s, p_blend_s, p_c0_s, months_s)

        slice_results[name] = {
            "n": int(s_mask.sum()),
            "c0": res_c0,
            "candidate_standalone": res_stand,
            "candidate_blended": res_blend,
        }
        stand_mw = res_stand["delta_vs_c0"]["delta"]
        stand_ci = [res_stand["delta_vs_c0"]["ci_lower"], res_stand["delta_vs_c0"]["ci_upper"]]
        blend_mw = res_blend["delta_vs_c0"]["delta"]
        blend_ci = [res_blend["delta_vs_c0"]["ci_lower"], res_blend["delta_vs_c0"]["ci_upper"]]
        print(f"Slice: {name:15s} (N={s_mask.sum():4d}) | Standalone ΔLL: {stand_mw:+.6f} [{stand_ci[0]:+.6f}, {stand_ci[1]:+.6f}] | Blended ΔLL: {blend_mw:+.6f} [{blend_ci[0]:+.6f}, {blend_ci[1]:+.6f}]")

    # 9. Verify non-collapse on OOV series
    oov_res = slice_results["oov_series"]
    oov_delta = oov_res["candidate_standalone"]["delta_vs_c0"]["delta"]
    print(f"\n[Verification] OOV-series (N=2,932) Standalone ΔLL vs C0: {oov_delta:+.6f}")
    assert oov_delta < 0.010872, f"OOV series suffered collapse: {oov_delta} vs 0.010872"
    print("✓ Successfully verified: model does NOT suffer the +0.01087 collapse of champion_history_20261007_01!")

    # 10. Save results
    print("\n[Step 8] Saving JSON summary and parquet predictions...")
    summary_report = {
        "schema": "track3_champion_archetypes_v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Continuous Champion Archetype Embeddings + 15% UNK Token Dropout (hist.champions)",
        "training": {
            "train_rows": len(train_ix),
            "cal_rows": len(cal_ix),
            "assess_rows": len(assess_ix),
            "seeds": list(SEEDS),
            "epochs": 8,
            "batch_size": 256,
            "token_dropout_rate": 0.15,
            "archetype_dims": 18,
            "discrete_embedding_dims": 8,
            "pool_feature_dims": 7,
            "cal_blend_weight": optimal_w,
            "candidate_rhos": candidate_rhos,
        },
        "archetype_metadata": arch_meta,
        "slices": slice_results,
    }

    json_path = output_dir / "track3_champion_archetypes.json"
    with open(json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved metrics: {json_path} ({json_path.stat().st_size:,} bytes)")

    pred_df = pd.DataFrame(
        {
            "golgg_match_id": assess_frame["golgg_match_id"].astype(str),
            "date": assess_frame["date"].astype(str),
            "best_of": bo_vals,
            "tournament_name": tnames.astype(str),
            "y": y_assess.astype(int),
            "p_c0": p_c0_full,
            "p_candidate_standalone": p_cand_standalone,
            "p_candidate_blended": p_cand_blended,
            "is_bo1": slice_definitions["bo1"],
            "is_bo3": slice_definitions["bo3"],
            "is_bo5": slice_definitions["bo5"],
            "is_major": slice_definitions["major"],
            "is_lpl": slice_definitions["lpl"],
            "is_unseen_players": slice_definitions["unseen_players"],
            "is_oov_series": slice_definitions["oov_series"],
        }
    )

    parquet_path = output_dir / "track3_predictions.parquet"
    pred_df.to_parquet(parquet_path, index=False)
    print(f"Saved predictions: {parquet_path} ({len(pred_df)} rows, {parquet_path.stat().st_size:,} bytes)")
    print("\n=== Track 3 Execution and Verification Complete ===")


if __name__ == "__main__":
    main()
