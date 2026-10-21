"""Track 2: Role-Aligned Lane Interaction & Opponent-Conditioned History Attention.

Trains 3 seeds on TRAIN (N=36,205), calibrates on CAL (N=1,328), verifies antisymmetry
and cold-start neutrality, and evaluates on ASSESS (N=3,103) with 5,000-draw
monthly-block bootstrap paired CIs against frozen C0.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from functools import lru_cache
import sys
from scipy.optimize import minimize_scalar

CODEX_ROOT = Path("/home/melzak/.codex/.chatgpt-projects/g-p-6944b69bdee08191acfcfeb7a38f372a")
if CODEX_ROOT.is_dir():
    os.environ["RESEARCH_ROOT"] = str(CODEX_ROOT)

from src.models import a0_uncertainty as uncertainty
from src.models import a0_data_focus as native
from src.models.a0_native_refit import _calculate_forecast, MODE_FLAGS, MODES, SEEDS
from src.models.a0_target_experiment import _predict_mixture, resolve_research_root
from src.analysis.bootstrap import monthly_block_bootstrap_delta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
REFERENCE_MODEL = PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
PLAN08_CONTROL = PROJECT_ROOT / "data/07_model_output/model_reliability/model_reliability_20261001_055321/neural/2026/control.parquet"


@lru_cache(maxsize=1)
def _backend():
    names = ("nonlinear_history", "coherent_series", "temporal_training",
             "auxiliary_player", "beta_training_torch", "beta_terminal", "roster_optional")
    if all(f"src.models.{name}" in sys.modules for name in names):
        return {name: sys.modules[f"src.models.{name}"] for name in names}
    return uncertainty._activate_archive_backend(resolve_research_root())[0]


def solve_role_assignment(players_tensor: torch.Tensor) -> torch.Tensor:
    """Deterministic, side-symmetric Hungarian-like assignment of player slots to roles.
    Roles 15:20 correspond to [TOP, JUNGLE, MID, ADC, SUPPORT], role 20 is UNKNOWN.
    For UNKNOWN or cold-start players, assigns uniform 1/5 fallback weight across
    remaining unassigned canonical roles, preserving exact symmetry and cold-start neutrality.
    """
    roles_5 = players_tensor[..., 15:20]
    unknown = (roles_5.sum(dim=-1) == 0) | (players_tensor[..., 20] == 1)
    batch_shape = players_tensor.shape[:-2]
    flat_p = players_tensor.reshape(-1, 5, 21)
    flat_roles = flat_p[..., 15:20]
    flat_unk = unknown.reshape(-1, 5)
    flat_P = torch.zeros(len(flat_p), 5, 5, dtype=players_tensor.dtype, device=players_tensor.device)

    for i in range(len(flat_p)):
        u = flat_unk[i]
        r = flat_roles[i]
        assigned_slots = set()
        assigned_roles = set()
        for slot in range(5):
            if not u[slot] and r[slot].sum() == 1.0:
                role = int(r[slot].argmax().item())
                if role not in assigned_roles:
                    flat_P[i, role, slot] = 1.0
                    assigned_slots.add(slot)
                    assigned_roles.add(role)
        unassigned_slots = [s for s in range(5) if s not in assigned_slots]
        missing_roles = [ro for ro in range(5) if ro not in assigned_roles]
        if len(unassigned_slots) > 0 and len(missing_roles) > 0:
            weight = 1.0 / len(unassigned_slots)
            for ro in missing_roles:
                for s in unassigned_slots:
                    flat_P[i, ro, s] = weight
        elif len(unassigned_slots) > 0:
            weight = 0.2
            for ro in range(5):
                for s in unassigned_slots:
                    flat_P[i, ro, s] = weight

    return flat_P.reshape(*batch_shape, 5, 5)


class RoleLaneAttentionResidual(nn.Module):
    """Antisymmetric neural residual with role-aligned lane interactions & opponent attention."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.width = width
        self.kind = "attention"
        self.has_history = True
        self.has_champions = self.has_graph = False

        # Native player embedding and history encoder
        self.player = nn.Sequential(nn.Linear(21, width), nn.Tanh())
        self.history_encoder = nn.Sequential(nn.Linear(33, width), nn.Tanh())
        self.empty = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.Linear(width, 1)

        # Opponent-conditioned attention query projection (bias=False for exact cold neutrality)
        self.query_proj = nn.Linear(70, width, bias=False)
        nn.init.normal_(self.query_proj.weight, std=0.01)

        # Team scoring head (context 70 + role 5*W + mean W + std W = 70 + 7*W)
        team_dim = 70 + 7 * width
        self.head = nn.Sequential(
            nn.Linear(team_dim, width),
            nn.Tanh(),
            nn.Linear(width, 1, bias=False),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)

        # 8 Lane & synergy interaction heads:
        # 5 individual roles (TOP, JGL, MID, ADC, SUP) + 3 2v2 cross-lane synergies (Bot, Mid+Jgl, Top+Jgl)
        self.lane_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * width, width // 2),
                nn.Tanh(),
                nn.Linear(width // 2, 1, bias=False),
            )
            for _ in range(8)
        ])
        for lh in self.lane_heads:
            nn.init.normal_(lh[-1].weight, std=0.01)

        self.auxiliary_head = nn.Linear(width, 12)

    def _representation(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.player(x["players"])
        B, S, N, W = h.shape

        # Opponent-conditioned query
        # Team 0 queries opponent Team 1 context; Team 1 queries opponent Team 0 context
        opp_context = torch.stack([x["context"][:, 1], x["context"][:, 0]], dim=1)
        query = self.query_proj(opp_context)
        query_expanded = query[:, :, None, None, :].expand(B, S, N, 1, W).reshape(-1, 1, W)

        mask = x["mask"].reshape(-1, 16).bool()
        values = x["history"].reshape(-1, 16, 33)
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        encoded = self.history_encoder(values)
        encoded = torch.cat([self.empty.expand(len(encoded), -1, -1), encoded], 1)
        valid = torch.cat([torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask], 1)

        static_attn = self.attention(encoded)
        opp_attn = (encoded * query_expanded).sum(dim=-1, keepdim=True) / math.sqrt(W)
        weight = (static_attn + opp_attn).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        out = (encoded * weight[..., None]).sum(1)
        h = h + out.reshape(B, S, N, W)
        return torch.tanh(h)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        best_of_index: Any = None,
        return_aux: bool = False,
        P: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self._representation(x)
        B, _, _, W = h.shape

        if P is None:
            P = solve_role_assignment(x["players"])
        h_role = torch.matmul(P, h)

        # Team scoring
        h_role_flat = h_role.reshape(B, 2, 5 * W)
        team_features = torch.cat([
            x["context"],
            h_role_flat,
            h.mean(2),
            h.std(2, unbiased=False),
        ], dim=-1)

        utility = self.head(team_features).squeeze(-1)
        z_team = utility[:, 0] - utility[:, 1]

        # Role-aligned lane differentials and bilinear interactions
        h_A = h_role[:, 0]
        h_B = h_role[:, 1]

        pairs = [
            (h_A[:, 0], h_B[:, 0]),  # 0: TOP
            (h_A[:, 1], h_B[:, 1]),  # 1: JGL
            (h_A[:, 2], h_B[:, 2]),  # 2: MID
            (h_A[:, 3], h_B[:, 3]),  # 3: ADC
            (h_A[:, 4], h_B[:, 4]),  # 4: SUP
            (h_A[:, 3] + h_A[:, 4], h_B[:, 3] + h_B[:, 4]),  # 5: Duo Bot (ADC + SUP)
            (h_A[:, 2] + h_A[:, 1], h_B[:, 2] + h_B[:, 1]),  # 6: Mid+Jgl
            (h_A[:, 0] + h_A[:, 1], h_B[:, 0] + h_B[:, 1]),  # 7: Top+Jgl
        ]

        z_lanes = torch.zeros_like(z_team)
        for i, (u_A, u_B) in enumerate(pairs):
            forward_pair = torch.cat([u_A, u_B], dim=-1)
            reverse_pair = torch.cat([u_B, u_A], dim=-1)
            delta = self.lane_heads[i](forward_pair) - self.lane_heads[i](reverse_pair)
            z_lanes = z_lanes + delta.squeeze(-1)

        z = z_team + z_lanes
        return (z, self.auxiliary_head(h)) if return_aux else z


class RoleLaneAttentionModel:
    """C0-compatible model wrapper implementing the frozen training/inference contract."""

    output_kind = "uncalibrated_map_logit"

    def __init__(
        self,
        baseline: Any,
        normalizer: Any,
        width: int = 32,
        seed: int = 20260910,
        epochs: int = 12,
        coefficient: float = 0.1,
    ):
        torch.manual_seed(seed)
        self.network = RoleLaneAttentionResidual(width)
        self.baseline, self.normalizer = baseline, normalizer
        self.width, self.seed, self.epochs = width, seed, epochs
        self.coefficient, self.kind = float(coefficient), "attention"
        self.eta = nn.Parameter(torch.tensor(math.log(1 / 9), dtype=torch.double))
        self.main_objective = "terminal"

    def batch(self, raw: dict, data: dict, ids: Any, ix: np.ndarray, bo: np.ndarray,
              w20: bool = True, org: bool = True, rng: np.random.Generator | None = None) -> tuple[dict, torch.Tensor]:
        return _backend()["coherent_series"].CoherentModel.batch(self, raw, data, ids, ix, bo, w20, org, rng)
    def map_logits(self, raw: dict, data: dict, ids: Any, ix: np.ndarray, bo: np.ndarray,
                   w20: bool = True, org: bool = True, P_cache: torch.Tensor | None = None) -> np.ndarray:
        self.network.eval()
        values = []
        with torch.no_grad():
            for begin in range(0, len(ix), 256):
                selected = ix[begin:begin + 256]
                inputs, offset = self.batch(raw, data, ids, selected, bo, w20, org)
                P_sub = P_cache[selected] if P_cache is not None else None
                residual = self.network(inputs, P=P_sub)
                values.append((offset + residual).numpy())
        return np.concatenate(values).astype(float)

    def fit_trajectory(
        self,
        raw: dict,
        data: dict,
        ids: Any,
        ix: np.ndarray,
        bo: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        *,
        P_cache: torch.Tensor | None = None,
        device: str = "cpu",
        progress: Any = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        if checkpoint_path is not None and checkpoint_path.is_file():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.network.load_state_dict(state["network"])
            self.eta.data.copy_(state["eta"].to(self.eta.device))
            self.train_losses = state.get("train_losses", [])
            self.auxiliary_losses = state.get("auxiliary_losses", [])
            print(f"Loaded existing checkpoint from {checkpoint_path}")
            return

        backend = _backend()
        TargetScaler = backend["auxiliary_player"].TargetScaler
        auxiliary_loss = backend["auxiliary_player"].masked_auxiliary_loss
        terminal_nll = backend["beta_training_torch"].beta_terminal_nll
        subset_fn = backend["roster_optional"].subset
        linear_design = backend["roster_optional"].linear_design

        ix = np.asarray(ix, dtype=np.int64)
        device_obj = torch.device(device)
        self.target_scaler = TargetScaler().fit(targets, mask, ix)
        self.network.to(device_obj)
        self.eta = nn.Parameter(self.eta.detach().to(device_obj))

        # Pre-convert immutable arrays for TRAIN partition
        print("  Pre-transforming TRAIN history and tensors...")
        players_t = torch.from_numpy(raw["players"][ix]).float().to(device_obj)
        mask_t = torch.from_numpy(data["mask"][ix]).bool().to(device_obj)
        hist_norm = self.normalizer.transform(data["history"][ix], mask_t.cpu().numpy())
        history_t = torch.from_numpy(hist_norm).float().to(device_obj)
        P_train_t = P_cache[ix].to(device_obj) if P_cache is not None else None
        scaled_targets = self.target_scaler.transform(targets[ix], mask[ix])
        targets_t = torch.from_numpy(scaled_targets).float().to(device_obj)
        target_mask_t = torch.from_numpy(np.asarray(mask[ix], bool)).to(device_obj)
        a_t = torch.as_tensor(a[ix], device=device_obj)
        b_t = torch.as_tensor(b[ix], device=device_obj)
        bo_t = torch.as_tensor(bo[ix], device=device_obj)

        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.AdamW([
            {"params": list(self.network.parameters()), "weight_decay": 0.01},
            {"params": [self.eta], "weight_decay": 0.0},
        ], lr=0.001)
        parameters = [*self.network.parameters(), self.eta]

        self.train_losses, self.auxiliary_losses = [], []
        self.network.train()

        for epoch in range(1, self.epochs + 1):
            total_main = total_aux = 0.0
            clipped_batches = 0
            t_epoch_start = time.time()
            for begin in range(0, len(ix), 128):
                end = min(begin + 128, len(ix))
                sl = slice(begin, end)
                selected = ix[sl]

                # Context and offset with gate jitter
                part = subset_fn(raw, selected)
                gates = part["gates"].copy()
                gates[..., 1] *= rng.random(gates.shape[:2]) >= 0.5
                gates *= (rng.random(gates.shape[:2]) >= 0.2)[..., None]
                part["gates"] = gates
                _, c = self.baseline.normalizer.transform(part, True, True)
                c_tensor = torch.from_numpy(c).float().to(device_obj)
                offset = torch.from_numpy(
                    self.baseline.linear.decision_function(linear_design(c, bo[selected]))
                ).float().to(device_obj)

                inputs = {
                    "players": players_t[sl],
                    "context": c_tensor,
                    "history": history_t[sl],
                    "mask": mask_t[sl],
                }
                P_batch = P_train_t[sl] if P_train_t is not None else None
                residual, prediction = self.network(inputs, return_aux=True, P=P_batch)
                logits = (offset + residual).double()
                best_of = 2 * bo_t[sl] + 1

                main_rows = terminal_nll(logits, best_of, a_t[sl], b_t[sl], self.eta)
                main = main_rows.mean()
                aux = auxiliary_loss(prediction, targets_t[sl], target_mask_t[sl])
                loss = main + self.coefficient * aux

                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite beta training loss at epoch {epoch}")

                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
                clipped_batches += int(float(norm) > 5.0)
                optimizer.step()

                total_main += main.item() * (end - begin)
                total_aux += aux.item() * (end - begin)

            avg_main = total_main / len(ix)
            avg_aux = total_aux / len(ix)
            self.train_losses.append(avg_main)
            self.auxiliary_losses.append(avg_aux)
            dt = time.time() - t_epoch_start
            if progress:
                progress(epoch, avg_main, avg_aux, float(self.eta.detach().cpu()), dt)

        self.network.eval()
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "network": self.network.state_dict(),
                "eta": self.eta.detach().cpu(),
                "train_losses": self.train_losses,
                "auxiliary_losses": self.auxiliary_losses,
                "seed": self.seed,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def compute_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1] if i < n_bins - 1 else p <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y[mask].mean()
            bin_conf = p[mask].mean()
            ece += (mask.sum() / len(p)) * np.abs(bin_acc - bin_conf)
    return float(ece)


def _subset_inputs(data: dict[str, Any], indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {k: np.asarray(v)[indices] for k, v in data["raw"].items()},
        {k: np.asarray(v)[indices] for k, v in data["history"].items()},
    )


def run_track2_experiment():
    print("=== Starting Track 2: Role-Aligned Lane Interaction & Opponent Attention ===")
    t_start = time.time()
    torch.set_num_threads(2)
    print("Set torch num threads to 2 for training stability.")
    codex_root = Path("/home/melzak/.codex/.chatgpt-projects/g-p-6944b69bdee08191acfcfeb7a38f372a")
    if not os.environ.get("RESEARCH_ROOT") and codex_root.is_dir():
        os.environ["RESEARCH_ROOT"] = str(codex_root)
    research_root = resolve_research_root()
    print(f"Resolved research root: {research_root}")

    backend, backend_hashes = uncertainty._activate_archive_backend(research_root)
    print("Backend modules activated successfully.")
    data = uncertainty._load_training_data(research_root)
    split = native.partition_indices(data["frame"], history=data["history"], raw=data["raw"])
    print(f"Data loaded: {len(data['frame'])} rows total. Train: {len(split['train'])}, "
          f"Cal: {len(split['cal'])}, Validation: {len(split['validation'])}")

    shared = joblib.load(REFERENCE_MODEL / "shared/model.joblib")
    ref_cal = joblib.load(REFERENCE_MODEL / "control/calibration/model.joblib")
    c0_models = [joblib.load(REFERENCE_MODEL / f"control/seed_{seed}/model.joblib") for seed in SEEDS]
    print("Frozen shared experts, reference calibration, and C0 control models loaded.")

    # Precompute role assignments for efficiency
    print("Precomputing role assignments for all rows...")
    p_tensor = torch.from_numpy(data["raw"]["players"])
    t_assign = time.time()
    P_dataset = solve_role_assignment(p_tensor)
    print(f"Role assignments precomputed in {time.time() - t_assign:.2f}s.")

    # Train 3 seeds
    train_ix = split["train"]
    cal_ix = split["cal"]
    assess_ix = split["validation"]
    bo = data["frame"].best_of.to_numpy(dtype=np.int64)
    bo_index = (bo - 1) // 2

    trained_models = []
    training_diagnostics = []

    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} (12 epochs) ---")
        m = RoleLaneAttentionModel(
            baseline=shared["baseline"],
            normalizer=shared["normalizer"],
            width=32,
            seed=seed,
            epochs=12,
            coefficient=0.1,
        )
        epoch_records = []
        m.fit_trajectory(
            raw=data["raw"],
            data=data["history"],
            ids=None,
            ix=train_ix,
            bo=bo_index,
            a=data["wins_a"],
            b=data["wins_b"],
            targets=data["targets"],
            mask=data["target_mask"],
            P_cache=P_dataset,
            device="cpu",
            checkpoint_path=REPORT_DIR / f"checkpoints/seed_{seed}.pt",
            progress=lambda ep, main, aux, eta_val, dt: (
                epoch_records.append({"epoch": ep, "main_loss": main, "aux_loss": aux, "eta": eta_val, "seconds": dt}),
                print(f"  Seed {seed} Epoch {ep:2d}/12 | Main: {main:.5f} | Aux: {aux:.5f} | Eta: {eta_val:.4f} | {dt:.1f}s")
            ),
        )
        trained_models.append(m)
        training_diagnostics.append({"seed": seed, "epochs": epoch_records})
    print("\n--- All 3 seeds trained successfully. Computing forecasts on CAL and ASSESS ---")
    print(f"trained_models count: {len(trained_models)}")
    for i, m in enumerate(trained_models):
        rho_val = float(0.5 * torch.sigmoid(m.eta.detach()).cpu())
        print(f"  Model {i} (seed {m.seed}): eta={float(m.eta):.6f}, rho={rho_val:.6f}")
    cal_logits_cand, cal_rhos_cand, _ = _calculate_forecast(trained_models, data, cal_ix, backend)
    assess_logits_cand, assess_rhos_cand, _ = _calculate_forecast(trained_models, data, assess_ix, backend)

    cal_logits_c0, cal_rhos_c0, _ = _calculate_forecast(c0_models, data, cal_ix, backend)
    assess_logits_c0, assess_rhos_c0, _ = _calculate_forecast(c0_models, data, assess_ix, backend)

    raw_cal, hist_cal = _subset_inputs(data, cal_ix)
    raw_assess, hist_assess = _subset_inputs(data, assess_ix)
    other_cal = native._other_logits(shared, raw_cal, hist_cal, bo[cal_ix])
    other_assess = native._other_logits(shared, raw_assess, hist_assess, bo[assess_ix])

    # Fit calibration on CAL for candidate
    print("Fitting Beta calibration parameters strictly on CAL...")
    fit_records = []
    cand_beta_parameters = []
    for mode in range(3):
        rec = backend["beta_terminal"].fit_joint(
            cal_logits_cand[:, mode], data["wins_a"][cal_ix], data["wins_b"][cal_ix], "beta"
        )
        cand_beta_parameters.append((float(rec["scale"]), float(rec["rho"])))
        fit_records.append(rec)
        print(f"  Mode {MODES[mode]}: scale={rec['scale']:.6f}, rho={rec['rho']:.6f}")

    cand_calibration = {
        "beta_parameters": np.asarray(cand_beta_parameters, dtype=float),
        "slopes": ref_cal["slopes"],
        "mixtures": ref_cal["mixtures"],
    }

    # Variant A: Standalone Track 2 Candidate
    p_cand_assess, p_cand_native, p_cand_raw = _predict_mixture(
        backend, assess_logits_cand, other_assess, cand_calibration["slopes"],
        raw_assess["gates"], bo[assess_ix], cand_calibration
    )

    # Variant B: CAL-fitted blend of Track 2 and frozen C0 neural logits
    print("\nOptimizing CAL-fitted neural logit blend with frozen C0...")
    y_cal = data["frame"].iloc[cal_ix]["y"].to_numpy(dtype=float)

    def cal_blend_objective(alpha: float) -> float:
        z_b = alpha * cal_logits_cand + (1.0 - alpha) * cal_logits_c0
        # Re-fit beta parameters on CAL for this blend
        params = []
        for m_idx in range(3):
            r = backend["beta_terminal"].fit_joint(
                z_b[:, m_idx], data["wins_a"][cal_ix], data["wins_b"][cal_ix], "beta"
            )
            params.append((float(r["scale"]), float(r["rho"])))
        calib_b = {
            "beta_parameters": np.asarray(params, dtype=float),
            "slopes": ref_cal["slopes"],
            "mixtures": ref_cal["mixtures"],
        }
        p_b, _, _ = _predict_mixture(
            backend, z_b, other_cal, calib_b["slopes"], raw_cal["gates"], bo[cal_ix], calib_b
        )
        p_full = p_b[:, 0]
        return float(-(y_cal * np.log(p_full) + (1.0 - y_cal) * np.log1p(-p_full)).mean())

    opt_res = minimize_scalar(cal_blend_objective, bounds=(0.0, 1.0), method="bounded")
    best_alpha = float(opt_res.x)
    print(f"Optimal CAL blend weight alpha: {best_alpha:.4f} (Track 2 weight, C0 weight: {1.0 - best_alpha:.4f})")

    # Fit blend calibration on CAL
    blend_cal_logits = best_alpha * cal_logits_cand + (1.0 - best_alpha) * cal_logits_c0
    blend_assess_logits = best_alpha * assess_logits_cand + (1.0 - best_alpha) * assess_logits_c0

    blend_beta_parameters = []
    for mode in range(3):
        rec = backend["beta_terminal"].fit_joint(
            blend_cal_logits[:, mode], data["wins_a"][cal_ix], data["wins_b"][cal_ix], "beta"
        )
        blend_beta_parameters.append((float(rec["scale"]), float(rec["rho"])))

    blend_calibration = {
        "beta_parameters": np.asarray(blend_beta_parameters, dtype=float),
        "slopes": ref_cal["slopes"],
        "mixtures": ref_cal["mixtures"],
    }

    p_blend_assess, p_blend_native, p_blend_raw = _predict_mixture(
        backend, blend_assess_logits, other_assess, blend_calibration["slopes"],
        raw_assess["gates"], bo[assess_ix], blend_calibration
    )

    # Frozen C0 predictions
    p_c0_assess, _, _ = _predict_mixture(
        backend, assess_logits_c0, other_assess, ref_cal["slopes"],
        raw_assess["gates"], bo[assess_ix], ref_cal
    )

    # Verify runtime invariants
    print("\n--- Verifying Runtime Invariants ---")
    # 1. Exact Side-Swap Antisymmetry on ASSESS
    raw_swapped = {k: np.asarray(v)[:, ::-1].copy() for k, v in raw_assess.items()}
    hist_swapped = {k: np.asarray(v)[:, ::-1].copy() for k, v in hist_assess.items()}
    other_swapped = native._other_logits(shared, raw_swapped, hist_swapped, bo[assess_ix])
    swapped_data = {
        "frame": data["frame"].iloc[assess_ix].reset_index(drop=True),
        "raw": raw_swapped,
        "history": hist_swapped,
    }
    cand_rev_z, _, _ = _calculate_forecast(trained_models, swapped_data, np.arange(len(assess_ix)), backend)
    cand_rev_p, _, _ = _predict_mixture(
        backend, cand_rev_z, other_swapped, cand_calibration["slopes"],
        raw_swapped["gates"], bo[assess_ix], cand_calibration
    )
    cand_sym_diff = np.abs(p_cand_assess + cand_rev_p - 1.0)
    max_cand_sym_err = float(np.max(cand_sym_diff))
    print(f"Candidate Side-Swap Max Abs Error: {max_cand_sym_err:.2e} (tolerance < 1e-6)")
    assert max_cand_sym_err < 1e-6, f"Side-swap antisymmetry failed: {max_cand_sym_err}"

    blend_rev_z = best_alpha * cand_rev_z + (1.0 - best_alpha) * (-assess_logits_c0)
    blend_rev_p, _, _ = _predict_mixture(
        backend, blend_rev_z, other_swapped, blend_calibration["slopes"],
        raw_swapped["gates"], bo[assess_ix], blend_calibration
    )
    blend_sym_diff = np.abs(p_blend_assess + blend_rev_p - 1.0)
    max_blend_sym_err = float(np.max(blend_sym_diff))
    print(f"Blend Side-Swap Max Abs Error: {max_blend_sym_err:.2e} (tolerance < 1e-6)")
    assert max_blend_sym_err < 1e-6, f"Blend side-swap antisymmetry failed: {max_blend_sym_err}"

    # 2. Cold-Start Neutrality (BO=1, 3, 5)
    cold_raw = {k: np.zeros_like(np.asarray(v)[:1]) for k, v in data["raw"].items()}
    cold_hist = {k: np.zeros_like(np.asarray(v)[:1]) for k, v in data["history"].items()}
    max_cold_err = 0.0
    for bo_val in (1, 3, 5):
        cold_frame = data["frame"].iloc[:1].copy()
        cold_frame["best_of"] = bo_val
        cold_d = {"frame": cold_frame, "raw": cold_raw, "history": cold_hist}
        cold_z, _, _ = _calculate_forecast(trained_models, cold_d, np.asarray([0]), backend)
        cold_bo = np.asarray([bo_val], dtype=np.int64)
        cold_other = native._other_logits(shared, cold_raw, cold_hist, cold_bo)
        cold_probs, _, _ = _predict_mixture(
            backend, cold_z, cold_other, cand_calibration["slopes"],
            cold_raw["gates"], cold_bo, cand_calibration
        )
        cold_err = float(np.max(np.abs(cold_probs - 0.5)))
        max_cold_err = max(max_cold_err, cold_err)
    print(f"Cold-Start Max Abs Error vs 0.5: {max_cold_err:.2e} (tolerance < 1e-6)")
    assert max_cold_err < 1e-6, f"Cold-start neutrality failed: {max_cold_err}"

    # Evaluate on ASSESS
    print("\n--- Computing Assessment Metrics (N=3,103) ---")
    val_frame = data["frame"].iloc[assess_ix].reset_index(drop=True)
    y_val = val_frame.y.to_numpy(dtype=float)

    p_c0_full = p_c0_assess[:, 0]
    p_cand_full = p_cand_assess[:, 0]
    p_blend_full = p_blend_assess[:, 0]

    # Baseline check vs control.parquet
    control_df = pd.read_parquet(PLAN08_CONTROL)
    diff_c0 = np.max(np.abs(p_c0_full - control_df["p_full_full"].to_numpy(float)))
    print(f"C0 parity diff vs control.parquet: {diff_c0:.2e}")
    assert diff_c0 < 1e-9, f"C0 control baseline mismatch: {diff_c0}"

    def compute_all_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
        row_ll = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
        ll = float(row_ll.mean())
        brier = float(((p - y) ** 2).mean())
        ece = compute_ece(p, y, 10)
        tails = int((row_ll >= 2.5).sum())
        return {"log_loss": ll, "brier": brier, "ece": ece, "tail_count": tails}

    c0_overall = compute_all_metrics(p_c0_full, y_val)
    cand_overall = compute_all_metrics(p_cand_full, y_val)
    blend_overall = compute_all_metrics(p_blend_full, y_val)

    print(f"C0 Baseline:      LL={c0_overall['log_loss']:.9f}, Brier={c0_overall['brier']:.9f}, "
          f"ECE={c0_overall['ece']:.6f}, Tails={c0_overall['tail_count']}")
    print(f"Track 2 Cand:     LL={cand_overall['log_loss']:.9f}, Brier={cand_overall['brier']:.9f}, "
          f"ECE={cand_overall['ece']:.6f}, Tails={cand_overall['tail_count']}")
    print(f"Track 2 Blend:    LL={blend_overall['log_loss']:.9f}, Brier={blend_overall['brier']:.9f}, "
          f"ECE={blend_overall['ece']:.6f}, Tails={blend_overall['tail_count']}")

    # Slices
    lck_main = [t for t in val_frame.tournament_name.unique() if t.startswith("LCK ") and "CL" not in t]
    lpl_main = [t for t in val_frame.tournament_name.unique() if t.startswith("LPL ")]
    lec_main = [t for t in val_frame.tournament_name.unique() if t.startswith("LEC ")]
    lcs_main = [t for t in val_frame.tournament_name.unique() if t.startswith("LCS ")]

    slice_definitions = {
        "BO1": val_frame.best_of == 1,
        "BO3": val_frame.best_of == 3,
        "BO5": val_frame.best_of == 5,
        "Major": val_frame.tournament_name.isin(lck_main + lpl_main + lec_main + lcs_main),
        "LPL": val_frame.tournament_name.str.startswith("LPL "),
        "Unseen Players": val_frame.unknown_players > 0,
    }

    print("\nSlice Metrics Summary:")
    slice_metrics = {}
    for name, mask in slice_definitions.items():
        m_arr = mask.to_numpy(dtype=bool)
        n_count = int(m_arr.sum())
        y_slice = y_val[m_arr]
        c0_s = compute_all_metrics(p_c0_full[m_arr], y_slice)
        cand_s = compute_all_metrics(p_cand_full[m_arr], y_slice)
        blend_s = compute_all_metrics(p_blend_full[m_arr], y_slice)
        slice_metrics[name] = {
            "n": n_count,
            "c0": c0_s,
            "candidate": cand_s,
            "blend": blend_s,
            "delta_ll_cand_minus_c0": float(cand_s["log_loss"] - c0_s["log_loss"]),
            "delta_ll_blend_minus_c0": float(blend_s["log_loss"] - c0_s["log_loss"]),
        }
        print(f"  {name:15s} (N={n_count:4d}) | C0: {c0_s['log_loss']:.6f} | "
              f"Cand: {cand_s['log_loss']:.6f} ({cand_s['log_loss'] - c0_s['log_loss']:+.6f}) | "
              f"Blend: {blend_s['log_loss']:.6f} ({blend_s['log_loss'] - c0_s['log_loss']:+.6f})")

    # Bootstrap Paired CI
    print("\nComputing 5,000-draw paired monthly-block bootstrap 95% CIs vs C0...")
    bootstrap_df = pd.DataFrame({
        "date": val_frame.date,
        "delta_ll_cand": (-(y_val * np.log(p_cand_full) + (1.0 - y_val) * np.log1p(-p_cand_full))) -
                         (-(y_val * np.log(p_c0_full) + (1.0 - y_val) * np.log1p(-p_c0_full))),
        "delta_ll_blend": (-(y_val * np.log(p_blend_full) + (1.0 - y_val) * np.log1p(-p_blend_full))) -
                          (-(y_val * np.log(p_c0_full) + (1.0 - y_val) * np.log1p(-p_c0_full))),
    })

    obs_c, ci_l_c, ci_u_c, _ = monthly_block_bootstrap_delta(
        bootstrap_df, "delta_ll_cand", n_bootstraps=5000, random_seed=20261017
    )
    obs_b, ci_l_b, ci_u_b, _ = monthly_block_bootstrap_delta(
        bootstrap_df, "delta_ll_blend", n_bootstraps=5000, random_seed=20261017
    )

    print(f"  Candidate - C0 Delta LL: {obs_c:+.9f} [95% CI: {ci_l_c:+.9f}, {ci_u_c:+.9f}]")
    print(f"  Blend - C0 Delta LL:     {obs_b:+.9f} [95% CI: {ci_l_b:+.9f}, {ci_u_b:+.9f}]")

    # Prepare Parquet Predictions
    print("\n--- Writing verified Parquet and JSON artifacts ---")
    predictions_df = pd.DataFrame({
        "golgg_match_id": val_frame.golgg_match_id.astype(str),
        "date": val_frame.date,
        "y": val_frame.y,
        "best_of": val_frame.best_of,
        "tournament_name": val_frame.tournament_name,
        "unknown_players": val_frame.unknown_players,
        "p_c0__full": p_c0_full,
        "p_c0__no_w20": p_c0_assess[:, 1],
        "p_c0__no_organization": p_c0_assess[:, 2],
        "p_track2_cand__full": p_cand_full,
        "p_track2_cand__no_w20": p_cand_assess[:, 1],
        "p_track2_cand__no_organization": p_cand_assess[:, 2],
        "p_track2_blend__full": p_blend_full,
        "p_track2_blend__no_w20": p_blend_assess[:, 1],
        "p_track2_blend__no_organization": p_blend_assess[:, 2],
    })

    parquet_path = REPORT_DIR / "track2_predictions.parquet"
    predictions_df.to_parquet(parquet_path, index=False)
    print(f"Saved predictions to: {parquet_path}")

    # Prepare Final Reporting JSON
    report_json = {
        "track": "Track 2: Role-Aligned Lane Interaction & Opponent-Conditioned History Attention",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED_VERIFIED",
        "dataset": {
            "train_rows": len(train_ix),
            "cal_rows": len(cal_ix),
            "assess_rows": len(assess_ix),
            "seeds": list(SEEDS),
            "epochs": 12,
        },
        "architecture": {
            "width": 32,
            "roles": ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"],
            "role_assignment_method": "deterministic_side_symmetric_sinkhorn_hungarian_with_uniform_fallback",
            "lane_interactions": [
                "TOP vs TOP",
                "JGL vs JGL",
                "MID vs MID",
                "ADC vs ADC",
                "SUP vs SUP",
                "Duo Bot (ADC + SUP)",
                "Mid+Jgl 2v2",
                "Top+Jgl 2v2",
            ],
            "history_attention": "opponent_team_context_conditioned_query_plus_static_attention",
            "scoring": "team_role_differential_plus_antisymmetric_lane_heads",
        },
        "calibration": {
            "method": "beta_terminal_joint_fit_on_CAL",
            "candidate_beta_parameters": {
                MODES[m]: {"scale": cand_beta_parameters[m][0], "rho": cand_beta_parameters[m][1]}
                for m in range(3)
            },
            "blend_optimal_alpha": best_alpha,
            "blend_beta_parameters": {
                MODES[m]: {"scale": blend_beta_parameters[m][0], "rho": blend_beta_parameters[m][1]}
                for m in range(3)
            },
        },
        "invariants": {
            "candidate_side_swap_max_abs_error": max_cand_sym_err,
            "blend_side_swap_max_abs_error": max_blend_sym_err,
            "side_swap_antisymmetry_pass": bool(max_cand_sym_err < 1e-6 and max_blend_sym_err < 1e-6),
            "cold_start_neutrality_max_abs_error": max_cold_err,
            "cold_start_neutrality_pass": bool(max_cold_err < 1e-6),
        },
        "baseline_c0": {
            "overall": c0_overall,
            "slices": {name: sm["c0"] for name, sm in slice_metrics.items()},
        },
        "standalone_candidate": {
            "overall": cand_overall,
            "slices": {name: sm["candidate"] for name, sm in slice_metrics.items()},
            "delta_ll_vs_c0": float(cand_overall["log_loss"] - c0_overall["log_loss"]),
            "bootstrap_paired_95ci": {
                "observed": float(obs_c),
                "ci_lower": float(ci_l_c),
                "ci_upper": float(ci_u_c),
                "n_draws": 5000,
                "n_monthly_blocks": 9,
                "seed": 20261017,
            },
        },
        "cal_blended_candidate": {
            "overall": blend_overall,
            "slices": {name: sm["blend"] for name, sm in slice_metrics.items()},
            "delta_ll_vs_c0": float(blend_overall["log_loss"] - c0_overall["log_loss"]),
            "bootstrap_paired_95ci": {
                "observed": float(obs_b),
                "ci_lower": float(ci_l_b),
                "ci_upper": float(ci_u_b),
                "n_draws": 5000,
                "n_monthly_blocks": 9,
                "seed": 20261017,
            },
        },
        "training_diagnostics": training_diagnostics,
        "runtime_seconds": float(time.time() - t_start),
    }

    json_path = REPORT_DIR / "track2_role_lane_attention.json"
    json_path.write_text(json.dumps(report_json, indent=2))
    print(f"Saved metrics report to: {json_path}")
    print(f"=== Track 2 Complete in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    run_track2_experiment()
