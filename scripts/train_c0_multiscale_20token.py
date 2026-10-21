#!/usr/bin/env python3
"""Train and Evaluate the 20-Token Multi-Scale Horizon History Attention Architecture.

Adds each player's strictly causal 90-Day (3-Month / Split) and 180-Day (6-Month / Half-Year)
Anchor Tokens alongside the 365-Day Whole-Year Anchor Token and 16 Recent Maps:
- [empty_token, year_365d_token, half_year_180d_token, quarter_90d_token, map_1, ..., map_16] = 20 tokens
- MultiScale20TokenCommonMapResidual(width=32): 6,702 parameters per seed (6,606 + 3 * 32)
- Normalization: Exact same TRAIN HistoryNormalizer as C0 (z-scored coordinate alignment)
- Distillation: 16 epochs on TRAIN (N=36,205) using Teacher B (Multi-Capacity Super-Teacher)
  with distillation_weight=0.60 (0.40 * Hard_Beta_NLL + 0.60 * Soft_Teacher_KL + 0.10 * Aux_SmoothL1)

Outputs saved to:
- data/08_reporting/c0_upgrade_and_distillation_20261017/multiscale_20token_distilled.json
- data/08_reporting/c0_upgrade_and_distillation_20261017/multiscale_20token_predictions.parquet
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import torch
import torch.nn as nn

torch.set_num_threads(2)
torch.set_num_threads(8)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.metrics import calculate_ece
from src.models import a0_capacity_experiment as cap_exp
from src.models import a0_capacity_runner as runner
from src.models import a0_data_focus as data_focus
from src.models import a0_uncertainty as uncertainty
from scripts.train_c0_year_anchor_distilled import (
    BANK_DIR,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    MODELS_DIR,
    REPORT_DIR,
    SEEDS,
    compute_tail_count,
    compute_unclipped_log_loss,
    evaluate_probability_stream,
    fit_beta_calibration_and_assemble,
    resolve_research_root,
)

OUTPUT_JSON_PATH = REPORT_DIR / "multiscale_20token_distilled.json"
OUTPUT_PARQUET_PATH = REPORT_DIR / "multiscale_20token_predictions.parquet"


def _summarize_window_maps(maps_list: list[tuple[int, np.ndarray]], cur_ord: int) -> np.ndarray:
    """Compute the 33-feature summary token from a list of (map_ord, vec_33) tuples."""
    arr = np.stack([v for _, v in maps_list], axis=0)
    out = np.zeros(33, dtype=np.float32)
    # 0..11: masked mean of each boxscore stat; 12..23: indicator whether stat was observed
    for j in range(12):
        valid_j = arr[:, 12 + j] > 0.5
        if np.any(valid_j):
            out[j] = float(arr[valid_j, j].mean())
            out[12 + j] = 1.0
    # 24: mean win rate
    out[24] = float(arr[:, 24].mean())
    # 25: mean log1p(age_days) at cur_ord
    ages = np.array([math.log1p(max(1, cur_ord - m_ord)) for m_ord, _ in maps_list], dtype=np.float32)
    out[25] = float(ages.mean())
    # 26: mean opponent prior Glicko
    out[26] = float(arr[:, 26].mean())
    # 27..32: mean role distribution
    out[27:33] = arr[:, 27:33].mean(axis=0)
    return out


def build_or_load_multiscale_banks(research_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build or load strictly causal 90d (3-month) and 180d (half-year) anchor banks."""
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    h90_path = BANK_DIR / "quarter_90d_history.npy"
    m90_path = BANK_DIR / "quarter_90d_mask.npy"
    h180_path = BANK_DIR / "half_year_180d_history.npy"
    m180_path = BANK_DIR / "half_year_180d_mask.npy"

    if all(p.is_file() for p in (h90_path, m90_path, h180_path, m180_path)):
        print("Loading cached 90d and 180d anchor banks...")
        return (
            np.load(h180_path, mmap_mode="r"),
            np.load(m180_path, mmap_mode="r"),
            np.load(h90_path, mmap_mode="r"),
            np.load(m90_path, mmap_mode="r"),
        )

    print("Building strictly causal 90-day and 180-day multi-scale anchor banks across 40,636 series...")
    t0 = time.monotonic()
    bank_dir = research_root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    meta = pd.read_parquet(bank_dir / "metadata.parquet")
    hist = np.load(bank_dir / "history.npy", mmap_mode="r")
    mask = np.load(bank_dir / "mask.npy", mmap_mode="r")
    champs = np.load(bank_dir / "champions.npy", mmap_mode="r")

    n_rows = len(meta)
    days = pd.to_datetime(meta["date"]).map(lambda d: d.toordinal()).to_numpy(dtype=np.int32)
    ra_list = [[str(x) for x in r] for r in meta["roster_a"]]
    rb_list = [[str(x) for x in r] for r in meta["roster_b"]]

    h90 = np.zeros((n_rows, 2, 5, 1, 33), dtype=np.float16)
    m90 = np.zeros((n_rows, 2, 5, 1), dtype=np.uint8)
    h180 = np.zeros((n_rows, 2, 5, 1, 33), dtype=np.float16)
    m180 = np.zeros((n_rows, 2, 5, 1), dtype=np.uint8)

    player_maps: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    seen_sigs: dict[str, set[tuple[int, int, int, int, int]]] = defaultdict(set)

    for r in range(n_rows):
        cur_ord = int(days[r])
        for s, rlist in ((0, ra_list[r]), (1, rb_list[r])):
            for j, pid in enumerate(rlist):
                m_row = mask[r, s, j]
                h_row = hist[r, s, j]
                c_row = champs[r, s, j]
                p_sigs = seen_sigs[pid]
                p_list = player_maps[pid]

                # 1. Ingest any newly released prior maps from the 16-map window
                for k in range(16):
                    if m_row[k] == 0:
                        continue
                    vec_f32 = h_row[k].astype(np.float32)
                    age_d = int(round(math.expm1(float(vec_f32[25]))))
                    if age_d > 180:
                        continue
                    map_ord = cur_ord - age_d
                    sig = (
                        map_ord,
                        int(c_row[k]),
                        int(round(float(vec_f32[0]) * 10)),
                        int(round(float(vec_f32[1]) * 10)),
                        int(round(float(vec_f32[2]) * 10)),
                    )
                    if sig not in p_sigs:
                        p_sigs.add(sig)
                        p_list.append((map_ord, vec_f32))

                # 2. Prune maps older than 180 days
                if p_list:
                    p_list[:] = [item for item in p_list if 0 < (cur_ord - item[0]) <= 180]

                # 3. Populate 180d and 90d summary tokens
                if p_list:
                    h180[r, s, j, 0] = _summarize_window_maps(p_list, cur_ord).astype(np.float16)
                    m180[r, s, j, 0] = 1
                    p_90 = [item for item in p_list if (cur_ord - item[0]) <= 90]
                    if p_90:
                        h90[r, s, j, 0] = _summarize_window_maps(p_90, cur_ord).astype(np.float16)
                        m90[r, s, j, 0] = 1

        if (r + 1) % 10000 == 0 or r == n_rows - 1:
            print(f"  Processed {r + 1}/{n_rows} series ({time.monotonic() - t0:.1f}s)...")

    np.save(h90_path, h90)
    np.save(m90_path, m90)
    np.save(h180_path, h180)
    np.save(m180_path, m180)
    print(f"Saved 90d & 180d banks in {time.monotonic() - t0:.1f}s.")
    return h180, m180, h90, m90


class MultiScale20TokenCommonMapResidual(cap_exp.ParametricCommonMapResidual):
    """20-Token Multi-Scale Horizon History Attention Architecture.

    Tokens:
    - Slot 0: self.empty (learnable empty token)
    - Slot 1: 365-day whole-year anchor token + self.year_token_bias
    - Slot 2: 180-day half-year anchor token + self.half_year_token_bias
    - Slot 3: 90-day quarter/split anchor token + self.quarter_token_bias
    - Slots 4..19: 16 recent maps
    Total: 20 tokens in masked self-attention layer.
    """

    def __init__(self, width: int = 32):
        super().__init__(width)
        self.year_token_bias = nn.Parameter(torch.zeros(1, 1, width))
        self.half_year_token_bias = nn.Parameter(torch.zeros(1, 1, width))
        self.quarter_token_bias = nn.Parameter(torch.zeros(1, 1, width))
        expected_params = (2 * width**2 + 142 * width + 13) + 3 * width  # 6605 + 96 = 6701 (+ eta = 6702)
        actual_params = sum(p.numel() for p in self.parameters())
        if actual_params != expected_params:
            raise AssertionError(f"Expected {expected_params} parameters, got {actual_params}")

    def _representation(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.player(x["players"])
        n = h.shape[0]

        # x['mask'] has shape (B, 2, 5, 19) -> flatten to (B*10, 19)
        mask_19 = x["mask"].reshape(-1, 19).bool()
        val_19 = x["history"].reshape(-1, 19, 33)
        val_19 = torch.where(mask_19[..., None], val_19, torch.zeros_like(val_19))

        year_val = val_19[:, 0:1, :]
        half_val = val_19[:, 1:2, :]
        quar_val = val_19[:, 2:3, :]
        recent_val = val_19[:, 3:, :]

        year_mask = mask_19[:, 0:1]
        half_mask = mask_19[:, 1:2]
        quar_mask = mask_19[:, 2:3]
        recent_mask = mask_19[:, 3:]

        year_enc = self.history_encoder(year_val) + self.year_token_bias
        half_enc = self.history_encoder(half_val) + self.half_year_token_bias
        quar_enc = self.history_encoder(quar_val) + self.quarter_token_bias
        recent_enc = self.history_encoder(recent_val)

        empty_token = self.empty.expand(len(val_19), -1, -1)
        encoded_20 = torch.cat([empty_token, year_enc, half_enc, quar_enc, recent_enc], dim=1)

        empty_valid = torch.ones((len(mask_19), 1), dtype=torch.bool, device=mask_19.device)
        valid_20 = torch.cat([empty_valid, year_mask, half_mask, quar_mask, recent_mask], dim=1)

        weight = self.attention(encoded_20).squeeze(-1).masked_fill(~valid_20, -torch.inf).softmax(1)
        out = (encoded_20 * weight[..., None]).sum(1)
        h = h + out.reshape(n, 2, 5, self.width)
        return torch.tanh(h)


class MultiScale20TokenNonlinearHistoryModel(cap_exp.ParametricNonlinearHistoryModel):
    """Container model matching the ParametricNonlinearHistoryModel API for 20 tokens."""

    def __init__(
        self,
        baseline: Any,
        normalizer: Any,
        width: int = 32,
        seed: int = 20260910,
        epochs: int = 16,
        coefficient: float = 0.1,
    ):
        if epochs < 1 or not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("positive epochs and nonnegative auxiliary coefficient required")
        torch.manual_seed(seed)
        self.network = MultiScale20TokenCommonMapResidual(width)
        self.baseline, self.normalizer = baseline, normalizer
        self.width, self.seed, self.epochs = width, seed, epochs
        self.coefficient, self.kind = float(coefficient), "attention"
        self.eta = nn.Parameter(torch.tensor(math.log(1 / 9), dtype=torch.double))
        self.main_objective = "terminal"


def train_or_load_20token_model(
    width: int,
    seed: int,
    epochs: int,
    distillation_weight: float,
    teacher_probs: np.ndarray,
    ckpt_dir: Path,
    data_19: dict[str, Any],
    train_ix: np.ndarray,
    bo_index_train: np.ndarray,
    shared: dict[str, Any],
) -> MultiScale20TokenNonlinearHistoryModel:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_epoch{epochs}.pt"

    if ckpt_path.is_file():
        print(f"Loading existing 20-token checkpoint: {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inst = MultiScale20TokenNonlinearHistoryModel(
            shared["baseline"], shared["normalizer"], width=width, seed=seed, epochs=epochs, coefficient=0.1
        )
        inst.network.load_state_dict(state["network_state_dict"])
        inst.eta = torch.nn.Parameter(state["eta"])
        inst.target_scaler = state["target_scaler"]
        inst.epochs = epochs
        inst.epoch_diagnostics = state.get("epoch_diagnostics", [])
        inst.network.eval()
        return inst

    print(f"Training 20-Token MultiScale model: width={width}, seed={seed}, epochs={epochs}, lambda={distillation_weight}...")
    t0 = time.monotonic()
    model_inst = MultiScale20TokenNonlinearHistoryModel(
        shared["baseline"], shared["normalizer"], width=width, seed=seed, epochs=epochs, coefficient=0.1
    )
    snapshots = model_inst.fit_trajectory(
        data_19["raw"], data_19["history"], None, train_ix, bo_index_train,
        data_19["wins_a"], data_19["wins_b"], data_19["targets"], data_19["target_mask"],
        checkpoint_dir=ckpt_dir, checkpoint_epochs=(epochs,), device="cpu",
        teacher_probabilities=teacher_probs, distillation_weight=distillation_weight,
        progress=lambda ep, loss, aux, rec: (
            print(f"  [20T w{width} s{seed}] epoch {ep}/{epochs}: loss={loss:.5f} aux={aux:.5f} ({time.monotonic()-t0:.1f}s)")
            if ep in (1, epochs // 4, epochs // 2, 3 * epochs // 4, epochs) else None
        ),
    )
    print(f"Finished training 20T w{width} seed {seed} in {time.monotonic() - t0:.1f}s")
    return snapshots[epochs]


def main() -> int:
    t_start = time.monotonic()
    print("=" * 80)
    print("20-TOKEN MULTI-SCALE HORIZON ATTENTION ARCHITECTURE (90d + 180d + 365d + 16m)")
    print("=" * 80)

    research_root = resolve_research_root()
    backend, _ = uncertainty._activate_archive_backend(research_root)
    data = uncertainty._load_training_data(research_root)
    frame = data["frame"]

    from src.models import competition_tiers as tiers
    identities = [
        tiers.classify_competition(name, pd.Timestamp(day).date())
        for name, day in zip(frame.tournament_name, frame.date)
    ]
    frame["competition_tier"] = [identity.tier.value for identity in identities]

    split = runner._partition(frame, smoke=False, history=data["history"], raw=data["raw"])
    train_ix, cal_ix, assess_ix = split["train"], split["cal"], split["validation"]

    shared = joblib.load(PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/shared/model.joblib")
    ref_cal = joblib.load(PROJECT_ROOT / "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/calibration/model.joblib")

    y_hist = np.load(BANK_DIR / "year_anchor_history.npy", mmap_mode="r")
    y_mask = np.load(BANK_DIR / "year_anchor_mask.npy", mmap_mode="r")
    h180, m180, h90, m90 = build_or_load_multiscale_banks(research_root)

    h_19 = np.concatenate([y_hist, h180, h90, data["history"]["history"]], axis=3)
    m_19 = np.concatenate([y_mask, m180, m90, data["history"]["mask"]], axis=3)
    print(f"Assembled 19-map tensor: history={h_19.shape}, mask={m_19.shape}")

    data_19 = {
        "raw": data["raw"],
        "history": {"history": h_19, "mask": m_19, "champions": data["history"].get("champions")},
        "wins_a": data["wins_a"],
        "wins_b": data["wins_b"],
        "targets": data["targets"],
        "target_mask": data["target_mask"],
        "frame": data["frame"],
    }
    # Verify architectural invariants
    torch.manual_seed(20260910)
    test_net = MultiScale20TokenCommonMapResidual(width=32)
    x_cold = {
        "players": torch.zeros(2, 2, 5, 21),
        "context": torch.zeros(2, 2, 70),
        "history": torch.zeros(2, 2, 5, 19, 33),
        "mask": torch.zeros(2, 2, 5, 19, dtype=torch.bool),
    }
    p_cold = float(expit(test_net(x_cold, torch.zeros(2, dtype=torch.long))[0].item()))
    assert abs(p_cold - 0.5) < 1e-6, f"Cold start failed: {p_cold}"
    print(f"Verified 20-Token Invariants: params/seed=6702, cold_start={p_cold:.6f}")

    teacher_probs_B = np.load(REPORT_DIR / "teacher_probs_B.npy")
    p1 = pd.read_parquet(REPORT_DIR / "track1_predictions.parquet")
    p_18t = pd.read_parquet(REPORT_DIR / "year_anchor_distilled.parquet")
    p_c0_full = p1["p_c0__full"].to_numpy(dtype=np.float64)
    p_c0d_18t = p_18t["p_c0_d_year_core__full"].to_numpy(dtype=np.float64)

    assess_frame = frame.iloc[assess_ix].reset_index(drop=True)
    y_assess = assess_frame["y"].to_numpy(dtype=int)
    y_cal = frame.iloc[cal_ix]["y"].to_numpy(dtype=int)
    dates_assess = assess_frame["date"].to_numpy(dtype=str)

    bo_train = frame.best_of.iloc[train_ix].to_numpy(dtype=np.int64)
    bo_index_train = (bo_train - 1) // 2

    ms_dir = MODELS_DIR / "multiscale_20token_distilled"
    models_20t = []
    for seed in SEEDS:
        m = train_or_load_20token_model(
            width=32, seed=seed, epochs=16, distillation_weight=0.60,
            teacher_probs=teacher_probs_B, ckpt_dir=ms_dir / f"seed_{seed}",
            data_19=data_19, train_ix=train_ix, bo_index_train=bo_index_train, shared=shared,
        )
        models_20t.append(m)

    p_20t_full, p_20t_neural, _, cal_summary, cal_20t_full, _ = fit_beta_calibration_and_assemble(
        models_20t, data_19, data, cal_ix, assess_ix, backend, shared, ref_cal
    )

    # Also evaluate 1-seed C0-Slim-20T (seed 20260910)
    p_slim_20t_full, _, _, _, _, _ = fit_beta_calibration_and_assemble(
        [models_20t[0]] * 3, data_19, data, cal_ix, assess_ix, backend, shared, ref_cal
    )

    eval_20t = evaluate_probability_stream(
        y_assess, p_20t_full, dates_assess, assess_frame, p_c0_full, p_c0d_18t
    )
    eval_slim_20t = evaluate_probability_stream(
        y_assess, p_slim_20t_full, dates_assess, assess_frame, p_c0_full, p_c0d_18t
    )

    print("\n=== 20-TOKEN MULTI-SCALE RESULTS ON 2026 ASSESSMENT (N=3,103) ===")
    print(f"17-Token Legacy C0  : LL = {compute_unclipped_log_loss(y_assess, p_c0_full):.6f}")
    print(f"18-Token C0-D (365d): LL = {compute_unclipped_log_loss(y_assess, p_c0d_18t):.6f}")
    print(f"20-Token C0-D-20T   : LL = {eval_20t['unclipped_log_loss']:.6f} | Brier = {eval_20t['brier_score']:.6f} | ECE = {eval_20t['ece_10_bins']*100:.2f}% | Tails = {eval_20t['tail_count_ll_ge_2_5']}")
    print(f"  ΔLL vs 17T C0     : {eval_20t['bootstrap_vs_c0']['observed_delta']:+.6f} (95% CI: [{eval_20t['bootstrap_vs_c0']['ci95_lower']:+.6f}, {eval_20t['bootstrap_vs_c0']['ci95_upper']:+.6f}])")
    print(f"  ΔLL vs 18T C0-D   : {eval_20t['bootstrap_vs_c0_d_core']['observed_delta']:+.6f} (95% CI: [{eval_20t['bootstrap_vs_c0_d_core']['ci95_lower']:+.6f}, {eval_20t['bootstrap_vs_c0_d_core']['ci95_upper']:+.6f}])")
    print(f"20-Token C0-Slim-20T: LL = {eval_slim_20t['unclipped_log_loss']:.6f} | Brier = {eval_slim_20t['brier_score']:.6f} | ECE = {eval_slim_20t['ece_10_bins']*100:.2f}% | Tails = {eval_slim_20t['tail_count_ll_ge_2_5']}")
    for s_name, s_val in eval_20t["slices"].items():
        print(f"  Slice {s_name:<15s}: {s_val}")
    out_payload = {
        "experiment": "c0_20_token_multiscale_distilled",
        "elapsed_seconds": time.monotonic() - t_start,
        "c0_d_20t_core": eval_20t,
        "c0_slim_20t": eval_slim_20t,
        "calibration": cal_summary,
    }
    OUTPUT_JSON_PATH.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    out_df = p_18t.copy()
    out_df["p_c0_d_20t_core__full"] = p_20t_full
    out_df["p_c0_slim_20t__full"] = p_slim_20t_full
    out_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"Saved results to {OUTPUT_JSON_PATH} and {OUTPUT_PARQUET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
