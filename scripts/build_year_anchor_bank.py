#!/usr/bin/env python3
"""Build the strictly prior 365-day player-history anchor bank."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_c0_multiscale_20token import _summarize_window_maps
from src.analysis.replacement_guards import assert_chronological_order

REPORT_DIR = ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
BANK_DIR = REPORT_DIR / "year_anchor_bank"
DEFAULT_OUTPUT_DIR = REPORT_DIR / "year_anchor_bank_rebuilt"
BANK = "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"


def _build_from_sources(
    meta: pd.DataFrame,
    history: np.ndarray,
    mask: np.ndarray,
    champions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert_chronological_order(meta["date"], label="release_corrected_bank metadata.parquet dates")
    n = len(meta)
    if len(history) != n or len(mask) != n or len(champions) != n:
        raise ValueError("metadata and source arrays must have matching row counts")
    days = pd.to_datetime(meta["date"]).map(lambda d: d.toordinal()).to_numpy(dtype=np.int32)
    rosters = [[list(map(str, r)) for r in meta[col]] for col in ("roster_a", "roster_b")]
    out = np.zeros((n, 2, 5, 1, 33), dtype=np.float16)
    out_mask = np.zeros((n, 2, 5, 1), dtype=np.uint8)
    counts = np.zeros((n, 2, 5, 1), dtype=np.uint16)
    # The supplied feature bank exposes neither exact map IDs nor raw release
    # dates. Signatures therefore approximate identity from its quantized age
    # and map features; causal guarantees apply to row ingestion, not this
    # lossy identity reconstruction.
    player_maps: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    seen_sigs: dict[str, set[tuple]] = defaultdict(set)
    for r in range(n):
        cur = int(days[r])
        for side in range(2):
            for slot, pid in enumerate(rosters[side][r]):
                maps = player_maps[pid]
                seen = seen_sigs[pid]
                for k in range(16):
                    if not mask[r, side, slot, k]:
                        continue
                    vec = history[r, side, slot, k].astype(np.float32)
                    map_day = cur - int(round(np.expm1(float(vec[25]))))
                    sig = (map_day, int(champions[r, side, slot, k]),
                           *(int(round(float(vec[j]) * 10)) for j in range(3)))
                    if sig not in seen:
                        seen.add(sig)
                        maps.append((map_day, vec))
                # Preserve the existing strict (<365) horizon convention.
                maps[:] = [(day, vec) for day, vec in maps if 0 < cur - day < 365]
                if maps:
                    out[r, side, slot, 0] = _summarize_window_maps(maps, cur).astype(np.float16)
                    out_mask[r, side, slot, 0] = 1
                    counts[r, side, slot, 0] = min(len(maps), np.iinfo(np.uint16).max)
    return out, out_mask, counts


def build(research_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load source files and return history, mask and release counts."""
    bank = research_root / BANK
    meta = pd.read_parquet(bank / "metadata.parquet")
    history = np.load(bank / "history.npy", mmap_mode="r")
    mask = np.load(bank / "mask.npy", mmap_mode="r")
    champions = np.load(bank / "champions.npy", mmap_mode="r")
    return _build_from_sources(meta, history, mask, champions)


def verify(research_root: Path) -> None:
    fresh, fresh_mask, fresh_counts = build(research_root)
    hp, mp = BANK_DIR / "year_anchor_history.npy", BANK_DIR / "year_anchor_mask.npy"
    cp = BANK_DIR / "year_map_counts.npy"
    saved = np.load(hp, mmap_mode="r")
    saved_mask = np.load(mp, mmap_mode="r")
    saved_counts = np.load(cp, mmap_mode="r")
    mismatches = []
    for label, actual, expected in (
        ("history", fresh, saved),
        ("mask", fresh_mask, saved_mask),
        ("counts", fresh_counts, saved_counts),
    ):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            mismatches.append(
                f"{label} shape/dtype saved={expected.shape}/{expected.dtype} "
                f"rebuilt={actual.shape}/{actual.dtype}"
            )
            continue
        unequal = actual != expected
        count = int(np.count_nonzero(unequal))
        if count:
            first = tuple(int(i) for i in np.argwhere(unequal)[0])
            mismatches.append(f"{label}: {count} mismatched values; first index {first}")

    bank = research_root / BANK
    meta = pd.read_parquet(bank / "metadata.parquet")
    ids_path = BANK_DIR / "year_anchor_row_ids.npy"
    if ids_path.exists():
        saved_ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
        expected_ids = meta["golgg_match_id"].astype(str).to_numpy(dtype=str)
        if not np.array_equal(saved_ids, expected_ids):
            mismatches.append("row-ID sidecar differs from source metadata")
    else:
        print(f"Historical row-ID sidecar missing: {ids_path}; it was not reconstructed.")

    history = np.array(np.load(bank / "history.npy", mmap_mode="r"), copy=True)
    mask = np.array(np.load(bank / "mask.npy", mmap_mode="r"), copy=True)
    champions = np.array(np.load(bank / "champions.npy", mmap_mode="r"), copy=True)
    split = len(meta) // 2
    if split == 0 or split == len(meta):
        raise AssertionError("future invariance verification requires rows both before and after the split")
    original_history = history[split:].copy()
    original_champions = champions[split:].copy()
    # Preserve masks and age channels: changed rows remain valid source maps.
    history[split:, ..., 0] += np.float16(1)
    champions[split:] += 1
    if np.array_equal(history[split:], original_history) or np.array_equal(champions[split:], original_champions):
        raise AssertionError("future source perturbation did not change source arrays")
    perturbed, perturbed_mask, perturbed_counts = _build_from_sources(meta, history, mask, champions)
    earlier = np.arange(split)
    np.testing.assert_array_equal(fresh_mask[earlier], perturbed_mask[earlier])
    np.testing.assert_array_equal(fresh_counts[earlier], perturbed_counts[earlier])
    np.testing.assert_array_equal(fresh[earlier], perturbed[earlier])
    print(
        "Future-row perturbation invariance passed for the supplied feature bank. "
        "Map identity/release dates are approximated from quantized features; "
        "this does not certify exact raw-map identity."
    )
    if mismatches:
        details = "; ".join(mismatches)
        print(f"Historical artifact mismatch against causal rebuild: {details}")
        raise AssertionError(
            "saved historical bank is not reproduced by the chronological builder; "
            "preserve the artifact and review the reported differences"
        )
    print("Full saved bank and available row-ID sidecar match; saved bank unchanged.")




def _configured_research_root() -> Path:
    value = os.environ.get("ENSEMBLE_RESEARCH_ROOT", "").strip()
    config = ROOT / "data/research_root.txt"
    if not value:
        if not config.exists():
            raise FileNotFoundError(f"research root not supplied and configuration is missing: {config}")
        value = config.read_text().strip()
    configured = Path(value)
    return configured if configured.is_absolute() else ROOT / configured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    research_root = args.research_root if args.research_root is not None else _configured_research_root()
    if args.verify:
        verify(research_root)
        return
    output_dir = args.output_dir.resolve()
    if output_dir == BANK_DIR.resolve() or output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing year anchor bank path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    h, m, counts = build(research_root)
    source_meta = pd.read_parquet(research_root / BANK / "metadata.parquet")
    row_ids = source_meta["golgg_match_id"].astype(str).to_numpy(dtype=str)
    np.save(output_dir / "year_anchor_history.npy", h)
    np.save(output_dir / "year_anchor_mask.npy", m)
    np.save(output_dir / "year_map_counts.npy", counts)
    np.save(output_dir / "year_anchor_row_ids.npy", row_ids, allow_pickle=False)


if __name__ == "__main__":
    main()
