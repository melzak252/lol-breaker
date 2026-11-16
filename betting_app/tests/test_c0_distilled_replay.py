"""Contracts for the frozen C0-D teacher and research replay."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.c0_distilled_replay import (
    build_research_batch,
    load_research_bundle,
    predict_research_batch,
    validate_teacher_probabilities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    PROJECT_ROOT
    / "data/05_model_input/c0_distilled_verification/c0d_repro_20261021_01/protocol.json"
)
ARTIFACT_ROOT = PROJECT_ROOT / "data/08_reporting/c0_upgrade_and_distillation_20261017"
CANONICAL_BANK = (
    PROJECT_ROOT
    / "data/05_model_input/canonical_benchmark/a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
)


def test_teacher_distribution_accepts_training_rows_and_preserves_values() -> None:
    ids = [f"train-{index}" for index in range(36_205)]
    values = np.tile(np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25]), (36_205, 1))

    validated = validate_teacher_probabilities(values, ids)

    assert validated.shape == (36_205, 6)
    np.testing.assert_array_equal(validated, values)


def test_teacher_distribution_accepts_small_valid_fixture() -> None:
    values = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])

    validated = validate_teacher_probabilities(values, ["a", "b"], expected_bins=3)

    np.testing.assert_array_equal(validated, values)


def test_teacher_distribution_rejects_wrong_matrix_shape() -> None:
    values = np.ones((2, 5)) / 5

    with pytest.raises(ValueError):
        validate_teacher_probabilities(values, ["a", "b"])


def test_teacher_distribution_rejects_legacy_sized_vector_with_aligned_fixture_ids() -> (
    None
):
    values = np.full(40_636, 1 / 40_636)
    fixture_ids = [f"fixture-{index}" for index in range(40_636)]

    with pytest.raises(ValueError):
        validate_teacher_probabilities(values, fixture_ids)


def _arrays(batch):
    if isinstance(batch, dict):
        for value in batch.values():
            yield from _arrays(value)
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            yield from _arrays(value)
    elif isinstance(batch, np.ndarray):
        yield batch


def _swap_sides(batch):
    swapped = deepcopy(batch)
    if isinstance(swapped, dict):
        return {key: _swap_sides(value) for key, value in swapped.items()}
    if isinstance(swapped, list):
        return [_swap_sides(value) for value in swapped]
    if isinstance(swapped, tuple):
        return tuple(_swap_sides(value) for value in swapped)
    if isinstance(swapped, np.ndarray) and swapped.ndim >= 2 and swapped.shape[1] == 2:
        return swapped[:, ::-1].copy()
    return swapped


def _cold_start(batch):
    cold = deepcopy(batch)
    if isinstance(cold, dict):
        return {key: _cold_start(value) for key, value in cold.items()}
    if isinstance(cold, list):
        return [_cold_start(value) for value in cold]
    if isinstance(cold, tuple):
        return tuple(_cold_start(value) for value in cold)
    if isinstance(cold, np.ndarray) and np.issubdtype(cold.dtype, np.number):
        return np.zeros_like(cold)
    return cold


def _assert_saved_candidate_replay(
    candidate: str, prediction_file: str, prediction_column: str
) -> None:
    saved = pd.read_parquet(ARTIFACT_ROOT / prediction_file)
    sample = saved.iloc[:24]
    bundle = load_research_bundle(PROTOCOL, candidate=candidate)
    assert set(sample["golgg_match_id"]).issubset(set(bundle["assess_row_ids"]))
    batch = build_research_batch(bundle, sample["golgg_match_id"].tolist())
    best_of = sample["best_of"].to_numpy()
    original = deepcopy(batch)

    # The saved legacy bank has no row-ID sidecar: this checks numerical
    # reproduction under the explicitly unverified positional-lineage assumption,
    # not causal provenance or certification.
    replay = predict_research_batch(bundle, batch, best_of=best_of, mode="full")

    np.testing.assert_allclose(
        replay, sample[prediction_column].to_numpy(), rtol=0, atol=2e-6
    )
    for before, after in zip(_arrays(original), _arrays(batch)):
        np.testing.assert_array_equal(before, after)

    swapped = predict_research_batch(
        bundle, _swap_sides(batch), best_of=best_of, mode="full"
    )
    np.testing.assert_allclose(swapped, 1.0 - replay, rtol=0, atol=2e-6)

    cold = predict_research_batch(
        bundle, _cold_start(batch), best_of=best_of, mode="full"
    )
    np.testing.assert_allclose(cold, 0.5, rtol=0, atol=2e-6)

    projections = [
        predict_research_batch(
            bundle, batch, best_of=np.full(len(sample), format_), mode="full"
        )
        for format_ in (1, 3, 5)
    ]
    for projected in projections:
        assert projected.shape == (len(sample),)
        assert np.isfinite(projected).all()
        assert ((projected >= 0) & (projected <= 1)).all()
    assert any(
        not np.allclose(projections[left], projections[right], rtol=0, atol=1e-9)
        for left, right in ((0, 1), (1, 2), (0, 2))
    )

    singletons = np.concatenate(
        [
            predict_research_batch(
                bundle,
                build_research_batch(bundle, [match_id]),
                best_of=np.array([format_]),
                mode="full",
            )
            for match_id, format_ in zip(
                sample["golgg_match_id"].iloc[:4].tolist(),
                best_of[:4],
            )
        ]
    )
    np.testing.assert_allclose(singletons, replay[:4], rtol=0, atol=2e-6)


@pytest.mark.requires_artifacts(
    PROTOCOL,
    ARTIFACT_ROOT / "year_anchor_bank/year_anchor_history.npy",
    ARTIFACT_ROOT / "year_anchor_bank/year_anchor_mask.npy",
    CANONICAL_BANK / "history.npy",
    CANONICAL_BANK / "mask.npy",
    *[
        CANONICAL_BANK / f"{name}.npy"
        for name in ("players", "core", "team", "w20", "gates", "champions")
    ],
    CANONICAL_BANK / "metadata.parquet",
    *[
        ARTIFACT_ROOT
        / f"models/year_anchor_distilled/seed_{seed}/checkpoint_epoch16.pt"
        for seed in (20260910, 20260911, 20260912)
    ],
    ARTIFACT_ROOT / "year_anchor_distilled.json",
    ARTIFACT_ROOT / "year_anchor_distilled_predictions.parquet",
    ARTIFACT_ROOT / "teacher_probs_B.npy",
)
def test_saved_18t_replay_uses_genuine_weights_and_preserves_symmetries() -> None:
    _assert_saved_candidate_replay(
        "18t", "year_anchor_distilled_predictions.parquet", "p_c0_d_year_core__full"
    )


@pytest.mark.requires_artifacts(
    PROTOCOL,
    ARTIFACT_ROOT / "year_anchor_bank/year_anchor_history.npy",
    ARTIFACT_ROOT / "year_anchor_bank/year_anchor_mask.npy",
    ARTIFACT_ROOT / "year_anchor_bank/half_year_180d_history.npy",
    ARTIFACT_ROOT / "year_anchor_bank/half_year_180d_mask.npy",
    ARTIFACT_ROOT / "year_anchor_bank/quarter_90d_history.npy",
    ARTIFACT_ROOT / "year_anchor_bank/quarter_90d_mask.npy",
    CANONICAL_BANK / "history.npy",
    CANONICAL_BANK / "mask.npy",
    *[
        CANONICAL_BANK / f"{name}.npy"
        for name in ("players", "core", "team", "w20", "gates", "champions")
    ],
    CANONICAL_BANK / "metadata.parquet",
    *[
        ARTIFACT_ROOT
        / f"models/multiscale_20token_distilled/seed_{seed}/checkpoint_epoch16.pt"
        for seed in (20260910, 20260911, 20260912)
    ],
    ARTIFACT_ROOT / "multiscale_20token_distilled.json",
    ARTIFACT_ROOT / "multiscale_20token_predictions.parquet",
    ARTIFACT_ROOT / "teacher_probs_B.npy",
)
def test_saved_20t_replay_uses_genuine_weights_and_preserves_symmetries() -> None:
    _assert_saved_candidate_replay(
        "20t", "multiscale_20token_predictions.parquet", "p_c0_d_20t_core__full"
    )


@pytest.mark.parametrize(
    "row",
    [
        [np.nan, 0.2, 0.2, 0.2, 0.2, 0.2],
        [np.inf, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.01, 0.21, 0.2, 0.2, 0.2, 0.2],
        [1.01, -0.01, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ],
)
def test_teacher_distribution_rejects_invalid_probability_rows(
    row: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_teacher_probabilities(np.array([row]), ["train-row"])


def test_teacher_distribution_rejects_duplicate_training_ids() -> None:
    values = np.full((2, 6), 1 / 6)

    with pytest.raises(ValueError):
        validate_teacher_probabilities(values, ["same-row", "same-row"])


def test_teacher_distribution_rejects_row_id_count_mismatch() -> None:
    values = np.full((2, 6), 1 / 6)

    with pytest.raises(ValueError):
        validate_teacher_probabilities(values, ["only-one-id"])
