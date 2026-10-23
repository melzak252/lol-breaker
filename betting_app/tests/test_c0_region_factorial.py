from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.c0_region_factorial import (
    assemble_four_configurations,
    paired_monthly_contrast,
    score_probabilities,
    transform_batch_once,
)

from src.analysis.c0_region_factorial import _require_finite_probabilities


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.5, np.nan]),
        np.array([0.5, np.inf]),
        np.array([0.5]),
    ],
)
def test_native_probability_coverage_rejects_missing_or_nonfinite_rows(values):
    with pytest.raises(ValueError, match="fresh20.*2"):
        _require_finite_probabilities(values, 2, "fresh20")


@pytest.mark.parametrize(
    ("source_key", "counts"),
    [
        ("raw_history", (41915, 69370, 693700)),
        ("extended_history", (42177, 70029, 700290)),
    ],
)
def test_raw_map_cache_isolated_by_source_and_validates_source_cardinality(
    tmp_path, source_key, counts
):
    import hashlib
    import pickle

    from src.analysis.c0_region_factorial import _get_raw_map_cache, _sha256

    raw_paths = {
        "raw_history": tmp_path / "raw.parquet",
        "extended_history": tmp_path / "extended.parquet",
    }
    for path in raw_paths.values():
        path.write_bytes(path.name.encode())
    release = tmp_path / "release.json"
    release.write_bytes(b"release")
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    external = tmp_path / "frozen_worlds.pkl"
    series, maps, records_count = counts
    source = raw_paths[source_key]
    record_receipt = {
        "raw_history_sha256": _sha256(source),
        "release_map_sha256": _sha256(release),
        "source_series_count": series,
        "source_map_count": maps,
        "map_record_count": records_count,
        "player_count": 1,
        "source_map_membership_sha256": "maps",
        "source_series_membership_sha256": "series",
    }
    envelope = {
        "schema": "c0_full_records_v1",
        "raw_history_sha256": _sha256(source),
        "release_map_sha256": _sha256(release),
        "records_by_player": {"p": []},
        "receipt": record_receipt,
    }
    cache = (
        external
        if source_key == "extended_history"
        else feature_root / "historical_records.pkl"
    )
    cache.write_bytes(pickle.dumps(envelope))
    if source_key == "raw_history":
        external.write_bytes(b"extended snapshot must not be reused")
    protocol = {
        "inputs": {
            **{key: str(path) for key, path in raw_paths.items()},
            "release_map_relative": "release.json",
            "worlds_records_cache": str(external),
            "worlds_records_cache_sha256": hashlib.sha256(
                external.read_bytes()
            ).hexdigest(),
        },
        "raw_map_cache_contracts": {
            key: {
                "source_series_count": vals[0],
                "source_map_count": vals[1],
                "map_record_count": vals[2],
            }
            for key, vals in (
                ("raw_history", (41915, 69370, 693700)),
                ("extended_history", (42177, 70029, 700290)),
            )
        },
        "feature_root": str(feature_root),
    }
    records, receipt = _get_raw_map_cache(
        protocol, tmp_path, source_path=source, cache_name="historical_records.pkl"
    )
    assert records == {"p": []}
    assert receipt["source_series_count"] == series
    protocol["raw_map_cache_contracts"][source_key]["source_series_count"] += 1
    with pytest.raises(RuntimeError, match="receipt differ"):
        _get_raw_map_cache(
            protocol, tmp_path, source_path=source, cache_name="historical_records.pkl"
        )


CONFIGURATIONS = ("C0", "C0+regional", "fresh20", "fresh20+regional")


@pytest.mark.parametrize(
    ("source_name", "bound_name"),
    [("Gen.G", "Gen.G"), ("Anyone s Legend", "Anyone's Legend")],
)
def test_worlds_native_wire_uses_separate_proof_and_binding_metadata(
    tmp_path, source_name, bound_name
):
    from src.analysis.c0_region_factorial import _worlds_request_metadata

    match_id = 146609
    name_path = tmp_path / "team_names.json"
    name_path.write_text(
        '{"schema":"retained_worlds_team_name_binding_v1","missing":[],"names":'
        f'{{"2804":["{source_name}"],"2813":["Team Liquid"]}}}}'
    )
    wire = {
        "team1_id": "2804",
        "team2_id": "2813",
        "roster_a": ["a1", "a2", "a3", "a4", "a5"],
        "roster_b": ["b1", "b2", "b3", "b4", "b5"],
        "best_of": 1,
        "decision_at": "2026-10-20T12:00:00Z",
        "competition_context": "Worlds",
        "start_at": "2026-10-20T13:00:00Z",
        "mode": "full",
    }
    protocol = {
        "inputs": {"worlds_team_name_binding": str(name_path)},
        "worlds": {
            "match_ids": [match_id],
            "team_id_bindings": [
                {
                    "match_id": match_id,
                    "team1_id": "2804",
                    "team2_id": "2813",
                    "team_a_name": bound_name,
                    "team_b_name": "Team Liquid",
                }
            ],
        },
    }
    proof = {
        "opportunity_id": str(match_id),
        "canonical_match_id": match_id,
        "cutoff_day": "2026-10-20",
    }

    orientation, regional = _worlds_request_metadata(
        protocol, [wire], [str(match_id)], [proof]
    )
    assert orientation == [True]
    assert regional == [
        {
            "opportunity_id": str(match_id),
            "cutoff_day": "2026-10-20",
            "team1_id": "2804",
            "team2_id": "2813",
            "roster_a": wire["roster_a"],
            "roster_b": wire["roster_b"],
        }
    ]
    wrong_team = {**wire, "team1_id": "2813"}
    with pytest.raises(ValueError, match="team IDs differ"):
        _worlds_request_metadata(protocol, [wrong_team], [str(match_id)], [proof])
    wrong_name_protocol = {
        **protocol,
        "worlds": {
            **protocol["worlds"],
            "team_id_bindings": [
                {**protocol["worlds"]["team_id_bindings"][0], "team_a_name": "Gen.Gx"}
            ],
        },
    }
    with pytest.raises(ValueError, match="team name differs"):
        _worlds_request_metadata(wrong_name_protocol, [wire], [str(match_id)], [proof])


def test_worlds_query_requests_retain_exact_wire_fields_with_native_sorted_slots():
    from src.models.c0_live_query import _worlds_wire_request_with_native_slots

    original = {
        "team1_id": "2804",
        "team2_id": "2850",
        "roster_a": ["685", "1115", "700", "701", "702"],
        "roster_b": ["803", "804", "805", "806", "807"],
        "best_of": 1,
        "decision_at": "2026-10-20T10:15:00+00:00",
        "competition_context": "Worlds",
        "start_at": "2026-10-20T11:00:00+00:00",
        "mode": "full",
    }
    validated = {
        **original,
        "roster_a": sorted(original["roster_a"]),
        "roster_b": sorted(original["roster_b"]),
    }

    returned = _worlds_wire_request_with_native_slots(original, validated)

    assert set(returned) == {
        "team1_id",
        "team2_id",
        "roster_a",
        "roster_b",
        "best_of",
        "decision_at",
        "competition_context",
        "start_at",
        "mode",
    }
    assert returned["roster_a"] == ["1115", "685", "700", "701", "702"]
    assert returned["roster_a"] == validated["roster_a"]
    assert returned["roster_b"] == validated["roster_b"]
    assert returned["decision_at"] == original["decision_at"]
    assert returned["start_at"] == original["start_at"]


def test_assembly_keeps_all_four_cells_and_requires_nineteen_fresh20_observed_slots():
    canonical = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "canonical_a_is_team1": [True, False],
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "match_id": match_id,
                "configuration": configuration,
                "canonical_a_is_team1": match_id == "m1",
                "probability": 0.6,
                "status": "observed",
                "observed_history_slots": 19
                if configuration.startswith("fresh20")
                else 0,
            }
            for match_id in canonical.match_id
            for configuration in CONFIGURATIONS
        ]
    )

    with pytest.raises(ValueError, match="19.*observed"):
        assemble_four_configurations(
            canonical,
            predictions.assign(
                observed_history_slots=np.where(
                    predictions.configuration.str.startswith("fresh20"), 16, 0
                )
            ),
        )

    assembled = assemble_four_configurations(canonical, predictions)
    assert len(assembled) == 2 * 4
    assert set(assembled.configuration) == set(CONFIGURATIONS)
    assert assembled.groupby("match_id").size().eq(4).all()


def test_assembly_rejects_prediction_orientation_that_conflicts_with_canonical_identity():
    canonical = pd.DataFrame({"match_id": ["m2"], "canonical_a_is_team1": [False]})
    contradictory = pd.DataFrame(
        [
            {
                "match_id": "m2",
                "configuration": "C0",
                "canonical_a_is_team1": True,
                "probability": 0.6,
                "status": "observed",
            },
        ]
    )

    with pytest.raises(ValueError, match="orientation differs"):
        assemble_four_configurations(canonical, contradictory)


def test_assembly_joins_on_original_id_and_canonical_orientation_without_positional_matching():
    canonical = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "canonical_a_is_team1": [True, False],
            "y": [1, 0],
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "match_id": "m2",
                "configuration": "C0",
                "canonical_a_is_team1": False,
                "probability": 0.2,
                "status": "observed",
            },
            {
                "match_id": "m1",
                "configuration": "C0",
                "canonical_a_is_team1": True,
                "probability": 0.8,
                "status": "observed",
            },
        ]
    )

    result = assemble_four_configurations(canonical, predictions)
    c0 = result.loc[result.configuration.eq("C0")].set_index("match_id")
    assert c0.loc["m1", "probability"] == pytest.approx(0.8)
    assert c0.loc["m1", "y"] == 1
    assert c0.loc["m2", "probability"] == pytest.approx(0.2)
    assert c0.loc["m2", "y"] == 0
    assert len(result) == 8


def test_assembly_retains_explicit_absences_and_full_denominator():
    canonical = pd.DataFrame(
        {"match_id": ["m1", "m2"], "canonical_a_is_team1": [True, True]}
    )
    predictions = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "configuration": "C0",
                "canonical_a_is_team1": True,
                "probability": 0.7,
                "status": "observed",
            },
            {
                "match_id": "m2",
                "configuration": "C0",
                "canonical_a_is_team1": True,
                "probability": 0.4,
                "status": "observed",
            },
        ]
    )

    result = assemble_four_configurations(canonical, predictions)
    missing = result.loc[result.configuration.eq("fresh20")]
    assert len(result) == 8
    assert len(missing) == 2
    assert missing.probability.isna().all()
    assert missing.status.eq("absent_prediction").all()
    assert result.groupby("configuration").size().eq(2).all()


def test_regional_transform_changes_all_nineteen_observed_slots_once_and_preserves_masks():
    raw = {
        "players": np.zeros((1, 2, 5, 21), dtype=np.float32),
        "core": np.zeros((1, 2, 5), dtype=np.float32),
        "team": np.zeros((1, 2, 11), dtype=np.float32),
        "w20": np.full((1, 2, 10), 0.5, dtype=np.float32),
        "gates": np.ones((1, 2, 2), dtype=np.float32),
    }
    history = {
        "history": np.zeros((1, 2, 5, 19, 33), dtype=np.float16),
        "mask": np.ones((1, 2, 5, 19), dtype=np.uint8),
        "champions": np.zeros((1, 2, 5, 16), dtype=np.uint16),
    }
    history["history"][..., 12] = 1.0  # presence marker for stat index zero
    history["history"][..., 0] = (
        np.arange(19, dtype=np.float16)[None, None, None, :] / 100
    )
    history["history"][..., 24] = 0.5
    history["history"][..., 26] = 0.1
    history["mask"][..., 3] = 0
    history["history"][..., 3, :] = 0

    raw_once, history_once = transform_batch_once(
        raw, history, np.array([180.0]), np.array([0.0])
    )
    assert raw_once["players"][0, 0, 0, 1] != raw["players"][0, 0, 0, 1]
    np.testing.assert_array_equal(
        history_once["history"][0, 0, :, :, 0] != history["history"][0, 0, :, :, 0],
        history["mask"][0, 0],
    )
    np.testing.assert_array_equal(
        history_once["history"][..., 3, :], history["history"][..., 3, :]
    )
    np.testing.assert_array_equal(history_once["mask"], history["mask"])
    np.testing.assert_array_equal(history_once["champions"], history["champions"])
    assert not np.array_equal(raw_once["players"], raw["players"])
    assert not np.array_equal(history_once["history"], history["history"])


def test_mixed_batch_keeps_zero_difference_float32_rows_bit_identical():
    raw = {
        "players": np.zeros((2, 2, 5, 21), dtype=np.float32),
        "core": np.zeros((2, 2, 5), dtype=np.float32),
        "team": np.zeros((2, 2, 11), dtype=np.float32),
        "w20": np.full((2, 2, 10), 0.5, dtype=np.float32),
        "gates": np.ones((2, 2, 2), dtype=np.float32),
    }
    history = {
        "history": np.full((2, 2, 5, 19, 33), 0.12345679, dtype=np.float32),
        "mask": np.ones((2, 2, 5, 19), dtype=np.uint8),
        "champions": np.zeros((2, 2, 5, 16), dtype=np.uint16),
    }
    history["history"][..., 12] = 1.0
    history["history"][..., 24] = 0.5
    history["history"][..., 26] = 0.1

    raw_once, history_once = transform_batch_once(
        raw, history, np.array([0.0, 180.0]), np.array([0.0, 0.0])
    )
    for key in raw:
        np.testing.assert_array_equal(raw_once[key][0], raw[key][0])
    np.testing.assert_array_equal(history_once["history"][0], history["history"][0])
    assert raw_once["players"][1, 0, 0, 1] != raw["players"][1, 0, 0, 1]
    assert not np.array_equal(history_once["history"][1], history["history"][1])
    np.testing.assert_array_equal(
        history_once["history"][1],
        history_once["history"][1].astype(np.float16).astype(np.float32),
    )


def test_proper_scores_do_not_hide_clipping_of_extreme_probabilities():
    correct = score_probabilities(np.array([1, 0]), np.array([1.0, 0.0]))
    wrong = score_probabilities(np.array([0, 1]), np.array([1.0, 0.0]))
    assert correct["n"] == 2
    assert correct["log_loss"] == pytest.approx(0.0)
    assert correct["brier"] == pytest.approx(0.0)
    assert np.isinf(wrong["log_loss"])
    assert (
        "auc" in correct
        and "accuracy" in correct
        and "ece" in correct
        and "tails" in correct
    )


def test_paired_monthly_contrast_matches_rows_and_keeps_unequal_month_sizes():
    frame = pd.DataFrame(
        {
            "month": ["2026-01"] * 3 + ["2026-02"],
            "y": [1, 0, 1, 0],
            "reference": [0.8, 0.2, 0.6, 0.4],
            "candidate": [0.7, 0.3, 0.5, 0.1],
        }
    )
    result = paired_monthly_contrast(
        frame, "candidate", "reference", draws=5000, seed=19, metric="brier"
    )
    assert result["n_rows"] == 4
    assert result["n_months"] == 2
    assert result["match_weighted_delta"] == pytest.approx(
        np.mean((frame.candidate - frame.y) ** 2 - (frame.reference - frame.y) ** 2)
    )
    expected_equal_month = np.mean(
        [
            (
                (frame.loc[:2, "candidate"] - frame.loc[:2, "y"]) ** 2
                - (frame.loc[:2, "reference"] - frame.loc[:2, "y"]) ** 2
            ).mean(),
            (
                (frame.loc[3:, "candidate"] - frame.loc[3:, "y"]) ** 2
                - (frame.loc[3:, "reference"] - frame.loc[3:, "y"]) ** 2
            ).mean(),
        ]
    )
    assert result["equal_month_delta"] == pytest.approx(expected_equal_month)
    assert result["match_weighted_delta"] != pytest.approx(result["equal_month_delta"])


@pytest.mark.parametrize(
    ("query_clock", "forecast_clock", "source_day", "prior_proof", "admitted"),
    [
        ("00:00", "00:00", "2026-05-28", True, True),
        ("12:00", "12:00", "2026-05-28", True, True),
        ("13:00", "13:00", "2026-05-28", True, False),
        ("12:00", "13:00", "2026-05-28", True, False),
        ("12:00", "12:00", "2026-05-29", True, False),
        ("12:00", "12:00", "2026-05-28", False, False),
    ],
)
def test_first_ready_projects_only_prior_admissible_same_day_states(
    tmp_path, query_clock, forecast_clock, source_day, prior_proof, admitted
):
    from src.analysis.c0_region_factorial import _build_first_ready_predictions

    edge = tmp_path / "edge"
    (edge / "alpha").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "opportunity_id": "fixture-book",
                "canonical_match_id": "1",
                "decision_at": "2026-05-29T12:00:00Z",
                "forecast_at": "2026-05-29T12:00:00Z",
            }
        ]
    ).to_parquet(edge / "alpha" / "prepared_cohort.parquet", index=False)
    daily_path = tmp_path / "daily.parquet"
    probability_columns = ["p_C0", "p_C0_regional", "p_fresh20", "p_fresh20_regional"]
    pd.DataFrame(
        [
            {
                "canonical_match_id": "1",
                "requested_utc_day": "2026-05-29",
                "decision_at": f"2026-05-29T{query_clock}:00Z",
                "forecast_at": f"2026-05-29T{forecast_clock}:00Z",
                "feature_source_max_day": source_day,
                "daily_state_cutoff_day": "2026-05-29",
                "strict_prior_feature_day_proof": prior_proof,
                "usable_for_same_day_projection": True,
                "regional_status": "ok",
                "regional_state_sha256": "prior-state",
                **dict.fromkeys(probability_columns, 0.7),
            }
        ]
    ).to_parquet(daily_path, index=False)
    protocol = {
        "inputs": {"edge_root": str(edge)},
        "historical": {
            "market_first_ready_expected_rows": 1,
            "market_first_ready_expected_fixtures": 1,
        },
    }
    output = tmp_path / "first_ready.parquet"
    _build_first_ready_predictions(protocol, daily_path, output)
    result = pd.read_parquet(output)
    assert bool(result[probability_columns].notna().all(axis=1).iloc[0]) is admitted
    assert (result.factorial_query_status.iloc[0] == "observed") is admitted


def _write_sealed_factorial_fixture(tmp_path):
    from pathlib import Path
    import hashlib
    import json

    from src.analysis.c0_region_factorial import _validated_sealed_daily_predictions

    old_protocol = {
        "schema": "c0_region_factorial_v1",
        "run_id": "historical-v4",
        "inputs": {"raw_history": "pinned-source"},
        **{
            key: {"fixed": key}
            for key in (
                "configurations",
                "anchors",
                "regional",
                "historical",
                "financial",
                "bootstrap",
                "decision",
                "prohibited",
            )
        },
    }
    protocol_path = tmp_path / "historical_protocol.json"
    protocol_path.write_text(json.dumps(old_protocol))
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    binding = {
        "schema": "c0_region_factorial_execution_binding_v1",
        "run_id": "historical-v4",
        "protocol_sha256": protocol_hash,
    }
    binding_path = tmp_path / "historical_binding.json"
    binding_path.write_text(json.dumps(binding))
    binding_hash = hashlib.sha256(
        json.dumps(binding, sort_keys=True).encode()
    ).hexdigest()
    daily_path = tmp_path / "daily_predictions.parquet"
    pd.DataFrame({"row": np.arange(1126)}).to_parquet(daily_path, index=False)
    daily_hash = hashlib.sha256(daily_path.read_bytes()).hexdigest()
    stage = {
        "completed_at_utc": "2026-10-22T00:00:00Z",
        "schema": "c0_region_factorial_stage_v1",
        "stage": "historical",
        "binding_sha256": binding_hash,
        "receipt": {
            "status": "COMPLETE",
            "canonical_rows": 3103,
            "daily_rows": 1126,
            "daily_predictions": str(daily_path),
        },
        "artifacts": {
            "daily_model_predictions": {
                "path": str(daily_path),
                "sha256": daily_hash,
            }
        },
    }
    stage_path = tmp_path / "historical_stage.json"
    stage_path.write_text(json.dumps(stage))
    sources = {
        "sealed_daily_predictions": str(daily_path),
        "sealed_historical_stage": str(stage_path),
        "sealed_historical_protocol": str(protocol_path),
        "sealed_historical_binding": str(binding_path),
    }
    for key, source in tuple(sources.items()):
        sources[f"{key}_sha256"] = hashlib.sha256(
            (tmp_path / Path(source).name).read_bytes()
        ).hexdigest()
    new_protocol = {
        **old_protocol,
        "inputs": {**old_protocol["inputs"], **sources},
    }
    return _validated_sealed_daily_predictions, new_protocol, daily_path, stage_path


def test_sealed_completed_historical_artifacts_are_validated_from_real_files(tmp_path):
    from pathlib import Path

    validate, protocol, daily_path, _ = _write_sealed_factorial_fixture(tmp_path)
    daily, source_hashes = validate(protocol)
    assert daily == Path(daily_path)
    assert source_hashes["daily_predictions"]
    assert len(pd.read_parquet(daily)) == 1126


@pytest.mark.parametrize("tamper", ["daily", "binding", "science"])
def test_sealed_historical_rejects_tampered_artifacts_and_scientific_settings(
    tmp_path, tamper
):
    import hashlib

    validate, protocol, daily_path, stage_path = _write_sealed_factorial_fixture(
        tmp_path
    )
    if tamper == "daily":
        pd.DataFrame({"row": np.arange(1125)}).to_parquet(daily_path, index=False)
        protocol["inputs"]["sealed_daily_predictions_sha256"] = hashlib.sha256(
            daily_path.read_bytes()
        ).hexdigest()
    elif tamper == "binding":
        stage = __import__("json").loads(stage_path.read_text())
        stage["binding_sha256"] = "0" * 64
        stage_path.write_text(__import__("json").dumps(stage))
        protocol["inputs"]["sealed_historical_stage_sha256"] = hashlib.sha256(
            stage_path.read_bytes()
        ).hexdigest()
    else:
        protocol["financial"] = {"fixed": "changed"}
    with pytest.raises(ValueError):
        validate(protocol)
