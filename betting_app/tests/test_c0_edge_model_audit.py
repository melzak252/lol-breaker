from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.c0_edge_model_audit import (
    _verify_prior_roster_source,
    attach_model_probabilities,
    build_prior_query_requests,
    build_prior_roster_anchors,
    FRESH_PROTOCOL,
    LEGACY_MANIFEST,
    model_identity,
)



def test_prior_roster_anchor_preserves_slots_masks_release_cutoff_and_side_swap():
    vector_a = np.zeros(33, dtype=np.float32)
    vector_a[12] = 1.0
    vector_a[0] = 4.0
    vector_b = np.zeros(33, dtype=np.float32)
    vector_b[0] = 7.0
    records = {
        "p1": [
            {"map_id": "a-eligible", "series_id": "s1", "player_id": "p1",
             "played_day": 999, "release_day": 999, "vector": vector_a},
            {"map_id": "a-cutoff", "series_id": "s2", "player_id": "p1",
             "played_day": 999, "release_day": 1000, "vector": vector_a},
            {"map_id": "a-future-release", "series_id": "s4", "player_id": "p1",
             "played_day": 998, "release_day": 1001, "vector": vector_a},
        ],
        "q1": [
            {"map_id": "b-eligible", "series_id": "s3", "player_id": "q1",
             "played_day": 999, "release_day": 999, "vector": vector_b},
        ],
        "p3": [
            {"map_id": "missing-kills", "series_id": "s5", "player_id": "p3",
             "played_day": 999, "release_day": 999, "vector": vector_a * 0 + np.eye(1, 33, 0, dtype=np.float32)[0]},
        ],
    }
    roster_a = ["p2", "p1", "p3", "p4", "p5"]
    roster_b = ["q1", "q2", "q3", "q4", "q5"]
    forward = build_prior_roster_anchors(
        records, roster_a=roster_a, roster_b=roster_b, decision_day=1000
    )
    swapped = build_prior_roster_anchors(
        records, roster_a=roster_b, roster_b=roster_a, decision_day=1000
    )

    assert forward["count_365"][0, 1] == 1
    assert forward["history_365"][0, 1, 0, 25] == pytest.approx(np.log1p(1))
    assert forward["history_365"][0, 2, 0, 0] == 0.0
    assert forward["history_365"][0, 2, 0, 12] == 0.0
    assert forward["history_365"][0, 1, 0, 0] == pytest.approx(4.0)
    assert forward["mask_365"][0, 1, 0]
    assert forward["count_365"][0, 0] == 0
    assert not forward["mask_365"][0, 0, 0]
    assert forward["count_365"][1, 0] == 1
    assert forward["history_365"].shape == (2, 5, 1, 33)
    assert forward["mask_365"].shape == (2, 5, 1)
    assert forward["count_365"].shape == (2, 5)
    for name in ("history_365", "mask_365", "count_365"):
        np.testing.assert_array_equal(forward[name][0], swapped[name][1])
        np.testing.assert_array_equal(forward[name][1], swapped[name][0])

def test_prior_roster_anchor_rejects_nonfive_player_rosters():
    with pytest.raises(ValueError, match="five"):
        build_prior_roster_anchors(
            {},
            roster_a=["p1"],
            roster_b=["q1", "q2", "q3", "q4", "q5"],
            decision_day=1000,
        )


def test_saved_slim_inference_exercises_the_manifest_bound_single_seed():
    if not FRESH_PROTOCOL.is_file():
        pytest.skip("frozen research artifacts are not present in this checkout")
    from src.models.c0_distilled_replay import (
        build_research_batch,
        load_research_bundle,
    )
    from src.models.c0_edge_model_audit import infer_saved_slim

    bundle = load_research_bundle(
        "data/05_model_input/c0_distilled_verification/c0d_repro_20261021_01/protocol.json",
        candidate="18t",
    )
    batch = build_research_batch(bundle, ["73043"])
    result = infer_saved_slim(batch, best_of=[1], mode="full")
    expected = 0.5980010427521782
    assert result.shape == (1,)
    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(expected, abs=2e-6)



def test_registry_aliases_and_distinct_17t_core_have_correct_artifact_identities():
    if not FRESH_PROTOCOL.is_file() or not LEGACY_MANIFEST.is_file():
        pytest.skip("frozen research artifacts are not present in this checkout")
    core = model_identity("C0-D-Core")
    alias = model_identity("C0-D")
    slim = model_identity("C0-Slim")
    core17 = model_identity("C0-D-Core-17T")
    fresh18 = model_identity("18t")

    assert core["artifact_identity"] == alias["artifact_identity"]
    assert len(core["checkpoint_sha256"]) == 3
    assert set(slim["checkpoint_sha256"].values()) < set(core["checkpoint_sha256"].values())
    assert slim["calibration_sha256"] != core["calibration_sha256"]
    assert core17["artifact_identity"] not in {
        core["artifact_identity"], slim["artifact_identity"], fresh18["artifact_identity"]
    }
    assert core17["lineage_status"] == "unverified_legacy_positional"
    assert fresh18["lineage_status"] == "verified_fresh_raw_causal"

def test_query_builder_normalizes_native_parquet_array_rosters_and_nested_source_metadata():
    import pandas as pd

    roster_a = np.array(["a1", "a2", "a3", "a4", "a5"], dtype=object)
    roster_b = np.array(["b1", "b2", "b3", "b4", "b5"], dtype=object)

    def source(source_id, team_id, roster):
        return {
            "source_match_id": source_id,
            "effective_release_day": "2026-06-30",
            "competition_context": "regional",
            "team_id": team_id,
            "roster": np.array(roster, dtype=object),
            "source_series_order": np.array([1, 2], dtype=np.int64),
        }

    row = {
        "canonical_match_id": "target-fixture",
        "opportunity_id": "book-1",
        "decision_at": "2026-07-01T12:00:00Z",
        "start_at": "2026-07-02T12:00:00Z",
        "team1_id": "team-1",
        "team2_id": "team-2",
        "canonical_a_is_team1": True,
        "prior_roster_orientation": "native_team1_team2",
        "best_of": 3,
        "competition_context": "regional",
        "prior_roster_a": roster_a,
        "prior_roster_b": roster_b,
        "prior_source_a": source("prior-a", "team-1", roster_a),
        "prior_source_b": source("prior-b", "team-2", roster_b),
    }

    requests, absence = build_prior_query_requests(pd.DataFrame([row]))

    assert not absence
    assert len(requests) == 1
    assert requests[0]["roster_a"] == roster_a.tolist()
    assert requests[0]["prior_source_a"]["roster"] == roster_a.tolist()

def test_query_builder_preserves_native_team1_orientation_when_canonical_a_is_team2():
    import pandas as pd
    import json

    roster_a = ["a1", "a2", "a3", "a4", "a5"]
    roster_b = ["b1", "b2", "b3", "b4", "b5"]

    def source(source_id, roster):
        return {
            "source_match_id": source_id,
            "effective_release_day": "2026-06-30",
            "competition_context": "regional",
            "roster": roster,
        }

    source_a = source("prior-a", roster_a)
    source_b = source("prior-b", roster_b)
    row = {
        "canonical_match_id": "target-fixture",
        "decision_at": "2026-07-01T12:00:00Z",
        "start_at": "2026-07-02T12:00:00Z",
        "team1_id": "team-1",
        "team2_id": "team-2",
        "canonical_a_is_team1": False,
        "prior_roster_orientation": "native_team1_team2",
        "best_of": 3,
        "competition_context": "regional",
        "prior_roster_a": json.dumps(roster_a),
        "prior_roster_b": json.dumps(roster_b),
        "prior_source_a": json.dumps(source_a),
        "prior_source_b": json.dumps(source_b),
    }
    rows = pd.DataFrame([
        {**row, "opportunity_id": "book-1"},
        {**row, "opportunity_id": "book-2"},
    ])
    requests, absence = build_prior_query_requests(rows)

    assert not absence
    assert len(requests) == 1
    request = requests[0]
    assert request["roster_a"] == roster_a
    assert request["roster_b"] == roster_b
    assert request["prior_source_a"]["source_match_id"] == "prior-a"
    assert request["prior_source_b"]["source_match_id"] == "prior-b"
    _verify_prior_roster_source(request)
    daily_rows = pd.DataFrame([
        {**row, "opportunity_id": "book-1"},
        {
            **row,
            "opportunity_id": "book-2",
            "decision_at": "2026-07-01T12:05:00Z",
        },
    ])
    daily_requests, daily_absence = build_prior_query_requests(
        daily_rows, group_by_decision=True
    )
    assert not daily_absence
    assert len(daily_requests) == 1
    assert len(daily_requests[0]["query_aliases"]) == 2

    request["prior_source_a"]["effective_release_day"] = "2026-07-01"
    with pytest.raises(ValueError, match="strictly prior"):
        _verify_prior_roster_source(request)


def test_query_builder_rejects_duplicate_or_cross_side_native_rosters():
    import pandas as pd

    source = {"source_match_id": "prior", "effective_release_day": "2026-06-30",
              "competition_context": "regional", "roster": ["a1", "a2", "a3", "a4", "a5"]}
    row = {
        "canonical_match_id": "fixture", "opportunity_id": "op",
        "decision_at": "2026-07-01T12:00:00Z", "start_at": "2026-07-02T12:00:00Z",
        "team1_id": "t1", "team2_id": "t2", "canonical_a_is_team1": True,
        "best_of": 3, "competition_context": "regional",
        "prior_roster_orientation": "native_team1_team2",
        "prior_roster_a": ["a1", "a1", "a3", "a4", "a5"],
        "prior_roster_b": ["b1", "b2", "b3", "b4", "b5"],
        "prior_source_a": source, "prior_source_b": source,
    }
    _, absence = build_prior_query_requests(pd.DataFrame([row]))
    assert absence[0]["reason"] == "prior_roster_invalid"


def test_prior_query_missing_required_schema_raises_instead_of_recording_absence():
    import pandas as pd

    with pytest.raises(ValueError, match="missing required native prior-query columns"):
        build_prior_query_requests(pd.DataFrame([{"canonical_match_id": "fixture"}]))


def test_model_probability_join_never_matches_fixture_id_without_exact_query_key():
    import pandas as pd

    source_a = {"source_match_id": "prior-a"}
    source_b = {"source_match_id": "prior-b"}
    row = {
        "canonical_match_id": "fixture-1",
        "decision_at": "2026-07-01T12:00:00Z",
        "prior_roster_a": ["a1", "a2", "a3", "a4", "a5"],
        "prior_roster_b": ["b1", "b2", "b3", "b4", "b5"],
        "prior_source_a": source_a,
        "prior_source_b": source_b,
    }
    rows = pd.DataFrame([
        {**row, "opportunity_id": "same-query"},
        {**row, "opportunity_id": "different-query", "decision_at": "2026-07-01T12:01:00Z"},
    ])
    predictions = pd.DataFrame([{
        **row,
        "p_c0d_18t": 0.61,
        "a_c0d_18t": 0.61,
        "b_c0d_18t": 0.39,
        "p_c0d_20t": 0.62,
        "a_c0d_20t": 0.62,
        "b_c0d_20t": 0.38,
    }])

    from src.models.c0_edge_model_audit import attach_model_probabilities

    joined = attach_model_probabilities(rows, predictions)
    assert joined.loc[0, "model_query_matched"]
    assert joined.loc[0, "p_c0d_18t"] == pytest.approx(0.61)
    assert joined.loc[0, "a_c0d_18t"] == pytest.approx(0.61)
    assert joined.loc[0, "b_c0d_18t"] == pytest.approx(0.39)
    assert not joined.loc[1, "model_query_matched"]
    assert pd.isna(joined.loc[1, "p_c0d_18t"])


def test_isolated_native_query_preserves_parent_archive_namespace():
    import os
    import sys

    from src.analysis import c0_edge_alpha_audit
    from src.models import c0_edge_model_audit as audit

    protocol_path = (
        "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json"
    )
    if not Path(protocol_path).is_file():
        pytest.skip("frozen native query protocol is not present in this checkout")
    protocol = audit._read_json(protocol_path)
    protocol["_protocol_sha256"] = audit._sha256(protocol_path)
    cohort = c0_edge_alpha_audit._load_prepared(protocol)
    if cohort is None:
        cohort = c0_edge_alpha_audit.prepare_edge_cohort(protocol)
    daily = cohort["daily_native_states"].loc[
        cohort["daily_native_states"].usable_for_same_day_projection.eq(True)
    ].copy()
    daily = daily.rename(columns={
        "representative_opportunity_id": "opportunity_id",
        "representative_decision_at": "decision_at",
    })
    requests, absence = audit.build_prior_query_requests(daily, group_by_decision=True)

    before_worker = os.environ.get("BETTING_APP_C0_WORKER")
    scripts_modules = {
        name: module for name, module in sys.modules.items()
        if name == "scripts" or name.startswith("scripts.")
    }
    ratings_modules = {
        name: module for name, module in sys.modules.items()
        if name == "src.ratings" or name.startswith("src.ratings.")
    }
    query = audit._build_causal_query_batches_isolated(requests[:1])

    assert query["p_c0"].shape == (1,)
    assert 0.0 < query["p_c0"][0] < 1.0
    assert query["query_proofs"][0]["native_input_hashes"] == requests[0]["native_input_hashes"]
    assert len(query["query_proofs"][0]["native_input_hashes"]) == 8
    assert os.environ.get("BETTING_APP_C0_WORKER") == before_worker
    assert {
        name: module for name, module in sys.modules.items()
        if name == "scripts" or name.startswith("scripts.")
    } == scripts_modules
    assert {
        name: module for name, module in sys.modules.items()
        if name == "src.ratings" or name.startswith("src.ratings.")
    } == ratings_modules
