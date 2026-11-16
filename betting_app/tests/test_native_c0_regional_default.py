from datetime import UTC, datetime

import numpy as np
import pytest

from betting_app.core.models.native_c0 import NativeC0Adapter


def _adapter():
    from collections import OrderedDict

    adapter = object.__new__(NativeC0Adapter)
    adapter._load_history = lambda: ("history.json", "raw", "release", {"release_path": "release.json"})
    adapter._regional_replay_cache = OrderedDict()
    return adapter


def _request():
    return {
        "team1_id": "1",
        "team2_id": "2",
        "roster_a": ["a1", "a2", "a3", "a4", "a5"],
        "roster_b": ["b1", "b2", "b3", "b4", "b5"],
        "decision_at": datetime(2026, 10, 1, tzinfo=UTC),
        "mode": "full",
    }


def _tensors(mask):
    from betting_app.core.models.native_c0 import _RAW_SHAPES

    raw = {key: np.zeros(shape, dtype=np.float32) for key, shape in _RAW_SHAPES.items()}
    hist = {
        "history": np.zeros((2, 5, 19, 33), dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.uint8).reshape(2, 5, 19),
        "champions": np.zeros((2, 5, 16), dtype=np.uint16),
    }
    return raw, hist


def test_regional_adjustment_preserves_neutral_cold_start_without_replay():
    adapter = _adapter()
    raw, hist = _tensors(np.zeros(190, dtype=np.uint8))

    proof = adapter._apply_regional_default(_request(), {}, raw, hist)

    assert proof["status"] == "neutral_cold_start"
    assert proof["family_a"] is proof["family_b"] is None
    assert not np.any(raw["players"])


def test_regional_adjustment_fails_closed_when_supported_history_has_unknown_family(monkeypatch):
    from src.models import c0_regional_replay

    monkeypatch.setattr(
        c0_regional_replay,
        "build_regional_query_states",
        lambda *args, **kwargs: {
            "rows": [{"status": "unavailable_family_or_support", "offset_a": None, "offset_b": 10.0}],
            "receipt": {"raw_history_sha256": "raw", "release_map_sha256": "release"},
        },
    )
    adapter = _adapter()
    raw, hist = _tensors(np.ones(190, dtype=np.uint8))

    with pytest.raises(ValueError, match="strictly prior domestic-family identity"):
        adapter._apply_regional_default(_request(), {}, raw, hist)


def test_evidence_snapshot_recomputes_exact_adjustment_after_adapter_restart(monkeypatch):
    from src.models import c0_regional_replay

    replay_result = {
        "rows": [{
            "status": "ok", "cutoff_day": "2026-10-01", "family_a": "LCK",
            "family_b": "LEC", "offset_a": 300.0, "offset_b": 100.0,
            "state_sha256": "state-hash", "max_release_day": "2026-09-30",
        }],
        "receipt": {
            "raw_history_sha256": "raw", "release_map_sha256": "release",
            "max_release_day": "2026-10-20", "parameter_law": {"tau": 0.5},
            "implementation_binding": {"normalizer": "law-hash"},
        },
    }
    monkeypatch.setattr(c0_regional_replay, "build_regional_query_states", lambda *args, **kwargs: replay_result)
    adapter = _adapter()
    raw, hist = _tensors(np.ones(190, dtype=np.uint8))
    target = {}
    issued = adapter._apply_regional_default(_request(), target, raw, hist)
    # The adjusted snapshot must remain encodable by the public fresh20 wire contract.
    from betting_app.core.models.native_c0 import encode_snapshot
    assert hist["history"].dtype == np.dtype("float32")
    encoded = encode_snapshot({"target": target, "raw": raw, "hist": hist})
    assert encoded["hist"]["history"]["shape"] == [2, 5, 19, 33]
    assert encoded["hist"]["history"]["dtype"] == "float32"
    adjusted_raw = {key: value.copy() for key, value in raw.items()}
    adjusted_hist = {key: value.copy() for key, value in hist.items()}

    restarted_adapter = _adapter()
    replayed = restarted_adapter._apply_regional_default(
        _request(), target, adjusted_raw, adjusted_hist,
    )

    assert replayed["status"] == "verified_evidence_snapshot"
    assert replayed["tensor_hashes"] == issued["tensor_hashes"]
    assert issued["max_release_day"] == "2026-09-30"
    assert issued["source_max_release_day"] == "2026-10-20"


def test_regional_replay_cache_separates_cutoff_and_invalidates_changed_sources(monkeypatch):
    from src.models import c0_regional_replay

    calls = []

    def replay(requests, *, raw_history_path, release_map_path):
        cutoff = requests[0]["cutoff_day"]
        calls.append((cutoff, raw_history_path, release_map_path))
        return {
            "rows": [{
                "status": "ok", "cutoff_day": cutoff, "state_sha256": f"state-{cutoff}",
                "max_release_day": "2026-10-01" if cutoff == "2026-10-02" else "2026-09-30",
            }],
            "receipt": {
                "raw_history_sha256": current["raw"],
                "release_map_sha256": current["release"],
                "max_release_day": "2026-10-20",
            },
        }

    monkeypatch.setattr(c0_regional_replay, "build_regional_query_states", replay)
    current = {"raw": "raw-v1", "release": "release-v1"}
    adapter = _adapter()
    adapter._load_history = lambda: (
        "history.json", current["raw"], current["release"],
        {"release_path": "release.json"},
    )
    early = _request()
    later = {**early, "decision_at": datetime(2026, 10, 2, tzinfo=UTC)}

    early_state, _ = adapter._regional_state(early)
    assert adapter._regional_state(early)[0] == early_state
    later_state, _ = adapter._regional_state(later)
    early_row, early_receipt = adapter._regional_state(early)
    assert early_row["max_release_day"] == "2026-09-30"
    assert early_row["max_release_day"] < early_row["cutoff_day"]
    assert early_receipt["max_release_day"] == "2026-10-20"
    assert later_state["max_release_day"] == "2026-10-01"
    assert later_state["max_release_day"] < later_state["cutoff_day"]
    assert early_state["state_sha256"] != later_state["state_sha256"]
    assert len(calls) == 2

    current["raw"] = "raw-v2"
    changed_source_state, _ = adapter._regional_state(early)
    assert changed_source_state["state_sha256"] == early_state["state_sha256"]
    assert len(calls) == 3


@pytest.mark.parametrize("tamper", ["source", "law", "tensor", "mode"])
def test_portable_regional_evidence_rejects_tampered_source_law_or_tensors(monkeypatch, tamper):
    from copy import deepcopy

    from src.models import c0_regional_replay

    replay_result = {
        "rows": [{
            "status": "ok", "cutoff_day": "2026-10-01", "family_a": "LCK",
            "family_b": "LEC", "offset_a": 300.0, "offset_b": 100.0,
            "state_sha256": "state-hash", "max_release_day": "2026-09-30",
        }],
        "receipt": {
            "raw_history_sha256": "raw", "release_map_sha256": "release",
            "max_release_day": "2026-10-20", "parameter_law": {"tau": 0.5},
            "implementation_binding": {"normalizer": "law-hash"},
        },
    }
    monkeypatch.setattr(c0_regional_replay, "build_regional_query_states", lambda *args, **kwargs: replay_result)
    adapter = _adapter()
    raw, hist = _tensors(np.ones(190, dtype=np.uint8))
    target = {}
    adapter._apply_regional_default(_request(), target, raw, hist)
    changed_target = deepcopy(target)
    changed_raw = {key: value.copy() for key, value in raw.items()}
    changed_hist = {key: value.copy() for key, value in hist.items()}
    if tamper == "source":
        changed_target["regional_adjustment"]["raw_history_sha256"] = "forged"
    elif tamper == "law":
        changed_target["regional_adjustment"]["law_version"] = "other-law"
    elif tamper == "mode":
        changed_target["regional_adjustment"]["mode"] = "no_w20"
    else:
        changed_raw["players"][0, 0, 0] += 1.0

    with pytest.raises(ValueError, match="regional"):
        _adapter()._apply_regional_default(_request(), changed_target, changed_raw, changed_hist)

def test_domestic_equal_offsets_leave_all_tensors_and_temporal_masks_unchanged(monkeypatch):
    from src.models import c0_regional_replay

    monkeypatch.setattr(
        c0_regional_replay,
        "build_regional_query_states",
        lambda *args, **kwargs: {
            "rows": [{
                "status": "ok", "cutoff_day": "2026-10-01", "family_a": "LEC",
                "family_b": "LEC", "offset_a": 100.0, "offset_b": 100.0,
                "state_sha256": "state-hash", "max_release_day": "2026-09-30",
            }],
            "receipt": {
                "raw_history_sha256": "raw", "release_map_sha256": "release",
                "max_release_day": "2026-10-20", "parameter_law": {"tau": 0.5},
                "implementation_binding": {"normalizer": "law-hash"},
            },
        },
    )
    adapter = _adapter()
    raw, hist = _tensors(np.tile(np.asarray([0, 1], dtype=np.uint8), 95))
    raw_before = {key: value.copy() for key, value in raw.items()}
    hist_before = {key: value.copy() for key, value in hist.items()}

    proof = adapter._apply_regional_default(_request(), {}, raw, hist)

    assert proof["status"] == "adjusted"
    assert all(np.array_equal(raw[key], value) for key, value in raw_before.items())
    assert all(np.array_equal(hist[key], value) for key, value in hist_before.items())


@pytest.mark.parametrize(
    ("mode", "changed_context"),
    [("no_w20", "w20"), ("no_organization", "w20"), ("no_organization", "team")],
)
def test_excluded_context_cannot_indirectly_change_regional_player_inputs(mode, changed_context):
    adapter = _adapter()
    adapter._regional_state = lambda request: (
        {
            "status": "ok", "cutoff_day": "2026-10-01", "family_a": "LCK",
            "family_b": "LEC", "offset_a": 300.0, "offset_b": 100.0,
            "state_sha256": "state", "max_release_day": "2026-09-30",
        },
        {
            "raw_history_sha256": "raw", "release_map_sha256": "release",
            "max_release_day": "2026-10-20", "parameter_law": {"tau": 0.5},
            "implementation_binding": {"normalizer": "law"},
        },
    )
    request = {**_request(), "mode": mode}
    first_raw, first_hist = _tensors(np.ones(190, dtype=np.uint8))
    first_raw["gates"].fill(1.0)
    second_raw = {key: value.copy() for key, value in first_raw.items()}
    second_hist = {key: value.copy() for key, value in first_hist.items()}
    first_raw[changed_context][0, 0] = 0.05
    second_raw[changed_context][0, 0] = 0.95

    adapter._apply_regional_default(request, {}, first_raw, first_hist)
    adapter._apply_regional_default(request, {}, second_raw, second_hist)

    for key in first_raw:
        np.testing.assert_array_equal(first_raw[key], second_raw[key])
    for key in first_hist:
        np.testing.assert_array_equal(first_hist[key], second_hist[key])
