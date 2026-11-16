from __future__ import annotations

import numpy as np
import pandas as pd


def test_build_refuses_historical_output_before_computing(monkeypatch, tmp_path):
    import sys
    from scripts import build_year_anchor_bank as builder


    historical = tmp_path / "historical"
    historical.mkdir()
    monkeypatch.setattr(builder, "BANK_DIR", historical)
    monkeypatch.setattr(builder, "build", lambda _root: (_ for _ in ()).throw(AssertionError("build called")))
    monkeypatch.setattr(sys, "argv", ["build_year_anchor_bank", "--research-root", str(tmp_path),
                                     "--output-dir", str(historical)])

    with np.testing.assert_raises(FileExistsError):
        builder.main()


def test_build_refuses_existing_output_without_modifying_it(monkeypatch, tmp_path):
    import sys

    from scripts import build_year_anchor_bank as builder

    output = tmp_path / "rebuilt"
    output.mkdir()
    sentinel = output / "keep.npy"
    sentinel.write_bytes(b"historical")
    monkeypatch.setattr(builder, "build", lambda _root: (_ for _ in ()).throw(AssertionError("build called")))
    monkeypatch.setattr(sys, "argv", ["build_year_anchor_bank", "--research-root", str(tmp_path),
                                     "--output-dir", str(output)])

    with np.testing.assert_raises(FileExistsError):
        builder.main()
    assert sentinel.read_bytes() == b"historical"


def test_build_defaults_to_distinct_new_directory(monkeypatch, tmp_path):
    import sys

    from scripts import build_year_anchor_bank as builder

    output = tmp_path / "rebuilt"
    monkeypatch.setattr(builder, "BANK_DIR", tmp_path / "historical")
    monkeypatch.setattr(builder, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(builder, "build", lambda _root: (
        np.zeros((1,)), np.zeros((1,)), np.zeros((1,))))
    monkeypatch.setattr(builder.pd, "read_parquet", lambda _path: pd.DataFrame({"golgg_match_id": ["id"]}))
    monkeypatch.setattr(builder, "DEFAULT_OUTPUT_DIR", output)
    monkeypatch.setattr(sys, "argv", ["build_year_anchor_bank", "--research-root", str(tmp_path)])

    builder.main()
    assert output.is_dir()
    assert (output / "year_anchor_history.npy").exists()


def test_empty_research_root_environment_falls_back_to_config(monkeypatch, tmp_path):
    from scripts import build_year_anchor_bank as builder

    config = tmp_path / "data" / "research_root.txt"
    config.parent.mkdir()
    config.write_text("research")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setenv("ENSEMBLE_RESEARCH_ROOT", "")

    assert builder._configured_research_root() == tmp_path / "research"
from scripts.build_year_anchor_bank import _build_from_sources


def _inputs(rows: int = 1):
    dates = pd.date_range("2024-01-01", periods=rows).strftime("%Y-%m-%d")
    meta = pd.DataFrame({
        "date": dates,
        "golgg_match_id": [f"match-{i}" for i in range(rows)],
        "roster_a": [[f"player-{i}" for i in range(5)] for _ in range(rows)],
        "roster_b": [[f"enemy-{i}" for i in range(5)] for _ in range(rows)],
    })
    history = np.zeros((rows, 2, 5, 16, 33), dtype=np.float32)
    mask = np.zeros((rows, 2, 5, 16), dtype=np.uint8)
    champions = np.zeros((rows, 2, 5, 16), dtype=np.uint16)
    return meta, history, mask, champions


def test_build_excludes_same_day_and_exactly_365_day_maps():
    meta, history, mask, champions = _inputs()
    for k, age, value in ((0, 0, 100), (1, 365, 200), (2, 364, 7)):
        mask[0, 0, 0, k] = 1
        history[0, 0, 0, k, 0] = value
        history[0, 0, 0, k, 12] = 1
        history[0, 0, 0, k, 25] = np.log1p(age)
        champions[0, 0, 0, k] = k + 1

    result, result_mask, counts = _build_from_sources(meta, history, mask, champions)

    assert result_mask[0, 0, 0, 0] == 1
    assert counts[0, 0, 0, 0] == 1
    assert result[0, 0, 0, 0, 0] == 7


def test_player_without_released_maps_has_zero_anchor_and_mask():
    meta, history, mask, champions = _inputs()

    result, result_mask, counts = _build_from_sources(meta, history, mask, champions)

    assert not result.any()
    assert not result_mask.any()
    assert not counts.any()


def test_future_source_perturbation_cannot_change_earlier_anchor():
    meta, history, mask, champions = _inputs(rows=2)
    mask[0, 0, 0, 0] = mask[1, 0, 0, 0] = 1
    history[0, 0, 0, 0, 0] = history[1, 0, 0, 0, 0] = 3
    history[0, 0, 0, 0, 12] = history[1, 0, 0, 0, 12] = 1
    history[0, 0, 0, 0, 25] = history[1, 0, 0, 0, 25] = np.log1p(1)
    champions[0, 0, 0, 0] = champions[1, 0, 0, 0] = 1
    base = _build_from_sources(meta, history, mask, champions)
    changed_history, changed_mask, changed_champions = history.copy(), mask.copy(), champions.copy()
    changed_history[1] += 17
    changed_mask[1] ^= 1
    changed_champions[1] += 1
    changed = _build_from_sources(meta, changed_history, changed_mask, changed_champions)

    for before, after in zip(base, changed):
        np.testing.assert_array_equal(before[0], after[0])

def test_future_valid_age_map_cannot_leak_into_earlier_anchor():
    meta, history, mask, champions = _inputs(rows=2)
    mask[1, 0, 0, 0] = 1
    history[1, 0, 0, 0, 0] = 9
    history[1, 0, 0, 0, 12] = 1
    history[1, 0, 0, 0, 25] = np.log1p(2)
    champions[1, 0, 0, 0] = 7

    base = _build_from_sources(meta, history, mask, champions)
    changed_history, changed_mask, changed_champions = history.copy(), mask.copy(), champions.copy()
    changed_history[1, 0, 0, 0, 0] = 19
    changed_champions[1, 0, 0, 0] = 8
    changed = _build_from_sources(meta, changed_history, changed_mask, changed_champions)

    assert not base[1][0, 0, 0, 0]
    for before, after in zip(base, changed):
        np.testing.assert_array_equal(before[0], after[0])

def test_build_rejects_non_chronological_metadata():
    meta, history, mask, champions = _inputs(rows=2)
    meta.loc[1, "date"] = "2023-12-31"
    with np.testing.assert_raises(ValueError):
        _build_from_sources(meta, history, mask, champions)
