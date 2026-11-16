"""Regression tests for GOL.GG team-candidate loading."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from betting_app.services import mapping_service


def test_load_golgg_team_candidates_json_fallback_reads_match_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    matches = [
        {
            "sname_t1": "Alpha Fallback",
            "sname_t2": "Beta Fallback",
            "games": [{"t1_name": "Alpha Fallback Full", "t2_name": "Beta Fallback Full"}],
        },
        "not-a-dict",
    ]
    (data_dir / "golgg_matches.json").write_text(json.dumps(matches), encoding="utf-8")

    monkeypatch.setattr(mapping_service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mapping_service, "_FILE_CANDIDATES_CACHE", None)
    monkeypatch.setattr(mapping_service, "query_df", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/prod")

    assert mapping_service.load_golgg_team_candidates() == [
        "Alpha Fallback",
        "Alpha Fallback Full",
        "Beta Fallback",
        "Beta Fallback Full",
    ]
    monkeypatch.setattr(mapping_service, "_FILE_CANDIDATES_CACHE", None)
