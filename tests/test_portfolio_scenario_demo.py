from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path


def test_demo_cli_emits_synthetic_gated_portfolio_trace() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    isolated_db = "postgresql://unused:unused@127.0.0.1:1/unused"
    env.update({
        "TEST_DATABASE_URL": isolated_db,
        "DATABASE_URL": isolated_db,
        "PLAN11_SCHEMA_DATABASE_URL": isolated_db,
    })
    result = subprocess.run(
        [sys.executable, "scripts/run_unified_betting_evaluation.py", "--demo"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "SYNTHETIC_SCENARIO_NOT_MODEL_EVIDENCE"

    qualification = payload["qualification"]
    abstentions = {row["opportunity_id"] for row in qualification["abstentions"]}
    assert "demo-missing-uncertainty" in abstentions
    offers = qualification["qualified_offers"]
    assert "demo-missing-uncertainty" not in {
        row["opportunity_id"] for row in offers
    }
    offers_by_fixture: dict[str, list[dict[str, object]]] = {}
    for offer in offers:
        offers_by_fixture.setdefault(offer["canonical_match_id"], []).append(offer)
    assert all(len({offer["y"] for offer in rows}) == 1 for rows in offers_by_fixture.values())

    best_by_fixture: dict[str, dict[str, object]] = {}
    for offer in offers:
        fixture = offer["canonical_match_id"]
        current = best_by_fixture.get(fixture)
        if current is None or offer["odds"] > current["odds"]:
            best_by_fixture[fixture] = offer

    portfolio = payload["portfolio"]
    configuration = portfolio["configuration"]
    ledger = portfolio["ledger"]
    funded_by_fixture: dict[str, dict[str, object]] = {}
    for entry in ledger:
        fixture = entry["canonical_match_id"]
        assert fixture not in funded_by_fixture
        funded_by_fixture[fixture] = entry
    assert set(funded_by_fixture) <= set(best_by_fixture)
    assert all(
        entry["opportunity_id"] == best_by_fixture[fixture]["opportunity_id"]
        for fixture, entry in funded_by_fixture.items()
    )

    tax_rate = configuration["tax_rate"]
    settled = [entry for entry in ledger if entry["settled"]]
    assert {entry["y"] for entry in settled} == {0, 1}
    for entry in settled:
        expected_pnl = (
            entry["stake"] * (entry["odds"] * (1.0 - tax_rate) - 1.0)
            if entry["y"] == 1
            else -entry["stake"]
        )
        assert math.isclose(entry["realized_pnl"], expected_pnl, rel_tol=1e-9, abs_tol=1e-8)
        assert math.isclose(
            entry["realized_return"], expected_pnl / entry["stake"],
            rel_tol=1e-9, abs_tol=1e-8,
        )

    pending = [entry for entry in ledger if not entry["settled"]]
    summary = portfolio["summary"]
    assert pending
    assert math.isclose(
        summary["final_reserved"], sum(entry["stake"] for entry in pending),
        rel_tol=1e-9, abs_tol=1e-8,
    )
    assert math.isclose(
        summary["settled_turnover"], sum(entry["stake"] for entry in settled),
        rel_tol=1e-9, abs_tol=1e-8,
    )
    assert math.isclose(
        summary["turnover"], sum(entry["stake"] for entry in ledger),
        rel_tol=1e-9, abs_tol=1e-8,
    )

    exposure_cap = configuration["max_portfolio_exposure"]
    for event in portfolio["capital_path"]:
        cash = event["cash"]
        reserved = event["reserved"]
        equity = event["equity"]
        assert cash >= -1e-8
        assert reserved >= -1e-8
        assert math.isclose(cash + reserved, equity, rel_tol=1e-9, abs_tol=1e-8)
        if event["event"] == "decision":
            assert reserved <= equity * exposure_cap + 1e-8
