import pandas as pd

from src.analysis.c0_interval_runner import (
    _evaluate,
    _validate_daily_state_contract,
    band_membership,
    choose_random_grid_checkpoints,
    select_timing_candidates,
)


def test_repeated_fixture_changes_qualifying_side_using_each_days_native_state():
    clocks = pd.to_datetime(["2026-01-01T14:00Z", "2026-01-02T14:00Z"], utc=True)
    opportunities = pd.DataFrame([
        {"opportunity_id": f"day-{i}", "canonical_match_id": "a",
         "decision_at": clock, "start_at": pd.Timestamp("2026-01-03T12:00Z"),
         "league": "LCK", "best_of": 3}
        for i, clock in enumerate(clocks)
    ])
    forecasts = pd.DataFrame([
        {"opportunity_id": f"day-{i}", "canonical_match_id": "a",
         "forecast_at": clock, "p_base": probability,
         "forecast_available": True, "identity_verified": True}
        for i, (clock, probability) in enumerate(zip(clocks, [0.55, 0.45]))
    ])
    quotes = pd.DataFrame([
        {"quote_id": f"{i}-{book}", "canonical_match_id": "a",
         "bookmaker_id": book, "bookmaker": str(book), "quoted_at": clock,
         "odds_a": (2.5 if i == 0 else 5 / 3) if book == 1 else 2.0,
         "odds_b": (5 / 3 if i == 0 else 2.5) if book == 1 else 2.0,
         "identity_verified": True, "raw_a_is_canonical_a": True, "is_live": False}
        for i, clock in enumerate(clocks) for book in (1, 2, 3)
    ])
    _, candidates = _evaluate(
        opportunities, quotes, forecasts,
        {"policy": {}, "checkpoints": {"freshness_seconds": 300}},
    )
    qualified = candidates.loc[candidates.eligible].set_index("opportunity_id")
    assert qualified.loc["day-0", "side"] == "a"
    assert qualified.loc["day-1", "side"] == "b"



def test_same_day_native_forecast_contract_mismatch_is_rejected():
    forecasts = pd.DataFrame(
        [
            {"canonical_match_id": "a", "daily_state_cutoff_day": "2026-01-02",
             "daily_state_key_sha256": "same-key", "p_base": 0.31, "forecast_available": True},
            {"canonical_match_id": "a", "daily_state_cutoff_day": "2026-01-02",
             "daily_state_key_sha256": "same-key", "p_base": 0.72, "forecast_available": True},
        ]
    )
    try:
        _validate_daily_state_contract(forecasts)
    except ValueError as exc:
        assert "p_base" in str(exc)
    else:
        raise AssertionError("same-day native state mismatch must not silently select a forecast")

def test_random_schedule_is_unconditioned_on_eligibility_and_retains_missing_state_fixtures():
    opportunities = pd.DataFrame(
        [
            {"opportunity_id": "a0", "canonical_match_id": "a", "is_grid": True, "decision_at": "2026-01-01T00:00Z"},
            {"opportunity_id": "a1", "canonical_match_id": "a", "is_grid": True, "decision_at": "2026-01-01T02:00Z"},
            {"opportunity_id": "b0", "canonical_match_id": "b", "is_grid": True, "decision_at": "2026-01-01T00:00Z"},
            {"opportunity_id": "b1", "canonical_match_id": "b", "is_grid": True, "decision_at": "2026-01-01T02:00Z"},
            {"opportunity_id": "c-snapshot", "canonical_match_id": "c", "is_grid": False, "decision_at": "2026-01-01T00:00Z"},
        ]
    )
    # Fixture b has no eligible candidate state and fixture c has no grid
    # state; neither omission conditions or resamples the random schedule.
    schedule = choose_random_grid_checkpoints(opportunities, seed=20261022)
    assert set(schedule) == {"a", "b", "c"}
    assert schedule["a"] in {"a0", "a1"}
    assert schedule["b"] in {"b0", "b1"}
    assert schedule["c"] is None


def test_selected_rows_are_invariant_to_outcome_perturbation():
    opportunities = pd.DataFrame(
        [
            {"opportunity_id": "a0", "canonical_match_id": "a", "decision_at": "2026-01-01T00:00:00Z", "start_at": "2026-01-02T00:00:00Z"},
            {"opportunity_id": "a1", "canonical_match_id": "a", "decision_at": "2026-01-01T02:00:00Z", "start_at": "2026-01-02T00:00:00Z"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"opportunity_id": "a0", "canonical_match_id": "a", "model_ev": 0.08, "eligible": True, "y": 0},
            {"opportunity_id": "a1", "canonical_match_id": "a", "model_ev": 0.12, "eligible": True, "y": 1},
        ]
    )
    selected = select_timing_candidates(
        candidates, opportunities, policy="max_coverage_model_ev", band="0_48h"
    )
    changed = candidates.copy()
    changed["y"] = 1 - changed["y"]
    perturbed = select_timing_candidates(
        changed, opportunities, policy="max_coverage_model_ev", band="0_48h"
    )
    assert selected.opportunity_id.tolist() == perturbed.opportunity_id.tolist()


def test_fixture_duplication_never_creates_second_selected_bet():
    opportunities = pd.DataFrame(
        [
            {"opportunity_id": "a0", "canonical_match_id": "a", "decision_at": "2026-01-01T00:00:00Z", "start_at": "2026-01-02T00:00:00Z"},
            {"opportunity_id": "a1", "canonical_match_id": "a", "decision_at": "2026-01-01T02:00:00Z", "start_at": "2026-01-02T00:00:00Z"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"opportunity_id": "a0", "canonical_match_id": "a", "model_ev": 0.08, "eligible": True},
            {"opportunity_id": "a1", "canonical_match_id": "a", "model_ev": 0.12, "eligible": True},
        ]
    )
    duplicated = pd.concat([candidates, candidates.iloc[[1]]], ignore_index=True)
    selected = select_timing_candidates(
        duplicated, opportunities, policy="max_coverage_model_ev", band="0_48h"
    )
    assert selected.canonical_match_id.is_unique
    assert len(selected) <= 1


def test_band_edges_are_half_open_and_decision_must_precede_start():
    start = "2026-01-02T00:00:00Z"
    opportunities = pd.DataFrame(
        [
            {"decision_at": "2026-01-01T18:00:00Z", "start_at": start},  # 6h
            {"decision_at": "2026-01-01T12:00:00Z", "start_at": start},  # 12h
            {"decision_at": "2026-01-01T00:00:00Z", "start_at": start},  # 24h
            {"decision_at": "2025-12-31T00:00:00Z", "start_at": start},  # 48h
            {"decision_at": start, "start_at": start},  # not before start
        ]
    )
    assert band_membership(opportunities, "0_6h").tolist() == [False, False, False, False, False]
    assert band_membership(opportunities, "6_12h").tolist() == [True, False, False, False, False]
    assert band_membership(opportunities, "12_24h").tolist() == [False, True, False, False, False]
    assert band_membership(opportunities, "24_48h").tolist() == [False, False, True, False, False]
    assert band_membership(opportunities, "0_48h").tolist() == [True, True, True, False, False]


def test_unknown_winner_export_preserves_null_and_reserved_capital(tmp_path, monkeypatch):
    import json
    from src.analysis import c0_interval_runner as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    clock = pd.Timestamp("2026-01-02T14:00Z")
    opportunities = pd.DataFrame([{
        "opportunity_id": "pending", "canonical_match_id": 1,
        "decision_at": clock, "start_at": clock + pd.Timedelta(hours=4),
        "settle_at": clock + pd.Timedelta(days=1), "league": "LCK",
        "best_of": 3, "is_grid": True, "is_snapshot": True, "y": float("nan"),
    }])
    forecasts = pd.DataFrame([{
        "opportunity_id": "pending", "canonical_match_id": 1,
        "forecast_at": clock, "p_base": 0.55,
        "forecast_available": True, "identity_verified": True,
    }])
    quotes = pd.DataFrame([{
        "quote_id": book, "canonical_match_id": 1, "bookmaker_id": book,
        "bookmaker": str(book), "quoted_at": clock,
        "odds_a": 2.5 if book == 1 else 2.0,
        "odds_b": 5 / 3 if book == 1 else 2.0,
        "identity_verified": True, "raw_a_is_canonical_a": True, "is_live": False,
    } for book in (1, 2, 3)])
    binding = tmp_path / "binding.json"
    binding.write_text("{}")
    (tmp_path / "daily_equivalence_receipt.json").write_text("{}")
    protocol = {
        "output_root": "result", "policy": {"tax_rate": 0.12},
        "portfolio": {"initial_bankroll": 10000, "kelly_fraction": 0.25,
                      "max_stake": 150, "min_stake": 2, "max_portfolio_exposure": 0.25},
        "random": {"draws": 1, "seed": 20261022},
        "bootstrap": {"draws": 100, "seed": 20261022, "confidence_per_interval": 0.99},
        "checkpoints": {"freshness_seconds": 300}, "interpretation": "Synthetic smoke",
    }
    runner._run({
        "opportunities": opportunities, "forecasts": forecasts, "quotes": quotes,
        "evidence": {"fixture_accounting": opportunities[["canonical_match_id", "y"]]},
    }, protocol, binding)
    accounting = json.loads((tmp_path / "result/fixture_accounting.json").read_text())
    assert accounting[0]["y"] is None
    results = json.loads((tmp_path / "result/results.json").read_text())
    summary = results["timing_policies"]["0_6h"]["first_eligible_2h"]["summary"]
    assert summary["pending_bets_count"] == 1
    assert float(summary["final_reserved"]) > 0
    assert float(summary["profit"]) == 0


def test_snapshot_oracle_includes_non_grid_offers_but_two_hour_policy_does_not():
    opportunities = pd.DataFrame([
        {"opportunity_id": "grid", "canonical_match_id": "a", "is_grid": True,
         "decision_at": "2026-01-02T10:00Z", "start_at": "2026-01-02T15:00Z"},
        {"opportunity_id": "snapshot", "canonical_match_id": "a", "is_grid": False,
         "decision_at": "2026-01-02T10:55Z", "start_at": "2026-01-02T15:00Z"},
    ])
    candidates = opportunities.assign(eligible=True, model_ev=[0.06, 0.12])
    oracle = select_timing_candidates(
        candidates, opportunities, policy="max_coverage_model_ev", band="0_6h",
    )
    rolling = select_timing_candidates(
        candidates, opportunities, policy="first_eligible_2h", band="0_6h",
    )
    assert oracle.opportunity_id.tolist() == ["snapshot"]
    assert rolling.opportunity_id.tolist() == ["grid"]
