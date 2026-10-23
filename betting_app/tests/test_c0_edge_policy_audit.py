import json
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from src.analysis.c0_edge_policy_audit import (
    build_policy_configs,
    choose_first_eligible,
    vectorized_is_bet_eligible,
)
from decimal import Decimal

from src.analysis.c0_edge_policy_audit import BASE_POLICY, _summary_row
from betting_app.services.bet_qualification_service import (
    SIGNAL_GATE_KWARGS,
    is_bet_eligible,
)


def _scalar_rows(frame, policy):
    return [
        is_bet_eligible(
            prob_model=float(row.prob_model),
            odds=float(row.odds),
            prob_market_novig=float(row.prob_market_novig),
            prob_base=float(row.prob_base),
            base_prob_required=True,
            competition_tier=row.competition_tier,
            best_of=int(row.best_of),
            rating_disagreement=float(row.rating_disagreement),
            prob_market_close_novig=float(row.prob_market_close_novig),
            uncertainty_required=False,
            uncertainty_basis="mean",
            **policy,
        )[0]
        for row in frame.itertuples(index=False)
    ]


def _one_case(**updates):
    row = {
        "prob_model": 0.62,
        "odds": 2.0,
        "prob_market_novig": 0.55,
        "prob_base": 0.60,
        "competition_tier": "Tier-1 International",
        "best_of": 3,
        "rating_disagreement": 0.0,
        "prob_market_close_novig": 0.55,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_vectorized_gate_matches_scalar_helper_at_active_gate_boundaries():
    defaults = dict(SIGNAL_GATE_KWARGS)
    cases = [
        (
            _one_case(prob_model=(1 + 0.05) / (2.0 * 0.88)),
            {**defaults, "dynamic_hurdle": False},
        ),
        (_one_case(odds=4.0, prob_model=0.70), defaults),
        (
            _one_case(
                odds=2.0,
                prob_model=1.25 / (2.0 * 0.88),
                prob_base=1.25 / (2.0 * 0.88),
                prob_market_novig=1.25 / (2.0 * 0.88),
            ),
            defaults,
        ),
        (
            _one_case(odds=3.0, prob_model=0.5, prob_base=0.5, prob_market_novig=0.5),
            {
                **defaults,
                "max_ev_net": None,
                "max_ev_longshot": None,
                "wasserstein_epsilon": 0.5 - 1.0 / (3.0 * 0.88),
            },
        ),
        (
            _one_case(best_of=1, prob_model=0.70, prob_base=0.70, prob_market_novig=0.55),
            {**defaults, "max_bo1_market_gap": 0.15},
        ),
        (
            _one_case(
                competition_tier="Tier-1 International",
                prob_model=0.63,
                prob_base=0.63,
                prob_market_novig=0.55,
            ),
            {**defaults, "max_tier1_market_gap": 0.08},
        ),
        (
            _one_case(
                odds=2.0,
                prob_model=(1.0 + 0.05 + (0.2 / 2.2 * 0.08)) / (2.0 * 0.88),
                prob_base=(1.0 + 0.05 + (0.2 / 2.2 * 0.08)) / (2.0 * 0.88),
                prob_market_novig=(1.0 + 0.05 + (0.2 / 2.2 * 0.08)) / (2.0 * 0.88),
            ),
            {**defaults, "dynamic_hurdle": True, "min_ev_net": 0.05},
        ),
        (
            _one_case(odds=2.4, rating_disagreement=0.1),
            {
                **defaults,
                "universal_rating_guard": True,
                "max_rating_disagreement": 0.1,
            },
        ),
        (
            _one_case(prob_market_close_novig=0.535),
            {**defaults, "max_negative_clv_drift": -0.015},
        ),
        (
            _one_case(odds=3.6, prob_model=0.5, prob_base=0.5, prob_market_novig=0.5),
            {**defaults, "max_ev_net": None, "max_ev_longshot": 0.584},
        ),
        (
            _one_case(
                odds=3.0,
                best_of=1,
                prob_model=0.7,
                prob_base=0.7,
                prob_market_novig=0.7,
            ),
            {
                **defaults,
                "bo1_max_odds": 3.0,
                "max_ev_net": None,
                "max_ev_longshot": None,
            },
        ),
        (
            _one_case(odds=2.0, prob_model=0.6, prob_base=0.6, prob_market_novig=0.55),
            {**defaults, "max_coinflip_market_gap": 0.05},
        ),
        (
            _one_case(prob_base=0.70, prob_model=0.70, prob_market_novig=0.55),
            {
                **defaults,
                "max_logit_disagreement": abs(
                    np.log(0.7 / 0.3) - np.log(0.55 / 0.45)
                ),
            },
        ),
    ]
    for frame, policy in cases:
        assert vectorized_is_bet_eligible(frame, policy).tolist() == _scalar_rows(frame, policy)


def test_vectorized_gate_matches_scalar_helper_on_seeded_random_rows():
    rng = np.random.default_rng(20261022)
    count = 512
    frame = pd.DataFrame(
        {
            "prob_model": rng.uniform(0.08, 0.92, count),
            "odds": rng.uniform(1.25, 5.5, count),
            "prob_market_novig": rng.uniform(0.08, 0.92, count),
            "prob_base": rng.uniform(0.08, 0.92, count),
            "competition_tier": rng.choice(
                ["Tier-1 International", "Tier-1 Domestic", "tier2", None], count
            ),
            "best_of": rng.choice([1, 3, 5], count),
            "rating_disagreement": rng.uniform(0, 0.2, count),
            "prob_market_close_novig": rng.uniform(0.08, 0.92, count),
        }
    )
    policy = dict(SIGNAL_GATE_KWARGS)
    policy.update({"min_ev_net": 0.025, "wasserstein_epsilon": 0.01})
    assert vectorized_is_bet_eligible(frame, policy).tolist() == _scalar_rows(frame, policy)


def test_vectorized_gate_handles_threshold_neighbors_without_rounding():
    threshold = 0.08
    rows = pd.concat(
        [
            _one_case(prob_model=0.65, prob_base=0.65, prob_market_novig=0.65 - value)
            for value in (np.nextafter(threshold, 0), threshold, np.nextafter(threshold, 1))
        ],
        ignore_index=True,
    )
    policy = {**dict(SIGNAL_GATE_KWARGS), "max_tier1_market_gap": threshold}
    assert vectorized_is_bet_eligible(rows, policy).tolist() == _scalar_rows(rows, policy)




def test_vectorized_gate_accepts_empty_candidate_frame():
    columns = [
        "prob_model",
        "odds",
        "prob_market_novig",
        "prob_base",
        "competition_tier",
        "best_of",
        "rating_disagreement",
        "prob_market_close_novig",
    ]
    empty = pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    assert vectorized_is_bet_eligible(empty, dict(SIGNAL_GATE_KWARGS)).tolist() == []


def test_quote_triggered_selection_uses_first_observed_eligible_state():
    events = pd.DataFrame(
        [
            {
                "canonical_match_id": 77,
                "opportunity_id": "first-ready-rejected",
                "decision_at": pd.Timestamp("2026-06-01T10:00:00Z"),
                "eligible": False,
                "ev_net": 0.25,
                "side": "a",
                "odds": 2.0,
                "bookmaker_id": 1,
            },
            {
                "canonical_match_id": 77,
                "opportunity_id": "first-eligible",
                "decision_at": pd.Timestamp("2026-06-01T11:00:00Z"),
                "eligible": True,
                "ev_net": 0.07,
                "side": "b",
                "odds": 2.2,
                "bookmaker_id": 2,
            },
            {
                "canonical_match_id": 77,
                "opportunity_id": "future-best-ev",
                "decision_at": pd.Timestamp("2026-06-01T12:00:00Z"),
                "eligible": True,
                "ev_net": 0.80,
                "side": "a",
                "odds": 2.4,
                "bookmaker_id": 3,
            },
        ]
    )

    selected = choose_first_eligible(
        events,
        events["eligible"].to_numpy(),
        "quote_triggered_first_eligible",
    )

    assert selected.opportunity_id.tolist() == ["first-eligible"]
def test_same_clock_selection_prioritizes_net_ev_before_odds():
    events = pd.DataFrame(
        [
            {
                "canonical_match_id": 77,
                "opportunity_id": "higher-odds-lower-ev",
                "decision_at": pd.Timestamp("2026-06-01T11:00:00Z"),
                "ev_net": 0.06,
                "odds": 4.0,
                "bookmaker_id": 1,
                "side": "b",
            },
            {
                "canonical_match_id": 77,
                "opportunity_id": "lower-odds-higher-ev",
                "decision_at": pd.Timestamp("2026-06-01T11:00:00Z"),
                "ev_net": 0.08,
                "odds": 2.0,
                "bookmaker_id": 2,
                "side": "a",
            },
        ]
    )

    selected = choose_first_eligible(events, np.ones(len(events), dtype=bool), "quote_triggered_first_eligible")

    assert selected.opportunity_id.tolist() == ["lower-odds-higher-ev"]




def test_fixed_utc_phases_require_exact_hour_grid_not_nearby_snapshots():
    events = pd.DataFrame(
        [
            {
                "canonical_match_id": 88,
                "opportunity_id": "exact-even",
                "decision_at": pd.Timestamp("2026-06-01T04:00:00Z"),
                "ev_net": 0.1,
                "odds": 2.0,
                "bookmaker_id": 1,
                "side": "a",
            },
            {
                "canonical_match_id": 88,
                "opportunity_id": "nearby-not-grid",
                "decision_at": pd.Timestamp("2026-06-01T04:55:00Z"),
                "ev_net": 0.9,
                "odds": 2.0,
                "bookmaker_id": 1,
                "side": "a",
            },
            {
                "canonical_match_id": 88,
                "opportunity_id": "exact-odd",
                "decision_at": pd.Timestamp("2026-06-01T05:00:00Z"),
                "ev_net": 0.2,
                "odds": 2.0,
                "bookmaker_id": 1,
                "side": "b",
            },
        ]
    )
    eligible = np.ones(len(events), dtype=bool)

    even = choose_first_eligible(events, eligible, "utc_even_2h_first_eligible")
    odd = choose_first_eligible(events, eligible, "utc_odd_2h_first_eligible")

    assert even.opportunity_id.tolist() == ["exact-even"]
    assert odd.opportunity_id.tolist() == ["exact-odd"]






def test_canonical_compact_json_hash_survives_null_roundtrip():
    from src.analysis.c0_edge_alpha_audit import canonical_json_dumps, canonical_json_sha256

    value = {
        "unknown_winner": np.nan,
        "missing_clock": pd.NaT,
        "observed_at": pd.Timestamp("2026-08-20T04:00:00Z"),
        "nested": {"not_available": float("inf"), "finite": 0.25},
    }
    compact = canonical_json_dumps(value)
    decoded = json.loads(compact)
    assert decoded["unknown_winner"] is None
    assert decoded["missing_clock"] is None
    assert decoded["nested"]["not_available"] is None
    assert canonical_json_sha256(value) == canonical_json_sha256(decoded)
    assert canonical_json_dumps(decoded) == compact


def test_frozen_policy_search_contains_all_core_and_active_gate_variants():
    protocol_path = Path(
        "data/05_model_input/c0_edge_recovery/"
        "c0_edge_recovery_20261022_01/protocol.json"
    )
    if not protocol_path.is_file():
        pytest.skip("frozen research protocol is not present in this checkout")
    protocol = json.loads(protocol_path.read_text())
    configs = build_policy_configs(protocol)
    core = [config for config in configs if config["family"] == "core"]
    grid = protocol["policy_grid"]
    policy_keys = [key for key in configs[0] if key not in {"family", "alpha", "config_id"}]

    def signature(config):
        return (config["alpha"], *(config[key] for key in policy_keys))

    expected = set()
    for alpha in grid["core"]["alpha"]:
        for min_ev in grid["core"]["min_ev_net"]:
            for epsilon in grid["core"]["wasserstein_epsilon"]:
                config = dict(core[0])
                config.update({"alpha": alpha, "min_ev_net": min_ev, "wasserstein_epsilon": epsilon})
                expected.add(signature(config))
    base = next(
        config for config in core
        if config["alpha"] == grid["baseline_core"]["alpha"]
        and config["min_ev_net"] == grid["baseline_core"]["min_ev_net"]
        and config["wasserstein_epsilon"] == grid["baseline_core"]["wasserstein_epsilon"]
    )
    for alpha in protocol["alpha_grid"]:
        for axis in ("one_factor_at_each_alpha", "availability_sensitivities"):
            for gate, values in grid[axis].items():
                for value in values:
                    config = dict(base)
                    config.update(
                        {
                            "alpha": alpha,
                            "min_ev_net": grid["baseline_core"]["min_ev_net"],
                            "wasserstein_epsilon": grid["baseline_core"]["wasserstein_epsilon"],
                            gate: value,
                        }
                    )
                    expected.add(signature(config))

    assert len(core) == len(grid["core"]["alpha"]) * len(grid["core"]["min_ev_net"]) * len(
        grid["core"]["wasserstein_epsilon"]
    )
    assert {signature(config) for config in configs} == expected
    assert len({config["config_id"] for config in configs}) == len(configs)

def test_frozen_policy_grid_preserves_every_none_disabled_one_factor_value():
    protocol_path = Path(
        "data/05_model_input/c0_edge_recovery/"
        "c0_edge_recovery_20261022_01/protocol.json"
    )
    if not protocol_path.is_file():
        pytest.skip("frozen research protocol is not present in this checkout")
    protocol = json.loads(protocol_path.read_text())
    configs = build_policy_configs(protocol)
    one_factor = protocol["policy_grid"]["one_factor_at_each_alpha"]
    for alpha in protocol["alpha_grid"]:
        for gate, values in one_factor.items():
            for value in values:
                assert any(
                    config["alpha"] == alpha
                    and config["family"] == f"one_factor:{gate}"
                    and config[gate] == value
                    for config in configs
                )

def test_numeric_policy_identity_is_stable_across_protocol_json_number_types():
    root = Path("data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01")
    if not all((root / name).is_file() for name in ("protocol.json", "protocol_v2.json")):
        pytest.skip("frozen research protocols are not present in this checkout")
    protocol_v1 = json.loads((root / "protocol.json").read_text())
    protocol_v2 = json.loads((root / "protocol_v2.json").read_text())
    primary_v1 = build_policy_configs(protocol_v1)
    primary_v2 = build_policy_configs(protocol_v2)
    assert [config["config_id"] for config in primary_v1] == [
        config["config_id"] for config in primary_v2
    ]





def test_simulation_preserves_selected_event_clock_and_settled_profit():
    from src.analysis.c0_edge_policy_audit import _simulate

    decision = pd.Timestamp("2026-08-05T10:00:00Z")
    selected = pd.DataFrame({
        "opportunity_id": ["quote-triggered-17"],
        "canonical_match_id": [7],
        "decision_at": [decision],
        "start_at": [pd.Timestamp("2026-08-05T12:00:00Z")],
        "settle_at": [pd.Timestamp("2026-08-05T18:00:00Z")],
        "partition": ["ASSESS"],
        "side": ["a"],
        "prob_for_ev": [0.7],
        "odds": [2.5],
        "y": [1],
    })
    protocol = {
        "cutoff_at": "2026-08-01T00:00:00Z",
        "assessment_end_at": "2026-10-01T00:00:00Z",
        "portfolio": {
            "initial_bankroll": 1000.0,
            "tax_rate": 0.12,
            "kelly_fraction": 0.25,
            "max_stake": 150.0,
            "min_stake": 2.0,
            "max_portfolio_exposure": 0.25,
        },
    }

    result = _simulate(selected, "ASSESS", protocol)
    assert result["ledger"]["decision_at"].tolist() == [decision]
    assert result["ledger"]["opportunity_id"].tolist() == ["quote-triggered-17"]
    assert float(result["ledger"].iloc[0]["stake"]) == pytest.approx(112.5)
    assert float(result["ledger"].iloc[0]["realized_pnl"]) == pytest.approx(135.0)
    assert float(result["summary"]["roi"]) == pytest.approx(1.2)


def test_proper_scores_accept_a_continuous_alpha_off_the_frozen_grid():
    from src.analysis.c0_edge_policy_audit import _proper_scores

    rows = pd.DataFrame(
        {
            "canonical_match_id": [1, 2, 3, 4],
            "p_base": [0.8, 0.2, 0.8, 0.2],
            "market_prob_a": [0.6, 0.4, 0.6, 0.4],
            "y": [1.0, 0.0, 1.0, 0.0],
            "fixture_weight": [1.0] * 4,
        }
    )
    continuous = _proper_scores(rows, 0.42)
    grid_alpha = _proper_scores(rows, 0.5)
    assert continuous["log_loss"] is not None
    assert continuous["log_loss"] != grid_alpha["log_loss"]


def test_market_only_alpha_zero_does_not_require_or_impute_sports_base():
    from src.analysis.c0_edge_policy_audit import _with_alpha

    panel = pd.DataFrame(
        {
            "p_base": [np.nan],
            "prob_market_novig": [0.62],
            "odds": [2.0],
        }
    )
    scored = _with_alpha(panel, 0.0)
    assert scored["prob_model"].tolist() == [0.62]
    assert scored["p_base"].isna().all()



def test_cash_clock_does_not_authorize_boundary_excluded_train_actions():
    from src.analysis.c0_edge_policy_audit import _action_partition_rows

    cutoff = pd.Timestamp("2026-08-01T00:00:00Z")
    protocol = {
        "cutoff_at": cutoff.isoformat(),
        "assessment_end_at": "2026-10-01T00:00:00Z",
    }
    candidates = pd.DataFrame(
        {
            "canonical_match_id": [1, 2],
            "partition": ["TRAIN", "BOUNDARY_EXCLUDED"],
            "decision_at": [
                pd.Timestamp("2026-07-20T00:00:00Z"),
                pd.Timestamp("2026-07-20T00:00:00Z"),
            ],
            "label_available_at": [
                pd.Timestamp("2026-07-31T00:00:00Z"),
                pd.Timestamp("2026-08-02T00:00:00Z"),
            ],
            "cash_partition": ["ASSESS_CLOCK", "TRAIN_CLOCK"],
        }
    )
    assert _action_partition_rows(candidates, "TRAIN", protocol).canonical_match_id.tolist() == [1]


def test_daily_model_predictions_use_same_day_proof_and_side_oriented_gate_values():
    from src.analysis.c0_edge_policy_audit import (
        _apply_model_variant,
        _orient_model_probabilities,
        _with_alpha,
        vectorized_gate_passes,
    )

    events = pd.DataFrame({
        "canonical_match_id": [1, 1, 2],
        "decision_at": pd.to_datetime([
            "2026-07-20T01:00:00Z",
            "2026-07-20T01:00:00Z",
            "2026-07-21T01:00:00Z",
        ], utc=True),
        "side": ["a", "b", "a"],
    })
    daily = pd.DataFrame({
        "p_variant": [0.75, 0.75, 0.70],
        "requested_utc_day": ["2026-07-20", "2026-07-20", "2026-07-21"],
        "decision_at": pd.to_datetime([
            "2026-07-20T02:00:00Z",
            "2026-07-20T02:00:00Z",
            "2026-07-21T02:00:00Z",
        ], utc=True),
        "forecast_at": pd.to_datetime([
            "2026-07-20T00:00:00Z",
            "2026-07-20T00:00:00Z",
            "2026-07-20T00:00:00Z",
        ], utc=True),
        "usable_for_same_day_projection": [True, True, True],
        "strict_prior_feature_day_proof": [True, True, True],
        "daily_state_cutoff_day": ["2026-07-20", "2026-07-20", "2026-07-21"],
        "feature_source_max_day": ["2026-07-19", "2026-07-19", "2026-07-20"],
    }, index=pd.MultiIndex.from_tuples([
        ("1", "2026-07-20"),
        ("1", "2026-07-20"),
        ("2", "2026-07-21"),
    ]))
    canonical, candidate = _orient_model_probabilities(events, daily, "p_variant")
    assert canonical[:2].tolist() == [0.75, 0.75]
    assert candidate[:2].tolist() == [0.75, 0.25]
    assert np.isnan(canonical[2]) and np.isnan(candidate[2])

    panel = events.iloc[:2].copy()
    panel["p_base"] = [0.65, 0.65]
    panel["prob_base"] = [0.65, 0.35]
    panel["prob_market_novig"] = [0.60, 0.40]
    panel["prob_market_close_novig"] = [0.60, 0.40]
    panel["odds"] = [2.0, 2.0]
    model_variant = _apply_model_variant(panel, daily.iloc[:2], "p_variant")
    assert model_variant["p_base"].tolist() == [0.75, 0.75]
    assert model_variant["prob_base"].tolist() == [0.75, 0.25]
    hybrid = _with_alpha(model_variant, 0.5)
    assert hybrid["prob_model"].sum() == pytest.approx(1.0)
    market_control = _with_alpha(model_variant, 0.0)
    assert market_control["prob_model"].tolist() == pytest.approx([0.60, 0.40])
    logit_pass = vectorized_gate_passes(
        market_control, {"max_logit_disagreement": 0.5}
    )["logit_disagreement"]
    assert logit_pass.tolist() == [False, False]


def test_market_actions_partition_by_quote_readiness_not_native_or_cash_clock():
    from src.analysis.c0_edge_policy_audit import (
        _action_partition_rows,
        _market_ready_partition_panel,
    )

    events = pd.DataFrame({
        "canonical_match_id": [1, 2],
        "opportunity_id": ["market-train-boundary", "market-assess"],
        "decision_at": pd.to_datetime([
            "2026-07-20T10:00:00Z", "2026-08-05T10:00:00Z",
        ], utc=True),
        "start_at": pd.to_datetime([
            "2026-07-21T10:00:00Z", "2026-08-06T10:00:00Z",
        ], utc=True),
        "label_available_at": pd.to_datetime([
            "2026-08-02T00:00:00Z", "2026-08-10T00:00:00Z",
        ], utc=True),
        "y": [1.0, 0.0],
        "cash_partition": ["TRAIN_CLOCK", "TRAIN_CLOCK"],
    })
    protocol = {
        "cutoff_at": "2026-08-01T00:00:00Z",
        "assessment_end_at": "2026-10-01T00:00:00Z",
    }
    market = _market_ready_partition_panel(events, protocol)
    assert market["partition"].tolist() == ["BOUNDARY_EXCLUDED", "ASSESS"]
    assert _action_partition_rows(market, "TRAIN", protocol).empty
    assert _action_partition_rows(
        market, "ASSESS", protocol
    ).canonical_match_id.tolist() == [2]


def test_odd_market_quote_grid_does_not_require_or_fabricate_native_state(tmp_path):
    from src.analysis.c0_edge_policy_audit import _build_odd_grid_events

    decision = pd.Timestamp("2026-08-20T11:00:00Z")
    quotes = pd.DataFrame({
        "canonical_match_id": [1, 1, 1],
        "quote_id": [10, 11, 12],
        "bookmaker_id": [1, 2, 3],
        "bookmaker": ["a", "b", "c"],
        "quoted_at": [decision] * 3,
        "odds_a": [1.8, 1.9, 2.0],
        "odds_b": [2.1, 2.0, 1.9],
        "raw_a_is_canonical_a": [True] * 3,
        "identity_verified": [True] * 3,
        "is_live": [False] * 3,
    })
    quotes.to_parquet(tmp_path / "quotes.parquet", index=False)
    fixtures = pd.DataFrame({
        "canonical_match_id": [1],
        "start_at": [pd.Timestamp("2026-08-20T12:00:00Z")],
        "y": [1.0],
    })
    native = pd.DataFrame(columns=[
        "canonical_match_id", "forecast_available", "identity_verified",
        "forecast_at", "requested_utc_day", "daily_state_cutoff_day", "p_base",
    ])
    grid, provenance = _build_odd_grid_events(
        {
            "cohort": {
                "freshness_seconds": 300,
                "minimum_other_books": 2,
            }
        },
        fixtures,
        native,
        tmp_path,
        require_native=False,
    )
    target = grid.loc[grid["decision_at"].eq(decision)]
    assert len(target) == 6
    assert target["p_base"].isna().all()
    assert target["prob_base"].isna().all()
    assert provenance["native_state_required"] is False


def test_summary_stake_weighted_ev_error_accepts_decimal_ledger_values(monkeypatch):
    import src.analysis.c0_edge_policy_audit as policy

    monkeypatch.setattr(
        policy,
        "_monthly_portfolio_summary",
        lambda *args, **kwargs: {"near_ev_error_bet_weighted_95_ci": [-0.02, 0.02]},
    )
    config = {
        **BASE_POLICY,
        "config_id": "policy-00000001",
        "family": "core",
        "alpha": 0.5,
    }
    wallet = {
        "summary": {
            "bets_count": 1,
            "settled_bets_count": 1,
            "pending_bets_count": 0,
            "profit": Decimal("1.0"),
            "turnover": Decimal("10.0"),
            "settled_turnover": Decimal("10.0"),
            "roi": Decimal("0.1"),
            "final_cash": Decimal("101.0"),
            "final_reserved": Decimal("0"),
            "final_equity": Decimal("101.0"),
            "peak_reserved_fraction": Decimal("0.1"),
        },
        "ledger": pd.DataFrame({
            "settled": [True],
            "decision_at": [pd.Timestamp("2026-07-01T12:00:00Z")],
            "model_ev": [Decimal("0.20")],
            "realized_return": [Decimal("0.10")],
            "stake": [Decimal("10.0")],
        }),
        "execution_rejections": [],
    }
    result = _summary_row(
        config, "first_ready", "TRAIN", pd.DataFrame(), wallet, pd.DataFrame(),
        {"reliability_gates": {"near_ev_band": [-0.25, 0.25]}}, 1,
    )
    assert result["near_ev_error_stake_weighted"] == pytest.approx(0.1)



def test_same_day_future_model_state_is_rejected_across_datetime_resolutions():
    from src.analysis.c0_edge_policy_audit import _orient_model_probabilities

    events = pd.DataFrame({
        "side": ["a"],
        "decision_at": pd.array(
            ["2026-08-20T01:00:00Z"], dtype="datetime64[ns, UTC]"
        ),
    })
    daily = pd.DataFrame({
        "p_variant": [0.75],
        "requested_utc_day": ["2026-08-20"],
        "forecast_at": pd.array(
            ["2026-08-20T02:00:00Z"], dtype="datetime64[us, UTC]"
        ),
        "usable_for_same_day_projection": [True],
        "strict_prior_feature_day_proof": [True],
        "daily_state_cutoff_day": ["2026-08-20"],
        "feature_source_max_day": ["2026-08-19"],
    })

    canonical, selected_side = _orient_model_probabilities(events, daily, "p_variant")
    assert np.isnan(canonical[0])
    assert np.isnan(selected_side[0])
