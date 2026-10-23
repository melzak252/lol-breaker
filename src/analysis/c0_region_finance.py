"""Fixed-policy financial replay for the four C0/regional configurations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.analysis.c0_edge_policy_audit import (
    BASE_POLICY,
    ROOT,
    _action_partition_rows,
    _event_panel,
    _event_partition_panel,
    _load_or_build_odd_grid_events,
    _partition_fixtures,
    _selection_candidates,
    _summary_row,
    _wallet_export,
    _with_alpha,
    vectorized_gate_passes,
)


MODEL_COLUMNS = {
    "C0": "p_C0",
    "C0+regional": "p_C0_regional",
    "fresh20": "p_fresh20",
    "fresh20+regional": "p_fresh20_regional",
}
TIMINGS = (
    "first_ready",
    "quote_triggered_first_eligible",
    "utc_even_2h_first_eligible",
    "utc_odd_2h_first_eligible",
)
DAILY_PROOF_COLUMNS = (
    "canonical_match_id",
    "requested_utc_day",
    "decision_at",
    "forecast_at",
    "usable_for_same_day_projection",
    "strict_prior_feature_day_proof",
    "daily_state_cutoff_day",
    "feature_source_max_day",
    "model_query_matched",
)

FINANCIAL_OUTPUT_NAMES = (
    "financial_summaries.parquet",
    "financial_wallet_ledgers.parquet",
    "financial_wallets.json",
    "financial_gate_masks.parquet",
    "financial_event_attribution.parquet",
    "financial_selection_overlaps.parquet",
    "financial_selected_tickets.parquet",
    "financial_paired_monthly_contrasts.parquet",
    "financial_fixed_tickets_resized_stakes_effects.parquet",
    "financial_fixed_tickets_resized_stakes.parquet",
    "financial_baseline_parity.json",
    "financial_receipt.json",
)

BASELINE_POLICY_DIR = (
    ROOT / "data/08_reporting/c0_edge_recovery/c0_edge_recovery_20261022_01/policy"
)
PARITY_TOLERANCE_PLN = 1e-8
BASELINE_FINANCIAL_FIELDS = (
    "bets_count",
    "settled_bets_count",
    "pending_bets_count",
    "profit",
    "turnover",
    "settled_turnover",
    "final_cash",
    "final_reserved",
    "final_equity",
    "initial_bankroll",
)


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _numeric(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="raise").iloc[0])


def _baseline_parity_proof(
    wallets: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    wallet_path = BASELINE_POLICY_DIR / "locked_model_policy_wallets.json"
    comparison_path = BASELINE_POLICY_DIR / "locked_model_policy_comparisons.parquet"
    if not wallet_path.is_file() or not comparison_path.is_file():
        raise FileNotFoundError(
            f"accepted C0 baseline artifacts are missing: {wallet_path}, {comparison_path}"
        )
    accepted_wallets = json.loads(wallet_path.read_text(encoding="utf-8"))
    accepted_comparisons = pd.read_parquet(comparison_path)
    baseline_wallets = {
        (row["timing"], row["partition"]): row
        for row in accepted_wallets
        if row.get("role") == "C0:baseline"
    }
    baseline_comparison_rows = accepted_comparisons.loc[
        accepted_comparisons["model"].eq("C0")
        & accepted_comparisons["locked_policy_role"].eq("baseline")
    ]
    baseline_comparisons = {
        (row["timing"], row["partition"]): row
        for row in baseline_comparison_rows.to_dict("records")
    }
    new_wallets = {
        (row["timing"], row["partition"]): row
        for row in wallets
        if row.get("role") == "C0"
    }
    new_comparisons = {
        (row["timing"], row["partition"]): row
        for row in comparisons
        if row.get("model") == "C0"
    }
    keys = [
        (timing, partition) for timing in TIMINGS for partition in ("TRAIN", "ASSESS")
    ]
    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for key in keys:
        label = {"timing": key[0], "partition": key[1]}
        expected_wallet = baseline_wallets.get(key)
        actual_wallet = new_wallets.get(key)
        expected_summary = baseline_comparisons.get(key)
        actual_summary = new_comparisons.get(key)
        if any(
            value is None
            for value in (
                expected_wallet,
                actual_wallet,
                expected_summary,
                actual_summary,
            )
        ):
            failures.append(
                {**label, "error": "missing accepted or candidate C0 baseline row"}
            )
            continue

        def identity(row: Mapping[str, Any]) -> tuple[str, ...]:
            return tuple(
                str(_json_safe(row.get(field, "")))
                for field in (
                    "canonical_match_id",
                    "opportunity_id",
                    "bookmaker",
                    "side",
                    "decision_at",
                )
            )

        expected_ledger = sorted(identity(row) for row in expected_wallet["ledger"])
        actual_ledger = sorted(identity(row) for row in actual_wallet["ledger"])
        row_failures: list[str] = []
        if expected_ledger != actual_ledger:
            row_failures.append("accepted_ticket_identities")
        expected_by_id = {identity(row): row for row in expected_wallet["ledger"]}
        actual_by_id = {identity(row): row for row in actual_wallet["ledger"]}
        for ticket_id in set(expected_by_id) & set(actual_by_id):
            for field in ("stake", "realized_pnl"):
                if (
                    abs(
                        _numeric(expected_by_id[ticket_id][field])
                        - _numeric(actual_by_id[ticket_id][field])
                    )
                    > PARITY_TOLERANCE_PLN
                ):
                    row_failures.append(f"ticket_{field}")
                    break
            for field in ("odds", "settled"):
                if expected_by_id[ticket_id][field] != actual_by_id[ticket_id][field]:
                    row_failures.append(f"ticket_{field}")
                    break

        expected_pending = _numeric(expected_wallet["pending_funds"])
        actual_pending = _numeric(actual_wallet["pending_funds"])
        if abs(expected_pending - actual_pending) > PARITY_TOLERANCE_PLN:
            row_failures.append("pending_funds")
        expected_summary_wallet = expected_wallet["summary"]
        actual_summary_wallet = actual_wallet["summary"]
        for field in BASELINE_FINANCIAL_FIELDS:
            if (
                field not in expected_summary_wallet
                or field not in actual_summary_wallet
            ):
                row_failures.append(f"wallet_summary_missing_{field}")
            elif field.endswith("_count"):
                if int(expected_summary_wallet[field]) != int(
                    actual_summary_wallet[field]
                ):
                    row_failures.append(f"wallet_summary_{field}")
            elif (
                abs(
                    _numeric(expected_summary_wallet[field])
                    - _numeric(actual_summary_wallet[field])
                )
                > PARITY_TOLERANCE_PLN
            ):
                row_failures.append(f"wallet_summary_{field}")

        for baseline_field, candidate_field in (
            ("selected_candidate_count", "selected_candidate_count"),
            ("bets", "bets"),
            ("settled_bets", "settled_bets"),
            ("pending_bets", "pending_bets"),
        ):
            if int(expected_summary[baseline_field]) != int(
                actual_summary[candidate_field]
            ):
                row_failures.append(f"comparison_{candidate_field}")
        for baseline_field, candidate_field in (
            ("net_profit", "net_profit"),
            ("turnover", "turnover"),
            ("settled_turnover", "settled_turnover"),
            ("final_cash", "final_cash"),
            ("final_reserved", "final_reserved"),
            ("final_equity", "final_equity"),
        ):
            if (
                abs(
                    _numeric(expected_summary[baseline_field])
                    - _numeric(actual_summary[candidate_field])
                )
                > PARITY_TOLERANCE_PLN
            ):
                row_failures.append(f"comparison_{candidate_field}")
        checks.append(
            {
                **label,
                "accepted_ticket_count": len(expected_ledger),
                "candidate_ticket_count": len(actual_ledger),
                "status": "PASS" if not row_failures else "FAIL",
                "failed_checks": sorted(set(row_failures)),
            }
        )
        if row_failures:
            failures.append({**label, "failed_checks": sorted(set(row_failures))})
    return {
        "schema": "c0_financial_baseline_parity_v1",
        "status": "PASS" if not failures else "FAIL",
        "baseline_wallets_path": str(wallet_path),
        "baseline_wallets_sha256": hashlib.sha256(wallet_path.read_bytes()).hexdigest(),
        "baseline_comparisons_path": str(comparison_path),
        "baseline_comparisons_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "numeric_tolerance_pln": PARITY_TOLERANCE_PLN,
        "configurations_checked": len(keys),
        "checks": checks,
        "failures": failures,
    }


def _require_fresh_financial_outputs(output: Path) -> None:
    existing = [name for name in FINANCIAL_OUTPUT_NAMES if (output / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing financial outputs in {output}: {existing}"
        )


def _require_existing_odd_grid_cache(edge_protocol: Mapping[str, Any]) -> None:
    cache_dir = ROOT / "data/04_feature" / str(edge_protocol["run_id"]) / "policy"
    cache_paths = (
        cache_dir / "odd_grid_native.parquet",
        cache_dir / "odd_grid_native_receipt.json",
    )
    missing = [str(path) for path in cache_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Refusing to build or mutate the sealed odd-grid cache; missing bound "
            f"cache artifact/receipt: {missing}"
        )


def _as_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return pd.read_parquet(value) if isinstance(value, (str, Path)) else value.copy()


def _utc_day(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")


def replay_candidate_events(
    events: pd.DataFrame,
    daily_predictions: pd.DataFrame,
    *,
    probability_column: str,
    timing: str,
    policy: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    sports_alpha: float = 0.5,
    prepared_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Replay causally proven daily canonical-A probabilities over raw quote events.

    All qualification probabilities and EV values are recomputed here. Invalid,
    missing, unavailable, or future states abstain; their quote events stay in the
    returned panel and are represented in gate masks/rejection reasons.
    """
    if timing not in TIMINGS:
        raise ValueError(f"unknown timing policy: {timing}")
    event_required = {
        "opportunity_id",
        "canonical_match_id",
        "decision_at",
        "side",
        "odds",
    }
    if not {"market_prob", "market_prob_a"}.intersection(events.columns):
        raise ValueError(
            "event panel lacks the saved raw market probability needed for fixed alpha blending"
        )
    missing_events = event_required.difference(events.columns)
    if missing_events:
        raise ValueError(
            f"event panel lacks required identity/price fields: {sorted(missing_events)}"
        )
    missing_proofs = set(DAILY_PROOF_COLUMNS).difference(daily_predictions.columns)
    if missing_proofs:
        raise ValueError(
            f"daily predictions lack causal-key/proof fields: {sorted(missing_proofs)}"
        )
    if not (0.0 <= float(sports_alpha) <= 1.0):
        raise ValueError("sports_alpha must be in [0, 1]")
    if (
        "odds" not in events
        or pd.to_numeric(events["odds"], errors="coerce").isna().any()
    ):
        raise ValueError(
            "every replay event requires its saved decimal price; prices are never inferred"
        )

    predictions = daily_predictions.copy()
    predictions["_key_id"] = predictions["canonical_match_id"].astype(str)
    predictions["_key_day"] = _utc_day(predictions["requested_utc_day"])
    if predictions.duplicated(["_key_id", "_key_day"]).any():
        raise ValueError(
            "daily prediction keys must be unique by fixture and requested UTC day"
        )
    if probability_column not in predictions:
        predictions[probability_column] = np.nan
    panel = events.copy()
    panel["_key_id"] = panel["canonical_match_id"].astype(str)
    event_times = pd.to_datetime(panel["decision_at"], utc=True, errors="coerce")
    panel["_key_day"] = event_times.dt.strftime("%Y-%m-%d")
    prediction_index = predictions.set_index(["_key_id", "_key_day"], drop=False)
    key_tuples = list(zip(panel["_key_id"], panel["_key_day"]))
    mapped = prediction_index.reindex(
        pd.MultiIndex.from_tuples(key_tuples, names=["_key_id", "_key_day"])
    )
    canonical = pd.to_numeric(mapped[probability_column], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    requested = pd.to_datetime(mapped["requested_utc_day"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(mapped["daily_state_cutoff_day"], utc=True, errors="coerce")
    feature = pd.to_datetime(
        mapped["feature_source_max_day"], utc=True, errors="coerce"
    )
    forecast = pd.to_datetime(mapped["forecast_at"], utc=True, errors="coerce")
    model_decision = pd.to_datetime(mapped["decision_at"], utc=True, errors="coerce")
    event_ns = event_times.array.as_unit("ns").asi8
    requested_ns = requested.array.as_unit("ns").asi8
    cutoff_ns = cutoff.array.as_unit("ns").asi8
    feature_ns = feature.array.as_unit("ns").asi8
    forecast_ns = forecast.array.as_unit("ns").asi8
    model_decision_ns = model_decision.array.as_unit("ns").asi8
    nat = np.iinfo(np.int64).min
    usable = mapped["usable_for_same_day_projection"].eq(True).to_numpy()
    prior_proof = mapped["strict_prior_feature_day_proof"].eq(True).to_numpy()
    matched = mapped["model_query_matched"].eq(True).to_numpy()
    causal = (
        usable
        & prior_proof
        & matched
        & np.isfinite(canonical)
        & (canonical > 0.0)
        & (canonical < 1.0)
        & (requested_ns != nat)
        & (requested_ns == cutoff_ns)
        & (feature_ns != nat)
        & (feature_ns < requested_ns)
        & (forecast_ns != nat)
        & (model_decision_ns != nat)
        & (event_ns != nat)
        & (forecast_ns <= event_ns)
        & (model_decision_ns <= event_ns)
        & (_utc_day(mapped["forecast_at"]).to_numpy() == panel["_key_day"].to_numpy())
        & (
            _utc_day(mapped["requested_utc_day"]).to_numpy()
            == panel["_key_day"].to_numpy()
        )
        & (_utc_day(mapped["decision_at"]).to_numpy() == panel["_key_day"].to_numpy())
    )
    canonical[~causal] = np.nan
    side_b = (
        panel["side"]
        .astype(str)
        .str.lower()
        .isin(("b", "team_b", "away", "2"))
        .to_numpy()
    )
    oriented = np.where(side_b, 1.0 - canonical, canonical)

    fixture_columns = [
        c
        for c in (
            "canonical_match_id",
            "y",
            "start_at",
            "settle_at",
            "result_recorded_at",
            "label_available_at",
            "partition",
            "cash_partition",
            "cash_decision_at",
            "cash_clock_basis",
        )
        if c in panel
    ]
    fixtures = panel[fixture_columns].drop_duplicates("canonical_match_id")
    panel = _event_panel(panel, fixtures)
    panel["p_base"] = canonical
    panel["prob_base"] = oriented
    panel = _with_alpha(panel, sports_alpha)
    masks = vectorized_gate_passes(panel, policy)
    masks["causal_daily_state"] = causal
    eligible = np.logical_and.reduce(
        [np.asarray(mask, dtype=bool) for mask in masks.values()]
    )
    first_rows = prepared_rows if prepared_rows is not None else panel
    selected = _selection_candidates(panel, first_rows, eligible, timing)
    from src.analysis.unified_evaluation_engine import simulate_operational_portfolio

    wallet = simulate_operational_portfolio(
        selected,
        selected,
        initial_bankroll=portfolio["initial_bankroll"],
        tax_rate=policy["tax_rate"],
        fraction=portfolio["kelly_fraction"],
        max_stake=portfolio["max_stake"],
        min_stake=portfolio["min_stake"],
        max_portfolio_exposure=portfolio["max_portfolio_exposure"],
    )
    fail_labels = np.full(len(panel), "eligible", dtype=object)
    for gate_name, passed in masks.items():
        fail = ~np.asarray(passed, dtype=bool) & (fail_labels == "eligible")
        fail_labels[fail] = gate_name
    panel["first_rejection_gate"] = fail_labels
    panel["source_model_available"] = causal
    return {
        "panel": panel.drop(columns=["_key_id", "_key_day"], errors="ignore"),
        "gate_masks": masks,
        "rejections": fail_labels,
        "eligible": eligible,
        "selected": selected,
        "wallet": wallet,
    }


def _load_edge_protocol(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path_value = protocol.get("inputs", {}).get("edge_protocol")
    if not path_value:
        raise ValueError("factorial protocol must bind inputs.edge_protocol")
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"bound source edge protocol is unavailable: {path}")
    edge = json.loads(path.read_text(encoding="utf-8"))
    edge["_protocol_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        edge.get("run_id")
        != protocol.get("inputs", {}).get("edge_root", "").rstrip("/").split("/")[-1]
    ):
        # The protocol path is the binding; permit root layout names while refusing another run.
        expected = Path(protocol["inputs"]["edge_root"]).name
        if edge.get("run_id") != expected:
            raise ValueError(
                "source edge protocol run identity differs from the factorial binding"
            )
    return edge, path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if (
        pd.isna(value)
        if not isinstance(value, (dict, list, tuple, str, bytes))
        else False
    ):
        return None
    return value


def run_financial_comparison(
    protocol: Mapping[str, Any],
    daily_predictions: pd.DataFrame | str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run all 32 fixed model × timing × partition financial comparisons."""
    from src.analysis.c0_edge_alpha_audit import _load_prepared

    output = Path(output_dir)
    _require_fresh_financial_outputs(output)
    edge_protocol, edge_protocol_path = _load_edge_protocol(protocol)
    source_finance = protocol["financial"]
    cohort = _load_prepared(edge_protocol, include_events=True)
    if cohort is None:
        raise FileNotFoundError("receipt-validated edge prepared cohort is unavailable")
    if len(cohort["fixtures"]) != 1771:
        raise ValueError(
            f"source edge universe must retain 1,771 fixtures, found {len(cohort['fixtures'])}"
        )
    if len(cohort["daily_native_states"]) != 4983:
        raise ValueError(
            "source daily-state denominator changed; expected all 4,983 requested states"
        )
    if (
        float(source_finance["sports_alpha"]) != 0.5
        or float(source_finance["wasserstein_epsilon"]) != 0.025
        or float(source_finance["tax_rate"]) != 0.12
    ):
        raise ValueError(
            "factorial financial settings differ from the frozen C0 baseline"
        )
    policy = dict(BASE_POLICY)
    policy["tax_rate"] = float(source_finance["tax_rate"])
    policy["min_ev_net"] = float(source_finance["min_ev_net"])
    policy["wasserstein_epsilon"] = float(source_finance["wasserstein_epsilon"])
    if policy != dict(BASE_POLICY):
        raise ValueError("frozen baseline policy differs from the source protocol")

    predictions = _as_frame(daily_predictions)
    missing = set(DAILY_PROOF_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(
            f"daily model artifact lacks key/proof fields: {sorted(missing)}"
        )
    if len(predictions) != 1126:
        raise ValueError(
            f"daily prediction artifact must retain all 1,126 states, found {len(predictions)}"
        )
    timing_root = Path(edge_protocol["inputs"]["timing_root"])
    if not timing_root.is_absolute():
        timing_root = ROOT / timing_root
    _require_existing_odd_grid_cache(edge_protocol)
    odd_events, odd_provenance = _load_or_build_odd_grid_events(
        edge_protocol,
        cohort["fixtures"],
        cohort["daily_native_states"],
        timing_root,
        cohort,
        require_native=True,
    )
    observed = _event_partition_panel(
        _event_panel(cohort["events"], cohort["fixtures"]), cohort["rows"]
    )
    odd = _event_partition_panel(
        _event_panel(odd_events, cohort["fixtures"]), cohort["rows"]
    )
    panels = {"observed": observed, "odd_grid": odd}
    rows_out: list[dict[str, Any]] = []
    wallets: list[dict[str, Any]] = []
    selections: dict[tuple[str, str, str], pd.DataFrame] = {}
    wallet_ledgers: dict[tuple[str, str, str], pd.DataFrame] = {}
    replay_panels: dict[tuple[str, str], pd.DataFrame] = {}
    gate_records: list[pd.DataFrame] = []
    availability_records: list[pd.DataFrame] = []
    for model, column in MODEL_COLUMNS.items():
        for timing in TIMINGS:
            source_events = panels[
                "odd_grid" if timing == "utc_odd_2h_first_eligible" else "observed"
            ]
            replay = replay_candidate_events(
                source_events,
                predictions,
                probability_column=column,
                timing=timing,
                policy=policy,
                portfolio=source_finance["portfolio"],
                sports_alpha=float(source_finance["sports_alpha"]),
                prepared_rows=cohort["rows"] if timing == "first_ready" else None,
            )
            replay_panels[(model, timing)] = replay["panel"]
            selected = replay["selected"]
            for partition in ("TRAIN", "ASSESS"):
                universe = _partition_fixtures(cohort["fixtures"], partition)
                partition_selected = _action_partition_rows(
                    selected, partition, edge_protocol
                )
                from src.analysis.unified_evaluation_engine import (
                    simulate_operational_portfolio,
                )

                portfolio = source_finance["portfolio"]
                wallet = simulate_operational_portfolio(
                    partition_selected,
                    partition_selected,
                    initial_bankroll=portfolio["initial_bankroll"],
                    tax_rate=source_finance["tax_rate"],
                    fraction=portfolio["kelly_fraction"],
                    max_stake=portfolio["max_stake"],
                    min_stake=portfolio["min_stake"],
                    max_portfolio_exposure=portfolio["max_portfolio_exposure"],
                )
                config_id = hashlib.sha256(
                    f"{model}:{timing}:{partition}".encode()
                ).hexdigest()[:16]
                config = {
                    "config_id": config_id,
                    "family": "fixed_baseline",
                    "alpha": 0.5,
                    **{key: policy[key] for key in policy},
                }
                summary = _summary_row(
                    config,
                    timing,
                    partition,
                    partition_selected,
                    wallet,
                    universe,
                    edge_protocol,
                    1,
                    fixture_month_universe=universe,
                )
                summary.update(
                    {
                        "model": model,
                        "probability_column": column,
                        "source_event_set": "odd_grid"
                        if timing.endswith("odd_2h_first_eligible")
                        else "observed",
                        "source_unavailable_event_rows": int(
                            (~replay["panel"]["source_model_available"]).sum()
                        ),
                        "fixture_denominator": len(cohort["fixtures"]),
                        "partition_fixture_count": len(universe),
                    }
                )
                rows_out.append(summary)
                wallets.append(_wallet_export(model, timing, partition, wallet))
                selections[(model, timing, partition)] = partition_selected
                ledger = wallet["ledger"].copy()
                if not ledger.empty:
                    ledger["model"] = model
                    ledger["timing"] = timing
                    ledger["partition"] = partition
                wallet_ledgers[(model, timing, partition)] = ledger
            selected_ids = set(
                selected.get("canonical_match_id", pd.Series(dtype=object)).astype(str)
            )
            event_availability = replay["panel"][
                [
                    "opportunity_id",
                    "canonical_match_id",
                    "decision_at",
                    "source_model_available",
                    "first_rejection_gate",
                ]
            ].copy()
            event_availability["model"] = model
            event_availability["timing"] = timing
            event_availability["selected_fixture"] = (
                event_availability["canonical_match_id"].astype(str).isin(selected_ids)
            )
            availability_records.append(event_availability)
            masks_frame = pd.DataFrame(
                {
                    "opportunity_id": replay["panel"]["opportunity_id"],
                    "canonical_match_id": replay["panel"]["canonical_match_id"],
                    "decision_at": replay["panel"]["decision_at"],
                }
            )
            for name, mask in replay["gate_masks"].items():
                masks_frame[name] = np.asarray(mask, dtype=bool)
            masks_frame["model"] = model
            masks_frame["timing"] = timing
            gate_records.append(masks_frame)

    summary_frame = pd.DataFrame(rows_out)
    if len(summary_frame) != 32:
        raise ValueError(
            f"financial comparison must produce 32 rows, got {len(summary_frame)}"
        )
    parity_proof = _baseline_parity_proof(wallets, rows_out)
    _write_json_atomic(output / "financial_baseline_parity.json", parity_proof)
    if parity_proof["status"] != "PASS":
        raise ValueError(
            "C0 financial result failed accepted-baseline parity: "
            f"{parity_proof['failures']}"
        )
    summary_frame.to_parquet(output / "financial_summaries.parquet", index=False)
    pd.concat(gate_records, ignore_index=True).to_parquet(
        output / "financial_gate_masks.parquet", index=False
    )
    pd.concat(availability_records, ignore_index=True).to_parquet(
        output / "financial_event_attribution.parquet", index=False
    )
    ledger_records = [
        row for frame in wallet_ledgers.values() for row in frame.to_dict("records")
    ]
    pd.DataFrame(ledger_records).to_parquet(
        output / "financial_wallet_ledgers.parquet", index=False
    )
    _write_json_atomic(output / "financial_wallets.json", wallets)

    overlap_rows: list[dict[str, Any]] = []
    for timing in TIMINGS:
        for partition in ("TRAIN", "ASSESS"):
            fixture_sets = {
                model: set(
                    frame.get("canonical_match_id", pd.Series(dtype=object)).astype(str)
                )
                for model in MODEL_COLUMNS
                for frame in [selections[(model, timing, partition)]]
            }
            ticket_sets = {}
            for model in MODEL_COLUMNS:
                frame = selections[(model, timing, partition)]
                ticket_keys = [
                    name
                    for name in (
                        "canonical_match_id",
                        "opportunity_id",
                        "quote_id",
                        "bookmaker_id",
                        "side",
                    )
                    if name in frame
                ]
                ticket_sets[model] = set(
                    tuple(str(value) for value in row)
                    for row in frame[ticket_keys].itertuples(index=False, name=None)
                )
            baseline = fixture_sets["C0"]
            baseline_tickets = ticket_sets["C0"]
            for model, selected_ids in fixture_sets.items():
                selected_tickets = ticket_sets[model]
                overlap_rows.append(
                    {
                        "timing": timing,
                        "partition": partition,
                        "model": model,
                        "selected_fixtures": len(selected_ids),
                        "c0_selected_fixtures": len(baseline),
                        "overlap_with_c0": len(selected_ids & baseline),
                        "only_model": len(selected_ids - baseline),
                        "only_c0": len(baseline - selected_ids),
                        "jaccard_with_c0": len(selected_ids & baseline)
                        / len(selected_ids | baseline)
                        if selected_ids | baseline
                        else None,
                        "selected_ticket_count": len(selected_tickets),
                        "ticket_overlap_with_c0": len(
                            selected_tickets & baseline_tickets
                        ),
                        "only_model_tickets": len(selected_tickets - baseline_tickets),
                        "only_c0_tickets": len(baseline_tickets - selected_tickets),
                    }
                )
    overlap_frame = pd.DataFrame(overlap_rows)
    overlap_frame.to_parquet(
        output / "financial_selection_overlaps.parquet", index=False
    )
    selection_frames = []
    for (model, timing, partition), frame in selections.items():
        if frame.empty:
            continue
        out = frame.copy()
        out["model"] = model
        out["timing"] = timing
        out["partition"] = partition
        selection_frames.append(out)
    selected_frame = (
        pd.concat(selection_frames, ignore_index=True)
        if selection_frames
        else pd.DataFrame()
    )
    selected_frame.to_parquet(
        output / "financial_selected_tickets.parquet", index=False
    )
    from src.analysis.c0_edge_policy_audit import _paired_monthly_contrast

    paired_rows: list[dict[str, Any]] = []
    fixed_selection_rows: list[dict[str, Any]] = []
    fixed_ticket_frames: list[pd.DataFrame] = []
    for timing in TIMINGS:
        for partition in ("TRAIN", "ASSESS"):
            universe = _partition_fixtures(cohort["fixtures"], partition)
            c0_ledger = wallet_ledgers[("C0", timing, partition)]
            for model in MODEL_COLUMNS:
                if model != "C0":
                    contrast = _paired_monthly_contrast(
                        wallet_ledgers[(model, timing, partition)],
                        c0_ledger,
                        universe,
                        edge_protocol,
                        family_size=3,
                        seed_offset=int(
                            hashlib.sha256(
                                f"{model}:{timing}:{partition}".encode()
                            ).hexdigest()[:8],
                            16,
                        ),
                    )
                    paired_rows.append(
                        {
                            "model": model,
                            "control_model": "C0",
                            "timing": timing,
                            "partition": partition,
                            **contrast,
                        }
                    )
                c0_selection = selections[("C0", timing, partition)]
                model_panel = replay_panels[(model, timing)]
                identity = [
                    key
                    for key in ("opportunity_id", "quote_id", "bookmaker_id", "side")
                    if key in c0_selection and key in model_panel
                ]
                if not identity:
                    raise ValueError(
                        "fixed-selection attribution lacks ticket identity columns"
                    )
                fixed = model_panel.merge(
                    c0_selection[identity].drop_duplicates(),
                    on=identity,
                    how="inner",
                    validate="many_to_one",
                )
                fixed = _action_partition_rows(fixed, partition, edge_protocol)
                from src.analysis.unified_evaluation_engine import (
                    simulate_operational_portfolio,
                )

                portfolio = source_finance["portfolio"]
                fixed_wallet = simulate_operational_portfolio(
                    fixed,
                    fixed,
                    initial_bankroll=portfolio["initial_bankroll"],
                    tax_rate=source_finance["tax_rate"],
                    fraction=portfolio["kelly_fraction"],
                    max_stake=portfolio["max_stake"],
                    min_stake=portfolio["min_stake"],
                    max_portfolio_exposure=portfolio["max_portfolio_exposure"],
                )
                fixed_summary = fixed_wallet["summary"]
                fixed_selection_rows.append(
                    {
                        "model": model,
                        "timing": timing,
                        "partition": partition,
                        "selection_basis": "FIXED_TICKETS_RESIZED_STAKES",
                        "baseline_ticket_identity": True,
                        "stakes_fixed": False,
                        "interpretation": (
                            "C0 ticket identities repriced with each model's probability and resized via Kelly; "
                            "portfolio PnL is not a causal probability-only effect."
                        ),
                        "ticket_count": len(fixed),
                        "bets": fixed_summary["bets_count"],
                        "settled_bets": fixed_summary["settled_bets_count"],
                        "pending_bets": fixed_summary["pending_bets_count"],
                        "profit": float(fixed_summary["profit"]),
                        "turnover": float(fixed_summary["turnover"]),
                        "roi": float(fixed_summary["roi"])
                        if fixed_summary["roi"] is not None
                        else None,
                        "final_cash": float(fixed_summary["final_cash"]),
                        "final_reserved": float(fixed_summary["final_reserved"]),
                    }
                )
                if not fixed.empty:
                    fixed = fixed.copy()
                    fixed["model"] = model
                    fixed["timing"] = timing
                    fixed["partition"] = partition
                    fixed["selection_basis"] = "FIXED_TICKETS_RESIZED_STAKES"
                    fixed["baseline_ticket_identity"] = True
                    fixed["stakes_fixed"] = False
                    fixed_ticket_frames.append(fixed)
    pd.DataFrame(paired_rows).to_parquet(
        output / "financial_paired_monthly_contrasts.parquet", index=False
    )
    pd.DataFrame(fixed_selection_rows).to_parquet(
        output / "financial_fixed_tickets_resized_stakes_effects.parquet", index=False
    )
    fixed_tickets = (
        pd.concat(fixed_ticket_frames, ignore_index=True)
        if fixed_ticket_frames
        else pd.DataFrame()
    )
    fixed_tickets.to_parquet(
        output / "financial_fixed_tickets_resized_stakes.parquet", index=False
    )

    receipt = {
        "schema": "c0_region_finance_v1",
        "status": "COMPLETED_FIXED_POLICY_REPLAY",
        "run_id": protocol["run_id"],
        "edge_run_id": edge_protocol["run_id"],
        "edge_protocol_path": str(edge_protocol_path),
        "edge_protocol_sha256": hashlib.sha256(
            edge_protocol_path.read_bytes()
        ).hexdigest(),
        "prepared_cohort_receipt": cohort["receipt"],
        "odd_grid_provenance": odd_provenance,
        "baseline_parity": {
            "status": parity_proof["status"],
            "artifact": str(output / "financial_baseline_parity.json"),
            "sha256": hashlib.sha256(
                (output / "financial_baseline_parity.json").read_bytes()
            ).hexdigest(),
        },
        "summary_rows": len(summary_frame),
        "models": list(MODEL_COLUMNS),
        "timings": list(TIMINGS),
        "partition_counts": {
            partition: int(len(_partition_fixtures(cohort["fixtures"], partition)))
            for partition in ("TRAIN", "ASSESS")
        },
        "policy": policy,
        "sports_alpha": source_finance["sports_alpha"],
        "fixed_ticket_attribution": {
            "selection_basis": "FIXED_TICKETS_RESIZED_STAKES",
            "baseline_ticket_identity": True,
            "stakes_fixed": False,
            "causal_probability_only_pnl_claim": False,
            "fixed_stake_ev_overstatement_diagnostic": "NOT_COMPUTED",
        },
        "artifacts": {
            name: str(output / name)
            for name in (
                "financial_summaries.parquet",
                "financial_wallet_ledgers.parquet",
                "financial_wallets.json",
                "financial_gate_masks.parquet",
                "financial_event_attribution.parquet",
                "financial_selection_overlaps.parquet",
                "financial_selected_tickets.parquet",
                "financial_paired_monthly_contrasts.parquet",
                "financial_fixed_tickets_resized_stakes_effects.parquet",
                "financial_fixed_tickets_resized_stakes.parquet",
                "financial_baseline_parity.json",
            )
        },
        "unavailable_probability_rows": {
            model: int(
                pd.to_numeric(
                    predictions.get(column, pd.Series(dtype=float)), errors="coerce"
                )
                .isna()
                .sum()
            )
            for model, column in MODEL_COLUMNS.items()
        },
        "claims": {
            "source_authentication": False,
            "clv": False,
            "true_uncertainty_bound": False,
        },
    }
    _write_json_atomic(output / "financial_receipt.json", receipt)
    return receipt
