"""Historical interval-timing analysis for unchanged native C0 policy gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.c0_policy_evaluation import evaluate_policy_candidates
from src.analysis.c0_policy_runner import _closure as _prior_closure
from src.analysis.c0_policy_runner import _json_write
from src.analysis.unified_evaluation_engine import simulate_operational_portfolio

ROOT = Path(__file__).resolve().parents[2]
BANDS = {"0_6h": (0.0, 6.0), "6_12h": (6.0, 12.0), "12_24h": (12.0, 24.0),
         "24_48h": (24.0, 48.0), "0_48h": (0.0, 48.0)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def band_membership(opportunities: pd.DataFrame, band: str) -> pd.Series:
    """Return half-open time-to-start band membership with strict pre-start clocks."""
    if band not in BANDS:
        raise ValueError(f"unknown interval band: {band}")
    lower, upper = BANDS[band]
    decision = pd.to_datetime(opportunities.decision_at, utc=True, errors="coerce")
    start = pd.to_datetime(opportunities.start_at, utc=True, errors="coerce")
    hours = (start - decision).dt.total_seconds() / 3600
    return (decision < start) & hours.ge(lower) & hours.lt(upper)


def choose_random_grid_checkpoints(
    opportunities: pd.DataFrame, *, seed: int, band: str | None = None
) -> dict:
    """Uniformly schedule one grid checkpoint per fixture, before gates/prices/outcomes."""
    grid = opportunities.loc[opportunities.is_grid.fillna(False)].copy()
    if band is not None:
        grid = grid.loc[band_membership(grid, band)]
    grid = grid.sort_values(["canonical_match_id", "decision_at", "opportunity_id"], kind="mergesort")
    choices_by_match = {
        match_id: group.opportunity_id.to_numpy()
        for match_id, group in grid.groupby("canonical_match_id", sort=False)
    }
    rng = np.random.default_rng(seed)
    schedule = {}
    for match_id in opportunities.canonical_match_id.drop_duplicates():
        if pd.isna(match_id):
            continue
        choices = choices_by_match.get(match_id, ())
        schedule[match_id] = choices[int(rng.integers(0, len(choices)))] if len(choices) else None
    return schedule


def select_timing_candidates(candidates: pd.DataFrame, opportunities: pd.DataFrame, *,
                             policy: str, band: str, random_schedule=None) -> pd.DataFrame:
    """Choose at most one eligible side/quote per canonical fixture, without outcomes."""
    allowed = {"first_eligible_2h", "random_check", "max_coverage_model_ev"}
    if policy not in allowed:
        raise ValueError(f"unknown timing policy: {policy}")
    in_band = opportunities.loc[band_membership(opportunities, band)].copy()
    if policy == "first_eligible_2h":
        in_band = in_band.loc[in_band.is_grid.fillna(False)]
    elif policy == "random_check":
        schedule = random_schedule or {}
        in_band = in_band.loc[in_band.opportunity_id.isin(schedule.values())]
    if candidates.empty:
        return pd.DataFrame(
            columns=["opportunity_id", "canonical_match_id", "decision_at", "eligible",
                     "model_ev", "side", "odds"]
        )
    checkpoint_columns = ["opportunity_id", "canonical_match_id", "decision_at"]
    if policy == "first_eligible_2h":
        checkpoint_columns.append("is_grid")
    eligible = candidates.loc[candidates.eligible.fillna(False)].copy()
    eligible = eligible.merge(
        in_band[checkpoint_columns].drop_duplicates(["opportunity_id", "canonical_match_id"]),
        on=["opportunity_id", "canonical_match_id"],
        how="inner",
        suffixes=("", "_opp"),
    )
    if "decision_at_opp" in eligible:
        eligible["_selection_clock"] = eligible.pop("decision_at_opp")
    else:
        eligible["_selection_clock"] = eligible["decision_at"]
    if policy == "first_eligible_2h":
        eligible = eligible.loc[eligible.is_grid.fillna(False)]
    if eligible.empty:
        return eligible
    eligible["_ev"] = pd.to_numeric(eligible.model_ev, errors="coerce")
    eligible["_clock"] = pd.to_datetime(eligible["_selection_clock"], utc=True, errors="coerce")
    if policy == "first_eligible_2h":
        earliest = eligible.groupby("canonical_match_id")._clock.transform("min")
        eligible = eligible.loc[eligible._clock.eq(earliest)]
    order = [name for name in ["canonical_match_id", "_ev", "_clock", "bookmaker_id", "side"] if name in eligible]
    ascending = [True, False, True, True, True][:len(order)]
    eligible = eligible.sort_values(order, ascending=ascending, kind="mergesort")
    return eligible.drop_duplicates("canonical_match_id", keep="first").drop(
        columns=["_ev", "_clock", "_selection_clock", "is_grid"], errors="ignore"
    ).reset_index(drop=True)


def _binding(protocol_path, protocol):
    base_path = (ROOT / protocol["base_protocol"]).resolve()
    prior_dir = (ROOT / protocol["prepared_inputs"]).resolve()
    prior_protocol = json.loads(base_path.read_text())
    prior_binding_path = base_path.parent / "execution_binding.json"
    prior_binding = json.loads(prior_binding_path.read_text())
    if _prior_closure(base_path, prior_protocol) != prior_binding:
        raise ValueError("prior financial inputs/code/model do not match their immutable binding")
    source_paths = [
        Path(__file__).resolve(), protocol_path.resolve(), base_path, prior_binding_path,
        ROOT / "src/analysis/c0_interval_data.py",
        ROOT / "betting_app/tests/test_c0_interval_runner.py",
        protocol_path.parent / "daily_equivalence_receipt.json",
        ROOT / "betting_app/tests/test_c0_interval_data.py",
        prior_dir / "opportunities.parquet",
        prior_dir / "quotes.parquet",
        protocol_path.parent / "design_receipt.json",
        protocol_path.parent / "daily_equivalence_input.pkl",
        protocol_path.parent / "daily_equivalence_output.pkl",
        protocol_path.parent / "daily_equivalence_original_verifier.py",
    ]
    pins = {str(path.resolve()): sha(path) for path in sorted(set(source_paths))}
    return {
        "schema": "c0_interval_policy_execution_binding_v1",
        "protocol_sha256": sha(protocol_path),
        "prior_execution_binding_sha256": sha(prior_binding_path),
        "prior_financial_pins": prior_binding["pins"],
        "pins": pins,
    }


def _validate_daily_state_contract(forecasts: pd.DataFrame) -> None:
    if forecasts.empty or "daily_state_cutoff_day" not in forecasts:
        return
    grouped = forecasts.loc[forecasts.daily_state_cutoff_day.notna()].groupby(
        ["canonical_match_id", "daily_state_cutoff_day"], sort=False
    )
    for _, group in grouped:
        for column in (
            "daily_state_key_sha256",
            "native_feature_contract_sha256",
            "native_input_hashes",
            "p_base",
            "forecast_available",
            "native_absence",
        ):
            if column not in group:
                continue
            values = group[column].dropna().map(repr).unique()
            if len(values) > 1:
                raise ValueError(
                    f"fixture/day native state contract changed within {column}"
                )




def _policy(protocol):
    return {**protocol["policy"], "alpha": .5, "temperature": 1.0, "blending_mode": "logit_shrinkage"}




def _evaluate(opportunities, quotes, forecasts, protocol):
    """Evaluate exact checkpoint forecasts, batching only identical fixture/day states."""
    _validate_daily_state_contract(forecasts)
    if not forecasts.opportunity_id.is_unique:
        raise ValueError("Checkpoint forecast IDs must be unique")
    by_id = forecasts.set_index("opportunity_id", drop=False)
    if not opportunities.opportunity_id.isin(by_id.index).all():
        raise ValueError("Every checkpoint requires an explicit forecast or absence row")
    policy = _policy(protocol)
    quotes_by_match = {
        key: group for key, group in quotes.groupby("canonical_match_id", sort=False)
    }
    days = pd.to_datetime(opportunities.decision_at, utc=True).dt.strftime("%Y-%m-%d")
    candidate_parts, opportunity_parts = [], []
    for (match_id, _), group in opportunities.groupby(
        ["canonical_match_id", days], sort=True
    ):
        states = by_id.loc[group.opportunity_id].sort_values("forecast_at", kind="mergesort")
        for column in (
            "p_base", "forecast_available", "identity_verified", "native_absence",
            "daily_state_key_sha256", "native_feature_contract_sha256", "native_input_hashes",
        ):
            if column in states and states[column].map(repr).nunique(dropna=False) > 1:
                raise ValueError(f"Fixture/day forecast contract differs in {column}")
        # The legacy gate maps by fixture, so only the verified identical daily
        # state may represent this group. Its earliest clock precedes every check.
        representative = states.iloc[[0]]
        if pd.Timestamp(representative.iloc[0].forecast_at) > group.decision_at.min():
            raise ValueError("Daily representative forecast is later than a checkpoint")
        result = evaluate_policy_candidates(
            group, quotes_by_match.get(match_id, quotes.iloc[0:0]), representative,
            policy=policy, arm="complete",
            ttl_seconds=int(protocol["checkpoints"]["freshness_seconds"]),
        )
        opportunity_parts.append(result["opportunities"])
        candidate_parts.append(result["candidates"])
    return (
        pd.concat(opportunity_parts, ignore_index=True) if opportunity_parts else opportunities.copy(),
        pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame(),
    )


def _portfolio(selected, opportunities, protocol):
    p = protocol["portfolio"]
    wallet_inputs = opportunities.loc[
        opportunities.opportunity_id.isin(selected.opportunity_id)
    ]
    return simulate_operational_portfolio(selected, wallet_inputs,
        initial_bankroll=float(p["initial_bankroll"]), tax_rate=float(protocol["policy"]["tax_rate"]),
        fraction=float(p["kelly_fraction"]), max_stake=float(p["max_stake"]), min_stake=float(p["min_stake"]),
        max_portfolio_exposure=float(p["max_portfolio_exposure"]))


def _monthly_metrics(portfolio, opportunities, protocol, *, with_intervals=True):
    months = sorted(pd.to_datetime(opportunities.decision_at, utc=True).dt.strftime("%Y-%m").unique())
    cols = ["pnl", "stake", "n", "error", "weighted_error", "near_n", "near_error", "near_stake", "near_weighted_error"]
    monthly = pd.DataFrame(0.0, index=months, columns=cols)
    ledger = portfolio["ledger"]
    if not ledger.empty:
        for row in ledger.loc[ledger.settled].to_dict("records"):
            month = pd.Timestamp(row["decision_at"]).strftime("%Y-%m")
            ev = float(row["model_ev"]); stake = float(row["stake"])
            error = ev - float(row["realized_return"])
            monthly.loc[month, ["pnl", "stake", "n", "error", "weighted_error"]] += [float(row["realized_pnl"]), stake, 1, error, stake * error]
            if .05 < ev <= .10:
                monthly.loc[month, ["near_n", "near_error", "near_stake", "near_weighted_error"]] += [1, error, stake, stake * error]
    if not with_intervals:
        def estimate(numerator, denominator):
            total = float(monthly[numerator].sum())
            weight = float(monthly[denominator].sum())
            return total / weight if weight > 0 else None
        return {
            "roi": estimate("pnl", "stake"),
            "ev_overstatement_bet_weighted": estimate("error", "n"),
            "ev_overstatement_stake_weighted": estimate("weighted_error", "stake"),
            "near_5_10_ev_overstatement_bet_weighted": estimate("near_error", "near_n"),
            "near_5_10_ev_overstatement_stake_weighted": estimate("near_weighted_error", "near_stake"),
            "near_5_10_settled_bets": int(monthly.near_n.sum()),
            "true_pinnacle_close_count": 0,
            "true_clv": None,
        }, monthly
    draws, seed = int(protocol["bootstrap"]["draws"]), int(protocol["bootstrap"]["seed"])
    alpha = 1.0 - float(protocol["bootstrap"]["confidence_per_interval"])
    months_n = len(months)
    rng = np.random.default_rng(seed)
    sample_months = rng.integers(0, months_n, size=(draws, months_n)) if months_n else np.empty((draws, 0), dtype=int)
    def interval(numerator, denominator):
        num = monthly[numerator].to_numpy(dtype=float)
        den = monthly[denominator].to_numpy(dtype=float)
        point = float(num.sum() / den.sum()) if den.sum() > 0 else None
        equal_defined = den > 0
        equal = float(np.mean(num[equal_defined] / den[equal_defined])) if equal_defined.any() else None
        if months_n:
            sampled_num, sampled_den = num[sample_months].sum(axis=1), den[sample_months].sum(axis=1)
            valid = sampled_den > 0
            valid_count = int(valid.sum())
            ci = [float(x) for x in np.quantile(sampled_num[valid] / sampled_den[valid], [alpha / 2, 1 - alpha / 2])] if valid_count == draws and months_n >= 2 else None
        else:
            valid_count, ci = 0, None
        return {"estimate": point, "ci": ci, "confidence_level": 1 - alpha,
                "draws": draws, "undefined_replicates": draws - valid_count,
                "equal_month_conditional": equal,
                "unsupported_months": [months[i] for i in range(months_n) if den[i] == 0],
                "month_universe": months}
    metrics = {"roi": interval("pnl", "stake"),
               "ev_overstatement_bet_weighted": interval("error", "n"),
               "ev_overstatement_stake_weighted": interval("weighted_error", "stake"),
               "near_5_10_ev_overstatement_bet_weighted": interval("near_error", "near_n"),
               "near_5_10_ev_overstatement_stake_weighted": interval("near_weighted_error", "near_stake"),
               "near_5_10_settled_bets": int(monthly.near_n.sum()),
               "true_pinnacle_close_count": 0, "true_clv": None}
    return metrics, monthly

def _run(prepared, protocol, binding_path):
    output = ROOT / protocol["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    opportunities, candidates = _evaluate(
        prepared["opportunities"], prepared["quotes"], prepared["forecasts"], protocol
    )
    evidence = prepared.get("evidence", {})
    fixture_accounting = evidence.get("fixture_accounting")
    fixture_ids = (
        fixture_accounting.canonical_match_id
        if isinstance(fixture_accounting, pd.DataFrame)
        else opportunities.canonical_match_id.drop_duplicates()
    )
    fixture_universe = (
        fixture_accounting.drop_duplicates("canonical_match_id").copy()
        if isinstance(fixture_accounting, pd.DataFrame)
        else pd.DataFrame({"canonical_match_id": pd.Series(fixture_ids).drop_duplicates()})
    )
    opportunities.to_parquet(output / "opportunities.parquet", index=False)
    prepared["quotes"].to_parquet(output / "quotes.parquet", index=False)
    prepared["forecasts"].to_parquet(output / "forecasts.parquet", index=False)
    candidates.to_parquet(output / "candidates.parquet", index=False)
    _json_write(
        output / "fixture_accounting.json",
        json.loads(fixture_accounting.to_json(orient="records", date_format="iso", default_handler=str))
        if isinstance(fixture_accounting, pd.DataFrame)
        else [],
    )
    representative_manifest = evidence.get("representative_manifest")
    if isinstance(representative_manifest, pd.DataFrame):
        _json_write(
            output / "daily_representative_manifest.json",
            json.loads(representative_manifest.to_json(orient="records", date_format="iso", default_handler=str)),
        )
    report_evidence = {
        key: value for key, value in evidence.items()
        if not isinstance(value, pd.DataFrame)
        and key not in {"representative_manifest", "fixture_accounting"}
    }
    daily_receipt = json.loads(
        (binding_path.parent / "daily_equivalence_receipt.json").read_text()
    )
    report = {
        "status": "HISTORICAL_DEVELOPMENT_ONLY",
        "binding_sha256": sha(binding_path),
        "evidence": report_evidence,
        "daily_equivalence_receipt": daily_receipt,
        "timing_policies": {},
        "bootstrap": protocol["bootstrap"],
        "primary_interval_family": (
            "first_eligible_2h across five frozen windows; ROI and EV-error families "
            "are separate; no winning-window selection"
        ),
        "interpretation": protocol["interpretation"],
        "true_clv": None,
        "noncausal_oracle": (
            "max_coverage_model_ev uses future quote availability/prices; maximizes "
            "qualifying fixture coverage, not financial return; outcome/PnL never used."
        ),
    }
    for band in BANDS:
        group = opportunities.loc[band_membership(opportunities, band)].copy()
        report["timing_policies"][band] = {}
        schedules = [
            choose_random_grid_checkpoints(
                opportunities,
                seed=int(protocol["random"]["seed"]) + i,
                band=band,
            )
            for i in range(int(protocol["random"]["draws"]))
        ]
        for index, schedule in enumerate(schedules):
            pd.DataFrame(
                [{"canonical_match_id": match_id, "opportunity_id": opportunity_id}
                 for match_id, opportunity_id in schedule.items()]
            ).to_parquet(output / f"{band}_random_schedule_{index:03d}.parquet", index=False)
        for policy in ("first_eligible_2h", "random_check", "max_coverage_model_ev"):
            repetitions = schedules if policy == "random_check" else [None]
            runs = []
            for index, schedule in enumerate(repetitions):
                selected = select_timing_candidates(candidates, opportunities, policy=policy, band=band, random_schedule=schedule)
                case_opportunities = group
                if policy == "first_eligible_2h":
                    case_opportunities = group.loc[group.is_grid.fillna(False)]
                elif policy == "random_check":
                    case_opportunities = group.loc[group.opportunity_id.isin(schedule.values())]
                case_candidates = candidates.loc[
                    candidates.opportunity_id.isin(case_opportunities.opportunity_id)
                ]
                portfolio = _portfolio(selected, opportunities, protocol)
                tag = f"{band}/{policy}" + (f"/{index:03d}" if policy == "random_check" else "")
                directory = output / tag
                directory.mkdir(parents=True)
                selected.to_parquet(directory / "selected.parquet", index=False)
                for key in ("ledger", "capital_path", "execution_rejections"):
                    portfolio[key].to_csv(directory / f"{key}.csv", index=False)
                metrics, monthly = _monthly_metrics(
                    portfolio, group, protocol,
                    with_intervals=(policy == "first_eligible_2h"),
                )
                monthly.to_csv(directory / "monthly.csv")
                selected_counts = selected.groupby("canonical_match_id").size().rename("selected_bets").reset_index()
                ledger = portfolio["ledger"]
                executed_counts = (
                    ledger.groupby("canonical_match_id").size().rename("executed_bets").reset_index()
                    if not ledger.empty
                    else pd.DataFrame(columns=["canonical_match_id", "executed_bets"])
                )
                eligibility = (
                    case_candidates.groupby("canonical_match_id").eligible.sum()
                    .rename("eligible_candidate_sides").reset_index()
                    if not case_candidates.empty and "eligible" in case_candidates
                    else pd.DataFrame(columns=["canonical_match_id", "eligible_candidate_sides"])
                )
                checkpoint_counts = case_opportunities.groupby("canonical_match_id").size().rename("checked_checkpoint_count").reset_index()
                candidate_counts = (
                    case_candidates.groupby("canonical_match_id").size()
                    .rename("candidate_side_count").reset_index()
                    if not case_candidates.empty and "canonical_match_id" in case_candidates
                    else pd.DataFrame(columns=["canonical_match_id", "candidate_side_count"])
                )
                reason_source = case_opportunities
                absence_reasons = (
                    reason_source.groupby("canonical_match_id").candidate_absence_reason
                    .agg(lambda values: json.dumps(dict(Counter(values.fillna("ready"))), sort_keys=True))
                    .rename("checkpoint_absence_reason_counts").reset_index()
                    if "candidate_absence_reason" in reason_source
                    else pd.DataFrame(columns=["canonical_match_id", "checkpoint_absence_reason_counts"])
                )
                absence = fixture_universe.merge(
                    selected_counts, on="canonical_match_id", how="left"
                ).merge(executed_counts, on="canonical_match_id", how="left").merge(
                    eligibility, on="canonical_match_id", how="left"
                ).merge(checkpoint_counts, on="canonical_match_id", how="left").merge(
                    candidate_counts, on="canonical_match_id", how="left"
                ).merge(absence_reasons, on="canonical_match_id", how="left").fillna(
                    {"selected_bets": 0, "executed_bets": 0, "eligible_candidate_sides": 0,
                     "checked_checkpoint_count": 0, "candidate_side_count": 0}
                )
                if policy == "random_check":
                    absence["random_check_opportunity_id"] = absence.canonical_match_id.map(schedule)
                absence["eligible_candidate_sides"] = absence.eligible_candidate_sides.astype(int)
                absence.to_parquet(directory / "fixture_absence_ledger.parquet", index=False)
                runs.append({"summary": portfolio["summary"], "statistics": metrics,
                             "selected_candidates": len(selected), "ledger": portfolio["ledger"],
                             "coverage_fixtures": int(selected.canonical_match_id.nunique()) if not selected.empty else 0,
                             "zero_bet_fixtures": int(absence.executed_bets.eq(0).sum()),
                             "unresolved_bets": int(portfolio["summary"]["pending_bets_count"])})
            if policy == "random_check":
                summaries = [r["summary"] for r in runs]
                def quantiles(values):
                    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
                    return np.quantile(finite, [.05, .5, .95]).tolist() if finite else None
                report["timing_policies"][band][policy] = {
                    "draw_count": len(runs),
                    "roi_quantiles": quantiles([r["summary"]["roi"] for r in runs]),
                    "pnl_quantiles": quantiles([r["summary"]["profit"] for r in runs]),
                    "drawdown_quantiles": quantiles([r["summary"]["max_drawdown"] for r in runs]),
                    "exposure_quantiles": quantiles([r["summary"]["peak_reserved_fraction"] for r in runs]),
                    "coverage_quantiles": quantiles([r["coverage_fixtures"] for r in runs]),
                    "ev_error_quantiles": quantiles([r["statistics"]["ev_overstatement_bet_weighted"] for r in runs]),
                    "near_band_ev_error_quantiles": quantiles([r["statistics"]["near_5_10_ev_overstatement_bet_weighted"] for r in runs]),
                    "zero_bet_schedule_count": sum(r["summary"]["bets_count"] == 0 for r in runs),
                    "unresolved_bet_counts": [r["unresolved_bets"] for r in runs],
                    "schedule_summaries": summaries,
                    "quantile_scope": "5th/50th/95th timing quantiles conditional on a defined metric; undefined schedules counted separately, not independent sports evidence",
                    "undefined_schedule_counts": {
                        "roi": sum(r["summary"]["roi"] is None for r in runs),
                        "ev_error": sum(r["statistics"]["ev_overstatement_bet_weighted"] is None for r in runs),
                        "near_band_ev_error": sum(r["statistics"]["near_5_10_ev_overstatement_bet_weighted"] is None for r in runs),
                    },
                    "independent_match_samples": False,
                }
            else:
                run = runs[0]
                report["timing_policies"][band][policy] = {
                    "summary": run["summary"], "statistics": run["statistics"],
                    "selected_candidates": run["selected_candidates"],
                    "coverage_fixtures": run["coverage_fixtures"],
                    "zero_bet_fixtures": run["zero_bet_fixtures"],
                    "unresolved_bets": run["unresolved_bets"],
                    "monthly_intervals_primary_first_eligible": policy == "first_eligible_2h",
                }
    _json_write(output / "results.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="c0-interval-policy")
    parser.add_argument("--protocol", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bind", action="store_true"); mode.add_argument("--smoke", action="store_true"); mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol.resolve(); protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "c0_interval_policy_v1" or protocol.get("schema_version") != 1:
        raise ValueError("unsupported C0 interval protocol")
    binding = _binding(protocol_path, protocol); binding_path = protocol_path.parent / "execution_binding.json"
    if args.bind:
        _json_write(binding_path, binding); return
    if json.loads(binding_path.read_text()) != binding:
        raise ValueError("interval inputs/code/prior financial binding changed after immutable binding")
    from src.analysis.c0_interval_data import prepare_interval_inputs
    prepared = prepare_interval_inputs(protocol)
    if args.smoke:
        print(json.dumps(prepared.get("evidence", {}), default=str)); return
    _run(prepared, protocol, binding_path)


if __name__ == "__main__":
    main()
