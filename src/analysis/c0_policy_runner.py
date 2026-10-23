"""Frozen current-policy historical evaluation, invoked by the unified runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.c0_policy_data import prepare_policy_inputs
from src.analysis.c0_policy_evaluation import ARMS, evaluate_policy_candidates
from src.analysis.unified_evaluation_engine import simulate_operational_portfolio

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _closure(protocol_path, protocol):
    from betting_app.core.models.native_c0 import load_serving_config

    config, config_path, _ = load_serving_config()
    research = Path((ROOT / "data/research_root.txt").read_text().strip())
    if not research.is_absolute():
        research = ROOT / research
    files = {
        protocol_path,
        config_path,
        Path(__file__).resolve(),
        ROOT / "scripts/run_unified_betting_evaluation.py",
        ROOT / "src/analysis/c0_policy_data.py",
        ROOT / "src/analysis/c0_policy_evaluation.py",
        ROOT / "src/analysis/unified_evaluation_engine.py",
        ROOT / "src/analysis/bootstrap.py",
        ROOT / "src/analysis/c0_input_aligned_data.py",
        ROOT / "betting_app/core/staking.py",
        ROOT / "betting_app/core/models/engine.py",
        ROOT / "betting_app/core/models/native_c0.py",
        ROOT / "betting_app/services/c0_inference.py",
        ROOT / "betting_app/services/c0_worker.py",
        ROOT / "betting_app/services/bet_qualification_service.py",
        ROOT / "data/research_root.txt",
        ROOT / "conf/base/research_benchmark.json",
        ROOT / "data/08_reporting/native_c0_serving_source_contract_v2.json",
    }
    for key in ("raw_snapshot", "identity_snapshot"):
        files.update(
            path
            for path in (ROOT / protocol["inputs"][key]).rglob("*")
            if path.is_file()
        )
    receipt = json.loads(
        (
            ROOT / "data/08_reporting/native_c0_serving_source_contract_v2.json"
        ).read_text()
    )
    for name in receipt["preserved_provenance"]["artifact_sha256"]:
        files.add(ROOT / name)
    for name in receipt["serving_source_hashes"]:
        files.add(ROOT / name.split(":", 1)[1])
    files.add(ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json")
    files.add(
        research
        / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/release_days.json"
    )
    # Native source receipts bind additional archive dependencies at runtime; bind their bytes too.
    identity = json.loads(
        (
            ROOT / protocol["inputs"]["identity_snapshot"] / "native_identity.json"
        ).read_text()
    )
    for name in identity["training_source_hashes"]:
        if name.startswith("research:"):
            files.add(research / name.split(":", 1)[1])
    pins = {str(path.resolve()): sha(path) for path in sorted(files)}
    return {
        "schema": "c0_full_policy_execution_binding_v1",
        "protocol_sha256": sha(protocol_path),
        "pins": pins,
    }


def _json_write(path, value):
    def encode(item):
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        return str(item)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(
            value, stream, indent=2, sort_keys=True, default=encode, allow_nan=False
        )
        stream.write("\n")


def _statistics(ledger, opportunities):
    months = sorted(
        pd.to_datetime(opportunities.decision_at, utc=True)
        .dropna()
        .dt.strftime("%Y-%m")
        .unique()
    )
    columns = [
        "pnl",
        "stake",
        "n",
        "error",
        "weighted_error",
        "near_n",
        "near_error",
        "near_stake",
        "near_weighted_error",
    ]
    monthly = pd.DataFrame(0.0, index=months, columns=columns)
    if not ledger.empty:
        for row in ledger.loc[ledger.settled].to_dict("records"):
            month = pd.Timestamp(row["decision_at"]).strftime("%Y-%m")
            pnl, stake, ev = (
                float(row["realized_pnl"]),
                float(row["stake"]),
                float(row["model_ev"]),
            )
            error = ev - float(row["realized_return"])
            monthly.loc[month, ["pnl", "stake", "n", "error", "weighted_error"]] += [
                pnl,
                stake,
                1,
                error,
                stake * error,
            ]
            if 0.05 < ev <= 0.10:
                monthly.loc[
                    month, ["near_n", "near_error", "near_stake", "near_weighted_error"]
                ] += [1, error, stake, stake * error]

    def ratio(num, den):
        return block_ratio_interval(
            monthly[num].to_numpy(),
            monthly[den].to_numpy(),
            months,
            draws=5000,
            seed=20261021,
        )

    metrics = {
        "roi": ratio("pnl", "stake"),
        "ev_overstatement_bet_weighted": ratio("error", "n"),
        "ev_overstatement_stake_weighted": ratio("weighted_error", "stake"),
        "near_5_10_ev_overstatement_bet_weighted": ratio("near_error", "near_n"),
        "near_5_10_ev_overstatement_stake_weighted": ratio(
            "near_weighted_error", "near_stake"
        ),
    }
    metrics["near_5_10_settled_bets"] = int(monthly.near_n.sum())
    metrics["profit_per_opportunity"] = float(monthly.pnl.sum()) / len(opportunities)
    metrics["true_pinnacle_close_count"] = 0
    metrics["true_clv"] = None
    metrics["close_limit"] = (
        "No independently bound actual Pinnacle close in this snapshot; not replaced by contemporaneous consensus."
    )
    return metrics, monthly


def _paired(complete, control):
    months = complete.index
    rng = np.random.default_rng(20261021)
    samples = rng.integers(0, len(months), size=(5000, len(months)))

    def ratio(pnl, stake):
        return np.divide(pnl, stake, out=np.full_like(pnl, np.nan), where=stake > 0)

    left = ratio(
        complete.pnl.to_numpy()[samples].sum(axis=1),
        complete.stake.to_numpy()[samples].sum(axis=1),
    )
    right = ratio(
        control.pnl.to_numpy()[samples].sum(axis=1),
        control.stake.to_numpy()[samples].sum(axis=1),
    )
    delta = left - right
    lm = ratio(complete.pnl.to_numpy(), complete.stake.to_numpy())
    rm = ratio(control.pnl.to_numpy(), control.stake.to_numpy())
    equal = (lm - rm)[samples].mean(axis=1)

    def result(values):
        finite = np.isfinite(values)
        return {
            "undefined_replicates": int((~finite).sum()),
            "ci99": np.quantile(values, [0.005, 0.995]).tolist()
            if finite.all() and len(months) >= 2
            else None,
        }

    lp, ls, rp, rs = (
        complete.pnl.sum(),
        complete.stake.sum(),
        control.pnl.sum(),
        control.stake.sum(),
    )
    return {
        "definition": "complete minus control; own-policy selection and capital feedback, not fixed-bet attribution",
        "draws": 5000,
        "month_count": len(months),
        "family_size": 5,
        "family_confidence": 0.95,
        "weighted_roi_difference": float(lp / ls - rp / rs)
        if ls > 0 and rs > 0
        else None,
        "weighted": result(delta),
        "equal_month": result(equal),
        "equal_month_point_common_active": float(np.nanmean(lm - rm))
        if np.isfinite(lm - rm).any()
        else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="current-policy")
    parser.add_argument("--protocol", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bind", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    binding = _closure(protocol_path, protocol)
    binding_path = protocol_path.parent / "execution_binding.json"
    if args.bind:
        _json_write(binding_path, binding)
        print(json.dumps({"status": "BOUND", "binding_sha256": sha(binding_path)}))
        return
    if json.loads(binding_path.read_text()) != binding:
        raise ValueError("Policy inputs/code/model changed after immutable binding")
    prepared = prepare_policy_inputs(protocol)
    if args.smoke:
        print(json.dumps(prepared["evidence"], default=str))
        return
    output = ROOT / "data/08_reporting/c0_policy_evaluation" / protocol["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    for name in ("opportunities", "quotes", "forecasts"):
        prepared[name].to_parquet(output / f"{name}.parquet", index=False)
    report = {
        "status": "HISTORICAL_DEVELOPMENT_ONLY",
        "binding_sha256": sha(binding_path),
        "evidence": prepared["evidence"],
        "panels": {},
        "paired_controls": {},
        "limits": [
            "No genuine issuance, executable-betting or profit certification.",
            "Zero authenticated admissions is not a financial or reliability success.",
            "EXP039 and rating veto unavailable/not_evaluated, not fabricated agreement.",
        ],
    }
    authenticated = prepared["opportunities"].copy()
    authenticated["admitted"] = False
    authenticated["absence_reason"] = "source_and_historical_issuance_not_authenticated"
    authenticated.to_parquet(
        output / "authenticated_opportunities.parquet", index=False
    )
    report["panels"]["authenticated"] = {
        "opportunities": len(authenticated),
        "admitted": 0,
        "status": "BLOCKED_SOURCE_AND_ISSUANCE",
    }
    policy = {
        **protocol["policy"],
        "alpha": 0.5,
        "temperature": 1.0,
        "blending_mode": "logit_shrinkage",
    }
    for panel in protocol["panels"]:
        if panel["id"] == "authenticated":
            continue
        results, monthly = {}, {}
        for arm in ARMS:
            evaluated = evaluate_policy_candidates(
                prepared["opportunities"],
                prepared["quotes"],
                prepared["forecasts"],
                policy=policy,
                arm=arm,
                ttl_seconds=panel["freshness_seconds"],
            )
            portfolio = simulate_operational_portfolio(
                evaluated["selected"],
                prepared["opportunities"],
                initial_bankroll=10000,
                tax_rate=0.12,
                fraction=0.25,
                max_stake=150,
                min_stake=2,
                max_portfolio_exposure=0.25,
            )
            directory = output / panel["id"] / arm
            directory.mkdir(parents=True)
            for key in ("opportunities", "candidates", "selected"):
                evaluated[key].to_parquet(directory / f"{key}.parquet", index=False)
            for key in ("ledger", "capital_path", "execution_rejections"):
                portfolio[key].to_csv(directory / f"{key}.csv", index=False)
            metrics, monthly[arm] = _statistics(
                portfolio["ledger"], prepared["opportunities"]
            )
            monthly[arm].to_csv(directory / "monthly.csv")
            failures = (
                evaluated["candidates"].reason.value_counts().to_dict()
                if not evaluated["candidates"].empty
                else {}
            )
            results[arm] = {
                "summary": portfolio["summary"],
                "statistics": metrics,
                "selected_fixture_sides": len(evaluated["selected"]),
                "failure_reasons": failures,
                "opportunity_absences": evaluated["opportunities"]
                .candidate_absence_reason.value_counts()
                .to_dict(),
            }
        report["panels"][panel["id"]] = results
        report["paired_controls"][panel["id"]] = {
            arm: _paired(monthly["complete"], monthly[arm])
            for arm in ARMS
            if arm != "complete"
        }
    _json_write(output / "results.json", report)
    print(
        json.dumps(
            {"status": report["status"], "results": str(output / "results.json")}
        )
    )
