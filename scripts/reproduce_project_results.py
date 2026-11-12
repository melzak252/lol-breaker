#!/usr/bin/env python3
"""Replay project-report scores and flat-stake diagnostics from installed ledgers.

No fitting, model inference, database access, or network access is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from betting_app.core.ev import expected_value
from src.analysis.long_term_market_evaluation import (
    _month_universe,
    select_fixed_policy,
    summarize_unit_returns,
)
from src.analysis.metrics import binary_log_loss_vector, calculate_ece

ROOT = Path(__file__).resolve().parents[1]
YEAR_RANGE = tuple(range(2021, 2027))
DRAWS = 5000
SEED = 20261111
MODEL_RUN = "c0_model_improvement_20261026_01"
FINANCE_RUN = "fresh20_oddsportal_20261111_02"
FIRST = "OPEN_OR_FIRST_OBSERVED"

PATHS = {
    "model_predictions": "data/07_model_output/c0_model_improvement/c0_model_improvement_20261026_01/predictions.parquet",
    "model_protocol": "data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/protocol.json",
    "model_binding": "data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/execution_binding.json",
    "model_verification": "data/08_reporting/c0_model_improvement/c0_model_improvement_20261026_01/sealed_verification.json",
    "serving_config": "conf/base/c0_serving.json",
    "odds_protocol": "data/05_model_input/fresh20_oddsportal/fresh20_oddsportal_20261111_02/protocol.json",
    "odds_evaluation": "data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/oddsportal_annual_endpoint_ledger.parquet",
    "odds_binding": "data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/execution_binding.json",
    "later_quote_proxies": "data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/samebook_shrinkage_sensitivity_01/first_observed_quote_proxies.parquet",
    "events": "data/04_feature/long_term_market_edge/long_term_market_edge_20261022_01/oddsportal_identity_complete_v3/events.parquet",
    "selection_source": "src/analysis/long_term_market_evaluation.py",
    "score_source": "src/analysis/metrics.py",
    "bootstrap_source": "src/analysis/bootstrap.py",
    "ev_source": "betting_app/core/ev.py",
    "engine_source": "betting_app/core/models/engine.py",
    "guards_source": "src/analysis/betting_guards_a_to_d.py",
    "probability_metrics_source": "src/analysis/probability_metrics.py",
    "devig_source": "src/analysis/devig_comparison.py",
    "audit_serialization_source": "src/analysis/c0_edge_alpha_audit.py",
    "this_script": "scripts/reproduce_project_results.py",
}


def select_first_policy(panel: pd.DataFrame, *, alpha: float, tax: float) -> pd.DataFrame:
    """Apply the existing financial selection rules at one declared tax rate."""
    prepared = panel.loc[panel.timed_price_eligible.fillna(False)].copy()
    prepared["q"] = prepared["q_focal"]
    selected = select_fixed_policy(
        prepared,
        sports_alpha=alpha,
        tax_rate=tax,
        epsilon=0.025,
        minimum_ev=0.05,
        maximum_odds=4.0,
        timed=True,
    )
    return selected.reset_index(drop=True)


def score_probability(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(p)
    y, p = y[valid].astype(int), p[valid].astype(float)
    if not len(y):
        return {"n": 0, "log_loss": None, "brier": None, "ece10": None, "accuracy": None, "auc": None}
    if not ((p > 0) & (p < 1)).all():
        raise ValueError("all scored probabilities must be strictly between 0 and 1")
    return {
        "n": int(len(y)),
        "log_loss": float(np.mean(binary_log_loss_vector(y, p, epsilon=0.0))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece10": float(calculate_ece(y, p, n_bins=10)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "auc": float(roc_auc_score(y, p)),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recompute_fixed_tax(selected: pd.DataFrame, tax: float) -> pd.DataFrame:
    result = selected.copy()
    result["tax_rate"] = tax
    result["ev_net"] = [
        expected_value(float(probability), float(odds), tax)
        for probability, odds in zip(result["p_selected"], result["odds"])
    ]
    return result


def _financial_summary(selected: pd.DataFrame, months: Sequence[str]) -> dict[str, Any]:
    full = summarize_unit_returns(selected, months, draws=DRAWS, seed=SEED, stake=1.0)
    interval = full.get("return_interval", {})
    interval_keys = (
        "estimate", "ci_lower", "ci_upper", "blocks", "draws", "seed",
        "empty_replicates", "valid_replicates", "equal_block_estimate", "status",
    )
    result = {
        key: full.get(key)
        for key in (
            "bets", "known_labels", "unknown_labels", "wins", "turnover",
            "known_label_turnover", "profit", "known_label_profit", "roi",
            "known_label_roi", "status",
        )
    }
    result["return_interval"] = {key: interval.get(key) for key in interval_keys if key in interval}
    if int(full.get("unknown_labels") or 0) > 0:
        result["return_interval_scope"] = "known-label-only diagnostic; unresolved selected labels excluded, not whole-portfolio ROI"
    else:
        result["return_interval_scope"] = "all selected outcomes known; interval covers the full selected cohort"
    return result


def _check_inputs() -> dict[str, Path]:
    found = {key: ROOT / rel for key, rel in PATHS.items()}
    missing = [f"{key}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
               for key, path in found.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing replay prerequisites (install the bound local artifacts; no fallback is used):\n  "
                                + "\n  ".join(missing))
    return found


def _bound_hash(binding: dict[str, Any], relative_path: str) -> str:
    matches = [
        digest for name, digest in binding.items()
        if str(name).replace("\\", "/").endswith("/" + relative_path)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one bound hash for {relative_path}; found {len(matches)}")
    return str(matches[0])


def replay() -> tuple[dict[str, Any], dict[str, str]]:
    paths = _check_inputs()
    model_protocol = json.loads(paths["model_protocol"].read_text(encoding="utf-8"))
    odds_protocol = json.loads(paths["odds_protocol"].read_text(encoding="utf-8"))
    serving = json.loads(paths["serving_config"].read_text(encoding="utf-8"))
    if model_protocol.get("run_id") != MODEL_RUN or odds_protocol.get("run_id") != FINANCE_RUN:
        raise ValueError("installed protocols do not match the report's frozen run IDs")
    if serving.get("serving_model_version") != "c0-native-fresh20-2026-10-21-v1-regional-v1":
        raise ValueError("local serving identity differs from the report's stated C20R model")
    model_binding = json.loads(paths["model_binding"].read_text(encoding="utf-8"))
    model_verification = json.loads(paths["model_verification"].read_text(encoding="utf-8"))
    if model_binding.get("input_hashes", {}).get("protocol.json") != _sha256(paths["model_protocol"]):
        raise ValueError("model protocol hash does not match its saved execution binding")
    if model_verification.get("status") != "VERIFIED_PASS":
        raise ValueError("model output lacks a successful saved independent verification receipt")
    if model_verification.get("predictions_sha256") != _sha256(paths["model_predictions"]):
        raise ValueError("model predictions differ from the saved independent-verification hash")
    odds_binding = json.loads(paths["odds_binding"].read_text(encoding="utf-8"))
    bound_inputs = odds_binding.get("input_and_code_sha256", {})
    for key in ("events", "odds_protocol"):
        if _bound_hash(bound_inputs, PATHS[key]) != _sha256(paths[key]):
            raise ValueError(f"{key} hash does not match the saved OddsPortal evaluation binding")
    if not odds_binding.get("binding_precedes_scoring"):
        raise ValueError("OddsPortal binding does not declare pre-score binding")

    pred = pd.read_parquet(paths["model_predictions"])
    if len(pred) != 3103 or pred.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("expected 3,103 unique canonical assessment predictions")
    if int(pred["regional_supported"].sum()) != 3083:
        raise ValueError("regional support mask differs from the frozen 3,083-row cohort")
    y = pred["y"].to_numpy(dtype=int)
    support = pred["regional_supported"].to_numpy(dtype=bool)
    model_columns = {
        "historical_C0": "C0",
        "fresh20_without_regional": "fresh20",
        "annual_A0": "annual_A0",
    }
    model_scores: dict[str, Any] = {
        "full_3103": {
            label: score_probability(y, pred[column].to_numpy(dtype=float))
            for label, column in model_columns.items()
        },
        "regional_supported_3083": {
            **{
                label: score_probability(y[support], pred[column].to_numpy(dtype=float)[support])
                for label, column in model_columns.items()
            },
            "C20R": score_probability(
                y[support], pred["fresh20+regional"].to_numpy(dtype=float)[support]
            ),
        },
        "coverage": {
            "assessment_rows": int(len(pred)),
            "regional_supported_rows": int(support.sum()),
            "regional_unsupported_rows": int((~support).sum()),
            "regional_coverage": float(support.mean()),
            "unsupported_probability_status": "missing; excluded from regional model score, not imputed",
        },
    }

    events = pd.read_parquet(paths["events"])
    months = _month_universe(events, YEAR_RANGE)
    panel = pd.read_parquet(paths["odds_evaluation"])
    panel = panel.loc[panel.horizon.eq(FIRST)].copy()
    sports_zero = select_first_policy(panel, alpha=1.0, tax=0.0)
    samebook_zero = select_first_policy(panel, alpha=0.5, tax=0.0)
    samebook_fixed_tax = _recompute_fixed_tax(samebook_zero, 0.12)
    samebook_requalified_tax = select_first_policy(panel, alpha=0.5, tax=0.12)
    financial = {
        "policy": {
            "window": FIRST,
            "years": list(YEAR_RANGE),
            "tax_for_zero_tax_policies": 0.0,
            "sports_alpha": {"sports_only": 1.0, "samebook_50_50_logit": 0.5},
            "samebook_reference": "focal bookmaker multiplicative no-vig probability; not the application hybrid",
            "minimum_net_ev": 0.05,
            "dro_epsilon": 0.025,
            "maximum_odds": 4.0,
            "selection": "first eligible retained quote, at most one per canonical fixture",
            "stake": 1.0,
        },
        "sports_only_zero_tax": _financial_summary(sports_zero, months),
        "samebook_50_50_zero_tax": _financial_summary(samebook_zero, months),
        "samebook_50_50_fixed_tickets_tax_12_percent": _financial_summary(samebook_fixed_tax, months),
        "samebook_50_50_requalified_tax_12_percent": _financial_summary(samebook_requalified_tax, months),
    }

    proxy_rows = pd.read_parquet(paths["later_quote_proxies"])

    paired = proxy_rows.loc[
        proxy_rows.later_quote_pair_status.eq("matched_later_prestart_scenario")
    ].copy()
    matched_known = paired.loc[
        paired.label_status.eq("known") & paired.y.isin([0, 1])
    ].copy()
    if len(matched_known):
        win = np.where(matched_known.side.eq("a"), matched_known.y, 1 - matched_known.y)
        predicted_ev = matched_known.p_selected.to_numpy(float) * matched_known.odds.to_numpy(float) - 1.0
        odds_a = matched_known.odds_a_later.to_numpy(float)
        odds_b = matched_known.odds_b_later.to_numpy(float)
        fair_a = (1.0 / odds_a) / (1.0 / odds_a + 1.0 / odds_b)
        q_later = np.where(matched_known.side.eq("a"), fair_a, 1.0 - fair_a)
        later_fair_proxy = matched_known.odds.to_numpy(float) * q_later - 1.0
        realized = np.fromiter((expected_value(int(w), float(o), 0.0)
                                for w, o in zip(win, matched_known.odds)), dtype=float, count=len(win))
        proxy_summary = {
            "scope": "matched later recorded quote proxy; not true CLV or executable close",
            "matched_proxy_rows": int(len(paired)),
            "matched_known_label_rows": int(len(matched_known)),
            "mean_entry_model_ev_known": float(predicted_ev.mean()),
            "mean_later_fair_price_proxy_known": float(later_fair_proxy.mean()),
            "mean_realized_return_known": float(realized.mean()),
            "model_ev_minus_later_proxy_pp": float((predicted_ev.mean() - later_fair_proxy.mean()) * 100),
            "later_proxy_minus_realized_pp": float((later_fair_proxy.mean() - realized.mean()) * 100),
            "decomposition_identity_max_error": float(np.max(np.abs((predicted_ev - realized) -
                ((predicted_ev - later_fair_proxy) + (later_fair_proxy - realized))))),
            "certification": "not CLV; quote chronology proxy only, no fill or authenticated close",
        }
    else:
        proxy_summary = {"status": "NO_MATCHED_KNOWN_ROWS"}

    hashes = {key: _sha256(path) for key, path in paths.items()}
    snapshot = {
        "schema": "project_results_replay_v1",
        "status": "REPLAYED_FROM_INSTALLED_SAVED_ARTIFACTS",
        "reproduction": {
            "bootstrap": {
                "resamples": DRAWS,
                "seed": SEED,
                "blocks": "complete preselection UTC months",
                "month_universe": [str(month) for month in months],
            },
            "runtime_versions": {
                package: version(package)
                for package in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn")
            },
            "source_sha256": hashes,
        },
        "model": {
            "display_name": "C20R",
            "description": "20-token player-history ensemble with regional adjustment; 19 history input slots plus one learned token",
            "weights_model_version": serving["model_version"],
            "serving_model_version": serving["serving_model_version"],
            "feature_version": serving["serving_feature_version"],
            "historical_C0_is_distinct": True,
            "history_input_slots": 19,
            "learned_model_tokens": 1,
            "historical_evaluation_environment": model_binding.get("environment", {}),
        },
        "model_scores": model_scores,
        "financial_scenarios": financial,
        "later_quote_proxy": proxy_summary,
        "scope_limits": [
            "Historical development results; no prospective or held-out confirmation.",
            "ROI is known-label return per unit stake (turnover), not bankroll growth.",
            "Unknown labels leave whole-portfolio P&L and ROI undefined.",
            "First observed is not certified opening price; no authenticated activity, fills, two-other-book hybrid, or full application-gate replay.",
            "Endpoint policy portfolios overlap and are not additive.",
            "Later-price statistic is a recorded-price proxy, not true CLV or executable close.",
            "No fit, inference, acquisition, database access, or network access in this replay.",
        ],
    }
    return snapshot, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new exclusive directory for aggregate summary and replay receipt")
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"exclusive output directory already exists: {output_dir}")
    summary, hashes = replay()
    receipt = {
        "schema": "project_results_replay_receipt_v1",
        "status": "PASS",
        "bootstrap": {"resamples": DRAWS, "seed": SEED, "blocks": "complete preselection UTC months"},
        "source_sha256": hashes,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "project_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_value, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=_json_value, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "source_sha256": hashes},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
