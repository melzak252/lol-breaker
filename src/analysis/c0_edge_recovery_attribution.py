"""Audit retained legacy financial claims without fuzzy historical re-joins.

This module only reads the preserved ledgers and saved simulation outputs. It does
not refit, regenerate, or rescore the historical adaptive grid. Legacy bets are
attributable to a replay only when exact fixture, side, bookmaker, quote, price,
and outcome keys are available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.analysis.c0_policy_data import admit_policy_quotes


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = "data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol.json"
LEGACY_LEDGER_PATH = "data/08_reporting/optimal_strategy_simulation_ledger.csv"
LEGACY_REPORT_PATH = "reports/optimal_synergy_c0_thesis_simulation_20261012.md"
OLD_AUDIT_PATH = "review-reports/codebase_logic_audit_20261021.md"
OPTIMIZER_RESULT_PATH = (
    "data/08_reporting/c0_upgrade_and_distillation_20261017/"
    "clv_vs_optimizers_curse_optimization.json"
)
TRAJECTORY_PATH = "data/04_feature/production_2h_odds_trajectory.parquet"

POLICY_RUN_ROOT = (
    "data/08_reporting/c0_policy_evaluation/"
    "c0_full_policy_20261021_01_integrity_v2"
)
INTERVAL_RUN_ROOT = "data/08_reporting/c0_interval_policy/c0_interval_policy_20261022_01"
PRIOR_T1H_LEDGER_PATH = f"{POLICY_RUN_ROOT}/observed24h/complete/ledger.csv"
FIXED_2H_LEDGER_PATH = f"{INTERVAL_RUN_ROOT}/0_48h/first_eligible_2h/ledger.csv"
FIXED_2H_SELECTED_PATH = f"{INTERVAL_RUN_ROOT}/0_48h/first_eligible_2h/selected.parquet"
LEGACY_PREOVERWRITE_PATH = (
    "data/01_raw/c0_edge_recovery/c0_edge_recovery_20261022_01/legacy_50_preoverwrite.csv"
)
YEAR_ANCHOR_RESULT_PATH = "data/08_reporting/c0_upgrade_and_distillation_20261017/year_anchor_distilled.json"
YEAR_ANCHOR_SCRIPT_PATH = "scripts/train_c0_year_anchor_distilled.py"


def cohort_b_provenance(optimization: Mapping[str, Any]) -> dict[str, Any]:
    """Separate an unbound saved optimizer label from the current-source rule."""
    result = optimization.get("verification_results", {}).get("C0-D-Core", {})
    selected = result.get("top_unconstrained_config", {})
    cohort_a = selected.get("cohort_a_5check") or {}
    cohort_b = selected.get("cohort_b_5check") or {}
    bindings = {
        "canonical_fixture_ids": ("canonical_fixture_ids", "fixture_ids", "match_ids"),
        "cohort_time_bounds": ("cohort_time_bounds", "time_bounds", "start_at_bounds"),
        "model_probability_basis": ("model_probability_basis", "probability_basis", "model_basis"),
        "source_hashes": ("source_hashes", "source_hash", "code_hash"),
    }
    missing = [
        binding for binding, aliases in bindings.items()
        if not any(not _is_missing(optimization.get(alias)) for alias in aliases)
    ]
    return {
        "source": OPTIMIZER_RESULT_PATH,
        "saved_result_provenance_status": (
            "SAVED_B_LABEL_PROVENANCE_UNVERIFIED" if missing else "SAVED_B_LABEL_PROVENANCE_BOUND"
        ),
        "missing_binding_fields": missing,
        "saved_timestamp_utc": optimization.get("timestamp_utc"),
        "saved_selection": selected.get("config"),
        "saved_cohort_a": cohort_a,
        "saved_cohort_b_label": cohort_b,
        "probability_weight_convention": {
            "alpha_definition": "sports/model logit weight",
            "alpha_0_5": "equal model and market logit weights",
            "market_weight_0_65_equivalent_model_alpha": 0.35,
        },
        "current_source_rule": "later_match_disjoint_cohort_b",
        "interpretation": (
            "The saved JSON's Cohort-B aggregate is retained as an unbound historical label; "
            "it is not verified as a later/disjoint replay of the current source or as the "
            "actual later 21-bet cohort."
        ),
    }


def _time_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    decision = pd.to_datetime(row.get("decision_at"), utc=True, errors="coerce")
    start = pd.to_datetime(row.get("start_at"), utc=True, errors="coerce")
    quoted = pd.to_datetime(row.get("quoted_at"), utc=True, errors="coerce")
    lead = (start - decision).total_seconds() if not pd.isna(start) and not pd.isna(decision) else None
    age = (decision - quoted).total_seconds() if not pd.isna(quoted) and not pd.isna(decision) else None
    return {
        "decision_at": None if pd.isna(decision) else decision.isoformat(),
        "start_at": None if pd.isna(start) else start.isoformat(),
        "quoted_at": None if pd.isna(quoted) else quoted.isoformat(),
        "decision_lead_seconds": lead,
        "quote_age_seconds": age,
        "quote_within_300s_at_decision": age is not None and 0 <= age <= 300,
    }


def reconcile_t1h_bets(prior: pd.DataFrame, replays: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare prior tickets to replayed bets by exact fixture ID, not suffix or proximity."""
    required = {"opportunity_id", "canonical_match_id", "side", "bookmaker", "odds"}
    missing = required.difference(prior.columns)
    if missing:
        raise ValueError(f"Prior bets lack reconciliation fields: {sorted(missing)}")
    replay_required = {"canonical_match_id", "side", "bookmaker", "odds"}
    rows: list[dict[str, Any]] = []
    for timing, replay in replays.items():
        missing_replay = replay_required.difference(replay.columns)
        if missing_replay:
            raise ValueError(f"{timing} replay lacks reconciliation fields: {sorted(missing_replay)}")
        by_fixture: dict[str, list[dict[str, Any]]] = {}
        for row in replay.to_dict(orient="records"):
            fixture = row.get("canonical_match_id")
            if not _is_missing(fixture):
                by_fixture.setdefault(str(fixture), []).append(row)
        for old in prior.to_dict(orient="records"):
            fixture = old.get("canonical_match_id")
            matches = by_fixture.get(str(fixture), []) if not _is_missing(fixture) else []
            record: dict[str, Any] = {
                "legacy_opportunity_id": old.get("opportunity_id"),
                "canonical_match_id": fixture,
                "timing": timing,
                "legacy_side": old.get("side"),
                "legacy_bookmaker": old.get("bookmaker"),
                "legacy_odds": _finite_float(old.get("odds")),
                "legacy_quote_id": old.get("quote_id"),
                "legacy_reference_quote_id": old.get("reference_quote_id"),
                "legacy_reference_bookmaker_id": old.get("reference_bookmaker_id"),
                "legacy_y": _finite_float(old.get("y")),
                "legacy_stake": _finite_float(old.get("stake")),
                "legacy_realized_pnl": _finite_float(old.get("realized_pnl")),
                "replay_opportunity_id": None,
                "replay_side": None,
                "replay_bookmaker": None,
                "replay_odds": None,
                "replay_bookmaker_id": None,
                "replay_quote_id": None,
                "replay_y": None,
                "replay_stake": None,
                "replay_realized_pnl": None,
                "quote_identity_status": "unknown",
                **{f"legacy_{k}": v for k, v in _time_evidence(old).items()},
                "status": "fixture_not_selected",
            }
            if matches:
                side_match = [r for r in matches if _same(r.get("side"), old.get("side"))]
                if not side_match:
                    record["status"] = "fixture_selected_side_changed"
                else:
                    book_match = [r for r in side_match if _same(r.get("bookmaker"), old.get("bookmaker"))]
                    if not book_match:
                        record["status"] = "same_fixture_side_price_or_book_changed"
                    else:
                        exact_price = [
                            r for r in book_match
                            if _finite_float(r.get("odds")) == _finite_float(old.get("odds"))
                        ]
                        record["status"] = (
                            "same_fixture_side_price_book_match"
                            if exact_price else "same_fixture_side_price_or_book_changed"
                        )
                        chosen = exact_price[0] if exact_price else book_match[0]
                        record.update({
                            "replay_opportunity_id": chosen.get("opportunity_id"),
                            "replay_side": chosen.get("side"),
                            "replay_bookmaker": chosen.get("bookmaker"),
                            "replay_bookmaker_id": chosen.get("bookmaker_id"),
                            "replay_odds": _finite_float(chosen.get("odds")),
                            "replay_quote_id": chosen.get("quote_id"),
                            "replay_y": _finite_float(chosen.get("y")),
                            "replay_stake": _finite_float(chosen.get("stake")),
                            "replay_realized_pnl": _finite_float(chosen.get("realized_pnl")),
                            **{f"replay_{k}": v for k, v in _time_evidence(chosen).items()},
                        })
                        old_quote, new_quote = old.get("quote_id"), chosen.get("quote_id")
                        if not _is_missing(old_quote) and not _is_missing(new_quote):
                            record["quote_identity_status"] = "same" if _same(old_quote, new_quote) else "different"
                        elif _is_missing(old_quote) and _is_missing(new_quote):
                            record["quote_identity_status"] = "both_unknown"
                        else:
                            record["quote_identity_status"] = "partial_unknown"
            rows.append(record)
    result = pd.DataFrame(rows)
    result.attrs["exact_fixture_id_matching"] = True
    result.attrs["probability_matching_used"] = False
    result.attrs["opportunity_suffix_used_as_timing_evidence"] = False
    return result


# The audit's historical assertion and the checked-in current-source behavior are
# separate fields by design. “Historical” means preserved claims/results, not an
# assertion that an unavailable earlier source revision was reproduced here.
CLAIM_STATUS: tuple[dict[str, str], ...] = (
    {
        "claim_id": "F01",
        "claim": "The 90-cell alpha/epsilon/Guard-B/odds search selected an in-sample winner; old Cohort B was a superset of A.",
        "historical_status": "The in-sample 90-cell search is documented; the old Cohort-B-superset statement is not current-source behavior.",
        "current_source_status": "Current script has 90 cells per model; eligibility is selected on Cohort A, then its selected config is evaluated on a strictly later, match-disjoint Cohort B. B is a small descriptive comparison, not independent validation of the adaptive research family.",
        "lineage_and_scope": "scripts/optimize_clv_vs_optimizers_curse.py:221-240,852-857,888-915,932-; saved JSON has 90 rows per model and a selected C0-D-Core Cohort-B result of 21 bets.",
    },
    {
        "claim_id": "F02",
        "claim": "A 43-bet C0-D ledger is a re-filter of the 50 bets selected by C0 plus EXP-039, with stale prices and known outcomes.",
        "historical_status": "The original source says the current 43 trajectory fixtures are a separate support set; a recovered C0-D-Year-Core report has a different 43-bet approximate rescore of the imported 50. Neither establishes a canonical-ID old 43 ledger.",
        "current_source_status": "scripts/run_c0_distilled_financial_simulations.py:403-404 defines 43 trajectory IDs; :455-501 filters ledger rows using team names, date +/-1 day, and probability tolerance 1e-3. Separately, scripts/train_c0_year_anchor_distilled.py reports 43 C0-D-Year-Core bets after a +/-3-day nearest-probability mapping of imported ledger rows; exact IDs are not recovered.",
        "lineage_and_scope": "Old CSV lacks canonical_match_id, side, and quote_id. The fixed 50 and both 43 quantities remain separate; do not treat a fuzzy reconstructed row as exact-ID recovery.",
    },
    {
        "claim_id": "F03",
        "claim": "Probability-proximity/outcome joins can map the old ledger to the wrong match or reverse its side.",
        "historical_status": "A recovered pre-overwrite CSV confirms the same 50 rows and ordering but no exact match/side/quote keys. Session history documents a later rewrite join by same date/book and odds within 0.02; that is not canonical identity, and the original selection join remains unknown.",
        "current_source_status": "Current run_c0_distilled_financial_simulations join is team/date +/-1 day plus 1e-3 probability tolerance; current side orientation uses team identity. Neither makes the missing old ticket identity exact. No probability-only join is used by this recovery module.",
        "lineage_and_scope": "scripts/run_c0_distilled_financial_simulations.py:455-501; scripts/generate_marginal_blocked_bets_audit.py:127-174. Exact old identity support: zero rows established from the CSV schema.",
    },
    {
        "claim_id": "F04",
        "claim": "The advertised 0.65 zero-lookahead LOO rescore of the 50 has no generating code; hybrid alpha semantics were inverted inconsistently.",
        "historical_status": "The recovered earlier CSV predates a later inline p_hybrid/model_ev_pct rewrite. History identifies that later rewrite's approximate same-date/book/odds-within-0.02 join and alpha scan; it does not recover the original creator, original selection generator, canonical IDs, or quote-level LOO basis.",
        "current_source_status": "The current rescore explicitly reconstructs a market probability with (logit(p_hybrid)-0.65*logit(p_C0))/0.35. The current runtime sports weight is alpha=0.35 for a 65% market blend; a value of .65 is not runtime sports alpha.",
        "lineage_and_scope": "scripts/run_c0_distilled_financial_simulations.py:503-507; protocol alpha semantics use SPORTS weight. This current reconstruction does not establish old quote-level LOO provenance.",
    },
    {
        "claim_id": "F05",
        "claim": "The legacy two-hour q_open_loo used future quotes and a market-only price prefilter.",
        "historical_status": "Retained old ledger has no decision quote IDs or timestamp evidence; this timing claim cannot be independently reconstructed from that CSV.",
        "current_source_status": "Current optimizer uses quotes observed at or before decision time, latest known quote per book, and explicitly sets market_prefilter_applied=false. This is a current-code correction, not evidence about an unavailable historical run.",
        "lineage_and_scope": "scripts/optimize_clv_vs_optimizers_curse.py:70-78,245-308; missing old quote lineage remains unknown.",
    },
    {
        "claim_id": "F06",
        "claim": "Missing closing prices defaulted to p_close=0.5, artificially increasing CLV.",
        "historical_status": "The prior audit records this behavior in an earlier script state; no missing-close rows occur in the retained 50 CSV, and no old source archive was found in searched roots.",
        "current_source_status": "Current optimizer returns no close-CLV when close probability is absent; current production simulator emits None for a missing close. No 0.5 fallback in checked code paths.",
        "lineage_and_scope": "scripts/optimize_clv_vs_optimizers_curse.py:81-115; src/analysis/historical_production_simulator.py:324-329. Current behavior must not be retroactively imputed to old results.",
    },
    {
        "claim_id": "F07",
        "claim": "Fair-price cohorts used the same de-vig probability for price and CLV reference, manufacturing a zero or tax-only CLV.",
        "historical_status": "The audit identifies the construction; saved optimizer output explicitly marks Pinnacle fair-odds results clv_by_construction and no independent close reference.",
        "current_source_status": "The checked walk-forward code labels same-consensus fair-price CLV as by construction and nulls unsupported CLV; it does not establish an executable close benchmark.",
        "lineage_and_scope": "scripts/optimize_clv_vs_optimizers_curse.py:96-115,985-1006; saved JSON contains clv_by_construction metadata.",
    },
    {
        "claim_id": "F08",
        "claim": "The T-1h Polish cohort used the best closing odds and the same closing quotes as its price reference.",
        "historical_status": "Preserved audit describes this as a construction, not independent closing-line evidence.",
        "current_source_status": "Current 1,467-series output explicitly says placed odds use closing quotes and independent closing reference is unavailable; it marks CLV unavailable rather than positive evidence.",
        "lineage_and_scope": "scripts/run_walk_forward_1467_polish_and_pinnacle.py:628-637; no separate pre-close ticket quote is retained in the old 50 CSV.",
    },
    {
        "claim_id": "F09",
        "claim": "The multi-year 3,002-bet +7.60% / +6.35% CLV row and related rows were reproducible financial evidence.",
        "historical_status": "The preserved pipeline report marks the former 2021-2026 row NOT_RECOMPUTED; no code path reproduces the literal. The available multi-year calculation uses de-vigged closing averages as both price and reference, while model training overlaps 2021-2025.",
        "current_source_status": "No multi-year ROI/CLV claim is reproduced by this module or inferred from that output. The old claimed row remains unsupported, not corrected by rerunning it.",
        "lineage_and_scope": "data/08_reporting/simulation_pipeline/SIMULATION_REPORT.md:139-141; stage3_betting_simulation.json:223-224; src/analysis/simulation_pipeline.py:1578-1579; input manifest names data/model_predictions_2020_2026.parquet.",
    },
    {
        "claim_id": "F10",
        "claim": "A 99% guaranteed-profit statement at 2,792 bets followed from a normal approximation.",
        "historical_status": "The historical audit records a conditional normal-sampling calculation with assumed fixed mean and hard-coded SD; it is not a guarantee and does not model selected-policy uncertainty or execution availability.",
        "current_source_status": "No current source or recovery output describes a guarantee. Finite historical evidence and CLV-to-yield conversion are not guaranteed 1:1.",
        "lineage_and_scope": "review-reports/codebase_logic_audit_20261021.md:144-147; 2,792 is a conditional calculation, not a recovered sample-size requirement.",
    },
    {
        "claim_id": "F11",
        "claim": "Inference treated 50 clustered bets as IID using bet-level tests/bootstrap.",
        "historical_status": "The prior report explicitly records per-bet IID t-tests/bootstrap for bets clustered in July-August; this does not estimate month-level uncertainty.",
        "current_source_status": "The frozen recovery contract uses paired full-calendar-month resampling and retains undefined resamples. It does not rerun or endorse old IID inference.",
        "lineage_and_scope": "review-reports/codebase_logic_audit_20261021.md:148; protocol bootstrap unit is full UTC calendar month.",
    },
    {
        "claim_id": "F12",
        "claim": "Sharpe 1.56 was interpreted as an annualized Sharpe ratio.",
        "historical_status": "The prior audit identifies mean/sd*sqrt(N) as a t-statistic-like bet-level quantity; annualization and dependence assumptions are unsupported.",
        "current_source_status": "The retained 50 ledger's aggregate JSON records a Sharpe-style ratio of 1.47, not 1.56; neither number is annualized financial evidence without dated return-frequency/variance assumptions.",
        "lineage_and_scope": "review-reports/codebase_logic_audit_20261021.md:149; retained financial_simulation_results.json:409-424.",
    },
    {
        "claim_id": "F13",
        "claim": "The Zero-Disappointment Guarantee was a per-bet DRO bound.",
        "historical_status": "The cited formula is a stake-weighted mean subtraction, not a per-bet uncertainty guarantee; passing an average threshold does not bound each realized bet.",
        "current_source_status": "Current policy stress is p_stress=max(0,p_nominal-epsilon) and stressed-EV eligibility; it is not a distributionally validated ambiguity set or a probability guarantee.",
        "lineage_and_scope": "review-reports/codebase_logic_audit_20261021.md:150; recovery protocol portfolio/DRO definition and historical audit text.",
    },
    {
        "claim_id": "F14",
        "claim": "Trajectory CLV used a focal quote's own Shin reference and inconsistent research/app Shin implementations.",
        "historical_status": "The prior audit reports own-quote Shin as the reference for an earlier trajectory metric; focal-price edge is not independent close CLV.",
        "current_source_status": "Current historical_production_simulator stores p_close CLV separately from shin_edge_vs_focal; current optimizer uses a timestamped closing dictionary. The research betting_guards_a_to_d.devig_shin implementation was corrected from the wrong z=S-1 expression to the exact two-outcome additive solution (4 prior regressions RED, then GREEN; five independent cases match the closed form with zero probability error and at most 2.19e-14 root error). Its near-zero/negative-vig fallback threshold remains 1e-10 versus 1e-4 in the root implementation. Frozen financial q remains multiplicative; this repair does not reclaim ROI or change formal consensus, and does not erase the remaining implementation difference.",
        "lineage_and_scope": "src/analysis/historical_production_simulator.py:324-330; src/analysis/shin_devig.py; src/analysis/betting_guards_a_to_d.py:84-95.",
    },
    {
        "claim_id": "F15",
        "claim": "Some saved financial outputs inferred selected-side orientation from the observed outcome.",
        "historical_status": "The audit identified outcome-based orientation in the cited old generators. Old 50 CSV has selected team text but no side ID, so its side cannot be exact-ID validated.",
        "current_source_status": "Current trajectory reconstruction maps sides through team identities; current marginal audit uses explicit quote/team identity and exact IDs where present. That does not repair missing old ticket side/quote IDs.",
        "lineage_and_scope": "review-reports/codebase_logic_audit_20261021.md:152; scripts/run_c0_distilled_financial_simulations.py:415-443,455-501; scripts/generate_marginal_blocked_bets_audit.py:605-627.",
    },
)


LEGACY_REQUIRED = ("bet_id", "stake", "odds_taken", "closing_odds", "won", "profit", "raw_clv_pct", "net_clv_pct")


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _same(left: Any, right: Any) -> bool:
    if _is_missing(left) or _is_missing(right):
        return False
    return str(left) == str(right)


def attribute_legacy_bets(legacy: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    """Attribute only exact fixture/side/book/quote/price evidence; never fuzzy-match.

    The legacy frame needs `legacy_bet_id`, `canonical_match_id`, `side`, `odds`,
    `bookmaker_id`, and `payout_profit`. Replay uses the same identity fields plus
    `eligible`; quote IDs and outcome checks are required to expose recoverable
    realized profit. Missing legacy identity is preserved as unknown.
    """
    legacy_required = {"legacy_bet_id", "canonical_match_id", "side", "odds", "bookmaker_id", "payout_profit"}
    replay_required = {"canonical_match_id", "side", "odds", "bookmaker_id", "eligible"}
    missing_legacy = legacy_required.difference(legacy.columns)
    missing_replay = replay_required.difference(replay.columns)
    if missing_legacy:
        raise ValueError(f"Legacy bets lack attribution fields: {sorted(missing_legacy)}")
    if missing_replay:
        raise ValueError(f"Replay rows lack attribution fields: {sorted(missing_replay)}")
    if legacy["legacy_bet_id"].duplicated().any():
        raise ValueError("Legacy bet IDs must be unique")

    replay_by_match: dict[str, list[tuple[int, pd.Series]]] = {}
    for position, (_, row) in enumerate(replay.iterrows()):
        match_id = row["canonical_match_id"]
        if not _is_missing(match_id):
            replay_by_match.setdefault(str(match_id), []).append((position, row))

    legacy_match_ids = {
        str(value) for value in legacy["canonical_match_id"] if not _is_missing(value)
    }
    rows: list[dict[str, Any]] = []
    for _, old in legacy.iterrows():
        match_id = old["canonical_match_id"]
        quote_candidates = replay_by_match.get(str(match_id), []) if not _is_missing(match_id) else []
        record: dict[str, Any] = {
            "legacy_bet_id": old["legacy_bet_id"],
            "canonical_match_id": match_id,
            "side": old["side"],
            "bookmaker_id": old["bookmaker_id"],
            "legacy_odds": _finite_float(old["odds"]),
            "status": "identity_unknown",
            "matched_replay_rows": 0,
            "recoverable_profit": np.nan,
            "recoverable_profit_basis": "exact ticket quote and outcome identity required",
        }
        if _is_missing(match_id):
            rows.append(record)
            continue
        if not quote_candidates:
            record["status"] = "coverage_absent"
            rows.append(record)
            continue

        sides = [(position, row) for position, row in quote_candidates if _same(row["side"], old["side"])]
        if not sides:
            record["status"] = "side_changed"
            rows.append(record)
            continue
        books = [(position, row) for position, row in sides if _same(row["bookmaker_id"], old["bookmaker_id"])]
        if not books:
            record["status"] = "book_or_quote_changed"
            rows.append(record)
            continue
        old_odds = _finite_float(old["odds"])
        prices = [
            (position, row) for position, row in books
            if old_odds is not None and _finite_float(row["odds"]) is not None
            and _finite_float(row["odds"]) == old_odds
        ]
        if not prices:
            record["status"] = "exact_identity_price_changed"
            record["matched_replay_rows"] = len(books)
            rows.append(record)
            continue
        eligible_prices = [(position, row) for position, row in prices if row["eligible"] is True or _same(row["eligible"], True)]
        if not eligible_prices:
            record["status"] = "policy_ineligible"
            record["matched_replay_rows"] = len(prices)
            rows.append(record)
            continue

        record["matched_replay_rows"] = len(eligible_prices)
        old_quote = old.get("quote_id")
        exact_quotes = [
            (position, row) for position, row in eligible_prices
            if _same(old_quote, row.get("quote_id"))
        ]
        if not exact_quotes:
            record["status"] = "exact_identity_price_quote_unknown"
            rows.append(record)
            continue
        if "won" not in old.index or "y" not in exact_quotes[0][1].index:
            record["status"] = "exact_quote_outcome_unknown"
            rows.append(record)
            continue
        selected_won = _finite_float(old.get("won"))
        replay_y = _finite_float(exact_quotes[0][1].get("y"))
        expected_won = replay_y if _same(old["side"], "a") else (1.0 - replay_y if replay_y is not None else None)
        if selected_won is None or expected_won is None or selected_won != expected_won:
            record["status"] = "exact_quote_outcome_mismatch_or_unknown"
            rows.append(record)
            continue
        record["status"] = "exact_ticket_and_outcome_supported"
        record["recoverable_profit"] = _finite_float(old["payout_profit"])
        rows.append(record)

    result = pd.DataFrame(rows)
    result.attrs["unmatched_replay_rows"] = int(
        sum(
            1 for value in replay["canonical_match_id"]
            if _is_missing(value) or str(value) not in legacy_match_ids
        )
    )
    result.attrs["probability_matching_used"] = False
    result.attrs["name_or_date_matching_used"] = False
    return result


def reconcile_legacy_ledger(frame: pd.DataFrame) -> dict[str, Any]:
    """Reconcile stored 50-ledger arithmetic and report mismatches transparently."""
    missing = set(LEGACY_REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f"Legacy ledger lacks required accounting fields: {sorted(missing)}")
    n = len(frame)
    if n == 0:
        raise ValueError("Legacy ledger is empty")
    stake = pd.to_numeric(frame["stake"], errors="coerce")
    odds = pd.to_numeric(frame["odds_taken"], errors="coerce")
    won = pd.to_numeric(frame["won"], errors="coerce")
    profit = pd.to_numeric(frame["profit"], errors="coerce")
    raw = pd.to_numeric(frame["raw_clv_pct"], errors="coerce")
    net = pd.to_numeric(frame["net_clv_pct"], errors="coerce")
    closing = pd.to_numeric(frame["closing_odds"], errors="coerce")
    recomputed_profit = np.where(won.eq(1.0), stake * (odds * 0.88 - 1.0), -stake)
    total_stake = float(stake.sum())
    stored_profit = float(profit.sum())
    positive_profit = float(profit.loc[profit.gt(0)].sum())
    negative_profit = float(-profit.loc[profit.lt(0)].sum())
    report_turnover = 7194.04
    report_profit = 1711.67
    report_close_odds = 2.06
    columns_missing = [name for name in ("canonical_match_id", "side", "quote_id", "bookmaker_id") if name not in frame.columns]
    return {
        "source": LEGACY_LEDGER_PATH,
        "rows": int(n),
        "wins": int(won.sum()),
        "months": int(frame["month"].nunique()) if "month" in frame else None,
        "turnover_pln": round(total_stake, 2),
        "stored_profit_pln": round(stored_profit, 2),
        "recomputed_tax12_profit_pln": round(float(np.sum(recomputed_profit)), 5),
        "recomputed_minus_stored_profit_pln": round(float(np.sum(recomputed_profit) - stored_profit), 5),
        "yield_pct": round(stored_profit / total_stake * 100.0, 6) if total_stake else None,
        "bankroll_roi_pct_start_10000": round(stored_profit / 10000.0 * 100.0, 6),
        "mean_odds_taken": round(float(odds.mean()), 6),
        "mean_stored_closing_odds": round(float(closing.mean()), 6),
        "mean_raw_clv_pct": round(float(raw.mean()), 6),
        "median_raw_clv_pct": round(float(raw.median()), 6),
        "positive_raw_clv_count": int(raw.gt(0).sum()),
        "mean_net_clv_pct": round(float(net.mean()), 6),
        "positive_net_clv_count": int(net.gt(0).sum()),
        "profit_factor": round(positive_profit / negative_profit, 6) if negative_profit else None,
        "missing_close_rows": int(closing.isna().sum()),
        "legacy_identity_fields_missing": columns_missing,
        "exact_id_bet_or_price_joins_supported": 0,
        "source_exactness": "No canonical match, selected-side, bookmaker-ID, or quote-ID key exists in the retained CSV; names/date/probability are not substituted.",
        "published_report_comparison": {
            "report_turnover_pln": report_turnover,
            "csv_minus_report_turnover_pln": round(total_stake - report_turnover, 2),
            "report_profit_pln": report_profit,
            "csv_minus_report_profit_pln": round(stored_profit - report_profit, 2),
            "report_mean_close_odds": report_close_odds,
            "csv_minus_report_mean_close_odds": round(float(closing.mean()) - report_close_odds, 6),
            "interpretation": "Retained CSV/financial JSON match each other but do not match the Oct-12 report's turnover, profit, or mean closing odds. This does not identify which row-level ledger is historically correct.",
        },
        "preservation": "Read-only reconciliation; original CSV and historical report are not modified.",
    }




def _claim_csv_rows() -> list[dict[str, str]]:
    return [dict(item) for item in CLAIM_STATUS]


def _json_path(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_50_history_evidence(
    root: Path, retained_ledger_path: Path, version_receipt_path: Path
) -> dict[str, Any]:
    """Bind the recovered pre-overwrite CSV and append source-history limitations."""
    earlier_path = root / LEGACY_PREOVERWRITE_PATH
    earlier_hash = _sha256(earlier_path) if earlier_path.is_file() else None
    retained_hash = _sha256(retained_ledger_path) if retained_ledger_path.is_file() else None
    receipt = (
        json.loads(version_receipt_path.read_text(encoding="utf-8"))
        if version_receipt_path.is_file() else None
    )
    binding_matches = bool(
        receipt
        and receipt.get("earlier_version") == LEGACY_PREOVERWRITE_PATH
        and receipt.get("retained_version") == LEGACY_LEDGER_PATH
        and receipt.get("earlier_version_sha256") == earlier_hash
        and receipt.get("retained_version_sha256") == retained_hash
    )
    year_path = root / YEAR_ANCHOR_RESULT_PATH
    year_report = json.loads(year_path.read_text(encoding="utf-8")) if year_path.is_file() else {}
    year_case = (
        year_report.get("financial_simulations", {})
        .get("polish_production_ledger_cohort", {})
        .get("C0-D-Year-Core", {})
        .get("without_veto_50_cohort", {})
    )
    trajectory_path = root / TRAJECTORY_PATH
    trajectory_support: dict[str, Any] = {
        "source": TRAJECTORY_PATH,
        "interpretation": "Fixture support in the trajectory is not the imported-ledger 43-bet subset.",
    }
    if trajectory_path.is_file():
        trajectory = pd.read_parquet(trajectory_path)
        if "p_model_a" in trajectory and "canonical_match_id" in trajectory:
            with_model = trajectory.loc[trajectory["p_model_a"].notna()]
            trajectory_support.update({
                "rows_with_p_model_a": len(with_model),
                "distinct_canonical_match_ids_with_p_model_a": int(
                    with_model["canonical_match_id"].nunique()
                ),
            })
    history_paths = [
        earlier_path, retained_ledger_path, version_receipt_path,
        root / YEAR_ANCHOR_SCRIPT_PATH, year_path, trajectory_path,
    ]
    history_hashes = {
        str(path.relative_to(root)): _sha256(path) for path in history_paths if path.is_file()
    }
    return {
        "recovered_version_binding_status": (
            "BOTH_VERSION_HASHES_MATCH_RECEIPT" if binding_matches else "VERSION_RECEIPT_UNBOUND_OR_MISMATCHED"
        ),
        "earlier_version": {
            "path": LEGACY_PREOVERWRITE_PATH,
            "sha256": earlier_hash,
            "version_comparison_receipt": (
                str(version_receipt_path.relative_to(root)) if receipt else None
            ),
            "comparison": receipt,
        },
        "retained_version": {
            "path": LEGACY_LEDGER_PATH,
            "sha256": retained_hash,
        },
        "source_history": {
            "evidence_locator": {
                "session_change": "f3cd3c70",
                "operation_call": "call_409441",
                "result_message": "94e2d4ab",
            },
            "scope": "Evidence for a later inline overwrite, not the original creator or selection generator.",
            "original_creator": "UNKNOWN",
            "original_selection_generator": "NOT_RECOVERED",
            "later_overwrite": {
                "identity_join": (
                    "Same date and bookmaker, with odds within 0.02; not canonical fixture, side, or quote IDs."
                ),
                "probability_fallback": "Use own quote probability or 1/(odds*1.055) when the approximate match is unavailable.",
                "alpha_semantics": "alpha=0.65 sports/model logit weight; this is not 65% market weight.",
                "alpha_scan": [0.50, 0.60, 0.65, 0.70, 0.75],
                "selection_target": "Known net CLV in the retained outcome-bearing ledger.",
                "rewritten_fields": ["p_hybrid", "model_ev_pct"],
            },
            "identity_status": {
                "canonical_match_ids": "NOT_RECOVERED",
                "selected_sides": "NOT_RECOVERED",
                "ticket_quote_ids": "NOT_RECOVERED",
            },
        },
        "historical_year_core_rescore": {
            "status": "ALTERNATE_APPROXIMATE_IMPORTED_LEDGER_RESCORE",
            "source_result": YEAR_ANCHOR_RESULT_PATH if year_path.is_file() else None,
            "source_script": YEAR_ANCHOR_SCRIPT_PATH,
            "source_script_sha256": (
                _sha256(root / YEAR_ANCHOR_SCRIPT_PATH)
                if (root / YEAR_ANCHOR_SCRIPT_PATH).is_file() else None
            ),
            "reported_result": year_case or None,
            "reported_43_case": {
                "bets": year_case.get("n_bets"),
                "profit_pln": year_case.get("total_net_profit_pln"),
                "realized_yield_pct": year_case.get("realized_yield_pct"),
            } if year_case else None,
            "lineage": (
                "The script maps each imported ledger row to the nearest C0 probability or its complement "
                "within +/-3 calendar days, then transports the model delta and uses a 0.50 model/0.50 "
                "inverse-reconstructed market logit blend. The reported without-veto mask still applies "
                "Guard-B, odds, EV, and DRO gates; the thesis consensus veto is omitted."
            ),
            "canonical_identity_recovered": False,
            "is_the_trajectory_43_support_set": False,
        },
        "trajectory_43_support": trajectory_support,
        "global_absence_claim": False,
        "source_hashes": history_hashes,
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe scalar/missing values to strict JSON-native records."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def prior_t1h_freshness_counts(prior: pd.DataFrame, quote_states: pd.DataFrame) -> pd.DataFrame:
    """Count exact latest admitted books/peers at each prior decision using the policy helper."""
    required = {"opportunity_id", "canonical_match_id", "decision_at"}
    missing = required.difference(prior.columns)
    if missing:
        raise ValueError(f"Prior opportunities lack clock fields: {sorted(missing)}")
    opportunity_state = prior[["opportunity_id", "canonical_match_id", "decision_at"]].copy()
    fixture_ids = set(prior["canonical_match_id"].dropna().astype(str))
    quotes = quote_states.loc[
        quote_states["canonical_match_id"].notna()
        & quote_states["canonical_match_id"].astype(str).isin(fixture_ids)
    ].copy()
    admitted = admit_policy_quotes(opportunity_state, quotes, ttl_seconds=300)
    rows = []
    for old in prior.to_dict(orient="records"):
        books = admitted.loc[admitted["opportunity_id"].eq(old["opportunity_id"])]
        bookmaker_ids = books["bookmaker_id"].tolist()
        focal_id = old.get("reference_bookmaker_id")
        focal_known = not _is_missing(focal_id)
        focal_is_fresh = any(_same(book_id, focal_id) for book_id in bookmaker_ids) if focal_known else None
        rows.append({
            "opportunity_id": old["opportunity_id"],
            "canonical_match_id": old["canonical_match_id"],
            "decision_at": old["decision_at"],
            "selected_bookmaker_id": focal_id,
            "selected_book_fresh": focal_is_fresh,
            "fresh_bookmaker_count": len(bookmaker_ids),
            "fresh_peer_bookmaker_count": (
                sum(not _same(book_id, focal_id) for book_id in bookmaker_ids)
                if focal_known else None
            ),
            "fresh_bookmaker_ids": bookmaker_ids,
            "fresh_quote_ids": books["quote_id"].tolist(),
            "freshness_seconds": books["freshness_seconds"].tolist(),
        })
    return pd.DataFrame(rows)


def run_recovery_audit(
    protocol: Mapping[str, Any], *, root: Path = ROOT, protocol_path: Path | None = None
) -> dict[str, Any]:
    """Write preserved-claim and exact-ID replay reconciliations without refitting."""
    protocol_source_path = protocol_path or (root / PROTOCOL_PATH)
    if not protocol_source_path.is_absolute():
        protocol_source_path = root / protocol_source_path
    baseline_protocol_path = Path(protocol["baseline_protocol"])
    if not baseline_protocol_path.is_absolute():
        baseline_protocol_path = root / baseline_protocol_path
    output = root / protocol["output_root"] / "recovery"
    ledger_path = root / LEGACY_LEDGER_PATH
    optimizer_path = root / OPTIMIZER_RESULT_PATH
    prior_path = root / PRIOR_T1H_LEDGER_PATH
    fixed_path = root / FIXED_2H_LEDGER_PATH
    fixed_selected_path = root / FIXED_2H_SELECTED_PATH
    oracle_path = root / INTERVAL_RUN_ROOT / "0_48h/max_coverage_model_ev/ledger.csv"
    oracle_selected_path = root / INTERVAL_RUN_ROOT / "0_48h/max_coverage_model_ev/selected.parquet"

    ledger = pd.read_csv(ledger_path)
    optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
    reconciliation = reconcile_legacy_ledger(ledger)
    cohort_b = cohort_b_provenance(optimizer)
    legacy_history = _legacy_50_history_evidence(
        root, ledger_path, output / "legacy_50_version_comparison.json"
    )

    legacy = ledger.rename(columns={"bet_id": "legacy_bet_id", "odds_taken": "odds", "profit": "payout_profit"}).copy()
    for field in ("canonical_match_id", "side", "quote_id", "bookmaker_id"):
        if field not in legacy:
            legacy[field] = pd.NA
    legacy = legacy[["legacy_bet_id", "canonical_match_id", "side", "odds", "bookmaker_id", "payout_profit"]]
    # These are actual saved policy candidate rows. The legacy 50 still has no
    # identity keys, so candidate availability never turns into an attribution.
    reachable_replay = pd.read_parquet(root / f"{POLICY_RUN_ROOT}/observed24h/complete/candidates.parquet")
    replay = reachable_replay.rename(columns={"bookmaker": "bookmaker_id"}).copy()
    attribution = attribute_legacy_bets(legacy, replay)

    prior = pd.read_csv(prior_path)
    fixed = pd.read_csv(fixed_path)
    oracle = pd.read_csv(oracle_path)
    observed300_path = root / f"{POLICY_RUN_ROOT}/observed300/complete/ledger.csv"
    observed300 = pd.read_csv(observed300_path)
    opportunities_path = root / f"{POLICY_RUN_ROOT}/observed24h/complete/opportunities.parquet"
    quotes_path = root / f"{POLICY_RUN_ROOT}/quotes.parquet"
    opportunities = pd.read_parquet(opportunities_path)
    quote_states = pd.read_parquet(quotes_path)
    prior_selected_path = root / f"{POLICY_RUN_ROOT}/observed24h/complete/selected.parquet"
    prior_selected = pd.read_parquet(prior_selected_path)
    fixed_selected = pd.read_parquet(fixed_selected_path)
    oracle_selected = pd.read_parquet(oracle_selected_path)
    observed300_selected_path = root / f"{POLICY_RUN_ROOT}/observed300/complete/selected.parquet"
    observed300_selected = pd.read_parquet(observed300_selected_path)
    def with_quote_evidence(
        ledger_frame: pd.DataFrame, selected_frame: pd.DataFrame, *, prior_reference: bool = False
    ) -> pd.DataFrame:
        columns = [
            "canonical_match_id", "start_at", "side", "bookmaker", "bookmaker_id",
            "odds", "quote_id", "quoted_at",
        ]
        available = selected_frame[[c for c in columns if c in selected_frame.columns]].copy()
        if prior_reference:
            available = available.rename(columns={
                "quote_id": "reference_quote_id",
                "bookmaker_id": "reference_bookmaker_id",
            })
        keys = ["canonical_match_id", "side", "bookmaker", "odds"]
        return ledger_frame.merge(available, on=keys, how="left", validate="one_to_one")

    prior = with_quote_evidence(prior, prior_selected, prior_reference=True)
    prior_clock = prior[["opportunity_id", "canonical_match_id", "reference_bookmaker_id"]].merge(
        opportunities[["opportunity_id", "canonical_match_id", "decision_at", "start_at"]],
        on=["opportunity_id", "canonical_match_id"],
        how="left",
        validate="one_to_one",
    )
    freshness = prior_t1h_freshness_counts(prior_clock, quote_states)
    prior = prior.drop(columns=["decision_at", "start_at"], errors="ignore").merge(
        prior_clock[["opportunity_id", "decision_at", "start_at"]],
        on="opportunity_id", how="left", validate="one_to_one",
    )
    fixed = with_quote_evidence(fixed, fixed_selected)
    oracle = with_quote_evidence(oracle, oracle_selected)
    observed300 = with_quote_evidence(observed300, observed300_selected)
    comparisons = reconcile_t1h_bets(
        prior,
        {
            "first_eligible_fixed_2h": fixed,
            "all_snapshot_oracle": oracle,
            "observed300_comparator": observed300,
        },
    )

    attribution_path = output / "legacy_50_exact_id_attribution.csv"
    claims_path = output / "financial_claim_status_matrix.csv"
    comparison_path = output / "prior_t1h_exact_fixture_reconciliation.csv"
    attribution_path.parent.mkdir(parents=True, exist_ok=True)
    attribution.to_csv(attribution_path, index=False)
    freshness_path = output / "prior_t1h_fresh_book_counts.csv"
    comparisons.to_csv(comparison_path, index=False)
    pd.DataFrame(_claim_csv_rows()).to_csv(claims_path, index=False)
    freshness.to_csv(freshness_path, index=False)
    cohort_b["current_source_rule"] = "later_match_disjoint_cohort_b"
    cohort_b["prior_t1h_reconciliation"] = {
        "prior_ledger_source": PRIOR_T1H_LEDGER_PATH,
        "prior_bet_ids": prior["canonical_match_id"].astype(str).tolist(),
        "fixed_2h_first_eligible_source": FIXED_2H_LEDGER_PATH,
        "fixed_2h_bet_ids": fixed["canonical_match_id"].astype(str).tolist(),
        "all_snapshot_oracle_source": str(oracle_path.relative_to(root)),
        "all_snapshot_oracle_bet_ids": oracle["canonical_match_id"].astype(str).tolist(),
        "observed300_source": str(observed300_path.relative_to(root)),
        "observed300_bet_ids": observed300["canonical_match_id"].astype(str).tolist(),
        "fixed_2h_overlap_count": len(
            set(prior["canonical_match_id"].astype(str)) & set(fixed["canonical_match_id"].astype(str))
        ),
    }
    # Store the exact overlap identities separately from price and quote proof.
    fixed_ids = set(fixed["canonical_match_id"].astype(str))
    oracle_ids = set(oracle["canonical_match_id"].astype(str))
    prior_ids = set(prior["canonical_match_id"].astype(str))
    cohort_b["prior_t1h_reconciliation"]["fixed_2h_overlap_ids"] = sorted(prior_ids & fixed_ids)
    cohort_b["prior_t1h_reconciliation"]["oracle_overlap_ids"] = sorted(prior_ids & oracle_ids)
    cohort_b["prior_t1h_reconciliation"]["quote_identity_proven"] = False
    cohort_b["prior_t1h_reconciliation"]["interpretation"] = (
        f"The five prior 24h bets are compared by exact fixture ID. The fixed 2h "
        f"first-eligible saved ledger overlaps {len(prior_ids & fixed_ids)} prior fixture(s); "
        "the oracle and observed-300s comparisons are separate. Prices, source-candidate "
        "quote IDs, and missing old-ticket quote identity are reported independently. "
        "Opportunity suffixes are not treated as clocks."
    )
    observed300_ids = set(observed300["canonical_match_id"].astype(str))
    cohort_b["prior_t1h_reconciliation"].update({
        "prior_bet_count": len(prior),
        "fixed_2h_overlap_pct_of_prior": round(100.0 * len(prior_ids & fixed_ids) / len(prior), 2) if len(prior) else None,
        "fixed_2h_total_ledger_bet_count": len(fixed),
        "oracle_overlap_count": len(prior_ids & oracle_ids),
        "oracle_total_ledger_bet_count": len(oracle),
        "observed300_overlap_ids": sorted(prior_ids & observed300_ids),
        "observed300_total_ledger_bet_count": len(observed300),
        "observed24h_vs_observed300_note": (
            "Quote ages are measured from each saved selected candidate's quoted_at to decision_at. "
            "Observed24h/observed300 labels and opportunity suffixes are not substituted for clocks."
        ),
        "fixed_2h_missing_pct_of_prior": round(100.0 * (len(prior) - len(prior_ids & fixed_ids)) / len(prior), 2) if len(prior) else None,
        "fixed_2h_overlapping_rows": _json_records(comparisons.loc[
            comparisons["timing"].eq("first_eligible_fixed_2h")
            & comparisons["canonical_match_id"].astype(str).isin(prior_ids & fixed_ids)
        ]),
        "oracle_overlapping_rows": _json_records(comparisons.loc[
            comparisons["timing"].eq("all_snapshot_oracle")
            & comparisons["canonical_match_id"].astype(str).isin(prior_ids & oracle_ids)
        ]),
        "observed300_overlapping_rows": _json_records(comparisons.loc[
            comparisons["timing"].eq("observed300_comparator")
            & comparisons["canonical_match_id"].astype(str).isin(prior_ids & observed300_ids)
        ]),
        "fresh_quote_state_source": str(quotes_path.relative_to(root)),
        "fresh_book_counts_by_prior_fixture": _json_records(freshness),
    })
    source_paths = [
        protocol_source_path, baseline_protocol_path, ledger_path, optimizer_path, prior_path, fixed_path,
        prior_selected_path, fixed_selected_path, oracle_path, oracle_selected_path,
        observed300_path, observed300_selected_path, opportunities_path, quotes_path,
        root / "src/analysis/c0_edge_recovery_attribution.py",
        root / LEGACY_PREOVERWRITE_PATH, output / "legacy_50_version_comparison.json",
        root / YEAR_ANCHOR_SCRIPT_PATH, root / YEAR_ANCHOR_RESULT_PATH, root / TRAJECTORY_PATH,
    ]
    source_paths = [path for path in source_paths if path.is_file()]
    cohort_b["source_hashes"] = {
        str(path.relative_to(root)): _sha256(path) for path in source_paths
    }
    absence = {
        "searched_known_roots": [
            "src/analysis", "scripts", "reports", "review-reports", "data/08_reporting",
            "data/01_raw/c0_edge_recovery/c0_edge_recovery_20261022_01",
        ],
        "file_checks": [
            {
                "path": LEGACY_LEDGER_PATH,
                "result": "present: 50 rows; no canonical_match_id, side, quote_id, or bookmaker_id columns",
            },
            {
                "path": f"{LEGACY_LEDGER_PATH}.dvc",
                "result": "present: md5/size metadata only; no generator or bet/quote identities",
            },
            {
                "path": TRAJECTORY_PATH,
                "result": "present: 7,505 rows, 2,615 rows with p_model_a and 43 distinct canonical_match_id values; trajectory states are not a standalone retained 43-bet ledger",
            },
            {
                "path": "src/analysis/simulation_pipeline.py",
                "result": "loads and summarizes the 50-row CSV; no ledger generator",
            },
            {
                "path": "scripts/run_c0_distilled_financial_simulations.py",
                "result": "reads the 50-row CSV and current trajectory; 43 trajectory IDs; reconstructed ledger joins use team/date/probability rather than exact old ticket IDs",
            },
            {
                "path": OPTIMIZER_RESULT_PATH,
                "result": "present: 90-cell aggregate grid records including some 43-bet cells; not row-level provenance for the old 43-bet fixed-ledger subset",
            },
            {
                "path": LEGACY_REPORT_PATH,
                "result": "present: published 50-bet metrics differ from retained CSV turnover, profit, and mean closing odds",
            },
            {
                "path": OLD_AUDIT_PATH,
                "result": "present: Oct-21 historical claim inventory; current code must be classified separately",
            },
            {
                "path": LEGACY_PREOVERWRITE_PATH,
                "result": "recovered immutable earlier 50-row CSV; identity fields remain absent",
            },
            {
                "path": str((output / "legacy_50_version_comparison.json").relative_to(root)),
                "result": f"receipt-to-current/pre-overwrite hash status: {legacy_history['recovered_version_binding_status']}",
            },
            {
                "path": YEAR_ANCHOR_RESULT_PATH,
                "result": "saved C0-D-Year-Core 43-case aggregate; approximate imported-ledger reconstruction, not an exact-ID ledger",
            },
            {
                "path": YEAR_ANCHOR_SCRIPT_PATH,
                "result": "source code uses +/-3-day nearest-probability mapping and a no-consensus-veto rescore of the imported 50 rows",
            },
        ],
        "retained_43_row_level_ledger": (
            "No exact-ID 43-row ledger recovered. The C0-D-Year-Core 43-case is a separate approximate "
            "rescore of the imported 50; trajectory support of 43 canonical fixtures is a third, distinct quantity."
        ),
        "50_ledger_generator": (
            "Original creator and initial selection generator remain unknown. History locates a later "
            "inline rewrite of p_hybrid/model_ev_pct; the recovered pre-overwrite CSV does not establish "
            "how the initial 50 bets were selected."
        ),
        "probability_outcome_matches_substituted": False,
        "global_absence_claim": False,
        "scope_note": "This is a bounded search of referenced/known roots, not a claim that no such file exists anywhere in all storage.",
    }
    absence["legacy_50_version_binding_status"] = legacy_history["recovered_version_binding_status"]
    absence["legacy_50_rewrite_history"] = legacy_history["source_history"]
    absence["historical_year_core_rescore"] = legacy_history["historical_year_core_rescore"]
    absence["trajectory_43_support"] = legacy_history["trajectory_43_support"]
    _json_path(output / "legacy_50_reconciliation.json", reconciliation)
    _json_path(output / "cohort_b_provenance.json", cohort_b)
    _json_path(output / "legacy_lineage_search.json", absence)
    _json_path(output / "legacy_50_history.json", legacy_history)
    return {
        "claim_status_matrix": str(claims_path),
        "legacy_50_exact_id_attribution": str(attribution_path),
        "prior_t1h_exact_fixture_reconciliation": str(comparison_path),
        "legacy_50_reconciliation": str(output / "legacy_50_reconciliation.json"),
        "cohort_b_provenance": str(output / "cohort_b_provenance.json"),
        "legacy_lineage_search": str(output / "legacy_lineage_search.json"),
        "legacy_50_history": str(output / "legacy_50_history.json"),
        "prior_t1h_fresh_book_counts": str(freshness_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit preserved C0 financial claims without fuzzy joins or old-grid refits")
    parser.add_argument("--protocol", default=PROTOCOL_PATH)
    args = parser.parse_args(argv)
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = run_recovery_audit(protocol, protocol_path=protocol_path)
    print(json.dumps(paths, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
