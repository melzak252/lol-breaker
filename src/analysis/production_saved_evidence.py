"""Replay saved production wallets and recorded prestart-price references, not fills."""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from typing import Sequence

import numpy as np
import pandas as pd

from src.analysis.bootstrap import block_ratio_interval


_TRUE_CLV_STATUS = "not_available_no_certified_close_or_fill"


def _aware(value, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _months(months: Sequence[str]) -> list[str]:
    values = list(months)
    if len(values) != len(set(values)):
        raise ValueError("months must be unique")
    for value in values:
        try:
            parsed = pd.Period(value, freq="M")
        except (TypeError, ValueError) as exc:
            raise ValueError("months must contain YYYY-MM values") from exc
        if str(parsed) != value:
            raise ValueError("months must contain YYYY-MM values")
    return values


def _ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    required = ["opportunity_id", "canonical_match_id", "decision_at", "start", "settle_at", "side",
                "bookmaker", "odds", "tax_rate", "stake", "y", "realized_pnl", "settled"]
    missing = set(required) - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger missing fields: {sorted(missing)}")
    out = ledger.copy()
    if out.opportunity_id.isna().any() or out.opportunity_id.duplicated().any():
        raise ValueError("opportunity_id must be present and unique")
    if out.canonical_match_id.isna().any() or out.canonical_match_id.duplicated().any():
        raise ValueError("canonical_match_id must be present and unique per portfolio")
    if out[["bookmaker", "side"]].isna().any().any() or not out.side.isin(["team_a", "team_b"]).all():
        raise ValueError("invalid bookmaker or side")
    for col in ("decision_at", "start"):
        out[col] = pd.Series([_aware(v, col) for v in out[col]], index=out.index,
                             dtype="datetime64[ns, UTC]")
    if (out.decision_at >= out.start).any():
        raise ValueError("decision_at must precede start")
    odds = pd.to_numeric(out.odds, errors="coerce").to_numpy(float)
    taxes = pd.to_numeric(out.tax_rate, errors="coerce").to_numpy(float)
    if not np.isfinite(odds).all() or (odds <= 1).any() or not np.isfinite(taxes).all() or (taxes < 0).any() or (taxes >= 1).any():
        raise ValueError("invalid odds or tax_rate")
    out["odds"] = odds
    out["tax_rate"] = taxes
    stakes = []
    for value in out.stake:
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("invalid Decimal stake") from exc
        if not amount.is_finite() or amount <= 0:
            raise ValueError("stake must be finite and positive")
        stakes.append(amount)
    out["_stake_decimal"] = stakes
    out["stake"] = [float(v) for v in stakes]
    if not out.settled.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("settled must be boolean")
    for i, row in out.iterrows():
        if row.settled:
            if pd.isna(row.y) or row.y not in (0, 1):
                raise ValueError("settled y must be binary")
            if pd.isna(row.settle_at):
                raise ValueError("settled row requires settle_at")
            _aware(row.settle_at, "settle_at")
            try:
                actual = Decimal(str(row.realized_pnl))
            except (InvalidOperation, ValueError) as exc:
                raise ValueError("invalid realized_pnl") from exc
            if not actual.is_finite():
                raise ValueError("realized_pnl must be finite")
            stake = row["_stake_decimal"]
            expected = stake * (Decimal(str(row.odds)) * (Decimal(1) - Decimal(str(row.tax_rate))) - Decimal(1)) if int(row.y) == 1 else -stake
            if abs(actual - expected) > Decimal("1e-8"):
                raise ValueError("realized_pnl does not reconcile to saved result")
        elif not (pd.isna(row.y) and pd.isna(row.realized_pnl) and pd.isna(row.settle_at)):
            raise ValueError("pending outcomes must be null")
    return out


def _ratio(num, den, blocks, months, *, draws, seed):
    frame = pd.DataFrame({"month": blocks, "num": num, "den": den})
    if len(frame) and (~frame.month.isin(months)).any():
        raise ValueError("ticket month outside supplied month universe")
    grouped = frame.groupby("month", sort=False)[["num", "den"]].sum()
    n = [float(grouped.loc[m, "num"]) if m in grouped.index else 0.0 for m in months]
    d = [float(grouped.loc[m, "den"]) if m in grouped.index else 0.0 for m in months]
    return block_ratio_interval(n, d, months, draws=draws, seed=seed)


def summarize_production_wallet(ledger: pd.DataFrame, months: Sequence[str], *, draws=5000, seed=20261023) -> dict:
    data = _ledger(ledger)
    month_values = _months(months)
    settled = data[data.settled]
    count = len(data)
    settled_count = len(settled)
    committed = float(sum(data["_stake_decimal"], Decimal(0)))
    turnover = float(sum(settled["_stake_decimal"], Decimal(0)))
    profit = float(sum((Decimal(str(value)) for value in settled.realized_pnl), Decimal(0)))
    returns = settled.realized_pnl.astype(float).to_numpy() / settled.stake.to_numpy() if settled_count else np.array([])
    starts = settled.start.dt.strftime("%Y-%m").tolist()
    interval = _ratio(settled.realized_pnl.astype(float).tolist(), settled.stake.tolist(), starts, month_values, draws=draws, seed=seed)
    unresolved = settled_count != count
    settled_roi = profit / turnover if turnover else None
    return {"bets": count, "settled_bets": settled_count, "pending_bets": count-settled_count,
            "wins": int((settled.y == 1).sum()), "committed_stake": committed, "settled_turnover": turnover,
            "settled_profit": profit, "settled_roi": settled_roi,
            "profit": None if unresolved else profit, "roi": None if unresolved else settled_roi,
            "unit_yield": float(returns.mean()) if settled_count else None, "roi_interval": interval}


def _json_list(value, field):
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed {field}") from exc
    if not isinstance(result, list):
        raise ValueError(f"{field} must decode to a list")
    return result


def pair_production_closes(ledger: pd.DataFrame, entries: pd.DataFrame, *, reference: str = "300") -> pd.DataFrame:
    if reference not in ("300", "3600", "anyage"):
        raise ValueError("reference must be 300, 3600, or anyage")
    data = _ledger(ledger)
    entry_required = {"snapshot_id", "canonical_match_id", "bookmaker", "entry_clock", "fixture_start_utc", "entry_odds_team_a", "entry_odds_team_b"}
    if not entry_required.issubset(entries.columns):
        raise ValueError("entries missing identity fields")
    if entries.snapshot_id.isna().any() or entries.snapshot_id.duplicated().any():
        raise ValueError("entry snapshot_id must be present and unique")
    joined = []
    lookup = entries.set_index("snapshot_id", drop=False)
    for _, row in data.iterrows():
        parts = str(row.opportunity_id).rsplit(":", 1)
        if len(parts) != 2 or parts[1] != row.side:
            raise ValueError("opportunity_id suffix must match selected side")
        snap = parts[0]
        record = row.to_dict()
        record["snapshot_id"] = snap
        record["fixture_month"] = row.start.strftime("%Y-%m")
        record.update({"samebook_status": "missing_entry", "loo_status": "missing_entry",
                       "samebook_price_move": np.nan, "loo_gross_proxy": np.nan, "loo_net_proxy": np.nan})
        if snap in lookup.index:
            ref = lookup.loc[snap]
            if isinstance(ref, pd.DataFrame):
                raise ValueError("ambiguous entry")
            entry_clock = _aware(ref.entry_clock, "entry_clock")
            start = _aware(ref.fixture_start_utc, "fixture_start_utc")
            decision = _aware(row.decision_at, "decision_at")
            if (ref.canonical_match_id != row.canonical_match_id or ref.bookmaker != row.bookmaker or
                entry_clock != decision or start != row.start or not np.isclose(float(ref["entry_odds_team_a"] if row.side == "team_a" else ref["entry_odds_team_b"]), row.odds, rtol=0, atol=1e-12)):
                raise ValueError("entry metadata does not match selected ticket")
            for domain in ("samebook", "loo"):
                eligible_col = f"{domain}_{reference}_eligible"
                reason_col = f"{domain}_{reference}_missing_reason"
                if eligible_col not in ref.index or reason_col not in ref.index:
                    raise ValueError(f"missing {domain} reference proof")
                eligible = ref[eligible_col]
                if not isinstance(eligible, (bool, np.bool_)):
                    raise ValueError("eligibility must be boolean")
                if not eligible:
                    record[f"{domain}_status"] = str(ref[reason_col]) if pd.notna(ref[reason_col]) and str(ref[reason_col]) else "ineligible"
                    continue
                if domain == "samebook":
                    close = _aware(ref["samebook_close_clock"], "samebook_close_clock")
                    age = float(ref["samebook_age_seconds_to_start"])
                    oa, ob = float(ref["samebook_close_odds_team_a"]), float(ref["samebook_close_odds_team_b"])
                    actual_age = (start-close).total_seconds()
                    limit = float(reference) if reference != "anyage" else math.inf
                    if not (entry_clock < close < start and np.isfinite([age, oa, ob]).all() and oa > 1 and ob > 1 and abs(age-actual_age) <= 1e-6 and age >= 0 and age <= limit):
                        raise ValueError("contradictory samebook reference proof")
                    selected = oa if row.side == "team_a" else ob
                    record["samebook_price_move"] = row.odds / selected - 1
                    record["samebook_status"] = "matched_recorded_prestart_reference"
                else:
                    q = float(ref[f"loo_{reference}_fair_q_team_a"])
                    n = int(ref[f"loo_{reference}_n_books"])
                    books = _json_list(ref[f"loo_{reference}_contributor_books"], "contributor books")
                    ids = _json_list(ref[f"loo_{reference}_contributor_snapshot_ids"], "contributor snapshot ids")
                    clocks = [_aware(x, "contributor clock") for x in _json_list(ref[f"loo_{reference}_contributor_clocks"], "contributor clocks")]
                    ages = _json_list(ref[f"loo_{reference}_contributor_ages_seconds"], "contributor ages")
                    if n < 2 or not (len(books) == len(ids) == len(clocks) == len(ages) == n) or len(set(books)) != n or row.bookmaker in books or not np.isfinite(q) or not 0 < q < 1:
                        raise ValueError("invalid LOO contributor proof")
                    age_values = np.asarray(ages, dtype=float)
                    differences = np.asarray([(start-clock).total_seconds() for clock in clocks])
                    limit = float(reference) if reference != "anyage" else math.inf
                    if not np.isfinite(age_values).all() or not np.allclose(age_values, differences, rtol=0, atol=1e-6) or (age_values < 0).any() or (age_values > limit).any() or any(clock >= start for clock in clocks) or not any(clock > entry_clock for clock in clocks):
                        raise ValueError("invalid LOO contributor clocks or ages")
                    side_q = q if row.side == "team_a" else 1-q
                    record["loo_gross_proxy"] = row.odds * side_q - 1
                    record["loo_net_proxy"] = (1-row.tax_rate) * row.odds * side_q - 1
                    record["loo_status"] = "matched_recorded_prestart_reference"
        joined.append(record)
    return pd.DataFrame(joined, columns=list(dict.fromkeys([
        *data.columns, "snapshot_id", "fixture_month", "samebook_status", "loo_status",
        "samebook_price_move", "loo_gross_proxy", "loo_net_proxy",
    ])))


def summarize_production_closes(pairs: pd.DataFrame, months: Sequence[str], *, draws=5000, seed=20261023) -> dict:
    month_values = _months(months)
    data = pairs.copy()
    if not {"fixture_month", "samebook_status", "loo_status", "samebook_price_move", "loo_gross_proxy", "loo_net_proxy", "stake"}.issubset(data.columns):
        raise ValueError("pairs missing summary fields")
    if data.fixture_month.isna().any() or (~data.fixture_month.isin(month_values)).any():
        raise ValueError("pair month outside supplied month universe")
    def metric(col, weight=None):
        values = pd.to_numeric(data[col], errors="coerce").to_numpy(float)
        covered = data[f"{'samebook' if col == 'samebook_price_move' else 'loo'}_status"].eq("matched_recorded_prestart_reference").to_numpy()
        if not np.array_equal(np.isfinite(values), covered):
            raise ValueError(f"{col} availability inconsistent with status")
        idx = np.flatnonzero(covered)
        if weight is None:
            numerator, denominator = values[idx], np.ones(len(idx))
        else:
            weights = pd.to_numeric(data[weight], errors="coerce").to_numpy(float)[idx]
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("invalid stake weights")
            numerator, denominator = values[idx] * weights, weights
        blocks = data.fixture_month.iloc[idx].tolist()
        return _ratio(numerator.tolist(), denominator.tolist(), blocks, month_values, draws=draws, seed=seed)
    same_abs = data.samebook_status.value_counts(dropna=False).to_dict()
    loo_abs = data.loo_status.value_counts(dropna=False).to_dict()
    same_cov = int(data.samebook_status.eq("matched_recorded_prestart_reference").sum())
    loo_cov = int(data.loo_status.eq("matched_recorded_prestart_reference").sum())
    return {"bets": len(data), "samebook_covered": same_cov, "loo_covered": loo_cov,
            "samebook_absences": {str(k): int(v) for k,v in same_abs.items() if k != "matched_recorded_prestart_reference"},
            "loo_absences": {str(k): int(v) for k,v in loo_abs.items() if k != "matched_recorded_prestart_reference"},
            "samebook_price_move": metric("samebook_price_move"), "loo_gross_proxy": metric("loo_gross_proxy"),
            "loo_net_proxy": metric("loo_net_proxy"),
            "stake_weighted_loo_gross_proxy": metric("loo_gross_proxy", "stake"),
            "stake_weighted_loo_net_proxy": metric("loo_net_proxy", "stake"),
            "true_clv": None, "true_clv_status": _TRUE_CLV_STATUS,
            "reference_scope": "covered recorded prestart references; unknown outcomes retained"}
