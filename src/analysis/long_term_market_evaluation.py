"""Source-separated, fixed-policy historical endpoint diagnostics.

These reducers use real decimal prices and the application EV/shrinkage/DRO
operators. Archival endpoint diagnostics are not a cash-timeline backtest or
proof that all quoted books were simultaneously available.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from betting_app.core.ev import expected_value
from betting_app.core.models.engine import bayesian_logit_shrinkage
from src.analysis.betting_guards_a_to_d import apply_guard_w_wasserstein_dro, devig_multiplicative
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.c0_edge_alpha_audit import canonical_json_dumps
from src.analysis.probability_metrics import binary_log_loss_vector

ROOT = Path(__file__).resolve().parents[2]
KEY = ["source", "source_event_id", "horizon"]


def _utc(value):
    if value is None or pd.isna(value):
        return pd.NaT
    stamp = pd.Timestamp(value)
    return pd.NaT if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _valid_prices(frame: pd.DataFrame) -> pd.Series:
    a = pd.to_numeric(frame.odds_a, errors="coerce")
    b = pd.to_numeric(frame.odds_b, errors="coerce")
    return np.isfinite(a) & np.isfinite(b) & a.gt(1) & b.gt(1)


def endpoint_loo_consensus(quotes: pd.DataFrame) -> pd.DataFrame:
    """Exclude the focal book; require two distinct other endpoint prices."""
    result = quotes.copy()
    if result.duplicated(KEY + ["bookmaker"]).any():
        raise ValueError("duplicate source/event/horizon/book prices")
    result["q"] = np.nan
    result["peer_books"] = 0
    valid = _valid_prices(result)
    q = pd.Series(np.nan, index=result.index, dtype=float)
    for i, row in result.loc[valid].iterrows():
        q.at[i] = devig_multiplicative(float(row.odds_a), float(row.odds_b))[0]
    grouped = q.groupby([result[c] for c in KEY], dropna=False)
    count = grouped.transform("count")
    total = grouped.transform("sum")
    peers = count - valid.astype(int)
    result["peer_books"] = peers.astype(int)
    mask = valid & peers.ge(2)
    result.loc[mask, "q"] = (total[mask] - q[mask]) / peers[mask]
    result["q_focal"] = q
    return result


def timed_loo_consensus(quotes: pd.DataFrame, *, freshness_seconds: int = 300) -> pd.DataFrame:
    """Reconstruct peer endpoint states at each focal clock, never interpolate gaps.

    Sparse endpoint tapes establish only their retained states. Known expiry is
    exclusive. Missing active-state proof cannot qualify a timed market context.
    """
    result = quotes.copy().reset_index(drop=True)
    result["q_timed"] = np.nan
    result["timed_peer_books"] = 0
    result["timed_peer_evidence"] = "[]"
    if result.empty:
        return result
    work = result.copy()
    for name in ("quote_at", "quote_at_a", "quote_at_b", "source_start_at", "valid_until_exclusive_at"):
        work[name] = pd.to_datetime(work.get(name, pd.Series(None, index=work.index)).map(_utc), utc=True)
    work["_valid"] = _valid_prices(work)
    work["_active"] = work.get("active_state_proven", pd.Series(False, index=work.index)).fillna(False).eq(True)
    if not work["_active"].any():
        return result
    work["_row_index"] = work.index
    for _, event in work.groupby(["source", "source_event_id"], dropna=False, sort=False):
        if not event["_active"].any():
            continue
        rows = event.to_dict("records")
        books = {}
        for row in rows:
            books.setdefault(row["bookmaker"], []).append(row)
        for focal in rows:
            i, decision = focal["_row_index"], focal["quote_at"]
            if pd.isna(decision) or pd.isna(focal["source_start_at"]) or decision >= focal["source_start_at"]:
                continue
            evidence = []
            for book in sorted(books):
                if book == focal["bookmaker"]:
                    continue
                observed = [row for row in books[book] if pd.notna(row["quote_at"]) and row["quote_at"] <= decision]
                if not observed:
                    continue
                latest = max(row["quote_at"] for row in observed)
                states = [row for row in observed if row["quote_at"] == latest]
                # A later invalid/expired state supersedes an earlier price.
                # Conflicting simultaneous states cannot supply a context.
                keys = ("odds_a", "odds_b", "quote_at_a", "quote_at_b",
                        "valid_until_exclusive_at", "_active", "_valid")
                variants = {tuple(None if pd.isna(row[key]) else row[key] for key in keys) for row in states}
                if len(variants) != 1:
                    continue
                row = states[0]
                if not row["_valid"] or not row["_active"]:
                    continue
                ages = [(decision - row[c]).total_seconds() if pd.notna(row[c]) else math.inf
                        for c in ("quote_at_a", "quote_at_b")]
                if not all(0 <= age <= freshness_seconds for age in ages):
                    continue
                expiry = row["valid_until_exclusive_at"]
                if pd.notna(expiry) and decision >= expiry:
                    continue
                q = devig_multiplicative(float(row["odds_a"]), float(row["odds_b"]))[0]
                evidence.append({"bookmaker": str(book), "source_row": str(row.get("source_row", "")),
                                 "quote_at": row["quote_at"].isoformat(), "q": q})
            result.at[i, "timed_peer_books"] = len(evidence)
            result.at[i, "timed_peer_evidence"] = json.dumps(evidence, sort_keys=True)
            if len(evidence) >= 2:
                result.at[i, "q_timed"] = float(np.mean([v["q"] for v in evidence]))
    return result


def select_fixed_policy(panel: pd.DataFrame, *, sports_alpha: float, tax_rate: float,
                        epsilon: float, minimum_ev: float = 0.05,
                        maximum_odds: float = 4.0, timed: bool = False) -> pd.DataFrame:
    """Batch input filtering; canonical scalar operators decide financial boundaries."""
    if not math.isfinite(sports_alpha) or not 0 <= sports_alpha <= 1:
        raise ValueError("sports_alpha must be in [0, 1]")
    eligible = panel.get("archive_eligible", pd.Series(True, index=panel.index)).fillna(False).astype(bool)
    pool = panel.loc[eligible].copy()
    p = pd.to_numeric(pool.p_c0, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    q = pd.to_numeric(pool.q, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    required = np.ones(len(pool), dtype=bool)
    if sports_alpha > 0:
        required &= np.isfinite(p) & (p >= 0) & (p <= 1)
    if sports_alpha < 1:
        required &= np.isfinite(q) & (q > 0) & (q < 1)
    pool = pool.iloc[np.flatnonzero(required)].copy()
    p, q = p[required], q[required]
    if sports_alpha == 0:
        probabilities = np.clip(q, 1e-6, 1 - 1e-6)
    elif sports_alpha == 1:
        probabilities = np.clip(p, 1e-6, 1 - 1e-6)
    else:
        probabilities = np.fromiter(
            (bayesian_logit_shrinkage(float(a), float(b), sports_alpha) for a, b in zip(p, q)),
            dtype=float, count=len(p))
    candidates = []
    for side, probs in (("a", probabilities), ("b", 1 - probabilities)):
        odds = pd.to_numeric(pool[f"odds_{side}"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        positions = np.flatnonzero(np.isfinite(odds) & (odds > 1) & (odds <= maximum_odds))
        if not len(positions):
            continue
        guards = np.asarray([
            apply_guard_w_wasserstein_dro(float(probs[i]), float(odds[i]), tax_rate=tax_rate,
                                          base_min_ev=minimum_ev, epsilon=epsilon)
            for i in positions], dtype=float)
        accepted = guards[:, 0].astype(bool)
        positions = positions[accepted]
        if not len(positions):
            continue
        rows = pool.iloc[positions].copy()
        rows["side"] = side
        rows["p_selected"] = probs[positions]
        rows["odds"] = odds[positions]
        rows["sports_alpha"], rows["tax_rate"], rows["epsilon"] = sports_alpha, tax_rate, epsilon
        rows["ev_net"] = np.fromiter(
            (expected_value(float(a), float(b), tax_rate) for a, b in zip(probs[positions], odds[positions])),
            dtype=float, count=len(positions))
        rows["ev_dro"], rows["effective_hurdle"] = guards[accepted, 1], guards[accepted, 2]
        candidates.append(rows)
    if not candidates:
        return pd.DataFrame(columns=[*panel.columns, "side", "p_selected", "odds", "sports_alpha",
                                     "tax_rate", "epsilon", "ev_net", "ev_dro", "effective_hurdle"])
    result = pd.concat(candidates, ignore_index=True)
    fixture = ["source", "horizon", "golgg_match_id"]
    if timed:
        result["_decision_sort"] = pd.to_datetime(result.quote_at.map(_utc), utc=True)
        result = result.loc[result._decision_sort.notna()]
    order = fixture + (["_decision_sort"] if timed else []) + ["ev_net", "bookmaker", "side", "source_event_id"]
    ascending = [True] * len(fixture) + ([True] if timed else []) + [False, True, True, True]
    result = result.sort_values(order, ascending=ascending, kind="stable")
    return result.drop_duplicates(fixture, keep="first").drop(columns="_decision_sort", errors="ignore").reset_index(drop=True)



def summarize_unit_returns(selected: pd.DataFrame, universe_months: Sequence[str], *,
                           draws: int = 5000, seed: int = 20261022, stake: float = 100) -> dict[str, Any]:
    """Flat-price diagnostic, not an invented cash or settlement chronology."""
    months = sorted(set(map(str, universe_months)))
    n = len(selected)
    known = selected.y.isin([0, 1]) if "y" in selected else pd.Series(False, index=selected.index)
    if "label_status" in selected:
        known &= selected.label_status.eq("known")
    n_known = int(known.sum())
    observed = selected.loc[known]
    win_values = np.where(observed.side.eq("a"), observed.y, 1 - observed.y).astype(int)
    returns = np.fromiter(
        (expected_value(int(w), float(o), float(t)) for w, o, t in
         zip(win_values, observed.odds, observed.tax_rate)), dtype=float, count=len(observed))
    errors = pd.to_numeric(observed.ev_net).to_numpy(dtype=float) - returns
    observed_months = pd.to_datetime(observed.canonical_date, errors="raise").dt.strftime("%Y-%m").to_numpy()
    if not set(observed_months).issubset(months):
        raise ValueError("selected fixture outside complete month universe")
    near = observed.ev_net.gt(0.05).to_numpy() & observed.ev_net.le(0.10).to_numpy()
    totals = pd.DataFrame({"month": observed_months, "return": returns, "n": np.ones(len(observed)),
                           "error": errors, "near_return": np.where(near, returns, 0),
                           "near_n": near.astype(float), "near_error": np.where(near, errors, 0)})
    totals = totals.groupby("month").sum().reindex(months, fill_value=0.0)
    wins = int(win_values.sum())
    expected = math.fsum(pd.to_numeric(observed.p_selected).to_numpy(dtype=float))
    def interval(numerator, denominator):
        return block_ratio_interval(totals[numerator], totals[denominator], totals.index,
                                    draws=draws, seed=seed)
    roi = interval("return", "n")
    error = interval("error", "n")
    near_error = interval("near_error", "near_n")
    profit = math.fsum(returns) * stake
    unknown = n - n_known
    status = "EMPTY_SELECTION" if not n else "UNKNOWN_SELECTED_LABELS" if unknown else "KNOWN_LABELS_ONLY_DIAGNOSTIC"
    return {"bets": n, "known_labels": n_known, "unknown_labels": unknown, "wins": wins,
            "expected_wins_known_labels": expected, "turnover": stake * n,
            "known_label_turnover": stake * n_known, "profit": None if unknown else profit,
            "known_label_profit": profit, "roi": None if unknown else roi["estimate"],
            "known_label_roi": roi["estimate"], "return_interval": roi,
            "return_error_interval": error, "near_return_error_interval": near_error,
            "near_bets_known_labels": int(totals.near_n.sum()),
            "near_reliability_pass": bool(near_error["ci_lower"] is not None and
                                          near_error["ci_lower"] >= -0.02 and near_error["ci_upper"] <= 0.02 and not unknown),
            "status": status, "cash_timeline": "NOT_ESTABLISHED_BY_ENDPOINT_DIAGNOSTIC",
            "monthly_totals": totals.reset_index(names="month").to_dict("records")}


def _scores(y, p) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"n": 0, "log_loss": None, "brier": None}
    if not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
        raise ValueError("proper-score probabilities must be finite and strictly interior; never clip a corrupted native output")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("proper scores require known binary labels")
    losses = binary_log_loss_vector(y, p, epsilon=0.0)
    return {"n": len(y), "log_loss": float(losses.mean()), "brier": float(np.mean((p - y) ** 2))}


def _write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Preserve existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_dumps(value) + "\n")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_archival_panel(events: pd.DataFrame, quotes: pd.DataFrame,
                         predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep all price rows and annotate forecast/identity/label/clock eligibility."""
    events = events.copy()
    quotes = quotes.copy()
    predictions = predictions.copy()
    for frame in (events, predictions):
        frame["golgg_match_id"] = frame.golgg_match_id.astype("string")
    if predictions.golgg_match_id.dropna().duplicated().any():
        raise ValueError("duplicate canonical model predictions")
    if events.duplicated(["source", "source_event_id"]).any():
        raise ValueError("duplicate source events")
    event_columns = [c for c in events if c not in quotes or c in ("source", "source_event_id")]
    panel = quotes.merge(events[event_columns], on=["source", "source_event_id"], how="left", validate="many_to_one")
    pcol = "p_c0" if "p_c0" in predictions else "p_full_full"
    if pcol not in predictions:
        raise ValueError("native prediction stream has no assembled full-mode probability")
    needed = {"team1_id", "team2_id", "y"}
    if not needed.issubset(predictions.columns):
        raise ValueError("native predictions require exact ordered team IDs and labels")
    pred = predictions[["golgg_match_id", "team1_id", "team2_id", "y", pcol]].rename(
        columns={pcol: "p_native", "team1_id": "native_team_a_id",
                 "team2_id": "native_team_b_id", "y": "native_y"})
    panel = panel.merge(pred, on="golgg_match_id", how="left", validate="many_to_one")
    source_a, source_b = panel.canonical_team_a_id.astype("string"), panel.canonical_team_b_id.astype("string")
    native_a, native_b = panel.native_team_a_id.astype("string"), panel.native_team_b_id.astype("string")
    same = (source_a.eq(native_a) & source_b.eq(native_b)).fillna(False)
    reverse = (source_a.eq(native_b) & source_b.eq(native_a)).fillna(False)
    available = panel.p_native.notna()
    valid_pair = (same | reverse) & source_a.ne(source_b).fillna(False) & source_a.ne("").fillna(False)
    if (available & ~valid_pair).any():
        raise ValueError("native/source team pair mismatch; never infer orientation from outcomes")
    panel["native_a_is_canonical_a"] = same.where(available)
    panel["p_c0"] = np.where(reverse, 1 - panel.p_native, panel.p_native)
    panel["native_y_canonical"] = np.where(reverse, 1 - panel.native_y, panel.native_y)
    canonical_y = panel.get("canonical_y", panel.y)
    known_pair = available & canonical_y.isin([0, 1]) & panel.native_y_canonical.isin([0, 1])
    if (known_pair & canonical_y.ne(panel.native_y_canonical)).any():
        raise ValueError("native/canonical label disagreement after exact team-pair orientation")
    valid = _valid_prices(panel)
    before = pd.Series(True, index=panel.index)
    for name in ("quote_at_a", "quote_at_b", "source_start_at"):
        panel[name] = pd.to_datetime(panel[name].map(_utc), utc=True)
    for name in ("quote_at_a", "quote_at_b"):
        # Unknown source time remains an explicitly undated archival diagnostic;
        # a known after-start timestamp can never be repaired into prematch.
        before &= panel[name].isna() | panel.source_start_at.isna() | panel[name].lt(panel.source_start_at)
    panel["archive_eligible"] = valid & before & panel.golgg_match_id.notna()
    panel["clock_basis"] = np.where(panel[["quote_at_a", "quote_at_b", "source_start_at"]].notna().all(axis=1),
                                    "known_source_endpoint_clocks", "unknown_source_endpoint_clock")
    # Ineligible peers cannot define a supposedly prematch endpoint consensus.
    context = panel.loc[valid & before].copy()
    context = endpoint_loo_consensus(context)
    panel["q"] = np.nan
    panel["peer_books"] = 0
    panel.loc[context.index, "q"] = context.q
    panel.loc[context.index, "peer_books"] = context.peer_books
    return panel


def _combined_ledger(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Keep heterogeneous provider identifiers losslessly serializable."""
    result = pd.concat(frames, ignore_index=True)
    identifiers = [name for name in result if name.endswith("_id")]
    result[identifiers] = result[identifiers].astype("string")
    return result


def run_evaluation(protocol_path: Path, source_layout_path: Path, predictions_path: Path,
                   output_dir: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 2 or protocol.get("run_id") != "long_term_market_edge_20261022_01":
        raise ValueError("unsupported frozen long-term financial protocol")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new exclusive output directory: {output_dir}")
    sources = protocol["scope"]["sources"]
    source_layout = json.loads(source_layout_path.read_text())["sources"]
    inputs = {str(protocol_path): _sha(protocol_path), str(predictions_path): _sha(predictions_path),
              str(source_layout_path): _sha(source_layout_path)}
    for source in sources:
        directory = Path(source_layout[source]["root"])
        for name in ("events.parquet", "quotes.parquet", source_layout[source]["receipt"]):
            path = directory / name
            inputs[str(path)] = _sha(path)
    code_paths = [Path(__file__), ROOT / "betting_app/core/ev.py", ROOT / "betting_app/core/models/engine.py",
                  ROOT / "src/analysis/betting_guards_a_to_d.py", ROOT / "src/analysis/bootstrap.py",
                  ROOT / "src/analysis/metrics.py"]
    inputs.update({str(path.relative_to(ROOT)): _sha(path) for path in code_paths})
    _write_json(output_dir / "execution_binding.json", {"protocol_sha256": _sha(protocol_path), "input_and_code_sha256": inputs,
                "before_financial_scoring": True, "fixed_policy_selection": "NONE"})
    predictions = pd.read_parquet(predictions_path)
    summaries, quality, yearly, selected_frames, missingness = [], [], [], [], []
    finance = protocol["fixed_financial_controls"]
    for source in sources:
        directory = Path(source_layout[source]["root"])
        events = pd.read_parquet(directory / "events.parquet")
        quotes = pd.read_parquet(directory / "quotes.parquet")
        panel = build_archival_panel(events, quotes, predictions)
        panel["source_year"] = pd.to_numeric(panel.source_year, errors="coerce")
        panel = panel.loc[panel.source_year.isin(protocol["scope"]["years"])].copy()
        panel.to_parquet(output_dir / f"{source}_archival_panel.parquet", index=False)
        missingness.append({"source": source, "events_inventory": len(events), "events_requested_years": int(events.source_year.isin(protocol["scope"]["years"]).sum()),
                            "quote_rows": len(panel), "mapped_quote_rows": int(panel.golgg_match_id.notna().sum()),
                            "native_quote_rows": int(panel.p_c0.notna().sum()), "archive_eligible_rows": int(panel.archive_eligible.sum()),
                            "loo_supported_rows": int((panel.archive_eligible & panel.q.notna()).sum()),
                            "clock_basis": panel.clock_basis.value_counts().to_dict()})
        for horizon, group in panel.groupby("horizon", sort=True):
            source_dates = pd.to_datetime(events.loc[events.source_year.isin(protocol["scope"]["years"]), "source_start_at"], utc=True, errors="coerce").dropna()
            canonical_dates = pd.to_datetime(events.loc[events.source_year.isin(protocol["scope"]["years"]), "canonical_date"], utc=True, errors="coerce").dropna()
            dates = pd.concat([source_dates, canonical_dates]).dt.tz_localize(None)
            months = list(pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M").astype(str)) if len(dates) else []
            usable = group.loc[group.archive_eligible & group.q.notna() & group.p_c0.notna()]
            policy_pool = group.loc[group.archive_eligible]
            # One deterministic fixture row, not every book as an independent game.
            scored = usable.loc[usable.y.isin([0, 1]) & usable.label_status.eq("known")].sort_values(
                ["golgg_match_id", "bookmaker", "source_event_id"], kind="stable").drop_duplicates("golgg_match_id")
            for year, cohort in [("ALL", scored), *list(scored.groupby("source_year", sort=True))]:
                for alpha in finance["sports_alpha"]:
                    probs = np.array([bayesian_logit_shrinkage(float(p), float(q), alpha) for p, q in zip(cohort.p_c0, cohort.q)])
                    raw = _scores(cohort.y, cohort.p_c0)
                    metric = _scores(cohort.y, probs)
                    market = _scores(cohort.y, cohort.q)
                    contrast = None
                    if len(cohort):
                        delta = binary_log_loss_vector(cohort.y, probs, epsilon=0.0) - binary_log_loss_vector(cohort.y, cohort.q, epsilon=0.0)
                        blocks = pd.to_datetime(cohort.canonical_date).dt.strftime("%Y-%m")
                        contrast = block_ratio_interval(delta, np.ones(len(cohort)), blocks, draws=5000, seed=20261022)
                    quality.append({"source": source, "horizon": horizon, "scenario": "target_roster_daily_endpoint_comparison",
                                    "year": str(year), "sports_alpha": alpha, **metric, "raw_c0_log_loss": raw["log_loss"],
                                    "raw_c0_brier": raw["brier"], "market_log_loss": market["log_loss"],
                                    "market_brier": market["brier"], "delta_vs_market": contrast})
            for alpha, tax, epsilon in itertools.product(finance["sports_alpha"], finance["tax_fractions"], finance["dro_probability_haircut"]):
                policy_id = f"a{alpha:g}_tax{tax:g}_dro{epsilon:g}"
                selected = select_fixed_policy(policy_pool, sports_alpha=alpha, tax_rate=tax, epsilon=epsilon,
                    minimum_ev=finance["minimum_net_ev"], maximum_odds=finance["maximum_decimal_odds"])
                selected["policy_id"] = policy_id
                selected["scenario"] = "archival_endpoint_target_roster_PRICE_DIAGNOSTIC_NOT_EXECUTION"
                selected_frames.append(selected)
                result = summarize_unit_returns(selected, months, draws=5000, seed=20261022, stake=finance["static_stake"])
                summaries.append({"source": source, "horizon": horizon, "policy_id": policy_id,
                                  "sports_alpha": alpha, "tax_rate": tax, "epsilon": epsilon, **result})
                for year in protocol["scope"]["years"]:
                    subset = selected.loc[pd.to_datetime(selected.canonical_date, errors="coerce").dt.year.eq(year)]
                    ym = [m for m in months if m.startswith(str(year))]
                    yearly.append({"source": source, "horizon": horizon, "policy_id": policy_id, "year": year,
                                   **summarize_unit_returns(subset, ym, draws=5000, seed=20261022, stake=finance["static_stake"])})
    pd.DataFrame(quality).to_parquet(output_dir / "probability_quality.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(output_dir / "policy_summaries.parquet", index=False)
    pd.DataFrame(yearly).to_parquet(output_dir / "yearly_policy_summaries.parquet", index=False)
    _combined_ledger(selected_frames).to_parquet(output_dir / "selected_price_diagnostics.parquet", index=False)
    receipt = {"status": "ARCHIVAL_PRICE_DIAGNOSTICS_COMPLETE", "probability_rows": len(quality),
               "policy_rows": len(summaries), "yearly_policy_rows": len(yearly), "source_missingness": missingness,
               "timed_evaluation": "SEPARATE_REQUIRED_SOURCE_CLOCK_AND_PRIOR_ROSTER_STAGE",
               "cash_timeline": "NOT_RUN_ARCHIVAL_ENDPOINTS_DO_NOT_ESTABLISH_EXECUTION_CLOCKS",
               "policy_selected": None, "promotion": False, "source_overlap_independent_replication": False}
    _write_json(output_dir / "archival_receipt.json", receipt)
    return receipt


def canonical_start_bounds(source_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                           query_tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Minimum known start for an exact fixture across all retained sources."""
    records = []
    fields = ("source_start_at", "scheduled_start_at", "true_start_at",
              "quote_source_start_at", "quote_scheduled_start_at", "quote_true_start_at")
    for source, (events, quotes) in source_tables.items():
        event_ids = events.set_index("source_event_id").golgg_match_id
        frames = [events, quotes]
        if query_tables is not None and source in query_tables:
            frames.append(query_tables[source])
        for frame in frames:
            if frame.empty:
                continue
            ids = frame.golgg_match_id if "golgg_match_id" in frame else frame.source_event_id.map(event_ids)
            for field in fields:
                if field not in frame:
                    continue
                rows = pd.DataFrame({"golgg_match_id": ids.astype("string"),
                    "canonical_earliest_start_at": pd.to_datetime(frame[field].map(_utc), utc=True)})
                records.append(rows.dropna())
    if not records:
        return pd.DataFrame(columns=["golgg_match_id", "canonical_earliest_start_at"])
    return pd.concat(records, ignore_index=True).groupby(
        "golgg_match_id", as_index=False).canonical_earliest_start_at.min()


def build_timed_panel(events: pd.DataFrame, quotes: pd.DataFrame,
                      queries: pd.DataFrame, forecasts: pd.DataFrame,
                      target_releases: pd.DataFrame, canonical_starts: pd.DataFrame, *,
                      years: Sequence[int] | None = None) -> pd.DataFrame:
    """Join exact quote requests to native daily states; admit prices independently."""
    quotes = quotes.copy()
    quotes["quote_index"] = np.arange(len(quotes))
    event_columns = [c for c in events if c not in quotes or c in ("source", "source_event_id")]
    panel = quotes.merge(events[event_columns], on=["source", "source_event_id"],
                         how="left", validate="many_to_one")
    query_columns = ["quote_index", "forecast_key", "forecast_absence", "decision_at", "source_start_at"]
    request_map = queries[query_columns].rename(columns={
        "decision_at": "native_decision_at", "source_start_at": "native_source_start_at"})
    if request_map.quote_index.duplicated().any():
        raise ValueError("duplicate native quote requests")
    if not request_map.quote_index.isin(panel.quote_index).all():
        raise ValueError("native request ledger references unknown original quote indices")
    request_map["native_request_emitted"] = True
    panel = panel.merge(request_map, on="quote_index", how="left", validate="one_to_one")
    panel["native_request_emitted"] = panel.native_request_emitted.fillna(False).astype(bool)
    panel["requested_source_year"] = panel.source_year.isin(years) if years is not None else True
    missing = ~panel.native_request_emitted
    panel.loc[missing, "forecast_absence"] = np.where(
        panel.loc[missing, "requested_source_year"], "quote_request_not_emitted_by_native_preparation",
        "source_year_outside_requested_scope")
    native = forecasts[["forecast_key", "p_c0", "source_team_a_id", "source_team_b_id",
                        "decision_at", "model_year"]].rename(columns={
        "p_c0": "p_native", "source_team_a_id": "native_team_a_id",
        "source_team_b_id": "native_team_b_id", "decision_at": "native_representative_at"})
    panel = panel.merge(native, on="forecast_key", how="left", validate="many_to_one")
    a, b = panel.canonical_team_a_id.astype("string"), panel.canonical_team_b_id.astype("string")
    na, nb = panel.native_team_a_id.astype("string"), panel.native_team_b_id.astype("string")
    same = (a.eq(na) & b.eq(nb)).fillna(False)
    reverse = (a.eq(nb) & b.eq(na)).fillna(False)
    available = panel.p_native.notna()
    if (available & ~(same | reverse)).any():
        raise ValueError("timed native/source ordered team pair mismatch")
    panel["p_c0"] = np.where(reverse, 1 - panel.p_native, panel.p_native)
    for name in ("quote_at_a", "quote_at_b", "source_start_at", "scheduled_start_at", "true_start_at",
                 "native_decision_at", "native_source_start_at", "native_representative_at",
                 "valid_until_exclusive_at"):
        panel[name] = pd.to_datetime(panel.get(name, pd.Series(None, index=panel.index)).map(_utc), utc=True)
    panel["quote_at"] = panel[["quote_at_a", "quote_at_b"]].max(axis=1)
    panel["source_start_at"] = panel[["source_start_at", "scheduled_start_at", "true_start_at",
                                      "native_source_start_at"]].min(axis=1)
    panel["per_source_start_at"] = panel.source_start_at
    bounds = canonical_starts.copy()
    for frame in (bounds, panel):
        frame["golgg_match_id"] = frame.golgg_match_id.astype("string")
    panel = panel.merge(bounds, on="golgg_match_id", how="left", validate="many_to_one")
    panel["canonical_earliest_start_at"] = pd.to_datetime(
        panel.canonical_earliest_start_at.map(_utc), utc=True)
    panel["source_start_at"] = panel[["per_source_start_at", "canonical_earliest_start_at"]].min(axis=1)
    qa, qb, clock, start = panel.quote_at_a, panel.quote_at_b, panel.quote_at, panel.source_start_at
    ages = pd.concat([(clock - qa).dt.total_seconds(), (clock - qb).dt.total_seconds()], axis=1)
    fresh = ages.ge(0).all(axis=1) & ages.le(300).all(axis=1) & qa.notna() & qb.notna()
    expiry = panel.valid_until_exclusive_at
    release = target_releases[["golgg_match_id", "target_native_effective_release_day"]].copy()
    for frame in (release, panel):
        frame["golgg_match_id"] = frame.golgg_match_id.astype("string")
    panel = panel.merge(release, on="golgg_match_id", how="left", validate="many_to_one")
    released_day = pd.to_datetime(panel.target_native_effective_release_day, utc=True, errors="raise")
    clock = panel.quote_at
    already_released = released_day.notna() & released_day.lt(clock.dt.normalize())
    source_status = panel.quote_status.isin(["available_source_scenario", "unknown_active_status"])
    conditions = [
        ~panel.requested_source_year, panel.golgg_match_id.isna(), ~_valid_prices(panel), qa.isna() | qb.isna(),
        start.isna(), ~clock.lt(start), ~fresh, expiry.notna() & clock.ge(expiry),
        already_released, ~source_status,
    ]
    reasons = [
        "source_year_outside_requested_scope", "canonical_identity_unresolved", "paired_prices_missing_or_invalid", "paired_clock_unknown",
        "earliest_start_unknown", "not_before_earliest_known_start", "both_sides_not_fresh_300s",
        "known_quote_expired", "target_already_released_before_decision_day", "source_state_unavailable",
    ]
    panel["timed_price_absence"] = pd.Series(
        np.select([c.to_numpy(dtype=bool, na_value=False) for c in conditions], reasons, default=""),
        index=panel.index).replace("", None)
    panel["timed_price_eligible"] = panel.timed_price_absence.isna()
    panel["archive_eligible"] = panel.timed_price_eligible
    if (available & (panel.native_decision_at.ne(clock) |
                     panel.native_representative_at.gt(clock) |
                     panel.native_representative_at.dt.normalize().ne(clock.dt.normalize()) |
                     panel.model_year.ne(clock.dt.year))).any():
        raise ValueError("native daily projection violates actual quote clock/model-year contract")
    panel = timed_loo_consensus(panel, freshness_seconds=300)
    panel["q"] = panel.q_timed
    panel["q_focal"] = np.nan
    valid = _valid_prices(panel)
    panel.loc[valid, "q_focal"] = ((1 / panel.loc[valid, "odds_a"]) /
        (1 / panel.loc[valid, "odds_a"] + 1 / panel.loc[valid, "odds_b"]))
    panel["lead_hours"] = (panel.source_start_at - panel.quote_at).dt.total_seconds() / 3600
    return panel


def _month_universe(events: pd.DataFrame, years: Sequence[int]) -> list[str]:
    requested = events.loc[events.source_year.isin(years)]
    dates = pd.concat([pd.to_datetime(requested[c], utc=True, errors="coerce")
                       for c in ("source_start_at", "canonical_date")]).dropna().dt.tz_localize(None)
    return list(pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M").astype(str)) if len(dates) else []


def _timed_quality(group: pd.DataFrame, source: str, horizon: str, *, paired: bool) -> list[dict]:
    eligible = group.timed_price_eligible & group.p_c0.notna()
    if paired:
        eligible &= group.q.notna()
    # Establish the first native/context-ready state before looking at outcomes.
    first = group.loc[eligible].sort_values(["quote_at", "bookmaker", "source_event_id"],
                                            kind="stable").drop_duplicates("golgg_match_id")
    scored = first.loc[first.y.isin([0, 1]) & first.label_status.eq("known")]
    rows = []
    for year, cohort in [("ALL", scored), *list(scored.groupby("source_year", sort=True))]:
        market = cohort.q if paired else cohort.q_focal
        for alpha in ([0, 0.5, 1] if paired else [1]):
            probs = np.array([bayesian_logit_shrinkage(float(p), float(q), alpha)
                              for p, q in zip(cohort.p_c0, market)])
            contrast = None
            sensitivity = None
            if len(cohort):
                delta = binary_log_loss_vector(cohort.y, probs, epsilon=0.0) - binary_log_loss_vector(cohort.y, market, epsilon=0.0)
                blocks = pd.to_datetime(cohort.canonical_date).dt.strftime("%Y-%m")
                contrast = block_ratio_interval(delta, np.ones(len(cohort)), blocks, draws=5000, seed=20261022)
                proxies = cohort.competition_context.fillna("UNKNOWN").astype(str) + "|" + cohort.source_year.astype(str)
                sensitivity = block_ratio_interval(delta, np.ones(len(cohort)), proxies, draws=5000, seed=20261022)
            rows.append({"source": source, "horizon": horizon, "year": str(year), "sports_alpha": alpha,
                         "scenario": "prior_roster_source_clock_" + ("two_other_books" if paired else "same_focal_book"),
                         **_scores(cohort.y, probs), "raw_c0_log_loss": _scores(cohort.y, cohort.p_c0)["log_loss"],
                         "market_log_loss": _scores(cohort.y, market)["log_loss"],
                         "market_brier": _scores(cohort.y, market)["brier"],
                         "delta_vs_market": contrast, "competition_context_year_proxy_sensitivity": sensitivity,
                         "proxy_is_certified_tournament_edition": False})
    return rows


def run_timed_evaluation(protocol_path: Path, source_layout_path: Path, native_dir: Path,
                         target_releases_path: Path, timed_rules_path: Path, output_dir: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 2 or protocol.get("run_id") != "long_term_market_edge_20261022_01":
        raise ValueError("unsupported frozen long-term financial protocol")
    rules = json.loads(timed_rules_path.read_text())
    if rules.get("run_id") != protocol["run_id"] or rules.get("schema") != "long_term_timed_evaluation_rules_v1":
        raise ValueError("unsupported frozen timing rules")
    layout = json.loads(source_layout_path.read_text())["sources"]
    receipt_path = native_dir / "receipt.json"
    native_receipt = json.loads(receipt_path.read_text())
    native_binding_path = Path(native_receipt["input_binding_path"])
    if _sha(native_binding_path) != native_receipt["input_binding_sha256"]:
        raise ValueError("native pre-inference input binding changed")
    native_binding = json.loads(native_binding_path.read_text())
    if native_binding["source_layout"]["sha256"] != _sha(source_layout_path):
        raise ValueError("timed native source layout differs from financial source layout")
    if native_binding["protocol"]["sha256"] != _sha(protocol_path):
        raise ValueError("native and financial frozen policy protocol mismatch")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new exclusive output directory: {output_dir}")
    forecast_path = Path(native_receipt["forecast_path"])
    if _sha(forecast_path) != native_receipt["forecast_sha256"]:
        raise ValueError("native forecast bytes changed")
    inputs = {str(path): _sha(path) for path in [protocol_path, source_layout_path, receipt_path,
              native_binding_path, forecast_path, target_releases_path, timed_rules_path, Path(__file__)]}
    for path in [ROOT / "betting_app/core/ev.py", ROOT / "betting_app/core/models/engine.py",
                 ROOT / "src/analysis/betting_guards_a_to_d.py", ROOT / "src/analysis/bootstrap.py",
                 ROOT / "src/analysis/probability_metrics.py"]:
        inputs[str(path)] = _sha(path)
    for source, details in layout.items():
        root = Path(details["root"])
        for name in ("events.parquet", "quotes.parquet"):
            path = root / name
            inputs[str(path)] = _sha(path)
            if inputs[str(path)] != native_binding["source_inputs"][source][name.split(".")[0] + "_sha256"]:
                raise ValueError("native source input changed before financial scoring")
    for item in native_receipt["source_year_counts"].values():
        path = Path(item["query_path"])
        inputs[str(path)] = _sha(path)
        if inputs[str(path)] != item["query_sha256"]:
            raise ValueError("native quote request ledger changed")
    source_tables, query_tables = {}, {}
    for source in protocol["scope"]["sources"]:
        root = Path(layout[source]["root"])
        source_tables[source] = (pd.read_parquet(root / "events.parquet"),
                                 pd.read_parquet(root / "quotes.parquet"))
        query_paths = [Path(item["query_path"]) for key, item in native_receipt["source_year_counts"].items()
                       if key.startswith(source + ":")]
        query_tables[source] = pd.concat([pd.read_parquet(path) for path in query_paths], ignore_index=True)
    starts = canonical_start_bounds(source_tables, query_tables)
    output_dir.mkdir(parents=True)
    starts_path = output_dir / "canonical_start_bounds.parquet"
    starts.to_parquet(starts_path, index=False)
    inputs[str(starts_path)] = _sha(starts_path)
    _write_json(output_dir / "execution_binding.json", {"input_and_code_sha256": inputs,
                "before_timed_financial_scoring": True, "policy_selected": None})
    (output_dir / "long_term_market_evaluation.py.snapshot").write_bytes(Path(__file__).read_bytes())
    forecasts = pd.read_parquet(forecast_path)
    releases = pd.read_parquet(target_releases_path)
    summaries, yearly, paired, focal, selections, missingness = [], [], [], [], [], []
    finance = protocol["fixed_financial_controls"]
    for source in protocol["scope"]["sources"]:
        events, quotes = source_tables[source]
        panel = build_timed_panel(events, quotes, query_tables[source], forecasts, releases, starts,
                                  years=protocol["scope"]["years"])
        panel.to_parquet(output_dir / f"{source}_timed_panel.parquet", index=False)
        months = _month_universe(events, protocol["scope"]["years"])
        leads = panel.loc[panel.timed_price_eligible, "lead_hours"]
        missingness.append({"source": source, "quote_rows": len(panel),
            "requested_quote_rows": int(panel.requested_source_year.sum()),
            "native_query_rows_emitted": int(panel.native_request_emitted.sum()),
            "requested_native_query_rows_not_emitted": int((panel.requested_source_year & ~panel.native_request_emitted).sum()),
            "timed_price_eligible": int(panel.timed_price_eligible.sum()),
            "native_quote_rows": int(panel.p_c0.notna().sum()),
            "native_price_eligible_rows": int((panel.timed_price_eligible & panel.p_c0.notna()).sum()),
            "loo_price_eligible_rows": int((panel.timed_price_eligible & panel.q.notna()).sum()),
            "price_absences": panel.timed_price_absence.fillna("ELIGIBLE").value_counts().to_dict(),
            "native_absences": panel.forecast_absence.fillna("AVAILABLE").value_counts().to_dict(),
            "model_decision_years": panel.loc[panel.p_c0.notna(), "model_year"].value_counts().to_dict(),
            "lead_hours_quantiles": leads.quantile([0, .1, .5, .9, 1]).to_dict()})
        for horizon, group in panel.groupby("horizon", sort=True):
            focal.extend(_timed_quality(group, source, horizon, paired=False))
            paired.extend(_timed_quality(group, source, horizon, paired=True))
            for alpha, tax, epsilon in itertools.product(finance["sports_alpha"], finance["tax_fractions"], finance["dro_probability_haircut"]):
                policy_id = f"a{alpha:g}_tax{tax:g}_dro{epsilon:g}"
                selected = select_fixed_policy(group, sports_alpha=alpha, tax_rate=tax, epsilon=epsilon,
                    minimum_ev=finance["minimum_net_ev"], maximum_odds=finance["maximum_decimal_odds"], timed=True)
                selected["policy_id"] = policy_id
                selected["scenario"] = "prior_roster_source_clock_PRICE_ONLY_NOT_PRODUCTION"
                selections.append(selected)
                summaries.append({"source": source, "horizon": horizon, "policy_id": policy_id,
                    "sports_alpha": alpha, "tax_rate": tax, "epsilon": epsilon,
                    **summarize_unit_returns(selected, months, stake=finance["static_stake"])})
                for year in protocol["scope"]["years"]:
                    subset = selected.loc[pd.to_datetime(selected.canonical_date, errors="coerce").dt.year.eq(year)]
                    yearly.append({"source": source, "horizon": horizon, "policy_id": policy_id, "year": year,
                        **summarize_unit_returns(subset, [m for m in months if m.startswith(str(year))],
                                                 stake=finance["static_stake"])})
    pd.DataFrame(paired).to_parquet(output_dir / "probability_quality.parquet", index=False)
    pd.DataFrame(focal).to_parquet(output_dir / "native_first_probability_quality.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(output_dir / "policy_summaries.parquet", index=False)
    pd.DataFrame(yearly).to_parquet(output_dir / "yearly_policy_summaries.parquet", index=False)
    _combined_ledger(selections).to_parquet(output_dir / "selected_price_diagnostics.parquet", index=False)
    changed = [path for path, sha in inputs.items() if _sha(Path(path)) != sha]
    if changed:
        raise RuntimeError(f"Timed input/code changed during evaluation: {changed}")
    result = {"status": "TIMED_PRICE_ONLY_SCENARIOS_COMPLETE", "source_missingness": missingness,
              "policy_rows": len(summaries), "policy_selected": None, "production_admitted": 0,
              "cash_timeline": "NOT_RUN_SOURCE_ENDPOINTS_CANNOT_ESTABLISH_SETTLEMENT_EXECUTION",
              "authenticated_source": False, "promotion": False, "source_overlap_independent_replication": False}
    _write_json(output_dir / "timed_receipt.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-layout", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--timed-native-dir", type=Path)
    parser.add_argument("--target-releases", type=Path)
    parser.add_argument("--timed-rules", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.timed_native_dir:
        if not args.target_releases or not args.timed_rules:
            parser.error("--timed-native-dir requires --target-releases and --timed-rules")
        result = run_timed_evaluation(args.protocol, args.source_layout, args.timed_native_dir,
                                     args.target_releases, args.timed_rules, args.output_dir)
    else:
        if not args.predictions:
            parser.error("archival diagnostics require --predictions")
        result = run_evaluation(args.protocol, args.source_layout, args.predictions, args.output_dir)
    print(canonical_json_dumps(result))
    return 0
