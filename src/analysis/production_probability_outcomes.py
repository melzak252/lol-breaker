"""Exact event/market/run/clock pairing of exported two-sided outcome observations."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def pair_outcome_quotes(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def rows(name):
        with (raw_root / f"{name}.jsonl").open() as stream:
            for line in stream:
                yield json.loads(line)

    events = {str(r["bookmaker_event_id"]): r for r in rows("bookmaker_events")}
    markets = {(str(r["bookmaker_event_id"]), str(r["bookmaker_market_key"])): r
               for r in rows("bookmaker_markets")}
    fields = ("id", "bookmaker_event_id", "bookmaker_market_key", "scrape_run_id",
              "scraped_at", "outcome_side", "outcome_key", "decimal_odds")
    frame = pd.DataFrame({k: r.get(k) for k in fields} for r in rows("odds_outcome_snapshots"))
    grouped = frame.groupby(["bookmaker_event_id", "bookmaker_market_key", "scrape_run_id", "scraped_at"], dropna=False, sort=False)
    paired, accounting = [], []
    for key, group in grouped:
        event_key, market_key, run, clock = key
        event = events.get(str(event_key))
        market = markets.get((str(event_key), str(market_key)))
        reason = None
        kind = str((market or {}).get("market_type", "")).lower().replace("-", "_")
        names = " ".join(str((market or {}).get(k) or "") for k in ("market_name", "line_name")).casefold()
        if event is None:
            reason = "missing_bookmaker_event"
        elif market is None:
            reason = "missing_exact_market_metadata"
        elif kind not in {"match_winner", "series_winner"}:
            reason = "not_binary_series_winner"
        elif any(word in names for word in ("mapa", "map ", "kills", "handicap", "duration", "total")):
            reason = "non_series_market_name"
        elif len(group) != 2 or Counter(group.outcome_side.astype(str).str.lower()) != Counter({"a": 1, "b": 1}):
            reason = "not_exactly_two_unique_a_b_outcomes"
        else:
            a = group.loc[group.outcome_side.astype(str).str.lower().eq("a")].iloc[0]
            b = group.loc[group.outcome_side.astype(str).str.lower().eq("b")].iloc[0]
            ident = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:24]
            paired.append({
                "id": f"outcome-pair:{ident}", "bookmaker_id": event.get("bookmaker_id"),
                "canonical_match_id": event.get("canonical_match_id"), "match_id": str(event_key),
                "market_type": "match_winner", "raw_team_a": event.get("raw_team_a"),
                "raw_team_b": event.get("raw_team_b"), "odds_a": a.decimal_odds,
                "odds_b": b.decimal_odds, "is_live": None, "scraped_at": clock,
                "observation_origin": "paired_outcome_snapshots",
                "live_status_basis": "unknown_source_live_flag_scheduled_prestart_scenario_only",
            })
        for row_id in group.id:
            accounting.append({"outcome_snapshot_id": row_id, "pairing_status": reason or "paired_exact_two_sided",
                               "paired_snapshot_id": paired[-1]["id"] if reason is None else None})
    result = pd.DataFrame(paired)
    if not result.empty:
        # Carry a direct live flag only from an exactly matching recorded quote state.
        keys = ("canonical_match_id", "bookmaker_id", "scraped_at", "raw_team_a", "raw_team_b", "odds_a", "odds_b")
        recorded = {tuple(str(r.get(k)) for k in keys): r.get("is_live") for r in rows("odds_snapshots")}
        for index, row in result.iterrows():
            flag = recorded.get(tuple(str(row.get(k)) for k in keys))
            if flag is not None:
                result.at[index, "is_live"] = flag
                result.at[index, "live_status_basis"] = "exact_recorded_quote_state"
    return result, pd.DataFrame(accounting)
