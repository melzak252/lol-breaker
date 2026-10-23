"""Offline preparation of retained OddsPapi winner-market evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_ROOT = Path("data/01_raw/oddspapi/oddspapi_lol_multibook_20260928_165639")
PRIMARY_ROOT = Path("data/03_primary/oddspapi/oddspapi_lol_multibook_20260928_165639")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return None
    return timestamp.tz_convert("UTC")


def prepare_oddspapi(protocol_path: str | Path, output_dir: str | Path,
                     aliases_json: str | Path | None = None) -> dict[str, Any]:
    """Write source-preserving events and sparse first/last quote-pair ledgers."""
    protocol_path = Path(protocol_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite preparation output: {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    exports = protocol["source_exports"]
    fixture_path = Path(exports["oddspapi_fixtures"]["path"])
    sparse_path = Path(exports["oddspapi_sparse"]["path"])
    fixture_sha = _sha256(fixture_path)
    sparse_sha = _sha256(sparse_path)
    if fixture_sha != exports["oddspapi_fixtures"]["sha256"]:
        raise ValueError("OddsPapi fixture source hash differs from frozen protocol")
    if sparse_sha != exports["oddspapi_sparse"]["sha256"]:
        raise ValueError("OddsPapi sparse source hash differs from frozen protocol")
    fixtures = pd.read_parquet(fixture_path)
    pairs = pd.read_parquet(sparse_path)
    if len(fixtures) != exports["oddspapi_fixtures"]["rows"] or len(pairs) != exports["oddspapi_sparse"]["rows"]:
        raise ValueError("OddsPapi source row count differs from frozen protocol")
    from src.analysis import long_term_market_sources
    raw_golgg = Path(protocol["canonical_inputs"]["raw_source"])
    if not raw_golgg.exists():
        pointer = Path(protocol["canonical_inputs"]["research_root_pointer"]).read_text(encoding="utf-8").strip()
        candidate = Path(pointer) / protocol["canonical_inputs"]["raw_source"]
        if candidate.exists():
            raw_golgg = candidate
    if not raw_golgg.is_file():
        raise FileNotFoundError(f"canonical identity source not found: {raw_golgg}")
    helpers = Path(long_term_market_sources._MAPPER_PATH)
    alias_path = Path(aliases_json) if aliases_json is not None else None
    binding = {
        "schema": "long_term_market_source_binding_v1",
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256(Path(__file__)),
        "shared_mapper_path": str(Path(long_term_market_sources.__file__).resolve()),
        "shared_mapper_sha256": _sha256(Path(long_term_market_sources.__file__)),
        "identity_helper_path": str(helpers),
        "identity_helper_sha256": _sha256(helpers),
        "aliases_path": str(alias_path.resolve()) if alias_path else None,
        "aliases_sha256": _sha256(alias_path) if alias_path else None,
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_sha,
        "sparse_path": str(sparse_path),
        "sparse_sha256": sparse_sha,
        "canonical_raw_path": str(raw_golgg),
        "canonical_raw_sha256": _sha256(raw_golgg),
    }
    output_dir.mkdir(parents=True)
    binding_path = output_dir / "source_binding.json"
    binding_path.write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    binding_sha = _sha256(binding_path)

    event_columns = {
        "source": "oddspapi", "source_event_id": fixtures["fixtureId"].astype("string"),
        "source_start_at": fixtures["trueStartTime"].where(fixtures["trueStartTime"].notna(), fixtures["startTime"]),
        "source_year": pd.to_datetime(fixtures["startTime"], utc=True, errors="coerce").dt.year,
        "home_name": fixtures["participant1Name"], "away_name": fixtures["participant2Name"],
        "home_participant_id": fixtures["participant1Id"].astype("string"),
        "away_participant_id": fixtures["participant2Id"].astype("string"),
        "scheduled_start_at": fixtures["startTime"], "true_start_at": fixtures["trueStartTime"],
        "true_end_at": fixtures["trueEndTime"], "fixture_updated_at": fixtures["updatedAt"],
        "fixture_status_id": fixtures["statusId"], "fixture_status": fixtures["statusName"],
        "has_odds": fixtures["hasOdds"], "tournament": fixtures["tournamentName"],
        "tournament_id": fixtures["tournamentId"], "season_id": fixtures["seasonId"],
        "sport_id": fixtures["sportId"], "provider_ids": fixtures["externalProviders"],
        "source_path": str(fixture_path), "source_sha256": _sha256(fixture_path),
        "source_row": range(len(fixtures)),
    }
    events = pd.DataFrame(event_columns)
    events["source_start_at"] = events["true_start_at"].where(events["true_start_at"].notna(), events["scheduled_start_at"])
    from src.analysis.long_term_market_sources import map_source_events
    mapped = map_source_events(events, raw_golgg, aliases_json)
    if len(mapped) != len(events):
        raise ValueError("shared identity mapper changed the source event row count")
    events = mapped

    expanded: list[dict[str, Any]] = []
    for row_number, item in pairs.iterrows():
        if str(item["market_id"]) != "181":
            continue
        source_path = str(item["source_path"])
        source_row = row_number
        start_true = _timestamp(item.get("true_start_at"))
        start_scheduled = _timestamp(item.get("scheduled_start_at"))
        for marker, horizon in (("first_observed", "OPEN_OR_FIRST_OBSERVED"), ("last_prestart", "LAST_CHANGE_OR_LAST_PRESTART")):
            if not bool(item.get(marker, False)):
                continue
            at1, at2 = _timestamp(item.get("outcome_1_quote_at")), _timestamp(item.get("outcome_2_quote_at"))
            o1, o2 = item.get("odds_1"), item.get("odds_2")
            identity = events.loc[events["source_event_id"].astype(str).eq(str(item["fixture_id"]))]
            identity_row = identity.iloc[0] if len(identity) else None
            swapped = bool(identity_row["orientation_swapped"]) if identity_row is not None and not pd.isna(identity_row["orientation_swapped"]) else None
            mapped_identity = identity_row is not None and identity_row["golgg_match_id"] is not None and swapped is not None
            odds_a, odds_b = ((o2, o1) if swapped else (o1, o2)) if mapped_identity else (None, None)
            quote_status = "available_source_scenario"
            if pd.isna(o1) or pd.isna(o2) or float(o1) <= 1 or float(o2) <= 1:
                quote_status = "invalid_price"
            elif at1 is None or at2 is None:
                quote_status = "missing_clock"
            elif start_true is None and start_scheduled is None:
                quote_status = "unknown_start"
            elif (start_true is not None and max(at1, at2) >= start_true) or (start_true is None and start_scheduled is not None and max(at1, at2) >= start_scheduled):
                quote_status = "after_start"
            expanded.append({
                "source": "oddspapi", "source_event_id": str(item["fixture_id"]), "bookmaker": item["bookmaker"],
                "horizon": horizon, "odds_a": odds_a, "odds_b": odds_b,
                "quote_at_a": at2.isoformat() if swapped and at2 is not None else (at1.isoformat() if at1 is not None else None),
                "quote_at_b": at1.isoformat() if swapped and at1 is not None else (at2.isoformat() if at2 is not None else None),
                "quote_at": max(at1, at2).isoformat() if at1 is not None and at2 is not None else None,
                "quote_status": quote_status, "active_state_proven": True,
                "active_state_proof": "frozen sparse derivative selected active winner-market pair state",
                "source_path": source_path, "source_sha256": item["source_sha256"],
                "source_row": source_row, "source_orientation_swapped": swapped,
                "odds_1": o1, "odds_2": o2, "outcome_1_quote_at": item["outcome_1_quote_at"],
                "outcome_2_quote_at": item["outcome_2_quote_at"], "valid_until_exclusive_at": item["valid_until_exclusive_at"],
                "valid_until_exclusive_at_present": not pd.isna(item["valid_until_exclusive_at"]),
                "scheduled_start_at": item["scheduled_start_at"], "true_start_at": item["true_start_at"],
                "cutoff_at": item["cutoff_at"], "cutoff_basis": item["cutoff_basis"],
                "fixture_status_id": item["fixture_status_id"], "market_id": item["market_id"],
            })
    quotes = pd.DataFrame(expanded)
    events.to_parquet(output_dir / "events.parquet", index=False)
    quotes.to_parquet(output_dir / "quotes.parquet", index=False)
    years = sorted(int(year) for year in events["source_year"].dropna().unique())
    quote_years = pd.to_datetime(quotes["scheduled_start_at"], utc=True, errors="coerce").dt.year
    receipt = {
        "source": "oddspapi", "protocol_path": str(protocol_path), "output_dir": str(output_dir),
        "fixture_count": len(events), "sparse_pair_count": len(pairs), "quote_ledger_count": len(quotes),
        "source_year_counts": {str(year): int((events["source_year"] == year).sum()) for year in years},
        "quote_year_counts": {str(year): int((quote_years == year).sum()) for year in sorted(int(y) for y in quote_years.dropna().unique())},
        "source_date_span": {
            "first_scheduled_start_at": events["scheduled_start_at"].dropna().min() if events["scheduled_start_at"].notna().any() else None,
            "last_scheduled_start_at": events["scheduled_start_at"].dropna().max() if events["scheduled_start_at"].notna().any() else None,
            "first_true_start_at": events["true_start_at"].dropna().min() if events["true_start_at"].notna().any() else None,
            "last_true_start_at": events["true_start_at"].dropna().max() if events["true_start_at"].notna().any() else None,
        },
        "quote_missingness": {column: int(quotes[column].isna().sum()) for column in ("odds_a", "odds_b", "quote_at_a", "quote_at_b", "quote_at", "valid_until_exclusive_at")},
        "active_state_parsed_in_retained_source": True,
        "active_status_limitation": "parsed pair-level state only; no public issuance/authentic execution or activity beyond valid_until_exclusive_at established",
        "fixture_missingness": {column: int(events[column].isna().sum()) for column in ("source_start_at", "true_start_at", "scheduled_start_at", "fixture_status_id", "home_participant_id", "away_participant_id")},
        "quote_status_counts": {str(key): int(value) for key, value in quotes["quote_status"].value_counts(dropna=False).items()},
        "source_hashes": {str(fixture_path): fixture_sha, str(sparse_path): sparse_sha, str(protocol_path): _sha256(protocol_path), str(raw_golgg): _sha256(raw_golgg)},
        "binding_path": str(binding_path), "binding_sha256": binding_sha,
        "implementation_hashes": {str(Path(__file__).resolve()): _sha256(Path(__file__)), str(Path(long_term_market_sources.__file__).resolve()): _sha256(Path(long_term_market_sources.__file__)), str(helpers): _sha256(helpers)},
        "year_support": {str(year): bool((events["source_year"] == year).any()) for year in protocol["scope"]["years"]},
        "missing_years_not_synthesized": [year for year in protocol["scope"]["years"] if not (events["source_year"] == year).any()],
        "ledger_files": ["events.parquet", "quotes.parquet"],
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, default=_json_value) + "\n", encoding="utf-8")
    return receipt
