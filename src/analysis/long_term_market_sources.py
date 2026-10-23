"""Outcome-blind OddsPortal and shared source-event identity preparation."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_MAPPER_PATH = ROOT / "scripts/03_dane_pipeline/00_map_matches.py"
_spec = importlib.util.spec_from_file_location("_identity_only_v5", _MAPPER_PATH)
_identity = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _identity
_spec.loader.exec_module(_identity)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _datetime(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        # Date-only source rows have no certified clock; use UTC midnight strictly for identity day.
        return stamp.date()
    return stamp.tz_convert("UTC").to_pydatetime()


def map_source_events(events: pd.DataFrame, golgg_json: str | Path,
                      aliases_json: str | Path | None = None) -> pd.DataFrame:
    """Map normalized source events, retaining graph edges before any label processing."""
    required = {"source", "source_event_id", "source_start_at", "home_name", "away_name"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Normalized source events missing columns: {sorted(missing)}")
    aliases, scoped = ({}, {}) if aliases_json is None else _identity.load_aliases(aliases_json)
    by_identity = defaultdict(list)
    games_by_id = {}
    with Path(golgg_json).open("rb") as stream:
        for raw in _identity._golgg_records(stream):
            game = _identity._canonical_game(raw)
            game["best_of"] = _identity.golgg_schema.best_of(raw)
            mid = str(game["match_id"])
            if mid in games_by_id:
                raise ValueError(f"Duplicate GOL.GG match id: {mid}")
            games_by_id[mid] = game
            names = tuple(_identity.apply_alias(_identity.normalize_team_name(game[k]), aliases)
                          for k in ("name_1", "name_2"))
            if all(names) and names[0] != names[1]:
                by_identity[(date.fromisoformat(game["date"]), tuple(sorted(names)))].append((game, names))
    out = events.copy()
    out["source_competition_context"] = out.get("competition_context", out.get("tournament"))
    fields = {"golgg_match_id": [], "orientation_swapped": [], "date_offset_days": [],
              "identity_status": [], "candidate_ids": [], "canonical_date": [], "best_of": [],
              "competition_context": [], "canonical_team_a_id": [], "canonical_team_b_id": []}
    candidates_by_row = []
    claims = []
    for row in out.to_dict("records"):
        ts = _datetime(row.get("source_start_at"))
        day = ts.date() if isinstance(ts, datetime) else (ts or _datetime(row.get("source_date")))
        if isinstance(day, datetime):
            day = day.date()
        names = (row.get("home_name"), row.get("away_name"))
        ids = (row.get("home_participant_id"), row.get("away_participant_id"))
        tournament = row.get("competition_context") or row.get("tournament") or ""
        variants = tuple(_identity._source_name_variants(n, _identity._participant_id(i), day, tournament, aliases, scoped)
                         if day is not None else {} for n, i in zip(names, ids))
        cands = []
        if day is not None and all(variants):
            cands = _identity._identity_candidates(by_identity, day, variants)
            if not cands:
                cands = [c for offset in (-1, 1) for c in _identity._identity_candidates(
                    by_identity, day + timedelta(days=offset), variants, offset)]
        ids_found = sorted({str(c[0]["match_id"]) for c in cands})
        orientations = set().union(*(c[3] for c in cands)) if cands else set()
        status = ("invalid_date_or_identity" if day is None or not all(variants) else
                  "ambiguous_golgg_fixture" if len(cands) > 1 else
                  "ambiguous_source_orientation" if cands and len(orientations) != 1 else
                  "candidate_unique" if cands else "unmatched")
        candidates_by_row.append(cands)
        claims.extend(ids_found)
        fields["candidate_ids"].append(json.dumps(ids_found))
        fields["identity_status"].append(status)
    claim_counts = Counter(claims)
    for index, (row, cands, status) in enumerate(zip(out.to_dict("records"), candidates_by_row, fields["identity_status"])):
        game = None; swapped = None; offset = None
        if status == "candidate_unique":
            candidate, canonical, offset, orientations = cands[0]
            mid = str(candidate["match_id"])
            if claim_counts[mid] > 1:
                status = "ambiguous_competing_source_events"
            else:
                game, swapped = candidate, next(iter(orientations))
        fields["golgg_match_id"].append(str(game["match_id"]) if game else None)
        fields["orientation_swapped"].append(bool(swapped) if game else None)
        fields["date_offset_days"].append(offset if game else None)
        fields["identity_status"][index] = status
        fields["canonical_date"].append(game["date"] if game else None)
        fields["best_of"].append(game["best_of"] if game else None)
        fields["competition_context"].append(game["tournament"] if game else row.get("competition_context") or row.get("tournament"))
        fields["canonical_team_a_id"].append(str(game["tid_1"]) if game and game["tid_1"] is not None else None)
        fields["canonical_team_b_id"].append(str(game["tid_2"]) if game and game["tid_2"] is not None else None)
    for key, values in fields.items():
        out[key] = values
    # Reconcile source results only after the full identity graph and competing claims are fixed.
    label_status, source_result_status, ys, canonical_ys = [], [], [], []
    canonical_scores_a, canonical_scores_b = [], []
    for row in out.to_dict("records"):
        canonical = games_by_id.get(row["golgg_match_id"])
        canonical_y = int(canonical["t1_win"]) if canonical and not canonical["draw"] else None
        canonical_ys.append(canonical_y)
        canonical_scores_a.append(canonical["score_1"] if canonical else None)
        canonical_scores_b.append(canonical["score_2"] if canonical else None)
        home_result, away_result = row.get("homeResult"), row.get("awayResult")
        provider_missing = home_result is None or away_result is None or pd.isna(home_result) or pd.isna(away_result)
        if canonical_y is None:
            label_status.append("unknown")
            source_result_status.append("missing" if provider_missing else "unreconciled")
            ys.append(None)
            continue
        if provider_missing:
            # Canonical GOL.GG is the declared outcome authority. A resultless
            # provider inventory does not erase its independently mapped label.
            label_status.append("known")
            source_result_status.append("missing")
            ys.append(canonical_y)
            continue
        try:
            home_result = _identity._source_score(home_result)
            away_result = _identity._source_score(away_result)
        except ValueError:
            label_status.append("unsupported")
            source_result_status.append("unsupported")
            ys.append(None)
            continue
        source_a, source_b = ((away_result, home_result) if row["orientation_swapped"]
                              else (home_result, away_result))
        agrees = (source_a, source_b) == (canonical["score_1"], canonical["score_2"])
        label_status.append("known" if agrees else "conflict")
        source_result_status.append("agreement" if agrees else "conflict")
        ys.append(canonical_y if agrees else None)
    out["label_status"], out["source_result_status"], out["y"] = label_status, source_result_status, ys
    out["canonical_y"] = canonical_ys
    out["canonical_score_a"], out["canonical_score_b"] = canonical_scores_a, canonical_scores_b
    return out


def prepare_oddsportal(protocol_path: str | Path, output_dir: str | Path,
                       aliases_json: str | Path | None = None) -> dict:
    protocol_path, output = Path(protocol_path), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Exclusive output path already exists: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_spec = protocol["source_exports"]["oddsportal_full"]
    source = (ROOT / source_spec["path"]).resolve()
    if not source.is_file() or _sha(source) != source_spec["sha256"]:
        raise ValueError("Pinned OddsPortal full export missing or hash mismatch")
    raw_path = (ROOT / protocol["canonical_inputs"]["raw_source"]).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Canonical Golgg source missing: {raw_path}")
    output.mkdir(parents=True)
    code_path = Path(__file__).resolve()
    mapper_hash = _sha(_MAPPER_PATH)
    aliases_path = Path(aliases_json).resolve() if aliases_json else None
    binding = {"schema":"long_term_market_source_binding_v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_path":str(protocol_path.resolve()),"protocol_sha256":_sha(protocol_path),
        "implementation_path":str(code_path),"implementation_sha256":_sha(code_path),
        "identity_mapper_path":str(_MAPPER_PATH),"identity_mapper_sha256":mapper_hash,
        "source_path":str(source),"source_sha256":source_spec["sha256"],
        "canonical_golgg_path":str(raw_path),"canonical_golgg_sha256":_sha(raw_path),
        "aliases_path":str(aliases_path) if aliases_path else None,
        "aliases_sha256":_sha(aliases_path) if aliases_path else None}
    binding_path = output/"source_binding.json"
    binding_path.write_text(json.dumps(binding,indent=2)+"\n",encoding="utf-8")
    binding_sha = _sha(binding_path)
    frame = pd.read_csv(source, low_memory=False)
    if len(frame) != source_spec["rows"]:
        raise ValueError("OddsPortal export row count does not match frozen protocol")
    event_rows = []
    for ix, r in frame.iterrows():
        dt = _datetime(r.get("date"))
        event_rows.append({"source":"oddsportal", "source_event_id":str(r.get("event_id") or r.get("url") or ix),
            "source_start_at":dt.isoformat() if isinstance(dt, datetime) else None,
            "source_date":dt.isoformat() if dt is not None else None,
            "source_year":dt.year if isinstance(dt, datetime) else (dt.year if dt else None),
            "home_name":r.get("home_team"), "away_name":r.get("away_team"),
            "home_participant_id":r.get("source_home_participant_id"), "away_participant_id":r.get("source_away_participant_id"),
            "homeResult":r.get("homeResult"), "awayResult":r.get("awayResult"),
            "competition_context":r.get("tournament"), "source_row":int(ix)})
    events = map_source_events(pd.DataFrame(event_rows), raw_path, aliases_json)
    quotes=[]
    for ix, r in frame.iterrows():
        for c in frame.columns:
            match = _identity.QUOTE_COLUMN.fullmatch(str(c))
            if not match or match[3]: continue
            side, book, period = match[0], match[1], match[2]
        # Build one quote for every discovered bookmaker and observed horizon.
        for book in sorted({m[1] for c in frame.columns if (m:=_identity.QUOTE_COLUMN.fullmatch(str(c))) and not m[3]}):
            for period, horizon in (("open", "OPEN_OR_FIRST_OBSERVED"), ("close", "LAST_CHANGE_OR_LAST_PRESTART")):
                one, two = r.get(f"odds1_{book}_{period}"), r.get(f"odds2_{book}_{period}")
                clock1, clock2 = r.get(f"odds1_{book}_{period}_at"), r.get(f"odds2_{book}_{period}_at")
                if pd.isna(one) and pd.isna(two): continue
                mapped=events.iloc[ix]
                try: a,b=float(one),float(two)
                except (TypeError,ValueError): a=b=float("nan")
                valid=math.isfinite(a) and math.isfinite(b) and a>1 and b>1
                swapped=mapped.orientation_swapped
                if valid and swapped: a,b=b,a
                c1,c2=(_datetime(clock1),_datetime(clock2))
                status="unknown_active_status" if valid else "invalid_price"
                if valid and (c1 is None or c2 is None): status="missing_clock"
                quotes.append({"source":"oddsportal","source_event_id":events.iloc[ix].source_event_id,"bookmaker":book,
                    "horizon":horizon,"odds_a":a if valid and swapped is not None else (a if valid and mapped.golgg_match_id else None),
                    "odds_b":b if valid and swapped is not None else (b if valid and mapped.golgg_match_id else None),
                    "source_odds1":one,"source_odds2":two,"source_clock1":clock1,"source_clock2":clock2,
                    "quote_at_a":c2.isoformat() if swapped and c2 else (c1.isoformat() if c1 else None),
                    "quote_at_b":c1.isoformat() if swapped and c1 else (c2.isoformat() if c2 else None),
                    "quote_at":max(c1,c2).isoformat() if c1 and c2 else None,"quote_status":status,
                    "source_path":str(source),"source_sha256":source_spec["sha256"],"source_row":int(ix)})
    quote_frame = pd.DataFrame(quotes)
    if not quote_frame.empty:
        quote_frame["source_year"] = quote_frame.source_event_id.map(
            events.set_index("source_event_id").source_year.to_dict())
        quote_year = quote_frame.groupby("source_year", dropna=False).agg(
            rows=("source_event_id","size"), paired_clocks=("quote_at","count"),
            available_prices=("odds_a","count")).reset_index().to_dict("records")
    else:
        quote_year = []
    events.to_parquet(output/"events.parquet",index=False)
    quote_frame.to_parquet(output/"quotes.parquet",index=False)
    receipt={"schema":"long_term_market_source_receipt_v1","source":"oddsportal","input_path":str(source),
      "input_sha256":source_spec["sha256"],"source_rows":len(frame),"event_rows":len(events),"quote_rows":len(quotes),
      "source_binding_path":str(binding_path),"source_binding_sha256":binding_sha,
      "protocol_path":str(protocol_path.resolve()),"protocol_sha256":binding["protocol_sha256"],
      "golgg_path":str(raw_path),"golgg_sha256":binding["canonical_golgg_sha256"],"implementation_path":str(code_path),
      "implementation_sha256":binding["implementation_sha256"],"identity_mapper_path":str(_MAPPER_PATH),
      "identity_mapper_sha256":mapper_hash,"aliases_path":str(aliases_path) if aliases_path else None,
      "aliases_sha256":binding["aliases_sha256"],
      "identity_status_counts":events.identity_status.value_counts(dropna=False).to_dict(),
      "label_status_counts":events.label_status.value_counts(dropna=False).to_dict(),
      "per_year":events.groupby("source_year",dropna=False).size().to_dict(),
      "quote_per_year":quote_year,
      "quote_horizon_counts":quote_frame.horizon.value_counts(dropna=False).to_dict() if not quote_frame.empty else {},
      "quote_status_counts":quote_frame.quote_status.value_counts(dropna=False).to_dict() if not quote_frame.empty else {}}
    (output/"source_receipt.json").write_text(json.dumps(receipt,indent=2,default=str)+"\n",encoding="utf-8")
    return receipt
