import json

from src.analysis.production_probability_outcomes import pair_outcome_quotes


def _write(root, name, records):
    (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))


def test_outcomes_pair_only_within_exact_event_market_run_and_clock(tmp_path):
    event = {"bookmaker_event_id": "external", "bookmaker_id": 3, "canonical_match_id": 7,
             "raw_team_a": "Alpha", "raw_team_b": "Beta"}
    _write(tmp_path, "bookmaker_events", [event])
    _write(tmp_path, "bookmaker_markets", [{"bookmaker_event_id": "external", "bookmaker_market_key": "main", "market_type": "match_winner"}])
    clock = "2026-10-01T00:00:00+00:00"
    observations = []
    for ident, run, side in [(1, 1, "a"), (2, 1, "b"), (3, 2, "a"), (4, 3, "b")]:
        observations.append({"id": ident, "bookmaker_event_id": "external", "bookmaker_market_key": "main",
                             "scrape_run_id": run, "scraped_at": clock, "outcome_side": side, "decimal_odds": 2.0})
    _write(tmp_path, "odds_outcome_snapshots", observations)
    _write(tmp_path, "odds_snapshots", [{**event, "scraped_at": clock, "odds_a": 2.0, "odds_b": 2.0, "is_live": False}])
    pairs, accounting = pair_outcome_quotes(tmp_path)
    assert len(pairs) == 1
    assert pairs.iloc[0].odds_a == pairs.iloc[0].odds_b == 2.0
    assert pairs.iloc[0].is_live == False
    assert accounting.set_index("outcome_snapshot_id").pairing_status.to_dict() == {
        1: "paired_exact_two_sided", 2: "paired_exact_two_sided",
        3: "not_exactly_two_unique_a_b_outcomes", 4: "not_exactly_two_unique_a_b_outcomes"}


def test_conflicting_same_side_prices_do_not_choose_a_favorable_pair(tmp_path):
    _write(tmp_path, "bookmaker_events", [{"bookmaker_event_id": "e", "bookmaker_id": 1, "canonical_match_id": 1}])
    _write(tmp_path, "bookmaker_markets", [{"bookmaker_event_id": "e", "bookmaker_market_key": "m", "market_type": "match_winner"}])
    _write(tmp_path, "odds_outcome_snapshots", [
        {"id": i, "bookmaker_event_id": "e", "bookmaker_market_key": "m", "scrape_run_id": 1,
         "scraped_at": "2026-10-01T00:00:00Z", "outcome_side": side, "decimal_odds": price}
        for i, side, price in [(1, "a", 2), (2, "a", 3), (3, "b", 2)]])
    _write(tmp_path, "odds_snapshots", [])
    pairs, accounting = pair_outcome_quotes(tmp_path)
    assert pairs.empty
    assert accounting.pairing_status.tolist() == ["not_exactly_two_unique_a_b_outcomes"] * 3
