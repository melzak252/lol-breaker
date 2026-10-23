"""Native mean-probability EV consumers over isolated PostgreSQL rows and quotes."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from betting_app.core.db import get_session
from betting_app.core.models import C0_NATIVE
from betting_app.models.match import CanonicalMatch
from betting_app.models.odds import OddsSnapshot
from betting_app.models.prediction import CanonicalPrediction, ModelEvSignal
from betting_app.services import upcoming_inference_service as service


@pytest.fixture
def signal_clock(client, monkeypatch):
    clock = {"now": datetime(2026, 10, 2, 12, tzinfo=UTC)}

    class SignalDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = clock["now"]
            return current.astimezone(tz) if tz is not None else current.replace(tzinfo=None)

    monkeypatch.setattr(service, "datetime", SignalDatetime)
    return clock


def _seed_signal_inputs(clock):
    now = clock["now"]
    with get_session() as session:
        match = CanonicalMatch(
            canonical_key="native-mean-signals",
            team_a_name="Alpha", team_b_name="Beta",
            normalized_team_a="alpha", normalized_team_b="beta",
            start_time_normalized=(now + timedelta(hours=1)).isoformat(),
            best_of=3, league="Unmapped invitational", status="upcoming",
        )
        session.add(match)
        session.flush()
        predictions = [
            CanonicalPrediction(
                canonical_match_id=match.id,
                model_name=C0_NATIVE.name, model_version=C0_NATIVE.version,
                features_version=C0_NATIVE.feature_version,
                ratings_version=C0_NATIVE.ratings_version,
                predicted_at=(now - timedelta(minutes=age)).isoformat(),
                data_cutoff_at=(now - timedelta(days=1)).isoformat(),
                prob_a=prob_a, prob_b=1.0 - prob_a,
                prediction_status="active", diagnostics_json="{}",
            )
            for age, prob_a in ((3, 0.3), (1, 0.52))
        ]
        quotes = [
            OddsSnapshot(
                canonical_match_id=match.id, bookmaker_id=1,
                market_type="match_winner", raw_team_a="Alpha", raw_team_b="Beta",
                odds_a=odds, odds_b=odds, is_live=0,
                scraped_at=now + offset,
            )
            for offset, odds in (
                (timedelta(minutes=-2), 2.4),
                (timedelta(seconds=-30), 2.6),
                (timedelta(seconds=30), 6.0),
            )
        ]
        peer_quotes = [
            OddsSnapshot(
                canonical_match_id=match.id, bookmaker_id=bookmaker_id,
                market_type="match_winner", raw_team_a="Alpha", raw_team_b="Beta",
                odds_a=2.6, odds_b=2.6, is_live=0,
                scraped_at=now - timedelta(seconds=30),
            )
            for bookmaker_id in (2, 3)
        ]
        session.add_all([*predictions, *quotes, *peer_quotes])
        session.flush()
        ids = {
            "match": match.id,
            "prediction": predictions[-1].id,
            "quote": quotes[1].id,
            "fresh_quotes": [quotes[1].id, *(quote.id for quote in peer_quotes)],
        }
        session.commit()
    return ids


def test_c0_ev_uses_focal_leave_one_out_consensus_and_records_gates(signal_clock):
    ids = _seed_signal_inputs(signal_clock)

    signals = service.generate_model_ev_signals(bankroll=100)

    assert {signal["side"] for signal in signals} == {"a", "b"}
    assert len(signals) == 6
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, ids["prediction"])
        saved = session.scalars(select(ModelEvSignal).where(ModelEvSignal.status == "new")).all()
        assert len(saved) == 6
        for signal in signals:
            side = signal["side"]
            quote = session.get(OddsSnapshot, signal["odds_snapshot_id"])
            mean = getattr(prediction, f"prob_{side}")
            odds = getattr(quote, f"odds_{side}")
            assert signal["canonical_match_id"] == ids["match"]
            assert signal["quote_scraped_at"] == quote.scraped_at.isoformat()
            assert signal["evaluated_at"] == signal_clock["now"].isoformat()
            assert signal["model_prob"] == mean
            assert signal["prob_for_ev"] == mean
            assert signal["prob_conservative"] is None
            assert signal["uncertainty_basis"] == "mean"
            assert signal["ev"] == pytest.approx(mean * odds * 0.88 - 1)
            persisted = next(
                row for row in saved
                if row.side == side and row.odds_snapshot_id == quote.id
            )
            assert persisted.canonical_prediction_id == prediction.id
            assert persisted.model_prob == pytest.approx(mean)
            assert persisted.ev == pytest.approx(signal["ev"])
            context = json.loads(persisted.market_context_json)
            assert context["decision_at"] == signal_clock["now"].isoformat()
            assert [source["odds_snapshot_id"] for source in context["sources"]] == sorted(
                ids["fresh_quotes"],
                key=lambda quote_id: session.get(OddsSnapshot, quote_id).bookmaker_id,
            )
            focal = context["focal_consensus"]
            assert focal["excluded_bookmaker_id"] == quote.bookmaker_id
            assert quote.bookmaker_id not in focal["bookmaker_ids"]
            assert len(focal["bookmaker_ids"]) == 2
            expected_market = focal["prob_a"] if side == "a" else 1.0 - focal["prob_a"]
            assert persisted.market_prob == pytest.approx(expected_market)
            assert context["gates"]["min_ev_net"]["status"] == "pass"
            assert context["gates"]["max_ev_net"]["status"] == "pass"
            assert context["gates"]["exp039_veto"]["status"] == "not_evaluated"
            assert context["gates"]["exp039_veto"]["reason"] == "exp039_prediction_missing"
            assert context["gate_config"]["min_ev_net"] == 0.05
            assert context["gate_config"]["max_ev_net"] == 0.25
        assert json.loads(prediction.diagnostics_json) == {}




def test_swapping_c0_probability_sides_swaps_ev_sides(signal_clock):
    ids = _seed_signal_inputs(signal_clock)
    forward = {signal["side"]: signal for signal in service.generate_model_ev_signals()}
    assert set(forward) == {"a", "b"}
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, ids["prediction"])
        prediction.prob_a, prediction.prob_b = prediction.prob_b, prediction.prob_a
        session.commit()

    reversed_signals = {
        signal["side"]: signal for signal in service.generate_model_ev_signals()
    }

    assert set(reversed_signals) == {"a", "b"}
    for side, opposite in (("a", "b"), ("b", "a")):
        assert reversed_signals[side]["model_prob"] == forward[opposite]["model_prob"]
        assert reversed_signals[side]["ev"] == pytest.approx(forward[opposite]["ev"])
        assert reversed_signals[side]["prob_conservative"] is None
    with get_session() as session:
        saved = session.scalars(select(ModelEvSignal)).all()
        assert sum(row.status == "stale" for row in saved) == 6
        assert sum(row.status == "new" for row in saved) == 6


def test_one_book_is_withheld_with_explicit_consensus_reason(signal_clock):
    from sqlalchemy import delete

    ids = _seed_signal_inputs(signal_clock)
    with get_session() as session:
        session.execute(
            delete(OddsSnapshot).where(
                OddsSnapshot.canonical_match_id == ids["match"],
                OddsSnapshot.bookmaker_id.in_((2, 3)),
            )
        )
        session.commit()

    signals, stats = service.generate_model_ev_signals(return_stats=True)

    assert signals == []
    assert stats["withheld_markets"] == [{
        "canonical_match_id": ids["match"],
        "reason": "market_consensus_insufficient_books",
        "book_count": 1,
    }]



@pytest.mark.parametrize(
    "exp039_probability,expect_signal",
    [(0.71, True), (0.81, False)],
)
def test_hybrid_exp039_veto_compares_sports_probability_to_fresh_thesis(
    signal_clock, monkeypatch, exp039_probability, expect_signal
):
    from betting_app.core.models import PredictionEngine, get_active_hybrid

    ids = _seed_signal_inputs(signal_clock)
    now = signal_clock["now"]
    hybrid = get_active_hybrid()
    hybrid_probability = PredictionEngine.blend_with_market(0.70, 0.5, hybrid)
    with get_session() as session:
        for quote in session.scalars(
            select(OddsSnapshot).where(OddsSnapshot.canonical_match_id == ids["match"])
        ):
            quote.odds_a = 2.2
            quote.odds_b = 2.2
        session.add_all([
            CanonicalPrediction(
                canonical_match_id=ids["match"],
                model_name=hybrid.hybrid_model_name,
                model_version=hybrid.hybrid_model_version,
                predicted_at=(now - timedelta(minutes=1)).isoformat(),
                data_cutoff_at=(now - timedelta(days=1)).isoformat(),
                prob_a=hybrid_probability,
                prob_b=1.0 - hybrid_probability,
                prediction_status="active",
                diagnostics_json=json.dumps({
                    "base_model_name": C0_NATIVE.name,
                    "base_model_version": C0_NATIVE.version,
                    "base_prob_a": 0.70,
                    "market_features_used": True,
                    "alpha": hybrid.alpha,
                    "temperature": hybrid.temperature,
                    "blending_mode": hybrid.blending_mode,
                    "market_context": {
                        "model_name": hybrid.hybrid_model_name,
                        "model_version": hybrid.hybrid_model_version,
                        "decision_at": (now - timedelta(seconds=20)).isoformat(),
                    },
                }),
            ),
            CanonicalPrediction(
                canonical_match_id=ids["match"],
                model_name="Sym-Cal LR-ElasticNet-W20-Binomial",
                model_version="exp-039",
                predicted_at=(now - timedelta(minutes=2)).isoformat(),
                data_cutoff_at=(now - timedelta(days=1)).isoformat(),
                prob_a=exp039_probability,
                prob_b=1.0 - exp039_probability,
                prediction_status="active",
                diagnostics_json="{}",
            ),
        ])
        session.commit()
    monkeypatch.setattr(service, "hybrid_prediction_contract_is_current", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(service, "stored_market_context_is_current", lambda *args, **kwargs: (True, None))

    signals = service.generate_model_ev_signals(
        model_name=hybrid.hybrid_model_name,
        model_version=hybrid.hybrid_model_version,
        bankroll=100,
    )

    if not expect_signal:
        assert signals == []
        stats = service.generate_model_ev_signals.last_stats
        assert stats["exp039_veto"]["fail"] > 0
        assert stats["disqualified_breakdown"]["exp039_disagreement"] > 0
        # Veto failures count both sides; the breakdown only counts reached opportunities.
        with get_session() as session:
            assert session.scalars(select(ModelEvSignal).where(ModelEvSignal.status == "new")).all() == []
        return
    assert {signal["side"] for signal in signals} == {"a"}
    with get_session() as session:
        saved = session.scalars(select(ModelEvSignal).where(ModelEvSignal.status == "new")).all()
        assert saved
        for row in saved:
            context = json.loads(row.market_context_json)
            veto = context["gates"]["exp039_veto"]
            assert veto["status"] == "pass"
            assert veto["base_prob_a"] == pytest.approx(0.70)
            assert veto["exp039_prob_a"] == pytest.approx(exp039_probability)
            assert abs(veto["base_prob_a"] - veto["exp039_prob_a"]) <= veto["max_difference"]
            assert row.model_prob == pytest.approx(hybrid_probability)

def test_stale_latest_exp039_is_not_replaced_by_an_older_active_row(signal_clock):
    ids = _seed_signal_inputs(signal_clock)
    now = signal_clock["now"]
    with get_session() as session:
        session.add_all([
            CanonicalPrediction(
                canonical_match_id=ids["match"],
                model_name="Sym-Cal LR-ElasticNet-W20-Binomial",
                model_version="exp-039",
                predicted_at=(now - timedelta(minutes=5)).isoformat(),
                data_cutoff_at=(now - timedelta(days=1)).isoformat(),
                prob_a=0.52,
                prob_b=0.48,
                prediction_status="active",
                diagnostics_json="{}",
            ),
            CanonicalPrediction(
                canonical_match_id=ids["match"],
                model_name="Sym-Cal LR-ElasticNet-W20-Binomial",
                model_version="exp-039",
                predicted_at=(now - timedelta(minutes=1)).isoformat(),
                data_cutoff_at=(now - timedelta(days=1)).isoformat(),
                prob_a=0.99,
                prob_b=0.01,
                prediction_status="stale",
                diagnostics_json="{}",
            ),
        ])
        session.commit()

    signals = service.generate_model_ev_signals(bankroll=100)

    assert signals
    with get_session() as session:
        saved = session.scalars(select(ModelEvSignal).where(ModelEvSignal.status == "new")).first()
        context = json.loads(saved.market_context_json)
        assert context["gates"]["exp039_veto"]["status"] == "not_evaluated"
        assert context["gates"]["exp039_veto"]["reason"] == "prediction_inactive"

@pytest.mark.parametrize("clock_error", ["unknown_cutoff", "future_prediction", "cutoff_after_prediction"])
def test_c0_signals_refuse_predictions_without_actionable_clocks(signal_clock, clock_error):
    ids = _seed_signal_inputs(signal_clock)
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, ids["prediction"])
        if clock_error == "unknown_cutoff":
            prediction.data_cutoff_at = None
        elif clock_error == "future_prediction":
            prediction.predicted_at = (signal_clock["now"] + timedelta(seconds=1)).isoformat()
        else:
            prediction.data_cutoff_at = signal_clock["now"].isoformat()
        session.commit()

    assert service.generate_model_ev_signals() == []
    with get_session() as session:
        assert session.scalars(select(ModelEvSignal)).all() == []
