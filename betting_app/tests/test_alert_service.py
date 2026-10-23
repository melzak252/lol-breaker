from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_alert_requalifies_hybrid_with_current_focal_quote(monkeypatch):
    from betting_app.core.models import C0_NATIVE, get_active_hybrid
    from betting_app.core.models.engine import blend_market_probability
    from betting_app.services import alert_service as alerts
    from betting_app.services import prematch_quote_service as quotes_service
    from betting_app.services.prematch_quote_service import (
        PrematchMarketContext, PrematchQuote, quotes_consensus,
        serialize_focal_consensus, serialize_market_context,
    )

    active = get_active_hybrid()
    params = dict(alpha=active.alpha, temperature=active.temperature,
                  blending_mode=active.blending_mode)
    now = datetime.now(UTC)
    origin = now - timedelta(seconds=25)
    quotes = tuple(
        PrematchQuote(i, 201, i, f"book{i}", now - timedelta(seconds=29 + i),
                      "A", "B", odds, 2.0, None, None)
        for i, odds in ((1, 3.0), (2, 2.2), (3, 2.2))
    )
    context = PrematchMarketContext(
        201, "a", "b", "LCK", "upcoming", now + timedelta(hours=1),
        now, quotes, quotes_consensus(quotes).prob_a, (),
    )
    forecast = replace(context, decision_at=origin)
    focal = quotes_consensus(quotes, exclude_bookmaker_id=1)
    forecast_metadata = {
        **serialize_market_context(forecast, model_name=active.hybrid_model_name,
                                   model_version=active.hybrid_model_version),
        "base_model_name": C0_NATIVE.name, "base_model_version": C0_NATIVE.version,
    }
    diag = {
        **params, "uncertainty_required": True, "uncertainty_basis": "measured_bounds",
        "base_prob_a": 0.4, "p_low_a": 0.38, "p_low_b": 0.55,
        "epistemic_sigma_z": 0.15, "rating_disagreement": 0.0,
        "base_diagnostics": {"p_low_a": 0.38, "p_low_b": 0.55},
        "base_model_name": C0_NATIVE.name, "base_model_version": C0_NATIVE.version,
        "base_data_cutoff_at": (now - timedelta(minutes=1)).isoformat(),
        "market_context": forecast_metadata,
    }
    global_prob = blend_market_probability(0.4, context.prob_a, **params)
    focal_prob = blend_market_probability(0.4, focal.prob_a, **params)
    row = dict(
        signal_id=401, prediction_id=301, prediction_match_id=201,
        canonical_prediction_id=301, canonical_match_id=201, bookmaker_id=1,
        odds_snapshot_id=1, side="a", odds=3.0,
        model_name=active.hybrid_model_name, model_version=active.hybrid_model_version,
        prob_a=global_prob, prob_b=1-global_prob, model_prob=focal_prob,
        market_prob=focal.prob_a, ev=10.0, evaluated_at=now,
        status="new", prediction_status="active", predicted_at=now-timedelta(seconds=20),
        data_cutoff_at=origin, start_time_normalized=context.start_at.isoformat(),
        best_of=3, league="LCK", team_a_name="A", team_b_name="B", bookmaker_name="book1",
        diagnostics_json=diag,
        market_context_json={
            **serialize_market_context(context, model_name=active.hybrid_model_name,
                                       model_version=active.hybrid_model_version),
            "focal_consensus": serialize_focal_consensus(focal),
        },
    )
    cfg = SimpleNamespace(
        is_enabled=True, min_ev=0.0, min_odds=1.0, max_odds=12.0,
        cooldown_hours=6.0, discord_enabled=False, discord_webhook_url=None,
        telegram_enabled=False, telegram_bot_token=None, telegram_chat_id=None,
    )
    artifact = {
        "model_params_json": {**params, "base_model_name": C0_NATIVE.name,
                              "base_model_version": C0_NATIVE.version},
        "feature_schema_json": {"base_model": f"{C0_NATIVE.name}/{C0_NATIVE.version}"},
    }
    db = MagicMock()
    monkeypatch.setattr(alerts, "get_or_create_alert_config", lambda _: cfg)
    monkeypatch.setattr(alerts, "query_df", lambda *args: [row])
    monkeypatch.setattr(alerts, "load_prematch_quotes", lambda *args, **kwargs: {201: context})
    monkeypatch.setattr(quotes_service, "load_prematch_quotes",
                        lambda *args, decision_at, **kwargs: {201: replace(context, decision_at=decision_at)})
    monkeypatch.setattr(quotes_service, "query_df", lambda *args, **kwargs: pd.DataFrame([artifact]))
    monkeypatch.setattr(alerts, "is_alert_on_cooldown", lambda *args: False)
    monkeypatch.setattr(alerts, "load_config", lambda: SimpleNamespace(tax_rate=0.12))
    monkeypatch.setattr(alerts, "wallet_bankroll", lambda *args: None)
    with patch.object(alerts, "send_discord_notification"), patch.object(alerts, "send_telegram_notification"):
        result = alerts.scan_and_dispatch_ev_alerts(db)

    assert result["dispatched"] == 1
    expected_low = blend_market_probability(0.38, focal.prob_a, **params)
    assert result["alerts"][0]["ev"] == pytest.approx(expected_low * 3.0 * 0.88 - 1.0)
    assert result["alerts"][0]["ev"] != pytest.approx(global_prob * 3.0 * 0.88 - 1.0)
