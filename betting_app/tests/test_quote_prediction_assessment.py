from datetime import UTC, datetime

import pytest

from betting_app.services import prematch_quote_service as service

NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)


def _quote(book_id, odds_a, odds_b):
    return service.PrematchQuote(book_id, 1, book_id, f'Book {book_id}', NOW,
                                 'Alpha', 'Beta', odds_a, odds_b, None, None)


def _context(*quotes):
    return service.PrematchMarketContext(1, 'Alpha', 'Beta', 'LCK', 'scheduled', None,
                                         NOW, tuple(quotes), None, ())


def _hybrid_diagnostics():
    return {
        'market_features_used': True,
        'base_prob_a': 0.65,
        'alpha': 0.5,
        'temperature': 1.0,
        'blending_mode': 'logit_shrinkage',
        'base_diagnostics': {'p_low_a': 0.60, 'p_low_b': 0.30},
        'p_low_a': 0.62,
        'p_low_b': 0.32,
    }


def _assess(context, focal, diagnostics=None):
    return service.qualify_prediction_for_quote(
        context=context, quote=focal, prob_a=0.63, prob_b=0.37,
        diagnostics=diagnostics or _hybrid_diagnostics(), model_name='Hybrid',
        league='LCK', best_of=3,
    )


def test_two_total_books_are_insufficient_for_focal_assessment():
    focal, other = _quote(1, 2.0, 2.0), _quote(2, 2.0, 2.0)
    result = _assess(_context(focal, other), focal)
    assert result['reason'] == service.CONSENSUS_INSUFFICIENT_REASON
    assert all(not decision[0] for decision in result['qualified'].values())


def test_focal_price_change_does_not_change_consensus_reblend_or_qualification():
    focal, peer_a, peer_b = _quote(1, 2.0, 2.0), _quote(2, 1.6, 2.4), _quote(3, 2.4, 1.6)
    context = _context(focal, peer_a, peer_b)
    first = _assess(context, focal)
    changed_focal = _quote(1, 4.0, 1.2)
    second = _assess(_context(changed_focal, peer_a, peer_b), changed_focal)
    assert first['market_prob_a'] == second['market_prob_a']
    assert first['prob_a'] == pytest.approx(second['prob_a'])
    assert first['prob_a'] != 0.63
    assert first['diagnostics']['p_low_a'] != _hybrid_diagnostics()['p_low_a']
    assert first['diagnostics']['base_diagnostics'] == _hybrid_diagnostics()['base_diagnostics']


def test_changed_other_book_changes_reblend_and_uses_current_quote_odds():
    focal, peer_a, peer_b = _quote(1, 2.0, 2.0), _quote(2, 1.6, 2.4), _quote(3, 2.4, 1.6)
    first = _assess(_context(focal, peer_a, peer_b), focal)
    changed_peer = _quote(3, 1.3, 3.0)
    second = _assess(_context(focal, peer_a, changed_peer), focal)
    assert second['market_prob_a'] != first['market_prob_a']
    assert second['prob_a'] != pytest.approx(first['prob_a'])


def test_missing_hybrid_parameters_fail_closed_and_invalid_json_is_safe():
    focal = _quote(1, 2.0, 2.0)
    peers = (_quote(2, 1.8, 2.2), _quote(3, 2.2, 1.8))
    for diagnostics in (
        {'market_features_used': True, 'base_prob_a': 0.6},
        '{not json',
    ):
        result = service.qualify_prediction_for_quote(
            context=_context(focal, *peers), quote=focal, prob_a=0.7, prob_b=0.3,
            diagnostics=diagnostics, model_name='Hybrid',
        )
        assert all(not decision[0] for decision in result['qualified'].values())
        assert result['reason'] is not None


def test_native_probability_is_not_reblended_and_missing_raw_bounds_are_removed():
    focal = _quote(1, 2.0, 2.0)
    diagnostics = {'uncertainty_required': False, 'p_low_a': 0.99, 'p_low_b': 0.99}
    result = service.qualify_prediction_for_quote(
        context=_context(focal, _quote(2, 1.8, 2.2), _quote(3, 2.2, 1.8)),
        quote=focal, prob_a=0.55, prob_b=0.45, diagnostics=diagnostics,
        model_name='c0-native',
    )
    assert result['prob_a'] == 0.55
    assert result['diagnostics']['p_low_a'] == 0.99
    assert diagnostics['p_low_a'] == 0.99


def test_default_ev_cap_and_bo1_are_passed_to_qualification():
    focal = _quote(1, 4.0, 1.2)
    context = _context(focal, _quote(2, 1.8, 2.2), _quote(3, 2.2, 1.8))
    result = service.qualify_prediction_for_quote(
        context=context, quote=focal, prob_a=0.99, prob_b=0.01,
        diagnostics={'uncertainty_required': False}, model_name='c0-native', best_of=1,
    )
    assert result['qualified']['a'][0] is False
    assert result['qualified']['a'][1] != 'eligible'
    # Default EV ceiling is enforced, rather than disabling it for read-side consumers.
    assert 'max_ev_net' in result['qualified']['a'][2]
