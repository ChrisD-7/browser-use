"""Tests for StockRanker."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from chronos_trading.config import TradingConfig
from chronos_trading.predictor import ChronosPredictionResult, QuantilePrediction
from chronos_trading.ranker import StockRanker


def _make_pred(ticker, horizon, q10, q25, q50, q75, q90):
	return QuantilePrediction(ticker=ticker, prediction_date=date(2022, 1, 3), horizon=horizon, q10=q10, q25=q25, q50=q50, q75=q75, q90=q90)


def _make_result(preds):
	return ChronosPredictionResult(predictions=preds, model_id='amazon/chronos-2', prediction_date=date(2022, 1, 3), inference_time_seconds=1.0)


def _config(top_n=5, bottom_n=5):
	return TradingConfig(TOP_N=top_n, BOTTOM_N=bottom_n)


def _ranker(config=None):
	return StockRanker(config or _config())


def test_direction_equals_q50():
	r = _ranker()
	pred = _make_pred('AAPL', 5, -0.05, -0.02, 0.03, 0.07, 0.12)
	assert r._direction(pred) == 0.03


def test_conviction_symmetric_quantiles():
	r = _ranker()
	pred = _make_pred('X', 5, -0.01, 0.00, 0.01, 0.00, 0.02)
	assert r._conviction(pred) == 5.0


def test_conviction_wider_iqr_lower():
	r = _ranker()
	tight = _make_pred('T', 5, -0.10, 0.25, 0.30, 0.55, 0.80)
	wide = _make_pred('W', 5, -0.30, 0.05, 0.20, 0.65, 1.00)
	assert r._conviction(tight) > r._conviction(wide)


def test_asymmetry_right_skew_positive():
	r = _ranker()
	pred = _make_pred('R', 5, -0.01, 0.00, 0.01, 0.02, 0.06)
	assert r._asymmetry(pred) > 0


def test_asymmetry_left_skew_negative():
	r = _ranker()
	pred = _make_pred('L', 5, -0.06, -0.02, -0.01, 0.00, 0.01)
	assert r._asymmetry(pred) < 0


def test_asymmetry_symmetric_zero():
	r = _ranker()
	pred = _make_pred('S', 5, -0.03, -0.01, 0.0, 0.01, 0.03)
	assert r._asymmetry(pred) == pytest.approx(0.0)


def test_horizon_weights_sum_to_one():
	r = _ranker()
	weights = r._compute_horizon_weights([5, 10, 21, 42, 63])
	assert sum(weights.values()) == pytest.approx(1.0)


def test_shorter_horizons_get_more_weight():
	r = _ranker()
	weights = r._compute_horizon_weights([5, 63])
	assert weights[5] > weights[63]


def test_z_scores_mean_zero():
	r = _ranker()
	raw = {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0, 'E': 5.0}
	z = r._z_score_universe(raw)
	mean = sum(z.values()) / len(z)
	assert mean == pytest.approx(0.0, abs=1e-10)


def test_z_scores_unit_std():
	r = _ranker()
	raw = {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0, 'E': 5.0}
	z = r._z_score_universe(raw)
	vals = list(z.values())
	std = (sum((v**2 for v in vals)) / len(vals)) ** 0.5
	assert std == pytest.approx(1.0, rel=0.1)


def test_z_scores_all_identical_scores():
	r = _ranker()
	raw = {'A': 5.0, 'B': 5.0, 'C': 5.0}
	z = r._z_score_universe(raw)
	assert all(v == 0.0 for v in z.values())


def _make_universe_preds(n, seed=42):
	rng = np.random.default_rng(seed)
	preds = []
	for i in range(n):
		ticker = f'TICK{i:03d}'
		for h in [5, 10, 21, 42, 63]:
			q = np.sort(rng.normal(0, 0.03, 5))
			preds.append(_make_pred(ticker, h, *q))
	return preds


def test_rank_picks_correct_count():
	ranker = _ranker(_config(top_n=5, bottom_n=5))
	ranked = ranker.rank(_make_result(_make_universe_preds(20)))
	assert len(ranked.long_picks) == 5
	assert len(ranked.short_picks) == 5


def test_rank_long_and_short_disjoint():
	ranker = _ranker()
	ranked = ranker.rank(_make_result(_make_universe_preds(20)))
	assert set(ranked.long_picks).isdisjoint(set(ranked.short_picks))


def test_rank_long_higher_z_than_short():
	ranker = _ranker()
	ranked = ranker.rank(_make_result(_make_universe_preds(20)))
	score_map = {s.ticker: s.z_score for s in ranked.scores}
	min_long_z = min(score_map[t] for t in ranked.long_picks)
	max_short_z = max(score_map[t] for t in ranked.short_picks)
	assert min_long_z > max_short_z


def test_rank_scores_sorted_descending():
	ranker = _ranker()
	ranked = ranker.rank(_make_result(_make_universe_preds(15)))
	z_scores = [s.z_score for s in ranked.scores]
	assert z_scores == sorted(z_scores, reverse=True)


def test_rank_empty_predictions():
	ranker = _ranker()
	ranked = ranker.rank(_make_result([]))
	assert ranked.long_picks == []
	assert ranked.short_picks == []
