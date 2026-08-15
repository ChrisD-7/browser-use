"""Tests for RiskAnalyzer."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from chronos_trading.risk import RiskAnalyzer


def _monthly_returns(n=100, seed=0):
	rng = np.random.default_rng(seed)
	idx = pd.date_range('2015-01-01', periods=n, freq='MS')
	return pd.Series(rng.normal(0.01, 0.05, n), index=idx)


def test_historical_var_negative():
	returns = _monthly_returns(200)
	var = RiskAnalyzer.historical_var(returns, 0.95)
	assert var <= returns.quantile(0.10)


def test_historical_var_95_worse_than_median():
	returns = _monthly_returns()
	var95 = RiskAnalyzer.historical_var(returns, 0.95)
	assert var95 < float(returns.median())


def test_historical_var_99_worse_than_95():
	returns = _monthly_returns(200)
	var95 = RiskAnalyzer.historical_var(returns, 0.95)
	var99 = RiskAnalyzer.historical_var(returns, 0.99)
	assert var99 <= var95


def test_var_known_distribution():
	vals = list(range(-100, 0))
	returns = pd.Series(vals, dtype=float)
	var95 = RiskAnalyzer.historical_var(returns, 0.95)
	assert var95 == -95.0


def test_cvar_worse_than_or_equal_to_var():
	for seed in range(5):
		returns = _monthly_returns(120, seed=seed)
		for conf in [0.95, 0.99]:
			var = RiskAnalyzer.historical_var(returns, conf)
			cvar = RiskAnalyzer.cvar(returns, conf)
			assert cvar <= var + 1e-10


def test_cvar_is_mean_of_tail():
	returns = pd.Series(range(-100, 0), dtype=float)
	expected_cvar = (-100 + -99 + -98 + -97 + -96 + -95) / 6
	cvar = RiskAnalyzer.cvar(returns, 0.95)
	assert abs(cvar - expected_cvar) < 1.0


def test_parametric_var_standard_normal():
	rng = np.random.default_rng(999)
	returns = pd.Series(rng.standard_normal(10_000))
	var95 = RiskAnalyzer.parametric_var(returns, 0.95)
	assert abs(var95 - (-1.645)) < 0.05


def test_parametric_var_scales_with_sigma():
	rng = np.random.default_rng(42)
	r1 = pd.Series(rng.normal(0, 0.05, 1000))
	r2 = pd.Series(rng.normal(0, 0.10, 1000))
	v1 = RiskAnalyzer.parametric_var(r1, 0.95)
	v2 = RiskAnalyzer.parametric_var(r2, 0.95)
	assert v2 < v1
	assert abs(v2 / v1 - 2.0) < 0.3


def test_mc_shape_and_bounds():
	returns = _monthly_returns()
	mc = RiskAnalyzer.monte_carlo_gbm(returns, n_paths=1000, horizon_days=60)
	assert mc.percentile_5 < mc.percentile_50 < mc.percentile_95
	assert 0.0 <= mc.prob_loss <= 1.0


def test_mc_reproducible():
	returns = _monthly_returns()
	mc1 = RiskAnalyzer.monte_carlo_gbm(returns, n_paths=500, horizon_days=50, seed=42)
	mc2 = RiskAnalyzer.monte_carlo_gbm(returns, n_paths=500, horizon_days=50, seed=42)
	assert mc1.percentile_50 == mc2.percentile_50
	assert mc1.prob_loss == mc2.prob_loss


def test_mc_zero_vol_constant_path():
	returns = pd.Series([0.01] * 120, dtype=float)
	mc = RiskAnalyzer.monte_carlo_gbm(returns, n_paths=100, horizon_days=252)
	assert abs(mc.percentile_95 - mc.percentile_5) < 0.1


def test_mc_prob_loss_high_vol():
	rng = np.random.default_rng(7)
	high_vol_returns = pd.Series(rng.normal(0.005, 0.15, 120))
	mc = RiskAnalyzer.monte_carlo_gbm(high_vol_returns, n_paths=5000, horizon_days=252)
	assert mc.prob_loss > 0.05


def test_mc_var_negative():
	returns = _monthly_returns(seed=13)
	mc = RiskAnalyzer.monte_carlo_gbm(returns, n_paths=1000, horizon_days=252)
	assert math.isfinite(mc.var_95_mc)


def test_sortino_positive_returns():
	returns = pd.Series([0.05, 0.03, 0.04, 0.02, -0.01, 0.06, 0.04] * 10)
	assert RiskAnalyzer.sortino_ratio(returns) > 0


def test_sortino_all_negative():
	returns = pd.Series([-0.03, -0.05, -0.02, -0.04] * 20)
	assert RiskAnalyzer.sortino_ratio(returns) < 0


def test_sortino_insufficient_data():
	returns = pd.Series([0.01, 0.02])
	assert RiskAnalyzer.sortino_ratio(returns) == 0.0
