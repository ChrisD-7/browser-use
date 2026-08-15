"""Tests for the Black-Scholes engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from chronos_trading.options import BlackScholesEngine

BS = BlackScholesEngine()


def test_call_price_known_value():
	price = BS.call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
	assert abs(price - 10.45) < 0.05


def test_put_call_parity():
	S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.04, 0.25
	call = BS.call_price(S, K, T, r, sigma)
	put = BS.put_price(S, K, T, r, sigma)
	lhs = call - put
	rhs = S - K * math.exp(-r * T)
	assert abs(lhs - rhs) < 1e-8


def test_call_price_at_expiry_itm():
	assert abs(BS.call_price(110, 100, 0, 0.05, 0.20) - 10.0) < 1e-10


def test_call_price_at_expiry_otm():
	assert BS.call_price(90, 100, 0, 0.05, 0.20) == 0.0


def test_put_price_at_expiry_itm():
	assert abs(BS.put_price(90, 100, 0, 0.05, 0.20) - 10.0) < 1e-10


def test_call_price_positive():
	assert BS.call_price(100, 100, 1.0, 0.05, 0.30) > 0


def test_put_price_positive():
	assert BS.put_price(100, 100, 1.0, 0.05, 0.30) > 0


def test_atm_call_delta_near_half():
	delta = BS.call_delta(100, 100, 1.0, 0.04, 0.20)
	assert 0.50 < delta < 0.70


def test_deep_itm_call_delta_near_one():
	delta = BS.call_delta(200, 100, 1.0, 0.04, 0.20)
	assert delta > 0.99


def test_put_delta_range():
	delta = BS.put_delta(100, 100, 1.0, 0.04, 0.20)
	assert -1.0 < delta < 0.0


def test_call_put_delta_relationship():
	S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
	cd = BS.call_delta(S, K, T, r, sigma)
	pd_ = BS.put_delta(S, K, T, r, sigma)
	assert abs(pd_ - (cd - 1.0)) < 1e-12


def test_find_strike_call_delta_80():
	S, T, r, sigma = 100.0, 60 / 252, 0.04, 0.25
	K = BS.find_strike_for_delta(S, 0.80, T, r, sigma, 'call')
	actual_delta = BS.call_delta(S, K, T, r, sigma)
	assert abs(actual_delta - 0.80) < 1e-3


def test_find_strike_put_delta_minus80():
	S, T, r, sigma = 100.0, 60 / 252, 0.04, 0.25
	K = BS.find_strike_for_delta(S, -0.80, T, r, sigma, 'put')
	actual_delta = BS.put_delta(S, K, T, r, sigma)
	assert abs(actual_delta - (-0.80)) < 1e-3


def test_itm_call_strike_below_spot():
	S, T, r, sigma = 150.0, 60 / 252, 0.04, 0.20
	K = BS.find_strike_for_delta(S, 0.80, T, r, sigma, 'call')
	assert K < S


def test_itm_put_strike_above_spot():
	S, T, r, sigma = 150.0, 60 / 252, 0.04, 0.20
	K = BS.find_strike_for_delta(S, -0.80, T, r, sigma, 'put')
	assert K > S


def test_realized_vol_known_sigma():
	rng = np.random.default_rng(0)
	daily_std = 0.01
	returns = pd.Series(rng.normal(0, daily_std, 100))
	rv = BS.realized_vol(returns, window=30, vrp_multiplier=1.15)
	expected = daily_std * math.sqrt(252) * 1.15
	assert abs(rv - expected) / expected < 0.30


def test_realized_vol_insufficient_data():
	returns = pd.Series([0.01, 0.02, 0.01])
	rv = BS.realized_vol(returns, window=30)
	assert math.isnan(rv)


def test_realized_vol_vrp_multiplier():
	rng = np.random.default_rng(1)
	returns = pd.Series(rng.normal(0, 0.01, 60))
	rv_1 = BS.realized_vol(returns, vrp_multiplier=1.0)
	rv_115 = BS.realized_vol(returns, vrp_multiplier=1.15)
	assert abs(rv_115 / rv_1 - 1.15) < 1e-9
