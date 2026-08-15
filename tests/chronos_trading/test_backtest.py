"""Tests for WalkForwardBacktester utilities."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from chronos_trading.backtest import WalkForwardBacktester
from chronos_trading.config import TradingConfig
from chronos_trading.data.downloader import DataDownloader
from chronos_trading.data.universe import UniverseManager
from chronos_trading.options import BlackScholesEngine, OptionsPortfolioBuilder
from chronos_trading.predictor import ChronosPredictor
from chronos_trading.ranker import StockRanker


def _make_backtester(config=None):
	cfg = config or TradingConfig()
	return WalkForwardBacktester(
		config=cfg,
		universe_manager=UniverseManager(cfg),
		downloader=DataDownloader(cfg),
		predictor=ChronosPredictor(cfg),
		ranker=StockRanker(cfg),
		portfolio_builder=OptionsPortfolioBuilder(cfg, BlackScholesEngine()),
	)


def test_rebalance_dates_after_warmup():
	cfg = TradingConfig(BACKTEST_START='2014-01-01', BACKTEST_END='2020-01-01', WARMUP_YEARS=2)
	bt = _make_backtester(cfg)
	dates = bt._get_rebalance_dates()
	assert all(d >= date(2016, 1, 1) for d in dates)


def test_rebalance_dates_monthly():
	cfg = TradingConfig(BACKTEST_START='2020-01-01', BACKTEST_END='2022-01-01', WARMUP_YEARS=0)
	bt = _make_backtester(cfg)
	dates = bt._get_rebalance_dates()
	assert len(dates) >= 23
	gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
	assert all(20 <= g <= 40 for g in gaps)


def test_rebalance_dates_before_end():
	cfg = TradingConfig(BACKTEST_START='2018-01-01', BACKTEST_END='2019-01-01', WARMUP_YEARS=0)
	bt = _make_backtester(cfg)
	dates = bt._get_rebalance_dates()
	assert all(d < date(2019, 1, 1) for d in dates)


def test_equity_curve_accumulation():
	bt = _make_backtester()
	returns = [0.05, -0.03, 0.04]
	idx = pd.date_range('2020-01-01', periods=3, freq='MS')
	monthly = pd.Series(returns, index=idx)
	equity = (1.0 + monthly).cumprod()
	assert abs(equity.iloc[0] - 1.05) < 1e-10
	assert abs(equity.iloc[1] - 1.05 * 0.97) < 1e-10
	assert abs(equity.iloc[2] - 1.05 * 0.97 * 1.04) < 1e-10


def test_sharpe_known_series():
	bt = _make_backtester()
	returns = pd.Series([0.02, 0.03, 0.01, 0.02, 0.04, 0.01] * 10)
	mu = float(returns.mean())
	sigma = float(returns.std())
	expected = mu / sigma * math.sqrt(12)
	assert abs(bt._compute_sharpe(returns) - expected) < 1e-8


def test_sharpe_zero_vol():
	bt = _make_backtester()
	returns = pd.Series([0.01] * 12)
	assert bt._compute_sharpe(returns) == 0.0


def test_sharpe_insufficient_data():
	bt = _make_backtester()
	returns = pd.Series([0.01, 0.02])
	assert bt._compute_sharpe(returns) == 0.0


def test_max_drawdown_known():
	bt = _make_backtester()
	idx = pd.date_range('2020-01-01', periods=4, freq='MS')
	equity = pd.Series([1.0, 1.2, 0.9, 1.1], index=idx)
	dd = bt._compute_max_drawdown(equity)
	assert abs(dd - (1.2 - 0.9) / 1.2) < 1e-10


def test_max_drawdown_always_nonnegative():
	bt = _make_backtester()
	idx = pd.date_range('2020-01-01', periods=5, freq='MS')
	equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
	assert bt._compute_max_drawdown(equity) >= 0.0


def test_max_drawdown_all_decline():
	bt = _make_backtester()
	idx = pd.date_range('2020-01-01', periods=4, freq='MS')
	equity = pd.Series([1.0, 0.8, 0.6, 0.5], index=idx)
	dd = bt._compute_max_drawdown(equity)
	assert abs(dd - 0.5) < 1e-10
