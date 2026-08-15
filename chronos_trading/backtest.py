"""WalkForwardBacktester: monthly rebalancing walk-forward engine.

Timeline:
  2014-01 → 2016-01  Warmup: builds up 2 years of context data, no trades
  2016-01 → 2022-01  In-sample: 72 monthly rebalance steps
  2022-01 → 2024-01  Out-of-sample: 24 monthly steps

Each step:
  1. Get survivorship-bias-free universe for that date
  2. Download incremental price/covariate bars
  3. Run Chronos2 batch inference
  4. Rank stocks, select top-5 / bottom-5
  5. Close prior positions (re-price at today's market)
  6. Open new 60-DTE ITM calls + puts
  7. Record period P&L

No-lookahead-bias safeguards are enforced at every stage.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from pydantic import BaseModel, ConfigDict

from chronos_trading.config import TradingConfig
from chronos_trading.data.downloader import DataDownloader
from chronos_trading.data.universe import UniverseManager
from chronos_trading.options import OptionsPortfolioBuilder, Portfolio
from chronos_trading.predictor import ChronosPredictor
from chronos_trading.ranker import StockRanker

logger = logging.getLogger(__name__)


class BacktestPeriod(BaseModel):
	"""State and results for one monthly rebalance step."""

	model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

	step: int
	rebalance_date: date
	n_universe: int
	predictions_cached: bool = False
	portfolio: Portfolio | None = None
	period_pnl: float | None = None
	period_return: float | None = None


class BacktestResult(BaseModel):
	"""Aggregated results across all backtest periods."""

	model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

	periods: list[BacktestPeriod]
	equity_curve: pd.Series
	monthly_returns: pd.Series
	in_sample_sharpe: float
	oos_sharpe: float
	max_drawdown: float
	total_return: float
	annualized_return: float
	annualized_vol: float
	in_sample_start: date
	oos_start: date
	oos_end: date


class WalkForwardBacktester:
	"""Runs the full walk-forward backtest."""

	def __init__(
		self,
		config: TradingConfig,
		universe_manager: UniverseManager,
		downloader: DataDownloader,
		predictor: ChronosPredictor,
		ranker: StockRanker,
		portfolio_builder: OptionsPortfolioBuilder,
	) -> None:
		self._config = config
		self._universe = universe_manager
		self._downloader = downloader
		self._predictor = predictor
		self._ranker = ranker
		self._builder = portfolio_builder
		self._price_cache: dict[str, pd.DataFrame] = {}
		self._cov_cache: pd.DataFrame = pd.DataFrame()
		self._rf_cache: pd.Series = pd.Series(dtype=float)

	async def run(self) -> BacktestResult:
		"""Execute the full walk-forward backtest and return aggregated results."""
		backtest_start = date.fromisoformat(self._config.BACKTEST_START)
		backtest_end = date.fromisoformat(self._config.BACKTEST_END)
		oos_start = date.fromisoformat(self._config.OOS_START)

		logger.info('[Backtest] Pre-downloading all data %s → %s', backtest_start, backtest_end)
		all_tickers = self._universe.all_tickers_ever()

		self._cov_cache = await self._downloader.download_covariates(backtest_start, backtest_end)
		self._rf_cache = await self._downloader.download_risk_free_rates(backtest_start, backtest_end)
		self._price_cache = await self._downloader.download_prices(all_tickers, backtest_start, backtest_end)

		logger.info('[Backtest] Data ready: %d tickers with price data', len(self._price_cache))

		rebalance_dates = self._get_rebalance_dates()
		logger.info('[Backtest] %d rebalance steps (%s → %s)', len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1])

		periods: list[BacktestPeriod] = []
		prev_portfolio: Portfolio | None = None

		for step_idx, step_date in enumerate(rebalance_dates):
			self._log_backtest_progress(step_idx + 1, len(rebalance_dates), step_date)
			try:
				period = await self._run_step(step_idx, step_date, prev_portfolio)
				periods.append(period)
				prev_portfolio = period.portfolio
			except Exception as exc:
				logger.error('[Backtest] Step %d (%s) failed: %s', step_idx, step_date, exc, exc_info=True)
				periods.append(
					BacktestPeriod(step=step_idx, rebalance_date=step_date, n_universe=0, period_pnl=0.0, period_return=0.0)
				)

		return self._build_result(periods, oos_start, backtest_end)

	async def _run_step(self, step_idx: int, step_date: date, prev_portfolio: Portfolio | None) -> BacktestPeriod:
		universe_snap = self._universe.get_universe_at(step_date)
		active_tickers = universe_snap.tickers

		price_data_as_of: dict[str, pd.DataFrame] = {}
		for ticker in active_tickers:
			df = self._price_cache.get(ticker)
			if df is not None and not df.empty:
				sliced = df[df.index <= step_date]
				if not sliced.empty:
					price_data_as_of[ticker] = sliced

		cov_as_of = self._cov_cache[self._cov_cache.index <= step_date] if not self._cov_cache.empty else pd.DataFrame()

		predictions = self._predictor.predict(price_data_as_of, cov_as_of, as_of_date=step_date)
		ranked = self._ranker.rank(predictions)

		period_pnl = 0.0
		if prev_portfolio is not None:
			for pos in prev_portfolio.all_positions():
				closed = self._builder.price_position(pos, self._price_cache, self._rf_cache, step_date)
				if closed.pnl is not None:
					period_pnl += closed.pnl

		new_portfolio = self._builder.construct(ranked, price_data_as_of, self._rf_cache, step_date)

		deployed = (
			prev_portfolio.total_notional
			if prev_portfolio and prev_portfolio.total_notional > 0
			else self._config.INITIAL_CAPITAL
		)
		period_return = period_pnl / deployed if deployed > 0 else 0.0

		return BacktestPeriod(
			step=step_idx,
			rebalance_date=step_date,
			n_universe=len(active_tickers),
			predictions_cached=True,
			portfolio=new_portfolio,
			period_pnl=period_pnl,
			period_return=period_return,
		)

	def _build_result(self, periods: list[BacktestPeriod], oos_start: date, oos_end: date) -> BacktestResult:
		dates = [p.rebalance_date for p in periods]
		returns = [p.period_return or 0.0 for p in periods]
		monthly_returns = pd.Series(returns, index=pd.to_datetime(dates), name='monthly_return')
		equity_curve = (1.0 + monthly_returns).cumprod()
		equity_curve.iloc[0] = 1.0 + returns[0]
		is_mask = monthly_returns.index < pd.Timestamp(oos_start)
		oos_mask = monthly_returns.index >= pd.Timestamp(oos_start)
		is_returns = monthly_returns[is_mask]
		oos_returns = monthly_returns[oos_mask]
		in_sample_sharpe = self._compute_sharpe(is_returns)
		oos_sharpe = self._compute_sharpe(oos_returns)
		max_dd = self._compute_max_drawdown(equity_curve)
		n_months = len(returns)
		total_return = float(equity_curve.iloc[-1]) - 1.0
		annualized_return = (1 + total_return) ** (12 / max(n_months, 1)) - 1.0
		annualized_vol = float(monthly_returns.std()) * (12**0.5)
		return BacktestResult(
			periods=periods,
			equity_curve=equity_curve,
			monthly_returns=monthly_returns,
			in_sample_sharpe=in_sample_sharpe,
			oos_sharpe=oos_sharpe,
			max_drawdown=max_dd,
			total_return=total_return,
			annualized_return=annualized_return,
			annualized_vol=annualized_vol,
			in_sample_start=date.fromisoformat(self._config.BACKTEST_START),
			oos_start=oos_start,
			oos_end=oos_end,
		)

	def _get_rebalance_dates(self) -> list[date]:
		try:
			import pandas_market_calendars as mcal

			nyse = mcal.get_calendar('NYSE')
		except Exception:
			nyse = None
		backtest_start = date.fromisoformat(self._config.BACKTEST_START)
		backtest_end = date.fromisoformat(self._config.BACKTEST_END)
		trade_start = date(backtest_start.year + self._config.WARMUP_YEARS, backtest_start.month, backtest_start.day)
		rebalance_dates: list[date] = []
		current = trade_start
		while current < backtest_end:
			month_start = date(current.year, current.month, 1)
			first_trade = self._first_trading_day(month_start, nyse)
			if first_trade and first_trade < backtest_end:
				rebalance_dates.append(first_trade)
			if current.month == 12:
				current = date(current.year + 1, 1, 1)
			else:
				current = date(current.year, current.month + 1, 1)
		return rebalance_dates

	def _first_trading_day(self, month_start: date, calendar) -> date | None:
		if calendar is not None:
			try:
				end_of_month = date(month_start.year, month_start.month, 28)
				schedule = calendar.schedule(start_date=str(month_start), end_date=str(end_of_month))
				if not schedule.empty:
					return schedule.index[0].date()
			except Exception:
				pass
		candidate = month_start
		for _ in range(10):
			if candidate.weekday() < 5:
				return candidate
			candidate += timedelta(days=1)
		return month_start

	def _compute_sharpe(self, returns: pd.Series, periods_per_year: int = 12) -> float:
		if len(returns) < 3:
			return 0.0
		mu = float(returns.mean())
		sigma = float(returns.std())
		if sigma < 1e-12:
			return 0.0
		return mu / sigma * (periods_per_year**0.5)

	def _compute_max_drawdown(self, equity: pd.Series) -> float:
		rolling_max = equity.cummax()
		drawdowns = (equity - rolling_max) / rolling_max
		return float(abs(drawdowns.min()))

	def _log_backtest_progress(self, step: int, total: int, step_date: date) -> None:
		pct = 100 * step / total
		logger.info('[Backtest] Step %d/%d (%.0f%%) — %s', step, total, pct, step_date)
