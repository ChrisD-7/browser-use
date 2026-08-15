"""Black-Scholes pricing engine and options portfolio builder.

Synthetic IV = realized_vol(30d) * VRP_MULTIPLIER (default 1.15)
ITM strike selection via bisection on N(d1) for calls, N(d1)-1 for puts.
Position sizing: equal-weight notional across TOP_N + BOTTOM_N legs.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from chronos_trading.config import TradingConfig
from chronos_trading.ranker import RankedUniverse

logger = logging.getLogger(__name__)


class OptionSpec(BaseModel):
	"""Static specification for an options contract at entry."""

	model_config = ConfigDict(extra='forbid')

	ticker: str
	option_type: Literal['call', 'put']
	S: float
	K: float
	T: float
	r: float
	sigma: float
	entry_price: float
	entry_date: date
	target_delta: float
	dte: int


class OptionPosition(BaseModel):
	"""An open options position with optional exit information."""

	model_config = ConfigDict(extra='forbid')

	spec: OptionSpec
	n_contracts: int = Field(default=1, description='Number of contracts (1 contract = 100 shares)')
	notional: float = Field(description='entry_price * n_contracts * 100')
	exit_date: date | None = None
	exit_price: float | None = None
	pnl: float | None = None
	pnl_pct: float | None = None


class Portfolio(BaseModel):
	"""A rebalance-period portfolio of options positions."""

	model_config = ConfigDict(extra='forbid')

	as_of_date: date
	long_positions: list[OptionPosition]
	short_positions: list[OptionPosition]
	total_notional: float
	cash: float = 0.0

	def all_positions(self) -> list[OptionPosition]:
		return self.long_positions + self.short_positions


class BlackScholesEngine:
	"""Standard Black-Scholes formulas plus ITM strike finder via bisection."""

	@staticmethod
	def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
		if T <= 0 or sigma <= 0 or K <= 0:
			return 0.0
		return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

	@staticmethod
	def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
		return BlackScholesEngine.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

	@staticmethod
	def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
		if T <= 0:
			return max(S - K, 0.0)
		from scipy.stats import norm

		d1 = BlackScholesEngine.d1(S, K, T, r, sigma)
		d2 = d1 - sigma * math.sqrt(T)
		return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

	@staticmethod
	def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
		if T <= 0:
			return max(K - S, 0.0)
		from scipy.stats import norm

		d1 = BlackScholesEngine.d1(S, K, T, r, sigma)
		d2 = d1 - sigma * math.sqrt(T)
		return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

	@staticmethod
	def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
		"""N(d1) — ranges from 0 to 1."""
		from scipy.stats import norm

		return float(norm.cdf(BlackScholesEngine.d1(S, K, T, r, sigma)))

	@staticmethod
	def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
		"""N(d1) - 1 — ranges from -1 to 0."""
		return BlackScholesEngine.call_delta(S, K, T, r, sigma) - 1.0

	@staticmethod
	def find_strike_for_delta(
		S: float,
		target_delta: float,
		T: float,
		r: float,
		sigma: float,
		option_type: Literal['call', 'put'],
		tol: float = 1e-4,
		max_iter: int = 50,
	) -> float:
		"""Bisection to find strike K that produces the target delta."""
		if option_type == 'call':
			assert 0 < target_delta <= 1.0, f'Call target_delta must be in (0,1], got {target_delta}'
			lo, hi = 0.5 * S, S * 1.5

			def objective(K: float) -> float:
				return BlackScholesEngine.call_delta(S, K, T, r, sigma) - target_delta

		else:
			assert -1.0 <= target_delta < 0, f'Put target_delta must be in [-1,0), got {target_delta}'
			lo, hi = 0.5 * S, 2.0 * S

			def objective(K: float) -> float:
				return BlackScholesEngine.put_delta(S, K, T, r, sigma) - target_delta

		for _ in range(max_iter):
			mid = (lo + hi) / 2.0
			val = objective(mid)
			if abs(val) < tol:
				return mid
			# Higher K → lower d1 → lower delta (both calls and puts)
			if val > 0:
				lo = mid
			else:
				hi = mid
		return (lo + hi) / 2.0

	@staticmethod
	def realized_vol(returns: pd.Series, window: int = 30, vrp_multiplier: float = 1.15) -> float:
		"""Annualised realized vol from daily log-returns, scaled by VRP multiplier."""
		tail = returns.dropna().tail(window)
		if len(tail) < 5:
			return float('nan')
		daily_std = float(tail.std())
		return daily_std * math.sqrt(252) * vrp_multiplier


class OptionsPortfolioBuilder:
	"""Constructs and values the options portfolio from ranked stocks."""

	def __init__(self, config: TradingConfig, bs: BlackScholesEngine | None = None) -> None:
		self._config = config
		self._bs = bs or BlackScholesEngine()

	def construct(
		self,
		ranked: RankedUniverse,
		price_data: dict[str, pd.DataFrame],
		risk_free_rates: pd.Series,
		as_of_date: date,
	) -> Portfolio:
		long_positions: list[OptionPosition] = []
		short_positions: list[OptionPosition] = []
		r = self._get_risk_free_rate(risk_free_rates, as_of_date)
		T = self._config.TARGET_DTE / 252.0
		for ticker in ranked.long_picks:
			pos = self._build_position(ticker, 'call', price_data, r, T, as_of_date)
			if pos:
				long_positions.append(pos)
		for ticker in ranked.short_picks:
			pos = self._build_position(ticker, 'put', price_data, r, T, as_of_date)
			if pos:
				short_positions.append(pos)
		total_notional = sum(p.notional for p in long_positions + short_positions)
		portfolio = Portfolio(
			as_of_date=as_of_date,
			long_positions=long_positions,
			short_positions=short_positions,
			total_notional=total_notional,
		)
		self._log_portfolio_summary(portfolio)
		return portfolio

	def _build_position(
		self,
		ticker: str,
		option_type: Literal['call', 'put'],
		price_data: dict[str, pd.DataFrame],
		r: float,
		T: float,
		as_of_date: date,
	) -> OptionPosition | None:
		df = price_data.get(ticker)
		if df is None or df.empty:
			logger.debug('[Options] No price data for %s — skipping', ticker)
			return None
		df_prior = df[df.index <= as_of_date]
		if df_prior.empty:
			return None
		S = float(df_prior['adj_close'].iloc[-1])
		if S <= 0:
			return None
		log_returns = df_prior['log_return'].dropna() if 'log_return' in df_prior.columns else pd.Series(dtype=float)
		sigma = BlackScholesEngine.realized_vol(log_returns, vrp_multiplier=self._config.VRP_MULTIPLIER)
		if math.isnan(sigma) or sigma <= 0:
			sigma = 0.25
		target_delta = self._config.TARGET_DELTA if option_type == 'call' else -self._config.TARGET_DELTA
		try:
			K = BlackScholesEngine.find_strike_for_delta(S, target_delta, T, r, sigma, option_type)
		except Exception as exc:
			logger.debug('[Options] Strike finder failed for %s: %s', ticker, exc)
			K = S * (0.90 if option_type == 'call' else 1.10)
		price_fn = BlackScholesEngine.call_price if option_type == 'call' else BlackScholesEngine.put_price
		entry_price = price_fn(S, K, T, r, sigma)
		if entry_price <= 0:
			return None
		per_leg_capital = self._config.position_size
		n_contracts = max(1, int(per_leg_capital / (entry_price * 100)))
		notional = entry_price * n_contracts * 100
		spec = OptionSpec(
			ticker=ticker,
			option_type=option_type,
			S=S, K=K, T=T, r=r, sigma=sigma,
			entry_price=entry_price,
			entry_date=as_of_date,
			target_delta=abs(target_delta),
			dte=self._config.TARGET_DTE,
		)
		return OptionPosition(spec=spec, n_contracts=n_contracts, notional=notional)

	def price_position(
		self,
		position: OptionPosition,
		price_data: dict[str, pd.DataFrame],
		risk_free_rates: pd.Series,
		as_of_date: date,
	) -> OptionPosition:
		spec = position.spec
		df = price_data.get(spec.ticker)
		if df is None or df.empty:
			return position
		df_prior = df[df.index <= as_of_date]
		if df_prior.empty:
			return position
		S_exit = float(df_prior['adj_close'].iloc[-1])
		r_exit = self._get_risk_free_rate(risk_free_rates, as_of_date)
		days_held = (as_of_date - spec.entry_date).days
		T_exit = max((spec.dte - days_held) / 252.0, 0.0)
		log_returns = df_prior['log_return'].dropna() if 'log_return' in df_prior.columns else pd.Series(dtype=float)
		sigma_exit = BlackScholesEngine.realized_vol(log_returns, vrp_multiplier=self._config.VRP_MULTIPLIER)
		if math.isnan(sigma_exit) or sigma_exit <= 0:
			sigma_exit = spec.sigma
		price_fn = BlackScholesEngine.call_price if spec.option_type == 'call' else BlackScholesEngine.put_price
		exit_price = price_fn(S_exit, spec.K, T_exit, r_exit, sigma_exit)
		gross_pnl = (exit_price - spec.entry_price) * position.n_contracts * 100
		friction = position.notional * self._config.ROUND_TRIP_COST
		net_pnl = gross_pnl - friction
		pnl_pct = net_pnl / max(position.notional, 1e-6)
		return OptionPosition(
			spec=spec,
			n_contracts=position.n_contracts,
			notional=position.notional,
			exit_date=as_of_date,
			exit_price=exit_price,
			pnl=net_pnl,
			pnl_pct=pnl_pct,
		)

	def _get_risk_free_rate(self, risk_free_rates: pd.Series, as_of_date: date) -> float:
		shifted_date = as_of_date - timedelta(days=1)
		try:
			idx = pd.to_datetime([str(d) for d in risk_free_rates.index])
			mask = idx <= pd.Timestamp(shifted_date)
			if mask.any():
				return float(risk_free_rates.values[mask][-1])
		except Exception:
			pass
		return 0.04

	def _log_portfolio_summary(self, portfolio: Portfolio) -> None:
		logger.info(
			'[Options] %s — %d calls + %d puts | notional $%.0f',
			portfolio.as_of_date,
			len(portfolio.long_positions),
			len(portfolio.short_positions),
			portfolio.total_notional,
		)
		for pos in portfolio.long_positions[:3]:
			s = pos.spec
			logger.debug('  CALL %s S=%.2f K=%.2f σ=%.1f%% entry=$%.2f n=%d', s.ticker, s.S, s.K, s.sigma * 100, s.entry_price, pos.n_contracts)
		for pos in portfolio.short_positions[:3]:
			s = pos.spec
			logger.debug('  PUT  %s S=%.2f K=%.2f σ=%.1f%% entry=$%.2f n=%d', s.ticker, s.S, s.K, s.sigma * 100, s.entry_price, pos.n_contracts)
