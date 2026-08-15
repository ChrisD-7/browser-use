"""RiskAnalyzer: VaR, CVaR, and Monte Carlo risk analysis."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import kurtosis, norm, skew

from chronos_trading.backtest import BacktestResult
from chronos_trading.config import TradingConfig

logger = logging.getLogger(__name__)


class VaRResult(BaseModel):
	model_config = ConfigDict(extra='forbid')
	confidence: float
	historical_var: float
	parametric_var: float
	cvar: float


class MonteCarloResult(BaseModel):
	model_config = ConfigDict(extra='forbid')
	n_paths: int
	horizon_days: int
	initial_wealth: float
	percentile_5: float
	percentile_50: float
	percentile_95: float
	prob_loss: float
	var_95_mc: float
	mu_annualized: float
	sigma_annualized: float


class RiskReport(BaseModel):
	model_config = ConfigDict(extra='forbid')
	var_results: list[VaRResult]
	monte_carlo: MonteCarloResult
	skewness: float
	excess_kurtosis: float
	calmar_ratio: float
	sortino_ratio: float
	in_sample_periods: int
	oos_periods: int


class RiskAnalyzer:
	"""Computes VaR, CVaR, and Monte Carlo metrics from backtest results."""

	def __init__(self, config: TradingConfig) -> None:
		self._config = config

	def analyze(self, backtest_result: BacktestResult) -> RiskReport:
		returns = backtest_result.monthly_returns.dropna()
		oos_mask = returns.index >= pd.Timestamp(backtest_result.oos_start)
		is_returns = returns[~oos_mask]

		var_results: list[VaRResult] = []
		for confidence in self._config.VAR_CONFIDENCE_LEVELS:
			var_results.append(
				VaRResult(
					confidence=confidence,
					historical_var=self.historical_var(returns, confidence),
					parametric_var=self.parametric_var(returns, confidence),
					cvar=self.cvar(returns, confidence),
				)
			)

		mc = self.monte_carlo_gbm(
			returns=is_returns,
			n_paths=self._config.MC_PATHS,
			horizon_days=self._config.MC_HORIZON_DAYS,
			initial_wealth=1.0,
			seed=self._config.MC_SEED,
		)

		ret_vals = returns.values.astype(float)
		skewness = float(skew(ret_vals))
		excess_kurt = float(kurtosis(ret_vals))

		calmar = (
			backtest_result.annualized_return / max(backtest_result.max_drawdown, 1e-6)
			if backtest_result.max_drawdown > 0
			else 0.0
		)
		sortino = self.sortino_ratio(returns)

		report = RiskReport(
			var_results=var_results,
			monte_carlo=mc,
			skewness=skewness,
			excess_kurtosis=excess_kurt,
			calmar_ratio=calmar,
			sortino_ratio=sortino,
			in_sample_periods=int((~oos_mask).sum()),
			oos_periods=int(oos_mask.sum()),
		)
		self._log_risk_summary(report, backtest_result)
		return report

	@staticmethod
	def historical_var(returns: pd.Series, confidence: float) -> float:
		sorted_r = np.sort(returns.dropna().values)
		idx = int(math.floor(len(sorted_r) * (1.0 - confidence)))
		idx = max(0, min(idx, len(sorted_r) - 1))
		return float(sorted_r[idx])

	@staticmethod
	def cvar(returns: pd.Series, confidence: float) -> float:
		var_threshold = RiskAnalyzer.historical_var(returns, confidence)
		losses = returns[returns <= var_threshold]
		if losses.empty:
			return var_threshold
		return float(losses.mean())

	@staticmethod
	def parametric_var(returns: pd.Series, confidence: float) -> float:
		mu = float(returns.mean())
		sigma = float(returns.std())
		z_alpha = norm.ppf(1.0 - confidence)
		return mu + z_alpha * sigma

	@staticmethod
	def monte_carlo_gbm(
		returns: pd.Series,
		n_paths: int = 10_000,
		horizon_days: int = 252,
		initial_wealth: float = 1.0,
		seed: int = 42,
	) -> MonteCarloResult:
		"""GBM Monte Carlo using annualised parameters estimated from monthly returns."""
		log_rets = np.log(1.0 + returns.dropna().values.astype(float))
		mu_monthly = float(np.mean(log_rets))
		sigma_monthly = float(np.std(log_rets))
		mu_ann = mu_monthly * 12.0
		sigma_ann = sigma_monthly * math.sqrt(12.0)
		dt = 1.0 / 252.0
		rng = np.random.default_rng(seed)
		Z = rng.standard_normal((n_paths, horizon_days))
		log_returns_daily = (mu_ann - 0.5 * sigma_ann**2) * dt + sigma_ann * math.sqrt(dt) * Z
		W = initial_wealth * np.exp(np.cumsum(log_returns_daily, axis=1))
		final_wealth = W[:, -1]
		p5 = float(np.percentile(final_wealth, 5))
		p50 = float(np.percentile(final_wealth, 50))
		p95 = float(np.percentile(final_wealth, 95))
		prob_loss = float(np.mean(final_wealth < initial_wealth))
		var_95 = p5 - initial_wealth
		return MonteCarloResult(
			n_paths=n_paths, horizon_days=horizon_days, initial_wealth=initial_wealth,
			percentile_5=p5, percentile_50=p50, percentile_95=p95,
			prob_loss=prob_loss, var_95_mc=var_95,
			mu_annualized=mu_ann, sigma_annualized=sigma_ann,
		)

	@staticmethod
	def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
		clean = returns.dropna()
		if len(clean) < 3:
			return 0.0
		rf_monthly = risk_free / 12.0
		excess = clean - rf_monthly
		downside = clean[clean < rf_monthly]
		if len(downside) == 0:
			return float(excess.mean()) * math.sqrt(12.0) * 100 if excess.mean() > 0 else 0.0
		semi_deviation = float(np.sqrt((downside.values**2).mean()))
		if semi_deviation < 1e-12:
			return float(excess.mean()) * math.sqrt(12.0) * 100 if excess.mean() > 0 else 0.0
		return float(excess.mean()) / semi_deviation * math.sqrt(12.0)

	def _log_risk_summary(self, report: RiskReport, backtest: BacktestResult) -> None:
		logger.info('━' * 55 + ' Risk Summary ' + '━' * 10)
		logger.info('  Total return: %.1f%%  Ann. return: %.1f%%  Vol: %.1f%%',
			backtest.total_return * 100, backtest.annualized_return * 100, backtest.annualized_vol * 100)
		logger.info('  IS Sharpe: %.2f  OOS Sharpe: %.2f  Max DD: %.1f%%',
			backtest.in_sample_sharpe, backtest.oos_sharpe, backtest.max_drawdown * 100)
		logger.info('  Sortino: %.2f  Calmar: %.2f  Skew: %.2f  Ex.Kurt: %.2f',
			report.sortino_ratio, report.calmar_ratio, report.skewness, report.excess_kurtosis)
		for var_r in report.var_results:
			logger.info('  VaR %.0f%%: hist=%.2f%%  param=%.2f%%  CVaR=%.2f%%',
				var_r.confidence * 100, var_r.historical_var * 100, var_r.parametric_var * 100, var_r.cvar * 100)
		mc = report.monte_carlo
		logger.info('  MC (%dk paths, %dd): P5=$%.3f  P50=$%.3f  P95=$%.3f  P(loss)=%.1f%%',
			mc.n_paths // 1000, mc.horizon_days, mc.percentile_5, mc.percentile_50, mc.percentile_95, mc.prob_loss * 100)
