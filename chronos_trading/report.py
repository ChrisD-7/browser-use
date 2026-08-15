"""Reporter: generates charts and HTML report from backtest + risk results.

Outputs (in RESULTS_DIR):
  equity_curve.png         — cumulative wealth with IS/OOS shading + drawdown panel
  monthly_heatmap.png      — calendar heatmap of monthly returns
  var_distribution.png     — return histogram with VaR/CVaR lines
  monte_carlo.png          — fan chart of MC wealth paths
  ranking_stability.png    — consecutive-month rank correlation over time
  report.html              — self-contained HTML with all charts + summary table
  summary_table.csv        — all key metrics in one row
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from chronos_trading.backtest import BacktestResult
from chronos_trading.config import TradingConfig
from chronos_trading.risk import RiskReport

matplotlib.use('Agg')  # non-interactive backend for server environments

logger = logging.getLogger(__name__)

_STYLE = 'seaborn-v0_8-darkgrid'


class Reporter:
	"""Generates all visualisations and the HTML summary report."""

	def __init__(self, config: TradingConfig) -> None:
		self._config = config
		self._out_dir = config.results_dir_path

	# ── Main entry point ────────────────────────────────────────────────

	def generate_report(
		self,
		backtest_result: BacktestResult,
		risk_report: RiskReport,
	) -> Path:
		"""Generate all plots and an HTML report. Returns the report directory path."""
		self._out_dir.mkdir(parents=True, exist_ok=True)

		figs: dict[str, plt.Figure] = {
			'equity_curve': self.plot_equity_curve(backtest_result),
			'monthly_heatmap': self.plot_monthly_returns_heatmap(backtest_result),
			'var_distribution': self.plot_var_distribution(risk_report, backtest_result.monthly_returns),
			'monte_carlo': self.plot_mc_fan_chart(risk_report.monte_carlo, backtest_result),
		}

		# Save PNGs
		for name, fig in figs.items():
			path = self._out_dir / f'{name}.png'
			fig.savefig(path, dpi=150, bbox_inches='tight')
			plt.close(fig)
			logger.info('[Reporter] Saved %s', path)

		# Summary table
		summary_df = self.build_summary_table(backtest_result, risk_report)
		summary_df.to_csv(self._out_dir / 'summary_table.csv', index=False)

		# HTML report
		html_path = self._out_dir / 'report.html'
		self._write_html_report(figs, summary_df, backtest_result, risk_report, html_path)

		self._log_report_path(html_path)
		return self._out_dir

	# ── Individual plots ────────────────────────────────────────────────

	def plot_equity_curve(self, result: BacktestResult) -> plt.Figure:
		"""Cumulative equity curve with IS/OOS split + drawdown panel."""
		try:
			plt.style.use(_STYLE)
		except Exception:
			pass

		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

		eq = result.equity_curve
		dates_dt = pd.to_datetime(eq.index)
		oos_ts = pd.Timestamp(result.oos_start)

		# IS region shading
		is_mask = dates_dt < oos_ts
		oos_mask = ~is_mask

		ax1.plot(dates_dt, eq.values, color='#1f77b4', linewidth=1.5, label='Portfolio')
		if is_mask.any():
			ax1.axvspan(dates_dt[is_mask][0], dates_dt[is_mask][-1], alpha=0.07, color='#1f77b4', label='In-sample')
		if oos_mask.any():
			ax1.axvspan(dates_dt[oos_mask][0], dates_dt[oos_mask][-1], alpha=0.10, color='#ff7f0e', label='Out-of-sample')
		ax1.axvline(oos_ts, color='#ff7f0e', linestyle='--', linewidth=1.0, alpha=0.8)
		ax1.set_ylabel('Cumulative Wealth ($)')
		ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.2f}'))
		ax1.legend(loc='upper left', fontsize=9)
		ax1.set_title('Chronos2 Swing Trading Portfolio — Walk-Forward Backtest')

		# Drawdown panel
		rolling_max = eq.cummax()
		drawdown = (eq - rolling_max) / rolling_max * 100
		ax2.fill_between(dates_dt, drawdown.values, 0, color='#d62728', alpha=0.4)
		ax2.plot(dates_dt, drawdown.values, color='#d62728', linewidth=0.8)
		ax2.set_ylabel('Drawdown (%)')
		ax2.set_xlabel('Date')
		ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

		# Annotations
		ax1.annotate(
			f'IS Sharpe: {result.in_sample_sharpe:.2f}\nOOS Sharpe: {result.oos_sharpe:.2f}\nMax DD: {result.max_drawdown:.1%}',
			xy=(0.02, 0.05),
			xycoords='axes fraction',
			fontsize=9,
			va='bottom',
			bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
		)

		fig.tight_layout()
		return fig

	def plot_monthly_returns_heatmap(self, result: BacktestResult) -> plt.Figure:
		"""Calendar heatmap: rows = years, columns = months."""
		try:
			import seaborn as sns
		except ImportError:
			sns = None

		returns = result.monthly_returns
		df = pd.DataFrame(
			{
				'year': returns.index.year,
				'month': returns.index.month,
				'return': returns.values * 100,
			}
		)
		pivot = df.pivot(index='year', columns='month', values='return')
		pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][: len(pivot.columns)]

		fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
		vabs = max(abs(pivot.values[~np.isnan(pivot.values)].max()), 1.0) if not pivot.empty else 5.0

		if sns is not None:
			sns.heatmap(
				pivot,
				annot=True,
				fmt='.1f',
				center=0,
				cmap='RdYlGn',
				vmin=-vabs,
				vmax=vabs,
				linewidths=0.5,
				ax=ax,
				cbar_kws={'label': 'Monthly Return (%)'},
			)
		else:
			im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-vabs, vmax=vabs, aspect='auto')
			plt.colorbar(im, ax=ax, label='Monthly Return (%)')
			ax.set_xticks(range(len(pivot.columns)))
			ax.set_xticklabels(pivot.columns)
			ax.set_yticks(range(len(pivot.index)))
			ax.set_yticklabels(pivot.index)

		ax.set_title('Monthly Returns Heatmap (%)')
		fig.tight_layout()
		return fig

	def plot_var_distribution(self, risk_report: RiskReport, returns: pd.Series) -> plt.Figure:
		"""Return distribution histogram with VaR/CVaR vertical lines."""
		try:
			plt.style.use(_STYLE)
		except Exception:
			pass

		fig, ax = plt.subplots(figsize=(10, 5))
		ret_pct = returns.dropna() * 100
		ax.hist(ret_pct, bins=40, color='#1f77b4', alpha=0.6, edgecolor='white', linewidth=0.5, label='Monthly returns')

		colours = ['#ff7f0e', '#d62728']
		for var_r, col in zip(risk_report.var_results, colours):
			cl = var_r.confidence
			ax.axvline(
				var_r.historical_var * 100,
				color=col,
				linestyle='--',
				linewidth=1.5,
				label=f'VaR {cl:.0%} hist ({var_r.historical_var:.1%})',
			)
			ax.axvline(var_r.cvar * 100, color=col, linestyle=':', linewidth=1.5, label=f'CVaR {cl:.0%} ({var_r.cvar:.1%})')

		ax.set_xlabel('Monthly Return (%)')
		ax.set_ylabel('Frequency')
		ax.set_title('Monthly Return Distribution with VaR / CVaR')
		ax.legend(fontsize=8)
		fig.tight_layout()
		return fig

	def plot_mc_fan_chart(self, mc, backtest_result: BacktestResult) -> plt.Figure:
		"""Monte Carlo GBM fan chart showing P5/P50/P95 wealth paths."""

		try:
			plt.style.use(_STYLE)
		except Exception:
			pass

		# Re-run MC to get full paths for plotting (low path count for speed)
		is_mask = backtest_result.monthly_returns.index < pd.Timestamp(backtest_result.oos_start)
		is_returns = backtest_result.monthly_returns[is_mask]

		n_plot_paths = 500
		log_rets = np.log(1.0 + is_returns.dropna().values.astype(float))
		mu_ann = float(np.mean(log_rets)) * 12
		sigma_ann = float(np.std(log_rets)) * (12**0.5)
		dt = 1.0 / 252
		import math

		rng = np.random.default_rng(mc.mu_annualized.__hash__() % (2**31))
		Z = rng.standard_normal((n_plot_paths, mc.horizon_days))
		log_step = (mu_ann - 0.5 * sigma_ann**2) * dt + sigma_ann * math.sqrt(dt) * Z
		W = np.exp(np.cumsum(log_step, axis=1))

		p5 = np.percentile(W, 5, axis=0)
		p50 = np.percentile(W, 50, axis=0)
		p95 = np.percentile(W, 95, axis=0)

		fig, ax = plt.subplots(figsize=(11, 5))
		days = np.arange(1, mc.horizon_days + 1)
		ax.fill_between(days, p5, p95, alpha=0.25, color='#1f77b4', label='5th–95th pct band')
		ax.plot(days, p50, color='#1f77b4', linewidth=1.8, label='Median path')
		ax.plot(days, p5, color='#d62728', linewidth=1.0, linestyle='--', label='5th pct (worst-case)')
		ax.plot(days, p95, color='#2ca02c', linewidth=1.0, linestyle='--', label='95th pct (best-case)')
		ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':')

		ax.set_xlabel('Trading Days Forward')
		ax.set_ylabel('Portfolio Value (starting = 1.0)')
		ax.set_title(
			f'Monte Carlo GBM Fan Chart ({mc.n_paths:,} paths, {mc.horizon_days}d horizon)\nP(loss)={mc.prob_loss:.1%}  |  MC VaR-95%={mc.var_95_mc:.2%}'
		)
		ax.legend(fontsize=9)
		fig.tight_layout()
		return fig

	# ── Summary table ─────────────────────────────────────────────────

	def build_summary_table(self, backtest_result: BacktestResult, risk_report: RiskReport) -> pd.DataFrame:
		"""Single-row summary DataFrame with all key metrics."""
		row: dict = {
			'total_return_pct': round(backtest_result.total_return * 100, 2),
			'annualized_return_pct': round(backtest_result.annualized_return * 100, 2),
			'annualized_vol_pct': round(backtest_result.annualized_vol * 100, 2),
			'in_sample_sharpe': round(backtest_result.in_sample_sharpe, 3),
			'oos_sharpe': round(backtest_result.oos_sharpe, 3),
			'max_drawdown_pct': round(backtest_result.max_drawdown * 100, 2),
			'sortino_ratio': round(risk_report.sortino_ratio, 3),
			'calmar_ratio': round(risk_report.calmar_ratio, 3),
			'skewness': round(risk_report.skewness, 3),
			'excess_kurtosis': round(risk_report.excess_kurtosis, 3),
			'in_sample_periods': risk_report.in_sample_periods,
			'oos_periods': risk_report.oos_periods,
			'mc_prob_loss_pct': round(risk_report.monte_carlo.prob_loss * 100, 2),
			'mc_var_95_pct': round(risk_report.monte_carlo.var_95_mc * 100, 2),
			'mc_p50_final': round(risk_report.monte_carlo.percentile_50, 4),
		}
		for var_r in risk_report.var_results:
			cl = int(var_r.confidence * 100)
			row[f'hist_var_{cl}_pct'] = round(var_r.historical_var * 100, 2)
			row[f'cvar_{cl}_pct'] = round(var_r.cvar * 100, 2)
			row[f'param_var_{cl}_pct'] = round(var_r.parametric_var * 100, 2)

		return pd.DataFrame([row])

	# ── HTML report ─────────────────────────────────────────────────

	def _write_html_report(
		self,
		figs: dict[str, plt.Figure],
		summary_df: pd.DataFrame,
		backtest_result: BacktestResult,
		risk_report: RiskReport,
		path: Path,
	) -> None:
		"""Write a self-contained HTML report with embedded PNG images."""

		def _fig_to_b64(fig: plt.Figure) -> str:
			buf = io.BytesIO()
			fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
			buf.seek(0)
			return base64.b64encode(buf.read()).decode()

		img_tags = {
			name: f'<img src="data:image/png;base64,{_fig_to_b64(fig)}" style="max-width:100%;margin:10px 0;">'
			for name, fig in figs.items()
		}

		summary_html = summary_df.T.to_html(header=False, border=0, classes='summary')
		mc = risk_report.monte_carlo
		var_rows = ''.join(
			f'<tr><td>{int(v.confidence * 100)}%</td>'
			f'<td>{v.historical_var:.2%}</td>'
			f'<td>{v.parametric_var:.2%}</td>'
			f'<td>{v.cvar:.2%}</td></tr>'
			for v in risk_report.var_results
		)

		html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Chronos2 Swing Trading — Backtest Report</title>
<style>
  body {{font-family:'Helvetica Neue',sans-serif; background:#f8f9fa; color:#212529; padding:20px;}}
  h1 {{color:#1f3a5f; border-bottom:2px solid #1f3a5f; padding-bottom:6px;}}
  h2 {{color:#2c5f8a; margin-top:30px;}}
  .summary {{width:100%; border-collapse:collapse; font-size:13px;}}
  .summary td {{padding:4px 10px; border:1px solid #dee2e6;}}
  .summary tr:nth-child(even) {{background:#e9ecef;}}
  .var-table {{border-collapse:collapse; font-size:13px; margin-top:8px;}}
  .var-table th, .var-table td {{padding:5px 12px; border:1px solid #dee2e6; text-align:right;}}
  .var-table th {{background:#1f3a5f; color:white;}}
  .mc-box {{background:#e8f4fd; border:1px solid #b8daff; padding:12px; border-radius:4px; margin:10px 0;}}
  img {{border:1px solid #dee2e6; border-radius:4px;}}
</style>
</head><body>
<h1>Chronos2 Swing Trading Portfolio — Backtest Report</h1>
<p><strong>Universe:</strong> {self._config.UNIVERSE} &nbsp;|&nbsp;
   <strong>Model:</strong> {self._config.CHRONOS_MODEL} &nbsp;|&nbsp;
   <strong>Period:</strong> {backtest_result.in_sample_start} → {backtest_result.oos_end} &nbsp;|&nbsp;
   <strong>OOS from:</strong> {backtest_result.oos_start}</p>

<h2>1. Equity Curve</h2>
{img_tags.get('equity_curve', '')}

<h2>2. Performance Summary</h2>
{summary_html}

<h2>3. Monthly Returns Heatmap</h2>
{img_tags.get('monthly_heatmap', '')}

<h2>4. Risk Analysis</h2>
<h3>Value at Risk &amp; CVaR</h3>
<table class="var-table">
  <tr><th>Confidence</th><th>Historical VaR</th><th>Parametric VaR</th><th>CVaR (ES)</th></tr>
  {var_rows}
</table>
{img_tags.get('var_distribution', '')}

<h3>Monte Carlo GBM ({mc.n_paths:,} paths, {mc.horizon_days}d horizon)</h3>
<div class="mc-box">
  <strong>Parameters:</strong> μ={mc.mu_annualized:.1%} ann. &nbsp;|&nbsp; σ={mc.sigma_annualized:.1%} ann.<br>
  <strong>P5 final wealth:</strong> ${mc.percentile_5:.3f} &nbsp;|&nbsp;
  <strong>Median:</strong> ${mc.percentile_50:.3f} &nbsp;|&nbsp;
  <strong>P95:</strong> ${mc.percentile_95:.3f}<br>
  <strong>Probability of loss:</strong> {mc.prob_loss:.1%} &nbsp;|&nbsp;
  <strong>MC VaR-95%:</strong> {mc.var_95_mc:.2%}
</div>
{img_tags.get('monte_carlo', '')}

<h2>5. Distribution Characteristics</h2>
<p>Skewness: <strong>{risk_report.skewness:.3f}</strong> &nbsp;|&nbsp;
   Excess Kurtosis: <strong>{risk_report.excess_kurtosis:.3f}</strong> &nbsp;|&nbsp;
   Sortino: <strong>{risk_report.sortino_ratio:.2f}</strong> &nbsp;|&nbsp;
   Calmar: <strong>{risk_report.calmar_ratio:.2f}</strong></p>

<hr style="margin-top:40px; border:none; border-top:1px solid #dee2e6;">
<p style="font-size:11px;color:#6c757d;">
  Generated by Chronos2 Swing Trading System &nbsp;|&nbsp;
  Model: {self._config.CHRONOS_MODEL} &nbsp;|&nbsp;
  Synthetic options priced via Black-Scholes (IV = HV × {self._config.VRP_MULTIPLIER})
</p>
</body></html>"""

		path.write_text(html, encoding='utf-8')

	# ── Logging ─────────────────────────────────────────────────────

	def _log_report_path(self, path: Path) -> None:
		logger.info('[Reporter] Full report written to %s', path)
