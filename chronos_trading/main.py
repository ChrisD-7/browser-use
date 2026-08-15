"""Entry point for the Chronos2 swing trading pipeline.

Usage:
    python -m chronos_trading.main
    python -m chronos_trading.main --universe smallcap
    python -m chronos_trading.main --start 2018-01-01 --oos 2023-01-01

Runs:
  1. Download/cache survivorship-bias-free universe + price data
  2. Walk-forward backtest with Chronos2 quantile forecasting
  3. VaR/CVaR risk analysis + Monte Carlo simulation
  4. HTML report + PNG charts written to RESULTS_DIR
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s  %(levelname)-8s [%(name)s]  %(message)s',
	handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def run_pipeline(
	universe: str | None = None,
	start: str | None = None,
	oos: str | None = None,
	model: str | None = None,
) -> None:
	"""Full pipeline: data → predict → backtest → risk → report."""
	from chronos_trading.backtest import WalkForwardBacktester
	from chronos_trading.config import TradingConfig
	from chronos_trading.data.downloader import DataDownloader
	from chronos_trading.data.universe import UniverseManager
	from chronos_trading.options import BlackScholesEngine, OptionsPortfolioBuilder
	from chronos_trading.predictor import ChronosPredictor
	from chronos_trading.ranker import StockRanker
	from chronos_trading.report import Reporter
	from chronos_trading.risk import RiskAnalyzer

	# Config: allow CLI overrides
	config = TradingConfig()
	if universe:
		config = TradingConfig(UNIVERSE=universe)
	if start:
		config = TradingConfig(**{**config.model_dump(), 'BACKTEST_START': start})
	if oos:
		config = TradingConfig(**{**config.model_dump(), 'OOS_START': oos})
	if model:
		config = TradingConfig(**{**config.model_dump(), 'CHRONOS_MODEL': model})

	t0 = time.time()
	logger.info('═' * 60)
	logger.info('  Chronos2 Swing Trading Pipeline')
	logger.info('  Universe: %s   Model: %s', config.UNIVERSE, config.CHRONOS_MODEL)
	logger.info('  Period: %s → %s  (OOS from %s)', config.BACKTEST_START, config.BACKTEST_END, config.OOS_START)
	logger.info(
		'  Capital: $%,.0f   Positions: %d calls + %d puts @ %dDTE',
		config.INITIAL_CAPITAL,
		config.TOP_N,
		config.BOTTOM_N,
		config.TARGET_DTE,
	)
	logger.info('═' * 60)

	# ── 1. Universe ───────────────────────────────────────────────────────────────
	universe_mgr = UniverseManager(config)
	await universe_mgr.load_constituent_history()

	# ── 2. Data downloader ────────────────────────────────────────────────────────
	downloader = DataDownloader(config)

	# ── 3. Chronos2 model ─────────────────────────────────────────────────────────
	predictor = ChronosPredictor(config)
	predictor.load_model()

	# ── 4. Ranker ─────────────────────────────────────────────────────────────────
	ranker = StockRanker(config)

	# ── 5. Options engine ─────────────────────────────────────────────────────────
	bs_engine = BlackScholesEngine()
	portfolio_builder = OptionsPortfolioBuilder(config, bs_engine)

	# ── 6. Walk-forward backtest ──────────────────────────────────────────────────
	backtester = WalkForwardBacktester(
		config=config,
		universe_manager=universe_mgr,
		downloader=downloader,
		predictor=predictor,
		ranker=ranker,
		portfolio_builder=portfolio_builder,
	)
	backtest_result = await backtester.run()

	# ── 7. Risk analysis ──────────────────────────────────────────────────────────
	risk_analyzer = RiskAnalyzer(config)
	risk_report = risk_analyzer.analyze(backtest_result)

	# ── 8. Report ─────────────────────────────────────────────────────────────────
	reporter = Reporter(config)
	report_path = reporter.generate_report(backtest_result, risk_report)

	elapsed = time.time() - t0
	logger.info('═' * 60)
	logger.info('  Pipeline complete in %.1fs', elapsed)
	logger.info('  Report: %s', report_path / 'report.html')
	logger.info(
		'  Total return: %.1f%%  |  OOS Sharpe: %.2f  |  Max DD: %.1f%%',
		backtest_result.total_return * 100,
		backtest_result.oos_sharpe,
		backtest_result.max_drawdown * 100,
	)
	logger.info('═' * 60)


def main() -> None:
	parser = argparse.ArgumentParser(description='Chronos2 Swing Trading Portfolio')
	parser.add_argument('--universe', choices=['midcap', 'smallcap'], help='Stock universe')
	parser.add_argument('--start', help='Backtest start date (YYYY-MM-DD)')
	parser.add_argument('--oos', help='Out-of-sample start date (YYYY-MM-DD)')
	parser.add_argument('--model', help='Chronos model ID (default: amazon/chronos-2)')
	args = parser.parse_args()

	asyncio.run(
		run_pipeline(
			universe=args.universe,
			start=args.start,
			oos=args.oos,
			model=args.model,
		)
	)


if __name__ == '__main__':
	main()
