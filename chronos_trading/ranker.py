"""StockRanker: converts Chronos2 quantile predictions into ranked stock universe.

Scoring algorithm per (ticker, horizon):
  direction(h)   = q50                             (raw median log-return)
  conviction(h)  = clip(1 / (q75 - q25), 0, 5)    (inverse IQR — tight = confident)
  asymmetry(h)   = (q90 - q50) - (q50 - q10)      (right-skew is bullish)
  score_h        = 0.5*direction + 0.3*conviction + 0.2*asymmetry

Cross-horizon aggregation uses exponential decay weights (half-life = 21 trading days)
so that near-term predictions carry more weight. Final scores are z-scored cross-sectionally
to produce stable ranks across varying market regimes.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
from pydantic import BaseModel, ConfigDict

from chronos_trading.config import TradingConfig
from chronos_trading.predictor import ChronosPredictionResult, QuantilePrediction

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────────────────────────────


class StockScore(BaseModel):
	"""Composite score for a single ticker on a given prediction date."""

	model_config = ConfigDict(extra='forbid')

	ticker: str
	prediction_date: date
	raw_score: float
	z_score: float
	direction_component: float  # weighted average of q50 across horizons
	conviction_component: float  # weighted average of 1/IQR across horizons
	asymmetry_component: float  # weighted average of skew across horizons
	horizon_scores: dict[int, float]  # horizon (days) → per-horizon composite score


class RankedUniverse(BaseModel):
	"""Full ranked universe for a single rebalance date."""

	model_config = ConfigDict(extra='forbid')

	prediction_date: date
	scores: list[StockScore]  # all tickers, sorted descending by z_score
	long_picks: list[str]  # top N tickers → buy calls
	short_picks: list[str]  # bottom N tickers → buy puts
	long_watch: list[str]  # next 5 after long_picks (informational)
	short_watch: list[str]  # next 5 before short_picks (informational)


# ── Ranker ───────────────────────────────────────────────────────────────────────


class StockRanker:
	"""Converts Chronos2 quantile predictions into a ranked stock universe."""

	# Score component weights — must sum to 1.0
	_W_DIRECTION: float = 0.5
	_W_CONVICTION: float = 0.3
	_W_ASYMMETRY: float = 0.2

	# Conviction clip to prevent division-by-zero inflating the score
	_CONVICTION_CAP: float = 5.0

	# Exponential decay half-life for cross-horizon weighting (trading days)
	_HORIZON_HALF_LIFE: float = 21.0

	def __init__(self, config: TradingConfig) -> None:
		self._config = config

	# ── Public API ────────────────────────────────────────────────────────────────

	def rank(self, predictions: ChronosPredictionResult) -> RankedUniverse:
		"""Compute composite scores and return a fully ranked universe."""
		horizons = self._config.HORIZONS
		horizon_weights = self._compute_horizon_weights(horizons)

		# Group predictions by ticker
		by_ticker: dict[str, list[QuantilePrediction]] = {}
		for pred in predictions.predictions:
			by_ticker.setdefault(pred.ticker, []).append(pred)

		raw_scores: dict[str, float] = {}
		stock_details: dict[str, tuple[float, float, float, dict[int, float]]] = {}

		for ticker, preds in by_ticker.items():
			pred_by_horizon = {p.horizon: p for p in preds}
			horizon_scores: dict[int, float] = {}
			direction_parts: list[float] = []
			conviction_parts: list[float] = []
			asymmetry_parts: list[float] = []

			for h in horizons:
				pred = pred_by_horizon.get(h)
				if pred is None:
					continue
				d = self._direction(pred)
				c = self._conviction(pred)
				a = self._asymmetry(pred)
				h_score = self._W_DIRECTION * d + self._W_CONVICTION * c + self._W_ASYMMETRY * a
				horizon_scores[h] = h_score
				direction_parts.append(d)
				conviction_parts.append(c)
				asymmetry_parts.append(a)

			if not horizon_scores:
				continue

			# Weighted aggregate across horizons
			raw_score = self._aggregate_horizons(horizon_scores, horizons, horizon_weights)
			raw_scores[ticker] = raw_score

			# Store components for reporting (simple mean across horizons)
			stock_details[ticker] = (
				float(np.mean(direction_parts)) if direction_parts else 0.0,
				float(np.mean(conviction_parts)) if conviction_parts else 0.0,
				float(np.mean(asymmetry_parts)) if asymmetry_parts else 0.0,
				horizon_scores,
			)

		if not raw_scores:
			logger.warning('[Ranker] No scores computed — predictions may be empty')
			return RankedUniverse(
				prediction_date=predictions.prediction_date,
				scores=[],
				long_picks=[],
				short_picks=[],
				long_watch=[],
				short_watch=[],
			)

		# Cross-sectional z-score normalisation
		z_scores = self._z_score_universe(raw_scores)

		# Build StockScore objects sorted descending
		scores: list[StockScore] = []
		for ticker, z in z_scores.items():
			d, c, a, h_scores = stock_details[ticker]
			scores.append(
				StockScore(
					ticker=ticker,
					prediction_date=predictions.prediction_date,
					raw_score=raw_scores[ticker],
					z_score=z,
					direction_component=d,
					conviction_component=c,
					asymmetry_component=a,
					horizon_scores=h_scores,
				)
			)

		scores.sort(key=lambda s: s.z_score, reverse=True)

		n_long = self._config.TOP_N
		n_short = self._config.BOTTOM_N

		long_picks = [s.ticker for s in scores[:n_long]]
		long_watch = [s.ticker for s in scores[n_long : n_long + 5]]
		short_picks = [s.ticker for s in scores[-n_short:]]
		short_watch = [s.ticker for s in scores[-(n_short + 5) : -n_short]] if len(scores) > n_short + 5 else []

		ranked = RankedUniverse(
			prediction_date=predictions.prediction_date,
			scores=scores,
			long_picks=long_picks,
			short_picks=short_picks,
			long_watch=long_watch,
			short_watch=short_watch,
		)

		self._log_ranking_summary(ranked)
		return ranked

	# ── Score components ──────────────────────────────────────────────────────────

	def _direction(self, pred: QuantilePrediction) -> float:
		"""Median log-return is the raw directional signal."""
		return pred.q50

	def _conviction(self, pred: QuantilePrediction) -> float:
		"""Inverse of IQR (q75 - q25). Tight distribution = high conviction."""
		iqr = pred.q75 - pred.q25
		if iqr <= 0.0:
			return self._CONVICTION_CAP
		return min(1.0 / iqr, self._CONVICTION_CAP)

	def _asymmetry(self, pred: QuantilePrediction) -> float:
		"""Right-skew (q90-q50) > (q50-q10) → positive = bullish for calls."""
		upside = pred.q90 - pred.q50
		downside = pred.q50 - pred.q10
		return upside - downside

	# ── Aggregation & normalisation ───────────────────────────────────────────────

	def _compute_horizon_weights(self, horizons: list[int]) -> dict[int, float]:
		"""Exponential decay weights: w(h) = exp(-h / half_life), normalised to sum=1."""
		raw = {h: np.exp(-h / self._HORIZON_HALF_LIFE) for h in horizons}
		total = sum(raw.values())
		return {h: w / total for h, w in raw.items()}

	def _aggregate_horizons(
		self,
		horizon_scores: dict[int, float],
		horizons: list[int],
		weights: dict[int, float],
	) -> float:
		"""Weighted average of per-horizon scores."""
		total_w = 0.0
		weighted_sum = 0.0
		for h in horizons:
			if h in horizon_scores and h in weights:
				w = weights[h]
				weighted_sum += w * horizon_scores[h]
				total_w += w
		return weighted_sum / max(total_w, 1e-12)

	def _z_score_universe(self, raw_scores: dict[str, float]) -> dict[str, float]:
		"""Cross-sectional z-score: (score - mean) / std across all tickers."""
		values = np.array(list(raw_scores.values()), dtype=float)
		mu = float(np.mean(values))
		sigma = float(np.std(values))
		if sigma < 1e-12:
			return {t: 0.0 for t in raw_scores}
		return {ticker: (score - mu) / sigma for ticker, score in raw_scores.items()}

	# ── Logging ───────────────────────────────────────────────────────────────────

	def _log_ranking_summary(self, ranked: RankedUniverse) -> None:
		logger.info(
			'[Ranker] %s — %d stocks ranked | long: %s | short: %s',
			ranked.prediction_date,
			len(ranked.scores),
			ranked.long_picks,
			ranked.short_picks,
		)
		if ranked.scores:
			best = ranked.scores[0]
			worst = ranked.scores[-1]
			logger.debug(
				'[Ranker] Best: %s z=%.2f | Worst: %s z=%.2f',
				best.ticker,
				best.z_score,
				worst.ticker,
				worst.z_score,
			)
