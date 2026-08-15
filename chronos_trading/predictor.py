"""ChronosPredictor: wraps Chronos2Pipeline for batch stock return forecasting.

Chronos2 covariate convention (verified from docs):
  - Column in context_df ONLY  → treated as past-only covariate
  - Same column in BOTH context_df AND future_df → treated as known-future covariate
  - No column prefixes needed ('past_' / 'future_' are NOT used)

Multi-horizon trick: run predict_df once with prediction_length = max(HORIZONS),
then slice the output at each horizon index of interest.
"""

from __future__ import annotations

import logging
import pickle
import time
from datetime import date, timedelta

import pandas as pd
from pydantic import BaseModel, ConfigDict

from chronos_trading.config import TradingConfig

logger = logging.getLogger(__name__)


class QuantilePrediction(BaseModel):
	"""Chronos2 quantile forecast for a single (ticker, horizon) pair."""

	model_config = ConfigDict(extra='forbid')

	ticker: str
	prediction_date: date
	horizon: int
	q10: float
	q25: float
	q50: float
	q75: float
	q90: float


class ChronosPredictionResult(BaseModel):
	"""Collection of quantile predictions from one Chronos2 inference run."""

	model_config = ConfigDict(extra='forbid')

	predictions: list[QuantilePrediction]
	model_id: str
	prediction_date: date
	inference_time_seconds: float


class ChronosPredictor:
	"""Wraps Chronos2Pipeline for batch stock return quantile prediction."""

	def __init__(self, config: TradingConfig) -> None:
		self._config = config
		self._pipeline = None
		self._pred_dir = config.data_dir_path / 'predictions'
		self._pred_dir.mkdir(parents=True, exist_ok=True)

	def load_model(self) -> None:
		"""Lazy-load Chronos2 from HuggingFace (downloads once, caches locally)."""
		if self._pipeline is not None:
			return
		import os

		from chronos import Chronos2Pipeline  # type: ignore[import-untyped]

		if self._config.HF_TOKEN:
			os.environ.setdefault('HF_TOKEN', self._config.HF_TOKEN)
			os.environ.setdefault('HUGGING_FACE_HUB_TOKEN', self._config.HF_TOKEN)

		logger.info('[Predictor] Loading %s …', self._config.CHRONOS_MODEL)
		t0 = time.time()
		import torch

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self._pipeline = Chronos2Pipeline.from_pretrained(
			self._config.CHRONOS_MODEL,
			device_map=device,
		)
		logger.info('[Predictor] Model loaded in %.1fs on %s', time.time() - t0, device)

	def predict(
		self,
		price_data: dict[str, pd.DataFrame],
		covariate_data: pd.DataFrame,
		as_of_date: date,
		horizons: list[int] | None = None,
	) -> ChronosPredictionResult:
		"""Run Chronos2 predictions for all tickers in price_data."""
		assert self._pipeline is not None, 'Call load_model() first'
		horizons = horizons or self._config.HORIZONS

		cache_path = self._pred_dir / f'{as_of_date}.pkl'
		if cache_path.exists():
			logger.info('[Predictor] Loading cached predictions for %s', as_of_date)
			with open(cache_path, 'rb') as f:
				return pickle.load(f)

		t0 = time.time()
		tickers = list(price_data.keys())
		all_predictions: list[QuantilePrediction] = []

		batches = [tickers[i : i + self._config.BATCH_SIZE] for i in range(0, len(tickers), self._config.BATCH_SIZE)]
		for batch_idx, batch in enumerate(batches):
			logger.debug('[Predictor] Batch %d/%d (%d tickers)', batch_idx + 1, len(batches), len(batch))
			try:
				preds = self._predict_batch(batch, price_data, covariate_data, as_of_date, horizons)
				all_predictions.extend(preds)
			except Exception as exc:
				logger.warning('[Predictor] Batch %d failed: %s — skipping', batch_idx + 1, exc)

		result = ChronosPredictionResult(
			predictions=all_predictions,
			model_id=self._config.CHRONOS_MODEL,
			prediction_date=as_of_date,
			inference_time_seconds=time.time() - t0,
		)

		with open(cache_path, 'wb') as f:
			pickle.dump(result, f)

		self._log_prediction_stats(result)
		return result

	def _predict_batch(
		self,
		tickers: list[str],
		price_data: dict[str, pd.DataFrame],
		covariate_data: pd.DataFrame,
		as_of_date: date,
		horizons: list[int],
	) -> list[QuantilePrediction]:
		context_df = self._build_context_df(tickers, price_data, covariate_data, as_of_date)
		if context_df.empty:
			return []
		future_df = self._build_future_df(tickers, covariate_data, as_of_date, max(horizons))
		pred_df = self._pipeline.predict_df(
			context_df,
			future_df=future_df if not future_df.empty else None,
			prediction_length=max(horizons),
			quantile_levels=self._config.QUANTILE_LEVELS,
			id_column='id',
			timestamp_column='timestamp',
			target='target',
		)
		return self._parse_predictions(pred_df, tickers, horizons, as_of_date)

	def _build_context_df(
		self,
		tickers: list[str],
		price_data: dict[str, pd.DataFrame],
		covariate_data: pd.DataFrame,
		as_of_date: date,
	) -> pd.DataFrame:
		rows: list[pd.DataFrame] = []
		context_days = self._config.CONTEXT_WINDOW
		for ticker in tickers:
			df = price_data.get(ticker)
			if df is None or df.empty or 'log_return' not in df.columns:
				continue
			df_slice = df[df.index <= as_of_date].tail(context_days).copy()
			if len(df_slice) < 10:
				continue
			df_slice = df_slice.copy()
			df_slice.index = pd.to_datetime(pd.Index([str(d) for d in df_slice.index]))
			cov_aligned = self._align_covariates(covariate_data, df_slice.index)
			frame = pd.DataFrame({'id': ticker, 'timestamp': df_slice.index, 'target': df_slice['log_return'].values})
			past_cov_cols = ['vix', 'spy_r', 'qqq_r'] + [
				f'{e.lower()}_r' for e in ['xlk', 'xlf', 'xle', 'xlv', 'xli', 'xly', 'xlp', 'xlb', 'xlre', 'xlu', 'xlc']
			]
			for col in past_cov_cols:
				if col in cov_aligned.columns:
					frame[col] = cov_aligned[col].values
				else:
					frame[col] = 0.0
			rows.append(frame.dropna(subset=['target']))
		if not rows:
			return pd.DataFrame()
		combined = pd.concat(rows, ignore_index=True)
		combined['timestamp'] = pd.to_datetime(combined['timestamp'])
		combined = combined.sort_values(['id', 'timestamp'])
		return combined

	def _build_future_df(
		self,
		tickers: list[str],
		covariate_data: pd.DataFrame,
		as_of_date: date,
		horizon: int,
	) -> pd.DataFrame:
		business_dates = pd.date_range(
			start=pd.Timestamp(as_of_date) + timedelta(days=1),
			periods=horizon,
			freq='B',
		)
		last_spread = 0.0
		if 'vix3m_spread' in covariate_data.columns:
			cov_idx = pd.to_datetime([str(d) for d in covariate_data.index])
			mask = cov_idx <= pd.Timestamp(as_of_date)
			if mask.any():
				last_spread = float(covariate_data['vix3m_spread'].values[mask][-1])
		rows: list[dict] = []
		for ticker in tickers:
			for ts in business_dates:
				rows.append({'id': ticker, 'timestamp': ts, 'vix3m_spread': last_spread})
		if not rows:
			return pd.DataFrame()
		future_df = pd.DataFrame(rows)
		future_df['timestamp'] = pd.to_datetime(future_df['timestamp'])
		return future_df

	def _parse_predictions(
		self,
		pred_df: pd.DataFrame,
		tickers: list[str],
		horizons: list[int],
		prediction_date: date,
	) -> list[QuantilePrediction]:
		predictions: list[QuantilePrediction] = []
		q_cols = {'q10': '0.1', 'q25': '0.25', 'q50': '0.5', 'q75': '0.75', 'q90': '0.9'}
		available_cols = set(pred_df.columns)
		missing_q = [v for v in q_cols.values() if v not in available_cols]
		if missing_q:
			for k, v in list(q_cols.items()):
				alt = f'quantile_{v}'
				if alt in available_cols:
					q_cols[k] = alt
		for ticker in tickers:
			ticker_df = pred_df[pred_df['id'] == ticker].reset_index(drop=True)
			if ticker_df.empty:
				continue
			for horizon in horizons:
				step_idx = horizon - 1
				if step_idx >= len(ticker_df):
					step_idx = len(ticker_df) - 1
				row = ticker_df.iloc[step_idx]
				try:
					predictions.append(
						QuantilePrediction(
							ticker=ticker,
							prediction_date=prediction_date,
							horizon=horizon,
							q10=float(row[q_cols['q10']]),
							q25=float(row[q_cols['q25']]),
							q50=float(row[q_cols['q50']]),
							q75=float(row[q_cols['q75']]),
							q90=float(row[q_cols['q90']]),
						)
					)
				except (KeyError, ValueError) as exc:
					logger.debug('Failed to parse prediction for %s h=%d: %s', ticker, horizon, exc)
		return predictions

	def _align_covariates(self, covariate_data: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
		if covariate_data.empty:
			return pd.DataFrame(index=target_index)
		cov = covariate_data.copy()
		cov.index = pd.to_datetime([str(d) for d in cov.index])
		aligned = cov.reindex(target_index, method='ffill')
		return aligned.fillna(0.0)

	def _log_prediction_stats(self, result: ChronosPredictionResult) -> None:
		n = len(result.predictions)
		tickers = len({p.ticker for p in result.predictions})
		logger.info(
			'[Predictor] %d predictions for %d tickers in %.1fs (%.0f pred/s)',
			n,
			tickers,
			result.inference_time_seconds,
			n / max(result.inference_time_seconds, 0.001),
		)
