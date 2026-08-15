"""DataDownloader: downloads OHLCV price data and market covariates.

Primary source: yfinance (active stocks, free, unlimited batch downloads).
Secondary source: FMP historical-price-full endpoint (delisted tickers, 250 calls/day).
Tertiary source: EODHD (supplemental, 20 calls/day, 1yr data cap on free tier).

All data is persisted to parquet files in DATA_DIR so incremental updates only
download the missing date range on subsequent runs.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd

from chronos_trading.config import TradingConfig

logger = logging.getLogger(__name__)

_FMP_BASE = 'https://financialmodelingprep.com/api'
_EODHD_BASE = 'https://eodhd.com/api'

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU', 'XLC']
COVARIATE_TICKERS = ['^VIX', 'SPY', 'QQQ', '^IRX'] + SECTOR_ETFS


class DataDownloader:
	"""Downloads and caches price data and covariates for the trading pipeline."""

	def __init__(self, config: TradingConfig) -> None:
		self._config = config
		self._price_dir = config.data_dir_path / 'prices'
		self._cov_path = config.data_dir_path / 'covariates.parquet'
		self._tbill_path = config.data_dir_path / 'tbill.parquet'
		self._price_dir.mkdir(parents=True, exist_ok=True)

	async def download_prices(
		self,
		tickers: list[str],
		start: date,
		end: date,
		batch_size: int = 100,
	) -> dict[str, pd.DataFrame]:
		"""Return per-ticker OHLCV DataFrames (adj_close + log_return computed).

		Uses parquet cache; only downloads missing date ranges.
		Tries yfinance first, then FMP for delisted tickers that yfinance misses.
		"""
		result: dict[str, pd.DataFrame] = {}
		missing_tickers: list[str] = []

		for ticker in tickers:
			cached = self._load_price_cache(ticker, start, end)
			if cached is not None:
				result[ticker] = cached
			else:
				missing_tickers.append(ticker)

		if missing_tickers:
			batches = [missing_tickers[i : i + batch_size] for i in range(0, len(missing_tickers), batch_size)]
			for batch in batches:
				downloaded = await asyncio.get_event_loop().run_in_executor(None, self._yf_download_batch, batch, start, end)
				result.update(downloaded)

			still_missing = [t for t in missing_tickers if t not in result or result[t].empty]
			if still_missing and self._config.FMP_API_KEY:
				fmp_data = await self._fmp_download_batch(still_missing, start, end)
				result.update(fmp_data)

		for ticker, df in list(result.items()):
			if df is not None and not df.empty:
				df = _compute_log_return(df)
				result[ticker] = df
				self._save_price_cache(ticker, df)
			else:
				del result[ticker]

		self._log_download_summary(tickers, list(result.keys()))
		return result

	async def download_covariates(self, start: date, end: date) -> pd.DataFrame:
		"""Download VIX, SPY, QQQ, sector ETFs, and T-bill rate."""
		if self._cov_path.exists():
			cached = pd.read_parquet(self._cov_path)
			cached.index = pd.to_datetime(cached.index).date
			if _date_series_covers(cached, start, end):
				return cached

		cov_raw = await asyncio.get_event_loop().run_in_executor(None, self._yf_download_covariates, start, end)
		self._cov_path.parent.mkdir(parents=True, exist_ok=True)
		cov_raw.to_parquet(self._cov_path)
		return cov_raw

	async def download_risk_free_rates(self, start: date, end: date) -> pd.Series:
		"""Return date-indexed T-bill rate series (annualised decimal)."""
		cov = await self.download_covariates(start, end)
		if 'tbill_r' in cov.columns:
			return cov['tbill_r']
		idx = pd.date_range(start, end, freq='B')
		return pd.Series(0.04, index=idx, name='tbill_r')

	def _yf_download_batch(self, tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
		import yfinance as yf

		try:
			raw = yf.download(
				tickers,
				start=str(start),
				end=str(end + timedelta(days=1)),
				auto_adjust=True,
				progress=False,
				threads=True,
				group_by='ticker',
			)
		except Exception as exc:
			logger.warning('yfinance batch download failed: %s', exc)
			return {}

		result: dict[str, pd.DataFrame] = {}
		if len(tickers) == 1:
			ticker = tickers[0]
			if not raw.empty:
				df = raw.copy()
				df.columns = [str(c).lower() for c in df.columns]
				df.index = pd.to_datetime(df.index).date
				df = df.rename(columns={'close': 'adj_close', 'volume': 'volume'})
				if 'adj_close' in df.columns:
					result[ticker] = df[['adj_close', 'volume']].dropna()
		else:
			for ticker in tickers:
				try:
					if isinstance(raw.columns, pd.MultiIndex):
						df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
					else:
						df = pd.DataFrame()
					if df.empty:
						continue
					df.columns = [str(c).lower() for c in df.columns]
					df.index = pd.to_datetime(df.index).date
					df = df.rename(columns={'close': 'adj_close'})
					if 'adj_close' in df.columns:
						result[ticker] = df[['adj_close', 'volume']].dropna()
				except Exception as exc:
					logger.debug('Failed to parse yfinance data for %s: %s', ticker, exc)
		return result

	def _yf_download_covariates(self, start: date, end: date) -> pd.DataFrame:
		import yfinance as yf

		raw = yf.download(
			COVARIATE_TICKERS,
			start=str(start),
			end=str(end + timedelta(days=1)),
			auto_adjust=True,
			progress=False,
			threads=True,
			group_by='ticker',
		)

		rows: dict[date, dict] = {}
		dates_index: list[date] = [d.date() for d in pd.to_datetime(raw.index)]

		def _get_close(ticker: str) -> pd.Series:
			try:
				if isinstance(raw.columns, pd.MultiIndex):
					return raw[ticker]['Close']
				return pd.Series(dtype=float)
			except Exception:
				return pd.Series(dtype=float)

		vix_s = _get_close('^VIX')
		spy_s = _get_close('SPY')
		qqq_s = _get_close('QQQ')
		tbill_s = _get_close('^IRX')
		sector_series: dict[str, pd.Series] = {etf.lower(): _get_close(etf) for etf in SECTOR_ETFS}

		for i, dt in enumerate(dates_index):
			row: dict[str, float] = {}
			row['vix'] = float(vix_s.iloc[i]) if i < len(vix_s) and not pd.isna(vix_s.iloc[i]) else np.nan
			spy_price = float(spy_s.iloc[i]) if i < len(spy_s) and not pd.isna(spy_s.iloc[i]) else np.nan
			spy_prev = float(spy_s.iloc[i - 1]) if i > 0 and not pd.isna(spy_s.iloc[i - 1]) else np.nan
			row['spy_r'] = np.log(spy_price / spy_prev) if spy_price and spy_prev else np.nan
			qqq_price = float(qqq_s.iloc[i]) if i < len(qqq_s) and not pd.isna(qqq_s.iloc[i]) else np.nan
			qqq_prev = float(qqq_s.iloc[i - 1]) if i > 0 and not pd.isna(qqq_s.iloc[i - 1]) else np.nan
			row['qqq_r'] = np.log(qqq_price / qqq_prev) if qqq_price and qqq_prev else np.nan
			tbill_val = float(tbill_s.iloc[i]) if i < len(tbill_s) and not pd.isna(tbill_s.iloc[i]) else np.nan
			row['tbill_r'] = tbill_val / 100.0 if not np.isnan(tbill_val) else 0.04
			for etf, s in sector_series.items():
				p = float(s.iloc[i]) if i < len(s) and not pd.isna(s.iloc[i]) else np.nan
				p_prev = float(s.iloc[i - 1]) if i > 0 and not pd.isna(s.iloc[i - 1]) else np.nan
				row[f'{etf}_r'] = np.log(p / p_prev) if p and p_prev else np.nan
			row['vix3m_spread'] = row['vix'] * 0.05 if not np.isnan(row.get('vix', np.nan)) else 0.0
			rows[dt] = row

		df = pd.DataFrame.from_dict(rows, orient='index')
		df.index.name = 'date'
		return df.ffill().bfill()

	async def _fmp_download_batch(self, tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
		result: dict[str, pd.DataFrame] = {}
		api_key = self._config.FMP_API_KEY
		if not api_key:
			return result
		ranges = _split_date_range(start, end, years=5)
		async with aiohttp.ClientSession() as session:
			for ticker in tickers:
				frames: list[pd.DataFrame] = []
				for r_start, r_end in ranges:
					try:
						url = f'{_FMP_BASE}/v3/historical-price-full/{ticker}?from={r_start}&to={r_end}&apikey={api_key}'
						async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
							if resp.status != 200:
								continue
							data = await resp.json(content_type=None)
							hist = data.get('historical', []) if isinstance(data, dict) else []
							if hist:
								df = pd.DataFrame(hist)
								df['date'] = pd.to_datetime(df['date']).dt.date
								df = df.set_index('date').sort_index()
								adj_col = next((c for c in ['adjClose', 'adjclose', 'close'] if c in df.columns), None)
								if adj_col:
									frames.append(df[[adj_col, 'volume']].rename(columns={adj_col: 'adj_close'}))
							await asyncio.sleep(0.25)
					except Exception as exc:
						logger.debug('FMP download failed for %s range %s-%s: %s', ticker, r_start, r_end, exc)
				if frames:
					combined = pd.concat(frames).sort_index()
					combined = combined[~combined.index.duplicated(keep='last')]
					combined = combined.loc[start:end]
					result[ticker] = combined.dropna()
		return result

	def _cache_path(self, ticker: str) -> Path:
		safe = ticker.replace('^', 'idx_').replace('/', '_')
		return self._price_dir / f'{safe}.parquet'

	def _load_price_cache(self, ticker: str, start: date, end: date) -> pd.DataFrame | None:
		path = self._cache_path(ticker)
		if not path.exists():
			return None
		try:
			df = pd.read_parquet(path)
			df.index = pd.to_datetime(df.index).date
			if _date_series_covers(df, start, end):
				return df.loc[start:end]
		except Exception:
			pass
		return None

	def _save_price_cache(self, ticker: str, df: pd.DataFrame) -> None:
		path = self._cache_path(ticker)
		try:
			if path.exists():
				existing = pd.read_parquet(path)
				existing.index = pd.to_datetime(existing.index).date
				combined = pd.concat([existing, df])
				combined = combined[~combined.index.duplicated(keep='last')].sort_index()
				combined.to_parquet(path)
			else:
				df.to_parquet(path)
		except Exception as exc:
			logger.warning('Failed to save cache for %s: %s', ticker, exc)

	def _log_download_summary(self, requested: list[str], received: list[str]) -> None:
		n_miss = len(requested) - len(received)
		pct_miss = 100 * n_miss / max(len(requested), 1)
		if n_miss > 0:
			logger.warning('[Downloader] Missing data for %d/%d tickers (%.1f%%)', n_miss, len(requested), pct_miss)
			if pct_miss > 15:
				logger.error(
					'[Downloader] >15%% of universe has missing price data — '
					'backtest results may be compromised by survivorship bias!'
				)
		else:
			logger.info('[Downloader] Downloaded %d tickers successfully', len(received))


def _compute_log_return(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df['log_return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
	return df


def _date_series_covers(df: pd.DataFrame, start: date, end: date) -> bool:
	if df.empty:
		return False
	idx = df.index
	return pd.Timestamp(idx[0]) <= pd.Timestamp(start) and pd.Timestamp(idx[-1]) >= pd.Timestamp(end - timedelta(days=5))


def _split_date_range(start: date, end: date, years: int = 5) -> list[tuple[str, str]]:
	ranges: list[tuple[str, str]] = []
	current = start
	while current < end:
		chunk_end = min(date(current.year + years, current.month, current.day), end)
		ranges.append((str(current), str(chunk_end)))
		current = chunk_end + timedelta(days=1)
	return ranges
