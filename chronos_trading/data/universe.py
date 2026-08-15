"""UniverseManager: survivorship-bias-free stock universe construction.

Three-tier data strategy:
  Tier 1 – FMP API: historical S&P 500 constituent changes (additions/removals
            with exact dates) + delisted-companies list.
  Tier 2 – yfinance / Wikipedia: current S&P 400 constituents as a live seed.
  Tier 3 – CSV fallback: minimal hardcoded ticker list for offline mode.

All API responses are persisted as parquet files so they are never re-fetched.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from pathlib import Path

import aiohttp
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from chronos_trading.config import TradingConfig

logger = logging.getLogger(__name__)

_FMP_BASE = 'https://financialmodelingprep.com/api'


class ConstituentRecord(BaseModel):
	"""Single ticker's membership window in an index."""

	model_config = ConfigDict(extra='forbid')

	ticker: str
	name: str | None = None
	sector: str | None = None
	added_date: date
	removed_date: date | None = None

	@property
	def is_active(self) -> bool:
		return self.removed_date is None

	def was_active_on(self, as_of: date) -> bool:
		return self.added_date <= as_of and (self.removed_date is None or self.removed_date > as_of)


class UniverseSnapshot(BaseModel):
	"""Active universe at a given point in time."""

	model_config = ConfigDict(extra='forbid')

	as_of_date: date
	tickers: list[str]
	source: str = Field(default='constructed')

	def __len__(self) -> int:
		return len(self.tickers)


class UniverseManager:
	"""Builds and caches a survivorship-bias-free stock universe."""

	_SP400_WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'

	def __init__(self, config: TradingConfig) -> None:
		self._config = config
		self._records: list[ConstituentRecord] = []
		self._loaded = False

	async def load_constituent_history(self) -> None:
		"""Download and cache constituent history. Must be called once before get_universe_at()."""
		cache_path = self._config.data_dir_path / 'constituents.parquet'
		if cache_path.exists():
			self._records = self._load_parquet(cache_path)
			self._loaded = True
			self._log_load_summary('cache', len(self._records))
			return

		records: list[ConstituentRecord] = []

		# Tier 1: FMP
		if self._config.FMP_API_KEY:
			try:
				records = await self._download_fmp_history()
				self._log_load_summary('FMP', len(records))
			except Exception as exc:
				logger.warning('FMP constituent download failed: %s — falling back', exc)

		# Tier 2: yfinance Wikipedia scrape
		if not records:
			try:
				records = await asyncio.get_event_loop().run_in_executor(None, self._load_yfinance_current)
				self._log_load_summary('yfinance', len(records))
			except Exception as exc:
				logger.warning('yfinance constituent load failed: %s — using CSV fallback', exc)

		# Tier 3: CSV fallback
		if not records:
			records = self._load_csv_fallback()
			self._log_load_summary('CSV fallback', len(records))

		self._records = records
		self._save_parquet(records, cache_path)
		self._loaded = True

	def get_universe_at(self, as_of: date) -> UniverseSnapshot:
		"""Return active tickers for the given historical date."""
		assert self._loaded, 'Call load_constituent_history() first'
		active = [r.ticker for r in self._records if r.was_active_on(as_of)]
		seen: set[str] = set()
		tickers: list[str] = []
		for t in active:
			if t not in seen:
				seen.add(t)
				tickers.append(t)
		snap = UniverseSnapshot(as_of_date=as_of, tickers=tickers)
		self._log_snapshot(snap)
		return snap

	def all_tickers_ever(self) -> list[str]:
		"""All tickers that appeared in the universe at any point."""
		assert self._loaded, 'Call load_constituent_history() first'
		return list({r.ticker for r in self._records})

	async def _download_fmp_history(self) -> list[ConstituentRecord]:
		records: list[ConstituentRecord] = []
		async with aiohttp.ClientSession() as session:
			sp500 = await self._fmp_get(session, '/v3/historical/sp500_constituent')
			records.extend(self._parse_fmp_sp500_history(sp500))
			delisted = await self._fmp_get_paginated(session, '/v3/delisted-companies', page_size=1000)
			records.extend(self._parse_fmp_delisted(delisted, existing=records))
		return records

	async def _fmp_get(self, session: aiohttp.ClientSession, path: str, params: dict | None = None) -> list:
		params = params or {}
		params['apikey'] = self._config.FMP_API_KEY
		url = f'{_FMP_BASE}{path}'
		async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
			resp.raise_for_status()
			data = await resp.json(content_type=None)
			return data if isinstance(data, list) else data.get('historicalStockList', data.get('data', []))

	async def _fmp_get_paginated(self, session: aiohttp.ClientSession, path: str, page_size: int = 100) -> list:
		all_items: list = []
		page = 0
		while True:
			items = await self._fmp_get(session, path, params={'limit': page_size, 'page': page})
			if not items:
				break
			all_items.extend(items)
			if len(items) < page_size:
				break
			page += 1
		return all_items

	def _parse_fmp_sp500_history(self, data: list) -> list[ConstituentRecord]:
		records: list[ConstituentRecord] = []
		for row in data:
			ticker = row.get('symbol', '')
			if not ticker:
				continue
			added_raw = row.get('dateAdded') or row.get('date') or ''
			removed_raw = row.get('removedDate') or row.get('dateRemoved') or ''
			try:
				added = _parse_date(added_raw) or date(2000, 1, 1)
				removed = _parse_date(removed_raw)
				records.append(
					ConstituentRecord(
						ticker=ticker.upper().strip(),
						name=row.get('name') or row.get('company'),
						sector=row.get('sector') or row.get('gicsSubIndustry'),
						added_date=added,
						removed_date=removed,
					)
				)
			except Exception as exc:
				logger.debug('Skipping FMP row %s: %s', row, exc)
		return records

	def _parse_fmp_delisted(self, data: list, existing: list[ConstituentRecord]) -> list[ConstituentRecord]:
		existing_tickers = {r.ticker for r in existing}
		records: list[ConstituentRecord] = []
		for row in data:
			ticker = (row.get('symbol') or '').upper().strip()
			if not ticker or ticker in existing_tickers:
				continue
			ipo_raw = row.get('ipoDate') or ''
			delist_raw = row.get('delistedDate') or ''
			try:
				added = _parse_date(ipo_raw) or date(2000, 1, 1)
				removed = _parse_date(delist_raw)
				records.append(
					ConstituentRecord(ticker=ticker, name=row.get('companyName'), added_date=added, removed_date=removed)
				)
			except Exception as exc:
				logger.debug('Skipping delisted row %s: %s', row, exc)
		return records

	def _load_yfinance_current(self) -> list[ConstituentRecord]:
		tables = pd.read_html(self._SP400_WIKI_URL)
		df = tables[0]
		df.columns = [str(c).lower().strip() for c in df.columns]
		ticker_col = next((c for c in df.columns if 'ticker' in c or 'symbol' in c), df.columns[0])
		name_col = next((c for c in df.columns if 'name' in c or 'company' in c), None)
		sector_col = next((c for c in df.columns if 'sector' in c or 'gics' in c), None)
		records: list[ConstituentRecord] = []
		for _, row in df.iterrows():
			ticker = str(row[ticker_col]).upper().strip()
			if not ticker or ticker == 'NAN':
				continue
			records.append(
				ConstituentRecord(
					ticker=ticker,
					name=str(row[name_col]) if name_col else None,
					sector=str(row[sector_col]) if sector_col else None,
					added_date=date(2000, 1, 1),
					removed_date=None,
				)
			)
		return records

	def _load_csv_fallback(self) -> list[ConstituentRecord]:
		fallback_tickers = [
			'AGNC','AIRC','ALE','ALKS','ALIT','ALRM','AMG','AMKR','APAM','APG',
			'ARWR','ASH','ATR','ATUS','AVT','BCPC','BDN','BECN','BJ','BLKB',
			'BNL','BOX','BRKL','BURL','BWA','CABO','CAKE','CATY','CBU','CC',
			'CCCS','CCOI','CEIX','CENTA','CENT','CFR','CHRW','CIEN','CIR','CLH',
			'CLW','CMC','CNK','COHU','COLM','CRI','CRL','CRS','CSL','CSWI',
			'DAN','DDS','DIOD','DLB','DLX','DORM','DRH','EAT','EFC','EGP',
			'EME','ENVA','EPRT','ESAB','ETSY','EXEL','EXLS','EXPO','FELE','FFIN',
			'FIVN','FN','FR','FRSH','GFF','GHC','GKOS','GMS','GOLF','GPI',
			'GXO','HAE','HBI','HCC','HI','HLNE','HLX','HRI','HUBG','IBP',
			'IDCC','IESC','INN','INSP','ITCI','ITT','JACK','JBT','JHG','JXN',
			'KBH','KFY','KNF','KNSL','KRG','LGND','LNN','LNTH','LPLA','LRN',
			'LSCC','LSI','LSTR','MARA','MBUU','MCY','MDU','MGPI','MHO','MKSI',
			'MLI','MMSI','MMS','MOD','MTH','NARI','NBR','NCNO','NEU','NHI',
			'NNI','NOVT','NRC','NSA','NVT','NXST',
		]
		return [ConstituentRecord(ticker=t, added_date=date(2000, 1, 1), removed_date=None) for t in fallback_tickers]

	def _save_parquet(self, records: list[ConstituentRecord], path: Path) -> None:
		if not records:
			return
		df = pd.DataFrame([r.model_dump() for r in records])
		df.to_parquet(path, index=False)

	def _load_parquet(self, path: Path) -> list[ConstituentRecord]:
		df = pd.read_parquet(path)
		records: list[ConstituentRecord] = []
		for _, row in df.iterrows():
			try:
				records.append(ConstituentRecord(**row.to_dict()))
			except Exception:
				pass
		return records

	def _log_load_summary(self, source: str, n: int) -> None:
		logger.info('[UniverseManager] Loaded %d constituent records from %s', n, source)

	def _log_snapshot(self, snap: UniverseSnapshot) -> None:
		logger.debug('[UniverseManager] Universe at %s: %d tickers', snap.as_of_date, len(snap))


def _parse_date(raw: str | None) -> date | None:
	if not raw or str(raw).strip() in ('', 'None', 'NaT', 'nan'):
		return None
	for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y'):
		try:
			return datetime.strptime(str(raw).strip(), fmt).date()
		except ValueError:
			continue
	return None
