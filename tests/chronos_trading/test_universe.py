"""Tests for UniverseManager."""

from __future__ import annotations

from datetime import date

import pytest

from chronos_trading.config import TradingConfig
from chronos_trading.data.universe import ConstituentRecord, UniverseManager, _parse_date


def test_constituent_record_active_on():
	rec = ConstituentRecord(ticker='AAPL', added_date=date(2015, 1, 1), removed_date=None)
	assert rec.was_active_on(date(2020, 6, 1))
	assert rec.was_active_on(date(2015, 1, 1))


def test_constituent_record_not_active_before_added():
	rec = ConstituentRecord(ticker='AAPL', added_date=date(2015, 1, 1), removed_date=None)
	assert not rec.was_active_on(date(2014, 12, 31))


def test_constituent_record_removed_before_query():
	rec = ConstituentRecord(ticker='XYZ', added_date=date(2010, 1, 1), removed_date=date(2018, 6, 1))
	assert rec.was_active_on(date(2017, 1, 1))
	assert not rec.was_active_on(date(2018, 6, 1))
	assert not rec.was_active_on(date(2019, 1, 1))


def test_constituent_record_is_active_property():
	active = ConstituentRecord(ticker='A', added_date=date(2000, 1, 1), removed_date=None)
	inactive = ConstituentRecord(ticker='B', added_date=date(2000, 1, 1), removed_date=date(2020, 1, 1))
	assert active.is_active
	assert not inactive.is_active


def _manager_with_records(records: list[ConstituentRecord]) -> UniverseManager:
	cfg = TradingConfig()
	mgr = UniverseManager(cfg)
	mgr._records = records
	mgr._loaded = True
	return mgr


def test_get_universe_at_basic():
	records = [
		ConstituentRecord(ticker='AAPL', added_date=date(2015, 1, 1), removed_date=None),
		ConstituentRecord(ticker='GOOG', added_date=date(2017, 1, 1), removed_date=None),
		ConstituentRecord(ticker='XYZ', added_date=date(2010, 1, 1), removed_date=date(2016, 6, 1)),
	]
	mgr = _manager_with_records(records)
	snap = mgr.get_universe_at(date(2018, 1, 1))
	assert 'AAPL' in snap.tickers
	assert 'GOOG' in snap.tickers
	assert 'XYZ' not in snap.tickers


def test_get_universe_at_before_any_added():
	records = [ConstituentRecord(ticker='AAPL', added_date=date(2015, 1, 1), removed_date=None)]
	mgr = _manager_with_records(records)
	snap = mgr.get_universe_at(date(2014, 1, 1))
	assert snap.tickers == []


def test_get_universe_at_deduplicates():
	records = [
		ConstituentRecord(ticker='AAPL', added_date=date(2010, 1, 1), removed_date=date(2015, 1, 1)),
		ConstituentRecord(ticker='AAPL', added_date=date(2016, 1, 1), removed_date=None),
	]
	mgr = _manager_with_records(records)
	snap = mgr.get_universe_at(date(2020, 1, 1))
	assert snap.tickers.count('AAPL') == 1


def test_get_universe_requires_loaded():
	cfg = TradingConfig()
	mgr = UniverseManager(cfg)
	with pytest.raises(AssertionError):
		mgr.get_universe_at(date(2020, 1, 1))


def test_all_tickers_ever():
	records = [
		ConstituentRecord(ticker='AAPL', added_date=date(2015, 1, 1), removed_date=None),
		ConstituentRecord(ticker='XYZ', added_date=date(2010, 1, 1), removed_date=date(2018, 1, 1)),
	]
	mgr = _manager_with_records(records)
	all_t = mgr.all_tickers_ever()
	assert set(all_t) == {'AAPL', 'XYZ'}


@pytest.mark.parametrize(
	'raw,expected',
	[
		('2020-01-15', date(2020, 1, 15)),
		('01/15/2020', date(2020, 1, 15)),
		('2020/01/15', date(2020, 1, 15)),
		('', None),
		('None', None),
		('nan', None),
		('NaT', None),
	],
)
def test_parse_date(raw: str, expected: date | None):
	assert _parse_date(raw) == expected
