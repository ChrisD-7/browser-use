"""Tests for TradingConfig."""

from __future__ import annotations

import pytest

from chronos_trading.config import TradingConfig


def test_defaults_load():
	config = TradingConfig()
	assert config.UNIVERSE in ('midcap', 'smallcap')
	assert config.TOP_N == 5
	assert config.BOTTOM_N == 5
	assert config.TARGET_DELTA == 0.80
	assert config.TARGET_DTE == 60
	assert config.CHRONOS_MODEL == 'amazon/chronos-2'
	assert len(config.HORIZONS) == 5
	assert config.max_horizon == 63


def test_horizons_sorted():
	config = TradingConfig(HORIZONS=[63, 5, 21, 10, 42])
	assert config.HORIZONS == [5, 10, 21, 42, 63]


def test_position_size():
	config = TradingConfig(INITIAL_CAPITAL=1_000_000, TOP_N=5, BOTTOM_N=5)
	assert config.position_size == 100_000.0


def test_invalid_universe():
	with pytest.raises(ValueError, match='UNIVERSE'):
		TradingConfig(UNIVERSE='megacap')


def test_empty_horizons():
	with pytest.raises(ValueError, match='HORIZONS'):
		TradingConfig(HORIZONS=[])


def test_data_dir_created(tmp_path):
	config = TradingConfig(DATA_DIR=str(tmp_path / 'cache'))
	path = config.data_dir_path
	assert path.exists()


def test_results_dir_created(tmp_path):
	config = TradingConfig(RESULTS_DIR=str(tmp_path / 'results'))
	path = config.results_dir_path
	assert path.exists()
