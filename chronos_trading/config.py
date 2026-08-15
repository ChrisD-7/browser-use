"""TradingConfig: central configuration for the Chronos2 swing trading system."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingConfig(BaseSettings):
	"""All configuration for the trading pipeline.

	Loaded from environment variables (or .env at repo root).
	Every field has a sensible default so the system runs without any keys
	using yfinance-only mode.
	"""

	model_config = SettingsConfigDict(
		env_file='.env',
		env_file_encoding='utf-8',
		case_sensitive=True,
		extra='allow',
	)

	# ── API keys (optional – fall back to free sources if absent) ──────────────
	FMP_API_KEY: str | None = Field(default=None, description='Financial Modeling Prep API key')
	EODHD_API_KEY: str | None = Field(default=None, description='EODHD API key (supplemental only)')
	HF_TOKEN: str | None = Field(default=None, description='HuggingFace token for model download')

	# ── Universe ────────────────────────────────────────────────────────────────
	UNIVERSE: str = Field(default='midcap')
	MARKET_CAP_MIN_B: float = Field(default=2.0)
	MARKET_CAP_MAX_B: float = Field(default=10.0)

	# ── Backtest window ─────────────────────────────────────────────────────────
	BACKTEST_START: str = Field(default='2014-01-01')
	BACKTEST_END: str = Field(default='2024-01-01')
	OOS_START: str = Field(default='2022-01-01')
	WARMUP_YEARS: int = Field(default=2)

	# ── Forecasting ─────────────────────────────────────────────────────────────
	HORIZONS: list[int] = Field(default=[5, 10, 21, 42, 63])
	QUANTILE_LEVELS: list[float] = Field(default=[0.1, 0.25, 0.5, 0.75, 0.9])
	CHRONOS_MODEL: str = Field(default='amazon/chronos-2')
	CONTEXT_WINDOW: int = Field(default=200)
	BATCH_SIZE: int = Field(default=40)

	# ── Portfolio ────────────────────────────────────────────────────────────────
	TOP_N: int = Field(default=5)
	BOTTOM_N: int = Field(default=5)
	TARGET_DELTA: float = Field(default=0.80)
	TARGET_DTE: int = Field(default=60)
	HOLD_DAYS: int = Field(default=21)
	INITIAL_CAPITAL: float = Field(default=1_000_000.0)
	ROUND_TRIP_COST: float = Field(default=0.02)
	VRP_MULTIPLIER: float = Field(default=1.15)

	# ── Risk analysis ────────────────────────────────────────────────────────────
	VAR_CONFIDENCE_LEVELS: list[float] = Field(default=[0.95, 0.99])
	MC_PATHS: int = Field(default=10_000)
	MC_HORIZON_DAYS: int = Field(default=252)
	MC_SEED: int = Field(default=42)

	# ── Paths ────────────────────────────────────────────────────────────────────
	DATA_DIR: str = Field(default='./data_cache')
	RESULTS_DIR: str = Field(default='./results')

	@field_validator('UNIVERSE')
	@classmethod
	def validate_universe(cls, v: str) -> str:
		allowed = {'midcap', 'smallcap'}
		if v not in allowed:
			raise ValueError(f'UNIVERSE must be one of {allowed}, got {v!r}')
		return v

	@field_validator('HORIZONS')
	@classmethod
	def validate_horizons(cls, v: list[int]) -> list[int]:
		if not v:
			raise ValueError('HORIZONS must be non-empty')
		return sorted(v)

	@property
	def data_dir_path(self) -> Path:
		p = Path(self.DATA_DIR)
		p.mkdir(parents=True, exist_ok=True)
		return p

	@property
	def results_dir_path(self) -> Path:
		p = Path(self.RESULTS_DIR)
		p.mkdir(parents=True, exist_ok=True)
		return p

	@property
	def max_horizon(self) -> int:
		return max(self.HORIZONS)

	@property
	def position_size(self) -> float:
		return self.INITIAL_CAPITAL / (self.TOP_N + self.BOTTOM_N)


CONFIG = TradingConfig()
