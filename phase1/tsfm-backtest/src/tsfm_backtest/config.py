from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    cache_dir: str = "data/cache"
    vol_window: int = 20
    annualization_factor: int = 252


class BacktestConfig(BaseModel):
    window_type: Literal["expanding", "rolling"] = "expanding"
    rolling_window_days: int = 504
    rebalance_freq: str = "W-FRI"
    horizon: int = 5
    vol_budget: float = 0.10
    transaction_cost_bps: float = 7.5
    min_history_days: int = 252
    seed: int = 42


class ExperimentConfig(BaseModel):
    experiment_name: str
    data_config: str
    backtest_config: str
    forecasters: list[str] = Field(default_factory=lambda: ["naive", "arima", "chronos2"])
    portfolio_strategies: list[str] = Field(
        default_factory=lambda: ["chronos2_vol_target", "naive_vol_target", "equal_weight"]
    )
    output_dir: str = "results/phase1"
    generate_tearsheet: bool = True
    generate_plots: bool = True
    fan_chart_tickers: list[str] = Field(default_factory=lambda: ["SPY", "XLK"])
    chronos_model_id: str = "amazon/chronos-2"
    chronos_device: str = "cpu"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data_config(path: str | Path) -> DataConfig:
    return DataConfig(**load_yaml(path))


def load_backtest_config(path: str | Path) -> BacktestConfig:
    return BacktestConfig(**load_yaml(path))


def _project_root(config_path: Path) -> Path:
    if config_path.parent.name == "experiments":
        return config_path.parent.parent.parent
    return Path.cwd()


def load_experiment_config(path: str | Path, base_dir: Path | None = None) -> ExperimentConfig:
    path = Path(path)
    if base_dir is None:
        base_dir = _project_root(path)
    cfg = ExperimentConfig(**load_yaml(path))
    cfg.data_config = str((base_dir / cfg.data_config).resolve())
    cfg.backtest_config = str((base_dir / cfg.backtest_config).resolve())
    cfg.output_dir = str((base_dir / cfg.output_dir).resolve())
    return cfg
