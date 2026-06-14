from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from tsfm_backtest.config import BacktestConfig
from tsfm_backtest.data.loader import ETFPanel
from tsfm_backtest.engine.costs import apply_cost_to_return, turnover_cost
from tsfm_backtest.engine.portfolio import compute_turnover, portfolio_return
from tsfm_backtest.forecasters.base import Forecaster, ForecastDistribution
from tsfm_backtest.signals.vol_target import equal_weight, vol_target_weights


@dataclass
class RebalanceRecord:
    timestamp: pd.Timestamp
    forecast: ForecastDistribution
    weights: pd.Series
    gross_return: float
    cost: float
    net_return: float
    turnover: float


@dataclass
class BacktestResult:
    strategy: str
    forecaster: str
    records: list[RebalanceRecord] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    forecasts_log: list[dict] = field(default_factory=list)


def _rebalance_dates(returns: pd.DataFrame, freq: str, min_history: int) -> list[pd.Timestamp]:
    dates = returns.index[min_history:]
    grouped = dates.to_series().groupby(pd.Grouper(freq=freq))
    rebalance = [grp.max() for _, grp in grouped if len(grp) > 0]
    return sorted(rebalance)


def _slice_context(panel: ETFPanel, t: pd.Timestamp, config: BacktestConfig) -> pd.DataFrame:
    ctx = panel.get_panel(up_to=t)
    if config.window_type == "rolling":
        cutoff = t - pd.Timedelta(days=config.rolling_window_days)
        ctx = ctx[ctx["timestamp"] >= cutoff]
    return ctx


def run_walk_forward(
    panel: ETFPanel,
    forecaster: Forecaster,
    config: BacktestConfig,
    weight_fn: Callable[[ForecastDistribution, pd.DataFrame], pd.Series],
    strategy_name: str,
) -> BacktestResult:
    """Walk-forward backtest with a single time cursor — no lookahead."""
    returns = panel.get_return_series()
    tickers = panel.tickers()
    dates = _rebalance_dates(returns, config.rebalance_freq, config.min_history_days)

    result = BacktestResult(strategy=strategy_name, forecaster=forecaster.name)
    prev_weights: pd.Series | None = None
    equity = 1.0
    equity_points: list[tuple[pd.Timestamp, float]] = []
    net_returns: list[tuple[pd.Timestamp, float]] = []

    for i, t in enumerate(dates[:-1]):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{strategy_name}] rebalance {i + 1}/{len(dates) - 1} @ {t.date()}")
        end_t = dates[i + 1]
        context = _slice_context(panel, t, config)
        ret_hist = panel.get_return_series(up_to=t)

        forecast = forecaster.predict(context, config.horizon)
        weights = weight_fn(forecast, ret_hist)

        gross = portfolio_return(weights, returns, t, end_t)
        cost = turnover_cost(prev_weights, weights, config.transaction_cost_bps)
        net = apply_cost_to_return(gross, cost)
        turnover = compute_turnover(prev_weights, weights)

        record = RebalanceRecord(
            timestamp=t,
            forecast=forecast,
            weights=weights,
            gross_return=gross,
            cost=cost,
            net_return=net,
            turnover=turnover,
        )
        result.records.append(record)
        result.forecasts_log.append(
            {
                "timestamp": t,
                "mean": forecast.mean.iloc[0].to_dict(),
                "weights": weights.to_dict(),
            }
        )

        equity *= np.exp(net)
        equity_points.append((end_t, equity))
        net_returns.append((end_t, net))
        prev_weights = weights

    result.equity_curve = pd.Series(
        [e for _, e in equity_points],
        index=pd.DatetimeIndex([d for d, _ in equity_points]),
        name="equity",
    )
    result.returns = pd.Series(
        [r for _, r in net_returns],
        index=pd.DatetimeIndex([d for d, _ in net_returns]),
        name="return",
    )
    return result


def run_vol_target_backtest(
    panel: ETFPanel,
    forecaster: Forecaster,
    config: BacktestConfig,
    strategy_name: str | None = None,
) -> BacktestResult:
    name = strategy_name or f"{forecaster.name}_vol_target"

    def weight_fn(forecast: ForecastDistribution, ret_hist: pd.DataFrame) -> pd.Series:
        return vol_target_weights(forecast, ret_hist, config.vol_budget)

    return run_walk_forward(panel, forecaster, config, weight_fn, name)


def run_equal_weight_backtest(
    panel: ETFPanel,
    config: BacktestConfig,
) -> BacktestResult:
    tickers = panel.tickers()
    ew = equal_weight(tickers)

    class _StaticForecaster:
        name = "equal_weight"

        def predict(self, context: pd.DataFrame, horizon: int) -> ForecastDistribution:
            from tsfm_backtest.forecasters.baselines import NaiveForecaster

            return NaiveForecaster().predict(context, horizon)

    def weight_fn(_forecast: ForecastDistribution, _ret_hist: pd.DataFrame) -> pd.Series:
        return ew.copy()

    return run_walk_forward(panel, _StaticForecaster(), config, weight_fn, "equal_weight")
