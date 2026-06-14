from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from tsfm_backtest.forecasters.base import ForecastDistribution


def _context_to_vol_wide(context: pd.DataFrame) -> pd.DataFrame:
    """Convert long context panel to wide realized-vol DataFrame."""
    if "realized_vol" not in context.columns:
        raise ValueError("context must contain realized_vol column")
    vol = context.pivot(index="timestamp", columns="ticker", values="realized_vol")
    return vol.sort_index()


def _make_forecast_distribution(
    tickers: list[str],
    horizon: int,
    mean_values: np.ndarray,
    quantile_levels: list[float] | None = None,
) -> ForecastDistribution:
    if quantile_levels is None:
        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

    steps = np.arange(1, horizon + 1)
    mean_df = pd.DataFrame(mean_values, index=steps, columns=tickers)

    quantiles: dict[float, pd.DataFrame] = {}
    for q in quantile_levels:
        spread = 0.15 * (0.5 + abs(q - 0.5)) * mean_values
        quantiles[q] = pd.DataFrame(mean_values + spread * np.sign(q - 0.5), index=steps, columns=tickers)

    return ForecastDistribution(mean=mean_df, quantiles=quantiles, series_ids=tickers)


class NaiveForecaster:
    """Vol forecast = last observed realized vol (random walk on vol)."""

    name = "naive"

    def predict(self, context: pd.DataFrame, horizon: int) -> ForecastDistribution:
        vol = _context_to_vol_wide(context)
        tickers = list(vol.columns)
        last_vol = vol.iloc[-1].values
        mean_values = np.tile(last_vol, (horizon, 1))
        return _make_forecast_distribution(tickers, horizon, mean_values)


class SeasonalNaiveForecaster:
    """Vol forecast = realized vol from same weekday one season ago."""

    name = "seasonal_naive"
    season: int = 5

    def predict(self, context: pd.DataFrame, horizon: int) -> ForecastDistribution:
        vol = _context_to_vol_wide(context)
        tickers = list(vol.columns)
        forecasts = []
        for h in range(1, horizon + 1):
            idx = -self.season if len(vol) >= self.season else -1
            forecasts.append(vol.iloc[idx].values)
        mean_values = np.array(forecasts)
        return _make_forecast_distribution(tickers, horizon, mean_values)


class ARIMAForecaster:
    """Per-series ARIMA fit on the in-window vol history."""

    name = "arima"
    order: tuple[int, int, int] = (1, 0, 1)

    def predict(self, context: pd.DataFrame, horizon: int) -> ForecastDistribution:
        vol = _context_to_vol_wide(context)
        tickers = list(vol.columns)
        mean_values = np.zeros((horizon, len(tickers)))

        for j, ticker in enumerate(tickers):
            series = vol[ticker].dropna()
            if len(series) < 30:
                last = series.iloc[-1] if len(series) else 0.1
                mean_values[:, j] = last
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series.values, order=self.order)
                    fit = model.fit()
                    fc = fit.forecast(steps=horizon)
                    mean_values[:, j] = np.maximum(fc, 1e-6)
            except Exception:
                mean_values[:, j] = series.iloc[-1]

        return _make_forecast_distribution(tickers, horizon, mean_values)


def get_baseline_forecaster(name: str):
    mapping = {
        "naive": NaiveForecaster,
        "seasonal_naive": SeasonalNaiveForecaster,
        "arima": ARIMAForecaster,
    }
    if name not in mapping:
        raise ValueError(f"Unknown baseline forecaster: {name}")
    return mapping[name]()
