"""Lookahead regression test — inject future spike, past forecast must not change."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsfm_backtest.config import BacktestConfig, DataConfig
from tsfm_backtest.data.loader import ETFPanel
from tsfm_backtest.forecasters.baselines import NaiveForecaster, ARIMAForecaster


def _make_synthetic_panel(n_days: int = 300, n_tickers: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rows = []
    for ticker in [f"T{i}" for i in range(n_tickers)]:
        vol = 0.15 + 0.02 * np.sin(np.arange(n_days) / 20)
        for i, d in enumerate(dates):
            rows.append(
                {
                    "timestamp": d,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                    "log_return": 0.001,
                    "realized_vol": vol[i],
                }
            )
    return pd.DataFrame(rows)


class SyntheticPanel:
    """Minimal panel for lookahead tests without network."""

    def __init__(self, panel: pd.DataFrame, config: BacktestConfig):
        self._panel = panel
        self.config = config

    def get_panel(self, up_to: pd.Timestamp | str | None = None) -> pd.DataFrame:
        df = self._panel.copy()
        if up_to is not None:
            df = df[df["timestamp"] <= pd.Timestamp(up_to)]
        return df.reset_index(drop=True)


def _inject_future_spike(panel: pd.DataFrame, t: pd.Timestamp, multiplier: float = 100.0) -> pd.DataFrame:
    mutated = panel.copy()
    mask = mutated["timestamp"] > t
    mutated.loc[mask, "realized_vol"] *= multiplier
    mutated.loc[mask, "log_return"] *= multiplier
    return mutated


def _assert_forecasts_equal(fc_a, fc_b, rtol: float = 1e-9) -> None:
    pd.testing.assert_frame_equal(fc_a.mean, fc_b.mean, rtol=rtol, atol=1e-12)
    for q in fc_a.quantiles:
        pd.testing.assert_frame_equal(fc_a.quantiles[q], fc_b.quantiles[q], rtol=rtol, atol=1e-12)


def test_naive_no_lookahead():
    panel_df = _make_synthetic_panel()
    config = BacktestConfig()
    panel = SyntheticPanel(panel_df, config)
    t = panel_df["timestamp"].iloc[200]
    horizon = 5

    context_a = panel.get_panel(up_to=t)
    fc_a = NaiveForecaster().predict(context_a, horizon)

    mutated_full = _inject_future_spike(panel_df, t)
    panel_mutated = SyntheticPanel(mutated_full, config)
    context_b = panel_mutated.get_panel(up_to=t)
    fc_b = NaiveForecaster().predict(context_b, horizon)

    _assert_forecasts_equal(fc_a, fc_b)


def test_arima_no_lookahead():
    panel_df = _make_synthetic_panel()
    config = BacktestConfig()
    panel = SyntheticPanel(panel_df, config)
    t = panel_df["timestamp"].iloc[200]
    horizon = 5

    context_a = panel.get_panel(up_to=t)
    fc_a = ARIMAForecaster().predict(context_a, horizon)

    mutated_full = _inject_future_spike(panel_df, t)
    panel_mutated = SyntheticPanel(mutated_full, config)
    context_b = panel_mutated.get_panel(up_to=t)
    fc_b = ARIMAForecaster().predict(context_b, horizon)

    _assert_forecasts_equal(fc_a, fc_b)


def test_get_panel_point_in_time():
    config = DataConfig(
        tickers=["SPY"],
        start_date="2023-01-01",
        end_date="2023-06-30",
        cache_dir="data/cache",
    )
    panel = ETFPanel(config)
    full = panel.get_panel()
    cutoff = full["timestamp"].iloc[len(full) // 2]
    sliced = panel.get_panel(up_to=cutoff)
    assert sliced["timestamp"].max() <= cutoff
    assert len(sliced) < len(full)


def test_realized_vol_positive_finite():
    config = DataConfig(
        tickers=["SPY"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        cache_dir="data/cache",
    )
    panel = ETFPanel(config)
    vol = panel.get_vol_series().dropna()
    assert (vol > 0).all().all()
    assert np.isfinite(vol.values).all()
