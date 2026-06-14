from __future__ import annotations

import numpy as np
import pandas as pd

from tsfm_backtest.forecasters.base import ForecastDistribution


def vol_target_weights(
    forecast: ForecastDistribution,
    return_history: pd.DataFrame,
    vol_budget: float,
    lookback_corr: int = 60,
) -> pd.Series:
    """
    Convert forecasted vols to inverse-vol weights, scaled to a target portfolio vol.

    w_i ∝ 1/σ̂_i, normalized long-only, then scaled using forecast vol + sample correlations.
    """
    tickers = forecast.series_ids or list(forecast.mean.columns)
    vol_hat = forecast.mean.iloc[0].reindex(tickers).fillna(forecast.mean.iloc[0].median())
    vol_hat = vol_hat.clip(lower=1e-6)

    inv_vol = 1.0 / vol_hat
    raw_weights = inv_vol / inv_vol.sum()

    hist = return_history[tickers].dropna().tail(lookback_corr)
    if len(hist) < 10:
        return raw_weights

    corr = hist.corr().fillna(0).values.copy()
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vol_hat.values, vol_hat.values) * corr
    port_var = float(raw_weights.values @ cov @ raw_weights.values)
    port_vol = np.sqrt(max(port_var, 1e-12))

    if port_vol > 0:
        scale = min(vol_budget / port_vol, 1.0)
    else:
        scale = 1.0

    return pd.Series(raw_weights * scale, index=tickers)


def equal_weight(tickers: list[str]) -> pd.Series:
    w = 1.0 / len(tickers)
    return pd.Series(w, index=tickers)
