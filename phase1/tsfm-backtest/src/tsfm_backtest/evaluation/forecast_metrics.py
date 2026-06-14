from __future__ import annotations

import numpy as np
import pandas as pd


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    naive_forecast: np.ndarray,
) -> float:
    """Mean Absolute Scaled Error vs naive benchmark."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    naive_forecast = np.asarray(naive_forecast, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(forecast) & np.isfinite(naive_forecast)
    if mask.sum() == 0:
        return np.nan
    errors = np.abs(actual[mask] - forecast[mask])
    scale = np.mean(np.abs(actual[mask] - naive_forecast[mask]))
    if scale == 0:
        return np.nan
    return float(errors.mean() / scale)


def crps(
    actual: float,
    quantiles: dict[float, float],
) -> float:
    """Continuous Ranked Probability Score from quantile forecasts."""
    score = 0.0
    for q, pred in sorted(quantiles.items()):
        diff = actual - pred
        score += (q - (1 if diff < 0 else 0)) * diff
    return float(score)


def quantile_calibration(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actuals inside [lower, upper] band."""
    mask = np.isfinite(actuals) & np.isfinite(lower) & np.isfinite(upper)
    if mask.sum() == 0:
        return np.nan
    inside = (actuals[mask] >= lower[mask]) & (actuals[mask] <= upper[mask])
    return float(inside.mean())


def aggregate_forecast_metrics(
    records: list,
    realized_vol: pd.DataFrame,
    naive_records: list,
) -> dict[str, float]:
    """Compute MASE, mean CRPS, and 80% band calibration across rebalance steps."""
    actuals, forecasts, naive_fcs = [], [], []
    crps_vals = []
    lowers, uppers, actual_vals = [], [], []
    naive_scales: list[float] = []

    for i, (rec, naive_rec) in enumerate(zip(records, naive_records, strict=False)):
        eval_t = records[i + 1].timestamp if i + 1 < len(records) else rec.timestamp
        if eval_t not in realized_vol.index:
            continue
        realized = realized_vol.loc[eval_t]
        fc_mean = rec.forecast.mean.iloc[0]
        naive_mean = naive_rec.forecast.mean.iloc[0]

        for ticker in fc_mean.index:
            if ticker not in realized.index or not np.isfinite(realized[ticker]):
                continue
            actual = realized[ticker]
            actuals.append(actual)
            forecasts.append(fc_mean[ticker])
            naive_fcs.append(naive_mean[ticker])

            vol_series = realized_vol[ticker].dropna()
            if len(vol_series) > 1:
                naive_scales.append(float(vol_series.diff().abs().mean()))

            q = rec.forecast.quantiles
            if 0.1 in q and 0.9 in q:
                lowers.append(q[0.1].iloc[0][ticker])
                uppers.append(q[0.9].iloc[0][ticker])
                actual_vals.append(actual)
                qdict = {k: v.iloc[0][ticker] for k, v in q.items()}
                crps_vals.append(crps(actual, qdict))

    mase_val = mase(np.array(actuals), np.array(forecasts), np.array(naive_fcs))
    if np.isnan(mase_val) and naive_scales:
        errors = np.abs(np.array(actuals) - np.array(forecasts))
        scale = float(np.mean(naive_scales))
        if scale > 0:
            mase_val = float(errors.mean() / scale)

    return {
        "mase_vs_naive": mase_val,
        "mean_crps": float(np.nanmean(crps_vals)) if crps_vals else np.nan,
        "calibration_80pct": quantile_calibration(
            np.array(actual_vals), np.array(lowers), np.array(uppers)
        ),
        "n_obs": len(actuals),
    }
