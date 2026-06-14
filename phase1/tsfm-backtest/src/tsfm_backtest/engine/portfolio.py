from __future__ import annotations

import pandas as pd


def portfolio_return(
    weights: pd.Series,
    returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Compound log returns over [start, end] with fixed weights."""
    period = returns.loc[(returns.index > start) & (returns.index <= end)]
    if period.empty:
        return 0.0
    aligned = weights.reindex(period.columns).fillna(0.0)
    daily = (period * aligned).sum(axis=1)
    return float(daily.sum())


def compute_turnover(prev_weights: pd.Series | None, new_weights: pd.Series) -> float:
    if prev_weights is None:
        return float(new_weights.abs().sum())
    aligned_prev = prev_weights.reindex(new_weights.index).fillna(0.0)
    return float((new_weights - aligned_prev).abs().sum())
