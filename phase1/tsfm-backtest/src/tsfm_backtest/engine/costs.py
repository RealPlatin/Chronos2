from __future__ import annotations

import pandas as pd


def turnover_cost(
    prev_weights: pd.Series | None,
    new_weights: pd.Series,
    cost_bps: float,
) -> float:
    """Per-rebalance transaction cost as fraction of portfolio (bps × turnover)."""
    if prev_weights is None:
        return 0.0
    aligned_prev = prev_weights.reindex(new_weights.index).fillna(0.0)
    turnover = float((new_weights - aligned_prev).abs().sum())
    return turnover * (cost_bps / 10_000.0)


def apply_cost_to_return(gross_return: float, cost: float) -> float:
    return gross_return - cost
