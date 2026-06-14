from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 52) -> float:
    if returns.empty or returns.std() == 0:
        return np.nan
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 52) -> float:
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods_per_year: int = 52) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return np.nan
    ann_return = returns.mean() * periods_per_year
    return float(ann_return / mdd)


def portfolio_metrics(result) -> dict[str, float]:
    returns = result.returns
    equity = result.equity_curve
    turnovers = [r.turnover for r in result.records]
    costs = [r.cost for r in result.records]

    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(returns, equity),
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else np.nan,
        "mean_turnover": float(np.mean(turnovers)) if turnovers else np.nan,
        "total_cost": float(np.sum(costs)) if costs else np.nan,
        "n_rebalances": len(result.records),
    }


def metrics_table(results: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        m = portfolio_metrics(result)
        m["strategy"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("strategy")
