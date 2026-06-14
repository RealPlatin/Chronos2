"""Transaction cost model tests."""

from tsfm_backtest.engine.costs import apply_cost_to_return, turnover_cost
import pandas as pd


def test_turnover_cost_reduces_return():
    prev = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"A": 0.6, "B": 0.4})
    cost = turnover_cost(prev, new, cost_bps=10.0)
    gross = 0.01
    net = apply_cost_to_return(gross, cost)
    assert net < gross
    assert abs(net - (gross - cost)) < 1e-12


def test_first_rebalance_no_cost():
    new = pd.Series({"A": 0.5, "B": 0.5})
    assert turnover_cost(None, new, cost_bps=10.0) == 0.0
