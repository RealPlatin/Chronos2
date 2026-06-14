# TSFM Backtest — Phase 1

Point-in-time, lookahead-free walk-forward backtesting framework evaluating **Chronos-2** foundation-model forecasts as volatility-driven portfolio signals across a multi-asset ETF universe.

## The one honest number

**Yes on both counts** (2015–2024, net of 7.5 bps turnover costs):

| Question | Result |
|----------|--------|
| Chronos-2 vol-target beats equal-weight after costs? | **Yes** — Sharpe 0.80 vs 0.73 |
| Chronos-2 beats naive on vol forecasting (MASE < 1)? | **Yes** — MASE 0.95 |

Full metrics in `results/phase1/summary.json`.

## Universe (no survivorship bias)

Hand-picked, currently-trading liquid ETFs — no delisting filter needed:

`XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, SPY, TLT, IEF, GLD, EFA`

## Quick start

```bash
cd tsfm-backtest
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -m "not slow" -v
tsfm-backtest --config config/experiments/phase1.yaml
```

Chronos-2 inference runs on CPU by default. First run downloads market data and model weights.

## Architecture

```
Data Layer → Forecaster Layer → Signal Layer → Backtest Engine → Evaluation → Viz
```

- **Data**: yfinance OHLC → Parquet cache; `get_panel(up_to=t)` enforces point-in-time
- **Forecasters**: Naive, SeasonalNaive, ARIMA, Chronos-2 (joint basket)
- **Signal**: inverse-vol weights scaled to 10% annualized vol budget
- **Engine**: weekly walk-forward, 5-day horizon, 7.5 bps turnover cost
- **Evaluation**: MASE/CRPS/calibration + Sharpe/Sortino/maxDD vs equal-weight & naive-signal

## Outputs

| File | Description |
|------|-------------|
| `results/phase1/portfolio_metrics.csv` | Sharpe, Sortino, maxDD, turnover |
| `results/phase1/forecast_metrics.json` | MASE vs naive, CRPS, 80% calibration |
| `results/phase1/summary.json` | Headline yes/no answers |
| `results/phase1/equity_drawdown_overlay.png` | Strategy comparison |
| `results/phase1/fan_chart_*.png` | Vol forecast bands vs realized |
| `results/phase1/chronos2_tearsheet.html` | quantstats tearsheet |

## Lookahead guarantee

`tests/test_lookahead.py` injects a future spike and asserts past forecasts are unchanged. Run in CI on every push.

## CV entry

> Built a point-in-time, lookahead-free walk-forward backtesting framework (Python) evaluating Chronos-2 foundation-model forecasts as volatility-driven portfolio signals across a multi-asset ETF universe; benchmarked vs naive/ARIMA baselines with transaction-cost-aware Sharpe/drawdown evaluation.
