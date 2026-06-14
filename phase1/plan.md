# Phase 1 — Foundation Harness + Chronos-2 Vol-Targeted Backtest
### Standalone Project Build Guide

> Part 1 of 2. Companion to the master blueprint (`tsfm-backtesting-pipeline-plan.md`) and `phase-2-timesfm-benchmark-and-rigor.md`. This document is self-contained: it builds the entire harness from scratch and ships a defensible, presentable result using **Chronos-2 only**. TimesFM is deliberately out of scope here.

**CV entry this unlocks (paste only once shipped):**
> Built a point-in-time, lookahead-free walk-forward backtesting framework (Python) evaluating Chronos-2 foundation-model forecasts as volatility-driven portfolio signals across a multi-asset ETF universe; benchmarked vs naive/ARIMA baselines with transaction-cost-aware Sharpe/drawdown evaluation.

---

## 0. Definition of done (acceptance criteria)

Phase 1 is shipped when **all** of these are true:

- [ ] A fixed ETF basket downloads and caches to Parquet; a `get_panel(up_to=t)` accessor returns **nothing dated after `t`**.
- [ ] A `Forecaster` Protocol exists, with **Naive**, **SeasonalNaive**, **ARIMA**, and **Chronos-2** implementations behind it.
- [ ] A walk-forward engine runs expanding/rolling windows, and a **lookahead regression test passes** (injecting a future spike does not change a past forecast).
- [ ] Forecasts drive a **volatility-targeted** multi-asset portfolio; turnover and costs are modelled.
- [ ] Evaluation reports **MASE vs naive on the vol forecast** and **cost-aware Sharpe / Sortino / max-drawdown vs equal-weight and naive-signal benchmarks**.
- [ ] A **quantstats tearsheet** + an **equity/drawdown overlay** + one **forecast fan chart** are generated.
- [ ] The whole run is **reproducible from a config file** with fixed seeds, and the README states the one honest number.

**The one honest number:** does the Chronos-2 vol-targeted book beat equal-weight after costs, and does Chronos-2 beat naive on volatility forecasting (MASE < 1)?

---

## 1. Framing for Phase 1

- **The no-lookahead invariant is the entire credibility of this phase.** Build it as a structural property of the code, not a thing you hope you avoided.
- **Baselines are first-class.** Naive (random walk) and ARIMA are not afterthoughts — they are the bar. A foundation model that can't clear them is the finding.
- **Plumbing before models.** Get a full equity curve out of the *baselines* end-to-end before Chronos-2 ever loads. That checkpoint (the old "Phase 0") is ~60% of the project's value and needs no GPU.
- **Volatility is the target, on purpose.** Volatility clusters and is forecastable; raw return direction is not. Vol-targeting is where the economic story and the empirical edge align — and it runs fine on CPU.
- **Scope discipline:** one universe, one lead strategy, one foundation model. Resist adding TimesFM, more strategies, or fine-tuning — that is all Phase 2.

---

## 2. Architecture (this phase)

```
Data Layer ─► Forecaster Layer ─► Signal Layer ─► Backtest Engine ─► Evaluation ─► Viz
(ETF panel,    (Protocol:          (vol-target     (walk-forward      (forecast +    (tearsheet,
point-in-time) .predict→dist)       sizing)         time cursor)       portfolio)     equity, fan)
```

**The `Forecaster` Protocol (the keystone — every model implements it):**

```python
class Forecaster(Protocol):
    def predict(self, context: Panel, horizon: int) -> ForecastDistribution: ...
# ForecastDistribution: mean, quantiles{0.1..0.9}, optional sample paths.
# Zero-shot (Chronos-2): predict() is stateless.
# Classical (ARIMA): fit-on-window happens inside predict().
```

**The walk-forward engine (one function owns the time cursor → lookahead is impossible):**

```python
for t in rebalance_dates:
    window   = data.get_panel(up_to=t)        # POINT-IN-TIME: never anything after t
    fc       = forecaster.predict(window, h)  # forecasts realized-vol series, ≤ t only
    weights  = vol_target(fc, vol_budget)     # inverse-vol, scaled to a vol budget
    pnl      = engine.step(weights, costs)    # apply turnover cost, realize over h
    log(t, fc, weights, pnl)
```

**The lookahead regression test (write this early, keep it green):**

```python
# Take a window ending at t. Record forecast_A.
# Mutate the series AFTER t (inject a huge spike). Re-slice up_to=t. Record forecast_B.
# assert forecast_A == forecast_B   # if not, you have lookahead.
```

**Cross-sectional note:** Chronos-2 can forecast the whole basket jointly (group attention). The engine should accept both a per-series loop and a joint-panel call; in Phase 1 either is fine — pick joint to exercise Chronos-2's strength.

---

## 3. Repository scaffold (create this in full)

```
tsfm-backtest/
├── pyproject.toml              # uv-managed
├── README.md                   # the write-up + the one honest number
├── config/
│   ├── data.yaml               # basket tickers, date range, source
│   ├── backtest.yaml           # window type, rebalance freq, horizon, vol budget, costs
│   └── experiments/phase1.yaml # the run config
├── src/tsfm_backtest/
│   ├── data/loader.py          # download, cache Parquet, point-in-time panel
│   ├── forecasters/
│   │   ├── base.py             # Forecaster Protocol + ForecastDistribution
│   │   ├── baselines.py        # Naive, SeasonalNaive, ARIMA
│   │   └── chronos2.py         # Chronos2Forecaster
│   ├── signals/vol_target.py   # forecast vol → inverse-vol weights @ vol budget
│   ├── engine/
│   │   ├── walk_forward.py      # the time cursor
│   │   ├── costs.py             # per-turnover bps
│   │   └── portfolio.py         # weights → returns, turnover
│   ├── evaluation/
│   │   ├── forecast_metrics.py  # MASE, CRPS, quantile calibration
│   │   └── portfolio_metrics.py # Sharpe, Sortino, maxDD, turnover
│   ├── viz/plots.py             # equity/drawdown overlay, fan chart
│   └── config.py                # Pydantic schemas
├── tests/test_lookahead.py      # the regression test above
└── results/                     # gitignored
```

---

## 4. Tech stack (Phase 1 subset — all CPU-friendly)

| Layer | Tool | Note |
|---|---|---|
| Runtime / deps | Python 3.11+, **uv** | `uv venv && uv pip install -e .` |
| Data wrangling | **pandas** | + Parquet caching |
| Market data | **yfinance** | OHLC for the fixed basket; long clean histories |
| Foundation model | **`chronos-forecasting`** (≥2.1) | `amazon/chronos-2`, pandas API, CPU inference |
| Classical baselines | **statsmodels** | ARIMA/ETS |
| Portfolio tearsheet | **quantstats** | full HTML tearsheet, near-zero effort |
| Forecast metrics | custom + scipy | MASE, CRPS |
| Plotting | **matplotlib / plotly** | equity, drawdown, fan chart |
| Config / reproducibility | **Pydantic + YAML**, fixed seeds | every run reconstructable |
| Dev hygiene | pytest, ruff, mypy, pre-commit | the lookahead test lives here |

**The research universe (fixed, eliminates survivorship bias):** ~12–18 liquid ETFs spanning sectors and asset classes, e.g. sector SPDRs `XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU` + asset-class sleeves `SPY, TLT, IEF, GLD, EFA`. Hand-picked, currently trading, no delisting → no survivorship problem. State this explicitly in the README.

---

## 5. Build sequence (the core — ordered, each step has an acceptance check)

**Step 1 — Scaffold & environment.** Create the repo tree, `pyproject.toml`, pre-commit (ruff + mypy), a stub GitHub Action. ✅ *`uv pip install -e .` succeeds; `pytest` runs (zero tests ok).*

**Step 2 — Data layer.** Download the basket OHLC via yfinance, cache to Parquet, build `get_panel(up_to=t)` returning a tidy panel of log returns + OHLC. ✅ *`get_panel(up_to=t)` contains no rows dated after `t`.*

**Step 3 — Realized-vol target series.** For each asset, construct a daily **realized-volatility series** — start with 20-day annualized rolling std of log returns; note range-based estimators (Parkinson / Garman-Klass / Yang-Zhang from OHLC) as a later efficiency upgrade. *This series is what the models forecast.* ✅ *Vol series is positive, finite, aligned to the return panel.*

**Step 4 — Forecaster Protocol + baselines.** Implement `base.py` (Protocol + `ForecastDistribution`) and `baselines.py`: **Naive** (vol forecast = last realized vol — a *strong* baseline because vol is persistent), **SeasonalNaive**, **ARIMA**. ✅ *Each baseline returns a horizon-`h` forecast on a window.*

**Step 5 — Walk-forward engine + lookahead test.** Implement the time cursor (expanding/rolling), horizon `h` = rebalance frequency (e.g. forecast next 5 trading days' vol, rebalance weekly, hold 5d). Write `test_lookahead.py`. ✅ *The lookahead regression test passes.*

**Step 6 — Vol-targeting signal.** Convert forecasted vols to inverse-vol weights `w_i ∝ 1/σ̂_i`, normalize, then scale the whole book to a target annualized portfolio vol (e.g. 10%) using forecasted vols + sample correlations. ✅ *Weights sum to 1 (long-only) or to the leverage cap; turnover is computed per rebalance.*

**Step 7 — Cost model.** Apply a per-turnover transaction cost (e.g. 5–10 bps × turnover). ✅ *Net returns < gross by exactly the modelled cost.*

**Step 8 — End-to-end on baselines only (the "plumbing complete" checkpoint).** Run the full pipeline with the Naive/ARIMA forecasters. ✅ *A complete equity curve exists from baselines before any foundation model loads.*

**Step 9 — Plug in Chronos-2.** Implement `Chronos2Forecaster.predict`: pandas in → quantile forecast of the realized-vol series out, via `Chronos2Pipeline` (verify the exact call against current `chronos-forecasting` docs). Forecast the basket jointly to use group attention. ✅ *Chronos-2 produces vol forecasts + a full backtest through the same engine.*

**Step 10 — Evaluation.** Forecast-level: **MASE vs naive** on the vol forecast (MASE < 1 = you beat the random-walk-on-vol), CRPS, quantile-calibration check. Portfolio-level: Sharpe, Sortino, max-drawdown, turnover — always **vs equal-weight and naive-signal benchmarks**. ✅ *A metrics table answers the two headline questions.*

**Step 11 — Visualization + reproducibility wrap.** Generate a quantstats tearsheet; plot equity + drawdown overlay (Chronos-2 vol-target vs equal-weight vs naive); plot one forecast fan chart (predicted quantile band vs realized vol). Make the run config-driven with fixed seeds; write the README with the result. ✅ *A fresh clone reproduces the run and the figures from one command.*

---

## 6. Evaluation (Phase 1)

**Forecast-level (on the realized-vol series):**
- **MASE** vs the naive vol forecast — the sanity metric. Vol is persistent, so this is a genuinely hard bar.
- **CRPS / WQL** — you have quantiles; score the distribution, not just the point.
- **Quantile calibration** — do the 80% bands cover ~80% of realized vol?

**Portfolio-level (always vs benchmarks):**
- **Sharpe, Sortino, max-drawdown, Calmar, turnover, exposure.**
- Benchmarks: **equal-weight** rebalanced book + **naive-signal** book (vol-target using the naive vol forecast). The naive-signal book isolates "did the *model's* forecast add value over a trivial forecast?"

---

## 7. Visualization deliverables (Phase 1)

1. **quantstats HTML tearsheet** for the Chronos-2 vol-target strategy.
2. **Equity + drawdown overlay**: Chronos-2 vol-target vs equal-weight vs naive-signal, one axis.
3. **Forecast fan chart**: predicted vol quantile band vs realized, for 1–2 sample ETFs.

---

## 8. Pitfalls & guardrails

- **Survivorship bias** — eliminated by the hand-picked, currently-trading basket. *Say so in the README* (turning a handled bias into a stated design choice is the signal).
- **Lookahead** — handled by the cursor invariant + `test_lookahead.py`. Run that test in CI.
- **Forecast the right object** — forecast a **realized-vol series**, not price (price forecasting lets the model "cheat" by echoing the last value). Define the vol estimator and horizon precisely.
- **Overlapping-window leakage** — if your realized-vol estimator looks back `k` days, ensure the lookback at time `t` uses only data ≤ `t`. This interacts with the cursor; the lookahead test should catch violations.
- **Costs are not optional** — report net-of-cost everywhere; gross-only numbers are not defensible.
- **Don't peek** — no choosing the basket, vol budget, or horizon by what maximises in-sample Sharpe. Fix them up front from economic reasoning.

---

## 9. The CV entry this unlocks

Ship the acceptance checklist in §0, then use:

> Built a point-in-time, lookahead-free walk-forward backtesting framework (Python) evaluating Chronos-2 foundation-model forecasts as volatility-driven portfolio signals across a multi-asset ETF universe; benchmarked vs naive/ARIMA baselines with transaction-cost-aware Sharpe/drawdown evaluation.

When ready, proceed to **Phase 2** to add the TimesFM 2.5 benchmark arm, the multi-target experiment matrix, and the statistical rigour that upgrades the entry to a full model comparison with a defensible finding.
