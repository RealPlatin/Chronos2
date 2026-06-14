from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_backtest.config import (
    load_backtest_config,
    load_data_config,
    load_experiment_config,
)
from tsfm_backtest.data.loader import ETFPanel
from tsfm_backtest.engine.walk_forward import (
    run_equal_weight_backtest,
    run_vol_target_backtest,
)
from tsfm_backtest.evaluation.forecast_metrics import aggregate_forecast_metrics
from tsfm_backtest.evaluation.portfolio_metrics import metrics_table, portfolio_metrics
from tsfm_backtest.forecasters.baselines import get_baseline_forecaster
from tsfm_backtest.forecasters.chronos2 import Chronos2Forecaster
from tsfm_backtest.viz.plots import (
    generate_tearsheet,
    plot_equity_drawdown_overlay,
    plot_forecast_fan_chart,
)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def get_forecaster(name: str, experiment_cfg) -> object:
    if name in ("naive", "seasonal_naive", "arima"):
        return get_baseline_forecaster(name)
    if name == "chronos2":
        return Chronos2Forecaster(
            model_id=experiment_cfg.chronos_model_id,
            device=experiment_cfg.chronos_device,
        )
    raise ValueError(f"Unknown forecaster: {name}")


def run_experiment(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    exp_cfg = load_experiment_config(config_path)
    data_cfg = load_data_config(exp_cfg.data_config)
    bt_cfg = load_backtest_config(exp_cfg.backtest_config)

    set_seeds(bt_cfg.seed)
    output_dir = Path(exp_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = ETFPanel(data_cfg)
    _ = panel.panel  # trigger download/cache

    results: dict[str, object] = {}
    forecaster_results: dict[str, object] = {}

    naive_fc = get_baseline_forecaster("naive")
    naive_result = run_vol_target_backtest(panel, naive_fc, bt_cfg, "naive_vol_target")
    results["naive_vol_target"] = naive_result
    forecaster_results["naive"] = naive_result

    if "equal_weight" in exp_cfg.portfolio_strategies:
        results["equal_weight"] = run_equal_weight_backtest(panel, bt_cfg)

    for fc_name in exp_cfg.forecasters:
        if fc_name == "naive":
            continue
        print(f"\nRunning forecaster: {fc_name}")
        forecaster = get_forecaster(fc_name, exp_cfg)
        strategy_name = f"{fc_name}_vol_target"
        if strategy_name in exp_cfg.portfolio_strategies or fc_name == "chronos2":
            result = run_vol_target_backtest(panel, forecaster, bt_cfg, strategy_name)
            results[strategy_name] = result
            forecaster_results[fc_name] = result

    port_metrics = metrics_table(results)
    port_metrics.to_csv(output_dir / "portfolio_metrics.csv")

    for name, result in results.items():
        if not result.equity_curve.empty:
            result.equity_curve.to_csv(output_dir / f"equity_{name}.csv", header=True)

    forecast_metrics: dict[str, dict] = {}
    realized_vol = panel.get_vol_series()
    for fc_name, result in forecaster_results.items():
        if fc_name == "naive":
            continue
        forecast_metrics[fc_name] = aggregate_forecast_metrics(
            result.records, realized_vol, naive_result.records
        )

    with open(output_dir / "forecast_metrics.json", "w") as f:
        json.dump(forecast_metrics, f, indent=2, default=_json_default)

    summary = _build_summary(port_metrics, forecast_metrics, results)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    if exp_cfg.generate_plots:
        try:
            _generate_plots(exp_cfg, results, panel, output_dir)
        except Exception as exc:
            print(f"Warning: plot generation failed: {exc}")

    if exp_cfg.generate_tearsheet and "chronos2_vol_target" in results:
        chronos_returns = results["chronos2_vol_target"].returns
        if not chronos_returns.empty:
            generate_tearsheet(
                chronos_returns,
                output_dir / "chronos2_tearsheet.html",
                title="Chronos-2 Vol-Target",
            )

    _print_summary(summary)
    return summary


def _json_default(obj):
    if isinstance(obj, (np.floating, float)) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _build_summary(
    port_metrics: pd.DataFrame,
    forecast_metrics: dict,
    results: dict,
) -> dict:
    chronos_sharpe = port_metrics.loc["chronos2_vol_target", "sharpe"] if "chronos2_vol_target" in port_metrics.index else None
    ew_sharpe = port_metrics.loc["equal_weight", "sharpe"] if "equal_weight" in port_metrics.index else None
    chronos_mase = forecast_metrics.get("chronos2", {}).get("mase_vs_naive")
    naive_mase = 1.0

    beats_ew = bool(chronos_sharpe > ew_sharpe) if chronos_sharpe is not None and ew_sharpe is not None else None
    beats_naive_vol = bool(chronos_mase < naive_mase) if chronos_mase is not None else None

    return {
        "headline": {
            "chronos2_beats_equal_weight_after_costs": beats_ew,
            "chronos2_beats_naive_on_vol_mase": beats_naive_vol,
            "chronos2_sharpe": float(chronos_sharpe) if chronos_sharpe is not None else None,
            "equal_weight_sharpe": float(ew_sharpe) if ew_sharpe is not None else None,
            "chronos2_mase_vs_naive": float(chronos_mase) if chronos_mase is not None else None,
        },
        "portfolio_metrics": port_metrics.to_dict(),
        "forecast_metrics": forecast_metrics,
    }


def _generate_plots(exp_cfg, results, panel, output_dir: Path) -> None:
    equity_curves = {
        name: r.equity_curve
        for name, r in results.items()
        if not r.equity_curve.empty
    }
    if equity_curves:
        plot_equity_drawdown_overlay(equity_curves, output_dir / "equity_drawdown_overlay.png")

    if "chronos2_vol_target" not in results:
        return

    chronos_result = results["chronos2_vol_target"]
    vol_series = panel.get_vol_series()

    for ticker in exp_cfg.fan_chart_tickers:
        if ticker not in vol_series.columns:
            continue
        realized = vol_series[ticker].dropna()
        if realized.empty:
            continue

        timestamps = [r.timestamp for r in chronos_result.records[:20]]
        fc_means, fc_lowers, fc_uppers, fc_dates = [], [], [], []

        for rec in chronos_result.records[:20]:
            if ticker not in rec.forecast.mean.columns:
                continue
            fc_dates.append(rec.timestamp)
            fc_means.append(rec.forecast.mean.iloc[0][ticker])
            if 0.1 in rec.forecast.quantiles:
                fc_lowers.append(rec.forecast.quantiles[0.1].iloc[0][ticker])
                fc_uppers.append(rec.forecast.quantiles[0.9].iloc[0][ticker])

        if not fc_dates:
            continue

        plot_forecast_fan_chart(
            timestamps=timestamps,
            realized=realized,
            forecast_lower=pd.Series(fc_lowers, index=fc_dates),
            forecast_upper=pd.Series(fc_uppers, index=fc_dates),
            forecast_mean=pd.Series(fc_means, index=fc_dates),
            ticker=ticker,
            output_path=output_dir / f"fan_chart_{ticker}.png",
        )


def _print_summary(summary: dict) -> None:
    h = summary["headline"]
    print("\n=== Phase 1 Results ===")
    print(f"Chronos-2 Sharpe (net):     {h.get('chronos2_sharpe')}")
    print(f"Equal-weight Sharpe (net): {h.get('equal_weight_sharpe')}")
    print(f"Chronos-2 MASE vs naive:     {h.get('chronos2_mase_vs_naive')}")
    print(f"Beats equal-weight?         {h.get('chronos2_beats_equal_weight_after_costs')}")
    print(f"Beats naive on vol (MASE<1)? {h.get('chronos2_beats_naive_on_vol_mase')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TSFM Phase 1 backtest experiment")
    parser.add_argument(
        "--config",
        default="config/experiments/phase1.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
