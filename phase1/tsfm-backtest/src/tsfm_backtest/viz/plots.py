from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_drawdown_overlay(
    equity_curves: dict[str, pd.Series],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name, equity in equity_curves.items():
        axes[0].plot(equity.index, equity.values, label=name, linewidth=1.5)

    axes[0].set_ylabel("Equity")
    axes[0].set_title("Equity Curve Overlay")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, equity in equity_curves.items():
        peak = equity.cummax()
        dd = (equity - peak) / peak
        axes[1].plot(dd.index, dd.values, label=name, linewidth=1.5)

    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Drawdown Overlay")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_forecast_fan_chart(
    timestamps: list[pd.Timestamp],
    realized: pd.Series,
    forecast_lower: pd.Series,
    forecast_upper: pd.Series,
    forecast_mean: pd.Series,
    ticker: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(realized.index, realized.values, color="black", label="Realized vol", linewidth=1.2)
    ax.fill_between(
        forecast_mean.index,
        forecast_lower.values,
        forecast_upper.values,
        alpha=0.3,
        color="steelblue",
        label="80% forecast band",
    )
    ax.plot(forecast_mean.index, forecast_mean.values, color="steelblue", label="Forecast mean")

    for t in timestamps:
        ax.axvline(t, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    ax.set_title(f"Volatility Forecast Fan Chart — {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized realized vol")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_tearsheet(returns: pd.Series, output_path: Path, title: str = "Strategy") -> None:
    import quantstats as qs

    qs.extend_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(returns, output=str(output_path), title=title)
