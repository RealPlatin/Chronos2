from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from tsfm_backtest.config import DataConfig


class ETFPanel:
    """Download, cache, and serve point-in-time ETF panels."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._panel: pd.DataFrame | None = None

    @property
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = self._load_or_download()
        return self._panel

    def _cache_path(self) -> Path:
        tickers_key = "_".join(sorted(self.config.tickers))
        return self.cache_dir / f"etf_panel_{tickers_key}.parquet"

    def _normalize_yfinance_df(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Flatten yfinance output to a standard OHLCV frame."""
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(0):
                df = df[ticker].copy()
            else:
                df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        rename_map = {c: str(c).lower() for c in df.columns}
        df = df.rename(columns=rename_map)
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        return df

    def _load_or_download(self) -> pd.DataFrame:
        cache_path = self._cache_path()
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        raw = yf.download(
            self.config.tickers,
            start=self.config.start_date,
            end=self.config.end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        frames: list[pd.DataFrame] = []
        for ticker in self.config.tickers:
            if len(self.config.tickers) == 1:
                ticker_raw = raw
            elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                ticker_raw = raw[ticker]
            else:
                ticker_raw = raw
            ticker_df = self._normalize_yfinance_df(ticker_raw, ticker)
            ticker_df["ticker"] = ticker
            ticker_df["log_return"] = np.log(ticker_df["close"] / ticker_df["close"].shift(1))
            frames.append(ticker_df)

        panel = pd.concat(frames, ignore_index=True)
        panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)
        panel = panel.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        panel = self._add_realized_vol(panel)
        panel.to_parquet(cache_path, index=False)
        return panel

    def _add_realized_vol(self, panel: pd.DataFrame) -> pd.DataFrame:
        w = self.config.vol_window
        ann = self.config.annualization_factor
        vols: list[pd.Series] = []
        for ticker, grp in panel.groupby("ticker"):
            rolling_std = grp["log_return"].rolling(w, min_periods=w).std()
            vol = rolling_std * np.sqrt(ann)
            vols.append(vol)
        panel["realized_vol"] = pd.concat(vols).sort_index()
        return panel

    def get_panel(self, up_to: pd.Timestamp | str | None = None) -> pd.DataFrame:
        """Return panel with no rows dated after `up_to` (point-in-time accessor)."""
        df = self.panel.copy()
        if up_to is not None:
            cutoff = pd.Timestamp(up_to)
            df = df[df["timestamp"] <= cutoff]
        return df.reset_index(drop=True)

    def get_vol_series(self, up_to: pd.Timestamp | str | None = None) -> pd.DataFrame:
        """Wide-format realized vol panel indexed by timestamp."""
        panel = self.get_panel(up_to=up_to)
        vol = panel.pivot(index="timestamp", columns="ticker", values="realized_vol")
        return vol.sort_index()

    def get_return_series(self, up_to: pd.Timestamp | str | None = None) -> pd.DataFrame:
        """Wide-format log-return panel indexed by timestamp."""
        panel = self.get_panel(up_to=up_to)
        rets = panel.pivot(index="timestamp", columns="ticker", values="log_return")
        return rets.sort_index()

    def tickers(self) -> list[str]:
        return list(self.config.tickers)
