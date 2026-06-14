from __future__ import annotations

import pandas as pd

from tsfm_backtest.forecasters.base import ForecastDistribution


class Chronos2Forecaster:
    """Zero-shot Chronos-2 forecaster for realized-vol series (joint basket)."""

    name = "chronos2"

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str = "cpu",
        quantile_levels: list[float] | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.quantile_levels = quantile_levels or [0.1, 0.25, 0.5, 0.75, 0.9]
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
            )
        return self._pipeline

    def _prepare_context(self, context: pd.DataFrame) -> pd.DataFrame:
        vol_context = context[["timestamp", "ticker", "realized_vol"]].dropna().copy()
        vol_context = vol_context.rename(
            columns={"ticker": "id", "realized_vol": "target"}
        )
        vol_context["id"] = vol_context["id"].astype(str)
        vol_context["timestamp"] = pd.to_datetime(vol_context["timestamp"])

        frames = []
        for ticker, grp in vol_context.groupby("id"):
            s = grp.set_index("timestamp")["target"].sort_index()
            s = s.asfreq("B", method="ffill").dropna()
            frames.append(
                pd.DataFrame({"id": ticker, "timestamp": s.index, "target": s.values})
            )
        return pd.concat(frames, ignore_index=True)

    def predict(self, context: pd.DataFrame, horizon: int) -> ForecastDistribution:
        pipeline = self._load_pipeline()
        vol_context = self._prepare_context(context)

        pred_df = pipeline.predict_df(
            vol_context,
            prediction_length=horizon,
            quantile_levels=self.quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            cross_learning=True,
        )

        tickers = sorted(vol_context["id"].unique())
        steps = list(range(1, horizon + 1))

        mean_frames = []
        quantile_frames: dict[float, list[pd.DataFrame]] = {q: [] for q in self.quantile_levels}

        for ticker in tickers:
            ticker_pred = pred_df[pred_df["id"] == ticker].sort_values("timestamp")
            if len(ticker_pred) < horizon:
                raise ValueError(f"Chronos-2 returned insufficient forecasts for {ticker}")

            mean_col = "target" if "target" in ticker_pred.columns else "0.5"
            if mean_col not in ticker_pred.columns:
                mean_col = [c for c in ticker_pred.columns if c not in ("id", "timestamp")][0]

            mean_frames.append(
                pd.DataFrame({ticker: ticker_pred[mean_col].values[:horizon]}, index=steps)
            )
            for q in self.quantile_levels:
                qcol = str(q)
                if qcol in ticker_pred.columns:
                    quantile_frames[q].append(
                        pd.DataFrame({ticker: ticker_pred[qcol].values[:horizon]}, index=steps)
                    )

        mean_df = pd.concat(mean_frames, axis=1)
        quantiles = {
            q: pd.concat(quantile_frames[q], axis=1) if quantile_frames[q] else mean_df.copy()
            for q in self.quantile_levels
        }

        return ForecastDistribution(mean=mean_df, quantiles=quantiles, series_ids=tickers)
