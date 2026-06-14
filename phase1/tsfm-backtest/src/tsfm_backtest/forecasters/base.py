from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class ForecastDistribution:
    """Probabilistic forecast for one or more series over a horizon."""

    mean: pd.DataFrame
    quantiles: dict[float, pd.DataFrame]
    samples: np.ndarray | None = None
    series_ids: list[str] = field(default_factory=list)

    def point_forecast(self) -> pd.DataFrame:
        return self.mean


class Forecaster(Protocol):
    """All forecasters implement predict(context, horizon) -> ForecastDistribution."""

    name: str

    def predict(self, context: pd.DataFrame, horizon: int) -> ForecastDistribution:
        """Forecast realized-vol series from a point-in-time context panel."""
        ...
