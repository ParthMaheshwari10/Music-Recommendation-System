"""Audio feature engineering for similarity search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, normalize

from .config import BOUNDED_FEATURES, FEATURE_WEIGHTS


ENGINEERED_FEATURES = [
    *BOUNDED_FEATURES,
    "loudness",
    "tempo_log",
    "duration_log",
    "key_sin",
    "key_cos",
    "key_known",
    "mode",
    "time_signature",
]


class AudioFeatureTransformer:
    """Create a robust, weighted, unit-normalized audio embedding."""

    def __init__(self, feature_weights: dict[str, float] | None = None):
        self.feature_weights = feature_weights or FEATURE_WEIGHTS
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.medians_: pd.Series | None = None
        self._fitted = False

    @staticmethod
    def engineer(tracks: pd.DataFrame) -> pd.DataFrame:
        result = tracks[BOUNDED_FEATURES].astype(float).copy()
        result["loudness"] = tracks["loudness"].astype(float).clip(-60, 5)
        result["tempo_log"] = np.log1p(tracks["tempo"].astype(float).clip(20, 300))
        result["duration_log"] = np.log1p(
            tracks["duration_ms"].astype(float).clip(30_000, 3_600_000)
        )

        key = tracks["key"].astype(float)
        known = key.between(0, 11)
        radians = 2 * np.pi * key.where(known, 0) / 12.0
        result["key_sin"] = np.sin(radians) * known
        result["key_cos"] = np.cos(radians) * known
        result["key_known"] = known.astype(float)
        result["mode"] = tracks["mode"].astype(float)
        result["time_signature"] = tracks["time_signature"].astype(float)
        return result[ENGINEERED_FEATURES]

    def fit(self, tracks: pd.DataFrame) -> "AudioFeatureTransformer":
        engineered = self.engineer(tracks)
        self.scaler.fit(engineered)
        self.medians_ = tracks.median(numeric_only=True)
        self._fitted = True
        return self

    def transform(self, tracks: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AudioFeatureTransformer must be fitted before transform().")
        values = self.scaler.transform(self.engineer(tracks))
        weights = np.sqrt(
            np.array([self.feature_weights.get(name, 1.0) for name in ENGINEERED_FEATURES])
        )
        return normalize(values * weights, norm="l2").astype(np.float32)

    def fit_transform(self, tracks: pd.DataFrame) -> np.ndarray:
        return self.fit(tracks).transform(tracks)

    def mood_row(self, targets: dict[str, float]) -> pd.DataFrame:
        """Construct an otherwise-neutral synthetic row for a mood target."""
        if self.medians_ is None:
            raise RuntimeError("AudioFeatureTransformer must be fitted before mood_row().")
        values = self.medians_.to_dict()
        values.update(targets)
        return pd.DataFrame([values])

