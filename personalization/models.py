"""Model A (absolute), B (linear residual), and C (HGB residual) on shared features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

FEATURE_COLS_A = ("user_mean", "sim", "log_n")
FEATURE_COLS_B = ("sim", "log_n")
FEATURE_COLS_C = ("user_mean", "sim", "log_n", "n_hist", "global_ridge")


def _design(frame: pd.DataFrame, cols: tuple[str, ...], use_global_ridge: bool) -> np.ndarray:
    names = list(cols)
    if use_global_ridge:
        names.append("global_ridge")
    return frame[names].to_numpy(dtype=float)


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def clip_stars(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1.0, 5.0)


class ModelA:
    """Absolute-rating Ridge on user_mean + taste similarity (+ optional global ridge)."""

    def __init__(self, alpha: float = 10.0, use_global_ridge: bool = False):
        self.alpha = alpha
        self.use_global_ridge = use_global_ridge
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, feats: pd.DataFrame, y: np.ndarray) -> "ModelA":
        self.model.fit(_design(feats, FEATURE_COLS_A, self.use_global_ridge), y)
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return clip_stars(self.model.predict(_design(feats, FEATURE_COLS_A, self.use_global_ridge)))


class ModelB:
    """Two-stage: hard user base + residual Ridge on similarity features."""

    def __init__(
        self,
        base: str = "user_median",
        alpha: float = 10.0,
        use_global_ridge: bool = False,
        residual_shift: float = 0.0,
    ):
        if base not in {"user_mean", "user_median"}:
            raise ValueError(base)
        self.base = base
        self.alpha = alpha
        self.use_global_ridge = use_global_ridge
        self.residual_shift = residual_shift
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, feats: pd.DataFrame, y: np.ndarray) -> "ModelB":
        base = feats[self.base].to_numpy(dtype=float)
        residual = y - base
        self.model.fit(_design(feats, FEATURE_COLS_B, self.use_global_ridge), residual)
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        base = feats[self.base].to_numpy(dtype=float)
        delta = self.model.predict(_design(feats, FEATURE_COLS_B, self.use_global_ridge))
        return clip_stars(base + delta + self.residual_shift)

    def predict_raw_residual(self, feats: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_design(feats, FEATURE_COLS_B, self.use_global_ridge))


def tune_residual_shift(base: np.ndarray, residual: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """Pick constant shift on residual that minimizes MAE on validation."""
    best_c, best_mae = 0.0, float("inf")
    for c in grid:
        m = mae(y, clip_stars(base + residual + c))
        if m < best_mae:
            best_mae, best_c = m, float(c)
    return best_c, best_mae


class ModelC:
    """MSE-oriented two-stage model: hard user_mean base + HGB residual.

    Residual features can include user_mean / global_ridge so the tree can learn
    interactions (e.g. when content disagrees with the user level).
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        min_samples_leaf: int = 50,
        l2_regularization: float = 0.0,
        use_global_ridge: bool = True,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.use_global_ridge = use_global_ridge
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=0,
        )

    def _X(self, feats: pd.DataFrame) -> np.ndarray:
        cols = list(FEATURE_COLS_C) if self.use_global_ridge else [c for c in FEATURE_COLS_C if c != "global_ridge"]
        # engineered interactions for the tree
        base = feats[cols].to_numpy(dtype=float)
        sim = feats["sim"].to_numpy(dtype=float)
        log_n = feats["log_n"].to_numpy(dtype=float)
        user_mean = feats["user_mean"].to_numpy(dtype=float)
        ridge = feats["global_ridge"].to_numpy(dtype=float) if self.use_global_ridge else np.zeros(len(feats))
        extra = np.column_stack(
            [
                sim * log_n,
                ridge - user_mean,
                np.abs(sim),
            ]
        )
        return np.hstack([base, extra])

    def fit(self, feats: pd.DataFrame, y: np.ndarray) -> "ModelC":
        base = feats["user_mean"].to_numpy(dtype=float)
        self.model.fit(self._X(feats), y - base)
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        base = feats["user_mean"].to_numpy(dtype=float)
        return clip_stars(base + self.model.predict(self._X(feats)))
