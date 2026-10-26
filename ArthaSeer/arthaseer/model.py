"""Model training, walk-forward evaluation, and forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, make_dataset


def _models() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=7
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=7
        ),
    }


@dataclass
class ArthaSeerModel:
    ticker: str
    horizon: int = 1
    models: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    # ---- training -------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "ArthaSeerModel":
        X, y, y_dir = make_dataset(df, self.horizon)
        self.feature_names = list(X.columns)
        self.metrics = self._walk_forward(X, y, y_dir)
        self.models = {name: m.fit(X, y) for name, m in _models().items()}
        return self

    def _walk_forward(self, X, y, y_dir, splits: int = 5) -> dict:
        tscv = TimeSeriesSplit(n_splits=splits, gap=self.horizon)
        preds = {n: pd.Series(np.nan, index=y.index) for n in _models()}
        for tr, te in tscv.split(X):
            for name, model in _models().items():
                model.fit(X.iloc[tr], y.iloc[tr])
                preds[name].iloc[te] = model.predict(X.iloc[te])
        preds["ensemble"] = pd.concat(preds.values(), axis=1).mean(axis=1)

        mask = ~preds["ridge"].isna()
        out = {}
        for name, p in preds.items():
            yt, yp = y[mask], p[mask]
            out[name] = {
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "directional_accuracy": float(((yt > 0) == (yp > 0)).mean()),
            }
        out["baseline_always_up"] = {"directional_accuracy": float((y[mask] > 0).mean())}
        self.cv_predictions = pd.DataFrame({"actual": y[mask], "predicted": preds["ensemble"][mask]})
        return out

    # ---- inference ------------------------------------------------------
    def predict_next(self, df: pd.DataFrame) -> dict:
        feats = build_features(df).iloc[[-1]][self.feature_names]
        if feats.isna().any(axis=None):
            raise ValueError("Not enough history to compute features for the latest row.")
        per_model = {n: float(m.predict(feats)[0]) for n, m in self.models.items()}
        ens = float(np.mean(list(per_model.values())))
        last = float(df["Close"].iloc[-1])
        spread = float(np.std(list(per_model.values())))
        resid = float(self.cv_predictions.eval("actual - predicted").std())
        return {
            "last_close": last,
            "predicted_return": ens,
            "predicted_price": last * (1 + ens),
            "low_price": last * (1 + ens - 1.96 * resid),
            "high_price": last * (1 + ens + 1.96 * resid),
            "signal": "BUY" if ens > 0.002 else "SELL" if ens < -0.002 else "HOLD",
            "model_disagreement": spread,
            "per_model_return": per_model,
        }

    def forecast_path(self, df: pd.DataFrame, days: int = 10) -> pd.DataFrame:
        """Iterative multi-day forecast (rolls predictions forward; uncertainty grows)."""
        work = df.copy()
        rows = []
        for _ in range(days):
            p = self.predict_next(work)
            nxt = work.index[-1] + pd.offsets.BDay(1)
            price = p["predicted_price"]
            rows.append({"Date": nxt, "forecast": price, "low": p["low_price"], "high": p["high_price"]})
            work.loc[nxt] = [price, price * 1.003, price * 0.997, price, work["Volume"].tail(20).mean()]
        out = pd.DataFrame(rows).set_index("Date")
        widen = np.sqrt(np.arange(1, days + 1))
        mid = out["forecast"]
        out["low"] = mid - (mid - out["low"]) * widen
        out["high"] = mid + (out["high"] - mid) * widen
        return out

    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ArthaSeerModel":
        return joblib.load(path)
