"""Technical-indicator feature engineering (pure pandas/numpy, no extra deps)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to df's index (rows with NaNs retained)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)

    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)

    for n in (5, 10, 20, 50):
        sma = c.rolling(n).mean()
        f[f"sma_ratio_{n}"] = c / sma - 1
    f["ema_12_26"] = c.ewm(span=12).mean() / c.ewm(span=26).mean() - 1

    f["rsi_14"] = _rsi(c)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c

    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - (mid - 2 * std)) / (4 * std).replace(0, np.nan)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    f["vol_20"] = c.pct_change().rolling(20).std()
    f["hl_range"] = (h - l) / c
    f["vol_chg"] = v.pct_change().replace([np.inf, -np.inf], np.nan)
    f["vol_ratio_20"] = v / v.rolling(20).mean()

    for lag in (1, 2, 3, 5):
        f[f"lag_ret_{lag}"] = c.pct_change().shift(lag)

    f["dow"] = df.index.dayofweek
    return f.replace([np.inf, -np.inf], np.nan)


def make_dataset(df: pd.DataFrame, horizon: int = 1):
    """X, y_return, y_direction, and the index. Target = return `horizon` days ahead."""
    feats = build_features(df)
    fut_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    data = feats.assign(target_ret=fut_ret).dropna()
    X = data.drop(columns="target_ret")
    y = data["target_ret"]
    return X, y, (y > 0).astype(int)
