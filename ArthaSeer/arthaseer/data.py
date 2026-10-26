"""Data loading. Uses Yahoo Finance (yfinance); falls back to synthetic data if offline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(days: int = 1500, seed: int = 42, start_price: float = 100.0) -> pd.DataFrame:
    """Generate a realistic-looking OHLCV series (geometric Brownian motion + regimes)."""
    rng = np.random.default_rng(seed)
    drift = 0.0004 + 0.0003 * np.sin(np.linspace(0, 12, days))
    vol = 0.012 + 0.006 * (np.sin(np.linspace(0, 25, days)) > 0.6)
    returns = rng.normal(drift, vol)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, days)))
    volume = rng.integers(1_000_000, 8_000_000, days).astype(float)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )
    df.index.name = "Date"
    return df


def load_prices(ticker: str, period: str = "5y", allow_synthetic: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source). source is 'yahoo' or 'synthetic'."""
    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 300:
            raise ValueError(f"Only {len(df)} rows returned for {ticker}")
        return df, "yahoo"
    except Exception as exc:  # network down, bad ticker, etc.
        if not allow_synthetic:
            raise RuntimeError(f"Could not download data for {ticker}: {exc}") from exc
        print(f"[ArthaSeer] Live data unavailable ({exc}). Using synthetic demo data.")
        return synthetic_prices(), "synthetic"
