"""Simple long/flat backtest of the model's directional signal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(cv_predictions: pd.DataFrame, threshold: float = 0.0, cost: float = 0.0005) -> dict:
    """Go long when predicted return > threshold, else stay in cash. Includes trading cost."""
    pos = (cv_predictions["predicted"] > threshold).astype(int)
    trades = pos.diff().abs().fillna(0)
    strat = pos * cv_predictions["actual"] - trades * cost
    buyhold = cv_predictions["actual"]

    def stats(r: pd.Series) -> dict:
        eq = (1 + r).cumprod()
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
        dd = (eq / eq.cummax() - 1).min()
        return {"total_return": float(eq.iloc[-1] - 1), "sharpe": float(sharpe), "max_drawdown": float(dd)}

    return {
        "strategy": stats(strat),
        "buy_and_hold": stats(buyhold),
        "equity_strategy": (1 + strat).cumprod(),
        "equity_buyhold": (1 + buyhold).cumprod(),
    }
