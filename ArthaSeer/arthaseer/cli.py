"""Command-line interface:  python -m arthaseer.cli train|predict ..."""

from __future__ import annotations

import argparse
import os

from .backtest import backtest
from .data import load_prices
from .model import ArthaSeerModel

MODEL_DIR = "models"


def _path(ticker: str, horizon: int) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, f"{ticker.replace('^', '')}_h{horizon}.joblib")


def cmd_train(a):
    df, src = load_prices(a.ticker, a.period)
    print(f"Loaded {len(df)} rows for {a.ticker} ({src})")
    model = ArthaSeerModel(a.ticker, a.horizon).fit(df)
    print("\nWalk-forward validation:")
    for name, m in model.metrics.items():
        print(f"  {name:20s}", {k: round(v, 4) for k, v in m.items()})
    bt = backtest(model.cv_predictions)
    print("\nBacktest  strategy:", {k: round(v, 3) for k, v in bt["strategy"].items()})
    print("Backtest buy&hold:", {k: round(v, 3) for k, v in bt["buy_and_hold"].items()})
    p = _path(a.ticker, a.horizon)
    model.save(p)
    print(f"\nModel saved to {p}")


def cmd_predict(a):
    p = _path(a.ticker, a.horizon)
    if not os.path.exists(p):
        raise SystemExit(f"No trained model at {p}. Run:  python -m arthaseer.cli train {a.ticker}")
    model = ArthaSeerModel.load(p)
    df, src = load_prices(a.ticker, "2y")
    res = model.predict_next(df)
    print(f"\nArthaSeer prediction for {a.ticker}  (data: {src})")
    print(f"  Last close      : {res['last_close']:.2f}")
    print(f"  Predicted price : {res['predicted_price']:.2f}  ({res['predicted_return']*100:+.2f}%)")
    print(f"  95% range       : {res['low_price']:.2f} - {res['high_price']:.2f}")
    print(f"  Signal          : {res['signal']}")
    if a.days > 1:
        print("\nMulti-day forecast:")
        print(model.forecast_path(df, a.days).round(2).to_string())
    print("\nEducational use only - not financial advice.")


def main():
    ap = argparse.ArgumentParser(prog="arthaseer", description="ArthaSeer stock predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("predict", cmd_predict)):
        s = sub.add_parser(name)
        s.add_argument("ticker", help="e.g. AAPL, TCS.NS, RELIANCE.NS, ^NSEI")
        s.add_argument("--horizon", type=int, default=1, help="days ahead to predict")
        if name == "train":
            s.add_argument("--period", default="5y")
        else:
            s.add_argument("--days", type=int, default=1, help="multi-day forecast length")
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
