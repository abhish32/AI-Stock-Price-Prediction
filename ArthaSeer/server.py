"""ArthaSeer web server: JSON API + front end.   Run:  python server.py  ->  http://localhost:8000"""

from __future__ import annotations

import os
import re
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from arthaseer.backtest import backtest
from arthaseer.data import load_prices
from arthaseer.model import ArthaSeerModel

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")

TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")
CACHE_TTL = 3600  # seconds
_cache: dict = {}
_lock = threading.Lock()


def _get_model(ticker: str, period: str, horizon: int):
    key = (ticker, period, horizon)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit["t"] < CACHE_TTL:
            return hit["model"], hit["df"], hit["src"]
        df, src = load_prices(ticker, period)
        model = ArthaSeerModel(ticker, horizon).fit(df)
        _cache[key] = {"t": time.time(), "model": model, "df": df, "src": src}
        return model, df, src


def _downsample(series, n=200):
    step = max(1, len(series) // n)
    return series.iloc[::step]


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/predict")
def predict():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    period = request.args.get("period", "5y")
    try:
        horizon = int(request.args.get("horizon", 1))
        days = int(request.args.get("days", 10))
    except ValueError:
        return jsonify(error="horizon and days must be integers"), 400
    if not TICKER_RE.match(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    if period not in {"2y", "5y", "10y"} or not 1 <= horizon <= 5 or not 1 <= days <= 30:
        return jsonify(error="Parameter out of range"), 400

    try:
        model, df, src = _get_model(ticker, period, horizon)
        res = model.predict_next(df)
        fc = model.forecast_path(df, days)
        bt = backtest(model.cv_predictions)
    except Exception as exc:  # surface a friendly message
        return jsonify(error=str(exc)), 500

    hist = df["Close"].tail(120)
    rf = model.models["random_forest"]
    imp = sorted(zip(model.feature_names, rf.feature_importances_), key=lambda x: -x[1])[:8]
    es, eb = _downsample(bt["equity_strategy"]), _downsample(bt["equity_buyhold"])

    return jsonify(
        ticker=ticker,
        source=src,
        horizon=horizon,
        prediction={k: v for k, v in res.items() if k != "per_model_return"},
        per_model=res["per_model_return"],
        history={"dates": [d.strftime("%Y-%m-%d") for d in hist.index], "close": [float(x) for x in hist]},
        forecast={
            "dates": [d.strftime("%Y-%m-%d") for d in fc.index],
            "mid": [float(x) for x in fc["forecast"]],
            "low": [float(x) for x in fc["low"]],
            "high": [float(x) for x in fc["high"]],
        },
        metrics=model.metrics,
        backtest={
            "strategy": bt["strategy"],
            "buy_and_hold": bt["buy_and_hold"],
            "dates": [d.strftime("%Y-%m-%d") for d in es.index],
            "equity_strategy": [float(x) for x in es],
            "equity_buyhold": [float(x) for x in eb],
        },
        importance=[{"feature": f, "value": float(v)} for f, v in imp],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"ArthaSeer running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
