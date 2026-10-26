# 📈 ArthaSeer — AI Stock Price Predictor

**ArthaSeer** (*Artha* = wealth, *Seer* = one who foresees) is an end-to-end machine-learning project that downloads
stock data, engineers technical indicators, trains an ensemble of models, validates them honestly with
walk-forward testing, backtests the signal, and serves everything in an interactive dashboard.

> ⚠️ **Educational project only. Not financial advice.** Markets are extremely noisy; no model reliably
> predicts prices. Never trade real money based on this output alone.

---

## Features

- **Live data** from Yahoo Finance via `yfinance` (US stocks, Indian stocks like `TCS.NS`, indices like `^NSEI`)
- **Offline fallback**: automatically uses synthetic demo data if the internet / Yahoo is unavailable
- **~35 engineered features**: returns, SMA/EMA ratios, RSI, MACD, Bollinger position, ATR, volatility, volume ratios, lags
- **Ensemble of 3 models**: Ridge regression, Random Forest, Gradient Boosting (predicts next-N-day return)
- **Honest validation**: `TimeSeriesSplit` walk-forward (no look-ahead leakage) + comparison with an "always up" baseline
- **Backtest**: long/cash strategy with trading costs vs buy-and-hold (return, Sharpe, max drawdown)
- **Prediction output**: predicted price, 95% range, BUY/HOLD/SELL signal, multi-day forecast path
- **Custom web front end** (HTML/CSS/JS, no build step, works offline, dark/light mode, mobile-friendly) served by a Flask API
- Optional **Streamlit dashboard** and **command-line interface**
- **Unit tests**, including a look-ahead-leakage test

## Project structure

```
ArthaSeer/
├── README.md
├── requirements.txt
├── server.py               # Flask server: JSON API + web front end
├── static/                 # front end
│   ├── index.html
│   ├── style.css
│   └── app.js              # canvas charts, no external libraries
├── app.py                  # optional Streamlit dashboard
├── arthaseer/
│   ├── __init__.py
│   ├── data.py             # Yahoo Finance loader + synthetic fallback
│   ├── features.py         # technical indicators & dataset builder
│   ├── model.py            # ensemble, walk-forward CV, prediction, forecast, save/load
│   ├── backtest.py         # strategy vs buy-and-hold
│   └── cli.py              # command-line interface
└── tests/
    └── test_arthaseer.py
```

## Installation

Requires Python 3.9+.

```bash
cd ArthaSeer
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Web app (recommended)

```bash
python server.py
```

Open **http://localhost:8000**, enter a ticker (e.g. `AAPL`, `TCS.NS`, `RELIANCE.NS`, `^NSEI`), pick settings and click
**Train & Predict**. Use `PORT=5000 python server.py` to change the port. The first prediction for a ticker takes
roughly 10-30 seconds (training); results are cached for an hour.

API: `GET /api/predict?ticker=AAPL&period=5y&horizon=1&days=10` returns JSON (prediction, forecast, metrics, backtest,
feature importance). `GET /api/health` is a health check.

### 1b. Streamlit dashboard (alternative)

```bash
streamlit run app.py
```

### 2. Command line

```bash
# Train on 5 years of data and print validation + backtest results
python -m arthaseer.cli train AAPL

# Predict the next close, with a 10-day forecast path
python -m arthaseer.cli predict AAPL --days 10

# Predict 3 days ahead instead of 1
python -m arthaseer.cli train TCS.NS --horizon 3
python -m arthaseer.cli predict TCS.NS --horizon 3
```

Trained models are saved in `models/`.

### 3. Tests

```bash
pytest -q
```

## How it works

1. **Data** – OHLCV prices are downloaded (split/dividend adjusted).
2. **Features** – indicators are computed using only past data. Target = return `N` days ahead.
3. **Training** – three models are fit; their average is the ensemble prediction.
4. **Validation** – 5-fold walk-forward split: always train on the past, test on the future.
5. **Prediction** – the latest feature row produces a predicted return → price, with a 95% interval
   based on out-of-sample residuals. The multi-day path widens uncertainty with √t.
6. **Backtest** – go long when the predicted return is positive, otherwise hold cash (0.05% cost per trade).

## Reading the results honestly

On random-walk-like data (such as the built-in synthetic series), directional accuracy hovers around
50–53%, which is roughly the same as the "always up" baseline. **That is expected** — the project is designed to
show you this truthfully rather than hide it. Always compare the model against the baseline and buy-and-hold before
believing any result, and remember that backtests overstate real-world performance.

## Ideas for extending

- Add an LSTM / Transformer model (PyTorch or TensorFlow) to the ensemble
- Add news sentiment features (e.g. headlines scored with an NLP model)
- Add macro features (interest rates, USD/INR, VIX, crude oil)
- Hyperparameter search with Optuna
- Portfolio-level prediction and position sizing

## License

MIT — free to use, modify and share for learning purposes.
